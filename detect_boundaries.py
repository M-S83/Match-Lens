"""
detect_boundaries.py — Match Lens Step 1b

Two-phase boundary detection using Claude vision:
  Phase 1 — coarse scan (≤85 frames, every N seconds) → approximate 2-min regions
  Phase 2 — fine scan every 5s within each region → confirmed frame + timestamp

All API calls are chunked to ≤85 images. Merge: highest confidence wins;
if two candidates are within 30 s, the earlier one is preferred.
All images are resized to 640px wide (Pillow, JPEG quality 85) before encoding.

Post-detection validation checks five rules and retries failed boundaries
automatically. Writes match_boundaries.json.

Usage:
    python detect_boundaries.py [MATCH_DIR]
    python detect_boundaries.py [MATCH_DIR] --override   # accept even if validation fails
"""

import anthropic
import base64
import io
import json
import math
import os
import sys

from PIL import Image
from pipeline_schemas import stamp_schema_version

# Load API key from .env alongside this script.
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    from dotenv import dotenv_values as _dotenv_values
    for _k, _v in _dotenv_values(_env_path).items():
        if _v:
            os.environ[_k] = _v


# ── Constants ──────────────────────────────────────────────────────────────────

MODEL         = "claude-haiku-4-5-20251001"
CHUNK_SIZE    = 85        # max images per API request (hard limit is 100)
IMG_MAX_WIDTH = 640       # px — enough for scene classification
IMG_QUALITY   = 85        # JPEG quality
INTERVAL_P1   = 77        # seconds between Phase 1 frames (~90 frames / 115 min)
INTERVAL_P2   = 5         # seconds between Phase 2 frames

# Validation thresholds
MIN_KO1H_S    = 30        # KO 1H must be > 30s in (pre-match always precedes kickoff)
MIN_HALF_S    = 2400      # 40 min minimum half duration (non-league with stoppage)
MAX_HALF_S    = 3900      # 65 min maximum half duration
MIN_HT_BREAK  = 480       # 8 min minimum half-time break
MAX_HT_BREAK  = 1500      # 25 min maximum half-time break
MIN_TOTAL_S   = 5100      # 85 min minimum total match
MAX_TOTAL_S   = 7800      # 130 min maximum total match
RETRY_WINDOW  = 180       # ±3 min window expansion for validation retries


# ── Prompts ────────────────────────────────────────────────────────────────────

PHASE1_PROMPT = """\
You are identifying match boundaries in a football video. Do not analyse tactics.

Match: {match_info}

PHASE 1 — COARSE SCAN
Frames shown: sampled at regular intervals across the full recording.

For each boundary, identify the approximate 2-minute region where it occurs:

  KO_1H      — both teams in starting positions, ball at centre spot, referee present
  HT_WHISTLE — players stopping play, walking toward touchline or tunnel
  KO_2H      — teams back in position after halftime, ball at centre spot,
               attack direction reversed from first half
  FT_WHISTLE — final whistle: players celebrating, shaking hands, or sitting on pitch

Output ONLY raw JSON — no markdown, no commentary:

{{
  "coarse_scan_complete": true,
  "approximate_regions": {{
    "ko_1h":      {{"search_from": "XXmYYs", "search_to": "XXmYYs"}},
    "ht_whistle": {{"search_from": "XXmYYs", "search_to": "XXmYYs"}},
    "ko_2h":      {{"search_from": "XXmYYs", "search_to": "XXmYYs"}},
    "ft_whistle": {{"search_from": "XXmYYs", "search_to": "XXmYYs"}}
  }},
  "notes": "anything unusual"
}}"""

PHASE2_DESCRIPTIONS = {
    "ko_1h":      "first frame where the ball is at centre spot with both teams set in starting positions",
    "ht_whistle": "first frame where players have clearly stopped play (referee central, players moving toward touchline or tunnel)",
    "ko_2h":      "first frame where the ball is at centre spot for the second-half kickoff (attack direction reversed from first half)",
    "ft_whistle": "last frame of live play before any post-match activity (celebrations, handshakes, or players sitting on pitch)",
}

PHASE2_PROMPT = """\
Fine-scan to confirm {boundary_name}.

Frames shown: one every 5 seconds from {search_from} to {search_to}.

Identify the single frame where: {description}

Output ONLY raw JSON — no markdown, no commentary:

{{
  "boundary": "{boundary_name}",
  "confirmed_frame": "frame_XXmYYs.jpg",
  "confirmed_timestamp_seconds": 0,
  "confidence": 0.0,
  "notes": "any ambiguity or reason for low confidence"
}}"""

BOUNDARY_KEYS = ("ko_1h", "ht_whistle", "ko_2h", "ft_whistle")


# ── Image helpers ──────────────────────────────────────────────────────────────

def encode_image(path: str) -> tuple[str, str]:
    """Load image via Pillow, resize to IMG_MAX_WIDTH px wide, encode as base64 JPEG."""
    img = Image.open(path).convert("RGB")
    if img.width > IMG_MAX_WIDTH:
        ratio = IMG_MAX_WIDTH / img.width
        img   = img.resize((IMG_MAX_WIDTH, int(img.height * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=IMG_QUALITY)
    return base64.standard_b64encode(buf.getvalue()).decode("utf-8"), "image/jpeg"


def build_content(frame_paths: list[str], prompt: str) -> list[dict]:
    """Interleave [label, image, …] then append the prompt text."""
    content: list[dict] = []
    for path in frame_paths:
        fname = os.path.basename(path)
        secs  = frame_to_seconds(path)
        content.append({"type": "text",  "text": f"Frame: {fname} (t={secs}s)"})
        data, media_type = encode_image(path)
        content.append({"type": "image", "source": {"type": "base64",
                                                     "media_type": media_type,
                                                     "data": data}})
    content.append({"type": "text", "text": prompt})
    return content


# ── Frame helpers ──────────────────────────────────────────────────────────────

def frame_to_seconds(filename: str) -> int:
    """Convert frame_MMmSSs.jpg → total seconds."""
    name = os.path.basename(filename).replace("frame_", "").replace(".jpg", "")
    m, s = name.split("m")
    return int(m) * 60 + int(s.rstrip("s"))


def region_str_to_seconds(s: str) -> int:
    """Convert 'XXmYYs' → seconds. Tolerates missing padding."""
    s = s.strip().lower().rstrip("s")
    if "m" in s:
        parts = s.split("m")
        return int(parts[0]) * 60 + int(parts[1] or 0)
    return int(s)


def get_frames_in_range(frames_dir: str, start_s: int, end_s: int,
                        interval_s: int) -> list[str]:
    """Paths for 1-fps frames at interval_s steps from start_s to end_s."""
    result = []
    t = start_s
    while t <= end_s:
        m, s = divmod(t, 60)
        path = os.path.join(frames_dir, f"frame_{m:02d}m{s:02d}s.jpg")
        if os.path.exists(path):
            result.append(path)
        t += interval_s
    return result


# ── Chunked API helpers ────────────────────────────────────────────────────────

def _chunks(lst: list, n: int) -> list[list]:
    return [lst[i:i + n] for i in range(0, len(lst), n)]


def _merge_phase2(candidates: list[dict]) -> dict:
    """Highest-confidence result; prefer earlier if within 30 s."""
    best = max(candidates, key=lambda c: c["confidence"])
    for c in candidates:
        if c is best:
            continue
        diff = abs(c["confirmed_timestamp_seconds"] - best["confirmed_timestamp_seconds"])
        if diff <= 30 and c["confirmed_timestamp_seconds"] < best["confirmed_timestamp_seconds"]:
            best = c
    return best


def parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text.strip())


def run_phase2_scan(
    client: anthropic.Anthropic,
    frames: list[str],
    boundary_name: str,
    search_from_str: str,
    search_to_str: str,
    duration_s: int,
) -> dict:
    """Phase 2 scan with automatic chunking. Returns merged best result."""
    chunks     = _chunks(frames, CHUNK_SIZE)
    candidates = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            c_from = frame_to_seconds(chunk[0])
            c_to   = frame_to_seconds(chunk[-1])
            print(
                f"    chunk {i+1}/{len(chunks)}  "
                f"({c_from // 60}m{c_from % 60:02d}s–{c_to // 60}m{c_to % 60:02d}s, "
                f"{len(chunk)} frames)...",
                end=" ", flush=True,
            )
        prompt  = PHASE2_PROMPT.format(
            boundary_name=boundary_name,
            search_from=search_from_str,
            search_to=search_to_str,
            description=PHASE2_DESCRIPTIONS[boundary_name],
        )
        resp    = client.messages.create(
            model=MODEL, max_tokens=512,
            messages=[{"role": "user", "content": build_content(chunk, prompt)}],
        )
        result  = parse_json(resp.content[0].text)
        ts      = result.get("confirmed_timestamp_seconds", 0)
        if ts > duration_s:
            result["confirmed_timestamp_seconds"] = duration_s
            result["confidence"] = 0.0
        candidates.append(result)
        if len(chunks) > 1:
            ts = result["confirmed_timestamp_seconds"]
            print(f"→ {ts // 60}m{ts % 60:02d}s  conf {result['confidence']:.2f}")

    return _merge_phase2(candidates)


def run_phase1_scan(client: anthropic.Anthropic,
                    frames: list[str], prompt: str) -> dict:
    """Phase 1 coarse scan with chunking. Merges by taking narrowest region per boundary."""
    chunks  = _chunks(frames, CHUNK_SIZE)
    results = []

    for i, chunk in enumerate(chunks):
        if len(chunks) > 1:
            c_from = frame_to_seconds(chunk[0])
            c_to   = frame_to_seconds(chunk[-1])
            print(f"    P1 chunk {i+1}/{len(chunks)}  "
                  f"({c_from // 60}m–{c_to // 60}m)...", end=" ", flush=True)
        resp = client.messages.create(
            model=MODEL, max_tokens=1024,
            messages=[{"role": "user", "content": build_content(chunk, prompt)}],
        )
        r = parse_json(resp.content[0].text)
        results.append(r)
        if len(chunks) > 1:
            print("done")

    if len(results) == 1:
        return results[0]

    # Merge: narrowest region per boundary
    merged: dict = {}
    for key in BOUNDARY_KEYS:
        candidates = []
        for r in results:
            region = r.get("approximate_regions", {}).get(key)
            if region:
                w = abs(region_str_to_seconds(region["search_to"])
                        - region_str_to_seconds(region["search_from"]))
                candidates.append((w, region))
        if candidates:
            merged[key] = min(candidates, key=lambda x: x[0])[1]
    notes = " | ".join(r.get("notes", "") for r in results if r.get("notes"))
    return {"coarse_scan_complete": True, "approximate_regions": merged, "notes": notes}


# ── Boundary validation ────────────────────────────────────────────────────────

def _boundary_stats(confirmed: dict) -> dict:
    ko  = confirmed["ko_1h"]["seconds"]
    ht  = confirmed["ht_whistle"]["seconds"]
    k2  = confirmed["ko_2h"]["seconds"]
    ft  = confirmed["ft_whistle"]["seconds"]
    return {
        "ko_1h_s":     ko,
        "first_half":  ht - ko,
        "ht_break":    k2 - ht,
        "second_half": ft - k2,
        "total":       ft - ko,
    }


def _check_rules(stats: dict) -> list[tuple]:
    """Return list of (rule_id, boundary_key, direction, message) failures."""
    failures = []
    ko  = stats["ko_1h_s"]
    fh  = stats["first_half"]
    htb = stats["ht_break"]
    sh  = stats["second_half"]
    tot = stats["total"]

    if ko <= MIN_KO1H_S:
        failures.append(("ko_1h_too_early", "ko_1h", "later",
                         f"KO 1H at {ko}s (≤ {MIN_KO1H_S}s) — pre-match frame, not kickoff"))
    if fh < MIN_HALF_S:
        failures.append(("first_half_too_short", "ht_whistle", "later",
                         f"1H only {fh}s ({fh // 60}m) — minimum is {MIN_HALF_S // 60}m"))
    if fh > MAX_HALF_S:
        failures.append(("first_half_too_long", "ht_whistle", "earlier",
                         f"1H is {fh}s ({fh // 60}m) — maximum is {MAX_HALF_S // 60}m"))
    if htb < MIN_HT_BREAK:
        failures.append(("ht_break_too_short", "ko_2h", "later",
                         f"HT break only {htb}s ({htb // 60}m) — minimum is {MIN_HT_BREAK // 60}m"))
    if htb > MAX_HT_BREAK:
        failures.append(("ht_break_too_long", "ko_2h", "earlier",
                         f"HT break is {htb}s ({htb // 60}m) — maximum is {MAX_HT_BREAK // 60}m"))
    if sh < MIN_HALF_S:
        failures.append(("second_half_too_short", "ft_whistle", "later",
                         f"2H only {sh}s ({sh // 60}m) — minimum is {MIN_HALF_S // 60}m"))
    if sh > MAX_HALF_S:
        failures.append(("second_half_too_long", "ft_whistle", "earlier",
                         f"2H is {sh}s ({sh // 60}m) — maximum is {MAX_HALF_S // 60}m"))
    if tot < MIN_TOTAL_S or tot > MAX_TOTAL_S:
        direction = "earlier" if tot > MAX_TOTAL_S else "later"
        failures.append(("total_out_of_range", "ft_whistle", direction,
                         f"Total match {tot}s ({tot // 60}m) — expected 85–130 min"))
    return failures


def validate_and_retry(
    confirmed: dict,
    duration_s: int,
    frames_dir: str,
    client: anthropic.Anthropic,
) -> tuple[dict, bool]:
    """Validate boundary constraints; retry failed boundaries once.

    Returns (confirmed, validation_failed).
    """
    stats    = _boundary_stats(confirmed)
    failures = _check_rules(stats)

    if not failures:
        print("  Validation passed.")
        return confirmed, False

    print(f"\n  {'─'*53}")
    print(f"  Validation — {len(failures)} rule(s) failed:")
    for rule, boundary, direction, msg in failures:
        print(f"    [!] {rule}: {msg}")

    # Retry each failed boundary once
    retried = set()
    for rule, boundary, direction, msg in failures:
        if boundary in retried:
            continue
        retried.add(boundary)

        current_s = confirmed[boundary]["seconds"]
        if direction == "later":
            retry_from = max(MIN_KO1H_S if boundary == "ko_1h" else 0,
                             current_s + RETRY_WINDOW // 2)
            retry_to   = min(duration_s, current_s + RETRY_WINDOW + RETRY_WINDOW // 2)
        else:  # "earlier"
            retry_from = max(0, current_s - RETRY_WINDOW - RETRY_WINDOW // 2)
            retry_to   = max(0, current_s - RETRY_WINDOW // 2)

        print(
            f"\n  Retrying {boundary} ({direction}) in "
            f"{retry_from // 60}m{retry_from % 60:02d}s → "
            f"{retry_to // 60}m{retry_to % 60:02d}s..."
        )
        retry_frames = get_frames_in_range(frames_dir, retry_from, retry_to, INTERVAL_P2)
        if not retry_frames:
            print(f"    [!] No frames found in retry window — skipping.")
            continue

        p2r = run_phase2_scan(
            client, retry_frames, boundary,
            f"{retry_from // 60:02d}m{retry_from % 60:02d}s",
            f"{retry_to // 60:02d}m{retry_to % 60:02d}s",
            duration_s,
        )
        ts2, conf2 = p2r["confirmed_timestamp_seconds"], p2r["confidence"]
        confirmed[boundary] = {
            "frame":      p2r.get("confirmed_frame",
                                  f"frame_{ts2 // 60:02d}m{ts2 % 60:02d}s.jpg"),
            "seconds":    ts2,
            "confidence": conf2,
            "notes":      p2r.get("notes", ""),
        }
        print(f"    → {ts2 // 60}m{ts2 % 60:02d}s  conf {conf2:.2f}")

    # Re-validate
    stats2    = _boundary_stats(confirmed)
    failures2 = _check_rules(stats2)

    if not failures2:
        print("  Validation passed after retry.")
        return confirmed, False

    # Still failing — warn and flag
    print(f"\n  {'═'*53}")
    print(f"  BOUNDARY VALIDATION FAILED — manual override required.")
    for rule, boundary, _, msg in failures2:
        best = confirmed[boundary]["seconds"]
        print(f"  {boundary}: {msg}")
        print(f"    Best guess: {best // 60}m{best % 60:02d}s  "
              f"(conf {confirmed[boundary]['confidence']:.2f})")
    print(f"  Run with --override to accept current values.")
    print(f"  {'═'*53}")
    return confirmed, True


# ── Main ───────────────────────────────────────────────────────────────────────

def detect_boundaries(match_dir: str, override: bool = False) -> dict:
    # Locate frames directory
    frames_dir = os.path.join(match_dir, "frames", "frames_detailed")
    if not os.path.exists(frames_dir):
        frames_dir = os.path.join(match_dir, "frames")
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found in {match_dir}")

    # Load match config
    cfg: dict = {}
    match_info = "Unknown match"
    for p in (os.path.join(match_dir, "match_config.json"),
              os.path.join(match_dir, "match_config_draft.json")):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                cfg = json.load(f)
            match_info = (
                f"{cfg.get('match', 'Unknown')} | "
                f"{cfg.get('home_kit', '?')} vs {cfg.get('away_kit', '?')} | "
                f"FT {cfg.get('ft_score', '?')}"
            )
            break

    # Build sorted frame list
    import glob as _glob
    all_frames = sorted(
        _glob.glob(os.path.join(frames_dir, "frame_*.jpg")),
        key=lambda p: frame_to_seconds(p),
    )
    if not all_frames:
        raise FileNotFoundError(f"No frames found in {frames_dir}")

    total_frames = len(all_frames)
    duration_s   = frame_to_seconds(all_frames[-1])

    print(f"\n{'─'*55}")
    print(f"  Step 1b — Boundary detection (two-phase + validation)")
    print(f"{'─'*55}")
    print(f"  Total frames: {total_frames} ({duration_s // 60}m{duration_s % 60:02d}s)")

    # ── Manual override: skip detection entirely ────────────────────────────
    ov = cfg.get("boundaries_override", {})
    _ov_keys = ("ko_1h_seconds", "ht_whistle_seconds", "ko_2h_seconds", "ft_whistle_seconds")
    if ov and all(ov.get(k) is not None for k in _ov_keys):
        print(f"  Boundaries loaded from match_config.json override — skipping detection.")

        def _fmt(s: int) -> str:
            m, sec = divmod(s, 60)
            return f"frame_{m:02d}m{sec:02d}s.jpg"

        ko_1h = int(ov["ko_1h_seconds"])
        ht    = int(ov["ht_whistle_seconds"])
        ko_2h = int(ov["ko_2h_seconds"])
        ft    = int(ov["ft_whistle_seconds"])

        result = {
            "match":                  cfg.get("match", "Unknown"),
            "video_duration_seconds": duration_s,
            "boundaries": {
                "ko_1h":      {"frame": _fmt(ko_1h), "seconds": ko_1h, "confidence": 1.0, "notes": "manual override"},
                "ht_whistle": {"frame": _fmt(ht),    "seconds": ht,    "confidence": 1.0, "notes": "manual override"},
                "ko_2h":      {"frame": _fmt(ko_2h), "seconds": ko_2h, "confidence": 1.0, "notes": "manual override"},
                "ft_whistle": {"frame": _fmt(ft),    "seconds": ft,    "confidence": 1.0, "notes": "manual override"},
            },
            "live_play_seconds": {
                "first_half":  max(0, ht - ko_1h),
                "second_half": max(0, ft - ko_2h),
                "total":       max(0, ht - ko_1h) + max(0, ft - ko_2h),
            },
            "dead_time_seconds": {
                "pre_match":  ko_1h,
                "halftime":   max(0, ko_2h - ht),
                "post_match": max(0, duration_s - ft),
            },
            "notes":             "All boundaries set via match_config.json boundaries_override.",
            "validation_failed": False,
        }
        out_path = os.path.join(match_dir, "match_boundaries.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(stamp_schema_version(result, "match_boundaries"), f, indent=2)
        print(f"  KO 1H:      {ko_1h // 60}m{ko_1h % 60:02d}s")
        print(f"  HT whistle: {ht // 60}m{ht % 60:02d}s")
        print(f"  KO 2H:      {ko_2h // 60}m{ko_2h % 60:02d}s")
        print(f"  FT whistle: {ft // 60}m{ft % 60:02d}s")
        print(f"  1H play: {(ht - ko_1h) // 60}m | HT: {(ko_2h - ht) // 60}m | 2H play: {(ft - ko_2h) // 60}m")
        print(f"  Written: match_boundaries.json")
        return result

    if override:
        print(f"  [override mode — validation failures will be accepted]")

    client = anthropic.Anthropic()

    # ── Phase 1: coarse scan ────────────────────────────────────────────────
    p1_interval = max(INTERVAL_P1, math.ceil(duration_s / CHUNK_SIZE))
    p1_frames   = get_frames_in_range(frames_dir, 0, duration_s, interval_s=p1_interval)
    print(f"\n  Phase 1 — coarse scan: {len(p1_frames)} frames (every {p1_interval}s)")

    p1 = run_phase1_scan(client, p1_frames, PHASE1_PROMPT.format(match_info=match_info))

    print("  Approximate regions:")
    for key in BOUNDARY_KEYS:
        r = p1["approximate_regions"].get(key, {})
        print(f"    {key}: {r.get('search_from','?')} → {r.get('search_to','?')}")
    if p1.get("notes"):
        print(f"  Notes: {p1['notes']}")

    # ── Phase 2: fine scan per boundary ─────────────────────────────────────
    confirmed: dict = {}

    for key in BOUNDARY_KEYS:
        region = p1["approximate_regions"].get(key, {})
        from_s = max(0, region_str_to_seconds(region.get("search_from", "0m0s")))
        to_s   = min(duration_s,
                     region_str_to_seconds(
                         region.get("search_to", f"{duration_s // 60}m{duration_s % 60}s")
                     ))

        # KO 1H: never start before 30s — frame_00m00s is always pre-match
        if key == "ko_1h":
            from_s = max(MIN_KO1H_S, from_s)

        # Ensure at least 2-minute window
        if to_s - from_s < 120:
            mid    = (from_s + to_s) // 2
            from_s = max(MIN_KO1H_S if key == "ko_1h" else 0, mid - 60)
            to_s   = min(duration_s, mid + 60)

        p2_frames = get_frames_in_range(frames_dir, from_s, to_s, interval_s=INTERVAL_P2)
        print(
            f"\n  Phase 2 — {key}: {len(p2_frames)} frames "
            f"({from_s // 60}m{from_s % 60:02d}s → {to_s // 60}m{to_s % 60:02d}s, every {INTERVAL_P2}s)"
        )

        p2 = run_phase2_scan(
            client, p2_frames, key,
            search_from_str=region.get("search_from", f"{from_s // 60:02d}m{from_s % 60:02d}s"),
            search_to_str=region.get("search_to",   f"{to_s // 60:02d}m{to_s % 60:02d}s"),
            duration_s=duration_s,
        )

        ts   = p2["confirmed_timestamp_seconds"]
        conf = p2["confidence"]
        confirmed[key] = {
            "frame":      p2.get("confirmed_frame", f"frame_{ts // 60:02d}m{ts % 60:02d}s.jpg"),
            "seconds":    ts,
            "confidence": conf,
            "notes":      p2.get("notes", ""),
        }
        print(f"    → {ts // 60}m{ts % 60:02d}s  conf {conf:.2f}  {p2.get('notes','')}")

    # ── Validation + retry ───────────────────────────────────────────────────
    print(f"\n  {'─'*53}")
    print(f"  Validating boundaries...")
    confirmed, validation_failed = validate_and_retry(confirmed, duration_s, frames_dir, client)

    if validation_failed and not override:
        print(f"  Re-run with --override to write match_boundaries.json anyway.")
        sys.exit(1)

    # Validate ordering
    for prev, curr in zip(BOUNDARY_KEYS, BOUNDARY_KEYS[1:]):
        if confirmed[prev]["seconds"] >= confirmed[curr]["seconds"]:
            print(f"  [!] Ordering violation: {prev} ({confirmed[prev]['seconds']}s) "
                  f">= {curr} ({confirmed[curr]['seconds']}s)")

    ko_1h = confirmed["ko_1h"]["seconds"]
    ht    = confirmed["ht_whistle"]["seconds"]
    ko_2h = confirmed["ko_2h"]["seconds"]
    ft    = confirmed["ft_whistle"]["seconds"]

    result = {
        "match":                  cfg.get("match", "Unknown"),
        "video_duration_seconds": duration_s,
        "boundaries": {
            "ko_1h":      confirmed["ko_1h"],
            "ht_whistle": confirmed["ht_whistle"],
            "ko_2h":      confirmed["ko_2h"],
            "ft_whistle": confirmed["ft_whistle"],
        },
        "live_play_seconds": {
            "first_half":  max(0, ht - ko_1h),
            "second_half": max(0, ft - ko_2h),
            "total":       max(0, ht - ko_1h) + max(0, ft - ko_2h),
        },
        "dead_time_seconds": {
            "pre_match":  ko_1h,
            "halftime":   max(0, ko_2h - ht),
            "post_match": max(0, duration_s - ft),
        },
        "notes":             p1.get("notes", ""),
        "validation_failed": validation_failed,
    }

    out_path = os.path.join(match_dir, "match_boundaries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(result, "match_boundaries"), f, indent=2)

    print(f"\n{'─'*55}")
    print(f"  KO 1H:      {ko_1h // 60}m{ko_1h % 60:02d}s  (conf {confirmed['ko_1h']['confidence']:.2f})")
    print(f"  HT whistle: {ht // 60}m{ht % 60:02d}s  (conf {confirmed['ht_whistle']['confidence']:.2f})")
    print(f"  KO 2H:      {ko_2h // 60}m{ko_2h % 60:02d}s  (conf {confirmed['ko_2h']['confidence']:.2f})")
    print(f"  FT whistle: {ft // 60}m{ft % 60:02d}s  (conf {confirmed['ft_whistle']['confidence']:.2f})")
    print(f"  1H play: {(ht - ko_1h) // 60}m | HT: {(ko_2h - ht) // 60}m | 2H play: {(ft - ko_2h) // 60}m")
    if validation_failed:
        print(f"  [!] Written with validation_failed=true — manual review recommended.")
    else:
        print(f"  Written: match_boundaries.json")
    return result


if __name__ == "__main__":
    args     = [a for a in sys.argv[1:] if not a.startswith("--")]
    override = "--override" in sys.argv

    if not args:
        print("Usage: python detect_boundaries.py [MATCH_DIR] [--override]")
        sys.exit(1)

    match_dir = args[0]
    if not os.path.isdir(match_dir):
        print(f"[X] Directory not found: {match_dir}")
        sys.exit(1)
    try:
        detect_boundaries(match_dir, override=override)
    except Exception as e:
        print(f"[X] Boundary detection failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
