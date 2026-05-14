"""
frame_preprocessor.py -- Match Lens Python frame pre-processing pipeline

Runs BEFORE any API calls. Takes raw 1fps frames and produces:
  1. A filtered frame list per window (cost reduction)
  2. Frame metadata labels per frame (pre-labelling for agents)
  3. Targeted event flags (GK possession, set pieces, press moments)

Requires: Pillow (PIL). OpenCV optional but recommended for blur detection.

Usage:
    from frame_preprocessor import preprocess_window
    result = preprocess_window(match_dir, window_id, kit_config)
"""

import os, json
from pathlib import Path
from typing import Optional

try:
    from PIL import Image, ImageStat
    import numpy as np
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


# ── Kit colour configuration ──────────────────────────────────────────────────

def build_kit_config(home_kit: str, away_kit: str,
                     home_gk_kit: str, away_gk_kit: str) -> dict:
    """
    Map kit colour names to HSV ranges for PIL-based detection.
    Extend this table as you encounter new kit colours.
    """
    # HSV ranges: (H_min, H_max, S_min, V_min) in PIL HSV space (0-255 each)
    KIT_COLOURS = {
        "red":          [(0, 15, 100, 80), (240, 255, 100, 80)],  # wraps
        "red_white":    [(0, 15, 80, 120), (240, 255, 80, 120)],
        "blue":         [(140, 180, 80, 60)],
        "dark_blue":    [(150, 175, 80, 40)],
        "pink":         [(220, 245, 70, 120)],
        "yellow":       [(35, 60, 100, 120)],
        "yellow_green": [(40, 70, 80, 100)],
        "green":        [(70, 100, 80, 60)],
        "black":        [(0, 255, 0, 0)],    # any hue, low saturation/value
        "white":        [(0, 255, 0, 200)],
        "orange":       [(10, 30, 120, 100)],
        "sky_blue":     [(125, 150, 60, 120)],
        "purple":       [(180, 210, 60, 60)],
    }
    return {
        "home":    KIT_COLOURS.get(home_kit.lower().replace(" ", "_"), []),
        "away":    KIT_COLOURS.get(away_kit.lower().replace(" ", "_"), []),
        "home_gk": KIT_COLOURS.get(home_gk_kit.lower().replace(" ", "_"), []),
        "away_gk": KIT_COLOURS.get(away_gk_kit.lower().replace(" ", "_"), []),
    }


# ── Individual frame filters ───────────────────────────────────────────────────

def green_coverage(img: "Image.Image") -> float:
    """Fraction of pixels that are pitch-green."""
    if not HAS_PIL:
        return 0.5  # assume OK if PIL not available
    arr  = np.array(img.convert("RGB")).astype(float)
    r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]
    # Green: G channel dominant, moderate saturation
    is_green = (g > r * 1.1) & (g > b * 1.1) & (g > 60) & (g < 230)
    return float(is_green.sum()) / is_green.size


def blur_score(img: "Image.Image") -> float:
    """Laplacian variance — higher = sharper. Below 50 = blurry."""
    if not HAS_CV2:
        return 100.0  # assume OK if cv2 not available
    gray = cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def frame_similarity(img1: "Image.Image", img2: "Image.Image") -> float:
    """Histogram correlation 0-1. Above 0.97 = near-duplicate."""
    if not HAS_PIL:
        return 0.0
    h1 = img1.convert("RGB").histogram()
    h2 = img2.convert("RGB").histogram()
    n  = len(h1)
    mean1 = sum(h1) / n
    mean2 = sum(h2) / n
    num   = sum((a - mean1) * (b - mean2) for a, b in zip(h1, h2))
    den   = (sum((a - mean1)**2 for a in h1) *
             sum((b - mean2)**2 for b in h2)) ** 0.5
    return num / den if den > 0 else 0.0


def kit_colour_presence(img: "Image.Image", hsv_ranges: list) -> float:
    """Fraction of pixels matching a kit colour's HSV ranges."""
    if not HAS_PIL or not hsv_ranges:
        return 0.0
    hsv  = np.array(img.convert("RGB"))
    r, g, b = hsv[:,:,0]/255, hsv[:,:,1]/255, hsv[:,:,2]/255
    # Approximate HSV conversion
    mx   = np.maximum(np.maximum(r, g), b)
    mn   = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h    = np.zeros_like(mx)
    mask = diff > 0
    # H calculation
    rmax = (mx == r) & mask
    gmax = (mx == g) & mask
    bmax = (mx == b) & mask
    h[rmax] = (60 * ((g[rmax] - b[rmax]) / diff[rmax]) % 360)
    h[gmax] = (60 * ((b[gmax] - r[gmax]) / diff[gmax]) + 120)
    h[bmax] = (60 * ((r[bmax] - g[bmax]) / diff[bmax]) + 240)
    s = np.where(mx > 0, diff / mx, 0)
    v = mx

    # Convert ranges from 0-255 to 0-1 / 0-360
    total_match = np.zeros(mx.shape, dtype=bool)
    for (h_min, h_max, s_min, v_min) in hsv_ranges:
        h_min_n, h_max_n = h_min * 1.41, h_max * 1.41  # 255->360 scale
        s_min_n, v_min_n = s_min / 255, v_min / 255
        if h_min < h_max:
            match = (h >= h_min_n) & (h <= h_max_n) & (s >= s_min_n) & (v >= v_min_n)
        else:  # wraps (e.g. red)
            match = ((h >= h_min_n) | (h <= h_max_n)) & (s >= s_min_n) & (v >= v_min_n)
        total_match |= match

    return float(total_match.sum()) / total_match.size


def estimate_possession(img: "Image.Image", kit_config: dict) -> str:
    """Rough possession estimate from kit colour dominance."""
    home_pct = kit_colour_presence(img, kit_config.get("home", []))
    away_pct = kit_colour_presence(img, kit_config.get("away", []))
    if home_pct < 0.01 and away_pct < 0.01:
        return "unclear"
    ratio = home_pct / (home_pct + away_pct + 0.001)
    if ratio > 0.65:
        return "home_kit"
    if ratio < 0.35:
        return "away_kit"
    return "contested"


def lateral_zone_estimate(img: "Image.Image") -> str:
    """
    Rough ball-carrier zone estimate from horizontal position of dominant action.
    Divides frame into thirds: left_channel / central / right_channel.
    Uses local colour activity as a proxy for ball position.
    """
    if not HAS_PIL:
        return "unknown"
    arr  = np.array(img.convert("L"))  # greyscale
    w    = arr.shape[1]
    # Activity (variance) in each horizontal third
    left_var   = float(arr[:, :w//3].std())
    centre_var = float(arr[:, w//3:2*w//3].std())
    right_var  = float(arr[:, 2*w//3:].std())
    max_var    = max(left_var, centre_var, right_var)
    if max_var < 10:
        return "unknown"
    if left_var == max_var and left_var > centre_var * 1.2:
        return "left_channel"
    if right_var == max_var and right_var > centre_var * 1.2:
        return "right_channel"
    return "central"


def gk_possession_likely(img: "Image.Image", kit_config: dict,
                          attack_direction: str = "right") -> bool:
    """
    Check if a GK kit colour is dominant near the goal area.
    attack_direction: which end the focus team attacks toward.
    """
    if not HAS_PIL:
        return False
    arr = np.array(img.convert("RGB"))
    h, w = arr.shape[:2]

    # Sample goal area region (bottom or top third depending on attack direction)
    if attack_direction == "right":
        goal_region = Image.fromarray(arr[h//3:, :w//4])     # left goal
    else:
        goal_region = Image.fromarray(arr[h//3:, 3*w//4:])   # right goal

    for kit_name in ["home_gk", "away_gk"]:
        ranges = kit_config.get(kit_name, [])
        if kit_colour_presence(goal_region, ranges) > 0.03:
            return True
    return False


def inter_frame_motion(img1: "Image.Image", img2: "Image.Image") -> float:
    """Mean absolute pixel difference between frames. Near 0 = dead ball."""
    if not HAS_PIL:
        return 10.0
    a1 = np.array(img1.convert("L")).astype(float)
    a2 = np.array(img2.convert("L")).astype(float)
    return float(np.abs(a1 - a2).mean())


# ── Window preprocessor ───────────────────────────────────────────────────────

def preprocess_window(match_dir: str, window_id: str,
                      kit_config: dict,
                      attack_direction: str = "right") -> dict:
    """
    Run all filters on a window's frames.

    Returns:
        filtered_frames:     list of frame paths passing all quality filters
        metadata:            per-frame labels {possession, zone, phase_context, ...}
        flagged_events:      list of {timestamp, event_type} for targeting
        stats:               summary of what was removed and why
    """
    frames_dir = Path(match_dir) / "frames" / window_id
    if not frames_dir.exists():
        frames_dir = Path(match_dir) / "frames"

    all_frames = sorted([
        str(p) for p in frames_dir.glob("*.jpg")
    ] + [str(p) for p in frames_dir.glob("*.png")])

    if not all_frames or not HAS_PIL:
        return {
            "filtered_frames": all_frames,
            "metadata":        {},
            "flagged_events":  [],
            "stats":           {"total": len(all_frames), "removed": 0, "reason": "PIL not available"},
        }

    filtered    = []
    metadata    = {}
    flagged     = []
    stats       = {"total": len(all_frames), "dead_ball": 0, "duplicate": 0,
                   "low_coverage": 0, "blurry": 0, "passed": 0}

    prev_img    = None
    prev_poss   = None

    for i, fpath in enumerate(all_frames):
        fname = os.path.basename(fpath)
        try:
            img = Image.open(fpath, encoding="utf-8").convert("RGB")
        except Exception:
            stats["blurry"] += 1
            continue

        meta = {
            "frame": fname,
            "green_coverage": None,
            "blur_score":     None,
            "possession":     "unclear",
            "zone":           "unknown",
            "phase_context":  "unknown",
            "gk_possession":  False,
            "possession_changed": False,
            "motion":         None,
        }

        # ── Filter 1: low pitch coverage ───────────────────────────────────────
        cov = green_coverage(img)
        meta["green_coverage"] = round(cov, 3)
        if cov < 0.22:
            stats["low_coverage"] += 1
            continue

        # ── Filter 2: dead ball (near-zero inter-frame motion) ─────────────────
        if prev_img is not None:
            motion = inter_frame_motion(prev_img, img)
            meta["motion"] = round(motion, 2)
            if motion < 2.5:
                stats["dead_ball"] += 1
                prev_img = img
                continue
        else:
            meta["motion"] = 99.9

        # ── Filter 3: duplicate frame ──────────────────────────────────────────
        if prev_img is not None:
            sim = frame_similarity(prev_img, img)
            if sim > 0.974:
                stats["duplicate"] += 1
                continue

        # ── Filter 4: camera blur ──────────────────────────────────────────────
        bl = blur_score(img)
        meta["blur_score"] = round(bl, 1)
        if bl < 35:
            stats["blurry"] += 1
            prev_img = img
            continue

        # ── Frame passes all filters — generate metadata ───────────────────────
        poss = estimate_possession(img, kit_config)
        zone = lateral_zone_estimate(img)
        gk   = gk_possession_likely(img, kit_config, attack_direction)

        meta["possession"]     = poss
        meta["zone"]           = zone
        meta["gk_possession"]  = gk

        # Possession change detection
        if prev_poss and prev_poss not in ("unclear", "contested") and \
           poss not in ("unclear", "contested") and poss != prev_poss:
            meta["possession_changed"] = True
            flagged.append({
                "frame": fname, "event_type": "possession_change",
                "from": prev_poss, "to": poss,
            })

        # GK possession flag
        if gk:
            flagged.append({"frame": fname, "event_type": "gk_in_possession"})

        # Phase context assignment (simplified — combine signals)
        if meta["motion"] and meta["motion"] < 5:
            meta["phase_context"] = "dead_ball"
        elif meta["possession_changed"]:
            meta["phase_context"] = "transition"
        elif gk:
            meta["phase_context"] = "transition"
        else:
            meta["phase_context"] = "settled"

        filtered.append(fpath)
        metadata[fname] = meta
        stats["passed"] += 1

        prev_img  = img
        prev_poss = poss if poss not in ("unclear", "contested") else prev_poss

    stats["removed"] = stats["total"] - stats["passed"]
    stats["reduction_pct"] = round(stats["removed"] / max(stats["total"], 1) * 100, 1)

    return {
        "filtered_frames": filtered,
        "metadata":        metadata,
        "flagged_events":  flagged,
        "stats":           stats,
    }


def build_agent_frame_context(metadata: dict) -> str:
    """
    Build a compact context block from frame metadata to prepend to agent prompts.
    Reduces agent token use by pre-answering possession, zone, and phase questions.
    """
    if not metadata:
        return ""

    poss_counts = {}
    zone_counts = {}
    gk_frames   = []
    transitions  = []

    for fname, m in metadata.items():
        poss = m.get("possession", "unclear")
        zone = m.get("zone", "unknown")
        poss_counts[poss] = poss_counts.get(poss, 0) + 1
        zone_counts[zone] = zone_counts.get(zone, 0) + 1
        if m.get("gk_possession"):
            gk_frames.append(fname)
        if m.get("possession_changed"):
            transitions.append(fname)

    total = max(sum(poss_counts.values()), 1)
    dominant_poss = max(poss_counts, key=poss_counts.get) if poss_counts else "unclear"
    dominant_zone = max(zone_counts, key=zone_counts.get) if zone_counts else "unknown"

    lines = [
        "=== FRAME METADATA (pre-computed by Python preprocessor) ===",
        f"Frames analysed: {total} (after dead ball, duplicate, and quality filtering)",
        f"Dominant possession: {dominant_poss} ({poss_counts.get(dominant_poss, 0)}/{total} frames)",
        f"Dominant ball zone: {dominant_zone}",
        f"Possession changes detected: {len(transitions)} (frames: {', '.join(transitions[:5])}{'...' if len(transitions)>5 else ''})",
        f"GK possession frames: {len(gk_frames)} (frames: {', '.join(gk_frames[:3])}{'...' if len(gk_frames)>3 else ''})",
        "",
        "Use these labels to confirm or override. Pre-computed labels are estimates",
        "based on kit colour detection — correct them if the frame shows otherwise.",
        "=== END FRAME METADATA ===",
    ]
    return "\n".join(lines)


def save_metadata(match_dir: str, window_id: str, result: dict):
    """Write preprocessor output to disk for audit and reuse."""
    out_dir = Path(match_dir) / "frame_metadata"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{window_id}_metadata.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "window_id":      window_id,
            "stats":          result["stats"],
            "flagged_events": result["flagged_events"],
            "frame_count":    len(result["filtered_frames"]),
            "metadata":       result["metadata"],
        }, f, indent=2)
    return str(out_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 4:
        print("Usage: python frame_preprocessor.py [MATCH_DIR] [WINDOW_ID] [HOME_KIT] [AWAY_KIT] [HOME_GK_KIT] [AWAY_GK_KIT]")
        print("Example: python frame_preprocessor.py . 1H_00-00_05-00 red pink yellow yellow_green")
        sys.exit(1)

    match_dir  = sys.argv[1]
    window_id  = sys.argv[2]
    home_kit   = sys.argv[3] if len(sys.argv) > 3 else "red"
    away_kit   = sys.argv[4] if len(sys.argv) > 4 else "blue"
    home_gk    = sys.argv[5] if len(sys.argv) > 5 else "yellow"
    away_gk    = sys.argv[6] if len(sys.argv) > 6 else "yellow_green"

    kit_config = build_kit_config(home_kit, away_kit, home_gk, away_gk)
    result     = preprocess_window(match_dir, window_id, kit_config)

    print(f"Window: {window_id}")
    print(f"Total frames: {result['stats']['total']}")
    print(f"Passed:        {result['stats']['passed']} ({100-result['stats']['reduction_pct']:.0f}%)")
    print(f"Removed:       {result['stats']['removed']} ({result['stats']['reduction_pct']:.0f}%)")
    print(f"  Dead ball:   {result['stats']['dead_ball']}")
    print(f"  Duplicate:   {result['stats']['duplicate']}")
    print(f"  Low coverage:{result['stats']['low_coverage']}")
    print(f"  Blurry:      {result['stats']['blurry']}")
    print(f"Flagged events: {len(result['flagged_events'])}")
    for ev in result["flagged_events"][:5]:
        print(f"  {ev['event_type']} @ {ev['frame']}")

    path = save_metadata(match_dir, window_id, result)
    print(f"\nMetadata saved: {path}")
    print("\nAgent context block:")
    print(build_agent_frame_context(result["metadata"]))
