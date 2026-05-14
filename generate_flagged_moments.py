"""
Generate flagged_moments.md for a Match Lens match directory.

Usage:
    python generate_flagged_moments.py <match_dir>

Reads: match_config.json, running_summary.json, match_boundaries.json
       (optional), source_profile.json (optional)
Writes: <match_dir>/flagged_moments.md

Video timestamps are computed from match_boundaries.json when available;
otherwise the heading uses match-clock minutes only and omits the
"Video: ~MMmSSs" field.
"""

import json
import os
import re
import sys

# SCRIPTS_DIR self-resolves so the script runs from any cwd / any machine.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Exemplar style template — used to seed the LLM with the report's format.
# Lives outside the repo (see .gitignore) since real match exemplars contain
# real team/player data. Override the directory via env var MATCH_LENS_EXEMPLAR_DIR.
EXEMPLAR_DIR = os.environ.get(
    "MATCH_LENS_EXEMPLAR_DIR",
    os.path.join(SCRIPTS_DIR, "exemplars"),
)
EXEMPLAR = os.path.join(EXEMPLAR_DIR, "flagged_moments.md")

sys.path.insert(0, SCRIPTS_DIR)
from pipeline_accessors import get_source_limitations_note
from synthesis_agent import (
    _call_synthesis,
    build_input_bundle,
    _summarise_bundle,
    build_roster_block,
)


# ── Timestamp helpers ─────────────────────────────────────────────────────

def load_boundaries(match_dir: str):
    """Return (ko_1h, ht, ko_2h) in seconds, or None if file absent."""
    path = os.path.join(match_dir, "match_boundaries.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        b = json.load(f)["boundaries"]
    return (
        b["ko_1h"]["seconds"],
        b["ht_whistle"]["seconds"],
        b["ko_2h"]["seconds"],
    )


def video_seconds(match_minute: int, ko_1h: int, ko_2h: int) -> int:
    if match_minute <= 45:
        return ko_1h + match_minute * 60
    return ko_2h + (match_minute - 45) * 60


def fmt_mmss(seconds: int) -> str:
    m, s = divmod(seconds, 60)
    return f"{m:02d}m{s:02d}s"


def patch_timestamps(text: str, ko_1h: int, ko_2h: int) -> tuple[str, int]:
    """Rewrite heading video timestamps using boundaries. Returns (patched_text, count)."""
    blocks  = re.split(r"\n---\n", text)
    patched = []
    count   = 0

    heading_rx = re.compile(
        r"^(##\s+\d+\.\s+.*?\|\s+(\d+)'\s+\|\s+Video:\s+~?)([0-9]+m[0-9]+s)",
        re.MULTILINE,
    )
    frame_rx = re.compile(
        r"(\*\*Supporting frame:\*\*\s+frame_)([0-9]+m[0-9]+s)(\.jpg)"
    )

    for block in blocks:
        m = heading_rx.search(block)
        if not m:
            patched.append(block)
            continue

        match_minute = int(m.group(2))
        old_video    = m.group(3)
        new_seconds  = video_seconds(match_minute, ko_1h, ko_2h)
        new_video    = fmt_mmss(new_seconds)

        if new_video == old_video:
            patched.append(block)
            continue

        new_block = heading_rx.sub(lambda mm: f"{mm.group(1)}{new_video}", block, count=1)
        new_block = frame_rx.sub(
            lambda mm: f"{mm.group(1)}{new_video}{mm.group(3)}", new_block, count=1
        )
        patched.append(new_block)
        count += 1

    return "\n---\n".join(patched), count


# ── Main ──────────────────────────────────────────────────────────────────

def generate(match_dir: str) -> None:
    bundle = build_input_bundle(match_dir)
    mc     = bundle["match_config"]
    home   = mc.get("home_team", "Home")
    away   = mc.get("away_team", "Away")

    boundaries = load_boundaries(match_dir)
    has_boundaries = boundaries is not None
    if has_boundaries:
        ko_1h, ht, ko_2h = boundaries
        boundary_block = (
            f"Match boundaries available — use for video timestamp computation:\n"
            f"  ko_1h_s = {ko_1h}   (first half kick-off in video seconds)\n"
            f"  ht_s    = {ht}      (half-time whistle)\n"
            f"  ko_2h_s = {ko_2h}   (second half kick-off)\n\n"
            f"Convert match minute M to video seconds:\n"
            f"  M <= 45: video_s = {ko_1h} + M * 60\n"
            f"  M >  45: video_s = {ko_2h} + (M - 45) * 60\n"
            f"Format as MMmSSs. Use '~' prefix on heading for ET minutes."
        )
        video_rule = (
            "Include '| Video: ~MMmSSs' in every heading using the formula above."
        )
    else:
        boundary_block = (
            "Match boundaries NOT available — video timestamps cannot be computed.\n"
            "Omit the 'Video: ~MMmSSs' part from headings. Use match-clock minute only.\n"
            "Heading format becomes: ## N. [Title] | M'"
        )
        video_rule = (
            "Omit the '| Video: ~MMmSSs' field from all headings. "
            "Match-clock minute only. Supporting frame line: use frame_M_prime.jpg as placeholder."
        )

    sp_path = os.path.join(match_dir, "source_profile.json")
    source_type = "broadcast_fixed_wide"
    source_note = ""
    if os.path.exists(sp_path):
        with open(sp_path, encoding="utf-8") as f:
            sp = json.load(f)
        source_type = sp.get("source_type", source_type)
        source_note = get_source_limitations_note(sp)

    # Warn-and-continue if exemplar is missing — losing the style template
    # degrades report quality but should not crash the pipeline.
    if os.path.exists(EXEMPLAR):
        with open(EXEMPLAR, encoding="utf-8") as f:
            exemplar = f.read()
    else:
        print(f"  [WARN] Exemplar not found at {EXEMPLAR}")
        print(f"  [WARN] Set MATCH_LENS_EXEMPLAR_DIR or place flagged_moments.md "
              f"in {EXEMPLAR_DIR}")
        print(f"  [WARN] Proceeding without exemplar — report style may suffer")
        exemplar = ""

    roster   = build_roster_block(mc)
    data_str = _summarise_bundle(bundle)

    system_prompt = f"""
You are compiling the flagged_moments.md chronological report for a football
match analysed by the Match Lens pipeline.

EVIDENCE RULES:
- Goals, cards, and substitutions are CONFIRMED FACTS from match_config.
- Tactical patterns and set pieces from running_summary.key_moments and
  running_summary.set_pieces — include only high or medium confidence.
- Deduplicate by timestamp ± 30 seconds. Sort chronologically by match minute.
- Target 10-20 moments total.

VIDEO TIMESTAMP RULE:
{video_rule}
{boundary_block}

WINDOW LANGUAGE RULE: Never reference "windows", "observation windows",
"X/Y windows", or pipeline-internal time-slicing. Use match-clock language.

NO SUGGESTIONS RULE: Describe observations only. Never suggest tactics,
give recommendations, or instruct coaching staff. PROHIBITED:
  "Recommendation:", "teams should", "should press",
  "opposition teams should", "a ball in behind would be effective",
  imperative verbs directed at the reader.

ROSTER RULE: Every player name must appear in match_config for the correct
team. STARTER vs SUBSTITUTE is authoritative from the roster block.

FORMAT: Match the Gorleston exemplar exactly. Every moment must have:
  - Heading line: ## N. [Type] — [Player/title] | M' | Video: ~MMmSSs
    (or ## N. [Type] — [Player/title] | M' if no boundaries)
  - **Type:** ...
  - **Team:** ...
  - **Confidence:** High / Medium / Suggestive (downgraded — <family>)
  - Narrative paragraph (past tense, factual, no opinion/recommendation)
  - **Supporting frame:** frame_MMmSSs.jpg (or frame_M_prime.jpg if no boundaries)

Close with:
*Flagged moments report generated by Match Lens. Source: {source_type}. \\
Timestamps are approximate to ±1 second.*

Output as markdown. No preamble before the first heading.
"""

    user_msg = f"""
{roster}

Match: {home} vs {away}
Source: {source_type}
Source note: {source_note or "Near-ball visibility. Sequences represent observed chains of play."}

EXEMPLAR (match this format exactly):

{exemplar}

COMPLETE PIPELINE DATA:

{data_str}

Produce flagged_moments.md for {home} vs {away}.
Target 10-20 chronological moments. Include all confirmed goals and cards.
Include 2-3 tactically meaningful substitutions. Include the most notable
set pieces and tactical patterns from running_summary (high/medium only).
Apply all language rules. No preamble before the first heading.
"""

    print(f"  Calling synthesis API — flagged_moments.md for {home} vs {away}...",
          flush=True)
    result = _call_synthesis(system_prompt, user_msg, max_tokens=8000)

    # Patch video timestamps if boundaries are available
    if has_boundaries:
        result, patched = patch_timestamps(result, ko_1h, ko_2h)
        if patched:
            print(f"  {patched} video timestamp(s) re-anchored from boundaries")

    out_path = os.path.join(match_dir, "flagged_moments.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  Written: {out_path} ({len(result):,} chars)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_flagged_moments.py <match_dir>")
        sys.exit(1)
    generate(sys.argv[1])
