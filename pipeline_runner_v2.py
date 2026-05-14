"""
pipeline_runner_v2.py — Match Lens pipeline with full interruption resilience

Before running any match:
    python cost_estimator.py [MATCH_DIR]   # check cost first
    python pipeline_runner_v2.py [MATCH_DIR] --quality standard

If interrupted at any point:
    python pipeline_runner_v2.py [MATCH_DIR] --resume

Progress:
    python pipeline_state.py [MATCH_DIR]
"""

import glob, json, os, sys, time, argparse
from pathlib import Path
from dotenv import load_dotenv
from pipeline_accessors import (
    get_window_id,
    get_formation_home,
    get_formation_away,
)
from pipeline_paths import find_agent_output, find_merged_window

# Fix 42: bump on every fix. Surfaced into the report manifest (Section 5).
MATCH_LENS_VERSION = "0.42"

# Force UTF-8 stdout/stderr on Windows (default cp1252 breaks non-ASCII print output)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Load .env — use absolute resolved path with override=True (Fix 5)
_loaded_env = False
for _env_path in [Path(__file__).resolve().parent/'.env',
                  Path(__file__).resolve().parent.parent/'.env',
                  Path.home()/'.env']:
    if _env_path.exists():
        load_dotenv(_env_path, override=True)
        _loaded_env = True
        import anthropic as _a_test
        _key = _a_test.Anthropic().api_key or ""
        print(f"  [ENV] Loaded .env from {_env_path} (key ...{_key[-4:] if _key else 'NOT FOUND'})")
        break
if not _loaded_env:
    print("  [WARN] No .env file found. Ensure ANTHROPIC_API_KEY is set in environment.")


def _kits(mc: dict) -> dict:
    """
    Normalise kit colour fields. match_config.json may use:
      flat keys:   mc["home_kit"], mc["away_kit"]
      nested keys: mc["kits"]["home"], mc["kits"]["away"]
    Returns dict with home, away, home_gk, away_gk keys. (Fix 2)
    """
    if "kits" in mc and isinstance(mc["kits"], dict):
        k = mc["kits"]
        return {
            "home":    k.get("home",    mc.get("home_kit",    "unknown")),
            "away":    k.get("away",    mc.get("away_kit",    "unknown")),
            "home_gk": k.get("home_gk", mc.get("home_gk_kit", "unknown")),
            "away_gk": k.get("away_gk", mc.get("away_gk_kit", "unknown")),
        }
    return {
        "home":    mc.get("home_kit",    "unknown"),
        "away":    mc.get("away_kit",    "unknown"),
        "home_gk": mc.get("home_gk_kit", "unknown"),
        "away_gk": mc.get("away_gk_kit", "unknown"),
    }

from pipeline_state  import (init_state, load_state, mark_window, mark_step,
                               is_window_done, is_step_done, pending_windows,
                               failed_windows, print_progress, PIPELINE_STEPS)
from batch_runner    import submit_batch, poll_batch, collect_results, with_retry
# Fix 42: surface the API model string into pipeline state and the report manifest.
from batch_runner    import MODEL as API_MODEL
from cost_estimator  import load_match_data, calculate_cost, print_estimate


# ── Frame resize helper ───────────────────────────────────────────────────────

def _load_frame_as_base64(frame_path: str,
                           resize_w: int = None,
                           resize_h: int = None) -> str:
    """
    Load a frame JPEG and return base64-encoded bytes.
    If resize_w/resize_h are given, resize in-memory before encoding.
    Original file is never modified.
    Uses Pillow if available; falls back to raw file read if not.
    """
    import base64
    if resize_w and resize_h:
        try:
            from PIL import Image
            import io as _io
            with Image.open(frame_path) as img:
                img_resized = img.resize((resize_w, resize_h), Image.LANCZOS)
                buf = _io.BytesIO()
                img_resized.save(buf, format="JPEG", quality=85)
                return base64.b64encode(buf.getvalue()).decode("utf-8")
        except ImportError:
            pass  # Pillow not available — fall through to raw read
    with open(frame_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _prepare_frames(frame_paths: list,
                     resize_w: int = None,
                     resize_h: int = None) -> list:
    """
    Return frames ready for build_request.
    If resize dimensions are set, pre-encode as base64 image dicts so
    batch_runner passes them through unchanged (no double-encoding).
    Otherwise return raw file paths — batch_runner handles loading.
    """
    if not resize_w or not resize_h:
        return frame_paths
    return [
        {
            "type": "image",
            "source": {
                "type":       "base64",
                "media_type": "image/jpeg",
                "data":       _load_frame_as_base64(p, resize_w, resize_h),
            },
        }
        for p in frame_paths
    ]


# ── Frame sampling ────────────────────────────────────────────────────────────

QUALITY_PROFILES = {
    "economy":   {"frames_per_window": 10,  "event_frames": 10,  "resize_w": None, "resize_h": None},
    "standard":  {"frames_per_window": 30,  "event_frames": 30,  "resize_w": None, "resize_h": None},
    "full":      {"frames_per_window": 60,  "event_frames": 60,  "resize_w": None, "resize_h": None},
    "full_1fps": {"frames_per_window": 300, "event_frames": 300, "resize_w": 512,  "resize_h": 288},
}
# NOTE: Event frames capped to frames_per_window to stay within API request size.
# The event agent uses a focused prompt — fewer targeted frames are more reliable
# than a large payload that triggers "Too much memory" rejections. (Fix 7)
# full_1fps: 300 frames at 512×288 ≈ 51,000 tokens/window (cheaper than standard).

def sample_frames(all_frames: list, n: int) -> list:
    """Select n evenly spaced frames from a list."""
    if len(all_frames) <= n:
        return all_frames
    step = max(1, len(all_frames) // n)
    return all_frames[::step][:n]

def _frame_seconds(fname: str) -> float:
    """Parse seconds from frame_MMmSSs.jpg filenames. Returns -1 if unparseable."""
    import re
    m = re.search(r'frame_([0-9]+)m([0-9]+)s', fname)
    if m:
        return int(m.group(1)) * 60 + int(m.group(2))
    # Also handle frame_NNNN.jpg (sequential number) — pass through
    return -1

def get_window_frames(match_dir: str, window: dict, n: int) -> list:
    """
    Get frame file paths for a window, sampled to n frames.
    Supports both subdirectory layout (frames/01/) and flat layout (frames/).
    In flat layout, filters by window start_s / end_s using filename timestamp.
    """
    window_id  = get_window_id(window)
    start_s    = window.get("start_s", window.get("start_seconds", 0))
    end_s      = window.get("end_s",   window.get("end_seconds",   start_s + 300))

    # Try per-window subdirectory first
    subdir = os.path.join(match_dir, "frames", window_id)
    if os.path.exists(subdir):
        all_frames = sorted([
            os.path.join(subdir, f) for f in os.listdir(subdir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        return sample_frames(all_frames, n)

    # Flat directory — filter by timestamp in filename
    flat_dir = os.path.join(match_dir, "frames")
    if not os.path.exists(flat_dir):
        return []

    all_files = sorted([
        f for f in os.listdir(flat_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])

    filtered = []
    for fname in all_files:
        secs = _frame_seconds(fname)
        if secs < 0:
            # Can't parse timestamp — include everything (fallback)
            filtered.append(os.path.join(flat_dir, fname))
        elif start_s <= secs < end_s:
            filtered.append(os.path.join(flat_dir, fname))

    if not filtered:
        print(f"  [WARN] No frames found for window {window_id} "
              f"({start_s:.0f}s-{end_s:.0f}s) in {flat_dir}")
    return sample_frames(filtered, n)


# ── Prompt builders ───────────────────────────────────────────────────────────

def load_skill_section(section_name: str) -> str:
    """Load a prompt section from SKILL.md."""
    skill_path = os.path.join(os.path.dirname(__file__), "..", "SKILL.md")
    if not os.path.exists(skill_path):
        return f"[{section_name} prompt -- SKILL.md not found]"
    with open(skill_path, encoding="utf-8") as f:
        content = f.read()
    # Extract section between ### {section_name} and the next ###
    start = content.find(f"### {section_name}")
    if start == -1:
        return f"[{section_name} not found in SKILL.md]"
    end = content.find("\n### ", start + 1)
    return content[start:end] if end != -1 else content[start:]


def _load_source_profile(match_dir: str) -> dict:
    """
    Load source_profile.json from match directory.
    Returns veo_ball_tracking defaults if file not found.
    """
    path = os.path.join(match_dir, "source_profile.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    # Default to veo constraints if no profile present
    return {
        "source_type": "veo_ball_tracking",
        "full_pitch_visible": False,
        "far_side_observable": False,
        "camera": "ball_tracking",
    }


SOURCE_CAPABILITY_VEO = """
SOURCE TYPE: veo_ball_tracking
Camera: Ball-tracking. Camera follows the ball.

CAPABILITY CONSTRAINTS - these apply to every window:
- Far-side defensive shape: NOT OBSERVABLE. Do not log shape or spacing
  for players on the weak side. Flag as suppressed if asked.
- Lateral zone codes: UNRELIABLE. Use central_channel as default unless
  ball is clearly in a wide area. Do not assume halfspace occupation.
- Line height: Express as percentage of visible pitch depth only.
  Do not estimate in metres.
- Both fullbacks simultaneously: NOT VISIBLE when play is tight.
  Log only the near-side fullback when the other is out of frame.
- Shots under 1 second: NOT OBSERVABLE at 1fps. Log shot context only.
- Formation verification: PARTIAL. Near-ball side only. Only confirm
  the formation if both lines are visible in the frame. If the ball
  is wide and the camera is tight, mark formation as uncertain and
  carry forward the lineup formation as the default.
"""

SOURCE_CAPABILITY_BROADCAST = """
SOURCE TYPE: broadcast_fixed_wide
Camera: Fixed elevated wide-angle. Full pitch visible in most frames.

CAPABILITY CONSTRAINTS - these apply to every window:
- Far-side defensive shape: OBSERVABLE. Log shape and spacing for both
  sides of the pitch. Far-side fullback position should be logged.
- Lateral zone codes: RELIABLE AND EXPECTED. Use left_channel,
  right_channel, left_halfspace, right_halfspace, central_channel
  on every sequence. Do not default to central_channel unless play
  is genuinely central.
- Line height: Express in pitch-relative terms using the penalty area
  depth as reference. State as "high" (above halfway), "medium"
  (between penalty areas), or "low" (within own penalty area zone).
  Also log as percentage of visible pitch depth for metric compatibility.
- Both fullbacks simultaneously: VISIBLE. Log both fullback positions
  when relevant to shape or pressing observations.
- Defensive block width: OBSERVABLE across the full pitch width.
  Log whether the block defends the full width or is narrow centrally.
- Off-side line coordination: OBSERVABLE. Log whether the defensive
  line moves as a unit or has gaps.
- Shots under 1 second: Still not observable at 1fps. Event agent
  at 5fps handles this. Log shot context only in structural windows.
- Formation verification: RELIABLE from this source. The full pitch
  view allows both teams' shapes to be confirmed simultaneously.
  Verify the match_config lineup formation against the frames each
  window. Note if the team shifts to a different defensive shape.
"""

SOURCE_CAPABILITY_TACTICAL_WIDE = """
SOURCE TYPE: tactical_wide_static
Camera: Fixed elevated tactical camera, high stand position.

CAPABILITY CONSTRAINTS — these apply to every window:
- Far-side defensive shape: OBSERVABLE and REQUIRED. Both sides
  of the pitch are clearly visible. Log far_side_shape on every
  window using the BROADCAST_FARSIDE_SCHEMA format.
- Lateral zone codes: RELIABLE AND EXPECTED. Use left_channel,
  right_channel, left_halfspace, right_halfspace, central_channel
  on every sequence. Halfspace occupation is particularly readable
  from the overhead angle.
- Line height: MEASURABLE and RELIABLE. The elevated overhead
  angle makes defensive line depth the most readable of any source
  type. Express as high/mid/deep AND as a percentage of pitch depth.
  Both defensive lines are simultaneously visible.
- Both fullbacks simultaneously: VISIBLE on every window. Log
  both fullback positions every time.
- Defensive block width: HIGHLY OBSERVABLE. Log whether block is
  narrow, standard, or wide on every window.
- Off-side line coordination: CLEARLY OBSERVABLE. The overhead
  angle is the optimal view for this — note any coordinated
  off-side traps.
- Jersey numbers: NOT LEGIBLE at this camera distance and height.
  Identify players by position, kit colour, movement pattern,
  and build only. Do not attempt to read shirt numbers. Use
  positional identifiers (home_CB_left, away_RB etc.) rather
  than names where uncertain.
- Scoreboard overlay: NOT PRESENT. Do not reference a clock
  or score from an overlay. Match state comes from match_config
  only.
- Shots under 1 second: Not observable at 1fps. Event agent
  at 5fps handles this for goal windows.
- Formation verification: HIGH CONFIDENCE from this source.
  The elevated overhead angle is the optimal view for shape reads.
  Both teams' full structures are visible simultaneously. You should
  be able to clearly identify the double pivot in a 4-2-3-1,
  the flat three in a 4-3-3, and the wing-back positioning in a
  three-at-the-back system. Override the lineup formation if the
  frames show a different shape consistently.
"""


BROADCAST_FARSIDE_SCHEMA = """
FAR-SIDE SHAPE FIELD (broadcast footage only):
Because the full pitch is visible, populate this field in your JSON
as a TOP-LEVEL key alongside your other findings:

"far_side_shape": {
  "observable": true,
  "home_far_side": "<description of home team's weak-side shape>",
  "away_far_side": "<description of away team's weak-side shape>",
  "home_fullback_position": "high / mid / deep / not_visible",
  "away_fullback_position": "high / mid / deep / not_visible",
  "confidence": "high / medium / low"
}

Populate this field for every window where far-side players
are visible. If the camera angle limits visibility for a
specific player, set their field to "not_visible".
Do not populate this field for veo_ball_tracking source type.
"""


def _source_capability_block(source_profile: dict) -> str:
    source_type = source_profile.get("source_type", "veo_ball_tracking")
    if source_type == "broadcast_fixed_wide":
        return SOURCE_CAPABILITY_BROADCAST
    if source_type == "tactical_wide_static":
        return SOURCE_CAPABILITY_TACTICAL_WIDE
    return SOURCE_CAPABILITY_VEO


STRUCTURAL_OUTPUT_SCHEMA = """
=== REQUIRED OUTPUT SCHEMA ===
Your response MUST be valid JSON containing exactly these top-level fields.
Do not add extra top-level fields. Do not rename these fields.

{
  "timestamp_range": "MM:SS-MM:SS (match clock, not broadcast time)",
  "half": "1H / 2H / ET1 / ET2",
  "match_state": "level / home_winning / away_winning",
  "score_home": 0,
  "score_away": 0,

  "formation": {
    "home": "Formation string e.g. 4-2-3-1. Use the match_config lineup formation as your starting hypothesis. Confirm it if consistent with frames. Override it with evidence if the shape visibly differs.",
    "away": "Same — hypothesis from lineup, verify from frames.",
    "home_shape": "block_shape: compact / mid / high (describes vertical compactness of the team block, not the line height)",
    "away_shape": "block_shape: same",
    "home_variation": "null or describe IP/OOP shape difference e.g. 'defends as 4-5-1 — wingers track back'",
    "away_variation": "null or describe IP/OOP shape difference e.g. 'shape consistent in both phases'",
    "home_formation_basis": "confirmed_from_frames | overridden | hypothesis_unverified",
    "away_formation_basis": "confirmed_from_frames | overridden | hypothesis_unverified"
  },

  "defensive_line": {
    "home_height_pct": 0.0,
    "away_height_pct": 0.0,
    "home_descriptor": "high / mid / deep",
    "away_descriptor": "high / mid / deep"
  },

  "pressing": {
    "home_intensity": 0.0,
    "away_intensity": 0.0,
    "home_trigger": "description or null",
    "away_trigger": "description or null"
  },

  "pass_sequences": [
    {
      "team": "home_kit or away_kit",
      "start_zone": "zone code",
      "end_zone": "zone code",
      "passes": 0,
      "outcome": "retained / turnover / shot / clearance",
      "sequence_type": "build_up / transition / set_piece / recovery"
    }
  ],

  "gk_distribution": [
    {
      "team": "home_kit or away_kit",
      "target_zone": "zone code",
      "length": "short / medium / long",
      "outcome": "retained / contested / clearance"
    }
  ],

  "set_pieces": [
    {
      "timestamp":          "[MMmSSs]",
      "timestamp_inferred": false,
      "type":               "[corner_left / corner_right / direct_fk / indirect_fk / throw_final_third / kickoff]",
      "team":               "[home_kit or away_kit]",
      "taker_position":     "[kit colour #N, or null if unclear]",
      "delivery_observed":  true,
      "delivery_zone":      "[near_post / far_post / penalty_spot / edge_of_box / short]",
      "bodies_in_box":      0,
      "marking":            "[zonal / man / mixed / unclear]",
      "rest_defence":       0,
      "outcome":            "[goal / cleared_near_post / cleared_far_post / gk_claim / gk_punch / second_phase / lost_possession]",
      "delivery_type":      null,
      "runners":            null,
      "wall_size":          null,
      "wall_position":      null
    }
  ],

  "key_moments": [
    {
      "minute": "MM:SS",
      "type": "goal / card / substitution / tactical_shift",
      "team": "home_kit or away_kit",
      "description": "brief factual description"
    }
  ],

  "far_side_shape": {
    "observable": true,
    "home_fullback_position": "high / mid / deep / not_visible",
    "away_fullback_position": "high / mid / deep / not_visible",
    "home_far_side": "description",
    "away_far_side": "description",
    "confidence": "high / medium / low"
  },

  "confidence": 0.85,
  // confidence is on a 0.0-1.0 scale ONLY. Do NOT use a 0-10 scale.
  // 0.9 = near-certain, 0.7 = probable, 0.5 = uncertain, 0.3 = poorly observable.
  "source_limitations": "brief note on what was not visible"
}

Zone codes: left_channel, right_channel, left_halfspace,
right_halfspace, central_channel, defending_third,
middle_third, attacking_third.

Leave individual_observations OUT of this output -- the Player
Agent (Step 3b) handles those separately. If a field is not
observable from the frames, set it to null rather than omitting it.
"""


def _farside_schema_block(source_profile: dict) -> str:
    """Inject the far-side schema instruction only for broadcast footage."""
    if source_profile.get("source_type") == "broadcast_fixed_wide":
        return BROADCAST_FARSIDE_SCHEMA
    return ""


FORMATION_RECOGNITION_GUIDE = """
=== FORMATION RECOGNITION GUIDE ===

You have been given the lineup formations in match_config.
Treat these as hypotheses to verify, not as facts to repeat.

DISTINGUISHING CRITERIA (observable from elevated/wide camera):

4-4-2 FLAT:
  - Two banks of four at similar depth
  - No player clearly ahead of or behind the midfield line
  - Width from wide midfielders, not advanced forwards
  - Two strikers at similar depth

4-4-1-1:
  - Four defenders, four midfielders (same as 4-4-2)
  - ONE player clearly between the midfield and striker
  - ONE striker ahead of everyone
  - The #10 is visibly ahead of the midfield line but behind the striker

4-2-3-1:
  - TWO midfielders clearly deeper than the rest (double pivot)
  - THREE players between the pivot and the striker
  - The pivot pair sits noticeably behind the attacking three
  - ONE striker furthest forward
  KEY SIGNAL: visible gap between the two deep midfielders
  and the three ahead of them

4-3-3:
  - THREE midfielders at roughly similar depth (no clear pivot pair)
  - THREE forwards — two wide, one central
  - Wide forwards positioned high and wide
  - No player clearly between lines as a #10
  KEY SIGNAL: midfield line is flat (not split), forwards are wide

3-4-3 / 3-4-2-1:
  - THREE centre-backs visible — no fullbacks in the defensive line
  - TWO wing-backs very wide, higher than the CBs
  - Defensive line has three players, not two pairs

3-5-2:
  - THREE centre-backs
  - TWO wing-backs
  - THREE midfielders (single pivot or three flat)
  - TWO strikers

4-3-3 vs 4-2-3-1 quick test:
  Is there a visible gap between two deeper midfielders and
  three ahead of them? -> 4-2-3-1
  Is the midfield line flat with no obvious split? -> 4-3-3

4-4-2 vs 4-4-1-1 quick test:
  Are the two frontmost players at the same depth? -> 4-4-2
  Is one clearly ahead of the other with space between? -> 4-4-1-1

IF UNCERTAIN: State what you observe ("I see what appears to be
a double pivot with three ahead") and give your best formation
label with a confidence note. A qualified observation is more
valuable than a confident wrong answer.

IP/OOP OBSERVATION RULE:
The formation field records the PRIMARY shape (what the team
looks like in possession). The variation field records how the
shape changes out of possession.

Always observe both phases within a window if both are visible.
The most important IP/OOP variations to catch:

4-3-3 -> 4-5-1:  Wide forwards track back (most common)
4-2-3-1 -> 4-4-2: The #10 drops alongside the pivot
4-4-2 -> 4-4-2:  Shape consistent (many non-league teams)
3-4-3 -> 5-4-1:  Wing-backs drop into the back line
4-3-3 -> 4-3-3:  Shape consistent (high-press teams hold shape)

If only one phase is visible in this window, note which one
and set variation to "only [in/out of] possession phase observed".

FORMATION BASIS — set home_formation_basis and away_formation_basis:
  confirmed_from_frames  — you observed the shape clearly from frames
                           (whether it matches the lineup or not)
  overridden             — you observed a shape that clearly DIFFERS
                           from the lineup formation provided
  hypothesis_unverified  — frames don't give sufficient signal to
                           confirm or deny; you are defaulting to lineup
"""


SET_PIECE_GUIDE = """
=== SET PIECES ===
For every set piece (corner, free kick awarded in the attacking or middle third,
final-third throw-in, kickoff) log a set_piece entry.

TIMESTAMP is REQUIRED. Use the MMmSSs frame when the ball is struck or released
(frame of contact, not the foul or walk-up). If contact is obscured, use the
closest visible setup frame and set timestamp_inferred: true. NEVER omit timestamp
-- without it the downstream 5fps burst cannot run and the entry will be rejected.

TYPE values:
  corner_left / corner_right / direct_fk / indirect_fk /
  throw_final_third / kickoff

TEAM: home_kit or away_kit -- whichever team is taking the set piece.

TAKER: set taker_position to kit colour + shirt number if visible (e.g. "home_kit #7").
       Set null if you cannot clearly see who struck the ball. Do NOT guess.

READ AT 1fps (populate where visible):
  delivery_zone:  near_post / far_post / penalty_spot / edge_of_box / short
                  WHERE the ball was delivered to (from the frame after contact)
  bodies_in_box:  count of attacking-team players inside the 18-yard box,
                  from the frame just before delivery
  marking:        zonal / man / mixed / unclear
                  defensive setup from the frame before delivery
  rest_defence:   count of attacking-team players who stayed outside the 18-yard box
                  during their own delivery (transition cover indicator)
  outcome:        goal / cleared_near_post / cleared_far_post / gk_claim /
                  gk_punch / second_phase / lost_possession

DELIVERY_OBSERVED: Set to false if you cannot see the delivery itself.
  Still log the entry with the setup frame timestamp -- the 5fps burst will
  recover delivery detail.

SETUP DETECTION -- log a set_piece entry whenever you observe ANY of:
  - A player standing over the ball at a corner flag
  - A defensive wall forming (3+ players aligned, facing the kicker)
  - The referee signalling a free kick in the attacking or middle third
  - Players gathering inside the 18-yard box without an active sequence in progress
  - A throw-in being taken from the final third

DO NOT attempt to populate at 1fps (leave as null -- filled by 5fps burst):
  delivery_type, runners, wall_size, wall_position
"""


def _kit_identification_block(mc: dict) -> str:
    """
    Authoritative kit identification block.

    On broadcast footage under stadium lighting, white kits can read as
    cream / off-white / yellow depending on shadow and exposure, and the
    GK kit (often yellow at non-league) can bleed into outfield identity.
    This block tells the agent to trust match_config rather than pixel
    inference.
    """
    kits = _kits(mc)
    return f"""
=== KIT IDENTIFICATION (AUTHORITATIVE) ===
The following kit colours are CONFIRMED from match data.
Do NOT infer kit colours from frame pixel analysis.
Do NOT let stadium lighting affect your team identification.
The GK kit is different from outfield kit - do not confuse them.

HOME TEAM: {mc.get('home_team', 'Home')}
Home outfield kit: {kits.get('home', 'unknown')}

AWAY TEAM: {mc.get('away_team', 'Away')}
Away outfield kit: {kits.get('away', 'unknown')}

When identifying which team has the ball or which player you are
observing: use kit colour as a guide but treat the above as ground
truth. If a player appears in a colour that does not match either
outfield kit, they are likely a goalkeeper. Do not relabel a team
based on what you see - use the confirmed names above.
"""


def _match_state_at_window(mc: dict, window: dict) -> dict:
    """
    Calculate the match score and state at the START of a given window.
    Uses confirmed goals from match_config and window start_s timestamp.
    Returns dict with home_score, away_score, state_label.
    Fix 33a (B3): replaces unpopulated window['match_state']/score_home/score_away.
    """
    goals     = mc.get("goals", []) or []
    home_team = mc.get("home_team", "")
    # Fix 55b: ko_1h_s/ko_2h_s can be None on recordings without
    # scoreboard overlay (tactical_wide_static, some veo). Default to
    # 0 / 2700 so goal-time mapping degrades to "broadcast clock ≈ match
    # clock" rather than crashing on None+int. Score state becomes
    # approximate, which matches the recording's actual signal quality.
    ko_1h_s   = mc.get("ko_1h_s")
    ko_2h_s   = mc.get("ko_2h_s")
    if ko_1h_s is None:
        ko_1h_s = 0
    if ko_2h_s is None:
        ko_2h_s = ko_1h_s + 2700

    win_start_s = window.get("start_s", window.get("start_seconds", 0))

    home_score = 0
    away_score = 0

    for goal in goals:
        # Convert goal minute to broadcast seconds
        t = goal.get("time", {})
        elapsed = t.get("elapsed", goal.get("minute", 0)) if isinstance(t, dict) \
                  else goal.get("minute", 0)

        # Map match minute to broadcast second.
        # Fix 36: extended for extra-time matches. Fall back to 2H mapping
        # if ET timing is missing so legacy non-ET matches behave identically.
        ko_et1_s = mc.get("ko_et1_s")
        ko_et2_s = mc.get("ko_et2_s")
        if elapsed <= 45:
            goal_broadcast_s = ko_1h_s + elapsed * 60
        elif elapsed <= 90:
            goal_broadcast_s = ko_2h_s + (elapsed - 45) * 60
        elif elapsed <= 105 and ko_et1_s:
            goal_broadcast_s = ko_et1_s + (elapsed - 90) * 60
        elif ko_et2_s:
            goal_broadcast_s = ko_et2_s + (elapsed - 105) * 60
        else:
            # No ET timing available -- treat as late 2H (preserves legacy behaviour)
            goal_broadcast_s = ko_2h_s + (elapsed - 45) * 60

        # Only count goals that occurred BEFORE this window started
        if goal_broadcast_s < win_start_s:
            team = goal.get("team", {})
            team_name = team.get("name", str(team)) if isinstance(team, dict) \
                        else str(team)
            if team_name == home_team:
                home_score += 1
            else:
                away_score += 1

    # Fix 33b: use neutral home/away framing instead of focus/opp.
    if home_score > away_score:
        state = "home_winning"
    elif home_score < away_score:
        state = "away_winning"
    else:
        state = "level"

    return {
        "home_score": home_score,
        "away_score": away_score,
        "state":      state,
        "label":      f"{home_score}-{away_score} ({state})",
    }


def build_structural_prompt(match_dir: str, window: dict,
                              mc: dict, state: dict,
                              blind_formation: bool = False) -> str:
    """Build the Step 3a structural agent prompt for a window.

    blind_formation: if True, strips the 'formation' key from every lineup
    entry before injection — used by --blind-formation to test whether agents
    observe formations independently or echo the lineup hypothesis.
    """
    source_profile = _load_source_profile(match_dir)
    source_type    = source_profile.get("source_type", "veo_ball_tracking")
    source_block   = _source_capability_block(source_profile)
    kit_block      = _kit_identification_block(mc)
    farside_block  = _farside_schema_block(source_profile)

    # Only inject Veo zone defaults for ball-tracking footage.
    # Broadcast footage uses SOURCE_CAPABILITY_BROADCAST for zone guidance,
    # which is more specific (lateral zones expected on every sequence) and
    # would directly contradict the central-default rules below.
    if source_type == "broadcast_fixed_wide":
        zone_encoding_block = ""
    else:
        zone_encoding_block = """
=== ZONE ENCODING ===
DEFAULT when ball is within ~20m of a touchline: left_channel or right_channel.
Do NOT use defending_third/middle near touchlines.
Central play only: defending_third / middle / attacking_third.
"""

    kits        = _kits(mc)

    # --blind-formation: remove 'formation' from each lineup entry so the
    # agent must derive the shape purely from frames (diagnostic mode).
    lineups = mc.get("lineups", [])
    if blind_formation:
        import copy as _copy
        lineups = _copy.deepcopy(lineups)
        for lineup in lineups:
            lineup.pop("formation", None)

    match_context = json.dumps({
        "match":       mc.get("match"),
        "home_team":   mc.get("home_team"),
        "away_team":   mc.get("away_team"),
        "home_kit":    kits["home"],
        "away_kit":    kits["away"],
        "home_gk_kit": kits["home_gk"],
        "away_gk_kit": kits["away_gk"],
        "score":       mc.get("ft_score"),
        "lineups":     lineups,
    }, indent=2)

    # Fix 33a B3: derive match state from confirmed goals + window start_s.
    ms          = _match_state_at_window(mc, window)
    score_h     = ms["home_score"]
    score_a     = ms["away_score"]
    match_state = ms["state"]
    window_id   = get_window_id(window)

    # Fix 33b: use explicit home/away naming. Reports cover both teams
    # independently; there is no single "focus" perspective at prompt time.
    home_team   = mc.get("home_team", "")
    away_team   = mc.get("away_team", "")

    # Fix 43E: select attack direction from the window's half. Previously
    # always used attack_direction_1h, so every 2H/ET window prompt carried
    # the wrong orientation. Teams swap ends only at HT — ET1 inherits 2H's
    # direction; ET2 swaps back to the 1H direction.
    _half = window.get("half", "1H")
    if _half in ("2H", "ET1"):
        _attack_dir = mc.get("attack_direction_2h",
                             mc.get("attack_direction_1h", "unknown"))
    elif _half == "ET2":
        _attack_dir = mc.get("attack_direction_1h", "unknown")
    else:
        _attack_dir = mc.get("attack_direction_1h", "unknown")

    return f"""You are a football tactical analyst. Review the frames below and produce a structured JSON output ONLY. No prose. No preamble. No markdown fences.

=== MATCH CONTEXT ===
{match_context}

HOME TEAM: {home_team}
AWAY TEAM: {away_team}
ATTACK DIR: {_attack_dir}
WINDOW:     {window_id}
HALF:       {window.get('half', '?')}

MATCH STATE: {score_h}-{score_a} ({match_state})
Do not use this to assume intent or judge performance.
{kit_block}
=== SOURCE CONTEXT ===
{source_block}
{farside_block}
=== SCOUTING PRIMER ===
Focus players from teamsheet: {json.dumps([
    p.get('player',{}).get('name','?') + ' #' + str(p.get('player',{}).get('number','?'))
    for lineup in mc.get('lineups',[]) for p in lineup.get('startXI',[])
][:11])}

=== GK DISTRIBUTION -- LOG FOR BOTH TEAMS ===
Every GK kick (goal kick, from hands, back pass played out) -> add to gk_kicks[].
Log for BOTH teams.

=== BOTH TEAMS SEQUENCES ===
Log pass sequences for BOTH teams.
Tag each: "team": "home_kit" or "team": "away_kit"
{zone_encoding_block}
{FORMATION_RECOGNITION_GUIDE}
{SET_PIECE_GUIDE}
{STRUCTURAL_OUTPUT_SCHEMA}
"""


def build_player_prompt(match_dir: str, window: dict, mc: dict,
                         structural_context=None) -> str:
    """
    Build the Step 3b player agent prompt.

    Fix 33a (A+B2): the 4th arg is now the 3a structural output dict for THIS
    window (not a free-text prior summary). Replaces the SKILL.md-mandated
    STRUCTURAL CONTEXT block that was previously absent.
    """
    if structural_context is None:
        structural_context = {}

    source_profile = _load_source_profile(match_dir)
    source_block   = _source_capability_block(source_profile)
    kit_block      = _kit_identification_block(mc)
    window_id      = get_window_id(window)

    # Fix 33b: explicit home/away naming. No single "focus" perspective.
    _home_p     = mc.get("home_team", "")
    _away_p     = mc.get("away_team", "")

    # Fix 33a A: compose STRUCTURAL CONTEXT block from 3a output for this window.
    formation = structural_context.get("formation", {}) or {}
    pressing  = structural_context.get("pressing", {}) or {}
    def_line  = structural_context.get("defensive_line", {}) or {}
    ms        = _match_state_at_window(mc, window)

    structural_block = f"""=== STRUCTURAL CONTEXT (from 3a output for this window) ===
Formation:         home={formation.get('home','?')} / away={formation.get('away','?')}
Home shape:        {formation.get('home_shape','?')}
Away shape:        {formation.get('away_shape','?')}
Home line height:  {def_line.get('home_descriptor','?')}
Away line height:  {def_line.get('away_descriptor','?')}
Home pressing:     {pressing.get('home_intensity','?')}/10
Away pressing:     {pressing.get('away_intensity','?')}/10
Match state:       {ms['label']}
"""

    # Fix 47: build a roster lookup string so the player agent uses
    # exact roster names rather than ad-hoc strings like "Cray Wanderers #9"
    # or "Wingate striker #9". The accumulator joins individual_observations[]
    # by name, and the synthesis PLAYER TENDENCY / EXPLOITABLE PATTERN rules
    # require named players. Pre-Fix-47, 86 of 88 Wingate observation strings
    # missed the join.
    home_lineup_lines = []
    away_lineup_lines = []
    for lineup in mc.get("lineups", []):
        side = lineup.get("team_side", "")
        for p in lineup.get("startXI", []) + lineup.get("substitutes", []):
            player = p.get("player", p)
            name   = player.get("name", "")
            number = player.get("number", "?")
            pos    = player.get("pos", "")
            if not name:
                continue
            entry = f"  {name} (#{number}, {pos})" if pos else f"  {name} (#{number})"
            (home_lineup_lines if side == "home" else away_lineup_lines).append(entry)

    roster_block = f"""
=== PLAYER IDENTIFICATION ===
When naming a player in individual_observations[], use their EXACT
name from the roster below. Format: "First Last (#number)" — e.g.
"Charlie Stallard (#11)" not "Wingate striker #11", not
"Cray Wanderers #9", not "home_ST".

If the jersey number is not legible, infer the most likely player from
position, kit, and movement and write that exact roster name. Only fall
back to a positional identifier ("home_CB_left", "away_RB") if the
player is genuinely unidentifiable from any of: kit, position, build,
or relative location to teammates.

HOME TEAM — {_home_p}:
{chr(10).join(home_lineup_lines)}

AWAY TEAM — {_away_p}:
{chr(10).join(away_lineup_lines)}
"""

    # OCR: inject jersey number sightings for this window if available
    ocr_block = ""
    _ocr_path = os.path.join(match_dir, 'jersey_number_map.json')
    if os.path.exists(_ocr_path):
        try:
            with open(_ocr_path, encoding='utf-8') as _ocrf:
                _ocr_data = json.load(_ocrf)
            _frame_detail = _ocr_data.get('frame_detail', {})
            _start_f = window.get('start_frame', '')
            _end_f   = window.get('end_frame', '')
            _window_sightings = {}
            for _fname, _sightings in _frame_detail.items():
                if _start_f <= _fname <= _end_f:
                    for _s in _sightings:
                        if _s.get('name'):
                            _n = _s['name']
                            _window_sightings[_n] = _window_sightings.get(_n, 0) + 1
            if _window_sightings:
                ocr_block = "\n=== JERSEY NUMBERS CONFIRMED (OCR) ===\n"
                ocr_block += "These players were visually confirmed in this window:\n"
                for _name, _count in sorted(_window_sightings.items(),
                                             key=lambda x: -x[1]):
                    ocr_block += f"  {_name} — seen {_count} time(s)\n"
                ocr_block += ("Use exact names from the roster when you observe "
                               "these players.\n")
        except Exception:
            pass  # OCR data malformed — skip silently

    # Fix 51: canonical action vocabulary for run-type aggregation.
    action_vocab_block = """
=== ACTION VOCABULARY ===
When you populate the `action` field of an individual_observation,
use one of these canonical terms wherever it fits. The accumulator
uses these tokens to detect run-type and movement patterns across
the match. Free-text descriptions still go in `observation`.

MOVEMENT PATTERNS (pick the closest one for `action`):
  "run in behind"     — runs beyond the last defender into space
  "drop deep"         — drops into midfield or defensive third to receive
  "overlap"           — overlapping run beyond a wide teammate
  "underlap"          — inside run beyond a wide teammate, into the halfspace
  "channel run"       — diagonal run into the channel between CB and FB
  "late run into box" — arrives in the penalty area from deep
  "hold up play"      — receives with back to goal, holds and lays off
  "run at defender"   — carries the ball directly at an opponent

DEFENSIVE PATTERNS:
  "press"             — closes down the ball carrier
  "track back"        — defensive recovery run
  "screen"            — positions to block central passing lanes
  "aerial duel"       — challenges for a headed ball

If the action does not fit a canonical term, use a brief
descriptive phrase (e.g. "switches play", "crosses from wide").
But prefer the canonical terms above whenever they apply —
they enable pattern aggregation across the whole match.
"""

    return f"""You are a football player analyst. Review the same frames as the structural agent.
Your ONLY task is individual player observations, duels, and physical profiles.
Do NOT re-read formation, sequences, or pressing scores.

=== MATCH CONTEXT ===
HOME TEAM: {_home_p}
AWAY TEAM: {_away_p}
WINDOW:    {window_id}
{kit_block}
=== SOURCE CONTEXT ===
{source_block}

{structural_block}
{roster_block}{ocr_block}
{action_vocab_block}
=== MINIMUM REQUIREMENTS ===
- At least 5 individual_observations per window
- At least 2 for opposition players
- At least 1 duels[] entry per visible physical contest
- For every attacker in possession: also log 1 out_of_possession observation

Return JSON with: player_agent=true, individual_observations[], duels[]
Follow the Step 3b Player Agent schema from SKILL.md.
"""


def _ev_minute(ev: dict):
    """Extract event minute from either ev['minute'] or ev['time']['elapsed']. (Fix 3)"""
    if "minute" in ev:
        return ev["minute"]
    return ev.get("time", {}).get("elapsed", "?")

def _ev_player(ev: dict) -> str:
    """Safe player name extraction -- field may be str or dict. (Fix 3)"""
    p = ev.get("player", "?")
    if isinstance(p, dict): return p.get("name", "?")
    return str(p) if p else "?"

def _ev_team(ev: dict) -> str:
    """Safe team name extraction -- field may be str or dict. (Fix 3)"""
    t = ev.get("team", "?")
    if isinstance(t, dict): return t.get("name", "?")
    return str(t) if t else "?"

def _build_player_lookup(mc: dict) -> str:
    """Build a confirmed number→position lookup injected into event prompts."""
    lines = ["CONFIRMED PLAYER POSITIONS (use these in build-up chains):"]
    _kits_lu = _kits(mc)
    home     = mc.get("home_team", "")
    for lineup in mc.get("lineups", []):
        t = lineup.get("team", {})
        team_name = t.get("name", str(t)) if isinstance(t, dict) else str(t)
        colour = _kits_lu["home"] if team_name == home else _kits_lu["away"]
        lines.append(f"  {team_name} ({colour}):")
        for p in lineup.get("startXI", []):
            pd = p.get("player", p)
            if isinstance(pd, dict):
                num = pd.get("number", "?")
                name = pd.get("name", "?")
                pos = pd.get("pos", "?")
                lines.append(f"    #{num} {name} — {pos}")
    lines.append("Use #number and confirmed position in build-up chains, e.g. [red] #5 CB.")
    lines.append("A player appearing forward of their position is making a run — do not relabel them.")
    return "\n".join(lines)


def build_event_prompt(mc: dict, window: dict,
                        event: dict, structural_context: str) -> str:
    """Build the Step 3d-EV event agent prompt."""
    event_type = event.get("type", "unknown")
    minute     = _ev_minute(event)
    player     = _ev_player(event)
    team       = _ev_team(event)
    player_lookup = _build_player_lookup(mc)

    return f"""You are a football event analyst. A key event occurred in this window.
Your task is to answer specific questions. Do NOT re-run the full structural scan.

=== CONFIRMED PLAYER POSITIONS ===
{player_lookup}

=== STRUCTURAL CONTEXT (from Step 3a) ===
{structural_context}

=== EVENT ===
Type:   {event_type}
Minute: {minute}'
Player: {player}
Team:   {team}

{"=== GOAL QUESTIONS ===" if event_type == "goal" else "=== SUBSTITUTION QUESTIONS ==="}
{"1. Shot origin zone (use lateral codes: left_channel / right_channel / halfspace / central)" if event_type=="goal" else "1. Position taken by player coming on"}
{"2. Shot foot (right/left/header/other)" if event_type=="goal" else "2. Did the formation change?"}
{"3. Target zone (top_left/top_right/bottom_left/bottom_right/central_high/central_low)" if event_type=="goal" else "3. Did the defensive line height change?"}
{"4. Full build-up chain in 30s before shot using confirmed positions: [kit] #number pos →[F/S/B] ..." if event_type=="goal" else "4. Did pressing intensity change?"}
{"5. Defensive shape at moment of shot" if event_type=="goal" else ""}

Return JSON with: event_agent=true, events[], shape_vs_structural_agent{{agrees, disputes}}
Follow the Step 3d Event Agent schema from SKILL.md.
"""


def build_setpiece_prompt(match_dir: str, queue_item: dict,
                           mc: dict, set_piece_record: dict,
                           state: dict) -> str:
    """Build the Step 3d-SP set piece burst prompt.

    queue_item:        routed entry from confirmation_queue.json -- has
                       anchor timestamp, team, escalation_target_fps, window.
    set_piece_record:  the 1fps record from the merged window file for this
                       set piece -- provides the values for the agent to confirm.
    """
    source_profile = _load_source_profile(match_dir)
    source_block   = _source_capability_block(source_profile)
    kit_block      = _kit_identification_block(mc)
    kits           = _kits(mc)

    anchor_ts      = queue_item.get("timestamp", "")
    team           = queue_item.get("team", "")
    window_id      = queue_item.get("window", "")
    target_fps     = queue_item.get("escalation_target_fps", 5)
    sp_type        = set_piece_record.get("type", "")
    taker_pos      = set_piece_record.get("taker_position")

    # 1fps confirmation fields
    bodies_1fps         = set_piece_record.get("bodies_in_box")
    delivery_zone_1fps  = set_piece_record.get("delivery_zone")
    marking_1fps        = set_piece_record.get("marking") or set_piece_record.get("marking_system")
    outcome_1fps        = set_piece_record.get("outcome")

    home_team  = mc.get("home_team", "")
    away_team  = mc.get("away_team", "")
    team_name  = home_team if team == "home_kit" else away_team

    return f"""You are a set piece analyst. You are reviewing a {target_fps}fps burst of frames (±3 seconds) around a set piece that was logged at 1fps. Your job is to confirm or correct the fields 1fps could read, and to fill in the fields 1fps could not.

Output JSON ONLY. No prose. No preamble. No markdown fences.

=== MATCH CONTEXT ===
HOME: {home_team} (kit: {kits.get("home", "unknown")})
AWAY: {away_team} (kit: {kits.get("away", "unknown")})
{kit_block}
=== SOURCE CONTEXT ===
{source_block}

=== ANCHOR EVENT ===
Timestamp:      {anchor_ts}
Type:           {sp_type}
Team:           {team} ({team_name})
Taker position: {taker_pos if taker_pos else "null (not observed at 1fps)"}

=== 1FPS OBSERVATIONS TO CONFIRM ===
The 1fps pass recorded the following. Confirm each field or correct based on
what the {target_fps}fps frames show. For any field you correct, log the prior
value and a brief reason in the corrections array.

bodies_in_box:   {bodies_1fps}
delivery_zone:   {delivery_zone_1fps}
marking_system:  {marking_1fps}
outcome:         {outcome_1fps}

=== BURST-ONLY FIELDS ===
Fill these fields. 1fps could not read them.

delivery_type:        inswinger / outswinger / driven / lofted / flick_on / short_routine / null
                      (null only if delivery itself was not visible)
delivery_target_zone: near_post / penalty_spot / far_post / edge_of_box / short_corner / null
                      (where the delivery was AIMED -- may differ from where it ENDED UP)
runners:              For each attacking-team runner visible in the burst, log:
                        runner_id          -- shirt number if visible, else kit+position label
                        start_zone         -- edge_of_box / penalty_spot / near_post / far_post / outside_box
                        run_zone           -- where the runner ended at point of contact
                        run_type           -- near_post_run / far_post_run / back_post / front_post /
                                             penalty_spot_hold / blocking_run / second_phase_runner / decoy
                        defender_assigned  -- shirt number or position label, or null if unmarked
                      Three to five visible runners is typical for a corner. Do not invent runners.
wall_size:            number, for direct free kicks within shooting range only. null otherwise
wall_position:        near_post / central / far_post / null
second_phase:         object with:
                        winner           -- home_kit / away_kit / contested / null
                        first_touch_zone -- near_post / penalty_spot / far_post / edge_of_box / outside_box / null
                        follow_up_action -- shot / cross / lost_possession / clearance /
                                           new_set_piece_awarded / end_of_burst_window / null
                        notes            -- <=30 words free text

=== ANCHORING RULE ===
This burst is anchored to ONE set piece at {anchor_ts}. Frames may show events
before or after (lead-up, second phase, a recycled corner). Focus output on the
set piece at {anchor_ts}. Related events go in second_phase, NOT as new records.
The pipeline has already queued separate bursts for separate events if needed.

=== OUTPUT FORMAT ===
{{
  "set_piece_agent":     true,
  "anchor_timestamp":    "{anchor_ts}",
  "window":              "{window_id}",
  "team":                "{team}",
  "burst_resolved":      true,
  "burst_fps":           {target_fps},
  "confirmed_fields": {{
    "bodies_in_box":     [count],
    "delivery_zone":     "[zone]",
    "marking_system":    "[system]",
    "outcome":           "[outcome]"
  }},
  "corrections": [],
  "burst_fields": {{
    "delivery_type":         "[type or null]",
    "delivery_target_zone":  "[zone or null]",
    "runners":               [],
    "wall_size":             null,
    "wall_position":         null,
    "second_phase":          {{
      "winner":            null,
      "first_touch_zone":  null,
      "follow_up_action":  null,
      "notes":             ""
    }}
  }},
  "burst_notes": ""
}}
"""


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(match_dir: str, quality: str = "standard",
                 resume: bool = False, force_reports: bool = False):

    print(f"\n{'='*60}")
    print(f"  Match Lens Pipeline v2")
    print(f"  Match dir: {match_dir}")
    print(f"  Quality:   {quality}")
    print(f"{'='*60}\n")

    # Load match config
    mc_path = os.path.join(match_dir, "match_config.json")
    wp_path = os.path.join(match_dir, "window_plan.json")
    if not os.path.exists(mc_path):
        print("ERROR: match_config.json not found. Run build_readiness_check.py first.")
        sys.exit(1)
    if not os.path.exists(wp_path):
        print("ERROR: window_plan.json not found. Run window_plan.py first.")
        sys.exit(1)
    with open(mc_path, encoding="utf-8") as f: mc = json.load(f)
    with open(wp_path, encoding="utf-8")  as f: wp = json.load(f)

    windows     = wp.get("windows", [])
    # Cast to str so window_ids match JSON state keys (always strings).
    window_ids  = [str(get_window_id(w)) for w in windows]
    profile     = QUALITY_PROFILES[quality]
    fpw         = profile["frames_per_window"]
    evf         = profile["event_frames"]
    resize_w    = profile.get("resize_w")
    resize_h    = profile.get("resize_h")

    # ── Cost check ────────────────────────────────────────────────────────────
    match_data = load_match_data(match_dir)
    estimate   = calculate_cost(match_data, quality)
    print(f"  Estimated cost: ${estimate['total_cost_usd']:.2f} ({quality})")
    print(f"  API calls:      {estimate['api_calls_total']}")
    print(f"  Frames/window:  {fpw}")
    print(f"\n  Using Message Batches API (50% cheaper, async, resumable)")
    print(f"  Check balance: https://console.anthropic.com/settings/billing\n")

    # ── State ─────────────────────────────────────────────────────────────────
    if resume:
        state = load_state(match_dir)
        if not state:
            print("  No state file found. Starting fresh.")
            state = init_state(match_dir, mc.get("match","?"), window_ids, quality)
        else:
            print("  Resuming from checkpoint:")
            print_progress(state)
    else:
        state = init_state(match_dir, mc.get("match","?"), window_ids, quality)

    # Fix 41: clear stale errors so the run-summary "Errors: N" line at the
    # end of the run reflects only this session. state["errors"] is otherwise
    # append-only; without this, errors from past failed runs persist after
    # the underlying step has since succeeded — e.g. 9 stale 3k_metrics
    # crashes from pre-Fix-39 runs lingered in state and prompted the
    # unnecessary Fix 40 investigation.
    #
    # Keep only errors whose step is still in "failed" status (genuinely
    # unresolved). Drop entries for steps that are now "complete" (the error
    # has been resolved) or "pending" (the step is about to be retried this
    # session, will produce its own fresh error if it fails again).
    failed_steps_now = {
        s for s, status in state.get("steps", {}).items() if status == "failed"
    }
    for wsteps in state.get("windows", {}).values():
        for s, status in wsteps.items():
            if status == "failed":
                failed_steps_now.add(s)
    state["errors"] = [
        e for e in state.get("errors", [])
        if e.get("step") in failed_steps_now
    ]
    # Fix 42: stamp the API model and runner version into state so the report
    # manifest (Section 5) can read them back without re-deriving.
    state["api_model"]          = API_MODEL
    state["match_lens_version"] = MATCH_LENS_VERSION
    # Persist run-start mutations immediately. If no step actually runs this
    # session (e.g. fully-cached resume), no mark_step / mark_window call
    # would otherwise save them and the values would not stick.
    from pipeline_state import _save as _ps_save_errors
    _ps_save_errors(match_dir, state)

    # ── STEP 2b: Jersey number OCR ────────────────────────────────────────────
    if not is_step_done(state, "2b_jersey_ocr"):
        print("\n  STEP 2b: Jersey number OCR...")
        _ocr_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "jersey_ocr.py")
        if not os.path.exists(_ocr_script):
            print("  [WARN] jersey_ocr.py not found — skipping 2b")
            mark_step(match_dir, state, "2b_jersey_ocr", "skipped")
        else:
            import subprocess as _sub2b
            try:
                _result = _sub2b.run(
                    [sys.executable, _ocr_script, match_dir,
                     "--sample-rate", "30"],
                    capture_output=True, text=True, timeout=600
                )
                if _result.returncode == 0:
                    mark_step(match_dir, state, "2b_jersey_ocr", "complete")
                    if _result.stdout.strip():
                        print(_result.stdout.rstrip())
                    print("  [OK] 2b_jersey_ocr")
                else:
                    err = (_result.stderr.strip().splitlines()[-1]
                           if _result.stderr.strip() else "unknown error")
                    print(f"  [WARN] 2b_jersey_ocr failed: {err} "
                          f"(continuing without OCR data)")
                    mark_step(match_dir, state, "2b_jersey_ocr", "skipped")
            except Exception as _e2b:
                print(f"  [WARN] 2b_jersey_ocr error: {_e2b} (continuing)")
                mark_step(match_dir, state, "2b_jersey_ocr", "skipped")
    else:
        print("  STEP 2b: Jersey OCR — already complete")

    # ── PHASE 1: Step 3a Structural (batch) ───────────────────────────────────
    pending_3a = pending_windows(state, "3a")
    if pending_3a:
        print(f"\n  PHASE 1: Step 3a Structural ({len(pending_3a)} windows pending)")

        requests = []
        for wid in pending_3a:
            win = next((w for w in windows
                        if str(get_window_id(w)) == str(wid)), {})
            frames  = get_window_frames(match_dir, win, fpw)
            prompt  = build_structural_prompt(match_dir, win, mc, state,
                                              blind_formation=args.blind_formation)

            from batch_runner import build_request
            requests.append(build_request(wid, prompt,
                                          _prepare_frames(frames, resize_w, resize_h),
                                          step="3a"))

        batch_id = with_retry(lambda: submit_batch(match_dir, state, requests, "3a"))
        batch    = poll_batch(batch_id)
        collect_results(batch_id, match_dir, state, "3a")
    else:
        print("  PHASE 1: Step 3a — already complete (all windows)")

    # ── PHASE 2: Step 3b Player agent (batch) ─────────────────────────────────
    pending_3b = pending_windows(state, "3b")
    if pending_3b:
        print(f"\n  PHASE 2: Step 3b Player ({len(pending_3b)} windows pending)")

        requests = []
        for i, wid in enumerate(window_ids):
            if wid not in pending_3b:
                continue
            win = next((w for w in windows
                        if str(get_window_id(w)) == str(wid)), {})
            frames  = get_window_frames(match_dir, win, fpw)

            # Fix 33a A+B2 + F2: previously used exact path
            # agent_{wid}_structural.json which never matched (batch_runner
            # writes labeled filenames), so structural_context was always {}
            # and player prompts shipped with no structural seeding.
            structural_context = {}
            _structural_path = find_agent_output(
                os.path.join(match_dir, "agent_logs"), wid, "structural"
            )
            if _structural_path:
                try:
                    with open(_structural_path, encoding="utf-8") as _f:
                        structural_context = json.load(_f)
                except Exception:
                    pass

            prompt  = build_player_prompt(match_dir, win, mc, structural_context)

            from batch_runner import build_request
            requests.append(build_request(wid, prompt,
                                          _prepare_frames(frames, resize_w, resize_h),
                                          step="3b"))

        batch_id = with_retry(lambda: submit_batch(match_dir, state, requests, "3b"))
        batch    = poll_batch(batch_id)
        collect_results(batch_id, match_dir, state, "3b")
    else:
        print("  PHASE 2: Step 3b — already complete")

    # ── PHASE 3: Event windows at 5fps ────────────────────────────────────────
    # Tag each event with its type so build_event_prompt shows the right questions.
    # Goals and subs in match_config have no "type" field; without this, all events
    # default to type="unknown" and the prompt renders substitution questions for goals.
    events = []
    for _g in mc.get("goals", []):
        _g2 = dict(_g); _g2.setdefault("type", "goal"); events.append(_g2)
    for _s in mc.get("substitutions", []):
        _s2 = dict(_s); _s2.setdefault("type", "sub"); events.append(_s2)

    # Load KO boundaries so match minutes can be converted to video seconds.
    # Without this, match minute 45 lands on wrong video frames (off by ~KO offset).
    # Priority: match_boundaries.json -> window_plan.json -> match_config.json -> 0.
    # Falling through to 0 silently breaks event-window time matching, so any of
    # the three sources is preferred when available.
    _ko_1h_s = _ko_2h_s = _ht_s = 0
    _boundaries_path = os.path.join(match_dir, "match_boundaries.json")
    if os.path.exists(_boundaries_path):
        with open(_boundaries_path, encoding="utf-8") as _bf:
            _b = json.load(_bf)
        _ko_1h_s = _b["boundaries"]["ko_1h"]["seconds"]
        _ko_2h_s = _b["boundaries"]["ko_2h"]["seconds"]
        _ht_s    = _b["boundaries"]["ht_whistle"]["seconds"]
    else:
        # Fall back to window_plan / match_config (both carry these fields).
        _ko_1h_s = wp.get("ko_1h_s") or mc.get("ko_1h_s") or 0
        _ko_2h_s = wp.get("ko_2h_s") or mc.get("ko_2h_s") or 0
        _ht_s    = wp.get("ht_s")    or mc.get("ht_s")    or 0

    _first_half_live = (_ht_s - _ko_1h_s) if _ht_s > _ko_1h_s else 45 * 60

    # Fix 36: extra-time KO offsets so post-90' events map to the correct
    # broadcast second instead of being projected into a stretched 2H.
    _ko_et1_s = (mc.get("ko_et1_s") or wp.get("ko_et1_s")
                 or (_b.get("boundaries", {}).get("ko_et1", {}) or {}).get("seconds")
                 if os.path.exists(_boundaries_path) else None)
    _ko_et2_s = (mc.get("ko_et2_s") or wp.get("ko_et2_s")
                 or (_b.get("boundaries", {}).get("ko_et2", {}) or {}).get("seconds")
                 if os.path.exists(_boundaries_path) else None)

    def _ev_to_video_s(minute):
        if not isinstance(minute, (int, float)):
            return None
        # Fix 36: 4-branch mapping for 1H / 2H / ET1 / ET2 with legacy
        # fallback so non-ET matches behave identically.
        if minute <= 45:
            vs_1h = _ko_1h_s + minute * 60
            return vs_1h if vs_1h <= _ht_s else _ko_2h_s + (minute - 45) * 60
        if minute <= 90:
            return _ko_2h_s + (minute - 45) * 60
        if minute <= 105 and _ko_et1_s:
            return _ko_et1_s + (minute - 90) * 60
        if _ko_et2_s:
            return _ko_et2_s + (minute - 105) * 60
        return _ko_2h_s + (minute - 45) * 60

    from pipeline_state import failed_windows as _failed_windows
    # Include both pending and previously-failed event windows (for retry)
    event_windows_pending = [
        wid for wid in window_ids
        if (not is_window_done(state, wid, "3d_event"))
        and _window_has_event(wid, events, windows)
    ] + [
        wid for wid in _failed_windows(state, "3d_event")
        if _window_has_event(wid, events, windows)
        and wid not in window_ids  # avoid duplicates
    ]
    event_windows_pending = list(dict.fromkeys(event_windows_pending))  # dedupe preserve order
    if event_windows_pending:
        failed_ev = _failed_windows(state, "3d_event")
        retry_msg = f" ({len(failed_ev)} retries)" if failed_ev else ""
        print(f"\n  PHASE 3: Step 3d-EV Event agent ({len(event_windows_pending)} windows{retry_msg})")

        import glob as _glob
        requests = []
        for wid in event_windows_pending:
            win    = next((w for w in windows
                           if str(get_window_id(w)) == str(wid)), {})
            frames = get_window_frames(match_dir, win, evf)

            # Get structural context from 3a output (filename includes window label)
            a_file = find_agent_output(
                os.path.join(match_dir, "agent_logs"), wid, "structural"
            )
            struct_ctx = ""
            if a_file:
                with open(a_file, encoding="utf-8") as f:
                    a = json.load(f)
                struct_ctx = (f"Formation home: {get_formation_home(a)}, "
                              f"Formation away: {get_formation_away(a)}, "
                              f"Line: {a.get('defensive_line',{}).get('avg_pct')}%")

            # Match events to this window using video seconds (not match minutes)
            win_start_s = win.get("start_s", win.get("start_seconds", 0))
            win_end_s   = win.get("end_s",   win.get("end_seconds",   0))
            for ev in events:
                ev_video_s = _ev_to_video_s(_ev_minute(ev))
                if ev_video_s is not None and win_start_s <= ev_video_s <= win_end_s:
                    prompt = build_event_prompt(mc, win, ev, struct_ctx)
                    from batch_runner import build_request
                    requests.append(build_request(wid, prompt,
                                                  _prepare_frames(frames, resize_w, resize_h),
                                                  "3d_event"))
                    break

        if requests:
            batch_id = with_retry(lambda: submit_batch(match_dir, state,
                                                        requests, "3d_event"))
            batch    = poll_batch(batch_id)
            collect_results(batch_id, match_dir, state, "3d_event")

    # ── PHASE 4: Python processing (no API calls) ─────────────────────────────
    # Mark any event windows that repeatedly failed as non-blocking
    # so 3e_merge can proceed on all structural data
    from pipeline_state import failed_windows as _fw, mark_window as _mw
    still_failed_ev = _fw(state, "3d_event")
    if still_failed_ev:
        print(f"\n  [INFO] {len(still_failed_ev)} event window(s) failed after retry.")
        print(f"  [INFO] Structural + player data is complete. Merge will proceed.")
        print(f"  [INFO] Failed event windows: {still_failed_ev}")
        for wid in still_failed_ev:
            _mw(match_dir, state, wid, "3d_event", "skipped",
                "Event agent failed after retry -- non-blocking, structural data complete")

    print(f"\n  PHASE 4: Steps 3e-3k (Python processing)")

    # Phase 4 is split: 3e–3i run first, then Phase 3b (set piece bursts)
    # reads the confirmation_queue.json that 3i_escalation just built, then
    # 3j–3l run with burst-enriched data in running_summary.
    def _run_python_steps(steps):
        for step_id, module, fn in steps:
            if is_step_done(state, step_id):
                print(f"  [OK] {step_id} (already complete)")
                continue
            try:
                print(f"  - {step_id}...", end="", flush=True)
                import importlib
                mod  = importlib.import_module(module)
                func = getattr(mod, fn, None)
                if func is None:
                    raise AttributeError(f"{module}.{fn} not found")
                if module == "deep_skill_metrics":
                    # Fix 33b: deep_skill_metrics treats "both" as the
                    # explicit no-focus default (see signature in that module).
                    func(match_dir, "both")
                else:
                    func(match_dir)
                mark_step(match_dir, state, step_id, "complete")
                # 3f_sequences also covers 3g_summary (same accumulation pass)
                if step_id == "3f_sequences":
                    mark_step(match_dir, state, "3g_summary", "complete")
                print(" done")
            except Exception as e:
                mark_step(match_dir, state, step_id, "failed", str(e))
                print(f" FAILED: {e}")

    _run_python_steps([
        ("3e_merge",        "merge_utils",       "merge_all_windows"),
        ("3f_shots",        "accumulator",        "build_shots_log"),
        ("3f_sequences",    "accumulator",        "accumulate_all_windows"),
        ("3h_ground_truth", "ground_truth",       "build_ground_truth_check"),
        ("3i_escalation",   "escalation_router",  "build_escalation_queue"),
    ])

    # ── PHASE 3b: Set piece 5fps bursts ──────────────────────────────────────
    # Reads confirmation_queue.json (built by 3i_escalation above), filters for
    # unresolved set_piece_delivery items, and runs a 5fps burst against each.
    # Extracts fresh frames from source video at target_fps -- NOT a resample
    # of the 1fps pool. Burst output is merged back into the matching
    # set_piece record via setpiece_writeback before 3j_readiness runs.
    try:
        from frame_extraction import (
            find_source_video, extract_segment,
            timestamp_to_seconds as _fe_ts2s,
        )
        from batch_runner import build_request as _build_req

        _queue_path = os.path.join(match_dir, "confirmation_queue.json")
        if os.path.exists(_queue_path):
            with open(_queue_path, encoding="utf-8") as _f:
                _cq = json.load(_f)

            _sp_items = [
                _item for _item in _cq.get("items", [])
                if _item.get("event_type") == "set_piece_delivery"
                and not _item.get("resolved")
            ]

            if _sp_items:
                print(f"\n  PHASE 3b: Step 3d-SP Set piece burst ({len(_sp_items)} items)")

                _video_path = find_source_video(match_dir)
                _burst_root = os.path.join(match_dir, "frames_burst")
                os.makedirs(_burst_root, exist_ok=True)

                _sp_requests = []
                _sp_skipped  = []

                for _item in _sp_items:
                    _anchor_ts  = _item.get("timestamp")
                    _team       = _item.get("team")
                    _window_id  = _item.get("window", "")
                    _target_fps = _item.get("escalation_target_fps", 5)
                    _end_ts     = _item.get("rerun_window_end", "")
                    _padding    = (
                        (_fe_ts2s(_end_ts) - _fe_ts2s(_anchor_ts))
                        if _end_ts else 3.0
                    )
                    if _padding <= 0:
                        _padding = 3.0

                    # Find merged window via canonical lookup
                    _merged_path = find_merged_window(
                        os.path.join(match_dir, "agent_logs"), _window_id
                    )
                    if _merged_path is None:
                        _sp_skipped.append((_anchor_ts, "no_merged_file"))
                        continue

                    with open(_merged_path, encoding="utf-8") as _f:
                        _merged = json.load(_f)

                    _sp_rec = next(
                        (_sp for _sp in _merged.get("set_pieces", [])
                         if _sp.get("timestamp") == _anchor_ts
                         and _sp.get("team") == _team),
                        None,
                    )
                    if _sp_rec is None:
                        _sp_skipped.append((_anchor_ts, "no_matching_1fps_record"))
                        continue

                    _anchor_dir = os.path.join(
                        _burst_root, f"sp_{_window_id}_{_anchor_ts}"
                    )
                    try:
                        _frames = extract_segment(
                            video_path     = _video_path,
                            anchor_seconds = _fe_ts2s(_anchor_ts),
                            out_dir        = _anchor_dir,
                            target_fps     = _target_fps,
                            padding_s      = _padding,
                        )
                    except Exception as _e:
                        _sp_skipped.append((_anchor_ts, f"extract_error: {_e}"))
                        continue

                    if not _frames:
                        _sp_skipped.append((_anchor_ts, "no_frames_extracted"))
                        continue

                    _prompt   = build_setpiece_prompt(
                        match_dir, _item, mc, _sp_rec, state
                    )
                    _burst_id = f"{_window_id}_{_anchor_ts}"
                    _sp_requests.append(_build_req(
                        _burst_id,
                        _prompt,
                        _prepare_frames(_frames, resize_w, resize_h),
                        "3d_setpiece",
                    ))

                if _sp_skipped:
                    print(f"  [INFO] Skipped {len(_sp_skipped)} set piece burst(s):")
                    for _ts, _reason in _sp_skipped:
                        print(f"    {_ts}: {_reason}")

                if _sp_requests:
                    _sp_batch_id = with_retry(lambda: submit_batch(
                        match_dir, state, _sp_requests, "3d_setpiece"
                    ))
                    poll_batch(_sp_batch_id)
                    collect_results(_sp_batch_id, match_dir, state, "3d_setpiece")

                    from setpiece_writeback import writeback_all_bursts
                    writeback_all_bursts(match_dir)
            else:
                print(f"\n  PHASE 3b: No unresolved set piece bursts (skipping)")
        else:
            print(f"\n  PHASE 3b: No confirmation_queue.json found (skipping)")

    except ImportError as _ie:
        print(f"\n  PHASE 3b: Skipped (frame_extraction unavailable: {_ie})")
    except Exception as _e:
        print(f"\n  PHASE 3b: Error — {_e} (non-blocking, continuing)")

    _run_python_steps([
        ("3j_readiness",    "build_readiness_check",  "build_readiness_check"),
        ("3k_metrics",      "deep_skill_metrics",      "build_deep_skill_metrics"),
        ("3l_synthesis",    "synthesis_agent",         "run_synthesis"),
    ])

    # ── PHASE 5: Reports ──────────────────────────────────────────────────────
    # Skip 4a/4b if 3l_synthesis has already produced the reports, unless the
    # caller explicitly requests --force-reports to run the older single-call
    # report generators instead.
    synthesis_complete = is_step_done(state, "3l_synthesis")

    if synthesis_complete:
        # 3l_synthesis is the preferred report path. If --force-reports was
        # passed, the CLI handler already reset 3l_synthesis to pending and
        # the python_steps loop above re-ran it. Either way, skip 4a/4b
        # here to avoid producing duplicate, lower-quality reports.
        print("\n  PHASE 5: 3l_synthesis output is current. Skipping 4a/4b.")
        print("  tactical_report.md and opposition reports written by 3l_synthesis.")
    else:
        import anthropic as _anthropic

        report_level = mc.get("report_level", "standard")

        if not is_step_done(state, "4a_tactical_report"):
            print(f"\n  PHASE 5: Step 4a Tactical report...")
            try:
                prompt = build_tactical_prompt(match_dir, mc, report_level)
                client = _anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}]
                )
                report_text = response.content[0].text
                out_path = os.path.join(match_dir, "tactical_report.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                mark_step(match_dir, state, "4a_tactical_report", "complete")
                print(f"  [OK] tactical_report.md written ({len(report_text):,} chars)")
            except Exception as e:
                mark_step(match_dir, state, "4a_tactical_report", "failed", str(e))
                print(f"  [FAIL] tactical report: {e}")

        if not is_step_done(state, "4b_opposition_report"):
            print(f"\n  PHASE 5: Step 4b Opposition report...")
            try:
                prompt = build_opposition_prompt(match_dir, mc,
                                                 mc.get("home_team", ""), report_level)
                client = _anthropic.Anthropic()
                response = client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}]
                )
                report_text = response.content[0].text
                opp_name = (mc.get("home_team", "opposition")
                            .lower().replace(" ", "_").replace("&", "and")[:20])
                out_path = os.path.join(match_dir, f"opposition_report_{opp_name}.md")
                with open(out_path, "w", encoding="utf-8") as f:
                    f.write(report_text)
                mark_step(match_dir, state, "4b_opposition_report", "complete")
                print(f"  [OK] {os.path.basename(out_path)} written ({len(report_text):,} chars)")
            except Exception as e:
                mark_step(match_dir, state, "4b_opposition_report", "failed", str(e))
                print(f"  [FAIL] opposition report: {e}")

    # ── PHASE 6: Step 4c flagged_moments + 4d pass_network ──────────────────
    import subprocess as _sp

    def _run_generator(script_name: str, match_dir: str,
                       step_key: str, state: dict) -> bool:
        if is_step_done(state, step_key):
            print(f"  - {step_key} (already complete, skipping)")
            return True

        scripts_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(scripts_dir, script_name)

        if not os.path.exists(script_path):
            print(f"  [WARN] {script_name} not found in scripts dir — skipping {step_key}")
            return False

        print(f"  - {step_key}...")
        result = _sp.run(
            [sys.executable, script_path, match_dir],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            mark_step(match_dir, state, step_key, "complete")
            if result.stdout.strip():
                print(result.stdout.rstrip())
            print(f"  [OK] {step_key}")
            return True
        else:
            err = (result.stderr.strip().splitlines()[-1]
                   if result.stderr.strip() else "unknown error")
            print(f"  [WARN] {step_key} failed: {err}")
            mark_step(match_dir, state, step_key, "failed", err)
            return False

    print(f"\n  PHASE 6: Step 4c/4d — Flagged moments + Pass network")

    _run_generator("generate_flagged_moments.py", match_dir,
                   "4c_flagged_moments", state)

    _pass_seq = os.path.join(match_dir, "pass_sequences.json")
    if os.path.exists(_pass_seq):
        _run_generator("generate_pass_network.py", match_dir,
                       "4d_pass_network", state)
    else:
        print("  - 4d_pass_network (skipped — pass_sequences.json not found)")

    print_progress(state)
    print(f"\n  Pipeline complete. Reports in {match_dir}")


def build_report_roster_block(mc: dict) -> str:
    """
    Build the confirmed player roster block to embed at the top of every
    report prompt. This is the primary defence against cross-match
    player name contamination.

    The roster is injected directly into the prompt text -- the agent
    reads it in its immediate context, not from a file it might forget to check.
    """
    kits = _kits(mc)

    def format_lineup(lineup_entry: dict) -> str:
        lines = []
        for p in lineup_entry.get("startXI", []):
            player = p.get("player", {})
            name   = player.get("name", "?") if isinstance(player, dict) else str(player)
            num    = player.get("number", "?") if isinstance(player, dict) else "?"
            pos    = player.get("pos", "") or ""
            lines.append(f"  #{num} {name}" + (f" ({pos})" if pos else ""))
        subs = []
        for p in lineup_entry.get("substitutes", []):
            player = p.get("player", {})
            name   = player.get("name", "?") if isinstance(player, dict) else str(player)
            num    = player.get("number", "?") if isinstance(player, dict) else "?"
            subs.append(f"  #{num} {name}")
        starting  = "  Starting XI:\n" + "\n".join(lines)
        sub_block = ("\n  Substitutes:\n" + "\n".join(subs)) if subs else ""
        return starting + sub_block

    sections = []
    for lineup in mc.get("lineups", []):
        if not isinstance(lineup, dict):
            continue
        team_raw  = lineup.get("team", {})
        team_name = team_raw.get("name", "?") if isinstance(team_raw, dict) else str(team_raw)
        sections.append(f"{team_name}:\n{format_lineup(lineup)}")

    roster = "\n\n".join(sections)

    match_name = mc.get("match") or (mc.get("home_team","?") + " vs " + mc.get("away_team","?"))
    date       = mc.get("date","?")

    return (
        "\n=== CONFIRMED PLAYER ROSTER -- THE ONLY NAMES YOU MAY USE ===\n"
        f"Match: {match_name}\n"
        f"Date:  {date}\n\n"
        f"{roster}\n\n"
        "ABSOLUTE RULE: You may only write a player name if it appears in the lists above.\n"
        "If you find yourself writing a name not in these lists:\n"
        "  STOP. Delete it. Do not include it, reference it, or mention its absence.\n"
        "  Not even to say a player is not listed. They do not exist for this report.\n"
        "This applies to every player from every previous match, club, or competition.\n"
        "Your knowledge of other matches does not exist for this report.\n"
        "=== END ROSTER ===\n"
    )


def _load_skill_block(section_start: str, section_end: str) -> str:
    """Extract a named block from SKILL.md between two marker strings."""
    skill_path = os.path.join(os.path.dirname(__file__), "..", "SKILL.md")
    if not os.path.exists(skill_path):
        return ""
    with open(skill_path, encoding="utf-8") as f:
        skill = f.read()
    start = skill.find(section_start)
    if start < 0:
        return ""
    end = skill.find(section_end, start)
    if end < 0:
        return skill[start:]
    return skill[start : end + len(section_end)]


def build_tactical_prompt(match_dir: str, mc: dict,
                           report_level: str = "standard") -> str:
    """Build the Step 4a tactical report prompt from pipeline data files."""
    files = {}
    for fname in ["running_summary.json", "pass_sequences.json",
                  "deep_skill_metrics.json", "source_profile.json",
                  "report_readiness.json"]:
        path = os.path.join(match_dir, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                files[fname] = json.load(f)
        else:
            files[fname] = {}

    source    = files.get("source_profile.json", {})
    readiness = files.get("report_readiness.json", {})
    roster    = build_report_roster_block(mc)

    player_id_ceiling = readiness.get("player_id_ceiling", "probable")
    source_type       = source.get("source_type", "veo_ball_tracking")
    # G2 fix: broadcast source_profile.json stores the limitation under "notes",
    # not "source_limitations_note". Prefer either-key, fall back to the Veo
    # default only when both are absent so legacy Veo behaviour is preserved.
    source_note       = (source.get("source_limitations_note")
                         or source.get("notes")
                         or "Ball-follow camera. Near-ball framing limits "
                            "weak-side observation.")

    rs  = files["running_summary.json"]
    psq = files["pass_sequences.json"]
    dsm = files["deep_skill_metrics.json"]

    data_block = json.dumps({
        "match_config": {k: mc.get(k) for k in [
            "match", "home_team", "away_team",
            "date", "ft_score", "ht_score", "goals", "cards", "substitutions"]},
        "line_height_by_window":    (rs.get("line_height_by_window", []) or [])[:30],
        "pressing_by_window":       (rs.get("pressing_by_window", []) or [])[:30],
        "match_state_by_window":    (rs.get("match_state_by_window", []) or [])[:30],
        "set_pieces":               (rs.get("set_pieces", []) or [])[:20],
        "transitions":              (rs.get("transitions", []) or [])[:30],
        "shots_for":                rs.get("shots_for", []) or [],
        "shots_against":            rs.get("shots_against", []) or [],
        "individual_observations":  (rs.get("individual_observations", []) or [])[:60],
        "key_moments":              (rs.get("key_moments", []) or [])[:20],
        "gk_kicks":                 (rs.get("gk_kicks", []) or [])[:20],
        "pass_sequences_sample":    ((psq.get("sequences", []) if isinstance(psq, dict)
                                      else []) or [])[:50],
        "deep_skill_metrics":       (dsm.get("metrics", []) if isinstance(dsm, dict)
                                     else []) or [],
    }, indent=1, default=str)

    constraint_block = _load_skill_block(
        "=== MATCH LENS REPORT CONSTRAINTS -- NON-NEGOTIABLE ===",
        "=== END CONSTRAINTS ===")

    report_structure = _load_skill_block(
        "### 4a -- Tactical Report prompt",
        "### 4b -- Opposition Report prompt")

    return f"""You are a football tactical analyst writing a match report for a coaching staff.
Read the data provided. Write from the data. Do not use memory from other matches.

{constraint_block}

REPORT LEVEL: {report_level}
PLAYER ID CEILING: {player_id_ceiling}
SOURCE TYPE: {source_type}
SOURCE LIMITATION: {source_note}

{roster}

=== MANDATORY PRE-FLIGHT (output this block verbatim before any prose) ===
Before writing the report, output this block exactly, filling in each line:
MATCH: [copy match field from match_config.json]
HOME TEAM CONFIRMED: [yes/no]
AWAY TEAM CONFIRMED: [yes/no]
PLAYER COUNT HOME: [count of home players in lineup]
PLAYER COUNT AWAY: [count of away players in lineup]
PREVIOUS MATCH KNOWLEDGE: EXCLUDED
WINDOW REFERENCES IN REPORT: [count -- must be 0]
AGENT LANGUAGE IN REPORT: [count -- must be 0]
If WINDOW REFERENCES or AGENT LANGUAGE is not 0: fix those before submitting.
=== END MANDATORY PRE-FLIGHT ===

=== PIPELINE DATA (read this, write from this) ===
{data_block}
=== END DATA ===

{report_structure}
"""


def build_opposition_prompt(match_dir: str, mc: dict,
                             opponent_team: str,
                             report_level: str = "standard") -> str:
    """Build the Step 4b opposition scouting report prompt."""
    files = {}
    for fname in ["running_summary.json", "pass_sequences.json",
                  "deep_skill_metrics.json", "source_profile.json",
                  "report_readiness.json"]:
        path = os.path.join(match_dir, fname)
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                files[fname] = json.load(f)
        else:
            files[fname] = {}

    source    = files.get("source_profile.json", {})
    readiness = files.get("report_readiness.json", {})
    roster    = build_report_roster_block(mc)

    player_id_ceiling = readiness.get("player_id_ceiling", "probable")
    source_type       = source.get("source_type", "veo_ball_tracking")
    # G2 fix: see build_tactical_prompt for rationale.
    source_note       = (source.get("source_limitations_note")
                         or source.get("notes")
                         or "Ball-follow camera.")

    # Fix 33b: opp_name is the team this report profiles. The opponent_team
    # arg is the team being scouted; the "other" team is the home team unless
    # opponent_team IS the home team, in which case the other side is away.
    home_team = mc.get("home_team", "")
    away_team = mc.get("away_team", "")
    opp_name  = opponent_team or away_team
    other_team = home_team if opp_name != home_team else away_team

    rs  = files["running_summary.json"]
    psq = files["pass_sequences.json"]
    dsm = files["deep_skill_metrics.json"]

    data_block = json.dumps({
        "match_config": {k: mc.get(k) for k in [
            "match", "home_team", "away_team",
            "date", "ft_score", "ht_score", "goals", "cards", "substitutions"]},
        "line_height_by_window":   (rs.get("line_height_by_window", []) or [])[:30],
        "pressing_by_window":      (rs.get("pressing_by_window", []) or [])[:30],
        "match_state_by_window":   (rs.get("match_state_by_window", []) or [])[:30],
        "set_pieces":              (rs.get("set_pieces", []) or [])[:20],
        "transitions":             (rs.get("transitions", []) or [])[:30],
        "shots_against":           rs.get("shots_against", []) or [],
        "individual_observations": (rs.get("individual_observations", []) or [])[:60],
        "key_moments":             (rs.get("key_moments", []) or [])[:20],
        "gk_kicks":                (rs.get("gk_kicks", []) or [])[:20],
        "pass_sequences_sample":   ((psq.get("sequences", []) if isinstance(psq, dict)
                                     else []) or [])[:50],
        "deep_skill_metrics":      (dsm.get("metrics", []) if isinstance(dsm, dict)
                                    else []) or [],
    }, indent=1, default=str)

    constraint_block = _load_skill_block(
        "=== MATCH LENS REPORT CONSTRAINTS -- NON-NEGOTIABLE ===",
        "=== END CONSTRAINTS ===")

    report_structure = _load_skill_block(
        "### 4b -- Opposition Report prompt",
        "### 4c -- Flagged Moments prompt")

    return f"""You are writing an opposition scouting report for a football coaching staff.
Read the data provided. Write from the data. Do not use memory from other matches.

{constraint_block}

REPORT LEVEL: {report_level}
PLAYER ID CEILING: {player_id_ceiling}
SOURCE TYPE: {source_type}
SOURCE LIMITATION: {source_note}
OPPONENT: {opp_name}

{roster}

=== MANDATORY PRE-FLIGHT (output this block verbatim before any prose) ===
Before writing the report, output this block exactly, filling in each line:
MATCH: [copy match field from match_config.json]
OPPONENT CONFIRMED: [yes/no]
PREVIOUS MATCH KNOWLEDGE: EXCLUDED
WINDOW REFERENCES IN REPORT: [count -- must be 0]
AGENT LANGUAGE IN REPORT: [count -- must be 0]
If WINDOW REFERENCES or AGENT LANGUAGE is not 0: fix those before submitting.
=== END MANDATORY PRE-FLIGHT ===

=== PIPELINE DATA ===
{data_block}
=== END DATA ===

{report_structure}
"""


def _window_has_event(wid: str, events: list, windows: list) -> bool:
    """
    Use the event_window flag set by window_plan.mark_event_windows. (Fix 4)
    Falls back to time-range matching if flag is absent.
    """
    win = next((w for w in windows
                if str(get_window_id(w)) == str(wid)), {})
    # Primary: use the flag already set in window_plan
    if "event_window" in win:
        return bool(win["event_window"])
    # Fallback: time-range match using correct key names
    t_start = win.get("start_s", win.get("start_seconds", 0)) / 60
    t_end   = win.get("end_s",   win.get("end_seconds",   0)) / 60
    return any(t_start <= _ev_minute(ev) <= t_end
               for ev in events
               if isinstance(_ev_minute(ev), (int, float)))


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Match Lens pipeline v2")
    parser.add_argument("match_dir", help="Match directory path")
    parser.add_argument("--quality",  choices=["economy","standard","full","full_1fps"],
                        default="standard")
    parser.add_argument("--resume",  action="store_true",
                        help="Resume from checkpoint")
    parser.add_argument("--blind-formation", action="store_true",
                        help="Strip formation from lineup data before injection "
                             "(diagnostic: tests whether agents observe vs echo lineup). "
                             "Automatically resets 3a + downstream when all windows are complete.")
    parser.add_argument("--force-structural", action="store_true",
                        help="Reset all 3a structural windows to pending and rerun "
                             "(also resets 3b and 3e downstream)")
    parser.add_argument("--estimate-only", action="store_true",
                        help="Show cost estimate and exit")
    parser.add_argument("--force-reports", action="store_true",
                        help="Regenerate reports even if pipeline steps are incomplete")
    parser.add_argument("--force-merge", action="store_true",
                        help="Force 3e merge and downstream steps to rerun")
    args = parser.parse_args()

    if args.estimate_only:
        md  = load_match_data(args.match_dir)
        est = [calculate_cost(md, q) for q in ["economy","standard","full","full_1fps"]]
        print_estimate(md, est)
        sys.exit(0)

    # --blind-formation auto-implies --force-structural when 3a is already done
    if args.blind_formation:
        _bs = load_state(args.match_dir)
        if _bs and all(
            _bs.get("windows", {}).get(wid, {}).get("3a") == "complete"
            for wid in _bs.get("windows", {})
        ):
            print("  [blind-formation] All 3a windows complete — auto-resetting "
                  "structural + downstream for blind rerun.")
            args.force_structural = True

    if args.force_reports or args.force_merge or args.force_structural:
        state = load_state(args.match_dir)
        if not state:
            print("No pipeline state found. Run without --force-reports first.")
            sys.exit(1)
        if args.force_merge:
            # Reset merge and all downstream steps so they rerun
            for step in ["3e_merge","3f_shots","3f_sequences","3g_summary",
                         "3h_ground_truth","3i_escalation","3j_readiness",
                         "3k_metrics","3l_synthesis",
                         "4a_tactical_report","4b_opposition_report",
                         "4c_flagged_moments","4d_pass_network"]:
                state["steps"][step] = "pending"
                print(f"  Reset: {step}")
            # Also reset failed event windows to skipped so merge proceeds
            for wid, steps in state["windows"].items():
                if steps.get("3d_event") == "failed":
                    steps["3d_event"] = "skipped"
            from pipeline_state import _save as _ps_save
            _ps_save(args.match_dir, state)
            print("  State reset for merge + reports. Re-running...")
        if args.force_structural:
            # Reset 3a + 3b windows to pending, and 3e downstream steps
            for wid in state.get("windows", {}):
                for wstep in ("3a", "3b"):
                    state["windows"][wid][wstep] = "pending"
            for step in ["3e_merge","3f_shots","3f_sequences","3g_summary",
                         "3h_ground_truth","3i_escalation","3j_readiness",
                         "3k_metrics","3l_synthesis",
                         "4a_tactical_report","4b_opposition_report",
                         "4c_flagged_moments","4d_pass_network"]:
                state["steps"][step] = "pending"
            from pipeline_state import _save as _ps_save_s
            _ps_save_s(args.match_dir, state)
            print(f"  Reset: all {len(state['windows'])} windows → 3a + 3b pending, "
                  f"3e–4d reset. Re-running from structural pass...")
        elif args.force_reports:
            # Prefer re-running 3l_synthesis (richer agent) when it has run before.
            # Fall back to legacy 4a/4b only when 3l never produced output.
            if state["steps"].get("3l_synthesis") == "complete":
                state["steps"]["3l_synthesis"] = "pending"
                for step in ["4c_flagged_moments", "4d_pass_network"]:
                    state["steps"][step] = "pending"
                print("  Reset: 3l_synthesis + 4c/4d (preferred report path)")
            else:
                for step in ["4a_tactical_report","4b_opposition_report",
                             "4c_flagged_moments","4d_pass_network"]:
                    state["steps"][step] = "pending"
                print("  Reset: 4a/4b + 4c/4d (3l_synthesis unavailable)")
            from pipeline_state import _save as _ps_save
            _ps_save(args.match_dir, state)
            print("  Reports reset to pending. Re-running...")

    run_pipeline(args.match_dir, args.quality, args.resume,
                 force_reports=args.force_reports)
