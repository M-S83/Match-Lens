#!/usr/bin/env python3
"""
pipeline_runner.py — Match Lens analysis execution engine + pipeline orchestrator.

Claude API steps (Steps 3a, 3c, 3d, 3i) call the Anthropic API directly.
All other programmatic steps delegate to dedicated modules.

Usage:
    python pipeline_runner.py MATCH_DIR [options]

    --video PATH        Video file (required for steps 1, 1f, 3i)
    --focus-team TEAM   home | away | both  (default: both)
    --step STEP_ID      Run one specific step
    --auto              Run all ready programmatic steps sequentially
    --status            Show pipeline state and next steps
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import pathlib

# Load .env from multiple locations
for env_path in [
    pathlib.Path(__file__).parent / '.env',
    pathlib.Path(__file__).parent.parent / '.env',
    pathlib.Path.home() / '.env',
]:
    if env_path.exists():
        load_dotenv(env_path)
        key = os.environ.get('ANTHROPIC_API_KEY', '')
        print(f"  API key loaded from {env_path} (****{key[-4:] if key else 'NOT FOUND'})")
        break

import json
import os
import sys
import glob
import base64
import io
import time
import subprocess
import argparse
from datetime import datetime
from PIL import Image

# ── Path setup ────────────────────────────────────────────────────────────────

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

SKILL_MD = os.path.join(os.path.dirname(SCRIPTS_DIR), "SKILL.md")

# ── Module imports ─────────────────────────────────────────────────────────────

from job_logger            import JobLogger
from pipeline_accessors import get_source_limitations_note
from source_profiler       import sample_frames, build_source_profile, CLASSIFICATION_PROMPT
from escalation_router     import build_escalation_queue
from deep_skill_metrics    import build_deep_skill_metrics
from build_readiness_check import build_readiness_check
from merge_utils           import merge_dual_agents, merge_single_agent, merge_all_windows
from accumulator           import (update_running_summary, accumulate_all_windows,
                                   apply_confirmation_to_summary, accumulate_pass_sequences)
from ground_truth          import build_ground_truth_check, parse_timestamp_to_seconds as _parse_ts
from window_plan           import build_window_plan, mark_event_windows
from md_to_docx            import convert_all_reports


# ═══════════════════════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _load(match_dir: str, fname: str, default=None):
    p = os.path.join(match_dir, fname)
    if not os.path.exists(p):
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def _banner(title: str):
    print(f"\n{'-' * 55}")
    print(f"  {title}")
    print(f"{'-' * 55}")


def _timestamp_to_seconds(ts: str) -> float:
    """Convert 'MMmSSs' or 'MM:SS' to seconds."""
    ts = str(ts).strip()
    if "m" in ts and ts.endswith("s"):
        try:
            parts = ts.rstrip("s").split("m")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass
    if ":" in ts:
        try:
            parts = ts.split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass
    try:
        return float(ts)
    except Exception:
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Claude API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def encode_image(path: str, max_width: int = 640) -> str:
    """
    Resize frame to max_width px wide (height scaled proportionally),
    encode as JPEG quality 85, return base64 string.
    """
    with Image.open(path, encoding="utf-8") as img:
        orig_w, orig_h = img.size
        if orig_w > max_width:
            new_h = int(orig_h * max_width / orig_w)
            img = img.resize((max_width, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.standard_b64encode(buf.getvalue()).decode("utf-8")


def _estimate_tokens_per_frame(frames: list, max_width: int = 640) -> int:
    """
    Sample the first frame to estimate tokens per image after resize.
    Uses Anthropic's approximate formula: (width * height) / 750.
    """
    if not frames:
        return 300  # fallback: ~640x360 / 750
    try:
        with Image.open(frames[0], encoding="utf-8") as img:
            w, h = img.size
            if w > max_width:
                h = int(h * max_width / w)
                w = max_width
            return max(150, (w * h) // 750)
    except Exception:
        return 300


def call_claude(client, prompt_text: str, image_paths: list,
                model: str, max_tokens: int = 4096, max_width: int = 640):
    """
    Build content blocks and call Claude API.
    Frames are resized to max_width px wide before encoding.
    Returns (response_text, input_tokens, output_tokens).
    """
    content = [{"type": "text", "text": prompt_text}]
    for path in image_paths:
        if os.path.exists(path):
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": encode_image(path, max_width=max_width),
                },
            })

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": content}],
    )
    text = response.content[0].text
    return text, response.usage.input_tokens, response.usage.output_tokens


def _parse_json_response(text: str, agent_id: str, window_label: str) -> dict:
    """
    Parse Claude JSON response. Strips markdown fences if present.
    Returns parsed dict, or failure record on error.
    """
    raw = text.strip()

    # Strip markdown fences
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Remove opening fence (```json or ```)
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "agent_id":    agent_id,
            "window":      window_label,
            "status":      "parse_failed",
            "raw_response": text[:2000],
            "findings":    [],
            "frames":      [],
        }

    # Ensure required fields
    data.setdefault("agent_id", agent_id)
    data.setdefault("window", window_label)
    data.setdefault("findings", [])
    data.setdefault("frames", [])

    return data


def _collect_frames(match_dir: str, start_frame: str, end_frame: str,
                    cap: int = 300) -> list:
    """
    Collect frame file paths between start_frame and end_frame (inclusive).
    Looks in frames/frames_detailed/ first, then frames/.
    Returns sorted list of existing paths, capped at `cap`.
    """
    # Resolve frames directory
    detailed_dir = os.path.join(match_dir, "frames", "frames_detailed")
    flat_dir     = os.path.join(match_dir, "frames")

    frames_dir = detailed_dir if os.path.isdir(detailed_dir) else flat_dir

    all_frames = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    if not all_frames:
        return []

    # Normalise start/end to "frame_XXmYYs.jpg" regardless of input format
    def _norm(s):
        s = os.path.basename(s)
        if not s.startswith("frame_"):
            s = "frame_" + s
        if not s.endswith(".jpg"):
            s += ".jpg"
        return s

    start_name = _norm(start_frame)
    end_name   = _norm(end_frame)

    in_window = []
    for path in all_frames:
        name = os.path.basename(path)
        if start_name <= name <= end_name:
            in_window.append(path)

    return in_window[:cap]


def _build_source_injection(match_dir: str) -> str:
    """Build SOURCE CONTEXT block from source_profile.json and result_family_gates.json."""
    sp = _load(match_dir, "source_profile.json", {})
    gd = _load(match_dir, "result_family_gates.json", {})
    gates = gd.get("gates", {})

    downgraded = [f for f, s in gates.items() if s == "downgraded"]
    allowed    = [f for f, s in gates.items() if s == "allowed"]

    return (
        f"Source type: {sp.get('source_type', 'unknown')}\n"
        f"Downgraded families (produce findings but add limitations_note):\n"
        f"  {', '.join(downgraded) if downgraded else 'none'}\n"
        f"Allowed families: {', '.join(allowed[:8])}{'...' if len(allowed) > 8 else ''}\n"
        f"Source limitation: {get_source_limitations_note(sp)}"
    )


def _load_match_context(match_dir: str) -> dict:
    """Load and return match context dict from match_config.json."""
    config = _load(match_dir, "match_config.json", {})
    # Fix 33b: legacy focus team removed from match_ctx -- pipeline produces
    # reports for both teams independently. Prompts use home_team explicitly.
    return {
        "match":            config.get("match", "Unknown Match"),
        "home_team":        config.get("home_team", ""),
        "away_team":        config.get("away_team", ""),
        "home_kit":         config.get("home_kit", ""),
        "away_kit":         config.get("away_kit", ""),
        "gk_name":          config.get("gk_name", ""),
        "gk_number":        config.get("gk_number", ""),
        "gk_kit":           config.get("gk_kit", ""),
        "attack_dir_1h":    config.get("attack_direction_1h", ""),
        "attack_dir_2h":    config.get("attack_direction_2h", ""),
        "goals":            config.get("goals", []),
        "substitutions":    config.get("substitutions", config.get("subs_used", [])),
        "cards":            config.get("cards", []),
    }


def _build_tier1_prompt(match_dir: str, window: dict,
                        match_ctx: dict, source_injection: str,
                        match_data_block: str) -> str:
    """Build the complete Tier 1 agent prompt for a window."""
    half    = window.get("half", "1H")
    atk_dir = match_ctx["attack_dir_1h"] if half == "1H" else match_ctx["attack_dir_2h"]
    atk_dir_upper = atk_dir.upper() if atk_dir else "LEFT"

    # Determine team/opponent from focus
    # Fix 33b: home/away naming -- no single focus team.
    focus   = match_ctx["home_team"]
    opp     = match_ctx["away_team"]
    fkit    = match_ctx["home_kit"]
    okit    = match_ctx["away_kit"]

    gk_line = ""
    if match_ctx["gk_name"]:
        gk_line = (f"GK {match_ctx['gk_name']} (#{match_ctx['gk_number']}) "
                   f"in {match_ctx['gk_kit']}.")

    prompt = f"""Football tactical analyst. {match_ctx['match']}. {focus} in {fkit}, {opp} in {okit}.
{gk_line} {focus} attack {atk_dir_upper} in this half.

=== SOURCE CONTEXT ===
{source_injection}

=== MATCH DATA ===
{match_data_block}

=== WINDOW ===
Window: {window.get('label', window.get('agent_id', ''))}
View EVERY SECOND (1fps) from {window.get('start_frame')} to {window.get('end_frame')}.
Each window covers 5 minutes — approximately 300 frames.

=== CONFIDENCE SCORING ===
For every frame or frame group you review, output a confidence block alongside
your observations. Score 0.0-1.0 based on how reliably you can read the tactical
picture. Use these reason codes when confidence is below 1.0:

  occlusion       - player(s) obscured, position or identity unclear
  camera_motion   - pan or zoom blur during the action
  kit_ambiguity   - cannot reliably distinguish teams
  ball_not_visible - ball position inferred, not observed
  cluster         - multiple players tightly grouped, individual tracking unreliable
  partial_frame   - pitch edge cut off, line height or shape unmeasurable
  low_contrast    - poor lighting or overexposure

Any frame scoring below 0.7 will be automatically queued for a targeted re-run.
Be honest - underconfidence wastes one re-run; overconfidence corrupts the report.

=== PASS TRACKING ===
For every possession sequence you observe, log it as a chain:
  [Player #N] ->[dir] [Player #N] ->[dir] [outcome]
Direction codes: F=forward  S=sideways  B=backward
Outcomes: shot / cross / lost_possession / clearance / set_piece / end_of_window
Example: "#1 GK ->F #5 CB ->S #17 CM ->F #11 LM ->F lost_possession"
Log EVERY sequence visible. Minimum 20 sequences per window.

=== PRESSING INTENSITY ===
Score pressing intensity 0-10 per frame group:
  0  = no players within pressing distance of ball carrier
  3  = one player closing, others passive
  5  = organised first press with cover shadow
  7  = coordinated press cutting passing lanes
  10 = full high press, ball carrier has no safe outlet
Record score and any press triggers observed:
(GK distribution / back pass / defender facing own goal / keeper in hands)

=== DEFENSIVE LINE HEIGHT ===
Estimate FOCUS TEAM defensive line as % of pitch (0%=own goal, 100%=opp goal).
Calibration: penalty box edge = ~16% from goal line.
Record at start, middle, and end of each frame group.
Note significant shifts with timestamp and cause.

=== SHOT TRACKING ===
Track shots from BOTH teams. Every shot must be logged regardless of which team
took it. Use the team field to distinguish.

ORIGIN ZONE:
  Columns:  left_channel / left_of_centre / central / right_of_centre / right_channel
  Rows:     six_yard_box / penalty_spot / edge_of_box / outside_box

SHOT TYPE: foot_right / foot_left / header / deflection

OUTCOME: goal / on_target / off_target / blocked / post_bar

TARGET ZONE: top_left / top_centre / top_right / bottom_left / bottom_centre /
             bottom_right / blocked_before_goal

POSSESSION SEQUENCE TO SHOT - log the full passing chain:
  "#N ->F #N ->S #N ->F SHOT"
Include how possession was won:
  open_play_win / set_piece / turnover / GK_distribution / kickoff

=== SET PIECES ===
For every set piece (corner, free kick, final-third throw):
- Type: corner_left / corner_right / direct_fk / indirect_fk / throw_final_third
- Delivery zone: near_post / far_post / penalty_spot / edge_of_box / short
- Delivery type: inswinger / outswinger / driven / flick_on / short_routine
- Bodies in box: [number]
- Marking system: zonal / man / mixed
- Outcome: goal / shot / cleared / keeper / wasted

Output ONLY raw JSON. No preamble, no explanation, no markdown fences.

{{
  "agent_id": "{window.get('agent_id', 'NN')}",
  "window": "{window.get('label', '')}",
  "frames_reviewed": [number],
  "scan_interval_seconds": 1,

  "frames": [
    {{
      "frame": "frame_XXmYYs.jpg",
      "confidence": {{
        "score": [0.0-1.0],
        "flags": ["[reason_code]"],
        "affected_players": ["#N"],
        "affected_metrics": ["line_height", "pressing_intensity", "formation", "pass_tracking"]
      }},
      "observations": {{
        "formation_shape": "[e.g. 4-3-3]",
        "pressing_score": [0-10],
        "press_trigger": "[description or null]",
        "line_height_pct": [0-100],
        "ball_visible": [true/false],
        "ball_zone": "[defending_third / middle / attacking_third or null]",
        "notes": "[anything notable]"
      }}
    }}
  ],

  "possession_summary": {{
    "{focus}_pct": [0-100],
    "{opp}_pct": [0-100],
    "dominant_zone": "[defending_third / middle / attacking_third]",
    "territory_notes": "[description]"
  }},

  "formation": {{
    "shape_in_possession": "[e.g. 4-4-2]",
    "shape_out_of_possession": "[e.g. 4-4-2]",
    "compactness": "[compact / stretched / disorganised]",
    "notes": "[any mid-window variation]"
  }},

  "defensive_line": {{
    "start_pct": [0-100],
    "mid_pct": [0-100],
    "end_pct": [0-100],
    "avg_pct": [0-100],
    "notable_shifts": ["[timestamp]: dropped from X% to Y% after [event]"],
    "frames_excluded": []
  }},

  "pressing": {{
    "scores": [
      {{"frame_group": "[MMmSSs]", "score": [0-10], "trigger": "[description or null]"}}
    ],
    "avg_score": [0-10],
    "peak_score": [0-10],
    "peak_timestamp": "[MMmSSs]",
    "coordinated_press_observed": [true/false],
    "press_triggers_identified": ["[description]"],
    "frames_excluded": []
  }},

  "pass_sequences": [
    {{
      "start_frame": "[MMmSSs]",
      "sequence": "[chain string]",
      "length": [number],
      "zone_start": "[defending_third / middle / attacking_third]",
      "zone_end": "[defending_third / middle / attacking_third]",
      "outcome": "[shot/cross/lost_possession/clearance/set_piece/end_of_window]",
      "progressive": [true/false]
    }}
  ],

  "set_pieces": [],
  "shot_attempts": [],
  "attacking": {{}},
  "defensive": {{}},

  "key_moments": [
    {{
      "timestamp": "[MMmSSs]",
      "type": "[goal/chance/sub/set_piece/tactical_shift/individual/disciplinary]",
      "description": "[what happened]",
      "tactical_significance": "[why it matters]",
      "frames": ["frame_XXmYYs.jpg"]
    }}
  ],

  "individual_observations": [],
  "flaggable_moments": [],

  "window_summary": "[3-4 sentences: shape, press, key events, tone]",

  "window_confidence": {{
    "overall_score": [0.0-1.0],
    "low_confidence_frame_count": [number],
    "unresolvable_frame_count": [number],
    "data_gap_warning": [true/false]
  }},

  "confirmation_queue": [],

  "findings": [
    {{
      "analysis_scope": "[match / opposition / player]",
      "finding_type": "[description]",
      "result_family": "[family name]",
      "team": "[home / away / both]",
      "subject_player_id": null,
      "subject_player_label": null,
      "time_start": "[MMmSSs]",
      "time_end": "[MMmSSs]",
      "evidence_tier": "[direct / repeated_pattern / suggestive]",
      "confidence": 0.0,
      "result_family_status": "allowed",
      "supporting_frames": ["frame_XXmYYs.jpg"],
      "limitations_note": null,
      "escalation_reason": null,
      "escalation_target_fps": null,
      "opposition_focus_type": null
    }}
  ]
}}"""
    return prompt


def _build_deep_scan_additions(window: dict, match_ctx: dict) -> str:
    """Build the DEEP SCAN additions block for event windows."""
    lines = ["\n=== DEEP SCAN MODE ===\nThis is a dual-agent deep scan. "
             "You are one of two independent agents analysing this window.\n"]

    event_types = window.get("event_types", [])

    if "goal" in event_types:
        for goal in match_ctx.get("goals", []):
            g_time = goal.get("time", {})
            g_min  = g_time.get("elapsed", 0)
            # Check if this goal is roughly in this window
            w_start_min = window.get("start_s", 0) / 60
            w_end_min   = window.get("end_s", 0) / 60
            if w_start_min <= g_min <= w_end_min:
                scorer = goal.get("player", {}).get("name", "unknown")
                team   = goal.get("team", {}).get("name", "unknown")
                lines.append(
                    f"{scorer} ({team}) scores at {g_min}' - THIS GOAL IS IN THIS WINDOW.\n"
                    f"Find it (celebrations, centre restart). When found, view every frame for\n"
                    f"30 seconds before and describe the full build-up sequence frame by frame.\n"
                )

    if "sub" in event_types:
        for sub in match_ctx.get("substitutions", []):
            s_time = sub.get("time", {})
            s_min  = s_time.get("elapsed", 0)
            w_start_min = window.get("start_s", 0) / 60
            w_end_min   = window.get("end_s", 0) / 60
            if w_start_min <= s_min <= w_end_min:
                on   = sub.get("assist", {}).get("name", "unknown")
                off  = sub.get("player", {}).get("name", "unknown")
                team = sub.get("team", {}).get("name", "unknown")
                lines.append(
                    f"{on} on for {off} ({team}) at {s_min}' - "
                    f"note exact timestamp, starting position, and any immediate "
                    f"shape or line-height change.\n"
                )

    lines.append("SCAN INTERVAL: 1 SECOND for this window.\n")
    return "\n".join(lines)


def _build_rerun_prompt(item: dict, match_ctx: dict) -> str:
    """Build targeted re-run prompt from SKILL.md Step 3c template."""
    frame  = item.get("frame", "")
    score  = item.get("confidence_score", 0.0)
    flags  = item.get("flags", [])
    players = item.get("affected_players", []) or ["none specified"]
    metrics = item.get("affected_metrics", [])

    # Fix 33b: home/away naming -- no single focus team.
    focus   = match_ctx["home_team"]
    opp     = match_ctx["away_team"]
    fkit    = match_ctx["home_kit"]
    okit    = match_ctx["away_kit"]

    # Determine attack direction from window half — default to 1H
    atk_dir = match_ctx.get("attack_dir_1h", "left").upper()

    flag_guidance = []
    if any(f in flags for f in ("occlusion", "cluster")):
        flag_guidance.append(
            f"  - Can you confirm the position and team assignment of "
            f"{', '.join(players)}?\n"
            f"  - If still obscured, give your best positional estimate and why."
        )
    if "kit_ambiguity" in flags:
        flag_guidance.append(
            "  - Look for shirt number, hair, height, or context to confirm team "
            "assignment for ambiguous players. List your reasoning."
        )
    if "ball_not_visible" in flags:
        flag_guidance.append(
            "  - Can the ball position be inferred from player body orientation, "
            "gaze direction, or foot contact? State your estimate and confidence."
        )
    if "partial_frame" in flags:
        flag_guidance.append(
            "  - Estimate defensive line height from visible players only.\n"
            "    Note which players are off-frame and how that affects the estimate."
        )
    if any(f in flags for f in ("camera_motion", "low_contrast")):
        flag_guidance.append(
            "  - Use player silhouettes and pitch markings to estimate formation "
            "shape and defensive line. State which elements you can and cannot resolve."
        )

    if not flag_guidance:
        flag_guidance.append("  - Resolve the uncertainty and improve confidence on all affected metrics.")

    return f"""Re-analyse {frame}.

Previous Tier 1 scan was low confidence ({score:.2f}) due to: {', '.join(flags)}
Uncertain players: {', '.join(players)}
Affected metrics: {', '.join(metrics)}

{match_ctx['match']}: {focus} in {fkit}, {opp} in {okit}.
{focus} attack {atk_dir} in this half.

Examine this frame carefully and address only the following:

{chr(10).join(flag_guidance)}

Return ONLY a JSON object for this single frame:

{{
  "frame": "{frame}",
  "rerun": true,
  "confidence": {{
    "score": [0.0-1.0],
    "flags": [],
    "affected_players": [],
    "affected_metrics": [],
    "unresolvable": [true/false]
  }},
  "resolved_observations": {{
    "formation_shape": "[or null if unresolvable]",
    "pressing_score": [0-10 or null],
    "line_height_pct": [0-100 or null],
    "ball_visible": [true/false],
    "ball_zone": "[zone or null]",
    "player_positions": [
      {{"player": "#N", "confirmed": [true/false], "position_estimate": "[description]"}}
    ],
    "notes": "[resolution summary]"
  }}
}}"""


def _build_confirmation_prompt(item: dict, frame_names: list,
                               match_ctx: dict, source_profile: dict) -> str:
    """Build confirmation prompt from SKILL.md Step 3i template."""
    focus  = match_ctx["home_team"]  # Fix 33b: explicit home/away
    opp    = match_ctx["away_team"] if focus == match_ctx["home_team"] else match_ctx["home_team"]
    fkit   = match_ctx["home_kit"] if focus == match_ctx["home_team"] else match_ctx["away_kit"]
    okit   = match_ctx["away_kit"] if focus == match_ctx["home_team"] else match_ctx["home_kit"]

    fps        = item.get("escalation_target_fps", 3)
    event_type = item.get("event_type", item.get("result_family", ""))
    scope      = item.get("analysis_scope", "match")
    timestamp  = item.get("timestamp", "")
    w_start    = item.get("rerun_window_start", "")
    w_end      = item.get("rerun_window_end", "")
    reason     = item.get("reason", "")
    src_type   = source_profile.get("source_type", "unknown")

    frames_list = "\n".join(f"  {n}" for n in frame_names)

    return f"""Confirm this event. This is a short segment extracted at {fps}fps.
Source type: {src_type}

Event type:     {event_type}
Analysis scope: {scope}
Timestamp:      {timestamp}
Rerun window:   {w_start} to {w_end}
Question:       {reason}

MATCH: {focus} in {fkit}, {opp} in {okit}.

View frames in order:
{frames_list}

Answer ONLY the specific question. Do not re-analyse shape, pressing, or
territory - those are covered by the 1fps scan. Focus only on the fast or
ambiguous event described above.

Output ONLY raw JSON:

{{
  "timestamp": "{timestamp}",
  "event_type": "{event_type}",
  "analysis_scope": "{scope}",
  "confirmed_outcome": "[description of what happened - descriptive, no opinion]",
  "player_involved": "[#N Name or position label]",
  "confidence_after_rerun": 0.0,
  "evidence_tier": "escalated_confirmation",
  "result_family_status": "allowed",
  "recommended_report_wording": "[one sentence using descriptive language only]",
  "limitations_note": null,
  "frames_used": {json.dumps(frame_names)},
  "status": "confirmed"
}}"""


# ── extract_segment (ffmpeg-based, no cv2 dependency) ─────────────────────────

def extract_segment(video_path: str, timestamp_seconds: float,
                    out_dir: str, fps_target: int = 3,
                    window_seconds: int = 4) -> dict:
    """
    Extract a short segment around a timestamp at higher fps using ffmpeg.
    Returns {"success": bool, "frames": [paths], "count": N, "error": str}.
    """
    if not video_path or not os.path.exists(video_path):
        return {"success": False, "frames": [], "count": 0,
                "error": f"Video not found: {video_path}"}

    start_s  = max(0.0, timestamp_seconds - window_seconds / 2)
    duration = float(window_seconds)
    os.makedirs(out_dir, exist_ok=True)

    tmp_pattern = os.path.join(out_dir, "tmp_%04d.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_s),
        "-i",  video_path,
        "-t",  str(duration),
        "-vf", f"fps={fps_target}",
        "-q:v", "2",
        tmp_pattern,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return {"success": False, "frames": [], "count": 0,
                "error": result.stderr[-400:]}

    # Rename tmp_NNNN.jpg → confirm_XXmYYsN.jpg
    tmp_files   = sorted(glob.glob(os.path.join(out_dir, "tmp_*.jpg")))
    frame_paths = []
    for i, src in enumerate(tmp_files):
        t   = start_s + i / fps_target
        m, s = divmod(int(t), 60)
        ms  = int((t - int(t)) * 10)
        dst = os.path.join(out_dir, f"confirm_{m:02d}m{s:02d}s{ms}.jpg")
        os.rename(src, dst)
        frame_paths.append(dst)

    return {"success": True, "frames": frame_paths, "count": len(frame_paths)}


# ── Step 3b — Rerun queue builder (no dedicated module) ───────────────────────

def build_rerun_queue(match_dir: str, confidence_threshold: float = 0.7) -> dict:
    """
    Scan all agent_logs/*.json, flag windows with avg confidence below threshold.
    Writes rerun_queue.json.
    """
    logs_dir    = os.path.join(match_dir, "agent_logs")
    agent_files = sorted(glob.glob(os.path.join(logs_dir, "*.json")))
    agent_files = [f for f in agent_files
                   if "_merged" not in f and "_rerun" not in f]

    rerun_queue = []
    ok_windows  = []

    for path in agent_files:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        agent_id   = os.path.basename(path).replace(".json", "")
        confidence = (data.get("window_confidence", {}).get("overall_score")
                      or data.get("confidence_avg"))
        data_gap   = data.get("window_confidence", {}).get("data_gap_warning", False)
        window     = data.get("window", agent_id)

        if confidence is not None and confidence < confidence_threshold:
            rerun_queue.append({
                "agent_id":    agent_id,
                "source_file": os.path.basename(path),
                "window":      window,
                "confidence":  confidence,
                "reason":      f"below threshold ({confidence_threshold})",
                "data_gap":    data_gap,
                "status":      "pending",
            })
        else:
            ok_windows.append(agent_id)

    output = {
        "confidence_threshold": confidence_threshold,
        "windows_ok":           len(ok_windows),
        "rerun_required":       len(rerun_queue),
        "rerun_queue":          rerun_queue,
        "generated_at":         datetime.now().isoformat(),
    }

    out_path = os.path.join(match_dir, "rerun_queue.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"  Rerun queue: {len(rerun_queue)} windows flagged "
          f"(threshold {confidence_threshold})")
    for item in rerun_queue:
        print(f"    -> {item['agent_id']} (conf: {item['confidence']})")

    return output


CHUNK_SIZE = 75  # stay well under the Anthropic 100-image-per-request limit


# ═══════════════════════════════════════════════════════════════════════════════
# Chunked scanning helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_window_chunked(client, prompt_text: str, frame_paths: list,
                         model: str, agent_id: str, window_label: str) -> tuple:
    """
    Split a window's frames into CHUNK_SIZE batches. Call Claude once per chunk.
    Merges all chunk results into a single window dict.
    Returns (merged_dict, total_input_tokens, total_output_tokens).
    """
    chunks = [frame_paths[i:i + CHUNK_SIZE]
              for i in range(0, len(frame_paths), CHUNK_SIZE)]

    chunk_results = []
    for idx, chunk in enumerate(chunks):
        first_name = os.path.basename(chunk[0])
        last_name  = os.path.basename(chunk[-1])
        chunk_prompt = (prompt_text +
                        f"\n\nCHUNK {idx + 1} of {len(chunks)}: "
                        f"frames {first_name} to {last_name}")

        for attempt in range(2):
            try:
                text, n_in, n_out = call_claude(client, chunk_prompt, chunk, model)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"      [!] Chunk {idx + 1} attempt 1 failed: {e} — retrying 30s")
                    time.sleep(30)
                else:
                    print(f"      [FAIL] Chunk {idx + 1}: {e}")
                    n_in = n_out = 0
                    text = None

        if text is None:
            raise RuntimeError(f"Chunk {idx + 1} failed after retry")

        result = _parse_json_response(text, agent_id, window_label)
        result["chunk_index"]       = idx
        result["chunk_frame_count"] = len(chunk)
        # BUG 1 diagnostic: warn when a chunk returns no frames
        if not result.get("frames"):
            print(f"      [DBG] chunk {idx + 1} empty frames — raw[:200]: {text[:200]!r}")
        chunk_results.append((result, n_in, n_out))

        if idx < len(chunks) - 1:
            time.sleep(1)  # brief pause between chunks

    merged    = _merge_chunks(chunk_results, agent_id, window_label)
    total_in  = sum(t[1] for t in chunk_results)
    total_out = sum(t[2] for t in chunk_results)
    return merged, total_in, total_out


def _merge_chunks(chunk_results: list, agent_id: str, window_label: str) -> dict:
    """
    Merge chunk dicts into a single window output.
    Lists are concatenated; scalars are averaged or taken from boundary chunks.
    """
    if not chunk_results:
        return {"error": "no chunks", "agent_id": agent_id, "window": window_label,
                "findings": [], "frames": []}

    results = [r for r, _, _ in chunk_results]
    first   = results[0]

    # BUG 1 guard: if every chunk returned empty frames, the API produced no
    # frame-level output — return a distinguishable error record rather than
    # silently writing an all-empty file.
    if not any(r.get("frames") for r in results):
        print(f"  [WARN] all {len(results)} chunks returned empty frames for {agent_id} / {window_label}")
        return {
            "agent_id":    agent_id,
            "window":      window_label,
            "error":       "all_chunks_empty",
            "frames":      [],
            "findings":    [],
            "key_moments": [],
            "shot_attempts": [],
            "set_pieces":  [],
            "pass_sequences": [],
            "individual_observations": [],
            "flaggable_moments": [],
            "pressing":    {"scores": [], "avg_score": None, "peak_score": None, "peak_timestamp": None},
            "defensive_line": {"avg_pct": None, "start_pct": None, "end_pct": None, "notable_shifts": []},
            "formation":   {"shape_in_possession": None, "shape_out_of_possession": None, "compactness": None},
            "possession_summary": {},
            "confirmation_queue": [],
            "window_confidence": {"overall_score": None, "low_confidence_frame_count": 0, "data_gap_warning": True},
            "window_summary": " | ".join(f"[Chunk {i+1}] {r.get('window_summary','')}"
                                         for i, r in enumerate(results)),
            "chunks_processed": len(results),
            "frames_reviewed": 0,
            "merge_type": "chunked_1fps",
        }

    def most_common(values):
        values = [v for v in values if v]
        return max(set(values), key=values.count) if values else None

    def safe_mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 2) if values else None

    def dedup_by_ts(items, tolerance_s=30):
        # BUG 2 fix: _parse_ts returns None for unparseable timestamps;
        # arithmetic abs(None - None) raises TypeError. Guard here.
        seen, out = [], []
        for item in items:
            secs = _parse_ts(item.get("timestamp", ""))
            if secs is None:
                out.append(item)   # can't deduplicate without a timestamp; include as-is
                continue
            if not any(abs(secs - s) <= tolerance_s for s in seen):
                seen.append(secs)
                out.append(item)
        return out

    # Concatenated lists
    all_frames    = []
    all_findings  = []
    all_cq        = []
    all_moments   = []
    all_shots     = []
    all_setpieces = []
    all_obs       = []
    all_flags     = []
    all_passes    = []
    all_pressing  = []
    all_shifts    = []

    for r in results:
        all_frames.extend(r.get("frames", []))
        all_findings.extend(r.get("findings", []))
        all_cq.extend(r.get("confirmation_queue", []))
        all_moments.extend(r.get("key_moments", []))
        all_shots.extend(r.get("shot_attempts", []))
        all_setpieces.extend(r.get("set_pieces", []))
        all_obs.extend(r.get("individual_observations", []))
        all_flags.extend(r.get("flaggable_moments", []))
        all_passes.extend(r.get("pass_sequences", []))
        all_pressing.extend(r.get("pressing", {}).get("scores", []))
        all_shifts.extend(r.get("defensive_line", {}).get("notable_shifts", []))

    # Deduplicate findings by (result_family, time_start)
    seen_keys, deduped_findings = set(), []
    for f in all_findings:
        key = (f.get("result_family"), f.get("time_start"))
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_findings.append(f)

    press_avgs  = [r.get("pressing", {}).get("avg_score")  for r in results]
    press_peaks = [r.get("pressing", {}).get("peak_score") for r in results]
    dl_avgs     = [r.get("defensive_line", {}).get("avg_pct") for r in results]
    shapes_ip   = [r.get("formation", {}).get("shape_in_possession")     for r in results]
    shapes_oop  = [r.get("formation", {}).get("shape_out_of_possession") for r in results]
    conf_scores = [r.get("window_confidence", {}).get("overall_score") for r in results]
    conf_low    = sum(r.get("window_confidence", {}).get("low_confidence_frame_count", 0)
                      for r in results)
    conf_gap    = any(r.get("window_confidence", {}).get("data_gap_warning", False)
                      for r in results)
    summaries   = [f"[Chunk {i + 1}] {r.get('window_summary', '')}"
                   for i, r in enumerate(results)]

    return {
        "agent_id":             agent_id,
        "window":               window_label,
        "frames_reviewed":      sum(r.get("frames_reviewed", len(r.get("frames", [])))
                                    for r in results),
        "chunks_processed":     len(results),
        "merge_type":           "chunked_1fps",
        "frames":               all_frames,
        "findings":             deduped_findings,
        "confirmation_queue":   dedup_by_ts(all_cq, 30),
        "key_moments":          dedup_by_ts(all_moments, 30),
        "shot_attempts":        dedup_by_ts(all_shots, 15),
        "set_pieces":           dedup_by_ts(all_setpieces, 15),
        "individual_observations": all_obs,
        "flaggable_moments":    all_flags,
        "pass_sequences":       all_passes,
        "pressing": {
            "scores":           all_pressing,
            "avg_score":        safe_mean(press_avgs),
            "peak_score":       max((p for p in press_peaks if p is not None), default=None),
            "peak_timestamp":   None,
        },
        "defensive_line": {
            "avg_pct":          safe_mean(dl_avgs),
            "start_pct":        results[0].get("defensive_line", {}).get("start_pct"),
            "end_pct":          results[-1].get("defensive_line", {}).get("end_pct"),
            "notable_shifts":   all_shifts,
        },
        "formation": {
            "shape_in_possession":     most_common(shapes_ip),
            "shape_out_of_possession": most_common(shapes_oop),
            "compactness":             first.get("formation", {}).get("compactness"),
        },
        "possession_summary":   first.get("possession_summary", {}),
        "window_confidence": {
            "overall_score":              safe_mean(conf_scores),
            "low_confidence_frame_count": conf_low,
            "data_gap_warning":           conf_gap,
        },
        "window_summary": " | ".join(s for s in summaries if s.strip() != "[Chunk 1] "),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3a — Tier 1 scan
# ═══════════════════════════════════════════════════════════════════════════════

def run_tier1_scan(match_dir: str, model: str = "claude-sonnet-4-6",
                   api_key: str = None) -> dict:
    """
    Run Tier 1 (1fps) scan for every window in window_plan.json.
    Skips windows that already have an output file in agent_logs/.
    Returns {"windows_complete": N, "windows_failed": N, "total_tokens": N}.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set and no api_key provided")

    client = anthropic.Anthropic(api_key=api_key)

    plan_path = os.path.join(match_dir, "window_plan.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"window_plan.json not found in {match_dir}")

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    windows   = plan.get("windows", [])
    total     = len(windows)
    logs_dir  = os.path.join(match_dir, "agent_logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Load shared context once
    match_ctx        = _load_match_context(match_dir)
    source_injection = _build_source_injection(match_dir)
    match_data_block = ""
    mdb_path = os.path.join(match_dir, "match_data_block.txt")
    if os.path.exists(mdb_path):
        with open(mdb_path, encoding="utf-8") as f:
            match_data_block = f.read()

    complete = 0
    failed   = 0
    total_in = 0
    total_out = 0
    _resize_logged = False  # print actual post-resize dims once per run

    for i, window in enumerate(windows):
        agent_id = window.get("agent_id", window.get("id", f"{i+1:02d}"))
        label    = window.get("label", agent_id)
        safe_label = label.replace(" ", "_").replace(":", "-").replace("\u2013", "-").replace("\u2014", "-")

        out_path = os.path.join(logs_dir, f"agent_{agent_id}_{safe_label}.json")
        if os.path.exists(out_path):
            print(f"  [{i+1:02d}/{total}] {label} | SKIPPED (exists)")
            complete += 1
            continue

        frames = _collect_frames(
            match_dir,
            window.get("start_frame", ""),
            window.get("end_frame", ""),
        )
        missing = window.get("frame_count", 0) - len(frames)
        est_tpf = _estimate_tokens_per_frame(frames)

        # One-shot: print actual post-resize dimensions on first real frame
        if not _resize_logged and frames:
            try:
                with Image.open(frames[0], encoding="utf-8") as _img:
                    orig_w, orig_h = _img.size
                new_w = min(orig_w, 640)
                new_h = int(orig_h * new_w / orig_w) if orig_w > 640 else orig_h
                print(f"  [resize check] {os.path.basename(frames[0])}: "
                      f"original {orig_w}x{orig_h} -> resized {new_w}x{new_h} "
                      f"(max_width=640, ~{(new_w * new_h) // 750} tokens/frame)")
                _resize_logged = True
            except Exception as _e:
                print(f"  [resize check] could not open {os.path.basename(frames[0])}: {_e}")

        n_chunks  = max(1, (len(frames) + CHUNK_SIZE - 1) // CHUNK_SIZE)
        est_k     = (min(len(frames), CHUNK_SIZE) * est_tpf + 3000) // 1000
        m_short   = model.replace("claude-", "")
        print(f"  [{i+1:02d}/{total}] {label} | {len(frames)} frames | "
              f"{n_chunks} chunks | ~{est_k}k tokens/chunk | {m_short}")

        prompt = _build_tier1_prompt(
            match_dir, window, match_ctx, source_injection, match_data_block
        )

        try:
            data, n_in, n_out = _scan_window_chunked(
                client, prompt, frames, model, agent_id, label
            )
            total_in  += n_in
            total_out += n_out
        except Exception as e:
            print(f"    [FAIL] {label}: {e}")
            err_record = {
                "agent_id": agent_id, "window": label,
                "status": "failed", "error": str(e),
                "findings": [], "frames": [],
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(err_record, f, indent=2)
            failed += 1
            time.sleep(2)
            continue

        data["agent_id"]      = agent_id
        data["window"]        = label
        data["missing_frames"] = missing
        data["scanned_at"]    = datetime.now().isoformat()
        data["model"]         = model

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        status = "parse_failed" if data.get("status") == "parse_failed" else "ok"
        if status == "ok":
            complete += 1
        else:
            failed += 1

        conf = data.get("window_confidence", {}).get("overall_score", "?")
        print(f"    done | actual: {n_in}+{n_out} tok | conf: {conf} | {status}")

        time.sleep(2)

    print(f"\n  Tier 1 complete: {complete}/{total} ok | {failed} failed "
          f"| tokens: {total_in}in/{total_out}out")

    return {
        "windows_complete": complete,
        "windows_failed":   failed,
        "total_tokens":     total_in + total_out,
        "input_tokens":     total_in,
        "output_tokens":    total_out,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3d — Deep scan (event windows only)
# ═══════════════════════════════════════════════════════════════════════════════

def run_deep_scan(match_dir: str, model: str = "claude-opus-4-5",
                  api_key: str = None) -> dict:
    """
    Run dual-agent deep scan for all windows where deep_scan=True.
    Agent A and Agent B are independent calls on the same prompt + frames.
    Returns {"deep_scans_complete": N, "deep_scans_failed": N}.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set and no api_key provided")

    client = anthropic.Anthropic(api_key=api_key)

    plan_path = os.path.join(match_dir, "window_plan.json")
    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"window_plan.json not found in {match_dir}")

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    event_windows = [w for w in plan.get("windows", []) if w.get("deep_scan")]
    if not event_windows:
        print("  No event windows marked for deep scan.")
        return {"deep_scans_complete": 0, "deep_scans_failed": 0}

    logs_dir  = os.path.join(match_dir, "agent_logs")
    os.makedirs(logs_dir, exist_ok=True)

    match_ctx        = _load_match_context(match_dir)
    source_injection = _build_source_injection(match_dir)
    match_data_block = ""
    mdb_path = os.path.join(match_dir, "match_data_block.txt")
    if os.path.exists(mdb_path):
        with open(mdb_path, encoding="utf-8") as f:
            match_data_block = f.read()

    complete = 0
    failed   = 0

    for window in event_windows:
        agent_id   = window.get("agent_id", window.get("id", "NN"))
        label      = window.get("label", agent_id)
        safe_label = label.replace(" ", "_").replace(":", "-").replace("\u2013", "-").replace("\u2014", "-")

        a_path = os.path.join(logs_dir, f"agent_{agent_id}_{safe_label}.json")
        b_path = os.path.join(logs_dir, f"agent_{agent_id}_agentB.json")

        frames = _collect_frames(
            match_dir,
            window.get("start_frame", ""),
            window.get("end_frame", ""),
        )

        # BUG 3 diagnostic: print first/last frame so miscollection is visible
        if frames:
            print(f"  [{agent_id}] frames: {os.path.basename(frames[0])} "
                  f"-> {os.path.basename(frames[-1])} ({len(frames)} total)")
        else:
            print(f"  [{agent_id}] WARNING: no frames collected for {label} "
                  f"(start_frame={window.get('start_frame')!r}, "
                  f"end_frame={window.get('end_frame')!r})")

        # Build prompt: Tier 1 base + deep scan additions
        base_prompt = _build_tier1_prompt(
            match_dir, window, match_ctx, source_injection, match_data_block
        )
        deep_additions = _build_deep_scan_additions(window, match_ctx)
        prompt = base_prompt + "\n" + deep_additions

        window_ok = True

        n_chunks = max(1, (len(frames) + CHUNK_SIZE - 1) // CHUNK_SIZE)
        m_short  = model.replace("claude-", "")
        print(f"  [{agent_id}] {label} | {len(frames)} frames | "
              f"{n_chunks} chunks | {m_short} (deep scan)")

        for agent_label, out_path in [("A", a_path), ("B", b_path)]:
            if os.path.exists(out_path):
                # BUG 3 fix: skip only if the existing file is not a failure/empty record
                try:
                    with open(out_path, encoding="utf-8") as _f:
                        _existing = json.load(_f)
                    _is_bad = (
                        _existing.get("status") == "failed"
                        or _existing.get("error") in ("all_chunks_empty",)
                        or (_existing.get("frames") == [] and
                            _existing.get("window_confidence", {}).get("overall_score") is None)
                    )
                except Exception:
                    _is_bad = True
                if not _is_bad:
                    print(f"  [{agent_id}] Agent {agent_label} SKIPPED (exists)")
                    continue
                print(f"  [{agent_id}] Agent {agent_label} re-running (previous output was bad: "
                      f"{_existing.get('error') or _existing.get('status')})")

            a_id = f"{agent_id}{agent_label.lower()}"
            try:
                data, n_in, n_out = _scan_window_chunked(
                    client, prompt, frames, model, a_id, label
                )
            except Exception as e:
                print(f"    [FAIL] [{agent_id}] Agent {agent_label}: {e}")
                err_record = {
                    "agent_id": a_id, "window": label,
                    "status": "failed", "error": str(e),
                    "findings": [], "frames": [],
                }
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(err_record, f, indent=2)
                window_ok = False
                time.sleep(2)
                continue

            data["agent_id"]        = a_id
            data["window"]          = label
            data["deep_scan"]       = True
            data["agent_pass"]      = agent_label
            data["scanned_at"]      = datetime.now().isoformat()
            data["model"]           = model

            # BUG 1 diagnostic: surface empty results before write
            conf = data.get("window_confidence", {}).get("overall_score")
            n_frames   = len(data.get("frames", []))
            n_moments  = len(data.get("key_moments", []))
            print(f"    Agent {agent_label} result: frames={n_frames}, "
                  f"moments={n_moments}, confidence={conf}")
            if data.get("error") == "all_chunks_empty":
                print(f"    [WARN] Agent {agent_label} all-chunks-empty — "
                      f"file written but will be treated as bad on next run")

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            print(f"    Agent {agent_label} done | {n_in}+{n_out} tok | conf: {conf}")

            time.sleep(2)

        if window_ok:
            # Compare agreement on key moments
            a_data = {}
            if os.path.exists(a_path) and os.path.exists(b_path):
                with open(a_path, encoding="utf-8") as f: a_data = json.load(f)
                with open(b_path, encoding="utf-8") as f: b_data = json.load(f)
                a_ts = {m["timestamp"] for m in a_data.get("key_moments", [])}
                b_ts = {m["timestamp"] for m in b_data.get("key_moments", [])}
                agreements = len(a_ts & b_ts)
                a_conf = a_data.get("window_confidence", {}).get("overall_score", "?")
                b_conf = b_data.get("window_confidence", {}).get("overall_score", "?")
                print(f"  [{agent_id}] Deep scan complete | A conf: {a_conf} "
                      f"| B conf: {b_conf} | agreements: {agreements}")
            complete += 1
        else:
            failed += 1

    print(f"\n  Deep scan complete: {complete} ok | {failed} failed")
    return {"deep_scans_complete": complete, "deep_scans_failed": failed}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3c — Targeted reruns
# ═══════════════════════════════════════════════════════════════════════════════

def run_targeted_reruns(match_dir: str, model: str = "claude-sonnet-4-6",
                        api_key: str = None) -> dict:
    """
    Process all queued items in rerun_queue.json with targeted re-run prompts.
    Each call sends 5 frames (2 before, flagged frame, 2 after).
    Returns {"resolved": N, "unresolvable": N}.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set and no api_key provided")

    client = anthropic.Anthropic(api_key=api_key)

    rq_path = os.path.join(match_dir, "rerun_queue.json")
    if not os.path.exists(rq_path):
        print("  [X] rerun_queue.json not found — run step 3b first")
        return {"resolved": 0, "unresolvable": 0}

    with open(rq_path, encoding="utf-8") as f:
        rq = json.load(f)

    queued_items = [i for i in rq.get("rerun_queue", [])
                    if i.get("status") == "queued"]

    if not queued_items:
        print("  No queued items in rerun_queue.json")
        return {"resolved": 0, "unresolvable": 0}

    logs_dir  = os.path.join(match_dir, "agent_logs")
    match_ctx = _load_match_context(match_dir)

    # Resolve frames directory
    frames_dir = os.path.join(match_dir, "frames", "frames_detailed")
    if not os.path.isdir(frames_dir):
        frames_dir = os.path.join(match_dir, "frames")

    resolved     = 0
    unresolvable = 0

    for item in queued_items:
        frame_name  = item.get("frame", "")
        agent_id    = item.get("agent_id", "")
        source_file = item.get("source_file", "").replace(".json", "")

        # Collect 5-frame context window: 2 before, target, 2 after
        all_frames  = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
        frame_names = [os.path.basename(p) for p in all_frames]
        try:
            idx = frame_names.index(frame_name)
        except ValueError:
            idx = -1

        if idx >= 0:
            context_paths = [all_frames[j] for j in
                             range(max(0, idx - 2), min(len(all_frames), idx + 3))]
        else:
            context_paths = []
            print(f"  [!] Frame not found: {frame_name}")

        prompt   = _build_rerun_prompt(item, match_ctx)
        out_path = os.path.join(logs_dir, f"{source_file}_rerun.json")

        for attempt in range(2):
            try:
                text, n_in, n_out = call_claude(client, prompt, context_paths, model)
                break
            except Exception as e:
                if attempt == 0:
                    print(f"    [!] Rerun attempt 1 failed: {e} — retrying 30s")
                    time.sleep(30)
                else:
                    print(f"    [FAIL] {frame_name}: {e}")
                    text = None

        if text is None:
            item["status"] = "unresolvable"
            unresolvable += 1
            continue

        data = _parse_json_response(text, agent_id, frame_name)
        data["source_item"] = item
        data["rerun_at"]    = datetime.now().isoformat()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Determine resolution
        post_conf = data.get("confidence", {}).get("score", 0.0)
        still_bad = data.get("confidence", {}).get("unresolvable", False)

        if still_bad or (isinstance(post_conf, (int, float)) and post_conf < 0.7):
            item["status"] = "unresolvable"
            unresolvable += 1
            resolved_flag = False
        else:
            item["status"] = "resolved"
            resolved += 1
            resolved_flag = True

        flags_str = ", ".join(item.get("flags", []))
        m_short   = model.replace("claude-", "")
        print(f"  [{agent_id}] {frame_name} | flags: [{flags_str}] "
              f"| resolved: {resolved_flag} | conf: {post_conf} | {m_short}")

        time.sleep(2)

    # Write updated queue
    with open(rq_path, "w", encoding="utf-8") as f:
        json.dump(rq, f, indent=2)

    print(f"\n  Reruns complete: {resolved} resolved | {unresolvable} unresolvable")
    return {"resolved": resolved, "unresolvable": unresolvable}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3i — Confirmation segment
# ═══════════════════════════════════════════════════════════════════════════════

def run_confirmation_segment(match_dir: str, item: dict,
                             model: str = "claude-opus-4-5",
                             api_key: str = None,
                             video_path: str = None) -> dict:
    """
    Run higher-fps confirmation for one item from confirmation_queue.json.
    Extracts segment via ffmpeg, sends to Claude, returns confirmation result dict.
    """
    import anthropic

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not set")

    client = anthropic.Anthropic(api_key=api_key)

    timestamp  = item.get("timestamp", "00m00s")
    fps_target = item.get("escalation_target_fps", 3)
    w_start    = item.get("rerun_window_start", timestamp)
    w_end      = item.get("rerun_window_end", timestamp)

    # Extract segment
    ts_seconds  = _timestamp_to_seconds(timestamp)
    ts_start    = _timestamp_to_seconds(w_start)
    ts_end      = _timestamp_to_seconds(w_end)
    window_secs = int(ts_end - ts_start) or 4

    conf_dir = os.path.join(match_dir, "confirmations",
                             f"confirm_{timestamp.replace(':', '').replace('m', '_').replace('s', '')}")

    seg_result = extract_segment(
        video_path    = video_path,
        timestamp_seconds = ts_start if ts_start > 0 else ts_seconds,
        out_dir       = conf_dir,
        fps_target    = fps_target,
        window_seconds = window_secs,
    )

    if not seg_result["success"]:
        return {
            "timestamp":    timestamp,
            "event_type":   item.get("event_type", ""),
            "status":       "extraction_failed",
            "error":        seg_result.get("error", "unknown"),
            "frames_used":  [],
        }

    frame_paths = seg_result["frames"]
    frame_names = [os.path.basename(p) for p in frame_paths]

    match_ctx      = _load_match_context(match_dir)
    source_profile = _load(match_dir, "source_profile.json", {})
    prompt = _build_confirmation_prompt(item, frame_names, match_ctx, source_profile)

    for attempt in range(2):
        try:
            text, n_in, n_out = call_claude(client, prompt, frame_paths, model,
                                             max_tokens=2048, max_width=960)
            break
        except Exception as e:
            if attempt == 0:
                print(f"    [!] Confirmation attempt 1 failed: {e} — retrying 30s")
                time.sleep(30)
            else:
                return {
                    "timestamp":  timestamp,
                    "event_type": item.get("event_type", ""),
                    "status":     "api_failed",
                    "error":      str(e),
                    "frames_used": frame_names,
                }

    result = _parse_json_response(text, "confirm", timestamp)
    result.setdefault("timestamp",   timestamp)
    result.setdefault("event_type",  item.get("event_type", ""))
    result.setdefault("status",      "confirmed")
    result.setdefault("frames_used", frame_names)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# API key resolution
# ═══════════════════════════════════════════════════════════════════════════════

def _resolve_api_key(match_dir: str = None) -> str | None:
    """
    Resolve the Anthropic API key from multiple sources in priority order:
      1. Environment variable ANTHROPIC_API_KEY
      2. .env in match_dir
      3. .env in scripts directory (same dir as this file)
      4. .env in user home directory

    Returns the key (str) or None if not found.
    Prints a masked confirmation of the source used.
    """
    def _read_dotenv(path: str) -> str | None:
        """Parse KEY=VALUE lines from a .env file. Returns value for ANTHROPIC_API_KEY."""
        try:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    if k.strip() == "ANTHROPIC_API_KEY":
                        return v.strip().strip('"').strip("'")
        except OSError:
            pass
        return None

    def _try_dotenv_lib(path: str) -> str | None:
        """Try python-dotenv if available; falls back to manual parser."""
        try:
            from dotenv import dotenv_values
            vals = dotenv_values(path)
            return vals.get("ANTHROPIC_API_KEY")
        except ImportError:
            return _read_dotenv(path)

    def _mask(key: str) -> str:
        if len(key) > 8:
            return key[:8] + "..." + key[-4:]
        return "****"

    # 1 — environment
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        print(f"  API key loaded from environment: {_mask(key)}")
        return key

    # 2 — match directory .env
    if match_dir:
        path = os.path.join(match_dir, ".env")
        key = _try_dotenv_lib(path)
        if key:
            print(f"  API key loaded from match dir .env: {_mask(key)}")
            return key

    # 3 — scripts directory .env (same dir as this file)
    scripts_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    key = _try_dotenv_lib(scripts_env)
    if key:
        print(f"  API key loaded from scripts .env: {_mask(key)}")
        return key

    # 4 — home directory .env
    home_env = os.path.join(os.path.expanduser("~"), ".env")
    key = _try_dotenv_lib(home_env)
    if key:
        print(f"  API key loaded from home .env: {_mask(key)}")
        return key

    print("  [!] ANTHROPIC_API_KEY not found in environment or any .env file")
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# PipelineRunner — orchestrator class
# ═══════════════════════════════════════════════════════════════════════════════

class PipelineRunner:
    """
    Orchestrates the full Match Lens pipeline for a single match directory.

    Programmatic steps run directly (including Claude API steps 3a/3c/3d/3i).
    Remaining agent steps (1b, 1d, 1e, 4) require manual Claude Code sessions.
    """

    PROGRAMMATIC_STEPS = [
        "1_extract",
        "1f_source",
        "1c_window_plan",
        "2_match_data",
        "3a_tier1",
        "3b_rerun_queue",
        "3c_reruns",
        "3d_deep_scan",
        "3e_merge",
        "3f_pass",
        "3g_summary",
        "3h_ground_truth",
        "3i_escalation",
        "3j_readiness",
        "3k_metrics",
        "5_convert",
    ]

    AGENT_STEPS = [
        "1b_boundaries",
        "1d_team_sheet",
        "1e_attack_dir",
        "4_reports",
    ]

    def __init__(self, match_dir: str, video_path: str = None,
                 team_label: str = "both", api_key: str = None):
        self.match_dir  = os.path.abspath(match_dir)
        self.video_path = video_path
        # Fix 33b: parameter renamed from legacy focus name.
        self.team_label = team_label
        self.api_key    = api_key or _resolve_api_key(self.match_dir)
        self.frames_dir = os.path.join(self.match_dir, "frames")
        self.logs_dir   = os.path.join(self.match_dir, "agent_logs")

    # ── Status ────────────────────────────────────────────────────────────────

    def status(self):
        """Print current pipeline state."""
        _banner(f"Pipeline status: {os.path.basename(self.match_dir)}")

        checks = [
            (os.path.isdir(self.frames_dir),
             "Step 1:   frames extracted"),
            (os.path.exists(self._p("source_profile.json")),
             "Step 1f:  source classified"),
            (os.path.exists(self._p("match_boundaries.json")),
             "Step 1b:  boundaries detected"),
            (os.path.exists(self._p("window_plan.json")),
             "Step 1c:  window plan built"),
            (self._config_verified(),
             "Step 1d:  team sheet verified"),
            (os.path.exists(self._p("match_data_block.txt")),
             "Step 2:   match data block written"),
            (self._agent_log_count() > 0,
             f"Step 3a:  tier-1 scan ({self._agent_log_count()} logs)"),
            (os.path.exists(self._p("rerun_queue.json")),
             "Step 3b:  rerun queue built"),
            (self._merged_count() > 0,
             f"Step 3e:  merge ({self._merged_count()} merged)"),
            (os.path.exists(self._p("pass_sequences.json")),
             "Step 3f:  pass sequences accumulated"),
            (self._summary_windows() > 0,
             f"Step 3g:  running summary ({self._summary_windows()} windows)"),
            (os.path.exists(self._p("ground_truth_check.json")),
             "Step 3h:  ground truth validated"),
            (os.path.exists(self._p("confirmation_queue.json")),
             "Step 3i:  escalation queue built"),
            (os.path.exists(self._p("report_readiness.json")),
             "Step 3j:  readiness checked"),
            (os.path.exists(self._p("deep_skill_metrics.json")),
             "Step 3k:  deep metrics computed"),
            (self._report_count() > 0,
             f"Step 4:   reports written ({self._report_count()} .md files)"),
            (self._docx_count() > 0,
             f"Step 5:   Word conversion ({self._docx_count()} .docx files)"),
        ]

        for done, label in checks:
            tick = "[OK]" if done else "."
            print(f"  {tick}  {label}")

        print()
        self._print_next_action()

    def _print_next_action(self):
        if not os.path.isdir(self.frames_dir):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 1_extract --video VIDEO")
        elif not self._config_verified():
            print("  NEXT: Complete Steps 1b -> 1e (agent steps - see SKILL.md)")
        elif not os.path.exists(self._p("match_data_block.txt")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 2_match_data")
        elif self._agent_log_count() == 0:
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3a_tier1")
        elif not os.path.exists(self._p("rerun_queue.json")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3b_rerun_queue")
        elif self._merged_count() == 0:
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3e_merge")
        elif self._summary_windows() == 0:
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3g_summary")
        elif not os.path.exists(self._p("ground_truth_check.json")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3h_ground_truth")
        elif not os.path.exists(self._p("confirmation_queue.json")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3i_escalation")
        elif not os.path.exists(self._p("report_readiness.json")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3j_readiness")
        elif not os.path.exists(self._p("deep_skill_metrics.json")):
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 3k_metrics")
        elif self._report_count() == 0:
            print("  NEXT: Run Step 4 (agent step - report writing, see SKILL.md)")
        elif self._docx_count() == 0:
            print("  NEXT: python pipeline_runner.py MATCH_DIR --step 5_convert")
        else:
            print("  Pipeline complete.")

    # ── Helper checks ─────────────────────────────────────────────────────────

    def _p(self, fname: str) -> str:
        return os.path.join(self.match_dir, fname)

    def _config_verified(self) -> bool:
        c = _load(self.match_dir, "match_config.json")
        return bool(c and c.get("verified"))

    def _agent_log_count(self) -> int:
        if not os.path.isdir(self.logs_dir):
            return 0
        return len([f for f in os.listdir(self.logs_dir)
                    if f.endswith(".json") and "_merged" not in f
                    and "_rerun" not in f])

    def _merged_count(self) -> int:
        return len(glob.glob(os.path.join(self.logs_dir, "*_merged.json")))

    def _summary_windows(self) -> int:
        s = _load(self.match_dir, "running_summary.json")
        return (s or {}).get("windows_complete", 0)

    def _report_count(self) -> int:
        return len(glob.glob(os.path.join(self.match_dir, "*.md")))

    def _docx_count(self) -> int:
        return len(glob.glob(os.path.join(self.match_dir, "*.docx")))

    # ── Step dispatcher ───────────────────────────────────────────────────────

    def run_step(self, step_id: str) -> bool:
        """Run a named step. Returns True on success."""
        dispatch = {
            # Programmatic
            "1_extract":       self.step_1_extract,
            "1f_source":       self.step_1f_source,
            "1c_window_plan":  self.step_1c_window_plan,
            "2_match_data":    self.step_2_match_data,
            "3a_tier1":        self.step_3a_tier1,
            "3b_rerun_queue":  self.step_3b_rerun_queue,
            "3c_reruns":       self.step_3c_reruns,
            "3d_deep_scan":    self.step_3d_deep_scan,
            "3e_merge":        self.step_3e_merge,
            "3f_pass":         self.step_3f_pass,
            "3g_summary":      self.step_3g_summary,
            "3h_ground_truth": self.step_3h_ground_truth,
            "3i_escalation":   self.step_3i_escalation,
            "3j_readiness":    self.step_3j_readiness,
            "3k_metrics":      self.step_3k_metrics,
            "5_convert":       self.step_5_convert,
            # Agent-info
            "1b_boundaries":   lambda: self._agent_info("1b_boundaries"),
            "1d_team_sheet":   lambda: self._agent_info("1d_team_sheet"),
            "1e_attack_dir":   lambda: self._agent_info("1e_attack_dir"),
            "4_reports":       lambda: self._agent_info("4_reports"),
        }
        fn = dispatch.get(step_id)
        if fn is None:
            print(f"  Unknown step: {step_id}")
            print(f"  Valid: {', '.join(sorted(dispatch))}")
            return False
        return fn()

    def auto(self):
        """Run all programmatic steps in order."""
        _banner("Auto-running programmatic steps")
        results = {}
        for step in self.PROGRAMMATIC_STEPS:
            print(f"\n  --- {step} ---")
            try:
                ok = self.run_step(step)
                results[step] = "ok" if ok else "skipped"
            except Exception as e:
                print(f"  [FAIL] {step}: {e}")
                results[step] = f"error: {e}"
        print("\n  Summary:")
        for step, result in results.items():
            print(f"    {step}: {result}")

    # =========================================================================
    # Programmatic step implementations
    # =========================================================================

    def step_1_extract(self) -> bool:
        """Step 1 — Extract 1fps frames from the video via ffmpeg."""
        if not self.video_path:
            print("  [X] --video required for step 1_extract")
            return False
        if not os.path.exists(self.video_path):
            print(f"  [X] Video not found: {self.video_path}")
            return False

        _banner("Step 1 — Frame extraction (1fps via ffmpeg)")
        detailed_dir = os.path.join(self.frames_dir, "frames_detailed")
        os.makedirs(detailed_dir, exist_ok=True)

        tmp_pattern = os.path.join(detailed_dir, "tmp_%06d.jpg")
        cmd = ["ffmpeg", "-y", "-i", self.video_path,
               "-vf", "fps=1", "-q:v", "2", tmp_pattern]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [X] ffmpeg failed:\n{result.stderr[-500:]}")
            return False

        tmp_files = sorted(glob.glob(os.path.join(detailed_dir, "tmp_*.jpg")))
        for i, src in enumerate(tmp_files):
            m, s = divmod(i, 60)
            dst = os.path.join(detailed_dir, f"frame_{m:02d}m{s:02d}s.jpg")
            os.replace(src, dst)

        count = len(glob.glob(os.path.join(detailed_dir, "frame_*.jpg")))
        print(f"  Frames extracted: {count} (frames_detailed/)")
        return count > 0

    def step_1f_source(self) -> bool:
        """Step 1f — Sample frames and print source classification prompt."""
        if not self.video_path:
            print("  [X] --video required for step 1f_source")
            return False

        _banner("Step 1f — Source classification")
        sample_dir = os.path.join(self.frames_dir, "source_samples")
        samples = sample_frames(self.video_path, sample_dir)

        print("\n" + "=" * 60)
        print("SOURCE CLASSIFICATION PROMPT — send to Claude with these frames:")
        print("=" * 60)
        print(CLASSIFICATION_PROMPT)
        print("\nFrames to include:")
        for s in samples:
            print(f"  {s}")
        print("\nThen call: build_source_profile(match_dir, <returned JSON>)")
        return len(samples) > 0

    def step_1c_window_plan(self) -> bool:
        """Step 1c — Build window_plan.json from match_boundaries.json."""
        _banner("Step 1c — Window plan")
        boundary_file = self._p("match_boundaries.json")
        if not os.path.exists(boundary_file):
            print("  [X] match_boundaries.json not found — run Step 1b first")
            return False

        config = _load(self.match_dir, "match_config.json", {})
        event_windows = {}
        goal_windows  = config.get("goal_windows") or []
        if goal_windows:
            event_windows["goal"] = goal_windows

        plan = build_window_plan(self.match_dir, event_windows or None)
        print(f"  window_plan.json written: {plan['total_windows']} windows")
        return True

    def step_2_match_data(self) -> bool:
        """Step 2 — Write match_data_block.txt and initialise accumulators."""
        _banner("Step 2 — Match data block")

        cfg_path = self._p("match_config.json")
        if not os.path.exists(cfg_path):
            print("  [X] match_config.json missing")
            return False
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)

        if not cfg.get("verified"):
            print("  [X] match_config.json not verified — complete Step 1d first")
            return False

        home = cfg.get("home_team", "Home")
        away = cfg.get("away_team", "Away")
        match_label = cfg.get("match", f"{home} vs {away}")
        # Fix 33b: legacy focus removed; downstream init writes home/away.
        date  = cfg.get("date", "")
        comp  = cfg.get("competition", "")
        venue = cfg.get("venue", "")
        result = cfg.get("ft_score", "")
        ad1   = cfg.get("attack_direction_1h") or "unknown"
        ad2   = cfg.get("attack_direction_2h") or "unknown"
        home_kit    = cfg.get("home_kit", "unknown")
        away_kit    = cfg.get("away_kit", "unknown")
        home_gk_kit = cfg.get("home_gk_kit", "unknown")
        away_gk_kit = cfg.get("away_gk_kit", "unknown")

        def _fmt_lineup(team_name: str) -> str:
            lineups = cfg.get("lineups", [])
            tl = next((l for l in lineups if l.get("team", {}).get("name") == team_name), None)
            if not tl:
                return f"  {team_name}: lineup not available"
            lines = [f"  {team_name}:"]
            for p in tl.get("startXI", []):
                pl = p.get("player", {})
                num  = pl.get("number", "?")
                name = pl.get("name", "Unknown")
                lines.append(f"    #{num}  {name}")
            bench = [p.get("player", {}) for p in tl.get("substitutes", [])]
            if bench:
                bench_str = ", ".join(f"#{p.get('number','?')} {p.get('name','')}" for p in bench)
                lines.append(f"    Bench: {bench_str}")
            return "\n".join(lines)

        def _fmt_goals() -> str:
            goals = cfg.get("goals", [])
            if not goals:
                return "  None recorded"
            return "\n".join(
                f"  {g['time']['elapsed']}'  {g['team']['name']} — {g['player']['name']}"
                for g in goals
            )

        def _fmt_cards() -> str:
            cards = cfg.get("cards", [])
            if not cards:
                return "  None recorded"
            return "\n".join(
                f"  {c['time']['elapsed']}'  {c['team']['name']} — {c['player']['name']} ({c['detail']})"
                for c in cards
            )

        def _fmt_subs() -> str:
            subs = cfg.get("substitutions", [])
            if not subs:
                return "  None recorded"
            return "\n".join(
                f"  {s['time']['elapsed']}'  {s['team']['name']}: {s['assist']['name']} ON for {s['player']['name']}"
                for s in subs
            )

        block = f"""MATCH: {match_label}
DATE: {date}
COMPETITION: {comp}
VENUE: {venue}
RESULT: {result}
FOCUS TEAM: {focus}
ATTACK DIRECTION 1H: {ad1}
ATTACK DIRECTION 2H: {ad2}

HOME — {home}:
{_fmt_lineup(home)}

AWAY — {away}:
{_fmt_lineup(away)}

KEY EVENTS — GOALS:
{_fmt_goals()}

KEY EVENTS — CARDS:
{_fmt_cards()}

KEY EVENTS — SUBSTITUTIONS:
{_fmt_subs()}

KIT COLOURS:
  Home: {home_kit} / Home GK: {home_gk_kit}
  Away: {away_kit} / Away GK: {away_gk_kit}
"""

        block_path = self._p("match_data_block.txt")
        with open(block_path, "w", encoding="utf-8") as f:
            f.write(block)
        print(f"  Written: match_data_block.txt")

        # Initialise running_summary.json if absent
        summary_path = self._p("running_summary.json")
        if not os.path.exists(summary_path):
            summary = {
                "match":                    match_label,
                "windows_complete":         0,
                "formation_history":        [],
                "pressing_by_window":       [],
                "line_height_by_window":    [],
                "shots_for":                [],
                "shots_against":            [],
                "flagged_moments":          [],
                "key_moments":              [],
                "individual_observations":  [],
                "set_pieces":               [],
                "possession_by_window":     [],
                "data_gap_windows":         [],
                "data_gap_rescanned":       [],
                "findings":                 [],
                "confirmation_queue":       [],
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"  Written: running_summary.json (initialised)")
        else:
            print(f"  Skipped: running_summary.json already exists")

        # Initialise pass_sequences.json if absent
        ps_path = self._p("pass_sequences.json")
        if not os.path.exists(ps_path):
            ps = {
                "match":           match_label,
                "home_team":       home,
                "away_team":       away,
                "total_sequences": 0,
                "sequences":       [],
            }
            with open(ps_path, "w", encoding="utf-8") as f:
                json.dump(ps, f, indent=2)
            print(f"  Written: pass_sequences.json (initialised)")
        else:
            print(f"  Skipped: pass_sequences.json already exists")

        return True

    def step_3a_tier1(self) -> bool:
        """Step 3a — Run Tier 1 scan via Claude API."""
        _banner("Step 3a — Tier 1 scan")
        if not os.path.exists(self._p("window_plan.json")):
            print("  [X] window_plan.json missing — run Steps 1b/1c first")
            return False
        if not os.path.exists(self._p("match_data_block.txt")):
            print("  [X] match_data_block.txt missing — run Step 2 first")
            return False
        result = run_tier1_scan(self.match_dir, api_key=self.api_key)
        return result["windows_complete"] > 0

    def step_3b_rerun_queue(self) -> bool:
        """Step 3b — Build rerun queue from agent confidence data."""
        _banner("Step 3b — Confidence aggregation / rerun queue")
        if self._agent_log_count() == 0:
            print("  [X] No agent logs found — run Step 3a first")
            return False
        build_rerun_queue(self.match_dir)
        return True

    def step_3c_reruns(self) -> bool:
        """Step 3c — Run targeted reruns for low-confidence frames."""
        _banner("Step 3c — Targeted reruns")
        if not os.path.exists(self._p("rerun_queue.json")):
            print("  [X] rerun_queue.json missing — run Step 3b first")
            return False
        result = run_targeted_reruns(self.match_dir, api_key=self.api_key)
        return result["resolved"] >= 0

    def step_3d_deep_scan(self) -> bool:
        """Step 3d — Run dual-agent deep scan for event windows."""
        _banner("Step 3d — Deep scan (event windows)")
        if not os.path.exists(self._p("window_plan.json")):
            print("  [X] window_plan.json missing")
            return False
        result = run_deep_scan(self.match_dir, api_key=self.api_key)
        return result["deep_scans_complete"] >= 0

    def step_3e_merge(self) -> bool:
        """Step 3e — Merge all agent window JSONs."""
        _banner("Step 3e — Programmatic merge")
        if not os.path.isdir(self.logs_dir):
            print("  [X] agent_logs/ not found")
            return False

        plan_path = self._p("window_plan.json")
        if os.path.exists(plan_path):
            result = merge_all_windows(self.match_dir)
            return result["merged"] > 0

        # Fallback: pattern-detect dual-agent pairs without a window plan
        print("  [!]  window_plan.json not found — using filename pattern detection")
        all_logs = sorted(glob.glob(os.path.join(self.logs_dir, "*.json")))
        all_logs = [f for f in all_logs
                    if "_merged" not in f and "_rerun" not in f]

        by_base = {}
        for path in all_logs:
            fname = os.path.basename(path)
            if "_agentA" in fname or fname.endswith("_a.json"):
                base = fname.replace("_agentA.json", "").replace("_a.json", "")
                by_base.setdefault(base, {})["A"] = path
            elif "_agentB" in fname or fname.endswith("_b.json"):
                base = fname.replace("_agentB.json", "").replace("_b.json", "")
                by_base.setdefault(base, {})["B"] = path
            else:
                base = fname.replace(".json", "")
                by_base.setdefault(base, {})["single"] = path

        merged_count = 0
        for base, paths in sorted(by_base.items()):
            out_path = os.path.join(self.logs_dir, f"{base}_merged.json")
            if os.path.exists(out_path):
                continue
            if "A" in paths and "B" in paths:
                merge_dual_agents(paths["A"], paths["B"], out_path,
                                  base, self.match_dir)
            elif "single" in paths:
                merge_single_agent(paths["single"], out_path,
                                   base, self.match_dir)
            elif "A" in paths:
                merge_single_agent(paths["A"], out_path, base, self.match_dir)
            merged_count += 1

        print(f"  Merged: {merged_count} windows")
        return merged_count > 0

    def step_3f_pass(self) -> bool:
        """Step 3f — Accumulate pass sequences from merged windows."""
        _banner("Step 3f — Pass sequence accumulation")
        merged_files = sorted(glob.glob(os.path.join(self.logs_dir, "*_merged.json")))
        if not merged_files:
            print("  [X] No merged files — run Step 3e first")
            return False

        ps_path = self._p("pass_sequences.json")
        # Reset pass_sequences.json before accumulating — accumulate_pass_sequences
        # always appends, so re-running 3f without a reset doubles the count.
        if os.path.exists(ps_path):
            with open(ps_path, encoding="utf-8") as _f:
                _ps = json.load(_f)
            _ps["sequences"] = []
            _ps["total_sequences"] = 0
            with open(ps_path, "w", encoding="utf-8") as _f:
                json.dump(_ps, _f, indent=2)

        total = 0
        for path in merged_files:
            total += accumulate_pass_sequences(path, ps_path)

        print(f"  Pass sequences accumulated: {total} from {len(merged_files)} windows")
        return True

    def step_3g_summary(self) -> bool:
        """Step 3g — Accumulate running summary from all merged windows."""
        _banner("Step 3g — Running summary accumulation")
        merged_files = sorted(glob.glob(os.path.join(self.logs_dir, "*_merged.json")))
        if not merged_files:
            print("  [X] No merged files — run Step 3e first")
            return False

        result = accumulate_all_windows(self.match_dir)
        return result.get("windows_processed", 0) > 0

    def step_3h_ground_truth(self) -> bool:
        """Step 3h — Ground truth validation."""
        _banner("Step 3h — Ground truth validation")
        try:
            result = build_ground_truth_check(self.match_dir)
            return result.get("pipeline_ready", False)
        except FileNotFoundError as e:
            print(f"  [X] {e}")
            return False

    def step_3i_escalation(self) -> bool:
        """
        Step 3i — Build escalation queue, then run all confirmation segments.
        Requires --video if confirmation items need higher-fps extraction.
        """
        _banner("Step 3i — Escalation + confirmation")
        if self._merged_count() == 0:
            print("  [X] No merged files — run Step 3e first")
            return False

        # 1. Build the queue
        build_escalation_queue(self.match_dir)

        # 2. Run confirmations
        cq_path = self._p("confirmation_queue.json")
        if not os.path.exists(cq_path):
            print("  [!] confirmation_queue.json not written — check escalation_router")
            return True  # queue build succeeded even if no items

        with open(cq_path, encoding="utf-8") as f:
            cq_doc = json.load(f)

        items = [i for i in cq_doc.get("items", cq_doc.get("queue", []))
                 if i.get("status") == "queued"]

        if not items:
            print("  No queued confirmation items.")
            return True

        if not self.video_path:
            print(f"  [!] {len(items)} confirmation items queued but --video not set.")
            print("      Re-run with --video to extract higher-fps segments.")
            return True

        summary_path = self._p("running_summary.json")
        done = 0
        for item in items:
            ts = item.get("timestamp", "?")
            print(f"  Confirming: {ts} ({item.get('event_type', '')})")
            result = run_confirmation_segment(
                match_dir  = self.match_dir,
                item       = item,
                api_key    = self.api_key,
                video_path = self.video_path,
            )
            item["status"]            = result.get("status", "confirmed")
            item["confirmed_outcome"] = result.get("confirmed_outcome")
            item["confidence_after_rerun"] = result.get("confidence_after_rerun")

            if result.get("status") not in ("extraction_failed", "api_failed"):
                apply_confirmation_to_summary(result, summary_path)
                done += 1

            time.sleep(2)

        # Write updated queue
        with open(cq_path, "w", encoding="utf-8") as f:
            json.dump(cq_doc, f, indent=2)

        print(f"  Confirmations complete: {done}/{len(items)}")
        return True

    def step_3j_readiness(self) -> bool:
        """Step 3j — Report readiness gate."""
        _banner("Step 3j — Report readiness check")
        return build_readiness_check(self.match_dir)

    def step_3k_metrics(self) -> bool:
        """Step 3k — Deep skill metrics."""
        _banner("Step 3k — Deep skill metrics")
        if not os.path.exists(self._p("running_summary.json")):
            print("  [X] running_summary.json missing — run Step 3g first")
            return False
        result = build_deep_skill_metrics(self.match_dir, self.team_label)
        return result is not None

    def step_5_convert(self) -> bool:
        """Step 5 — Convert all .md reports to .docx."""
        _banner("Step 5 — Word conversion")
        md_files = glob.glob(os.path.join(self.match_dir, "*.md"))
        if not md_files:
            print("  [X] No .md files found — run Step 4 first")
            return False
        result    = convert_all_reports(self.match_dir)
        converted = result.get("converted", 0)
        failed    = result.get("failed", 0)
        print(f"  Converted: {converted} | Failed: {failed}")
        return converted > 0

    # =========================================================================
    # Agent step info
    # =========================================================================

    def _agent_info(self, step_id: str) -> bool:
        descriptions = {
            "1b_boundaries": (
                "Step 1b — Boundary detection",
                "Send 10-15 spread frames from frames/ to Claude. "
                "Detect KO, HT, and FT boundaries. Writes match_boundaries.json."
            ),
            "1d_team_sheet": (
                "Step 1d — Team sheet verification",
                "Send match_config.json + sample frames. Verify kit colours and squad. "
                "Sets verified: true in match_config.json."
            ),
            "1e_attack_dir": (
                "Step 1e — Attack direction detection",
                "Send KO frame + a few 1H frames. Identify attack direction per half. "
                "Updates match_config.json attack_direction fields."
            ),
            "2_match_data": (
                "Step 2 — Match data block",
                "Read match_config.json -> write match_data_block.txt "
                "with the pre-formatted context block for all Tier 1 agents."
            ),
            "4_reports": (
                "Step 4 — Report writing",
                "Read running_summary.json and pass_sequences.json. "
                "Apply SKILL.md Step 4 constraint block. "
                "Write tactical_report.md, opposition_report_[team].md, "
                "flagged_moments.md, pass_network.md."
            ),
        }

        title, desc = descriptions.get(step_id, (step_id, "See SKILL.md."))
        _banner(f"Agent step: {title}")
        print(f"\n  {desc}")
        print(f"\n  Match dir : {self.match_dir}")
        print(f"  SKILL.md  : {SKILL_MD}")
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience entry point
# ═══════════════════════════════════════════════════════════════════════════════

def run(match_dir: str, step: str = None, video: str = None,
        team_label: str = "both", auto: bool = False,
        api_key: str = None) -> "PipelineRunner":
    """
    Importable entry point for Claude Code sessions.

        from pipeline_runner import run
        pr = run(MATCH_DIR, step="3a_tier1")
        pr = run(MATCH_DIR, auto=True, api_key=API_KEY)
        pr.status()
    """
    pr = PipelineRunner(match_dir, video_path=video,
                        team_label=team_label, api_key=api_key)
    if auto:
        pr.auto()
    elif step:
        pr.run_step(step)
    else:
        pr.status()
    return pr


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Match Lens pipeline orchestrator + analysis execution engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("match_dir")
    parser.add_argument("--video",      help="Video file path (steps 1, 1f, 3i)")
    parser.add_argument("--api-key",    help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--team-label", default="both",
                        choices=["home", "away", "both"],
                        help="Team perspective for metric output (Fix 33b: was --focus-team)")
    parser.add_argument("--step",       help="Run one specific step")
    parser.add_argument("--auto",       action="store_true",
                        help="Run all ready programmatic steps sequentially")
    parser.add_argument("--status",     action="store_true",
                        help="Show pipeline status")

    args = parser.parse_args()

    if not os.path.isdir(args.match_dir):
        print(f"Error: '{args.match_dir}' is not a valid directory.")
        sys.exit(1)

    pr = PipelineRunner(
        match_dir  = args.match_dir,
        video_path = args.video,
        team_label = args.team_label,
        api_key    = args.api_key,
    )

    if args.status:
        pr.status()
    elif args.auto:
        pr.auto()
    elif args.step:
        ok = pr.run_step(args.step)
        sys.exit(0 if ok else 1)
    else:
        pr.status()


if __name__ == "__main__":
    main()
