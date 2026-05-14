"""
window_plan.py -- Match Lens Step 1c

Generates the analysis window list from match_boundaries.json.
Reads container_profile.json to snap window edges to segment boundaries.
Windows cover live play only -- halftime, pre-match, and post-match excluded.

Usage:
    from window_plan import build_window_plan
    plan = build_window_plan(MATCH_DIR)

    # or standalone:
    python window_plan.py [MATCH_DIR]
"""

import json
import os
import sys
from datetime import datetime

WINDOW_SECONDS = 300    # 5-minute windows
SNAP_THRESHOLD = 15     # snap window edge to container boundary if within 15s


# -- Frame naming --------------------------------------------------------------

def seconds_to_frame(s: float) -> str:
    """Convert seconds to frame filename format: frame_MMmSSs.jpg"""
    m, sec = divmod(int(s), 60)
    return f"frame_{m:02d}m{sec:02d}s.jpg"


def seconds_to_label(s: float, offset: float = 0.0) -> str:
    """Convert seconds to match-time label (relative to kick-off offset)."""
    relative = s - offset
    m, sec = divmod(int(relative), 60)
    return f"{m:02d}:{sec:02d}"



# -- Match state calculation ---------------------------------------------------

def calc_match_state(window_start_s: float, goals: list,
                     home_team: str) -> dict:
    """
    Calculate the score and match state at the start of a window.

    Fix 33b: home/away framing only -- no single-team perspective.

    Args:
        window_start_s: window start in video seconds
        goals:          list of goal dicts from match_config.json
                        each has: time.elapsed (minutes), team.name
        home_team:      the home team name

    Returns:
        {
            "score_home": int,
            "score_away": int,
            "match_state": "level" | "home_winning" | "away_winning",
            "goals_before_window": int
        }
    """
    score_home = 0
    score_away = 0

    for goal in goals:
        goal_minute = goal.get("time", {}).get("elapsed", 0)
        goal_second = goal_minute * 60
        if goal_second < window_start_s:
            team_raw = goal.get("team", "")
            team = team_raw.get("name", "") if isinstance(team_raw, dict) else str(team_raw)
            if team == home_team:
                score_home += 1
            else:
                score_away += 1

    if score_home > score_away:
        match_state = "home_winning"
    elif score_home < score_away:
        match_state = "away_winning"
    else:
        match_state = "level"

    return {
        "score_home":          score_home,
        "score_away":          score_away,
        "match_state":         match_state,
        "goals_before_window": score_home + score_away,
    }

# -- Window generation ---------------------------------------------------------

def make_windows(start: float, end: float, half_label: str,
                 boundaries: list = None) -> list:
    """
    Generate analysis windows covering one half of live play.
    If container boundaries are provided, window edges snap to them
    so higher-fps extraction never targets a timestamp near a join.

    Args:
        start:      kick-off timestamp in video seconds
        end:        whistle timestamp in video seconds
        half_label: "1H" or "2H"
        boundaries: list of container segment boundary timestamps in seconds

    Returns:
        List of window dicts
    """
    windows = []
    cursor  = start

    while cursor < end:
        natural_end = min(cursor + WINDOW_SECONDS, end)

        # Snap to nearby container segment boundary
        if boundaries:
            for b in boundaries:
                if cursor < b < natural_end:
                    if abs(b - natural_end) <= SNAP_THRESHOLD:
                        natural_end = b        # pull end back to boundary
                        break
                    elif abs(b - cursor) <= SNAP_THRESHOLD:
                        cursor = b             # push start forward past boundary
                        natural_end = min(cursor + WINDOW_SECONDS, end)
                        break

        boundary_nearby = bool(boundaries) and any(
            abs(b - cursor) <= SNAP_THRESHOLD or abs(b - natural_end) <= SNAP_THRESHOLD
            for b in (boundaries or [])
        )

        windows.append({
            "half":             half_label,
            "start_s":          round(cursor, 2),
            "end_s":            round(natural_end, 2),
            "start_frame":      seconds_to_frame(cursor),
            "end_frame":        seconds_to_frame(natural_end - 1),
            "duration_s":       round(natural_end - cursor, 2),
            "frame_count":      int(natural_end - cursor),   # 1fps
            "event_window":     False,
            "deep_scan":        False,
            "boundary_nearby":  boundary_nearby,
        })
        cursor = natural_end

    return windows


# -- Mark event windows --------------------------------------------------------

def mark_event_windows(plan: dict, event_windows: dict) -> dict:
    """
    Mark goal and sub windows in the plan.
    event_windows = {"goal": ["01","04"], "sub": ["11","13"]}
    """
    goal_ids = set(event_windows.get("goal", []))
    sub_ids  = set(event_windows.get("sub",  []))

    for w in plan["windows"]:
        aid = w["agent_id"]
        if aid in goal_ids or aid in sub_ids:
            w["event_window"] = True
            w["deep_scan"]    = True
            w["event_types"]  = []
            if aid in goal_ids: w["event_types"].append("goal")
            if aid in sub_ids:  w["event_types"].append("sub")

    plan["event_window_count"] = sum(1 for w in plan["windows"] if w["event_window"])
    return plan


# -- Main build function -------------------------------------------------------

def build_window_plan(match_dir: str,
                      event_windows: dict = None) -> dict:
    """
    Build and write window_plan.json from match_boundaries.json.
    Reads container_profile.json if present for boundary alignment.

    Args:
        match_dir:      path to match directory
        event_windows:  optional dict of {"goal": [ids], "sub": [ids]}
                        can be applied later via mark_event_windows()

    Returns:
        The completed plan dict (also written to window_plan.json)
    """

    # -- Load match config for match state tagging ----------------------------
    config_path = os.path.join(match_dir, "match_config.json")
    goals     = []
    home_team = ""
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        goals     = config.get("goals", [])
        home_team = config.get("home_team", "")
    else:
        print("  [!]  match_config.json not found -- match_state will not be tagged")

    # -- Load boundaries -------------------------------------------------------
    boundary_file = os.path.join(match_dir, "match_boundaries.json")
    if not os.path.exists(boundary_file):
        raise FileNotFoundError(
            f"match_boundaries.json not found in {match_dir} -- run Step 1b first"
        )
    with open(boundary_file, encoding="utf-8") as f:
        b = json.load(f)

    ko_1h = b["boundaries"]["ko_1h"]["seconds"]
    ht    = b["boundaries"]["ht_whistle"]["seconds"]
    ko_2h = b["boundaries"]["ko_2h"]["seconds"]
    ft    = b["boundaries"]["ft_whistle"]["seconds"]

    # -- Resolve optional top-level fields with fallbacks ----------------------
    if "match" not in b:
        # Try match_config.json, then fall back to dir name
        mc_path = os.path.join(match_dir, "match_config.json")
        if os.path.exists(mc_path):
            with open(mc_path, encoding="utf-8") as f:
                _mc = json.load(f)
            b["match"] = _mc.get("match", os.path.basename(match_dir))
        else:
            b["match"] = os.path.basename(match_dir)

    if "video_duration_seconds" not in b:
        # Try counting frames (1fps = 1s per frame), else ft + 120s buffer
        frames_dir = os.path.join(match_dir, "frames")
        if os.path.isdir(frames_dir):
            import glob as _glob
            n = len(_glob.glob(os.path.join(frames_dir, "*.jpg")))
            b["video_duration_seconds"] = n if n > 0 else int(ft) + 120
        else:
            b["video_duration_seconds"] = int(ft) + 120
        print(f"  [!]  video_duration_seconds not in boundaries — "
              f"derived as {b['video_duration_seconds']}s")

    # -- Load container boundaries ---------------------------------------------
    container_boundaries = []
    container_path = os.path.join(match_dir, "container_profile.json")
    if os.path.exists(container_path):
        with open(container_path, encoding="utf-8") as f:
            cp = json.load(f)
        container_boundaries = cp.get("boundary_timestamps_s", [])
        if container_boundaries:
            print(f"  Container: {len(container_boundaries)} segment "
                  f"boundary(ies) at {container_boundaries}s -- windows will snap")
        else:
            print(f"  Container: clean -- no boundary snapping needed")
    else:
        print(f"  [!]  container_profile.json not found -- run Step 1a first "
              f"(proceeding without boundary alignment)")

    # -- Generate windows ------------------------------------------------------
    first_half  = make_windows(ko_1h, ht,  "1H", container_boundaries)
    second_half = make_windows(ko_2h, ft,  "2H", container_boundaries)
    all_windows = first_half + second_half

    # Number sequentially, add labels, and tag match state
    for i, w in enumerate(all_windows):
        w["agent_id"] = f"{i+1:02d}"
        offset = ko_1h if w["half"] == "1H" else ko_2h
        w["label"] = (
            f"{w['half']} {seconds_to_label(w['start_s'], offset)}"
            f"–{seconds_to_label(w['end_s'], offset)}"
        )
        # Tag match state at start of this window
        if goals is not None:
            w["match_state"] = calc_match_state(
                w["start_s"], goals, home_team
            )
        else:
            w["match_state"] = {"match_state": "unknown"}

    # -- Apply event windows if provided ---------------------------------------
    plan = {
        "match":                        b["match"],
        "total_windows":                len(all_windows),
        "first_half_windows":           len(first_half),
        "second_half_windows":          len(second_half),
        "event_window_count":           0,
        "halftime_excluded_seconds":    int(ko_2h - ht),
        "pre_match_excluded_seconds":   int(ko_1h),
        "post_match_excluded_seconds":  int(b["video_duration_seconds"] - ft),
        "container_boundaries_applied": container_boundaries,
        "windows":                      all_windows,
        "generated_at":                 datetime.now().isoformat(),
    }

    if event_windows:
        plan = mark_event_windows(plan, event_windows)

    # -- Write and print -------------------------------------------------------
    out_path = os.path.join(match_dir, "window_plan.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)

    print(f"\n{'-'*55}")
    print(f"  Window plan written: {len(all_windows)} windows")
    print(f"  Halftime excluded:   {int(ko_2h - ht)}s "
          f"({(ko_2h - ht)/60:.1f} min)")
    print(f"  Pre-match excluded:  {int(ko_1h)}s")
    print(f"  Post-match excluded: {int(b['video_duration_seconds'] - ft)}s")
    if container_boundaries:
        snapped = sum(1 for w in all_windows if w["boundary_nearby"])
        print(f"  Boundary-snapped windows: {snapped}")
    print(f"{'-'*55}")
    for w in all_windows:
        snap  = " [!]" if w["boundary_nearby"] else ""
        ev    = " [EVENT]" if w["event_window"] else ""
        state = w.get("match_state", {})
        score = (f" [{state.get('score_home',0)}-{state.get('score_away',0)}"
                 f" {state.get('match_state','')}]"
                 if state.get("match_state") else "")
        print(f"  [{w['agent_id']}] {w['label']} | "
              f"{w['start_frame']} -> {w['end_frame']} | "
              f"{w['frame_count']} frames{score}{snap}{ev}")
    print(f"{'-'*55}")

    return plan


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    build_window_plan(match_dir)
