# ── Addition to accumulator.py ─────────────────────────────────────────────
# Place this function after accumulate_all_windows()
# Called from pipeline_runner_v2.py as step 3f_shots (after 3e merge)
# No changes to existing functions required.

import json, os, glob, sys
from pathlib import Path
from pipeline_accessors import get_window_id, get_window_start_seconds, get_window_end_seconds, get_match_id


def build_shots_log(match_dir: str) -> dict:
    """
    Aggregate goal shot data from match_config.json and event agent outputs.
    Writes shots_log.json to match_dir.
    Returns the shots log dict.

    Data sources (in priority order):
      1. match_config.json  -- confirmed goal events (scorer, minute, team)
      2. event agent output -- shot detail (origin zone, foot, target zone,
                               build-up chain, defensive shape)
      3. set piece agent    -- if the goal came from a set piece

    Goals without event agent output are still recorded from match facts
    with evidence_grade C and source 'match_facts_only'.
    """

    mc_path = os.path.join(match_dir, "match_config.json")
    wp_path = os.path.join(match_dir, "window_plan.json")

    if not os.path.exists(mc_path):
        print("  shots_log: match_config.json not found -- skipping")
        return {}

    with open(mc_path, encoding="utf-8") as f:
        mc = json.load(f)
    with open(wp_path, encoding="utf-8") as f:
        wp = json.load(f)

    windows   = wp.get("windows", [])
    match_id  = get_match_id(match_dir, mc)
    if not mc.get("match"):
        print(f"  [!] WARNING: match_config.json missing 'match' key -- using dirname '{match_id}'", file=sys.stderr)
    home_team = mc.get("home_team", "home")
    away_team = mc.get("away_team", "away")

    # ── Load match boundaries for minute → video-second conversion ──────────
    _ko_1h_s = _ht_s = _ko_2h_s = 0
    _boundaries_path = os.path.join(match_dir, "match_boundaries.json")
    if os.path.exists(_boundaries_path):
        with open(_boundaries_path, encoding="utf-8") as _bf:
            _b = json.load(_bf)
        _ko_1h_s = _b["boundaries"]["ko_1h"]["seconds"]
        _ko_2h_s = _b["boundaries"]["ko_2h"]["seconds"]
        _ht_s    = _b["boundaries"]["ht_whistle"]["seconds"]

    def _minute_to_video_s(minute: int) -> float:
        """Convert scoreboard minute to video seconds (same logic as ground_truth.py)."""
        vs_1h = _ko_1h_s + minute * 60
        if vs_1h <= _ht_s:
            return vs_1h
        return _ko_2h_s + (minute - 45) * 60

    # ── Helper: find which window contains a given match minute ──────────────
    def _window_for_minute(minute: int) -> dict:
        video_s = _minute_to_video_s(minute)
        for w in windows:
            start = get_window_start_seconds(w)
            end   = get_window_end_seconds(w)
            if start <= video_s <= end:
                return w
        return {}

    # ── Helper: load event agent output for a window ──────────────────────
    def _load_event_output(window_id: str) -> dict:
        logs_dir = os.path.join(match_dir, "agent_logs")
        # Try both naming conventions
        patterns = [
            os.path.join(logs_dir, f"*{window_id}*event*.json"),
            os.path.join(logs_dir, f"agent_{window_id}_event.json"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], encoding="utf-8") as f:
                    return json.load(f)
        return {}

    # ── Helper: load set piece output for a window ────────────────────────
    def _load_setpiece_output(window_id: str) -> dict:
        logs_dir = os.path.join(match_dir, "agent_logs")
        patterns = [
            os.path.join(logs_dir, f"*{window_id}*setpiece*.json"),
            os.path.join(logs_dir, f"agent_{window_id}_setpiece.json"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], encoding="utf-8") as f:
                    return json.load(f)
        return {}

    # ── Helper: resolve team name from match_config event format ──────────
    def _team_name(ev: dict) -> str:
        team = ev.get("team", {})
        if isinstance(team, dict):
            return team.get("name", "unknown")
        return str(team)

    def _team_side(team_name: str) -> str:
        if team_name == home_team:
            return "home"
        if team_name == away_team:
            return "away"
        return "unknown"

    def _player_name(ev: dict) -> str:
        player = ev.get("player", {})
        if isinstance(player, dict):
            return player.get("name", "unknown")
        return str(player)

    def _player_number(ev: dict, team_side: str) -> int | None:
        """Try to find shirt number from lineups."""
        pname = _player_name(ev)
        target = home_team if team_side == "home" else away_team
        for lineup in mc.get("lineups", []):
            # lineup["team"] may be a dict {"name": "..."} or a plain string
            t = lineup.get("team", "")
            lineup_team = t.get("name", "") if isinstance(t, dict) else str(t)
            if lineup_team != target:
                continue
            for p in lineup.get("startXI", []) + lineup.get("substitutes", []):
                pdata = p.get("player", p)
                if isinstance(pdata, dict) and pdata.get("name") == pname:
                    return pdata.get("number")
        return None

    def _minute(ev: dict) -> int:
        t = ev.get("time", {})
        if isinstance(t, dict):
            return t.get("elapsed", ev.get("minute", 0))
        return ev.get("minute", 0)

    def _score_at_minute(minute: int, goals: list) -> str:
        h, a = 0, 0
        for g in goals:
            if _minute(g) <= minute:
                if _team_side(_team_name(g)) == "home":
                    h += 1
                else:
                    a += 1
        return f"{h}-{a}"

    def _match_state(team_side: str, score: str) -> str:
        h, a = map(int, score.split("-"))
        if team_side == "home":
            diff = h - a
        else:
            diff = a - h
        if diff > 0:
            return "winning"
        if diff < 0:
            return "losing"
        return "level"

    # ── Build shots log ───────────────────────────────────────────────────
    shots = []
    goals = mc.get("goals", [])

    for ev in goals:
        minute    = _minute(ev)
        team_name = _team_name(ev)
        team_side = _team_side(team_name)
        score     = _score_at_minute(minute - 1, goals)  # score before this goal
        state     = _match_state(team_side, score)
        player    = _player_name(ev)
        number    = _player_number(ev, team_side)

        # Base shot record from match facts
        shot = {
            "match_id":        match_id,
            "competition":     mc.get("competition", "unknown"),
            "match_date":      mc.get("date", "unknown"),
            "home_team":       home_team,
            "away_team":       away_team,
            "team_side":       team_side,
            "team_name":       team_name,
            "player":          player,
            "shirt_number":    number,
            "minute":          minute,
            "match_state":     state,
            "score_at_shot":   score,

            # Shot detail fields -- populated from event agent below
            "origin_zone":     None,
            "shot_foot":       None,
            "target_zone":     None,
            "outcome":         "goal",   # confirmed from match facts
            "shot_quality":    None,
            "set_piece":       False,
            "set_piece_type":  None,
            "build_up_length": None,
            "build_up_chain":  None,

            # Provenance
            "evidence_grade":  "C",      # upgraded to A/B if event agent confirms
            "source":          "match_facts_only",
        }

        # ── Try to enrich from event agent output ─────────────────────────
        window     = _window_for_minute(minute)
        window_id  = get_window_id(window)

        if window_id:
            ev_output = _load_event_output(window_id)

            if ev_output and ev_output.get("events"):
                for ev_detail in ev_output["events"]:
                    if ev_detail.get("type") != "goal":
                        continue
                    # Match by minute proximity (within 2 minutes)
                    # Event agent returns formats like "6'", "45'", "6", 45, "45m30s"
                    ev_min = ev_detail.get("minute", ev_detail.get("timestamp", ""))
                    try:
                        import re as _re
                        _m = _re.search(r'\d+', str(ev_min))
                        ev_min_int = int(_m.group()) if _m else minute
                    except (ValueError, AttributeError):
                        ev_min_int = minute  # default to match

                    if abs(ev_min_int - minute) <= 2:
                        # Enrich shot record
                        shot["origin_zone"]     = ev_detail.get("shot_origin_zone")
                        shot["shot_foot"]       = ev_detail.get("shot_foot")
                        shot["target_zone"]     = ev_detail.get("target_zone")
                        shot["shot_quality"]    = ev_detail.get("shot_quality")
                        shot["evidence_grade"]  = "A"
                        shot["source"]          = "event_agent"

                        # Build-up chain length
                        chain = ev_detail.get("build_up_sequence", "")
                        if chain:
                            shot["build_up_chain"]  = chain
                            shot["build_up_length"] = len(
                                [x for x in chain.split("->") if x.strip()]
                            )
                        break

            # ── Try to enrich from set piece agent ────────────────────────
            sp_output = _load_setpiece_output(window_id)
            if sp_output and sp_output.get("set_piece_agent"):
                sp_min = sp_output.get("timestamp", "")
                try:
                    import re as _re2
                    _sm = _re2.search(r'\d+', str(sp_min))
                    sp_min_int = int(_sm.group()) if _sm else minute
                except (ValueError, AttributeError):
                    sp_min_int = minute

                if abs(sp_min_int - minute) <= 2 and sp_output.get("outcome") == "goal":
                    shot["set_piece"]     = True
                    shot["set_piece_type"] = sp_output.get("type", "corner")
                    if shot["evidence_grade"] != "A":
                        shot["evidence_grade"] = "B"

        shots.append(shot)

    # ── Write shots_log.json ──────────────────────────────────────────────
    shots_log = {
        "match_id":     match_id,
        "match_date":   mc.get("date", "unknown"),
        "home_team":    home_team,
        "away_team":    away_team,
        "total_goals":  len(shots),
        "goals":        shots,
        "data_note":    (
            "Goals confirmed from match facts. Shot detail (origin_zone, "
            "shot_foot, target_zone) from event agent where available. "
            "evidence_grade A = event agent confirmed. "
            "evidence_grade C = match facts only, no event agent output "
            "for this window."
        )
    }

    out_path = os.path.join(match_dir, "shots_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(shots_log, f, indent=2, ensure_ascii=False)

    confirmed = sum(1 for s in shots if s["evidence_grade"] == "A")
    facts_only = sum(1 for s in shots if s["evidence_grade"] == "C")
    print(f"  shots_log: {len(shots)} goals written "
          f"({confirmed} event agent confirmed, {facts_only} match facts only)")

    return shots_log


# ── Season aggregator ──────────────────────────────────────────────────────
# Run separately against a directory of match directories to combine
# all shots_log.json files into a single dataset for visualisation.

def aggregate_shots(matches_root: str, output_path: str = None) -> list:
    """
    Reads all shots_log.json files from subdirectories of matches_root.
    Returns a flat list of shot records suitable for pandas or JSON export.
    Writes to output_path if provided.
    """
    all_shots = []
    for match_dir in sorted(Path(matches_root).iterdir()):
        log_path = match_dir / "shots_log.json"
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)
            all_shots.extend(log.get("goals", []))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_shots, f, indent=2, ensure_ascii=False)
        print(f"Aggregated {len(all_shots)} goals from "
              f"{matches_root} -> {output_path}")

    return all_shots

