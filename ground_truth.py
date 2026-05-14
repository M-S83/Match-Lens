"""
ground_truth.py -- Match Lens Step 3h

Validates known match events against what the analysis pipeline found.
Reads KEY EVENTS from match_config.json and checks each one appears in
running_summary.json key_moments with confirmed consensus.

Events not found trigger a mandatory deep scan re-run before Step 4.

Usage:
    from ground_truth import build_ground_truth_check

    result = build_ground_truth_check(MATCH_DIR)
    if result["missed"] > 0:
        print("Re-run deep scan for:", result["rerun_required"])
"""

import json
import os
import sys
from datetime import datetime


# -- Timing tolerance ----------------------------------------------------------

TIMESTAMP_TOLERANCE_SECONDS = 90   # events within 90s of expected time count as found
                                    # wider than you'd expect because 1fps + window edges
                                    # mean the exact frame varies


# -- Helpers -------------------------------------------------------------------

def parse_timestamp_to_seconds(ts: str) -> float:
    """
    Parse a timestamp string to seconds.
    Accepts: "18m47s", "18:47", "18'47", "1127" (seconds), "18.47" (min.sec)
    Returns float seconds, or None if unparseable.
    """
    if ts is None:
        return None
    ts = str(ts).strip()

    # MMmSSs format: "18m47s"
    if "m" in ts and ts.endswith("s"):
        try:
            parts = ts.rstrip("s").split("m")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            pass

    # MM:SS or HH:MM:SS
    if ":" in ts:
        try:
            parts = ts.split(":")
            if len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            elif len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        except Exception:
            pass

    # Minute with prime: "18'"
    if ts.endswith("'"):
        try:
            return int(ts.rstrip("'")) * 60
        except Exception:
            pass

    # Plain number (seconds)
    try:
        return float(ts)
    except Exception:
        pass

    return None


def seconds_match(ts_a: str, ts_b: str,
                  tolerance: int = TIMESTAMP_TOLERANCE_SECONDS) -> bool:
    """Return True if two timestamps are within tolerance seconds."""
    a = parse_timestamp_to_seconds(ts_a)
    b = parse_timestamp_to_seconds(ts_b)
    if a is None or b is None:
        return False
    return abs(a - b) <= tolerance


def find_event_in_moments(event: dict, key_moments: list) -> dict | None:
    """
    Search key_moments for a match to a known event.
    Match criteria:
      1. Timestamp within TIMESTAMP_TOLERANCE_SECONDS
      2. Type match (goal/sub/card) if both specified
    Returns the matching moment dict or None.
    """
    event_ts   = event.get("timestamp") or f"{event.get('minute',0)*60}"
    event_type = event.get("type", "").lower()

    for moment in key_moments:
        moment_ts   = moment.get("timestamp", "")
        moment_type = moment.get("type", "").lower()

        ts_ok   = seconds_match(event_ts, moment_ts)
        # Normalise type aliases: "card" matches disciplinary/yellow_card/red_card;
        # "sub" matches substitution
        _TYPE_ALIASES = {"card": {"disciplinary", "yellow_card", "red_card", "yellow", "red"},
                         "sub":  {"substitution", "sub"}}
        if event_type and moment_type:
            aliases = _TYPE_ALIASES.get(event_type, set())
            type_ok = (event_type in moment_type
                       or moment_type in event_type
                       or moment_type in aliases)
        else:
            type_ok = True

        if ts_ok and type_ok:
            return moment

    return None


def event_window_id(event_seconds: float, windows: list) -> str | None:
    """Return the agent_id of the window containing this event."""
    for w in windows:
        if w["start_s"] <= event_seconds <= w["end_s"]:
            return w["agent_id"]
    return None


# -- Main validation -----------------------------------------------------------

def build_ground_truth_check(match_dir: str) -> dict:
    """
    Compare known events from match_config.json against running_summary.json.
    Writes ground_truth_check.json and returns the result dict.

    Return dict includes:
        events_checked, confirmed, partial, missed, results, rerun_required
    """

    # -- Load inputs -----------------------------------------------------------
    config_path  = os.path.join(match_dir, "match_config.json")
    summary_path = os.path.join(match_dir, "running_summary.json")
    plan_path    = os.path.join(match_dir, "window_plan.json")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"match_config.json not found in {match_dir}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"running_summary.json not found in {match_dir} "
                                f"-- run Steps 3e/3g first")

    with open(config_path, encoding="utf-8")  as f: config  = json.load(f)
    with open(summary_path, encoding="utf-8") as f: summary = json.load(f)
    windows = []
    if os.path.exists(plan_path):
        with open(plan_path, encoding="utf-8") as f:
            plan    = json.load(f)
            windows = plan.get("windows", [])

    # -- Load match boundaries for minute-to-video-time conversion ---------------
    ko_1h_s         = 0
    ko_2h_s         = 0
    first_half_live = 45 * 60  # default: 45 minutes
    ht_s            = first_half_live   # safe default; overwritten if boundaries present
    boundaries_path = os.path.join(match_dir, "match_boundaries.json")
    if os.path.exists(boundaries_path):
        with open(boundaries_path, encoding="utf-8") as f:
            b = json.load(f)
        ko_1h_s         = b["boundaries"]["ko_1h"]["seconds"]
        ko_2h_s         = b["boundaries"]["ko_2h"]["seconds"]
        ht_s            = b["boundaries"]["ht_whistle"]["seconds"]
        first_half_live = ht_s - ko_1h_s

    def _minute_to_video_s(minute):
        """Convert match minute (scoreboard time) to video seconds.
        Match minutes are always 0-45 for 1H and 45+ for 2H regardless of
        stoppage time, so we cannot use first_half_live as the split point.
        Instead: try the 1H formula; if that would land past the HT whistle,
        it must be a 2H event.
        """
        if minute is None:
            return None
        vs_1h = ko_1h_s + minute * 60
        if vs_1h <= ht_s:
            return vs_1h
        return ko_2h_s + (minute - 45) * 60

    key_moments  = summary.get("key_moments", [])
    known_events = []

    def _name(val):
        """Return a name string from either a dict {'name': ...} or a plain string."""
        if isinstance(val, dict):
            return val.get("name", "?")
        return str(val) if val else "?"

    def _minute(event):
        """Return minute int from either flat 'minute' field or nested 'time.elapsed'."""
        if "minute" in event:
            return event["minute"]
        return event.get("time", {}).get("elapsed") if isinstance(event.get("time"), dict) else None

    # -- Build event list from config ------------------------------------------
    # Goals
    for goal in config.get("goals", []):
        minute = _minute(goal)
        video_s = _minute_to_video_s(minute)
        known_events.append({
            "type":        "goal",
            "minute":      minute,
            "timestamp":   str(video_s) if video_s is not None else str((minute or 0) * 60),
            "description": f"{_name(goal.get('team'))} goal -- {_name(goal.get('player'))}",
            "team":        _name(goal.get("team")),
        })

    # Substitutions
    for sub in config.get("substitutions", []):
        minute  = _minute(sub)
        video_s = _minute_to_video_s(minute)
        known_events.append({
            "type":        "sub",
            "minute":      minute,
            "timestamp":   str(video_s) if video_s is not None else str((minute or 0) * 60),
            "description": (f"{_name(sub.get('team'))}: "
                            f"{_name(sub.get('assist') or sub.get('player_on'))} on for "
                            f"{_name(sub.get('player') or sub.get('player_off'))}"),
            "team":        _name(sub.get("team")),
        })

    # Cards
    for card in config.get("cards", []):
        minute  = _minute(card)
        video_s = _minute_to_video_s(minute)
        known_events.append({
            "type":        "card",
            "minute":      minute,
            "timestamp":   str(video_s) if video_s is not None else str((minute or 0) * 60),
            "description": (f"{_name(card.get('team'))} "
                            f"{card.get('type', card.get('detail','?'))} -- "
                            f"{_name(card.get('player'))}"),
            "team":        _name(card.get("team")),
        })

    # -- Validate each event ---------------------------------------------------
    results      = []
    rerun_needed = []

    for event in known_events:
        found_moment = find_event_in_moments(event, key_moments)
        event_secs   = parse_timestamp_to_seconds(event["timestamp"])
        win_id       = event_window_id(event_secs, windows) if event_secs else None

        if found_moment:
            consensus = found_moment.get("consensus", "unknown")
            if consensus == "confirmed":
                status = "confirmed"
            elif "partial" in str(consensus):
                status = "partial"
                rerun_needed.append(
                    f"agent_{win_id} -- {event['description']} "
                    f"(partial consensus: {consensus})"
                )
            else:
                status = "partial"
        else:
            status = "missed"
            rerun_needed.append(
                f"agent_{win_id} -- {event['description']} NOT FOUND in key_moments"
            )

        results.append({
            "event":            event["description"],
            "type":             event["type"],
            "minute":           event.get("minute"),
            "expected_seconds": event_secs,
            "expected_window":  f"agent_{win_id}" if win_id else "unknown",
            "status":           status,
            "found_at":         found_moment.get("timestamp") if found_moment else None,
            "consensus":        found_moment.get("consensus") if found_moment else None,
        })

    # -- Build counts ----------------------------------------------------------
    confirmed = sum(1 for r in results if r["status"] == "confirmed")
    partial   = sum(1 for r in results if r["status"] == "partial")
    missed    = sum(1 for r in results if r["status"] == "missed")

    output = {
        "match":           config.get("match", ""),
        "events_checked":  len(results),
        "confirmed":       confirmed,
        "partial":         partial,
        "missed":          missed,
        "pipeline_ready":  missed == 0,
        "results":         results,
        "rerun_required":  rerun_needed,
        "generated_at":    datetime.now().isoformat(),
    }

    # -- Write output ----------------------------------------------------------
    out_path = os.path.join(match_dir, "ground_truth_check.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # -- Print summary ---------------------------------------------------------
    print(f"\n{'-'*55}")
    print(f"  Ground Truth Validation")
    print(f"{'-'*55}")
    print(f"  Events checked: {len(results)}")
    print(f"  Confirmed:      {confirmed}")
    print(f"  Partial:        {partial}")
    print(f"  Missed:         {missed}")

    for r in results:
        icon = "[OK]" if r["status"] == "confirmed" else ("[ ]" if r["status"] == "partial" else "[X]")
        print(f"  {icon} {r['event'][:55]:<55} [{r['status']}]")

    if rerun_needed:
        print(f"\n  [!]  Deep scan re-runs required:")
        for r in rerun_needed:
            print(f"     - {r}")

    if missed == 0:
        print(f"\n  [OK] All events accounted for -- pipeline may proceed to Step 4")
    else:
        print(f"\n  [FAIL] {missed} event(s) missed -- resolve before Step 4")

    print(f"{'-'*55}")

    return output


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    result = build_ground_truth_check(match_dir)
    sys.exit(0 if result["pipeline_ready"] else 1)
