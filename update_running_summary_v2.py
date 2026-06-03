"""
update_running_summary_v2.py

Drop-in replacement for the running_summary.json update function.
Adds new accumulators required by the v2 schema:

  vertical_progression_counts -- per-window dict counting each progression type
  defending_third_turnover_count
  watch_list_confirmations    -- collected across all windows
  between_lines_events        -- list of every observation with between_lines != null
  line_height_m_by_window     -- metres + width_m + space_behind_m per window
  pocket_observations         -- shortcut filter: obs with between_def_mid
  quiet_windows               -- list of windows flagged quiet by player agent

Call once per merged window file, in window order, as in the existing pipeline.

Usage:
    from update_running_summary_v2 import update_running_summary
    update_running_summary(merged_path, summary_path)
"""

import json
import os


SUMMARY_INITIAL = {
    "match":                          "",
    "windows_complete":               0,
    "formation_history":              [],
    "pressing_by_window":             [],
    "line_height_by_window":          [],
    "line_height_m_by_window":        [],   # NEW: metres
    "shots_for":                      [],
    "shots_against":                  [],
    "flagged_moments":                [],
    "key_moments":                    [],
    "individual_observations":        [],
    "set_pieces":                     [],
    "possession_by_window":           [],
    "data_gap_windows":               [],
    "quiet_windows":                  [],   # NEW
    "vertical_progression_counts":    [],   # NEW: per-window dict
    "vertical_progression_totals":    {     # NEW: cumulative across windows
        "same_third":                       0,
        "defending_to_middle":              0,
        "middle_to_attacking":              0,
        "defending_to_attacking":           0,
        "regression_middle_to_defending":   0,
        "regression_attacking_to_middle":   0,
        "regression_attacking_to_defending":0,
        "unknown":                          0,
    },
    "defending_third_turnover_count": 0,    # NEW
    "defending_third_sequence_count": 0,    # NEW (denominator)
    "watch_list_confirmations":       [],   # NEW
    "between_lines_events":           [],   # NEW
    "duels":                          [],   # NEW: collected duels w/ outcomes
}


def _count_vertical_progressions(pass_sequences):
    """Count vertical_progression values across a window's pass sequences."""
    counts = {
        "same_third":                        0,
        "defending_to_middle":               0,
        "middle_to_attacking":               0,
        "defending_to_attacking":            0,
        "regression_middle_to_defending":    0,
        "regression_attacking_to_middle":    0,
        "regression_attacking_to_defending": 0,
        "unknown":                           0,
    }
    for seq in pass_sequences:
        vp = seq.get("vertical_progression", "unknown")
        if vp in counts:
            counts[vp] += 1
        else:
            counts["unknown"] += 1
    return counts


def _count_defending_turnovers(pass_sequences):
    """
    Returns (turnover_count, sequence_count) for sequences starting in
    the defending third. Turnover = outcome in {lost_possession, clearance}.
    """
    turnover_count = 0
    sequence_count = 0
    for seq in pass_sequences:
        zs = seq.get("zone_start") or {}
        v_start = zs.get("vertical_third")
        if v_start == "defending":
            sequence_count += 1
            outcome = seq.get("outcome")
            if outcome in {"lost_possession", "clearance"}:
                turnover_count += 1
    return turnover_count, sequence_count


def update_running_summary(merged_path, summary_path):
    """
    Reads a merged window JSON, appends/accumulates into running_summary.json.
    Creates the summary file if it does not exist.

    Idempotent ONLY if called once per merged window in window order. If you
    re-run the pipeline, reset running_summary.json to SUMMARY_INITIAL first.
    """
    with open(merged_path) as f:
        w = json.load(f)

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
        # Backfill any missing v2 fields if reading an older summary
        for key, default in SUMMARY_INITIAL.items():
            if key not in summary:
                summary[key] = (
                    default.copy() if isinstance(default, (list, dict)) else default
                )
    else:
        summary = json.loads(json.dumps(SUMMARY_INITIAL))  # deep copy

    summary["windows_complete"] += 1

    # --- Formation ---
    summary["formation_history"].append({
        "window": w.get("window"),
        "shape":  w.get("formation", {}).get("shape_in_possession"),
    })

    # --- Pressing ---
    summary["pressing_by_window"].append({
        "window":    w.get("window"),
        "avg_score": w.get("pressing", {}).get("avg_score"),
        "peak":      w.get("pressing", {}).get("peak_score"),
    })

    # --- Line height (pct, legacy) ---
    summary["line_height_by_window"].append({
        "window":  w.get("window"),
        "avg_pct": w.get("defensive_line", {}).get("avg_pct"),
        "shifts":  w.get("defensive_line", {}).get("notable_shifts", []),
    })

    # --- Line height (metres, NEW) ---
    dl = w.get("defensive_line", {}) or {}
    summary["line_height_m_by_window"].append({
        "window":              w.get("window"),
        "avg_m_approx":        dl.get("avg_m_approx"),
        "line_width_m_approx": dl.get("line_width_m_approx"),
        "space_behind_m":      dl.get("space_behind_m"),
        "line_coordination":   dl.get("line_coordination"),
        "phase_context":       dl.get("phase_context"),
    })

    # --- Shots ---
    focus_team = w.get("focus_team")
    for shot in w.get("shot_attempts", []):
        target = (
            summary["shots_for"]
            if shot.get("team") == focus_team
            else summary["shots_against"]
        )
        target.append(shot)

    # --- Moments / observations / set pieces ---
    summary["flagged_moments"].extend(w.get("flaggable_moments", []))
    summary["key_moments"].extend(w.get("key_moments", []))
    summary["individual_observations"].extend(w.get("individual_observations", []))
    summary["set_pieces"].extend(w.get("set_pieces", []))

    # --- Possession ---
    summary["possession_by_window"].append({
        "window":  w.get("window"),
        "summary": w.get("possession_summary", {}),
    })

    # --- Data gap windows (from confidence aggregator) ---
    if w.get("window_confidence", {}).get("data_gap_warning"):
        summary["data_gap_windows"].append(w.get("window"))

    # --- Quiet windows (NEW, from player agent) ---
    quiet = w.get("quiet_window") or {}
    if quiet.get("flagged"):
        summary["quiet_windows"].append({
            "window":                w.get("window"),
            "reason":                quiet.get("reason"),
            "observations_produced": quiet.get("observations_produced"),
        })

    # --- Vertical progression (NEW) ---
    seq = w.get("pass_sequences", [])
    window_counts = _count_vertical_progressions(seq)
    summary["vertical_progression_counts"].append({
        "window": w.get("window"),
        "counts": window_counts,
    })
    for k, v in window_counts.items():
        summary["vertical_progression_totals"][k] += v

    # --- Defending-third turnovers (NEW) ---
    turnovers, total_def_seq = _count_defending_turnovers(seq)
    summary["defending_third_turnover_count"] += turnovers
    summary["defending_third_sequence_count"] += total_def_seq

    # --- Watch list confirmations (NEW) ---
    for c in w.get("watch_list_confirmations", []):
        c_with_window = dict(c)
        c_with_window["window"] = w.get("window")
        summary["watch_list_confirmations"].append(c_with_window)

    # --- Between-lines events (NEW) ---
    for obs in w.get("individual_observations", []):
        zone = obs.get("zone", {}) or {}
        if zone.get("between_lines"):
            summary["between_lines_events"].append({
                "window":        w.get("window"),
                "timestamp":     obs.get("timestamp"),
                "player":        obs.get("player"),
                "team":          obs.get("team"),
                "between_lines": zone.get("between_lines"),
                "lateral_lane":  zone.get("lateral_lane"),
                "vertical_third":zone.get("vertical_third"),
                "frequency":     obs.get("frequency"),
                "outcome":       obs.get("outcome"),
            })

    # --- Duels (NEW, with post_duel_outcome) ---
    for duel in w.get("duels", []):
        duel_with_window = dict(duel)
        duel_with_window["window"] = w.get("window")
        summary["duels"].append(duel_with_window)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python update_running_summary_v2.py [MERGED_FILE] [SUMMARY_FILE]")
        sys.exit(1)
    update_running_summary(sys.argv[1], sys.argv[2])
