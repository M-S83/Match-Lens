"""
update_running_summary_v2_1_patch.py

Patch on top of update_running_summary_v2.py. Adds accumulators for:
    - fouls_committed (from Step 3a)
    - conditional_pattern_observations (filtered view for fast access)
    - temperament_observations (filtered view, for source-coverage analytics)

Use this in place of v2's update_running_summary if you have v2.1 prompts
deployed. Backwards-compatible -- if windows don't have these fields, the
accumulators stay empty.
"""

import json
import os

# Import the v2 version so we inherit all its accumulation logic
try:
    from update_running_summary_v2 import (
        SUMMARY_INITIAL as V2_SUMMARY_INITIAL,
        update_running_summary as v2_update,
        _count_vertical_progressions,
        _count_defending_turnovers,
    )
except ImportError:
    # Allow running stand-alone for testing
    V2_SUMMARY_INITIAL = {}
    v2_update = None


# v2.1 adds these accumulators on top of v2
V2_1_ADDITIONAL = {
    "fouls_committed":                  [],   # NEW: structural log from Step 3a
    "conditional_pattern_observations": [],   # NEW: filter view
    "temperament_observations":         [],   # NEW: filter view
}


def update_running_summary_v2_1(merged_path, summary_path):
    """
    Runs v2 update first, then layers v2.1 accumulators.
    """
    # Step 1: run the v2 update
    if v2_update is None:
        raise RuntimeError(
            "v2 update_running_summary not importable -- ensure "
            "update_running_summary_v2.py is on the Python path"
        )
    v2_update(merged_path, summary_path)

    # Step 2: read in the now-updated summary and layer v2.1 fields
    with open(merged_path) as f:
        w = json.load(f)
    with open(summary_path) as f:
        summary = json.load(f)

    # Backfill v2.1 fields if missing
    for k, default in V2_1_ADDITIONAL.items():
        if k not in summary:
            summary[k] = list(default) if isinstance(default, list) else default

    # --- Fouls committed (NEW) ---
    for foul in w.get("fouls_committed", []) or []:
        f_with_window = dict(foul)
        f_with_window["window"] = w.get("window")
        summary["fouls_committed"].append(f_with_window)

    # --- Conditional pattern observations (filter view) ---
    for obs in w.get("individual_observations", []) or []:
        if obs.get("condition") and obs.get("condition_absent_outcome"):
            obs_with_window = dict(obs)
            obs_with_window["window"] = w.get("window")
            summary["conditional_pattern_observations"].append(obs_with_window)

    # --- Temperament observations (filter view) ---
    for obs in w.get("individual_observations", []) or []:
        if obs.get("action_category") == "temperament_observation":
            obs_with_window = dict(obs)
            obs_with_window["window"] = w.get("window")
            summary["temperament_observations"].append(obs_with_window)

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


# Keep the same public API name as v2 so the orchestrator can just swap the import
update_running_summary = update_running_summary_v2_1


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python update_running_summary_v2_1_patch.py [MERGED_FILE] [SUMMARY_FILE]")
        sys.exit(1)
    update_running_summary(sys.argv[1], sys.argv[2])
