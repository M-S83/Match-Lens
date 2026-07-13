"""
regenerate_match_data.py -- Match Lens Step 26

WHY THIS FILE EXISTS
---------------------
The user asked to start improving the tactical/opposition REPORTS first,
because those are the client-facing deliverable -- the actual thing a coach
reads. But investigating showed the reports aren't currently bottlenecked
by the report-writing prompt or model at all: they're bottlenecked by STALE
DATA underneath them.

Concretely, for the real Gorleston vs Tilbury match:
  - running_summary.json on disk was last written BEFORE the Step 25 fix
    to accumulator.py. Proof, not guesswork: its pressing observations use
    "defender_facing_own_goal" and "other" as trigger values -- neither is
    part of the real 9-code canonical vocabulary. Those are the exact wrong
    values the old (pre-Step-25) _normalise_trigger() substring-matcher
    used to produce. It also still carries the "peak"/"peak_ts" fields that
    Step 25 deliberately removed as dead weight.
  - deep_skill_metrics.json on disk is built FROM running_summary.json, so
    it inherited the same staleness: pressing_trigger_consistency's
    dominant_trigger is "defender_facing_own_goal" (should be
    "cb_receiving_turned"), and pressing_intensity_score is computed from
    the wrong observation set.
  - report_readiness.json on disk says event_validation_passed: false with
    a blocking issue ("16 events missed -- re-run deep scan"). Running the
    SAME check fresh, right now, against the SAME real merged logs returns
    missed: 0, pipeline_ready: True. The blocker was stale, not real.

None of this is a report-writing problem. It's a "the report is reading
last week's homework" problem. Editing the tactical/opposition prompts or
upgrading the synthesis model before fixing this would optimize on top of
wrong inputs -- garbage in, garbage out, no matter how good the writer is.

WHAT THIS SCRIPT DOES
----------------------
Re-runs the three real pipeline steps that produce match_dir's stored
analysis artifacts, IN THE CORRECT DEPENDENCY ORDER, so every file ends up
consistent with the CURRENT code (including Step 25's fix):

  1. accumulate_all_windows(match_dir)
       Rebuilds running_summary.json + pass_sequences.json from the real
       agent_logs/*_merged.json files. This is the step Step 25 actually
       fixed, so its output changes the most.

  2. build_deep_skill_metrics(match_dir)
       Rebuilds deep_skill_metrics.json. Must run AFTER step 1, because it
       reads running_summary.json off disk as its input -- if it ran first,
       it would just re-inherit the stale pressing data.

  3. build_readiness_check(match_dir)
       Rebuilds report_readiness.json + confidence_reliability_report.json,
       including the ground-truth event check. Must run AFTER steps 1-2,
       since readiness reflects the state of the summary/metrics files.

This script deliberately does NOT touch the actual report TEXT
(tactical_report.md / opposition_report_*.md). Regenerating those requires
a real Anthropic API call through synthesis_agent.py -- a genuine cost, and
an action the user should explicitly approve, not something a data-sync
script should trigger as a side effect.
"""

import json
import os
import sys

# These are the three real, existing pipeline functions -- nothing new is
# invented here, we're just calling them in the right order.
from accumulator import accumulate_all_windows
from deep_skill_metrics import build_deep_skill_metrics
from build_readiness_check import build_readiness_check
from ground_truth import build_ground_truth_check


def regenerate_match_artifacts(match_dir: str) -> dict:
    """
    Re-sync a match's stored JSON artifacts with the current codebase.

    Raises FileNotFoundError if match_dir itself doesn't exist -- we want a
    loud, immediate failure here rather than silently creating a fresh
    empty match folder, since that would hide a typo'd path as "0 windows
    processed" instead of "this match doesn't exist."

    Returns a small summary dict so callers (and tests) can assert on the
    outcome without re-parsing every file.
    """
    if not os.path.isdir(match_dir):
        raise FileNotFoundError(f"match_dir does not exist: {match_dir}")

    # Step 1: running_summary.json + pass_sequences.json.
    # accumulate_all_windows() already handles "no merged files found"
    # gracefully (returns windows_processed: 0 rather than raising), which
    # matters for a match that's still mid-capture -- that's not an error
    # condition, it's an expected state during a live pipeline run.
    accumulate_result = accumulate_all_windows(match_dir)

    # Step 2: deep_skill_metrics.json. Only meaningful once step 1 has
    # actually produced windows -- if there were zero merged files, running
    # this against an empty running_summary.json wouldn't be wrong, just
    # not useful. We still run it for consistency (matches production
    # pipeline_runner_v2.py's behaviour), but note the zero-window case in
    # the returned summary so tests/callers can distinguish "ran on real
    # data" from "ran on nothing."
    build_deep_skill_metrics(match_dir)

    # Step 3: report_readiness.json (incl. the ground-truth event check)
    # + confidence_reliability_report.json.
    build_readiness_check(match_dir)

    # Re-read the ground truth result directly so the caller gets the
    # concrete missed-events number without re-parsing report_readiness.json.
    #
    # Edge case: build_ground_truth_check() raises FileNotFoundError if
    # match_config.json is absent (no key events list to check against at
    # all). build_readiness_check() above already handles that same case
    # internally and wrote a "NOT READY" report_readiness.json for it --
    # so here we treat "no ground truth possible" as a real, distinct
    # outcome (None / not ready) rather than letting it crash the whole
    # regeneration. A match with no config genuinely has nothing to
    # validate against; that's a fact about the match, not a bug in this
    # script.
    try:
        ground_truth = build_ground_truth_check(match_dir)
        gt_missed = ground_truth["missed"]
        gt_ready  = ground_truth["pipeline_ready"]
    except FileNotFoundError:
        gt_missed = None
        gt_ready  = False

    return {
        "windows_processed": accumulate_result.get("windows_processed", 0),
        "windows_complete":  accumulate_result.get("windows_complete", 0),
        "total_sequences":   accumulate_result.get("total_sequences", 0),
        "ground_truth_missed":        gt_missed,
        "ground_truth_pipeline_ready": gt_ready,
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python regenerate_match_data.py <match_dir>")
        sys.exit(1)
    result = regenerate_match_artifacts(sys.argv[1])
    print(json.dumps(result, indent=2))
