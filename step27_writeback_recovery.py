"""
step27_writeback_recovery.py -- Match Lens Step 27

WHY THIS FILE EXISTS
---------------------
Before spending real API cost regenerating the tactical/opposition report
TEXT, we checked whether we were wasting tokens anywhere else in the
pipeline first. We found a real, concrete case of it on the Gorleston vs
Tilbury match:

  - 10 set-piece deliveries were escalated for a 5fps "burst" re-analysis
    (5x the frame density of a normal window -- the expensive, high-
    fidelity tier). All 10 completed successfully: confirmation_queue.json
    shows resolved: true for every one, and real *_setpiece.json result
    files exist on disk with genuinely useful scouting detail -- delivery
    type, exact runner roles and marking assignments, wall size and
    position, second-phase outcomes.
  - None of that detail reached running_summary.json. Every one of the 20
    set_pieces entries there still shows the crude 1fps-only version
    (burst_resolved: false, delivery_type: null, runners: null, ...).
  - We proved the merge logic itself isn't broken: calling
    setpiece_writeback.writeback_all_bursts() directly against a safe copy
    of this exact real data applied all 10 bursts cleanly, 0 errors.
  - We could NOT pin down with certainty why the merge never landed in the
    real files -- it may have been a silently-swallowed exception in
    pipeline_runner_v2.py's PHASE 3b (wrapped in a broad
    `except Exception: print(...)  # non-blocking` handler), or a later
    re-run of the ordinary window-merge step overwriting the patched
    merged window file with a freshly-rebuilt one that has no knowledge a
    burst confirmation had already been applied. Both are plausible.

Rather than gamble on a single guessed root cause, the fix here is a
detect-and-repair pattern that's correct regardless of WHICH of those
happened:
  1. setpiece_writeback.check_writeback_integrity() (added in this step)
     is now wired into build_readiness_check.py, so report_readiness.json
     will ALWAYS flag this specific kind of gap on every future run --
     unattended, zero human review, exactly like the existing ground-truth
     check already does for missed events.
  2. writeback_all_bursts() is already idempotent (it skips records
     already burst_resolved: True), so re-running it whenever a gap is
     flagged is always safe -- it can never double-apply or corrupt
     already-merged data.

WHAT THIS SCRIPT DOES
----------------------
Composes the full, correct recovery + resync order for a match:

  1. writeback_all_bursts(match_dir)
       Merges any already-resolved-but-unmerged set-piece burst data into
       the merged window files (and mirrors the patch into
       running_summary.json directly, though step 2 below rebuilds it
       fresh anyway). Must run FIRST -- everything else reads from the
       merged window files, so if this doesn't run first, steps 2-4 would
       just re-inherit the same gap they're supposed to fix.

  2-4. regenerate_match_artifacts(match_dir)  [Step 26, reused as-is]
       Rebuilds running_summary.json/pass_sequences.json from the now-
       patched merged window files, then deep_skill_metrics.json, then
       report_readiness.json (which will now correctly show
       writeback_gap_count: 0, assuming step 1 closed every gap).

This deliberately reuses Step 26's regenerate_match_artifacts() rather
than duplicating its logic -- Step 26's own tests already cover that
function's correctness; this script only needs to prove that running
writeback FIRST, then Step 26's existing chain, produces a fully-merged
result.
"""

import json
import sys

from setpiece_writeback import writeback_all_bursts, check_writeback_integrity
from regenerate_match_data import regenerate_match_artifacts


def regenerate_match_artifacts_full(match_dir: str) -> dict:
    """
    Full recovery + resync: writeback first, then Step 26's existing chain.

    Raises FileNotFoundError if match_dir itself doesn't exist -- delegated
    to regenerate_match_artifacts()'s own check, called first below so we
    fail loudly before attempting any writeback work on a nonexistent path.

    Returns a summary dict combining writeback stats with Step 26's
    existing regeneration stats, so callers/tests can assert on the whole
    pipeline in one call.
    """
    # Fail fast on a bad path, same guarantee Step 26 already provides --
    # doing this check via the writeback call itself would be a confusing
    # place for a FileNotFoundError to first surface.
    import os
    if not os.path.isdir(match_dir):
        raise FileNotFoundError(f"match_dir does not exist: {match_dir}")

    # Step 1 (new in Step 27): merge any resolved-but-unmerged set-piece
    # bursts into the merged window files. Safe to call even if there are
    # zero burst files (writeback_all_bursts handles that -- returns
    # applied: 0) or zero already-applied ones (idempotent skip).
    writeback_result = writeback_all_bursts(match_dir)

    # Steps 2-4 (Step 26, unchanged): rebuild running_summary.json,
    # deep_skill_metrics.json, and report_readiness.json from the now-
    # patched merged window files.
    regen_result = regenerate_match_artifacts(match_dir)

    # Re-check integrity after the full chain, so the caller gets a direct
    # answer to "did this actually close the gap?" rather than having to
    # infer it from writeback_result's applied/orphans/errors counts alone.
    integrity_after = check_writeback_integrity(match_dir)

    return {
        "writeback_applied": writeback_result["applied"],
        "writeback_orphans":  writeback_result["orphans"],
        "writeback_errors":   writeback_result["errors"],
        "windows_processed":  regen_result["windows_processed"],
        "windows_complete":   regen_result["windows_complete"],
        "total_sequences":    regen_result["total_sequences"],
        "ground_truth_missed":         regen_result["ground_truth_missed"],
        "ground_truth_pipeline_ready": regen_result["ground_truth_pipeline_ready"],
        "writeback_gap_count_after":   integrity_after["gap_count"],
        "writeback_merged_count_after": integrity_after["merged_count"],
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python step27_writeback_recovery.py <match_dir>")
        sys.exit(1)
    result = regenerate_match_artifacts_full(sys.argv[1])
    print(json.dumps(result, indent=2))
