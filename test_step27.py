"""
test_step27.py -- tests for Step 27: set-piece write-back integrity check
and recovery.

Covers:
  1. check_writeback_integrity() correctly detects the REAL gap that
     exists right now on the real (not-yet-fixed) Gorleston match --
     proves the bug is real, not hypothetical, before we fix it.
  2. check_writeback_integrity() edge cases: no confirmation_queue.json
     at all; a synthetic already-clean case (resolved item whose summary
     record already shows burst_resolved: True).
  3. build_readiness_check() folds a write-back gap into blocking_issues
     and exposes the three new writeback_* fields.
  4. regenerate_match_artifacts_full() end-to-end on a throwaway copy of
     the real match: after running it, all 10 real set pieces show
     burst_resolved: True and the gap count drops to 0.
  5. regenerate_match_artifacts_full() raises FileNotFoundError on a
     nonexistent match_dir, same guarantee Step 26 provides.

All tests that touch the real match operate on a tempfile.mkdtemp() copy,
never the original -- same pattern as test_step26.py.
"""

import json
import os
import shutil
import tempfile
import unittest

from setpiece_writeback import check_writeback_integrity, writeback_all_bursts
from build_readiness_check import build_readiness_check
from step27_writeback_recovery import regenerate_match_artifacts_full

REAL_MATCH_DIR = os.path.join(
    os.path.dirname(__file__), "matches", "2026-04-11_gorleston_vs_tilbury"
)


class TestWritebackIntegrityRealGap(unittest.TestCase):
    """
    Step 27 originally proved the bug was real here: BEFORE the fix was
    applied, the real match directory had 10 resolved-but-unmerged
    set-piece confirmations (gap_count == 10, merged_count == 0). That real
    recovery has since been run against the actual match directory (not a
    copy), so this test now serves as a REGRESSION GUARD instead: it proves
    the gap stays fixed and doesn't silently reopen. This test reads the
    REAL match_dir directly (read-only -- it never writes to it).
    """

    def test_real_match_gap_stays_closed_after_recovery(self):
        result = check_writeback_integrity(REAL_MATCH_DIR)
        self.assertEqual(result["resolved_count"], 10)
        self.assertEqual(result["gap_count"], 0)
        self.assertEqual(result["merged_count"], 10)
        self.assertEqual(result["gap_items"], [])


class TestWritebackIntegrityEdgeCases(unittest.TestCase):

    def test_no_confirmation_queue_file_returns_all_zero(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            # Empty match dir: no confirmation_queue.json at all.
            result = check_writeback_integrity(tmp_dir)
            self.assertEqual(result, {
                "resolved_count": 0, "merged_count": 0,
                "gap_count": 0, "gap_items": [],
            })
        finally:
            shutil.rmtree(tmp_dir)

    def test_already_merged_item_reports_no_gap(self):
        """
        Synthetic clean case: a confirmation_queue.json with one resolved
        set-piece item, and a running_summary.json whose matching
        set_piece record already shows burst_resolved: True. Should report
        zero gap -- proves the check doesn't false-positive on data that's
        already correctly merged.
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            cq = {
                "items": [{
                    "event_type": "set_piece_delivery",
                    "timestamp": "10m00s",
                    "team": "home_kit",
                    "resolved": True,
                }]
            }
            summary = {
                "set_pieces": [{
                    "timestamp": "10m00s",
                    "team": "home_kit",
                    "burst_resolved": True,
                }]
            }
            with open(os.path.join(tmp_dir, "confirmation_queue.json"), "w") as f:
                json.dump(cq, f)
            with open(os.path.join(tmp_dir, "running_summary.json"), "w") as f:
                json.dump(summary, f)

            result = check_writeback_integrity(tmp_dir)
            self.assertEqual(result["resolved_count"], 1)
            self.assertEqual(result["gap_count"], 0)
            self.assertEqual(result["merged_count"], 1)
        finally:
            shutil.rmtree(tmp_dir)

    def test_resolved_item_with_no_summary_file_is_a_gap(self):
        """
        A resolved confirmation exists, but running_summary.json is
        entirely missing. Should report the item as a gap (not silently
        return zero), since the whole point of this check is to never
        under-report a real gap.
        """
        tmp_dir = tempfile.mkdtemp()
        try:
            cq = {
                "items": [{
                    "event_type": "set_piece_delivery",
                    "timestamp": "20m00s",
                    "team": "away_kit",
                    "resolved": True,
                }]
            }
            with open(os.path.join(tmp_dir, "confirmation_queue.json"), "w") as f:
                json.dump(cq, f)
            # Deliberately no running_summary.json written.

            result = check_writeback_integrity(tmp_dir)
            self.assertEqual(result["resolved_count"], 1)
            self.assertEqual(result["gap_count"], 1)
            self.assertEqual(result["merged_count"], 0)
        finally:
            shutil.rmtree(tmp_dir)


class TestReadinessCheckSurfacesGap(unittest.TestCase):
    """
    Confirms build_readiness_check() actually folds a write-back gap into
    blocking_issues and exposes the new fields -- not just that the
    underlying check function works in isolation.
    """

    def test_real_match_readiness_is_blocked_by_writeback_gap(self):
        # The real match's gap has since been fixed for real (Step 27's
        # recovery ran against the actual match directory, not a copy), so
        # we can no longer rely on the real data having a gap to test
        # against. Instead, take a throwaway copy of the (now-fixed) real
        # match and artificially reopen ONE gap by flipping a single
        # set_piece record's burst_resolved back to False -- simulating
        # exactly the bug this step fixed -- while confirmation_queue.json
        # (still correctly resolved: true) is left untouched. Proves the
        # readiness gate still correctly flips to NOT READY whenever this
        # specific mismatch exists, regardless of the real match's current
        # (fixed) state.
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
        try:
            rs_path = os.path.join(tmp_dir, "running_summary.json")
            with open(rs_path) as f:
                rs = json.load(f)
            reopened = False
            for sp in rs.get("set_pieces", []):
                if sp.get("burst_resolved") is True:
                    sp["burst_resolved"] = False
                    reopened = True
                    break
            self.assertTrue(reopened, "expected at least one burst_resolved:True record to reopen")
            with open(rs_path, "w") as f:
                json.dump(rs, f)

            report_ready = build_readiness_check(tmp_dir)
            with open(os.path.join(tmp_dir, "report_readiness.json")) as f:
                readiness = json.load(f)

            self.assertFalse(report_ready)
            self.assertEqual(readiness["writeback_resolved_count"], 10)
            self.assertGreaterEqual(readiness["writeback_gap_count"], 1)
            self.assertTrue(any(
                "write-back gap" in issue.lower() for issue in readiness["blocking_issues"]
            ))
        finally:
            shutil.rmtree(tmp_dir)

    def test_readiness_is_not_blocked_once_gap_is_closed(self):
        # Same copy, but writeback_all_bursts() runs first this time --
        # the gap should close and this specific blocking issue should
        # disappear (other blocking issues are irrelevant to this test,
        # so we only assert on the writeback-specific ones).
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
        try:
            writeback_all_bursts(tmp_dir)
            build_readiness_check(tmp_dir)
            with open(os.path.join(tmp_dir, "report_readiness.json")) as f:
                readiness = json.load(f)

            self.assertEqual(readiness["writeback_gap_count"], 0)
            self.assertEqual(readiness["writeback_merged_count"], 10)
            self.assertFalse(any(
                "write-back gap" in issue.lower() for issue in readiness["blocking_issues"]
            ))
        finally:
            shutil.rmtree(tmp_dir)


class TestFullRecoveryEndToEnd(unittest.TestCase):

    def test_full_recovery_closes_the_real_gap(self):
        tmp_dir = tempfile.mkdtemp()
        shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
        try:
            result = regenerate_match_artifacts_full(tmp_dir)

            self.assertEqual(result["writeback_applied"], 10)
            self.assertEqual(result["writeback_orphans"], 0)
            self.assertEqual(result["writeback_errors"], [])
            self.assertEqual(result["writeback_gap_count_after"], 0)
            self.assertEqual(result["writeback_merged_count_after"], 10)
            # Step 26's own checks should still hold post-recovery.
            self.assertEqual(result["ground_truth_missed"], 0)
            self.assertTrue(result["ground_truth_pipeline_ready"])

            # Direct confirmation on the actual file: every one of the 10
            # resolved set pieces now shows burst_resolved: True.
            with open(os.path.join(tmp_dir, "running_summary.json")) as f:
                summary = json.load(f)
            resolved = [sp for sp in summary["set_pieces"] if sp.get("burst_resolved")]
            self.assertEqual(len(resolved), 10)

            # Each resolved record should be either a real "confirmed"
            # verdict (with genuine burst-only detail like delivery_type or
            # runners populated) or a legitimate "inconclusive" verdict
            # (the model couldn't confirm extra detail, so it correctly
            # marks burst_resolved: True with a rejection_reason instead of
            # fabricating detail it didn't actually see). Both are correct
            # outcomes of a successful writeback -- only a record stuck at
            # neither (still null/unresolved) would be a real bug.
            confirmed_with_detail = [
                sp for sp in resolved
                if sp.get("burst_verdict") == "confirmed"
                and (sp.get("delivery_type") is not None or sp.get("runners") is not None)
            ]
            inconclusive = [
                sp for sp in resolved if sp.get("burst_verdict") == "inconclusive"
            ]
            self.assertEqual(len(confirmed_with_detail) + len(inconclusive), 10)
            # This real match's data has exactly one inconclusive burst
            # (92m15s/away_kit, per the writeback log) -- pin that down too,
            # so a future change that silently makes MORE bursts
            # inconclusive doesn't slip past unnoticed.
            self.assertEqual(len(inconclusive), 1)
            self.assertEqual(len(confirmed_with_detail), 9)
        finally:
            shutil.rmtree(tmp_dir)

    def test_nonexistent_match_dir_raises(self):
        with self.assertRaises(FileNotFoundError):
            regenerate_match_artifacts_full("/tmp/definitely_does_not_exist_step27")


if __name__ == "__main__":
    unittest.main()
