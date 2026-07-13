"""
test_step26.py -- Match Lens Step 26

Tests regenerate_match_data.py's regenerate_match_artifacts(), which re-syncs
a match's stored JSON artifacts (running_summary.json, deep_skill_metrics.json,
report_readiness.json) with the current codebase -- specifically proving that
the real Gorleston match's on-disk artifacts were stale relative to Step 25's
pressing fix, and that regenerating them fixes that.

We run against a COPY of the real match directory, never the original --
this is training work, and we don't want a test run to be the thing that
silently overwrites the user's real match data. The actual "apply this to
the real match_dir" action happens separately, outside the test suite,
with its own explicit before/after diff shown to the user.
"""

import json
import os
import shutil
import tempfile
import unittest

from regenerate_match_data import regenerate_match_artifacts

REAL_MATCH_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "matches", "2026-04-11_gorleston_vs_tilbury",
)

# The real, current 9-code canonical pressing trigger vocabulary
# (pipeline_runner_v2.py lines ~1108-1132). Anything outside this set in a
# regenerated file is proof the fix didn't actually take effect.
CANONICAL_TRIGGERS = {
    "back_pass", "gk_in_possession", "cb_receiving_turned",
    "cb_receiving_open", "fb_receiving", "free_kick_restart",
    "throw_in_restart", "no_trigger_observed", "null",
}


class TestRegenerateRealMatchValidCase(unittest.TestCase):
    """
    The main valid case: regenerating the real Gorleston match's artifacts
    against the CURRENT codebase should produce fresh, non-stale output,
    and should flip the ground-truth check from "blocked" to "ready" --
    because the underlying data was never actually broken, only the stored
    snapshot of it was out of date.
    """

    @classmethod
    def setUpClass(cls):
        # Work on a throwaway copy so this test suite never mutates the
        # user's real match folder.
        cls.tmp_dir = tempfile.mkdtemp(prefix="step26_")
        cls.match_copy = os.path.join(cls.tmp_dir, "match")
        shutil.copytree(REAL_MATCH_DIR, cls.match_copy)
        cls.result = regenerate_match_artifacts(cls.match_copy)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_all_21_real_windows_were_processed(self):
        # Confirms the regeneration actually ran against the real 21 merged
        # agent logs for this match, not zero / a subset.
        self.assertEqual(self.result["windows_processed"], 21)
        self.assertEqual(self.result["windows_complete"], 21)

    def test_running_summary_has_no_stale_peak_fields(self):
        # Step 25 deliberately removed peak/peak_ts as dead fields nothing
        # reads. A fresh regeneration must not reintroduce them.
        with open(os.path.join(self.match_copy, "running_summary.json")) as f:
            rs = json.load(f)
        for window in rs["pressing_by_window"]:
            self.assertNotIn("peak", window)
            self.assertNotIn("peak_ts", window)

    def test_running_summary_uses_only_canonical_triggers(self):
        # This is the direct proof the fix flows through end to end: every
        # trigger value written must come from the real 9-code vocabulary,
        # not the old substring-matched values like "defender_facing_own_goal"
        # or "other" that the stale on-disk file (pre-regeneration) contained.
        with open(os.path.join(self.match_copy, "running_summary.json")) as f:
            rs = json.load(f)
        seen_triggers = set()
        for window in rs["pressing_by_window"]:
            for obs in window.get("observations", []):
                seen_triggers.add(obs["trigger"])
        self.assertTrue(seen_triggers, "expected at least one real trigger observation")
        non_canonical = seen_triggers - CANONICAL_TRIGGERS
        self.assertEqual(
            non_canonical, set(),
            f"found non-canonical trigger values (stale-fix evidence): {non_canonical}",
        )

    def test_deep_skill_metrics_pressing_consistency_uses_canonical_trigger(self):
        # deep_skill_metrics.json is derived FROM running_summary.json, so if
        # step 1 ran first and step 2 ran second (correct order), this must
        # also be canonical. This is the metric a stale run reported as
        # dominant_trigger="defender_facing_own_goal" -- proof the whole
        # chain, not just one file, is now consistent.
        with open(os.path.join(self.match_copy, "deep_skill_metrics.json")) as f:
            dsm = json.load(f)
        metrics = dsm.get("metrics", dsm)
        consistency = next(
            m for m in metrics if m.get("metric_name") == "pressing_trigger_consistency"
        )
        dominant = consistency["value"]["dominant_trigger"]
        self.assertIn(dominant, CANONICAL_TRIGGERS)
        self.assertNotEqual(dominant, "defender_facing_own_goal")

    def test_ground_truth_check_is_no_longer_blocked(self):
        # The stale report_readiness.json on disk says missed=16,
        # pipeline_ready=False. Running the SAME real events against the
        # SAME real merged logs, fresh, must show missed=0, ready=True --
        # proving the original block was a stale-snapshot artifact, not a
        # real data gap.
        self.assertEqual(self.result["ground_truth_missed"], 0)
        self.assertTrue(self.result["ground_truth_pipeline_ready"])

    def test_report_readiness_file_matches_fresh_ground_truth(self):
        with open(os.path.join(self.match_copy, "report_readiness.json")) as f:
            readiness = json.load(f)
        self.assertTrue(readiness["event_validation_passed"])
        self.assertNotIn(
            "Ground truth", " ".join(readiness.get("blocking_issues", [])),
        )


class TestRegenerateEdgeCases(unittest.TestCase):
    """
    Edge cases the established test pattern calls for: what happens when
    the input isn't a healthy, complete real match.
    """

    def test_match_dir_with_no_merged_logs_does_not_crash(self):
        # A match still mid-capture (agent_logs/ exists but empty, or the
        # match_dir has no agent_logs/ at all) is a normal, expected state
        # during a live pipeline run -- not an error. Must return cleanly
        # with windows_processed=0 rather than raising.
        #
        # This case also has no match_config.json, so there's no key-events
        # list to check ground truth against at all -- that's a real,
        # distinct "not applicable" outcome (ground_truth_missed=None),
        # not a crash. See the try/except in regenerate_match_artifacts().
        tmp_dir = tempfile.mkdtemp(prefix="step26_empty_")
        try:
            os.makedirs(os.path.join(tmp_dir, "agent_logs"))
            result = regenerate_match_artifacts(tmp_dir)
            self.assertEqual(result["windows_processed"], 0)
            self.assertIsNone(result["ground_truth_missed"])
            self.assertFalse(result["ground_truth_pipeline_ready"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def test_nonexistent_match_dir_raises_filenotfounderror(self):
        # A typo'd path should fail loudly and immediately, not silently
        # process "0 windows" and let the caller think everything's fine.
        with self.assertRaises(FileNotFoundError):
            regenerate_match_artifacts("/tmp/this_match_dir_does_not_exist_step26")


if __name__ == "__main__":
    unittest.main(verbosity=2)
