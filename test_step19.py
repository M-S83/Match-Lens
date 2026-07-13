"""
test_step19.py -- Match Lens training, Step 19

WHAT THIS STEP CHANGED (see the WHY comments in ground_truth.py for the
full reasoning -- this is the short version):

Before this step, build_ground_truth_check() treated goals, subs, and
cards identically: if a known event from match_config.json wasn't ALSO
independently found in running_summary.json's key_moments, it was marked
"missed", and a non-zero "missed" count blocks the whole pipeline in
build_readiness_check.py (unless the source is broadcast_fixed_wide).

That's correct for GOALS -- the "Goals Scored" report section genuinely
needs agent-observed build-up/shot data that only the video pipeline can
supply, so an undetected goal is a real coverage gap worth a deep-scan
retry.

It was WRONG for subs/cards -- their who/when is already a certain fact
from match_config.json, so failing to also find them in key_moments is
not a defect. What actually matters for report quality is whether there
is ANY structural data (of any type) near that timestamp for the
synthesis prompt to reason about impact with -- if there is, the report
can say something; if not, the report states the bare fact plainly
(which synthesis_agent.py's own prompt already explicitly allows).

This step:
  1. Keeps goals on the strict confirmed/partial/missed path (unchanged).
  2. Gives subs/cards two new, non-blocking statuses instead of "missed":
       - "fact_only_context_available" (something nearby, of any type)
       - "fact_only_no_context"        (nothing nearby at all)
  3. Narrows "missed" (and therefore pipeline_ready / the readiness gate's
     blocking condition) to goals only, since that's the only case a
     rerun could actually fix.

These tests prove all three points with real assert/raise checks -- not
just prints -- against both synthetic fixtures (to isolate each behavior
precisely) and the real Gorleston vs Tilbury match data (to prove the
change actually lands as intended on data we already know is honest).
"""

import json
import os
import shutil
import tempfile
import unittest

from ground_truth import build_ground_truth_check, has_nearby_structural_context


# -- Shared fixture builder ----------------------------------------------------
# WHY a shared helper instead of copy-pasting JSON in every test: every
# scenario needs the same three files (match_config.json, running_summary.json,
# match_boundaries.json) in the same schema shape build_ground_truth_check()
# actually reads. Centralising it means a real schema change only needs
# fixing in one place, and keeps each test method focused on what's
# DIFFERENT about that scenario (the one thing being tested) rather than
# repeating unrelated boilerplate.

BOUNDARIES = {
    "boundaries": {
        "ko_1h":      {"frame": 0, "seconds": 0,    "confidence": 1.0},
        "ht_whistle": {"frame": 0, "seconds": 2700, "confidence": 1.0},
        "ko_2h":      {"frame": 0, "seconds": 2760, "confidence": 1.0},
        "ft_whistle": {"frame": 0, "seconds": 5460, "confidence": 1.0},
    },
    "live_play_seconds": {},
}


def _make_match_dir(goals=None, substitutions=None, cards=None, key_moments=None):
    d = tempfile.mkdtemp()
    config = {
        "match": "Test FC vs Example Town",
        "goals": goals or [],
        "substitutions": substitutions or [],
        "cards": cards or [],
    }
    with open(os.path.join(d, "match_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f)
    with open(os.path.join(d, "running_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"key_moments": key_moments or []}, f)
    with open(os.path.join(d, "match_boundaries.json"), "w", encoding="utf-8") as f:
        json.dump(BOUNDARIES, f)
    return d


class TestSubCardNoLongerBlocksOnMissing(unittest.TestCase):
    """A sub/card absent from key_moments must never produce 'missed'."""

    def setUp(self):
        # Sub at minute 60 (2H): vs_1h = 0 + 60*60 = 3600s, which is past
        # ht_s=2700, so it correctly routes to the 2H formula instead:
        # ko_2h_s + (60-45)*60 = 2760 + 900 = 3660s.
        self.work_dir = _make_match_dir(
            substitutions=[{"team": "Test FC", "player_off": "A. Player",
                            "player_on": "B. Player", "time": {"elapsed": 60}}],
            key_moments=[],   # nothing observed anywhere -- the "no context" case
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_sub_with_zero_nearby_moments_is_fact_only_no_context_not_missed(self):
        result = build_ground_truth_check(self.work_dir)
        self.assertEqual(result["events_checked"], 1)
        # The core assertion of this step: an unmatched sub must NEVER be
        # "missed" -- that field is goals-only now.
        self.assertEqual(result["missed"], 0)
        self.assertEqual(result["fact_only_no_context"], 1)
        self.assertEqual(result["fact_only_context_available"], 0)
        self.assertEqual(result["results"][0]["status"], "fact_only_no_context")
        # Nothing to rerun -- a deep-scan retry cannot manufacture context
        # that plain never existed near this timestamp.
        self.assertEqual(result["rerun_required"], [])
        # And critically: this must NOT block the pipeline.
        self.assertTrue(result["pipeline_ready"])


class TestSubCardWithNearbyContext(unittest.TestCase):
    """A sub/card near SOME key_moment (any type) gets the 'context available' status."""

    def setUp(self):
        # Same sub as above (minute 60 -> 3660s), but this time there's an
        # unrelated key_moment (type "shot", not "substitution") 30s away --
        # close enough (tolerance is 90s) for the report to have SOMETHING
        # to reason with, even though it never independently detected the
        # sub itself.
        self.work_dir = _make_match_dir(
            substitutions=[{"team": "Test FC", "player_off": "A. Player",
                            "player_on": "B. Player", "time": {"elapsed": 60}}],
            key_moments=[{"type": "shot", "minute": "60:30", "consensus": "confirmed"}],
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_unrelated_nearby_moment_counts_as_context_available(self):
        result = build_ground_truth_check(self.work_dir)
        self.assertEqual(result["missed"], 0)
        self.assertEqual(result["fact_only_context_available"], 1)
        self.assertEqual(result["fact_only_no_context"], 0)
        self.assertEqual(result["results"][0]["status"], "fact_only_context_available")
        self.assertTrue(result["pipeline_ready"])


class TestGoalStillStrict(unittest.TestCase):
    """Goals must keep the OLD strict behavior -- this step must not weaken that gate."""

    def setUp(self):
        # Goal at minute 20 (1H): vs_1h = 0 + 20*60 = 1200s, <= ht_s=2700,
        # so it stays on the 1H formula as expected. No key_moments at all,
        # so this goal is genuinely undetected.
        self.work_dir = _make_match_dir(
            goals=[{"team": "Test FC", "player": "C. Striker", "time": {"elapsed": 20}}],
            key_moments=[],
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_undetected_goal_is_still_missed_and_blocks(self):
        result = build_ground_truth_check(self.work_dir)
        # This is the control case proving Step 19 narrowed the CHECK, not
        # the STANDARD -- a goal with zero evidence anywhere must still be
        # a real, blocking gap, exactly as before this step.
        self.assertEqual(result["missed"], 1)
        self.assertEqual(result["fact_only_no_context"], 0)
        self.assertEqual(result["fact_only_context_available"], 0)
        self.assertEqual(result["results"][0]["status"], "missed")
        self.assertEqual(len(result["rerun_required"]), 1)
        self.assertIn("C. Striker", result["rerun_required"][0])
        self.assertFalse(result["pipeline_ready"])


class TestMixedMatchOnlyGoalCountsTowardMissed(unittest.TestCase):
    """A match with one undetected goal AND one undetected sub: only the
    goal should count toward 'missed'; the sub must land in the new,
    non-blocking bucket. Proves the two paths don't leak into each other."""

    def setUp(self):
        self.work_dir = _make_match_dir(
            goals=[{"team": "Test FC", "player": "C. Striker", "time": {"elapsed": 20}}],
            substitutions=[{"team": "Test FC", "player_off": "A. Player",
                            "player_on": "B. Player", "time": {"elapsed": 60}}],
            key_moments=[],
        )

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_missed_count_excludes_the_undetected_sub(self):
        result = build_ground_truth_check(self.work_dir)
        self.assertEqual(result["events_checked"], 2)
        # If this were still the pre-Step-19 behavior, missed would be 2.
        self.assertEqual(result["missed"], 1)
        self.assertEqual(result["fact_only_no_context"], 1)
        statuses = {r["event"]: r["status"] for r in result["results"]}
        self.assertTrue(any(k.startswith("Test FC goal") for k in statuses))
        self.assertEqual(
            [v for k, v in statuses.items() if k.startswith("Test FC goal")][0],
            "missed",
        )
        self.assertEqual(
            [v for k, v in statuses.items() if "on for" in k][0],
            "fact_only_no_context",
        )


class TestHasNearbyStructuralContextEdgeCases(unittest.TestCase):
    """Direct unit tests on the new helper, including malformed input that
    must degrade gracefully (return False) rather than raise."""

    def test_empty_key_moments_returns_false(self):
        self.assertFalse(has_nearby_structural_context("1000", []))

    def test_garbage_moment_timestamps_do_not_raise(self):
        # Every one of these moments has an unparseable or missing time
        # field. parse_timestamp_to_seconds() returns None for all of
        # them, and seconds_match() already treats None as "not a match" --
        # this test proves has_nearby_structural_context() inherits that
        # safety rather than crashing on garbage real-world data (e.g. a
        # key_moment written by an older pipeline version with a field
        # missing or malformed).
        garbage_moments = [
            {"minute": "not-a-timestamp"},
            {"timestamp": None, "minute": None},
            {},
            {"minute": ""},
        ]
        try:
            result = has_nearby_structural_context("1000", garbage_moments)
        except Exception as exc:  # pragma: no cover - this is the failure mode we're guarding against
            self.fail(f"has_nearby_structural_context() raised on malformed input: {exc!r}")
        self.assertFalse(result)

    def test_moment_exactly_at_tolerance_boundary_counts(self):
        # tolerance is inclusive (<=), so exactly 90s away must still match.
        self.assertTrue(
            has_nearby_structural_context("1000", [{"timestamp": "1090"}], tolerance=90)
        )

    def test_moment_just_outside_tolerance_does_not_count(self):
        self.assertFalse(
            has_nearby_structural_context("1000", [{"timestamp": "1091"}], tolerance=90)
        )


class TestRealGorlestonMatchStaysPipelineReady(unittest.TestCase):
    """The actual point of this whole step, proven against real match data:
    the real Gorleston vs Tilbury match, which has 13 real subs/cards with
    zero nearby key_moments of any kind, must now be pipeline_ready, not
    blocked -- because none of those 13 are goals."""

    FIXTURE_DIR = os.path.join("tests", "fixtures", "gorleston_vs_tilbury_gt")

    def setUp(self):
        self.work_dir = tempfile.mkdtemp()
        for fname in ("match_config.json", "running_summary.json", "match_boundaries.json"):
            shutil.copy(os.path.join(self.FIXTURE_DIR, fname),
                        os.path.join(self.work_dir, fname))

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_real_match_thirteen_non_goal_misses_do_not_block(self):
        result = build_ground_truth_check(self.work_dir)
        self.assertEqual(result["events_checked"], 16)
        self.assertEqual(result["confirmed"], 0)
        self.assertEqual(result["partial"], 3)          # the 3 real goals
        self.assertEqual(result["missed"], 0)            # <- was 13 before Step 19
        self.assertEqual(result["fact_only_no_context"], 13)
        self.assertEqual(result["fact_only_context_available"], 0)
        self.assertTrue(result["pipeline_ready"])
        self.assertEqual(result["rerun_required"], [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
