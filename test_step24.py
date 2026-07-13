"""
test_step24.py -- Match Lens training, Step 24.

Covers the fix to _count_vertical_progressions() in accumulator.py (was
always reading a vertical_progression field that never exists on real
sequences -- always resolved to "unknown") plus the new
progression_directness_score metric in deep_skill_metrics.py that makes
the fixed data reportable for the first time (nothing consumed this data
in the active v3 codebase before this step).

Like Step 23's defensive_third_turnover_rate fix, this is a deliberate
PARTIAL fix: the real zone taxonomy is single-dimension per sequence, so
a sequence labelled by channel (rather than by third) at either end has
a genuinely unknown vertical progression and is excluded, not guessed.
"""
import unittest

import accumulator
import deep_skill_metrics as dsm

REAL_MATCH_DIR = "matches/2026-04-11_gorleston_vs_tilbury"


def _seq(start_zone=None, end_zone=None):
    d = {}
    if start_zone is not None:
        d["start_zone"] = start_zone
    if end_zone is not None:
        d["end_zone"] = end_zone
    return d


class TestDeriveVerticalProgression(unittest.TestCase):
    """accumulator._derive_vertical_progression(): the matrix lookup itself."""

    def test_same_third_for_all_three_thirds(self):
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", "defending_third"),
            "same_third")
        self.assertEqual(
            accumulator._derive_vertical_progression("middle_third", "middle_third"),
            "same_third")
        self.assertEqual(
            accumulator._derive_vertical_progression("attacking_third", "attacking_third"),
            "same_third")

    def test_forward_progressions(self):
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", "middle_third"),
            "defending_to_middle")
        self.assertEqual(
            accumulator._derive_vertical_progression("middle_third", "attacking_third"),
            "middle_to_attacking")
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", "attacking_third"),
            "defending_to_attacking")

    def test_backward_regressions(self):
        self.assertEqual(
            accumulator._derive_vertical_progression("middle_third", "defending_third"),
            "regression_middle_to_defending")
        self.assertEqual(
            accumulator._derive_vertical_progression("attacking_third", "middle_third"),
            "regression_attacking_to_middle")
        self.assertEqual(
            accumulator._derive_vertical_progression("attacking_third", "defending_third"),
            "regression_attacking_to_defending")

    def test_channel_at_either_end_is_unknown(self):
        # The deliberate partial-fix boundary: a channel label at EITHER
        # end makes the whole progression unknowable, even if the other
        # end is a perfectly good third.
        self.assertEqual(
            accumulator._derive_vertical_progression("left_channel", "attacking_third"),
            "unknown")
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", "right_channel"),
            "unknown")
        self.assertEqual(
            accumulator._derive_vertical_progression("central_channel", "central_channel"),
            "unknown")

    def test_non_zone_outcome_values_are_unknown(self):
        # e.g. end_zone == "end_of_window" (sequence ran out of window
        # time rather than changing zone) -- must not crash or be
        # miscategorised.
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", "end_of_window"),
            "unknown")

    def test_missing_fields_are_unknown(self):
        self.assertEqual(accumulator._derive_vertical_progression(None, None), "unknown")
        self.assertEqual(
            accumulator._derive_vertical_progression("defending_third", None), "unknown")


class TestCountVerticalProgressions(unittest.TestCase):
    """accumulator._count_vertical_progressions(): the per-sequence tally."""

    def test_valid_mixed_sequences_tally_correctly(self):
        sequences = [
            _seq("defending_third", "middle_third"),      # defending_to_middle
            _seq("defending_third", "middle_third"),      # defending_to_middle
            _seq("middle_third", "attacking_third"),       # middle_to_attacking
            _seq("attacking_third", "defending_third"),    # regression_attacking_to_defending
            _seq("middle_third", "middle_third"),           # same_third
        ]
        counts = accumulator._count_vertical_progressions(sequences)
        self.assertEqual(counts["defending_to_middle"], 2)
        self.assertEqual(counts["middle_to_attacking"], 1)
        self.assertEqual(counts["regression_attacking_to_defending"], 1)
        self.assertEqual(counts["same_third"], 1)
        self.assertEqual(counts["unknown"], 0)
        self.assertEqual(sum(counts.values()), 5)

    def test_channel_labelled_sequences_land_in_unknown_not_dropped(self):
        # Unlike _count_defending_turnovers (which drops non-defending-third
        # sequences from the denominator entirely), every sequence here
        # must land in SOME bucket -- channel-labelled ones go to
        # "unknown", they are never simply absent from the count dict.
        sequences = [_seq("left_channel", "right_channel")]
        counts = accumulator._count_vertical_progressions(sequences)
        self.assertEqual(counts["unknown"], 1)
        self.assertEqual(sum(counts.values()), 1)

    def test_old_dict_shaped_vertical_progression_field_is_no_longer_read(self):
        # Regression guard: the OLD (always-unknown) field must not be
        # read even if present -- proves the switch to deriving from
        # start_zone/end_zone is real, not an additional accepted input.
        sequences = [
            {"vertical_progression": "defending_to_middle",
             "start_zone": "attacking_third", "end_zone": "attacking_third"},
        ]
        counts = accumulator._count_vertical_progressions(sequences)
        # Must be derived from start_zone/end_zone (same_third), NOT read
        # from the stale vertical_progression field (defending_to_middle).
        self.assertEqual(counts["same_third"], 1)
        self.assertEqual(counts["defending_to_middle"], 0)

    def test_missing_or_empty_sequences_list_does_not_crash(self):
        counts = accumulator._count_vertical_progressions(None)
        self.assertEqual(sum(counts.values()), 0)
        counts = accumulator._count_vertical_progressions([])
        self.assertEqual(sum(counts.values()), 0)

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        seqs = data["passes"].get("sequences", [])
        counts = accumulator._count_vertical_progressions(seqs)
        # Confirmed by direct inspection this step.
        self.assertEqual(counts["same_third"], 138)
        self.assertEqual(counts["defending_to_middle"], 79)
        self.assertEqual(counts["middle_to_attacking"], 69)
        self.assertEqual(counts["defending_to_attacking"], 17)
        self.assertEqual(counts["regression_middle_to_defending"], 2)
        self.assertEqual(counts["regression_attacking_to_middle"], 0)
        self.assertEqual(counts["regression_attacking_to_defending"], 0)
        self.assertEqual(counts["unknown"], 59)
        self.assertEqual(sum(counts.values()), len(seqs))


class TestProgressionDirectnessMetric(unittest.TestCase):
    """deep_skill_metrics._metric_progression_directness(): the new metric."""

    def test_valid_totals_produce_a_real_rate(self):
        summary = {
            "vertical_progression_totals": {
                "same_third": 2, "defending_to_middle": 3,
                "middle_to_attacking": 2, "defending_to_attacking": 1,
                "regression_middle_to_defending": 1,
                "regression_attacking_to_middle": 0,
                "regression_attacking_to_defending": 0,
                "unknown": 5,
            },
            "windows_complete": 5,
        }
        metric = dsm._metric_progression_directness(summary, "tactical_wide_static")
        # progressed = 3+2+1 = 6, regressed = 1, same = 2, total_known = 9
        self.assertEqual(metric["value"]["progressed_count"], 6)
        self.assertEqual(metric["value"]["regressed_count"], 1)
        self.assertEqual(metric["value"]["same_third_count"], 2)
        self.assertEqual(metric["value"]["total_known"], 9)
        self.assertEqual(metric["value"]["progressive_rate"], round(6 / 9, 2))
        self.assertIn("channel", metric["limitation_note"])

    def test_fewer_than_three_known_is_unavailable(self):
        summary = {"vertical_progression_totals": {
            "same_third": 1, "defending_to_middle": 1, "unknown": 50}}
        metric = dsm._metric_progression_directness(summary, "tactical_wide_static")
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["value_type"], "unavailable")

    def test_missing_totals_key_entirely_is_unavailable_not_a_crash(self):
        metric = dsm._metric_progression_directness({}, "tactical_wide_static")
        self.assertIsNone(metric["value"])

    def test_calculation_basis_names_the_real_fields(self):
        summary = {"vertical_progression_totals": {
            "same_third": 2, "defending_to_middle": 3, "middle_to_attacking": 2}}
        metric = dsm._metric_progression_directness(summary, "tactical_wide_static")
        self.assertIn("start_zone", metric["calculation_basis"])
        self.assertIn("end_zone", metric["calculation_basis"])

    def test_metric_name_does_not_collide_with_existing_metrics(self):
        # Guards the naming decision made this step: this metric must stay
        # distinct from the pre-existing build_up_effectiveness_score and
        # pattern_reliability_score, which measure different things.
        summary = {"vertical_progression_totals": {
            "same_third": 2, "defending_to_middle": 3, "middle_to_attacking": 2}}
        metric = dsm._metric_progression_directness(summary, "tactical_wide_static")
        self.assertEqual(metric["metric_name"], "progression_directness_score")
        self.assertNotIn(metric["metric_name"],
                          {"build_up_effectiveness_score", "pattern_reliability_score"})

    def test_real_match_data_end_to_end(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        seqs = data["passes"].get("sequences", [])
        totals = accumulator._count_vertical_progressions(seqs)
        summary = dict(data["summary"])
        summary["vertical_progression_totals"] = totals
        metric = dsm._metric_progression_directness(summary, "veo_ball_tracking")
        self.assertIsNotNone(metric["value"])
        self.assertEqual(metric["value"]["progressive_rate"], 0.54)
        self.assertEqual(metric["value"]["progressed_count"], 165)
        self.assertEqual(metric["value"]["regressed_count"], 2)
        self.assertEqual(metric["value"]["same_third_count"], 138)
        self.assertEqual(metric["value"]["total_known"], 305)
        self.assertFalse(metric["severely_limited"])

    def test_registered_in_full_pipeline_metric_list(self):
        # Confirms the metric is actually wired into build_deep_skill_metrics(),
        # not just defined and orphaned like vertical_progression_totals was
        # before this step. Checked via unavailable_metrics rather than the
        # confidence-filtered metrics list: the on-disk fixture for this real
        # match predates both the Step 23 and Step 24 accumulator fixes (its
        # counts are still the old always-zero/always-unknown values), so
        # this metric is currently computed as unavailable here and gets
        # dropped by filter_metrics() at the default confidence level -- that
        # is filter_metrics() working as designed, not a sign the metric
        # isn't wired in. unavailable_metrics is populated before filtering
        # and proves the function was called and returned a real result.
        result = dsm.build_deep_skill_metrics(REAL_MATCH_DIR)
        self.assertIn("progression_directness_score", result["unavailable_metrics"])


if __name__ == "__main__":
    unittest.main()
