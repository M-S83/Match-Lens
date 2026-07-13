"""
test_step22.py -- Match Lens training, Step 22.

Covers the 7 field-name / dead-field bugs fixed this step, all sharing
just two root causes:

  1. line_height_by_window's avg_pct is a dead field on real data (the
     structural agent never emits it -- only home_height_pct/
     away_height_pct). line_height_m_by_window is the real, populated
     sibling field. Affects: calc_line_height_range, calc_compactness.

  2. Several functions read "zone_start"/"zone_end" but real pass-sequence
     records use "start_zone"/"end_zone". Affects: calc_width_usage,
     calc_pattern_reliability (plus a second, independent bug inside it --
     BASELINE_ROUTES used the bare word "middle" instead of the real
     "middle_third"), calc_build_up_route_diversity,
     calc_build_up_effectiveness.

  Plus one unrelated field-name bug: calc_attacking_support_score read
  "length"/"sequence_length" keys that don't exist on real records; the
  real field is "passes".

WHY these specific cases: each one is a real bug found by directly
inspecting the real Gorleston vs Tilbury match data before writing the
fix -- see deep_skill_metrics.py's "Step 22 fix" comments on each
function for the full reasoning. defensive_third_turnover_rate was
investigated in the same pass but intentionally NOT fixed this step
(only a partial fix is possible -- flat start_zone strings only give a
known vertical third for 85% of sequences); it's left untouched.
"""
import unittest

import deep_skill_metrics as dsm

REAL_MATCH_DIR = "matches/2026-04-11_gorleston_vs_tilbury"


def _seq(start_zone=None, end_zone=None, outcome=None, passes=None, progressive=None):
    """Build a minimal real-shaped pass-sequence record for synthetic tests."""
    d = {}
    if start_zone is not None:
        d["start_zone"] = start_zone
    if end_zone is not None:
        d["end_zone"] = end_zone
    if outcome is not None:
        d["outcome"] = outcome
    if passes is not None:
        d["passes"] = passes
    if progressive is not None:
        d["progressive"] = progressive
    return d


class TestLineHeightRange(unittest.TestCase):
    """calc_line_height_range: dead line_height_by_window -> real line_height_m_by_window."""

    def test_valid_real_shaped_windows_give_a_range(self):
        # Two windows with real-shaped avg_pct values -- should produce a
        # genuine highest/lowest split, not None.
        summary = {"line_height_m_by_window": [
            {"window": "1H_00-00_05-00", "avg_pct": 40.0, "shifts": []},
            {"window": "1H_05-00_10-00", "avg_pct": 55.0, "shifts": []},
        ]}
        value, windows, tiers = dsm.calc_line_height_range(summary)
        self.assertIsNotNone(value)
        self.assertEqual(value["highest_pct"], 55.0)
        self.assertEqual(value["lowest_pct"], 40.0)
        self.assertEqual(value["range_pct"], 15.0)
        self.assertEqual(windows, 2)
        self.assertEqual(tiers, ["repeated_pattern"])

    def test_old_dead_field_alone_is_no_longer_read(self):
        # Regression guard: if only the OLD field is present (as it would be
        # from a stale/legacy summary), the function must not silently pull
        # numbers from it -- it should report unavailable, not a fake range.
        summary = {"line_height_by_window": [
            {"window": "1H_00-00_05-00", "avg_pct": None, "start_pct": None, "end_pct": None, "shifts": []},
        ]}
        value, windows, tiers = dsm.calc_line_height_range(summary)
        self.assertIsNone(value)
        self.assertEqual(windows, 0)

    def test_fewer_than_two_values_returns_none(self):
        summary = {"line_height_m_by_window": [
            {"window": "1H_00-00_05-00", "avg_pct": 40.0, "shifts": []},
        ]}
        value, windows, tiers = dsm.calc_line_height_range(summary)
        self.assertIsNone(value)
        self.assertEqual(windows, 1)  # windows counted, but not enough distinct values

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        value, windows, tiers = dsm.calc_line_height_range(data["summary"])
        self.assertIsNotNone(value, "real match has 21 real windows -- must not be unavailable")
        self.assertEqual(windows, 21)
        self.assertEqual(value["total_windows"], 21)
        # Real numbers confirmed by direct inspection this step.
        self.assertEqual(value["highest_m"], 57.8)
        self.assertEqual(value["lowest_m"], 42.0)
        self.assertEqual(value["range_m"], 15.8)


class TestCompactness(unittest.TestCase):
    """calc_compactness: same dead-field root cause as line_height_range."""

    def test_valid_real_shaped_windows(self):
        summary = {"line_height_m_by_window": [
            {"avg_pct": 40.0}, {"avg_pct": 50.0},
        ], "pressing_by_window": [{"avg_score": 5.0}]}
        score, avg_m, windows, tiers, category = dsm.calc_compactness(summary)
        self.assertGreater(score, 0.0)
        self.assertEqual(windows, 2)
        self.assertIsNotNone(avg_m)

    def test_missing_new_field_falls_back_to_documented_zero(self):
        # If line_height_m_by_window is absent entirely (e.g. an even older
        # summary shape), the function must degrade to the documented 0.0 /
        # "suggestive" fallback -- not raise, and not silently invent a
        # plausible-looking number from the old dead field.
        summary = {"line_height_by_window": [{"avg_pct": None}]}
        score, avg_m, windows, tiers, category = dsm.calc_compactness(summary)
        self.assertEqual(score, 0.0)
        self.assertIsNone(avg_m)
        self.assertEqual(tiers, ["suggestive"])

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        score, avg_m, windows, tiers, category = dsm.calc_compactness(data["summary"])
        self.assertEqual(windows, 21)
        self.assertEqual(avg_m, 52.2)
        self.assertEqual(score, 0.49)
        self.assertEqual(category, "moderate")


class TestWidthUsage(unittest.TestCase):
    """calc_width_usage: zone_start/zone_end -> real start_zone/end_zone."""

    def test_valid_zone_labelled_sequences_use_accurate_method(self):
        sequences = {"sequences": [
            _seq(start_zone="left_channel", end_zone="attacking_third", outcome="retained"),
            _seq(start_zone="defending_third", end_zone="middle_third", outcome="retained"),
        ]}
        score, total, tiers, method = dsm.calc_width_usage(sequences)
        self.assertEqual(method, "zone_labels")
        self.assertEqual(tiers, ["repeated_pattern"])
        self.assertEqual(score, 0.5)  # 1 of 2 sequences touches a wide channel

    def test_no_zone_labels_falls_back_to_cross_proxy(self):
        # If real records happened to have no channel labels at all, the
        # function must fall back gracefully to the weaker cross-outcome
        # proxy -- not crash, not silently report 0 with high confidence.
        sequences = {"sequences": [
            _seq(start_zone="defending_third", end_zone="middle_third", outcome="cross"),
            _seq(start_zone="middle_third", end_zone="attacking_third", outcome="retained"),
        ]}
        score, total, tiers, method = dsm.calc_width_usage(sequences)
        self.assertEqual(method, "cross_outcomes_proxy")
        self.assertEqual(tiers, ["suggestive"])
        self.assertEqual(score, 0.5)  # 1 of 2 sequences ended in a cross

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        score, total, tiers, method = dsm.calc_width_usage(data["passes"])
        self.assertEqual(method, "zone_labels")
        self.assertEqual(total, 364)
        self.assertEqual(score, 0.13)


class TestPatternReliabilityAndBaselineRoutes(unittest.TestCase):
    """
    calc_pattern_reliability: field-name fix, PLUS the independent
    BASELINE_ROUTES "middle" vs real "middle_third" bug found while fixing
    the field name.
    """

    def test_baseline_routes_use_real_zone_vocabulary(self):
        # Direct check on the constant itself: it must contain the real
        # "middle_third" value, never the old bare "middle".
        for start, end in dsm.BASELINE_ROUTES:
            self.assertNotEqual(start, "middle", "BASELINE_ROUTES must not use the fake 'middle' value")
            self.assertNotEqual(end, "middle", "BASELINE_ROUTES must not use the fake 'middle' value")
        self.assertIn(("defending_third", "middle_third"), dsm.BASELINE_ROUTES)
        self.assertIn(("middle_third", "attacking_third"), dsm.BASELINE_ROUTES)

    def test_baseline_sequences_are_correctly_excluded(self):
        # A mix of baseline and genuinely non-baseline routes -- only the
        # non-baseline one should count once there are enough of them.
        # (Function requires >=10 non-baseline sequences to trust the
        # non-baseline path, so we pad with repeats of both kinds.)
        sequences = {"sequences":
            [_seq(start_zone="defending_third", end_zone="middle_third") for _ in range(20)] +   # baseline
            [_seq(start_zone="defending_third", end_zone="attacking_third") for _ in range(12)]  # non-baseline
        }
        score, top_route, top_count, total, tiers = dsm.calc_pattern_reliability(sequences)
        # total must reflect ONLY the non-baseline pool (12), not all 32
        self.assertEqual(total, 12)
        self.assertEqual(top_count, 12)
        self.assertEqual(score, 1.0)

    def test_fewer_than_ten_non_baseline_falls_back_to_all_sequences(self):
        sequences = {"sequences":
            [_seq(start_zone="defending_third", end_zone="middle_third") for _ in range(20)] +   # baseline
            [_seq(start_zone="defending_third", end_zone="attacking_third") for _ in range(3)]   # non-baseline, <10
        }
        score, top_route, top_count, total, tiers = dsm.calc_pattern_reliability(sequences)
        # fallback path uses ALL 23 sequences as the denominator
        self.assertEqual(total, 23)
        self.assertIn("3-zone encoding", top_route)

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        score, top_route, top_count, total, tiers = dsm.calc_pattern_reliability(data["passes"])
        # Confirmed by direct inspection this step: fixing BOTH the field
        # name AND BASELINE_ROUTES drops non-baseline count from a wrongly
        # inflated 293 down to the genuine 78.
        self.assertEqual(total, 78)
        self.assertEqual(top_count, 17)
        self.assertEqual(score, 0.22)
        self.assertEqual(top_route, "own third -> final third")


class TestBuildUpRouteDiversity(unittest.TestCase):
    """calc_build_up_route_diversity: same zone_start/zone_end -> start_zone/end_zone fix."""

    def test_valid_sequences_count_distinct_routes(self):
        sequences = {"sequences": [
            _seq(start_zone="defending_third", end_zone="middle_third"),
            _seq(start_zone="middle_third", end_zone="attacking_third"),
            _seq(start_zone="defending_third", end_zone="middle_third"),  # duplicate route
        ]}
        score, total, distinct, tiers = dsm.calc_build_up_route_diversity(sequences)
        self.assertEqual(total, 3)
        self.assertEqual(distinct, 2)  # only 2 unique (start,end) pairs

    def test_missing_zone_fields_are_skipped_not_crashed(self):
        sequences = {"sequences": [_seq(outcome="retained")]}  # no zones at all
        score, total, distinct, tiers = dsm.calc_build_up_route_diversity(sequences)
        self.assertEqual(distinct, 0)
        self.assertEqual(score, 0.0)

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        score, total, distinct, tiers = dsm.calc_build_up_route_diversity(data["passes"])
        self.assertEqual(total, 364)
        self.assertEqual(distinct, 22)
        self.assertEqual(score, 1.0)


class TestAttackingSupportScore(unittest.TestCase):
    """calc_attacking_support_score: length/sequence_length -> real 'passes' field."""

    def test_valid_threat_sequences_use_real_passes_field(self):
        passes = {"sequences": [
            _seq(outcome="shot", passes=3),
            _seq(outcome="cross", passes=5),
        ]}
        summary = {"shots_for": []}
        value, samples, tiers = dsm.calc_attacking_support_score(passes, summary)
        self.assertIsNotNone(value)
        self.assertEqual(samples, 2)
        self.assertEqual(value["avg_passes_before_shot"], 4.0)

    def test_zero_passes_sequence_is_excluded(self):
        # passes=0 is falsy, so it's filtered by the truthy check in the
        # list comprehension, then filtered again by the explicit `l > 0`
        # guard -- both agree, so this documents intended, non-contradictory
        # behaviour rather than a bug.
        passes = {"sequences": [
            _seq(outcome="shot", passes=0),
            _seq(outcome="shot", passes=4),
            _seq(outcome="cross", passes=6),
        ]}
        summary = {"shots_for": []}
        value, samples, tiers = dsm.calc_attacking_support_score(passes, summary)
        self.assertEqual(samples, 2)  # the passes=0 sequence dropped out

    def test_fewer_than_two_valid_samples_returns_none(self):
        passes = {"sequences": [_seq(outcome="shot", passes=3)]}
        summary = {"shots_for": []}
        value, samples, tiers = dsm.calc_attacking_support_score(passes, summary)
        self.assertIsNone(value)
        self.assertEqual(tiers, ["suggestive"])

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        value, samples, tiers = dsm.calc_attacking_support_score(data["passes"], data["summary"])
        self.assertIsNotNone(value)
        self.assertEqual(samples, 70)
        self.assertEqual(value["avg_passes_before_shot"], 3.8)


class TestBuildUpEffectiveness(unittest.TestCase):
    """calc_build_up_effectiveness: zone_end/zone_start -> real end_zone/start_zone."""

    def test_valid_sequences_reaching_attacking_third_are_counted(self):
        sequences = {"sequences": [
            _seq(start_zone="defending_third", end_zone="attacking_third", outcome="retained"),
            _seq(start_zone="attacking_third", end_zone="attacking_third", outcome="retained"),  # excluded: started there
            _seq(start_zone="middle_third", end_zone="middle_third", outcome="retained"),          # never reaches it
        ]}
        (score, total, progressive, threats, final_third,
         progressive_rate, final_third_rate, conversion_rate, tiers) = dsm.calc_build_up_effectiveness(sequences)
        self.assertEqual(total, 3)
        self.assertEqual(final_third, 1)  # only the first genuinely progresses into the final third

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        result = dsm.calc_build_up_effectiveness(data["passes"])
        total, final_third, final_third_rate = result[1], result[4], result[6]
        self.assertEqual(total, 364)
        self.assertEqual(final_third, 117)
        self.assertEqual(final_third_rate, 0.321)


class TestFullBuildIntegration(unittest.TestCase):
    """
    End-to-end check: after all 7 fixes, the real match's full metrics
    build must report each fixed metric as genuinely computed (sample
    status "sufficient"), and none of the 7 should still appear in the
    low_confidence_metrics diagnostic bucket.
    """

    FIXED_METRICS = [
        "line_height_range", "compactness_score", "width_usage_score",
        "pattern_reliability_score", "build_up_route_diversity",
        "attacking_support_score", "build_up_effectiveness_score",
    ]

    def test_fixed_metrics_are_sufficient_and_not_low_confidence(self):
        result = dsm.build_deep_skill_metrics(REAL_MATCH_DIR)
        by_name = {m["metric_name"]: m for m in result["metrics"]}
        low_conf = set(result.get("low_confidence_metrics", []))
        for name in self.FIXED_METRICS:
            self.assertIn(name, by_name, f"{name} missing from output entirely")
            self.assertEqual(by_name[name]["sample_status"], "sufficient",
                              f"{name} should be a real, sufficiently-sampled reading now")
            self.assertNotIn(name, low_conf, f"{name} should no longer be low-confidence")

    def test_untouched_genuine_gaps_remain_unavailable(self):
        # Regression guard: the metrics we deliberately did NOT touch this
        # step (real upstream data gaps, or explicitly skipped this step)
        # must still report as unavailable/low-confidence -- proving we
        # didn't accidentally paper over them with a fake fix.
        result = dsm.build_deep_skill_metrics(REAL_MATCH_DIR)
        still_gapped = {"halfspace_occupation_score", "transition_efficiency_score",
                         "compactness_geometry_score", "defensive_third_turnover_rate",
                         "watch_list_summary"}
        self.assertEqual(still_gapped, set(result.get("low_confidence_metrics", [])) & still_gapped)
        for name in still_gapped:
            self.assertIn(name, result.get("low_confidence_metrics", []))

    def test_predictability_score_now_computes_from_real_inputs(self):
        # predictability_score is downstream of pattern_reliability_score and
        # build_up_route_diversity -- confirming it went from None (both
        # inputs were None pre-fix) to a real number is a good end-to-end
        # signal that the fix propagated correctly through the pipeline.
        result = dsm.build_deep_skill_metrics(REAL_MATCH_DIR)
        by_name = {m["metric_name"]: m for m in result["metrics"]}
        value = by_name["predictability_score"]["value"]
        self.assertIsNotNone(value)
        self.assertEqual(value["score"], 0.13)
        self.assertEqual(value["category"], "unpredictable")


if __name__ == "__main__":
    unittest.main()
