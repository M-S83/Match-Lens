"""
test_step25.py -- Match Lens training, Step 25.

Covers the pressing_by_window double-bug fix:

  1. accumulator._extract_pressing_window() used to read a "scores" array
     off each window's pressing dict that NEVER exists in real data (the
     model returns flat home_intensity/away_intensity/home_trigger/
     away_trigger fields instead -- verified directly against a real
     Gorleston merged log). dict.get() on a missing key doesn't raise, it
     silently returns None/[], so avg_score/peak/observations were always
     empty for every match, every window, since this was written.

  2. Two DOWNSTREAM consumers were each separately built assuming the
     fields THIS function now actually writes:
       - deep_skill_metrics.calc_momentum_by_window() reads
         pressing_by_window[i]["home_intensity"]/["away_intensity"]
         directly for its per-team momentum split -- those keys never
         existed on the dict before this fix, so momentum's pressing
         component silently fell back to its neutral 0.5 default in
         every window of every match.
       - deep_skill_metrics.calc_pressing_intensity() reads "avg_score"
         (always None before) for the match-wide pressing_intensity_score
         metric.
       - deep_skill_metrics.calc_pressing_trigger_consistency() reads
         each window's "observations" list of {"trigger": ...} dicts
         (always [] before) to find the dominant press trigger.

This file tests the extraction helper directly (unit tests) AND exercises
the two real downstream metric functions against summaries built purely
from _extract_pressing_window() output (integration tests) -- proving the
fix doesn't just look right in isolation, it actually unblocks the
metrics that were silently broken because of it.
"""
import unittest

import accumulator
import deep_skill_metrics as dsm


def _window(window_id="1H_00-00_05-00", agent_id="structural_1",
            home_intensity=None, away_intensity=None,
            home_trigger=None, away_trigger=None,
            pressing_is_dict=True):
    """Build a fake merged-window dict with a pressing block, matching
    the real schema (pipeline_runner_v2.py's PRESSING TRIGGER
    VOCABULARY section) rather than the old, never-real scores[] shape."""
    w = {"window": window_id, "agent_id": agent_id}
    if pressing_is_dict:
        w["pressing"] = {
            "home_intensity": home_intensity,
            "away_intensity": away_intensity,
            "home_trigger":   home_trigger,
            "away_trigger":   away_trigger,
        }
    else:
        w["pressing"] = pressing_is_dict  # deliberately wrong type, e.g. None or a string
    return w


class TestExtractPressingWindowValidCases(unittest.TestCase):
    """Real-shaped input, both teams reporting."""

    def test_both_teams_present_matches_real_log_shape(self):
        # Exact values pulled from the real merged log inspected during
        # the investigation: {"home_intensity": 3.5, "away_intensity": 4.0,
        # "home_trigger": "cb_receiving_turned", "away_trigger": "back_pass"}.
        w = _window(home_intensity=3.5, away_intensity=4.0,
                    home_trigger="cb_receiving_turned", away_trigger="back_pass")
        entry = accumulator._extract_pressing_window(w)

        self.assertEqual(entry["home_intensity"], 3.5)
        self.assertEqual(entry["away_intensity"], 4.0)
        # avg_score: blended match-wide figure, (3.5 + 4.0) / 2 = 3.75
        self.assertEqual(entry["avg_score"], 3.75)
        # Both teams reported a trigger -> two observations.
        self.assertEqual(len(entry["observations"]), 2)
        triggers = {o["trigger"] for o in entry["observations"]}
        self.assertEqual(triggers, {"cb_receiving_turned", "back_pass"})

    def test_cb_receiving_turned_is_not_corrupted_by_normalise_trigger(self):
        # This is the specific bug that WOULD be introduced if we ran
        # home_trigger through _normalise_trigger(): that helper matches
        # the substring "turned" and would misclassify the real canonical
        # code "cb_receiving_turned" as "defender_facing_own_goal".
        w = _window(home_intensity=5.0, home_trigger="cb_receiving_turned")
        entry = accumulator._extract_pressing_window(w)
        home_obs = [o for o in entry["observations"] if o["team"] == "home"]
        self.assertEqual(len(home_obs), 1)
        self.assertEqual(home_obs[0]["trigger"], "cb_receiving_turned")

    def test_no_trigger_observed_is_kept_as_a_real_observation(self):
        # The prompt defines "no_trigger_observed" as a meaningful finding
        # (pressing happened, pattern unclear) -- distinct from null (no
        # pressing at all). It must NOT be filtered out like null is.
        w = _window(home_intensity=2.0, home_trigger="no_trigger_observed")
        entry = accumulator._extract_pressing_window(w)
        self.assertEqual(len(entry["observations"]), 1)
        self.assertEqual(entry["observations"][0]["trigger"], "no_trigger_observed")


class TestExtractPressingWindowEdgeCases(unittest.TestCase):
    """Missing data, one-sided data, and malformed input."""

    def test_only_one_team_reporting_averages_that_team_alone(self):
        w = _window(home_intensity=6.0, away_intensity=None,
                    home_trigger="gk_in_possession", away_trigger=None)
        entry = accumulator._extract_pressing_window(w)
        self.assertEqual(entry["home_intensity"], 6.0)
        self.assertIsNone(entry["away_intensity"])
        # avg_score should be exactly the one present value, not skewed
        # by treating the missing team as zero.
        self.assertEqual(entry["avg_score"], 6.0)
        self.assertEqual(len(entry["observations"]), 1)
        self.assertEqual(entry["observations"][0]["team"], "home")

    def test_null_trigger_string_is_dropped_not_kept(self):
        # Real logs can emit the literal string "null" for a trigger,
        # matching the prompt's own vocabulary entry for "no pressing
        # observed". This must be filtered out, same as None is.
        w = _window(home_intensity=None, away_intensity=None,
                    home_trigger="null", away_trigger="none")
        entry = accumulator._extract_pressing_window(w)
        self.assertEqual(entry["observations"], [])

    def test_both_teams_entirely_unobservable(self):
        w = _window(home_intensity=None, away_intensity=None,
                    home_trigger=None, away_trigger=None)
        entry = accumulator._extract_pressing_window(w)
        self.assertIsNone(entry["home_intensity"])
        self.assertIsNone(entry["away_intensity"])
        self.assertIsNone(entry["avg_score"])
        self.assertEqual(entry["observations"], [])

    def test_missing_pressing_key_entirely(self):
        # A window dict with no "pressing" key at all (e.g. a malformed
        # or partial merged log) must not raise -- everything degrades
        # to the same "unobservable" shape as the explicit-None case.
        w = {"window": "1H_00-00_05-00", "agent_id": "structural_1"}
        entry = accumulator._extract_pressing_window(w)
        self.assertIsNone(entry["home_intensity"])
        self.assertIsNone(entry["avg_score"])
        self.assertEqual(entry["observations"], [])

    def test_pressing_value_is_wrong_type_not_a_dict(self):
        # Defensive case: pressing is present but is e.g. a bare string
        # or None instead of a dict (malformed upstream data).
        w = _window(pressing_is_dict=None)
        entry = accumulator._extract_pressing_window(w)
        self.assertIsNone(entry["home_intensity"])
        self.assertEqual(entry["observations"], [])

    def test_peak_and_peak_ts_are_not_present_at_all(self):
        # Regression guard: the old dead "peak"/"peak_ts" fields must be
        # gone entirely, not just present-and-None -- nothing downstream
        # reads them (confirmed by repo-wide grep during the investigation)
        # and keeping a permanently-null field implies data that can never
        # actually appear under the real schema.
        w = _window(home_intensity=3.0)
        entry = accumulator._extract_pressing_window(w)
        self.assertNotIn("peak", entry)
        self.assertNotIn("peak_ts", entry)

    def test_window_and_agent_id_are_still_carried_through(self):
        w = _window(window_id="2H_10-00_15-00", agent_id="structural_7",
                    home_intensity=1.0)
        entry = accumulator._extract_pressing_window(w)
        self.assertEqual(entry["window"], "2H_10-00_15-00")
        self.assertEqual(entry["agent_id"], "structural_7")


class TestDownstreamMomentumNoLongerUsesNeutralFallback(unittest.TestCase):
    """
    Integration check: deep_skill_metrics.calc_momentum_by_window() reads
    home_intensity/away_intensity directly off pressing_by_window. Before
    this fix, those keys never existed, so _momentum_component() always
    fell back to press_n = 0.5 (neutral) for every window. Build a
    summary purely from the new extraction and confirm real intensity
    values now flow through instead of the neutral fallback.
    """

    def test_high_home_pressing_produces_above_neutral_component(self):
        w = _window(home_intensity=9.0, away_intensity=1.0,
                    home_trigger="back_pass", away_trigger="gk_in_possession")
        pressing_entry = accumulator._extract_pressing_window(w)
        summary = {
            "home_team": "Gorleston", "away_team": "Tilbury",
            "pressing_by_window": [pressing_entry],
            "line_height_m_by_window": [],
            "possession_by_window": [],
        }
        result = dsm.calc_momentum_by_window(summary, match_config={})
        self.assertEqual(len(result), 1)
        home_components = result[0]["home_components"]
        away_components = result[0]["away_components"]
        # Before the fix these would both silently read 0.5 (neutral).
        # home_intensity=9.0 -> press_n = min(9.0/10.0, 1.0) = 0.9
        self.assertAlmostEqual(home_components["pressing"], 0.9)
        # away_intensity=1.0 -> press_n = min(1.0/10.0, 1.0) = 0.1
        self.assertAlmostEqual(away_components["pressing"], 0.1)
        # The two teams must now produce DIFFERENT momentum figures --
        # under the old dead-field bug both would have been identical
        # (both stuck on the same 0.5 neutral pressing component).
        self.assertNotEqual(result[0]["home_momentum"], result[0]["away_momentum"])


class TestDownstreamPressingIntensityScore(unittest.TestCase):
    """
    Integration check: deep_skill_metrics.calc_pressing_intensity() reads
    "avg_score" off each pressing_by_window entry for the match-wide
    pressing_intensity_score metric. Before this fix avg_score was always
    None, so this always returned (0.0, 0.0, 0, ["suggestive"]).
    """

    def test_real_shaped_windows_produce_a_real_nonzero_average(self):
        windows = [
            _window(window_id="1H_00-00_05-00", home_intensity=3.5, away_intensity=4.0),
            _window(window_id="1H_05-00_10-00", home_intensity=5.5, away_intensity=6.5),
        ]
        pressing_by_window = [accumulator._extract_pressing_window(w) for w in windows]
        summary = {"pressing_by_window": pressing_by_window}

        avg_10, avg_01, sample_size, tier = dsm.calc_pressing_intensity(summary)

        # Window 1 avg_score = (3.5+4.0)/2 = 3.75; window 2 = (5.5+6.5)/2 = 6.0
        # Match-wide average of the two windows' avg_score = (3.75+6.0)/2 = 4.875 -> 4.88
        self.assertEqual(avg_10, 4.88)
        self.assertEqual(sample_size, 2)
        self.assertEqual(tier, ["direct"])
        # Before the fix this was unconditionally (0.0, 0.0, 0, ["suggestive"]).
        self.assertNotEqual(sample_size, 0)


class TestDownstreamPressingTriggerConsistency(unittest.TestCase):
    """
    Integration check: deep_skill_metrics.calc_pressing_trigger_consistency()
    reads window.get("observations", []) across all pressing_by_window
    entries. Before this fix "observations" was always [] for every
    window (the dead scores[] read never matched anything), so this
    metric always fell through to the weak key_moments-inference path.
    """

    def test_dominant_trigger_is_computed_from_real_observations(self):
        windows = [
            _window(window_id="1H_00-00_05-00", home_intensity=4.0, home_trigger="back_pass"),
            _window(window_id="1H_05-00_10-00", home_intensity=3.0, home_trigger="back_pass"),
            _window(window_id="1H_10-00_15-00", home_intensity=5.0, home_trigger="gk_in_possession"),
        ]
        pressing_by_window = [accumulator._extract_pressing_window(w) for w in windows]
        summary = {"pressing_by_window": pressing_by_window, "transitions": []}

        result, total_obs, tier = dsm.calc_pressing_trigger_consistency(summary)

        self.assertIsNotNone(result)
        self.assertEqual(total_obs, 3)
        self.assertEqual(tier, ["direct"])
        self.assertEqual(result["dominant_trigger"], "back_pass")
        self.assertEqual(result["count_using_dominant_trigger"], 2)
        # 2 of 3 = 67%
        self.assertEqual(result["pct_using_dominant_trigger"], 67)

    def test_no_real_observations_still_falls_back_gracefully(self):
        # Windows with genuinely no pressing observed at all (both teams
        # null) must not fabricate a dominant trigger -- confirms the
        # fallback path (key_moments inference) is still reachable and
        # this fix didn't just mask the "no data" case with fake data.
        windows = [_window(window_id="1H_00-00_05-00")]
        pressing_by_window = [accumulator._extract_pressing_window(w) for w in windows]
        summary = {"pressing_by_window": pressing_by_window, "key_moments": [], "transitions": []}

        result, total_obs, tier = dsm.calc_pressing_trigger_consistency(summary)
        self.assertIsNone(result)
        self.assertEqual(tier, ["suggestive"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
