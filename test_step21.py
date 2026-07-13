"""
test_step21.py -- Match Lens training, Step 21.

Covers calc_substitution_impact()'s rewrite: the field-name crash fix,
the window-matching parser fix, and the per-team split, plus the new
window-label helper functions it's built on.

WHY these specific cases: each one is a real bug this step found and
fixed, checked directly against the real Gorleston match before writing
the fix -- see deep_skill_metrics.py's WHY comments on
calc_substitution_impact(), _parse_window_bounds(), and
_window_for_match_minute() for the full reasoning behind each case here.
"""
import copy
import os
import shutil
import tempfile
import unittest

import deep_skill_metrics as dsm

REAL_MATCH_DIR = os.path.join("matches", "2026-04-11_gorleston_vs_tilbury")

REAL_WINDOWS = [
    "1H_00-00_05-00", "1H_05-00_10-00", "1H_10-00_15-00", "1H_15-00_20-00",
    "1H_20-00_25-00", "1H_25-00_30-00", "1H_30-00_35-00", "1H_35-00_40-00",
    "1H_40-00_45-00", "1H_45-00_48-13",
    "2H_00-00_05-00", "2H_05-00_10-00", "2H_10-00_15-00", "2H_15-00_20-00",
    "2H_20-00_25-00", "2H_25-00_30-00", "2H_30-00_35-00", "2H_35-00_40-00",
    "2H_40-00_45-00", "2H_45-00_50-00", "2H_50-00_50-12",
]


class TestParseWindowBounds(unittest.TestCase):
    """
    The regex parser that replaced the old str.replace()+split("-")
    logic. That old logic silently computed end=0 for every window
    (confirmed directly -- see the WHY comment above _WINDOW_LABEL_RE)
    because it split on every hyphen in the label, including the one
    inside the MM-SS pair itself.
    """

    def test_plain_five_minute_window(self):
        # "05-00" is 5 minutes 0 seconds, not a second independent number.
        self.assertEqual(dsm._parse_window_bounds("2H_10-00_15-00"), (2, 10.0, 15.0))

    def test_stoppage_time_window_seconds_component(self):
        # Real window from the real match: 1st-half stoppage running to
        # 48 minutes 13 seconds. If seconds were misread as a second
        # minute value (the old bug's failure mode), this would come out
        # as (1, 45.0, 48.0) or worse -- not 48 + 13/60.
        half, start, end = dsm._parse_window_bounds("1H_45-00_48-13")
        self.assertEqual(half, 1)
        self.assertEqual(start, 45.0)
        self.assertAlmostEqual(end, 48 + 13 / 60, places=4)

    def test_old_bug_would_have_zeroed_every_end(self):
        # Direct regression check for the exact bug this step found:
        # under the OLD parsing logic, every window's `end` came out as
        # 0 regardless of the label. Confirm the NEW parser does not
        # reproduce that -- end must always be >= start for a real window.
        for w in REAL_WINDOWS:
            half, start, end = dsm._parse_window_bounds(w)
            self.assertGreaterEqual(
                end, start,
                f"{w}: end ({end}) < start ({start}) -- old zeroing bug reproduced")

    def test_malformed_label_returns_none_not_a_guess(self):
        # No half prefix, wrong shape, empty string -- all must return
        # None outright rather than a partially-parsed, silently-wrong
        # tuple.
        self.assertIsNone(dsm._parse_window_bounds("garbage"))
        self.assertIsNone(dsm._parse_window_bounds("3H_00-00_05-00"))
        self.assertIsNone(dsm._parse_window_bounds(""))
        self.assertIsNone(dsm._parse_window_bounds(None))


class TestWindowForMatchMinute(unittest.TestCase):
    """
    The half-aware minute->window lookup that replaced window_index().
    window_index() had two independent bugs: it never subtracted 45 for
    2nd-half minutes (so it was comparing match-elapsed minutes directly
    against within-half window bounds), AND it inherited the end=0
    parsing bug above -- so in practice it returned None for every real
    substitution in the match, silently, before this rewrite.

    Real match_boundaries.json shape, used below for the stoppage-time
    disambiguation tests (mirrors the real Gorleston file's structure).
    """

    REAL_BOUNDARIES = {
        "boundaries": {
            "ko_1h":       {"seconds": 351},
            "ht_whistle":  {"seconds": 3244},
            "ko_2h":       {"seconds": 4269},
            "ft_whistle":  {"seconds": 7281},
        }
    }

    def test_first_half_minute_resolves_correctly(self):
        # Minute 12 in the 1st half should land in the window that
        # actually spans 10-15 minutes -- no offset needed pre-HT.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 12)
        self.assertEqual(REAL_WINDOWS[idx], "1H_10-00_15-00")

    def test_second_half_minute_needs_the_45_minute_rebase(self):
        # Real sub: elapsed=57. Within the 2nd half's OWN clock that's
        # minute 12 (57-45), which should land in "2H_10-00_15-00" --
        # NOT in the 1st half's "1H_10-00_15-00" window of the same
        # relative minute, and not in whatever "1H_55-.._60-.." would
        # be (doesn't exist -- 1H windows stop at 48:13 in this match).
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 57)
        self.assertEqual(REAL_WINDOWS[idx], "2H_10-00_15-00")

    def test_boundary_minute_45_is_treated_as_first_half(self):
        # elapsed<=45 is the 1H branch under the naive fallback -- minute
        # 45 exactly must resolve into the 1st half's final
        # (stoppage-extended) window, not the 2nd half's first window.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 45)
        self.assertEqual(REAL_WINDOWS[idx], "1H_40-00_45-00")

    def test_stoppage_minute_WITHOUT_boundaries_is_a_known_wrong_answer(self):
        # Documents the naive fallback's real limitation rather than
        # hiding it: with no match_boundaries.json available, elapsed=47
        # (plausible first-half stoppage) is indistinguishable from an
        # early second-half minute and resolves into the WRONG half's
        # window. This is the accepted approximation for matches missing
        # the boundaries file -- see _resolve_half_and_within_minute()'s
        # WHY comment. If this assertion ever starts failing, the
        # fallback's behaviour has changed and the docstring needs
        # re-checking against this test, not the other way around.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 47, match_boundaries=None)
        self.assertEqual(REAL_WINDOWS[idx], "2H_00-00_05-00")

    def test_stoppage_minute_WITH_boundaries_resolves_correctly(self):
        # Same elapsed=47, but now with the real half-time whistle
        # timestamp available -- this is the actual bug fix. 47 minutes
        # past kickoff (ko_1h_s=351) is still well before the real
        # half-time whistle (ht_s=3244), so it correctly resolves as
        # first-half stoppage, landing in "1H_45-00_48-13".
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 47, self.REAL_BOUNDARIES)
        self.assertEqual(REAL_WINDOWS[idx], "1H_45-00_48-13")

    def test_real_second_half_minute_still_correct_with_boundaries(self):
        # Boundaries must not break the ordinary, unambiguous case --
        # elapsed=57 with real boundaries supplied must still resolve
        # exactly like the no-boundaries case above.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 57, self.REAL_BOUNDARIES)
        self.assertEqual(REAL_WINDOWS[idx], "2H_10-00_15-00")

    def test_malformed_boundaries_falls_back_gracefully(self):
        # A boundaries dict missing expected keys must not crash --
        # it should fall through to the plain elapsed<=45 split.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 12, {"boundaries": {"wrong_key": {}}})
        self.assertEqual(REAL_WINDOWS[idx], "1H_10-00_15-00")

    def test_minute_past_every_window_returns_none(self):
        # elapsed=200 is far past every window in this match, in either
        # half -- must resolve to no match at all, not an out-of-range
        # guess.
        idx = dsm._window_for_match_minute(REAL_WINDOWS, 200)
        self.assertIsNone(idx)

    def test_none_minute_returns_none(self):
        self.assertIsNone(dsm._window_for_match_minute(REAL_WINDOWS, None))


class TestCalcSubstitutionImpactSynthetic(unittest.TestCase):
    """
    calc_substitution_impact() against small, hand-built inputs -- the
    field-name crash fix and the per-team attribution logic, isolated
    from real-match noise.
    """

    def _summary(self):
        return {
            "home_team": "Home FC",
            "away_team": "Away FC",
            "pressing_by_window": [
                {"window": "1H_00-00_05-00", "home_intensity": 5.0, "away_intensity": 5.0},
                {"window": "1H_05-00_10-00", "home_intensity": 5.0, "away_intensity": 5.0},
                {"window": "1H_10-00_15-00", "home_intensity": 5.0, "away_intensity": 5.0},
                {"window": "1H_15-00_20-00", "home_intensity": 8.0, "away_intensity": 2.0},
                {"window": "1H_20-00_25-00", "home_intensity": 8.0, "away_intensity": 2.0},
                {"window": "1H_25-00_30-00", "home_intensity": 8.0, "away_intensity": 2.0},
            ],
            "line_height_m_by_window": [
                {"window": "1H_00-00_05-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
                {"window": "1H_05-00_10-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
                {"window": "1H_10-00_15-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
                {"window": "1H_15-00_20-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
                {"window": "1H_20-00_25-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
                {"window": "1H_25-00_30-00", "home_height_pct": 40.0, "away_height_pct": 40.0},
            ],
            "possession_by_window": [
                {"window": "1H_00-00_05-00", "focus_pct": 50.0},
                {"window": "1H_05-00_10-00", "focus_pct": 50.0},
                {"window": "1H_10-00_15-00", "focus_pct": 50.0},
                {"window": "1H_15-00_20-00", "focus_pct": 50.0},
                {"window": "1H_20-00_25-00", "focus_pct": 50.0},
                {"window": "1H_25-00_30-00", "focus_pct": 50.0},
            ],
        }

    def test_no_crash_on_real_field_shapes(self):
        # This is the direct regression test for the bug found: team is
        # a plain string, player_off/player_on are plain strings -- the
        # OLD code called .get() on the string and raised AttributeError
        # on every real match. This must not raise.
        match_config = {
            "substitutions": [
                {"team": "Home FC", "player_off": "A. Smith", "player_on": "B. Jones",
                 "time": {"elapsed": 17}},
            ]
        }
        try:
            result = dsm.calc_substitution_impact(self._summary(), match_config)
        except AttributeError as e:
            self.fail(f"calc_substitution_impact crashed on real field shapes: {e}")
        self.assertEqual(len(result), 1)

    def test_player_names_extracted_correctly_not_always_question_mark(self):
        # The old code read sub["player"]/sub["assist"] (goal-event field
        # names) instead of the real player_off/player_on -- so names
        # always silently came out as "?". Confirm the real names now
        # come through.
        match_config = {
            "substitutions": [
                {"team": "Home FC", "player_off": "A. Smith", "player_on": "B. Jones",
                 "time": {"elapsed": 17}},
            ]
        }
        result = dsm.calc_substitution_impact(self._summary(), match_config)
        self.assertEqual(result[0]["player_off"], "A. Smith")
        self.assertEqual(result[0]["player_on"], "B. Jones")

    def test_home_substitution_shows_positive_momentum_shift(self):
        # Sub at minute 17 (window index 3, "1H_15-00_20-00") for the
        # home team. Home's press jumps from 5.0 to 8.0 right at that
        # window while away's drops 5.0->2.0 -- home's own momentum
        # should read higher after than before, and the shift should be
        # attributed to "home" specifically, not blended with away's
        # opposite-direction move.
        match_config = {
            "substitutions": [
                {"team": "Home FC", "player_off": "A. Smith", "player_on": "B. Jones",
                 "time": {"elapsed": 17}},
            ]
        }
        result = dsm.calc_substitution_impact(self._summary(), match_config)
        entry = result[0]
        self.assertEqual(entry["substituting_side"], "home")
        self.assertIsNotNone(entry["team_momentum_shift"])
        self.assertGreater(entry["team_momentum_shift"], 0)
        # And the away side's own momentum in the "after" snapshot must
        # have MOVED THE OTHER WAY -- proving the two sides are genuinely
        # independent, not two views of one blended number.
        self.assertLess(entry["after"]["away"]["momentum"], entry["before"]["away"]["momentum"])

    def test_unresolvable_team_name_does_not_guess_a_side(self):
        # Team name matching neither roster entry -- a real-world data
        # inconsistency this function must not paper over by guessing.
        match_config = {
            "substitutions": [
                {"team": "Some Other FC", "player_off": "X", "player_on": "Y",
                 "time": {"elapsed": 17}},
            ]
        }
        result = dsm.calc_substitution_impact(self._summary(), match_config)
        entry = result[0]
        self.assertIsNone(entry["substituting_side"])
        self.assertIsNone(entry["team_momentum_shift"])
        # Both sides' snapshots must still be populated (context is still
        # useful even when we can't attribute it to a side), just the
        # attribution-specific fields are None.
        self.assertIsNotNone(entry["before"]["home"]["momentum"])
        self.assertIsNotNone(entry["before"]["away"]["momentum"])

    def test_no_substitutions_returns_empty_list(self):
        match_config = {"substitutions": []}
        self.assertEqual(dsm.calc_substitution_impact(self._summary(), match_config), [])

    def test_missing_substitutions_key_returns_empty_list(self):
        match_config = {}
        self.assertEqual(dsm.calc_substitution_impact(self._summary(), match_config), [])

    def test_minute_with_no_matching_window_is_skipped_not_crashed(self):
        # elapsed=500 resolves to no window at all in this tiny synthetic
        # window set -- must be silently skipped (documented "continue"
        # behaviour), not raise or fabricate a result.
        match_config = {
            "substitutions": [
                {"team": "Home FC", "player_off": "A", "player_on": "B",
                 "time": {"elapsed": 500}},
            ]
        }
        result = dsm.calc_substitution_impact(self._summary(), match_config)
        self.assertEqual(result, [])


class TestRealGorlestonSubstitutionImpact(unittest.TestCase):
    """
    Integration check against the real match. Uses a tempfile copy so
    the real deep_skill_metrics.json is never overwritten by this test
    suite (same precaution test_step20.py already established).
    """

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp()
        cls.match_copy = os.path.join(cls.tmp_dir, "match")
        shutil.copytree(REAL_MATCH_DIR, cls.match_copy)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def _load(self):
        import json
        with open(os.path.join(self.match_copy, "running_summary.json")) as f:
            summary = json.load(f)
        with open(os.path.join(self.match_copy, "match_config.json")) as f:
            match_config = json.load(f)
        boundaries_path = os.path.join(self.match_copy, "match_boundaries.json")
        with open(boundaries_path) as f:
            match_boundaries = json.load(f)
        return summary, match_config, match_boundaries

    def test_all_ten_real_substitutions_are_matched(self):
        # Real match has exactly 10 substitutions, all in the 2nd half.
        # Before this fix, ALL 10 were silently dropped (window_index
        # returned None for every one, due to the end=0 parsing bug
        # compounded with the missing half-rebase).
        summary, match_config, match_boundaries = self._load()
        result = dsm.calc_substitution_impact(summary, match_config, match_boundaries)
        self.assertEqual(len(result), 10)

    def test_every_real_substitution_resolves_to_a_known_side(self):
        # Real match's substitution team names are exactly "Gorleston"
        # or "Tilbury" -- both must match a roster side (no unresolved
        # None-side entries in real, clean data).
        summary, match_config, match_boundaries = self._load()
        result = dsm.calc_substitution_impact(summary, match_config, match_boundaries)
        for entry in result:
            self.assertIn(entry["substituting_side"], ("home", "away"))

    def test_shift_values_are_not_all_identical(self):
        # If this were still silently reading one blended series (the
        # bug this step fixes), or if the per-team split were wired
        # wrong, every substitution's shift could plausibly come out
        # identical or trivially zero. Real subs at different minutes
        # against a changing match should show genuine variation.
        summary, match_config, match_boundaries = self._load()
        result = dsm.calc_substitution_impact(summary, match_config, match_boundaries)
        shifts = {e["team_momentum_shift"] for e in result if e["team_momentum_shift"] is not None}
        self.assertGreater(len(shifts), 1, f"shifts lack variation: {shifts}")

    def test_wired_into_full_metric_build(self):
        # End-to-end: build_deep_skill_metrics() must now actually call
        # calc_substitution_impact() and register its result -- the
        # producer/consumer gap this step closes. Checked via the
        # returned dict's full (pre-active-filter) metrics list.
        result = dsm.build_deep_skill_metrics(self.match_copy)
        names = [m["metric_name"] for m in result["metrics"]] + \
                result.get("low_confidence_metrics", [])
        self.assertIn("substitution_impact", names)

    def test_registered_metric_value_matches_direct_call(self):
        summary, match_config, match_boundaries = self._load()
        direct = dsm.calc_substitution_impact(summary, match_config, match_boundaries)
        result = dsm.build_deep_skill_metrics(self.match_copy)
        # substitution_impact is low-confidence for this source and so
        # is filtered out of the "active" list by the pre-existing
        # confidence gating (same bucket as compactness_score and
        # several other established metrics for this same source) --
        # find it via the unfiltered per-metric loop instead of assuming
        # it's in result["metrics"].
        found = None
        for m in result["metrics"]:
            if m["metric_name"] == "substitution_impact":
                found = m
                break
        if found is not None:
            self.assertEqual(len(found["value"]), len(direct))
        else:
            # Confirm it really is the confidence gate excluding it, not
            # a silent computation failure -- it must appear in the
            # diagnostic low-confidence bucket instead.
            self.assertIn("substitution_impact", result.get("low_confidence_metrics", []))


if __name__ == "__main__":
    unittest.main()
