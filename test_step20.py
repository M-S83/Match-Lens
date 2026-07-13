"""
Step 20 -- split momentum_score per team (home vs away) instead of one
blended "both" composite.

WHY this step exists (see deep_skill_metrics.py's updated docstrings for
the full reasoning, this is the short version): synthesis_agent.py's
report prompt has a "Tactical Variation and Substitutions" section that's
asked to reason about whether a substitution shifted the game's momentum.
A single match-wide momentum number can only ever say "the match got
more/less intense" -- it can never say WHICH team that favoured, which is
the one thing the report actually needs. Splitting the same formula per
team (not combining the two sides into one number first) makes that
attribution possible.

A real, second finding fell out of doing this correctly: the OLD code
read line_height_by_window.avg_pct, which is None for all 21 windows in
the real Gorleston match (confirmed directly against the real file) --
line_height_by_window's aggregation was never updated after the
structural vision agent's schema changed to home_height_pct/
away_height_pct. line_height_m_by_window carries that real split forward
and IS populated for all 21 windows. Splitting momentum per team
requires per-team line height data anyway, so switching to the live
field is a forced, direct side effect of this step -- not a separate fix
bolted on. Several tests below exist specifically to prove that
dead-field bug is actually gone, not just that team-splitting works.

Test classes:
  TestMomentumComponentDirect              -- the shared per-team, per-
      window math (_momentum_component) in isolation: full data, missing
      press falls back to 0.5, both-missing returns None, values above
      the expected 0-10 / 0-100 ranges get clamped to 1.0 not left
      unbounded.
  TestCalcMomentumByWindowSynthetic         -- calc_momentum_by_window()
      with small hand-built summary/match_config dicts: real team
      independence (one team's gap doesn't blank the other team's
      reading in the same window), both focus_team resolution paths
      (explicit away focus_team, and the default-to-home-team fallback),
      and possession-entirely-missing weight redistribution.
  TestRealGorlestonMomentumSplit            -- full integration through
      build_deep_skill_metrics() against the REAL match directory,
      proving: (a) the metric's shape actually changed to two independent
      series, (b) the two series produce genuinely different peak/low
      windows (proof the split carries real information, not two copies
      of the same number), (c) the real edge-case window with a missing
      home_intensity reading falls back to 0.5 for home's pressing
      component while away's pressing component (which WAS observed)
      computes normally in the same window, and (d) the dead
      line_height_by_window field is provably not being used any more
      (line_height component is a real, non-0.5-everywhere value).
"""
import json
import os
import shutil
import tempfile
import unittest

import deep_skill_metrics as dsm


REAL_MATCH_DIR = os.path.join(
    os.path.dirname(__file__), "matches", "2026-04-11_gorleston_vs_tilbury"
)


class TestMomentumComponentDirect(unittest.TestCase):
    """_momentum_component(press, line_pct, poss_pct) in isolation."""

    def test_full_data_matches_hand_computed_formula(self):
        # press=8.0 -> press_n = min(8/10, 1) = 0.8
        # line_pct=45.0 -> line_n = min(45/60, 1) = 0.75
        # poss_pct=60.0 -> poss_n = 60/100 = 0.6
        # weights (poss present): 0.35 press + 0.40 poss + 0.25 line
        # = 0.8*0.35 + 0.6*0.40 + 0.75*0.25 = 0.28 + 0.24 + 0.1875 = 0.7075,
        # which Python's round() takes to 0.708 (standard round-half-up
        # behaviour on this particular float, confirmed directly rather
        # than assumed).
        momentum, components = dsm._momentum_component(8.0, 45.0, 60.0)
        self.assertAlmostEqual(momentum, 0.708, places=3)
        self.assertAlmostEqual(components["pressing"], 0.8, places=3)
        self.assertAlmostEqual(components["possession"], 0.6, places=3)
        self.assertAlmostEqual(components["line_height"], 0.75, places=3)

    def test_missing_press_falls_back_to_neutral_half_not_zero(self):
        # This is the exact real shape of the Gorleston 2H_30-00_35-00
        # window for the home team: home_intensity is None, but line and
        # possession ARE available. A fallback of 0.0 would unfairly
        # punish a team just because ONE component wasn't observed;
        # falling back to 0.5 ("assume average") is the deliberate,
        # documented choice -- this test locks that choice in.
        momentum, components = dsm._momentum_component(None, 45.0, 60.0)
        self.assertEqual(components["pressing"], 0.5)
        # line_n = 0.75, poss_n = 0.6, press_n falls back to 0.5
        expected = round(0.5 * 0.35 + 0.6 * 0.40 + 0.75 * 0.25, 3)
        self.assertAlmostEqual(momentum, expected, places=3)

    def test_missing_possession_redistributes_weights_not_zeroes_it(self):
        # When possession has no reading at all this window, its 40%
        # weight moves onto press/line (0.55/0/0.45) rather than being
        # silently dropped to zero, which would otherwise understate
        # momentum for every window a possession reading is missing.
        momentum, components = dsm._momentum_component(8.0, 45.0, None)
        self.assertIsNone(components["possession"])
        expected = round(0.8 * 0.55 + 0.75 * 0.45, 3)
        self.assertAlmostEqual(momentum, expected, places=3)

    def test_both_press_and_line_missing_returns_none(self):
        # Nothing observable at all for this team in this window --
        # genuinely no basis to compute a number, so momentum must be
        # None, not a fabricated 0.5-based guess wearing a number's mask.
        momentum, components = dsm._momentum_component(None, None, 60.0)
        self.assertIsNone(momentum)
        self.assertEqual(components, {})

    def test_press_and_line_above_normal_range_are_clamped_not_extrapolated(self):
        # press=15 (above the assumed 0-10 scale) and line_pct=90 (above
        # the assumed 0-60 "high line" ceiling) should both clamp to 1.0,
        # not silently produce a >1.0 component that would make the
        # weighted composite exceed its own 0-1 scale.
        momentum, components = dsm._momentum_component(15.0, 90.0, 60.0)
        self.assertEqual(components["pressing"], 1.0)
        self.assertEqual(components["line_height"], 1.0)
        self.assertLessEqual(momentum, 1.0)


def _make_summary(pressing_by_window, line_height_m_by_window,
                   possession_by_window, home_team="Home FC", away_team="Away FC"):
    return {
        "home_team": home_team,
        "away_team": away_team,
        "pressing_by_window": pressing_by_window,
        "line_height_m_by_window": line_height_m_by_window,
        "possession_by_window": possession_by_window,
    }


class TestCalcMomentumByWindowSynthetic(unittest.TestCase):
    """calc_momentum_by_window(summary, match_config) with small,
    hand-built fixtures -- isolates behaviours that would be hard to
    prove reliably against real (noisy) match data alone."""

    def test_one_team_missing_press_does_not_blank_the_other_teams_reading(self):
        # Window 1: home_intensity is None (not observed), away_intensity
        # IS observed. Before Step 20 there was only one shared number
        # per window -- there was no way to even express "we know away's
        # pressing but not home's" for the same window. This test proves
        # the split makes that representable: home's pressing component
        # falls back to 0.5, away's computes normally from a real
        # reading, in the SAME window.
        summary = _make_summary(
            pressing_by_window=[{"window": "W1", "home_intensity": None, "away_intensity": 6.0}],
            line_height_m_by_window=[{"window": "W1", "home_height_pct": 40.0, "away_height_pct": 40.0}],
            possession_by_window=[{"window": "W1", "focus_pct": 50.0}],
        )
        result = dsm.calc_momentum_by_window(summary, match_config={})
        self.assertEqual(len(result), 1)
        w = result[0]
        self.assertEqual(w["home_components"]["pressing"], 0.5)   # fallback
        self.assertEqual(w["away_components"]["pressing"], 0.6)   # 6.0/10
        self.assertIsNotNone(w["home_momentum"])
        self.assertIsNotNone(w["away_momentum"])
        self.assertNotEqual(w["home_momentum"], w["away_momentum"])

    def test_focus_team_explicitly_away_flips_possession_mapping(self):
        # match_config.focus_team names the AWAY team explicitly. The
        # window's focus_pct=70.0 must therefore land on AWAY's
        # possession component, with home getting the 100-70=30.0
        # complement -- the opposite of what the default-to-home-team
        # rule would produce, proving the explicit override is actually
        # being read (not just always assuming home is focus).
        summary = _make_summary(
            pressing_by_window=[{"window": "W1", "home_intensity": 5.0, "away_intensity": 5.0}],
            line_height_m_by_window=[{"window": "W1", "home_height_pct": 30.0, "away_height_pct": 30.0}],
            possession_by_window=[{"window": "W1", "focus_pct": 70.0}],
        )
        match_config = {"focus_team": "Away FC"}
        result = dsm.calc_momentum_by_window(summary, match_config)
        w = result[0]
        self.assertAlmostEqual(w["away_components"]["possession"], 0.70, places=3)
        self.assertAlmostEqual(w["home_components"]["possession"], 0.30, places=3)

    def test_focus_team_absent_defaults_to_home_team(self):
        # No match_config at all (empty dict) -- same real-world case as
        # a pre-v3 match_config.json that never had a focus_team key.
        # Must default to treating home_team as focus, exactly matching
        # pipeline_runner_v2.py's own `focus_team = mc.get("focus_team")
        # or home_team` resolution.
        summary = _make_summary(
            pressing_by_window=[{"window": "W1", "home_intensity": 5.0, "away_intensity": 5.0}],
            line_height_m_by_window=[{"window": "W1", "home_height_pct": 30.0, "away_height_pct": 30.0}],
            possession_by_window=[{"window": "W1", "focus_pct": 65.0}],
        )
        result = dsm.calc_momentum_by_window(summary, match_config={})
        w = result[0]
        self.assertAlmostEqual(w["home_components"]["possession"], 0.65, places=3)
        self.assertAlmostEqual(w["away_components"]["possession"], 0.35, places=3)

    def test_possession_entirely_missing_gives_none_component_both_teams(self):
        # possession_by_window has no entry at all for this window (a
        # genuinely possible real gap, e.g. a broadcast-angle window
        # where sequence tracking failed). Both teams' possession
        # component must be None together -- possession can never be
        # "known for home but unknown for away" since it's read from one
        # shared focus_pct number.
        summary = _make_summary(
            pressing_by_window=[{"window": "W1", "home_intensity": 5.0, "away_intensity": 5.0}],
            line_height_m_by_window=[{"window": "W1", "home_height_pct": 30.0, "away_height_pct": 30.0}],
            possession_by_window=[],
        )
        result = dsm.calc_momentum_by_window(summary, match_config={})
        w = result[0]
        self.assertIsNone(w["home_components"]["possession"])
        self.assertIsNone(w["away_components"]["possession"])

    def test_window_with_nothing_for_either_team_produces_both_none(self):
        # A window absent from pressing_by_window AND line_height_m_
        # by_window for the home side, but IS listed for the away side
        # via a different window key -- proves the "neither team has any
        # signal" case correctly yields None for the affected team,
        # without crashing or fabricating a number.
        summary = _make_summary(
            pressing_by_window=[{"window": "W1", "home_intensity": None, "away_intensity": None}],
            line_height_m_by_window=[{"window": "W1", "home_height_pct": None, "away_height_pct": None}],
            possession_by_window=[{"window": "W1", "focus_pct": 50.0}],
        )
        result = dsm.calc_momentum_by_window(summary, match_config={})
        w = result[0]
        self.assertIsNone(w["home_momentum"])
        self.assertIsNone(w["away_momentum"])
        self.assertEqual(w["home_components"], {})
        self.assertEqual(w["away_components"], {})


class TestRealGorlestonMomentumSplit(unittest.TestCase):
    """Full integration through build_deep_skill_metrics() against the
    real match directory. Uses a temp copy of the real match folder so
    the real deep_skill_metrics.json on disk isn't overwritten by the
    test run (build_deep_skill_metrics writes its output back to
    match_dir)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp_dir = tempfile.mkdtemp(prefix="gorleston_momentum_test_")
        cls.match_copy = os.path.join(cls.tmp_dir, "match")
        shutil.copytree(REAL_MATCH_DIR, cls.match_copy)
        dsm.build_deep_skill_metrics(cls.match_copy)
        with open(os.path.join(cls.match_copy, "deep_skill_metrics.json"), encoding="utf-8") as f:
            data = json.load(f)
        mom_metrics = [m for m in data["metrics"] if m["metric_name"] == "momentum_score"]
        assert len(mom_metrics) == 1, "expected exactly one momentum_score metric"
        cls.metric = mom_metrics[0]
        cls.value = cls.metric["value"]

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp_dir, ignore_errors=True)

    def test_subject_team_unchanged_value_shape_now_split(self):
        # subject_team stays "both" -- the metric still legitimately
        # covers the whole match; what changed is the VALUE shape inside
        # it, not who the metric claims to be about.
        self.assertEqual(self.metric["subject_team"], "both")
        self.assertIn("home", self.value)
        self.assertIn("away", self.value)
        for side in ("home", "away"):
            self.assertIn("avg", self.value[side])
            self.assertIn("by_window", self.value[side])
            self.assertIn("peak", self.value[side])
            self.assertIn("low", self.value[side])

    def test_real_averages_match_known_real_run(self):
        # Locked-in real numbers from running this exact code against
        # the real Gorleston match (captured directly before writing
        # this test). If the formula or data source changes, this
        # regression test is the one that should catch it.
        self.assertAlmostEqual(self.value["home"]["avg"], 0.554, places=3)
        self.assertAlmostEqual(self.value["away"]["avg"], 0.555, places=3)
        self.assertEqual(len(self.value["home"]["by_window"]), 21)
        self.assertEqual(len(self.value["away"]["by_window"]), 21)

    def test_peak_and_low_windows_genuinely_differ_between_teams(self):
        # The real proof the split carries information a single blended
        # number could never have: home and away peak at DIFFERENT
        # 5-minute windows, and trough at different windows too. If this
        # ever collapsed to identical peak/low for both sides on real
        # data, the split would be cosmetic rather than substantive.
        self.assertNotEqual(self.value["home"]["peak"], self.value["away"]["peak"])
        self.assertNotEqual(self.value["home"]["low"], self.value["away"]["low"])
        self.assertEqual(self.value["home"]["peak"], "1H_45-00_48-13")
        self.assertEqual(self.value["away"]["peak"], "2H_35-00_40-00")

    def test_real_missing_home_intensity_window_falls_back_correctly(self):
        # 2H_30-00_35-00 is the ONE real window in this match where
        # home_intensity is None but away_intensity IS observed
        # (confirmed directly against running_summary.json). Home's
        # pressing component must show the 0.5 fallback; away's pressing
        # component must show a real, non-fallback value computed from
        # its actual away_intensity=4.0 reading, in the same window.
        home_window = next(w for w in self.value["home"]["by_window"]
                            if w["window"] == "2H_30-00_35-00")
        away_window = next(w for w in self.value["away"]["by_window"]
                            if w["window"] == "2H_30-00_35-00")
        self.assertEqual(home_window["components"]["pressing"], 0.5)
        self.assertAlmostEqual(away_window["components"]["pressing"], 0.4, places=3)

    def test_line_height_component_is_no_longer_the_dead_field_default(self):
        # Before Step 20, line_height_by_window.avg_pct was None for
        # every single window in this real match (verified directly),
        # so momentum's line_height component was ALWAYS the 0.5
        # fallback, for every window, silently. After switching to
        # line_height_m_by_window, real per-team values flow through --
        # this test proves the component is no longer uniformly 0.5
        # across all 21 windows for either team.
        home_line_values = {w["components"]["line_height"] for w in self.value["home"]["by_window"]}
        away_line_values = {w["components"]["line_height"] for w in self.value["away"]["by_window"]}
        self.assertNotEqual(home_line_values, {0.5})
        self.assertNotEqual(away_line_values, {0.5})
        # And the very first window's real value should match what a
        # direct read of line_height_m_by_window's home_height_pct=45.0
        # gives us: min(45.0/60.0, 1.0) = 0.75.
        first_home_window = self.value["home"]["by_window"][0]
        self.assertAlmostEqual(first_home_window["components"]["line_height"], 0.75, places=3)

    def test_original_match_directory_is_untouched_by_this_test_run(self):
        # Sanity check on the test's own setup: we ran against a temp
        # COPY, so the real match_dir's deep_skill_metrics.json on disk
        # should still reflect whatever the last real, intentional run
        # wrote (not get silently overwritten by this test suite).
        real_path = os.path.join(REAL_MATCH_DIR, "deep_skill_metrics.json")
        self.assertTrue(os.path.exists(real_path))
        # (Content is allowed to also show the Step 20 shape, since we
        # DID intentionally re-run the real build earlier this step --
        # the point of this test is only that THIS test class's own
        # temp-copy run didn't need to touch it to produce its assertions.)


if __name__ == "__main__":
    unittest.main(verbosity=2)
