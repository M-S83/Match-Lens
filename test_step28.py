"""
Step 28: rest_defence_security_score was always wrong (score=1.0, every
match, regardless of what actually happened defensively).

Root cause had two layers stacked on top of each other:
  1. calc_rest_defence_security() read line_height_by_window, a field Step 22
     already proved is dead on real data (avg_pct/start_pct/end_pct always
     null). The live field is line_height_m_by_window.
  2. Unlike Step 22's compactness fix, a field-name swap alone doesn't work
     here: line_height_m_by_window's own "shifts" sub-list is ALSO always
     empty on every real window, because it's sourced from a
     defensive_line.notable_shifts field that belonged to an older
     per-window extraction schema and was never carried into the current
     merged-window format (confirmed directly against the real match's raw
     agent_logs/*_merged.json files -- notable_shifts simply isn't a key
     there anymore).

The fix redefines the signal: instead of waiting for an LLM-narrated
"shift" that structurally never arrives, compute backward line-height
movement directly from consecutive windows' avg_pct values in
line_height_m_by_window -- data that IS real and populated on every real
window. Because this new signal is a bounded 0-1 rate (each transition is
either backward or not) rather than an unbounded narrated-shift count,
rest_defence_category()'s thresholds were rescaled to match.

Four test classes:
  TestDeadFieldConfirmation        -- proves the OLD signal source really
                                       is structurally dead, not just wrong
                                       on this one match.
  TestCalcRestDefenceSecurityCorrectness -- proves the NEW calculation is
                                       correct against real data and clean
                                       synthetic all-forward / all-backward
                                       cases.
  TestEdgeCases                    -- no data, single data point, and
                                       None-valued windows are all handled
                                       honestly (unavailable, not a fake
                                       number).
  TestCategoryRescale              -- proves the full 0-1 range of the new,
                                       correctly-bounded signal actually
                                       reaches all five category bands.
"""
import json
import os
import unittest

from deep_skill_metrics import calc_rest_defence_security, rest_defence_category

REAL_MATCH_DIR = os.path.join(
    os.path.dirname(__file__), "matches", "2026-04-11_gorleston_vs_tilbury"
)


class TestDeadFieldConfirmation(unittest.TestCase):
    """Prove the OLD signal source is structurally dead, not just unlucky."""

    def test_old_field_shifts_always_empty_on_real_match(self):
        rs_path = os.path.join(REAL_MATCH_DIR, "running_summary.json")
        rs = json.load(open(rs_path))
        old_field = rs.get("line_height_by_window", [])
        self.assertTrue(len(old_field) > 0, "expected the dead field to still be present")
        for w in old_field:
            # avg_pct is the dead field Step 22 already documented.
            self.assertIsNone(w.get("avg_pct"))
            # shifts is the ADDITIONAL dead sub-field this step is fixing.
            self.assertEqual(w.get("shifts", []), [])

    def test_notable_shifts_never_populated_in_raw_merged_files(self):
        agent_logs_dir = os.path.join(REAL_MATCH_DIR, "agent_logs")
        merged_files = [
            f for f in os.listdir(agent_logs_dir) if f.endswith("_merged.json")
        ]
        self.assertTrue(len(merged_files) > 0, "expected merged window files to exist")
        checked = 0
        for fname in merged_files:
            d = json.load(open(os.path.join(agent_logs_dir, fname)))
            dl = d.get("defensive_line")
            if dl is None:
                continue
            checked += 1
            # This is the actual claim being tested: the real, current
            # per-window schema never carries a notable_shifts key at all --
            # it's not that the key exists and is empty, it's genuinely gone.
            self.assertNotIn("notable_shifts", dl)
            self.assertIn("home_height_pct", dl)
            self.assertIn("away_height_pct", dl)
        self.assertTrue(checked > 0, "expected at least one merged file with defensive_line")


class TestCalcRestDefenceSecurityCorrectness(unittest.TestCase):
    """Prove the new calculation is correct, on both real and synthetic data."""

    def test_real_match_gives_meaningful_nonconstant_score(self):
        rs = json.load(open(os.path.join(REAL_MATCH_DIR, "running_summary.json")))
        score, backward_rate, windows, tags = calc_rest_defence_security(rs)
        # Pinned to the exact real numbers as a regression guard: 6 backward
        # transitions out of 20 possible (21 windows) = 0.3 exactly.
        self.assertEqual(windows, 21)
        self.assertAlmostEqual(backward_rate, 0.3, places=3)
        self.assertAlmostEqual(score, 0.70, places=2)
        self.assertEqual(rest_defence_category(backward_rate), "moderate")
        # The old, broken behaviour was always score=1.0 -- assert we're not
        # accidentally still there.
        self.assertNotEqual(score, 1.0)

    def test_all_forward_pushes_gives_perfect_security(self):
        summary = {"line_height_m_by_window": [
            {"window": f"w{i}", "avg_pct": pct}
            for i, pct in enumerate([30, 35, 40, 45, 50])  # strictly rising = all forward
        ]}
        score, backward_rate, windows, tags = calc_rest_defence_security(summary)
        self.assertEqual(backward_rate, 0.0)
        self.assertEqual(score, 1.0)
        self.assertEqual(windows, 5)
        self.assertEqual(rest_defence_category(backward_rate), "very_secure")

    def test_all_backward_drops_gives_worst_security(self):
        summary = {"line_height_m_by_window": [
            {"window": f"w{i}", "avg_pct": pct}
            for i, pct in enumerate([55, 50, 45, 40, 35])  # strictly falling = all backward
        ]}
        score, backward_rate, windows, tags = calc_rest_defence_security(summary)
        self.assertEqual(backward_rate, 1.0)
        self.assertEqual(score, 0.0)
        self.assertEqual(rest_defence_category(backward_rate), "very_vulnerable")

    def test_equal_consecutive_values_are_not_counted_as_backward(self):
        # A hold (line staying exactly level) is not a backward shift --
        # only a genuine drop should count, per the metric's own definition.
        summary = {"line_height_m_by_window": [
            {"window": "w0", "avg_pct": 45},
            {"window": "w1", "avg_pct": 45},
            {"window": "w2", "avg_pct": 45},
        ]}
        score, backward_rate, windows, tags = calc_rest_defence_security(summary)
        self.assertEqual(backward_rate, 0.0)
        self.assertEqual(score, 1.0)


class TestEdgeCases(unittest.TestCase):
    """No data, insufficient data, and gappy data must be handled honestly."""

    def test_no_data_returns_unavailable_not_a_fake_number(self):
        score, backward_rate, windows, tags = calc_rest_defence_security({})
        self.assertEqual(score, 0.0)
        self.assertIsNone(backward_rate)
        self.assertEqual(windows, 0)
        self.assertIn("suggestive", tags)

    def test_single_window_cannot_compute_a_transition(self):
        summary = {"line_height_m_by_window": [{"window": "w0", "avg_pct": 45}]}
        score, backward_rate, windows, tags = calc_rest_defence_security(summary)
        # Exactly one data point -- there is nothing to compare it against.
        # Must not silently pretend a rate of 0.0 with high confidence.
        self.assertIsNone(backward_rate)
        self.assertEqual(windows, 1)
        self.assertIn("suggestive", tags)

    def test_none_valued_windows_are_skipped_not_treated_as_zero(self):
        # A None avg_pct in the middle must not be silently treated as 0
        # (which would look like a huge, fake backward drop). It should
        # simply be excluded from the comparison entirely.
        summary = {"line_height_m_by_window": [
            {"window": "w0", "avg_pct": 50},
            {"window": "w1", "avg_pct": None},
            {"window": "w2", "avg_pct": 55},  # still a rise vs. the last REAL value
        ]}
        score, backward_rate, windows, tags = calc_rest_defence_security(summary)
        self.assertEqual(windows, 2)  # only the two real data points counted
        self.assertEqual(backward_rate, 0.0)  # 50 -> 55 is a forward push, not backward
        self.assertEqual(score, 1.0)


class TestCategoryRescale(unittest.TestCase):
    """Prove the new bounded 0-1 signal actually reaches all five bands."""

    def test_full_range_reaches_all_five_categories(self):
        self.assertEqual(rest_defence_category(0.0), "very_secure")
        self.assertEqual(rest_defence_category(0.20), "secure")
        self.assertEqual(rest_defence_category(0.40), "moderate")
        self.assertEqual(rest_defence_category(0.60), "vulnerable")
        self.assertEqual(rest_defence_category(1.0), "very_vulnerable")

    def test_none_returns_unknown(self):
        self.assertEqual(rest_defence_category(None), "unknown")


if __name__ == "__main__":
    unittest.main()
