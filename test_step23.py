"""
test_step23.py -- Match Lens training, Step 23.

Covers the partial fix to defensive_third_turnover_rate:
_count_defending_turnovers() in accumulator.py was written waiting for a
v3 dict-shaped zone_start.vertical_third contract that never arrived --
real data's start_zone is a flat string. Switched to reading it directly.

This is deliberately a PARTIAL fix, not a full one: the real zone
taxonomy is single-dimension per sequence (a sequence is labelled by its
vertical third OR its horizontal channel, never both), so channel-labelled
sequences are correctly excluded rather than guessed at. See
accumulator.py's _count_defending_turnovers() docstring and
deep_skill_metrics.py's _metric_defensive_third_turnover_rate() docstring
for the full reasoning behind each case tested here.
"""
import unittest

import accumulator
import deep_skill_metrics as dsm

REAL_MATCH_DIR = "matches/2026-04-11_gorleston_vs_tilbury"


def _seq(start_zone=None, outcome=None):
    d = {}
    if start_zone is not None:
        d["start_zone"] = start_zone
    if outcome is not None:
        d["outcome"] = outcome
    return d


class TestCountDefendingTurnovers(unittest.TestCase):
    """accumulator._count_defending_turnovers(): the flat-string fix."""

    def test_valid_defending_third_turnover_is_counted(self):
        sequences = [
            _seq(start_zone="defending_third", outcome="lost_possession"),
            _seq(start_zone="defending_third", outcome="clearance"),
        ]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 2)
        self.assertEqual(turnovers, 2)

    def test_valid_defending_third_non_turnover_counts_denominator_only(self):
        sequences = [
            _seq(start_zone="defending_third", outcome="retained"),
        ]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 1)
        self.assertEqual(turnovers, 0)

    def test_non_defending_third_sequences_are_excluded(self):
        # A sequence starting in the middle or attacking third is correctly
        # NOT counted as a defending-third sequence at all -- it should not
        # appear in either the numerator or the denominator.
        sequences = [
            _seq(start_zone="middle_third", outcome="lost_possession"),
            _seq(start_zone="attacking_third", outcome="clearance"),
        ]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 0)
        self.assertEqual(turnovers, 0)

    def test_channel_labelled_sequences_are_excluded_not_guessed(self):
        # This is the deliberate partial-fix boundary: a sequence labelled
        # by channel instead of third has a genuinely unknown vertical
        # third, and must be excluded -- NOT assumed to be non-defending,
        # and NOT assumed to be defending either.
        sequences = [
            _seq(start_zone="left_channel", outcome="lost_possession"),
            _seq(start_zone="right_channel", outcome="clearance"),
            _seq(start_zone="central_channel", outcome="lost_possession"),
        ]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 0)
        self.assertEqual(turnovers, 0)

    def test_old_dict_shaped_zone_start_is_no_longer_read(self):
        # Regression guard: the OLD v3-contract dict shape must not be
        # silently accepted as if it still worked -- proves the switch to
        # the flat string is real, not just an additional accepted shape.
        sequences = [
            {"zone_start": {"vertical_third": "defending"}, "outcome": "lost_possession"},
        ]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 0)
        self.assertEqual(turnovers, 0)

    def test_missing_or_null_sequences_list_does_not_crash(self):
        turnovers, total = accumulator._count_defending_turnovers(None)
        self.assertEqual((turnovers, total), (0, 0))
        turnovers, total = accumulator._count_defending_turnovers([])
        self.assertEqual((turnovers, total), (0, 0))

    def test_sequence_missing_outcome_field_counts_denominator_not_numerator(self):
        # A malformed/incomplete real record (no outcome at all) should
        # still count toward the denominator (it genuinely started in the
        # defending third) but must not be misclassified as a turnover.
        sequences = [_seq(start_zone="defending_third")]
        turnovers, total = accumulator._count_defending_turnovers(sequences)
        self.assertEqual(total, 1)
        self.assertEqual(turnovers, 0)

    def test_real_match_data(self):
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        seqs = data["passes"].get("sequences", [])
        turnovers, total = accumulator._count_defending_turnovers(seqs)
        # Confirmed by direct inspection this step: 126 real sequences
        # start in the defending third (71 lost_possession + 31 clearance
        # + 24 other outcomes = 126), of which 102 are turnovers.
        self.assertEqual(total, 126)
        self.assertEqual(turnovers, 102)


class TestDefensiveThirdTurnoverRateMetric(unittest.TestCase):
    """
    deep_skill_metrics._metric_defensive_third_turnover_rate(): no logic
    change needed here (it already read the right summary keys) -- these
    tests confirm the metric correctly reflects whatever
    _count_defending_turnovers() now produces, end to end, plus the new
    partial-coverage caveat text.
    """

    def test_valid_counts_produce_a_real_rate(self):
        summary = {"defending_third_turnover_count": 6,
                   "defending_third_sequence_count": 10,
                   "windows_complete": 5}
        metric = dsm._metric_defensive_third_turnover_rate(summary, "tactical_wide_static")
        self.assertEqual(metric["value"]["rate"], 0.6)
        self.assertEqual(metric["value"]["turnovers"], 6)
        self.assertEqual(metric["value"]["total_defending_third_seqs"], 10)
        self.assertIn("channel", metric["limitation_note"])  # partial-coverage caveat present

    def test_fewer_than_three_sequences_is_unavailable(self):
        # Guards the < 3 threshold -- must still gate on genuinely tiny
        # samples, exactly like before this fix (this behaviour is
        # unchanged; only the upstream counts feeding it were broken).
        summary = {"defending_third_turnover_count": 1,
                   "defending_third_sequence_count": 2}
        metric = dsm._metric_defensive_third_turnover_rate(summary, "tactical_wide_static")
        self.assertIsNone(metric["value"])
        self.assertEqual(metric["value_type"], "unavailable")

    def test_zero_counts_is_unavailable_not_a_fake_zero_rate(self):
        # Documents the "genuinely no defending-third sequences this match"
        # case, still distinguishable from "we have data but it's a low
        # rate" -- 0/0 must not become a misleading rate of 0.0.
        summary = {"defending_third_turnover_count": 0,
                   "defending_third_sequence_count": 0}
        metric = dsm._metric_defensive_third_turnover_rate(summary, "tactical_wide_static")
        self.assertIsNone(metric["value"])

    def test_calculation_basis_names_the_real_field(self):
        # Regression guard against the old, now-inaccurate documentation
        # string that named the never-real zone_start.vertical_third shape.
        summary = {"defending_third_turnover_count": 6,
                   "defending_third_sequence_count": 10}
        metric = dsm._metric_defensive_third_turnover_rate(summary, "tactical_wide_static")
        self.assertIn("start_zone", metric["calculation_basis"])
        self.assertNotIn("vertical_third", metric["calculation_basis"])

    def test_real_match_data_end_to_end(self):
        # Simulates the fixed accumulator output feeding the metric
        # function, using the real match's real pass sequences.
        data = dsm.load_pipeline_data(REAL_MATCH_DIR)
        seqs = data["passes"].get("sequences", [])
        turnovers, total = accumulator._count_defending_turnovers(seqs)
        summary = dict(data["summary"])
        summary["defending_third_turnover_count"] = turnovers
        summary["defending_third_sequence_count"] = total
        metric = dsm._metric_defensive_third_turnover_rate(summary, "veo_ball_tracking")
        self.assertIsNotNone(metric["value"])
        self.assertEqual(metric["value"]["rate"], 0.81)
        self.assertEqual(metric["value"]["turnovers"], 102)
        self.assertEqual(metric["value"]["total_defending_third_seqs"], 126)
        self.assertFalse(metric["severely_limited"])


if __name__ == "__main__":
    unittest.main()
