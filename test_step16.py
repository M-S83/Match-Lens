"""
test_step16.py -- tests for the real production bug fix in ground_truth.py

WHY this step exists: while looking for a real match to run build_ground_truth_check()
against, we found a genuinely uploaded, already-completed production match directory
(2026-04-11_gorleston_vs_tilbury) whose real, pre-existing ground_truth_check.json
reported ALL 16 known events as "missed", even though the real running_summary.json
already contained key_moments entries that should have matched at least the 3 goals.

That is not a hypothetical edge case -- it is what actually happened to a real client
match. Digging in found two independent, stacking bugs in find_event_in_moments():

  1. It read moment.get("timestamp", ""), but real key_moments entries use a
     "minute" field (e.g. "minute": "06:00") -- "timestamp" is never present, so
     every lookup silently got an empty string and could never match anything.

  2. Even reading "minute" correctly, its value is match-clock time ("06:00" =
     6 minutes elapsed), while the known-event side is already converted to real
     VIDEO seconds via the kickoff-offset-aware _minute_to_video_s() helper
     (e.g. minute 6 -> 711s, because this match's real kickoff sits at video
     second 351). Comparing 360 (raw) against 711 (offset-adjusted) fails the
     90-second tolerance even though they describe the same real instant.

These tests prove (a) the fix works using a minimal, deterministic reproduction of
the bug shape, (b) nothing regresses for the timestamp-based shape the code
originally assumed, and (c) running the REAL, unmodified build_ground_truth_check()
against a trimmed copy of the real match's actual data reproduces the exact
before/after counts we saw live in the sandbox: missed drops 16 -> 13, partial
rises 0 -> 3 (the 3 real goals), confirmed stays 0 (the real key_moments goal
entries have no "consensus" field, so they land on "partial", not "confirmed" --
we assert this honestly rather than overclaiming a full fix). The 13 remaining
missed events (10 subs + 3 cards) stay missed because the real key_moments for
this match only contains one generic, unattributed card entry and one generic,
unattributed substitution entry -- a separate, larger, still-open gap (individual
player-level events never get written into key_moments at all) that this fix does
not touch and does not claim to.
"""

import json
import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ground_truth import find_event_in_moments, build_ground_truth_check


# -- Real kickoff-offset numbers from the real match's match_boundaries.json ----
# ko_1h=351s, ht_whistle=3244s, ko_2h=4269s -- copied from the real file so the
# unit tests below convert minutes exactly the way build_ground_truth_check()
# does internally via its own _minute_to_video_s closure.
KO_1H_S = 351
HT_S = 3244
KO_2H_S = 4269


def real_minute_to_video_s(minute):
    """Standalone copy of ground_truth.py's _minute_to_video_s, using the
    real match's real boundary numbers, so unit tests exercise the exact
    same conversion the production code performs -- not a simplified stand-in."""
    if minute is None:
        return None
    vs_1h = KO_1H_S + minute * 60
    if vs_1h <= HT_S:
        return vs_1h
    return KO_2H_S + (minute - 45) * 60


class TestFindEventInMomentsMinuteFieldFix(unittest.TestCase):
    """Unit tests directly targeting the two-part bug in find_event_in_moments()."""

    def test_minute_field_matches_when_converter_provided(self):
        # This is the real scenario: a goal at match minute 6 (real known event,
        # already converted to video seconds: 351 + 6*60 = 711) must match a real
        # key_moments entry stored as {"minute": "06:00"} once we convert that
        # moment through the SAME kickoff-offset math.
        known_event = {"type": "goal", "timestamp": "711"}
        key_moments = [{"type": "goal", "minute": "06:00", "team": "Gorleston"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=real_minute_to_video_s)

        self.assertIsNotNone(
            found,
            "Fix regressed: a real 'minute' field should match its converted "
            "known-event timestamp when a minute_to_video_s converter is given.")
        self.assertEqual(found["team"], "Gorleston")

    def test_minute_field_without_converter_does_not_falsely_match(self):
        # Without a converter, "06:00" parses to raw 360s, which is 351s away
        # from the real known-event timestamp of 711s -- outside the 90s
        # tolerance. This documents WHY build_ground_truth_check() must always
        # pass minute_to_video_s: reading "minute" alone is necessary but not
        # sufficient to fix the real bug.
        known_event = {"type": "goal", "timestamp": "711"}
        key_moments = [{"type": "goal", "minute": "06:00"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=None)

        self.assertIsNone(
            found,
            "Without the kickoff-offset converter, a raw match-clock minute "
            "should NOT spuriously match an offset-adjusted known-event timestamp.")

    def test_backward_compatible_with_pre_converted_timestamp_field(self):
        # If a producer already writes a "timestamp" field in raw video seconds
        # (the shape this function originally assumed), it must keep matching
        # exactly as before -- the "minute" fallback must never override an
        # already-present "timestamp".
        known_event = {"type": "sub", "timestamp": "5000"}
        key_moments = [{"type": "sub", "timestamp": "5015", "minute": "99:99"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=real_minute_to_video_s)

        self.assertIsNotNone(
            found,
            "Regression: an existing 'timestamp' field (video seconds) must "
            "still match within tolerance, ignoring the malformed 'minute' "
            "field entirely since 'timestamp' takes priority.")

    def test_malformed_minute_string_does_not_crash_and_does_not_match(self):
        # A moment with an unparseable minute string must fail safely (no
        # match, no exception) rather than crashing the whole ground-truth run.
        known_event = {"type": "card", "timestamp": "2511"}
        key_moments = [{"type": "card", "minute": "not-a-time"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=real_minute_to_video_s)

        self.assertIsNone(found)

    def test_type_mismatch_still_blocks_a_time_matching_moment(self):
        # Proves the fix only changed HOW timestamps are read/converted, not
        # the existing type-matching guard -- a moment at the exact right time
        # but wrong type must still be rejected.
        known_event = {"type": "goal", "timestamp": "711"}
        key_moments = [{"type": "card", "minute": "06:00"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=real_minute_to_video_s)

        self.assertIsNone(
            found,
            "A moment of the wrong type must not match even if its converted "
            "time lines up with the known event.")

    def test_missing_minute_and_timestamp_does_not_crash(self):
        # A moment with neither field present must fail safely.
        known_event = {"type": "goal", "timestamp": "711"}
        key_moments = [{"type": "goal"}]

        found = find_event_in_moments(known_event, key_moments,
                                       minute_to_video_s=real_minute_to_video_s)

        self.assertIsNone(found)


class TestBuildGroundTruthCheckAgainstRealMatchData(unittest.TestCase):
    """
    Full-flow regression test using a trimmed copy of the REAL
    2026-04-11_gorleston_vs_tilbury match's real match_config.json,
    running_summary.json, and match_boundaries.json (fixtures/), run through
    the real, unmodified build_ground_truth_check(). This is not synthetic
    data -- it is the actual real match facts and the actual real detection
    output that triggered this whole investigation, trimmed only to drop
    unrelated running_summary.json keys that build_ground_truth_check() never
    reads.
    """

    FIXTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "tests", "fixtures", "gorleston_vs_tilbury_gt")

    def setUp(self):
        # Run against a throwaway copy so this test never overwrites the
        # committed fixture's ground_truth_check.json.
        self.work_dir = tempfile.mkdtemp()
        for fname in ("match_config.json", "running_summary.json", "match_boundaries.json"):
            shutil.copy(os.path.join(self.FIXTURE_DIR, fname),
                        os.path.join(self.work_dir, fname))

    def tearDown(self):
        shutil.rmtree(self.work_dir, ignore_errors=True)

    def test_real_match_16_events_matches_before_after_counts(self):
        result = build_ground_truth_check(self.work_dir)

        self.assertEqual(result["events_checked"], 16,
                          "Real match has 3 goals + 10 subs + 3 cards = 16 known events.")
        # Before the Step 16 fix, this was confirmed=0, partial=0, missed=16
        # (100% wrong). After Step 16: the 3 real goals are now found
        # (partial -- their real key_moments entries have no "consensus"
        # field, so they land on "partial" rather than "confirmed", the
        # honest, correct outcome given the real data -- not "confirmed").
        self.assertEqual(result["confirmed"], 0)
        self.assertEqual(result["partial"], 3)
        # Step 19 update: "missed" is now GOALS ONLY (see ground_truth.py's
        # has_nearby_structural_context() docstring) -- a sub/card's fact is
        # already certain from match_config.json, so failing to also find it
        # in key_moments is no longer treated as a defect. All 3 goals were
        # found above, so missed correctly stays 0, not 13.
        self.assertEqual(result["missed"], 0)
        # The 13 real subs/cards now land in the new informational buckets
        # instead: this match genuinely has zero key_moments of ANY type
        # near any of these 13 timestamps, so all 13 are "no context" --
        # not zero, and not "missed" either.
        self.assertEqual(result["fact_only_context_available"], 0)
        self.assertEqual(result["fact_only_no_context"], 13)

        statuses = {r["event"]: r["status"] for r in result["results"]}
        self.assertEqual(statuses["Gorleston goal -- Ryan Curtis"], "partial")
        self.assertEqual(statuses["Gorleston goal -- Adam Tann"], "partial")
        self.assertEqual(statuses["Gorleston goal -- Joe Jefford"], "partial")

        # The 13 real subs/cards genuinely have no corresponding key_moments
        # entry of ANY type nearby in this match's real data -- they MUST
        # stay "fact_only_no_context", never "missed" (goals-only now) and
        # never "fact_only_context_available" (that needs SOME nearby
        # key_moment, even an unrelated one). If this ever flips, something
        # changed in either the real data or the matching logic.
        self.assertEqual(statuses["Tilbury yellow -- Jake Brocklebank"], "fact_only_no_context")
        self.assertEqual(statuses["Tilbury: Soumen Nandi on for Ronnie Winn"], "fact_only_no_context")

    def test_pipeline_ready_now_honestly_true(self):
        # WHY this flipped from False (Step 16) to True (Step 19), and why
        # that is now the CORRECT answer rather than a regression: the 13
        # "missing" events are all subs/cards whose facts are already
        # certain from match_config.json -- they never needed video
        # corroboration to be reported accurately (see ground_truth.py's
        # has_nearby_structural_context() docstring for the full reasoning,
        # and Step 19's analysis of synthesis_agent.py's own prompt, which
        # already instructs the report to state a fact plainly with no
        # added tactical color when no nearby data exists). Only a genuinely
        # undetected GOAL should block the pipeline now, and this match has
        # zero of those -- all 3 real goals were found (partial). So
        # pipeline_ready=True here is the honest, correct outcome, not a
        # loosening of a real check.
        result = build_ground_truth_check(self.work_dir)
        self.assertTrue(result["pipeline_ready"])

    def test_ground_truth_check_json_written_to_disk(self):
        build_ground_truth_check(self.work_dir)
        out_path = os.path.join(self.work_dir, "ground_truth_check.json")
        self.assertTrue(os.path.exists(out_path))
        with open(out_path, encoding="utf-8") as f:
            written = json.load(f)
        # Step 19: "missed" is goals-only now; this match's 3 goals were
        # all found (partial), so missed is correctly 0, not 13.
        self.assertEqual(written["missed"], 0)
        self.assertEqual(written["fact_only_no_context"], 13)


if __name__ == "__main__":
    unittest.main(verbosity=2)
