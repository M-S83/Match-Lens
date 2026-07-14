"""
Step 29: shots_for/shots_against are always empty [] -- but not for the
reason we might first assume.

Tracing the real root cause: shots_for/shots_against are populated in
accumulate_window() from each merged window's shot_attempts field. Checking
the real match's merged window files directly confirms shot_attempts is
never populated by the extraction agent, on any window, for any match seen
so far -- a genuine upstream extraction gap, already documented (but not
fixed) in calc_attacking_support_score's own comment in deep_skill_metrics.py.
That part genuinely isn't fixable without new upstream extraction work, and
this step does NOT pretend otherwise.

But there IS real, 100%-confirmed shot data already sitting on disk that
never gets used: shots_log.json, built by build_shots_log() (phase 3f_shots)
straight from match_config.json's confirmed scorer/minute facts. That data
just never gets copied into running_summary.json's shots_for/shots_against,
so it's invisible to every shot-based metric even though we're completely
sure it happened.

Step 29 closes exactly that gap (and documents, rather than fixes, the
larger one): merge_confirmed_goals_into_shots() (accumulator.py) merges
shots_log.json's goals into shots_for/shots_against, honestly tagged as
match-facts-only where that's all we have. Also fixed one silently-
confident consumer: build_deep_skill_metrics_v2.py's shots_from_defending_third
used to report a confident "0" that was indistinguishable from "we checked
and there genuinely were zero" -- now reports None when no shot has any
known zone info at all.

Five test classes:
  TestUpstreamGapConfirmation   -- proves shot_attempts really is
                                    structurally dead (not fixed here).
  TestMergeConfirmedGoals       -- proves the new merge function is
                                    correct: classification, idempotency,
                                    honesty about unknown fields.
  TestRealMatchIntegration      -- runs the merge against a copy of the
                                    real match and checks the real numbers.
  TestChanceCreationProfileMergeDoesNotRegress -- the merge itself created
                                    a second bug that had to be fixed in the
                                    same step: deep_skill_metrics.py's real,
                                    LIVE calc_chance_creation_profile() used
                                    to treat "any shots_for entries at all"
                                    as a reason to fully discard its 70-real-
                                    entry pass-sequence fallback and use ONLY
                                    the (now 3-entry) shots_for list. Fixed
                                    to union both sources, deduping by
                                    timestamp so a goal isn't double-counted.
                                    This class is the regression guard.
  TestShotsFromDefendingThirdHonesty -- separate, smaller finding: while
                                    investigating, the session's earlier
                                    diagnosis of "the real unguarded bug" in
                                    build_deep_skill_metrics_v2.py turned out
                                    to be in a LEGACY REFERENCE file that is
                                    NOT wired into the live pipeline at all
                                    (confirmed by grep: nothing imports
                                    build_deep_skill_metrics_v2.build_deep_
                                    skill_metrics; individual functions get
                                    "ported verbatim" from it into the real
                                    deep_skill_metrics.py instead, per that
                                    file's own comments). The live version of
                                    this metric (calc_build_up_effectiveness)
                                    doesn't use shots_for at all, so it never
                                    had this bug. The fix there is kept as a
                                    harmless honesty improvement to the
                                    reference file (in case it's ever ported
                                    forward), but does NOT affect any real
                                    match's actual output -- this class
                                    documents that distinction rather than
                                    claiming a live fix.

NOTE -- separately discovered, NOT fixed here (kept in scope for a future
step, listed in the glossary as a new deferred finding): calc_chance_
creation_profile's fallback branch reads sequence fields "zone_start"/
"zone_end", but real pass_sequences.json records use "start_zone"/
"end_zone" -- the same wrong-key-name pattern Step 22 already fixed in
calc_build_up_effectiveness's final_third calculation, just not yet fixed
here. This means origin_zone/route for the 70 fallback chances is silently
"unknown" on every real match today, independent of anything this step
changes. Flagged, not fixed, to keep this step's diff reviewable.
"""
import json
import os
import shutil
import tempfile
import unittest

from accumulator import merge_confirmed_goals_into_shots, build_shots_log
from deep_skill_metrics import calc_chance_creation_profile
from build_deep_skill_metrics_v2 import _metric_build_up_effectiveness

REAL_MATCH_DIR = os.path.join(
    os.path.dirname(__file__), "matches", "2026-04-11_gorleston_vs_tilbury"
)


class TestUpstreamGapConfirmation(unittest.TestCase):
    """The non-goal shot_attempts gap is real and NOT fixed by this step."""

    def test_shot_attempts_never_populated_on_real_match(self):
        agent_logs_dir = os.path.join(REAL_MATCH_DIR, "agent_logs")
        merged_files = [f for f in os.listdir(agent_logs_dir) if f.endswith("_merged.json")]
        checked = 0
        for fname in merged_files:
            d = json.load(open(os.path.join(agent_logs_dir, fname)))
            if "shot_attempts" in d:
                checked += 1
                self.assertEqual(d["shot_attempts"], [])
        self.assertTrue(checked > 0, "expected some merged files to have the shot_attempts key")


class TestMergeConfirmedGoals(unittest.TestCase):
    """The new merge function: correctness, idempotency, honesty."""

    def _make_match(self, tmp_dir, goals, home_team="Home FC", away_team="Away FC",
                     existing_shots_for=None, existing_shots_against=None):
        shots_log = {"home_team": home_team, "away_team": away_team,
                     "total_goals": len(goals), "goals": goals}
        with open(os.path.join(tmp_dir, "shots_log.json"), "w") as f:
            json.dump(shots_log, f)
        summary = {"home_team": home_team, "away_team": away_team,
                   "shots_for": existing_shots_for or [],
                   "shots_against": existing_shots_against or []}
        with open(os.path.join(tmp_dir, "running_summary.json"), "w") as f:
            json.dump(summary, f)

    def test_home_goal_classified_as_shots_for(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            goals = [{"minute": 23, "team_name": "Home FC", "team_side": "home",
                      "player": "A. Striker", "evidence_grade": "C", "set_piece": False}]
            self._make_match(tmp_dir, goals)
            result = merge_confirmed_goals_into_shots(tmp_dir)
            self.assertEqual(result["goals_added"], 1)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            self.assertEqual(len(summary["shots_for"]), 1)
            self.assertEqual(len(summary["shots_against"]), 0)
            shot = summary["shots_for"][0]
            self.assertEqual(shot["outcome"], "goal")
            self.assertEqual(shot["minute"], 23)
            self.assertEqual(shot["source"], "shots_log_goal")
        finally:
            shutil.rmtree(tmp_dir)

    def test_away_goal_classified_as_shots_against(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            goals = [{"minute": 67, "team_name": "Away FC", "team_side": "away",
                      "player": "B. Poacher", "evidence_grade": "C", "set_piece": False}]
            self._make_match(tmp_dir, goals)
            merge_confirmed_goals_into_shots(tmp_dir)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            self.assertEqual(len(summary["shots_for"]), 0)
            self.assertEqual(len(summary["shots_against"]), 1)
        finally:
            shutil.rmtree(tmp_dir)

    def test_unknown_detail_fields_stay_none_not_guessed(self):
        # Match-facts-only goals genuinely don't carry origin/target/sequence
        # detail -- the merge must not invent plausible-looking values for
        # fields it doesn't actually know.
        tmp_dir = tempfile.mkdtemp()
        try:
            goals = [{"minute": 10, "team_name": "Home FC", "team_side": "home",
                      "player": "A. Striker", "evidence_grade": "C", "set_piece": False}]
            self._make_match(tmp_dir, goals)
            merge_confirmed_goals_into_shots(tmp_dir)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            shot = summary["shots_for"][0]
            self.assertIsNone(shot["origin_row"])
            self.assertIsNone(shot["origin_column"])
            self.assertIsNone(shot["shot_type"])
            self.assertIsNone(shot["sequence_start_zone"])
        finally:
            shutil.rmtree(tmp_dir)

    def test_rerun_is_idempotent_no_duplicates(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            goals = [{"minute": 15, "team_name": "Home FC", "team_side": "home",
                      "player": "A. Striker", "evidence_grade": "C", "set_piece": False}]
            self._make_match(tmp_dir, goals)
            first = merge_confirmed_goals_into_shots(tmp_dir)
            second = merge_confirmed_goals_into_shots(tmp_dir)
            self.assertEqual(first["goals_added"], 1)
            self.assertEqual(second["goals_added"], 0)
            self.assertEqual(second["goals_already_present"], 1)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            self.assertEqual(len(summary["shots_for"]), 1)  # not duplicated
        finally:
            shutil.rmtree(tmp_dir)

    def test_missing_shots_log_is_a_graceful_noop_not_a_crash(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            summary = {"home_team": "Home FC", "shots_for": [], "shots_against": []}
            with open(os.path.join(tmp_dir, "running_summary.json"), "w") as f:
                json.dump(summary, f)
            result = merge_confirmed_goals_into_shots(tmp_dir)
            self.assertEqual(result["goals_added"], 0)
            self.assertIn("skipped_reason", result)
        finally:
            shutil.rmtree(tmp_dir)


class TestRealMatchIntegration(unittest.TestCase):
    """Run against a throwaway copy of the real match."""

    def test_real_match_three_goals_merge_correctly(self):
        # NOTE: the real match directory now has the Step 29 fix already
        # applied (it was run for real, not just in a test copy -- same
        # discipline as Steps 27/28's "stays fixed" regression guards). So
        # this copy of REAL_MATCH_DIR may already contain the 3 goals in
        # shots_for before this test's own merge call runs. Assert on the
        # END STATE (3 goals present, correctly classified) via
        # goals_added + goals_already_present, rather than assuming a
        # specific "added" count that depends on whether this is the first
        # time the merge has ever run against this data.
        tmp_dir = tempfile.mkdtemp()
        try:
            shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
            # Regenerate shots_log.json fresh, matching production ordering
            # (3f_shots runs before the merge in both the live pipeline and
            # the regeneration script).
            build_shots_log(tmp_dir)
            result = merge_confirmed_goals_into_shots(tmp_dir)
            self.assertEqual(result["goals_in_log"], 3)
            self.assertEqual(result["goals_added"] + result["goals_already_present"], 3)

            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            # All 3 real goals are Gorleston (home/focus team) per shots_log.json.
            self.assertEqual(len(summary["shots_for"]), 3)
            self.assertEqual(len(summary["shots_against"]), 0)
            minutes = sorted(s["minute"] for s in summary["shots_for"])
            self.assertEqual(minutes, [6, 45, 59])

            # Re-running again must stay idempotent regardless of starting state.
            result2 = merge_confirmed_goals_into_shots(tmp_dir)
            self.assertEqual(result2["goals_added"], 0)
            self.assertEqual(result2["goals_already_present"], 3)
            summary2 = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            self.assertEqual(len(summary2["shots_for"]), 3)  # still not duplicated
        finally:
            shutil.rmtree(tmp_dir)


class TestChanceCreationProfileMergeDoesNotRegress(unittest.TestCase):
    """deep_skill_metrics.py's REAL calc_chance_creation_profile must union
    shots_for goals with the threat_sequences fallback, not override it."""

    def test_fallback_alone_when_shots_for_empty(self):
        # Matches pre-Step-29 behaviour exactly when there are no goals yet.
        summary = {"shots_for": []}
        passes = {"sequences": [
            {"outcome": "shot",  "start_frame": "10m00s", "zone_start": "middle_third",
             "zone_end": "attacking_third", "length": 4},
            {"outcome": "cross", "start_frame": "40m00s", "zone_start": "middle_third",
             "zone_end": "attacking_third", "length": 3},
        ]}
        value, count, tiers = calc_chance_creation_profile(summary, passes)
        self.assertEqual(value["total_chances"], 2)
        self.assertEqual(value["data_source"]["shots_for_count"], 0)
        self.assertEqual(value["data_source"]["threat_sequences_fallback_count"], 2)
        self.assertEqual(tiers, ["repeated_pattern"])

    def test_goal_chance_adds_to_fallback_not_replaces_it(self):
        # This is the regression: adding ONE goal to shots_for must not
        # cause the OTHER unrelated threat_sequences to disappear.
        summary = {"shots_for": [
            {"timestamp": "10m00s", "outcome": "goal", "minute": 10,
             "origin_column": None, "shot_type": None, "target_zone": None,
             "sequence_length": None},
        ]}
        passes = {"sequences": [
            {"outcome": "shot",  "start_frame": "10m00s", "zone_start": "middle_third",
             "zone_end": "attacking_third", "length": 4},   # this IS the goal -- must be deduped
            {"outcome": "shot",  "start_frame": "25m00s", "zone_start": "middle_third",
             "zone_end": "attacking_third", "length": 5},   # unrelated shot -- must survive
            {"outcome": "cross", "start_frame": "70m00s", "zone_start": "middle_third",
             "zone_end": "attacking_third", "length": 2},   # unrelated cross -- must survive
        ]}
        value, count, tiers = calc_chance_creation_profile(summary, passes)
        # 1 goal chance + 2 surviving unrelated chances = 3, NOT 1.
        self.assertEqual(value["total_chances"], 3)
        self.assertEqual(value["data_source"]["shots_for_count"], 1)
        self.assertEqual(value["data_source"]["threat_sequences_fallback_count"], 2)
        self.assertIn("direct", tiers)
        self.assertIn("repeated_pattern", tiers)

    def test_real_match_goals_added_without_losing_fallback_chances(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
            passes = json.load(open(os.path.join(tmp_dir, "pass_sequences.json")))

            summary_before = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            summary_before["shots_for"] = []
            value_before, _, tiers_before = calc_chance_creation_profile(summary_before, passes)
            before_total = value_before["total_chances"]

            build_shots_log(tmp_dir)
            merge_confirmed_goals_into_shots(tmp_dir)
            summary_after = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            value_after, _, tiers_after = calc_chance_creation_profile(summary_after, passes)

            # Must have GAINED chances (the 3 goals), never lost the fallback ones.
            self.assertGreater(value_after["total_chances"], before_total)
            self.assertEqual(value_after["data_source"]["threat_sequences_fallback_count"],
                              value_before["data_source"]["threat_sequences_fallback_count"])
            self.assertEqual(value_after["data_source"]["shots_for_count"], 3)
        finally:
            shutil.rmtree(tmp_dir)


class TestShotsFromDefendingThirdHonesty(unittest.TestCase):
    """The silently-confident-zero is now an honest None when warranted."""

    def _base_summary(self, shots_for=None):
        # total_known must be >= 3 or _metric_build_up_effectiveness returns
        # an _unavailable() stub instead of a real value dict -- keep enough
        # known-progression sequences that the metric actually computes.
        return {
            "shots_for": shots_for or [],
            "vertical_progression_totals": {
                "defending_to_middle": 5,
                "middle_to_attacking": 3,
                "defending_to_attacking": 0,
                "regression_middle_to_defending": 1,
                "regression_attacking_to_middle": 0,
                "regression_attacking_to_defending": 0,
                "same_third": 1,
            },
        }

    def test_no_shots_with_known_zone_reports_none_not_zero(self):
        summary = self._base_summary(shots_for=[
            {"outcome": "goal", "sequence_start_zone": None},  # goal-log entry, zone unknown
        ])
        result = _metric_build_up_effectiveness(summary, "vision_1fps")
        self.assertIsNone(result["value"]["shots_from_defending_third"])

    def test_shots_with_known_zone_reports_a_real_count(self):
        summary = self._base_summary(shots_for=[
            {"outcome": "shot", "sequence_start_zone": "defending_third"},
            {"outcome": "shot", "sequence_start_zone": "middle_third"},
        ])
        result = _metric_build_up_effectiveness(summary, "vision_1fps")
        self.assertEqual(result["value"]["shots_from_defending_third"], 1)

    def test_empty_shots_for_reports_none_not_zero(self):
        summary = self._base_summary(shots_for=[])
        result = _metric_build_up_effectiveness(summary, "vision_1fps")
        self.assertIsNone(result["value"]["shots_from_defending_third"])


if __name__ == "__main__":
    unittest.main()
