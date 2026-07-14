"""
test_step30.py -- Step 30: dead-code / lost-value audit.

Two things happened in this step, and they are DIFFERENT in kind, which is
why they get different treatment:

1. home_formation_evidence / away_formation_evidence (accumulator.py,
   formation_history[] entries) -- a REAL "computed but thrown away" bug,
   same family as Step 27's set-piece write-back gap and Step 29's shots
   gap. The 3a structural prompt REQUIRES the model to state the visible
   evidence for its formation call (an anti-hallucination discipline: no
   confirmatory structure visible -> the model must say so explicitly
   rather than guess). That text is real, paid-for, and useful for QA/
   report review -- but accumulator.py's formation_history.append() never
   carried it from the merged window file into running_summary.json. FIXED
   in this step: the two fields are now written through, honestly left
   None when a window's merged file genuinely doesn't have them (e.g. a
   legacy-schema window from before the evidence requirement existed).

2. frame_preprocessor.py's preprocess_window() -- investigated, confirmed
   genuinely unreferenced by any real pipeline code (only self-referencing
   in its own docstring and its own __main__ block), but NOT removed and
   NOT wired in during this step. Unlike (1), turning it on changes what
   frames the vision agents actually see on every future match -- a real
   product/quality decision, not a bug fix, and deleting ~450 lines of
   working kit-colour-detection logic is not something to do unilaterally
   either. This file documents the investigation as a permanent record;
   the activate-or-remove decision is intentionally left to the user.

Two test classes:
  TestFormationEvidenceWriteback  -- proves the real gap, proves the fix,
                                      proves honesty when evidence is absent.
  TestFramePreprocessorDeadCodeConfirmed -- proves (not just asserts) that
                                      preprocess_window is unreferenced by
                                      any live pipeline module, so this
                                      finding doesn't silently go stale if
                                      someone wires it in later without
                                      updating this file.
"""
import ast
import json
import os
import shutil
import tempfile
import unittest

from accumulator import accumulate_all_windows

REAL_MATCH_DIR = os.path.join(
    os.path.dirname(__file__), "matches", "2026-04-11_gorleston_vs_tilbury"
)
REPO_ROOT = os.path.dirname(__file__)

LIVE_PIPELINE_FILES = [
    "pipeline_runner_v2.py",
    "regenerate_match_data.py",
    "accumulator.py",
    "deep_skill_metrics.py",
    "synthesis_agent.py",
    "report_filter.py",
]


class TestFormationEvidenceWriteback(unittest.TestCase):
    """accumulator.py must carry home/away_formation_evidence into
    formation_history[], not silently drop it."""

    def test_upstream_evidence_is_real_in_merged_files(self):
        # Confirm the raw material this fix depends on actually exists on
        # the real match -- if this ever stops being true (schema change),
        # this test fails loudly instead of the fix silently doing nothing.
        merged_dir = os.path.join(REAL_MATCH_DIR, "agent_logs")
        sample = os.path.join(merged_dir, "agent_01_1H_00-00_05-00_merged.json")
        d = json.load(open(sample))
        formation = d.get("structural_context", {}).get("formation") or d.get("formation", {})
        self.assertTrue(formation.get("home_formation_evidence"))
        self.assertTrue(formation.get("away_formation_evidence"))

    def test_regenerated_real_match_carries_evidence_through(self):
        tmp_dir = tempfile.mkdtemp()
        try:
            shutil.copytree(REAL_MATCH_DIR, tmp_dir, dirs_exist_ok=True)
            accumulate_all_windows(tmp_dir)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            fh = summary["formation_history"]
            self.assertEqual(len(fh), 21)
            with_evidence = [f for f in fh if f.get("home_formation_evidence")]
            # Every real window on this match has the evidence field upstream,
            # so every entry in formation_history must now carry it.
            self.assertEqual(len(with_evidence), 21)
            self.assertIn("visible", fh[0]["home_formation_evidence"])
        finally:
            shutil.rmtree(tmp_dir)

    def _run_single_synthetic_window(self, formation_extra):
        """Helper: write one synthetic merged window file to a throwaway
        match_dir and run the real accumulate_all_windows() against it,
        exactly like the live pipeline would."""
        tmp_dir = tempfile.mkdtemp()
        try:
            logs_dir = os.path.join(tmp_dir, "agent_logs")
            os.makedirs(logs_dir)
            window = {
                "window": "1H_00-00_05-00",
                "agent_id": "01",
                "formation": {
                    "home": "4-4-2", "away": "4-3-3",
                    "home_shape": "mid", "away_shape": "high",
                    **formation_extra,
                },
            }
            with open(os.path.join(logs_dir, "agent_01_1H_00-00_05-00_merged.json"), "w") as f:
                json.dump(window, f)
            accumulate_all_windows(tmp_dir)
            summary = json.load(open(os.path.join(tmp_dir, "running_summary.json")))
            return summary["formation_history"][0]
        finally:
            shutil.rmtree(tmp_dir)

    def test_missing_evidence_reports_none_not_empty_string_or_omitted(self):
        # A legacy-schema window with no evidence field at all must not
        # crash, and must not fabricate an empty string that looks like
        # "the model said nothing" when really "the field doesn't exist
        # in this window's schema at all."
        entry = self._run_single_synthetic_window({})
        self.assertIn("home_formation_evidence", entry)
        self.assertIsNone(entry["home_formation_evidence"])
        self.assertIsNone(entry["away_formation_evidence"])

    def test_present_evidence_is_carried_through_verbatim(self):
        entry = self._run_single_synthetic_window({
            "home_formation_evidence": "Back four visible with two banks of four.",
            "away_formation_evidence": "No confirmatory structure visible; defaulting to lineup.",
        })
        self.assertEqual(entry["home_formation_evidence"],
                          "Back four visible with two banks of four.")
        self.assertEqual(entry["away_formation_evidence"],
                          "No confirmatory structure visible; defaulting to lineup.")


class TestFramePreprocessorDeadCodeConfirmed(unittest.TestCase):
    """Confirms frame_preprocessor.preprocess_window() is genuinely
    unreferenced by any live pipeline file -- documents the finding as a
    regression-detecting test rather than a one-off manual grep, so if
    someone wires it in later without updating this file, this test starts
    failing and forces a conscious decision rather than a silent drift."""

    def test_preprocess_window_not_imported_by_any_live_pipeline_file(self):
        for fname in LIVE_PIPELINE_FILES:
            path = os.path.join(REPO_ROOT, fname)
            if not os.path.exists(path):
                continue
            source = open(path).read()
            tree = ast.parse(source, filename=fname)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "frame_preprocessor":
                    self.fail(
                        f"{fname} now imports frame_preprocessor -- "
                        f"the dead-code finding from Step 30 is stale. "
                        f"Update this test and the glossary to reflect the "
                        f"activation decision."
                    )
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name == "frame_preprocessor":
                            self.fail(
                                f"{fname} now imports frame_preprocessor -- "
                                f"the dead-code finding from Step 30 is stale."
                            )

    def test_kit_config_source_data_already_exists_for_future_activation(self):
        # Not a bug, just documentation-by-test: if this step's finding
        # that "activation is closer than it looks" (match_config.json
        # already stores kits) stops being true, this test catches it.
        mc = json.load(open(os.path.join(REAL_MATCH_DIR, "match_config.json")))
        self.assertIn("kits", mc)
        self.assertIn("home", mc["kits"])
        self.assertIn("away", mc["kits"])


if __name__ == "__main__":
    unittest.main()
