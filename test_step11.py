import json
import os
import tempfile

from readiness_graph import build_match_graph, MatchState


def _write(match_dir, fname, data):
    with open(os.path.join(match_dir, fname), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_ready_match_dir() -> str:
    """A match_dir with every file build_readiness_check() wants to see,
    all passing -- should come out report_ready=True."""
    d = tempfile.mkdtemp(prefix="matchlens_ready_")
    # WHY these four specific keys (ko_1h/ht_whistle/ko_2h/ft_whistle), not
    # the simpler kickoff/half_time keys this fixture used before Step 18:
    # build_readiness_check()'s boundary-confidence check is generic (it
    # just averages whatever confidence values are present under any keys),
    # but ground_truth.py's minute-to-video-time conversion is NOT generic --
    # it reads these four specific keys with direct bracket access, because
    # that's the real schema real match_boundaries.json files use (see
    # tests/fixtures/gorleston_vs_tilbury_gt/match_boundaries.json). Now that
    # Step 18 wires ground_truth_node in front of readiness_gate_node in the
    # real graph, this fixture has to satisfy BOTH readers, not just one --
    # a mismatch here is exactly the kind of integration bug wiring two
    # previously-separate functions together for the first time exposes.
    _write(d, "match_boundaries.json", {
        "boundaries": {
            "ko_1h":      {"seconds": 0,    "confidence": 0.95},
            "ht_whistle": {"seconds": 2700, "confidence": 0.95},
            "ko_2h":      {"seconds": 2760, "confidence": 0.90},
            "ft_whistle": {"seconds": 5460, "confidence": 0.90},
        },
    })
    _write(d, "match_config.json", {
        "verified": True, "enrichment_level": "identity_only",
        "player_id_ceiling": "tentative", "match": "Test FC vs Test United",
    })
    _write(d, "window_plan.json", {"total_windows": 2})
    _write(d, "running_summary.json", {"windows_complete": 2, "data_gap_windows": []})
    # WHY ground_truth_check.json is NOT hand-written here any more (it was,
    # before Step 18): the graph now has a real ground_truth_node that runs
    # build_ground_truth_check() itself and writes this file for real before
    # readiness_gate_node ever reads it -- a hand-written stand-in here would
    # just get silently overwritten the moment the graph runs, which would
    # be misleading to read later. Since this fixture's match_config.json
    # (below) declares zero goals/substitutions/cards, the real check will
    # correctly compute events_checked=0, missed=0 -- same "passes" outcome
    # the hand-written stand-in used to assert, but now actually earned by
    # running the real function instead of assumed by fixture fiat.
    _write(d, "rerun_queue.json", {"rerun_queue": []})
    _write(d, "confirmation_queue.json", {"total": 0, "skipped": 0})
    _write(d, "source_profile.json", {
        "source_type": "tactical_wide_static", "classification_confidence": 0.9, "split_aware": False,
    })
    _write(d, "result_family_gates.json", {"gates": {}})
    _write(d, "deep_skill_metrics.json", {
        "total_metrics": 0, "active_metrics": 0, "suppressed_metrics": [], "avg_confidence": 0.0,
    })
    # required by synthesis_agent.build_input_bundle() (REQUIRED_FILES) --
    # doesn't need real pass data, just needs to exist, since Step 11's
    # synthesize_node calls the real run_synthesis() end to end.
    _write(d, "pass_sequences.json", {"sequences": []})
    return d


def _make_not_ready_match_dir() -> str:
    """Same as above but WITHOUT match_boundaries.json -- boundary_ok
    fails ('boundaries file missing'), so report_ready should be False."""
    d = _make_ready_match_dir()
    os.remove(os.path.join(d, "match_boundaries.json"))
    return d


app = build_match_graph()

print("--- test 1: a match that IS ready -> should route to synthesize, no email ---")
ready_dir = _make_ready_match_dir()
result_ready = app.invoke(MatchState(match_dir=ready_dir))
print(f"  report_ready={result_ready['report_ready']}, blocking_issues={result_ready['blocking_issues']}")
assert result_ready["report_ready"] is True
assert result_ready["blocking_issues"] == []

# The real run_synthesis() should have actually run (LLM call stubbed)
# and written three real .md files to disk in the temp match_dir.
synth = result_ready["synthesis_result"]
print(f"  synthesis_result={synth}")
assert synth["status"] == "complete"
for fname in ("tactical_report.md", "opposition_report_home.md", "opposition_report_away.md"):
    fpath = os.path.join(ready_dir, fname)
    assert os.path.exists(fpath), f"expected {fname} to be written by run_synthesis()"
    with open(fpath, encoding="utf-8") as f:
        content = f.read()
    assert "STUB" in content, f"{fname} should contain the stub marker"
    print(f"  {fname}: {len(content)} chars written, contains stub marker")

print("Confirmed: ready match routed to synthesize, which ran the REAL run_synthesis() "
      "(bundle loading, roster block, prompt building, file writing) with only the network "
      "call stubbed -- three real .md files landed on disk, no '[STUB EMAIL]' lines.")

print("\n--- test 2: a match that is NOT ready -> should route to insufficient_data + send an email ---")
not_ready_dir = _make_not_ready_match_dir()
result_not_ready = app.invoke(MatchState(match_dir=not_ready_dir))
print(f"  report_ready={result_not_ready['report_ready']}")
print(f"  blocking_issues={result_not_ready['blocking_issues']}")
assert result_not_ready["report_ready"] is False
assert any("boundar" in issue.lower() for issue in result_not_ready["blocking_issues"])
print("Confirmed: not-ready match routed to insufficient_data, and the missing-boundaries "
      "issue appears in blocking_issues (see the '[STUB EMAIL]' lines above -- that's what "
      "would land in your inbox).")
