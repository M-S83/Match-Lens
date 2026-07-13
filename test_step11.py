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
    _write(d, "match_boundaries.json", {
        "boundaries": {"kickoff": {"confidence": 0.95}, "half_time": {"confidence": 0.90}},
    })
    _write(d, "match_config.json", {
        "verified": True, "enrichment_level": "identity_only",
        "player_id_ceiling": "tentative", "match": "Test FC vs Test United",
    })
    _write(d, "window_plan.json", {"total_windows": 2})
    _write(d, "running_summary.json", {"windows_complete": 2, "data_gap_windows": []})
    _write(d, "ground_truth_check.json", {"missed": 0, "total": 5})
    _write(d, "rerun_queue.json", {"rerun_queue": []})
    _write(d, "confirmation_queue.json", {"total": 0, "skipped": 0})
    _write(d, "source_profile.json", {
        "source_type": "tactical_wide_static", "classification_confidence": 0.9, "split_aware": False,
    })
    _write(d, "result_family_gates.json", {"gates": {}})
    _write(d, "deep_skill_metrics.json", {
        "total_metrics": 0, "active_metrics": 0, "suppressed_metrics": [], "avg_confidence": 0.0,
    })
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
print("Confirmed: ready match routed straight to synthesize (see '[synthesize]' log line above, "
      "no '[STUB EMAIL]' lines).")

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
