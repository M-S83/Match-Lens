# WHY asyncio.run(run_match_pipeline(...)) instead of calling it
# directly: run_match_pipeline is `async def` now, purely because
# run_window is, purely because the window-level graph it drives
# contains async nodes (agent_a_scan/agent_b_scan). asyncio.run() is
# how you enter the async world from ordinary top-level script code --
# it starts an event loop, runs the coroutine to completion, and
# returns the plain result. Note build_match_graph()'s app.invoke(...)
# calls below stay UNCHANGED and synchronous -- readiness_graph.py's
# own nodes were never touched, so that graph has no async nodes and
# no reason to switch entry points.
import asyncio
import json
import os
import tempfile

import pipeline_graph
from match_pipeline import WindowSpec, run_match_pipeline
from readiness_graph import build_match_graph, MatchState


def _write(match_dir, fname, data):
    with open(os.path.join(match_dir, fname), "w", encoding="utf-8") as f:
        json.dump(data, f)


def _make_match_dir_missing_window_files() -> str:
    """Every match-level file build_readiness_check()/run_synthesis()
    need EXCEPT window_plan.json / running_summary.json / source_profile.json
    -- those three are exactly what run_match_pipeline() produces, so we
    deliberately don't pre-write them here. This is the same fixture
    shape as test_step11's _make_ready_match_dir(), minus the three
    files this step's aggregator now owns."""
    d = tempfile.mkdtemp(prefix="matchlens_agg_")
    _write(d, "match_boundaries.json", {
        "boundaries": {"kickoff": {"confidence": 0.95}, "half_time": {"confidence": 0.90}},
    })
    _write(d, "match_config.json", {
        "verified": True, "enrichment_level": "identity_only",
        "player_id_ceiling": "tentative", "match": "Test FC vs Test United",
    })
    _write(d, "ground_truth_check.json", {"missed": 0, "total": 5})
    _write(d, "rerun_queue.json", {"rerun_queue": []})
    _write(d, "confirmation_queue.json", {"total": 0, "skipped": 0})
    _write(d, "result_family_gates.json", {"gates": {}})
    _write(d, "deep_skill_metrics.json", {
        "total_metrics": 0, "active_metrics": 0, "suppressed_metrics": [], "avg_confidence": 0.0,
    })
    _write(d, "pass_sequences.json", {"sequences": []})
    return d


def _make_three_window_specs():
    return [
        WindowSpec(window_id="01", start_s=0.0,   end_s=60.0,  half="1H", frame_paths=["01_a.jpg"]),
        WindowSpec(window_id="02", start_s=60.0,  end_s=120.0, half="1H", frame_paths=["02_a.jpg"]),
        WindowSpec(window_id="03", start_s=120.0, end_s=180.0, half="1H", frame_paths=["03_a.jpg"]),
    ]


print("--- test 1: all 3 windows succeed -> aggregation should report 3/3, then the full match graph should run all the way to synthesis ---")
match_dir = _make_match_dir_missing_window_files()
specs = _make_three_window_specs()

window_results = asyncio.run(run_match_pipeline(match_dir, specs))
assert len(window_results) == 3, f"expected 3 successful windows, got {len(window_results)}"

with open(os.path.join(match_dir, "window_plan.json"), encoding="utf-8") as f:
    window_plan = json.load(f)
with open(os.path.join(match_dir, "running_summary.json"), encoding="utf-8") as f:
    running_summary = json.load(f)
with open(os.path.join(match_dir, "source_profile.json"), encoding="utf-8") as f:
    source_profile_doc = json.load(f)

print(f"  window_plan.json:      {window_plan}")
print(f"  running_summary.json:  {running_summary}")
print(f"  source_profile.json:   {source_profile_doc}")

assert window_plan["total_windows"] == 3
assert running_summary["windows_complete"] == 3
assert running_summary["data_gap_windows"] == []
assert source_profile_doc["classification_confidence"] == 0.91
print("Confirmed: window_plan/running_summary/source_profile were derived correctly from 3 real window-graph runs.")

# Now run the actual match-level graph (readiness gate -> synthesis) on
# top of what the aggregator just wrote, alongside the other match-wide
# files the fixture pre-supplied. This is the full three-tier chain:
# window graph -> aggregation -> match graph -> synthesis.
app = build_match_graph()
match_result = app.invoke(MatchState(match_dir=match_dir))
print(f"  report_ready={match_result['report_ready']}, blocking_issues={match_result['blocking_issues']}")
assert match_result["report_ready"] is True, match_result["blocking_issues"]
assert match_result["synthesis_result"]["status"] == "complete"
print("Confirmed: the aggregated match-level files were sufficient for the readiness gate to pass, "
      "and synthesis ran end to end on top of them.")


print("\n--- test 2: one window fails mid-run -> should NOT crash the whole match, and readiness should correctly block ---")

# Monkeypatch the window-level agent stub to fail for window "02" only,
# simulating a real failure inside one window's processing (e.g. a bad
# stub response, a validation error). This is the actual thing this
# step's fan-in has to handle that a single-window test never faced.
original_stub_a = pipeline_graph._stub_agent_a_response

# WHY this replacement is `async def` and does `await original_stub_a(...)`:
# same reasoning as test_step9's patched stub -- agent_a_scan_node does
# `raw = await _stub_agent_a_response(...)`, so whatever function sits
# in that slot has to be awaitable. Calling the real original_stub_a
# (itself now async) requires awaiting it too, not just calling it.
async def _flaky_stub_agent_a_response(window_id: str) -> dict:
    if window_id == "02":
        raise RuntimeError("simulated agent failure for window 02")
    return await original_stub_a(window_id)

pipeline_graph._stub_agent_a_response = _flaky_stub_agent_a_response
try:
    match_dir_2 = _make_match_dir_missing_window_files()
    specs_2 = _make_three_window_specs()
    window_results_2 = asyncio.run(run_match_pipeline(match_dir_2, specs_2))
finally:
    # Same finally-block discipline as synthesize_node's monkeypatch --
    # restore the real stub whether or not run_match_pipeline raised.
    pipeline_graph._stub_agent_a_response = original_stub_a

assert len(window_results_2) == 2, f"expected 2 successful windows (window 02 should have failed), got {len(window_results_2)}"

with open(os.path.join(match_dir_2, "running_summary.json"), encoding="utf-8") as f:
    running_summary_2 = json.load(f)
print(f"  running_summary.json (after 1 failure): {running_summary_2}")
assert running_summary_2["windows_complete"] == 2
assert running_summary_2["data_gap_windows"] == ["02"]
print("Confirmed: window 02's failure did not crash the run -- windows 01 and 03 still completed, "
      "and window 02 is correctly recorded as a data-gap window.")

match_result_2 = app.invoke(MatchState(match_dir=match_dir_2))
print(f"  report_ready={match_result_2['report_ready']}")
print(f"  blocking_issues={match_result_2['blocking_issues']}")
assert match_result_2["report_ready"] is False
assert any("incomplete" in issue.lower() or "data gap" in issue.lower()
           for issue in match_result_2["blocking_issues"])
print("Confirmed: the readiness gate correctly blocked the report because of the incomplete/data-gap "
      "window, and (per the earlier email-alert step) an owner alert would have been sent for it.")
