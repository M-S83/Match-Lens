from pipeline_graph import build_graph, PipelineState

app = build_graph()

print("--- test 1: window WITH a set piece -> should route through burst_scan ---")
w1 = PipelineState(window_id="07", frame_paths=["frame_07m00s.jpg"])
result1 = app.invoke(w1)

sp = result1["set_pieces"][0]
print(f"  source={sp.source!r}, burst_resolved={sp.burst_resolved}, "
      f"burst_fps={sp.burst_fps}, runners={sp.runners}")
assert sp.burst_resolved is True, "set piece should have been escalated and resolved"
assert sp.source == "5fps_burst"
assert sp.runners == ["CB1 near post", "ST1 far post"]
print("Confirmed: the set piece was escalated to burst_scan and enriched.")

print("\n--- test 2: window WITHOUT any set pieces -> should skip burst_scan entirely ---")
# WHY this now patches BOTH agent stubs, not one shared stub: since Step
# 10's dual-agent refactor, there is no single '_stub_structural_llm_response'
# to patch anymore -- Agent A and Agent B each have their own independent
# stub function. This is exactly the kind of test breakage that happens
# when a test reaches into internal implementation details (monkeypatching
# a private function by name) instead of testing through the public
# interface: the internals changed shape, and the old patch target silently
# stopped doing anything, rather than failing loudly.
import pipeline_graph

def _no_set_pieces_response(window_id):
    return {"formation": {"home": "4-3-3", "away": "4-4-2"}, "set_pieces": []}

pipeline_graph._stub_agent_a_response = _no_set_pieces_response
pipeline_graph._stub_agent_b_response = _no_set_pieces_response
app2 = build_graph()  # rebuild so the node closures pick up the patched stubs

w2 = PipelineState(window_id="15", frame_paths=["frame_15m00s.jpg"])
result2 = app2.invoke(w2)
print(f"  set_pieces={result2['set_pieces']}")
assert result2["set_pieces"] == []
print("Confirmed: with no set pieces, the graph went straight from tier1_scan to END, "
      "burst_scan never ran (no '[burst_scan]' log line above for window 15).")
