# WHY this test now uses asyncio.run(app.ainvoke(...)) instead of
# app.invoke(...): once ANY node in the graph is `async def` (agent_a_scan
# and agent_b_scan now are, for real concurrency -- see test_step13.py),
# LangGraph refuses to run the graph via the synchronous .invoke() at all
# (it raises "No synchronous function provided"). Every caller of this
# graph has to switch to the async entry point, even tests like this one
# that don't care about concurrency themselves -- that propagation is a
# real, well-known property of asyncio: introducing async anywhere in a
# call chain forces async all the way up to wherever that chain starts.
import asyncio
from pipeline_graph import build_graph, PipelineState

print("--- test 1: run the compiled graph with a starting state ---")
app = build_graph()

starting_state = PipelineState(
    window_id="07",
    frame_paths=["frame_00m00s.jpg", "frame_01m00s.jpg", "frame_02m00s.jpg"],
)
print("Before running the graph:")
print(starting_state.model_dump_json(indent=2))

result = asyncio.run(app.ainvoke(starting_state))

print("\nAfter running the graph (source_profile was filled in by the node):")
import json
print(json.dumps(result, indent=2, default=str))

print("\n--- test 2: confirm the result validates back into our WindowResult-compatible shape ---")
from models import SourceProfile
sp = result["source_profile"]
if not isinstance(sp, SourceProfile):
    sp = SourceProfile.model_validate(sp)
print(f"source_type={sp.source_type}, confidence={sp.classification_confidence}")
assert sp.source_type == "tactical_wide_static"
assert result["window_id"] == "07"
print("Graph correctly carried window_id through untouched, and added source_profile.")
