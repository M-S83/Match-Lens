from pipeline_graph import build_graph, PipelineState

print("--- test 1: run the compiled graph with a starting state ---")
app = build_graph()

starting_state = PipelineState(
    window_id="07",
    frame_paths=["frame_00m00s.jpg", "frame_01m00s.jpg", "frame_02m00s.jpg"],
)
print("Before running the graph:")
print(starting_state.model_dump_json(indent=2))

result = app.invoke(starting_state)

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
