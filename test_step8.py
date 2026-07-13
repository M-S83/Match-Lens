from pipeline_graph import build_graph, PipelineState

print("--- test 1: run the full 2-node graph (profile_source -> tier1_scan) ---")
app = build_graph()

starting_state = PipelineState(
    window_id="07",
    frame_paths=["frame_07m00s.jpg", "frame_07m30s.jpg"],
)
result = app.invoke(starting_state)

print("\nfinal state:")
formation = result["formation"]
print(f"  home in_possession={formation.home.in_possession!r}, "
      f"out_of_possession={formation.home.out_of_possession!r} (basis={formation.home.basis!r})")
print(f"  away in_possession={formation.away.in_possession!r}, "
      f"out_of_possession={formation.away.out_of_possession!r} (basis={formation.away.basis!r})")

set_pieces = result["set_pieces"]
print(f"  {len(set_pieces)} set piece(s) parsed:")
for sp in set_pieces:
    print(f"    {sp.timestamp} {sp.type} ({sp.team}) marking_system={sp.marking_system!r} window={sp.window!r}")

print("\n--- test 2: confirm the 'marking' -> 'marking_system' rename (F4 fix) actually happened ---")
assert set_pieces[0].marking_system == "zonal", "marking -> marking_system rename did not happen"
print("Confirmed: raw 'marking': 'zonal' from the stub LLM response became marking_system='zonal'.")

print("\n--- test 3: confirm out_of_possession is honestly None, not guessed ---")
assert formation.home.out_of_possession is None
assert formation.away.out_of_possession is None
print("Confirmed: out_of_possession stayed None -- the raw schema doesn't give us that cleanly yet.")

print("\n--- test 4: a raw set piece with an invalid type gets rejected, not silently dropped-but-unnoticed ---")
from pipeline_graph import _parse_set_pieces
bad_raw = {"set_pieces": [{"timestamp": "10m 00s", "type": "penalty", "team": "home_kit"}]}
parsed = _parse_set_pieces(bad_raw, "10")
assert len(parsed) == 0, "invalid set piece should have been rejected, not parsed"
print("Confirmed: invalid type was rejected (see the 'rejected a set piece' log line above) and excluded from state.")
