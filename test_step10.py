from pipeline_graph import build_graph, PipelineState

app = build_graph()

print("--- test: dual-agent scan + merge on a window where the two agents disagree ---")
w = PipelineState(window_id="09", frame_paths=["frame_09m00s.jpg"])
result = app.invoke(w)

print("\nDisagreements logged:")
for d in result["disagreements"]:
    print(f"  - {d}")

formation = result["formation"]
print(f"\nMerged home formation: {formation.home.in_possession!r} (basis={formation.home.basis!r})")
print(f"Merged away formation: {formation.away.in_possession!r} (basis={formation.away.basis!r})")

# -- Assertions -----------------------------------------------------------

# 1. Home formation disagreed (A said 4-3-3, B said 4-2-3-1) -> longer
#    string ("4-2-3-1") should win, and basis should say "resolved" not "agreed".
assert formation.home.in_possession == "4-2-3-1", "longer-string heuristic should have picked B's reading"
assert formation.home.basis == "dual_agent_resolved"
print("Confirmed: home formation disagreement resolved via the longer-string heuristic.")

# 2. Away formation agreed (both said 4-4-2) -> basis should say "agreed".
assert formation.away.in_possession == "4-4-2"
assert formation.away.basis == "dual_agent_agreed"
print("Confirmed: away formation agreement recorded as agreed, not resolved.")

# 3. Set pieces: should have 2 entries -- the 7m02s corner (seen by both,
#    A's marking kept) and the 12m15s throw-in (seen by B only).
timestamps = sorted(sp.timestamp for sp in result["set_pieces"])
assert timestamps == ["12m 15s", "7m 02s"], f"unexpected set piece timestamps: {timestamps}"
corner = next(sp for sp in result["set_pieces"] if sp.timestamp == "7m 02s")
assert corner.marking_system == "zonal", "should have kept Agent A's marking reading"
print("Confirmed: set pieces merged correctly -- one shared event (A's marking kept), "
      "one Agent-B-only event, no duplicates.")

# 4. Disagreements log should mention all three: home formation, marking
#    mismatch, and the Agent-B-only throw-in.
joined = " | ".join(result["disagreements"])
assert "home formation" in joined
assert "marking A=" in joined
assert "seen by Agent B only" in joined
print("Confirmed: all three disagreement types were logged for human review.")

# 5. Because the corner at 7m02s is not yet burst-resolved, escalation
#    routing should still have kicked in same as before.
corner_after = next(sp for sp in result["set_pieces"] if sp.timestamp == "7m 02s")
assert corner_after.burst_resolved is True
assert corner_after.source == "5fps_burst"
print("Confirmed: merged set piece still correctly escalated to burst_scan afterwards.")
