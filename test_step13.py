"""
test_step13.py -- proving agent_a_scan and agent_b_scan actually run
CONCURRENTLY now, not just that the code changed shape.

WHY this test measures wall-clock TIME instead of just checking the
final merged result is correct: correctness was already fully proven
by test_step10.py, and none of today's changes touched
merge_dual_agents_node's logic at all -- it still receives the exact
same two inputs it always did. The ONLY thing that changed is HOW
those two inputs get produced (concurrently vs. one-after-another), and
the only way to observe THAT is to measure how long it takes. A test
that only checked the merged output would pass identically whether the
graph secretly went back to running the agents sequentially -- it
would give false confidence about the one thing this step was actually
for.
"""

import asyncio
import time

from pipeline_graph import (
    build_graph,
    PipelineState,
    _stub_agent_a_response,
    _stub_agent_b_response,
    AGENT_A_SIMULATED_LATENCY_S,
    AGENT_B_SIMULATED_LATENCY_S,
)

SEQUENTIAL_EXPECTED_S = AGENT_A_SIMULATED_LATENCY_S + AGENT_B_SIMULATED_LATENCY_S  # 0.4 + 0.6 = 1.0s
PARALLEL_EXPECTED_S = max(AGENT_A_SIMULATED_LATENCY_S, AGENT_B_SIMULATED_LATENCY_S)  # 0.6s

print("--- test 1: isolate the concept -- await the two stubs one at a time vs. together ---")
print(f"  Agent A simulated latency: {AGENT_A_SIMULATED_LATENCY_S}s")
print(f"  Agent B simulated latency: {AGENT_B_SIMULATED_LATENCY_S}s")


async def _sequential_calls():
    # WHY this is the "before" comparison, not just a description of
    # the old code: awaiting one call, then awaiting the next, on
    # separate lines, is EXACTLY what the old
    # profile_source -> agent_a_scan -> agent_b_scan edge chain did at
    # the graph level. This reproduces that behaviour directly with the
    # coroutines themselves, no graph involved, so the comparison below
    # isn't confused by any other node's overhead.
    a = await _stub_agent_a_response("t")
    b = await _stub_agent_b_response("t")
    return a, b


async def _concurrent_calls():
    # WHY asyncio.gather(...) here, not two separate `await`s: gather()
    # schedules BOTH coroutines onto the event loop up front and only
    # returns once both are done -- this is the raw asyncio primitive
    # that LangGraph's fan-out/fan-in is built on top of. Two `await`s
    # in a row would go back to the sequential behaviour above, even
    # though both functions are async -- `async def` alone doesn't make
    # anything concurrent, it just makes concurrency POSSIBLE. gather()
    # (or LangGraph's superstep scheduler) is what actually cashes that
    # possibility in.
    a, b = await asyncio.gather(
        _stub_agent_a_response("t"),
        _stub_agent_b_response("t"),
    )
    return a, b


start = time.monotonic()
asyncio.run(_sequential_calls())
sequential_elapsed = time.monotonic() - start
print(f"  Sequential (await, then await): {sequential_elapsed:.2f}s "
      f"(expected ~{SEQUENTIAL_EXPECTED_S}s)")

start = time.monotonic()
asyncio.run(_concurrent_calls())
concurrent_elapsed = time.monotonic() - start
print(f"  Concurrent (asyncio.gather):     {concurrent_elapsed:.2f}s "
      f"(expected ~{PARALLEL_EXPECTED_S}s)")

# Generous margins to absorb sandbox scheduling jitter, but still tight
# enough that sequential-vs-concurrent can't be confused with noise:
# sequential must be close to the SUM (1.0s), concurrent close to the
# MAX (0.6s) -- a ~0.4s gap that's impossible to get by accident.
assert sequential_elapsed >= SEQUENTIAL_EXPECTED_S - 0.05, (
    f"sequential calls finished suspiciously fast ({sequential_elapsed:.2f}s) -- "
    f"did they secretly run concurrently?"
)
assert concurrent_elapsed < SEQUENTIAL_EXPECTED_S - 0.15, (
    f"concurrent calls took {concurrent_elapsed:.2f}s, too close to the sequential "
    f"time of {sequential_elapsed:.2f}s -- gather() may not be overlapping the sleeps"
)
print("Confirmed: gather() finished noticeably faster than sequential awaits -- "
      "the two coroutines genuinely overlapped in time, not just in source-code order.")


print("\n--- test 2: the same effect, but through the REAL compiled LangGraph, "
      "not a hand-written gather() ---")
app = build_graph()
state = PipelineState(window_id="13", frame_paths=["frame_13m00s.jpg"])

start = time.monotonic()
result = asyncio.run(app.ainvoke(state))
graph_elapsed = time.monotonic() - start
print(f"  Full graph run (profile_source -> [agent_a_scan, agent_b_scan] -> "
      f"merge_dual_agents -> ...): {graph_elapsed:.2f}s")

# WHY this threshold is a bit looser than test 1's (SEQUENTIAL_EXPECTED_S
# - 0.15 vs -0.10 here): the full graph run does strictly more work than
# the isolated gather() call above (profile_source_node, merge, escalation
# routing, burst_scan all run too), so a little extra wall-clock slack is
# expected and legitimate -- the important comparison is still "closer to
# 0.6s than to 1.0s", not "identical to test 1's number".
assert graph_elapsed < SEQUENTIAL_EXPECTED_S - 0.10, (
    f"full graph run took {graph_elapsed:.2f}s -- too close to the ~{SEQUENTIAL_EXPECTED_S}s "
    f"sequential time; agent_a_scan and agent_b_scan may not actually be running concurrently "
    f"inside the compiled graph"
)
print("Confirmed: the compiled LangGraph itself -- not just a hand-written asyncio.gather() -- "
      "ran agent_a_scan and agent_b_scan concurrently via the fan-out/fan-in wiring.")


print("\n--- test 3: correctness is untouched -- same merged result shape as before the rewiring ---")
# This deliberately reuses the same assertions test_step8/test_step9 already
# made on this exact stub data, on window "07" specifically (matching
# their fixtures), to prove the rewiring changed nothing about WHAT comes
# out of the graph, only HOW FAST.
state2 = PipelineState(window_id="07", frame_paths=["frame_07m00s.jpg"])
result2 = asyncio.run(app.ainvoke(state2))
formation = result2["formation"]
assert formation.home.basis == "dual_agent_resolved"  # A and B disagree on home shape
assert formation.away.basis == "dual_agent_agreed"     # A and B agree on away shape
set_piece = next(sp for sp in result2["set_pieces"] if sp.timestamp == "7m 02s")
assert set_piece.burst_resolved is True  # escalation to burst_scan still fires
print("Confirmed: merge, disagreement handling, and escalation routing all still produce "
      "exactly the same results as before -- concurrency changed timing, not behaviour.")
