# WHY this is a separate file from test_step14.py rather than adding
# cases to it: test_step14.py proved the tool's three ORIGINAL, expected
# outcomes work through the real MCP dispatch path. This file proves
# the hardening added on top of that -- three genuinely new failure
# modes that could not have been triggered before this step, because
# the code paths they exercise (timeout, exception handling, the
# partial-synthesis check) didn't exist yet. Keeping them separate
# means test_step14.py still stands on its own as "the basic contract
# works" evidence.
import asyncio
import os

import mcp_server
import readiness_graph
import synthesis_agent
from test_step12 import _make_match_dir_missing_window_files, _make_three_window_specs
from match_pipeline import run_match_pipeline


print("--- test 1: empty match_dir -> a clean error, not a silent not_started or a crash ---")

content, structured = asyncio.run(
    mcp_server.mcp.call_tool("get_match_report", {"match_dir": ""})
)
print(f"  structured result: {structured}")
assert structured["status"] == "error"
assert structured["error_message"] == "match_dir must not be empty"
print("Confirmed: an empty match_dir is correctly distinguished from an unknown-but-validly-named "
      "match -- it's a caller mistake, not a match that simply hasn't started yet.")


print("\n--- test 2: readiness/synthesis genuinely hangs -> the tool times out cleanly instead of hanging forever ---")

# WHY we monkeypatch build_match_graph (not something deeper inside the
# real graph) to return a fake object whose ainvoke() just sleeps: the
# point of this test is ONLY to prove get_match_report's own
# asyncio.wait_for(...) wrapper actually fires and returns cleanly when
# the thing it's wrapping doesn't finish in time. We don't need a real
# hang inside LangGraph to prove that -- we need ANY awaitable that
# doesn't finish inside the timeout, and a fake stand-in is the most
# direct way to guarantee that deterministically, in well under a second,
# instead of actually waiting 45+ real seconds for the real timeout to
# fire against genuinely slow (but eventually-completing) real code.
class _HangingApp:
    async def ainvoke(self, state):
        await asyncio.sleep(3600)  # far longer than any timeout we'll set below
        raise AssertionError("should never actually get here")


original_build_match_graph = mcp_server.build_match_graph
original_timeout = mcp_server._READINESS_TIMEOUT_S
mcp_server.build_match_graph = lambda: _HangingApp()
mcp_server._READINESS_TIMEOUT_S = 0.2  # shrink the real 45s constant just for this test
try:
    # match_dir just needs to exist -- the fake app never touches the
    # filesystem, so any real directory (even an empty temp one) works.
    hang_dir = _make_match_dir_missing_window_files()
    content, structured = asyncio.run(
        mcp_server.mcp.call_tool("get_match_report", {"match_dir": hang_dir})
    )
finally:
    # WHY finally, same reasoning as every other monkeypatch in this
    # training: if the assertions below ever failed, leaving the real
    # build_match_graph/timeout permanently swapped out would silently
    # break every later test in the same process.
    mcp_server.build_match_graph = original_build_match_graph
    mcp_server._READINESS_TIMEOUT_S = original_timeout

print(f"  structured result: {structured}")
assert structured["status"] == "error"
assert "did not finish within" in structured["error_message"]
print("Confirmed: a readiness/synthesis call that never finishes is cut off by asyncio.wait_for(...) "
      "and reported as a clean status=\"error\", not an indefinite hang.")


print("\n--- test 3: synthesis reports report_ready=True but one document genuinely failed -> caught before a crash ---")

# WHY this reuses the REAL synthesis_agent.run_synthesis pipeline for two
# of the three documents and only fakes the third one's failure, instead
# of faking the whole function: we want to trigger the EXACT real gap
# this hardening pass found in readiness_graph.py (synthesize_node
# ignores synthesis_result["status"]) -- not a hypothetical stand-in for
# it. Patching run_synthesis itself (not _call_synthesis, which
# synthesize_node overwrites with its own stub on every call -- see its
# try/finally block) is the seam that actually lets our fake take effect.
original_run_synthesis = synthesis_agent.run_synthesis


def _fake_partial_run_synthesis(match_dir: str) -> dict:
    # Simulates exactly the real failure mode: two documents wrote
    # successfully, one (tactical) did not -- so tactical_report.md
    # genuinely does not exist on disk, even though the overall match
    # was ready enough to attempt synthesis at all.
    with open(os.path.join(match_dir, "opposition_report_away.md"), "w", encoding="utf-8") as f:
        f.write("# STUB opposition report (away)\n\nSimulated success.\n")
    with open(os.path.join(match_dir, "opposition_report_home.md"), "w", encoding="utf-8") as f:
        f.write("# STUB opposition report (home)\n\nSimulated success.\n")
    return {
        "status": "partial",
        "tactical": "failed",
        "opp_away": "complete",
        "opp_home": "complete",
    }


readiness_graph.synthesis_agent.run_synthesis = _fake_partial_run_synthesis
try:
    partial_dir = _make_match_dir_missing_window_files()
    specs = _make_three_window_specs()
    window_results = asyncio.run(run_match_pipeline(partial_dir, specs))
    assert len(window_results) == 3, "fixture regression -- expected all 3 windows to succeed"

    content, structured = asyncio.run(
        mcp_server.mcp.call_tool("get_match_report", {"match_dir": partial_dir})
    )
finally:
    readiness_graph.synthesis_agent.run_synthesis = original_run_synthesis

print(f"  structured result: {structured}")
assert structured["status"] == "error"
assert "synthesis did not complete" in structured["error_message"]
assert structured["tactical_report"] is None
assert structured["opposition_report_away"] is None  # never populated on the error path, even though the file exists
# Sanity: confirm this scenario really would have crashed the OLD version
# of this tool -- tactical_report.md genuinely isn't on disk.
assert not os.path.exists(os.path.join(partial_dir, "tactical_report.md"))
assert os.path.exists(os.path.join(partial_dir, "opposition_report_away.md"))
print("Confirmed: a partial synthesis failure (report_ready=True, but one of three documents didn't "
      "write its file) is caught by the synthesis_result status check and reported as a clean "
      "status=\"error\" naming which part failed -- not an unhandled FileNotFoundError from _read().")

print("\nAll test_step15 checks passed.")
