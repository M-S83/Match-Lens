# WHY this test calls mcp.call_tool("get_match_report", {...}) instead of
# just calling get_match_report(match_dir) directly as a plain async
# function: calling the bare function would only prove our business
# logic works -- it would NOT prove the MCP layer itself is wired up
# correctly. mcp.call_tool() goes through the exact same path a real
# client (Claude Desktop, Claude Code, any MCP client) would go
# through: look up "get_match_report" in the registered tool table,
# validate the incoming arguments dict against the auto-generated
# inputSchema, invoke the underlying function, then convert whatever it
# returns into the structured content blocks the protocol sends back.
# That's the thing we actually need to prove works, since it's the new
# surface this step adds.
import asyncio
import json
import os
import tempfile

import mcp_server
from match_pipeline import run_match_pipeline

# WHY importing these two helpers also re-runs ALL of test_step12.py's own
# module-level test/assert code, right here, before anything below runs:
# that's just how Python import works -- importing any name from a module
# executes the whole module top-to-bottom once. That's a genuine, honest
# side effect (you'll see test_step12's own "---test 1---"/"---test 2---"
# output appear first), not a bug -- and it's harmless here because
# test_step12.py's asserts either pass silently or raise, same as running
# it standalone. Reusing its fixtures this way avoids a THIRD copy of
# _make_three_window_specs() drifting out of sync with the first two.
from test_step12 import _make_match_dir_missing_window_files, _make_three_window_specs


def _write(match_dir, fname, data):
    with open(os.path.join(match_dir, fname), "w", encoding="utf-8") as f:
        json.dump(data, f)


print("--- test 0: the tool is actually discoverable, with a schema derived from our type hints ---")

tools = asyncio.run(mcp_server.mcp.list_tools())
tool_names = [t.name for t in tools]
print(f"  registered tools: {tool_names}")
assert tool_names == ["get_match_report"], (
    "expected exactly one curated tool right now -- analyse_match is a "
    "deliberately deferred next step, not a missing tool (see mcp_server.py docstring)"
)

report_tool = tools[0]
print(f"  description (this is our docstring, verbatim): {report_tool.description!r}")
print(f"  inputSchema:  {report_tool.inputSchema}")
print(f"  outputSchema: {report_tool.outputSchema}")
assert report_tool.inputSchema["required"] == ["match_dir"]
assert report_tool.inputSchema["properties"]["match_dir"]["type"] == "string"
# WHY we assert on outputSchema containing "status" with our three literal
# values: this proves FastMCP actually read MatchReportResult's fields
# and Literal[...] type, not just "some dict came back" -- the schema a
# calling model sees is genuinely generated from our Pydantic model.
assert set(report_tool.outputSchema["properties"]["status"]["enum"]) == {
    "not_started", "not_ready", "ready",
}
print("Confirmed: the tool is discoverable via the standard MCP tools/list call, and its input/output "
      "schemas were generated automatically from get_match_report's type hints and MatchReportResult -- "
      "nobody hand-wrote a JSON schema anywhere.")


print("\n--- test 1: match_dir that was never analysed at all -> not_started, not a crash ---")

nonexistent_dir = os.path.join(tempfile.gettempdir(), "matchlens_never_created_xyz")
assert not os.path.isdir(nonexistent_dir)  # sanity: make sure it really doesn't exist

content, structured = asyncio.run(
    mcp_server.mcp.call_tool("get_match_report", {"match_dir": nonexistent_dir})
)
print(f"  structured result: {structured}")
assert structured["status"] == "not_started"
assert structured["blocking_issues"] == []
assert structured["tactical_report"] is None
print("Confirmed: an unknown/never-created match_dir comes back as a clean not_started status, "
      "not an exception -- exactly what a client polling for a report needs to be able to rely on.")


print("\n--- test 2: match_dir that exists but is missing the window-level files -> not_ready, with real blocking_issues ---")

not_ready_dir = _make_match_dir_missing_window_files()
# Deliberately do NOT call run_match_pipeline() here -- window_plan.json /
# running_summary.json / source_profile.json are left missing, same gap
# test_step11's "not ready" case exercised.

content, structured = asyncio.run(
    mcp_server.mcp.call_tool("get_match_report", {"match_dir": not_ready_dir})
)
print(f"  structured result: {structured}")
assert structured["status"] == "not_ready"
assert len(structured["blocking_issues"]) > 0
assert structured["tactical_report"] is None
assert structured["opposition_report_home"] is None
print(f"Confirmed: a match_dir that exists but is missing required inputs comes back as not_ready with "
      f"{len(structured['blocking_issues'])} real blocking issue(s) from the actual readiness gate, "
      f"not a fabricated or empty list.")


print("\n--- test 3: fully ready match -> ready, with the real report content read back off disk ---")

ready_dir = _make_match_dir_missing_window_files()
specs = _make_three_window_specs()
window_results = asyncio.run(run_match_pipeline(ready_dir, specs))
assert len(window_results) == 3, "fixture regression -- expected all 3 windows to succeed"

content, structured = asyncio.run(
    mcp_server.mcp.call_tool("get_match_report", {"match_dir": ready_dir})
)
print(f"  status: {structured['status']}")
assert structured["status"] == "ready"
assert structured["blocking_issues"] == []
assert "STUB" in structured["tactical_report"]
assert "STUB" in structured["opposition_report_home"]
assert "STUB" in structured["opposition_report_away"]

# WHY we also check the raw `content` (the TextContent blocks), not just
# `structured`: call_tool returns BOTH a human-readable text rendering
# and the structured dict, because not every MCP client understands
# structured output -- some only read the text content blocks. A real
# tool should make sense read either way.
assert len(content) == 1
assert "ready" in content[0].text
print("Confirmed: for a match that genuinely passed the readiness gate, get_match_report -- called "
      "through the real MCP tool-dispatch path -- returns the actual tactical and opposition report "
      "text that run_synthesis() wrote to disk, in both structured and text form.")

print("\nAll test_step14 checks passed.")
