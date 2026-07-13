"""
mcp_server.py -- the first Match Lens MCP tool: get_match_report.

WHY this file exists at all, instead of just importing readiness_graph.py
directly into whatever calls it: everything we've built so far
(pipeline_graph.py, match_pipeline.py, readiness_graph.py) is only
reachable by writing a Python script that imports it -- exactly what
every test_stepN.py file does. That's perfect for training, but it's
not a SERVICE. Nothing outside this sandbox (a website, a support tool,
a future orchestrator) can safely call "analyse this match" without
reading our source code and knowing our internal function names, state
models, and file layout. MCP exists specifically to solve that: it
lets us expose a small, curated set of OUTCOME-level actions through
one standard protocol that any MCP-compatible client can discover and
call, without needing to know anything about LangGraph, Pydantic
states, or which JSON files live in a match directory.

WHY this step ships ONLY get_match_report, not also analyse_match (the
other tool named in the training plan): "curate tools ruthlessly" is
one of the core principles for writing good agent tools. A genuinely
outcome-level analyse_match(video_path, teamsheet) needs frame
extraction, boundary detection, and window planning to turn a raw video
into the WindowSpec list match_pipeline.run_match_pipeline() expects --
none of which have been rebuilt as part of this training's rewrite
(frame_extraction.py / detect_boundaries.py / window_plan.py are real,
existing files in this repo, but they were never brought into
pipeline_graph.py's world). Faking that step -- accepting a video_path
and pretending to turn it into windows -- would be exactly the kind of
fabricated-completeness mistake match_pipeline.py's own docstring
explicitly rejected for match_boundaries.json/ground_truth_check.json.
Shipping ONE tool that is genuinely, fully wired end-to-end is more
honest and more useful than shipping TWO tools where one is a facade.
analyse_match is a deliberately named next step, not an oversight.
"""

import os
from typing import List, Literal, Optional

from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP

from readiness_graph import build_match_graph, MatchState


# WHY this Pydantic model instead of returning a plain dict: it's the
# exact same reasoning as every other model in this training, applied
# at a new boundary -- the MCP tool's OUTPUT. FastMCP reads this
# model's fields and auto-generates the tool's outputSchema (the
# structured shape a calling model is told to expect back), the same
# way it reads get_match_report's parameter type hints to generate the
# tool's inputSchema. Callers get a stable, typed, self-documenting
# contract instead of an unstructured dict that could silently change
# shape between versions.
class MatchReportResult(BaseModel):
    # WHY three states instead of a plain boolean "ready": a caller (or
    # an LLM deciding what to do next) needs to react differently to
    # "this match_id doesn't exist at all" (probably a typo, or
    # analyse_match hasn't been called yet) versus "this match exists
    # but is genuinely blocked" (something needs fixing, or it's still
    # processing) versus "here's your report." Collapsing all three
    # into one boolean would throw away information the caller actually
    # needs to decide what to do next.
    status: Literal["not_started", "not_ready", "ready"]
    blocking_issues: List[str] = []
    tactical_report: Optional[str] = None
    opposition_report_home: Optional[str] = None
    opposition_report_away: Optional[str] = None


# WHY this is `FastMCP`, the high-level SDK class, instead of hand-rolling
# JSON-RPC message handling ourselves: MCP's wire format (tools/list,
# tools/call, capability negotiation, content-block encoding) is
# genuinely fiddly protocol plumbing that has nothing to do with Match
# Lens. FastMCP handles all of that and lets us focus purely on: what
# tools do we have, and what do they do. This is the same "don't
# reinvent a solved problem" reasoning as reusing validate_set_piece()
# back in pipeline_graph.py instead of writing a second normalizer.
mcp = FastMCP("match-lens")


# WHY the docstring below is written the way it is, not just "gets the
# report": FastMCP takes this exact docstring and hands it to whatever
# AI client calls this tool, as the tool's `description` field -- that
# description is literally the ONLY context the calling model has to
# decide when and how to use this tool. A vague docstring produces a
# model that calls the tool at the wrong time or with the wrong
# arguments; a docstring that states inputs, outputs, AND the possible
# outcomes (ready and here's why not) gives the calling model enough to
# reason correctly about what happens next.
@mcp.tool()
async def get_match_report(match_dir: str) -> MatchReportResult:
    """Get the tactical and opposition reports for a match, if ready.

    Runs the real readiness check against the match's on-disk files and,
    if every requirement is satisfied, returns the tactical report and
    both opposition reports. If the match has blocking issues (missing
    or incomplete data), returns status="not_ready" with the specific
    blocking_issues instead of an error -- this is a normal, expected
    outcome for a match still being processed, not a failure. If
    match_dir does not exist at all, returns status="not_started".

    Args:
        match_dir: Path to the match's working directory (the same
            directory analyse_match writes its window and match-level
            files into).
    """
    # WHY this checks os.path.isdir() FIRST, before doing anything else:
    # a match_id that was never actually analysed (typo, or the caller
    # is polling before analyse_match has finished creating the
    # directory) is a fundamentally different situation from a match
    # that exists but is missing specific files -- see the WHY comment
    # on MatchReportResult.status above. Checking this first also means
    # we never even touch the readiness gate for a directory that can't
    # possibly contain anything.
    if not os.path.isdir(match_dir):
        return MatchReportResult(status="not_started")

    # WHY this re-runs the REAL match-level graph on every call instead
    # of reading back a possibly-stale report_readiness.json some
    # earlier run wrote: the whole point of an outcome-level tool is
    # that the caller shouldn't have to know or care that readiness
    # checking and synthesis are two separate internal steps -- they
    # ask "is the report ready, and if so what does it say," and get a
    # single, freshly-computed answer. Re-running readiness_gate_node
    # and (if ready) run_synthesis() is cheap and deterministic given
    # the same on-disk files, so there's no real cost to always
    # computing a fresh answer instead of trusting a cached one that
    # could be out of date if a window was reprocessed since.
    app = build_match_graph()
    result = await app.ainvoke(MatchState(match_dir=match_dir))

    if not result["report_ready"]:
        return MatchReportResult(
            status="not_ready",
            blocking_issues=result["blocking_issues"],
        )

    # WHY we read the .md files back from disk instead of returning
    # result["synthesis_result"] (the dict run_synthesis() already
    # returned): that dict only says {"status": "complete", ...} --
    # it's a completion RECORD, not the report CONTENT. The actual
    # tactical/opposition text lives in the .md files run_synthesis()
    # wrote to match_dir. This is the same "read back what was actually
    # written, don't assume you already have it in memory" pattern
    # test_step11.py used to prove the reports genuinely landed on disk.
    def _read(fname: str) -> str:
        with open(os.path.join(match_dir, fname), encoding="utf-8") as f:
            return f.read()

    return MatchReportResult(
        status="ready",
        tactical_report=_read("tactical_report.md"),
        opposition_report_home=_read("opposition_report_home.md"),
        opposition_report_away=_read("opposition_report_away.md"),
    )


# WHY this file can also just be run directly (`python mcp_server.py`)
# to start a real server, separate from how test_step14.py exercises
# the tool: `mcp.run()` starts an actual MCP server listening over
# stdio, the transport a local client like Claude Desktop or Claude
# Code would connect through. Training and tests call get_match_report
# through mcp.call_tool(...) directly (see test_step14.py) rather than
# spinning up a real subprocess+stdio connection, because that tests
# the exact same schema-validation-and-dispatch machinery a real client
# would go through, without the overhead of an actual subprocess.
if __name__ == "__main__":
    mcp.run()
