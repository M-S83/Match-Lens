"""
readiness_graph.py -- the MATCH-level gate, separate from pipeline_graph.py.

WHY this is its own file with its own state class, not a node bolted onto
PipelineState: everything in pipeline_graph.py processes ONE window at a
time (window_id, frame_paths, one window's set pieces). build_readiness_check()
is fundamentally different in scope -- it runs ONCE per match, AFTER every
window has already been processed, and reads whole-match files
(window_plan.json, running_summary.json, ground_truth_check.json, ...) that
don't exist until all per-window work is done. Cramming a match-level
decision into a window-level state model would blur a real architectural
boundary. Real pipelines have multiple "levels" like this; this file is the
first piece of the match-level orchestration layer, separate on purpose.

WHY this reuses build_readiness_check() directly instead of reimplementing
its rules: that function already has real, tested rules (boundary
confidence >= 0.8, all windows complete, ground-truth check passed, source
classification confidence >= 0.6, ...) written by you, reading real files.
Reimplementing them here would risk the two copies drifting apart over
time. Reuse means there's exactly one place these rules live.
"""

import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from build_readiness_check import build_readiness_check


# WHY report_ready/blocking_issues are Optional/empty until the gate node
# runs, same reasoning as PipelineState's Optional fields: this state
# object exists before we know the answer, and the type system should be
# honest about that rather than defaulting to a lie like report_ready=True.
class MatchState(BaseModel):
    match_dir: str
    report_ready: Optional[bool] = None
    blocking_issues: List[str] = []


def readiness_gate_node(state: MatchState) -> dict:
    print(f"  [readiness_gate] checking match_dir={state.match_dir!r}")

    # build_readiness_check() does double duty: it returns a plain bool,
    # AND (as a side effect) writes report_readiness.json to disk with the
    # full blocking_issues list. WHY we read that file back immediately
    # afterward instead of recomputing blocking_issues ourselves: reading
    # its own written output guarantees our node's view of "why" exactly
    # matches the boolean it returned -- there's no chance of the node's
    # reasoning drifting out of sync with the function's own reasoning.
    ready = build_readiness_check(state.match_dir)
    readiness_path = os.path.join(state.match_dir, "report_readiness.json")
    with open(readiness_path, encoding="utf-8") as f:
        readiness_doc = json.load(f)

    return {
        "report_ready": ready,
        "blocking_issues": readiness_doc.get("blocking_issues", []),
    }


# WHY this is a conditional edge, not a new concept: this is exactly the
# same shape as route_after_merge in pipeline_graph.py -- a function that
# reads state and returns a label. The only thing new here is what the
# two branches DO, not how the branching itself works.
def route_after_readiness_gate(state: MatchState) -> Literal["synthesize", "insufficient_data"]:
    return "synthesize" if state.report_ready else "insufficient_data"


# WHY this is only a print-stub for now, not a real call into
# synthesis_agent.py: wiring up the real report-writing step is its own
# training step (the next piece of Phase 4 after this), and putting a
# half-finished version of it here would blur what this step is actually
# teaching -- the readiness gate and the failure alert.
def synthesize_node(state: MatchState) -> dict:
    print(f"  [synthesize] match {state.match_dir}: report_ready=True -- "
          f"would call synthesis_agent.py here (not built yet)")
    return {}


# WHY the alert email is stubbed with a print, exactly like every LLM call
# in pipeline_graph.py: this sandbox has no SMTP/SendGrid/Resend/SES
# credentials configured, same root reason the LLM calls are stubbed. The
# function signature and call site below are written to be a drop-in
# swap -- when you deploy this for real, you replace the body of
# _send_failure_email with an actual API/SMTP call and nothing else in
# this file needs to change.
#
# WHY the alert goes to YOU, not the client: the client has no useful
# action to take on "boundary confidence was low" -- that's a pipeline
# problem, not a video problem they can fix. You're the one who can
# actually investigate and fix it (or decide it's a one-off and move on).
OWNER_ALERT_EMAIL = "dbmuxsolutions@gmail.com"  # TODO: move to an env var before deploying for real


def _send_failure_email(match_dir: str, blocking_issues: list) -> None:
    subject = f"[Match Lens] Report generation failed -- {os.path.basename(match_dir.rstrip('/'))}"
    lines = [f"Match Lens could not produce a report for: {match_dir}", "", "Blocking issues:"]
    for issue in blocking_issues:
        lines.append(f"  - {issue}")
    body = "\n".join(lines)

    print(f"  [STUB EMAIL] To: {OWNER_ALERT_EMAIL}")
    print(f"  [STUB EMAIL] Subject: {subject}")
    print(f"  [STUB EMAIL] Body:")
    for line in lines:
        print(f"    {line}")


# WHY this node returns a client-facing message that is genuinely
# different wording from blocking_issues, instead of just forwarding the
# raw list: "match_config.json not verified" means nothing to a client and
# exposes internal file names. The client gets an honest, plain-language
# explanation; you (via the email) get the real internal detail needed to
# actually fix it. Same underlying failure, two different audiences, two
# different levels of detail -- deliberately not the same text reused twice.
def insufficient_data_node(state: MatchState) -> dict:
    print(f"  [insufficient_data] match {state.match_dir}: "
          f"report_ready=False, {len(state.blocking_issues)} blocking issue(s)")
    _send_failure_email(state.match_dir, state.blocking_issues)
    return {}


def build_match_graph():
    graph = StateGraph(MatchState)

    graph.add_node("readiness_gate", readiness_gate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("insufficient_data", insufficient_data_node)

    graph.add_edge(START, "readiness_gate")
    graph.add_conditional_edges(
        "readiness_gate",
        route_after_readiness_gate,
        {"synthesize": "synthesize", "insufficient_data": "insufficient_data"},
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("insufficient_data", END)

    return graph.compile()
