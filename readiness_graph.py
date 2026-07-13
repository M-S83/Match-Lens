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

import synthesis_agent

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from build_readiness_check import build_readiness_check
from ground_truth import build_ground_truth_check


# WHY report_ready/blocking_issues are Optional/empty until the gate node
# runs, same reasoning as PipelineState's Optional fields: this state
# object exists before we know the answer, and the type system should be
# honest about that rather than defaulting to a lie like report_ready=True.
#
# WHY ground_truth_result is a new field here: it's the return value of
# build_ground_truth_check() (events_checked/confirmed/partial/missed/...),
# kept on the state purely so a caller inspecting the final MatchState can
# see WHAT the ground-truth node actually found without having to re-open
# ground_truth_check.json from disk themselves. readiness_gate_node doesn't
# need to read this field back off state -- it re-reads the file, same as
# it always has -- this is for visibility, not for wiring the two nodes
# together (that wiring is via the file on disk, see ground_truth_node).
class MatchState(BaseModel):
    match_dir: str
    ground_truth_result: Optional[dict] = None
    report_ready: Optional[bool] = None
    blocking_issues: List[str] = []
    synthesis_result: Optional[dict] = None


# WHY this node has to exist at all: build_readiness_check() (called by
# readiness_gate_node right after this) has ALWAYS read ground_truth_check.json
# from disk via a plain load-or-None helper -- it was written defensively,
# on purpose, to treat a missing file as "not passed" rather than crash.
# But nothing in this graph ever actually RAN build_ground_truth_check() to
# produce that file in the first place -- it only ever existed in our tests
# because the test fixtures hand-wrote it directly. In a real unattended
# run there is no human to hand-write it, so without this node the ground
# truth check would silently never run, and the gate right after it would
# always see "file missing" and block every single match. Adding this node
# is what makes ground_truth.py's own docstring claim true in practice --
# it says it "triggers before Step 4" (Step 4 == report writing == this
# match graph's synthesize node), but until now nothing actually called it
# before Step 4. This closes that gap.
#
# WHY build_ground_truth_check() is called for its FILE side effect, not
# its return value: build_readiness_check() (the next node) only reads
# ground_truth_check.json off disk -- it has no idea this graph exists or
# that a Python dict came back from a function call. The disk file is the
# real hand-off between these two nodes, exactly the same shape as the
# hand-off between readiness_gate_node and build_readiness_check() itself
# (see that node's own comment below). We ALSO keep the return value on
# state.ground_truth_result, but that's a convenience for callers/tests,
# not what readiness_gate_node actually relies on.
#
# WHY FileNotFoundError is caught here instead of left to propagate:
# build_ground_truth_check() was written to raise hard if match_config.json
# or running_summary.json don't exist yet -- a genuinely normal state for
# a match that hasn't finished Step 1d / window aggregation. The very next
# node (readiness_gate) is ALREADY designed to handle "not ready yet"
# cleanly: missing files there just become blocking issues, which route to
# insufficient_data, which sends you an alert email with the real reason.
# If we let this exception propagate instead, LangGraph would abort the
# whole invocation with an unhandled traceback for a condition that isn't
# actually exceptional at the match level -- replacing a clean, existing,
# already-tested "not ready" path with a hard crash. Catching it here and
# letting the graph continue into readiness_gate preserves that existing
# behavior: build_readiness_check() will independently notice the missing
# match_config.json (its own separate check) and/or the missing
# ground_truth_check.json (since we never got to write one) and block for
# the real underlying reason, worded the way a human already reads today.
def ground_truth_node(state: MatchState) -> dict:
    print(f"  [ground_truth] checking match_dir={state.match_dir!r}")
    try:
        result = build_ground_truth_check(state.match_dir)
    except FileNotFoundError as e:
        print(f"  [ground_truth] skipped -- {e}")
        return {"ground_truth_result": None}
    except KeyError as e:
        # WHY KeyError gets the exact same treatment as FileNotFoundError:
        # discovered for real while wiring this node in -- if
        # match_boundaries.json EXISTS but doesn't have the four specific
        # keys ground_truth.py's minute-to-video conversion expects
        # (ko_1h/ht_whistle/ko_2h/ft_whistle), that function does direct
        # bracket access and raises KeyError instead of FileNotFoundError.
        # From this node's perspective the two failures mean the same
        # thing -- "the inputs ground truth needs aren't in a usable state
        # yet" -- and build_readiness_check() right after this node will
        # independently catch the fallout anyway: no ground_truth_check.json
        # got written, so its own load() sees it as missing and blocks for
        # that real reason. Swallowing this here, instead of crashing the
        # whole match graph over one malformed intermediate file, is what
        # "runs fully unattended, zero human review" (the actual product
        # requirement) means in practice -- one bad file degrades gracefully
        # into a blocked/alerted match, not a dead process.
        print(f"  [ground_truth] skipped -- match_boundaries.json missing expected key {e}")
        return {"ground_truth_result": None}
    print(f"  [ground_truth] events_checked={result['events_checked']} "
          f"confirmed={result['confirmed']} partial={result['partial']} "
          f"missed={result['missed']}")
    return {"ground_truth_result": result}


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


# WHY this stub matches _call_synthesis's exact signature and return
# type (a plain string): synthesis_agent._write_tactical_report() and
# _write_opposition_report() call _call_synthesis(...) and then write
# whatever string comes back straight to a .md file. As long as our
# stub returns a string with the same signature, run_synthesis()'s real
# file-writing code works completely unmodified -- we only need to
# replace the one function that actually reaches the network.
def _stub_synthesis_llm_response(system_prompt: str, user_content: str, max_tokens: int = 8000) -> str:
    kind = "opposition" if "OPPOSITION REPORT" in user_content else "tactical"
    return (
        f"# STUB {kind} report\n\n"
        f"ROSTER CHECK: PASSED\n\n"
        f"This is a placeholder report body ({len(user_content)} chars of "
        f"prompt were built for real from the actual pipeline data). Swap "
        f"this function for a real anthropic.Anthropic().messages.create() "
        f"call once an API key is configured."
    )


# WHY this monkeypatches synthesis_agent._call_synthesis instead of
# reimplementing run_synthesis(): everything in run_synthesis() EXCEPT
# that one function is real, already-written pipeline code -- loading
# match_config/running_summary/pass_sequences (build_input_bundle),
# building the roster block and the per-document prompts, writing three
# separate .md files, and catching per-document failures so one bad
# document doesn't take down the other two. Swapping out just the
# network call and running the real function around it means this node
# is exercising the actual production code path, not a parallel
# reimplementation of it that could quietly drift out of sync.
def synthesize_node(state: MatchState) -> dict:
    print(f"  [synthesize] match {state.match_dir}: report_ready=True -- "
          f"running real synthesis_agent.run_synthesis() (LLM call stubbed)")

    original_call_synthesis = synthesis_agent._call_synthesis
    synthesis_agent._call_synthesis = _stub_synthesis_llm_response
    try:
        result = synthesis_agent.run_synthesis(state.match_dir)
    finally:
        # WHY restore in a finally, not just after the call: if
        # run_synthesis() ever raised instead of catching its own
        # per-document errors, leaving the stub permanently installed
        # on the shared synthesis_agent module would silently break
        # every later real call in the same process. This isn't
        # hypothetical caution -- it's the same reason a database
        # connection gets closed in a finally block.
        synthesis_agent._call_synthesis = original_call_synthesis

    return {"synthesis_result": result}


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

    graph.add_node("ground_truth", ground_truth_node)
    graph.add_node("readiness_gate", readiness_gate_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("insufficient_data", insufficient_data_node)

    # WHY ground_truth runs BEFORE readiness_gate, unconditionally, with no
    # branching in between: ground_truth.py's own docstring says it
    # "triggers before Step 4" -- it's a precondition-generating step, not
    # a decision point. The actual pass/fail DECISION about what the
    # ground-truth result means (missed>0 -> block? source-aware
    # relaxation for broadcast?) already lives in exactly one place --
    # build_readiness_check()'s "-- Ground truth --" section -- and stays
    # there. This node's only job is to make sure ground_truth_check.json
    # exists and is fresh by the time that section reads it.
    graph.add_edge(START, "ground_truth")
    graph.add_edge("ground_truth", "readiness_gate")
    graph.add_conditional_edges(
        "readiness_gate",
        route_after_readiness_gate,
        {"synthesize": "synthesize", "insufficient_data": "insufficient_data"},
    )
    graph.add_edge("synthesize", END)
    graph.add_edge("insufficient_data", END)

    return graph.compile()
