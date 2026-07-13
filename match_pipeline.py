"""
match_pipeline.py -- the FAN-IN layer between the window-level graph
(pipeline_graph.py) and the match-level graph (readiness_graph.py).

WHY this is a new file, not a method on PipelineState or MatchState:
it doesn't belong to either scope. It CONSUMES many PipelineState
results (one per window) and PRODUCES the match-level files that
MatchState's readiness_gate_node reads. Neither existing state model
is the right home for "run N windows, then combine them" -- that's a
distinct responsibility sitting between the two levels, which is
exactly why it gets its own file instead of being bolted onto whichever
of the two happened to be open last.

WHY this file does NOT try to fabricate every match-level file
build_readiness_check() reads: boundary detection (match_boundaries.json),
the deep ground-truth scan (ground_truth_check.json), the human
confirmation queue (confirmation_queue.json / rerun_queue.json), and
deep skill metrics (deep_skill_metrics.json) are all separate pipeline
stages in the real Match Lens system. None of them are derived from
window_id/formation/set_pieces -- they come from entirely different
agents that were never built as part of pipeline_graph.py's per-window
scan. Faking plausible-looking values for them here would hide a real
gap behind a false sense of completeness. This file only aggregates
what running the window-level graph N times actually produces:
- how many windows completed vs. failed (window_plan.json /
  running_summary.json)
- a match-wide source_profile.json, taken from the windows that were
  actually classified (see the WHY comment on _pick_match_source_profile
  below for the honest caveat on this one)
Everything else still genuinely needs its own, not-yet-built pipeline
stage. Noticing and naming that gap honestly is the actual lesson here,
not papering over it with fabricated JSON.
"""

import json
import os
from typing import List, Literal, Optional

from pydantic import BaseModel

from pipeline_graph import build_graph, PipelineState
from models import WindowResult, SourceProfile


# WHY WindowSpec is its own tiny model instead of just passing raw
# start_s/end_s/half/frame_paths as loose function arguments: it's the
# same reasoning as every other model in this training -- callers of
# run_match_pipeline() get validation for free (e.g. half must be "1H"
# or "2H", not "first-half"), and the function signature stays at one
# argument (a list of WindowSpec) instead of four parallel lists that
# could silently get out of sync with each other.
class WindowSpec(BaseModel):
    window_id: str
    start_s: float
    end_s: float
    half: Literal["1H", "2H"]
    event_window: bool = False
    frame_paths: List[str] = []


# WHY the window-level graph is built ONCE at module load, not once per
# window inside the loop: build_graph() just wires nodes and edges
# together -- it doesn't hold any per-window data. Building it fresh
# for every window would be pure wasted work repeated N times for zero
# benefit, the same reason you don't recompile a function every time
# you call it.
_window_graph = build_graph()


def run_window(spec: WindowSpec) -> WindowResult:
    """Run ONE window through the real window-level graph and translate
    its result into a WindowResult -- the match-level record of what
    happened in this window. Raises whatever the graph raises; the
    caller (run_match_pipeline) decides how to handle a failed window,
    not this function."""
    result = _window_graph.invoke(
        PipelineState(window_id=spec.window_id, frame_paths=spec.frame_paths)
    )
    return WindowResult(
        window_id=spec.window_id,
        start_s=spec.start_s,
        end_s=spec.end_s,
        half=spec.half,
        event_window=spec.event_window,
        source_profile=result.get("source_profile"),
        formation=result.get("formation"),
        set_pieces=result.get("set_pieces", []),
    )


# WHY source profiling gets picked from ONE representative window
# instead of, say, averaged across all of them: source_type and
# classification_confidence describe the FOOTAGE (camera angle, motion,
# occlusion), not any individual window's events -- every window of the
# same match shares the same footage source, so in a real pipeline this
# would run ONCE per match, not once per window. Our window-level graph
# runs it per window anyway (a training simplification), so at
# aggregation time we treat the first successfully-classified window's
# reading as representative of the whole match, and say so explicitly
# rather than silently averaging confidences into a number nobody
# actually computed.
def _pick_match_source_profile(results: List[WindowResult]) -> Optional[SourceProfile]:
    for r in results:
        if r.source_profile is not None:
            return r.source_profile
    return None


def run_match_pipeline(match_dir: str, window_specs: List[WindowSpec]) -> List[WindowResult]:
    """
    Run every window in window_specs through the real window-level
    graph, then write the match-level files that are ACTUALLY derivable
    from those runs: window_plan.json, running_summary.json,
    source_profile.json.

    A window that raises during processing is recorded as a data-gap
    window and the run continues -- one bad window must not take down
    the whole match. This is the actual new idea in this step: a
    fan-in over multiple independent runs has to tolerate partial
    failure, which a single per-window test never had to think about.
    """
    os.makedirs(match_dir, exist_ok=True)

    successful: List[WindowResult] = []
    data_gap_windows: List[str] = []

    for spec in window_specs:
        try:
            successful.append(run_window(spec))
        except Exception as e:
            # WHY we catch Exception broadly here, not a narrower type:
            # this boundary's job is "don't let one window's failure
            # take the whole match down," and a window can fail for
            # many different reasons (a bad Pydantic validation inside
            # a node, a KeyError from malformed stub data, anything).
            # The specific failure gets printed for visibility; the
            # match-level file just needs to know THAT it failed.
            print(f"  [run_match_pipeline] window {spec.window_id} failed: "
                  f"{type(e).__name__}: {e}")
            data_gap_windows.append(spec.window_id)

    window_plan = {"total_windows": len(window_specs)}
    with open(os.path.join(match_dir, "window_plan.json"), "w", encoding="utf-8") as f:
        json.dump(window_plan, f, indent=2)

    running_summary = {
        "windows_complete": len(successful),
        "data_gap_windows": data_gap_windows,
    }
    with open(os.path.join(match_dir, "running_summary.json"), "w", encoding="utf-8") as f:
        json.dump(running_summary, f, indent=2)

    match_source_profile = _pick_match_source_profile(successful)
    source_profile_doc = (
        match_source_profile.model_dump() if match_source_profile is not None else {}
    )
    with open(os.path.join(match_dir, "source_profile.json"), "w", encoding="utf-8") as f:
        json.dump(source_profile_doc, f, indent=2)

    print(f"  [run_match_pipeline] {len(successful)}/{len(window_specs)} windows complete"
          + (f", data gaps: {data_gap_windows}" if data_gap_windows else ""))

    return successful
