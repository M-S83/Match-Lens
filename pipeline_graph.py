"""
pipeline_graph.py -- LangGraph rebuild of the Match Lens pipeline.

WHY this file exists: this replaces the "write JSON to disk, glob for it
later" pattern that caused F2 (merged filename naming drift), F8, F13,
F14, F15 in AUDIT.md. Instead of separate scripts each reading/writing
their own files with their own filename conventions, every step here
is a plain function that takes the current PipelineState in, and
returns only the fields it changed. LangGraph handles moving that
state from one node to the next -- there is no file, and therefore no
filename pattern to get wrong.

This file starts with ONE node (profile_source) to prove the wiring
works end to end. More nodes (Tier 1 scan, escalation routing, merge,
synthesis) get added the same way, one at a time.
"""

from typing import List, Optional
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END

from models import SourceProfile, VisibilityScores, Formation, SetPiece


# WHY the state is a Pydantic model, not a plain dict or TypedDict (the
# two other common LangGraph state options): we already have validation
# rules on SourceProfile/Formation/SetPiece from Phase 1. Using a
# Pydantic model as the state schema means LangGraph validates the
# WHOLE state -- not just each model individually -- every time a node
# returns an update. A node that tried to write a malformed
# SourceProfile would fail immediately, at the graph level, instead of
# silently corrupting state that a much later node discovers is broken.
class PipelineState(BaseModel):
    window_id: str
    frame_paths: List[str] = []

    source_profile: Optional[SourceProfile] = None
    formation: Optional[Formation] = None
    set_pieces: List[SetPiece] = []


# WHY this node is a stub, not a real LLM call: this sandbox has no
# Anthropic API key configured, and the point of THIS step is proving
# the graph mechanics (state in -> node runs -> updated state out) work
# correctly -- not re-testing an LLM call you already have working in
# source_profiler.py. When you run this for real, replace the body of
# this function with your existing sample_frames() + LLM classification
# call from source_profiler.py; everything else in the graph is
# unaffected, because nodes only care about the shape of the state,
# never about how another node's return value was produced.
#
# WHY a node returns a dict of ONLY the fields it changed, not the
# whole state: this is how LangGraph merges results back in. Returning
# {"source_profile": ...} means "update just this field" -- returning
# the entire state object back would work too, but returning only what
# changed makes it obvious, just by reading the function, exactly what
# this node is responsible for.
def profile_source_node(state: PipelineState) -> dict:
    print(f"  [profile_source] window {state.window_id}: "
          f"examining {len(state.frame_paths)} sampled frame(s) (stubbed)")

    stub_profile = SourceProfile(
        source_type="tactical_wide_static",
        classification_confidence=0.91,
        split_aware=False,
        visibility_scores=VisibilityScores(
            full_pitch_visibility_score=0.9, weakside_visibility_score=0.7,
            off_ball_coverage_score=0.8, camera_motion_score=0.2,
            zoom_variability_score=0.1, stability_score=0.95,
            orientation_consistency_score=0.9, occlusion_score=0.1,
            ball_follow_bias=0.3,
        ),
        source_limitations_note="STUB: replace with real classification call.",
    )
    return {"source_profile": stub_profile}


# WHY build_graph() is a function, not module-level code that runs on
# import: it lets test files (and later, other scripts) get a fresh
# compiled graph on demand, and it's the natural place to keep adding
# .add_node(...) / .add_edge(...) calls as more steps are ported over --
# one function, growing one line at a time, mirrors how we added one
# Pydantic model at a time in Phase 1.
def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("profile_source", profile_source_node)

    # WHY START -> profile_source -> END, spelled out explicitly, instead
    # of just letting the single node run: with only one node the edges
    # look redundant, but this is the exact same wiring pattern we'll
    # reuse once there are 6 nodes and real branching logic -- learning
    # the explicit form now means the next node is just one more
    # add_node + add_edge call, not a new concept.
    graph.add_edge(START, "profile_source")
    graph.add_edge("profile_source", END)

    return graph.compile()
