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

from models import SourceProfile, VisibilityScores, Formation, TeamFormation, SetPiece
from accumulator import validate_set_piece


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
# WHY this node returns raw dicts internally (a stubbed structural LLM
# response) before converting to our models, instead of stubbing typed
# objects directly like profile_source_node did: this is realistic. A
# real LLM call returns raw JSON matching STRUCTURAL_OUTPUT_SCHEMA (see
# pipeline_runner_v2.py), not a Pydantic object. Every real node in this
# graph will need a parsing/adapter step exactly like this one -- raw
# dict in, validated model out -- so it's worth seeing that step clearly
# even while the LLM call itself is stubbed.
def _stub_structural_llm_response(window_id: str) -> dict:
    return {
        "formation": {
            "home": "4-3-3",
            "away": "4-4-2",
            "home_formation_basis": "confirmed_from_frames",
            "away_formation_basis": "confirmed_from_frames",
        },
        "set_pieces": [
            {
                "timestamp": "7m 02s",
                "type": "corner_left",
                "team": "home_kit",
                "delivery_zone": "near_post",
                "marking": "zonal",   # note: raw schema still says "marking", not "marking_system"
                "outcome": "cleared_near_post",
            }
        ],
    }


# WHY the raw schema only gives us a starting point for Formation, not
# a complete one: STRUCTURAL_OUTPUT_SCHEMA asks the LLM for one "home"/
# "away" formation string plus a free-text "variation" note for when
# in-possession and out-of-possession shape differ (e.g. "defends as
# 4-5-1"). That's unstructured text, not a clean formation string --
# we can't safely auto-parse "defends as 4-5-1 -- wingers track back"
# into out_of_possession="4-5-1" without risking silently wrong data.
# So this adapter deliberately leaves out_of_possession as None for now
# and is honest about why, rather than guessing. The real fix belongs
# one level up, in the PROMPT: ask the LLM for in_possession and
# out_of_possession as two separate clean fields, the same shape our
# model already expects. That's a concrete, specific follow-up task for
# whenever you touch pipeline_runner_v2.py's prompt text.
def _parse_formation(raw: dict) -> Formation:
    f = raw.get("formation", {})
    return Formation(
        home=TeamFormation(
            in_possession=f.get("home"),
            basis=f.get("home_formation_basis"),
        ),
        away=TeamFormation(
            in_possession=f.get("away"),
            basis=f.get("away_formation_basis"),
        ),
    )


# WHY this reuses validate_set_piece() from accumulator.py instead of
# writing a second normalizer from scratch: that function already does
# exactly the field normalization we need (including get_marking()'s
# marking -> marking_system rename, and all the same defaults our
# SetPiece model uses). Reusing it means the F4 fix stays in one place.
# validate_set_piece() returns a plain dict with one extra key
# ("delivery", an alias pair with delivery_type) that our SetPiece
# model doesn't define -- Pydantic's default behaviour is to silently
# ignore fields a model doesn't recognise, so **normalized just works
# without any extra config.
def _parse_set_pieces(raw: dict, window_id: str) -> list[SetPiece]:
    parsed = []
    for sp in raw.get("set_pieces", []):
        is_valid, normalized, reason = validate_set_piece(sp, window_id)
        if not is_valid:
            print(f"  [tier1_scan] rejected a set piece: {reason}")
            continue
        parsed.append(SetPiece(**normalized))
    return parsed


def tier1_scan_node(state: PipelineState) -> dict:
    print(f"  [tier1_scan] window {state.window_id}: running 1fps structural scan (stubbed)")
    raw = _stub_structural_llm_response(state.window_id)

    formation = _parse_formation(raw)
    set_pieces = _parse_set_pieces(raw, state.window_id)

    return {"formation": formation, "set_pieces": set_pieces}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("profile_source", profile_source_node)
    graph.add_node("tier1_scan", tier1_scan_node)

    # WHY START -> profile_source -> tier1_scan -> END: this is a
    # straight-line chain for now (no branching yet) -- escalation
    # routing, the next piece, is where we introduce a CONDITIONAL edge
    # that picks between two different next nodes based on what tier1_scan
    # found, instead of always going to the same next step.
    graph.add_edge(START, "profile_source")
    graph.add_edge("profile_source", "tier1_scan")
    graph.add_edge("tier1_scan", END)

    return graph.compile()
