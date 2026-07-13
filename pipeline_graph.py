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

from typing import List, Optional, Literal
from datetime import datetime, timezone
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

    # WHY these four "agent_a_*" / "agent_b_*" fields exist even though
    # they never appear in a final report: not every state field has to
    # be a finished, user-facing result. These are SCRATCH fields --
    # their only job is letting agent_a_scan_node and agent_b_scan_node
    # each write down their own independent reading, so merge_dual_agents_node
    # has something to compare a moment later. Once merged, `formation`
    # and `set_pieces` below are the only fields anything downstream
    # (burst_scan, reports) should ever read.
    agent_a_formation: Optional[Formation] = None
    agent_b_formation: Optional[Formation] = None
    agent_a_set_pieces: List[SetPiece] = []
    agent_b_set_pieces: List[SetPiece] = []

    # WHY a plain list of strings, mirroring merge_utils.py's own
    # `review_required` list almost exactly: when two independent
    # readings disagree and we don't have (or don't trust) a rule to
    # pick a winner, the honest move is to surface it for a human to
    # look at -- not silently discard one agent's reading. This list is
    # that surfaced record.
    disagreements: List[str] = []

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
# WHY there are TWO stub functions now, not one: they stand in for two
# genuinely independent LLM calls over the same window -- same frames,
# same prompt, but a fresh model context each time, the same way two
# human analysts watching the same replay can walk away with slightly
# different reads. To make that realistic (and to give us something
# real to reconcile), these stubs deliberately disagree in three ways
# that mirror real disagreement patterns:
#   1. Home formation: "4-3-3" vs "4-2-3-1" -- a genuine categorical
#      disagreement, the kind merge_categorical() exists to resolve.
#   2. The corner at 7m 02s: both agents saw it (this is the "agreed
#      existence, disagreed detail" case) but disagree on marking.
#   3. The throw-in at 12m 15s: Agent B saw it, Agent A didn't at all --
#      a "seen by one agent only" case, the same shape as
#      merge_dual_agents()'s partial_a_only/partial_b_only key-moment logic.
def _stub_agent_a_response(window_id: str) -> dict:
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


def _stub_agent_b_response(window_id: str) -> dict:
    return {
        "formation": {
            "home": "4-2-3-1",   # disagrees with Agent A on home shape
            "away": "4-4-2",     # agrees with Agent A
            "home_formation_basis": "confirmed_from_frames",
            "away_formation_basis": "confirmed_from_frames",
        },
        "set_pieces": [
            {
                "timestamp": "7m 02s",          # same set piece as Agent A...
                "type": "corner_left",
                "team": "home_kit",
                "delivery_zone": "near_post",
                "marking": "man_to_man",         # ...but disagrees on marking
                "outcome": "cleared_near_post",
            },
            {
                "timestamp": "12m 15s",          # ...and saw one Agent A missed entirely
                "type": "throw_in",
                "team": "away_kit",
                "delivery_zone": "attacking_third",
                "marking": None,
                "outcome": "retained",
            },
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


def agent_a_scan_node(state: PipelineState) -> dict:
    print(f"  [agent_a_scan] window {state.window_id}: running Agent A's 1fps scan (stubbed)")
    raw = _stub_agent_a_response(state.window_id)
    return {
        "agent_a_formation": _parse_formation(raw),
        "agent_a_set_pieces": _parse_set_pieces(raw, state.window_id),
    }


def agent_b_scan_node(state: PipelineState) -> dict:
    print(f"  [agent_b_scan] window {state.window_id}: running Agent B's 1fps scan (stubbed)")
    raw = _stub_agent_b_response(state.window_id)
    return {
        "agent_b_formation": _parse_formation(raw),
        "agent_b_set_pieces": _parse_set_pieces(raw, state.window_id),
    }


# --- Dual-agent merge -----------------------------------------------------
# WHY this port of merge_categorical keeps the SAME "prefer longer
# string" rule as merge_utils.py, rather than something smarter: this
# is a deliberate, honest port, not an improvement. "Longer string wins"
# is a genuinely crude heuristic -- "4-2-3-1" beats "4-3-3" just because
# it has more characters, with no actual tactical reasoning behind it.
# It happens to work often enough (more detailed formations tend to have
# longer names) but it's worth naming as a known weak spot, the same way
# we flagged F1/F4/F5 earlier, rather than pretending it's more
# principled than it is. A real improvement here would weigh each
# agent's own confidence score, which the raw schema does not currently
# give us per-field.
def _merge_categorical(a, b) -> tuple:
    """Returns (merged_value, agreed: bool)."""
    if a == b:
        return a, True
    if a is None:
        return b, True
    if b is None:
        return a, True
    return (a if len(str(a)) >= len(str(b)) else b), False


# WHY formation and set_pieces use TWO DIFFERENT strategies here, not
# one shared "merge everything the same way" function: they are
# different KINDS of disagreement. Formation is one value per side --
# there can only be one "true" home formation, so disagreement forces a
# pick, which is exactly what _merge_categorical's heuristic is for. Set
# pieces are a LIST of discrete events -- there's no need to "pick one"
# between two agents who both saw the same corner AND a throw-in the
# other missed; both are real, so both belong in the merged list. The
# only real decision for set pieces is de-duplication (same event, seen
# twice) versus genuine addition (a new event, seen once). Applying
# _merge_categorical's pick-a-winner logic to a list would be the wrong
# tool for this job.
def merge_dual_agents_node(state: PipelineState) -> dict:
    disagreements = list(state.disagreements)

    # -- formation: one value per side, must pick a winner on disagreement --
    a_home = state.agent_a_formation.home.in_possession
    b_home = state.agent_b_formation.home.in_possession
    a_away = state.agent_a_formation.away.in_possession
    b_away = state.agent_b_formation.away.in_possession

    home_value, home_agreed = _merge_categorical(a_home, b_home)
    away_value, away_agreed = _merge_categorical(a_away, b_away)

    if not home_agreed:
        disagreements.append(
            f"window {state.window_id}: home formation -- "
            f"A={a_home!r}, B={b_home!r} -> used {home_value!r} (longer-string heuristic)"
        )
    if not away_agreed:
        disagreements.append(
            f"window {state.window_id}: away formation -- "
            f"A={a_away!r}, B={b_away!r} -> used {away_value!r} (longer-string heuristic)"
        )

    merged_formation = Formation(
        home=TeamFormation(
            in_possession=home_value,
            basis="dual_agent_agreed" if home_agreed else "dual_agent_resolved",
        ),
        away=TeamFormation(
            in_possession=away_value,
            basis="dual_agent_agreed" if away_agreed else "dual_agent_resolved",
        ),
    )

    # -- set pieces: a list, so the question is dedupe vs. genuine addition --
    # WHY keyed by timestamp alone, not (timestamp, type): a real
    # disagreement about WHAT happened at a given moment (one agent
    # calls it a corner, the other a throw-in) is a more serious
    # conflict than a marking-system mismatch, and we want that surfaced
    # too, not silently treated as "two different events" just because
    # the type field differs.
    by_timestamp: dict = {}
    for sp in state.agent_a_set_pieces:
        by_timestamp[sp.timestamp] = {"a": sp, "b": None}
    for sp in state.agent_b_set_pieces:
        by_timestamp.setdefault(sp.timestamp, {"a": None, "b": None})
        by_timestamp[sp.timestamp]["b"] = sp

    merged_set_pieces = []
    for ts, pair in by_timestamp.items():
        sp_a, sp_b = pair["a"], pair["b"]
        if sp_a and sp_b:
            # Seen by both -- confirmed to exist. WHY we still don't try
            # to pick a "best" reading field-by-field here (unlike
            # formation): SetPiece has many fields (marking_system,
            # delivery_zone, outcome, ...) and blindly running
            # _merge_categorical on each would multiply disagreement
            # noise for very little gain, when burst_scan is about to
            # re-observe this exact set piece at 5fps anyway (see
            # ALWAYS_ESCALATE_TYPES) and get a much more reliable answer
            # than either 1fps agent could give.
            merged_set_pieces.append(sp_a)
            if sp_a.marking_system != sp_b.marking_system:
                disagreements.append(
                    f"window {state.window_id}: set piece at {ts} -- "
                    f"marking A={sp_a.marking_system!r}, B={sp_b.marking_system!r} "
                    f"-> kept A's reading (burst_scan will re-observe this anyway)"
                )
        elif sp_a:
            merged_set_pieces.append(sp_a)
            disagreements.append(
                f"window {state.window_id}: set piece at {ts} ({sp_a.type}) "
                f"seen by Agent A only"
            )
        else:
            merged_set_pieces.append(sp_b)
            disagreements.append(
                f"window {state.window_id}: set piece at {ts} ({sp_b.type}) "
                f"seen by Agent B only"
            )

    print(f"  [merge_dual_agents] window {state.window_id}: "
          f"{len(disagreements) - len(state.disagreements)} new disagreement(s) logged")

    return {
        "formation": merged_formation,
        "set_pieces": merged_set_pieces,
        "disagreements": disagreements,
    }


# --- Escalation routing --------------------------------------------------
# WHY this is a *conditional* edge, the first one in this graph: every
# edge so far has always gone to the same next node. Escalation needs
# to ask a question first -- "does this window have anything that needs
# a closer 5fps look?" -- and go to a DIFFERENT node depending on the
# answer. A conditional edge is a function that looks at the current
# state and returns a label; a path_map then says which node each label
# leads to.
#
# WHY this mirrors escalation_router.py's rule exactly, with no
# confidence threshold: ALWAYS_ESCALATE_TYPES in that file includes
# "set_piece_delivery" with no confidence check at all -- unlike, say,
# "pressing", which only escalates on genuine uncertainty. Set pieces
# always escalate because 1fps sampling (one frame per second) simply
# cannot capture fast-moving detail like runners or wall shape, at any
# confidence level. So the routing question here isn't "how confident
# are we" -- it's just "does an unresolved set piece exist at all."
def route_after_merge(state: PipelineState) -> Literal["burst_scan", "skip"]:
    if any(not sp.burst_resolved for sp in state.set_pieces):
        return "burst_scan"
    return "skip"


# WHY this stub only fills fields that 1fps genuinely cannot see
# (runners, delivery_type, bodies_in_box) rather than re-guessing
# fields tier1_scan already set (timestamp, type, team): a real 5fps
# burst re-watch CONFIRMS or CORRECTS the 1fps read on some fields and
# FILLS IN others that were never observable at 1fps at all (see the
# "1fps confirmation fields" vs "1fps could not read them" split in
# pipeline_runner_v2.py's actual burst prompt). This stub keeps that
# same split, rather than overwriting everything indiscriminately.
def _stub_burst_llm_response() -> dict:
    return {
        "runners":       ["CB1 near post", "ST1 far post"],
        "delivery_type": "inswinger",
        "bodies_in_box": 6,
    }


# WHY model_copy(update={...}) instead of mutating the SetPiece object
# directly: it produces a new, independently-validated object rather
# than reaching into an existing one and changing fields in place. This
# matters here specifically because burst_resolved/source/resolved_at
# all need to change together, atomically, as one clear "this record
# just got upgraded" event -- not several separate in-place edits that
# could, in principle, be interrupted halfway through.
def burst_scan_node(state: PipelineState) -> dict:
    unresolved = [sp for sp in state.set_pieces if not sp.burst_resolved]
    print(f"  [burst_scan] window {state.window_id}: escalating "
          f"{len(unresolved)} unresolved set piece(s) to 5fps (stubbed)")

    updated = []
    for sp in state.set_pieces:
        if sp.burst_resolved:
            updated.append(sp)
            continue
        enrichment = _stub_burst_llm_response()
        updated.append(sp.model_copy(update={
            **enrichment,
            "source":         "5fps_burst",
            "burst_resolved": True,
            "burst_fps":      5,
            "resolved_at":    datetime.now(timezone.utc).isoformat(),
        }))
    return {"set_pieces": updated}


def build_graph():
    graph = StateGraph(PipelineState)

    graph.add_node("profile_source", profile_source_node)
    graph.add_node("agent_a_scan", agent_a_scan_node)
    graph.add_node("agent_b_scan", agent_b_scan_node)
    graph.add_node("merge_dual_agents", merge_dual_agents_node)
    graph.add_node("burst_scan", burst_scan_node)

    graph.add_edge(START, "profile_source")

    # WHY profile_source -> agent_a_scan -> agent_b_scan -> merge_dual_agents
    # runs the two agents ONE AFTER ANOTHER rather than truly in parallel:
    # LangGraph does support parallel fan-out (multiple edges from one
    # node running concurrently, then converging back into a fan-in
    # node), and that WOULD be the right optimization here since Agent A
    # and Agent B don't depend on each other's output at all. We're
    # deliberately not reaching for that yet -- the reconciliation logic
    # in merge_dual_agents_node is the new concept this step, and
    # sequential edges keep that the only new moving part. Parallel
    # fan-out is a genuine follow-up once this is solid, not a
    # correctness requirement now.
    graph.add_edge("profile_source", "agent_a_scan")
    graph.add_edge("agent_a_scan", "agent_b_scan")
    graph.add_edge("agent_b_scan", "merge_dual_agents")

    # WHY the conditional edge now attaches to "merge_dual_agents"
    # instead of "tier1_scan": escalation routing reads state.set_pieces,
    # and that field is only ever written once, by the merge node -- the
    # two agent-scan nodes only ever write to their own agent_a_/agent_b_
    # scratch fields. Escalation decisions must be made on the
    # RECONCILED set of set pieces, not either agent's raw, possibly
    # incomplete list.
    graph.add_conditional_edges(
        "merge_dual_agents",
        route_after_merge,
        {"burst_scan": "burst_scan", "skip": END},
    )
    graph.add_edge("burst_scan", END)

    return graph.compile()
