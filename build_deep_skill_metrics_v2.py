"""
build_deep_skill_metrics_v2.py

Drop-in replacement for deep_skill_metrics.build_deep_skill_metrics().
Computes the v2 metrics layer using the new v2 schema fields:
    - vertical_progression from pass sequences
    - between_lines from individual observations
    - avg_m_approx / line_width_m_approx / space_behind_m from defensive_line
    - post_duel_outcome from duels

Public function:
    build_deep_skill_metrics(match_dir, focus_team, confidence_level=2)

Confidence inheritance (unchanged from existing skill):
    repeated_pattern evidence  -> cap at 0.75
    suggestive evidence        -> cap at 0.4
    required family downgraded -> reduce by 0.2
    < 3 windows contributing   -> reduce by 0.15
    source = unknown           -> cap at 0.5
    source = veo_ball_tracking -> shape/spacing/territory metrics downgraded
    source = broadcast_tv      -> all structural metrics downgraded

Output: deep_skill_metrics.json
"""

import json
import os
import statistics
from collections import Counter


# ----- Confidence helpers ----------------------------------------------------

def _apply_confidence_caps(confidence, evidence_tier, source_type,
                           required_family_downgraded, windows_contributing):
    """Apply the standard cap/reduction rules."""
    c = confidence

    if evidence_tier == "suggestive":
        c = min(c, 0.4)
    elif evidence_tier == "repeated_pattern":
        c = min(c, 0.75)

    if required_family_downgraded:
        c = max(0, c - 0.2)

    if windows_contributing is not None and windows_contributing < 3:
        c = max(0, c - 0.15)

    if source_type == "unknown":
        c = min(c, 0.5)

    return round(c, 2)


def _is_structural_metric(metric_name):
    return metric_name in {
        "compactness_score",
        "compactness_geometry_score",
        "line_height_range",
        "width_usage_score",
        "halfspace_occupation_score",
        "rest_defence_security_score",
        "pattern_reliability_score",
        "build_up_route_diversity",
    }


def _source_caps_for_metric(metric_name, source_type):
    """
    Returns (downgraded_by_source, cap_value, source_note).
    """
    if source_type == "broadcast_tv" and _is_structural_metric(metric_name):
        return True, None, "Broadcast framing limits whole-team structural readings."
    if source_type == "veo_ball_tracking" and metric_name in {
        "compactness_score", "compactness_geometry_score",
        "width_usage_score", "halfspace_occupation_score",
        "rest_defence_security_score",
    }:
        return True, 0.6, "Ball-follow source -- structural readings zone-limited."
    return False, None, None


# ----- Metric builders -------------------------------------------------------

def _metric_compactness_geometry(summary, source_type):
    """
    NEW metric: combines line height (m) and line width (m).
    Compact = narrow line + lower height (deep block).
    Stretched = wide line + high height (advanced press).
    Returns a profile, not a single 0-1 score.
    """
    rows = summary.get("line_height_m_by_window", [])
    heights = [r["avg_m_approx"] for r in rows if r.get("avg_m_approx") is not None]
    widths = [r["line_width_m_approx"] for r in rows if r.get("line_width_m_approx") is not None]
    behinds = [r["space_behind_m"] for r in rows if r.get("space_behind_m") is not None]

    if not heights or not widths:
        return _unavailable("compactness_geometry_score",
                            "Insufficient metres-anchored line data")

    avg_height = statistics.mean(heights)
    avg_width = statistics.mean(widths)
    avg_behind = statistics.mean(behinds) if behinds else None

    downgraded, _, note = _source_caps_for_metric(
        "compactness_geometry_score", source_type
    )

    return {
        "metric_name":   "compactness_geometry_score",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "avg_line_height_m":    round(avg_height, 1),
            "avg_line_width_m":     round(avg_width, 1),
            "avg_space_behind_m":   round(avg_behind, 1) if avg_behind else None,
            "windows_contributing": len(heights),
        },
        "value_type":              "profile",
        "supporting_result_families": ["shape"],
        "evidence_tier":           "direct",
        "confidence":              _apply_confidence_caps(
                                       0.85, "direct", source_type,
                                       downgraded, len(heights)),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    len(heights),
        "fps_context":             "1fps observation",
        "source_limitations":      note,
        "calculation_basis": (
            "Average line height in metres, line width in metres, and space "
            "behind defensive line across all windows. Cross-window means."
        ),
        "traceable_to":            ["line_height_m_by_window"],
    }


def _metric_line_height_range(summary, source_type):
    """
    Returns BOTH pct range and metres range (v2).
    """
    rows_pct = summary.get("line_height_by_window", [])
    rows_m = summary.get("line_height_m_by_window", [])

    pct_values = [r["avg_pct"] for r in rows_pct if r.get("avg_pct") is not None]
    m_values = [r["avg_m_approx"] for r in rows_m if r.get("avg_m_approx") is not None]

    if not pct_values and not m_values:
        return _unavailable("line_height_range", "No line height data")

    downgraded, _, note = _source_caps_for_metric("line_height_range", source_type)

    return {
        "metric_name":   "line_height_range",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "range_pct":        (
                round(max(pct_values) - min(pct_values), 1) if pct_values else None
            ),
            "range_m":          (
                round(max(m_values) - min(m_values), 1) if m_values else None
            ),
            "min_pct":          round(min(pct_values), 1) if pct_values else None,
            "max_pct":          round(max(pct_values), 1) if pct_values else None,
            "min_m_approx":     round(min(m_values), 1) if m_values else None,
            "max_m_approx":     round(max(m_values), 1) if m_values else None,
        },
        "value_type":              "profile",
        "supporting_result_families": ["shape"],
        "evidence_tier":           "direct",
        "confidence":              _apply_confidence_caps(
                                       0.85, "direct", source_type,
                                       downgraded, len(pct_values or m_values)),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    max(len(pct_values), len(m_values)),
        "fps_context":             "1fps observation",
        "source_limitations":      note,
        "calculation_basis":       "Max minus min of per-window avg line height "
                                    "in both pct and m.",
        "traceable_to":            ["line_height_by_window", "line_height_m_by_window"],
    }


def _metric_build_up_effectiveness(summary, source_type):
    """
    v2: uses vertical_progression directly. Numerator = sequences that
    actually progressed forward. Denominator = sequences starting in defending
    or middle thirds.
    """
    totals = summary.get("vertical_progression_totals", {})

    progressed = (
        totals.get("defending_to_middle", 0)
        + totals.get("middle_to_attacking", 0)
        + totals.get("defending_to_attacking", 0)
    )
    regressed = (
        totals.get("regression_middle_to_defending", 0)
        + totals.get("regression_attacking_to_middle", 0)
        + totals.get("regression_attacking_to_defending", 0)
    )
    same = totals.get("same_third", 0)
    total_known = progressed + regressed + same

    if total_known < 3:
        return _unavailable("build_up_effectiveness_score",
                            "Fewer than 3 sequences with known vertical progression")

    rate = progressed / total_known if total_known else 0

    # Threat conversion: sequences with vertical_progression that ended in
    # shot or cross. Read this from pass_sequences if you have access to the
    # accumulated file; for now compute from running summary's recorded shots.
    shots_from_progression = len([
        s for s in summary.get("shots_for", []) if s.get("sequence_start_zone") == "defending_third"
    ])

    downgraded, _, note = _source_caps_for_metric(
        "build_up_effectiveness_score", source_type
    )

    return {
        "metric_name":   "build_up_effectiveness_score",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "progressive_rate":          round(rate, 2),
            "progressed_count":          progressed,
            "regressed_count":           regressed,
            "same_third_count":          same,
            "total_known":               total_known,
            "shots_from_defending_third":shots_from_progression,
        },
        "value_type":              "profile",
        "supporting_result_families": ["build_up"],
        "evidence_tier":           "direct",
        "confidence":              _apply_confidence_caps(
                                       0.8, "direct", source_type,
                                       downgraded, total_known),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps observation",
        "source_limitations":      note,
        "calculation_basis": (
            "Sequences with vertical_progression in {defending_to_middle, "
            "middle_to_attacking, defending_to_attacking} divided by "
            "total sequences with known vertical progression."
        ),
        "traceable_to":            ["vertical_progression_totals"],
    }


def _metric_defensive_third_turnover_rate(summary, source_type):
    """NEW v2 metric -- previously impossible to compute."""
    turnovers = summary.get("defending_third_turnover_count", 0)
    total = summary.get("defending_third_sequence_count", 0)

    if total < 3:
        return _unavailable("defensive_third_turnover_rate",
                            "Fewer than 3 sequences starting in defending third")

    rate = turnovers / total

    downgraded, _, note = _source_caps_for_metric(
        "build_up_effectiveness_score", source_type  # similar gate
    )

    return {
        "metric_name":   "defensive_third_turnover_rate",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "rate":                       round(rate, 2),
            "turnovers":                  turnovers,
            "total_defending_third_seqs": total,
        },
        "value_type":              "profile",
        "supporting_result_families": ["build_up", "transitions"],
        "evidence_tier":           "direct",
        "confidence":              _apply_confidence_caps(
                                       0.8, "direct", source_type,
                                       downgraded, total),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps observation",
        "source_limitations":      note,
        "calculation_basis": (
            "Sequences with zone_start.vertical_third = defending AND outcome "
            "in {lost_possession, clearance} divided by total sequences "
            "starting in defending third."
        ),
        "traceable_to":            [
            "defending_third_turnover_count",
            "defending_third_sequence_count",
        ],
    }


def _metric_between_lines_receiving_rate(summary, source_type):
    """
    NEW v2 per-player metric. For each player observed receiving, what
    fraction of those observations were between lines?

    Returns a profile dict keyed by player. Players with fewer than 3
    receiving observations are marked severely_limited.
    """
    receiving_categories = {
        "ball_carrying", "distribution", "hold_up_play",
        "receiving_orientation", "first_touch_direction",
    }

    per_player = {}
    for obs in summary.get("individual_observations", []):
        cat = obs.get("action_category")
        if cat not in receiving_categories:
            continue
        player = obs.get("player")
        if not player:
            continue
        rec = per_player.setdefault(player, {
            "team":             obs.get("team"),
            "position":         obs.get("position"),
            "total_receiving":  0,
            "between_lines":    0,
            "between_def_mid":  0,
            "between_mid_fwd":  0,
            "between_fb_cb":    0,
        })
        rec["total_receiving"] += 1
        zone = obs.get("zone", {}) or {}
        bl = zone.get("between_lines")
        if bl in {"between_def_mid", "between_mid_fwd", "between_fb_cb"}:
            rec["between_lines"] += 1
            rec[bl] += 1

    profiles = {}
    for player, rec in per_player.items():
        if rec["total_receiving"] == 0:
            continue
        rate = rec["between_lines"] / rec["total_receiving"]
        profiles[player] = {
            **rec,
            "rate":              round(rate, 2),
            "severely_limited":  rec["total_receiving"] < 3,
        }

    if not profiles:
        return _unavailable(
            "between_lines_receiving_rate",
            "No receiving observations with between_lines data"
        )

    downgraded, _, note = _source_caps_for_metric(
        "build_up_effectiveness_score", source_type
    )

    return {
        "metric_name":   "between_lines_receiving_rate",
        "analysis_scope":"player",
        "subject_team":  "both",
        "value":         profiles,
        "value_type":    "profile",
        "supporting_result_families": [
            "player_role", "player_positioning", "player_movement",
        ],
        "evidence_tier":           "repeated_pattern",
        "confidence":              _apply_confidence_caps(
                                       0.7, "repeated_pattern", source_type,
                                       downgraded, len(profiles)),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps observation; 3fps player-action confirmation if escalated",
        "source_limitations":      note,
        "calculation_basis": (
            "Per-player: observations with between_lines != null among "
            "receiving-category observations, divided by total receiving "
            "observations for that player."
        ),
        "traceable_to":            ["individual_observations", "between_lines_events"],
    }


def _metric_pattern_reliability(summary, source_type):
    """
    Updated v2 -- richer route codes now distinguish halfspace_left -> central
    from wide_left -> central. Reads pass_sequences indirectly via the
    vertical_progression_totals or per-window counts.
    """
    counts = summary.get("vertical_progression_totals", {})
    total = sum(counts.values())

    if total < 3:
        return _unavailable("pattern_reliability_score",
                            "Insufficient sequences with vertical_progression")

    most_common = max(counts.values()) if counts else 0
    score = most_common / total if total else 0

    downgraded, _, note = _source_caps_for_metric(
        "pattern_reliability_score", source_type
    )

    return {
        "metric_name":   "pattern_reliability_score",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "score":            round(score, 2),
            "most_common":      max(counts, key=counts.get) if counts else None,
            "most_common_n":    most_common,
            "total":            total,
        },
        "value_type":              "profile",
        "supporting_result_families": ["build_up", "phase"],
        "evidence_tier":           "repeated_pattern",
        "confidence":              _apply_confidence_caps(
                                       0.7, "repeated_pattern", source_type,
                                       downgraded, summary.get("windows_complete", 0)),
        "result_family_status":    "downgraded" if downgraded else "allowed",
        "severely_limited":        downgraded,
        "limitation_note":         note,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps observation",
        "source_limitations":      note,
        "calculation_basis": (
            "Frequency of the most common vertical_progression type divided "
            "by total sequences with known vertical progression."
        ),
        "traceable_to":            ["vertical_progression_totals"],
    }


def _metric_duel_effectiveness(summary, source_type):
    """
    NEW v2 metric -- uses post_duel_outcome to distinguish "wins duels"
    from "wins duels and retains."
    """
    duels = summary.get("duels", [])
    if not duels:
        return _unavailable("duel_effectiveness",
                            "No duels logged with post_duel_outcome")

    per_player = {}
    for d in duels:
        for player in d.get("players_visible", []):
            rec = per_player.setdefault(player, {
                "total":              0,
                "won":                0,
                "retained":           0,
                "lost_to_second":     0,
                "won_free_kick":      0,
            })
            rec["total"] += 1
            winner = d.get("winner")
            won_kit = "home_kit" in player if winner == "home_kit" else "away_kit" in player
            # Crude: assume player kit substring matches winner
            if (winner == "home_kit" and "home_kit" in player) or \
               (winner == "away_kit" and "away_kit" in player):
                rec["won"] += 1
                outcome = d.get("post_duel_outcome")
                if outcome == "retained_possession":
                    rec["retained"] += 1
                elif outcome == "lost_to_second_ball":
                    rec["lost_to_second"] += 1
                elif outcome == "free_kick_won":
                    rec["won_free_kick"] += 1

    profiles = {}
    for p, rec in per_player.items():
        if rec["total"] == 0:
            continue
        win_rate = rec["won"] / rec["total"]
        retention = rec["retained"] / rec["won"] if rec["won"] else 0
        profiles[p] = {
            **rec,
            "win_rate":         round(win_rate, 2),
            "retention_rate":   round(retention, 2),
            "severely_limited": rec["total"] < 3,
        }

    if not profiles:
        return _unavailable("duel_effectiveness", "No usable duel data")

    return {
        "metric_name":   "duel_effectiveness",
        "analysis_scope":"player",
        "subject_team":  "both",
        "value":         profiles,
        "value_type":    "profile",
        "supporting_result_families": ["player_duels", "local_duels"],
        "evidence_tier":           "direct",
        "confidence":              _apply_confidence_caps(
                                       0.7, "direct", source_type,
                                       False, len(profiles)),
        "result_family_status":    "allowed",
        "severely_limited":        False,
        "limitation_note":         None,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps duel logging; 5fps confirmation if escalated",
        "source_limitations":      None,
        "calculation_basis": (
            "Per-player: won duels and post_duel_outcome distribution. "
            "Retention rate = retained_possession outcomes / total wins."
        ),
        "traceable_to":            ["duels"],
    }


def _metric_watch_list_summary(summary):
    """
    NEW v2 -- summarises watch_list_confirmations across all windows.
    """
    confirmations = summary.get("watch_list_confirmations", [])
    if not confirmations:
        return _unavailable("watch_list_summary",
                            "No watch list entries to confirm")

    by_id = {}
    for c in confirmations:
        wid = c.get("watch_list_id")
        if not wid:
            continue
        rec = by_id.setdefault(wid, {
            "confirmed":              0,
            "refuted":                0,
            "not_observed":           0,
            "windows": [],
        })
        status = c.get("status")
        if status == "confirmed":
            rec["confirmed"] += 1
        elif status == "refuted":
            rec["refuted"] += 1
        elif status == "not_observed_this_window":
            rec["not_observed"] += 1
        rec["windows"].append({
            "window": c.get("window"),
            "status": status,
            "notes":  c.get("notes"),
        })

    # Final per-item verdict
    verdicts = {}
    for wid, rec in by_id.items():
        if rec["confirmed"] >= 2 and rec["confirmed"] > rec["refuted"]:
            verdict = "confirmed"
        elif rec["refuted"] >= 2 and rec["refuted"] > rec["confirmed"]:
            verdict = "refuted"
        elif rec["confirmed"] > 0 and rec["refuted"] == 0:
            verdict = "tentatively_confirmed"
        elif rec["refuted"] > 0 and rec["confirmed"] == 0:
            verdict = "tentatively_refuted"
        else:
            verdict = "inconclusive"
        verdicts[wid] = {**rec, "verdict": verdict}

    return {
        "metric_name":   "watch_list_summary",
        "analysis_scope":"opposition",
        "subject_team":  "opposition",
        "value":         verdicts,
        "value_type":    "profile",
        "supporting_result_families": ["opposition_identity", "opposition_patterns"],
        "evidence_tier":           "repeated_pattern",
        "confidence":              0.8,
        "result_family_status":    "allowed",
        "severely_limited":        False,
        "limitation_note":         None,
        "windows_contributing":    summary.get("windows_complete", 0),
        "fps_context":             "1fps observation",
        "source_limitations":      None,
        "calculation_basis": (
            "Per watch_list item: verdict from per-window confirmations. "
            "Confirmed if 2+ windows confirm. Refuted if 2+ refute. "
            "Tentatively_* if only one direction observed. Inconclusive otherwise."
        ),
        "traceable_to":            ["watch_list_confirmations"],
    }


def _unavailable(name, reason):
    return {
        "metric_name":      name,
        "value":            None,
        "value_type":       "unavailable",
        "confidence":       0.0,
        "severely_limited": True,
        "limitation_note":  reason,
    }


# ----- Public entrypoint -----------------------------------------------------

def build_deep_skill_metrics(match_dir, focus_team, confidence_level=2):
    """
    Reads:
        running_summary.json
        source_profile.json
        match_config.json

    Writes:
        deep_skill_metrics.json
    """
    summary_path = os.path.join(match_dir, "running_summary.json")
    profile_path = os.path.join(match_dir, "source_profile.json")

    with open(summary_path) as f:
        summary = json.load(f)
    with open(profile_path) as f:
        source_profile = json.load(f)
    source_type = source_profile.get("source_type", "unknown")

    metrics = [
        _metric_compactness_geometry(summary, source_type),
        _metric_line_height_range(summary, source_type),
        _metric_build_up_effectiveness(summary, source_type),
        _metric_defensive_third_turnover_rate(summary, source_type),
        _metric_between_lines_receiving_rate(summary, source_type),
        _metric_pattern_reliability(summary, source_type),
        _metric_duel_effectiveness(summary, source_type),
        _metric_watch_list_summary(summary),
    ]

    available = [m for m in metrics if m.get("value_type") != "unavailable"]
    unavailable = [m for m in metrics if m.get("value_type") == "unavailable"]

    avg_confidence = (
        round(statistics.mean([m["confidence"] for m in available]), 2)
        if available else 0.0
    )

    low_confidence_metrics = [
        m["metric_name"] for m in available
        if m["confidence"] < 0.5
    ]

    output = {
        "focus_team":                focus_team,
        "confidence_level_applied":  confidence_level,
        "source_type":               source_type,
        "metrics":                   available,
        "unavailable_metrics":       [m["metric_name"] for m in unavailable],
        "unavailable_reasons": {
            m["metric_name"]: m.get("limitation_note") for m in unavailable
        },
        "avg_confidence":            avg_confidence,
        "low_confidence_metrics":    low_confidence_metrics,
    }

    out_path = os.path.join(match_dir, "deep_skill_metrics.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[deep_skill_metrics_v2] Wrote {out_path}")
    print(f"  Available metrics:   {len(available)}")
    print(f"  Unavailable metrics: {len(unavailable)} ({output['unavailable_metrics']})")
    print(f"  Avg confidence:      {avg_confidence}")
    if low_confidence_metrics:
        print(f"  Low-confidence:      {low_confidence_metrics}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python build_deep_skill_metrics_v2.py [MATCH_DIR] [FOCUS_TEAM] [CONFIDENCE_LEVEL=2]")
        sys.exit(1)
    level = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    build_deep_skill_metrics(sys.argv[1], sys.argv[2], level)
