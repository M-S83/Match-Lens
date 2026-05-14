"""
deep_skill_metrics.py -- Match Lens Step 3k

Translates aggregated findings into 20 performance-level tactical insights.

Metric set (20 total):
  System-level (5):     compactness, build_up_effectiveness, rest_defence_security,
                        pressing_intensity, line_height_range
  Behavioural (6):      width_usage, halfspace_occupation, transition_efficiency,
                        chance_creation_profile, pattern_reliability, build_up_route_diversity
  Tactical insight (4): predictability, pressing_trigger_consistency,
                        set_piece_delivery_profile, attacking_support_score
  Player (5):           role_consistency, positioning_stability, decision_profile,
                        movement_contribution, duel_effectiveness

REMOVED from previous version:
  shape_stability_score      -- always 1.0 for consistent teams, no signal
  spacing_control_score      -- proxy for line height already captured elsewhere
  weakside_utilisation_score -- requires weakside_visibility > 0.5, ball-tracking never achieves
  control_score              -- composite of composites, signal diluted twice
  dominance_score            -- meaningless without reliable shot data
  structural_integrity_score -- composite of composites, low value
  chaos_index                -- inverse of shape_stability, redundant

Reads:
  running_summary.json
  pass_sequences.json
  result_family_gates.json
  source_profile.json
  confirmation_queue.json

Writes:
  deep_skill_metrics.json
"""

import json
import os
import sys
import re
from datetime import datetime
from collections import Counter


EVIDENCE_TIER_CAP = {
    "escalated_confirmation": 1.0,
    "direct":                 0.9,
    "repeated_pattern":       0.75,
    "suggestive":             0.4,
}

STATUS_CONFIDENCE_PENALTY = {
    "allowed":    0.0,
    "downgraded": 0.2,
}

SOURCE_GLOBAL_CAP = {
    "tactical_wide_static":        1.0,
    "tactical_wide_auto_tracking": 0.85,
    "drone_high_wide":             0.85,
    "veo_ball_tracking":           0.6,
    "drone_follow":                0.6,
    "dual_panoramic":              0.7,
    "behind_goal":                 0.5,
    "broadcast_tv":                0.5,
    # Fix 39: production broadcast extractor writes "broadcast_fixed_wide",
    # which fell through to the 0.5 default and capped every Wingate /
    # Billericay metric. Set to match the highest per-family ceiling in the
    # broadcast source_profile.json (shape_analysis = 0.85).
    "broadcast_fixed_wide":        0.85,
    "tracking_overlay":            0.75,
    "unknown":                     0.5,
}

PITCH_LENGTH_M = 105.0
MIN_WINDOWS    = 3



# -- Categorisation helpers ----------------------------------------------------
# All categories use football language, not data-science language.
# The goal: a coach reads the metric and immediately understands it.

def pct_to_metres(pct):
    """Convert pitch percentage (0-100) to metres from own goal."""
    return round(pct / 100.0 * PITCH_LENGTH_M, 1) if pct is not None else None


def line_height_category(metres):
    """
    Categorise defensive line height in metres from own goal.
      own goal line   =  0m     own penalty box = 16.5m
      own third       = 35m     halfway line    = 52.5m
      opp third       = 70m     opp penalty box = 88.5m
    """
    if metres is None: return "unknown"
    if metres > 63:    return "very_high"
    if metres > 47:    return "high"
    if metres > 35:    return "medium"
    if metres > 21:    return "low"
    return "very_low"


def pressing_category(score_0_10):
    """Categorise average pressing score (0-10 scale)."""
    if score_0_10 is None: return "unknown"
    if score_0_10 >= 8:    return "intense"
    if score_0_10 >= 6:    return "high"
    if score_0_10 >= 4:    return "moderate"
    if score_0_10 >= 2:    return "low"
    return "non_existent"


def compactness_category(score):
    """Categorise compactness composite score (0-1)."""
    if score is None: return "unknown"
    if score >= 0.75: return "very_compact"
    if score >= 0.60: return "compact"
    if score >= 0.45: return "moderate"
    if score >= 0.30: return "stretched"
    return "disorganised"


def rest_defence_category(shifts_per_window):
    """Categorise from avg shifts per window."""
    if shifts_per_window is None: return "unknown"
    if shifts_per_window < 0.5:   return "very_secure"
    if shifts_per_window < 1.0:   return "secure"
    if shifts_per_window < 2.0:   return "moderate"
    if shifts_per_window < 3.5:   return "vulnerable"
    return "very_vulnerable"


def predictability_category(score):
    """Categorise build-up predictability (0-1)."""
    if score is None: return "unknown"
    if score >= 0.75: return "highly_predictable"
    if score >= 0.60: return "predictable"
    if score >= 0.40: return "mixed"
    if score >= 0.25: return "varied"
    return "unpredictable"


def support_depth_category(avg_passes):
    """Categorise attacking support depth from avg passes before shot."""
    if avg_passes is None: return "unknown"
    if avg_passes >= 7:    return "deep_support"
    if avg_passes >= 5:    return "good_support"
    if avg_passes >= 3:    return "moderate_support"
    if avg_passes >= 1.5:  return "limited_support"
    return "isolated"


ZONE_LABELS = {
    "defending_third": "own third",
    "middle":          "midfield",
    "middle_third":    "midfield",
    "attacking_third": "final third",
    "own_half":        "own half",
    "unknown":         "unknown",
}

def football_zone(zone_code):
    return ZONE_LABELS.get(str(zone_code).lower(), str(zone_code))



# -- Prose interpretation templates --------------------------------------------
# Convert category labels into tactical sentences.
# The report writer reads prose_interpretation, not the raw label.

PRESS_PROSE = {
    "intense":      "a high-intensity press applied across most phases of play",
    "high":         "a committed press, frequently hunting the ball in opposition territory",
    "moderate":     "selective pressure applied in specific trigger situations",
    "low":          "limited pressing activity -- predominantly holding shape and allowing the opposition to circulate",
    "non_existent": "no organised pressing observed -- sitting in a passive block throughout",
}


# ============================================================
# OBSERVATION AND PROFILE GRADING
# ============================================================


# ============================================================
# NEW METRICS — Phase 4 implementation
# ============================================================

def calc_possession_balance(summary: dict) -> dict:
    """Per-window possession balance from both-teams sequence counts."""
    pb = summary.get("possession_by_window", [])
    if not pb:
        return {}
    total_windows = len(pb)
    focus_avg = round(sum((w.get("focus_pct") or 50) for w in pb) / max(total_windows, 1), 1)
    opp_avg   = round(100 - focus_avg, 1)
    windows_dominant = sum(1 for w in pb if (w.get("focus_pct") or 50) >= 55)
    return {
        "focus_avg_pct":      focus_avg,
        "opposition_avg_pct": opp_avg,
        "windows_focus_dominant": windows_dominant,
        "windows_total":          total_windows,
        "balance": ("dominant" if focus_avg >= 55 else
                    "contested" if focus_avg >= 45 else "dominated"),
    }


def calc_direct_play_ratio(summary: dict) -> dict:
    """Proportion of sequences that are long balls, and second ball win rate."""
    seqs = summary.get("pass_sequences", {}).get("sequences",
           summary.get("sequences", []))
    if not seqs:
        return {}
    total       = len(seqs)
    long_balls  = [s for s in seqs if s.get("is_long_ball")]
    lb_count    = len(long_balls)
    sb_won      = sum(1 for s in long_balls
                      if s.get("second_ball_contest") == "won")
    sb_lost     = sum(1 for s in long_balls
                      if s.get("second_ball_contest") == "lost")
    sb_total    = sb_won + sb_lost
    return {
        "long_ball_count":       lb_count,
        "long_ball_pct":         round(lb_count / max(total, 1) * 100, 1),
        "style": ("direct" if lb_count/max(total,1) > 0.25 else
                  "mixed"  if lb_count/max(total,1) > 0.12 else "possession"),
        "second_ball_win_rate":  round(sb_won / max(sb_total, 1), 2) if sb_total else None,
        "second_ball_sample":    sb_total,
    }


def calc_aerial_dominance(summary: dict) -> dict:
    """Aerial and ground duel win rate by zone from duels[] array."""
    duels = summary.get("duels", [])
    if not duels:
        return {}
    # Fix 33b: home/away naming.
    home_team = summary.get("home_team", "")
    aerials = [d for d in duels if d.get("type") == "aerial"]
    ground  = [d for d in duels if d.get("type") in ("ground", "tackle")]

    def win_rate(duel_list, team):
        won  = sum(1 for d in duel_list if d.get("winner") == team)
        total = sum(1 for d in duel_list if d.get("winner") in (team, "opposition"))
        return round(won / max(total, 1), 2) if total else None

    aerial_rate = win_rate(aerials, home_team)
    ground_rate = win_rate(ground,  home_team)
    return {
        "aerial_duels_total":    len(aerials),
        "aerial_win_rate":       aerial_rate,
        "ground_duels_total":    len(ground),
        "ground_win_rate":       ground_rate,
        "overall_duels":         len(duels),
        "duel_dominance": ("dominant" if (aerial_rate or 0) >= 0.6 else
                           "competitive" if (aerial_rate or 0) >= 0.4 else
                           "second best" if aerial_rate is not None else "no data"),
    }


def calc_gk_distribution_consistency(summary: dict) -> dict:
    """GK distribution target zone consistency from gk_kicks[] array."""
    kicks = summary.get("gk_kicks", [])
    # Fix 33b: filter by home_team rather than the legacy literal sentinel.
    home_team = summary.get("home_team", "")
    focus_kicks = [k for k in kicks if k.get("team") == home_team]
    if not focus_kicks:
        return {}
    from collections import Counter
    zones   = Counter(k.get("target_zone") for k in focus_kicks if k.get("target_zone"))
    lengths = Counter(k.get("length") for k in focus_kicks if k.get("length"))
    dominant_zone   = zones.most_common(1)[0] if zones else (None, 0)
    dominant_length = lengths.most_common(1)[0] if lengths else (None, 0)
    consistency = round(dominant_zone[1] / max(len(focus_kicks), 1), 2)
    return {
        "total_kicks":         len(focus_kicks),
        "dominant_zone":       dominant_zone[0],
        "dominant_zone_pct":   round(dominant_zone[1]/max(len(focus_kicks),1)*100, 1),
        "dominant_length":     dominant_length[0],
        "zone_consistency":    consistency,
        "consistent":          consistency >= 0.6,
        "zone_breakdown":      dict(zones),
    }


def calc_momentum_by_window(summary: dict) -> list:
    """
    Per-window momentum score from pressing, possession, and line height.
    Returns list of {window, momentum, components} dicts.
    """
    pb_win = {w["window"]: w for w in summary.get("pressing_by_window", [])
              if w.get("window")}
    lh_win = {w["window"]: w for w in summary.get("line_height_by_window", [])
              if w.get("window")}
    po_win = {w["window"]: w for w in summary.get("possession_by_window", [])
              if w.get("window")}

    windows_seen = sorted(set(list(pb_win) + list(lh_win)), key=str)
    result = []
    for win in windows_seen:
        pb  = pb_win.get(win, {})
        lh  = lh_win.get(win, {})
        po  = po_win.get(win, {})

        press     = pb.get("avg_score")
        line_pct  = lh.get("avg_pct")
        poss      = po.get("focus_pct")

        if press is None and line_pct is None:
            result.append({"window": win, "momentum": None, "components": {}})
            continue

        # Normalise each to 0-1
        press_n = min(press / 10.0, 1.0) if press is not None else 0.5
        line_n  = min(line_pct / 60.0, 1.0) if line_pct is not None else 0.5
        poss_n  = poss / 100.0 if poss is not None else 0.5

        # Weighted composite: press 35%, possession 40%, line height 25%
        weights = (0.35, 0.40, 0.25) if poss is not None else (0.55, 0.0, 0.45)
        momentum = round(press_n * weights[0] + poss_n * weights[1] + line_n * weights[2], 3)

        result.append({
            "window":   win,
            "momentum": momentum,
            "components": {
                "pressing":   round(press_n, 3),
                "possession": round(poss_n, 3) if poss is not None else None,
                "line_height":round(line_n, 3),
            },
        })
    return result


def calc_substitution_impact(summary: dict, match_config: dict) -> list:
    """
    For each substitution, compare 3-window average before vs after
    across pressing, possession, and line height.
    """
    subs = []
    for team_lineup in match_config.get("lineups", []):
        team_name = team_lineup.get("team", {}).get("name", "")
    # Get subs from match_config
    all_subs = match_config.get("substitutions", [])
    if not all_subs:
        return []

    pb_win = {w["window"]: w.get("avg_score") for w in summary.get("pressing_by_window", []) if w.get("window")}
    lh_win = {w["window"]: w.get("avg_pct")   for w in summary.get("line_height_by_window",[]) if w.get("window")}
    po_win = {w["window"]: w.get("focus_pct") for w in summary.get("possession_by_window",[]) if w.get("window")}

    # Fix 33b follow-up: defensive key=str. Window keys are normally label
    # strings ("1H_00-05min") but if any agent emits an int/None we'd hit
    # the same str-vs-int collision the previous L293 fix addressed.
    windows_ordered = sorted(pb_win.keys(), key=str)

    def window_index(win_list, target_minute):
        # Very rough: find windows near the target minute
        for i, w in enumerate(win_list):
            parts = w.replace("1H_","").replace("2H_","").split("-")
            try:
                start = int(parts[0].split("_")[-1])
                end   = int(parts[1].split("_")[0]) if len(parts) > 1 else start + 5
                if start <= target_minute <= end:
                    return i
            except:
                pass
        return None

    impacts = []
    for sub in all_subs:
        minute = sub.get("time", {}).get("elapsed", 0)
        team   = sub.get("team", {}).get("name", "")
        player_off = sub.get("player", {}).get("name", "?")
        player_on  = sub.get("assist", {}).get("name", "?")

        idx = window_index(windows_ordered, minute)
        if idx is None:
            continue

        before_windows = windows_ordered[max(0, idx-3):idx]
        after_windows  = windows_ordered[idx:min(len(windows_ordered), idx+3)]

        def avg_metric(win_list, metric_dict):
            vals = [metric_dict.get(w) for w in win_list if metric_dict.get(w) is not None]
            return round(sum(vals)/len(vals), 2) if vals else None

        impacts.append({
            "minute":     minute,
            "team":       team,
            "player_off": player_off,
            "player_on":  player_on,
            "before": {
                "pressing":   avg_metric(before_windows, pb_win),
                "line_height":avg_metric(before_windows, lh_win),
                "possession": avg_metric(before_windows, po_win),
            },
            "after": {
                "pressing":   avg_metric(after_windows, pb_win),
                "line_height":avg_metric(after_windows, lh_win),
                "possession": avg_metric(after_windows, po_win),
            },
            "windows_before": before_windows,
            "windows_after":  after_windows,
        })
    return impacts

def observation_grade(confidence: str, frequency: str) -> str:
    """
    Grade a single individual observation.
    A = Confirmed pattern, high confidence (act on this directly)
    B = Likely pattern, moderate confidence (use with care)
    C = Single clear sighting or low-visibility pattern
    D = Suggestive only (omit from main profile)
    """
    conf  = {"high": 3, "medium": 2, "low": 1}.get((confidence or "").lower(), 1)
    freq  = {"consistent": 3, "repeated": 2, "single": 1}.get((frequency or "").lower(), 1)
    score = conf * freq
    if score >= 6: return "A"
    if score >= 3: return "B"
    if score >= 2: return "C"
    return "D"


def player_profile_grade(observations: list) -> tuple:
    """
    Grade a player profile.
    Returns (grade, summary_str, ab_count, c_count, d_count, window_count)
    A = Strong intelligence, patterns confirmed across multiple windows
    B = Reasonable read, patterns identified with some gaps
    C = Limited data, treat as preliminary
    D = Insufficient data, match record only
    """
    if not observations:
        return "D", "No observations recorded", 0, 0, 0, 0

    grades   = [observation_grade(o.get("confidence","low"), o.get("frequency","single"))
                for o in observations]
    ab_count = sum(1 for g in grades if g in ("A","B"))
    c_count  = sum(1 for g in grades if g == "C")
    d_count  = sum(1 for g in grades if g == "D")
    windows  = len(set(o.get("timestamp","")[:5] for o in observations if o.get("timestamp")))

    if ab_count >= 4 and windows >= 3:
        grade   = "A"
        meaning = "Strong intelligence -- patterns confirmed across multiple windows"
    elif ab_count >= 2 or (ab_count >= 1 and c_count >= 3):
        grade   = "B"
        meaning = "Reasonable read -- patterns identified, some gaps"
    elif ab_count >= 1 or c_count >= 2:
        grade   = "C"
        meaning = "Limited data -- preliminary, treat with caution"
    else:
        grade   = "D"
        meaning = "Insufficient data -- match record only"

    summary = (f"Grade {grade}: {ab_count} confirmed, {c_count} preliminary, "
               f"{d_count} suggestive ({len(observations)} total, {windows} windows). "
               f"{meaning}.")
    return grade, summary, ab_count, c_count, d_count, windows


def grade_new_metric_sample(metric_name: str, value, summary: dict) -> str:
    """Sample status for fitness_trajectory and line_reaction_to_goals."""
    if metric_name == "fitness_trajectory":
        n = len(summary.get("pressing_by_window", []))
        if n >= 10: return "sufficient"
        if n >= 6:  return "preliminary"
        return "context_only"
    if metric_name == "line_reaction_to_goals":
        if not isinstance(value, dict): return "context_only"
        state_windows = [value.get(s, {}).get("windows", 0)
                         for s in ("level","winning","losing") if s in value]
        min_w = min(state_windows) if state_windows else 0
        if min_w >= 3: return "sufficient"
        if min_w >= 2: return "preliminary"
        return "context_only"
    return "sufficient"


COMPACTNESS_PROSE = {
    "very_compact": ("back line and midfield unit holding very tight vertical proximity, "
                     "with minimal space between the lines"),
    "compact":      ("back line and midfield operating in close vertical proximity, "
                     "distances between lines held tight across most phases"),
    "moderate":     "reasonable defensive organisation with occasional stretching between the lines",
    "stretched":    "notable gaps between the defensive and midfield lines -- space available between the blocks",
    "disorganised": "inconsistent defensive shape with significant gaps between units",
}

REST_DEFENCE_PROSE = {
    "very_secure":   "the back line holds its position almost completely after possession changes",
    "secure":        "the back line settles quickly after possession changes with few adjustments required",
    "moderate":      ("roughly one positional adjustment per possession phase -- "
                      "a reactive rest-defence rather than a committed resting shape"),
    "vulnerable":    "the back line makes frequent adjustments after possession changes -- unsettled rest-defence shape",
    "very_vulnerable":"the defensive structure is regularly disrupted following possession changes",
}

PREDICTABILITY_PROSE = {
    "highly_predictable":("extremely predictable build-up -- a small number of routes repeated heavily, "
                          "making the shape readable from short observation"),
    "predictable":       ("limited route variety makes the build-up readable -- "
                          "the likely options are identifiable and can be covered"),
    "mixed":             "a moderate mix of recurring routes and variation -- neither formulaic nor unpredictable",
    "varied":            "good variety in build-up routes -- not easily pressed from a preset shape",
    "unpredictable":     "high variety in build-up patterns -- the shape resists systematic pressing",
}

SUPPORT_PROSE = {
    "deep_support":    ("shots arrive at the end of patient multi-phase build-up with supporting runners "
                        "consistently in position -- not a direct or counter-attacking side"),
    "good_support":    ("shots come from sustained build-up with runners arriving into the final third "
                        "before the attempt"),
    "moderate_support":"a mix of quick and sustained attacks -- neither consistently direct nor consistently patient",
    "limited_support": ("most attacks reach the final third with limited supporting movement -- "
                        "few runners beyond the ball"),
    "isolated":        "the primary striker is largely isolated -- most attempts arrive with minimal supporting runners",
}

LINE_HEIGHT_LANDMARKS = {
    "very_high": "brushing the halfway line or beyond",
    "high":      "between the halfway line and the opponent's final third",
    "medium":    "in the central zone between the thirds",
    "low":       "inside their own defensive third",
    "very_low":  "deep inside their own penalty area",
}

# -- Sample-size discipline ----------------------------------------------------
# Below these counts, a metric is downgraded to context_only.
# context_only metrics appear in Data Quality Notes, not as tactical reads.

SAMPLE_THRESHOLDS = {
    "pressing_trigger_consistency": {"sufficient": 5,  "preliminary": 3},
    "set_piece_delivery_profile":   {"sufficient": 5,  "preliminary": 2},
    "chance_creation_profile":      {"sufficient": 5,  "preliminary": 3},
    "transition_efficiency_score":  {"sufficient": 5,  "preliminary": 3},
    "width_usage_score":            {"sufficient": 10, "preliminary": 5},
    "pattern_reliability_score":    {"sufficient": 15, "preliminary": 5},
    "build_up_route_diversity":     {"sufficient": 15, "preliminary": 5},
    "attacking_support_score":      {"sufficient": 5,  "preliminary": 3},
}


def sample_status(metric_name, sample_count, extra=None):
    """
    Returns "sufficient" | "preliminary" | "context_only".
    extra: dict of additional checks (e.g. method="cross_outcomes_proxy" for width_usage).
    """
    # Width usage: if method is proxy, always context_only regardless of count
    if metric_name == "width_usage_score" and extra:
        if extra.get("method") == "cross_outcomes_proxy":
            return "context_only"

    thresholds = SAMPLE_THRESHOLDS.get(metric_name)
    if not thresholds:
        return "sufficient"
    if sample_count >= thresholds["sufficient"]:
        return "sufficient"
    if sample_count >= thresholds["preliminary"]:
        return "preliminary"
    return "context_only"


def build_prose_interpretation(metric_name, value, sample_count=0):
    """
    Convert structured metric value into a tactical prose sentence.
    Returns a string ready for the report writer to use directly.
    Returns None if the metric is context_only (insufficient data).
    """
    if value is None:
        return None

    # -- compactness -----------------------------------------------------------
    if metric_name == "compactness_score":
        if not isinstance(value, dict): return None
        cat    = value.get("category", "unknown")
        avg_m  = value.get("avg_line_height_m")
        base   = COMPACTNESS_PROSE.get(cat, cat)
        if avg_m is not None:
            return f"{base} -- average defensive line at {avg_m}m from own goal"
        return base

    # -- pressing intensity ----------------------------------------------------
    if metric_name == "pressing_intensity_score":
        if isinstance(value, dict):
            avg_10 = value.get("avg_score_0_10", 0)
            cat    = value.get("category", "low")
            return f"{avg_10}/10 -- {PRESS_PROSE.get(cat, cat)}"
        return None

    # -- build-up effectiveness ------------------------------------------------
    if metric_name == "build_up_effectiveness_score":
        if isinstance(value, dict):
            ft   = value.get("final_third_rate_pct", 0)
            conv = value.get("conversion_rate_pct", 0)
            total= value.get("total_sequences", 0)
            if total < 10:
                return None
            if conv < 3:
                return (f"{ft}% of {total} sequences reached the final third, "
                        f"but only {conv}% ended in a shot or cross -- "
                        f"consistent progression without end product")
            return (f"{ft}% of {total} sequences reached the final third; "
                    f"{conv}% ended in a shot or cross")
        return None

    # -- rest defence ---------------------------------------------------------
    if metric_name == "rest_defence_security_score":
        if isinstance(value, dict):
            shifts = value.get("avg_backward_shifts_per_window")
            cat    = value.get("category", "moderate")
            if shifts is None:
                return None
            # Fix 53C: previously emitted "X backward line shifts per window".
            # The "per window" suffix leaked into synthesis prose and broke
            # WINDOW LANGUAGE RULE on broadcast reports. Drop it; the average
            # is per observation interval, which the synthesis prose can frame
            # in match-clock language.
            return f"{REST_DEFENCE_PROSE.get(cat, cat)} (avg {shifts} backward line shifts)"
        return None

    # -- line height range -----------------------------------------------------
    if metric_name == "line_height_range":
        if isinstance(value, dict):
            hi_m   = value.get("highest_m")
            lo_m   = value.get("lowest_m")
            rng_m  = value.get("range_m")
            hi_cat = value.get("highest_category", "")
            lo_cat = value.get("lowest_category", "")
            if hi_m is None or lo_m is None:
                return None
            hi_land = LINE_HEIGHT_LANDMARKS.get(hi_cat, "")
            lo_land = LINE_HEIGHT_LANDMARKS.get(lo_cat, "")
            val_dict = value  # access window counts
            hi_wins  = val_dict.get("high_line_windows", 0)
            lo_wins  = val_dict.get("low_line_windows", 0)
            tot_wins = val_dict.get("total_windows", 1)
            context  = ""
            if hi_wins == 1 and lo_wins > 5:
                context = f" (the high position was a spike, not a pattern -- {lo_wins} of {tot_wins} windows at low depth)"
            elif hi_wins > tot_wins * 0.4 and lo_wins > tot_wins * 0.4:
                context = f" ({hi_wins} windows at high depth, {lo_wins} at low depth -- genuinely oscillating)"
            return (f"The defensive line moved between {lo_m}m ({lo_land}) "
                    f"and {hi_m}m ({hi_land}) -- "
                    f"a {rng_m}m range indicating a genuinely reactive line "
                    f"that adjusts to ball position rather than holding a preset depth{context}")
        return None

    # -- pattern reliability ---------------------------------------------------
    if metric_name == "pattern_reliability_score":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            pct   = value.get("top_route_pct", 0)
            route = value.get("top_route", "unknown")
            total = value.get("total_sequences", 0)
            prefix = "preliminary reading: " if status == "preliminary" else ""
            return (f"{prefix}{pct}% of sequences used the dominant route "
                    f"({route}) across {total} observed sequences")
        return None

    # -- build-up route diversity ----------------------------------------------
    if metric_name == "build_up_route_diversity":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            distinct = value.get("distinct_routes", 0)
            prefix   = "preliminary reading: " if status == "preliminary" else ""
            if distinct <= 1:
                return (f"{prefix}only {distinct} distinct non-baseline route observed -- "
                        f"build-up is highly predictable beyond standard forward movement")
            if distinct <= 2:
                return (f"{prefix}{distinct} distinct non-baseline routes -- "
                        f"limited variety, the shape is readable")
            if distinct <= 3:
                return f"{prefix}{distinct} distinct non-baseline routes -- moderate variety"
            return (f"{prefix}{distinct}+ distinct non-baseline routes -- "
                    f"high variety in build-up, difficult to press from a preset shape")
        return None

    # -- predictability --------------------------------------------------------
    if metric_name == "predictability_score":
        if isinstance(value, dict):
            cat = value.get("category", "mixed")
            return PREDICTABILITY_PROSE.get(cat, cat)
        return None

    # -- pressing trigger consistency ------------------------------------------
    if metric_name == "pressing_trigger_consistency":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            pct       = value.get("pct_using_dominant_trigger", 0)
            count     = value.get("count_using_dominant_trigger", 0)
            total_obs = value.get("total_observations", 0)
            trigger   = value.get("dominant_trigger", "unknown")
            prefix    = "preliminary reading: " if status == "preliminary" else ""

            # Human-readable trigger names
            trigger_labels = {
                "gk_in_possession":           "the goalkeeper receiving the ball",
                "back_pass":                  "a back pass into the defensive third",
                "defender_facing_own_goal":   "a defender receiving with their back to goal",
                "free_kick_restart":          "a free-kick restart",
                "other":                      "varied triggers",
            }
            trigger_label = trigger_labels.get(trigger, trigger.replace("_", " "))

            if pct >= 90:
                return (f"{prefix}{count} of {total_obs} observed pressing triggers occurred "
                        f"only when {trigger_label} -- "
                        f"they do not press during recycling between defenders, midfield receptions, "
                        f"or after goal kicks once the ball leaves the goalkeeper")
            elif pct >= 70:
                return (f"{prefix}the dominant press trigger is {trigger_label} "
                        f"({count} of {total_obs} triggers, {pct}%) -- "
                        f"other triggers occur but this is the most consistent activation point")
            else:
                triggers = value.get("trigger_counts", {})
                top_two  = sorted(triggers.items(), key=lambda x: x[1], reverse=True)[:2]
                return (f"{prefix}press triggers are varied across {total_obs} observations -- "
                        f"most common: {trigger_labels.get(top_two[0][0], top_two[0][0])} "
                        f"({top_two[0][1]} times)")
        return None

    # -- set piece delivery profile --------------------------------------------
    if metric_name == "set_piece_delivery_profile":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            total = value.get("total_set_pieces", 0)
            prefix = "preliminary reading (small sample): " if status == "preliminary" else ""
            top_zone = max(value.get("by_zone",{}),
                          key=value["by_zone"].get) if value.get("by_zone") else "unknown"
            top_del  = max(value.get("by_delivery_type",{}),
                          key=value["by_delivery_type"].get) if value.get("by_delivery_type") else "unknown"
            top_outcome = max(value.get("by_outcome",{}),
                             key=value["by_outcome"].get) if value.get("by_outcome") else "unknown"
            return (f"{prefix}{total} set pieces observed: "
                    f"most from {top_zone.replace('_',' ')}, "
                    f"predominantly {top_del.replace('_',' ')} deliveries, "
                    f"most common outcome: {top_outcome.replace('_',' ')}")
        return None

    # -- chance creation profile -----------------------------------------------
    if metric_name == "chance_creation_profile":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            total   = value.get("total_chances", 0)
            by_zone = value.get("by_origin_zone", {})
            by_route= value.get("by_route", {})
            avg_seq = value.get("avg_sequence_length")
            prefix  = "preliminary reading: " if status == "preliminary" else ""

            top_zone  = max(by_zone,  key=by_zone.get)  if by_zone  else "unknown"
            top_route = max(by_route, key=by_route.get) if by_route else "unknown"

            zone_label  = {"own_half":"own half","midfield":"the middle third",
                          "own third":"own third","final third":"the final third"}.get(
                          football_zone(top_zone), football_zone(top_zone))
            route_label = top_route.replace("_"," ")

            seq_note = (f", typically after {avg_seq}-pass sequences" if avg_seq else "")
            return (f"{prefix}creative threat concentrated in {zone_label} "
                    f"via {route_label} combinations{seq_note} -- "
                    f"{total} chances observed")
        return None

    # -- attacking support -----------------------------------------------------
    if metric_name == "attacking_support_score":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if isinstance(value, dict):
            avg     = value.get("avg_passes_before_shot")
            cat     = value.get("support_depth", "moderate_support")
            samples = value.get("samples", 0)
            if avg is None:
                return None
            prefix = "preliminary reading: " if status == "preliminary" else ""
            return f"{prefix}{SUPPORT_PROSE.get(cat, cat)} -- {avg} passes before shot on average ({samples} samples)"
        return None

    # -- transition efficiency ------------------------------------------------
    if metric_name == "transition_efficiency_score":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if value is None:
            return None
        prefix = "preliminary reading: " if status == "preliminary" else ""
        if isinstance(value, dict):
            att_rate  = value.get("attack_threat_rate")
            def_rate  = value.get("defence_exposure_rate")
            att_n     = value.get("attack_transitions", 0)
            def_n     = value.get("defence_transitions", 0)
            parts = []
            if att_rate is not None and att_n > 0:
                pct = round(att_rate * 100)
                if att_rate >= 0.4:
                    parts.append(f"{pct}% of winning-the-ball transitions led to a counter "
                                 f"({att_n} observed) -- threatening in transition")
                else:
                    parts.append(f"{pct}% of winning-the-ball transitions led to a counter "
                                 f"({att_n} observed) -- most transitions were absorbed")
            if def_rate is not None and def_n > 0:
                pct = round(def_rate * 100)
                if def_rate >= 0.4:
                    parts.append(f"{pct}% of losing-the-ball transitions conceded a counter "
                                 f"({def_n} observed) -- exposed in defensive transition")
                else:
                    parts.append(f"{pct}% of losing-the-ball transitions conceded a counter "
                                 f"({def_n} observed) -- defensive transition reasonably secure")
            if parts:
                return prefix + "; ".join(parts)
        return None

    # -- fitness trajectory ----------------------------------------------------
    if metric_name == "fitness_trajectory":
        if not isinstance(value, dict): return None
        traj   = value.get("trajectory", "stable")
        early  = value.get("early_avg", 0)
        late   = value.get("late_avg", 0)
        delta  = value.get("delta", 0)
        if traj == "rising":
            return (f"Pressing intensity increased across the match -- "
                    f"averaged {early}/10 in the opening phase, {late}/10 in the closing phase. "
                    f"The team pressed more as the match progressed, not less.")
        elif traj == "falling":
            return (f"Pressing intensity dropped across the match -- "
                    f"{early}/10 in the opening phase, falling to {late}/10 late on. "
                    f"The team pressed less as the match progressed.")
        else:
            return (f"Pressing intensity was consistent across the match -- "
                    f"{early}/10 early, {late}/10 late (delta: {delta:+.1f}).")

    # -- line reaction to goals ------------------------------------------------
    if metric_name == "line_reaction_to_goals":
        if not isinstance(value, dict): return None
        delta = value.get("delta_losing_vs_level_m")
        level = value.get("level", {}).get("avg_m")
        losing= value.get("losing", {}).get("avg_m")
        winning=value.get("winning", {}).get("avg_m")
        parts = []
        if level is not None:
            parts.append(f"level: {level}m")
        if losing is not None:
            parts.append(f"losing: {losing}m")
        if winning is not None:
            parts.append(f"winning: {winning}m")
        base = "Defensive line by scoreline -- " + ", ".join(parts)
        if delta is not None:
            if delta < -3:
                base += (f". The line dropped {abs(delta)}m deeper when behind -- "
                         f"the team defends lower after conceding.")
            elif delta > 3:
                base += (f". The line pushed {delta}m higher when losing -- "
                         f"the team sought to compress space rather than protect depth.")
            else:
                base += ". Minimal change in line height with the scoreline."
        return base

    # -- halfspace occupation --------------------------------------------------
    if metric_name == "halfspace_occupation_score":
        status = sample_status(metric_name, sample_count)
        if status == "context_only":
            return None
        if value is None:
            return None
        prefix = "preliminary reading: " if status == "preliminary" else ""
        pct = round(value * 100, 1)
        if value >= 0.3:
            return (f"{prefix}half-space zones were frequently occupied -- "
                    f"{pct}% of sequences involved the inside channels, "
                    f"suggesting deliberate use of the spaces between the centre and wide areas")
        elif value >= 0.15:
            return (f"{prefix}moderate half-space activity -- "
                    f"{pct}% of sequences passed through the inside channels")
        else:
            return (f"{prefix}limited half-space occupation -- "
                    f"{pct}% of sequences used the inside channels; "
                    f"play was predominantly central or wide")

    # -- width usage -----------------------------------------------------------
    # -- width usage -----------------------------------------------------------
    if metric_name == "width_usage_score":
        method = value.get("method") if isinstance(value, dict) else None
        status = sample_status(metric_name, sample_count, extra={"method": method})
        if status == "context_only":
            return None  # Do not surface in report -- source artifact
        if isinstance(value, dict):
            pct = round(value.get("score", 0) * 100)
            return f"{pct}% of sequences used wide channels"
        return None

    return None



def load_pipeline_data(match_dir):
    def load(fname, default=None):
        p = os.path.join(match_dir, fname)
        if not os.path.exists(p):
            return default
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {
        "summary":       load("running_summary.json",     {}),
        "passes":        load("pass_sequences.json",      {"sequences": []}),
        "gates":         load("result_family_gates.json", {"gates": {}}),
        "source":        load("source_profile.json",      {"source_type": "unknown", "visibility_scores": {}}),
        "confirmations": load("confirmation_queue.json",  {"items": []}),
    }


def get_gate(family, gates):
    return gates.get("gates", {}).get(family, "downgraded")


def worst_evidence_tier(tiers):
    order = ["escalated_confirmation", "direct", "repeated_pattern", "suggestive"]
    for t in reversed(order):
        if t in tiers:
            return t
    return "suggestive"


def compute_metric_confidence(base, required_families, gates, evidence_tiers, windows_contributing, source_type):
    confidence = base
    worst_tier = worst_evidence_tier(evidence_tiers)
    cap = EVIDENCE_TIER_CAP.get(worst_tier, 0.4)
    confidence = min(confidence, cap)

    max_penalty = 0.0
    statuses = []
    for family in required_families:
        status = get_gate(family, gates)
        statuses.append(status)
        penalty = STATUS_CONFIDENCE_PENALTY.get(status, 0.0)
        max_penalty = max(max_penalty, penalty)
    confidence -= max_penalty

    if 0 < windows_contributing < MIN_WINDOWS:
        confidence -= 0.15

    global_cap = SOURCE_GLOBAL_CAP.get(source_type, 0.5)
    confidence = min(confidence, global_cap)
    confidence = max(0.0, round(confidence, 2))

    if any(s == "downgraded" for s in statuses):
        status_out = "downgraded"
        limitation = f"Confidence reduced -- required families downgraded for {source_type}"
    else:
        status_out = "allowed"
        limitation = None

    return confidence, status_out, limitation


# -- System-level --------------------------------------------------------------

def calc_compactness(summary):
    """
    Compactness: measured from actual average line height, not stability of movement.
    Lower line = more compact. Higher line = more expansive.
    A team sitting at 25m is compact regardless of whether the line moves.
    A team sitting at 52m is not compact regardless of stability.
    Score 0-1: higher = more compact (line sits deeper).
    """
    line_data = summary.get("line_height_by_window", [])
    pressing  = summary.get("pressing_by_window",   [])
    heights   = [w.get("avg_pct") for w in line_data if w.get("avg_pct") is not None]
    if not heights:
        return 0.0, None, 0, ["suggestive"], "unknown"
    avg_height_pct = sum(heights) / len(heights)
    avg_height_m   = round(avg_height_pct / 100.0 * PITCH_LENGTH_M, 1)
    # Invert: lower line = higher compactness (0m = 1.0, 105m = 0.0)
    height_score = max(0.0, min(1.0, 1.0 - (avg_height_pct / 100.0)))
    scores     = [w.get("avg_score") for w in pressing if w.get("avg_score") is not None]
    press_norm = (sum(scores) / len(scores) / 10.0) if scores else 0.3
    # 80% line height position, 20% pressing activity
    score = round(height_score * 0.80 + press_norm * 0.20, 2)
    return score, avg_height_m, len(heights), ["repeated_pattern"], compactness_category(score)



def calc_pressing_intensity(summary):
    pressing = summary.get("pressing_by_window", [])
    scores   = [w.get("avg_score") for w in pressing if w.get("avg_score") is not None]
    if not scores:
        return 0.0, 0.0, 0, ["suggestive"]
    avg_10 = round(sum(scores) / len(scores), 2)
    avg_01 = round(avg_10 / 10.0, 2)
    return avg_10, avg_01, len(scores), ["direct"]


def calc_build_up_effectiveness(passes):
    sequences  = passes.get("sequences", [])
    if not sequences:
        return 0.0, 0, 0, 0, 0, 0.0, 0.0, 0.0, ["suggestive"]
    progressive = sum(1 for s in sequences if s.get("progressive") is True)
    threats     = sum(1 for s in sequences if s.get("outcome") in ("shot", "cross", "goal"))
    # final_third: sequences that actually reached the attacking third (more meaningful than 'progressive')
    final_third = sum(1 for s in sequences
                      if s.get("zone_end") in ("attacking_third", "right_channel", "left_channel",
                                                "right_halfspace", "left_halfspace")
                      and s.get("zone_start") not in ("attacking_third",))
    total       = len(sequences)
    score            = round(min((progressive + threats) / total, 1.0), 2) if total > 0 else 0.0
    progressive_rate = round(progressive / total, 3) if total > 0 else 0.0
    final_third_rate = round(final_third / total, 3) if total > 0 else 0.0
    conversion_rate  = round(threats / total, 3) if total > 0 else 0.0
    return score, total, progressive, threats, final_third, progressive_rate, final_third_rate, conversion_rate, ["repeated_pattern"]


def calc_rest_defence_security(summary):
    """
    Rest-defence security: only backward defensive line shifts indicate vulnerability.
    Forward shifts (line pushing higher) are tactically positive and not penalised.
    """
    line_data = summary.get("line_height_by_window", [])
    if not line_data:
        return 0.0, 0, 0, ["suggestive"]
    windows        = len(line_data)
    total_backward = 0
    for w in line_data:
        for shift in w.get("shifts", []):
            if isinstance(shift, dict):
                from_p = shift.get("from_pct", 0)
                to_p   = shift.get("to_pct", 0)
                if from_p > to_p:
                    total_backward += 1
            elif isinstance(shift, str):
                import re as _re2
                nums = _re2.findall(r"(\d+(?:\.\d+)?)%", shift)
                if len(nums) >= 2 and float(nums[0]) > float(nums[1]):
                    total_backward += 1
                elif not nums:
                    total_backward += 1  # conservative if unparseable
    backward_per_win = round(total_backward / windows, 2) if windows > 0 else 0.0
    score = max(0.0, round(1.0 - (backward_per_win * 0.25), 2))
    return score, backward_per_win, windows, ["repeated_pattern"]



def calc_line_height_range(summary):
    line_data  = summary.get("line_height_by_window", [])
    if not line_data:
        return None, 0, ["suggestive"]
    all_values = []
    for w in line_data:
        for field in ("avg_pct", "start_pct", "end_pct"):
            v = w.get(field)
            if v is not None:
                all_values.append(v)
        for shift in w.get("shifts", []):
            if isinstance(shift, dict):
                for key in ("from_pct", "to_pct", "pct"):
                    v = shift.get(key)
                    if v is not None:
                        all_values.append(v)
            elif isinstance(shift, str):
                # Legacy string format: "07m30s: dropped from 45% to 32% after goal"
                # Extract numeric percentages
                import re as _re
                nums = _re.findall(r"(\d+(?:\.\d+)?)%", shift)
                for n in nums:
                    try:
                        all_values.append(float(n))
                    except ValueError:
                        pass
    if len(all_values) < 2:
        return None, len(all_values), ["suggestive"]
    range_pct = max(all_values) - min(all_values)
    highest_m = pct_to_metres(max(all_values))
    lowest_m  = pct_to_metres(min(all_values))

    # Count windows where the line sat in high vs low bands
    # High band: avg_pct > 45% (~47m); Low band: avg_pct < 33% (~35m)
    window_avgs    = [w.get("avg_pct") for w in line_data if w.get("avg_pct") is not None]
    high_windows   = sum(1 for p in window_avgs if p > 45)
    low_windows    = sum(1 for p in window_avgs if p < 33)
    total_windows  = len(window_avgs)

    return {
        "highest_m":        highest_m,
        "lowest_m":         lowest_m,
        "range_m":          round(highest_m - lowest_m, 1),
        "highest_category": line_height_category(highest_m),
        "lowest_category":  line_height_category(lowest_m),
        "highest_pct":      round(max(all_values), 1),
        "lowest_pct":       round(min(all_values), 1),
        "range_pct":        round(range_pct, 1),
        "high_line_windows":  high_windows,
        "low_line_windows":   low_windows,
        "total_windows":      total_windows,
    }, len(line_data), ["repeated_pattern"]


# -- Behavioural ---------------------------------------------------------------

def calc_width_usage(passes):
    sequences    = passes.get("sequences", [])
    if not sequences:
        return 0.0, 0, ["suggestive"], "none"
    wide_keywords = ("left_channel", "right_channel", "left_flank", "right_flank",
                     "left_wide", "right_wide", "wide_left", "wide_right")
    zone_wide = sum(1 for s in sequences
                    if any(kw in str(s.get("zone_start", "")).lower() or
                           kw in str(s.get("zone_end", "")).lower()
                           for kw in wide_keywords))
    cross_wide = sum(1 for s in sequences if s.get("outcome") == "cross")
    wide_count = zone_wide if zone_wide > 0 else cross_wide
    method     = "zone_labels" if zone_wide > 0 else "cross_outcomes_proxy"
    score      = round(min(wide_count / len(sequences), 1.0), 2)
    tiers      = ["repeated_pattern"] if zone_wide > 0 else ["suggestive"]
    return score, len(sequences), tiers, method


def calc_halfspace_occupation(passes):
    sequences = passes.get("sequences", [])
    if not sequences:
        return None, 0, ["suggestive"]
    hs_keywords = ("halfspace", "half_space", "inside_channel")
    hs_seqs = sum(1 for s in sequences
                  if any(kw in str(s.get("zone_start", "")).lower() or
                         kw in str(s.get("zone_end", "")).lower()
                         for kw in hs_keywords))
    if hs_seqs == 0:
        return None, 0, ["suggestive"]
    return round(hs_seqs / len(sequences), 2), len(sequences), ["repeated_pattern"]


def calc_transition_efficiency(summary, confirmations):
    """
    Split transition analysis into attack and defence dimensions:

    attack_efficiency: when THIS team wins the ball, how often does the
    transition lead to a counter or shot?
    defence_exposure: when THIS team LOSES the ball, how often does the
    opponent launch a counter?

    A team can be good at attacking transitions but poor at defending them.
    These are different tactical concerns and should not be combined.
    """
    direct_tr = summary.get("transitions", [])
    flagged   = [m for m in summary.get("flagged_moments", [])
                 if "transition" in str(m.get("type", "")).lower()
                 or "transition" in str(m.get("reason", "")).lower()]
    key_tr    = [m for m in summary.get("key_moments", [])
                 if "transition" in str(m.get("type", "")).lower()]
    conf_tr   = [i for i in confirmations.get("items", [])
                 if i.get("result_family") == "transitions" and i.get("status") == "resolved"]
    all_tr    = direct_tr + flagged + key_tr + conf_tr

    if len(all_tr) < 3:
        return None, len(all_tr), ["suggestive"]

    # Split by direction
    attack_tr  = [t for t in all_tr if t.get("direction") == "defence_to_attack"]
    defence_tr = [t for t in all_tr if t.get("direction") == "attack_to_defence"]

    threat_kw  = ("shot","goal","chance","cross_into_box","counter_launched","shot_on_goal")
    vuln_kw    = ("counter_launched","counter_attack","shot","goal")

    # Attack: how many defence_to_attack transitions led to a threat
    att_threats = sum(1 for t in attack_tr
                      if any(kw in str(t.get("outcome","")).lower() for kw in threat_kw))
    att_rate    = round(att_threats / len(attack_tr), 2) if attack_tr else None

    # Defence: how many attack_to_defence transitions led to opponent counter
    def_exposed = sum(1 for t in defence_tr
                      if any(kw in str(t.get("outcome","")).lower() for kw in vuln_kw))
    def_rate    = round(def_exposed / len(defence_tr), 2) if defence_tr else None

    # Combined score for backwards compatibility -- attack weighted slightly higher
    if att_rate is not None and def_rate is not None:
        # High att_rate is good, high def_rate is bad
        combined = round(att_rate * 0.6 + (1.0 - def_rate) * 0.4, 2)
    elif att_rate is not None:
        combined = att_rate
    elif def_rate is not None:
        combined = round(1.0 - def_rate, 2)
    else:
        combined = None

    tiers = []
    if conf_tr:               tiers.append("escalated_confirmation")
    if flagged or key_tr:     tiers.append("repeated_pattern")
    if direct_tr and not tiers: tiers.append("repeated_pattern")
    if not tiers:             tiers = ["suggestive"]

    result = {
        "score":               combined,
        "attack_transitions":  len(attack_tr),
        "attack_threat_rate":  att_rate,
        "defence_transitions": len(defence_tr),
        "defence_exposure_rate": def_rate,
        "total_transitions":   len(all_tr),
    }
    return result, len(all_tr), tiers


def _parse_ts(ts):
    if not ts: return 0.0
    m = re.match(r"(\d+)m(\d+)s", str(ts))
    if m: return int(m.group(1)) * 60 + int(m.group(2))
    try:    return float(ts)
    except: return 0.0


def _infer_route(seq):
    if not seq: return "unknown"
    combined = (str(seq.get("zone_start", "")) + " " +
                str(seq.get("zone_end", "")) + " " +
                str(seq.get("sequence", ""))).lower()
    if "left" in combined:  return "left_channel"
    if "right" in combined: return "right_channel"
    return "central"


def calc_chance_creation_profile(summary, passes):
    shots = summary.get("shots_for", [])
    seqs  = passes.get("sequences", [])
    chances = []
    for shot in shots:
        shot_ts  = _parse_ts(shot.get("timestamp"))
        match_seq = next((s for s in seqs
                          if s.get("outcome") in ("shot", "cross", "goal")
                          and abs(_parse_ts(s.get("start_frame")) - shot_ts) < 30), None)
        chances.append({
            "timestamp":      shot.get("timestamp"),
            "origin_zone":    match_seq.get("zone_start") if match_seq else "unknown",
            "route":          _infer_route(match_seq),
            "sequence_length":shot.get("sequence_length") or
                              (match_seq.get("length") if match_seq else None),
            "shot_zone":      shot.get("origin_column", "unknown"),
            "shot_type":      shot.get("shot_type", "unknown"),
            "outcome":        shot.get("outcome", "unknown"),
            "target_zone":    shot.get("target_zone", "unknown"),
            "source":         "shots_for",
        })
    if not chances:
        threat_seqs = [s for s in seqs if s.get("outcome") in ("shot", "cross")]
        for s in threat_seqs:
            chances.append({
                "timestamp":      s.get("start_frame"),
                "origin_zone":    s.get("zone_start", "unknown"),
                "route":          _infer_route(s),
                "sequence_length":s.get("length"),
                "shot_zone":      s.get("zone_end", "unknown"),
                "shot_type":      "unknown",
                "outcome":        s.get("outcome"),
                "target_zone":    "unknown",
                "source":         "threat_sequences_fallback",
            })
    if not chances:
        return None, 0, ["suggestive"]
    seq_lengths = [c["sequence_length"] for c in chances if c["sequence_length"]]
    by_origin  = dict(Counter(c["origin_zone"] for c in chances))
    by_route   = dict(Counter(c["route"] for c in chances))
    by_outcome = dict(Counter(c["outcome"] for c in chances))
    top_origin = max(by_origin, key=by_origin.get) if by_origin else "unknown"
    top_route  = max(by_route,  key=by_route.get)  if by_route  else "unknown"

    # Zone quality: high-danger = six_yard_box + penalty_spot, low = edge/outside
    high_danger_zones = {"six_yard_box", "penalty_spot"}
    high_danger = sum(1 for c in chances
                      if c.get("shot_zone") in high_danger_zones
                      or c.get("origin_row") in high_danger_zones)
    low_danger  = len(chances) - high_danger
    danger_pct  = round(high_danger / len(chances) * 100) if chances else 0

    profile = {
        "total_chances":       len(chances),
        "chances":             chances,
        "by_origin_zone":      by_origin,
        "by_route":            by_route,
        "by_outcome":          by_outcome,
        "high_danger_chances": high_danger,
        "low_danger_chances":  low_danger,
        "high_danger_pct":     danger_pct,
        "avg_sequence_length": round(sum(seq_lengths)/len(seq_lengths), 1) if seq_lengths else None,
        "data_source":         "shots_for" if shots else "threat_sequences_fallback",
        "summary":             (f"{len(chances)} chances: {danger_pct}% from high-danger zones "
                                f"(six-yard box or penalty spot); most from {football_zone(top_origin)}"
                                f" via {top_route} route"),
    }
    tiers = ["direct"] if shots else ["repeated_pattern"]
    return profile, len(chances), tiers



# Baseline routes that every team uses -- exclude from predictability
# since they reflect the shape of football, not tactical predictability
BASELINE_ROUTES = {
    ("defending_third", "middle"),
    ("middle", "attacking_third"),
    ("defending_third", "defending_third"),
    ("middle", "middle"),
    ("attacking_third", "middle"),
    ("attacking_third", "attacking_third"),
}


def calc_pattern_reliability(passes):
    """
    Most-used NON-BASELINE zone-to-zone route / non-baseline total.

    Baseline routes (defending_third->middle, middle->attacking_third etc.) are
    excluded because they appear in every team's build-up by default.
    Predictability is only meaningful in the routes beyond baseline movement.

    If fewer than 10 non-baseline sequences exist, falls back to full route analysis
    with a note that the 3-zone encoding limits discriminating power.
    """
    sequences = passes.get("sequences", [])
    if not sequences:
        return 0.0, None, 0, 0, ["suggestive"]

    # Try non-baseline first
    non_baseline = [s for s in sequences
                    if (s.get("zone_start"), s.get("zone_end")) not in BASELINE_ROUTES
                    and s.get("zone_start") and s.get("zone_end")]

    use_sequences = non_baseline if len(non_baseline) >= 10 else sequences
    fallback = len(non_baseline) < 10

    zone_pairs = Counter(
        (s.get("zone_start"), s.get("zone_end"))
        for s in use_sequences
        if s.get("zone_start") and s.get("zone_end")
    )
    if not zone_pairs:
        return 0.0, None, 0, 0, ["suggestive"]

    total          = sum(zone_pairs.values())
    top_pair, top_count = zone_pairs.most_common(1)[0]
    score          = round(top_count / total, 2)
    top_route_label = (f"{football_zone(top_pair[0])} -> {football_zone(top_pair[1])}"
                       if top_pair else None)

    if fallback:
        # 3-zone encoding with no lateral zones -- metric has low discriminating power
        top_route_label = f"{top_route_label} (3-zone encoding -- limited discriminating power)"

    return score, top_route_label, top_count, total, ["repeated_pattern"]


def calc_build_up_route_diversity(passes):
    sequences = passes.get("sequences", [])
    if not sequences:
        return 0.0, 0, 0, ["suggestive"]
    zone_pairs = set((s.get("zone_start"), s.get("zone_end"))
                     for s in sequences if s.get("zone_start") and s.get("zone_end"))
    score = round(min(len(zone_pairs) / 4.0, 1.0), 2)  # 4 = realistic max for non-baseline routes
    return score, len(sequences), len(zone_pairs), ["repeated_pattern"]


# -- Tactical insight ----------------------------------------------------------

def calc_predictability(pattern_reliability, route_diversity):
    if pattern_reliability is None or route_diversity is None:
        return None, None, ["suggestive"]
    score = round(pattern_reliability * 0.6 + (1.0 - route_diversity) * 0.4, 2)
    return score, predictability_category(score), ["repeated_pattern"]


def calc_pressing_trigger_consistency(summary):
    all_obs = []
    for window in summary.get("pressing_by_window", []):
        all_obs.extend(window.get("observations", []))
    if all_obs:
        triggers = [o.get("trigger", "unknown") for o in all_obs]
        tc = Counter(triggers)
        top_count = tc.most_common(1)[0][1]
        top_type  = tc.most_common(1)[0][0]
        pct = round(top_count/len(triggers) * 100)

        # Estimate press success rate from transitions following high-press windows
        # A successful press leads to defence_to_attack transition (turnover won)
        # An unsuccessful press = opponent plays through = no transition won
        transitions = summary.get("transitions", [])
        press_wins  = sum(1 for t in transitions
                          if t.get("direction") == "defence_to_attack"
                          and t.get("trigger") in ("interception", "clearance"))
        press_total = len(transitions)
        success_rate = round(press_wins / press_total, 2) if press_total >= 3 else None

        result = {
            "pct_using_dominant_trigger": pct,
            "count_using_dominant_trigger": top_count,
            "total_observations": len(triggers),
            "dominant_trigger": top_type,
            "trigger_counts": dict(tc),
            "summary": f"{top_count}/{len(triggers)} pressing triggers were {top_type} ({pct}%)",
        }
        if success_rate is not None:
            result["press_success_rate"] = success_rate
            result["summary"] += f"; press won ball in {round(success_rate*100)}% of transitions observed"

        return result, len(triggers), ["direct"]
    press_moments = [m for m in summary.get("key_moments", [])
                     if "press" in str(m.get("type", "")).lower()
                     or "press" in str(m.get("description", "")).lower()]
    if len(press_moments) < 3:
        return None, len(press_moments), ["suggestive"]
    trigger_map = {
        "gk_in_possession":        ("gk", "goalkeeper", "goal kick"),
        "back_pass":               ("back pass",),
        "defender_facing_own_goal":("facing own goal", "turned", "back to goal"),
        "free_kick_restart":       ("free kick", "restart"),
    }
    inferred = []
    for m in press_moments:
        desc  = str(m.get("description", "") or m.get("reason", "")).lower()
        found = "other"
        for t_type, kws in trigger_map.items():
            if any(kw in desc for kw in kws):
                found = t_type
                break
        inferred.append(found)
    tc  = Counter(inferred)
    top = tc.most_common(1)
    pct = round(top[0][1]/len(inferred) * 100)
    return {"pct_using_dominant_trigger": pct,
            "count_using_dominant_trigger": top[0][1],
            "dominant_trigger": top[0][0], "trigger_counts": dict(tc),
            "total_observations": len(inferred),
            "data_source": "inferred_from_key_moments",
            "summary": f"{top[0][1]}/{len(inferred)} pressing triggers were {top[0][0]} ({pct}%)"
            }, len(inferred), ["repeated_pattern"]


def calc_set_piece_delivery_profile(summary):
    sps = summary.get("set_pieces", [])
    if not sps:
        return None, 0, ["suggestive"]
    by_zone     = Counter(sp.get("zone", sp.get("type", "unknown")) for sp in sps)
    by_delivery = Counter(sp.get("delivery", "unknown") for sp in sps)
    by_outcome  = Counter(sp.get("outcome", "unknown") for sp in sps)
    by_marking  = Counter(sp.get("marking_system", "unknown") for sp in sps)
    bodies      = [sp.get("bodies_in_box") for sp in sps if sp.get("bodies_in_box")]
    zone_detail = {}
    for sp in sps:
        z = sp.get("zone", sp.get("type", "unknown"))
        if z not in zone_detail:
            zone_detail[z] = {"count": 0, "deliveries": [], "outcomes": []}
        zone_detail[z]["count"]      += 1
        zone_detail[z]["deliveries"].append(sp.get("delivery", "unknown"))
        zone_detail[z]["outcomes"].append(sp.get("outcome", "unknown"))
    for z in zone_detail:
        d = zone_detail[z]
        d["most_common_delivery"] = Counter(d["deliveries"]).most_common(1)[0][0] if d["deliveries"] else None
        d["most_common_outcome"]  = Counter(d["outcomes"]).most_common(1)[0][0]  if d["outcomes"]   else None
        del d["deliveries"], d["outcomes"]
    top_del_count  = max(by_delivery.values()) if by_delivery else 0
    top_del_type   = max(by_delivery, key=by_delivery.get) if by_delivery else None
    consistency_pct= round(top_del_count / len(sps) * 100) if sps else 0

    profile = {
        "total_set_pieces":        len(sps),
        "by_zone":                 dict(by_zone),
        "by_delivery_type":        dict(by_delivery),
        "by_outcome":              dict(by_outcome),
        "by_marking_system":       dict(by_marking),
        "avg_bodies_in_box":       round(sum(bodies)/len(bodies), 1) if bodies else None,
        "zone_detail":             zone_detail,
        "dominant_delivery":       top_del_type,
        "dominant_delivery_pct":   consistency_pct,
    }
    return profile, len(sps), ["direct"]


def calc_attacking_support_score(passes, summary):
    threat_seqs  = [s for s in passes.get("sequences", []) if s.get("outcome") in ("shot", "cross")]
    shot_lengths = [s.get("sequence_length") for s in summary.get("shots_for", []) if s.get("sequence_length")]
    all_lengths  = [s.get("length", s.get("sequence_length")) for s in threat_seqs
                    if s.get("length") or s.get("sequence_length")] + shot_lengths
    all_lengths  = [l for l in all_lengths if l is not None and l > 0]
    if len(all_lengths) < 2:
        return None, len(all_lengths), ["suggestive"]
    avg = round(sum(all_lengths) / len(all_lengths), 1)
    # Cross-reference with shot zones if available in summary
    shots_for = summary.get("shots_for", [])
    box_shots  = sum(1 for s in shots_for
                     if s.get("origin_row") in ("six_yard_box","penalty_spot"))
    edge_shots = sum(1 for s in shots_for
                     if s.get("origin_row") in ("edge_of_box","outside_box"))
    cat = support_depth_category(avg)
    summary_str = f"avg {avg} passes before shot ({cat})"
    if box_shots + edge_shots > 0:
        box_pct = round(box_shots / (box_shots + edge_shots) * 100)
        summary_str += f"; {box_pct}% of shots from inside or near the box"

    return {"avg_passes_before_shot": avg,
            "support_depth": cat,
            "box_shots": box_shots,
            "edge_shots": edge_shots,
            "samples": len(all_lengths),
            "summary": summary_str
            }, len(all_lengths), ["repeated_pattern"]


# -- Player metrics ------------------------------------------------------------

def calc_player_metrics(summary, confirmations):
    observations = summary.get("individual_observations", [])
    if not observations:
        return []
    by_player = {}
    for obs in observations:
        pid = obs.get("player", "unknown")
        if pid not in by_player:
            by_player[pid] = {"player": pid, "position": obs.get("position"),
                               "ratings": [], "obs": []}
        by_player[pid]["ratings"].append(obs.get("rating", 3))
        by_player[pid]["obs"].append(obs.get("observation", ""))
    player_metrics = []
    for pid, data in by_player.items():
        ratings = data["ratings"]
        if not ratings:
            continue
        avg_r = sum(ratings) / len(ratings)
        var   = sum((r - avg_r) ** 2 for r in ratings) / len(ratings)
        escalated = [i for i in confirmations.get("items", [])
                     if i.get("analysis_scope") == "player"
                     and pid in str(i.get("player_involved", ""))
                     and i.get("status") == "resolved"]
        player_metrics.append({
            "player":                      pid,
            "position":                    data["position"],
            "player_role_consistency":     round(max(0.0, 1.0 - (var / 4.0)), 2),
            "player_positioning_stability":round(avg_r / 5.0, 2),
            "observations_count":          len(ratings),
            "evidence_tier":               "escalated_confirmation" if escalated else "repeated_pattern",
            "fps_context":                 "1fps for role/positioning; 3-5fps for decision/movement",
        })
    return player_metrics


# -- Main build function -------------------------------------------------------

def calc_fitness_trajectory(summary):
    """
    Compare pressing intensity in the first third of windows vs the last third.

    A team pressing MORE late in the match = superior conditioning or tactical shift.
    A team pressing LESS late = tiring or reacting to scoreline.

    Returns:
        trajectory: "rising" / "falling" / "stable"
        early_avg: pressing score in first 33% of windows
        late_avg:  pressing score in last 33% of windows
        delta:     late - early (positive = rising)
    """
    pressing = summary.get("pressing_by_window", [])
    scores   = [w.get("avg_score") for w in pressing if w.get("avg_score") is not None]
    if len(scores) < 6:
        return None, 0, 0, 0, ["suggestive"]

    third = max(1, len(scores) // 3)
    early_avg = round(sum(scores[:third]) / third, 2)
    late_avg  = round(sum(scores[-third:]) / third, 2)
    delta     = round(late_avg - early_avg, 2)

    if delta > 0.8:
        trajectory = "rising"
    elif delta < -0.8:
        trajectory = "falling"
    else:
        trajectory = "stable"

    return trajectory, early_avg, late_avg, delta, ["repeated_pattern"]



def calc_line_reaction_to_goals(summary):
    """
    Compare average defensive line height when losing vs when level or winning.

    This tells you: does this team drop deeper after conceding?
    A team that drops 8m after conceding has a very different shape to exploit
    immediately after you score.

    Returns dict with avg_pct per match_state, or None if data insufficient.
    """
    line_data  = summary.get("line_height_by_window", [])
    state_data = summary.get("match_state_by_window", [])

    if not line_data or not state_data:
        return None, 0, ["suggestive"]

    # Match windows by index (both ordered chronologically)
    state_by_window = {s.get("window"): s.get("match_state") for s in state_data}

    by_state = {}
    for w in line_data:
        state = state_by_window.get(w.get("window"), "unknown")
        pct   = w.get("avg_pct")
        if pct is not None and state != "unknown":
            by_state.setdefault(state, []).append(pct)

    if len(by_state) < 2:
        return None, 0, ["suggestive"]

    result = {}
    for state, vals in by_state.items():
        result[state] = {
            "avg_pct":     round(sum(vals)/len(vals), 1),
            "avg_m":       round(sum(vals)/len(vals)/100*105, 1),
            "windows":     len(vals),
        }

    # Compute the most interesting delta: losing vs level
    losing_m = result.get("losing", {}).get("avg_m")
    level_m  = result.get("level", {}).get("avg_m")
    winning_m= result.get("winning", {}).get("avg_m")

    delta_losing_vs_level = None
    if losing_m is not None and level_m is not None:
        delta_losing_vs_level = round(losing_m - level_m, 1)

    result["delta_losing_vs_level_m"] = delta_losing_vs_level
    result["summary"] = _line_reaction_summary(result, delta_losing_vs_level)

    return result, len(line_data), ["repeated_pattern"]


def _line_reaction_summary(result, delta):
    parts = []
    for state in ("level", "winning", "losing"):
        if state in result:
            parts.append(f"{state}: {result[state]['avg_m']}m")
    summary = "Line height by match state -- " + ", ".join(parts)
    if delta is not None:
        if delta < -3:
            summary += f"; dropped {abs(delta)}m deeper when losing"
        elif delta > 3:
            summary += f"; pushed {delta}m higher when losing (pressing for equaliser)"
        else:
            summary += "; minimal change with scoreline"
    return summary



# -----------------------------------------------------------------------------
# CONFIDENCE FILTER
# Three levels applied to metrics, findings, and observations.
#
# Level 1 -- Confirmed: only high-confidence, sufficient-sample metrics.
#             No hedging. For head coach / match prep use.
# Level 2 -- Standard (default): all non-context_only metrics with sample notes.
#             For analyst + coaching staff weekly use.
# Level 3 -- Full: everything including context_only and suggestive.
#             For analyst QA and pipeline review.
# -----------------------------------------------------------------------------

def assign_confidence_level(metric: dict) -> int:
    """
    Assign the minimum confidence_level a reader needs to see this metric.
    Returns 1 (confirmed), 2 (standard), or 3 (full/review only).

    Level 1: sufficient sample, confidence >= 0.5, not context_only
    Level 2: any non-context_only metric with confidence >= 0.3 or preliminary sample
    Level 3: context_only, unavailable, or very low confidence
    """
    ctx     = metric.get("context_only", False)
    status  = metric.get("sample_status", "context_only")
    conf    = metric.get("confidence", 0.0) or 0.0
    # evidence_tier is a single string on the metric dict
    tier    = metric.get("evidence_tier", "suggestive")
    value   = metric.get("value")

    # Unavailable or context_only = Level 3 only
    if ctx or value is None:
        return 3

    if status == "context_only":
        return 3

    # Preliminary samples = Level 2 minimum
    if status == "preliminary":
        return 2

    # Sufficient sample + decent confidence = Level 1 or 2
    if status == "sufficient":
        if conf >= 0.5:
            return 1
        if conf >= 0.3:
            return 2
        return 3

    return 2


def filter_metrics(metrics: list, confidence_level: int) -> list:
    """
    Filter metrics list to only include those visible at the given level.
    Adds 'confidence_level' field to each metric.
    """
    result = []
    for m in metrics:
        level = assign_confidence_level(m)
        m = dict(m)
        m["confidence_level"] = level
        if level <= confidence_level:
            result.append(m)
    return result


def filter_observations(observations: list, confidence_level: int) -> list:
    """
    Filter individual_observations by confidence level.
    Level 1: repeated/consistent, high confidence
    Level 2: any frequency, medium/high confidence
    Level 3: everything
    """
    if confidence_level >= 3:
        return observations

    result = []
    for obs in observations:
        freq = obs.get("frequency", "single")
        conf = obs.get("confidence", "low")

        if confidence_level == 1:
            if freq in ("repeated", "consistent") and conf == "high":
                result.append(obs)
        elif confidence_level == 2:
            if conf in ("medium", "high"):
                result.append(obs)
    return result


def filter_findings(findings: list, confidence_level: int) -> list:
    """
    Filter findings by evidence tier and result family status.
    Level 1: direct or escalated_confirmation, allowed only
    Level 2: direct, repeated_pattern, escalated_confirmation (downgraded allowed with note)
    Level 3: everything including suggestive
    """
    if confidence_level >= 3:
        return findings

    result = []
    for f in findings:
        tier   = f.get("evidence_tier", "suggestive")
        status = f.get("result_family_status", "downgraded")

        if confidence_level == 1:
            if tier in ("direct", "escalated_confirmation") and status == "allowed":
                result.append(f)
        elif confidence_level == 2:
            if tier in ("direct", "repeated_pattern", "escalated_confirmation"):
                result.append(f)
    return result



def build_deep_skill_metrics(match_dir, team_label="both", confidence_level=2):
    # Fix 33b: parameter renamed from legacy focus name. Pipeline produces
    # reports for both teams independently; default "both" applies metrics
    # generically. Metric-internal team identity uses summary["home_team"] /
    # summary["away_team"] rather than a single focus.

    data          = load_pipeline_data(match_dir)
    summary       = data["summary"]
    passes        = data["passes"]
    gates         = data["gates"]
    source_prof   = data["source"]
    confirmations = data["confirmations"]
    source_type   = source_prof.get("source_type", "unknown")
    global_cap    = SOURCE_GLOBAL_CAP.get(source_type, 0.5)
    total_windows = summary.get("windows_complete", 1)

    metrics = []

    def make_metric(name, scope, required_families, value, value_type,
                    evidence_tiers, base_confidence, windows_contributing,
                    calculation_basis, traceable_to=None, subject_team=team_label,
                    fps_ctx=None, sample_n=None, sample_status_override=None):
        confidence, status, limitation = compute_metric_confidence(
            base_confidence, required_families, gates,
            evidence_tiers, windows_contributing, source_type)

        # Generate prose interpretation and sample status
        sample_count = sample_n if sample_n is not None else windows_contributing
        s_status     = sample_status_override if sample_status_override is not None else \
                       sample_status(name, sample_count,
                                    extra=value if isinstance(value, dict) else None)
        prose        = build_prose_interpretation(name, value, sample_count)

        # context_only metrics should not appear in tactical report sections
        is_context_only = (s_status == "context_only")

        return {
            "metric_name":                name,
            "analysis_scope":             scope,
            "subject_team":               subject_team,
            "value":                      value,
            "value_type":                 value_type,
            "supporting_result_families": required_families,
            "evidence_tier":              worst_evidence_tier(evidence_tiers),
            "confidence":                 confidence if confidence is not None else 0.0,
            "result_family_status":       status,
            "severely_limited":           (confidence is not None and confidence < 0.2) or value is None,
            "limitation_note":            limitation,
            "windows_contributing":       windows_contributing,
            "fps_context":                fps_ctx or "1fps",
            # Fix 44A: same key-drift as Fix 38 / Fix 39B. Broadcast and
            # tactical_wide_static profiles store the prose under "notes".
            "source_limitations":         (source_prof.get("source_limitations_note")
                                            or source_prof.get("notes", "")),
            "calculation_basis":          calculation_basis,
            "traceable_to":               traceable_to or required_families,
            "prose_interpretation":       prose,
            "sample_status":              s_status,
            "context_only":               is_context_only,
        }

    # -- System-level (5) -----------------------------------------------------

    v, avg_h_m, w, t, compact_cat = calc_compactness(summary)
    metrics.append(make_metric("compactness_score", "match", ["shape", "pressing"],
        {"score": v, "category": compact_cat,
         "avg_line_height_m": avg_h_m,
         "summary": f"defensive compactness: {compact_cat} (avg line {avg_h_m}m from own goal)"},
        "profile", t, 0.75, w,
        "Line height stability (60%) + pressing intensity (40%)",
        traceable_to=["line_height_by_window", "pressing_by_window"]))

    avg_10, avg_01, w, t = calc_pressing_intensity(summary)
    press_cat = pressing_category(avg_10)
    metrics.append(make_metric("pressing_intensity_score", "match", ["pressing"],
        {"avg_score_0_10": avg_10, "avg_score_0_1": avg_01,
         "category": press_cat,
         "summary": f"{avg_10}/10 ({press_cat})"},
        "profile", t, 0.85, w,
        "Average pressing score across windows (0-10 scale) with categorical label",
        fps_ctx="1fps structural; 3fps confirmation"))

    v, total_seqs, prog, threats, ft, prog_rate, ft_rate, conv_rate, t = calc_build_up_effectiveness(passes)
    metrics.append(make_metric("build_up_effectiveness_score", "match", ["build_up"],
        {"progressive_sequences": prog,
         "final_third_sequences": ft,
         "threat_sequences":      threats,
         "total_sequences":       total_seqs,
         "final_third_rate_pct":  round(ft_rate * 100, 1),
         "conversion_rate_pct":   round(conv_rate * 100, 1),
         "summary": f"{round(ft_rate*100)}% of sequences reached the final third; "
                    f"{round(conv_rate*100, 1)}% ended in shot/cross"},
        "profile", t, 0.75, total_seqs,
        "Progressive sequences and shot/cross-ending sequences as percentage of total",
        traceable_to=["pass_sequences"]))

    v, backward_per_win, w, t = calc_rest_defence_security(summary)
    rest_cat = rest_defence_category(backward_per_win)
    # Fix 55c: rephrase per-window prose so synthesis cannot import it
    # verbatim. The calculation_basis is read by synthesis as context;
    # "per window" leaks WINDOW LANGUAGE RULE violations into reports.
    metrics.append(make_metric("rest_defence_security_score", "match", ["rest_defence"],
        {"avg_backward_shifts_per_phase": backward_per_win,
         "category": rest_cat,
         "summary": f"{backward_per_win} avg backward line shifts ({rest_cat})"},
        "profile", t, 0.75, w,
        "Average defensive line shifts per possession phase with categorical label",
        traceable_to=["line_height_by_window"]))

    v, w, t = calc_line_height_range(summary)
    if v:
        v["summary"] = (f"highest: {v['highest_m']}m ({v['highest_category']}), "
                        f"lowest: {v['lowest_m']}m ({v['lowest_category']}), "
                        f"range: {v['range_m']}m")
    metrics.append(make_metric("line_height_range", "match", ["shape"],
        v, "profile", t, 0.80, w,
        "Highest and lowest line height in metres from own goal with categorical bands",
        traceable_to=["line_height_by_window"]))

    # -- Behavioural (6) ------------------------------------------------------

    v, w, t, method = calc_width_usage(passes)
    metrics.append(make_metric("width_usage_score", "match", ["spacing"],
        {"score": v, "method": method,
         "summary": (f"{round(v*100)}% of sequences used wide channels"
                     + (" (likely undercount -- ball-follow footage)" if method=="cross_outcomes_proxy" else ""))
         },
        "profile", t, 0.55, w,
        f"Wide zone sequences / total ({method})",
        traceable_to=["pass_sequences"]))

    v, w, t = calc_halfspace_occupation(passes)
    metrics.append(make_metric("halfspace_occupation_score", "match", ["phase", "territory"],
        v, "numeric_0_1" if v is not None else "unavailable", t, 0.65, w,
        "Proportion of sequences involving half-space zones (requires granular zone encoding)",
        traceable_to=["pass_sequences"]))

    v, w, t = calc_transition_efficiency(summary, confirmations)
    te_val = v.get("score") if isinstance(v, dict) else v
    metrics.append(make_metric("transition_efficiency_score", "match", ["transitions"],
        v, "profile" if v is not None else "unavailable", t, 0.70, w,
        "Attack and defence transition rates (split by direction)",
        sample_n=w,
        fps_ctx="5fps minimum for confirmation",
        traceable_to=["flagged_moments", "key_moments", "confirmation_queue"]))

    v, w, t = calc_chance_creation_profile(summary, passes)
    metrics.append(make_metric("chance_creation_profile", "match", ["chance_patterns"],
        v, "profile", t, 0.80, w,
        "Three-layer profile: origin zone -> route -> end point for each observed chance",
        sample_n=(v["total_chances"] if v else 0),
        fps_ctx="3fps for confirmation",
        traceable_to=["shots_for", "pass_sequences"]))

    v, top_route, top_count, total_seqs_p, t = calc_pattern_reliability(passes)
    pattern_val = v
    metrics.append(make_metric("pattern_reliability_score", "match", ["build_up", "phase"],
        {"score": v,
         "top_route": top_route,
         "top_route_count": top_count,
         "total_sequences": total_seqs_p,
         "top_route_pct": round(v * 100, 1),
         "summary": (f"{round(v*100)}% of sequences used route: {top_route}"
                     if top_route else "insufficient data")},
        "profile", t, 0.75, total_seqs_p,
        "Most-common zone-to-zone route as percentage of total sequences",
        traceable_to=["pass_sequences"]))

    v, w, distinct, t = calc_build_up_route_diversity(passes)
    div_val = v
    metrics.append(make_metric("build_up_route_diversity", "match", ["build_up"],
        {"score": v, "distinct_routes": distinct,
         "summary": f"{distinct} distinct passing routes observed across the match"},
        "profile", t, 0.75, w,
        "Distinct zone-to-zone route pairs / 12 (recalibrated ceiling)",
        traceable_to=["pass_sequences"]))


    v, pred_cat, t = calc_predictability(pattern_val, div_val)
    metrics.append(make_metric("predictability_score", "match", ["build_up", "phase"],
        {"score": v, "category": pred_cat,
         "summary": f"build-up predictability: {pred_cat}"} if v is not None else None,
        "profile" if v is not None else "unavailable", t, 0.70, total_windows,
        "Pattern reliability (60%) + inverse route diversity (40%); "
        "categories: highly_predictable / predictable / mixed / varied / unpredictable",
        traceable_to=["pattern_reliability_score", "build_up_route_diversity"]))

    v, w, t = calc_pressing_trigger_consistency(summary)
    metrics.append(make_metric("pressing_trigger_consistency", "match", ["pressing"],
        v, "profile", t, 0.75, w,
        "Consistency of press trigger type across observed pressing moments",
        sample_n=w,
        fps_ctx="1fps observation; 3fps confirmation",
        traceable_to=["pressing_by_window", "key_moments", "flagged_moments"]))

    v, w, t = calc_set_piece_delivery_profile(summary)
    if v:
        top_zone = max(v["by_zone"], key=v["by_zone"].get) if v["by_zone"] else "unknown"
        top_del  = max(v["by_delivery_type"], key=v["by_delivery_type"].get) if v["by_delivery_type"] else "unknown"
        v["summary"] = (f"{v['total_set_pieces']} set pieces: "
                        f"most from {top_zone.replace('_',' ')}, "
                        f"most common delivery: {top_del.replace('_',' ')}")
    metrics.append(make_metric("set_piece_delivery_profile", "match", ["set_pieces"],
        v, "profile", t, 0.85, w,
        "Delivery type, bodies in box, marking system and outcome by zone for all set pieces",
        sample_n=(v["total_set_pieces"] if v else 0),
        fps_ctx="1fps structure; 3fps delivery confirmation",
        traceable_to=["set_pieces"]))

    v, w, t = calc_attacking_support_score(passes, summary)
    # -- Fitness trajectory ----------------------------------------------------
    ft_traj, ft_early, ft_late, ft_delta, ft_t = calc_fitness_trajectory(summary)
    if ft_traj is not None:
        press_windows = len(summary.get("pressing_by_window", []))
        ft_status     = grade_new_metric_sample("fitness_trajectory",
                            {"trajectory":ft_traj}, summary)
        metrics.append(make_metric("fitness_trajectory", "match", ["pressing"],
            {"trajectory": ft_traj, "early_avg": ft_early,
             "late_avg": ft_late, "delta": ft_delta},
            "profile", ft_t, 0.60, press_windows,
            "Pressing intensity trend: first third vs last third of windows",
            sample_n=press_windows,
            sample_status_override=ft_status))

    # -- Line reaction to goals -------------------------------------------------
    lr_val, lr_w, lr_t = calc_line_reaction_to_goals(summary)
    if lr_val is not None and len(summary.get("match_state_by_window",[])) >= 5:
        lr_status = grade_new_metric_sample("line_reaction_to_goals", lr_val, summary)
        metrics.append(make_metric("line_reaction_to_goals", "match", ["rest_defence", "shape"],
            lr_val,
            "profile", lr_t, 0.60, lr_w,
            "Defensive line height compared across match states (level/winning/losing)",
            sample_n=lr_w,
            sample_status_override=lr_status))

    metrics.append(make_metric("attacking_support_score", "match", ["chance_patterns", "build_up"],
        v, "profile", t, 0.70, w,
        "Average sequence length before shots/crosses (longer = better collective support)",
        sample_n=(v["samples"] if v else 0),
        fps_ctx="3fps for confirmation",
        traceable_to=["pass_sequences", "shots_for"]))

    # -- Player metrics --------------------------------------------------------

    for pm in calc_player_metrics(summary, confirmations):
        confidence, status, limitation = compute_metric_confidence(
            0.70, ["player_role", "player_positioning"], gates,
            [pm["evidence_tier"]], pm["observations_count"], source_type)
        base = {"analysis_scope": "player", "subject_team": team_label,
                "subject_player": pm["player"], "subject_position": pm["position"],
                "evidence_tier": pm["evidence_tier"],
                "confidence": confidence if confidence is not None else 0.0,
                "result_family_status": status,
                "severely_limited": confidence is not None and confidence < 0.2,
                "limitation_note": limitation,
                "windows_contributing": pm["observations_count"],
                "fps_context": pm["fps_context"],
                # Fix 44A: see line ~1812 for rationale.
                "source_limitations": (source_prof.get("source_limitations_note")
                                        or source_prof.get("notes", "")),
                "traceable_to": ["individual_observations"],
                "supporting_result_families": ["player_role", "player_positioning"]}
        for mname, val, basis in [
            ("player_role_consistency",     pm["player_role_consistency"],
             "Rating variance across observations (1 - normalised variance)"),
            ("player_positioning_stability",pm["player_positioning_stability"],
             "Average rating normalised to 0-1"),
        ]:
            metrics.append({**base, "metric_name": mname, "value": val,
                             "value_type": "numeric_0_1", "calculation_basis": basis})

    # ── Phase 4 new metrics ──────────────────────────────────────────────────────

    # Possession balance (from possession_by_window, populated when both teams logged)
    poss_val = calc_possession_balance(summary)
    if poss_val:
        metrics.append(make_metric(
            "possession_balance", "match", ["build_up"], poss_val, "profile",
            "repeated_pattern", 0.55, len(summary.get("possession_by_window",[])),
            "Focus team possession share from both-teams sequence counts",
            sample_n=len(summary.get("possession_by_window",[]))))

    # Direct play ratio (from is_long_ball on sequences)
    _sq_path = os.path.join(match_dir, "pass_sequences.json")
    if os.path.exists(_sq_path):
        with open(_sq_path, encoding="utf-8") as _f: _sq_data = json.load(_f)
        _dp_seqs = _sq_data.get("sequences", [])
        dp_val = calc_direct_play_ratio({"pass_sequences": {"sequences": _dp_seqs}})
        if dp_val and dp_val.get("long_ball_count", 0) > 0:
            metrics.append(make_metric(
                "direct_play_ratio", "match", ["build_up"], dp_val, "profile",
                "repeated_pattern", 0.55, dp_val.get("long_ball_count", 0),
                "Proportion of sequences that are long balls; second ball win rate",
                sample_n=dp_val.get("long_ball_count", 0)))

    # Aerial dominance (from duels[] array)
    ad_val = calc_aerial_dominance({"duels": summary.get("duels",[]),
                                    "home_team": summary.get("home_team","")})
    if ad_val and ad_val.get("overall_duels", 0) > 0:
        metrics.append(make_metric(
            "aerial_dominance", "match", ["phase"], ad_val, "profile",
            "repeated_pattern", 0.55, ad_val.get("overall_duels", 0),
            "Aerial and ground duel win rate from duels[] array",
            sample_n=ad_val.get("overall_duels", 0)))

    # GK distribution consistency (from gk_kicks[] array)
    gk_val = calc_gk_distribution_consistency({"gk_kicks": summary.get("gk_kicks",[]),
                                                "home_team": summary.get("home_team","")})
    if gk_val and gk_val.get("total_kicks", 0) > 0:
        metrics.append(make_metric(
            "gk_distribution_consistency", "match", ["build_up"], gk_val, "profile",
            "repeated_pattern", 0.55, gk_val.get("total_kicks", 0),
            "GK target zone and length consistency from gk_kicks[] array",
            sample_n=gk_val.get("total_kicks", 0)))

    # Momentum score (composite per window)
    mom_windows = calc_momentum_by_window(summary)
    valid_mom   = [w for w in mom_windows if w.get("momentum") is not None]
    if valid_mom:
        avg_mom = round(sum(w["momentum"] for w in valid_mom)/len(valid_mom), 3)
        metrics.append(make_metric(
            "momentum_score", "match", ["pressing","build_up","shape"],
            {"avg": avg_mom,
             "by_window": valid_mom,
             "peak": max(valid_mom, key=lambda w: w["momentum"])["window"],
             "low":  min(valid_mom, key=lambda w: w["momentum"])["window"]},
            "profile", "repeated_pattern", 0.50, len(valid_mom),
            "Composite momentum: pressing 35% + possession 40% + line height 25%",
            sample_n=len(valid_mom)))

        # -- Output ----------------------------------------------------------------

    avg_confidence = round(sum(m["confidence"] for m in metrics)/len(metrics), 2) if metrics else 0.0
    unavailable    = [m["metric_name"] for m in metrics if m.get("value") is None]
    low_conf       = [m["metric_name"] for m in metrics if m.get("confidence", 1) < 0.3]
    severe         = [m["metric_name"] for m in metrics if m.get("severely_limited")]
    context_only   = [m["metric_name"] for m in metrics if m.get("context_only")]


    # Apply confidence filter to metrics before output
    for m in metrics:
        m["confidence_level"] = assign_confidence_level(m)
    filtered = filter_metrics(metrics, confidence_level)

    output = {
        "match":                    summary.get("match", "unknown"),
        "team_label":               team_label,
        "source_type":              source_type,
        "global_confidence_cap":    global_cap,
        "total_metrics":            len(metrics),
        "active_metrics":           len(metrics) - len(unavailable),
        "unavailable_metrics":      unavailable,
        "context_only_metrics":     context_only,
        "low_confidence_metrics":   low_conf,
        "severely_limited_metrics": severe,
        "avg_confidence":           avg_confidence,
        "confidence_level_applied":  confidence_level,
        "metrics":                  filtered,
        "generated_at":             datetime.now().isoformat(),
    }

    out_path = os.path.join(match_dir, "deep_skill_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n{chr(45)*55}")
    print(f"  Deep skill metrics written")
    print(f"  Source:          {source_type} (cap: {global_cap})")
    print(f"  Total metrics:   {len(metrics)}")
    print(f"  Active:          {len(metrics) - len(unavailable)}")
    print(f"  Unavailable:     {unavailable}")
    print(f"  Context only:    {context_only}")
    print(f"  Low conf (<0.3): {low_conf}")
    print(f"  Avg confidence:  {avg_confidence}")
    print(f"{chr(45)*55}")
    return output


if __name__ == "__main__":
    match_dir  = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    team_label = sys.argv[2] if len(sys.argv) > 2 else "both"
    if not os.path.isdir(match_dir):
        print(f"Error: '{match_dir}' is not a valid directory.")
        sys.exit(1)
    build_deep_skill_metrics(match_dir, team_label)
