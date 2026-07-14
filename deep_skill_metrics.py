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
import statistics  # v3 Step 8 merge: used by _metric_compactness_geometry
import os
import sys
import re
from datetime import datetime
from collections import Counter
from pipeline_accessors import get_marking
from pipeline_schemas import stamp_schema_version


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


def rest_defence_category(backward_rate):
    """
    Categorise from the fraction of window-to-window transitions where the
    defensive line dropped deeper (backward_rate, 0-1).

    Step 28: rescaled from the old "avg narrated shifts per window" scale
    (unbounded -- a window could in theory have several narrated shifts) to
    this bounded 0-1 rate (each transition is either backward or not, so the
    rate can never exceed 1.0). Keeping the old thresholds (0.5/1.0/2.0/3.5)
    unchanged after switching to a bounded signal would have made the
    "vulnerable"/"very_vulnerable" bands permanently unreachable -- silently
    just as broken as the bug this rescale is fixing.
    """
    if backward_rate is None: return "unknown"
    if backward_rate < 0.15:  return "very_secure"
    if backward_rate < 0.30:  return "secure"
    if backward_rate < 0.50:  return "moderate"
    if backward_rate < 0.70:  return "vulnerable"
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


def _momentum_component(press, line_pct, poss_pct):
    """
    Shared per-team, per-window composite math. Pulled out into its own
    function (rather than written twice, once for home and once for away)
    for the exact DRY reason we wrote up in the Step 19 glossary entry:
    two copies of the same formula is how one of them silently goes stale
    the next time the weighting or normalisation needs a fix.

    press, line_pct: 0-10 / 0-100 raw readings for ONE team, or None if
    that team's reading wasn't observable in this window.
    poss_pct: 0-100 possession share for ONE team, or None if the
    possession_by_window entry for this window has no usable data at all
    (both teams get None together -- possession is always a same-window,
    both-teams-or-neither reading, never lopsided).

    Returns (momentum, components_dict) -- momentum is None only when BOTH
    press and line_pct are None (i.e. genuinely nothing observable for
    this team in this window; matches the original single-team logic's
    "nothing to compute from" case, just applied per team now instead of
    once for the whole match).
    """
    if press is None and line_pct is None:
        return None, {}

    # Normalise each raw reading to a common 0-1 scale so the weighted sum
    # below is comparing like with like. A team with no reading for a
    # given component falls back to 0.5 (neutral -- "assume average",
    # not "assume zero") rather than dropping that component silently,
    # which is exactly how the original single-team version behaved.
    press_n = min(press / 10.0, 1.0) if press is not None else 0.5
    line_n  = min(line_pct / 60.0, 1.0) if line_pct is not None else 0.5
    poss_n  = poss_pct / 100.0 if poss_pct is not None else 0.5

    # Weighted composite: press 35%, possession 40%, line height 25%.
    # When possession has no data at all this window, its 40% weight is
    # redistributed onto press/line rather than silently zeroing it out --
    # same rebalancing the original function already did.
    weights = (0.35, 0.40, 0.25) if poss_pct is not None else (0.55, 0.0, 0.45)
    momentum = round(press_n * weights[0] + poss_n * weights[1] + line_n * weights[2], 3)

    return momentum, {
        "pressing":    round(press_n, 3),
        "possession":  round(poss_n, 3) if poss_pct is not None else None,
        "line_height": round(line_n, 3),
    }


def calc_momentum_by_window(summary: dict, match_config: dict) -> list:
    """
    Per-window momentum score, split by team (home vs away).

    WHY split by team at all: synthesis_agent.py's report prompt has a
    dedicated "Tactical Variation and Substitutions" section that's asked
    to reason about whether a substitution shifted the game's momentum.
    A single match-wide number (the old behaviour: subject_team="both",
    one blended avg) can only ever say "the match got more/less intense"
    -- it cannot say WHICH team that shift favoured, which is the one
    thing a report actually needs to credit a substitution with having
    worked (or not). Splitting the same formula per team, instead of
    combining the two sides into one number first, is what makes that
    attribution possible.

    WHY read line_height_m_by_window instead of line_height_by_window:
    checked directly against the real Gorleston match -- line_height_
    by_window[*].avg_pct is None for all 21 windows, every single one.
    accumulator.py's own Step-541 comment already documents why: the
    structural vision agent's schema was changed to emit home_height_pct
    / away_height_pct (see pipeline_runner_v2.py's defensive_line block),
    but line_height_by_window's aggregation code was never updated and
    still only ever looks for the old, no-longer-emitted avg_pct key.
    line_height_m_by_window was added later specifically to carry the
    real home/away split forward -- it's populated for all 21 real
    windows. Momentum's line-height component has therefore been running
    on the 0.5 neutral fallback for every window, in every match, since
    it was written; splitting momentum per team requires per-team line
    height data anyway, so this dead-field bug gets fixed as a direct
    side effect of doing the split correctly, not as a separate fix.

    WHY match_config is needed: pressing_by_window and line_height_m_
    by_window already label their readings home_intensity/away_intensity
    and home_height_pct/away_height_pct directly -- no resolution needed.
    possession_by_window does NOT: it labels its single number focus_pct
    (whichever team match_config.json's optional focus_team names, or
    home_team if focus_team isn't set -- see pipeline_runner_v2.py line
    1761's identical resolution: `focus_team = mc.get("focus_team") or
    home_team`). Reusing that exact resolution rule here (rather than
    inventing a different one) is what turns focus_pct into a real
    home_pct/away_pct pair.
    """
    pb_win = {w["window"]: w for w in summary.get("pressing_by_window", [])
              if w.get("window")}
    lh_win = {w["window"]: w for w in summary.get("line_height_m_by_window", [])
              if w.get("window")}
    po_win = {w["window"]: w for w in summary.get("possession_by_window", [])
              if w.get("window")}

    home_team = summary.get("home_team", "")
    away_team = summary.get("away_team", "")
    # Same fallback rule pipeline_runner_v2.py already uses when building
    # the extraction prompt itself -- an explicit match_config.focus_team
    # wins, otherwise the home team is treated as focus by convention.
    focus_team = (match_config or {}).get("focus_team") or home_team
    focus_is_home = (focus_team == home_team)

    windows_seen = sorted(set(list(pb_win) + list(lh_win)), key=str)
    result = []
    for win in windows_seen:
        pb = pb_win.get(win, {})
        lh = lh_win.get(win, {})
        po = po_win.get(win, {})

        press_home = pb.get("home_intensity")
        press_away = pb.get("away_intensity")
        line_home  = lh.get("home_height_pct")
        line_away  = lh.get("away_height_pct")

        # focus_pct is a single number FROM the focus team's perspective.
        # Turn it into a real home_pct/away_pct pair using the same
        # focus-team resolution as above. Possession is always a
        # both-teams-or-neither reading for a given window (there's no
        # such thing as "we know the home team's possession share but not
        # the away team's" -- they're complements of the same number),
        # so both come from focus_pct together, or both are None together.
        focus_pct = po.get("focus_pct")
        if focus_pct is None:
            poss_home = poss_away = None
        elif focus_is_home:
            poss_home, poss_away = focus_pct, round(100.0 - focus_pct, 1)
        else:
            poss_away, poss_home = focus_pct, round(100.0 - focus_pct, 1)

        home_momentum, home_components = _momentum_component(press_home, line_home, poss_home)
        away_momentum, away_components = _momentum_component(press_away, line_away, poss_away)

        result.append({
            "window":         win,
            "home_momentum":  home_momentum,
            "away_momentum":  away_momentum,
            "home_components": home_components,
            "away_components": away_components,
        })
    return result


# WHY an anchored regex instead of the old str.replace()+str.split("-")
# parser: a window label like "2H_10-00_15-00" has a hyphen INSIDE each
# minute-second pair, not just between the pair and the next pair. The
# old code split on every hyphen in the string, so "05-00" (5 minutes,
# 0 seconds) got torn into "05" and "00" as if they were two unrelated
# numbers -- and because int("00") parses without error, the surrounding
# bare `except: pass` had nothing to catch. It silently computed an
# `end` of 0 for every window in the match, every time, which is why
# calc_substitution_impact() below found zero matching windows for
# every real substitution once the earlier crash bug was fixed --
# checked directly: window_index() returned None for all 10 real
# Gorleston subs before this rewrite. An anchored regex either matches
# the exact 4-number shape or returns None outright -- no partial,
# silently-wrong parse is possible.
_WINDOW_LABEL_RE = re.compile(r"^([12])H_(\d+)-(\d+)_(\d+)-(\d+)$")


def _parse_window_bounds(window_label: str):
    """
    Parse a window label into (half, start_minute, end_minute) as a float
    minute pair, e.g. "2H_10-00_15-00" -> (2, 10.0, 15.0) and
    "1H_45-00_48-13" -> (1, 45.0, 48.2166...). The MM-SS trailing pair is
    minutes-and-seconds of the SAME instant, not two independent minute
    readings, so seconds are divided by 60 and added, never read as a
    second minute value.

    Returns None if the label doesn't match the expected shape, rather
    than guessing -- callers must treat None as "skip this window", the
    same explicit-skip contract the rest of this file already uses for
    unresolvable data.
    """
    m = _WINDOW_LABEL_RE.match(window_label or "")
    if not m:
        return None
    half, start_min, start_sec, end_min, end_sec = (int(g) for g in m.groups())
    start = start_min + start_sec / 60.0
    end   = end_min + end_sec / 60.0
    return half, start, end


def _resolve_half_and_within_minute(elapsed_minute, match_boundaries=None):
    """
    Decide which half a match-elapsed minute (e.g. a substitution's
    time.elapsed) falls in, and re-base it onto THAT half's own clock --
    which is what window labels are anchored to ("2H_10-00_15-00" means
    minutes 10-15 of the second half, not match minutes 55-60).

    WHY this can't just be "elapsed <= 45 -> first half": real stoppage
    time runs PAST 45 minutes while still in the first half -- this real
    match's own windows extend to "1H_45-00_48-13" (48:13 of genuine
    first-half stoppage). A naive elapsed<=45 split misclassifies any
    first-half stoppage minute (46, 47, 48...) as second-half instead,
    which then gets the wrong -45 rebase applied and searches entirely
    the wrong half's windows. Caught directly by this step's own test
    suite: elapsed=47 (a plausible first-half-stoppage minute) resolved
    to "2H_00-00_05-00" under the naive split -- silently wrong, not a
    crash, so it would never have been noticed without a test built
    specifically to probe the 45-50 minute boundary band.

    WHY reuse ground_truth.py's boundary-whistle check rather than invent
    a different one: ground_truth.py's _minute_to_video_s() (this file's
    sibling) already solved the exact same ambiguity for the exact same
    reason -- match minutes are "0-45 for 1H and 45+ for 2H regardless of
    stoppage time", so its fix converts the minute AS IF first-half and
    checks whether that lands before or after the REAL half-time whistle
    (match_boundaries.json's ko_1h/ht_whistle video-second timestamps).
    Only once that would land past the whistle is it genuinely a
    second-half minute. Reusing that check here means both files agree
    on what "first half" means for the same match, rather than each
    having its own, potentially-drifting definition.

    Falls back to the plain elapsed<=45 split when match_boundaries.json
    isn't available for a match (mirrors ground_truth.py's own graceful
    default for the same missing-file case) -- a known approximation,
    but one that still resolves correctly for every event outside the
    narrow ~45-50 minute stoppage-ambiguity band.
    """
    if elapsed_minute is None:
        return None, None

    boundaries = (match_boundaries or {}).get("boundaries")
    if boundaries:
        try:
            ko_1h_s = boundaries["ko_1h"]["seconds"]
            ko_2h_s = boundaries["ko_2h"]["seconds"]
            ht_s    = boundaries["ht_whistle"]["seconds"]
            vs_1h   = ko_1h_s + elapsed_minute * 60
            if vs_1h <= ht_s:
                return 1, elapsed_minute
            return 2, elapsed_minute - 45
        except (KeyError, TypeError):
            pass  # malformed boundaries file -- fall through to the plain split

    return (1, elapsed_minute) if elapsed_minute <= 45 else (2, elapsed_minute - 45)


def _window_for_match_minute(win_labels: list, elapsed_minute, match_boundaries=None):
    """
    Find the index (within win_labels, already time-ordered) of the
    window whose half + minute range contains a given match-elapsed
    minute.

    WHY no `extra` (stoppage-time) field handling on the elapsed number
    itself: checked directly -- grepped every producer in this codebase
    for a time.extra key and found none; this pipeline's real
    match_config.json substitutions (and goals/cards) only ever carry a
    single `elapsed` number, with stoppage already folded into it
    directly. _resolve_half_and_within_minute() above is what handles
    the half-ambiguity that creates; there's no separate extra-minutes
    field to add on top of it.
    """
    target_half, within_half = _resolve_half_and_within_minute(elapsed_minute, match_boundaries)
    if target_half is None:
        return None
    for i, label in enumerate(win_labels):
        bounds = _parse_window_bounds(label)
        if bounds is None:
            continue
        half, start, end = bounds
        if half == target_half and start <= within_half <= end:
            return i
    return None


def calc_substitution_impact(summary: dict, match_config: dict, match_boundaries: dict = None) -> list:
    """
    For each substitution, compare a 3-window average before vs after,
    split per team, using the SAME per-team momentum series Step 20
    already built.

    WHY this function existed but never ran: it was written, fully
    formed, but never called from build_deep_skill_metrics() and never
    registered as a metric -- the exact same producer/consumer gap
    shape as ground_truth.py before Step 18 (a function whose params
    promise a capability -- match_config -- that no caller ever
    supplied). Wiring it in below is what actually lets a substitution's
    effect reach a report.

    WHY it crashed before this rewrite: checked directly by calling it
    against the real Gorleston match -- it raised
    `AttributeError: 'str' object has no attribute 'get'`. The old code
    assumed sub["team"] was a nested dict (`sub.get("team", {}).get(
    "name", "")`), copied from how goal-event scorer/assist fields are
    shaped elsewhere. A real substitution's "team" field is a plain
    string ("Tilbury"), and calling .get() on a string is a hard crash,
    not a silent wrong answer. The old code also read sub["player"] and
    sub["assist"] for the outgoing/incoming player -- goal-event field
    names again -- when the real substitution fields are player_off and
    player_on directly (confirmed against real match_config.json).

    WHY per-team instead of one blended before/after: the whole point of
    a substitution-impact metric is to say whether a substitution helped
    the team that MADE it -- a single match-wide before/after number
    (the old behaviour, reading un-split pressing_by_window/
    line_height_by_window/possession_by_window fields) can only describe
    the whole match getting more or less intense, exactly the same
    attribution gap Step 20 fixed for momentum_score itself. Reusing
    calc_momentum_by_window() here (rather than re-deriving a second,
    independent blended reading straight from the raw fields, which is
    what the old code did) means this metric automatically inherits
    every future fix to the per-team momentum math instead of silently
    drifting from it.

    match_boundaries is optional and defaults to None -- passed through
    to _window_for_match_minute() so a first-half-stoppage substitution
    (e.g. elapsed=47) resolves to the correct half using the real
    half-time whistle timestamp, instead of the naive elapsed<=45 split
    silently misreading it as second-half. Callers that don't have
    match_boundaries.json loaded still get the naive-split fallback --
    correct for every event outside the ~45-50 minute ambiguity band.
    """
    all_subs = match_config.get("substitutions", [])
    if not all_subs:
        return []

    home_team = summary.get("home_team", "")
    away_team = summary.get("away_team", "")

    momentum_by_window = calc_momentum_by_window(summary, match_config)
    win_labels  = [w["window"] for w in momentum_by_window]
    by_label    = {w["window"]: w for w in momentum_by_window}

    def side_snapshot(win_list, side):
        # side is "home", "away", or None (substituting team couldn't be
        # matched to either roster name -- see the None branch below).
        # None is propagated as an all-None snapshot rather than
        # guessing a side, the same explicit-uncertainty contract
        # _momentum_component() already uses for missing readings.
        if side is None:
            return {"momentum": None, "pressing": None, "line_height": None, "possession": None}
        mom, press, line, poss = [], [], [], []
        for wl in win_list:
            w = by_label.get(wl)
            if w is None:
                continue
            m = w.get(f"{side}_momentum")
            if m is not None:
                mom.append(m)
            comps = w.get(f"{side}_components") or {}
            if comps.get("pressing") is not None:
                press.append(comps["pressing"])
            if comps.get("line_height") is not None:
                line.append(comps["line_height"])
            if comps.get("possession") is not None:
                poss.append(comps["possession"])

        def avg(vals):
            return round(sum(vals) / len(vals), 3) if vals else None

        return {
            "momentum":    avg(mom),
            "pressing":    avg(press),
            "line_height": avg(line),
            "possession":  avg(poss),
        }

    impacts = []
    for sub in all_subs:
        minute     = (sub.get("time") or {}).get("elapsed", 0)
        team       = sub.get("team", "")          # real shape: plain string
        player_off = sub.get("player_off", "?")    # real field name
        player_on  = sub.get("player_on", "?")     # real field name

        idx = _window_for_match_minute(win_labels, minute, match_boundaries)
        if idx is None:
            continue

        before_windows = win_labels[max(0, idx - 3):idx]
        after_windows  = win_labels[idx:min(len(win_labels), idx + 3)]

        if team == home_team:
            sub_side, opp_side = "home", "away"
        elif team == away_team:
            sub_side, opp_side = "away", "home"
        else:
            # Data inconsistency: the sub's team name matches neither
            # roster name. Recording team_momentum_shift as None here
            # (rather than guessing a side) keeps a bad name from ever
            # silently attributing a shift to the wrong team.
            sub_side = opp_side = None

        before = {"home": side_snapshot(before_windows, "home"),
                  "away": side_snapshot(before_windows, "away")}
        after  = {"home": side_snapshot(after_windows, "home"),
                  "away": side_snapshot(after_windows, "away")}

        team_momentum_shift = None
        if sub_side is not None:
            b, a = before[sub_side]["momentum"], after[sub_side]["momentum"]
            if b is not None and a is not None:
                team_momentum_shift = round(a - b, 3)

        impacts.append({
            "minute":              minute,
            "team":                team,
            "player_off":          player_off,
            "player_on":           player_on,
            "substituting_side":   sub_side,   # "home" / "away" / None
            "before":              before,      # {"home": {...}, "away": {...}}
            "after":               after,
            "team_momentum_shift": team_momentum_shift,
            "windows_before":      before_windows,
            "windows_after":       after_windows,
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
            # `or "unknown"` guards against None keys in the by_* dicts.
            top_zone = (max(value.get("by_zone",{}),
                          key=value["by_zone"].get) if value.get("by_zone") else "unknown") or "unknown"
            top_del  = (max(value.get("by_delivery_type",{}),
                          key=value["by_delivery_type"].get) if value.get("by_delivery_type") else "unknown") or "unknown"
            top_outcome = (max(value.get("by_outcome",{}),
                             key=value["by_outcome"].get) if value.get("by_outcome") else "unknown") or "unknown"
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

            # `or "unknown"` guards against None keys in by_zone / by_route.
            top_zone  = (max(by_zone,  key=by_zone.get)  if by_zone  else "unknown") or "unknown"
            top_route = (max(by_route, key=by_route.get) if by_route else "unknown") or "unknown"

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
        # Step 20: momentum_score needs to know which team is "focus" so it
        # can turn possession_by_window's focus_pct/opp_seqs framing into a
        # real home/away split. calc_substitution_impact() (line 337) has
        # ALWAYS declared match_config as a required parameter, but nothing
        # in this loader ever supplied it -- another producer/consumer gap
        # of exactly the same shape we already found and fixed in
        # ground_truth.py (Step 19): a function assumes a caller wires in
        # data that, in practice, no caller ever did.
        "match_config":  load("match_config.json",        {}),
        # Step 21: calc_substitution_impact() needs the real half-time
        # whistle timestamp to correctly resolve first-half stoppage-time
        # substitutions (see _resolve_half_and_within_minute()'s WHY
        # comment) -- defaults to {} so matches without this file still
        # work, just with the coarser elapsed<=45 fallback split.
        "match_boundaries": load("match_boundaries.json", {}),
    }


def get_gate(family, gates):
    g = gates.get("gates", {}).get(family, "downgraded")
    # Canonical result_family_gates.json stores each gate as
    # {"status": "...", "note": ...}. Older/flat files store a bare string.
    # Return the status string in both cases (callers compare/key on a string).
    if isinstance(g, dict):
        return g.get("status", "downgraded")
    return g


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


# -- v3 Step 8 merge helpers (block 1 of 3) ----------------------------------
# Ported verbatim from build_deep_skill_metrics_v2.py (the v3 reference
# module that the Step 8 merge subsumes). These helpers are used ONLY by
# the 5 v3-distinctive metric functions below; they intentionally don't
# share infrastructure with v2's make_metric system. Both styles coexist
# inside this module; build_deep_skill_metrics's orchestration treats
# the resulting metric dicts uniformly.

def _apply_confidence_caps(confidence, evidence_tier, source_type,
                            required_family_downgraded, windows_contributing):
    """Apply the standard cap/reduction rules. v3 helper."""
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
    """v3 helper: identifies metrics whose readings depend on whole-team
    visibility. Used by _source_caps_for_metric for source-aware caps."""
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
    """Returns (downgraded_by_source, cap_value, source_note). v3 helper.

    The v3 metric functions read this to apply source-type-aware caps
    (broadcast_tv and veo_ball_tracking limit structural readings)."""
    if source_type == "broadcast_tv" and _is_structural_metric(metric_name):
        return True, None, "Broadcast framing limits whole-team structural readings."
    if source_type == "veo_ball_tracking" and metric_name in {
        "compactness_score", "compactness_geometry_score",
        "width_usage_score", "halfspace_occupation_score",
        "rest_defence_security_score",
    }:
        return True, 0.6, "Ball-follow source -- structural readings zone-limited."
    return False, None, None


def _unavailable(name, reason):
    """v3 helper: standard 'metric cannot be computed' return value.
    The downstream filter_metrics path treats value=None as unavailable
    (v2's existing convention); this helper just standardises the v3
    metric functions' return shape."""
    return {
        "metric_name":      name,
        "value":            None,
        "value_type":       "unavailable",
        "confidence":       0.0,
        "severely_limited": True,
        "limitation_note":  reason,
    }


# -- System-level --------------------------------------------------------------

def calc_compactness(summary):
    """
    Compactness: measured from actual average line height, not stability of movement.
    Lower line = more compact. Higher line = more expansive.
    A team sitting at 25m is compact regardless of whether the line moves.
    A team sitting at 52m is not compact regardless of stability.
    Score 0-1: higher = more compact (line sits deeper).
    """
    # Step 22 fix: line_height_by_window's avg_pct is a DEAD field on real data --
    # the structural agent never emits a flat avg_pct for the defensive line, only
    # home_height_pct/away_height_pct (see accumulator.py's own comment at the
    # v3 line_height_m_by_window block). Step 20 already built the fix -- a
    # sibling field, line_height_m_by_window, that derives avg_pct as the
    # home/away mean -- but this function was never pointed at it, so it kept
    # silently returning 0.0 ("least compact / fully expansive", the bottom
    # of this metric's 0-1 scale) on every real match. It's tagged
    # ["suggestive"] so it isn't presented as high-confidence, but it's still
    # a made-up number rather than an honest "unavailable".
    line_data = summary.get("line_height_m_by_window", [])
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
    # Step 22 fix: real pass-sequence records use start_zone/end_zone, never
    # zone_start/zone_end. Reading the wrong key means s.get("zone_end") is
    # always None, so this comparison was always False -- final_third silently
    # computed as 0 on every real match, on an ACTIVE metric already shown to
    # users (unlike the others below, this one was never on the low-confidence
    # list, which is exactly why it's more serious: nothing flagged it as broken).
    final_third = sum(1 for s in sequences
                      if s.get("end_zone") in ("attacking_third", "right_channel", "left_channel",
                                                "right_halfspace", "left_halfspace")
                      and s.get("start_zone") not in ("attacking_third",))
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

    Step 28 fix (same family as Step 22's calc_compactness fix, one layer
    deeper): this function used to read line_height_by_window's per-window
    "shifts" list, itself sourced from a defensive_line.notable_shifts field
    (see accumulator.py). Two problems stacked on top of each other:

      1. line_height_by_window itself is the dead field Step 22 already found
         (avg_pct/start_pct/end_pct always null on real data) -- the live
         field is line_height_m_by_window.
      2. Unlike compactness, simply switching field names is NOT enough here:
         line_height_m_by_window's own "shifts" list is ALSO always empty on
         every real window, because notable_shifts was part of an OLDER
         per-window extraction schema. Checking the real merged agent window
         files directly (agent_XX_..._merged.json) confirms defensive_line
         only ever contains home_height_pct/away_height_pct/home_descriptor/
         away_descriptor -- notable_shifts was never carried into the current
         schema. So total_backward was structurally guaranteed to stay 0 on
         every match, forever -- not a stale-pointer bug fixable by a field
         rename, but a signal source that no longer exists.

    The fix: stop waiting for an LLM-narrated "notable shift" and compute the
    signal directly from data that IS real and populated on every match --
    line_height_m_by_window's per-window avg_pct. Comparing avg_pct between
    consecutive windows tells us directly whether the defensive line dropped
    deeper (backward = vulnerable, per this metric's own stated definition)
    or held/pushed up. This is arguably a MORE honest signal than the old
    narrative-shift-list approach ever was: it's a direct numeric comparison
    instead of trusting a vision model to remember to narrate every shift.

    Because the new signal is a 0/1-per-transition rate (bounded to [0, 1])
    rather than an unbounded count of narrated shifts, the category
    thresholds in rest_defence_category() were rescaled to match -- the old
    thresholds (calibrated for values that could exceed 1.0) would have made
    the "vulnerable"/"very_vulnerable" bands unreachable under the new,
    correctly-bounded definition.
    """
    line_data = summary.get("line_height_m_by_window", [])
    # Ordered list of (window_label, avg_pct) for windows that actually have a
    # numeric line-height reading. Order matters here -- we're measuring
    # window-to-window transitions, so we must preserve match chronology.
    valid_points = [(w.get("window"), w.get("avg_pct"))
                    for w in line_data if w.get("avg_pct") is not None]
    if len(valid_points) < 2:
        # Can't measure a "shift" with zero or one data point -- there's
        # nothing to compare against. Honestly report "unavailable" rather
        # than fabricating a number, same convention calc_compactness uses.
        return 0.0, None, len(valid_points), ["suggestive"]
    total_backward = 0
    total_transitions = 0
    for (_, prev_pct), (_, next_pct) in zip(valid_points, valid_points[1:]):
        total_transitions += 1
        if next_pct < prev_pct:
            # Line sits at a LOWER pct in the next window than this one --
            # i.e. it dropped deeper / retreated. That's the "backward shift"
            # this metric penalises. A rise (pushing higher) or a hold
            # (equal) is not penalised, per this metric's own definition.
            total_backward += 1
    backward_rate = round(total_backward / total_transitions, 3) if total_transitions > 0 else 0.0
    score = max(0.0, round(1.0 - backward_rate, 2))
    return score, backward_rate, len(valid_points), ["repeated_pattern"]



def calc_line_height_range(summary):
    # Step 22 fix: same root cause as calc_compactness above -- switch from the
    # dead line_height_by_window (avg_pct always None on real data) to
    # line_height_m_by_window (avg_pct derived from home/away height by
    # accumulator.py's Step 20 fix). One difference from the old field: the
    # new field never had start_pct/end_pct keys either (accumulator.py never
    # wrote them -- they were dead in the OLD field too, always None), so we
    # only collect avg_pct per window now. That's still enough to compute a
    # genuine highest-window-vs-lowest-window range across all 21 windows.
    line_data  = summary.get("line_height_m_by_window", [])
    if not line_data:
        return None, 0, ["suggestive"]
    all_values = []
    for w in line_data:
        v = w.get("avg_pct")
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
    # Step 22 fix: real pass-sequence records key these start_zone/end_zone,
    # not zone_start/zone_end -- reading the wrong key made zone_wide always 0,
    # which silently forced every real match onto the inferior
    # cross_outcomes_proxy fallback method below instead of the more accurate
    # zone_labels method. Real data does contain left_channel/right_channel
    # labels, so the accurate method is usable once the key is fixed.
    zone_wide = sum(1 for s in sequences
                    if any(kw in str(s.get("start_zone", "")).lower() or
                           kw in str(s.get("end_zone", "")).lower()
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
    matched_seq_timestamps = []  # Step 29: track which threat_seqs got claimed by a shots_for match
    for shot in shots:
        shot_ts  = _parse_ts(shot.get("timestamp"))
        match_seq = next((s for s in seqs
                          if s.get("outcome") in ("shot", "cross", "goal")
                          and abs(_parse_ts(s.get("start_frame")) - shot_ts) < 30), None)
        if match_seq is not None:
            matched_seq_timestamps.append(_parse_ts(match_seq.get("start_frame")))
        chances.append({
            "timestamp":      shot.get("timestamp"),
            "origin_zone":    match_seq.get("zone_start") if match_seq else "unknown",
            "route":          _infer_route(match_seq),
            "sequence_length":shot.get("sequence_length") or
                              (match_seq.get("length") if match_seq else None),
            "shot_zone":      shot.get("origin_column") or "unknown",
            "shot_type":      shot.get("shot_type") or "unknown",
            "outcome":        shot.get("outcome", "unknown"),
            "target_zone":    shot.get("target_zone") or "unknown",
            "source":         "shots_for",
        })
    # Step 29: shots_for used to be structurally always empty (the extraction
    # agent never emits shot_attempts), so "if not chances" was a safe stand-
    # in for "we have no shots_for data at all, use the pass-sequence proxy
    # instead." Now that shots_for can be non-empty (Step 29's merge adds
    # confirmed goals to it), that binary either/or would be a regression:
    # 3 goal entries would fully replace 70 real threat_sequences chances
    # instead of adding to them, making this profile LESS complete right
    # after we made shots_for MORE complete. shots_for's goals and
    # threat_sequences' shot/cross outcomes describe overlapping but not
    # identical populations (goals are a subset of all shots; threat_seqs
    # covers shots that missed/were saved too, which shots_for still can't
    # see at all -- see the module-level note on the still-open upstream
    # shot_attempts gap). So: always add threat_sequences chances, but skip
    # any whose timestamp already matched a shots_for entry above, so a
    # goal isn't double-counted as two separate "chances."
    threat_seqs = [s for s in seqs if s.get("outcome") in ("shot", "cross")]
    for s in threat_seqs:
        s_ts = _parse_ts(s.get("start_frame"))
        if any(abs(s_ts - claimed) < 30 for claimed in matched_seq_timestamps):
            continue  # already represented via its matching shots_for/goal entry
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
        # Step 29: no longer a binary either/or -- report how many of each
        # actually made up this profile, since both sources can now
        # genuinely contribute at once.
        "data_source":         {
            "shots_for_count":               sum(1 for c in chances if c["source"] == "shots_for"),
            "threat_sequences_fallback_count": sum(1 for c in chances if c["source"] == "threat_sequences_fallback"),
        },
        "summary":             (f"{len(chances)} chances: {danger_pct}% from high-danger zones "
                                f"(six-yard box or penalty spot); most from {football_zone(top_origin)}"
                                f" via {top_route} route"),
    }
    # Step 29: tiers now reflects actual composition instead of a binary
    # "any shots_for at all -> whole profile is direct" claim. worst_evidence_
    # tier() (used by make_metric) takes the WORST tier present, so listing
    # both when both are present correctly caps confidence at the fallback
    # tier's level rather than overclaiming "direct" evidence for a profile
    # that's mostly built from the pass-sequence proxy.
    has_shots_for_chances = any(c["source"] == "shots_for" for c in chances)
    has_fallback_chances  = any(c["source"] == "threat_sequences_fallback" for c in chances)
    tiers = []
    if has_shots_for_chances:
        tiers.append("direct")
    if has_fallback_chances:
        tiers.append("repeated_pattern")
    if not tiers:
        tiers = ["suggestive"]
    return profile, len(chances), tiers



# Baseline routes that every team uses -- exclude from predictability
# since they reflect the shape of football, not tactical predictability
# Step 22 fix: these tuples used the bare word "middle", but the real zone
# vocabulary produced by the structural agent is "middle_third" (confirmed via
# grep: every real start_zone/end_zone value is attacking_third, middle_third,
# defending_third, left_channel, right_channel, or central_channel -- "middle"
# by itself never appears). This wasn't just inert -- it actively broke the
# baseline-route exclusion below: with "middle", real (start,end) pairs like
# ("defending_third", "middle_third") never matched any BASELINE_ROUTES tuple,
# so almost every sequence touching midfield was wrongly treated as
# "non-baseline" (293 of 364 real sequences, instead of the genuine 78) --
# diluting the very signal this metric exists to isolate.
BASELINE_ROUTES = {
    ("defending_third", "middle_third"),
    ("middle_third", "attacking_third"),
    ("defending_third", "defending_third"),
    ("middle_third", "middle_third"),
    ("attacking_third", "middle_third"),
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

    # Step 22 fix: real records use start_zone/end_zone -- reading
    # zone_start/zone_end meant this was always None, so non_baseline was
    # always [] and this ALWAYS fell back to the full-sequence, low-power
    # path below, silently, on every real match.
    # Try non-baseline first
    non_baseline = [s for s in sequences
                    if (s.get("start_zone"), s.get("end_zone")) not in BASELINE_ROUTES
                    and s.get("start_zone") and s.get("end_zone")]

    use_sequences = non_baseline if len(non_baseline) >= 10 else sequences
    fallback = len(non_baseline) < 10

    zone_pairs = Counter(
        (s.get("start_zone"), s.get("end_zone"))
        for s in use_sequences
        if s.get("start_zone") and s.get("end_zone")
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
    # Step 22 fix: same zone_start/zone_end vs real start_zone/end_zone
    # mismatch as calc_width_usage and calc_pattern_reliability -- this made
    # zone_pairs always empty, silently reporting 0 route diversity ("only
    # one route ever used") on every real match regardless of actual variety.
    zone_pairs = set((s.get("start_zone"), s.get("end_zone"))
                     for s in sequences if s.get("start_zone") and s.get("end_zone"))
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
    # NOTE: use `or "unknown"` rather than dict.get(default) because set_piece
    # entries often carry explicit None for these fields (taker, delivery,
    # outcome) -- the default kwarg only kicks in when the key is MISSING,
    # not when the value is None. Without this coercion, by_delivery_type
    # ends up {None: N, ...}, max() returns None, and downstream `.replace()`
    # crashes 3k_metrics. (Bayern vs PSG 2026-05-06 regression.)
    by_zone     = Counter((sp.get("zone") or sp.get("type") or "unknown") for sp in sps)
    by_delivery = Counter((sp.get("delivery") or "unknown") for sp in sps)
    by_outcome  = Counter((sp.get("outcome") or "unknown") for sp in sps)
    by_marking  = Counter(get_marking(sp) or "unknown" for sp in sps)
    bodies      = [sp.get("bodies_in_box") for sp in sps if sp.get("bodies_in_box")]
    zone_detail = {}
    for sp in sps:
        z = sp.get("zone") or sp.get("type") or "unknown"
        if z not in zone_detail:
            zone_detail[z] = {"count": 0, "deliveries": [], "outcomes": []}
        zone_detail[z]["count"]      += 1
        zone_detail[z]["deliveries"].append(sp.get("delivery") or "unknown")
        zone_detail[z]["outcomes"].append(sp.get("outcome") or "unknown")
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
    # shots_for/shots_against are a genuine, separate data-collection gap (the
    # structural agent never emits a shot_attempts category at all, in any
    # window, for any match seen so far) -- so shot_lengths is expected to
    # stay [] until that's addressed upstream. That's fine: it's kept as a
    # forward-compatible fallback, not the fix for this step.
    shot_lengths = [s.get("sequence_length") for s in summary.get("shots_for", []) if s.get("sequence_length")]
    # Step 22 fix: pass-sequence records don't have "length" or
    # "sequence_length" keys at all -- the real field is "passes" (an int,
    # 2-7 in real data, never null). Reading nonexistent keys made
    # all_lengths always [] (plus whatever shot_lengths contributed, which is
    # also always [] per the note above), so this metric reported
    # "unavailable" despite 70 real shot/cross sequences sitting right there.
    all_lengths  = [s.get("passes") for s in threat_seqs if s.get("passes")] + shot_lengths
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



# -- v3 Step 8 merge: 5 new metric functions (block 2 of 3) -------------------
# Ported verbatim from build_deep_skill_metrics_v2.py. These metrics
# extend v2's coverage with v3-distinctive readings -- they don't
# replace any v2 metric. The 3 v3 functions whose names overlap with
# v2 (build_up_effectiveness_score, line_height_range,
# pattern_reliability_score) were INTENTIONALLY NOT ported -- v2's
# implementations stay canonical (preservation-not-replacement, the
# Step 5 / Step 8 pattern).
#
# All 5 use v3's own _apply_confidence_caps / _source_caps_for_metric /
# _unavailable helpers above; they don't go through v2's make_metric.
# The orchestration block at the end of build_deep_skill_metrics() adds
# sample_status to each so filter_metrics treats them like v2 metrics.

def _metric_compactness_geometry(summary, source_type):
    """NEW v3: combines line height (m) and line width (m).
    Compact = narrow line + lower height (deep block).
    Stretched = wide line + high height (advanced press).
    Returns a profile, not a single 0-1 score."""
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


def _metric_defensive_third_turnover_rate(summary, source_type):
    """NEW v3 metric (was impossible to compute pre-Step-5).

    Step 23: this function itself needed no logic change -- it was
    already correctly reading defending_third_turnover_count /
    defending_third_sequence_count from the summary. The bug was
    entirely upstream, in accumulator.py's _count_defending_turnovers(),
    which populates those two fields and was always writing (0, 0). See
    that function's docstring for the fix and for why it's a partial fix
    (real sequences are labelled by EITHER vertical third OR horizontal
    channel, never both, so ~15% of real sequences have a genuinely
    unknowable third and are excluded from this metric's denominator).
    That caveat is surfaced below in the metric's own limitation_note /
    source_limitations fields so report readers see it, not just this
    codebase's own comments.
    """
    turnovers = summary.get("defending_third_turnover_count", 0)
    total = summary.get("defending_third_sequence_count", 0)

    if total < 3:
        return _unavailable("defensive_third_turnover_rate",
                            "Fewer than 3 sequences starting in defending third")

    rate = turnovers / total
    downgraded, _, source_note = _source_caps_for_metric(
        "build_up_effectiveness_score", source_type  # similar structural gate
    )
    # Step 23: always-present caveat about the partial-coverage nature of
    # this metric's denominator (see _count_defending_turnovers() for the
    # full reasoning) -- distinct from source_note, which is about the
    # capture source (broadcast/veo) and may be None for this source type.
    partial_coverage_note = (
        "Sequences are labelled by EITHER vertical third OR horizontal "
        "channel, never both -- channel-labelled sequences (~15% on the "
        "real Gorleston match) are excluded because their vertical third "
        "is genuinely unknown, not just unread. This rate is an honest "
        "undercount of the denominator, not a wrong one."
    )
    note = f"{source_note} {partial_coverage_note}" if source_note else partial_coverage_note

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
            "Sequences with start_zone == 'defending_third' AND outcome "
            "in {lost_possession, clearance} divided by total sequences "
            "starting in defending third (channel-labelled sequences "
            "excluded -- see limitation_note)."
        ),
        "traceable_to":            [
            "defending_third_turnover_count",
            "defending_third_sequence_count",
        ],
    }


def _metric_between_lines_receiving_rate(summary, source_type):
    """NEW v3 per-player metric. For each player observed receiving,
    what fraction of those observations were between lines?

    Returns a profile dict keyed by player. Players with fewer than 3
    receiving observations are marked severely_limited."""
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


def _metric_duel_effectiveness(summary, source_type):
    """NEW v3 metric -- uses post_duel_outcome to distinguish 'wins duels'
    from 'wins duels and retains'."""
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


def _metric_progression_directness(summary, source_type):
    """Step 24: NEW v3 metric, built on top of the Step 24 fix to
    vertical_progression_totals.

    Why this exists: vertical_progression_totals was computed by the
    accumulator ever since v3 Step 5, but no metric function anywhere in
    this file ever read it -- confirmed by grepping the whole active v3
    codebase for "progression" before writing this function. Before the
    Step 24 accumulator fix, that data was also always wrong (every
    sequence resolved to 'unknown'), so there was nothing worth reading
    yet anyway. Now that the counts are real, this metric makes them
    reportable: how often does a team that has the ball in its defending
    or middle third actually move it forward a third, versus holding
    it in the same third or playing it backward?

    This is distinct from the existing build_up_effectiveness_score
    (which measures reaching the final third and shot/cross conversion)
    and pattern_reliability_score (which measures route repetition) --
    both already exist in this file with their own calculation bases.
    This metric is specifically about directness of vertical ball
    progression, independent of whether it ends in a shot.

    Same partial-fix caveat as defensive_third_turnover_rate (Step 23):
    sequences with a channel label (rather than a third label) at
    either end have a genuinely unknown progression and are excluded
    from total_known via the 'unknown' bucket, rather than guessed at.
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
        return _unavailable("progression_directness_score",
                            "Fewer than 3 sequences with a known vertical progression")

    rate = progressed / total_known

    # Same structural gate as defensive_third_turnover_rate /
    # between_lines_receiving_rate -- none of these three have their own
    # entry in _source_caps_for_metric's per-metric-name table, so they
    # all borrow the build_up_effectiveness_score gate as the closest
    # analogous structural-limitation check for this source type.
    downgraded, _, source_note = _source_caps_for_metric(
        "build_up_effectiveness_score", source_type
    )
    partial_coverage_note = (
        "Sequences are labelled by EITHER vertical third OR horizontal "
        "channel at each end, never both -- sequences with a channel label "
        "at either end (~16% on the real Gorleston match) are excluded "
        "because their progression is genuinely unknown, not just unread. "
        "This rate is an honest undercount of the denominator, not a "
        "wrong one."
    )
    note = f"{source_note} {partial_coverage_note}" if source_note else partial_coverage_note

    return {
        "metric_name":   "progression_directness_score",
        "analysis_scope":"match",
        "subject_team":  "focus",
        "value": {
            "progressive_rate": round(rate, 2),
            "progressed_count": progressed,
            "regressed_count":  regressed,
            "same_third_count": same,
            "total_known":      total_known,
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
            "Sequences whose start_zone/end_zone are both known vertical "
            "thirds, categorised via the progression matrix (forward move, "
            "same third, or backward move). progressive_rate = forward-move "
            "sequences divided by all sequences with a known progression "
            "(channel-labelled sequences excluded -- see limitation_note)."
        ),
        "traceable_to":            ["vertical_progression_totals"],
    }


def _metric_watch_list_summary(summary):
    """NEW v3 -- summarises watch_list_confirmations across all windows."""
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


# -- End v3 Step 8 merge (block 2 of 3) ---------------------------------------


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
    match_config  = data["match_config"]
    match_boundaries = data["match_boundaries"]  # Step 21: real half-time whistle timestamp
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
        # Step 22 fix: traceable_to metadata updated to match the field the
        # function actually reads now -- this string is shown to report
        # readers/synthesis as "where did this number come from", so it must
        # stay honest about the real source.
        traceable_to=["line_height_m_by_window", "pressing_by_window"]))

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

    v, backward_rate, w, t = calc_rest_defence_security(summary)
    rest_cat = rest_defence_category(backward_rate)
    # Fix 55c: rephrase per-window prose so synthesis cannot import it
    # verbatim. The calculation_basis is read by synthesis as context;
    # "per window" leaks WINDOW LANGUAGE RULE violations into reports.
    # Step 28: field renamed avg_backward_shifts_per_window -> backward_shift_rate_pct
    # to match the new bounded-rate definition (was a raw shift count before;
    # now it's "what fraction of transitions were backward", so we report it
    # as a percentage, which is the more natural unit for a bounded rate).
    metrics.append(make_metric("rest_defence_security_score", "match", ["rest_defence"],
        {"backward_shift_rate_pct": round(backward_rate * 100, 1) if backward_rate is not None else None,
         "category": rest_cat,
         "summary": f"{rest_cat} ({round(backward_rate * 100, 1) if backward_rate is not None else 'n/a'}% of phases saw the line drop deeper)"},
        "profile", t, 0.75, w,
        "Fraction of window-to-window transitions where the defensive line height dropped, with categorical label",
        traceable_to=["line_height_m_by_window"]))

    v, w, t = calc_line_height_range(summary)
    if v:
        v["summary"] = (f"highest: {v['highest_m']}m ({v['highest_category']}), "
                        f"lowest: {v['lowest_m']}m ({v['lowest_category']}), "
                        f"range: {v['range_m']}m")
    metrics.append(make_metric("line_height_range", "match", ["shape"],
        v, "profile", t, 0.80, w,
        "Highest and lowest line height in metres from own goal with categorical bands",
        # Step 22 fix: same metadata correction as compactness_score above.
        traceable_to=["line_height_m_by_window"]))

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
        # `or "unknown"` guards against by_zone/by_delivery_type containing a
        # None key (set_pieces can carry explicit None for these fields);
        # without the guard max() returns None and .replace() crashes.
        top_zone = (max(v["by_zone"], key=v["by_zone"].get) if v["by_zone"] else "unknown") or "unknown"
        top_del  = (max(v["by_delivery_type"], key=v["by_delivery_type"].get) if v["by_delivery_type"] else "unknown") or "unknown"
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

    # Momentum score (composite per window, split home vs away -- Step 20).
    #
    # Before this step, momentum_score reported ONE blended number per
    # window (subject_team="both"), which can only say "the match got
    # more/less intense" -- never which side that favoured. The report
    # prompt (synthesis_agent.py's "Tactical Variation and Substitutions"
    # section) needs to attribute a shift to a specific team around a
    # substitution, so the value shape below carries two independent
    # series (home/away) instead of one composite -- see
    # calc_momentum_by_window()'s docstring for the full reasoning.
    mom_windows = calc_momentum_by_window(summary, match_config)

    def _team_summary(side):
        # side is "home" or "away"; picks that team's momentum/components
        # back out of the shared per-window list, dropping windows where
        # THIS team specifically had nothing observable (the other team
        # may still have a valid reading for the same window -- that's
        # exactly the point of splitting, so one team's gap doesn't zero
        # out or blank the other team's real reading).
        valid = [
            {"window": w["window"], "momentum": w[f"{side}_momentum"],
             "components": w[f"{side}_components"]}
            for w in mom_windows if w.get(f"{side}_momentum") is not None
        ]
        if not valid:
            return None, valid
        avg = round(sum(w["momentum"] for w in valid) / len(valid), 3)
        return {
            "avg": avg,
            "by_window": valid,
            "peak": max(valid, key=lambda w: w["momentum"])["window"],
            "low":  min(valid, key=lambda w: w["momentum"])["window"],
        }, valid

    home_summary, home_valid = _team_summary("home")
    away_summary, away_valid = _team_summary("away")

    # Only register the metric at all if AT LEAST ONE side produced a
    # usable reading -- mirrors the old "if valid_mom" guard, just
    # widened from one series to two so a match isn't skipped just
    # because (for example) the away team's pressing was never observed.
    if home_summary is not None or away_summary is not None:
        metrics.append(make_metric(
            "momentum_score", "match", ["pressing","build_up","shape"],
            {"home": home_summary, "away": away_summary},
            "profile", "repeated_pattern", 0.50,
            max(len(home_valid), len(away_valid)),
            "Composite momentum per team: pressing 35% + possession 40% + line height 25%",
            sample_n=max(len(home_valid), len(away_valid))))

    # Substitution impact (per-team before/after momentum -- Step 21).
    #
    # WHY registered here at all: calc_substitution_impact() was fully
    # written but never called anywhere in build_deep_skill_metrics() --
    # the same producer/consumer gap Step 18 found in ground_truth.py --
    # so a substitution's effect never reached synthesis_agent.py's
    # report no matter how good the underlying math was. Calling it and
    # appending its result here is what actually wires it in.
    #
    # WHY base_confidence is slightly lower than momentum_score's (0.45
    # vs 0.50) despite reusing the exact same per-team momentum series:
    # momentum_score's average is drawn from every window in the match,
    # but each substitution_impact data point only ever averages 3
    # windows before and 3 after that one event -- same data source,
    # smaller sample per reading, so a slightly more conservative base
    # confidence is the honest reflection of that, not an arbitrary
    # number.
    sub_impacts = calc_substitution_impact(summary, match_config, match_boundaries)
    if sub_impacts:
        metrics.append(make_metric(
            "substitution_impact", "match", ["pressing","build_up","shape"],
            sub_impacts, "profile", "repeated_pattern", 0.45, len(sub_impacts),
            "Per-substitution 3-window average momentum before vs after, "
            "split by the substituting team's own home/away momentum series",
            sample_n=len(sub_impacts)))

    # -- v3 Step 8 merge (block 3 of 3): register the 5 v3-distinctive
    # metric functions. Each function returns its metric dict directly
    # (v3 doesn't use make_metric); we wrap with _v3_with_sample_status
    # so the downstream assign_confidence_level / filter_metrics path
    # treats v3 entries the same as v2's make_metric output. Without
    # this, v3 metrics would default sample_status="context_only" and
    # be filtered out at confidence_level=2 (the runtime default).
    def _v3_with_sample_status(m):
        if m.get("value") is None:
            m["sample_status"] = "unavailable"
        else:
            w = m.get("windows_contributing") or 0
            m["sample_status"] = "preliminary" if w < 3 else "sufficient"
        # context_only is a v2-side flag; v3 entries don't set it,
        # default to False so assign_confidence_level treats them
        # as standard metrics.
        m.setdefault("context_only", False)
        return m

    metrics.append(_v3_with_sample_status(
        _metric_compactness_geometry(summary, source_type)))
    metrics.append(_v3_with_sample_status(
        _metric_defensive_third_turnover_rate(summary, source_type)))
    metrics.append(_v3_with_sample_status(
        _metric_between_lines_receiving_rate(summary, source_type)))
    metrics.append(_v3_with_sample_status(
        _metric_duel_effectiveness(summary, source_type)))
    metrics.append(_v3_with_sample_status(
        _metric_progression_directness(summary, source_type)))
    metrics.append(_v3_with_sample_status(
        _metric_watch_list_summary(summary)))

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
        json.dump(stamp_schema_version(output, "deep_skill_metrics"), f, indent=2)

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
