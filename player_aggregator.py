"""
player_aggregator.py

Builds per-player summary cards by aggregating across all windows.
Produces player_summary_cards.json -- the per-player deliverable read by
Step 4 report writers.

Run order in the pipeline:
    After update_running_summary_v2 finishes all windows
    After zone_helpers.walk_findings_apply_zone_helpers runs
    After player_escalation_router merges confirmations
    Before build_deep_skill_metrics_v2
    Before Step 4 report writing

Public entrypoint:
    build_player_summary_cards(match_dir)

Inputs read:
    running_summary.json
    match_config.json
    source_profile.json

Output written:
    player_summary_cards.json

All derivations are deterministic from the data. The agent does not produce
any of these fields directly.
"""

import json
import os
import re
import statistics
from collections import Counter, defaultdict


# ============================================================================
# Constants
# ============================================================================

INSUFFICIENT_TIME_THRESHOLD_MINUTES = 15

# Action categories considered "active" vs "passive" for temporal arc detection
ACTIVE_CATEGORIES = {
    "ball_carrying",
    "pressing_behaviour",
    "finishing",
    "movement_off_ball",
    "set_piece_delivery",
    "distribution",
    "first_touch_direction",
    "receiving_orientation",
}

PASSIVE_CATEGORIES = {
    "defensive_positioning",
    "recovery_runs",
    "positional_tendency",
}

# Position role mismatch rules. Each entry: (flag_id, predicate)
# A predicate returns True for an observation that contributes to the flag.
def _is_aerial_weakness_under_pressure(obs):
    return (
        obs.get("action_category") == "aerial_ability"
        and obs.get("observation_type") == "weakness"
        and obs.get("trigger_context") == "under_pressure"
    )


def _is_beaten_1v1(obs):
    cat = obs.get("action_category")
    text = (obs.get("observation") or "").lower()
    return (
        cat in {"defensive_positioning", "duels"}
        and obs.get("observation_type") == "weakness"
        and (
            "beaten" in text
            or "1v1" in text
            or "1 v 1" in text
            or "pace" in text
        )
    )


def _is_hold_up_weakness(obs):
    return (
        obs.get("action_category") == "hold_up_play"
        and obs.get("observation_type") == "weakness"
    )


def _is_finishing_weakness(obs):
    return (
        obs.get("action_category") == "finishing"
        and obs.get("observation_type") == "weakness"
    )


POSITION_MISMATCH_FLAGS = {
    "lb": [
        ("aerial_under_pressure_for_full_back", _is_aerial_weakness_under_pressure),
        ("beaten_1v1_recurring_for_full_back",  _is_beaten_1v1),
    ],
    "rb": [
        ("aerial_under_pressure_for_full_back", _is_aerial_weakness_under_pressure),
        ("beaten_1v1_recurring_for_full_back",  _is_beaten_1v1),
    ],
    "st": [
        ("hold_up_weakness_for_striker",        _is_hold_up_weakness),
        ("finishing_weakness_for_striker",      _is_finishing_weakness),
    ],
    "cf": [
        ("hold_up_weakness_for_striker",        _is_hold_up_weakness),
        ("finishing_weakness_for_striker",      _is_finishing_weakness),
    ],
}


# ============================================================================
# Player ID helpers
# ============================================================================

def extract_shirt_number(player_string):
    """
    '#10 Aaron Bullent'    -> 10
    '#10 [striker]'        -> 10
    '#10'                  -> 10
    'striker'              -> None
    None                   -> None
    """
    if not player_string:
        return None
    m = re.search(r"#(\d+)", str(player_string))
    return int(m.group(1)) if m else None


def player_matches(obs_player_string, target_shirt, target_name=None):
    """True if an observation player string matches a target shirt (and name)."""
    obs_shirt = extract_shirt_number(obs_player_string)
    if obs_shirt is not None and target_shirt is not None:
        return obs_shirt == target_shirt
    if target_name:
        return target_name.lower() in (obs_player_string or "").lower()
    return False


# ============================================================================
# Roster extraction
# ============================================================================

def _resolve_team_side(team_name, match_config):
    """Returns 'home' or 'away' for a team name from API lineup."""
    if not team_name:
        return None
    if team_name == match_config.get("home_team"):
        return "home"
    if team_name == match_config.get("away_team"):
        return "away"
    return None


def extract_roster(match_config):
    """
    Returns a list of player records: every starter + every substitute who
    came on. Each record:
      {
        "shirt":           int,
        "name":            str,
        "team":            "home" | "away",
        "position":        "lb" | ...
        "is_starter":      bool,
        "sub_on_minute":   int | None,
        "sub_off_minute":  int | None,
      }
    """
    roster = []
    seen_keys = set()  # (team, shirt) -- avoid duplicate entries

    lineups = match_config.get("lineups", []) or []
    subs = match_config.get("substitutions", []) or []

    # v3 port Step 6: production match_config.json uses two shapes for
    # team and player fields:
    #
    #   lineups[i].team         = {"name": "..."}              (dict)
    #   substitutions[i].team   = "..."                         (string)
    #   substitutions[i].player_off / player_on = "name string" (NOT
    #     the dict-with-number 'player' / 'assist' fields the
    #     football-data.org API uses; the production match_config
    #     extractor writes string-name fields instead)
    #
    # The aggregator was originally built against the football-data.org
    # API shape. Per Step 6 direction-of-fix rule, patch the v3 reader
    # to handle what production writes -- both shapes -- rather than
    # demanding production conform to the reference.

    def _team_name(entity):
        """Read a team field that may be dict {'name': ...} OR plain string."""
        if isinstance(entity, dict):
            return entity.get("name")
        if isinstance(entity, str):
            return entity
        return None

    # Pre-build a name -> (team_side, shirt) index from lineups so we
    # can resolve string-name substitution entries back to shirt numbers.
    name_to_key = {}
    for lineup in lineups:
        _t_side = _resolve_team_side(_team_name(lineup.get("team")), match_config)
        if _t_side is None:
            continue
        for p_wrap in (lineup.get("startXI", []) or []) + (lineup.get("substitutes", []) or []):
            p = p_wrap.get("player", {}) if isinstance(p_wrap, dict) and "player" in p_wrap else p_wrap
            if not isinstance(p, dict):
                continue
            _shirt = p.get("number")
            _name  = p.get("name")
            if _shirt is not None and _name:
                name_to_key[_name] = (_t_side, int(_shirt))

    # Pre-index substitutions
    sub_on_for = {}   # (team, shirt) -> minute_on
    sub_off_for = {}  # (team, shirt) -> minute_off
    for s in subs:
        team_name = _team_name(s.get("team"))
        team = _resolve_team_side(team_name, match_config)
        minute = s.get("time", {}).get("elapsed")

        # Player going OFF -- accept both dict-shape ("player" field)
        # and string-name shape ("player_off" field).
        off_entity = s.get("player") if s.get("player") is not None else s.get("player_off")
        if isinstance(off_entity, dict):
            shirt = off_entity.get("number")
            if team and shirt is not None:
                sub_off_for[(team, int(shirt))] = minute
        elif isinstance(off_entity, str):
            key = name_to_key.get(off_entity)
            if key:
                sub_off_for[key] = minute

        # Player coming ON -- accept both dict-shape ("assist" field)
        # and string-name shape ("player_on" field).
        on_entity = s.get("assist") if s.get("assist") is not None else s.get("player_on")
        if isinstance(on_entity, dict):
            shirt = on_entity.get("number")
            if team and shirt is not None:
                sub_on_for[(team, int(shirt))] = minute
        elif isinstance(on_entity, str):
            key = name_to_key.get(on_entity)
            if key:
                sub_on_for[key] = minute

    # Starters
    for lineup in lineups:
        team_name = _team_name(lineup.get("team"))
        team = _resolve_team_side(team_name, match_config)
        for p_wrap in lineup.get("startXI", []) or []:
            p = p_wrap.get("player", {}) if "player" in p_wrap else p_wrap
            shirt = p.get("number")
            if shirt is None:
                continue
            shirt = int(shirt)
            key = (team, shirt)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            roster.append({
                "shirt":          shirt,
                "name":           p.get("name", ""),
                "team":           team,
                "position":       (p.get("pos") or "unknown").lower(),
                "is_starter":     True,
                "sub_on_minute":  None,
                "sub_off_minute": sub_off_for.get(key),
            })

    # Substitutes who came on
    for lineup in lineups:
        team_name = _team_name(lineup.get("team"))
        team = _resolve_team_side(team_name, match_config)
        for p_wrap in lineup.get("substitutes", []) or []:
            p = p_wrap.get("player", {}) if "player" in p_wrap else p_wrap
            shirt = p.get("number")
            if shirt is None:
                continue
            shirt = int(shirt)
            key = (team, shirt)
            if key in sub_on_for and key not in seen_keys:
                seen_keys.add(key)
                roster.append({
                    "shirt":          shirt,
                    "name":           p.get("name", ""),
                    "team":           team,
                    "position":       (p.get("pos") or "unknown").lower(),
                    "is_starter":     False,
                    "sub_on_minute":  sub_on_for.get(key),
                    "sub_off_minute": sub_off_for.get(key),
                })

    return roster


# ============================================================================
# Per-player computations
# ============================================================================

def compute_minutes_visible(player_rec, match_length=90):
    """
    Approximation -- does NOT subtract dead time. Returns (minutes, source).
    """
    starter = player_rec["is_starter"]
    on = player_rec["sub_on_minute"]
    off = player_rec["sub_off_minute"]

    if starter:
        if off is not None:
            return max(0, off), "starter_subbed_off"
        return match_length, "starter_played_full"
    else:
        if on is None:
            return 0, "substitute_unused"
        end = off if off is not None else match_length
        return max(0, end - on), "substitute_subbed_on"


def filter_player_observations(player_rec, all_observations):
    """Return only observations whose player field matches this player's shirt."""
    result = []
    for obs in all_observations:
        obs_team = obs.get("team")
        if obs_team and obs_team != player_rec["team"]:
            continue
        if player_matches(
            obs.get("player"),
            player_rec["shirt"],
            target_name=player_rec["name"],
        ):
            result.append(obs)
    return result


def compute_observation_counts(observations):
    """Per-window, per-category, per-type counts."""
    by_window = defaultdict(int)
    by_category = defaultdict(int)
    by_type = defaultdict(int)

    for obs in observations:
        window = obs.get("window") or obs.get("window_id")
        if window:
            by_window[window] += 1
        cat = obs.get("action_category")
        if cat:
            by_category[cat] += 1
        otype = obs.get("observation_type")
        if otype:
            by_type[otype] += 1

    return dict(by_window), dict(by_category), dict(by_type)


def _window_key_sort(window_id):
    """Sort key for window IDs like '00-05min' -> 0."""
    if not window_id:
        return 999
    m = re.match(r"(\d+)-", window_id)
    return int(m.group(1)) if m else 999


def compute_temporal_arc(observations, player_rec, match_length=90):
    """
    Returns (arc_label, evidence_dict). Strict 3-condition gate for fading_late.
    """
    if not observations:
        return "insufficient_data", {
            "still_on_pitch_at_end": False,
            "avg_obs_early_windows": 0,
            "avg_obs_late_windows":  0,
            "active_ratio_early":    None,
            "active_ratio_late":     None,
        }

    by_window_counts = Counter()
    by_window_categories = defaultdict(Counter)
    for obs in observations:
        w = obs.get("window") or obs.get("window_id")
        if not w:
            continue
        by_window_counts[w] += 1
        cat = obs.get("action_category")
        if cat:
            by_window_categories[w][cat] += 1

    windows_with_obs = sorted(by_window_counts.keys(), key=_window_key_sort)

    # Still on pitch at end?
    still_on = player_rec["sub_off_minute"] is None or player_rec["sub_off_minute"] >= (match_length - 1)

    if len(windows_with_obs) < 3:
        return "insufficient_data", {
            "still_on_pitch_at_end": still_on,
            "avg_obs_early_windows": 0,
            "avg_obs_late_windows":  0,
            "active_ratio_early":    None,
            "active_ratio_late":     None,
        }

    # Last 2 windows vs earlier windows
    late = windows_with_obs[-2:]
    early = windows_with_obs[:-2]

    if not early:
        return "insufficient_data", {
            "still_on_pitch_at_end": still_on,
            "avg_obs_early_windows": 0,
            "avg_obs_late_windows":  statistics.mean([by_window_counts[w] for w in late]) if late else 0,
            "active_ratio_early":    None,
            "active_ratio_late":     None,
        }

    avg_early = statistics.mean([by_window_counts[w] for w in early])
    avg_late = statistics.mean([by_window_counts[w] for w in late]) if late else 0

    def _active_ratio(windows):
        active = 0
        passive = 0
        for w in windows:
            cats = by_window_categories[w]
            for cat, n in cats.items():
                if cat in ACTIVE_CATEGORIES:
                    active += n
                elif cat in PASSIVE_CATEGORIES:
                    passive += n
        denom = active + passive
        if denom == 0:
            return None
        return active / denom

    ratio_early = _active_ratio(early)
    ratio_late = _active_ratio(late)

    evidence = {
        "still_on_pitch_at_end":     still_on,
        "avg_obs_early_windows":     round(avg_early, 2),
        "avg_obs_late_windows":      round(avg_late, 2),
        "active_ratio_early":        round(ratio_early, 2) if ratio_early is not None else None,
        "active_ratio_late":         round(ratio_late, 2) if ratio_late is not None else None,
    }

    # Fading-late gate -- all 3 conditions
    density_drop = avg_early > 0 and avg_late < (avg_early * 0.5)
    category_shift = (
        ratio_early is not None
        and ratio_late is not None
        and ratio_late < (ratio_early - 0.3)
    )

    if still_on and density_drop and category_shift:
        return "fading_late", evidence

    # Building late -- inverse pattern
    if (
        still_on
        and avg_late > (avg_early * 1.5)
        and ratio_late is not None
        and ratio_early is not None
        and ratio_late > (ratio_early + 0.2)
    ):
        return "building_late", evidence

    # Front loaded -- early dominance without late presence
    if avg_early > (avg_late * 1.5) and not still_on:
        return "front_loaded", evidence

    # Inconsistent -- high variance
    counts = [by_window_counts[w] for w in windows_with_obs]
    if len(counts) >= 3:
        mean = statistics.mean(counts)
        try:
            stdev = statistics.stdev(counts)
        except statistics.StatisticsError:
            stdev = 0
        if mean > 0 and (stdev / mean) > 0.8:
            return "inconsistent", evidence

    return "stable", evidence


def compute_signature_actions(observations):
    """
    frequency=consistent + observation_type in {strength, trait}
    + seen across 3+ windows.
    """
    candidates = defaultdict(lambda: {"windows": set(), "occurrences": 0, "exemplar": None})

    for obs in observations:
        if obs.get("frequency") != "consistent":
            continue
        if obs.get("observation_type") not in {"strength", "trait"}:
            continue
        cat = obs.get("action_category")
        obs_text = (obs.get("observation") or "")[:60]
        key = (cat, obs_text)
        rec = candidates[key]
        rec["windows"].add(obs.get("window") or obs.get("window_id"))
        rec["occurrences"] += 1
        if rec["exemplar"] is None:
            rec["exemplar"] = obs

    signatures = []
    for (cat, _), rec in candidates.items():
        windows_clean = {w for w in rec["windows"] if w}
        if len(windows_clean) >= 3:
            signatures.append({
                "action_category":   cat,
                "summary":           rec["exemplar"].get("observation"),
                "windows_observed":  sorted(windows_clean, key=_window_key_sort),
                "occurrences":       rec["occurrences"],
            })
    signatures.sort(key=lambda s: s["occurrences"], reverse=True)
    return signatures


def link_decisive_moments(player_rec, key_moments):
    """
    Walk key_moments and surface ones referencing this player by shirt # or
    name in the description.
    """
    matched = []
    shirt_pattern = re.compile(rf"#\b{player_rec['shirt']}\b")
    name_lower = (player_rec["name"] or "").lower()

    for km in key_moments or []:
        desc = km.get("description") or ""
        if shirt_pattern.search(desc) or (name_lower and name_lower in desc.lower()):
            matched.append({
                "timestamp":   km.get("timestamp"),
                "type":        km.get("type"),
                "description": desc,
            })
    return matched


def compute_partnerships(observations):
    """
    Aggregate link_up_with across all observations for this player.
    Filter out partners with < 2 shared observations.
    """
    partner_data = defaultdict(lambda: {
        "shared_observations": 0,
        "shared_categories":   Counter(),
        "shared_zones":        Counter(),
    })

    for obs in observations:
        link_up = obs.get("link_up_with") or []
        if not isinstance(link_up, list):
            continue
        cat = obs.get("action_category")
        zone = obs.get("zone") or {}
        lane = zone.get("lateral_lane")

        for partner in link_up:
            if not partner:
                continue
            rec = partner_data[partner]
            rec["shared_observations"] += 1
            if cat:
                rec["shared_categories"][cat] += 1
            if lane:
                rec["shared_zones"][lane] += 1

    partnerships = []
    for partner, data in partner_data.items():
        if data["shared_observations"] < 2:
            continue
        partnerships.append({
            "with":                 partner,
            "shared_observations":  data["shared_observations"],
            "shared_categories":    dict(data["shared_categories"]),
            "shared_zones":         dict(data["shared_zones"]),
        })
    partnerships.sort(key=lambda p: p["shared_observations"], reverse=True)
    return partnerships


def compute_match_aggregate_counts(observations, fouls_for_player):
    """
    Counts of specific event types for the player. Used for descriptors like
    'beaten 1v1' and aerial duel record.
    """
    counts = {
        "ball_carrying_strengths":            0,
        "ball_carrying_weaknesses":           0,
        "aerial_duels_won":                   0,
        "aerial_duels_lost":                  0,
        "beaten_1v1":                         0,
        "received_under_pressure":            0,
        "fouls_committed":                    0,
        "fouls_committed_transition_against": 0,
    }

    for obs in observations:
        cat = obs.get("action_category")
        otype = obs.get("observation_type")
        outcome = obs.get("outcome")
        trigger = obs.get("trigger_context")

        if cat == "ball_carrying" and otype == "strength":
            counts["ball_carrying_strengths"] += 1
        elif cat == "ball_carrying" and otype == "weakness":
            counts["ball_carrying_weaknesses"] += 1

        if cat == "aerial_ability":
            if outcome == "success":
                counts["aerial_duels_won"] += 1
            elif outcome == "failure":
                counts["aerial_duels_lost"] += 1

        if _is_beaten_1v1(obs):
            counts["beaten_1v1"] += 1

        if trigger == "under_pressure" and cat in {
            "receiving_orientation",
            "first_touch_direction",
            "hold_up_play",
            "ball_carrying",
        }:
            counts["received_under_pressure"] += 1

    counts["fouls_committed"] = len(fouls_for_player)
    counts["fouls_committed_transition_against"] = sum(
        1 for f in fouls_for_player
        if f.get("match_state_context") == "transition_against"
    )

    return counts


def compute_position_mismatch_flags(observations, position):
    """Apply the position-specific mismatch rules. Returns list of flags."""
    rules = POSITION_MISMATCH_FLAGS.get(position, [])
    if not rules:
        return []

    flags = []
    for flag_id, predicate in rules:
        matching = [obs for obs in observations if predicate(obs)]
        if len(matching) < 2:
            continue
        if len(matching) >= 5:
            weight = "high"
        elif len(matching) >= 3:
            weight = "medium"
        else:
            weight = "low"
        flags.append({
            "flag":                    flag_id,
            "supporting_observations": [obs.get("timestamp") for obs in matching],
            "weight":                  weight,
        })
    return flags


def compute_tactical_fouling_indicator(fouls_for_player):
    """
    Returns the tactical_fouling_indicator block or None if below threshold.
    Threshold: 2+ transition_against fouls.
    """
    transition_against = [
        f for f in fouls_for_player
        if f.get("match_state_context") == "transition_against"
    ]
    total = len(fouls_for_player)

    if len(transition_against) < 2:
        return None

    return {
        "transition_against_fouls": len(transition_against),
        "total_fouls":              total,
        "ratio":                    round(len(transition_against) / total, 2) if total else 0,
        "threshold_met":            True,
        "fouls":                    [f.get("timestamp") for f in transition_against],
    }


def compute_temperament_summary(observations):
    """Counts by subcode, plus notable_pattern if 3+ of any type."""
    subcode_counts = Counter()
    for obs in observations:
        if obs.get("action_category") != "temperament_observation":
            continue
        sub = obs.get("temperament_subcode")
        if sub:
            subcode_counts[sub] += 1

    if not subcode_counts:
        return None

    notable = None
    for sub, n in subcode_counts.items():
        if n >= 3:
            notable = f"{sub} observed {n} times"
            break

    return {
        "visible_frustration_count":     subcode_counts.get("visible_frustration", 0),
        "organising_behaviour_count":    subcode_counts.get("organising_behaviour", 0),
        "assertiveness_in_duels_count":  subcode_counts.get("assertiveness_in_duels", 0),
        "reaction_after_error_count":    subcode_counts.get("reaction_after_error", 0),
        "notable_pattern":               notable,
    }


def compute_conditional_patterns(observations):
    """Extract observations with non-null condition + condition_absent_outcome."""
    patterns = []
    for obs in observations:
        cond = obs.get("condition")
        outcome = obs.get("condition_absent_outcome")
        if cond and outcome:
            patterns.append({
                "observation":                obs.get("observation"),
                "condition":                  cond,
                "condition_absent_outcome":   outcome,
                "frequency":                  obs.get("frequency"),
                "action_category":            obs.get("action_category"),
            })
    return patterns


# ============================================================================
# Cross-player computations
# ============================================================================

def compute_comparative_rankings(cards):
    """
    Assign rank_in_position for each player by observation count, strength
    count, and weakness count.
    """
    by_position = defaultdict(list)
    for c in cards:
        by_position[(c["team"], c["position"])].append(c)

    for (team, pos), players in by_position.items():
        if len(players) <= 1:
            continue

        # By total observations
        sorted_obs = sorted(players, key=lambda p: p["observation_count_total"], reverse=True)
        for i, p in enumerate(sorted_obs, 1):
            p["comparative_rank_in_position"]["rank_by_observation_count"] = (
                f"{i} of {len(sorted_obs)} {pos.upper()}s in {team} lineup"
            )

        # By strength count
        sorted_str = sorted(
            players,
            key=lambda p: p["observation_count_by_type"].get("strength", 0),
            reverse=True,
        )
        for i, p in enumerate(sorted_str, 1):
            p["comparative_rank_in_position"]["rank_by_strength_count"] = f"{i} of {len(sorted_str)}"

        # By weakness count
        sorted_wk = sorted(
            players,
            key=lambda p: p["observation_count_by_type"].get("weakness", 0),
            reverse=True,
        )
        for i, p in enumerate(sorted_wk, 1):
            p["comparative_rank_in_position"]["rank_by_weakness_count"] = f"{i} of {len(sorted_wk)}"


def compute_scarcity_traits(cards, observations_index):
    """
    Identify team-relative rarities. Currently: left-footedness within team.
    """
    # Index preferred_foot per player from observations
    foot_by_player = {}
    for c in cards:
        feet_seen = Counter()
        for obs in observations_index.get((c["team"], c["shirt"]), []):
            f = obs.get("preferred_foot")
            if f and f != "unknown":
                feet_seen[f] += 1
        if feet_seen:
            foot_by_player[(c["team"], c["shirt"])] = feet_seen.most_common(1)[0][0]

    # Count left-footed per team
    left_footed_by_team = defaultdict(list)
    for (team, shirt), foot in foot_by_player.items():
        if foot == "left":
            left_footed_by_team[team].append(shirt)

    for c in cards:
        team = c["team"]
        shirt = c["shirt"]
        traits = []
        if shirt in left_footed_by_team[team] and len(left_footed_by_team[team]) <= 3:
            n = len(left_footed_by_team[team])
            traits.append(f"left-footed (one of {n} left-footed player{'s' if n != 1 else ''} in {team} team)")
        c["scarcity_traits"] = traits


# ============================================================================
# Foul indexing
# ============================================================================

def index_fouls_by_player(running_summary):
    """
    Returns dict: (team, shirt) -> list of foul records.
    Reads from running_summary.fouls_committed (accumulated by v2.1 running
    summary updater).
    """
    by_player = defaultdict(list)
    fouls = running_summary.get("fouls_committed", []) or []
    for f in fouls:
        team = f.get("fouler_team")
        shirt = extract_shirt_number(f.get("fouler"))
        if team is not None and shirt is not None:
            by_player[(team, shirt)].append(f)
    return by_player


# ============================================================================
# Card builder
# ============================================================================

def build_card(player_rec, running_summary, match_config, source_profile,
               fouls_index, observations_index):
    """Build a single player_summary_card."""
    observations = observations_index.get((player_rec["team"], player_rec["shirt"]), [])
    fouls = fouls_index.get((player_rec["team"], player_rec["shirt"]), [])

    minutes, source = compute_minutes_visible(player_rec)
    by_window, by_cat, by_type = compute_observation_counts(observations)
    arc_label, arc_evidence = compute_temporal_arc(observations, player_rec)
    signatures = compute_signature_actions(observations)
    decisive = link_decisive_moments(player_rec, running_summary.get("key_moments", []))
    partnerships = compute_partnerships(observations)
    aggregate_counts = compute_match_aggregate_counts(observations, fouls)
    mismatch_flags = compute_position_mismatch_flags(observations, player_rec["position"])
    tactical_fouling = compute_tactical_fouling_indicator(fouls)
    temperament = compute_temperament_summary(observations)
    conditional = compute_conditional_patterns(observations)

    # Source-driven gate notes
    src_note = None
    off_ball_cov = (
        source_profile.get("visibility_scores", {}).get("off_ball_coverage_score", 1)
    )
    if off_ball_cov < 0.4 and temperament is None:
        src_note = (
            f"Source off_ball_coverage_score {off_ball_cov:.2f} below 0.4 -- "
            "temperament observations were not produced; partnership and "
            "off-ball movement coverage limited."
        )

    return {
        "player":                          f"#{player_rec['shirt']} {player_rec['name']}".strip(),
        "shirt":                           player_rec["shirt"],
        "team":                            player_rec["team"],
        "position":                        player_rec["position"],

        "minutes_visible_approx":          minutes,
        "minutes_source":                  source,
        "insufficient_observation_time":   minutes < INSUFFICIENT_TIME_THRESHOLD_MINUTES,

        "observation_count_total":         len(observations),
        "observation_count_by_window":     by_window,
        "observation_count_by_category":   by_cat,
        "observation_count_by_type":       by_type,

        "temporal_arc":                    arc_label,
        "temporal_arc_evidence":           arc_evidence,

        "signature_actions":               signatures,
        "decisive_moments":                decisive,
        "partnerships":                    partnerships,

        "comparative_rank_in_position": {
            "rank_by_observation_count":   None,
            "rank_by_strength_count":      None,
            "rank_by_weakness_count":      None,
        },

        "match_aggregate_counts":          aggregate_counts,
        "scarcity_traits":                 [],   # filled in cross-player phase
        "position_role_mismatch_flags":    mismatch_flags,
        "tactical_fouling_indicator":      tactical_fouling,
        "temperament_summary":             temperament,
        "conditional_patterns":            conditional,
        "source_limitations_applied":      src_note,
    }


# ============================================================================
# Observations index
# ============================================================================

def index_observations_by_player(running_summary):
    """Returns dict: (team, shirt) -> list of observation records."""
    by_player = defaultdict(list)
    for obs in running_summary.get("individual_observations", []) or []:
        team = obs.get("team")
        shirt = extract_shirt_number(obs.get("player"))
        if team and shirt is not None:
            by_player[(team, shirt)].append(obs)
    return by_player


# ============================================================================
# Public entrypoint
# ============================================================================

def build_player_summary_cards(match_dir):
    """
    Reads:
        running_summary.json
        match_config.json
        source_profile.json
    Writes:
        player_summary_cards.json
    """
    paths = {
        "summary":  os.path.join(match_dir, "running_summary.json"),
        "config":   os.path.join(match_dir, "match_config.json"),
        "profile":  os.path.join(match_dir, "source_profile.json"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required input: {p}")

    with open(paths["summary"]) as f:
        running_summary = json.load(f)
    with open(paths["config"]) as f:
        match_config = json.load(f)
    with open(paths["profile"]) as f:
        source_profile = json.load(f)

    roster = extract_roster(match_config)
    obs_index = index_observations_by_player(running_summary)
    foul_index = index_fouls_by_player(running_summary)

    cards = []
    for player_rec in roster:
        card = build_card(
            player_rec, running_summary, match_config, source_profile,
            foul_index, obs_index,
        )
        cards.append(card)

    # Cross-player phase
    compute_comparative_rankings(cards)
    compute_scarcity_traits(cards, obs_index)

    output = {
        "match":             match_config.get("match"),
        "focus_team":        match_config.get("focus_team"),
        "schema_version":    "v2.1",
        "cards":             cards,
        "card_count":        len(cards),
        "source_summary": {
            "source_type":              source_profile.get("source_type"),
            "off_ball_coverage_score":  (
                source_profile.get("visibility_scores", {}).get("off_ball_coverage_score")
            ),
        },
    }

    out_path = os.path.join(match_dir, "player_summary_cards.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[player_aggregator] Wrote {out_path}")
    print(f"  Cards produced: {len(cards)}")
    n_insufficient = sum(1 for c in cards if c["insufficient_observation_time"])
    print(f"  Insufficient observation time: {n_insufficient}")
    n_fading = sum(1 for c in cards if c["temporal_arc"] == "fading_late")
    print(f"  Fading-late flagged: {n_fading}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python player_aggregator.py [MATCH_DIR]")
        sys.exit(1)
    build_player_summary_cards(sys.argv[1])
