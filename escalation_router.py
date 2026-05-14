"""
escalation_router.py -- Match Lens Step 3i escalation logic

Replaces the simple event-type whitelist with a rules-based router that:
  - reads all confirmation_queue entries from merged window JSONs
  - determines the correct escalation fps tier per result_family / evidence_tier
  - enforces the cap: 10 items max UNLESS there are more than 10 goals
    (if goals > 10, all goals are processed; other high-priority items fill remaining cap)
  - writes confirmation_queue.json

Cap rule:
  - If total goals ≤ 10: standard cap of 10 high-priority items applies
  - If total goals > 10: goals are uncapped; other high-priority items capped at
    max(0, 10 - len(non_goal_high)) to respect operator intent on unusual matches

Usage:
    python escalation_router.py [MATCH_DIR]
"""

import json
import os
import sys
import glob
from datetime import datetime


STANDARD_CAP_HIGH   = 10
MAX_MEDIUM_ALLOWED  = 4    # only if high count < 6

ELIGIBLE_FAMILIES = {
    "transitions", "counterpress", "box_entries", "line_breaking_actions",
    "local_duels", "pressing", "build_up", "chance_patterns", "phase",
    "opposition_transitions", "opposition_pressing",
    "player_duels", "player_movement",
}

ALWAYS_ESCALATE_TYPES = {
    "goal", "gk_claim", "gk_punch", "gk_parry",
    "cross_six_yard", "goalmouth_scramble", "rebound",
    "set_piece_delivery",
}

DEFAULT_ESCALATION_RULES = {
    "transitions":            {"target_fps": 5, "reason": "fast_event",   "padding_s": 5},
    "counterpress":           {"target_fps": 5, "reason": "fast_event",   "padding_s": 5},
    "box_entries":            {"target_fps": 5, "reason": "fast_event",   "padding_s": 3},
    "line_breaking_actions":  {"target_fps": 5, "reason": "fast_event",   "padding_s": 5},
    "local_duels":            {"target_fps": 5, "reason": "fast_event",   "padding_s": 3},
    "pressing":               {"target_fps": 3, "reason": "uncertainty",  "padding_s": 5},
    "build_up":               {"target_fps": 3, "reason": "uncertainty",  "padding_s": 8},
    "chance_patterns":        {"target_fps": 3, "reason": "uncertainty",  "padding_s": 5},
    "phase":                  {"target_fps": 3, "reason": "uncertainty",  "padding_s": 5},
    "opposition_transitions": {"target_fps": 5, "reason": "fast_event",   "padding_s": 5},
    "opposition_pressing":    {"target_fps": 3, "reason": "uncertainty",  "padding_s": 5},
    "player_duels":           {"target_fps": 5, "reason": "fast_event",   "padding_s": 3},
    "player_movement":        {"target_fps": 3, "reason": "uncertainty",  "padding_s": 5},
    "goal":                   {"target_fps": 5, "reason": "importance",   "padding_s": 6},
    "gk_claim":               {"target_fps": 3, "reason": "uncertainty",  "padding_s": 3},
    "gk_punch":               {"target_fps": 3, "reason": "uncertainty",  "padding_s": 3},
    "gk_parry":               {"target_fps": 3, "reason": "uncertainty",  "padding_s": 3},
    "cross_six_yard":         {"target_fps": 3, "reason": "uncertainty",  "padding_s": 3},
    "goalmouth_scramble":     {"target_fps": 5, "reason": "fast_event",   "padding_s": 3},
    "rebound":                {"target_fps": 5, "reason": "fast_event",   "padding_s": 3},
    "set_piece_delivery":     {"target_fps": 5, "reason": "density",     "padding_s": 3},
}


def load_config_overrides(match_dir: str) -> dict:
    """Load source_profiles_config.json escalation rules if available."""
    config_path = os.path.join(
        os.path.dirname(__file__), "source_profiles_config.json"
    )
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg.get("_escalation_rules", {})
    return {}


def timestamp_to_seconds(ts: str) -> float:
    """Convert 'MMmSSs' to seconds."""
    try:
        ts = ts.replace("m", ":").replace("s", "")
        parts = ts.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except Exception:
        return 0.0


def seconds_to_timestamp(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:02d}m{sec:02d}s"


def determine_escalation(item: dict, rules: dict) -> dict:
    """
    Determine target fps, reason, and rerun window for a single queue item.
    item must have: event_type OR result_family, timestamp, priority.
    """
    event_type    = item.get("event_type", "")
    result_family = item.get("result_family", "")
    evidence_tier = item.get("evidence_tier", "suggestive")
    confidence    = item.get("confidence_before_rerun", 0.5)

    # Determine rule key -- event_type takes priority over result_family
    rule_key = None
    if event_type in rules:
        rule_key = event_type
    elif result_family in rules:
        rule_key = result_family

    if rule_key is None:
        # Default: escalate suggestive to 3fps, high-importance to 5fps
        if evidence_tier == "suggestive" or confidence < 0.7:
            target_fps = 5 if item.get("priority") == "high" else 3
            reason     = "uncertainty"
            padding    = 5
        else:
            return None  # No escalation needed
    else:
        rule       = rules[rule_key]
        target_fps = rule["target_fps"]
        reason     = rule["reason"]
        # Accept both "padding_s" (new) and "default_padding_s" (legacy config key)
        padding    = rule.get("padding_s") or rule.get("default_padding_s") or 5

    # Compute rerun window with padding
    ts_s   = timestamp_to_seconds(item.get("timestamp", "00m00s"))
    w_start = max(0.0, ts_s - padding)
    w_end   = ts_s + padding

    item["escalation_target_fps"]  = target_fps
    item["escalation_reason"]      = reason
    item["rerun_window_start"]     = seconds_to_timestamp(w_start)
    item["rerun_window_end"]       = seconds_to_timestamp(w_end)
    item["status"]                 = "queued"
    return item


def is_goal_item(item: dict) -> bool:
    return item.get("event_type") == "goal"


def count_match_goals(match_dir: str) -> int:
    """Read goal count from match_config.json."""
    config_path = os.path.join(match_dir, "match_config.json")
    if not os.path.exists(config_path):
        return 0
    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    goals = config.get("goals") or []
    return len(goals)


def build_escalation_queue(match_dir: str) -> dict:
    logs_dir = os.path.join(match_dir, "agent_logs")

    # Load escalation rules (config overrides > defaults)
    rules = dict(DEFAULT_ESCALATION_RULES)
    rules.update(load_config_overrides(match_dir))

    # Collect all queue items from merged window JSONs
    raw_items = []
    # Check both naming conventions used by pipeline_runner.py
    merged_paths = (
        sorted(glob.glob(os.path.join(logs_dir, "*_merged.json"))) +
        sorted(glob.glob(os.path.join(match_dir, "merged_windows", "merged_*.json"))) +
        sorted(glob.glob(os.path.join(logs_dir, "agent_*_merged.json")))
    )
    # Deduplicate
    seen = set()
    merged_paths = [p for p in merged_paths if not (p in seen or seen.add(p))]
    for path in merged_paths:
        with open(path, encoding="utf-8") as f:
            w = json.load(f)
        # "window" is set by dual-merge; single-agent merges use "agent_id"
        window = w.get("window") or w.get("agent_id") or ""
        source = os.path.basename(path)
        for item in w.get("confirmation_queue", []):
            item["window"] = window
            item["source"] = source
            raw_items.append(item)

        # Auto-generate set_piece_delivery items from set_pieces[] that have
        # timestamps. Structural agents populate set_pieces[], not
        # confirmation_queue[], so these never appear via the loop above.
        # Dedup by (timestamp, team) against anything already in raw_items.
        _sp_seen = {
            (i.get("timestamp"), i.get("team"))
            for i in raw_items
            if i.get("event_type") == "set_piece_delivery"
        }
        for sp in (w.get("set_pieces") or []):
            ts   = sp.get("timestamp")
            team = sp.get("team")
            if not ts or not team:
                continue  # legacy records without timestamp -- skip
            if sp.get("burst_resolved"):
                continue  # already processed by 5fps burst -- do not re-queue
            if (ts, team) in _sp_seen:
                continue  # already queued
            _sp_seen.add((ts, team))
            raw_items.append({
                "event_type":     "set_piece_delivery",
                "timestamp":      ts,
                "team":           team,
                "priority":       "high",
                "evidence_tier":  "suggestive",
                "window":         window,
                "source":         source,
                "set_piece_type": sp.get("type"),
            })

    # Determine escalation params for each item
    escalated = []
    skipped_ineligible = 0
    for item in raw_items:
        et = item.get("event_type", "")
        rf = item.get("result_family", "")

        # Check eligibility
        if et not in ALWAYS_ESCALATE_TYPES and rf not in ELIGIBLE_FAMILIES and et not in rules and rf not in rules:
            skipped_ineligible += 1
            print(f"  SKIPPED (ineligible): {et or rf} at {item.get('timestamp')}")
            continue

        result = determine_escalation(item, rules)
        if result is not None:
            escalated.append(result)

    # Separate by type and priority
    goals      = [i for i in escalated if is_goal_item(i)]
    non_goals  = [i for i in escalated if not is_goal_item(i)]
    high       = [i for i in non_goals if i.get("priority") == "high"]
    medium     = [i for i in non_goals if i.get("priority") == "medium"]

    # Determine cap
    match_goal_count = count_match_goals(match_dir)
    goals_uncapped   = match_goal_count > STANDARD_CAP_HIGH

    if goals_uncapped:
        print(f"  [!]  {match_goal_count} goals detected -- goal cap lifted (standard cap: {STANDARD_CAP_HIGH})")
        # All goals processed; remaining cap slots for non-goal high items
        remaining_cap    = max(0, STANDARD_CAP_HIGH - len(goals))
        capped_high      = high[:remaining_cap]
        cap_note         = f"Goals uncapped ({match_goal_count}); {remaining_cap} slots for other high-priority items"
        all_high         = goals + capped_high
    else:
        # Goals count toward cap; combined list is capped at STANDARD_CAP_HIGH
        total_high   = goals + high
        capped_high  = total_high[:STANDARD_CAP_HIGH]
        cap_note     = f"Standard cap {STANDARD_CAP_HIGH} applied"
        all_high     = capped_high   # capped_high already contains both goals and non-goal high

    # Medium items: only if total high count < 6
    effective_high_count = len(all_high)
    med_allowed  = medium if effective_high_count < 6 else []
    med_skipped  = len(medium) - len(med_allowed)
    if med_skipped > 0:
        print(f"  {med_skipped} medium-priority items skipped (high count >= 6)")

    all_items    = all_high + med_allowed
    skipped_cap  = len(escalated) - len(all_items)

    # Write confirmation_queue.json
    goals_in_q      = sum(1 for i in all_items if is_goal_item(i))
    high_other_in_q = sum(1 for i in all_items if not is_goal_item(i) and i.get("priority") == "high")

    out = {
        "total":                len(all_items),
        "goals":                goals_in_q,
        "high_priority_other":  high_other_in_q,
        "medium_priority":      len(med_allowed),
        "skipped_ineligible":   skipped_ineligible,
        "skipped_by_cap":       skipped_cap,
        "cap_note":             cap_note,
        "match_goal_count":     match_goal_count,
        "goals_uncapped":       goals_uncapped,
        "items":                all_items,
        "generated_at":         datetime.now().isoformat(),
    }

    out_path = os.path.join(match_dir, "confirmation_queue.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n  Escalation queue built:")
    print(f"    Goals:           {goals_in_q}")
    print(f"    Other high:      {high_other_in_q}")
    print(f"    Medium:          {len(med_allowed)}")
    print(f"    Skipped (cap):   {skipped_cap}")
    print(f"    Skipped (ineligible): {skipped_ineligible}")
    print(f"    Cap rule: {cap_note}")

    for item in all_items:
        print(
            f"  [{item.get('priority','?').upper()}] {item.get('timestamp')} "
            f"-- {item.get('event_type') or item.get('result_family')} "
            f"-> {item.get('escalation_target_fps')}fps "
            f"[{item.get('rerun_window_start')}–{item.get('rerun_window_end')}]"
        )

    return out


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    if not os.path.isdir(match_dir):
        print(f"Error: '{match_dir}' is not a valid directory.")
        sys.exit(1)
    build_escalation_queue(match_dir)
