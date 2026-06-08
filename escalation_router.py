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
import re as _re
import sys
import glob
from datetime import datetime
from pipeline_paths import find_all_merged_windows
from pipeline_schemas import stamp_schema_version


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


def _load_match_boundaries(match_dir: str) -> dict:
    """Load match_boundaries.json. Returns the boundaries dict or {} if absent."""
    path = os.path.join(match_dir, "match_boundaries.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get("boundaries", {}) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _infer_half(merged_dict: dict, merged_filename: str, timestamp: str) -> str:
    """Determine which half a merged window covers.

    Resolution order:
      1. merged_dict["half"] if present and non-empty (modern merge_utils
         writes this; Bayern's older files sometimes have it on 2H windows
         but not 1H).
      2. merged_dict["window"] label if it carries a "1H_" / "2H_" / "ET1_"
         / "ET2_" prefix (Gorleston-era merge output).
      3. The filename pattern (e.g. "agent_agent_11_W11_2H_50-55min_merged.json").
         Robust because window_plan-driven merge naming embeds the half.
      4. A timestamp heuristic as last resort: MM < 45 -> "1H", >= 45 -> "2H".
         Imperfect (early-2H stoppage time can roll above 45 too), used only
         when filename + dict both fail.

    Returns "1H" / "2H" / "ET1" / "ET2" / "" if nothing matched.
    """
    half = merged_dict.get("half")
    if isinstance(half, str) and half.upper() in ("1H", "2H", "ET1", "ET2"):
        return half.upper()
    label = merged_dict.get("window") or ""
    if isinstance(label, str):
        for prefix in ("1H", "2H", "ET1", "ET2"):
            if label.startswith(prefix + "_") or label.startswith(prefix + " "):
                return prefix
    if merged_filename:
        for prefix in ("1H", "2H", "ET1", "ET2"):
            if f"_{prefix}_" in merged_filename:
                return prefix
    if isinstance(timestamp, str):
        m = _re.match(r"^(\d+)m\d+s$", timestamp.strip())
        if m:
            mm = int(m.group(1))
            return "1H" if mm < 45 else "2H"
    return ""


def _match_to_video_seconds(timestamp: str, half: str, boundaries: dict) -> int:
    """v3.0.1 bundle Task 141b: convert match-clock timestamp to video-clock
    seconds using match_boundaries.json kickoff offsets.

    Why: confirmation_queue.json items store match-clock timestamps
    ("02m45s" = 2 min 45 sec into the first half), which are correct for
    display in reports and joining to merged-window set_piece records (which
    also use match-clock). But the burst frame extraction uses the timestamp
    as a video-clock offset, which is wrong when kickoff is not at video
    time 0 (typical: 30-90s of warmup before kickoff).

    This helper computes the video-clock equivalent. Queue items get an
    additional `anchor_video_s` field with the correct video seconds; the
    burst block reads that field for extract_segment instead of re-parsing
    the match-clock string.

    `half` is one of "1H" / "2H" / "ET1" / "ET2" (determined by
    _infer_half above).

    Returns None if conversion not possible (missing boundaries, unparseable
    input, or unknown half). Falls back to legacy match-clock behavior at
    the read site if None.
    """
    if not boundaries:
        return None
    if not timestamp or not isinstance(timestamp, str):
        return None
    m = _re.match(r"^(\d+)m(\d+)s$", timestamp.strip())
    if not m:
        return None
    mc_sec = int(m.group(1)) * 60 + int(m.group(2))
    half = (half or "").upper()
    if half == "1H":
        ko = boundaries.get("ko_1h", {}).get("seconds")
        if ko is None: return None
        return mc_sec + ko
    if half == "2H":
        ko = boundaries.get("ko_2h", {}).get("seconds")
        if ko is None: return None
        # 2H queue timestamps continue from 45:00 (e.g. "63m42s" = 63 min
        # match-clock). Subtract 45 min to get into-2H seconds, then add the
        # video-clock kickoff offset.
        return (mc_sec - 45 * 60) + ko
    if half in ("ET1", "ET2"):
        # Extra time: rare. Use ko_2h as best-effort anchor; downstream
        # consumers should refine when extra-time matches appear.
        ko = boundaries.get("ko_2h", {}).get("seconds")
        if ko is None: return None
        return (mc_sec - 45 * 60) + ko
    return None


def build_escalation_queue(match_dir: str) -> dict:
    logs_dir = os.path.join(match_dir, "agent_logs")

    # Load escalation rules (config overrides > defaults)
    rules = dict(DEFAULT_ESCALATION_RULES)
    rules.update(load_config_overrides(match_dir))

    # v3.0.1 bundle Task 141b: load match_boundaries once for video-clock conversion.
    boundaries = _load_match_boundaries(match_dir)

    # Collect all queue items from merged window JSONs
    raw_items = []
    # Find all merged windows, including legacy v1 merged_windows/ dir
    merged_paths = find_all_merged_windows(
        logs_dir,
        extra_dirs=[os.path.join(match_dir, "merged_windows")],
    )
    for path in merged_paths:
        with open(path, encoding="utf-8") as f:
            w = json.load(f)
        # v3.0.1 bundle (Task 140): unify on agent_id as the queue's window
        # field. Modern merged JSONs populate BOTH "window" (label like
        # "1H_00-00_05-00") and "agent_id" (numeric like "01"). The original
        # `window or agent_id` preference picked the label, which downstream
        # readers (find_merged_window in pipeline_paths.py, the burst block
        # in pipeline_runner_v2.py, and setpiece_writeback.py) cannot resolve
        # because find_merged_window's patterns expect the numeric form. Flip
        # the preference: agent_id wins; label kept as legacy fallback for
        # older matches that only populated `window`.
        window = w.get("agent_id") or w.get("window") or ""
        source = os.path.basename(path)
        # v3.0.1 bundle Task 141b: determine which half this window covers so
        # we can compute video-clock seconds for queue items. _infer_half
        # walks several fallbacks: merged.half field, merged.window label,
        # filename pattern, timestamp heuristic.
        # ARCHITECTURAL INVARIANT (F8):
        # The standalone confirmation_queue.json is the canonical source of
        # truth for confirmation state. Embedded confirmation_queue arrays
        # inside agent_*_merged.json files are write-once data, produced by
        # accumulator.py and read only here for the consolidation pass below.
        # No other consumer reads them.
        #
        # As a consequence, setpiece_writeback.py marks items as resolved:true
        # ONLY in the standalone file. The embedded copies retain their
        # original unresolved state forever. This is intentional, not a bug.
        #
        # This invariant depends on the pipeline being run at most once per
        # match directory. If a re-run path is ever added (recovery workflow,
        # partial re-processing, repeated confirmation passes), the embedded
        # queues will be re-consolidated and resolved items will re-appear in
        # the standalone queue. See SKILL.md "Pipeline Invariants" section.
        for item in w.get("confirmation_queue", []):
            item["window"] = window
            item["source"] = source
            # v3.0.1 bundle Task 141b: video-clock for frame extraction.
            _half = _infer_half(w, source, item.get("timestamp"))
            item["anchor_video_s"] = _match_to_video_seconds(
                item.get("timestamp"), _half, boundaries
            )
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
            _half = _infer_half(w, source, ts)
            raw_items.append({
                "event_type":     "set_piece_delivery",
                "timestamp":      ts,
                # v3.0.1 bundle Task 141b: video-clock seconds for frame
                # extraction. None if match_boundaries.json absent or half
                # cannot be inferred; burst block falls back to legacy
                # _ts2s(timestamp) at that point.
                "anchor_video_s": _match_to_video_seconds(ts, _half, boundaries),
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
        json.dump(stamp_schema_version(out, "confirmation_queue"), f, indent=2)

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
