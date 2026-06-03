"""
watch_list_aggregator.py

Helpers for managing the watch_list across the pipeline.

Functions:
    generate_watch_list_id(item) -> str
        Stable hash for a watch_list entry so confirmations across windows
        can refer back to the same item.

    inject_watch_list_into_prompt(match_config) -> str
        Returns the formatted block to paste into the player agent prompt
        at the [WATCH LIST] placeholder.

    aggregate_confirmations(match_dir) -> dict
        Reads all watch_list_confirmations from running_summary.json,
        returns per-item verdicts. (Final verdict logic also lives in
        build_deep_skill_metrics_v2 -- this is for ad-hoc inspection.)
"""

import hashlib
import json
import os


def generate_watch_list_id(item):
    """
    Stable short hash from player + team + watch_for.
    Returns a 12-char hex string.

    item -- dict like {"player": "#7", "team": "away",
                       "watch_for": "cuts inside onto left foot"}
    """
    key = f"{item.get('player', '')}|{item.get('team', '')}|{item.get('watch_for', '')}"
    h = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return h[:12]


def inject_watch_list_into_prompt(match_config):
    """
    Build the formatted watch list block for the player agent prompt.
    Returns 'Watch list empty for this match.' if no watch_list present.
    """
    items = match_config.get("watch_list") or []
    if not items:
        return "Watch list empty for this match."

    lines = []
    for item in items:
        wid = generate_watch_list_id(item)
        lines.append(
            f"  watch_list_id: {wid}\n"
            f"    player:    {item.get('player')}\n"
            f"    team:      {item.get('team')}\n"
            f"    watch_for: {item.get('watch_for')}\n"
            f"    source:    {item.get('source')}"
        )
    return "\n\n".join(lines)


def aggregate_confirmations(match_dir):
    """
    Reads running_summary.json and returns per-item confirmation counts.
    Used for ad-hoc inspection -- the canonical verdict logic is in
    build_deep_skill_metrics_v2._metric_watch_list_summary().
    """
    summary_path = os.path.join(match_dir, "running_summary.json")
    if not os.path.exists(summary_path):
        return {}

    with open(summary_path) as f:
        summary = json.load(f)

    confirmations = summary.get("watch_list_confirmations", [])
    by_id = {}
    for c in confirmations:
        wid = c.get("watch_list_id")
        if not wid:
            continue
        rec = by_id.setdefault(wid, {
            "confirmed":    0,
            "refuted":      0,
            "not_observed": 0,
            "details":      [],
        })
        status = c.get("status")
        if status == "confirmed":
            rec["confirmed"] += 1
        elif status == "refuted":
            rec["refuted"] += 1
        elif status == "not_observed_this_window":
            rec["not_observed"] += 1
        rec["details"].append({
            "window": c.get("window"),
            "status": status,
            "notes":  c.get("notes"),
        })
    return by_id


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python watch_list_aggregator.py [MATCH_DIR]")
        sys.exit(1)
    result = aggregate_confirmations(sys.argv[1])
    print(json.dumps(result, indent=2))
