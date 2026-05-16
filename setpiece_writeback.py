"""
setpiece_writeback.py -- Step 3d-SP-WB

Merges 5fps burst outputs (from the set piece agent) back into the
matching set_piece records in the merged window file and running_summary.json.

Called after all set piece bursts have completed (Step 3d-SP).

Usage:
    from setpiece_writeback import writeback_all_bursts, writeback_burst

    # Process all burst files in one pass:
    writeback_all_bursts(match_dir)

    # Or process a single burst:
    writeback_burst(burst_path, merged_window_path, summary_path)
"""

import glob
import json
import os
from datetime import datetime
from pipeline_paths import find_merged_window
from pipeline_schemas import stamp_schema_version

# Fields the burst confirms (1fps read them; 5fps re-reads for accuracy)
BURST_CONFIRMS = {"bodies_in_box", "delivery_zone", "marking_system", "outcome"}

# Fields the burst alone populates (1fps leaves them null)
BURST_ONLY = {
    "delivery_type", "delivery_target_zone", "runners",
    "wall_size", "wall_position", "second_phase",
}


def find_matching_set_piece(set_pieces: list, anchor_ts: str, team: str):
    """
    Find the set_piece record this burst output corresponds to.
    Match key is (timestamp, team) -- timestamps are unique within a
    team because two set pieces from the same team cannot happen at the
    same MMmSSs.
    Returns the matching dict (mutable, in place), or None.
    """
    for sp in set_pieces:
        if sp.get("timestamp") == anchor_ts and sp.get("team") == team:
            return sp
    return None


def apply_burst_to_record(record: dict, burst: dict) -> tuple:
    """
    Patch the set_piece record in place with burst output.
    Returns (changed_fields, corrections_logged).
    Idempotent: if burst_resolved is already True, skips re-application.
    """
    if record.get("burst_resolved"):
        return [], []

    changed = []
    corrections_logged = []

    # Confirmation fields: write burst value, log if it differs from 1fps
    confirmed = burst.get("confirmed_fields", {})
    for field in BURST_CONFIRMS:
        if field not in confirmed:
            continue
        new_val = confirmed[field]
        old_val = record.get(field)
        if old_val != new_val:
            corrections_logged.append({
                "field":       field,
                "prior_value": old_val,
                "new_value":   new_val,
                "source":      "burst_correction",
            })
        record[field] = new_val
        changed.append(field)

    # Explicit corrections the agent flagged (with reasons)
    for corr in burst.get("corrections", []):
        corrections_logged.append({**corr, "source": "burst_agent_explicit"})

    # Burst-only fields: write directly (1fps left them null)
    burst_fields = burst.get("burst_fields", {})
    for field in BURST_ONLY:
        if field in burst_fields:
            record[field] = burst_fields[field]
            changed.append(field)

    # Audit trail
    record["burst_resolved"] = True
    record["burst_fps"]      = burst.get("burst_fps", 5)
    record["source"]         = "1fps_plus_5fps_burst"
    record["corrections"]    = record.get("corrections", []) + corrections_logged
    record["burst_notes"]    = burst.get("burst_notes", "")
    record["resolved_at"]    = datetime.utcnow().isoformat() + "Z"

    return changed, corrections_logged


def writeback_burst(burst_path: str, merged_window_path: str, summary_path: str,
                    queue_path: "str | None" = None) -> bool:
    """
    Apply one burst output to the matching set_piece record in the merged
    window file, then mirror the patch into running_summary.json and mark
    the confirmation_queue entry resolved.

    Returns True if a matching record was found and patched, False on orphan.
    """
    with open(burst_path, encoding="utf-8") as f:
        burst = json.load(f)

    anchor_ts = burst.get("anchor_timestamp")
    if not anchor_ts:
        raise ValueError(f"Burst output missing anchor_timestamp: {burst_path}")

    team = burst.get("team") or burst.get("source_record", {}).get("team")
    if not team:
        raise ValueError(f"Burst output missing team: {burst_path}")

    # --- Patch the merged window file ---
    with open(merged_window_path, encoding="utf-8") as f:
        merged = json.load(f)

    record = find_matching_set_piece(merged.get("set_pieces", []), anchor_ts, team)

    if record is None:
        # Orphan: burst for a set piece that doesn't exist in the merged window.
        # Should be impossible if auto-queue is working correctly.
        orphan_path = merged_window_path.replace(".json", "_orphan_setpieces.json")
        with open(orphan_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "anchor_timestamp": anchor_ts,
                "team":             team,
                "burst_path":       burst_path,
                "detected_at":      datetime.utcnow().isoformat() + "Z",
            }) + "\n")
        print(f"  [ORPHAN] No matching set_piece for {anchor_ts}/{team} "
              f"in {os.path.basename(merged_window_path)}")
        return False

    changed, corrections = apply_burst_to_record(record, burst)

    with open(merged_window_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(merged, "agent_merged"), f, indent=2)

    # --- Mirror patch into running_summary.json ---
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        summary_record = find_matching_set_piece(
            summary.get("set_pieces", []), anchor_ts, team
        )
        if summary_record is not None:
            apply_burst_to_record(summary_record, burst)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(stamp_schema_version(summary, "running_summary"), f, indent=2)

    # --- Mark confirmation_queue entry resolved ---
    _queue_path = queue_path or os.path.join(
        os.path.dirname(summary_path), "confirmation_queue.json"
    )
    if os.path.exists(_queue_path):
        with open(_queue_path, encoding="utf-8") as f:
            queue = json.load(f)

        for item in queue.get("items", []):
            if (item.get("event_type") == "set_piece_delivery"
                    and item.get("timestamp") == anchor_ts
                    and item.get("team") == team):
                item["resolved"]    = True
                item["resolved_at"] = datetime.utcnow().isoformat() + "Z"
                item["resolved_by"] = "3d-SP"
                break

        with open(_queue_path, "w", encoding="utf-8") as f:
            json.dump(stamp_schema_version(queue, "confirmation_queue"), f, indent=2)

    print(f"  [WB] {anchor_ts}/{team} -- "
          f"{len(changed)} fields updated, "
          f"{len(corrections)} corrections logged")
    return True


def writeback_all_bursts(match_dir: str, summary_path: "str | None" = None,
                          queue_path: "str | None" = None) -> dict:
    """
    Walk agent_logs/ for *_setpiece.json burst output files and apply each to
    the matching merged window file.

    Uses the 'window' field inside the burst JSON to locate the merged file --
    filename parsing is unreliable when window_id contains underscores (which
    it does: 'agent_07_20m03s'). Falls back to glob if the direct path misses.

    Idempotent: apply_burst_to_record skips records already marked
    burst_resolved: True.

    Returns summary dict with counts.
    """
    logs_dir     = os.path.join(match_dir, "agent_logs")
    _summary     = summary_path or os.path.join(match_dir, "running_summary.json")

    burst_files  = sorted([
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if f.endswith("_setpiece.json")
    ])

    if not burst_files:
        print("  [WB] No set piece burst outputs found")
        return {"applied": 0, "orphans": 0, "errors": []}

    applied  = 0
    orphans  = 0
    errors   = []

    print(f"\n  Step 3d-SP-WB -- Set piece burst write-back ({len(burst_files)} file(s))")
    print(f"  {'-'*50}")

    for burst_path in burst_files:
        base = os.path.basename(burst_path)
        try:
            with open(burst_path, encoding="utf-8") as f:
                burst = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  [WB] Skip {base}: {e}")
            errors.append(f"{base}: {e}")
            continue

        # Window id is canonical -- read from inside the burst JSON.
        window_id = burst.get("window")
        if not window_id:
            print(f"  [WB] Skip {base}: no window field in burst JSON")
            errors.append(f"{base}: no window field")
            continue

        # Find merged file via canonical lookup (handles label suffixes).
        merged_path = find_merged_window(logs_dir, window_id)
        if merged_path is None:
            print(f"  [WB] Skip {base}: no merged file for window {window_id!r}")
            errors.append(f"{base}: no merged file for window {window_id!r}")
            continue

        try:
            ok = writeback_burst(burst_path, merged_path, _summary, queue_path)
            if ok:
                applied += 1
            else:
                orphans += 1
        except Exception as e:
            errors.append(f"{base}: {e}")
            print(f"  [ERROR] {base}: {e}")

    print(f"  {'-'*50}")
    print(f"  Applied: {applied} | Orphans: {orphans} | Errors: {len(errors)}")

    return {"applied": applied, "orphans": orphans, "errors": errors}


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python setpiece_writeback.py <match_dir>")
        sys.exit(1)
    writeback_all_bursts(sys.argv[1])
