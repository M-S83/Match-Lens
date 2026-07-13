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

    Status-aware (v3.0.1 bundle): when burst.status == "inconclusive", do
    NOT merge confirmed_fields or burst_fields. Instead mark the 1fps record
    with burst_verdict + burst_rejection_reason so downstream consumers
    (synthesis, report writers) can treat the original 1fps emission with
    appropriate suspicion. This is the architectural escape that lets the
    5fps burst surface upstream 1fps fabrications without baking them into
    the merged dataset.
    """
    if record.get("burst_resolved"):
        return [], []

    # Inconclusive verdict: do not merge field data; flag for downstream suspicion.
    burst_status = burst.get("status")
    if burst_status == "inconclusive":
        rejection = burst.get("rejection_reason") or "(no reason provided)"
        record["burst_verdict"]          = "inconclusive"
        record["burst_rejection_reason"] = rejection
        record["burst_fps"]              = burst.get("burst_fps", 5)
        record["burst_resolved"]         = True
        record["resolved_at"]            = datetime.utcnow().isoformat() + "Z"
        return ["burst_verdict", "burst_rejection_reason"], []

    changed = []
    corrections_logged = []

    # Default behaviour (status="confirmed" or absent — legacy bursts):
    # merge confirmed_fields + burst_fields per existing logic.

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
    record["burst_verdict"]  = "confirmed"  # explicit (paired with the inconclusive branch above)
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


def check_writeback_integrity(match_dir: str) -> dict:
    """
    Step 27 -- the permanent safety net for this whole module.

    WHY THIS EXISTS
    ----------------
    We found a real case (the Gorleston vs Tilbury match) where 10 set-piece
    bursts were paid for and successfully resolved by the API (real,
    genuinely useful data: delivery type, runner roles, wall setup, second-
    phase outcome -- confirmation_queue.json shows resolved: true for all
    10, with real *_setpiece.json result files on disk), yet NONE of that
    data was present in running_summary.json's set_pieces[] -- every entry
    still showed the crude 1fps-only version (burst_resolved: false,
    delivery_type: null, runners: null, ...).

    We could not pin down with certainty WHY the merge was lost -- it may
    have been a silently-swallowed exception in pipeline_runner_v2.py's
    PHASE 3b (which wraps the whole set-piece burst phase in a broad
    `except Exception: print(...)  # non-blocking` handler, so a real
    failure there produces nothing but an easy-to-miss print line), or a
    later re-run of the ordinary window-merge step that rebuilt the merged
    window file from the original 1fps agent output and had no knowledge
    that a burst confirmation had already been applied to it. Both are
    plausible; we verified writeback_burst()'s own merge logic is correct
    (it applied cleanly to a test copy of this exact real data), so the bug
    is in WHEN/WHETHER the merge happens, not in the merge logic itself.

    Rather than patch one specific guessed cause, this function is a
    detect-and-flag check that catches the SYMPTOM regardless of root
    cause: "we have a resolved, paid-for confirmation that isn't reflected
    in the report-facing data." Paired with the fact that writeback_burst()
    is already idempotent (it skips records already burst_resolved: True),
    this turns into a safe, always-correct pattern: run this check on every
    match (via build_readiness_check, so it happens automatically, with
    zero human review needed -- required for the eventual unattended,
    per-client pipeline); if it reports a gap, re-running writeback_all_bursts()
    is always safe and always closes that specific gap.

    WHAT IT DOES
    ------------
    For every item in confirmation_queue.json with event_type ==
    "set_piece_delivery" and resolved == True (meaning: we already paid for
    and received this confirmation), check whether running_summary.json's
    matching set_piece record (same (timestamp, team) key writeback_burst()
    itself uses) shows burst_resolved: True. Any resolved item whose summary
    record does NOT show burst_resolved: True is a genuine gap -- money
    already spent, value not yet delivered to the reports.

    Only covers set-piece confirmations for now, since that's the one
    burst-based confirmation flow that writes into running_summary.json's
    set_pieces[] via this exact (timestamp, team) matching key. If another
    item type (e.g. player-action confirmations) gains an equivalent
    burst-merge flow in future, it will need its own branch here -- this
    function does not currently check anything about player_escalation_queue.json.

    Returns a dict: resolved_count, merged_count, gap_count, gap_items
    (each gap item is {"timestamp":..., "team":...} for diagnostics).
    A match with no confirmation_queue.json at all returns all-zero counts --
    "nothing was ever queued for confirmation" is a real, valid state (not
    an error), not a gap.
    """
    cq_path = os.path.join(match_dir, "confirmation_queue.json")
    summary_path = os.path.join(match_dir, "running_summary.json")

    if not os.path.exists(cq_path):
        return {"resolved_count": 0, "merged_count": 0, "gap_count": 0, "gap_items": []}

    with open(cq_path, encoding="utf-8") as f:
        cq = json.load(f)

    resolved_items = [
        item for item in cq.get("items", [])
        if item.get("event_type") == "set_piece_delivery" and item.get("resolved") is True
    ]

    if not resolved_items:
        return {"resolved_count": 0, "merged_count": 0, "gap_count": 0, "gap_items": []}

    # No confirmations were ever paid for, but running_summary.json is also
    # missing entirely -- that's a much bigger problem than this function is
    # meant to catch (build_readiness_check's own "windows incomplete" check
    # already blocks on that). Treat every resolved item as a gap here so the
    # caller still sees an accurate count rather than a silent divide-by-zero.
    if not os.path.exists(summary_path):
        gap_items = [{"timestamp": i.get("timestamp"), "team": i.get("team")} for i in resolved_items]
        return {
            "resolved_count": len(resolved_items),
            "merged_count": 0,
            "gap_count": len(gap_items),
            "gap_items": gap_items,
        }

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    set_pieces = summary.get("set_pieces", [])

    gap_items = []
    for item in resolved_items:
        ts   = item.get("timestamp")
        team = item.get("team")
        record = find_matching_set_piece(set_pieces, ts, team)
        if record is None or not record.get("burst_resolved"):
            gap_items.append({"timestamp": ts, "team": team})

    return {
        "resolved_count": len(resolved_items),
        "merged_count":   len(resolved_items) - len(gap_items),
        "gap_count":       len(gap_items),
        "gap_items":       gap_items,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python setpiece_writeback.py <match_dir>")
        sys.exit(1)
    writeback_all_bursts(sys.argv[1])
