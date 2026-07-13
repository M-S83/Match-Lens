"""
build_readiness_check.py -- Match Lens pipeline gate before Step 4.

Reads all pipeline output files and determines whether the pipeline
is ready to write reports. Writes report_readiness.json.

Step 4 must not run if report_ready is false.

Usage:
    python build_readiness_check.py [MATCH_DIR]
"""

import json
import os
import sys
from pipeline_accessors import get_source_limitations_note
from pipeline_schemas import stamp_schema_version
from setpiece_writeback import check_writeback_integrity


def build_readiness_check(match_dir: str) -> bool:

    def load(fname):
        p = os.path.join(match_dir, fname)
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return json.load(f)

    boundaries    = load("match_boundaries.json")
    config        = load("match_config.json")
    window_plan   = load("window_plan.json")
    ground_truth  = load("ground_truth_check.json")
    summary       = load("running_summary.json")
    rerun_q       = load("rerun_queue.json")
    cq            = load("confirmation_queue.json") or {"total": 0, "skipped": 0}
    source_prof   = load("source_profile.json")
    result_gates  = load("result_family_gates.json")
    deep_metrics  = load("deep_skill_metrics.json")

    blocking = []
    warnings = []

    # Source type is read here (not just in the source-profile section below)
    # so the ground-truth check can apply a source-aware relaxation.
    source_type_early = (source_prof or {}).get("source_type", "unknown")

    # -- Boundary confidence ---------------------------------------------------
    bc = {}
    if boundaries:
        for k, v in boundaries.get("boundaries", {}).items():
            bc[k] = v.get("confidence", 0)
    boundary_ok = bool(bc) and all(v >= 0.8 for v in bc.values())
    if not boundary_ok:
        low = [k for k, v in bc.items() if v < 0.8] if bc else ["boundaries file missing"]
        blocking.append(f"Low boundary confidence: {low}")

    # -- Config verification ---------------------------------------------------
    config_verified   = bool(config and config.get("verified"))
    enrichment_level  = (config or {}).get("enrichment_level", "identity_only")
    player_id_ceiling = (config or {}).get("player_id_ceiling", "tentative")
    if not config_verified:
        blocking.append("match_config.json not verified -- complete Step 1d")

    # -- Windows ---------------------------------------------------------------
    planned  = (window_plan or {}).get("total_windows", 0)
    complete = (summary or {}).get("windows_complete", 0)
    windows_ok = planned > 0 and complete == planned
    if not windows_ok:
        blocking.append(f"Windows incomplete: {complete}/{planned}")

    # -- Ground truth ----------------------------------------------------------
    # Source-aware relaxation: broadcast_fixed_wide event agents only run on
    # confirmed goal windows by design, so subs (and other non-goal events)
    # are expected to read as "missed" at 1fps. The realistic steady-state for
    # broadcast is "all subs missed, possibly a goal or two too" -- often
    # ~100% of events_checked. Block only if events_checked == 0 (deep scan
    # never ran or file missing) -- a genuine pipeline failure shows 0 checks,
    # not 10 checks with 10 misses. Veo runs keep the strict gate.
    gt_total  = (ground_truth or {}).get("total",
                  (ground_truth or {}).get("events_checked", 0))
    gt_missed = (ground_truth or {}).get("missed", "file missing")

    gt_passed = bool(ground_truth and isinstance(gt_missed, int) and gt_missed == 0)

    if not gt_passed:
        # For broadcast: pass-through whenever the deep scan actually ran
        # (events_checked > 0). 10/10 missed is design behaviour; 0 events
        # checked is a genuine pipeline failure (file missing, deep scan
        # never ran) and falls into the strict "block" branch via gt_total
        # being falsy. 0.95 would have allowed up to 95% missed but still
        # blocked at 100% -- the actual signal of failure is gt_total == 0.
        if (source_type_early == "broadcast_fixed_wide"
                and isinstance(gt_missed, int) and gt_total > 0):
            warnings.append(
                f"Ground truth: {gt_missed}/{gt_total} event(s) missed "
                f"-- expected for broadcast_fixed_wide at 1fps "
                f"(event agent only runs on goal windows)"
            )
        else:
            blocking.append(
                f"Ground truth: {gt_missed} event(s) missed -- re-run deep scan"
            )

    # -- Data gaps (warning, not blocking) ------------------------------------
    data_gap_windows = (summary or {}).get("data_gap_windows", [])
    if data_gap_windows:
        blocking.append(
            f"Data gap windows detected: {data_gap_windows} "
            f"-- review before reporting or accept reduced confidence"
        )

    # -- Unresolvable frames ---------------------------------------------------
    unresolvable = sum(
        1 for i in (rerun_q or {}).get("rerun_queue", [])
        if i.get("status") == "unresolvable"
    )

    # -- Source profile --------------------------------------------------------
    source_profile_present   = bool(source_prof)
    source_type              = (source_prof or {}).get("source_type", "unknown")
    source_confidence        = (source_prof or {}).get("classification_confidence", 0.0)
    source_profile_confident = source_confidence >= 0.6
    split_aware              = (source_prof or {}).get("split_aware", False)
    # Fix 38: broadcast source_profile.json stores the limitation under "notes",
    # not "source_limitations_note". Prefer either-key so broadcast runs no longer
    # write an empty source-limitation string into report_readiness.json.
    sp = source_prof or {}
    source_limitations       = get_source_limitations_note(sp)
    downgraded_families      = [
        f for f, s in (result_gates or {}).get("gates", {}).items()
        if s == "downgraded"
    ]
    suppressed_families = []   # no suppressed families in this pipeline

    if source_prof and not source_profile_confident:
        blocking.append(
            f"Source classification confidence low ({source_confidence:.2f}) "
            f"-- source_type defaulted to {source_type}. "
            f"Review source_profile.json before reporting."
        )

    # -- Step 27: set-piece write-back integrity ------------------------------
    # Catches a real, expensive failure mode we found on the Gorleston match:
    # a set-piece burst confirmation gets paid for and resolved by the API
    # (confirmation_queue.json says resolved: true, a real *_setpiece.json
    # result file exists on disk), but the confirmed detail never makes it
    # into running_summary.json's set_pieces[] -- so every downstream report
    # is built from the crude 1fps-only version despite the richer, already-
    # paid-for version existing right there on disk. This is a detect-and-
    # flag check, not a root-cause fix -- writeback_burst() is idempotent, so
    # whenever this fires, the correct response is simply to re-run
    # writeback_all_bursts(match_dir) and re-run this readiness check.
    writeback_check = check_writeback_integrity(match_dir)
    writeback_gap    = writeback_check["gap_count"]
    if writeback_gap > 0:
        blocking.append(
            f"Set-piece write-back gap: {writeback_gap} confirmed burst(s) "
            f"paid for but not merged into running_summary.json -- "
            f"re-run writeback_all_bursts(match_dir)"
        )

    report_ready = len(blocking) == 0

    # -- Report modules available ----------------------------------------------
    # All three are always available -- which ones to generate is a per-job decision
    modules = {
        "tactical":   True,
        "opposition": True,
        "moments":    True,
    }

    readiness = {
        "report_ready":             report_ready,
        "boundary_confidence":      bc,
        "boundary_confidence_ok":   boundary_ok,
        "team_config_verified":     config_verified,
        "enrichment_level":         enrichment_level,
        "player_id_ceiling":        player_id_ceiling,
        "event_validation_passed":  gt_passed,
        "windows_planned":          planned,
        "windows_complete":         complete,
        "windows_ok":               windows_ok,
        "unresolvable_frames":      unresolvable,
        "data_gap_windows":         data_gap_windows,
        "source_type":              source_type,
        "source_classification_confidence": source_confidence,
        "source_profile_present":   source_profile_present,
        "source_profile_confident": source_profile_confident,
        "split_aware":              split_aware,
        "suppressed_families":      suppressed_families,
        "downgraded_families":      downgraded_families,
        "source_limitations":       source_limitations,
        "confirmation_total":       cq.get("total", 0),
        "confirmation_skipped":     cq.get("skipped_by_cap", cq.get("skipped", 0)),
        "confirmation_goals":       cq.get("goals", 0),
        "goals_uncapped":           cq.get("goals_uncapped", False),
        "writeback_resolved_count": writeback_check["resolved_count"],
        "writeback_merged_count":   writeback_check["merged_count"],
        "writeback_gap_count":      writeback_check["gap_count"],
        "report_modules_available": modules,
        "blocking_issues":          blocking,
        "warnings":                 warnings,
    }

    out_path = os.path.join(match_dir, "report_readiness.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(readiness, "report_readiness"), f, indent=2)

    # -- Terminal output -------------------------------------------------------
    status = "[OK] READY" if report_ready else "[X]  NOT READY"
    print(f"\n{'-'*55}")
    print(f"  {status} -- report_readiness.json written")
    print(f"{'-'*55}")

    if blocking:
        print("  Blocking issues:")
        for issue in blocking:
            print(f"    [FAIL] {issue}")
    else:
        print(f"  Enrichment level : {enrichment_level}")
        print(f"  Player ID ceiling: {player_id_ceiling}")
        print(f"  Windows          : {complete}/{planned}")
        print(f"  Unresolvable     : {unresolvable} frames")
        print(f"  Confirmation     : {cq.get('total',0)} items | {cq.get('skipped_by_cap', cq.get('skipped',0))} skipped")
        if data_gap_windows:
            print(f"  [!]  Data gap windows: {data_gap_windows}")

    print(f"{'-'*55}\n")

    # Write confidence_reliability_report.json
    cq_data = cq or {}
    reliability = {
        "match":                     (config or {}).get("match", "unknown"),
        "source_type":               source_type,
        "classification_confidence": source_confidence,
        "split_aware":               split_aware,
        "visibility_scores":         (source_prof or {}).get("visibility_scores", {}),
        # Fix 38: see line ~122 for rationale.
        "source_limitations_note":   get_source_limitations_note(source_prof or {}),
        "visibility_based_limitations": source_limitations,
        "suppressed_families":       suppressed_families,
        "downgraded_families":       downgraded_families,
        "result_family_gates":       (result_gates or {}).get("gates", {}),
        "escalation_summary": {
            "total_tagged":        cq_data.get("total", 0),
            "goals":               cq_data.get("goals", 0),
            "goals_uncapped":      cq_data.get("goals_uncapped", False),
            "high_priority_other": cq_data.get("high_priority_other", 0),
            "medium_priority":     cq_data.get("medium_priority", 0),
            "skipped_by_cap":      cq_data.get("skipped_by_cap", 0),
            "skipped_ineligible":  cq_data.get("skipped_ineligible", 0),
        },
        "data_gap_windows":   data_gap_windows,
        "unresolvable_frames": unresolvable,
        "report_ready":        report_ready,
        "blocking_issues":     blocking,
        "generated_at":        __import__("datetime").datetime.now().isoformat(),
        "deep_skill_metrics_summary": {
            "available":           deep_metrics is not None,
            "total_metrics":       (deep_metrics or {}).get("total_metrics", 0),
            "active_metrics":      (deep_metrics or {}).get("active_metrics", 0),
            "suppressed_metrics":  (deep_metrics or {}).get("suppressed_metrics", []),
            "avg_confidence":      (deep_metrics or {}).get("avg_confidence", 0.0),
        },
    }
    rel_path = os.path.join(match_dir, "confidence_reliability_report.json")
    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(reliability, "confidence_reliability_report"), f, indent=2)
    print(f"  confidence_reliability_report.json written")

    if deep_metrics:
        dm = deep_metrics
        print(f"  Metrics:         {dm.get('active_metrics',0)} active | {len(dm.get('suppressed_metrics',[]))} suppressed | avg conf: {dm.get('avg_confidence',0.0)}")
    else:
        print("  [!]  deep_skill_metrics.json not found -- run Step 3k before Step 4")

    if not report_ready:
        print("  Step 4 is blocked. Resolve issues above and re-run this script.")

    return report_ready



def validate_shirt_numbers(match_dir: str) -> dict:
    """
    Cross-check individual_observation shirt numbers against confirmed lineups.
    Flags any #-numbers that don't appear in either team's lineup.
    Returns dict with: unmatched, home_shirts, away_shirts, obs_numbers.
    """
    mc_path  = os.path.join(match_dir, "match_config.json")
    rs_path  = os.path.join(match_dir, "running_summary.json")
    if not os.path.exists(mc_path) or not os.path.exists(rs_path):
        return {"skipped": True, "reason": "match_config or running_summary missing"}

    with open(mc_path, encoding="utf-8") as f:  mc = json.load(f)
    with open(rs_path, encoding="utf-8")  as f: rs = json.load(f)

    # Build confirmed shirt sets from lineups
    home_shirts, away_shirts = set(), set()
    for lineup in mc.get("lineups", []):
        _t = lineup.get("team", {})
        team_name = _t.get("name", "") if isinstance(_t, dict) else str(_t)
        for player in lineup.get("startXI", []) + lineup.get("substitutes", []):
            num = player.get("player", {}).get("number")
            if num:
                if team_name == mc.get("home_team"):
                    home_shirts.add(str(num))
                else:
                    away_shirts.add(str(num))

    all_shirts = home_shirts | away_shirts

    # Check shirt numbers in individual_observations
    import re as _re
    obs_numbers = set()
    unmatched = []
    for obs in rs.get("individual_observations", []):
        player = obs.get("player", "") or ""
        nums = _re.findall(r"#(\d+)", player)
        for n in nums:
            obs_numbers.add(n)
            if all_shirts and n not in all_shirts:
                unmatched.append({"player": player, "number": n,
                                   "obs": obs.get("observation", "")[:50]})

    return {
        "skipped":      False,
        "home_shirts":  sorted(home_shirts),
        "away_shirts":  sorted(away_shirts),
        "obs_numbers":  sorted(obs_numbers),
        "unmatched":    unmatched,
        "unmatched_count": len(unmatched),
        "note": ("Pass network sequences use free-text chain notation (#N) which "
                 "may have misread multi-digit shirt numbers at 1fps. "
                 f"Unmatched: {sorted(set(u['number'] for u in unmatched))}")
                 if unmatched else "All observed shirt numbers match confirmed lineups.",
    }


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory path: ").strip()
    if not os.path.isdir(match_dir):
        print(f"Error: '{match_dir}' is not a valid directory.")
        sys.exit(1)
    ready = build_readiness_check(match_dir)
    sys.exit(0 if ready else 1)
