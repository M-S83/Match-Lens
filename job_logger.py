"""
job_logger.py — Match Lens pipeline runtime and cost logger.

Usage:
    from job_logger import JobLogger
    logger = JobLogger(MATCH_DIR)
    logger.step_start("1b_boundary_detection")
    # ... run step ...
    logger.step_end("1b_boundary_detection", frames_reviewed=42)
    logger.finish(reports_generated=["tactical", "opposition"])
"""

import json
import os
import time
from datetime import datetime
from pipeline_schemas import stamp_schema_version


class JobLogger:
    """
    Logs per-step runtime and operational counts to job_log.json.
    One file per match job. No database, no aggregation — flat JSON only.
    """

    STEPS = [
        "1_frame_extraction",
        "1b_boundary_detection",
        "1c_window_plan",
        "1d_team_sheet",
        "1e_attack_direction",
        "2_match_data_block",
        "3a_tier1_scan",
        "3b_confidence_aggregator",
        "3c_targeted_reruns",
        "3d_deep_scan",
        "3e_programmatic_merge",
        "3f_pass_accumulation",
        "3g_running_summary",
        "3h_ground_truth",
        "3j_report_readiness",
        "3i_confirmation",
        "4_report_writing",
        "5_word_conversion",
    ]

    def __init__(self, match_dir: str):
        self.match_dir = match_dir
        self.log_path  = os.path.join(match_dir, "job_log.json")
        self._step_start_times: dict = {}

        self.log = {
            "match_dir":             match_dir,
            "started_at":            datetime.now().isoformat(),
            "finished_at":           None,
            "total_runtime_seconds": None,
            "steps":                 {},
            "counts": {
                "frames_extracted":      0,
                "windows_planned":       0,
                "windows_complete":      0,
                "tier1_scans":           0,
                "rerun_frames":          0,
                "unresolvable_frames":   0,
                "deep_scan_windows":     0,
                "confirmation_items":    0,
                "confirmation_skipped":  0,
                "confirmation_goals":    0,
                "data_gap_windows":      0,
                "ground_truth_missed":   0,
                "suppressed_families":   0,
                "downgraded_families":   0,
            },
            "enrichment_level":   None,
            "player_id_ceiling":  None,
            "source_type":        None,
            "source_confidence":  None,
            "split_aware":        None,
            "metrics_active":     None,
            "metrics_suppressed": None,
            "metrics_avg_conf":   None,
            "reports_generated":  [],
            "blocking_issues":    [],
            "notes":              [],
        }
        self._save()

    def step_start(self, step_name: str):
        self._step_start_times[step_name] = time.time()
        print(f"  [{step_name}] started")

    def step_end(self, step_name: str, **counts):
        start = self._step_start_times.get(step_name, time.time())
        elapsed = round(time.time() - start, 1)

        self.log["steps"][step_name] = {
            "runtime_seconds": elapsed,
            **counts
        }

        # Accumulate into totals
        for k, v in counts.items():
            if k in self.log["counts"] and isinstance(v, (int, float)):
                self.log["counts"][k] += v

        self._save()
        print(f"  [{step_name}] done — {elapsed}s")

    def set_enrichment(self, enrichment_level: str, player_id_ceiling: str):
        self.log["enrichment_level"]  = enrichment_level
        self.log["player_id_ceiling"] = player_id_ceiling
        self._save()

    def set_source_profile(self, source_type: str, confidence: float,
                           split_aware: bool = False,
                           suppressed_count: int = 0, downgraded_count: int = 0):
        self.log["source_type"]       = source_type
        self.log["source_confidence"] = confidence
        self.log["split_aware"]       = split_aware
        self.log["counts"]["suppressed_families"] = suppressed_count
        self.log["counts"]["downgraded_families"] = downgraded_count
        self._save()

    def set_metrics(self, active: int, suppressed: int, avg_confidence: float):
        self.log["metrics_active"]     = active
        self.log["metrics_suppressed"] = suppressed
        self.log["metrics_avg_conf"]   = avg_confidence
        self._save()

    def add_blocking_issue(self, issue: str):
        self.log["blocking_issues"].append(issue)
        self._save()

    def note(self, msg: str):
        self.log["notes"].append(msg)
        self._save()

    def finish(self, reports_generated: list):
        self.log["finished_at"]           = datetime.now().isoformat()
        started                           = datetime.fromisoformat(self.log["started_at"])
        self.log["total_runtime_seconds"] = round(
            (datetime.now() - started).total_seconds(), 1
        )
        self.log["reports_generated"] = reports_generated
        self._save()

        mins = self.log["total_runtime_seconds"] / 60
        print(f"\n{'-'*50}")
        print(f"Job complete: {self.log['total_runtime_seconds']}s ({mins:.1f} min)")
        print(f"Reports:      {', '.join(reports_generated)}")
        print(f"Windows:      {self.log['counts']['windows_complete']} / {self.log['counts']['windows_planned']}")
        print(f"Re-runs:      {self.log['counts']['rerun_frames']} frames")
        print(f"Confirmations:{self.log['counts']['confirmation_items']} items (goals: {self.log['counts']['confirmation_goals']})")
        print(f"Enrichment:   {self.log['enrichment_level']} | ID ceiling: {self.log['player_id_ceiling']}")
        print(f"Source:       {self.log['source_type']} (conf: {self.log['source_confidence']}) | split: {self.log['split_aware']}")
        print(f"Families:     suppressed={self.log['counts']['suppressed_families']} | downgraded={self.log['counts']['downgraded_families']}")
        if self.log.get("metrics_active") is not None:
            print(f"Metrics:      {self.log['metrics_active']} active | {self.log['metrics_suppressed']} suppressed | avg conf: {self.log['metrics_avg_conf']}")
        if self.log["notes"]:
            for n in self.log["notes"]:
                print(f"  Note: {n}")
        print(f"{'-'*50}")

    def _save(self):
        os.makedirs(self.match_dir, exist_ok=True)
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(stamp_schema_version(self.log, "job_log"), f, indent=2)


def summarise_jobs(jobs_root: str):
    """
    Print a summary table across all job_log.json files found under jobs_root.
    Use this to track unit economics across multiple matches.
    """
    import glob

    logs = sorted(glob.glob(os.path.join(jobs_root, "**", "job_log.json"), recursive=True))
    if not logs:
        print("No job_log.json files found.")
        return

    rows = []
    for path in logs:
        with open(path, encoding="utf-8") as f:
            log = json.load(f)
        rows.append({
            "dir":         os.path.basename(os.path.dirname(path)),
            "runtime_min": round((log.get("total_runtime_seconds") or 0) / 60, 1),
            "windows":     log["counts"]["windows_complete"],
            "reruns":      log["counts"]["rerun_frames"],
            "confirms":    log["counts"]["confirmation_items"],
            "enrichment":  log.get("enrichment_level", "?"),
            "reports":     ", ".join(log.get("reports_generated", [])),
        })

    header = f"{'Match':<35} {'Min':>6} {'Win':>4} {'Rer':>4} {'Con':>4} {'Enrich':<15} Reports"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['dir']:<35} {r['runtime_min']:>6} {r['windows']:>4} "
            f"{r['reruns']:>4} {r['confirms']:>4} {r['enrichment']:<15} {r['reports']}"
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "summary":
        root = sys.argv[2] if len(sys.argv) > 2 else "."
        summarise_jobs(root)
    else:
        print("Usage: python job_logger.py summary [jobs_root_dir]")
