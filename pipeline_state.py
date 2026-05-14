"""
pipeline_state.py — checkpoint manager for Match Lens pipeline

Writes pipeline_state.json to the match directory after every step.
If a run is interrupted, re-running pipeline_runner.py will skip
completed steps and resume from the exact failure point.

State file structure:
  {
    "match":    "Felixstowe vs Lowestoft",
    "started":  "2026-04-23T14:00:00",
    "quality":  "standard",
    "windows": {
      "1H_00-00_05-00": {
        "3a": "complete",
        "3b": "complete",
        "3d_event": "skipped",
        "3e_merge": "complete"
      },
      ...
    },
    "steps": {
      "3c_triage":    "complete",
      "3f_sequences": "complete",
      "3g_summary":   "complete",
      "3h_ground_truth": "pending",
      ...
    },
    "batch_ids": {
      "3a_batch_id": "msgbatch_abc123",
      "3b_batch_id": null
    }
  }
"""

import json, os
from datetime import datetime

STATE_FILE = "pipeline_state.json"

WINDOW_STEPS  = ["3a", "3b", "3d_event", "3d_setpiece", "3d_recovery", "3e_merge"]
PIPELINE_STEPS = [
    "2b_jersey_ocr",
    "3c_triage", "3d_reruns", "3e_merge",
    "3f_shots", "3f_sequences", "3g_summary",
    "3h_ground_truth", "3i_escalation", "3j_readiness",
    "3k_metrics", "3l_synthesis",
    "4a_tactical_report", "4b_opposition_report",
    "4c_flagged_moments", "4d_pass_network",
]


def load_state(match_dir: str) -> dict:
    path = os.path.join(match_dir, STATE_FILE)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None


def init_state(match_dir: str, match_name: str, windows: list,
               quality: str = "standard") -> dict:
    """Create a fresh state file. Call once before starting the pipeline."""
    path = os.path.join(match_dir, STATE_FILE)
    if os.path.exists(path):
        existing = load_state(match_dir)
        # Don't overwrite a state that has completed windows
        done = sum(1 for w in existing.get("windows", {}).values()
                   if w.get("3a") == "complete")
        if done > 0:
            print(f"  [STATE] Existing state found: {done} windows complete.")
            print(f"  [STATE] Use resume_state() to continue from checkpoint.")
            return existing

    state = {
        "match":     match_name,
        "started":   datetime.now().isoformat(),
        "quality":   quality,
        "windows":   {w: {s: "pending" for s in WINDOW_STEPS} for w in windows},
        "steps":     {s: "pending" for s in PIPELINE_STEPS},
        "batch_ids": {},
        "errors":    [],
        "last_updated": datetime.now().isoformat(),
    }
    _save(match_dir, state)
    print(f"  [STATE] Initialised: {len(windows)} windows, quality={quality}")
    return state


def mark_window(match_dir: str, state: dict, window_id: str,
                step: str, status: str, error: str = None) -> dict:
    """Mark a window step as complete/failed/skipped."""
    if window_id not in state["windows"]:
        state["windows"][window_id] = {s: "pending" for s in WINDOW_STEPS}
    state["windows"][window_id][step] = status
    state["last_updated"] = datetime.now().isoformat()
    if error:
        state["errors"].append({
            "window": window_id, "step": step,
            "error": error, "time": datetime.now().isoformat()
        })
    _save(match_dir, state)
    return state


def mark_step(match_dir: str, state: dict, step: str,
              status: str, error: str = None) -> dict:
    """Mark a pipeline step as complete/failed/skipped."""
    state["steps"][step] = status
    state["last_updated"] = datetime.now().isoformat()
    if error:
        state["errors"].append({
            "step": step, "error": error,
            "time": datetime.now().isoformat()
        })
    _save(match_dir, state)
    return state


def is_window_done(state: dict, window_id: str, step: str) -> bool:
    return state.get("windows", {}).get(window_id, {}).get(step) == "complete"


def is_step_done(state: dict, step: str) -> bool:
    return state.get("steps", {}).get(step) == "complete"


def pending_windows(state: dict, step: str, include_failed: bool = False) -> list:
    """Return window IDs that still need a given step."""
    done = {"complete", "skipped"}
    if not include_failed:
        done.add("failed")
    return [wid for wid, steps in state.get("windows", {}).items()
            if steps.get(step) not in done]


def failed_windows(state: dict, step: str) -> list:
    """Return window IDs where a given step has failed."""
    return [wid for wid, steps in state.get("windows", {}).items()
            if steps.get(step) == "failed"]


def print_progress(state: dict):
    """Print a summary of pipeline progress."""
    windows  = state.get("windows", {})
    total    = len(windows)
    print(f"\n  Pipeline: {state.get('match','?')} ({state.get('quality','?')})")
    print(f"  Started:  {state.get('started','?')[:19]}")
    print()
    for step in WINDOW_STEPS:
        done    = sum(1 for w in windows.values() if w.get(step) == "complete")
        skipped = sum(1 for w in windows.values() if w.get(step) == "skipped")
        failed  = sum(1 for w in windows.values() if w.get(step) == "failed")
        pending = total - done - skipped - failed
        bar     = "#" * done + "." * pending
        print(f"  {step:<14} {bar} {done}/{total} "
              + (f"[{skipped} skipped] " if skipped else "")
              + (f"[{failed} FAILED] " if failed else ""))
    print()
    for step in PIPELINE_STEPS:
        status = state.get("steps", {}).get(step, "pending")
        icon   = "[OK]" if status == "complete" else "[FAIL]" if status == "failed" else "-"
        print(f"  {icon} {step}")
    if state.get("errors"):
        print(f"\n  Errors: {len(state['errors'])}")
        for e in state["errors"][-3:]:
            print(f"    {e.get('window','?')}/{e.get('step','?')}: {e.get('error','?')[:60]}")


def store_batch_id(match_dir: str, state: dict, key: str, batch_id: str) -> dict:
    state["batch_ids"][key] = batch_id
    state["last_updated"] = datetime.now().isoformat()
    _save(match_dir, state)
    return state


def _save(match_dir: str, state: dict):
    path = os.path.join(match_dir, STATE_FILE)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pipeline_state.py [MATCH_DIR]")
        sys.exit(1)
    state = load_state(sys.argv[1])
    if state:
        print_progress(state)
    else:
        print("No pipeline_state.json found in this directory.")
