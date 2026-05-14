"""Canonical file path lookups for pipeline output files.

Writers use complex naming patterns: agent_{id}_{label}_{suffix}.json
where {label} is a safe-format time range (e.g. "20m03s-25m03s"). The
label cannot be reconstructed at read time, so readers MUST NOT build
exact paths — call into this module instead.

Single source of truth for finding agent files. Adding a new file type
or naming variant is a one-line change here.
"""

import glob
import os


def find_agent_output(logs_dir: str, window_id, suffix: str):
    """Find an agent output file for a given window and suffix.

    `suffix` is the type marker without extension: 'structural', 'player',
    'event', 'setpiece', or 'recovery'. For merged files use
    find_merged_window instead.

    batch_runner.py writes these as agent_{id}_{label}_{suffix}.json.
    Returns the path or None.
    """
    wid_str = str(window_id).lstrip("agent_")
    wid_padded = wid_str.zfill(2) if wid_str.isdigit() else wid_str

    patterns = [
        f"agent_{wid_str}_{suffix}.json",
        f"agent_{wid_padded}_{suffix}.json",
        f"agent_{wid_str}_*_{suffix}.json",
        f"agent_{wid_padded}_*_{suffix}.json",
        f"*_{wid_str}_*_{suffix}.json",
        f"*_{wid_padded}_*_{suffix}.json",
    ]
    for pattern in patterns:
        matches = [m for m in glob.glob(os.path.join(logs_dir, pattern))
                   if "_merged" not in os.path.basename(m)
                   and "_rerun" not in os.path.basename(m)
                   and "agentB" not in os.path.basename(m)]
        if matches:
            matches.sort(key=lambda p: len(os.path.basename(p)))
            return matches[0]
    return None


def find_merged_window(logs_dir: str, window_id):
    """Find the merged file for a given window.

    merge_utils.py writes these as agent_{id}_{label}_merged.json. The
    label-free form agent_{id}_merged.json is NOT produced by the live
    pipeline — do not construct that path.
    Returns the path or None.
    """
    wid_str = str(window_id).lstrip("agent_")
    wid_padded = wid_str.zfill(2) if wid_str.isdigit() else wid_str

    patterns = [
        f"agent_{wid_str}_*_merged.json",
        f"agent_{wid_padded}_*_merged.json",
        f"*_{wid_str}_*_merged.json",
        f"*_{wid_padded}_*_merged.json",
    ]
    for pattern in patterns:
        matches = glob.glob(os.path.join(logs_dir, pattern))
        if matches:
            matches.sort(key=lambda p: len(os.path.basename(p)))
            return matches[0]
    return None


def find_all_merged_windows(logs_dir: str, extra_dirs=None):
    """Find every merged window file produced by the pipeline.

    `extra_dirs` is for legacy locations (e.g. v1's `merged_windows/`)
    that earlier pipeline versions wrote to. Pass an iterable of
    absolute paths to scan; non-existent dirs are skipped silently.
    Returns a deduplicated, sorted list.
    """
    paths = sorted(glob.glob(os.path.join(logs_dir, "*_merged.json")))
    if extra_dirs:
        for d in extra_dirs:
            if os.path.isdir(d):
                paths.extend(sorted(glob.glob(os.path.join(d, "merged_*.json"))))
    seen = set()
    return [p for p in paths if not (p in seen or seen.add(p))]
