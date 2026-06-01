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

    Naming variants this function must handle:

      1. agent_{id}_{suffix}.json
         e.g. agent_01_structural.json
         Canonical shape if batch_runner is given a bare numeric id.

      2. agent_agent_{id}_{suffix}.json
         e.g. agent_agent_01_structural.json
         **What batch_runner.collect_results actually produces today.** It
         builds ``agent_{window_id}_{suffix}.json`` where ``window_id``
         already starts with ``agent_`` (e.g. "agent_01"), giving the
         doubled prefix. Pre-fix readers that only knew shape #1 silently
         missed these files, causing 3e_merge to find 0/N inputs on
         fresh-only agent_logs.

      3. agent_{id}_{label}_{suffix}.json
         e.g. agent_01_W01_1H_00-05min_structural.json
         Canonical-with-label, if a future writer interleaves a label.

      4. agent_agent_{id}_{label}_{suffix}.json
         e.g. agent_agent_07_W07_1H_30-35min_setpiece.json
         Doubled prefix with label — the form May-13 archive runs used
         for setpiece + merged outputs.

    Returns the path or None.
    """
    wid_str = str(window_id).lstrip("agent_")
    wid_padded = wid_str.zfill(2) if wid_str.isdigit() else wid_str

    patterns = [
        # 1. canonical, no label
        f"agent_{wid_str}_{suffix}.json",
        f"agent_{wid_padded}_{suffix}.json",
        # 2. doubled prefix, no label (current batch_runner output)
        f"agent_agent_{wid_str}_{suffix}.json",
        f"agent_agent_{wid_padded}_{suffix}.json",
        # 3. canonical with label
        f"agent_{wid_str}_*_{suffix}.json",
        f"agent_{wid_padded}_*_{suffix}.json",
        # 4. doubled prefix with label (May-13 archive shape)
        f"agent_agent_{wid_str}_*_{suffix}.json",
        f"agent_agent_{wid_padded}_*_{suffix}.json",
        # 5. very loose fallback (any prefix, any middle)
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

    Like find_agent_output above, this also handles the doubled-prefix
    form (agent_agent_{id}_{label}_merged.json) used by the May-13
    archive. The very-loose fallback pattern (*_{id}_*_merged.json) catches
    that case incidentally but the explicit pattern makes the intent clear.

    Returns the path or None.
    """
    wid_str = str(window_id).lstrip("agent_")
    wid_padded = wid_str.zfill(2) if wid_str.isdigit() else wid_str

    patterns = [
        f"agent_{wid_str}_*_merged.json",
        f"agent_{wid_padded}_*_merged.json",
        f"agent_agent_{wid_str}_*_merged.json",
        f"agent_agent_{wid_padded}_*_merged.json",
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
