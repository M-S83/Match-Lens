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
import re


# Match a window label embedded in merged-file basenames.
# Recognized shapes:
#   agent_NN_<label>_merged.json
#   agent_agent_NN_<label>_merged.json
#   agent_NN_W##_<label>_merged.json          (W## index prefix)
#   agent_agent_NN_W##_<label>_merged.json
# Where <label> matches a half + minute range produced by Step 1c:
#   "1H_00-05min" / "2H_85-90min" / "ET1_00-05min" / etc.
_WINDOW_LABEL_RE = re.compile(
    r'(?:^|_)(?:W\d+_)?'                          # optional W## prefix
    r'((?:\d+H|ET\d)_\d+-\d+min)'                 # capture the label
    r'_merged\.json$'
)


def derive_window_label(merged_path: str) -> str:
    """Recover the canonical window label (e.g. '1H_00-05min') from a
    merged-file path.

    Bug A context: the structural agent does not emit a top-level
    `window` field, so merge_utils historically wrote `"window": null`
    into the merged file. Any reader that trusted that field saw None.
    Every merged file's BASENAME does carry the window label (it's
    embedded by the runner when the file is written), so the canonical
    way to recover it is to parse the basename — not to read the merged
    file's top-level field.

    Returns the label on match, or the basename stripped of the
    `agent_(agent_)?ID_` prefix and `_merged.json` suffix as a last
    resort. Never returns None — readers that expect a non-null string
    can rely on this contract."""
    bn = os.path.basename(merged_path)
    m = _WINDOW_LABEL_RE.search(bn)
    if m:
        return m.group(1)
    # Fallback: strip the prefix and suffix; return whatever remains.
    bn = re.sub(r'^agent_(?:agent_)?[^_]+_', '', bn)
    bn = re.sub(r'_merged\.json$', '', bn)
    return bn or "unknown"


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

    # v3.0.1 bundle (Task 140): defensive label-format fallback. Older /
    # in-flight queues may carry the label form (e.g. "1H_00-00_05-00",
    # "2H_45-00_50-00") as their window_id instead of the canonical agent_id
    # ("01", "12"). The patterns above won't match because the label is
    # embedded mid-filename, not in the leading agent_NN position. Recognize
    # the half-prefixed label pattern and add the matching glob.
    if isinstance(window_id, str) and (
        window_id.startswith("1H_")
        or window_id.startswith("2H_")
        or window_id.startswith("ET1_")
        or window_id.startswith("ET2_")
    ):
        label_pattern = f"agent_*_{window_id}_merged.json"
        matches = glob.glob(os.path.join(logs_dir, label_pattern))
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
