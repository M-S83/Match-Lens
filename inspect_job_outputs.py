"""
inspect_job_outputs.py -- Match Lens output reconnaissance

Read-only diagnostic for a finished match folder. Answers three persistence
questions that gate Phase 1 of the metrics harness:

  Q1. Does the OPEN-PLAY burst path persist its adjudication
      (status / rejection_reason), or only the SET-PIECE burst path?
  Q2. What does null-emission (no-observation marker) look like on disk?
  Q3. Are inconclusive rejection_reason strings varied or canned?

Discovery-based, not assumption-based: walks agent_logs/, buckets files by
filename heuristic, REPORTS what it found so bucketing errors are visible
BEFORE conclusions.

Usage:
  python inspect_job_outputs.py "<match_dir>" [--full-keys]
"""

import json, os, sys, glob, argparse
from collections import defaultdict, Counter

# -- Heuristics (edit at top of file if filename conventions differ) ----------

SET_PIECE_TOKENS  = ("setpiece", "set_piece")
BURST_TOKENS      = ("burst", "confirmation", "5fps", "5_fps")
PLAYER_TOKENS     = ("player",)
STRUCTURAL_TOKENS = ("structural",)
MERGED_TOKENS     = ("merged",)
EVENT_TOKENS      = ("event",)

NULL_MARKER_KEYS = {
    "is_null", "null_emission", "no_observation", "no_observations",
    "empty", "skip", "skipped", "emit", "is_silent", "silent",
    "no_event", "event_present", "has_observations", "observed",
}

OBSERVATION_LIST_KEYS = {
    "individual_observations", "observations", "events", "set_pieces",
    "duels", "key_moments", "items", "findings",
    "player_escalation_queue", "watch_list_confirmations",
    "top_observations_for_next_window",
}

BUCKET_ORDER = (
    "setpiece", "burst_dedicated", "player_1fps", "structural",
    "merged", "event_1fps", "other",
)

# Forensic backups / test fixtures -- excluded from production-file counts so
# they can't trigger a false "status IS persisted" verdict on bucket Q1.
FORENSIC_TOKENS = (
    ".test.", ".postStep", ".preStep", ".snapshot.",
    ".pre_zone_normalise", "_baseline.", ".postBugB", ".pre_",
)


def is_forensic(name: str) -> bool:
    return any(t in name for t in FORENSIC_TOKENS)


# -- Bucketing ----------------------------------------------------------------

def classify_file(name: str) -> str:
    n = name.lower()
    is_setpiece = any(t in n for t in SET_PIECE_TOKENS)
    is_burst    = any(t in n for t in BURST_TOKENS)
    if is_setpiece:
        return "setpiece"
    if is_burst:
        return "burst_dedicated"
    if any(t in n for t in MERGED_TOKENS):
        return "merged"
    if any(t in n for t in STRUCTURAL_TOKENS):
        return "structural"
    if any(t in n for t in PLAYER_TOKENS):
        return "player_1fps"
    if any(t in n for t in EVENT_TOKENS):
        return "event_1fps"
    return "other"


# -- JSON walking -------------------------------------------------------------

def walk(obj, path=""):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk(v, f"{path}.{k}" if path else k)
    elif isinstance(obj, list):
        yield path, obj
        for i, v in enumerate(obj):
            yield from walk(v, f"{path}[{i}]")


def find_field(obj, field_name):
    """Return list of (path, value) for every dict that contains field_name."""
    out = []
    for path, val in walk(obj):
        if isinstance(val, dict) and field_name in val:
            out.append((f"{path}.{field_name}" if path else field_name,
                        val[field_name]))
    return out


# -- Loader -------------------------------------------------------------------

def load_all(buckets):
    """Parse every file once. Return {filepath: parsed_obj} (None on error)."""
    parsed = {}
    for files in buckets.values():
        for fp in files:
            try:
                with open(fp, encoding="utf-8") as f:
                    parsed[fp] = json.load(f)
            except (OSError, json.JSONDecodeError):
                parsed[fp] = None
    return parsed


# -- Q1: status / rejection_reason persistence --------------------------------

def q1_status_persistence(buckets, parsed):
    print("\n" + "=" * 78)
    print("Q1 -- status / rejection_reason persistence per bucket")
    print("=" * 78)

    bucket_stats = {}
    for bucket in BUCKET_ORDER:
        files = buckets.get(bucket, [])
        if not files:
            continue
        n_status, n_rej = 0, 0
        sample_status_paths, sample_rej_paths = [], []
        for fp in files:
            data = parsed.get(fp)
            if data is None:
                continue
            shits = find_field(data, "status")
            rhits = find_field(data, "rejection_reason")
            if shits:
                n_status += 1
                if len(sample_status_paths) < 3:
                    sample_status_paths.append(shits[0][0])
            if rhits:
                n_rej += 1
                if len(sample_rej_paths) < 3:
                    sample_rej_paths.append(rhits[0][0])
        bucket_stats[bucket] = (n_status, n_rej, len(files))
        print(f"\n  [{bucket}]  {len(files)} files")
        print(f"    with `status`:           {n_status}/{len(files)}")
        print(f"    with `rejection_reason`: {n_rej}/{len(files)}")
        if sample_status_paths:
            print(f"    status field paths:      {sample_status_paths}")
        if sample_rej_paths:
            print(f"    rejection field paths:   {sample_rej_paths}")

    print("\n  READING")
    sp = bucket_stats.get("setpiece",        (0, 0, 0))
    pl = bucket_stats.get("player_1fps",     (0, 0, 0))
    bd = bucket_stats.get("burst_dedicated", (0, 0, 0))
    mg = bucket_stats.get("merged",          (0, 0, 0))

    open_play_paths = []
    if pl[0] > 0: open_play_paths.append(f"player_1fps ({pl[0]}/{pl[2]})")
    if bd[0] > 0: open_play_paths.append(f"burst_dedicated ({bd[0]}/{bd[2]})")
    if mg[0] > 0: open_play_paths.append(f"merged ({mg[0]}/{mg[2]})")

    if sp[0] > 0 and open_play_paths:
        print(f"    status IS persisted on open-play path(s): "
              f"{', '.join(open_play_paths)}.")
        print("    Both burst counters are pure-read; Phase 1 touches nothing.")
    elif sp[0] > 0 and not open_play_paths:
        print("    PERSISTENCE GAP -- set-piece persists status; open-play does not.")
        print("    Phase 1 burst counter would silently only cover set-pieces.")
        print("    Close the gap (persist open-play adjudication) before harness lands.")
    elif sp[0] == 0 and open_play_paths:
        print(f"    open-play persists but set-piece does not: "
              f"{', '.join(open_play_paths)}.")
        print("    Asymmetric -- inverse of the expected gap. Worth checking why.")
    else:
        print("    NO `status` found in either bucket. Either the field is named")
        print("    differently here, or no inconclusive items exist in this match.")
        print("    Re-run with --full-keys to see real top-level keys per bucket.")


# -- Q2: null-emission marker -------------------------------------------------

def q2_null_marker(buckets, parsed, full_keys):
    print("\n" + "=" * 78)
    print("Q2 -- null-emission marker on observation windows")
    print("=" * 78)

    # 2a -- explicit named markers
    print("\n  2a -- explicit null-marker keys (matched by name)")
    key_hits = Counter()
    key_examples = defaultdict(list)
    for bucket, files in buckets.items():
        for fp in files:
            data = parsed.get(fp)
            if data is None:
                continue
            for _, val in walk(data):
                if isinstance(val, dict):
                    for k in val:
                        if k in NULL_MARKER_KEYS:
                            key_hits[k] += 1
                            if len(key_examples[k]) < 2:
                                key_examples[k].append((bucket, os.path.basename(fp)))
    if key_hits:
        for k, c in key_hits.most_common():
            print(f"    `{k}`: {c} occurrences")
            for bucket, name in key_examples[k]:
                print(f"        e.g. [{bucket}] {name}")
    else:
        print("    no explicit null-marker keys matched the candidate list.")

    # 2b -- boolean flags whose key name suggests null/empty/silent
    print("\n  2b -- boolean flags suggesting null/empty/silent")
    flag_hits = Counter()
    flag_examples = defaultdict(list)
    for bucket, files in buckets.items():
        for fp in files:
            data = parsed.get(fp)
            if data is None:
                continue
            for _, val in walk(data):
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, bool):
                            kl = k.lower()
                            if any(suf in kl for suf in (
                                "null", "empty", "silent", "skip",
                                "no_obs", "no_event", "has_obs", "observed",
                            )):
                                flag_hits[k] += 1
                                if len(flag_examples[k]) < 1:
                                    flag_examples[k].append((bucket, os.path.basename(fp)))
    if flag_hits:
        for k, c in flag_hits.most_common(10):
            print(f"    `{k}`: {c} occurrences")
            for bucket, name in flag_examples[k]:
                print(f"        e.g. [{bucket}] {name}")
    else:
        print("    no boolean-flag null-marker patterns found.")

    # 2c -- empty-observation route (list present, length 0)
    print("\n  2c -- empty observation lists (list present, length 0)")
    obs_total = Counter()
    obs_empty = Counter()
    for bucket, files in buckets.items():
        for fp in files:
            data = parsed.get(fp)
            if data is None:
                continue
            for _, val in walk(data):
                if isinstance(val, dict):
                    for k, v in val.items():
                        if k in OBSERVATION_LIST_KEYS and isinstance(v, list):
                            obs_total[k] += 1
                            if not v:
                                obs_empty[k] += 1
    if obs_total:
        print("    observation-list keys (key: total occurrences / of which empty):")
        for k in obs_total:
            print(f"      `{k}`: {obs_total[k]} / {obs_empty[k]}")
    else:
        print("    no known observation-list keys found.")

    # 2d -- --full-keys fallback
    if full_keys:
        print("\n  2d -- top-level keys per bucket (--full-keys enabled)")
        for bucket in BUCKET_ORDER:
            files = buckets.get(bucket, [])
            if not files:
                continue
            tl = Counter()
            for fp in files[:8]:
                data = parsed.get(fp)
                if isinstance(data, dict):
                    for k in data:
                        tl[k] += 1
            print(f"    [{bucket}]  top-level keys: {dict(tl.most_common(20))}")


# -- Q3: rejection_reason variety --------------------------------------------

def q3_rejection_variety(buckets, parsed):
    print("\n" + "=" * 78)
    print("Q3 -- rejection_reason variety (canned boilerplate vs. real reasoning)")
    print("=" * 78)

    rows = []
    for bucket, files in buckets.items():
        for fp in files:
            data = parsed.get(fp)
            if data is None:
                continue
            for _, val in find_field(data, "rejection_reason"):
                if isinstance(val, str) and val.strip():
                    rows.append((bucket, os.path.basename(fp), val.strip()))

    if not rows:
        print("\n    no `rejection_reason` strings found on disk.")
        return

    total = len(rows)
    counter = Counter(r for _, _, r in rows)
    unique = len(counter)
    ratio = unique / total

    print(f"\n    total rejection_reason strings: {total}")
    print(f"    unique strings:                  {unique}")
    print(f"    variety ratio:                   {ratio:.2f}  "
          f"(1.0 = every reason distinct; near 0 = canned)")

    if counter.most_common(1)[0][1] > 1:
        print(f"\n    MOST-REPEATED strings (boilerplate risk):")
        for s, c in counter.most_common(3):
            if c < 2:
                break
            preview = s[:160] + ("..." if len(s) > 160 else "")
            print(f"      {c}x  {preview!r}")

    print(f"\n    SAMPLE -- up to 5 distinct strings:")
    seen = set()
    for bucket, name, r in rows:
        if r in seen:
            continue
        seen.add(r)
        preview = r[:260] + ("..." if len(r) > 260 else "")
        print(f"      [{bucket}] {name}")
        print(f"        {preview!r}")
        if len(seen) >= 5:
            break


# -- Main ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("match_dir")
    ap.add_argument("--full-keys", action="store_true",
                    help="Dump top-level keys per bucket if Q2 doesn't match.")
    args = ap.parse_args()

    md = args.match_dir
    if not os.path.isdir(md):
        print(f"ERROR: match_dir not found: {md}", file=sys.stderr)
        sys.exit(1)
    agent_dir = os.path.join(md, "agent_logs")
    if not os.path.isdir(agent_dir):
        print(f"ERROR: agent_logs/ subfolder not found in {md}", file=sys.stderr)
        sys.exit(1)

    all_json = sorted(glob.glob(os.path.join(agent_dir, "*.json")))
    prod_files     = [fp for fp in all_json if not is_forensic(os.path.basename(fp))]
    forensic_files = [fp for fp in all_json if     is_forensic(os.path.basename(fp))]

    buckets = defaultdict(list)
    for fp in prod_files:
        buckets[classify_file(os.path.basename(fp))].append(fp)

    print("=" * 78)
    print(f"inspect_job_outputs.py -- {os.path.basename(md)}")
    print("=" * 78)
    print(f"\nFILE INVENTORY  (under agent_logs/, total {len(all_json)} files)")
    print(f"  production:        {len(prod_files)}")
    if forensic_files:
        print(f"  forensic backups:  {len(forensic_files)}  "
              f"(excluded from Q1/Q2/Q3 -- e.g. {os.path.basename(forensic_files[0])})")
    for b in BUCKET_ORDER:
        if buckets.get(b):
            print(f"  {b:18s}  {len(buckets[b]):4d} files")

    print(f"\nFILENAME PATTERNS  (sample 3 per bucket -- sanity-check bucketing)")
    for b in BUCKET_ORDER:
        files = buckets.get(b, [])
        if files:
            print(f"  [{b}]")
            for fp in files[:3]:
                print(f"    {os.path.basename(fp)}")

    parsed = load_all(buckets)
    n_load_errors = sum(1 for v in parsed.values() if v is None)
    if n_load_errors:
        print(f"\n  WARN: {n_load_errors} files failed to parse "
              f"(skipped in answers below)")

    q1_status_persistence(buckets, parsed)
    q2_null_marker(buckets, parsed, args.full_keys)
    q3_rejection_variety(buckets, parsed)

    print("\n" + "=" * 78)
    print("Done. If FILE INVENTORY shows a bucket misclassified, edit the")
    print("SET_PIECE_TOKENS / BURST_TOKENS / etc. constants at top of script.")
    print("=" * 78)


if __name__ == "__main__":
    main()
