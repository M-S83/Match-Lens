"""check_accumulator_preservation.py

Pre-merge diligence for V3 porting Step 5
(the accumulator.update_running_summary merge).

Asserts that the 11 "hard requirement" preservation items from
V3_PORTING_PLAN.md Section 8 Step 5 are all present in a given
match's running_summary.json. Intended as both:

  - A BASELINE check (run before Step 5 to confirm we know what the
    pre-merge state is)
  - A REGRESSION check (run after Step 5 to assert no preservation
    item was dropped)

Usage:
    python scripts/diagnostics/check_accumulator_preservation.py <match_dir>

Exit code: 0 if all 11 items confirmed present (or PRESENT-empty,
which is structurally valid); 1 if any item is ABSENT or MALFORMED.

The script does NOT try to verify behaviour against agent inputs --
it only checks structural presence. A list that is empty because no
agent emitted anything for it is treated as PRESENT (the accumulator
correctly initialised it); a list that is missing entirely is
ABSENT (the accumulator failed to initialise it). A field with the
wrong shape (e.g. dict where a list is expected) is MALFORMED.

The 11 items checked are:
   1. schema_version field (F12 stamp)
   2. set_pieces_rejected list (F11)
   3. transitions list
   4. gk_kicks list
   5. match_state_by_window list, entries carry match_state
   6. formation_history list, entries carry home_formation / away_formation / home_shape / away_shape (F5)
   7. line_height_by_window list, entries carry both pct and metres
   8. pressing_by_window list, triggers in normalised vocabulary
   9. possession_by_window list, entries carry focus_pct and focus_seqs
  10. confirmation_queue list, deduplicated by timestamp
  11. individual_observations list, entries carry obs_grade where applicable
"""

import json
import os
import sys


# Canonical pressing trigger vocabulary -- from
# pipeline_runner_v2.py:build_structural_prompt "=== PRESSING TRIGGER
# VOCABULARY ===" block. _normalise_trigger collapses agent free-text
# into this set; if an entry uses a value outside this set it's a
# pre-merge regression.
CANONICAL_TRIGGERS = frozenset({
    "back_pass",
    "gk_in_possession",
    "cb_receiving_turned",
    "cb_receiving_open",
    "fb_receiving",
    "free_kick_restart",
    "throw_in_restart",
    "no_trigger_observed",
    None,
})


def _ok(label, example=None):
    return ("PRESENT", label, example)


def _ok_empty(label):
    return ("PRESENT-empty", label, "(empty list -- accumulator initialised it; no agent input fed into it)")


def _absent(label, reason):
    return ("ABSENT", label, reason)


def _malformed(label, reason, example=None):
    return ("MALFORMED", label, f"{reason}: {example!r}" if example is not None else reason)


def check_schema_version(s):
    """1. schema_version field (F12 stamp)"""
    v = s.get("schema_version")
    if v is None:
        return _absent("schema_version", "key not in running_summary.json")
    if not isinstance(v, str):
        return _malformed("schema_version", "expected str", v)
    return _ok("schema_version", f"'{v}'")


def check_set_pieces_rejected(s):
    """2. set_pieces_rejected list (F11)"""
    v = s.get("set_pieces_rejected", "__MISSING__")
    if v == "__MISSING__":
        return _absent("set_pieces_rejected", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("set_pieces_rejected", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("set_pieces_rejected")
    return _ok("set_pieces_rejected", v[0])


def check_transitions(s):
    """3. transitions list"""
    v = s.get("transitions", "__MISSING__")
    if v == "__MISSING__":
        return _absent("transitions", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("transitions", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("transitions")
    return _ok("transitions", v[0])


def check_gk_kicks(s):
    """4. gk_kicks list"""
    v = s.get("gk_kicks", "__MISSING__")
    if v == "__MISSING__":
        return _absent("gk_kicks", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("gk_kicks", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("gk_kicks")
    return _ok("gk_kicks", v[0])


def check_match_state_by_window(s):
    """5. match_state_by_window list, entries carry match_state"""
    v = s.get("match_state_by_window", "__MISSING__")
    if v == "__MISSING__":
        return _absent("match_state_by_window", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("match_state_by_window", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("match_state_by_window")
    sample = v[0]
    if not isinstance(sample, dict) or "match_state" not in sample:
        return _malformed("match_state_by_window",
                          "first entry missing match_state field", sample)
    return _ok("match_state_by_window",
               f"{len(v)} entries; first: window={sample.get('window')} "
               f"match_state={sample.get('match_state')}")


def check_formation_history(s):
    """6. formation_history list, entries carry home/away formation and shape (F5)"""
    v = s.get("formation_history", "__MISSING__")
    if v == "__MISSING__":
        return _absent("formation_history", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("formation_history", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("formation_history")
    sample = v[0]
    required = ("home_formation", "away_formation", "home_shape", "away_shape")
    missing = [k for k in required if k not in sample]
    if missing:
        return _malformed("formation_history",
                          f"first entry missing F5 fields: {missing}", sample)
    return _ok("formation_history",
               f"{len(v)} entries; first: home_formation={sample.get('home_formation')} "
               f"away_formation={sample.get('away_formation')}")


def check_line_height_by_window(s):
    """7. line_height_by_window list, entries carry both pct and metres (F4 / _pct_to_m)"""
    v = s.get("line_height_by_window", "__MISSING__")
    if v == "__MISSING__":
        return _absent("line_height_by_window", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("line_height_by_window", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("line_height_by_window")
    sample = v[0]
    has_pct = any(k.endswith("_pct") for k in sample.keys())
    has_m   = any(k.endswith("_m") or k.endswith("_m_approx") for k in sample.keys())
    if not (has_pct and has_m):
        return _malformed("line_height_by_window",
                          f"first entry needs both pct and metres fields; "
                          f"has_pct={has_pct}, has_m={has_m}", list(sample.keys()))
    return _ok("line_height_by_window",
               f"{len(v)} entries; first: avg_pct={sample.get('avg_pct')} "
               f"avg_m={sample.get('avg_m')}")


def check_pressing_by_window(s):
    """8. pressing_by_window list, triggers in normalised vocabulary"""
    v = s.get("pressing_by_window", "__MISSING__")
    if v == "__MISSING__":
        return _absent("pressing_by_window", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("pressing_by_window", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("pressing_by_window")
    # Scan observations across all windows for any non-canonical trigger
    bad_triggers = []
    for entry in v:
        for obs in entry.get("observations", []) if isinstance(entry, dict) else []:
            t = obs.get("trigger") if isinstance(obs, dict) else None
            if t is not None and t not in CANONICAL_TRIGGERS:
                bad_triggers.append(t)
    if bad_triggers:
        return _malformed("pressing_by_window",
                          f"non-canonical triggers found "
                          f"(_normalise_trigger should have normalised them)",
                          set(bad_triggers))
    sample = v[0]
    return _ok("pressing_by_window",
               f"{len(v)} entries; first: avg_score={sample.get('avg_score')} "
               f"obs_count={len(sample.get('observations', []))}")


def check_possession_by_window(s):
    """9. possession_by_window list, entries carry focus_pct and focus_seqs"""
    v = s.get("possession_by_window", "__MISSING__")
    if v == "__MISSING__":
        return _absent("possession_by_window", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("possession_by_window", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("possession_by_window")
    sample = v[0]
    required = ("focus_pct", "focus_seqs")
    missing = [k for k in required if k not in sample]
    if missing:
        return _malformed("possession_by_window",
                          f"first entry missing required fields: {missing}",
                          list(sample.keys()))
    return _ok("possession_by_window",
               f"{len(v)} entries; first: focus_pct={sample.get('focus_pct')} "
               f"focus_seqs={sample.get('focus_seqs')}")


def check_confirmation_queue(s):
    """10. confirmation_queue list, deduplicated by timestamp"""
    v = s.get("confirmation_queue", "__MISSING__")
    if v == "__MISSING__":
        return _absent("confirmation_queue", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("confirmation_queue", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("confirmation_queue")
    # Dedup-by-timestamp check: count unique timestamps vs total entries
    timestamps = [e.get("timestamp") for e in v if isinstance(e, dict)]
    if len(timestamps) != len(set(timestamps)):
        dup_count = len(timestamps) - len(set(timestamps))
        return _malformed("confirmation_queue",
                          f"timestamp dedup violated -- {dup_count} duplicate "
                          f"timestamps in {len(v)} entries (dedup expected)",
                          timestamps)
    return _ok("confirmation_queue",
               f"{len(v)} entries, all unique by timestamp")


def check_individual_observations(s):
    """11. individual_observations list, entries carry obs_grade where applicable"""
    v = s.get("individual_observations", "__MISSING__")
    if v == "__MISSING__":
        return _absent("individual_observations", "key not in running_summary.json")
    if not isinstance(v, list):
        return _malformed("individual_observations", "expected list", type(v).__name__)
    if not v:
        return _ok_empty("individual_observations")
    # Sample a few entries -- obs_grade is added "where applicable" by
    # deep_skill_metrics.observation_grade, so not every entry needs it.
    # We require at least one entry has obs_grade as evidence the
    # grading path ran. If literally none have it, the grading wiring
    # may have regressed.
    graded = [e for e in v if isinstance(e, dict) and "obs_grade" in e]
    if not graded:
        return _malformed("individual_observations",
                          f"no entry carries obs_grade across {len(v)} entries "
                          f"-- observation_grade wiring may have regressed",
                          v[0])
    return _ok("individual_observations",
               f"{len(v)} entries, {len(graded)} carry obs_grade; first graded: "
               f"obs_grade={graded[0].get('obs_grade')}")


CHECKS = [
    ("F12 schema_version",                check_schema_version),
    ("F11 set_pieces_rejected",           check_set_pieces_rejected),
    ("transitions",                       check_transitions),
    ("gk_kicks",                          check_gk_kicks),
    ("match_state_by_window",             check_match_state_by_window),
    ("F5 formation_history",              check_formation_history),
    ("F4 line_height_by_window pct+m",    check_line_height_by_window),
    ("pressing_by_window canonical trig", check_pressing_by_window),
    ("possession_by_window focus_pct",    check_possession_by_window),
    ("confirmation_queue dedup",          check_confirmation_queue),
    ("individual_observations obs_grade", check_individual_observations),
]


def main(match_dir):
    summary_path = os.path.join(match_dir, "running_summary.json")
    if not os.path.exists(summary_path):
        print(f"  [FAIL] running_summary.json not found at {summary_path}",
              file=sys.stderr)
        sys.exit(2)

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)

    print(f"\n{'=' * 75}")
    print(f"  Accumulator preservation check -- {os.path.basename(match_dir)}")
    print(f"{'=' * 75}\n")

    counts = {"PRESENT": 0, "PRESENT-empty": 0, "ABSENT": 0, "MALFORMED": 0}
    for label, fn in CHECKS:
        status, name, detail = fn(summary)
        counts[status] += 1
        marker = {"PRESENT": "[OK]", "PRESENT-empty": "[OK]",
                  "ABSENT": "[X] ", "MALFORMED": "[!] "}[status]
        print(f"  {marker} {label:<40} {status}")
        print(f"        {detail}")
        print()

    confirmed = counts["PRESENT"] + counts["PRESENT-empty"]
    total = len(CHECKS)
    print(f"{'=' * 75}")
    print(f"  Summary: {confirmed}/{total} preservation items confirmed present")
    print(f"  ({counts['PRESENT']} populated, {counts['PRESENT-empty']} empty-but-initialised)")
    if counts["ABSENT"]:
        print(f"  ABSENT:    {counts['ABSENT']}")
    if counts["MALFORMED"]:
        print(f"  MALFORMED: {counts['MALFORMED']}")
    print(f"{'=' * 75}\n")

    sys.exit(0 if confirmed == total else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <match_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
