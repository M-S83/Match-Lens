# Step 5 Round-Trip Test — Plan (sketch)

**Status:** sketch only. The round-trip test can only execute once
the Step 5 accumulator merge is in place. This document specifies
what the test must do so the merge can be reviewed against a clear
contract.

**Companion to:** `check_accumulator_preservation.py` (the 11-item
structural-preservation check; already runnable pre-merge).

---

## Goal

Prove that the Step 5 merge of v3 additions into
`accumulator.update_running_summary` is **strictly additive**:

- Every pre-merge top-level key in `running_summary.json` survives,
  with the same value type and structurally-equivalent contents.
- The only differences are the **10 new v3 accumulators** listed in
  V3_PORTING_PLAN.md Section 8 Step 5 block (2).
- Nothing silently disappears, no list becomes a dict, no field name
  shifts.

---

## Pre-merge snapshot (run ONCE before the merge lands)

```python
import json, os, sys
sys.path.insert(0, "scripts")
from accumulator import accumulate_all_windows

MATCH_DIR = "<bayern_match_dir>"

# Reset any stale summary so the snapshot reflects a fresh accumulator run.
# accumulate_all_windows() already resets pass_sequences.json and
# running_summary.json at the top of each invocation, so this is a no-op
# but documents intent.
accumulate_all_windows(MATCH_DIR)

with open(os.path.join(MATCH_DIR, "running_summary.json")) as f:
    pre = json.load(f)

snapshot = {
    "top_level_keys": sorted(pre.keys()),
    "type_per_key":   {k: type(v).__name__ for k, v in pre.items()},
    "list_lengths":   {k: len(v) for k, v in pre.items()
                       if isinstance(v, list)},
    "dict_subkeys":   {k: sorted(v.keys()) for k, v in pre.items()
                       if isinstance(v, dict)},
    # Sample entries from each list -- to detect shape drift, e.g. a
    # post-merge change that re-keys formation_history entries
    "list_first_entry_keys": {
        k: sorted(v[0].keys()) if (v and isinstance(v[0], dict)) else None
        for k, v in pre.items() if isinstance(v, list)
    },
}
json.dump(snapshot, open(os.path.join(MATCH_DIR, "running_summary.preStep5.snapshot.json"), "w"), indent=2)
```

The snapshot file lives in the match dir (not the scripts repo) and
acts as a per-match contract.

---

## Post-merge assertion (run AFTER Step 5 lands)

```python
import json, os, sys
sys.path.insert(0, "scripts")
from accumulator import accumulate_all_windows

MATCH_DIR = "<bayern_match_dir>"
EXPECTED_NEW_KEYS = {
    "line_height_m_by_window",             # block (1)
    "quiet_windows",                       # block (2)
    "vertical_progression_counts",         # block (2)
    "vertical_progression_totals",         # block (2)
    "defending_third_turnover_count",      # block (2)
    "defending_third_sequence_count",      # block (2)
    "watch_list_confirmations",            # block (2)
    "between_lines_events",                # block (2)
    "fouls_committed",                     # block (2)
    "conditional_pattern_observations",    # block (2)
    "temperament_observations",            # block (2)
}
# Note: line_height_m_by_window already exists in production (we saw it
# in the pre-merge baseline — both pct and metres). If the merge adds
# additional metres fields rather than a new top-level key, the test
# should distinguish "new top-level key" from "existing key extended".
# Adjust EXPECTED_NEW_KEYS based on final Step 5 design.

# Re-run accumulator to pick up Step 5 logic
accumulate_all_windows(MATCH_DIR)

with open(os.path.join(MATCH_DIR, "running_summary.json")) as f:
    post = json.load(f)
with open(os.path.join(MATCH_DIR, "running_summary.preStep5.snapshot.json")) as f:
    pre = json.load(f)

# 1. Strict superset: every pre-merge key survives
pre_keys  = set(pre["top_level_keys"])
post_keys = set(post.keys())
missing   = pre_keys - post_keys
assert not missing, f"Step 5 dropped pre-merge keys: {missing}"

# 2. Additions are exactly the expected v3 set
additions = post_keys - pre_keys
unexpected = additions - EXPECTED_NEW_KEYS
assert not unexpected, (
    f"Step 5 added keys not in EXPECTED_NEW_KEYS: {unexpected}\n"
    f"Either the merge introduced an unplanned field, or "
    f"EXPECTED_NEW_KEYS needs updating to match final Step 5 design."
)
missing_v3 = EXPECTED_NEW_KEYS - additions
# Empty v3 accumulators are allowed (no agent input yet) but the KEY
# must exist; absent keys mean the accumulator failed to initialise.
assert not missing_v3, f"Step 5 did not add v3 keys: {missing_v3}"

# 3. Type stability: no key changes value type
type_changes = {
    k: (pre["type_per_key"][k], type(post[k]).__name__)
    for k in pre_keys
    if pre["type_per_key"][k] != type(post[k]).__name__
}
assert not type_changes, (
    f"Step 5 changed value types on existing keys: {type_changes}\n"
    f"This is a structural regression; preserved keys must keep "
    f"their pre-merge type."
)

# 4. List-entry shape stability: for each list of dicts, the keys on
#    the first entry must be a superset of the pre-merge first-entry keys
shape_regressions = {}
for k, pre_entry_keys in pre["list_first_entry_keys"].items():
    if pre_entry_keys is None:
        continue
    post_list = post.get(k)
    if not (post_list and isinstance(post_list[0], dict)):
        continue
    post_entry_keys = set(post_list[0].keys())
    dropped = set(pre_entry_keys) - post_entry_keys
    if dropped:
        shape_regressions[k] = list(dropped)
assert not shape_regressions, (
    f"Step 5 dropped fields from list entries: {shape_regressions}\n"
    f"Per-entry shapes must additively grow, not shrink."
)

# 5. Re-run the structural-preservation checklist; assert 11/11 still pass
import subprocess
ret = subprocess.run(
    ["python", "scripts/diagnostics/check_accumulator_preservation.py", MATCH_DIR],
    capture_output=True, text=True,
)
assert ret.returncode == 0, (
    f"check_accumulator_preservation.py failed post-merge:\n{ret.stdout}\n{ret.stderr}"
)

print("Step 5 round-trip: PASS")
```

---

## What the test does NOT verify

- It does NOT verify that the new accumulators have correct VALUES.
  An empty `vertical_progression_totals` would pass the structural
  test even though it's "wrong" semantically (the source of empty is
  the upstream schema gap tracked as Step 18, not a Step 5 bug).
- It does NOT verify that v3 metric computations downstream of
  Step 5 produce sensible numbers. That's Step 6's territory
  (Confirm v3 reader modules accept production accumulator output).
- It does NOT replace human review of the actual code diff. The
  test catches structural regressions; it cannot catch logic
  regressions inside preserved code paths (e.g. a Step 5 edit that
  accidentally changes a `+=` to a `=`).

---

## Failure handling

If post-merge assertions fail:

| Failure | Most likely cause | Resolution |
|---|---|---|
| `Step 5 dropped pre-merge keys` | The merge accidentally swapped to v3 baseline's init dict rather than extending production's | Restore extension pattern; production init dict must be the base |
| `Step 5 added keys not in EXPECTED_NEW_KEYS` | Final Step 5 design differs from V3_PORTING_PLAN.md Section 8 Step 5 block (2) | Update `EXPECTED_NEW_KEYS` in the test if the addition is intentional; otherwise revert the merge edit that produced it |
| `Step 5 did not add v3 keys: {…}` | Initialisation block missed entries | Re-add to the init defaults; the per-window append logic without an init produces silent KeyErrors downstream |
| `Step 5 changed value types on existing keys` | A list got replaced by a dict, etc. | Hard regression; revert that specific edit |
| `Step 5 dropped fields from list entries` | A per-window append accidentally rebuilt the entry instead of mutating it | Check the per-window block for inadvertent dict replacement |
| `check_accumulator_preservation.py failed post-merge` | One of the 11 hard requirements regressed | The script's output identifies which item; fix at the corresponding line in the merged function |

---

## Multi-match validation

Run the round-trip against more than one match before declaring
Step 5 done. Suggested set:

- Bayern vs PSG — tactical_wide_static, full visibility scores, the
  match Step 4 was validated on
- Felixstowe vs Lowestoft — veo_ball_tracking, conforming
  visibility_scores from the original classifier era
- One match with non-empty `individual_observations` (if any exist
  post-Bug-B with full 3b runs against the new prompt) — needed to
  exercise the `obs_grade` preservation path

If a match's `individual_observations` is empty, the round-trip
still passes structurally (the field is present, just empty), but
the `obs_grade` check inside the preservation script returns
PRESENT-empty rather than PRESENT-populated. Plan accordingly.
