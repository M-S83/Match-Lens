# Step 8 Round-Trip Test — Plan (sketch)

**Status:** sketch only. Executes once the Step 8 metrics merge is in
place. Mirrors `step5_roundtrip_test_plan.md` — same discipline, same
shape, different module.

**Companion to:** `check_deep_skill_metrics_preservation.py` (the
16-item v2-name preservation check; already runnable pre-merge).

---

## Goal

Prove that the Step 8 merge of v3-distinctive metrics into
`deep_skill_metrics.build_deep_skill_metrics` is **strictly additive**:

- Every pre-merge v2 metric name survives, in any of the three lists
  (`metrics`, `unavailable_metrics`, `context_only_metrics`) — the
  preservation check (the script) enforces this.
- The only additions are the **5 v3-distinctive metric names** listed
  in `EXPECTED_NEW_METRICS` below.
- Top-level schema fields are unchanged.
- No type drift on existing metrics (value field shape preserved).

The pivot from "swap" (the original plan) to "additive merge into v2"
(the Step-5-style pattern, approved after Task 60 surfaced the
metric-set reduction risk) means v2 is canonical and v3 is the source
of NEW metric functions only. `build_deep_skill_metrics_v2.py` stays
on disk as documentation, uninvoked at runtime — same fate as
`update_running_summary_v2.py` after Step 5.

---

## Pre-merge snapshot (run ONCE before the merge lands)

```python
import json, os, sys
sys.path.insert(0, "scripts")
from deep_skill_metrics import build_deep_skill_metrics

MATCH_DIR = "<bayern_match_dir>"

# Fresh v2 run -- captures the canonical baseline shape under current
# inputs (post-Step-3 source_profile, post-Step-5 running_summary).
build_deep_skill_metrics(MATCH_DIR, "both")

with open(os.path.join(MATCH_DIR, "deep_skill_metrics.json")) as f:
    pre = json.load(f)

# Extract baseline structure for post-merge comparison.
snapshot = {
    "top_level_keys": sorted(pre.keys()),
    "type_per_key":   {k: type(v).__name__ for k, v in pre.items()},
    "active_count":   len([m for m in pre.get("metrics", [])
                           if isinstance(m, dict)]),
    "unavailable_count":  len(pre.get("unavailable_metrics", [])),
    "context_only_count": len(pre.get("context_only_metrics", [])),
    "total_metrics":      pre.get("total_metrics", 0),
    # Per-metric value type, captured by metric_name -- detects shape
    # drift (e.g. a metric whose value was a dict becoming a float).
    "metric_value_types": {
        m["metric_name"]: type(m.get("value")).__name__
        for m in pre.get("metrics", [])
        if isinstance(m, dict) and m.get("metric_name")
    },
    "metric_confidence_band": {
        m["metric_name"]: (
            round(m.get("confidence", 0.0), 2)
            if isinstance(m.get("confidence"), (int, float))
            else None
        )
        for m in pre.get("metrics", [])
        if isinstance(m, dict) and m.get("metric_name")
    },
}
json.dump(snapshot,
          open(os.path.join(MATCH_DIR, "deep_skill_metrics.preStep8.snapshot.json"), "w"),
          indent=2)
```

The snapshot file lives in the match dir (per-match contract). Bayern's
pre-Step-8 raw v2 file (`deep_skill_metrics.preStep8.json`) is also
preserved from the Task 61 diagnostic run.

---

## Post-merge assertion (run AFTER Step 8 merge lands)

```python
import json, os, sys
sys.path.insert(0, "scripts")
from deep_skill_metrics import build_deep_skill_metrics

MATCH_DIR = "<bayern_match_dir>"

# The 5 v3-distinctive metric names that the merge adds.
# Source: build_deep_skill_metrics_v2.py:625-633 (the v3 module's
# metric calls). Of the 8 calls there, 3 reuse v2 names
# (build_up_effectiveness_score, line_height_range,
# pattern_reliability_score) which the merge enhances or leaves as
# the v2 implementation; the 5 below are genuinely new.
EXPECTED_NEW_METRICS = {
    "compactness_geometry_score",
    "defensive_third_turnover_rate",
    "between_lines_receiving_rate",
    "duel_effectiveness",
    "watch_list_summary",
}

# Re-run the (now merged) deep_skill_metrics module
build_deep_skill_metrics(MATCH_DIR, "both")

with open(os.path.join(MATCH_DIR, "deep_skill_metrics.json")) as f:
    post = json.load(f)
with open(os.path.join(MATCH_DIR, "deep_skill_metrics.preStep8.snapshot.json")) as f:
    pre = json.load(f)


def _all_names(d):
    """Union of metric names across active / unavailable / context_only."""
    active = {m.get("metric_name") for m in (d.get("metrics", []) or [])
              if isinstance(m, dict) and m.get("metric_name")}
    return active | set(d.get("unavailable_metrics", []) or []) \
                  | set(d.get("context_only_metrics", []) or [])


# 1. Strict superset on metric names: every v2 name survives.
pre_names  = set(pre["metric_value_types"].keys()) \
           | set(pre.get("unavailable_count_names", [])) \
           | set(pre.get("context_only_count_names", []))
# NOTE: simplest implementation -- run the v2-baseline-list assertion
# via the preservation check rather than reconstructing from snapshot.
# Suggested:
import subprocess
ret = subprocess.run(
    ["python", "scripts/diagnostics/check_deep_skill_metrics_preservation.py", MATCH_DIR],
    capture_output=True, text=True,
)
assert ret.returncode == 0, (
    f"Preservation check failed:\n{ret.stdout}\n{ret.stderr}"
)

post_names = _all_names(post)

# 2. Additions == EXPECTED_NEW_METRICS
additions = post_names - set(pre["metric_value_types"].keys()) - \
            set(pre.get("unavailable_count_names", [])) - \
            set(pre.get("context_only_count_names", []))
# Filter to genuinely-new additions only (excluding any that happen
# to also be v2 names enhanced by the merge — those count as
# preservation, not addition).
genuine_new = additions & EXPECTED_NEW_METRICS
unexpected  = additions - EXPECTED_NEW_METRICS
assert not unexpected, (
    f"Step 8 added unexpected names: {unexpected}\n"
    f"Either the merge introduced an unplanned metric, or "
    f"EXPECTED_NEW_METRICS needs updating to match final design."
)
missing_v3 = EXPECTED_NEW_METRICS - additions
assert not missing_v3, (
    f"Step 8 did not add expected v3 names: {missing_v3}"
)

# 3. Top-level schema field stability
pre_top  = set(pre["top_level_keys"])
post_top = set(post.keys())
removed = pre_top - post_top
assert not removed, (
    f"Step 8 dropped top-level schema fields: {removed}\n"
    f"This is a structural regression; preserved keys must keep "
    f"their pre-merge presence."
)

# 4. Metric counts not LOWER than pre-merge.
# The merge is additive, so active/unavailable/context_only counts
# can only stay the same or grow. Total = sum of all three.
assert post.get("active_metrics", 0) >= pre["active_count"], (
    f"active count regressed: pre={pre['active_count']} "
    f"post={post.get('active_metrics', 0)}"
)
assert post.get("total_metrics", 0) >= pre["total_metrics"], (
    f"total count regressed: pre={pre['total_metrics']} "
    f"post={post.get('total_metrics', 0)}"
)

# 5. Per-metric value shape stability for preserved v2 metrics.
# A metric whose value was a dict pre-merge must still be a dict
# post-merge. Confidence may drift within reasonable bounds but
# the value field type stays.
for name, pre_type in pre["metric_value_types"].items():
    for m in (post.get("metrics", []) or []):
        if (isinstance(m, dict) and
            m.get("metric_name") == name):
            post_type = type(m.get("value")).__name__
            assert post_type == pre_type, (
                f"value type drift on {name}: pre={pre_type} "
                f"post={post_type}"
            )
            break

print("Step 8 round-trip: PASS")
```

---

## EXPECTED_NEW_METRICS

The five v3-distinctive metric names the merge adds to v2:

| Name | Source (v3 module) | What it depends on |
|---|---|---|
| `compactness_geometry_score` | `_metric_compactness_geometry` | `line_height_m_by_window.avg_m_approx` + `line_width_m_approx`. Available on Bayern (Step 5 produced both fields). |
| `defensive_third_turnover_rate` | `_metric_defensive_third_turnover_rate` | `defending_third_turnover_count` / `defending_third_sequence_count` (both Step 5 additions; Step 18 dependency for non-zero values). |
| `between_lines_receiving_rate` | `_metric_between_lines_receiving_rate` | `between_lines_events[]` (Step 5 addition; populated by future v3 player prompt). |
| `duel_effectiveness` | `_metric_duel_effectiveness` | `duels[]` entries with `post_duel_outcome` (Step 5 enriched duels block; agent must emit `post_duel_outcome` for non-zero values). |
| `watch_list_summary` | `_metric_watch_list_summary` | `watch_list_confirmations[]` (Step 5 addition; populated by Step 14 player-action confirmation loop). |

On Bayern (current pre-merge state), all 5 will resolve to
"unavailable" until their respective upstream dependencies land
(Steps 14 / 18 / v3 player prompt deployment). The merge SUCCESS
criterion is that the names appear in either `metrics` OR
`unavailable_metrics` — populated values come later.

---

## Failure-mode → resolution table

| Assertion failure | Most likely cause | Resolution |
|---|---|---|
| `Preservation check failed` (v2 name missing) | The merge replaced a v2 metric function instead of leaving it alone | Restore the v2 function; the merge is ADDITIVE — never delete or replace v2 logic |
| `Step 8 added unexpected names` | Merge accidentally introduced metric not in EXPECTED_NEW_METRICS | Update EXPECTED_NEW_METRICS if intentional; otherwise revert that addition |
| `Step 8 did not add expected v3 names` | One of the 5 v3 helpers wasn't ported | Add the missing `_metric_*` function call to the metrics list |
| `Step 8 dropped top-level schema fields` | Merge edits altered the output-dict construction | Restore the original output-dict pattern; additions go into the metrics list, not the top-level schema |
| `active count regressed` | A v2 metric that was active pre-merge is now unavailable | Check whether the merge accidentally altered v2 source-cap logic or sample-count thresholds |
| `value type drift on <name>` | Merge edits to v2 logic changed a metric's value-field shape | Check that the v2 function for that metric wasn't unintentionally edited |

---

## What the test does NOT verify

- **Numeric value correctness on the 5 v3 additions.** Their values are
  expected to be zero/null on Bayern because of the upstream
  dependencies (Step 18 / Step 14). Real validation happens when those
  steps land.
- **SKILL.md report-rule references to dropped-and-now-restored names.**
  Task 60 surfaced that 7 SKILL.md "Draw from:" lines reference v2
  metrics. The merge restores all 16 v2 names, so those references
  are no longer dangling — but the test doesn't validate that any
  specific report sentence still works.
- **Cross-match comparison.** The test runs against one match. Other
  matches with broader visibility (different `source_profile`) may
  produce a different metric count distribution. The contract is
  "no v2 name is silently dropped," not "exactly N metrics produced."
- **Logic-equivalence on enhanced v2 metrics.** Three v3 names overlap
  v2 (`build_up_effectiveness_score`, `line_height_range`,
  `pattern_reliability_score`). The merge keeps the v2 implementation
  unchanged. The v3 module's variants stay on disk as documentation.

---

## Multi-match validation

Same recommendation as Step 5's round-trip: run against more than one
match before declaring Step 8 done.

| Match | Why |
|---|---|
| Bayern vs PSG | tactical_wide_static, post-Step-3 reclassification, the canonical Step 8 baseline |
| Felixstowe vs Lowestoft | veo_ball_tracking, conforming visibility_scores, different source-cap behaviour |
| Wingate vs Cray or Billericay vs Brentwood | reclassified to tactical_wide_static at Task 53; tests behaviour against the originally-hand-curated cohort |

If a match shows a different active count distribution, that's expected
(visibility-score driven). What must hold uniformly: the preservation
check passes 16/16 and the 5 v3 names are present somewhere in the
output.
