"""check_deep_skill_metrics_preservation.py

Pre-merge diligence for V3 porting Step 8 (the metrics-module merge).

Mirrors check_accumulator_preservation.py. Asserts that every v2 metric
name produced by deep_skill_metrics.build_deep_skill_metrics survives
the Step 8 merge -- the additive merge of v3-distinctive metrics into
the v2 module must NOT silently drop any v2 metric.

The baseline list was captured by running v2 fresh against Bayern vs
PSG 2026-05-06 (post-Step-3 reclassification of source_profile, which
unlocks the structural metrics). The set of 16 metric names below is
the union of `metrics[*].metric_name`, `unavailable_metrics`, and
`context_only_metrics` from that run.

Usage:
    python scripts/diagnostics/check_deep_skill_metrics_preservation.py <match_dir>

Exit code: 0 if all 16 v2 metric names are present in the metrics
file (across the three lists -- active, unavailable, context_only),
1 if any are missing.

The check is location-agnostic: a metric reported as "context_only"
still satisfies preservation. Only TOTAL ABSENCE -- the name appears
in no list -- counts as a failure.

What this script does NOT check:
  - Numeric value correctness (each metric is a separate evaluation)
  - The 5 v3-distinctive additions are present (that's the round-trip
    test's EXPECTED_NEW_METRICS check, not preservation)
  - Schema-level top-level field stability (also round-trip's job)
"""

import json
import os
import sys


# v2 baseline -- 16 unique metric names produced by
# deep_skill_metrics.build_deep_skill_metrics(match_dir, "both") on
# Bayern vs PSG 2026-05-06 (post-Step-3 reclassification).
#
# 10 active + 4 unavailable + 4 context_only with 2 overlapping
# between unavailable and context_only -> 16 unique.
#
# Step 8 merge MUST preserve every name -- the merge is additive on
# top of v2, so this list is what "no metric was silently dropped"
# means in practice.
V2_BASELINE_METRIC_NAMES = (
    "attacking_support_score",
    "build_up_effectiveness_score",
    "build_up_route_diversity",
    "chance_creation_profile",
    "compactness_score",
    "halfspace_occupation_score",
    "line_height_range",
    "pattern_reliability_score",
    "possession_balance",
    "predictability_score",
    "pressing_intensity_score",
    "pressing_trigger_consistency",
    "rest_defence_security_score",
    "set_piece_delivery_profile",
    "transition_efficiency_score",
    "width_usage_score",
)


def _all_names_in_file(metrics_dict):
    """Union of metric names across the three lists v2 emits."""
    active = set()
    for m in (metrics_dict.get("metrics", []) or []):
        if isinstance(m, dict):
            n = m.get("metric_name") or m.get("name")
            if n:
                active.add(n)
    unavailable  = set(metrics_dict.get("unavailable_metrics", []) or [])
    context_only = set(metrics_dict.get("context_only_metrics", []) or [])
    return active, unavailable, context_only


def _describe_metric_in_active(metrics_dict, name):
    """Return a short shape description for a name found in active."""
    for m in (metrics_dict.get("metrics", []) or []):
        if isinstance(m, dict) and (m.get("metric_name") or m.get("name")) == name:
            conf = m.get("confidence")
            val  = m.get("value")
            val_type = type(val).__name__
            if isinstance(val, dict):
                val_keys = sorted(val.keys())[:5]
                val_repr = f"dict[{val_keys}]"
            elif isinstance(val, (int, float)):
                val_repr = f"{val_type}={val}"
            elif val is None:
                val_repr = "None"
            else:
                val_repr = f"{val_type}={str(val)[:30]}"
            return f"conf={conf} value={val_repr}"
    return ""


def main(match_dir):
    metrics_path = os.path.join(match_dir, "deep_skill_metrics.json")
    if not os.path.exists(metrics_path):
        print(f"  [FAIL] deep_skill_metrics.json not found at {metrics_path}",
              file=sys.stderr)
        sys.exit(2)

    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)

    active, unavailable, context_only = _all_names_in_file(metrics)
    all_present = active | unavailable | context_only

    print(f"\n{'=' * 78}")
    print(f"  Deep skill metrics preservation check -- {os.path.basename(match_dir)}")
    print(f"  ({len(active)} active, {len(unavailable)} unavailable, "
          f"{len(context_only)} context_only, "
          f"{len(all_present)} unique total)")
    print(f"{'=' * 78}\n")

    missing = []
    for name in V2_BASELINE_METRIC_NAMES:
        if name in active:
            detail = _describe_metric_in_active(metrics, name)
            print(f"  [OK] {name:<38} ACTIVE       {detail}")
        elif name in unavailable:
            print(f"  [OK] {name:<38} UNAVAILABLE  (declared in unavailable_metrics)")
        elif name in context_only:
            print(f"  [OK] {name:<38} CONTEXT_ONLY (declared in context_only_metrics)")
        else:
            missing.append(name)
            print(f"  [X]  {name:<38} ABSENT       (not in any of the three lists)")

    confirmed = len(V2_BASELINE_METRIC_NAMES) - len(missing)
    total = len(V2_BASELINE_METRIC_NAMES)
    print(f"\n{'=' * 78}")
    print(f"  Summary: {confirmed}/{total} v2 baseline metric names confirmed present")
    if missing:
        print(f"  ABSENT (Step 8 merge regression):")
        for n in missing:
            print(f"    - {n}")
    print(f"{'=' * 78}\n")

    sys.exit(0 if confirmed == total else 1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <match_dir>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
