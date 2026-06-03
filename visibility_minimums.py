"""
visibility_minimums.py

Computes scaled minimum observation counts for the player agent prompt,
based on source_profile.visibility_scores.off_ball_coverage_score.

Used at prompt-build time by the orchestrator. The values are injected
into the player agent prompt at [SCALED_TOTAL_MIN], [SCALED_OPP_MIN],
[SCALED_GK_FREQUENCY] placeholders.

Tiers:
    >= 0.7   high coverage     -> 5 total, 2 opp, GK 1-every-3 windows
    0.4-0.7  partial coverage  -> 4 total, 1 opp, GK 1-every-5 windows
    < 0.4    low coverage      -> 3 total, 1 opp, GK not enforced
"""

import json
import os


def compute_minimums(source_profile):
    """
    source_profile -- the dict from source_profile.json

    Returns:
        {
          "off_ball_coverage_score": float,
          "tier":                   "high | partial | low",
          "total_min":              int,
          "opposition_min":         int,
          "gk_frequency":           str  -- e.g. "1 every 3 windows" or
                                              "not enforced -- source limit"
        }
    """
    coverage = (
        source_profile.get("visibility_scores", {})
        .get("off_ball_coverage_score", 0)
    )

    if coverage >= 0.7:
        return {
            "off_ball_coverage_score": coverage,
            "tier":                    "high",
            "total_min":               5,
            "opposition_min":          2,
            "gk_frequency":            "1 every 3 windows",
        }
    if coverage >= 0.4:
        return {
            "off_ball_coverage_score": coverage,
            "tier":                    "partial",
            "total_min":               4,
            "opposition_min":          1,
            "gk_frequency":            "1 every 5 windows",
        }
    return {
        "off_ball_coverage_score": coverage,
        "tier":                    "low",
        "total_min":               3,
        "opposition_min":          1,
        "gk_frequency":            "not enforced -- source limit",
    }


def compute_for_match(match_dir):
    """
    Convenience -- reads match_dir/source_profile.json and returns the
    minimums dict.
    """
    path = os.path.join(match_dir, "source_profile.json")
    if not os.path.exists(path):
        # Default to high coverage if profile missing (don't silently relax)
        return compute_minimums({"visibility_scores": {"off_ball_coverage_score": 0.7}})
    with open(path) as f:
        source_profile = json.load(f)
    return compute_minimums(source_profile)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python visibility_minimums.py [MATCH_DIR]")
        sys.exit(1)
    mins = compute_for_match(sys.argv[1])
    print(json.dumps(mins, indent=2))
