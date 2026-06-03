"""
zone_helpers.py

Programmatic helpers for the v2 zone system.

Public functions:
    derive_vertical_progression(zone_start, zone_end) -> str
    resolve_named_zone(zone_obj) -> str | None
    normalise_zone(zone_obj) -> dict
    validate_between_lines(zone_obj, source_profile) -> dict
    walk_findings_apply_zone_helpers(match_dir) -> None

Zone object shape:
    {
      "vertical_third": "defending | middle | attacking",
      "lateral_lane":   "wide_left | halfspace_left | central | halfspace_right | wide_right",
      "named_zone":     str | None,
      "between_lines":  str | None
    }

This module is deterministic. It runs after agent JSON is parsed, before the
data feeds into running_summary and deep_skill_metrics. It does NOT call LLMs
or extract frames -- it just normalises and derives.

Run order in the pipeline:
    Step 3a writes agent_NN_*.json    ->  agent zone fields filled
    Step 3b writes _player.json       ->  agent zone fields filled
    Step 3e merges into _merged.json  ->  call walk_findings_apply_zone_helpers
    Step 3f appends to pass_sequences ->  vertical_progression already on each
"""

import json
import os
import glob


# ----- Constants --------------------------------------------------------------

VALID_VERTICAL_THIRDS = {"defending", "middle", "attacking"}

VALID_LATERAL_LANES = {
    "wide_left", "halfspace_left", "central", "halfspace_right", "wide_right",
}

VALID_NAMED_ZONES = {
    "zone_14",
    "second_six_yard_box",
    "six_yard_box",
    "penalty_box",
    "build_up_halfspace_left",
    "build_up_halfspace_right",
    "build_up_central",
    "final_third_wide_left",
    "final_third_wide_right",
}

VALID_BETWEEN_LINES = {
    "between_def_mid",
    "between_mid_fwd",
    "between_fb_cb",
}

# Vertical-third progression matrix.
# Index 0 = defending, 1 = middle, 2 = attacking.
_THIRD_INDEX = {"defending": 0, "middle": 1, "attacking": 2}

_PROGRESSION_MATRIX = {
    (0, 0): "same_third",
    (1, 1): "same_third",
    (2, 2): "same_third",
    (0, 1): "defending_to_middle",
    (1, 2): "middle_to_attacking",
    (0, 2): "defending_to_attacking",
    (1, 0): "regression_middle_to_defending",
    (2, 1): "regression_attacking_to_middle",
    (2, 0): "regression_attacking_to_defending",
}


# ----- Vertical progression ---------------------------------------------------

def derive_vertical_progression(zone_start, zone_end):
    """
    Derive vertical_progression from zone_start and zone_end objects.
    Returns one of:
        same_third
        defending_to_middle
        middle_to_attacking
        defending_to_attacking
        regression_middle_to_defending
        regression_attacking_to_middle
        regression_attacking_to_defending
        unknown  -- if either third is missing or invalid
    """
    if not isinstance(zone_start, dict) or not isinstance(zone_end, dict):
        return "unknown"

    v_start = zone_start.get("vertical_third")
    v_end = zone_end.get("vertical_third")

    if v_start not in _THIRD_INDEX or v_end not in _THIRD_INDEX:
        return "unknown"

    key = (_THIRD_INDEX[v_start], _THIRD_INDEX[v_end])
    return _PROGRESSION_MATRIX.get(key, "unknown")


# ----- Named zone auto-resolution --------------------------------------------

def resolve_named_zone(zone_obj):
    """
    Given a zone object with vertical_third and lateral_lane, resolve the
    named_zone overlay code automatically. Returns None if no named zone
    applies.

    This is geometry-based -- it does NOT need the agent to fill named_zone.
    The agent may fill it for hints (e.g. when the ball is inside the 18yd
    box and the agent can see that explicitly), but this function is the
    source of truth for grid-derived named zones.

    NOTE: penalty_box and six_yard_box require the agent to confirm the ball
    is INSIDE the box (not just within the lateral_lane). For those two,
    we only auto-resolve when the agent has already flagged them.
    """
    if not isinstance(zone_obj, dict):
        return None

    v = zone_obj.get("vertical_third")
    lane = zone_obj.get("lateral_lane")
    agent_supplied = zone_obj.get("named_zone")

    # Agent-supplied box overlays take precedence (they require inside-box
    # visibility that geometry alone cannot determine)
    if agent_supplied in {"six_yard_box", "penalty_box", "second_six_yard_box"}:
        return agent_supplied

    # Geometric overlays we can resolve from grid alone
    if v == "defending" and lane == "halfspace_left":
        return "build_up_halfspace_left"
    if v == "defending" and lane == "halfspace_right":
        return "build_up_halfspace_right"
    if v == "defending" and lane == "central":
        return "build_up_central"
    if v == "attacking" and lane == "central":
        # zone_14 is the area OUTSIDE the 18yd box but inside the third.
        # Agent flags inside-box overlays explicitly; if they didn't, default
        # to zone_14 for attacking central.
        return "zone_14"
    if v == "attacking" and lane == "wide_left":
        return "final_third_wide_left"
    if v == "attacking" and lane == "wide_right":
        return "final_third_wide_right"

    return None


# ----- Zone normalisation -----------------------------------------------------

def normalise_zone(zone_obj):
    """
    Returns a normalised copy of zone_obj:
      - validates vertical_third and lateral_lane
      - auto-resolves named_zone if agent left it null
      - clears between_lines if not in valid set
      - returns a fresh dict (does not mutate input)

    Invalid zones are returned with original values but with a _zone_invalid
    flag set to true. Downstream code can drop these from metric computation.
    """
    if not isinstance(zone_obj, dict):
        return {
            "vertical_third": None,
            "lateral_lane":   None,
            "named_zone":     None,
            "between_lines":  None,
            "_zone_invalid":  True,
        }

    v = zone_obj.get("vertical_third")
    lane = zone_obj.get("lateral_lane")
    named = zone_obj.get("named_zone")
    between = zone_obj.get("between_lines")

    invalid = False
    if v is not None and v not in VALID_VERTICAL_THIRDS:
        invalid = True
    if lane is not None and lane not in VALID_LATERAL_LANES:
        invalid = True
    if named is not None and named not in VALID_NAMED_ZONES:
        # Agent supplied an unknown code -- drop it but don't invalidate
        named = None
    if between is not None and between not in VALID_BETWEEN_LINES:
        between = None

    # Auto-resolve named_zone if agent left it null
    if named is None and not invalid:
        named = resolve_named_zone({
            "vertical_third": v,
            "lateral_lane":   lane,
            "named_zone":     None,
        })

    result = {
        "vertical_third": v,
        "lateral_lane":   lane,
        "named_zone":     named,
        "between_lines":  between,
    }
    if invalid:
        result["_zone_invalid"] = True
    return result


# ----- Between-lines validation ----------------------------------------------

def validate_between_lines(zone_obj, source_profile):
    """
    Returns a dict:
      {
        "between_lines_kept":    bool,
        "between_lines_value":   str | None,
        "downgrade_reason":      str | None
      }

    Rule: between_lines findings require off-ball coverage. If source has
    low off_ball_coverage_score, between_lines findings are downgraded to
    null and a reason is recorded for the limitations note.
    """
    if not isinstance(zone_obj, dict):
        return {
            "between_lines_kept":  False,
            "between_lines_value": None,
            "downgrade_reason":    "zone_obj missing",
        }

    between = zone_obj.get("between_lines")
    if between is None or between not in VALID_BETWEEN_LINES:
        return {
            "between_lines_kept":  False,
            "between_lines_value": None,
            "downgrade_reason":    None,
        }

    coverage = (
        source_profile.get("visibility_scores", {})
        .get("off_ball_coverage_score", 0)
    )

    if coverage < 0.4:
        return {
            "between_lines_kept":  False,
            "between_lines_value": None,
            "downgrade_reason": (
                f"Source off_ball_coverage_score {coverage:.2f} below 0.4 -- "
                "between_lines not reliably observable; downgraded to null."
            ),
        }

    return {
        "between_lines_kept":  True,
        "between_lines_value": between,
        "downgrade_reason":    None,
    }


# ----- Walk merged JSONs and apply helpers -----------------------------------

def walk_findings_apply_zone_helpers(match_dir):
    """
    Reads every *_merged.json in agent_logs/ and applies:
      - normalise_zone() to every zone object in pass_sequences,
        individual_observations, duels, findings
      - derive_vertical_progression() onto every pass_sequence
      - validate_between_lines() against source_profile.json

    Writes back to the same files. Idempotent -- safe to re-run.
    """
    logs_dir = os.path.join(match_dir, "agent_logs")
    profile_path = os.path.join(match_dir, "source_profile.json")

    if not os.path.exists(profile_path):
        print(f"[zone_helpers] source_profile.json missing -- skipping")
        return

    with open(profile_path) as f:
        source_profile = json.load(f)

    merged_files = sorted(glob.glob(os.path.join(logs_dir, "*_merged.json")))
    updated_count = 0

    for path in merged_files:
        with open(path) as f:
            data = json.load(f)

        # Pass sequences
        for seq in data.get("pass_sequences", []):
            zs = seq.get("zone_start")
            ze = seq.get("zone_end")
            if zs:
                seq["zone_start"] = normalise_zone(zs)
            if ze:
                seq["zone_end"] = normalise_zone(ze)
            seq["vertical_progression"] = derive_vertical_progression(
                seq.get("zone_start"), seq.get("zone_end")
            )

        # Individual observations
        for obs in data.get("individual_observations", []):
            zone = obs.get("zone")
            if zone:
                obs["zone"] = normalise_zone(zone)
                bl = validate_between_lines(obs["zone"], source_profile)
                if not bl["between_lines_kept"]:
                    obs["zone"]["between_lines"] = None
                    if bl["downgrade_reason"]:
                        existing = obs.get("limitations_note") or ""
                        obs["limitations_note"] = (
                            existing + " | " + bl["downgrade_reason"]
                        ).strip(" |")

        # Duels
        for duel in data.get("duels", []):
            zone = duel.get("zone")
            if zone:
                duel["zone"] = normalise_zone(zone)

        # Findings (universal finding objects)
        for finding in data.get("findings", []):
            zone = finding.get("zone")
            if zone:
                finding["zone"] = normalise_zone(zone)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        updated_count += 1

    print(f"[zone_helpers] Normalised {updated_count} merged window JSONs")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python zone_helpers.py [MATCH_DIR]")
        sys.exit(1)
    walk_findings_apply_zone_helpers(sys.argv[1])
