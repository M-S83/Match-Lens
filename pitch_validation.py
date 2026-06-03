"""
pitch_validation.py

Step 1d helper. Validates pitch_dimensions_assumed in match_config.json at
team-sheet confirmation time. Prints warnings for non-default venues that
have not been overridden.

Run from confirm_team_sheet() AFTER mandatory field validation and BEFORE
writing match_config.json with verified=True.
"""

import json
import os


# Venues known to have non-standard pitches. Extend as you encounter them.
# Add entries in the form:
#   "venue_name_substring": {"length_m": N, "width_m": M, "note": "..."}
KNOWN_NON_STANDARD_VENUES = {
    # Example entries -- replace with real venue data when known
    # "lowestoft": {"length_m": 100, "width_m": 64, "note": "Crown Meadow -- narrower than UEFA default"},
    # "felixstowe": {"length_m": 102, "width_m": 66, "note": "Goldstar Ground"},
}


def validate_pitch_dimensions(match_config):
    """
    match_config -- dict loaded from match_config.json

    Returns a dict:
      {
        "ok":             bool,
        "warnings":       [str, ...],
        "venue_match":    dict | None,   -- suggested override values
        "current_source": str
      }

    Does NOT mutate match_config. Caller decides whether to override.
    """
    pd = match_config.get("pitch_dimensions_assumed")

    # Block: missing entirely
    if not pd:
        return {
            "ok":             False,
            "warnings":       ["pitch_dimensions_assumed missing -- add to match_config.json before continuing"],
            "venue_match":    None,
            "current_source": None,
        }

    warnings = []
    source = pd.get("source", "default")
    length = pd.get("length_m")
    width = pd.get("width_m")

    if length is None or width is None:
        return {
            "ok":             False,
            "warnings":       ["pitch_dimensions_assumed missing length_m or width_m"],
            "venue_match":    None,
            "current_source": source,
        }

    # Plausibility bounds (FA min/max)
    if not (90 <= length <= 120):
        warnings.append(f"length_m {length} outside FA range 90-120 -- verify")
    if not (45 <= width <= 90):
        warnings.append(f"width_m {width} outside FA range 45-90 -- verify")

    # Check against known-non-standard venue list
    venue = (match_config.get("venue") or "").lower()
    venue_match = None
    for substring, dims in KNOWN_NON_STANDARD_VENUES.items():
        if substring in venue:
            venue_match = dims
            if source == "default":
                warnings.append(
                    f"Venue '{match_config.get('venue')}' is in the "
                    f"known-non-standard list (~{dims['length_m']}x{dims['width_m']}m, "
                    f"{dims['note']}) but pitch_dimensions_assumed.source is "
                    "still 'default'. Consider setting source='venue_known' and "
                    "overriding length_m/width_m."
                )
            break

    return {
        "ok":             len(warnings) == 0,
        "warnings":       warnings,
        "venue_match":    venue_match,
        "current_source": source,
    }


def print_validation_report(result):
    """Pretty-print the validation result for terminal review."""
    if result["ok"]:
        print(f"[pitch_validation] OK (source: {result['current_source']})")
        return

    print(f"[pitch_validation] WARNINGS (source: {result['current_source']})")
    for w in result["warnings"]:
        print(f"  - {w}")
    if result["venue_match"]:
        vm = result["venue_match"]
        print(f"  Suggested override: length_m={vm['length_m']}, "
              f"width_m={vm['width_m']}, source='venue_known'")


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python pitch_validation.py [MATCH_DIR]")
        sys.exit(1)
    config_path = os.path.join(sys.argv[1], "match_config.json")
    with open(config_path) as f:
        config = json.load(f)
    result = validate_pitch_dimensions(config)
    print_validation_report(result)
