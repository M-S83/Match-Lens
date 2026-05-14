"""
report_filter.py -- Match Lens report level filter

Three levels:
  brief     -- Pre-match brief. 1-2 pages. Manager reads it 30 minutes before kick-off.
  standard  -- Match analysis report. 4-6 pages. Full tactical review.
  technical -- Full technical analysis. 8-12 pages. Analyst-grade depth.

Usage:
    from report_filter import build_report_config, filter_metrics, filter_player_profiles

    config   = build_report_config(match_dir)   # reads report_level from match_config.json
    metrics  = filter_metrics(all_metrics, config)
    profiles = filter_player_profiles(observations_by_player, config)
    prompt   = build_report_prompt(config, report_type="opposition")
"""

import json
import os

# -- Level definitions --------------------------------------------------------

LEVEL_SPECS = {
    "brief": {
        "name":            "Pre-match brief",
        "target_reader":   "Manager, 30 minutes before kick-off",
        "page_target":     "1-2 pages",
        "include_metrics": False,
        "metric_statuses": [],          # no metrics surfaced in brief
        "profile_grades":  ["A"],       # Grade A observations only
        "max_profiles":    3,           # maximum 3 player profiles
        "sentences_per_profile": 4,     # brief -- 3-4 sentences per player
        "include_sections": [
            "formation",
            "key_threats",
            "set_pieces",
            "pressing_triggers",
            "key_players",
        ],
        "language_instruction": (
            "Write in plain, direct language. "
            "No hedging. No evidence tier language ('appeared to', 'suggested'). "
            "No metric values or confidence scores. "
            "State confirmed Grade A observations as fact. "
            "Every sentence must answer: what does the manager need to know right now? "
            "If a sentence doesn't help prepare for the next 90 minutes, cut it."
        ),
        "section_order": [
            "## Formation",
            "## Key Threats",
            "## Pressing Triggers",
            "## Set Pieces",
            "## Players to Watch",
        ],
    },
    "standard": {
        "name":            "Match analysis report",
        "target_reader":   "Coaching staff, pre-session review",
        "page_target":     "4-6 pages",
        "include_metrics": True,
        "metric_statuses": ["sufficient", "preliminary"],  # active only
        "profile_grades":  ["A", "B"],   # A and B in main profile
        "include_c_grades": True,        # C in preliminary sub-section
        "max_profiles":    None,         # all players meeting threshold
        "include_sections": "all",
        "language_instruction": (
            "Analytical but accessible. "
            "Grade A observations: state as fact. "
            "Grade B observations: use 'appeared to', 'in observed windows', 'consistently across the match'. "
            "Grade C observations: open with 'On the one observed occasion' or 'Preliminary indication'. "
            "Do not cite metric values numerically in prose -- use the prose_interpretation text directly. "
            "Evidence tiers are implied by language, not stated explicitly."
        ),
    },
    "technical": {
        "name":            "Full technical analysis",
        "target_reader":   "Analyst / head of analysis",
        "page_target":     "8-12 pages",
        "include_metrics": True,
        "metric_statuses": ["sufficient", "preliminary", "context_only"],
        "profile_grades":  ["A", "B", "C"],
        "include_d_grades": True,       # D listed as 'observed, unprofilable'
        "max_profiles":    None,
        "include_sections": "all",
        "include_data_quality": True,   # full data quality section
        "include_confidence_scores": True,
        "language_instruction": (
            "Full analyst register. "
            "Grade A: state as fact. "
            "Grade B: qualify appropriately. "
            "Grade C: explicit caveat -- 'single observation at [confidence] confidence'. "
            "Include evidence tiers where they affect interpretation. "
            "Metric values may be quoted alongside prose where they add precision. "
            "Source limitations are stated per finding where downgraded. "
            "Data quality section is written in full."
        ),
    },
}

# -- Load config ---------------------------------------------------------------

def build_report_config(match_dir: str) -> dict:
    """
    Read match_config.json and return the level spec for this match.
    Defaults to 'standard' if report_level not set.
    """
    config_path = os.path.join(match_dir, "match_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        level = config.get("report_level", "standard").lower()
    else:
        level = "standard"

    if level not in LEVEL_SPECS:
        print(f"  [WARN] Unknown report_level '{level}' -- defaulting to standard")
        level = "standard"

    spec = dict(LEVEL_SPECS[level])
    spec["level"] = level
    print(f"  Report level: {level} ({spec['name']})")
    print(f"  Target reader: {spec['target_reader']}")
    print(f"  Target length: {spec['page_target']}")
    return spec


# -- Metric filter -------------------------------------------------------------

def filter_metrics(metrics: list, config: dict) -> dict:
    """
    Filter metrics by report level.
    Returns dict with 'include' and 'context_only_notes' lists.
    """
    if not config.get("include_metrics", True):
        return {"include": [], "context_only_notes": [], "all_omitted": True}

    allowed_statuses = config.get("metric_statuses", ["sufficient"])
    include          = []
    context_notes    = []
    omitted          = []

    for m in metrics:
        if m.get("analysis_scope") == "player":
            continue  # player metrics handled separately
        status = m.get("sample_status", "context_only")
        ctx    = m.get("context_only", False)

        if ctx or status == "context_only":
            if "context_only" in allowed_statuses:
                context_notes.append({
                    "metric_name": m["metric_name"],
                    "reason":      "context_only",
                    "note":        f"{m['metric_name']}: insufficient data for a reliable reading.",
                })
            # always show context_only note in technical level
        elif status in allowed_statuses:
            include.append(m)
        else:
            omitted.append(m["metric_name"])

    return {
        "include":          include,
        "context_only_notes": context_notes,
        "omitted":          omitted,
    }


# -- Player profile filter -----------------------------------------------------

def filter_player_profiles(
    observations: list, config: dict, team: str = None
) -> dict:
    """
    Group observations by player, compute profile grade, filter by level.

    Returns:
        full_profiles:        list of {player, grade, grade_summary, obs_A, obs_B}
        preliminary_profiles: list of {player, grade, grade_summary, obs_C}
        unprofilable:         list of player identifiers (Grade D only)
    """
    try:
        from deep_skill_metrics import player_profile_grade, observation_grade
    except ImportError:
        return {"full_profiles": [], "preliminary_profiles": [], "unprofilable": []}

    allowed_main   = set(config.get("profile_grades", ["A", "B"]))
    include_c      = config.get("include_c_grades", False)
    include_d_list = config.get("include_d_grades", False)
    max_profiles   = config.get("max_profiles", None)

    # Filter by team if specified
    team_obs = [o for o in observations if team is None or o.get("team") == team]

    # Group by player identifier
    by_player = {}
    for obs in team_obs:
        pid = obs.get("player") or obs.get("player_id") or "unknown"
        by_player.setdefault(pid, []).append(obs)

    full_profiles        = []
    preliminary_profiles = []
    unprofilable         = []

    for player, obs_list in by_player.items():
        grade, summary_str, ab, c, d, windows = player_profile_grade(obs_list)

        obs_a = [o for o in obs_list
                 if observation_grade(o.get("confidence","low"), o.get("frequency","single")) == "A"]
        obs_b = [o for o in obs_list
                 if observation_grade(o.get("confidence","low"), o.get("frequency","single")) == "B"]
        obs_c = [o for o in obs_list
                 if observation_grade(o.get("confidence","low"), o.get("frequency","single")) == "C"]
        obs_d = [o for o in obs_list
                 if observation_grade(o.get("confidence","low"), o.get("frequency","single")) == "D"]

        profile_record = {
            "player":        player,
            "grade":         grade,
            "grade_summary": summary_str,
            "obs_A":         obs_a,
            "obs_B":         obs_b,
            "obs_C":         obs_c,
            "obs_D":         obs_d,
            "total_obs":     len(obs_list),
            "windows":       windows,
        }

        has_main_grade = any(g in allowed_main for g in
                             (["A"]*len(obs_a) + ["B"]*len(obs_b)))

        if grade == "D" and not obs_a and not obs_b and not obs_c:
            if include_d_list:
                unprofilable.append(player)
        elif has_main_grade:
            full_profiles.append(profile_record)
        elif obs_c and include_c:
            preliminary_profiles.append(profile_record)
        elif grade == "D":
            if include_d_list:
                unprofilable.append(player)

    # Sort by total A+B observations descending
    full_profiles.sort(key=lambda p: len(p["obs_A"]) + len(p["obs_B"]), reverse=True)

    # Cap if brief level
    if max_profiles and len(full_profiles) > max_profiles:
        full_profiles = full_profiles[:max_profiles]

    return {
        "full_profiles":        full_profiles,
        "preliminary_profiles": preliminary_profiles,
        "unprofilable":         unprofilable,
    }


# -- Prompt builder ------------------------------------------------------------

BRIEF_SECTIONS = """
# Pre-match Brief: [Opponent]
### [Home] vs [Away] -- [Competition] -- [Date]

---

## Formation
[Starting shape. One sentence. Name the formation and system -- e.g. "3-4-2-1, dropping to 5-3-1-1 out of possession."]

## Key Threats
[2-4 bullet points. Each is one specific, concrete observation at Grade A. No generalisation.]
- [Threat 1]
- [Threat 2]

## Pressing Triggers
[One sentence on whether and when they press. Source: pressing_trigger_consistency if available.]

## Set Pieces
[One sentence on attacking delivery if sufficient data. One sentence on defensive structure.]

## Players to Watch
[Maximum 3 players. Each is 3-4 sentences. Grade A observations only. Plain language.]

### [Player -- shirt number and position]
[Opening sentence: what they do most and where.]
[Sentence 2: their most useful vulnerability or habit for preparation.]
[Sentence 3: physical profile if captured.]
[Sentence 4: set piece involvement if relevant.]

---
Output as markdown. No preamble. No data quality section.
"""

SECTION_FILTER_BY_LEVEL = {
    "brief":     ["Formation", "Key Threats", "Pressing Triggers", "Set Pieces", "Players to Watch"],
    "standard":  "all",
    "technical": "all_with_data_quality",
}


def build_report_prompt(config: dict, report_type: str = "opposition") -> str:
    """
    Return the level-specific injection block for the report prompt.
    This is prepended to the standard report prompt in pipeline_runner.py.
    """
    level = config.get("level", "standard")
    spec  = LEVEL_SPECS[level]

    lines = [
        f"=== REPORT LEVEL: {level.upper()} ===",
        f"Target reader:  {spec['target_reader']}",
        f"Target length:  {spec['page_target']}",
        f"Player profiles: Grade {', '.join(config.get('profile_grades', ['A','B']))} only",
        f"Metrics:        {'yes -- active only' if config.get('include_metrics') else 'omit from output'}",
        "",
        "LANGUAGE INSTRUCTION:",
        spec["language_instruction"],
        "",
        "=== END REPORT LEVEL ===",
        "",
    ]

    if level == "brief":
        lines.append(BRIEF_SECTIONS)

    return "\n".join(lines)


# -- CLI -----------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python report_filter.py [MATCH_DIR] [brief|standard|technical]")
        sys.exit(1)
    match_dir = sys.argv[1]
    if len(sys.argv) == 3:
        level = sys.argv[2]
        config_path = os.path.join(match_dir, "match_config.json")
        if os.path.exists(config_path):
            with open(config_path, encoding="utf-8") as f:
                mc = json.load(f)
            mc["report_level"] = level
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(mc, f, indent=2)
            print(f"  Set report_level = {level} in match_config.json")
    config = build_report_config(match_dir)
    print(json.dumps(config, indent=2, default=str))
