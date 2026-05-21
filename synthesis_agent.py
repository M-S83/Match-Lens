"""
Match Lens — Synthesis Agent (Step 3l)
Reads all accumulated pipeline outputs and produces:
  - tactical_report.md      (both teams, match as it happened)
  - opposition_report_[team].md  (one per team, independent)

No raw frames are read. No individual window files are accessed.
All input comes from validated, accumulated pipeline outputs.

Usage (standalone):
    python synthesis_agent.py "C:/path/to/match_dir"

Called from pipeline_runner_v2.py as:
    ("3l_synthesis", "synthesis_agent", "run_synthesis")
"""

import json
import os
import sys
import re
import time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

# Fix 42: bump when the Section 3 §3.4 output contract firing rules change.
# Surfaced into the report manifest (Section 5).
CONTRACT_VERSION = "section-3.4-2026-05-08"

# ── Prompt (4 sections, assembled at runtime) ─────────────────────────────

SYNTHESIS_PROMPT = r"""
You are a football tactical analyst producing match reports for a non-league
coaching staff at Step 4 level -- Isthmian League, Northern Premier League,
and equivalent. Your reader is technically competent and expects precise
football language. You do not explain tactical concepts. You use them.

=== SECTION 1: ROLE AND INPUT ===

You have received validated, accumulated pipeline outputs listed below.
You do not re-read raw frames. You synthesise from these inputs only.

Your task is to produce two documents:

  Document 1 -- tactical_report.md
    The match as it happened. Both teams. Phase by phase. Includes key moments.

  Document 2 -- opposition_report_{away_team}.md
    The away team profiled in isolation.
    
  Document 3 -- opposition_report_{home_team}.md
    The home team profiled in isolation.

Before writing any prose output the following single line:
  ROSTER CHECK: PASSED
Confirm every player name you write appears in match_config for the correct team.

=== SECTION 2: EVIDENCE RULES ===

All statements must be grounded in the observation data provided.

You may connect observations into a pattern statement when THREE OR MORE
observations of the same phenomenon appear across different windows.
Below that threshold it is a single-observation note, graded accordingly.

A-D EVIDENCE SCALE (team tactical findings):
  A -- Pattern confirmed, multiple windows, high confidence. State as fact.
       "Their defensive line sat consistently deep in the second half."
  B -- Multiple windows medium confidence, OR single window high confidence.
       Light qualifier. "Throughout the observed phases, their press
       initiated from the striker engaging the centre-backs."
  C -- Single observation medium confidence, OR low confidence repeated.
       Explicit caveat. "On the one occasion where their shape was visible
       from a wide frame, the left side appeared narrow."
  D -- Observed but insufficient for a reliable reading. One sentence only.

A-E ATTRIBUTE SCALE (player profiles only):
  Relative to quality observed in this match at this level -- not absolute.
  A -- Highest quality observed for this attribute in this match.
  B -- Above average for the level as observed in this match.
  C -- Average for the level as observed in this match.
  D -- Below average for the level as observed in this match.
  E -- Significantly below average, or no usable evidence for this attribute.

Attribute scores come from observation data only. Score only attributes
with supporting observations. Do not estimate.

FORMATION RULE: When describing team formations, use the
canonical_formation from running_summary as the primary label.
However, if the formation distribution shows a meaningful split
across windows, describe the variation rather than stating a
single fixed formation.

A meaningful split is when the second-most-common formation
accounts for 25% or more of observed windows. In that case:
- State the primary formation as the default shape
- Describe the secondary formation as a phase-dependent variant
- Explain what player movement or tactical trigger drove the shift

Example of correct variation narrative:
"Cray operated predominantly in a 4-2-3-1 shape but showed a
4-1-4-1 variant across approximately a third of observed windows.
Raymond (#6) consistently screened as the single pivot; Hernandez's
(#10) positioning between the lines -- dropping deeper in defensive
phases, pushing higher in attacking transitions -- determined which
reading applied at any given moment."

Do NOT average formations into a hybrid label (e.g. "4-1.5-3-1").
Do NOT describe every window's formation individually.
Do use player names and shirt numbers to anchor the variation to
specific individuals whose movement caused the shape shift.

If the distribution is clear (one formation accounts for 70%+
of windows), state it as the settled shape without caveat.

IP SHAPE RULE:
If ip_shape_summary is present in running_summary with available: true,
describe in-possession shape variations in the Formation and Shape section.

Use home_variations and away_variations counts to determine frequency.
Only assert patterns with 3+ observations. Do not assert variations with
fewer than 2 observations.

Examples of correct IP shape language:
"In attacking phases Bayern regularly pushed both fullbacks into the
front line, creating a 3-2-5 structure with the double pivot staying
deep [A — observed in multiple phases]"

"PSG's 4-3-3 compressed to a 4-5-1 out of possession throughout the
match, with both wide forwards tracking back deep [A]"

Do NOT describe IP shape variations as formations — they are
phase-specific shape changes within the primary formation.
Do NOT use the word "shape" more than twice per paragraph.

OPPOSITION REPORT HEADER RULE: The Final Score line in all
opposition reports must use the full scoreline format:
"Home Team X-Y Away Team" or "Home Team X-Y Away Team (AET)"
for extra time matches. Never use perspective-relative
descriptions like "Lost 2-1" or "Won 2-1". Use the same
scoreline format across all three generated documents
(tactical report and both opposition reports).

FAR-SIDE SHAPE RULE: If far_side_shape_summary is present in
running_summary with available: true, include far-side fullback
positioning in the formation and shape section of both the
tactical report and opposition reports. Describe the dominant
fullback position observed (high / mid / deep) per team.
This data is only available on broadcast_fixed_wide source type.
Do not fabricate far-side observations for veo_ball_tracking runs.

PRESS TRIGGER RULE: If press_trigger_summary is present in
running_summary with available: true, describe the dominant
press triggers in the pressing and shape sections. Use the
home_top_triggers / away_top_triggers lists to name specific
triggers per team (e.g. "back passes to the goalkeeper",
"fullback receiving in wide areas"). Describe press triggers
as patterns observed across the match, not as isolated incidents.
If a team has no triggers identified (empty top_triggers list),
do not invent them.

RUNNER PATTERN RULE: If runner_pattern_summary is present with
available: true, reference the most_common_run_zones and
most_common_run_types in the transition and set piece sections
where relevant. Unmarked routes (most_common_unmarked_routes) in
particular are worth highlighting as potential vulnerabilities or
strengths. If available: false, do not fabricate runner patterns
-- 1fps structural agents typically don't emit runner data.

PLAYER TENDENCY RULE: If player_tendencies is present in
running_summary with available: true, use it to describe consistent
player patterns in opposition reports. Apply WINDOW LANGUAGE RULE
when stating frequency — never reference "windows" or "X/Y windows"
in any rendered sentence.

Apply these thresholds strictly:
- run_type_confidence: "consistent" (5+ observations) -> assert
  as fact, in match-clock or frequency language:
  "Kymani Thomas consistently dropped deep to receive throughout
   the match rather than running in behind [A]"
- run_type_confidence: "tendency" (3-4 observations) -> note with
  caveat:
  "Thomas frequently dropped into midfield to receive [B]"
- run_type_confidence: "insufficient" (1-2) -> do not report as a
  pattern. Individual note only if highly significant.

ACCUMULATOR AUTHORITY (Fix 54): The accumulator's confidence
classification is the authoritative source for evidence grading
in tendency statements. Do not downgrade based on observation text.

  accumulator "consistent" → always [A] in the report
  accumulator "tendency"   → always [B] in the report
  accumulator "insufficient" → do not report as a pattern

Never write [C] or [D] for an accumulator-classified tendency.
The accumulator has already applied the threshold (5+ observations
for consistent, 3-4 for tendency) — the evidence has been counted.
If the observation text seems mixed, the count supersedes the
impression. A player observed dropping deep 14 times is consistent
regardless of whether any single observation description reads as
uncertain. Quote-hedge ("appeared to", "seemed", "tendency was not
consistent enough") is forbidden when the accumulator says
consistent — state the pattern as fact with [A].

For fullbacks, always note the dominant positioning when
positioning_confidence is consistent or tendency:
"Reid pushed consistently high throughout the first half [A] —
his advanced position created width and left space behind on
defensive transitions"
"Laimer combined defensive discipline with frequent late box
arrivals [B]"

GK DISTRIBUTION RULE: If a goalkeeper has gk_total_kicks >= 5 in
player_tendencies, assert their distribution pattern explicitly in
the opposition report GK profile section.

GK distribution confidence classifications from player_tendencies
are authoritative (Fix 54):
  gk_zone_confidence "consistent"   → [A] in the report
  gk_zone_confidence "tendency"     → [B] in the report
  gk_length_confidence follows the same rule.

Do not downgrade based on a general assessment of the goalkeeper.
The kick count is the evidence. The accumulator already applied
both the absolute threshold (≥5 kicks) and the relative-share
floor (top zone ≥35% of total) before labelling "consistent".

For zone (state the distribution and any in-match consequence):
"Neuer distributed predominantly to the right channel throughout
the match (7 of 11 goal kicks -- left_channel 3, right_channel 7,
central 1) [A]. PSG's first-half pressing shape did not
consistently position a press trigger on his preferred
distribution side; in the second half Mbappe and Dembele
shifted to cover the right channel and Neuer's distribution
broke up on two occasions, producing turnovers in midfield [B]."

For length (state the distribution and any in-match consequence):
"Safonov favoured short distribution from goal kicks (8 of 11
kicks short to the centre-backs) [A]. Bayern engaged the
centre-backs early on five of those eight restarts, forcing
two ball recoveries in the defending third and three resets
back to Safonov [B]."

Do not mention distribution if gk_total_kicks < 5.
Do not guess distribution pattern if not in the data.

Both examples above describe what happened. Neither recommends,
suggests, or instructs. The "in-match consequence" sentence is
factual reporting of what actually occurred against the observed
pattern -- if the opposition did not engage the pattern, say so
plainly ("PSG did not consistently engage Neuer's right side
during the match"). Do not project, do not infer.

OBSERVED PATTERN RULE (Fix 53; replaces the prior EXPLOITABLE PATTERN
RULE): After identifying a consistent player tendency, state the
observation factually with its evidence grade. If the pattern was
exploited or attacked during the match, describe what actually
happened. Do NOT recommend, suggest, or instruct. The coach reads
the observation and decides what to do with it.

Formula: [observation with grade] -> [in-match consequence if any]
Examples:
"Reid maintained high right-back positioning throughout the first
half [A]. When possession turned over quickly, the space behind
him was briefly unoccupied before the back four recovered."

"PSG's fullbacks held mid positioning consistently when defending
[A], producing a compact shape with limited recovery distance to
cover. Bayern attempted no sustained wide overloads against this
structure during the match [B]."

"Kymani Thomas dropped deep consistently to receive [A], pulling
the Brentwood defensive line forward. The space he vacated was
attacked by midfield runs on three occasions, producing one shot
and two cleared deliveries."

If no consistent pattern exists, state that plainly:
"[Player] maintained positional discipline throughout [A]; no
recurring pattern observed."

This rule applies to every opposition report. Never include
"Recommendation:" lines. Never use imperative coaching instructions.

WINDOW LANGUAGE RULE (Fix 53): Never reference "windows", "agent
windows", "observation windows", or any pipeline-internal time-slicing
concept in any report. A coach reading this report does not know what
a window is and the term undermines credibility.

Use match clock and frequency language instead:
- "throughout the match" (not "in 20/20 windows")
- "consistently" or "in the majority of phases" (not "in 14/20 windows")
- "frequently" (not "in 8/12 windows")
- "occasionally" or "in isolated phases" (not "in 3/20 windows")
- "throughout the first half" (not "in 9/10 1H windows")
- "in the second half" (not "in 2H windows")
- "after the substitutions at 75 minutes" (use match clock)

Evidence grades [A/B/C/D] carry the confidence signal. Window counts
are internal pipeline data and must never appear in reports.

For numeric data that DOES belong in reports:
- Goalkeeper kicks: "17 of 21 kicks to the right channel" is fine —
  the unit is kicks, not windows.
- Pass counts, shots, set pieces: cite the action count if useful.
The ban applies only to "windows" / "1H windows" / "2H windows" etc.

NO SUGGESTIONS RULE (Fix 53): Reports describe what was observed.
They do not suggest tactics, make recommendations, or tell the
coaching staff what to do. The coach draws their own conclusions
from the observations.

PROHIBITED language — never use in any report:
- "Recommendation:" (in any form)
- "Teams should..." / "Opposition teams should..."
- "To exploit this..." / "This can be exploited by..."
- "Press his..." / "Force play..." (imperative coaching instructions)
- "A ball in behind would be effective"
- Any sentence beginning with an imperative verb directed at the reader

CORRECT form: State the observation plainly with its evidence grade.
If something was exploited during the match, describe what happened
factually.

WRONG: "Neuer's right-channel preference creates a predictable
        pattern — press his distribution to force play back."
RIGHT: "Neuer distributed predominantly to the right channel
        throughout the match [A]. This created a consistent
        first-ball pattern that PSG were unable to disrupt."

WRONG: "Both fullbacks leave space behind them — a ball in behind
        would be effective on the transition."
RIGHT: "Both fullbacks maintained high positioning throughout the
        first half [A]. When possession turned over quickly, the
        space behind them was briefly unoccupied before recovery."

The Exploitable Patterns section in advanced reports is renamed
Observed Patterns. Each pattern states what was seen and any
instances where it was exploited during the match. No
recommendation line. The coach decides what to do with the
observation.

SET PIECE GOAL RULE: Before attributing any goal to a set piece
(free kick, corner, penalty), verify it against the confirmed
goals list in match_config. If a goal in match_config has no
set_piece field or has set_piece: false, it must be described
as an open play goal regardless of what structural agents logged.
Structural agents may log set piece activity near a goal minute
that is unrelated to the goal itself.

PATTERN THRESHOLD: 3+ observations across different windows = pattern statement.
Below 3 = single-observation note with explicit caveat.

PROHIBITED LANGUAGE:
  - No prescriptive statements. Describe what was observed. The coaching
    staff draw their own conclusions.
    CORRECT:   "Their left side was consistently exposed in transition."
    INCORRECT: "Target their left side in transition."
  - No window identifiers (W01, W02 etc). Use match time and phase.
  - No cross-match references. This match only.
  - Do not repeat source limitation caveats. State them ONCE in the data
    quality note and nowhere else.
  - Do not pad. If data does not support a section, state that clearly
    and move on.

=== SECTION 3: DOCUMENT SPECIFICATIONS ===

--- DOCUMENT 1: tactical_report.md ---

# Match Header
Teams, score, date, competition. Confirmed lineups with shirt numbers
and positions. Substitutions with minutes.

# Match Overview
Two to three paragraphs. What each team set up to do, how those approaches
interacted, what the decisive factors were. Evidence graded A-D throughout.
This is the tactical picture -- not a results summary. The coaching staff
already know the score.

# Key Moments
Drawn from ground_truth events and flagged_moments combined.
Each moment:
  - Match time
  - Event type (goal / card / substitution / tactical shift / significant chance)
  - What happened -- factually, from confirmed data
  - Tactical context at that point -- graded A-D from accumulated window data
  - Match impact -- how it shifted the phase picture. If it did not change
    the tactical picture materially, state that.
Goals and cards are confirmed from match facts. Tactical context around
them is graded from what agents observed. Make this distinction visible.

# Match Phases
Natural phases defined by events AND substantive tactical shifts across
multiple consecutive windows. A single-window anomaly is not a phase break.
Minimum two phases. Each phase covers both teams:
  - Shape and structure
  - How possession moved and where it broke down
  - Pressing behaviour and triggers
  - Transition moments
  - Individual contributions that shaped the phase (name + shirt number)

EQUAL COVERAGE RULE: Every phase must describe both teams with equal analytical
depth. The losing team is not just a backdrop to the winning team's performance.
Describe what the losing team was trying to do with the ball, how they tried to
create, where their build-up was targeted, and what their tactical intentions
were -- not just what happened to them. A team that lost 3-0 still had a tactical
approach; describe it.

# Goals Scored (per team)
Each goal: build-up chain, shot origin zone, foot, target zone, outcome,
contributing pattern. Use event agent data where evidence grade is A.
Fall back to match facts description where event agent data is unavailable.

# Set Pieces
Attacking and defending for both teams. Delivery type, target zones,
aerial threats, marking system. Evidence grade stated per finding.

# Tactical Variation and Substitutions
What changed when substitutions were made. Whether they were tactical
adjustments or personnel rotations. Evidence graded A-D.

# Performance Metrics Summary
METRICS WARNING: All pipeline metrics (pressing intensity, line height, possession
balance, sequence counts, passing routes) are MATCH-LEVEL figures derived from
both teams' combined activity unless explicitly labelled per-team in the data.
Do NOT attribute a match-level metric to one team as if it describes only that
team. When writing opposition reports, either omit match-level metrics or
clearly frame them as match context (e.g. "match pressing averaged 4.37/10
across both teams"). The 10-distinct-passing-routes figure is a Gorleston
structural finding -- never attribute it to Tilbury or any opposition without
explicit per-team data confirming it.

SUPPRESSED METRICS: Any metric labelled _suppressed_metrics in the
input data has been pre-filtered as unreliable. Do not reference
suppressed metrics in reports. Do not state that data is unavailable
unless directly relevant to a coaching question.

Cover all 17 metrics in prose. For each metric:
  - If confidence is sufficient AND relevant to this match: full interpretation
  - If confidence is low BUT directly relevant: include with confidence stated
  - If confidence is low AND not relevant to what happened: omit
Do not present metrics as a data table. Prose only.
Do not say "confidence of 0.22" -- translate: "pressing intensity readings
are directional given the ball-tracking source -- the pattern is consistent
but the precise figure should be treated as approximate."

# Data Quality Note
ONE paragraph. Source type, what it means for the findings, which specific
observations are limited. State this ONCE. Do not repeat elsewhere.

--- DOCUMENT 2 and 3: opposition_report_[team].md ---

One document per team. Independent. Does not reference the opposing team
by name in analytical sections -- only in match context where necessary.

## Header
Team name, date, competition, final score, observed formation.

## Overview
The agent determines appropriate length from data richness. Covers:
tactical identity as observed in this match, what they set up to do,
how they executed it, where they succeeded and where they broke down.
Evidence graded A-D. This is the paragraph a manager reads before a session.

## Formation and Shape

### In Possession
Observed formation, build-up pattern, ball progression, width usage,
halfspace occupation, tempo.

### Out of Possession
Defensive shape, block type, press triggers, line height, compactness.

### Transitions
Speed of phase shift, directness of attacking transition, defensive
recovery behaviour.

## Set Piece Analysis

### Attacking Set Pieces
Delivery type and target zones, primary aerial threats, movement patterns.
Evidence grade per finding.

### Defending Set Pieces
GK behaviour (does the GK come for deliveries or stay on line),
marking system, organisation, any observed vulnerabilities -- stated
as findings, not prescriptions.

## Player Profiles

Grouped by position: GK, defensive line, midfield, attack, substitutes.

Position attribute sets (score only where observations exist):

  GK:            shot-stopping, aerial command, distribution-short,
                 distribution-long, footwork, composure, leadership
  Centre-backs:  aerial-duels, defensive-positioning, 1v1-defending,
                 ball-progression, passing-under-pressure, pace, leadership
  Fullbacks:     1v1-defending, recovery-pace, attacking-output, crossing,
                 defensive-tracking, positional-discipline
  CM (defensive): screening, aerial-duels, tackle-success, passing-range,
                 mobility, press-resistance
  CM (box-to-box): defensive-work-rate, forward-runs, passing-accuracy,
                 tempo, press-contribution
  AM:            touch, passing, movement-between-lines, press-contribution,
                 chance-creation, speed-of-play
  Wide forwards: pace, direct-dribbling, crossing, defensive-tracking,
                 final-ball-quality, press-contribution
  Strikers:      aerial-duels, hold-up-play, link-play, pressing-body-shape,
                 movement, pace, finishing

Each player with sufficient evidence (at least one attribute with observation):
  - Name, shirt number, position -- MUST use the exact pos from the roster block above.
    Never infer or rename a position. If the roster says RB, the heading says RB.
  - Attribute ratings A-E (listed only where observations exist)
  - Prose profile: lead with starting role, then describe observed movement tendencies.
    Did they hold position or get forward? Did a fullback overlap or sit deep?
    Did a midfielder stay disciplined or arrive late in the box? Describe what
    was actually observed -- do not assume default behaviour from the position label.
  - Strengths observed in this match
  - Weaknesses observed in this match
  - Overall profile grade A-D

Substitutes:
  - Sufficient impact: full profile
  - Limited impact: one sentence on what changed tactically
  - No observable impact (E): name, number, appearance minute only

## Key Weaknesses Observed
Ranked by evidence grade, highest first. Numbered. Findings only.
No prescriptive language.

## Key Strengths Observed
Ranked by evidence grade. Numbered. What the team demonstrated capability in.

## Risk Assessment
Markdown table.
Columns: Finding | Evidence Grade | Context observed

=== SECTION 4: LANGUAGE RULES ===

Football language throughout. Precise, technical, unhedged where evidence
supports it.

Do NOT:
  - Use window identifiers (W01, W02)
  - Make prescriptive statements
  - Reference opposing team by name in opposition report analytical sections
  - Reference any player not in match_config for the correct team
  - Write D-grade findings in A-grade language
  - Repeat source limitation caveats beyond the data quality note
  - Pad sections where data is thin -- state it and move on

DO:
  - Connect dots when 3+ observations support a pattern
  - State evidence grade in brackets where clarity requires it
  - Use match time and phase references instead of window numbers
  - Write player profiles as narrative, not bullet aggregation
  - Make the gap between confirmed and inferred visible to the reader

Formatting:
  - Markdown throughout
  - Headers consistent with existing pipeline output style
  - Player attributes as an indented list within each profile
  - Risk assessment as a markdown table
  - Horizontal rules between major sections
"""

# ── Input bundle loader ───────────────────────────────────────────────────

def _load_json(path: str) -> dict | list:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_text(path: str) -> str:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return f.read()
    return ""


# Required files — pipeline cannot produce a useful report without these
REQUIRED_FILES = [
    "match_config.json",
    "running_summary.json",
    "pass_sequences.json",
]

# Optional files — logged as warnings if missing but do not block
OPTIONAL_FILES = [
    "deep_skill_metrics.json",
    "ground_truth_check.json",
    "shots_log.json",
    "flagged_moments.md",
    "report_readiness.json",
]


def _should_suppress_metric(m: dict, context_only_names: set) -> bool:
    """
    Decide whether a single metric entry should be filtered out before
    reaching the synthesis agent. Suppress when:
      - metric_name is in the file's own context_only_metrics list
      - confidence is below 0.15
      - prose_interpretation begins with a zero-rate phrase
      - the nested value dict reports zero distinct routes / zero rate /
        all-None scores
    """
    if not isinstance(m, dict):
        return False

    name = m.get("metric_name") or m.get("name") or ""
    if name in context_only_names:
        return True

    try:
        if float(m.get("confidence", 1.0)) < 0.15:
            return True
    except (ValueError, TypeError):
        pass

    prose = str(m.get("prose_interpretation") or "").strip().lower()
    if prose.startswith(("0%", "0.0%", "0 distinct", "0 of ", "0 sequences")):
        return True

    val = m.get("value")
    if isinstance(val, dict):
        # Route diversity: zero distinct routes
        if val.get("distinct_routes") == 0 and (val.get("score") in (0, 0.0)):
            return True
        # Final-third penetration: 0% rate from a non-trivial sample
        ftr = val.get("final_third_rate")
        try:
            if ftr is not None and float(ftr) == 0.0 \
                    and val.get("total_sequences", 0) > 0:
                return True
        except (ValueError, TypeError):
            pass
        # Transition efficiency: ALL of the transition-specific keys
        # must be present (and degenerate) to suppress. Required keys
        # guard against false positives on unrelated metric shapes.
        if ("attack_transitions" in val and "attack_threat_rate" in val
                and "score" in val
                and val["score"] is None
                and val["attack_threat_rate"] is None
                and val["attack_transitions"] == 0):
            return True
    elif val is not None:
        try:
            if float(str(val).replace("%", "").strip()) == 0.0:
                return True
        except (ValueError, TypeError):
            pass

    return False


def _clean_metrics(metrics):
    """
    Filter zero-value / context-only / very-low-confidence entries from
    the deep_skill_metrics bundle before it reaches the synthesis agent.
    The model never sees the suppressed entries; only a summary note.
    """
    if not isinstance(metrics, dict):
        return metrics

    context_only = set(metrics.get("context_only_metrics") or [])
    severely_lim = set(metrics.get("severely_limited_metrics") or [])
    drop_names   = context_only | severely_lim

    metric_list = metrics.get("metrics")
    if not isinstance(metric_list, list):
        return metrics

    kept, dropped = [], []
    for m in metric_list:
        if _should_suppress_metric(m, drop_names):
            dropped.append(m.get("metric_name") or m.get("name") or "?")
        else:
            kept.append(m)

    cleaned = dict(metrics)  # shallow copy preserves top-level fields
    cleaned["metrics"] = kept
    if dropped:
        cleaned["_suppressed_metrics"] = (
            f"{len(dropped)} metric(s) suppressed (zero value, context-only, "
            f"or low confidence): {', '.join(dropped)}. "
            f"Do not reference these in reports."
        )
    return cleaned


def build_input_bundle(match_dir: str) -> dict:
    """
    Load all accumulated pipeline outputs into a single bundle.
    Raises FileNotFoundError if any required file is missing.
    Logs warnings for missing optional files.
    """
    missing_required = [
        f for f in REQUIRED_FILES
        if not os.path.exists(os.path.join(match_dir, f))
    ]
    if missing_required:
        raise FileNotFoundError(
            f"[3l_synthesis] Cannot proceed — required files missing:\n"
            + "\n".join(f"  {match_dir}\\{f}" for f in missing_required)
        )

    for fname in OPTIONAL_FILES:
        if not os.path.exists(os.path.join(match_dir, fname)):
            print(f"  [WARN] Optional file missing — "
                  f"related findings will be empty: {fname}")

    return {
        "match_config":       _load_json(os.path.join(match_dir, "match_config.json")),
        "window_plan":        _load_json(os.path.join(match_dir, "window_plan.json")),
        "running_summary":    _load_json(os.path.join(match_dir, "running_summary.json")),
        "pass_sequences":     _load_json(os.path.join(match_dir, "pass_sequences.json")),
        "deep_skill_metrics": _clean_metrics(
            _load_json(os.path.join(match_dir, "deep_skill_metrics.json"))),
        "ground_truth_check": _load_json(os.path.join(match_dir, "ground_truth_check.json")),
        "shots_log":          _load_json(os.path.join(match_dir, "shots_log.json")),
        "flagged_moments":    _load_text(os.path.join(match_dir, "flagged_moments.md")),
        "report_readiness":   _load_json(os.path.join(match_dir, "report_readiness.json")),
    }


def _summarise_bundle(bundle: dict) -> str:
    """
    Produce a compact JSON representation of the input bundle for the prompt.
    Truncates very large arrays to avoid token limits while preserving structure.
    """
    def _trim(obj, max_list=80):
        if isinstance(obj, list):
            if len(obj) > max_list:
                return obj[:max_list] + [f"... ({len(obj) - max_list} more items truncated)"]
            return [_trim(i, max_list) for i in obj]
        if isinstance(obj, dict):
            return {k: _trim(v, max_list) for k, v in obj.items()}
        return obj

    trimmed = _trim(bundle)
    return json.dumps(trimmed, indent=2, ensure_ascii=False)


# ── Roster block builder ──────────────────────────────────────────────────

def build_roster_block(mc: dict) -> str:
    """Build the confirmed lineup block injected into every synthesis prompt."""
    lines = [f"CONFIRMED LINEUPS FOR THIS MATCH\n{'='*40}"]
    lines.append("CRITICAL: Each player is labelled STARTER or SUBSTITUTE.")
    lines.append("A STARTER played from kick-off. A SUBSTITUTE did not start.")
    lines.append("Never describe a STARTER as a substitute or impact sub.")
    lines.append("Never describe a SUBSTITUTE as starting or playing from kick-off.")
    lines.append("")
    lines.append("THIS ROSTER BLOCK IS THE AUTHORITATIVE SOURCE FOR:")
    lines.append("  - Whether a player started or came on as a substitute")
    lines.append("  - Each player's confirmed starting position (GK, RB, CB, LB, RM, CM, LM, AM, ST)")
    lines.append("")
    lines.append("Event data and build-up chains describe observed roles in sequences,")
    lines.append("NOT starting positions. If a CB appears in the box to score, they")
    lines.append("are a CB making a forward run -- not a striker, not a substitute.")
    lines.append("Never override this roster block based on where a player appeared")
    lines.append("in a build-up chain or event description.")
    lines.append("")

    for lineup in mc.get("lineups", []):
        t = lineup.get("team", {})
        team_name = t.get("name", str(t)) if isinstance(t, dict) else str(t)
        lines.append(f"{team_name}:")
        lines.append("  STARTERS:")
        for p in lineup.get("startXI", []):
            pd = p.get("player", p)
            if isinstance(pd, dict):
                lines.append(f"    [STARTER] #{pd.get('number','?')} {pd.get('name','?')} "
                             f"({pd.get('pos', pd.get('position',''))})")
        subs = lineup.get("substitutes", [])
        if subs:
            lines.append("  SUBSTITUTES (did not start):")
            for p in subs:
                pd = p.get("player", p)
                if isinstance(pd, dict):
                    lines.append(f"    [SUBSTITUTE] #{pd.get('number','?')} {pd.get('name','?')}")
        lines.append("")

    lines.append(f"{'='*40}")
    lines.append("RULE: A player name may only appear in a report if it appears")
    lines.append("in the correct team's lineup above. No exceptions.")
    return "\n".join(lines)


# ── API call ──────────────────────────────────────────────────────────────

def _call_synthesis(system_prompt: str, user_content: str,
                    max_tokens: int = 8000) -> str:
    """Single API call for synthesis. Returns full text response."""
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(
        block.text for block in response.content
        if hasattr(block, "text")
    )


# ── Document writers ──────────────────────────────────────────────────────

def _write_tactical_report(match_dir: str, bundle: dict) -> str:
    """Generate and write tactical_report.md via synthesis agent."""
    mc         = bundle["match_config"]
    home_team  = mc.get("home_team", "Home")
    away_team  = mc.get("away_team", "Away")
    roster     = build_roster_block(mc)
    data_str   = _summarise_bundle(bundle)

    user_msg = f"""
{roster}

You are generating DOCUMENT 1 -- tactical_report.md

Match: {home_team} vs {away_team}

Here is the complete accumulated pipeline data for this match:

{data_str}

Produce the full tactical report now. Start with ROSTER CHECK: PASSED then
begin the report. Follow the document specification exactly.
"""
    print("  Calling synthesis API -- tactical report...", flush=True)
    result = _call_synthesis(SYNTHESIS_PROMPT, user_msg, max_tokens=8000)

    out_path = os.path.join(match_dir, "tactical_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  tactical_report.md written ({len(result):,} chars)")
    return result


def _write_opposition_report(match_dir: str, bundle: dict,
                              team_side: str) -> str:
    """Generate and write opposition_report_[team].md via synthesis agent."""
    mc        = bundle["match_config"]
    home_team = mc.get("home_team", "Home")
    away_team = mc.get("away_team", "Away")
    team_name = home_team if team_side == "home" else away_team
    roster    = build_roster_block(mc)
    data_str  = _summarise_bundle(bundle)

    user_msg = f"""
{roster}

You are generating an OPPOSITION REPORT for: {team_name} ({team_side} team)

Match: {home_team} vs {away_team}

This document profiles {team_name} in isolation. Do not reference the
opposing team by name in analytical sections -- only in match context
(scoreline, numerical situations).

Here is the complete accumulated pipeline data for this match:

{data_str}

Produce the full opposition report now. Start with ROSTER CHECK: PASSED then
begin the report. Follow the document specification exactly.
"""
    safe_name = re.sub(r"[^\w\-]", "_", team_name.lower())
    fname     = f"opposition_report_{safe_name}.md"

    print(f"  Calling synthesis API -- opposition report ({team_name})...",
          flush=True)
    result = _call_synthesis(SYNTHESIS_PROMPT, user_msg, max_tokens=8000)

    out_path = os.path.join(match_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  {fname} written ({len(result):,} chars)")
    return result


# ── Advanced synthesis tier (Fix 51) ──────────────────────────────────────

ADVANCED_SYNTHESIS_ADDITIONS = """
=== ADVANCED REPORT ADDITIONS ===

For ADVANCED quality reports, add the following sections to each document.
These are IN ADDITION to all standard sections, not replacements. Place
them where indicated. Do NOT remove or compress any standard section.

1. PLAYER TENDENCY TABLE (opposition reports only)
After the Player Profiles section, add a markdown table titled
"### Player Tendency Table". Columns:

| Player | Pos | Observed | Dominant Position | Run Type | Defensive |
|---|---|---:|---|---|---|

Populate from player_tendencies in running_summary. Include only
players with windows_appeared >= 3. Mark confidence:
  * = tendency (3-4 obs)
  ** = consistent (5+)
  blank = below threshold (omit row)
If a tendency field is null/insufficient, leave the cell blank.

2. GK DISTRIBUTION TABLE (both reports)
In each GK player profile, append a markdown table after the prose:

| Zone | Kicks | % |
|---|---:|---:|

Use gk_zone_distribution from player_tendencies. Sort descending by
kick count. Conclude with: "Total kicks: N | Dominant: <zone>
(<confidence>) | Length preference: <length> (<confidence>)".

3. PER-HALF FORMATION SUMMARY (tactical report only)
After Match Overview and before Key Moments, insert a section
titled "## Per-Half Formation Summary" with two sub-blocks:

**First Half**
- Home: <formation> — <shape> block
- Away: <formation> — <shape> block
- Notable: <any phase change observed>

**Second Half**
- Home: <formation> — <shape> block (note any change from 1H)
- Away: <formation> — <shape> block (note any change from 1H)
- Notable: <IP/OOP variations from formation_history.home_variation
  / away_variation>

Compute by splitting formation_history entries on the half field
("1H" / "2H" / "ET1" / "ET2") and majority-voting per half.

4. FULLBACK POSITIONING SUMMARY (both opposition reports)
After Formation and Shape, add a paragraph titled
"### Fullback Positioning Pattern":

Use far_side_shape_summary.home_fullback_distribution and
away_fullback_distribution. Quote concrete counts:
  "<Team> fullbacks held <position> positioning in N of M observed
   windows; secondary distribution: <other position> in K windows."
Then one sentence on the exploitable consequence if relevant.

5. OBSERVED PATTERNS SECTION (opposition reports only)
At the end of the report, BEFORE the Risk Assessment table, add:

## Observed Patterns

List 3–5 patterns observed across the match. Each one:
  **<Pattern name>** [<grade>]
  <One or two sentences stating what was observed and any
   in-match instances where the pattern was attacked or
   defended successfully. Use match-clock and frequency
   language; never reference "windows" or "X/Y windows".>

NO recommendation lines. NO imperative coaching instructions
("press his...", "force play...", "to exploit"). The coach
reads the observation and decides what to do with it.

Example:
  **GK Distribution Bias** [A]
  Neuer distributed predominantly to the right channel
  throughout the match (17 of 21 kicks). PSG attempted
  no sustained press of this side and Bayern's right-channel
  build-up went largely uncontested in the first half.

If fewer than 3 patterns clear the [B] threshold, add the explicit
note "Additional patterns below the [B] threshold; not enough
evidence to state with confidence." rather than padding.

6. TACTICAL NARRATIVE ARC (tactical report only)
After Goals Scored, insert a section titled
"## Tactical Narrative Arc" — 3-4 paragraphs that read like a
tactical explainer rather than a match report. Cover:
  - How each team adapted across the match
  - Which tactical adjustments worked, which did not
  - The decisive tactical factor
  - What both managers would take away

This is more analytical than the standard Match Overview. Use
formation_history, key_moments, and any IP/OOP variation notes
to anchor the narrative across phases.

=== EVIDENCE COVERAGE RULE FOR ADVANCED ===

For every [A] finding in this advanced report, ground the claim in
concrete data from running_summary. Cite action counts (kicks,
passes, set pieces) or match-clock evidence inline where useful —
e.g. "(17 of 21 kicks to the right channel)" or "throughout the
first half" or "after the 75th-minute substitution".

Do NOT cite "windows" or "X/Y windows" — this term is internal
pipeline data and breaks credibility (see WINDOW LANGUAGE RULE).
Use match clock and frequency language instead.

=== TABLE FORMATTING RULE ===

All tables must be valid GitHub-flavoured markdown. Keep column
widths reasonable; do not pad with whitespace. If a table row
would have all blank cells, omit the row.
"""


def _write_advanced_tactical_report(match_dir: str, bundle: dict) -> str:
    """Generate the ADVANCED tactical_report — saves to advanced_tactical_report.md."""
    mc        = bundle["match_config"]
    home_team = mc.get("home_team", "Home")
    away_team = mc.get("away_team", "Away")
    roster    = build_roster_block(mc)
    data_str  = _summarise_bundle(bundle)

    user_msg = f"""
{roster}

You are generating DOCUMENT 1 -- ADVANCED tactical_report.md.

Match: {home_team} vs {away_team}

Standard report content + advanced additions per the ADVANCED REPORT
ADDITIONS section in the system prompt. Both must be present; do not
choose between them.

Here is the complete accumulated pipeline data for this match:

{data_str}

{ADVANCED_SYNTHESIS_ADDITIONS}

Produce the full ADVANCED tactical report now. Start with
ROSTER CHECK: PASSED then begin the report. Follow the document
specification AND the advanced additions above.
"""
    print("  Calling synthesis API -- ADVANCED tactical report...", flush=True)
    result = _call_synthesis(SYNTHESIS_PROMPT, user_msg, max_tokens=12000)

    out_path = os.path.join(match_dir, "advanced_tactical_report.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  advanced_tactical_report.md written ({len(result):,} chars)")
    return out_path


def _write_advanced_opposition_report(match_dir: str, bundle: dict,
                                       team_side: str) -> str:
    """Generate ADVANCED opposition_report — saves to advanced_opposition_report_*.md."""
    mc        = bundle["match_config"]
    home_team = mc.get("home_team", "Home")
    away_team = mc.get("away_team", "Away")
    team_name = home_team if team_side == "home" else away_team
    roster    = build_roster_block(mc)
    data_str  = _summarise_bundle(bundle)

    user_msg = f"""
{roster}

You are generating an ADVANCED OPPOSITION REPORT for: {team_name} ({team_side} team)

Match: {home_team} vs {away_team}

This document profiles {team_name} in isolation. Do not reference the
opposing team by name in analytical sections -- only in match context
(scoreline, numerical situations).

Standard opposition report content + advanced additions per the
ADVANCED REPORT ADDITIONS section. Both must be present.

Here is the complete accumulated pipeline data for this match:

{data_str}

{ADVANCED_SYNTHESIS_ADDITIONS}

Produce the full ADVANCED opposition report now. Start with
ROSTER CHECK: PASSED then begin the report.
"""
    safe_name = re.sub(r"[^\w\-]", "_", team_name.lower())
    fname     = f"advanced_opposition_report_{safe_name}.md"

    print(f"  Calling synthesis API -- ADVANCED opposition ({team_name})...",
          flush=True)
    result = _call_synthesis(SYNTHESIS_PROMPT, user_msg, max_tokens=12000)

    out_path = os.path.join(match_dir, fname)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  {fname} written ({len(result):,} chars)")
    return out_path


def synthesise_advanced(match_dir: str) -> dict:
    """
    Produce three ADVANCED reports alongside the existing standard reports.
    Reads the same running_summary / match_config / report_readiness as the
    standard synthesis path; uses an enhanced prompt requesting additional
    analytical depth (per-player tendency tables, GK distribution table,
    per-half formation breakdown, fullback positioning paragraph,
    exploitable-patterns section, tactical narrative arc).

    Saves to:
        advanced_tactical_report.md
        advanced_opposition_report_<home_slug>.md
        advanced_opposition_report_<away_slug>.md

    Does NOT overwrite the standard reports. Returns the three paths.
    """
    print(f"\n[3l_synthesis_advanced] Starting advanced synthesis for:\n  {match_dir}\n")

    bundle = build_input_bundle(match_dir)

    tactical_path  = _write_advanced_tactical_report(match_dir, bundle)
    home_opp_path  = _write_advanced_opposition_report(match_dir, bundle, "home")
    away_opp_path  = _write_advanced_opposition_report(match_dir, bundle, "away")

    print("\n[3l_synthesis_advanced] Complete.")
    return {
        "tactical": tactical_path,
        "home_opp": home_opp_path,
        "away_opp": away_opp_path,
    }


# ── Main entry point ──────────────────────────────────────────────────────

def run_synthesis(match_dir: str) -> dict:
    """
    Pipeline entry point. Called from pipeline_runner_v2.py as:
        ("3l_synthesis", "synthesis_agent", "run_synthesis")

    Returns a summary dict for the pipeline state.
    """
    print(f"\n[3l_synthesis] Starting synthesis agent for:\n  {match_dir}\n")

    try:
        bundle = build_input_bundle(match_dir)
    except FileNotFoundError as e:
        print(str(e))
        return {"status": "failed", "reason": "missing_required_files"}
    mc     = bundle["match_config"]

    if not mc:
        print("  ERROR: match_config.json not found or empty. Cannot proceed.")
        return {"status": "failed", "reason": "no match_config"}

    home_team = mc.get("home_team", "Home")
    away_team = mc.get("away_team", "Away")

    results = {}

    # Document 1 -- tactical report
    try:
        results["tactical_report"] = _write_tactical_report(match_dir, bundle)
        results["tactical_status"] = "complete"
    except Exception as e:
        print(f"  ERROR generating tactical report: {e}")
        results["tactical_status"] = "failed"
        results["tactical_error"]  = str(e)

    time.sleep(2)  # brief pause between calls

    # Document 2 -- away team opposition report
    try:
        results["opposition_away"] = _write_opposition_report(
            match_dir, bundle, "away"
        )
        results["opposition_away_status"] = "complete"
    except Exception as e:
        print(f"  ERROR generating away opposition report: {e}")
        results["opposition_away_status"] = "failed"
        results["opposition_away_error"]  = str(e)

    time.sleep(2)

    # Document 3 -- home team opposition report
    try:
        results["opposition_home"] = _write_opposition_report(
            match_dir, bundle, "home"
        )
        results["opposition_home_status"] = "complete"
    except Exception as e:
        print(f"  ERROR generating home opposition report: {e}")
        results["opposition_home_status"] = "failed"
        results["opposition_home_error"]  = str(e)

    # Summary
    statuses = [
        results.get("tactical_status"),
        results.get("opposition_away_status"),
        results.get("opposition_home_status"),
    ]
    all_complete = all(s == "complete" for s in statuses)

    print(f"\n[3l_synthesis] Complete.")
    print(f"  Tactical report:    {results.get('tactical_status')}")
    print(f"  Away opposition:    {results.get('opposition_away_status')}")
    print(f"  Home opposition:    {results.get('opposition_home_status')}")

    return {
        "status":      "complete" if all_complete else "partial",
        "tactical":    results.get("tactical_status"),
        "opp_away":    results.get("opposition_away_status"),
        "opp_home":    results.get("opposition_home_status"),
    }


# ── Standalone run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python synthesis_agent.py <match_dir>")
        sys.exit(1)
    result = run_synthesis(sys.argv[1])
    print(f"\nFinal status: {result['status']}")
