"""
Generate pass_network.md for a Match Lens match directory.

Usage:
    python generate_pass_network.py <match_dir>

Requires: synthesis_agent.py in match-analysis/scripts (for _call_synthesis).

What it produces:
  - Summary table (total seqs, avg length, outcome rates)
  - Zone Flow per team
  - Sequence Outcomes per team
  - Pattern Analysis (shot-ending, set-piece-winning, build-up routes)
  - Note: Player Pairs section omitted — pass_sequences.json does not
    carry player-level data for these matches.

1H/2H split is omitted if window data is not available.
"""

import json
import os
import sys
from collections import Counter, defaultdict

# SCRIPTS_DIR self-resolves so the script runs from any cwd / any machine.
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Exemplar style template — used to seed the LLM with the report's format.
# Lives outside the repo (see .gitignore) since real match exemplars contain
# real team/player data. Override the directory via env var MATCH_LENS_EXEMPLAR_DIR.
EXEMPLAR_DIR = os.environ.get(
    "MATCH_LENS_EXEMPLAR_DIR",
    os.path.join(SCRIPTS_DIR, "exemplars"),
)
EXEMPLAR = os.path.join(EXEMPLAR_DIR, "pass_network.md")

sys.path.insert(0, SCRIPTS_DIR)
from pipeline_accessors import get_source_limitations_note
from synthesis_agent import _call_synthesis


# ── Pre-compute stats from pass_sequences.json ────────────────────────────

def compute_stats(sequences: list, home_team: str, away_team: str) -> dict:
    """
    Derive structured stats from raw pass_sequences.json entries.
    Maps 'home_kit' / 'away_kit' to actual team names.
    """
    kit_map = {"home_kit": home_team, "away_kit": away_team}

    per_team = {home_team: [], away_team: []}
    for s in sequences:
        team = kit_map.get(s.get("team", ""), "unknown")
        if team in per_team:
            per_team[team].append(s)

    def team_stats(seqs):
        if not seqs:
            return {}
        outcomes    = Counter(s.get("outcome", "unknown") for s in seqs)
        seq_types   = Counter(s.get("sequence_type", "unknown") for s in seqs)
        seq_confidence = Counter(s.get("sequence_confidence", "unknown") for s in seqs)
        start_zones = Counter(s.get("start_zone", "unknown") for s in seqs)
        end_zones   = Counter(s.get("end_zone", "unknown") for s in seqs)
        lengths     = [s.get("passes", 0) for s in seqs if isinstance(s.get("passes"), (int, float))]
        avg_len     = round(sum(lengths) / len(lengths), 1) if lengths else 0

        # Zone flow: group start->end pairs
        zone_flow = Counter(
            (s.get("start_zone", "?"), s.get("end_zone", "?")) for s in seqs
        )

        # Progressive: sequences that advance into or through the middle / attacking third
        FORWARD_ENDS = {"middle_third", "attacking_third", "left_channel", "right_channel",
                        "attacking_left", "attacking_right", "attacking_centre"}
        BACK_STARTS  = {"defending_third", "defensive_left", "defensive_right",
                        "defensive_centre", "goalkeeper"}
        progressive  = [s for s in seqs
                        if s.get("end_zone", "") in FORWARD_ENDS
                        or s.get("start_zone", "") in BACK_STARTS]

        # Half split — only if at least one sequence has a non-unknown window
        windows = [s.get("window") for s in seqs if s.get("window") not in (None, "unknown", "")]
        has_half_data = len(windows) > 0

        half_1 = [s for s in seqs if str(s.get("window", "")).startswith(("1H", "01", "02", "03", "04", "05",
                                                                            "06", "07", "08", "09", "10", "11"))]
        half_2 = [s for s in seqs if str(s.get("window", "")).startswith(("2H", "12", "13", "14", "15", "16",
                                                                            "17", "18", "19", "20"))]

        return {
            "total":           len(seqs),
            "avg_length":      avg_len,
            "outcomes":        dict(outcomes.most_common()),
            "sequence_types":  dict(seq_types.most_common()),
            "sequence_confidence_distribution": dict(seq_confidence.most_common()),
            "start_zones":     {k: round(v * 100 / len(seqs)) for k, v in start_zones.most_common()},
            "end_zones":       {k: round(v * 100 / len(seqs)) for k, v in end_zones.most_common()},
            "zone_flow_top10": [{"from": f, "to": t, "count": c}
                                for (f, t), c in zone_flow.most_common(10)],
            "progressive_count": len(progressive),
            "shot_sequences":  [{"start": s.get("start_zone"), "end": s.get("end_zone"),
                                  "passes": s.get("passes"), "type": s.get("sequence_type")}
                                 for s in seqs if s.get("outcome") in ("shot", "goal")],
            "set_piece_sequences": [{"start": s.get("start_zone"), "end": s.get("end_zone"),
                                      "passes": s.get("passes")}
                                     for s in seqs if s.get("outcome") in ("set_piece_won", "set_piece",
                                                                             "foul_won", "free_kick")],
            "has_half_data":   has_half_data,
            "half_1_count":    len(half_1) if has_half_data else None,
            "half_2_count":    len(half_2) if has_half_data else None,
        }

    return {
        home_team: team_stats(per_team[home_team]),
        away_team:  team_stats(per_team[away_team]),
        "total":    len(sequences),
        "home_team": home_team,
        "away_team": away_team,
    }


# ── Main ──────────────────────────────────────────────────────────────────

def generate_pass_network(match_dir: str) -> str:
    # Load inputs
    mc_path = os.path.join(match_dir, "match_config.json")
    ps_path = os.path.join(match_dir, "pass_sequences.json")
    sp_path = os.path.join(match_dir, "source_profile.json")

    with open(mc_path, encoding="utf-8") as f:
        mc = json.load(f)
    with open(ps_path, encoding="utf-8") as f:
        ps = json.load(f)

    home_team = mc.get("home_team", "Home")
    away_team = mc.get("away_team", "Away")
    date      = mc.get("date", "")
    comp      = mc.get("competition", "")

    source_type = "broadcast_fixed_wide"
    source_note = ""
    if os.path.exists(sp_path):
        with open(sp_path, encoding="utf-8") as f:
            sp = json.load(f)
        source_type = sp.get("source_type", source_type)
        source_note = get_source_limitations_note(sp)

    sequences = ps.get("sequences", [])
    stats     = compute_stats(sequences, home_team, away_team)
    stats_str = json.dumps(stats, indent=2)

    # Warn-and-continue if exemplar is missing.
    if os.path.exists(EXEMPLAR):
        with open(EXEMPLAR, encoding="utf-8") as f:
            exemplar = f.read()
    else:
        print(f"  [WARN] Exemplar not found at {EXEMPLAR}")
        print(f"  [WARN] Set MATCH_LENS_EXEMPLAR_DIR or place pass_network.md "
              f"in {EXEMPLAR_DIR}")
        print(f"  [WARN] Proceeding without exemplar — report style may suffer")
        exemplar = ""

    system_prompt = f"""
You are writing pass_network.md for a football match analysed by the
Match Lens pipeline. Follow every evidence and language rule below.

EVIDENCE RULES:
- All statistics come directly from pass_sequences.json data provided.
  The pre-computed stats block is authoritative — do not fabricate counts.
- Source type: {source_type}. Near-ball bias applies — sequences reflect
  near-ball chains, not total pass volume across the pitch.
- Confidence note: sequence zone data is reliable; player-level data is NOT
  available in pass_sequences.json for this match (omit Player Pairs section).
- If has_half_data is False, omit the "First Half vs Second Half" section
  entirely and replace with a single sentence noting window data was not
  recorded per-half for this match.

WINDOW LANGUAGE RULE: Never reference "windows", "observation windows",
"X/Y windows", or any pipeline-internal concept. Use match-clock language.

NO SUGGESTIONS RULE: Describe patterns observed. Do not tell coaching staff
what to do. No "Recommendation:", "teams should", "press his", imperative
coaching instructions, or "a ball in behind would be effective".

FORMAT: Match the Gorleston exemplar structure exactly where data supports it.
Sections to produce (in order):
  1. Header (match, date, source note)
  2. Summary table (per team: total seqs, avg length, progressive %, outcome rates)
  3. Zone Flow (start % and end % per team)
  4. Sequence Outcomes (raw counts per team)
  5. Pattern Analysis (shot-ending sequences, build-up progressions, notable routes)
  6. First Half vs Second Half — ONLY if has_half_data is True; otherwise one-line note
  7. Footer attribution line

OMIT the "Pass Network — Top Player Pairs" section entirely. Add a one-line
note after Sequence Outcomes: "Player pair data not available for this match —
pass_sequences.json does not carry player-level sequence attribution."

PATTERN ANALYSIS TIMESTAMP RULE: Do NOT include per-sequence timestamps
in the Pattern Analysis section. The pre-computed stats block fed below
does not contain per-sequence timestamps. List patterns by origin zone,
end zone, count, and average length — never by minute marker. If you
catch yourself writing "11m10s" or "74m54s", the timestamp is
fabricated; remove it and aggregate the sequences by route instead.

Output: pure markdown. No preamble before the first heading.
"""

    user_msg = f"""
Match: {home_team} vs {away_team}
Date: {date}
Competition: {comp}
Source: {source_type}
Source limitations: {source_note or "Near-ball visibility only. Sequences represent observed chains, not complete pass map."}

PRE-COMPUTED STATISTICS (authoritative — derived directly from pass_sequences.json):

{stats_str}

EXEMPLAR (Gorleston pass_network.md — match this format, adapt for available data):

{exemplar}

Produce pass_network.md for {home_team} vs {away_team} now.
Use only the statistics provided above. Do not invent counts.
Omit Player Pairs section (no data). Apply all language rules.
No preamble before the first heading.
"""

    print(f"  Generating pass_network.md for {home_team} vs {away_team}...", flush=True)
    result = _call_synthesis(system_prompt, user_msg, max_tokens=4000)

    out_path = os.path.join(match_dir, "pass_network.md")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result)
    print(f"  Written: {out_path} ({len(result):,} chars)")
    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_pass_network.py <match_dir>")
        sys.exit(1)
    generate_pass_network(sys.argv[1])
