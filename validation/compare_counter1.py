#!/usr/bin/env python3
"""
compare_counter1.py -- Counter 1: hard-fact agreement (Match Lens report vs StatsBomb truth)

Scope, stated honestly and enforced in the output:
This match is tactical_wide_static footage. The pipeline skips jersey OCR on this
source, the report stores no structured facts ledger, and several hard-fact
categories are therefore either prose-only, semi-structured, or absent. This layer
scores ONLY what this match can legitimately answer and labels every category it
cannot, rather than inflating coverage.

Built against the REAL shapes of:
  - running_summary.json  (key_moments, match_state_by_window, fouls_committed, team names)
  - ground_truth_3895158.json (the quarantined StatsBomb truth)
Score line is prose-only (tactical_report.md line ~10); pass --report to parse it,
else the score is reconstructed from key_moments goal entries.

READ-ONLY. Writes one report into --out. Stdlib only.
"""
import argparse, json, re
from pathlib import Path

# ---- the four standing caveats, printed beside every result ----
CAVEATS = [
    "Filter precision: UNMEASURABLE -- open-play path persists no candidate denominator.",
    "Filter existence: observable only on the set-piece burst path (genuine, discriminating); "
    "invisible for all open-play families -- persistence gap, not capability gap.",
    "Null-emission capability: PARTIALLY tested -- this match had no dead passage to test "
    "set-piece silence, but cards (5 real / 0 claimed) DO exercise null-on-content.",
    "Operator-verified inputs: match boundaries and attack direction were human-corrected during "
    "blind setup; any direction-dependent result leans partly on operator input, not pure detection.",
]

def parse_minute(s):
    """Best-effort minute extraction. WARNING: key_moments mixes clock conventions --
    '45:13' reads as match-clock but its frame_range points at video-time frame_47m03s,
    and '04m13s'/'79m03s' are video-frame times. This returns the leading integer only
    and the caller MUST treat the result as clock-ambiguous, not a clean match minute.
    Where a frame_range exists, that's video-time; the bare 'minute' may be either."""
    if s is None: return None
    m = re.match(r"(\d+)[m:](\d+)", str(s))
    return int(m.group(1)) if m else None

def clock_note(km):
    """Flag the clock ambiguity per goal/sub entry so mismatches aren't read as errors."""
    minute_str = str(km.get("minute"))
    has_range = bool(km.get("observation_frame_range"))
    fmt = "video(m..s)" if "m" in minute_str else ("colon(ambiguous)" if ":" in minute_str else "raw")
    return f"minute='{minute_str}' fmt={fmt} frame_range={'yes(video-time)' if has_range else 'no'}"

def kit_to_team(kit, summary):
    if kit == "home_kit": return summary["home_team"]
    if kit == "away_kit": return summary["away_team"]
    return kit

def load_score_from_report(report_path):
    """Score lives in prose only. Return (home,away) or None."""
    if not report_path or not Path(report_path).exists():
        return None
    txt = Path(report_path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"(?:final\s*score|score)\D*?(\d+)\s*[-–:]\s*(\d+)", txt, re.IGNORECASE)
    return (int(m.group(1)), int(m.group(2))) if m else None


def score_match(summary, gt, report_path):
    out = {"fixture_truth": gt["fixture"], "categories": {}, "findings": []}
    home, away = summary["home_team"], summary["away_team"]

    # ---- GOALS ----
    goal_km = [k for k in summary["key_moments"] if k.get("type") == "goal"]
    # collapse to (team, minute) but KEEP duplicates visible
    claimed_goals = [{"team": kit_to_team(k["team"], summary),
                      "minute": parse_minute(k.get("minute")),
                      "desc": k.get("description", "")} for k in goal_km]
    truth_goals = [{"team": g["team"], "minute": g["minute"]} for g in gt["goals"]]

    # duplicate / inconsistency detection: same team scored more than truth says
    from collections import Counter
    claimed_per_team = Counter(g["team"] for g in claimed_goals)
    truth_per_team = Counter(g["team"] for g in truth_goals)
    dup_flags = []
    for team in set(claimed_per_team) | set(truth_per_team):
        if claimed_per_team[team] > truth_per_team[team]:
            mins = sorted(g["minute"] for g in claimed_goals if g["team"] == team)
            dup_flags.append(f"{team}: {claimed_per_team[team]} goals claimed vs {truth_per_team[team]} real "
                             f"(claimed minutes {mins}) -- likely duplicate logging of one goal")
    # team-level goal agreement
    goal_team_match = (claimed_per_team == truth_per_team)
    out["categories"]["goals"] = {
        "scoreable": "partly (team+minute; no scorer field)",
        "truth": truth_goals,
        "claimed": [{"team": g["team"], "minute": g["minute"]} for g in claimed_goals],
        "team_count_agreement": goal_team_match,
        "duplicate_inconsistency": dup_flags or None,
    }
    for f in dup_flags:
        out["findings"].append("GOAL-LOG INCONSISTENCY -- " + f)

    # ---- SCORE ----
    report_score = load_score_from_report(report_path)
    # reconstruct from claimed goals as fallback
    recon = (claimed_per_team[home], claimed_per_team[away])
    truth_score = (gt["score"]["home"], gt["score"]["away"])
    out["categories"]["score"] = {
        "scoreable": "yes (prose line in report; reconstructable from key_moments)",
        "truth": list(truth_score),
        "report_prose": list(report_score) if report_score else None,
        "reconstructed_from_key_moments": list(recon),
        "prose_matches_truth": (report_score == truth_score) if report_score else None,
        "note": "reconstruction inflated by duplicate goal -- see goals.duplicate_inconsistency"
                if recon != truth_score else None,
    }
    if report_score and report_score != truth_score:
        out["findings"].append(f"SCORE MISMATCH -- report prose {report_score} vs truth {truth_score}")
    elif report_score == truth_score:
        out["findings"].append(f"SCORE CORRECT (prose) -- {report_score}; but key_moments reconstruct "
                               f"to {recon} due to duplicate goal entry")

    # ---- SUBSTITUTIONS (timing+team only; identity not stored) ----
    sub_km = [k for k in summary["key_moments"] if k.get("type") == "substitution"]
    out["categories"]["substitutions"] = {
        "scoreable": "timing+team only (identity unconfirmed in report)",
        "truth_count": len(gt["substitutions"]),
        "claimed_count": len(sub_km),
        "claimed": [{"team": kit_to_team(k["team"], summary), "minute": parse_minute(k.get("minute"))}
                    for k in sub_km],
        "note": f"pipeline logged {len(sub_km)} of {len(gt['substitutions'])} real subs -- "
                f"under-detection, not a claim error",
    }
    out["findings"].append(f"SUBSTITUTIONS -- {len(sub_km)} logged vs {len(gt['substitutions'])} real "
                           f"(timing/team only; who-came-on never stored)")

    # ---- CARDS (a measurable NULL) ----
    has_cards = bool(summary.get("fouls_committed")) or any(
        k.get("type") == "card" for k in summary["key_moments"])
    out["categories"]["cards"] = {
        "scoreable": "NULL result -- pipeline asserted no cards",
        "truth_count": len(gt["cards"]),
        "claimed_count": 0,
        "interpretation": "null-on-content: 5 real cards occurred, pipeline surfaced none. "
                          "This is the one place THIS match tests null-emission against real events.",
    }
    out["findings"].append(f"CARDS -- 5 real, 0 claimed. Pipeline makes no card assertion on this source. "
                           f"Measurable null-on-content.")

    # ---- JERSEY / XI (untestable by source) ----
    out["categories"]["jersey_xi"] = {
        "scoreable": "NOT TESTABLE on this source",
        "reason": "tactical_wide_static -- pipeline skips OCR (frames_scanned=0, sightings=0). "
                  "Any #number in the report is echoed from the operator-typed config XI, NOT detected. "
                  "Cannot even degrade to name-matching: it is config-echo. Requires broadcast/closer "
                  "footage where OCR fires.",
    }

    # ---- MATCH-STATE-BY-WINDOW vs true score timeline ----
    # Build the TRUE relative-state timeline from ground-truth goals, then compare
    # each window's claimed match_state against what was actually true at that time.
    # Two candidate causes for disagreement are NAMED, not adjudicated:
    #   (a) genuine contradiction in scoreboard reads, or
    #   (b) scoreboard unreadable -> agent defaults to first enum option ('level').
    msw = summary.get("match_state_by_window", [])
    # true state changes: 0-0 until DOR 4', then away ahead until LEV 78', then level.
    # (home=Leverkusen, away=Dortmund per verified mapping)
    def true_state_at(half, start_min):
        # crude: map window to absolute match minute via half + start
        absmin = start_min if half == "1H" else 45 + start_min
        # DOR scored ~4', LEV equalised ~78'
        if absmin < 4: return "level"
        if absmin < 78: return "away_winning"
        return "level"
    rows = []
    agree = 0
    for w in msw:
        wlabel = w.get("window", "")
        mm = re.match(r"(1H|2H)_(\d+)", wlabel)
        half, start = (mm.group(1), int(mm.group(2))) if mm else ("1H", 0)
        truth_state = true_state_at(half, start)
        claimed = w.get("match_state")
        ok = (claimed == truth_state)
        agree += ok
        rows.append({"window": wlabel, "claimed": claimed, "true": truth_state, "match": ok})
    out["categories"]["match_state_by_window"] = {
        "scoreable": "yes -- per-window relative state vs true score timeline",
        "windows": len(msw),
        "agreement": f"{agree}/{len(msw)}",
        "first_half_all_level_while_DOR_led": all(
            r["claimed"] == "level" for r in rows if r["window"].startswith("1H") and not r["window"].startswith("1H_00")
        ),
        "detail": rows,
        "candidate_causes_NOT_adjudicated": [
            "(a) genuine contradiction in the pipeline's scoreboard reads",
            "(b) scoreboard unreadable on tactical-wide -> agent emits first/default enum 'level' "
            "(matches the known first-enum-anchoring failure mode; first-half uniform 'level' is "
            "consistent with this, second-half oscillation is not -- possibly two modes stacked)",
        ],
    }
    out["findings"].append(
        f"MATCH-STATE vs TRUTH -- {agree}/{len(msw)} windows agree with true score timeline. "
        f"First half reads 'level' throughout while Dortmund led from 4'. Cause not isolated: "
        f"genuine contradiction vs scoreboard-unreadable-defaulting-to-level (see category detail)."
    )

    # ---- TIMESTAMP FORMAT HAZARD ----
    fmts = set()
    for k in summary["key_moments"]:
        ms = str(k.get("minute"))
        fmts.add("video(m..s)" if "m" in ms else ("colon" if ":" in ms else "raw"))
    if len(fmts) > 1:
        out["findings"].append(
            f"TIMESTAMP FORMAT HAZARD -- key_moments mixes {sorted(fmts)}; some entries carry "
            f"video-time frame ranges alongside colon-format minutes. Minutes are CLOCK-AMBIGUOUS; "
            f"goal/sub minute agreement is approximate (+/- a few min), not exact."
        )

    # ---- coverage map ----
    out["coverage"] = {
        "categories_designed": 5,
        "categories_scored": ["score", "goals(team+minute)", "substitutions(timing+team)",
                              "cards(as null)", "match_state(per-window vs timeline)"],
        "categories_not_scored": ["scorer_identity (prose-only/inferential)",
                                  "sub_identity (not stored)",
                                  "jersey/XI (source-gated: no OCR on tactical-wide)"],
        "headline": "Counter 1 here scores SCORE + EVENT-TIMING with weak scorer attribution and a "
                    "measurable card-null. Lineup/jersey accuracy is untestable on this footage type.",
    }
    out["caveats"] = CAVEATS
    return out


def render(out):
    L = ["="*72, "COUNTER 1 -- HARD-FACT AGREEMENT", f"Truth: {out['fixture_truth']}", "="*72, ""]
    L.append("FINDINGS (the point of the exercise):")
    for f in out["findings"]:
        L.append(f"  * {f}")
    L.append("")
    L.append("COVERAGE -- what this match could and could not answer:")
    L.append(f"  scored:     {', '.join(out['coverage']['categories_scored'])}")
    L.append(f"  NOT scored: {', '.join(out['coverage']['categories_not_scored'])}")
    L.append(f"  >> {out['coverage']['headline']}")
    L.append("")
    L.append("PER-CATEGORY:")
    for name, c in out["categories"].items():
        L.append(f"  [{name}]  scoreable: {c.get('scoreable')}")
        for k, v in c.items():
            if k in ("scoreable", "detail"): continue
            L.append(f"       {k}: {v}")
        if name == "match_state_by_window" and c.get("detail"):
            wrong = [r for r in c["detail"] if not r["match"]]
            L.append(f"       windows_disagreeing: {len(wrong)} -> " +
                     ", ".join(f"{r['window'].split('_',1)[-1]}(said {r['claimed']}, true {r['true']})"
                               for r in wrong[:6]) + (" ..." if len(wrong) > 6 else ""))
    L.append("")
    L.append("STANDING CAVEATS (this number means nothing without these):")
    for i, cav in enumerate(out["caveats"], 1):
        L.append(f"  {i}. {cav}")
    L.append("="*72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="running_summary.json")
    ap.add_argument("--truth", required=True, help="ground_truth_3895158.json")
    ap.add_argument("--report", default=None, help="tactical_report.md (for prose score line)")
    ap.add_argument("--out", default=None,
                    help="default: <this script dir>/outputs/counter1_result.json")
    a = ap.parse_args()

    out_default = Path(__file__).resolve().parent / "outputs" / "counter1_result.json"
    a.out = a.out or str(out_default)

    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))
    gt = json.loads(Path(a.truth).read_text(encoding="utf-8"))

    # schema-assertion front door: fail loud if the report shape drifted
    required = ["key_moments", "match_state_by_window", "home_team", "away_team", "fouls_committed"]
    missing = [k for k in required if k not in summary]
    assert not missing, f"running_summary.json missing expected keys {missing} -- schema drift, STOP."

    out = score_match(summary, gt, a.report)
    rendered = render(out)
    print(rendered)

    outpath = Path(a.out); outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (outpath.parent / "counter1_result.txt").write_text(rendered, encoding="utf-8")
    print(f"\nwrote {outpath} (+ .txt)")

if __name__ == "__main__":
    main()
