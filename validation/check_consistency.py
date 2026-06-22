"""
check_consistency.py -- Tier-1 internal-consistency checker (NO ground truth)

WHAT THIS IS
------------
A truth-stripped descendant of compare_counter1.py. It reads ONLY a pipeline
running_summary.json and asks one question: does the report agree with ITSELF?
It never loads ground truth, because its purpose is to run on footage where
ground truth does not and will never exist -- i.e. Veo / non-league production.

It therefore cannot say "the report is WRONG about reality." It can only say
"the report CONTRADICTS ITSELF" -- which is a defect regardless of what truly
happened, and is the whole point: a self-contradiction is footage-independent.

WHAT IT CHECKS (internal invariants only)
-----------------------------------------
  1. Goal-log self-consistency: no scoreline transition logged twice; the
     scoreline implied by the goal sequence advances monotonically.
  2. Match-state vs the report's OWN goal log: each window's match_state must be
     consistent with the scoreline the goal log implies at that time. (NOT vs an
     external timeline -- vs the report's own goals.)
  3. Timestamp sanity: flag mixed clock conventions and non-chronological order.
  4. Lossy-summary fields: flag fields known to be dead/under-counting where the
     summary fails to reflect agent-level truth.

BUILT TO THE INVARIANT, NOT THE SYMPTOM
---------------------------------------
Check (1) tests "does any scoreline transition repeat / does the score ever go
backwards", NOT "is there a goal at both 45 and 79". Written to the general
property it catches the known duplicate AND unseen siblings of the same
accumulator never-reconcile flaw.

PASS/FAIL
---------
Exits 0 if no inconsistency found, 1 if any invariant is violated. This makes it
usable as an acceptance test: run before a fix (expect FAIL on known bugs), run
after (expect PASS).

READ-ONLY. Reads one running_summary.json, writes one report into --out. Stdlib.
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path


def parse_minute(s):
    """Leading integer of a key_moments minute. CLOCK-AMBIGUOUS by design --
    '04m13s'/'79m03s' are video-time, '45:13' may be match-clock; callers must
    treat the result as approximate ordering, not an exact match minute."""
    if s is None:
        return None
    m = re.match(r"(\d+)[m:](\d+)", str(s))
    return int(m.group(1)) if m else None


def minute_fmt(s):
    ms = str(s)
    return "video(m..s)" if "m" in ms else ("colon" if ":" in ms else "raw")


def kit_to_team(kit, summary):
    if kit == "home_kit":
        return summary.get("home_team", "home_kit")
    if kit == "away_kit":
        return summary.get("away_team", "away_kit")
    return kit


def check(summary):
    out = {"fixture": summary.get("match", "unknown"), "findings": [], "ok": True}

    def fail(msg):
        out["findings"].append("FAIL  " + msg)
        out["ok"] = False

    def note(msg):
        out["findings"].append("note  " + msg)

    def unverifiable(msg):
        # a check the gate COULD NOT run -- must NEVER be silently treated as a pass
        out["findings"].append("UNVERIFIABLE  " + msg)
        out.setdefault("unverifiable", []).append(msg)

    home = summary.get("home_team", "home_team")
    away = summary.get("away_team", "away_team")
    km = summary.get("key_moments", [])
    goals = [k for k in km if k.get("type") == "goal"]
    subs = [k for k in km if k.get("type") == "substitution"]

    # ---------------------------------------------------------------
    # INVARIANT 1 -- goal-log self-consistency
    #   Spine = goals ordered by minute, each assigned its EXPECTED cumulative
    #   scoreline from the team tags (structured, reliable). Descriptions are a
    #   CROSS-CHECK only, never the source -- so phrasing/word-order cannot defeat
    #   the check, and a goal whose text omits a scoreline is still counted.
    #   (a) re-log: a goal naming an already-reached total but not its own new one
    #   (b) corroboration: entry count vs the highest total any description names
    # ---------------------------------------------------------------
    def mentioned_scores(desc):
        # scorelines named in free text, as a set of (x, y). Digit-adjacency
        # guards reject formation strings (4-2-3-1) and multi-digit numbers.
        return {(int(x), int(y)) for x, y in
                re.findall(r"(?<!\d)(?<!\d[-–])(\d)\s*[-–]\s*(\d)(?!\s*[-–]\s*\d)", desc or "")}

    goal_seq = sorted(
        ((parse_minute(g.get("minute")), kit_to_team(g.get("team"), summary), g) for g in goals),
        key=lambda x: x[0] if x[0] is not None else 10 ** 9,
    )
    claimed_per_team = Counter(kit_to_team(g.get("team"), summary) for g in goals)

    h = a = 0
    max_total_mentioned = 0
    for i, (m, team, g) in enumerate(goal_seq):
        if team == home:
            h += 1
        elif team == away:
            a += 1
        exp_total = h + a
        seen_totals = {x + y for x, y in mentioned_scores(g.get("description", ""))}
        if seen_totals:
            max_total_mentioned = max(max_total_mentioned, max(seen_totals))

        # (a) re-log detection -- orientation-free: a goal whose description names
        #     only scoreline totals already reached earlier (and not its own new
        #     expected total) is the same goal logged twice in another window.
        if seen_totals and exp_total not in seen_totals and any(t < exp_total for t in seen_totals):
            fail(f"GOAL-LOG: goal #{i + 1} (min {m}, {team}) names scoreline total(s) "
                 f"{sorted(seen_totals)} already reached earlier, not its own expected new "
                 f"total {exp_total} -- same goal re-logged in another window "
                 f"(accumulator never-reconcile defect).")

    out["goal_log"] = {
        "n_goal_entries": len(goals),
        "reconstructed_per_team": dict(claimed_per_team),
        "expected_final_scoreline_from_sequence": [h, a],
        "max_total_mentioned_in_descriptions": max_total_mentioned,
    }
    # (b) corroboration: more goal entries than the highest total any description
    #     names => surplus entry that adds no new score (duplicate or phantom).
    if goals and max_total_mentioned and len(goals) > max_total_mentioned:
        fail(f"GOAL-LOG: {len(goals)} goal entries but no description names a scoreline "
             f"beyond {max_total_mentioned} goals total -- {len(goals) - max_total_mentioned} "
             f"surplus entry/entries add no new score (duplicate or phantom).")

    # ---------------------------------------------------------------
    # INVARIANT 2 -- match_state_by_window vs the report's OWN goal log
    #   Build the scoreline timeline from THIS report's goals, then check each
    #   window's relative-state label against the report's own implied lead.
    #   NB: this is internal -- no external truth. We compare the report to itself.
    # ---------------------------------------------------------------
    msw = summary.get("match_state_by_window", [])

    # Build (minute -> implied lead) from the report's own goal transitions.
    # Lead sign from home/away perspective: +1 home ahead, -1 away ahead, 0 level.
    # Source the lead timeline from ALL team-tagged goals (the structured spine
    # built above), NOT from description-parsed transitions -- otherwise a goal
    # whose text lacks a full 'A-B to C-D' string (e.g. the opener '...to 0-1')
    # would be invisible here and the implied lead would silently undercount.
    lead_events = sorted((m if m is not None else 0, team) for m, team, _g in goal_seq)

    def implied_state_at(abs_min):
        h = a = 0
        for m, team in lead_events:
            if m <= abs_min:
                if team == home:
                    h += 1
                elif team == away:
                    a += 1
        if h > a:
            return "home_winning"
        if a > h:
            return "away_winning"
        return "level"

    def window_abs_min(label):
        # search (not match): the half marker may be embedded, e.g. "W02_1H_05-10min"
        mm = re.search(r"(1H|2H)_(\d+)", label or "")
        if not mm:
            return None
        half, start = mm.group(1), int(mm.group(2))
        return start if half == "1H" else 45 + start

    rows, disagree, n_placeable = [], 0, 0
    for w in msw:
        label = w.get("window", "")
        amin = window_abs_min(label)
        if amin is not None:
            n_placeable += 1
        claimed = w.get("match_state")
        implied = implied_state_at(amin) if amin is not None else None
        # only compare when we have both; 'winning' label may be either side
        ok = None
        if implied is not None and claimed is not None and claimed in ("home_winning", "away_winning", "level"):
            ok = (claimed == implied)
        if ok is False:
            disagree += 1
        rows.append({"window": label, "claimed": claimed, "implied_by_own_goals": implied, "match": ok})

    comparable = [r for r in rows if r["match"] is not None]
    if disagree:
        fail(f"MATCH-STATE: {disagree}/{len(comparable)} comparable windows disagree with the "
             f"report's OWN goal log. The report's per-window lead contradicts the lead its own "
             f"goals imply -- internal contradiction (no external truth needed).")
    # The gate must NEVER silently pass a check it could not run. If no window carries a
    # parseable time label, match_state cannot be placed against the goal log -> say so LOUD.
    if msw and n_placeable == 0:
        unverifiable(f"MATCH-STATE: 0/{len(msw)} windows have a parseable time label (the 'window' "
                     f"field is absent/unrecognised), so match_state could NOT be checked against "
                     f"the goal log. This is NOT a pass on this invariant.")
    elif 0 < n_placeable < len(msw):
        note(f"MATCH-STATE: only {n_placeable}/{len(msw)} windows had a parseable time label; "
             f"the remaining {len(msw) - n_placeable} were unplaceable and NOT checked.")
    out["match_state"] = {
        "windows": len(msw),
        "placeable": n_placeable,
        "comparable": len(comparable),
        "disagreeing": disagree,
        "candidate_causes_NOT_adjudicated": [
            "(a) genuine contradiction in the pipeline's per-window scoreboard reads",
            "(b) scoreboard unreadable -> agent defaults to first enum option 'level' "
            "(first-enum anchoring). Uniform 'level' across a stretch is consistent with (b); "
            "oscillation is more consistent with (a). Possibly both, stacked.",
        ],
        "detail": rows,
    }

    # ---------------------------------------------------------------
    # INVARIANT 3 -- timestamp sanity (hygiene; prevents 1&2 misfiring)
    # ---------------------------------------------------------------
    fmts = sorted({minute_fmt(k.get("minute")) for k in km if k.get("minute") is not None})
    if len(fmts) > 1:
        note(f"TIMESTAMP: key_moments mixes clock conventions {fmts}; some entries carry "
             f"video-time frame ranges alongside colon-format minutes. Minute ordering is "
             f"approximate, not exact -- match-state/goal alignment carries +/- slop.")
    goal_mins = [parse_minute(g.get("minute")) for g in goals if parse_minute(g.get("minute")) is not None]
    if goal_mins != sorted(goal_mins):
        note(f"TIMESTAMP: goal key_moments are not in chronological order ({goal_mins}) -- "
             f"accumulator concatenates per-window without sorting.")

    # ---------------------------------------------------------------
    # INVARIANT 4 -- lossy-summary fields (summary not reflecting agent truth)
    #   Known family: by-window narrative slots empty; set_pieces_rejected under-counts.
    #   These are NOT failures of self-contradiction -- flagged as 'note', the
    #   fourth instance of the accumulator concatenate-or-drop pattern.
    # ---------------------------------------------------------------
    def empty_rate(arr, field, kind):
        vals = [w.get(field) for w in arr]
        empty = sum(1 for v in vals if v in (None, "", [], {}))
        return empty, len(vals)
    for arr_name, field in [("pressing_by_window", "observations")]:
        arr = summary.get(arr_name, [])
        if arr:
            e, n = empty_rate(arr, field, None)
            if n and e == n:
                note(f"LOSSY-SUMMARY: {arr_name}[].{field} empty in {e}/{n} windows -- "
                     f"dead summary slot (data may exist upstream; summary drops it).")
    spr = summary.get("set_pieces_rejected", None)
    spf = summary.get("set_piece_filter_summary")
    if isinstance(spf, dict) and spf.get("available"):
        note(f"SET-PIECE FILTER: {spf.get('confirmed')}/{spf.get('bursts_run')} bursts confirmed, "
             f"{spf.get('not_confirmed')} not-confirmed ({spf.get('distinct_rejection_reasons')} distinct "
             f"reasons) -- filter existence IS observable here (set_piece_filter_summary). "
             f"set_pieces_rejected is schema-validation only.")
    elif isinstance(spr, list) and len(spr) == 0:
        note("LOSSY-SUMMARY: set_pieces_rejected is empty AND no set_piece_filter_summary -- "
             "the burst filter signal is not surfaced; read per-burst *_setpiece.json files.")

    return out


def render(out):
    L = ["=" * 72, "TIER-1 INTERNAL-CONSISTENCY CHECK  (no ground truth)",
         f"Report: {out['fixture']}", "=" * 72, ""]
    if not out["ok"]:
        verdict = "FAIL -- internal contradiction(s) found"
    elif out.get("unverifiable"):
        verdict = ("INCOMPLETE -- no contradiction found, but a check could NOT be run "
                   "(see UNVERIFIABLE below). This is NOT a clean pass.")
    else:
        verdict = "PASS -- no self-contradiction found"
    L.append(f"VERDICT: {verdict}")
    L.append("")
    L.append("FINDINGS:")
    if not out["findings"]:
        L.append("  (none)")
    for f in out["findings"]:
        L.append("  * " + f)
    L.append("")
    if "goal_log" in out:
        g = out["goal_log"]
        L.append("GOAL-LOG:")
        L.append(f"  entries={g['n_goal_entries']}  per_team={g['reconstructed_per_team']}  "
                 f"seq_final={g['expected_final_scoreline_from_sequence']}  "
                 f"max_total_in_text={g['max_total_mentioned_in_descriptions']}")
    if "match_state" in out:
        m = out["match_state"]
        L.append("MATCH-STATE vs OWN GOAL LOG:")
        L.append(f"  windows={m['windows']}  comparable={m['comparable']}  disagreeing={m['disagreeing']}")
        wrong = [r for r in m["detail"] if r["match"] is False]
        if wrong:
            L.append("  disagreeing -> " +
                     ", ".join(f"{r['window'].split('_',1)[-1]}(said {r['claimed']}, own-goals imply {r['implied_by_own_goals']})"
                               for r in wrong[:6]) + (" ..." if len(wrong) > 6 else ""))
        L.append("  candidate causes (NOT adjudicated):")
        for c in m["candidate_causes_NOT_adjudicated"]:
            L.append(f"    - {c}")
    L.append("")
    L.append("SCOPE: this checks self-consistency ONLY. PASS means 'the report does not")
    L.append("contradict itself', NEVER 'the report is accurate'. Accuracy needs ground")
    L.append("truth, which this footage class does not have. Necessary, not sufficient.")
    L.append("=" * 72)
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True, help="running_summary.json (the ONLY input)")
    ap.add_argument("--out", default=None,
                    help="default: <this script dir>/outputs/consistency_result.json")
    a = ap.parse_args()

    out_default = Path(__file__).resolve().parent / "outputs" / "consistency_result.json"
    a.out = a.out or str(out_default)

    summary = json.loads(Path(a.summary).read_text(encoding="utf-8"))

    # schema front door: fail loud if the report shape drifted
    required = ["key_moments", "match_state_by_window", "home_team", "away_team"]
    missing = [k for k in required if k not in summary]
    assert not missing, f"running_summary.json missing expected keys {missing} -- schema drift, STOP."

    out = check(summary)
    rendered = render(out)
    print(rendered)

    outpath = Path(a.out)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    (outpath.parent / "consistency_result.txt").write_text(rendered, encoding="utf-8")
    print(f"\nwrote {outpath} (+ .txt)")

    # exit code makes this usable as an acceptance test:
    #   0 = clean pass, 1 = contradiction found,
    #   2 = no contradiction but a check could NOT be run (unverifiable -- not a clean pass)
    sys.exit(1 if not out["ok"] else (2 if out.get("unverifiable") else 0))


if __name__ == "__main__":
    main()
