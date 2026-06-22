#!/usr/bin/env python3
"""
statsbomb_groundtruth.py -- verified hard-fact ground truth for
Bayer Leverkusen 1-1 Borussia Dortmund (StatsBomb match_id 3895158).

QUARANTINE / ONE-WAY VALVE
--------------------------
The JSON this writes is COMPARISON-ONLY. It must never be imported by, fed into,
or written near the Match Lens pipeline. The pipeline runs blind; ground truth
meets pipeline output only in the post-run comparison layer. Do not place this
file or its output inside any match job dir or pipeline_state.json.

READ-ONLY on the StatsBomb source. Writes one file into --out only. Stdlib only.
"""
import argparse, json, urllib.request
from pathlib import Path

MATCH_ID = 3895158
COMP_ID, SEASON_ID = 9, 281
RAW = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

# confirmed truths -- regression guard, verified from the real files
EXPECT = {"score": ("Bayer Leverkusen", 1, 1, "Borussia Dortmund"),
          "n_goals": 2, "n_cards": 5, "n_subs": 5,
          "formations": {"Bayer Leverkusen": 3421, "Borussia Dortmund": 4231}}
CARD_KEY = {"Foul Committed": "foul_committed", "Bad Behaviour": "bad_behaviour"}

def load(kind, data_dir):
    rel = f"matches/{COMP_ID}/{SEASON_ID}.json" if kind == "matches" else f"{kind}/{MATCH_ID}.json"
    if data_dir:
        return json.loads((Path(data_dir) / rel).read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{RAW}/{rel}", timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))

def extract(events, lineups, matches):
    m = next(x for x in matches if x["match_id"] == MATCH_ID)
    home, away = m["home_team"]["home_team_name"], m["away_team"]["away_team_name"]
    goals, cards, subs, starting, shifts = [], [], [], [], []
    for e in events:
        t = e["type"]["name"]
        if t == "Shot" and e.get("shot", {}).get("outcome", {}).get("name") == "Goal":
            goals.append({"minute": e["minute"], "scorer": e["player"]["name"],
                          "team": e["team"]["name"], "shot_type": e["shot"].get("type", {}).get("name")})
        elif t == "Own Goal For":
            goals.append({"minute": e["minute"], "scorer": "(own goal)",
                          "team": e["team"]["name"], "shot_type": "Own Goal"})
        elif t in CARD_KEY:
            card = e.get(CARD_KEY[t], {}).get("card", {}).get("name")
            if card:
                cards.append({"minute": e["minute"], "card": card,
                              "player": e["player"]["name"], "team": e["team"]["name"], "via": t})
        elif t == "Substitution":
            subs.append({"minute": e["minute"], "off": e["player"]["name"],
                         "on": e["substitution"]["replacement"]["name"], "team": e["team"]["name"]})
        elif t == "Starting XI":
            tac = e["tactics"]
            starting.append({"team": e["team"]["name"], "formation": tac["formation"],
                             "xi": [{"jersey": p["jersey_number"], "player": p["player"]["name"],
                                     "position": p["position"]["name"]} for p in tac["lineup"]]})
        elif t == "Tactical Shift":
            shifts.append({"minute": e["minute"], "team": e["team"]["name"],
                           "formation": e["tactics"]["formation"]})
    jersey = {}
    for team in lineups:
        jersey[team["team_name"]] = sorted(
            ({"jersey": p["jersey_number"], "player": p["player_name"]} for p in team["lineup"]),
            key=lambda r: (r["jersey"] is None, r["jersey"]))
    return {"match_id": MATCH_ID, "fixture": f"{home} {m['home_score']}-{m['away_score']} {away}",
            "score": {"home_team": home, "home": m["home_score"], "away": m["away_score"], "away_team": away},
            "goals": goals, "cards": cards, "substitutions": subs,
            "starting_xi": starting, "tactical_shifts": shifts, "squad_jersey_map": jersey}

def selfcheck(gt):
    s = gt["score"]
    assert (s["home_team"], s["home"], s["away"], s["away_team"]) == EXPECT["score"], "score mismatch"
    assert len(gt["goals"]) == EXPECT["n_goals"], f"goals {len(gt['goals'])}"
    assert len(gt["cards"]) == EXPECT["n_cards"], f"cards {len(gt['cards'])}"
    assert len(gt["substitutions"]) == EXPECT["n_subs"], f"subs {len(gt['substitutions'])}"
    got = {x["team"]: x["formation"] for x in gt["starting_xi"]}
    assert got == EXPECT["formations"], f"formations {got}"
    per = {}
    for g in gt["goals"]:
        per[g["team"]] = per.get(g["team"], 0) + 1
    assert per.get(s["home_team"], 0) == s["home"] and per.get(s["away_team"], 0) == s["away"], "goals dont reconcile with score"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None, help="local statsbomb/open-data clone (optional; fetches if omitted)")
    ap.add_argument("--out", default="validation/groundtruth")
    a = ap.parse_args()
    gt = extract(load("events", a.data_dir), load("lineups", a.data_dir), load("matches", a.data_dir))
    selfcheck(gt)
    outdir = Path(a.out); outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / f"ground_truth_{MATCH_ID}.json"
    outpath.write_text(json.dumps(gt, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK  wrote {outpath}")
    print(f"    {gt['fixture']}   goals={len(gt['goals'])} cards={len(gt['cards'])} subs={len(gt['substitutions'])}")
    print("\n  GOALS:")
    for g in gt["goals"]:        print(f"    {g['minute']:>2}' {g['scorer']} ({g['team']}) {g['shot_type']}")
    print("  CARDS:")
    for c in gt["cards"]:        print(f"    {c['minute']:>2}' {c['card']} {c['player']} ({c['team']}) via {c['via']}")
    print("  SUBS:")
    for s in gt["substitutions"]: print(f"    {s['minute']:>2}' {s['off']} -> {s['on']} ({s['team']})")
    print("  XI / FORMATION:")
    for x in gt["starting_xi"]:   print(f"    {x['team']}: {x['formation']} ({len(x['xi'])} players)")

if __name__ == "__main__":
    main()
