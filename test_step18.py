"""
test_step18.py -- proves the new ground_truth_node wiring in
readiness_graph.py actually works, in four separate scenarios:

  1. Happy path: a fresh match_dir with real-shaped inputs -> the graph
     itself computes and writes ground_truth_check.json (nobody hand-wrote
     it), and readiness_gate correctly reads that freshly-written file.
  2. Malformed match_boundaries.json (present but missing the keys
     ground_truth.py needs) -> must NOT crash the graph; ground_truth_node
     catches the KeyError and lets readiness_gate handle the fallout.
  3. Totally immature match_dir (no match_config.json / running_summary.json
     at all) -> must NOT crash the graph; ground_truth_node catches the
     FileNotFoundError and readiness_gate blocks for the real reason.
  4. Real-data regression: the actual Gorleston vs Tilbury fixture run
     through the FULL match graph end to end -- confirms the real 16/0/3/13
     ground-truth numbers land on disk and on state, and that report_ready
     correctly comes out False for real reasons (not just re-testing
     build_ground_truth_check() in isolation, which test_step16 already did --
     this test proves the GRAPH wiring reaches the same real numbers).
"""

import json
import os
import shutil
import tempfile

from readiness_graph import build_match_graph, MatchState


def _write(match_dir, fname, data):
    with open(os.path.join(match_dir, fname), "w", encoding="utf-8") as f:
        json.dump(data, f)


# Real production match_boundaries.json shape (ko_1h/ht_whistle/ko_2h/ft_whistle
# with "seconds" + "confidence"), reused across scenarios 1 and 4's setup so
# every test in this file that needs boundaries uses the SAME real schema --
# not a simplified test-only shape, since that mismatch is exactly what
# scenario 2 below is testing for.
def _real_boundaries():
    return {
        "boundaries": {
            "ko_1h":      {"seconds": 0,    "confidence": 0.95},
            "ht_whistle": {"seconds": 2700, "confidence": 0.95},
            "ko_2h":      {"seconds": 2760, "confidence": 0.90},
            "ft_whistle": {"seconds": 5460, "confidence": 0.90},
        },
    }


print("--- test 1: happy path -- graph itself computes ground_truth_check.json, nobody hand-writes it ---")
d1 = tempfile.mkdtemp(prefix="matchlens_step18_happy_")
_write(d1, "match_boundaries.json", _real_boundaries())
_write(d1, "match_config.json", {
    "verified": True, "enrichment_level": "identity_only",
    "player_id_ceiling": "tentative", "match": "Test FC vs Test United",
    "goals": [], "substitutions": [], "cards": [],   # zero known events on purpose
})
_write(d1, "window_plan.json", {"total_windows": 1})
_write(d1, "running_summary.json", {"windows_complete": 1, "data_gap_windows": [], "key_moments": []})
_write(d1, "rerun_queue.json", {"rerun_queue": []})
_write(d1, "confirmation_queue.json", {"total": 0, "skipped": 0})
_write(d1, "source_profile.json", {
    "source_type": "tactical_wide_static", "classification_confidence": 0.9, "split_aware": False,
})
_write(d1, "result_family_gates.json", {"gates": {}})
_write(d1, "deep_skill_metrics.json", {
    "total_metrics": 0, "active_metrics": 0, "suppressed_metrics": [], "avg_confidence": 0.0,
})
_write(d1, "pass_sequences.json", {"sequences": []})

# WHY this assertion matters: BEFORE this file's ground_truth_check.json is
# written by the graph, it does not exist on disk at all -- unlike test_step11
# /test_step12's fixtures (pre-Step-18), this test deliberately never hand-writes
# it. If ground_truth_node were somehow skipped or broken, build_readiness_check()
# would see "file missing" and block -- so seeing report_ready=True below is
# itself proof the node ran and produced a real file, not just a happy accident.
assert not os.path.exists(os.path.join(d1, "ground_truth_check.json")), \
    "test setup bug: ground_truth_check.json should not exist yet"

app = build_match_graph()
result1 = app.invoke(MatchState(match_dir=d1))
print(f"  report_ready={result1['report_ready']}, blocking_issues={result1['blocking_issues']}")
print(f"  ground_truth_result={result1['ground_truth_result']}")

assert result1["report_ready"] is True, result1["blocking_issues"]
assert result1["ground_truth_result"] is not None
assert result1["ground_truth_result"]["events_checked"] == 0
assert result1["ground_truth_result"]["missed"] == 0

# The real proof this is graph-produced, not fixture-produced: the file
# exists NOW, with content matching what the node returned on state.
gt_file_path = os.path.join(d1, "ground_truth_check.json")
assert os.path.exists(gt_file_path), "ground_truth_node should have written this file"
with open(gt_file_path, encoding="utf-8") as f:
    gt_on_disk = json.load(f)
assert gt_on_disk["missed"] == 0
assert gt_on_disk["events_checked"] == 0
print("Confirmed: ground_truth_check.json did not exist before the graph ran, the graph's own "
      "ground_truth_node produced it via a real build_ground_truth_check() call, and "
      "readiness_gate_node correctly read that freshly-written file (not a hand-written stand-in) "
      "to reach report_ready=True.")


print("\n--- test 2: match_boundaries.json PRESENT but wrong schema -> must not crash the graph ---")
d2 = tempfile.mkdtemp(prefix="matchlens_step18_badboundaries_")
# The OLD simplified test schema this file's sibling tests used to use --
# present, valid JSON, just missing the ko_1h/ht_whistle/ko_2h/ft_whistle
# keys ground_truth.py's minute-to-video conversion needs.
_write(d2, "match_boundaries.json", {
    "boundaries": {"kickoff": {"confidence": 0.95}, "half_time": {"confidence": 0.90}},
})
_write(d2, "match_config.json", {"goals": [], "substitutions": [], "cards": []})
_write(d2, "running_summary.json", {"key_moments": []})

result2 = app.invoke(MatchState(match_dir=d2))
print(f"  report_ready={result2['report_ready']}, blocking_issues={result2['blocking_issues']}")
print(f"  ground_truth_result={result2['ground_truth_result']}")

# The whole point: no exception propagated out of app.invoke() above --
# reaching this line at all is half the proof. The rest is checking the
# node degraded the way it's supposed to.
assert result2["ground_truth_result"] is None, \
    "ground_truth_node should have caught the KeyError and returned None, not a real result"
assert not os.path.exists(os.path.join(d2, "ground_truth_check.json")), \
    "no ground_truth_check.json should exist -- build_ground_truth_check() raised before writing"
assert result2["report_ready"] is False
# build_readiness_check() independently blocks on the missing config
# verification AND the now-missing ground_truth_check.json, for its own
# separate reasons -- we just check at least one blocking issue exists,
# since the exact wording isn't this test's concern.
assert len(result2["blocking_issues"]) > 0
print("Confirmed: a malformed-but-present match_boundaries.json raised KeyError inside "
      "build_ground_truth_check(), ground_truth_node caught it and returned ground_truth_result=None "
      "instead of crashing the whole graph, and readiness_gate correctly blocked afterward for its "
      "own real reasons -- exactly the 'one bad file degrades gracefully' behavior the product needs "
      "to run fully unattended.")


print("\n--- test 3: totally immature match_dir (no match_config.json / running_summary.json at all) ---")
d3 = tempfile.mkdtemp(prefix="matchlens_step18_immature_")
# Deliberately empty -- simulates a match that hasn't finished Step 1d /
# window aggregation yet. This is the FileNotFoundError branch, not KeyError.

result3 = app.invoke(MatchState(match_dir=d3))
print(f"  report_ready={result3['report_ready']}, blocking_issues={result3['blocking_issues']}")
print(f"  ground_truth_result={result3['ground_truth_result']}")

assert result3["ground_truth_result"] is None
assert result3["report_ready"] is False
assert any("not verified" in issue.lower() or "boundar" in issue.lower()
           for issue in result3["blocking_issues"])
print("Confirmed: an empty match_dir (no inputs at all) does not crash the graph either -- "
      "ground_truth_node caught the FileNotFoundError, and readiness_gate still ran and blocked "
      "for its own independent, correctly-worded reasons.")


print("\n--- test 4: real-data regression -- the actual Gorleston vs Tilbury fixture through the FULL match graph ---")
d4 = tempfile.mkdtemp(prefix="matchlens_step18_gorleston_")
shutil.copytree("tests/fixtures/gorleston_vs_tilbury_gt", d4, dirs_exist_ok=True)

# Same "prove the graph produced this, not the fixture" check as test 1 --
# this fixture DOES ship a ground_truth_check.json (test_step16 needed the
# already-fixed version present to test build_readiness_check() in
# isolation), so here we delete it first to prove THIS run recomputes it
# fresh via the graph, rather than silently reusing a stale copy.
stale_gt_path = os.path.join(d4, "ground_truth_check.json")
if os.path.exists(stale_gt_path):
    os.remove(stale_gt_path)

result4 = app.invoke(MatchState(match_dir=d4))
print(f"  report_ready={result4['report_ready']}")
print(f"  blocking_issues={result4['blocking_issues']}")
print(f"  ground_truth_result={result4['ground_truth_result']}")

# The real, manually-verified numbers for this match (same ones test_step16's
# regression test locks in against build_ground_truth_check() directly --
# this test locks in the same numbers reached through the GRAPH instead).
gtr = result4["ground_truth_result"]
assert gtr["events_checked"] == 16
assert gtr["confirmed"] == 0
assert gtr["partial"] == 3
# Step 19 update: "missed" is goals-only now (see ground_truth.py's
# has_nearby_structural_context() docstring) -- all 3 real goals were found
# (partial), so missed is correctly 0, not 13. The 13 real subs/cards land
# in the new informational fact_only_no_context bucket instead, since this
# match genuinely has zero key_moments of any type near any of them.
assert gtr["missed"] == 0
assert gtr["fact_only_context_available"] == 0
assert gtr["fact_only_no_context"] == 13

with open(stale_gt_path, encoding="utf-8") as f:
    gt_on_disk_4 = json.load(f)
assert gt_on_disk_4["missed"] == 0, "the freshly-written file on disk should match state"
assert gt_on_disk_4["fact_only_no_context"] == 13

# report_ready is STILL False here -- but now for only ONE real reason, not
# two. Before Step 19, both "match_config.json not verified" AND "ground
# truth: 13 missed" blocked this match. Step 19 removes the second reason
# (subs/cards never needed video corroboration to be reported correctly),
# leaving the genuinely-unrelated verification gap as the sole blocker --
# proving the ground-truth fix didn't accidentally paper over a DIFFERENT
# real issue this fixture also carries.
assert result4["report_ready"] is False
issues_lower = [i.lower() for i in result4["blocking_issues"]]
assert any("not verified" in i for i in issues_lower), result4["blocking_issues"]
assert not any("ground truth" in i for i in issues_lower), (
    "ground truth should no longer block this match post-Step-19 -- "
    f"got {result4['blocking_issues']}"
)
# WHY .get() here, not result4["synthesis_result"]: LangGraph's invoke()
# only returns channels that were actually written by some node during this
# run -- synthesize_node never ran on the insufficient_data path, so the
# key is simply absent from the returned dict, not present-with-None. This
# is a real, slightly surprising LangGraph behavior worth noting in the
# glossary: MatchState's Optional[dict]=None default describes what a field
# means before any node has run, but invoke()'s return value is closer to
# "the diff of channels written this run" than "the full state object".
assert result4.get("synthesis_result") is None, \
    "synthesize_node should never have run -- this match routed to insufficient_data"
print("Confirmed: running the REAL Gorleston vs Tilbury match data through the full match graph "
      "(ground_truth -> readiness_gate -> insufficient_data) reproduces the exact real numbers "
      "(16 checked, 0 confirmed, 3 partial, 0 missed post-Step-19, 13 fact_only_no_context) that "
      "were previously only ever reached by calling build_ground_truth_check() directly in "
      "test_step16 -- proving the new node in the graph is wired to the same real function, not "
      "a parallel copy of its logic.")

print("\nAll test_step18 scenarios passed.")
