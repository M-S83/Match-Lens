---
name: match-analysis
description: >
  Full pipeline for producing tactical match analysis reports from football match
  footage. Covers frame extraction, source profile classification, adaptive multi-fps
  escalation, tiered window analysis with per-frame confidence scoring, targeted
  re-runs, result family gating, pass network inference, pressing intensity scoring,
  defensive line measurement, set piece detail, ground truth validation, tactical
  report writing, opposition report writing, flagged moments logging, and confidence
  reliability reporting. Applies consistent evidence/confidence discipline to match,
  opposition, and player analysis. Use this skill whenever analysing match footage,
  reviewing frames, writing tactical reports, opposition reports, or flagged moments
  from video. Also use when the user mentions "run analysis", "analyse the match",
  "next match", "start agents", or wants to produce any match report from video.
---

# Match Analysis Pipeline

## Product Context

**Match Lens** is a service-first football analysis product for non-league clubs
without a full-time analyst department. The offer is analyst-style insight from
footage clubs already have -- delivered as written reports, not dashboards.

Three modular products:
- **Match Lens Tactical** -- anchor product, full match breakdown
- **Match Lens Opposition** -- add-on, opponent-facing scouting report
- **Match Lens Moments** -- add-on, timestamped key moments

Current operating model: receive footage -> receive match details -> run pipeline
-> review outputs -> deliver reports. Validate demand at service stage before
building self-serve software.

Customer footage is not used for model training. Pipeline improvement is done
manually using internal data only.

**Two-layer analysis model.**
Layer 1 -- 1fps full-match structural scan (always on).
Layer 2 -- targeted higher-fps confirmation (rule-based escalation only, Step 3i).

**1fps is the baseline.** Shape, territory, line height, pressing behaviour, and
recurring patterns are all readable at 1fps. Higher frame rates are used only when
a finding is tagged for escalation by event type, importance, or uncertainty.
The principle: capture the right patterns, not every frame.

**Source-aware interpretation.** The pipeline classifies the footage source type
(Step 1f) and scores its visibility dimensions. A finding's validity depends on
BOTH whether the fps used is sufficient AND whether the source type supports the
claim. 1fps from stable tactical-wide footage supports strong shape/territory
findings. 1fps from ball-follow footage supports only near-ball local findings.
This logic applies consistently to match, opposition, and player analysis.

**Descriptive only.** Reports describe what was visible and what repeated.
They never instruct, advise, judge, or suggest solutions. This is enforced at
the prompt level in Step 4, not left to style discretion.

---

Produces structured output files from a single match video:
- `tactical_report.md` -- full tactical breakdown of both teams
- `opposition_report_[team].md` -- opponent-facing scouting report
- `flagged_moments.md` -- 10-20 individually timestamped key moments
- `pass_network.md` -- full pass sequence log, network map, and pattern analysis

The pipeline runs in tiers. All windows are scanned at 1fps with a single agent.
Only low-confidence frames and event windows receive additional passes.
Token cost stays proportional to analytical uncertainty.

**Critical rule:** Write each JSON to disk immediately after completion.
Never rely on conversation memory between steps. If a window fails, re-run
only that window -- all other windows are already saved.

---

## Directory Structure

```
[match_dir]/
├-- frames/                                   ← extracted at 1fps by Step 1
├-- agent_logs/
│   ├-- agent_01_[start]-[end]min.json        ← Tier 1 scan output
│   ├-- agent_01_[start]-[end]min_rerun.json  ← Targeted re-run output (if needed)
│   ├-- agent_01_[start]-[end]min_deep_a.json ← Deep scan Agent A (event windows only)
│   ├-- agent_01_[start]-[end]min_deep_b.json ← Deep scan Agent B (event windows only)
│   ├-- agent_01_[start]-[end]min_merged.json ← Final merged output
│   └-- ...
├-- match_boundaries.json                     ← KO1, HT whistle, KO2, FT whistle timestamps
├-- window_plan.json                          ← Generated window list (live play only)
├-- teamsheet_api_raw.json                    ← Raw API response, never edited
├-- match_config.json                         ← Human-verified team sheet, source of truth for all agents
├-- source_profile.json                       ← Step 1f: source type, visibility scores, split_aware flag
├-- result_family_gates.json                  ← Step 1f: per-family gate states (allowed/downgraded)
├-- rerun_queue.json                          ← Low-confidence frames queued for re-run
├-- running_summary.json                      ← Accumulates data across all windows
├-- pass_sequences.json                       ← Pass chains accumulated across all windows
├-- confirmation_queue.json                   ← Escalation queue with fps tiers and rerun windows
├-- ground_truth_check.json                   ← Event validation results
├-- report_readiness.json                     ← Pipeline gate -- must be ready=true before Step 4
├-- confidence_reliability_report.json        ← Source limitations, gated families, evidence tier summary
├-- deep_skill_metrics.json                   ← Step 3k: performance metrics derived from findings
├-- job_log.json                              ← Runtime and cost metrics for this job
├-- tactical_report.md
├-- opposition_report_[team].md
├-- flagged_moments.md
└-- pass_network.md
```

---

## Pipeline Invariants

The pipeline relies on a small number of explicit assumptions about how
it is used. Violating any of these will produce subtle bugs that may
not surface immediately.

### One run per match directory

The pipeline is designed to be run exactly once per match directory,
start to finish. No step expects to be re-invoked on a directory that
already contains its outputs.

The most consequential dependency on this invariant is in
confirmation queue handling (see AUDIT.md F8):

- Embedded per-window `confirmation_queue` arrays inside
  `agent_*_merged.json` files are write-once. They are produced by
  `accumulator.py` and read only by `escalation_router.py` during its
  consolidation pass.
- The standalone `confirmation_queue.json` at the match directory root
  is the canonical source of truth for confirmation state.
- `setpiece_writeback.py` marks items as `resolved: true` only in the
  standalone file. The embedded copies retain their original
  unresolved state.

If `escalation_router.py` is ever re-invoked on a match directory
where it has already run, the embedded (still-unresolved) queues will
be re-consolidated and items that were resolved by writeback will
re-appear in the standalone queue as unresolved. Under the one-run
invariant this cannot happen.

Future work that violates this invariant (recovery passes, partial
re-processing, repeated confirmation rounds) must either reconcile the
embedded queues with the standalone resolutions or skip the
re-consolidation step for already-resolved items.

---

## Step 1 -- Frame Extraction

```python
import cv2, os

PYTHON = r"C:\Users\dbmux\AppData\Local\Programs\Python\Python313\python.exe"

video = r"[MATCH_DIR]\[VIDEO_FILE].mp4"
out_dir = r"[MATCH_DIR]\frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(video)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
dur = total / fps
print(f"Extracting {int(dur)} frames at 1fps from {dur/60:.1f} min video...")

count = 0
for sec in range(int(dur) + 1):
    frame_num = int(sec * fps)
    if frame_num >= total:
        break
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if ret:
        m, s = divmod(sec, 60)
        cv2.imwrite(
            os.path.join(out_dir, f"frame_{m:02d}m{s:02d}s.jpg"),
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )
        count += 1
        if count % 500 == 0:
            print(f"  {count} frames ({sec//60}m{sec%60}s)")

cap.release()
print(f"Done: {count} frames at 1fps")
```

Frame naming: `frame_XXmYYs.jpg` (e.g. `frame_17m09s.jpg`)

---

## Step 1a -- Container Analysis

Run before frame extraction. Reads the raw container structure via ffprobe
to determine whether video seeking is reliable and where segment boundaries
fall. Takes 2-5 seconds regardless of video length.

**Why it runs first:** if the container has segment joins or a long keyframe
interval, seeking to arbitrary timestamps (as used in higher-fps extraction)
may land on the wrong frame. This step detects those conditions before any
extraction runs, so downstream steps know what they can and cannot do safely.

**What it detects:**

| Condition | Detected via | Downstream effect |
|---|---|---|
| Segment joins (Veo-style) | DTS gaps >500ms | Windows snap to boundaries; seek blocked near joins |
| Fragmented container (phone) | major_brand iso5/iso6 | seek_reliable=False; remux before extraction |
| Long keyframe interval (DJI) | Keyframe interval >4s | higher_fps_extraction_safe=False; use ffmpeg not OpenCV |
| Edit list (iOS/Android) | elst atom present | pts may be offset; verify after first extraction |
| Clean container | No discontinuities | seek_reliable=True; all extraction modes available |

**Run:**

```python
from container_analyser import run_step_1a

MATCH_DIR  = r"[MATCH_DIR]"
VIDEO_FILE = r"[VIDEO_PATH]"

container_profile = run_step_1a(MATCH_DIR, VIDEO_FILE)
```

Writes: `container_profile.json`

**Key output fields:**

```json
{
  "seek_reliable":              true,
  "discontinuity_count":        0,
  "boundary_timestamps_s":      [],
  "source_pattern":             "clean_continuous",
  "is_fragmented":              false,
  "max_keyframe_interval_s":    2.0,
  "higher_fps_extraction_safe": true,
  "remux_recommended":          false,
  "notes":                      ["Clean container. Seeking reliable."]
}
```

**source_pattern values:**

| Pattern | Meaning |
|---|---|
| `clean_continuous` | No discontinuities. Seeking fully reliable. |
| `periodic_5min_segments` | Veo-style segment joins every ~5 minutes |
| `periodic_15min_segments` | GoPro chapter joins every ~17 minutes |
| `single_join` | Two recordings concatenated |
| `fragmented_with_edit_list` | Phone recording with edit list |
| `long_gop` | Long keyframe interval -- seek may land between keyframes |
| `irregular_discontinuities` | Broadcast capture or unpredictable joins |

**Step 1c reads `boundary_timestamps_s`** to snap window edges away from joins.
**Step 3i reads `seek_reliable` and `higher_fps_extraction_safe`** before
attempting higher-fps extraction. If either is false, it uses
`ffmpeg -ss [time] -i` instead of OpenCV seek.

**Note on synthetic testing:** ffmpeg concat always normalises DTS, so
Veo-style segment joins cannot be reproduced synthetically. The DTS >500ms
threshold is correct in principle but requires validation against real Veo
footage on first use.

---

## Step 1b -- Match Boundary Detection

Run before any analysis windows are allocated. Identifies the four live-play
boundaries so that halftime, pre-match warmup, and post-match frames are never
passed to analysis agents.

**Boundaries to detect:**

| Boundary | Signal |
|---|---|
| `ko_1h` | Both teams in formation, ball at centre spot, referee present |
| `ht_whistle` | Players walking toward tunnel, referee moving to centre, play stopped |
| `ko_2h` | Teams back in formation, ball at centre spot, attack direction swapped |
| `ft_whistle` | Players celebrating / dejected / handshakes, no restart |

**Two-phase scan:**

Phase 1 -- coarse scan at every 30 seconds across the full video duration.
Identify approximate 2-minute regions where each boundary likely falls.

Phase 2 -- fine scan at every 5 seconds within each identified region.
Confirm the precise boundary frame.

**Boundary detection prompt:**

```
You are identifying match boundaries in a football video to determine which frames
contain live play. Scan the frames listed below and identify the timestamp of each
boundary event. You are looking for four moments only -- do not analyse tactics.

[MATCH]: [Home] vs [Away]. [Home] in [COLOUR], [Away] in [COLOUR].

PHASE 1 -- COARSE SCAN
View every 30 seconds from frame_00m00s.jpg to [LAST_FRAME].jpg.

For each boundary, identify the approximate 2-minute region where it occurs:

KO_1H    -- both teams in starting positions, ball at centre spot, referee present,
           no warmup or broadcast graphics filling the frame
HT_WHISTLE -- players stopping play, walking toward touchline or tunnel,
             referee moving to centre circle area
KO_2H    -- teams back in position after halftime, ball at centre spot,
           attack direction reversed from first half
FT_WHISTLE -- final whistle blown, players reacting (celebrations, handshakes,
             sitting on pitch), no subsequent centre restart

Output ONLY raw JSON:

{
  "coarse_scan_complete": true,
  "approximate_regions": {
    "ko_1h":       {"search_from": "[MMmSSs]", "search_to": "[MMmSSs]"},
    "ht_whistle":  {"search_from": "[MMmSSs]", "search_to": "[MMmSSs]"},
    "ko_2h":       {"search_from": "[MMmSSs]", "search_to": "[MMmSSs]"},
    "ft_whistle":  {"search_from": "[MMmSSs]", "search_to": "[MMmSSs]"}
  },
  "notes": "[anything unusual -- e.g. broadcast delay, VEO cut, missing frames]"
}
```

After Phase 1 JSON is returned, run Phase 2 for each boundary:

```
Fine-scan to confirm [BOUNDARY_NAME].

View every 5 seconds from [search_from] to [search_to].

Identify the single frame where [BOUNDARY_DESCRIPTION]:
  ko_1h       -- first frame where the ball is at centre spot with both teams set
  ht_whistle  -- first frame where players have clearly stopped play (referee central,
                players moving off)
  ko_2h       -- first frame where ball is at centre spot for second-half kickoff
  ft_whistle  -- last frame of live play (before any post-match activity)

Output ONLY raw JSON:

{
  "boundary": "[boundary_name]",
  "confirmed_frame": "[frame_XXmYYs.jpg]",
  "confirmed_timestamp_seconds": [number],
  "confidence": [0.0-1.0],
  "notes": "[any ambiguity]"
}
```

Run Phase 2 four times -- once per boundary. Write all four results before continuing.

**Write `match_boundaries.json`:**

```json
{
  "match": "[Home vs Away]",
  "video_duration_seconds": [number],
  "boundaries": {
    "ko_1h":      {"frame": "frame_XXmYYs.jpg", "seconds": [N], "confidence": [0.0-1.0]},
    "ht_whistle": {"frame": "frame_XXmYYs.jpg", "seconds": [N], "confidence": [0.0-1.0]},
    "ko_2h":      {"frame": "frame_XXmYYs.jpg", "seconds": [N], "confidence": [0.0-1.0]},
    "ft_whistle": {"frame": "frame_XXmYYs.jpg", "seconds": [N], "confidence": [0.0-1.0]}
  },
  "live_play_seconds": {
    "first_half":  [ht_whistle_seconds - ko_1h_seconds],
    "second_half": [ft_whistle_seconds - ko_2h_seconds],
    "total":       [number]
  },
  "dead_time_seconds": {
    "pre_match":   [ko_1h_seconds],
    "halftime":    [ko_2h_seconds - ht_whistle_seconds],
    "post_match":  [video_duration_seconds - ft_whistle_seconds]
  }
}
```

**If any boundary confidence is below 0.8**, flag it and run Phase 2 again with
a 1-second scan interval within the same search region before writing the file.

**Do not proceed to Step 2b until all four boundaries are confirmed.**

---

## Step 1c -- Window Plan Generation

Read `match_boundaries.json` and generate the analysis window list. Windows cover
live play only. Halftime and any dead time are excluded entirely.

**Prerequisite:** Step 1a must have run. `container_profile.json` is read here to
snap window edges to any segment boundaries. If it does not exist the step
still runs but without boundary alignment.

```python
import json, os, math

MATCH_DIR    = r"[MATCH_DIR]"
BOUNDARY_FILE = os.path.join(MATCH_DIR, "match_boundaries.json")
WINDOW_FILE   = os.path.join(MATCH_DIR, "window_plan.json")
WINDOW_SECONDS = 300  # 5 minutes

with open(BOUNDARY_FILE) as f:
    b = json.load(f)

ko_1h  = b["boundaries"]["ko_1h"]["seconds"]
ht     = b["boundaries"]["ht_whistle"]["seconds"]
ko_2h  = b["boundaries"]["ko_2h"]["seconds"]
ft     = b["boundaries"]["ft_whistle"]["seconds"]

def seconds_to_frame(s):
    m, sec = divmod(int(s), 60)
    return f"frame_{m:02d}m{sec:02d}s.jpg"

# Load container profile if available -- used to snap windows to segment boundaries
container_profile_path = os.path.join(MATCH_DIR, "container_profile.json")
container_boundaries = []
if os.path.exists(container_profile_path):
    with open(container_profile_path) as f:
        cp = json.load(f)
    container_boundaries = cp.get("boundary_timestamps_s", [])
    if container_boundaries:
        print(f"  Container boundaries loaded: {container_boundaries}")
    else:
        print(f"  Container: clean -- no boundary snapping needed")
else:
    print(f"  ⚠  container_profile.json not found -- run Step 1a first")

SNAP_THRESHOLD = 15  # seconds -- snap window edge to boundary if within this range

def make_windows(start, end, half_label, boundaries=None):
    """
    Generate 5-minute analysis windows covering live play.
    If container_profile.json has segment boundaries, window edges snap to them
    so higher-fps extraction never targets a timestamp near a join.
    """
    windows = []
    cursor  = start

    while cursor < end:
        natural_end = min(cursor + WINDOW_SECONDS, end)

        # Snap end to any nearby container boundary
        if boundaries:
            for b in boundaries:
                if cursor < b < natural_end:
                    if abs(b - natural_end) <= SNAP_THRESHOLD:
                        natural_end = b  # pull end back to boundary
                        break
                    elif abs(b - cursor) <= SNAP_THRESHOLD:
                        cursor = b       # push start forward to boundary
                        natural_end = min(cursor + WINDOW_SECONDS, end)
                        break

        boundary_nearby = boundaries and any(
            abs(b - cursor) <= SNAP_THRESHOLD or abs(b - natural_end) <= SNAP_THRESHOLD
            for b in boundaries
        )

        windows.append({
            "half":              half_label,
            "start_s":          cursor,
            "end_s":            natural_end,
            "start_frame":      seconds_to_frame(cursor),
            "end_frame":        seconds_to_frame(natural_end - 1),
            "duration_s":       natural_end - cursor,
            "frame_count":      natural_end - cursor,  # 1fps = 1 frame per second
            "event_window":     False,
            "deep_scan":        False,
            "boundary_nearby":  boundary_nearby,
        })
        cursor = natural_end

    return windows

first_half_windows  = make_windows(ko_1h, ht, "1H", container_boundaries)
second_half_windows = make_windows(ko_2h, ft, "2H", container_boundaries)
all_windows = first_half_windows + second_half_windows

# Number sequentially across both halves
for i, w in enumerate(all_windows):
    w["agent_id"] = f"{i+1:02d}"
    label_start = w["start_s"] - (ko_1h if w["half"] == "1H" else ko_2h)
    label_end   = w["end_s"]   - (ko_1h if w["half"] == "1H" else ko_2h)
    m_s, s_s = divmod(int(label_start), 60)
    m_e, s_e = divmod(int(label_end),   60)
    w["label"] = f"{w['half']} {m_s:02d}:{s_s:02d}-{m_e:02d}:{s_e:02d}"

plan = {
    "match":              b["match"],
    "total_windows":      len(all_windows),
    "first_half_windows": len(first_half_windows),
    "second_half_windows":len(second_half_windows),
    "halftime_excluded_seconds": int(ko_2h - ht),
    "pre_match_excluded_seconds": int(ko_1h),
    "post_match_excluded_seconds": int(b["video_duration_seconds"] - ft),
    "windows": all_windows
}

with open(WINDOW_FILE, "w") as f:
    json.dump(plan, f, indent=2)

print(f"Window plan written: {len(all_windows)} windows")
print(f"Halftime excluded: {int(ko_2h - ht)}s ({(ko_2h-ht)/60:.1f} min)")
print(f"Pre-match excluded: {int(ko_1h)}s")
print(f"Post-match excluded: {int(b['video_duration_seconds'] - ft)}s")
for w in all_windows:
    print(f"  [{w['agent_id']}] {w['label']} | {w['start_frame']} -> {w['end_frame']} | {w['frame_count']} frames")
```

After generating `window_plan.json`, mark event windows manually:

```python
# Mark goal and sub windows -- update agent IDs to match your window_plan.json output
EVENT_WINDOWS = {
    "goal":  ["[NN]", "[NN]"],   # agent IDs where goals occur
    "sub":   ["[NN]", "[NN]"]    # agent IDs where subs occur
}

with open(WINDOW_FILE) as f:
    plan = json.load(f)

for w in plan["windows"]:
    if w["agent_id"] in EVENT_WINDOWS["goal"] or w["agent_id"] in EVENT_WINDOWS["sub"]:
        w["event_window"] = True
        w["deep_scan"] = True

with open(WINDOW_FILE, "w") as f:
    json.dump(plan, f, indent=2)

print("Event windows marked.")
```

**Do not proceed to Step 2 until `window_plan.json` is written and event windows are marked.**

---

## Step 1d -- Team Sheet Verification

Retrieve whatever data the API can provide, then **stop and verify manually**
before analysis begins. The pipeline does not hard-block on missing enrichment --
it degrades gracefully. Missing lineup data lowers the player ID confidence
ceiling; it does not stop the job.

**Mandatory fields** -- pipeline hard-blocks if any of these are missing or
still set to [FILL IN]:
- match identity (home team, away team, competition, date)
- kit colours (home, away, home GK, away GK)
- focus team

**Optional enrichment** -- populated from API if available, otherwise null.
When null, downstream effects are logged in report_readiness.json:
- venue, ht_score, ft_score
- lineups, substitutions, goals, cards
- shirt numbers

**Player ID ceiling by enrichment level:**

| Enrichment level | What's available | Max player ID confidence |
|---|---|---|
| Full | Lineups + shirt numbers from API | Confirmed |
| Partial | Score and goals but no lineups | Probable |
| Identity only | Match identity only | Tentative |

**API retrieval:**

```python
import json, os, requests

MATCH_DIR  = r"[MATCH_DIR]"
API_KEY    = "[YOUR_API_KEY]"
FIXTURE_ID = "[FIXTURE_ID]"   # confirmed from API fixture search

# Example using API-Football (api-sports.io)
url = "https://v3.football.api-sports.io/fixtures"
headers = {"x-apisports-key": API_KEY}
params  = {"id": FIXTURE_ID}

r = json.loads(requests.get(url, headers=headers, params=params).text)

raw_path = os.path.join(MATCH_DIR, "teamsheet_api_raw.json")
with open(raw_path, "w") as f:
    json.dump(r, f, indent=2)

print(f"Raw API response written to teamsheet_api_raw.json")
print("Now review the verification block below before proceeding.")
```

Save `teamsheet_api_raw.json` immediately. Never edit this file -- it is the
permanent record of what the API returned.

**Verification block -- print to terminal for manual review:**

```python
def print_verification_block(api_data, match_dir):
    """
    Parse API response and print a clean verification block.
    Adapt field paths to whichever API you are using.
    """
    fixture  = api_data["response"][0]
    home     = fixture["teams"]["home"]["name"]
    away     = fixture["teams"]["away"]["name"]
    date     = fixture["fixture"]["date"]
    venue    = fixture["fixture"]["venue"]["name"]
    ht_score = fixture["score"]["halftime"]
    ft_score = fixture["score"]["fulltime"]
    lineups  = fixture.get("lineups", [])
    events   = fixture.get("events", [])

    def format_lineup(team_lineup):
        out = []
        for p in team_lineup.get("startXI", []):
            pl = p["player"]
            out.append(f"  {pl.get('pos','??'):2s}  #{pl.get('number','?'):2}  {pl.get('name','Unknown')}")
        out.append("")
        bench = [p["player"] for p in team_lineup.get("substitutes", [])]
        out.append(f"  Bench: {', '.join(f\"#{p.get('number','?')} {p.get('name','')}\" for p in bench)}")
        return "\n".join(out)

    subs   = [e for e in events if e["type"] == "subst"]
    cards  = [e for e in events if e["type"] == "Card"]
    goals  = [e for e in events if e["type"] == "Goal"]

    null_fields = []
    if not lineups:          null_fields.append("lineups")
    if ft_score["home"] is None: null_fields.append("final score")

    block = f"""
TEAM SHEET VERIFICATION -- MANUAL CHECK REQUIRED
{'-' * 60}
Match:       {home} vs {away}
Date:        {date}
Venue:       {venue}
HT Score:    {ht_score['home']}-{ht_score['away']}
FT Score:    {ft_score['home']}-{ft_score['away']}

Goals:
{chr(10).join(f"  {e['time']['elapsed']}' {e['team']['name']} -- {e['player']['name']}" for e in goals) or '  None recorded'}

HOME -- {home}:
{format_lineup(lineups[0]) if lineups else '  API did not return lineup'}

AWAY -- {away}:
{format_lineup(lineups[1]) if len(lineups) > 1 else '  API did not return lineup'}

Substitutions:
{chr(10).join(f"  {e['time']['elapsed']}' {e['team']['name']}: {e['assist']['name']} ON for {e['player']['name']}" for e in subs) or '  None recorded'}

Cards:
{chr(10).join(f"  {e['time']['elapsed']}' {e['team']['name']} {e['detail']} -- {e['player']['name']}" for e in cards) or '  None recorded'}

{'-' * 60}
API SOURCE: api-football  |  FIXTURE ID: {fixture['fixture']['id']}
{f'⚠  MISSING FIELDS: {", ".join(null_fields)}' if null_fields else '✓  No missing fields detected'}

KIT COLOURS (not supplied by API -- fill in manually):
  Home kit: [COLOUR]
  Away kit: [COLOUR]
  Home GK:  [COLOUR]
  Away GK:  [COLOUR]

ATTACK DIRECTION (fill in from video):
  1H: [team] attack [left/right]
  2H: [team] attack [left/right]
{'-' * 60}
CHECK EACH ITEM:
  [ ] Correct fixture (not a different match between same teams)
  [ ] Starting XI matches actual lineup, not predicted
  [ ] Shirt numbers correct
  [ ] Substitutions complete with correct timings
  [ ] Goals and scorers correct
  [ ] Kit colours filled in above
  [ ] Attack directions filled in above

When satisfied, type CONFIRM to write match_config.json and continue.
Any corrections? Edit match_config_draft.json before confirming.
"""
    print(block)

    draft_path = os.path.join(match_dir, "match_config_draft.json")
    # Write a draft for optional manual editing before confirmation
    draft = {
        "match":        f"{home} vs {away}",
        "date":         date,
        "venue":        venue,
        "ht_score":     f"{ht_score['home']}-{ht_score['away']}",
        "ft_score":     f"{ft_score['home']}-{ft_score['away']}",
        "home_team":    home,
        "away_team":    away,
        "home_kit":     "[FILL IN]",
        "away_kit":     "[FILL IN]",
        "home_gk_kit":  "[FILL IN]",
        "away_gk_kit":  "[FILL IN]",
        "attack_direction_1h": "[FILL IN]",
        "attack_direction_2h": "[FILL IN]",
        "focus_team":   "[FILL IN]",
        "lineups":      lineups,
        "substitutions": subs,
        "goals":        goals,
        "cards":        cards,
        "report_level": "standard",
        "verified":     False
    }
    with open(draft_path, "w") as f:
        json.dump(draft, f, indent=2)
    print(f"\nDraft written to match_config_draft.json -- edit if needed, then confirm.")
```

**Confirmation step -- run after review:**

```python
def confirm_team_sheet(match_dir):
    """
    Call this after manual review. Copies the (optionally edited) draft
    to match_config.json and marks it as verified.
    """
    draft_path  = os.path.join(match_dir, "match_config_draft.json")
    config_path = os.path.join(match_dir, "match_config.json")

    with open(draft_path) as f:
        config = json.load(f)

    # Guard: only mandatory fields must be filled -- optional enrichment can be null
    MANDATORY = ["match", "home_team", "away_team", "competition", "date",
                 "home_kit", "away_kit", "home_gk_kit", "away_gk_kit", "focus_team"]
    unfilled = [k for k in MANDATORY if config.get(k) in (None, "[FILL IN]", "")]
    if unfilled:
        print(f"⚠  Cannot confirm -- mandatory fields missing: {', '.join(unfilled)}")
        print("Edit match_config_draft.json and run confirm_team_sheet() again.")
        return False

    # Determine enrichment level and log it
    has_lineups = bool(config.get("lineups"))
    has_score   = config.get("ft_score") not in (None, "[FILL IN]", "")
    config["enrichment_level"] = (
        "full"          if has_lineups else
        "partial"       if has_score  else
        "identity_only"
    )
    config["player_id_ceiling"] = (
        "confirmed" if config["enrichment_level"] == "full"    else
        "probable"  if config["enrichment_level"] == "partial" else
        "tentative"
    )

    config["verified"] = True
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("✓  match_config.json written and verified.")
    print("Pipeline may now proceed to Step 2.")
    return True
```

**Rules:**
- `match_config.json` is the only team sheet file agents ever read from
- `teamsheet_api_raw.json` is never modified after writing
- If the API returns the wrong fixture, discard the raw file, search again, restart Step 1d
- Missing optional fields (lineup, subs, cards) -> leave as null, do not force-fill
- Kit colours are always filled in manually -- no API supplies these reliably
- Attack directions are detected automatically in Step 1e from KO frames -- do not fill manually
- enrichment_level and player_id_ceiling are written automatically by confirm_team_sheet()

**Do not proceed to Step 1e until `match_config.json` exists and `verified` is `true`.**

---

## Step 1e -- Attack Direction Detection

Uses the confirmed KO1 and KO2 frames from `match_boundaries.json` and the
verified kit colours from `match_config.json` to determine which way each team
is attacking in each half. Removes the last manual input from the pipeline.

**Why kickoff frames work:** at KO1 and KO2, both teams are set up in their
starting formation with the ball at the centre spot. The team in the attacking
half (beyond the centre line toward the opponent's goal) is attacking in that
direction. Kit colours are already confirmed, so team assignment is unambiguous.

**Detection prompt:**

```
You are determining attack direction from a single football kickoff frame.
Kit colours are confirmed -- use them to identify which team is which.

HOME TEAM kit: [home_kit] (from match_config.json)
AWAY TEAM kit: [away_kit] (from match_config.json)

View this frame: [KO_FRAME_PATH]

The ball is at the centre spot. One team occupies the left half of the pitch,
one team occupies the right half.

Answer these questions only -- no tactical analysis:

1. Which team (Home or Away) is set up in the LEFT half of the pitch?
2. Which team (Home or Away) is set up in the RIGHT half of the pitch?
3. The team in the LEFT half is attacking toward the RIGHT goal -- confirm true/false.
4. The team in the RIGHT half is attacking toward the LEFT goal -- confirm true/false.

Output ONLY raw JSON:

{
  "frame": "[frame_name]",
  "half": "[1H or 2H]",
  "left_half_team": "[Home or Away]",
  "right_half_team": "[Home or Away]",
  "home_attacking_direction": "[left or right]",
  "away_attacking_direction": "[left or right]",
  "confidence": [0.0-1.0],
  "notes": "[anything that made this ambiguous, or null]"
}
```

Run this prompt twice -- once with the KO1 frame, once with the KO2 frame.

**Attack direction writer -- run after both prompts return:**

```python
import json, os

MATCH_DIR    = r"[MATCH_DIR]"
CONFIG_PATH  = os.path.join(MATCH_DIR, "match_config.json")
BOUNDARY_FILE = os.path.join(MATCH_DIR, "match_boundaries.json")

def write_attack_directions(ko1_result, ko2_result, match_dir):
    config_path = os.path.join(match_dir, "match_config.json")

    with open(config_path) as f:
        config = json.load(f)

    # Validate confidence
    for result, label in [(ko1_result, "KO1"), (ko2_result, "KO2")]:
        if result["confidence"] < 0.8:
            print(f"⚠  {label} attack direction confidence low ({result['confidence']})")
            print(f"   Notes: {result.get('notes')}")
            print(f"   Review frame manually: {result['frame']}")
            print(f"   Override by setting attack_direction_1h / attack_direction_2h manually in match_config.json")

    # Sanity check: directions should swap at HT
    if ko1_result["home_attacking_direction"] == ko2_result["home_attacking_direction"]:
        print(f"⚠  WARNING: Home team attacking same direction in both halves.")
        print(f"   This is unusual -- verify KO frames are correct before continuing.")

    config["attack_direction_1h"] = (
        f"Home attack {ko1_result['home_attacking_direction']}, "
        f"Away attack {ko1_result['away_attacking_direction']}"
    )
    config["attack_direction_2h"] = (
        f"Home attack {ko2_result['home_attacking_direction']}, "
        f"Away attack {ko2_result['away_attacking_direction']}"
    )
    config["attack_direction_source"] = "auto_detected"

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"✓  Attack directions written to match_config.json")
    print(f"   1H: {config['attack_direction_1h']}")
    print(f"   2H: {config['attack_direction_2h']}")
```

**Sanity checks built in:**
- Confidence below 0.8 -> prints a warning and the frame path for manual review
- Same direction in both halves -> hard warning (teams always swap ends at HT)
- Manual override available: set `attack_direction_1h` / `attack_direction_2h` directly in `match_config.json` if detection fails

**Update `confirm_team_sheet()` guard** -- remove `attack_direction_1h` and
`attack_direction_2h` from the `[FILL IN]` fields check, since they are now
populated by Step 1e not manual entry:

```python
# Fields that must be filled manually (kit colours only -- directions auto-detected)
MANUAL_REQUIRED = ["home_kit", "away_kit", "home_gk_kit", "away_gk_kit", "focus_team"]

unfilled = [k for k in MANUAL_REQUIRED if config.get(k) == "[FILL IN]"]
```

**Do not proceed to Step 2 until attack directions are written to `match_config.json`.**

---


---

## Step 1f -- Source Profiling

Run after attack direction is confirmed and before the match data block is
generated. Classifies the footage type, scores visibility dimensions, and
writes the result family gate states that govern every finding downstream.

**Why this runs before analysis:** the same window observation means different
things from tactical-wide vs ball-follow footage. Gating result families before
analysis runs prevents overclaiming at source, rather than correcting it later.

**Two outputs:**
- `source_profile.json` -- source type, visibility scores, split_aware flag
- `result_family_gates.json` -- per-family gate states for this job

**Step 1: Sample frames**

```python
from source_profiler import sample_frames, CLASSIFICATION_PROMPT

MATCH_DIR  = r"[MATCH_DIR]"
VIDEO_FILE = r"[VIDEO_PATH]"

sample_dir = os.path.join(MATCH_DIR, "frames", "source_samples")
samples    = sample_frames(VIDEO_FILE, sample_dir)

# Print the classification prompt -- use it with the sampled frames
print(CLASSIFICATION_PROMPT)
print("\nFrames to view:", samples)
```

**Step 2: Classification prompt** (run with sample frames)

```
You are classifying football match footage to determine the camera source type
and visibility characteristics. Examine the sample frames provided.

Classify the source type as exactly ONE of:
  tactical_wide_static         -- fixed wide-angle covering full/most of pitch
  tactical_wide_auto_tracking  -- wide angle but auto-tracking, crop may tighten
  veo_ball_tracking            -- Veo or similar system following the ball closely
  drone_high_wide              -- high-altitude drone with wide pitch coverage
  drone_follow                 -- lower drone following action
  dual_panoramic               -- two stacked half-pitch views (top + bottom)
  behind_goal                  -- fixed camera behind/near one goal
  broadcast_tv                 -- television broadcast feed
  tracking_overlay             -- tracking data overlay on video
  unknown                      -- cannot determine

Score each visibility dimension 0.0-1.0:
  full_pitch_visibility_score    -- how much of the pitch is visible typically
  weakside_visibility_score      -- how often the far/weak side is visible
  off_ball_coverage_score        -- how much off-ball player behaviour is visible
  camera_motion_score            -- degree of camera motion (0=static, 1=constant)
  zoom_variability_score         -- how much zoom changes (0=constant, 1=frequent)
  stability_score                -- overall frame stability (0=unstable, 1=perfect)
  orientation_consistency_score  -- pitch orientation consistency (0=none, 1=always)
  occlusion_score                -- player/pitch occlusion (0=none, 1=severe)
  ball_follow_bias               -- how strongly camera follows ball (0=none, 1=always)

For dual_panoramic footage: set split_aware: true.

Output ONLY raw JSON:

{
  "source_type": "...",
  "classification_confidence": 0.0,
  "split_aware": false,
  "visibility_scores": {
    "full_pitch_visibility_score": 0.0,
    "weakside_visibility_score": 0.0,
    "off_ball_coverage_score": 0.0,
    "camera_motion_score": 0.0,
    "zoom_variability_score": 0.0,
    "stability_score": 0.0,
    "orientation_consistency_score": 0.0,
    "occlusion_score": 0.0,
    "ball_follow_bias": 0.0
  },
  "source_limitations_note": "[one sentence describing the main limitation]"
}
```

**Step 3: Build source profile and gate result families**

```python
from source_profiler import build_source_profile

# classification_result = the JSON returned by the agent above
build_source_profile(MATCH_DIR, classification_result)
```

This writes both output files and prints a summary of downgraded families and their limitation notes.

**Dual panoramic handling:**
If `source_type` is `dual_panoramic`, the pipeline must:
- Set `split_aware: true` in source_profile.json (done automatically by the classifier)
- Interpret each half-pitch view independently
- Not spatially merge the two halves
- Qualify all spatial findings: "based on [home/away] half view"
- Cross-pitch spacing and territory claims are downgraded until split is confirmed
  and carry the note: 'cross-pitch spatial claim -- requires both panels in frame'

**If classification_confidence < 0.6:**
Source type defaults to `unknown`. All structural result families are downgraded.
The pipeline does not block -- it continues with conservative gates and notes the
limitation in `confidence_reliability_report.json`.

**Do not proceed to Step 2 until `source_profile.json` and `result_family_gates.json` exist.**

---

## Step 2 -- Match Data Block

Generated from `match_config.json`. Do not type this manually -- run the script
below to produce the block, then paste it into every agent prompt unchanged.

```python
import json, os

MATCH_DIR   = r"[MATCH_DIR]"
CONFIG_PATH = os.path.join(MATCH_DIR, "match_config.json")

with open(CONFIG_PATH) as f:
    c = json.load(f)

assert c.get("verified"), "match_config.json is not verified -- complete Step 1d first."

def format_team(team_name, kit, lineups, substitutions):
    lineup = next((l for l in lineups if l["team"]["name"] == team_name), None)
    if not lineup:
        return f"{team_name} -- {kit}:\n  [LINEUP NOT AVAILABLE]\n"
    players = "\n".join(
        f"  #{p['player']['number']:2}  {p['player']['name']}  ({p['player']['pos']})"
        for p in lineup["startXI"]
    )
    team_subs = [s for s in substitutions if s["team"]["name"] == team_name]
    sub_lines = "\n".join(
        f"  #{s['assist']['number']} {s['assist']['name']} on {s['time']['elapsed']}' for {s['player']['name']}"
        for s in team_subs
    ) or "  None"
    return f"{team_name} -- {kit}:\n{players}\n  Subs:\n{sub_lines}\n"

goals = "\n".join(
    f"  {g['time']['elapsed']}' {g['team']['name']} -- {g['player']['name']}"
    for g in c.get("goals", [])
) or "  None"

cards = "\n".join(
    f"  {k['time']['elapsed']}' {k['team']['name']} {k['detail']} -- {k['player']['name']}"
    for k in c.get("cards", [])
) or "  None"

home_block = format_team(c["home_team"], c["home_kit"], c["lineups"], c["substitutions"])
away_block = format_team(c["away_team"], c["away_kit"], c["lineups"], c["substitutions"])

block = f"""MATCH: {c['match']}
COMPETITION: {c.get('competition', '[FILL IN]')}
DATE: {c['date']}
VENUE: {c['venue']}
SCORE: HT {c['ht_score']} / FT {c['ft_score']}

HOME -- {home_block}
AWAY -- {away_block}
KEY EVENTS -- GOALS:
{goals}

KEY EVENTS -- CARDS:
{cards}

GK KIT: Home GK in {c['home_gk_kit']} / Away GK in {c['away_gk_kit']}
ATTACK DIRECTION 1H: {c['attack_direction_1h']}
ATTACK DIRECTION 2H: {c['attack_direction_2h']}
FOCUS TEAM: {c['focus_team']}"""

print(block)
with open(os.path.join(MATCH_DIR, "match_data_block.txt"), "w") as f:
    f.write(block)
print("\n✓  Saved to match_data_block.txt")
```

Copy the printed block into every agent prompt. Do not retype or paraphrase it.

---

## Frame-Rate and Source-Aware Analysis Philosophy

Match Lens uses a two-layer analysis model:

- **Layer 1 -- 1fps full-match structural scan** (always on)
  Used for: team shape, line structure, territory, spacing, phase clues, and
  recurring patterns. 1fps is preferred when behaviour lasts multiple seconds
  and is mainly structural.

- **Layer 2 -- targeted higher-fps confirmation** (rule-based escalation only)
  Used only when a finding is tagged for escalation by importance, uncertainty,
  or event type. Never runs on the full match.

A finding's validity depends on **both** dimensions:
1. Is the fps used sufficient for this type of tactical claim?
2. Does the source type and pitch visibility support this claim?

---

## Tactical Detection by Minimum FPS

### 1fps -- direct or repeated_pattern evidence

Suitable for structural and recurring observations that are visible across frames:
- Team shape / line structure
- Block height
- Territory / field tilt proxies
- Broad phase clues
- Width / spacing / occupation
- Repeated structural patterns across windows
- Set-piece starting structure
- Rest-defence in settled moments

### 1fps -- suggestive only (escalate if important or uncertain)

Indicated but not fully confirmable at 1fps:
- Build-up structure (confirm at 3fps)
- Pressing shape (confirm at 3fps)
- Chance-creation patterns (confirm at 3fps)
- Phase disambiguation in unstable sequences (confirm at 3fps)

### 3fps minimum -- behavioural confirmation

- Build-up structure confirmation
- Pressing shape confirmation
- Chance-pattern confirmation
- Phase disambiguation
- Rest-defence review around unstable possession moments
- Overload detection
- Unit relationship patterns (opposition and player analysis)

### 5fps minimum -- fast tactical moments

- Transitions and counterpress
- Duel windows / 1v1 outcomes
- Line-breaking pass confirmation
- Quick box-entry sequences
- Fast pressing / escape moments
- Opposition pressing triggers
- Player duels and movement timing

### 10fps -- selected clips only, never full match

- Goal / big chance detail review
- Very ambiguous moments where 5fps remains insufficient
- High-value validation clips

---

## Evidence Classes

Every finding carries an evidence_tier field:

| Tier | Meaning |
|---|---|
| `direct` | Directly observable at the fps used in a single frame or sequence |
| `repeated_pattern` | Confirmed through recurrence across multiple windows |
| `suggestive` | Indicated but not confirmable at current fps -- may need escalation |
| `escalated_confirmation` | Confirmed after fps escalation to Layer 2 |

---

## Escalation Principles

A finding is tagged for escalation if any of the following apply:
- Tactically important moment (goal, transition, key chance)
- Confidence below window threshold (< 0.7)
- Visibility poor, cluttered, or occluded
- Two competing interpretations exist
- Event type likely involves fast action

**Escalate to 5fps:** transitions, turnovers, box entries, line-breaking actions,
duel clusters, defensive line breaks, goalmouth scrambles.

**Escalate to 3fps:** pressing shape, build-up under pressure, overloads,
repeated but unclear patterns, phase ambiguity, GK aerial actions.

**Remain 1fps:** stable recycling possession, settled low block with high clarity,
long stable territorial periods.

**Rerun window padding:** default 5s before and 5s after the tagged moment.
Build-up patterns allow up to 8s before. Padding is configurable per tag type
in `source_profiles_config.json`.

**Cap rule:** maximum 10 escalation items per match. Exception: if the match
contains more than 10 goals, all goals are processed regardless of cap.
Non-goal high-priority items fill any remaining slots.

---

## Source Profile Framework

Defined in `source_profiles_config.json`. Ten source types supported:

| Code | Description |
|---|---|
| `tactical_wide_static` | Fixed wide-angle, full-pitch coverage |
| `tactical_wide_auto_tracking` | Auto-tracking wide, crop may tighten |
| `veo_ball_tracking` | Ball-follow system, near-ball only |
| `drone_high_wide` | High-altitude drone, full pitch |
| `drone_follow` | Follow drone, local coverage |
| `dual_panoramic` | Two stacked half-pitch views |
| `behind_goal` | Fixed behind one goal |
| `broadcast_tv` | Television feed |
| `tracking_overlay` | Tracking data overlaid |
| `unknown` | Could not be classified |

For each source type, result families are in one of three states:
- `allowed` -- full confidence support
- `downgraded` -- finding may be produced but must carry limitations_note
- `downgraded` -- finding produced with limitations_note; use suggestive evidence_tier if visibility is poor

Full rules are in `source_profiles_config.json`. Visibility override rules apply
on top -- if `full_pitch_visibility_score` or `weakside_visibility_score` are low,
additional families are downgraded regardless of source type.

---

## Match Analysis -- FPS and Source Rules

1fps structural findings (requires sufficient pitch visibility):
evidence_tier: `direct` or `repeated_pattern`

3fps confirmation findings:
evidence_tier: `escalated_confirmation`

5fps fast-event findings:
evidence_tier: `escalated_confirmation`

Source constraints:
- Ball-follow footage: downgrade or suppress all full-team structural claims
- Narrow/local framing: produce local-only findings only
- Full-pitch visibility required for: shape, spacing, rest-defence, weak-side conclusions

---

## Opposition Analysis -- FPS and Source Rules

Opposition findings use analysis_scope: `opposition`. Same evidence_tier and
result_family_status system as match analysis.

1fps (requires pitch visibility):
- Opposition identity / style -> `repeated_pattern`
- Base structure -> `direct` or `repeated_pattern`
- Defensive block type -> `repeated_pattern`
- Broad structural recurring strengths/weaknesses -> `repeated_pattern`

3fps:
- Build-up behaviour confirmation -> `escalated_confirmation`
- Attacking pattern confirmation -> `escalated_confirmation`
- Pressing shape confirmation -> `escalated_confirmation`
- Unit relationship patterns -> `escalated_confirmation`

5fps:
- Transitions, pressing triggers, counterpress -> `escalated_confirmation`
- Fast attacking sequences -> `escalated_confirmation`
- Line-breaking actions -> `escalated_confirmation`

Opposition weakness classification -- use `opposition_focus_type`:
- `structural_weakness` -- visible at 1fps from pitch-visible source
- `local_action_weakness` -- requires 3-5fps or near-ball source
- `transition_weakness` -- requires 5fps
- `build_up_weakness` -- requires 3fps confirmation

Source constraints:
- Ball-follow: limits full-team opposition structure to near-ball only
- `opposition_structure` and `opposition_patterns` downgraded for `veo_ball_tracking`
  and `broadcast_tv` unless full_pitch_visibility_score is high

---

## Player Analysis -- FPS and Source Rules

Player findings use analysis_scope: `player`. Same rules apply.

NOTE: Full player analysis passes are not yet implemented as dedicated pipeline
steps. Player findings are extracted during window analysis when a player is
the subject of a tagged moment. These rules define when those findings are valid.

1fps (structure only, team context required):
- `player_role` -> `repeated_pattern` (across multiple windows)
- `player_positioning` -> `direct` or `repeated_pattern`
- Broad role relationships to shape -> `repeated_pattern`

3-5fps:
- `player_movement` -> `escalated_confirmation`
- `player_decision_making` -> `escalated_confirmation`
- Off-ball runs, support movement -> `escalated_confirmation`

5-10fps:
- `player_duels` -> `escalated_confirmation`
- `player_technical_actions` -> `escalated_confirmation`
- `player_line_breaking_contributions` -> `escalated_confirmation`

Source constraints:
- `veo_ball_tracking`: near-ball player actions only; `player_role` downgraded with in-zone limitation
- `tactical_wide_static`: best for role, positioning, team context
- `broadcast_tv`: isolated action clips; suppress complete off-ball player behaviour
- If player is off-ball and not in frame: do not produce findings for that player

---

## Universal Finding Object

Every finding produced by agents must carry these fields:

```json
{
  "analysis_scope":          "match | opposition | player",
  "finding_type":            "[description of what was observed]",
  "result_family":           "[family name from result family list]",
  "team":                    "home | away | both",
  "subject_player_id":       null,
  "subject_player_label":    null,
  "time_start":              "[MMmSSs]",
  "time_end":                "[MMmSSs]",
  "evidence_tier":           "direct | repeated_pattern | suggestive | escalated_confirmation",
  "confidence":              0.0,
  "result_family_status":    "allowed | downgraded",
  "source_type":             "[from source_profile.json]",
  "supporting_frames":       ["frame_XXmYYs.jpg"],
  "limitations_note":        null,
  "escalation_reason":       null,
  "escalation_target_fps":   null,
  "rerun_window_start":      null,
  "rerun_window_end":        null,
  "confidence_before_rerun": null,
  "confidence_after_rerun":  null,
  "opposition_focus_type":   null
}
```

Agents do not need to fill every field -- null is acceptable for non-applicable fields.
The pipeline stamps `result_family_status` and `source_type` from `result_family_gates.json`
after each window is merged.

---

## Step 3 -- Analysis Pipeline

### Overview

```
Step 1b  Boundary detection  -- KO1, HT whistle, KO2, FT whistle confirmed
Step 1c  Window plan         -- live-play windows generated from boundaries
Step 1d  Team sheet          -- API retrieval -> manual verification -> match_config.json
Step 1e  Attack direction    -- auto-detected from KO1/KO2 frames using confirmed kit colours
Step 1f  Source profiling    -- classify footage type, score visibility, gate result families
           writes: source_profile.json, result_family_gates.json
Step 2   Match data block    -- generated from match_config.json, saved to match_data_block.txt
Step 3a  Structural agent    -- all windows, 1fps. Shape, sequences (both teams),
                                line height, pressing, GK kicks, set piece flags.
Step 3b  Player agent        -- all windows, 1fps. Runs after 3a using its output
                                as context. Individual observations, duels, foot preference.
Step 3c  Confidence triage   -- rerun_queue.json + source gates applied to findings
Step 3d  Targeted re-runs    -- low-confidence frames only, focused prompt
Step 3d-EV  Event agent      -- event windows only, 5fps. Shot zone, build-up chain.
                                Single agent replacing the old dual Agent A+B.
Step 3d-SP  Set piece agent  -- when 3a flags a set piece. 5fps burst ±15 frames.
                                Runner positions, delivery arc, defensive structure.
Step 3d-REC Recovery agent   -- null windows only, 3fps. Minimum viable data.
Step 3e  Programmatic merge  -- merges 3a + 3b (+ 3d-EV/SP/REC where applicable)
                                Flags shape_dispute and line_height_dispute.
Step 3f  Pass accumulation   -- append pass_sequences to running_summary.json
Step 3g  Running summary     -- update running_summary.json after each window
Step 3h  Ground truth validation
Step 3i  Escalation router   -- escalation_router.py routes findings to 3/5/10fps segments
           cap: 10 items unless match has >10 goals (then goals are uncapped)
Step 3j  Report readiness    -- build_readiness_check.py reads confirmation_queue + all prior outputs
           writes: report_readiness.json, confidence_reliability_report.json
Step 3k  Deep skill metrics  -- deep_skill_metrics.py translates findings into performance scores
           writes: deep_skill_metrics.json
           inherits: evidence_tier, confidence, source limitations from findings
```

---

### 3a -- Window Allocation (Tier 1 Scan)

Windows are not hardcoded. They are read from `window_plan.json` generated in
Step 1c. This guarantees that only live-play frames are ever passed to analysis
agents -- halftime, pre-match warmup, and post-match are excluded at source.

All windows run at **1fps (every second)** with a **single agent**.
Each window covers exactly 5 minutes of live play = ~300 frames.
Event windows (goal, sub) are marked in `window_plan.json` and receive dual-agent
deep scan treatment in Step 3d.

**Before running any window, read window_plan.json and confirm:**
- Total window count
- Which windows are marked `event_window: true`
- The exact start_frame and end_frame for each window

### 3a -- Tier 1 Agent Prompt Template

```
You are a football tactical analyst. Review the frames listed below and produce
a structured JSON output only. No prose. No preamble. No markdown fences.

=== MATCH CONTEXT ===
[PASTE FULL MATCH DATA BLOCK]

FOCUS TEAM: [FOCUS TEAM] -- kit: [COLOUR]
OPPONENT:   [OPPONENT]   -- kit: [COLOUR]
GK:         [Name] (#[N]) in [colour]
ATTACK DIR: [FOCUS TEAM] attack [LEFT/RIGHT] in this half

=== SOURCE CONTEXT -- READ BEFORE ANALYSING ===
Source type:    [source_type from source_profile.json]
Downgraded families (produce findings but add limitations_note inline):
  [list from result_family_gates.json where status = "downgraded"]
Source limitation: [source_limitations_note from source_profile.json]

MATCH STATE AT START OF THIS WINDOW:
  Score: [score_home]-[score_away] ([match_state: level / winning / losing])
  (Read from window_plan.json match_state field for this window's agent_id)
  Use this to contextualise patterns -- note when behaviour differs by scoreline.
  Do not use this to assume intent or judge performance.

=== SCOUTING PRIMER ===
Known key players from match_config (use to direct player observation attention):
  [Paste from match_config: lineups with names, shirt numbers, and any prior notes]
Do not limit observations to named players -- but use this list to direct focus
when visibility is limited and you must prioritise.

=== GK DISTRIBUTION -- LOG FOR BOTH TEAMS ===
Every time a goalkeeper takes a goal kick, distributes from hands, or receives
a back pass and plays it out, add an entry to gk_kicks[].
This is MANDATORY. GK kicks are identifiable at 1fps -- the keeper standing
on the ball before a goal kick is a clear visual signal.
Log for both teams, not just the focus team.

=== BOTH TEAMS SEQUENCES ===
Log pass sequences for BOTH teams. Tag every sequence with:
  "team": "home_kit"  -- when the home team has possession
  "team": "away_kit"  -- when the away team has possession
The kit colours are specified in the MATCH CONTEXT above.
Do not skip opposition sequences. They are the source of the opposition pass network.


Every result family must be attempted. There is no suppressed state.
For downgraded families:
- Produce findings based on what you can observe
- Set evidence_tier honestly: use suggestive if visibility is poor
- Add limitations_note to the finding from the source profile
- If genuinely nothing could be observed: produce a finding that states this
  explicitly -- e.g. "Shape could not be reliably read from this viewing angle"
  This is a real finding. It tells the coaching staff what the footage cannot support.

=== FRAMES TO REVIEW ===
FRAMES DIR: [MATCH_DIR]ramesView EVERY SECOND (1fps) from [start_frame].jpg to [end_frame].jpg.
Each window covers 5 minutes -- approximately 300 frames.

=== CONFIDENCE SCORING ===
For every frame, output a confidence block. Score 0.0-1.0.
Reason codes for scores below 1.0:
  occlusion        -- player(s) obscured, position or identity unclear
  camera_motion    -- pan or zoom blur
  kit_ambiguity    -- cannot reliably distinguish teams
  ball_not_visible -- ball position inferred, not observed
  cluster          -- multiple players tightly grouped
  partial_frame    -- pitch edge cut off, line height unmeasurable
  low_contrast     -- poor lighting or overexposure
Frames below 0.7 confidence are automatically re-run. Be accurate.

=== EVIDENCE TIER RULES ===
Every finding you produce must be assigned an evidence_tier:

  direct           -- you can see it clearly in one or more frames
                     Use for: formation shape, line height, shots, set pieces
  repeated_pattern -- you see the same thing recurring across multiple frames/sequences
                     Use for: build-up routes, pressing patterns, territory, phase tendencies
  suggestive       -- you think it is happening but cannot confirm it clearly
                     Use for: anything where visibility limits certainty
                     Suggestive findings should trigger a confirmation_queue entry if important

Do not assign direct to anything you cannot clearly see.
Do not assign repeated_pattern to something observed only once.

=== RESULT FAMILY RULES ===
Every finding must be assigned to a result_family. Use these codes:
  Match scope:      shape / spacing / territory / phase / pressing / build_up /
                    rest_defence / transitions / chance_patterns / set_pieces /
                    local_duels / line_breaking_actions / box_entries
  Opposition scope: opposition_identity / opposition_structure / opposition_patterns /
                    opposition_build_up / opposition_pressing / opposition_transitions /
                    opposition_strengths / opposition_weaknesses
  Player scope:     player_role / player_positioning / player_movement /
                    player_decision_making / player_duels / player_technical_actions /
                    player_line_breaking_contributions

IMPORTANT: produce findings for ALL result families. If a family is downgraded,
produce the finding honestly with your actual evidence tier, and add the
limitations_note from the source profile. If the footage did not permit a
reliable reading, your finding text should say so explicitly.

=== ANALYSIS SCOPE RULES ===
Tag each finding with analysis_scope:
  match      -- observations about overall match shape, territory, phases
  opposition -- observations specifically about the opponent's structure or patterns
  player     -- observations about a named or identified individual player

=== PASS TRACKING ===
Log every possession sequence as a chain:
  [#N] ->F [#N] ->S [#N] ->[outcome]
Direction: F=forward  S=sideways  B=backward
Outcomes: shot / cross / lost_possession / clearance / set_piece / end_of_window
Minimum 20 sequences per window.


Zone codes for zone_start and zone_end:

RULE: The DEFAULT when the ball is within ~20m of either touchline is left_channel or right_channel.
Do NOT use defending_third or middle when the ball is near a touchline.
Longitudinal codes (defending_third / middle / attacking_third) are for genuinely central play only.

  left_channel      -- wide left, ball within ~20m of left touchline
  right_channel     -- wide right, ball within ~20m of right touchline
  left_halfspace    -- 20-35m from left touchline, between channel and central
  right_halfspace   -- 20-35m from right touchline, between channel and central
  defending_third   -- genuinely central play in own half
  middle            -- genuinely central play in middle third
  attacking_third   -- genuinely central play in final third

Test: ask "is the ball near a touchline?" → yes = channel or halfspace. no = longitudinal.
Example: left back receives a throw-in in their own half → left_channel, NOT defending_third.
Example: striker receives centrally in the box → attacking_third, NOT right_channel.

=== PRESSING INTENSITY ===
=== PRESSING INTENSITY ===
Score 0-10 per frame group. Use these reference points to calibrate consistently:

  0  = no pressure -- ball carrier has 3+ seconds with no challenger within 5m.
       The team is fully passive. Example: GK takes 10 seconds to distribute.

  2  = token closing -- one player moves toward ball but pulls out before contact.
       No cover shadow. Example: lone striker jogs to 8m then stops.

  4  = organised passive block -- shape held without pressing. Players cut passing
       lanes but do not step out. Example: two banks of four at 35m, forcing wide.

  6  = active pressing -- one or two players commit aggressively. Cover shadow.
       Example: midfielder closes right CB while striker cuts lane to GK.

  8  = coordinated press -- multiple players press simultaneously with structure.
       Example: front two press both CBs while midfield steps to block the DM.

  10 = total press -- full team pressing with no safe outlet. Rare even at elite level.
       Only use 10 if you genuinely see this.

Use the full scale. A typical non-league low-block will score 1-3 most windows.
Only use 7-10 for genuine coordinated multi-player pressing with blocked outlets.

Record press triggers using these codes:
  back_pass                -- ball played back toward own goal under pressure
  defender_facing_own_goal -- outfield defender receives with back to goal
  free_kick_restart        -- team restarts from a free kick
  throw_in_restart         -- team restarts from a throw-in
  other                    -- triggered but cause unclear
  null                     -- no press trigger in this frame group

=== DEFENSIVE LINE HEIGHT ===
Estimate FOCUS TEAM line as % of pitch (0%=own goal, 100%=opp goal).
Calibration: penalty box edge = ~16% from goal line.
Record at start, middle, end of each frame group.
Note significant shifts with timestamp and cause.

=== SHOT TRACKING ===
Track shots from BOTH teams.
Origin columns: left_channel / left_of_centre / central / right_of_centre / right_channel
Origin rows:    six_yard_box / penalty_spot / edge_of_box / outside_box
Shot type:      foot_right / foot_left / header / deflection
Outcome:        goal / on_target / off_target / blocked / post_bar
Target zone:    top_left / top_centre / top_right / bottom_left / bottom_centre /
                bottom_right / blocked_before_goal
Possession won: open_play_win / set_piece / turnover / GK_distribution / kickoff

=== SET PIECES ===
For every set piece (corner, free kick, final-third throw, kickoff):
Type:           corner_left / corner_right / direct_fk / indirect_fk /
                throw_final_third / kickoff
Delivery zone:  near_post / far_post / penalty_spot / edge_of_box / short
Delivery type:  inswinger / outswinger / driven / lofted / flick_on / short_routine / penalty_spot
Bodies in box:  [number]
Marking:        zonal / man / mixed
Outcome:        goal / cleared_near_post / cleared_far_post / gk_claim / gk_punch /
                second_phase / lost_possession

Runners (attacking set pieces): for each run you can observe, log:
  - runner zone:   near_post / far_post / penalty_spot / edge_of_box / blocking_run / second_phase
  - run type:      near_post_run / far_post_run / back_post / front_post / hold / blocking
  - defender:      #N or position label assigned to this runner (null if unassigned)

Even approximate runner data is valuable. If shirt numbers are unclear, use
position labels (e.g. "right winger", "second striker"). Coaching staff use
this for set piece preparation.

=== TRANSITIONS ===
For every observable possession change, log:
  Direction:  attack_to_defence / defence_to_attack
  Trigger:    interception / clearance / loss_of_possession / gk_distribution /
              set_piece_won / set_piece_conceded / other
  Players in front of ball:   [number or null]
  Defensive shape speed:      immediate / slow / disorganised
  counter_press:              true / false -- did the team losing the ball immediately
                              try to press to win it back within the same phase?
  Outcome:    counter_launched / possession_retained / ball_out_of_play /
              foul / organised_recovery

At 1fps you may not catch the exact transition frame. Log what is visible
in surrounding frames and note "transition inferred between frame_A and frame_B".
Transitions are important for opposition preparation -- log every one you observe.

Note counter-pressing specifically: if the team that just lost the ball immediately
presses in numbers to win it back (within 2-3 seconds), set counter_press: true.
This is different from their structured block -- it is an immediate reaction to
the turnover. Teams with consistent counter_press: true are difficult to transition
against; teams with counter_press: false leave space in behind after turnovers.

=== INDIVIDUAL PLAYER OBSERVATIONS ===
Log observations for BOTH teams throughout the window.

PRIORITY OBSERVATION -- BODY ORIENTATION WHEN RECEIVING UNDER PRESSURE:
For every defender or midfielder you observe receiving the ball under pressure,
note which way they are facing and which side the ball arrives on.
Example: "Received side-on with ball arriving on right -- turned left, played short."
This tells a pressing team exactly how to approach and which side to force.
Use action_category: body_orientation.
Minimum targets (enforced per window):
  - At least 5 individual observations per window total
  - At least 2 observations for opposition players per window
  - At least 1 observation for the opposition GK every 3 windows
  - At least 1 out-of-possession observation per attacker you observe in possession
  For the opposition, prioritise: GK, striker(s), any player with repeated involvement.

If a window has fewer than 3 total observations, player profile cards will be Grade D.
Individual observations are for REPEATED BEHAVIOURAL PATTERNS across a window, not
one-off events (those belong in key_moments or flagged_moments).
For the opposition, prioritise: GK, striker(s), any player with repeated involvement.

For each observation:
- Assign action_category from the fixed list above
- Assign observation_type: strength / weakness / trait / neutral
- Describe what you saw specifically -- no evaluation, no ratings
- Note frequency: single (saw it once), repeated (2-3 times this window),
  consistent (pattern across the window or match)
- Give the timestamp and 1-2 supporting frame names

Strengths and weaknesses are observations, not judgements. A player who repeatedly
wins aerial duels in the box has an aerial_ability strength. A player who
consistently received with back to goal and lost the ball has a hold_up_play
weakness. State what happened; do not add opinion.

For opposition players specifically, ask: what would an analyst preparing
for this team want to know about this player? What did they do repeatedly?
What spaces did they exploit? What did they struggle with?

=== CONFIRMATION QUEUE RULES ===
Add an item to confirmation_queue when:
- You observe an event that 1fps cannot confirm (fast contact, transition, GK claim)
- You have a suggestive finding that is tactically important enough to confirm
- The exact outcome of an action is unclear (duel won/lost, ball caught/punched)

Set priority: high = goals, GK claims, transitions, box entries, rebounds
              medium = pressing shape, build-up patterns, phase ambiguities

Do NOT queue: open-play passing sequences, formation shape, pressing scores,
              line height, set-piece starting structures, anything a downgraded family's limitation note says cannot be confirmed

Output ONLY raw JSON. No preamble, no explanation, no markdown fences.

{
  "agent_id": "[NN]",
  "window": "[start]-[end]min",
  "frames_reviewed": [number],
  "scan_interval_seconds": 1,

  "frames": [
    {
      "frame": "frame_XXmYYs.jpg",
      "confidence": {
        "score": [0.0-1.0],
        "flags": ["[reason_code]", ...],
        "affected_players": ["#N", ...],
        "affected_metrics": ["line_height", "pressing_intensity", "formation", "pass_tracking", "shot_tracking"]
      },
      "observations": {
        "formation_shape": "[e.g. 4-3-3]",
        "pressing_score": [0-10],
        "press_trigger": "[description or null]",
        "line_height_pct": [0-100],
        "ball_visible": [true/false],
        "ball_zone": "[defending_third / middle / attacking_third or null]",
        "notes": "[anything notable in this frame]"
      }
    }
  ],

  "possession_summary": {
    "[focus_team]_pct": [0-100],
    "[opponent]_pct": [0-100],
    "dominant_zone": "[defending_third / middle / attacking_third]",
    "territory_notes": "[description]"
  },

  "formation": {
    "shape_in_possession": "[e.g. 4-4-2]",
    "shape_out_of_possession": "[e.g. 4-4-2]",
    "compactness": "[compact / stretched / disorganised]",
    "notes": "[any mid-window variation]"
  },

  "defensive_line": {
    "start_pct": [0-100],
    "mid_pct": [0-100],
    "phase_context": "[settled / pressing / transition / set_piece / unknown -- the tactical state when avg_pct was recorded]",
    "end_pct": [0-100],
    "avg_pct": [0-100],
    "line_width":     "[very_narrow / narrow / standard / wide -- horizontal spread of the back line]",
    "line_coordination": "[coordinated / ragged / unclear -- do defenders step together or individually?]",
    "notable_shifts": [
      {"timestamp": "[MMmSSs]", "from_pct": [number], "to_pct": [number],
       "cause": "[what triggered the shift]"}
    ],
    "frames_excluded": ["frame_XXmYYs.jpg"]
  },

  "pressing": {
    "scores": [
      {"frame_group":   "[MMmSSs]",
       "score":         [0-10],
       "trigger":       "[gk_in_possession / back_pass / defender_facing_own_goal / free_kick_restart / throw_in_restart / other / null]",
       "press_direction": "[force_wide / force_central / block_backward / unclear / null -- which direction does the press funnel the ball carrier?]",
       "press_initiator": "[#N or position label -- which player closes first, or null]"}
    ],
    "avg_score": [0-10],
    "peak_score": [0-10],
    "peak_timestamp": "[MMmSSs]",
    "coordinated_press_observed": [true/false],
    "press_triggers_identified": ["[description]"],
    "frames_excluded": ["frame_XXmYYs.jpg"]
  },

  "pass_sequences": [
    {
      "start_frame": "[MMmSSs]",
      "sequence": "[chain string]",
      "length": [number],
      "zone_start": "[defending_third / middle / attacking_third / left_channel / right_channel / left_halfspace / right_halfspace]",
      "zone_end": "[defending_third / middle / attacking_third / left_channel / right_channel / left_halfspace / right_halfspace]",
      "team":             "[home_kit / away_kit -- REQUIRED. Log sequences for BOTH teams, not just the focus team]",
      "outcome": "[shot/cross/lost_possession/clearance/set_piece/end_of_window]",
      "progressive":        [true/false],
      "is_long_ball":        [true/false -- REQUIRED for clearances, long kicks, goal kicks going long, and any sequence where zone_start → zone_end skips a zone],
      "second_ball_contest": "[won / lost / unclear / null -- REQUIRED when is_long_ball is true]",
      "second_ball_contest": "[won / lost / unclear / null -- for long balls only, who won the loose ball]"
    }
  ],

  "set_pieces": [
    {
      "timestamp":      "[MMmSSs]",
      "type":           "[corner_left / corner_right / free_kick_central / free_kick_wide_left / free_kick_wide_right / throw_in / kickoff]",
      "team":           "[team]",
      "delivery_zone":  "[zone]",
      "delivery_type":  "[inswinger / outswinger / near_post / far_post / driven / lofted / short / penalty_spot]",
      "bodies_in_box":  [number],
      "marking_system": "[zonal / man / mixed]",
      "runners": [
        {
          "runner_id":    "[#N or position label]",
          "run_zone":     "[near_post / far_post / penalty_spot / edge_of_box / second_phase]",
          "run_type":     "[near_post_run / far_post_run / back_post / front_post / penalty_spot_hold / blocking_run / second_phase]",
          "defender_assigned": "[#N or position label or null if unassigned]"
        }
      ],
      "delivery_target_zone": "[near_post / penalty_spot / far_post / edge_of_box]",
      "outcome":        "[goal / cleared_near_post / cleared_far_post / gk_claim / gk_punch / second_phase / lost_possession]",
      "outcome_detail": "[who cleared / who scored / who punched]",
      "wall_size":      [number of players in wall, for direct free kicks near goal, or null],
      "wall_position":  "[near_post / central / null -- where the wall is set]"
    }
  ],

  "transitions": [
    {
      "timestamp":          "[MMmSSs]",
      "direction":          "[attack_to_defence / defence_to_attack]",
      "trigger":            "[interception / clearance / loss_of_possession / gk_distribution / set_piece_won / set_piece_conceded / other]",
      "players_in_front_of_ball": [number or null],
      "defensive_shape_speed": "[immediate / slow / disorganised]",
      "outcome":            "[counter_launched / possession_retained / ball_out_of_play / foul / organised_recovery]",
      "counter_press":      [true / false],
      "frames":             ["frame_XXmYYs.jpg"]
    }
  ],

  "shot_attempts": [
    {
      "timestamp": "[MMmSSs]",
      "team": "[team name]",
      "player": "#[N] [Name]",
      "origin_column": "[left_channel / left_of_centre / central / right_of_centre / right_channel]",
      "origin_row": "[six_yard_box / penalty_spot / edge_of_box / outside_box]",
      "shot_type": "[foot_right / foot_left / header / deflection]",
      "outcome": "[goal / on_target / off_target / blocked / post_bar]",
      "target_zone": "[top_left / top_centre / top_right / bottom_left / bottom_centre / bottom_right / blocked_before_goal]",
      "from_set_piece": [true/false],
      "possession_won_by": "[open_play_win / set_piece / turnover / GK_distribution / kickoff]",
      "sequence_to_shot": "[#N ->F #N ->S #N ->F SHOT]",
      "sequence_length": [number],
      "sequence_start_zone": "[defending_third / middle / attacking_third]",
      "frames": ["frame_XXmYYs.jpg"]
    }
  ],

  "attacking": {
    "preferred_side": "[left / right / central / mixed]",
    "build_up_style": "[short / direct / mixed]",
    "forward_movement": "[description]",
    "width_usage": "[description]",
    "chances": [
      {"timestamp": "[MMmSSs]", "description": "[what]", "quality": "[low/medium/high]"}
    ]
  },

  "defensive": {
    "transition_speed": "[slow / medium / fast]",
    "vulnerabilities": ["[description]"],
    "gk_distribution": "[short / long / mixed]",
    "gk_positioning": "[on_line / sweeper / mixed]"
  },

  "key_moments": [
    {
      "timestamp": "[MMmSSs]",
      "type": "[goal/chance/sub/set_piece/tactical_shift/individual/disciplinary]",
      "description": "[what happened]",
      "tactical_significance": "[why it matters]",
      "frames": ["frame_XXmYYs.jpg"]
    }
  ],


  "individual_observations": [],

  NOTE: Individual player observations are captured by the Step 3b Player Agent,
  not this structural scan. Leave individual_observations as an empty array here.
  The player agent runs after this output and uses it as context.

    {
      "timestamp": "[MMmSSs]",
      "frames": ["frame_XXmYYs.jpg", "frame_XXmYYs.jpg"],
      "event_type": "[goal / gk_claim / gk_punch / gk_parry / cross_six_yard / goalmouth_scramble / rebound / set_piece_delivery / null]",
      "result_family": "[transitions / pressing / build_up / chance_patterns / local_duels / box_entries / line_breaking_actions / phase / opposition_transitions / opposition_pressing / player_duels / player_movement / null]",
      "analysis_scope": "[match / opposition / player]",
      "reason": "[what 1fps could not confirm -- specific question to answer]",
      "evidence_tier": "suggestive",
      "confidence_before_rerun": 0.0,
      "priority": "[high / medium]"
    }
  ],

  "findings": [
    {
      "analysis_scope": "[match / opposition / player]",
      "finding_type": "[description]",
      "result_family": "[family name]",
      "team": "[home / away / both]",
      "subject_player_id": null,
      "subject_player_label": null,
      "time_start": "[MMmSSs]",
      "time_end": "[MMmSSs]",
      "evidence_tier": "[direct / repeated_pattern / suggestive]",
      "confidence": 0.0,
      "result_family_status": "allowed",
      "supporting_frames": ["frame_XXmYYs.jpg"],
      "limitations_note": null,
      "escalation_reason": null,
      "escalation_target_fps": null,
      "opposition_focus_type": null
    }
  ]
}
```

Write immediately after completion:
`agent_logs/agent_[NN]_[start]-[end]min.json`

---


---

### 3b -- Player Agent (NEW)

Run immediately after Step 3a structural scan is complete for each window.
The player agent focuses exclusively on individual players -- it does NOT re-read
shape, sequences, or pressing. It receives the structural output as context.

**Purpose:** capture individual observations, duels, preferred foot, physical
profile, and cross-window behavioural patterns. This is the sole source of
data for player profile cards and the opposition Key Players section.

**Cross-window context:** before running, include a brief summary of the
prior window's top observations so the agent can correctly code
"repeated" or "consistent" frequency.

```
You are a football player analyst. Your only task is to observe individual players
in the frames listed below.

=== MATCH CONTEXT ===
[PASTE FULL MATCH DATA BLOCK]

FOCUS TEAM: [FOCUS TEAM] -- kit: [COLOUR]
OPPONENT:   [OPPONENT]   -- kit: [COLOUR]
ATTACK DIR: [FOCUS TEAM] attack [LEFT/RIGHT] in this half

=== STRUCTURAL CONTEXT (from Step 3a output for this window) ===
Formation: [formation from 3a]
Line height: [avg_pct from 3a]
Pressing avg: [avg_score from 3a]
Most involved players (from 3a pass sequences): [top shirt numbers by involvement]

=== PRIOR WINDOW PLAYER SUMMARY (cross-window context) ===
[For window 01: leave blank. For all other windows: paste top 3-4 individual
 observations from the previous window's player agent output, e.g.:
 "#10 Bullent: received between lines x3 (repeated), left foot preference noted"
 "#6 Ruffles: won aerial duels in right channel (consistent)"
 This context allows you to code 'repeated' or 'consistent' frequency correctly.]

=== SCOUTING PRIMER ===
Known key players: [from match_config lineups -- names, shirt numbers]
Pay particular attention to: [any flagged players from prior match intelligence]

=== FRAMES TO REVIEW ===
FRAMES DIR: [MATCH_DIR]
View EVERY SECOND (1fps) from [start_frame].jpg to [end_frame].jpg.

=== YOUR TASK ===
For every player you can clearly identify in the frames:

1. Log individual_observations entries (schema below)
2. Log duels[] entries for every physical contest you observe
3. Note preferred_foot from carrying, shooting, or crossing actions
4. Note physical_profile from visual impression

MINIMUM REQUIREMENTS (enforced):
  - At least 5 individual_observations per window total
  - At least 2 observations for opposition players
  - At least 1 observation for the opposition GK every 3 windows
  - For every attacker you observe in possession: also log one out_of_possession observation
  - For every duel you observe (aerial or ground): log it in duels[]

If the window is quiet and you genuinely cannot reach 5 observations: log what
you can see and note the reason (e.g. "ball-follow framing limits player visibility").
Do not invent observations. Do not inflate confidence.

=== OUTPUT FORMAT ===
Return JSON only. No prose. No preamble. No markdown fences.

{
  "window": "[window_id]",
  "player_agent": true,

  "individual_observations": [
    {
      "player":           "#[N] [Name or position label]",
      "team":             "[home / away]",
      "position":         "[gk/cb/lb/rb/cm/dm/am/lm/rm/lw/rw/st/cf]",
      "action_category":  "[ball_carrying / distribution / hold_up_play /
                           movement_off_ball / finishing / set_piece_delivery /
                           pressing_behaviour / defensive_positioning / aerial_ability /
                           duels / recovery_runs / gk_distribution / gk_positioning /
                           gk_shot_stopping / positional_tendency / body_orientation /
                           link_up_partner]",
      "observation_type": "[strength / weakness / trait / neutral]",
      "observation":      "[specific description -- what happened, not evaluation]",
      "outcome":          "[success / failure / neutral / unclear]",
      "obs_grade":        "[leave blank -- auto-computed from confidence + frequency]",
      "zone":             "[use lateral codes by default near touchlines:
                           left_channel / right_channel / left_halfspace / right_halfspace /
                           defending_third / middle / attacking_third / box]",
      "game_phase":       "[in_possession / out_of_possession / transition / set_piece]",
      "frequency":        "[single / repeated / consistent]",
      "timestamp":        "[MMmSSs]",
      "confidence":       "[high / medium / low]",
      "frames":           ["frame_XXmYYs.jpg"],
      "preferred_foot":   "[right / left / both / unknown]",
      "physical_profile": {
        "height_impression": "[tall / average / short / unknown]",
        "pace_impression":   "[quick / average / slow / unknown]",
        "build":             "[powerful / athletic / lean / unknown]"
      }
    }
  ],

  "duels": [
    {
      "timestamp":       "[MMmSSs]",
      "type":            "[aerial / ground / tackle]",
      "winner":          "[home_kit / away_kit / contested / unknown]",
      "zone":            "[zone code]",
      "players_visible": ["#N kit_colour", "#N kit_colour"]
    }
  ]
}
```

Write output to:
`agent_logs/agent_[NN]_[start]-[end]min_player.json`

The player agent output is merged with the structural output in Step 3e.

---


### 3b -- Confidence Aggregator and Source Gate Application

Run after all Tier 1 scans are complete. Reads every agent JSON, builds the
re-run queue, and applies source gate states to all findings.
Do not run per window -- run once across all windows.

**Source gate application:** after building the rerun queue, read
`result_family_gates.json` and stamp `result_family_status` onto every finding
in every Tier 1 output. Findings with `result_family_status: downgraded`
carry a limitations_note inline. No findings are excluded.
Findings with `result_family_status: downgraded` are included but carry a
`limitations_note` derived from `source_profile.json`.

**Gate design principle -- suppressed vs downgraded:**
`suppressed` means the evidence is structurally absent. Use it when the
camera geometry makes a result family physically unreadable -- e.g. shape from
a behind-goal camera where the pitch is viewed end-on.
`downgraded` means zone-limited or partial evidence. Real signal is present
but incomplete in scope. Ball-tracking footage (e.g. Veo) produces this for
shape, spacing, territory and rest-defence -- the camera shows the active zone
clearly but not the full pitch simultaneously. Downgraded findings carry a
limitations_note and use qualified language in reports:
  "The defensive shape in the visible zone appeared to be two banks of four"
  NOT: "The formation was 4-4-2"
Suppression should be reserved for footage types where no useful reading
is possible -- not applied defensively to avoid overclaiming.

```python
from source_profiler import apply_gate_to_finding
import json, os, glob

def apply_source_gates(match_dir):
    gates_path = os.path.join(match_dir, "result_family_gates.json")
    profile_path = os.path.join(match_dir, "source_profile.json")
    if not os.path.exists(gates_path):
        print("⚠  result_family_gates.json missing -- skipping gate application")
        return

    with open(gates_path) as f:
        gates_doc = json.load(f)
    gates = gates_doc.get("gates", {})

    with open(profile_path) as f:
        profile = json.load(f)
    source_note = profile.get("source_limitations_note", "")

    logs_dir = os.path.join(match_dir, "agent_logs")
    for path in sorted(glob.glob(os.path.join(logs_dir, "agent_*_*min.json"))):
        with open(path) as f:
            window = json.load(f)

        findings = window.get("findings", [])
        for finding in findings:
            apply_gate_to_finding(finding, gates)
            if finding.get("result_family_status") == "downgraded" and not finding.get("limitations_note"):
                finding["limitations_note"] = source_note
            finding["source_type"] = profile.get("source_type", "unknown")

        window["findings"] = findings
        window["source_gates_applied"] = True
        with open(path, "w") as f:
            json.dump(window, f, indent=2)

    print(f"  Source gates applied to all Tier 1 window JSONs")
```

```python
import json, os, glob

MATCH_DIR = r"[MATCH_DIR]"
LOGS_DIR  = os.path.join(MATCH_DIR, "agent_logs")
CONFIDENCE_THRESHOLD = 0.7
DATA_GAP_THRESHOLD   = 0.30  # flag window if >30% of frames are low confidence

def build_rerun_queue():
    rerun_queue   = []
    window_flags  = []
    tier1_files   = sorted(glob.glob(os.path.join(LOGS_DIR, "agent_*_*min.json")))

    for filepath in tier1_files:
        with open(filepath) as f:
            data = json.load(f)

        agent_id       = data.get("agent_id")
        window         = data.get("window")
        frames         = data.get("frames", [])
        total_frames   = len(frames)
        low_conf_count = 0

        for frame in frames:
            conf = frame.get("confidence", {})
            score = conf.get("score", 1.0)
            if score < CONFIDENCE_THRESHOLD:
                low_conf_count += 1
                rerun_queue.append({
                    "agent_id":         agent_id,
                    "window":           window,
                    "source_file":      os.path.basename(filepath),
                    "frame":            frame["frame"],
                    "confidence_score": score,
                    "flags":            conf.get("flags", []),
                    "affected_players": conf.get("affected_players", []),
                    "affected_metrics": conf.get("affected_metrics", []),
                    "status":           "queued"
                })

        low_conf_pct = low_conf_count / total_frames if total_frames > 0 else 0
        window_flags.append({
            "agent_id":          agent_id,
            "window":            window,
            "total_frames":      total_frames,
            "low_conf_frames":   low_conf_count,
            "low_conf_pct":      round(low_conf_pct, 2),
            "data_gap_warning":  low_conf_pct > DATA_GAP_THRESHOLD
        })

    output = {
        "total_frames_assessed": sum(w["total_frames"] for w in window_flags),
        "total_low_conf_frames": len(rerun_queue),
        "window_summary":        window_flags,
        "rerun_queue":           rerun_queue
    }

    out_path = os.path.join(MATCH_DIR, "rerun_queue.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"rerun_queue.json written.")
    print(f"Total frames assessed: {output['total_frames_assessed']}")
    print(f"Frames queued for re-run: {len(rerun_queue)}")
    for w in window_flags:
        gap = " ⚠ DATA GAP" if w["data_gap_warning"] else ""
        print(f"  {w['window']}: {w['low_conf_frames']}/{w['total_frames']} low conf ({w['low_conf_pct']*100:.0f}%){gap}")

build_rerun_queue()
```

---

### 3c -- Targeted Re-Run Prompt Template

One prompt per low-confidence frame. Built programmatically from rerun_queue.json.
Do not re-send match data block -- include only what is needed to resolve the
specific uncertainty.

```
Re-analyse [FRAME_FILENAME].

Previous Tier 1 scan was low confidence ([SCORE]) due to: [FLAGS]
Uncertain players: [AFFECTED_PLAYERS or "none specified"]
Affected metrics: [AFFECTED_METRICS]

[MATCH]: [FOCUS TEAM] in [COLOUR], [OPPONENT] in [COLOUR].
[FOCUS TEAM] attack [LEFT/RIGHT] in this half.

Examine this frame carefully and address only the following:

[IF occlusion or cluster]:
  -- Can you now confirm the position and team assignment of [AFFECTED_PLAYERS]?
  -- If still obscured, what is your best positional estimate and why?

[IF kit_ambiguity]:
  -- Look for shirt number, hair, height, or context to confirm team assignment
    for any ambiguous players. List your reasoning.

[IF ball_not_visible]:
  -- Can the ball position be inferred from player body orientation, gaze
    direction, or foot contact? State your estimate and confidence.

[IF partial_frame]:
  -- Estimate defensive line height from visible players only.
    Note which players are off-frame and how that affects the estimate.

[IF camera_motion or low_contrast]:
  -- Use player silhouettes and pitch markings to estimate formation shape
    and defensive line. State which elements you can and cannot resolve.

Return ONLY a JSON object for this single frame:

{
  "frame": "[FRAME_FILENAME]",
  "rerun": true,
  "confidence": {
    "score": [0.0-1.0],
    "flags": [...],
    "affected_players": [...],
    "affected_metrics": [...],
    "unresolvable": [true/false]
  },
  "resolved_observations": {
    "formation_shape": "[or null if unresolvable]",
    "pressing_score": [0-10 or null],
    "line_height_pct": [0-100 or null],
    "ball_visible": [true/false],
    "ball_zone": "[zone or null]",
    "player_positions": [
      {"player": "#N", "confirmed": [true/false], "position_estimate": "[description]"}
    ],
    "notes": "[resolution summary]"
  }
}
```

After each re-run JSON is returned, update its entry in rerun_queue.json:
- Set `"status": "resolved"` if confidence ≥ 0.7
- Set `"status": "unresolvable"` if confidence remains < 0.7

---

### 3d -- Event Agent (replaces dual deep scan)

Run on windows containing goals or substitutions ONLY.
Extracts at **5fps** for the event window -- not 1fps.
Single agent, focused prompt. No dual A+B duplication.

The event agent answers specific questions about the event.
It does NOT re-run the full structural scan.
The structural scan (3a) and player scan (3b) have already run on this window.

**Why 5fps:** At 1fps, a shot completed in 0.8 seconds is invisible.
At 5fps, the shot frame, ball trajectory, and goalkeeper position are all captured.
The 5fps extraction runs only for the event window, not the full match.

**Cross-agent validation:** The event agent output is compared against 3a/3b
for the same window. Disagreements on formation or line height are flagged
as shape_dispute in the merge (Step 3e). No silent resolution.

```
You are a football event analyst. A key event occurred in this window.
Your task is to answer specific questions about it, not to re-scan the full window.

=== MATCH CONTEXT ===
[PASTE FULL MATCH DATA BLOCK]

FOCUS TEAM: [FOCUS TEAM] -- kit: [COLOUR]
OPPONENT:   [OPPONENT]   -- kit: [COLOUR]

=== STRUCTURAL CONTEXT (from 3a output for this window) ===
Formation: [3a formation]
Line height at event: [3a line height near event timestamp]
Pressing score: [3a pressing avg]

[IF GOAL IN WINDOW]:
=== GOAL EVENT ===
[Player] (#[N] [Team]) scores at [minute]'. This goal is in this window.

Answer these questions using the 5fps frames:
1. Shot origin zone (use lateral codes: left_channel/right_channel/halfspace/central)
2. Shot foot (right / left / header / other)
3. Target zone (top_left / top_right / bottom_left / bottom_right / central_high / central_low)
4. Build-up: describe the sequence of passes in the 30 seconds before the shot.
   For each pass: [kit_colour] [position_label] →[F=forward/S=square/B=backward] [kit_colour] [position_label]
5. Defensive shape at the moment of the shot: where were the defenders?
   Was the shooter marked? Was there a runner making space?
6. Any pressing trigger that preceded the build-up?

[IF SUBSTITUTION IN WINDOW]:
=== SUBSTITUTION EVENT ===
[Player_off] replaced by [Player_on] at [minute]'.

Answer these questions:
1. Exact video timestamp when substitution occurred
2. What position did [Player_on] take up immediately?
3. Did the formation change? If yes: from what to what?
4. Did the defensive line change height? If yes: by how much?
5. Any immediate change in pressing intensity?

=== OUTPUT FORMAT ===
Return JSON only. No prose. No preamble.

{
  "window": "[window_id]",
  "event_agent": true,
  "events": [
    {
      "type":              "[goal / substitution]",
      "timestamp":         "[MMmSSs]",
      "team":              "[home / away]",
      "player":            "[name or #N]",

      // GOAL fields
      "shot_origin_zone":  "[zone code]",
      "shot_foot":         "[right / left / header / other]",
      "target_zone":       "[top_left / top_right / bottom_left / bottom_right / central_high / central_low]",
      "build_up_sequence": "[full chain as described above]",
      "defensive_shape_at_shot": "[description of defensive positions]",
      "shot_quality":      "[high_danger / low_danger / unclear -- based on origin zone and marking]",

      // SUBSTITUTION fields
      "player_on_position":    "[position taken up]",
      "formation_change":      "[null / from_4-4-2_to_4-3-3 etc]",
      "line_height_change_m":  "[null / +5 / -3 etc]",
      "pressing_change":       "[null / increased / decreased / unchanged]"
    }
  ],
  "shape_vs_structural_agent": {
    "agrees":   [true / false],
    "disputes": ["[any disagreement with 3a output]"]
  }
}
```

Write output to:
`agent_logs/agent_[NN]_[start]-[end]min_event.json`

The event agent output is merged in Step 3e alongside structural and player outputs.

---

### 3d-SP -- Set Piece Agent (NEW)

Triggered when Step 3a flags a set piece in its output.
Runs a **5fps burst of ±15 frames** around the flagged set piece timestamp.
Single agent, focused on delivery and runner positions only.

```
You are a set piece analyst. A set piece occurred at approximately [timestamp].
Review the frames in the ±15 frame window around this moment.

=== EVENT ===
Type: [corner_right / corner_left / free_kick_direct / free_kick_indirect / penalty]
Team: [home / away]
Timestamp: [MMmSSs]

Answer these questions:
1. Delivery type: inswinger / outswinger / driven / lofted / short / pullback
2. Delivery target zone: near_post / back_post / penalty_spot / edge_of_box / far_post
3. Bodies in box: [count]
4. Runner positions and roles:
   For each visible runner: shirt number or kit colour, starting zone, run zone,
   role (near_post_run / back_post_run / penalty_spot_hold / edge_blocker / runner_far_post)
5. Marking system: man_marking / zonal / mixed
6. Any unmarked runner?
7. Outcome: goal / cleared / second_phase / saved / off_target / other

=== OUTPUT FORMAT ===
{
  "window": "[window_id]",
  "set_piece_agent": true,
  "timestamp": "[MMmSSs]",
  "type": "[type]",
  "team": "[home / away]",
  "delivery": "[delivery type]",
  "target_zone": "[target zone]",
  "bodies_in_box": [count],
  "runners": [
    {
      "player_id":   "[#N or kit_colour + position]",
      "start_zone":  "[zone]",
      "run_zone":    "[zone]",
      "role":        "[role]",
      "marked":      [true / false / unclear]
    }
  ],
  "marking_system": "[man_marking / zonal / mixed]",
  "unmarked_runner": "[#N or null]",
  "outcome":         "[outcome]"
}
```

Write output to:
`agent_logs/agent_[NN]_[start]-[end]min_setpiece.json`

---

### 3d-REC -- Recovery Agent (NEW)

Triggered automatically for any window where Step 3a returned null or confidence < 0.5.
Runs at **3fps** with a stripped-down prompt. Goal: recover minimum viable data.
Output is marked `recovery_pass: true` and carries reduced confidence.

```
A previous scan of this window returned null or low confidence.
Review the frames at 3fps and produce minimum viable structural data only.

=== FRAMES ===
FRAMES DIR: [MATCH_DIR]
View every THIRD frame (3fps) from [start] to [end].

Report ONLY:
1. Dominant formation visible (or "unreadable")
2. Approximate defensive line height (% of pitch or "unreadable")
3. Which team had the ball for most of the window (or "unclear")
4. Any key event visible (goal celebration, substitution board, etc.)

{
  "window": "[window_id]",
  "recovery_pass": true,
  "formation":     "[4-4-2 / 4-3-3 / unreadable]",
  "line_height_pct": [number or null],
  "possession_est":  "[home / away / contested / unclear]",
  "key_event_visible": "[description or null]",
  "confidence":    0.4
}
```

Write to:
`agent_logs/agent_[NN]_[start]-[end]min_recovery.json`

---


### 3e -- Programmatic Merge

Merges Tier 1 (+ re-runs) for routine windows, and dual deep-scan outputs for
event windows. Produces one merged file per window.

```python
import json, os, glob

MATCH_DIR = r"[MATCH_DIR]"
LOGS_DIR  = os.path.join(MATCH_DIR, "agent_logs")
RERUN_Q   = os.path.join(MATCH_DIR, "rerun_queue.json")

def load_rerun_patches(agent_id):
    """Return dict of frame_name -> resolved_observations for a given agent."""
    with open(RERUN_Q) as f:
        rq = json.load(f)
    patches = {}
    for item in rq.get("rerun_queue", []):
        if item["agent_id"] == agent_id and item["status"] == "resolved":
            rerun_file = os.path.join(
                LOGS_DIR, item["source_file"].replace(".json", "_rerun.json")
            )
            if os.path.exists(rerun_file):
                with open(rerun_file) as f:
                    patches[item["frame"]] = json.load(f)
    return patches

def patch_frames(frames, patches):
    """Replace low-confidence frame observations with resolved re-run data."""
    patched = []
    for frame in frames:
        name = frame["frame"]
        if name in patches:
            rerun = patches[name]
            frame["confidence"]   = rerun["confidence"]
            frame["observations"] = rerun.get("resolved_observations", frame["observations"])
            frame["rerun_applied"] = True
        patched.append(frame)
    return patched

def merge_numeric(a, b):
    if a is None: return b
    if b is None: return a
    return round((a + b) / 2, 1)

def merge_categorical(a, b):
    if a == b: return a
    # prefer more specific (longer) value; fall back to A
    return a if len(str(a)) >= len(str(b)) else b

def merge_dual_agents(a_path, b_path, out_path, agent_id):
    with open(a_path) as f: a = json.load(f)
    with open(b_path) as f: b = json.load(f)

    patches_a = load_rerun_patches(a["agent_id"])
    patches_b = load_rerun_patches(b["agent_id"])
    a["frames"] = patch_frames(a.get("frames", []), patches_a)
    b["frames"] = patch_frames(b.get("frames", []), patches_b)

    review_required = []

    def resolve_line(fa, fb, key):
        va, vb = fa.get(key), fb.get(key)
        if va == vb: return va, "agreed"
        merged = merge_numeric(va, vb)
        return merged, "resolved"

    def resolve_cat(fa, fb, key):
        va, vb = fa.get(key), fb.get(key)
        if va == vb: return va, "agreed"
        merged = merge_categorical(va, vb)
        review_required.append(f"{key}: A={va}, B={vb} -> used {merged}")
        return merged, "resolved"

    line_avg, line_status = resolve_line(
        a.get("defensive_line", {}), b.get("defensive_line", {}), "avg_pct"
    )
    formation, form_status = resolve_cat(
        a.get("formation", {}), b.get("formation", {}), "shape_in_possession"
    )
    press_a = a.get("pressing", {}).get("avg_score")
    press_b = b.get("pressing", {}).get("avg_score")
    press_avg = merge_numeric(press_a, press_b)

    # Key moment cross-check
    a_moments = {m["timestamp"]: m for m in a.get("key_moments", [])}
    b_moments = {m["timestamp"]: m for m in b.get("key_moments", [])}
    merged_moments = []
    for ts, moment in a_moments.items():
        if ts in b_moments:
            moment["consensus"] = "confirmed"
        else:
            moment["consensus"] = "partial_a_only"
            review_required.append(f"key_moment at {ts} seen by A only")
        merged_moments.append(moment)
    for ts, moment in b_moments.items():
        if ts not in a_moments:
            moment["consensus"] = "partial_b_only"
            review_required.append(f"key_moment at {ts} seen by B only")
            merged_moments.append(moment)

    merged = {
        "agent_id":        agent_id,
        "window":          a["window"],
        "merge_type":      "dual_agent",
        "frames_reviewed": max(a.get("frames_reviewed", 0), b.get("frames_reviewed", 0)),
        "review_required": review_required,
        "defensive_line": {
            **a.get("defensive_line", {}),
            "avg_pct":    line_avg,
            "merge_note": line_status
        },
        "formation": {
            **a.get("formation", {}),
            "shape_in_possession": formation,
            "merge_note":          form_status
        },
        "pressing": {
            **a.get("pressing", {}),
            "avg_score":  press_avg,
            "merge_note": "averaged"
        },
        "pass_sequences":         a.get("pass_sequences", []) + b.get("pass_sequences", []),
        "set_pieces":             a.get("set_pieces", []),
        "shot_attempts":          a.get("shot_attempts", []) + b.get("shot_attempts", []),
        "attacking":              a.get("attacking", {}),
        "defensive":              a.get("defensive", {}),
        "key_moments":            merged_moments,
        "individual_observations":a.get("individual_observations", []),
        "flaggable_moments":      a.get("flaggable_moments", []) + b.get("flaggable_moments", []),
        "possession_summary":     a.get("possession_summary", {}),
        "window_summary":         a.get("window_summary", ""),
        "window_confidence": {
            "overall_score":          merge_numeric(
                a.get("window_confidence", {}).get("overall_score"),
                b.get("window_confidence", {}).get("overall_score")
            ),
            "low_confidence_frame_count": max(
                a.get("window_confidence", {}).get("low_confidence_frame_count", 0),
                b.get("window_confidence", {}).get("low_confidence_frame_count", 0)
            ),
            "data_gap_warning": (
                a.get("window_confidence", {}).get("data_gap_warning", False) or
                b.get("window_confidence", {}).get("data_gap_warning", False)
            )
        }
    }

    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged -> {os.path.basename(out_path)} | Issues: {len(review_required)}")

def merge_single_agent(a_path, out_path, agent_id):
    """For routine windows: apply re-run patches and write as merged."""
    with open(a_path) as f: a = json.load(f)
    patches = load_rerun_patches(a["agent_id"])
    a["frames"] = patch_frames(a.get("frames", []), patches)
    a["merge_type"] = "single_agent"
    with open(out_path, "w") as f:
        json.dump(a, f, indent=2)
    print(f"Patched -> {os.path.basename(out_path)}")
```

---

### 3f -- Pass Sequence Accumulation

After each merged file is written, append its `pass_sequences` to the running
`pass_sequences.json`. This builds a full-match pass log across all 16 windows.

```json
{
  "match": "[Home vs Away]",
  "focus_team": "[Team]",
  "total_sequences": [number],
  "sequences": [
    {
      "window": "[start]-[end]min",
      "start_frame": "[MMmSSs]",
      "sequence": "[chain]",
      "length": [number],
      "zone_start": "[zone]",
      "zone_end": "[zone]",
      "outcome": "[outcome]",
      "progressive": [true/false]
    }
  ]
}
```

---

### 3g -- Running Summary Accumulation

After each window is merged, append its key metrics to `running_summary.json`.
This is the single file the report writer reads in Step 4 -- it never reads raw
window JSONs.

```python
import json, os

def update_running_summary(merged_path, summary_path):
    with open(merged_path) as f:
        w = json.load(f)

    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {
            "match": "",
            "windows_complete": 0,
            "formation_history": [],
            "pressing_by_window": [],
            "line_height_by_window": [],
            "shots_for": [],
            "shots_against": [],
            "flagged_moments": [],
            "key_moments": [],
            "individual_observations": [],
            "set_pieces": [],
            "possession_by_window": [],
            "data_gap_windows": []
        }

    summary["windows_complete"] += 1

    summary["formation_history"].append({
        "window":  w["window"],
        "shape":   w.get("formation", {}).get("shape_in_possession")
    })
    summary["pressing_by_window"].append({
        "window":    w["window"],
        "avg_score": w.get("pressing", {}).get("avg_score"),
        "peak":      w.get("pressing", {}).get("peak_score")
    })
    summary["line_height_by_window"].append({
        "window":  w["window"],
        "avg_pct": w.get("defensive_line", {}).get("avg_pct"),
        "shifts":  w.get("defensive_line", {}).get("notable_shifts", [])
    })

    for shot in w.get("shot_attempts", []):
        target = summary["shots_for"] if shot["team"] == w.get("focus_team") else summary["shots_against"]
        target.append(shot)

    summary["flagged_moments"].extend(w.get("flaggable_moments", []))
    summary["key_moments"].extend(w.get("key_moments", []))
    summary["individual_observations"].extend(w.get("individual_observations", []))
    summary["set_pieces"].extend(w.get("set_pieces", []))
    summary["possession_by_window"].append({
        "window":  w["window"],
        "summary": w.get("possession_summary", {})
    })

    if w.get("window_confidence", {}).get("data_gap_warning"):
        summary["data_gap_windows"].append(w["window"])

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
```

---

### 3h -- Ground Truth Validation

After all 16 merged files exist, validate every known event from the match data
block. Search `running_summary.json` key_moments for each KEY EVENT:

1. Found in key_moments with `"consensus": "confirmed"` -> `"status": "confirmed"`
2. Found with `"consensus": "partial_*"` -> `"status": "partial"` -- re-run deep scan for that window
3. Not found -> `"status": "missed"` -- mandatory deep scan re-run before Step 4

Write to `ground_truth_check.json`:
```json
{
  "events_checked": [number],
  "confirmed": [number],
  "partial": [number],
  "missed": [number],
  "results": [
    {
      "event": "[minute]' [description]",
      "expected_window": "agent_[NN]",
      "status": "[confirmed / partial / missed]",
      "found_at": "[timestamp or null]"
    }
  ],
  "rerun_required": ["agent_[NN] -- [reason]"]
}
```

**Do not proceed to Step 4 until `missed` count is 0.**

---


---


---

### 3j -- Report Readiness Check

Run `build_readiness_check.py` after ground truth validation and before
confirmation. This is the pipeline gate for Step 4. Step 4 must not run if
`report_ready` is `false`.

```python
# build_readiness_check.py
import json, os

def build_readiness_check(match_dir):
    def load(fname):
        p = os.path.join(match_dir, fname)
        if not os.path.exists(p): return None
        with open(p) as f: return json.load(f)

    boundaries   = load("match_boundaries.json")
    config       = load("match_config.json")
    window_plan  = load("window_plan.json")
    ground_truth = load("ground_truth_check.json")
    summary      = load("running_summary.json")
    rerun_q      = load("rerun_queue.json")

    # Boundary confidence
    bc = {}
    if boundaries:
        for k, v in boundaries.get("boundaries", {}).items():
            bc[k] = v.get("confidence", 0)
    boundary_ok = all(v >= 0.8 for v in bc.values()) if bc else False

    # Config
    config_verified   = bool(config and config.get("verified"))
    enrichment_level  = (config or {}).get("enrichment_level", "identity_only")
    player_id_ceiling = (config or {}).get("player_id_ceiling", "tentative")

    # Ground truth
    gt_passed = bool(ground_truth and ground_truth.get("missed", 1) == 0)

    # Windows
    planned  = (window_plan or {}).get("total_windows", 0)
    complete = (summary or {}).get("windows_complete", 0)
    windows_ok = complete == planned and planned > 0

    # Reruns
    unresolvable = sum(
        1 for i in (rerun_q or {}).get("rerun_queue", [])
        if i.get("status") == "unresolvable"
    )

    # Confirmation
    cq = load("confirmation_queue.json") or {"total": 0, "skipped": 0}

    blocking = []
    if not boundary_ok:
        low = [k for k, v in bc.items() if v < 0.8]
        blocking.append(f"Low boundary confidence: {low}")
    if not config_verified:
        blocking.append("match_config.json not verified")
    if not gt_passed:
        missed = (ground_truth or {}).get("missed", "unknown")
        blocking.append(f"Ground truth: {missed} events missed -- re-run deep scan")
    if not windows_ok:
        blocking.append(f"Windows: {complete}/{planned} complete")

    data_gap_windows = (summary or {}).get("data_gap_windows", [])
    if data_gap_windows:
        blocking.append(f"Data gap windows: {data_gap_windows} -- review before reporting")

    report_ready = len(blocking) == 0

    readiness = {
        "report_ready":            report_ready,
        "boundary_confidence":     bc,
        "boundary_confidence_ok":  boundary_ok,
        "team_config_verified":    config_verified,
        "enrichment_level":        enrichment_level,
        "player_id_ceiling":       player_id_ceiling,
        "event_validation_passed": gt_passed,
        "windows_planned":         planned,
        "windows_complete":        complete,
        "windows_ok":              windows_ok,
        "unresolvable_frames":     unresolvable,
        "data_gap_windows":        data_gap_windows,
        "confirmation_total":      cq.get("total", 0),
        "confirmation_skipped":    cq.get("skipped", 0),
        "blocking_issues":         blocking,
        "report_modules_available": {
            "tactical":   True,
            "opposition": True,
            "moments":    True
        }
    }

    out_path = os.path.join(match_dir, "report_readiness.json")
    with open(out_path, "w") as f:
        json.dump(readiness, f, indent=2)

    status = "✓ READY" if report_ready else "✗ NOT READY"
    print(f"\n{status} -- report_readiness.json written")
    for issue in blocking:
        print(f"  ✗ {issue}")
    if report_ready:
        print(f"  Enrichment: {enrichment_level} | Player ID ceiling: {player_id_ceiling}")
        print(f"  Confirmation items: {cq.get('total',0)} | Skipped: {cq.get('skipped',0)}")
        print(f"  Data gap windows: {data_gap_windows or 'none'}")

    return report_ready

if __name__ == "__main__":
    import sys
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ")
    build_readiness_check(match_dir)
```

**Hard rule:** If `report_ready` is `false`, Step 4 does not run.
Resolve all blocking issues, re-run `build_readiness_check.py`, confirm ready before continuing.

**The `player_id_ceiling` field** is read by the Step 4 report writer to cap player
identification language. If ceiling is `tentative`, no full player names appear in
any report section regardless of what agents observed.

---

### 3i -- Hybrid Frame-Rate Confirmation

After report_readiness.json confirms ready=true, process the confirmation queue.

**Scope -- this step is confirmation only, not a second analysis pass.**
It answers one focused question per event using a short higher-fps segment.
It does not re-analyse shape, pressing, territory, or passing patterns.

**Eligible event types -- agents should only queue these:**

| Event type | Priority | fps | Window |
|---|---|---|---|
| Goal | always | 3fps | ±4s |
| Shot on target | always | 3fps | ±3s |
| GK claim / punch / parry | always | 3fps | ±3s |
| Cross into six-yard box | always | 3fps | ±3s |
| Goalmouth scramble / rebound | always | 5fps | ±3s |
| Set-piece delivery (ambiguous outcome) | high | 3fps | ±3s |
| Decisive transition (possession origin unclear) | high | 3fps | ±3s |

**Explicitly excluded -- do not queue these:**
- Open-play passing sequences
- Tackles and 50-50s away from goal
- Throw-ins and routine restarts
- Formation shape observations
- Pressing behaviour observations
- Anything already confirmed by multiple 1fps frames

**Hard cap -- enforced in script:**

```python
from escalation_router import build_escalation_queue

MATCH_DIR = r"[MATCH_DIR]"
build_escalation_queue(MATCH_DIR)
```

This script:
- Reads all confirmation_queue entries from merged window JSONs
- Determines escalation fps tier per result_family / event_type
- Applies the cap: 10 items max UNLESS match has >10 goals
  (if goals > 10, all goals process; other high-priority items fill remaining slots)
- Writes `confirmation_queue.json` with `escalation_target_fps` and `rerun_window_*` per item

**Escalation fps tiers:**

| Trigger | Target fps | Reason |
|---|---|---|
| goal | 5fps | importance |
| transitions, box_entries, local_duels | 5fps | fast_event |
| goalmouth_scramble, rebound | 5fps | fast_event |
| pressing, build_up, phase | 3fps | uncertainty |
| gk_claim, gk_punch, gk_parry | 3fps | uncertainty |
| cross_six_yard, set_piece_delivery | 3fps | uncertainty |
| suggestive finding, confidence < 0.7 | 3fps (medium) or 5fps (high) | uncertainty |

**Cap rule:** 10 items max. If match has > 10 goals, goals are uncapped;
non-goal high-priority items fill any remaining cap slots.
Skipped items are logged in `confirmation_queue.json` and reported in
`confidence_reliability_report.json`.
```

**Re-extraction script:**

```python
import cv2, os, json

def extract_segment(video_path, timestamp_seconds, out_dir,
                    fps_target=3, window_seconds=4, container_profile=None):
    """
    Extract a short segment at higher fps. Returns a structured dict, never a bare integer.
    Reads container_profile to decide seek method and detect boundary proximity.

    Returns:
        {
          "success": bool,
          "frames_extracted": int,
          "frames": [filenames],
          "reason": str or None,       # failure reason if success=False
          "detail": str or None,       # technical detail for pipeline logs
          "seek_method": "opencv" | "ffmpeg"
        }
    """
    import subprocess

    start_s = max(0, timestamp_seconds - window_seconds / 2)
    end_s   = timestamp_seconds + window_seconds / 2
    os.makedirs(out_dir, exist_ok=True)

    # -- Check container profile before attempting seek ------------------------
    if container_profile:
        # Block if timestamp is within 5s of a known segment boundary
        for boundary in container_profile.get("boundary_timestamps_s", []):
            if abs(timestamp_seconds - boundary) < 5.0:
                return {
                    "success":          False,
                    "frames_extracted": 0,
                    "frames":           [],
                    "reason":           "near_segment_boundary",
                    "detail":           (f"Timestamp {timestamp_seconds}s is within 5s of "
                                        f"segment boundary at {boundary}s. "
                                        f"Seek accuracy not guaranteed for this source."),
                    "seek_method":      None,
                }

        # Use ffmpeg seek for long-GOP or non-seekable sources
        use_ffmpeg = (
            not container_profile.get("seek_reliable", True)
            or not container_profile.get("higher_fps_extraction_safe", True)
            or container_profile.get("max_keyframe_interval_s", 0) > 4.0
        )
    else:
        use_ffmpeg = False

    # -- Remux if recommended and not already done -----------------------------
    if container_profile and container_profile.get("remux_recommended") and not use_ffmpeg:
        remuxed = video_path.replace(".mp4", "_remuxed.mp4")
        if not os.path.exists(remuxed):
            r = subprocess.run([
                "ffmpeg", "-y", "-i", video_path,
                "-c", "copy", "-movflags", "faststart", remuxed
            ], capture_output=True)
            if r.returncode != 0:
                use_ffmpeg = True  # fall back to ffmpeg seek
            else:
                video_path = remuxed
        else:
            video_path = remuxed

    frames = []

    # -- ffmpeg seek path (long-GOP and non-seekable sources) ------------------
    if use_ffmpeg:
        dur = end_s - start_s
        pattern = os.path.join(out_dir, "confirm_%02dm%02ds%1d.jpg")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_s),
            "-i", video_path,
            "-t", str(dur),
            "-vf", f"fps={fps_target}",
            "-q:v", "3",
            os.path.join(out_dir, "confirm_%04d.jpg"),
            "-loglevel", "error"
        ]
        r = subprocess.run(cmd, capture_output=True)
        if r.returncode != 0:
            return {
                "success":          False,
                "frames_extracted": 0,
                "frames":           [],
                "reason":           "ffmpeg_extraction_failed",
                "detail":           r.stderr.decode()[:300],
                "seek_method":      "ffmpeg",
            }
        frames = sorted(f for f in os.listdir(out_dir) if f.startswith("confirm_"))
        return {
            "success":          True,
            "frames_extracted": len(frames),
            "frames":           frames,
            "reason":           None,
            "detail":           None,
            "seek_method":      "ffmpeg",
        }

    # -- OpenCV seek path (clean containers) -----------------------------------
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "success":          False,
            "frames_extracted": 0,
            "frames":           [],
            "reason":           "video_open_failed",
            "detail":           f"cv2.VideoCapture failed to open {video_path}",
            "seek_method":      "opencv",
        }

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = src_fps / fps_target
    frame_idx = int(start_s * src_fps)

    # Verify seek accuracy before extracting
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    actual_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    seek_error = abs(actual_pos - start_s)

    if seek_error > 2.0:
        cap.release()
        return {
            "success":          False,
            "frames_extracted": 0,
            "frames":           [],
            "reason":           "seek_inaccurate",
            "detail":           (f"Requested {start_s}s, OpenCV landed at {actual_pos:.1f}s "
                                f"(error: {seek_error:.1f}s). "
                                f"Source may have segmented recording format."),
            "seek_method":      "opencv",
        }

    while frame_idx <= int(end_s * src_fps):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break
        t = frame_idx / src_fps
        m, s = divmod(int(t), 60)
        ms = int((t - int(t)) * 10)
        fname = f"confirm_{m:02d}m{s:02d}s{ms}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frames.append(fname)
        frame_idx += int(frame_interval)

    cap.release()

    if not frames:
        return {
            "success":          False,
            "frames_extracted": 0,
            "frames":           [],
            "reason":           "no_frames_extracted",
            "detail":           "cap.read() returned False on first frame -- check video integrity",
            "seek_method":      "opencv",
        }

    return {
        "success":          True,
        "frames_extracted": len(frames),
        "frames":           frames,
        "reason":           None,
        "detail":           None,
        "seek_method":      "opencv",
    }
```

**Confirmation prompt:**

Build one prompt per queued item. The question is specific to the event type and
the fps used. Read `escalation_target_fps` from `confirmation_queue.json` per item.

Before sending this prompt, check extract_segment() result:
- If `success: False` -- do NOT send a confirmation prompt. Record the failure:
  ```json
  {
    "status":              "extraction_failed",
    "reason":              "[reason from extract_segment return]",
    "detail":              "[detail from extract_segment return]",
    "seek_method":         "[opencv / ffmpeg / null]",
    "confirmed_outcome":   null,
    "confidence_after_rerun": 0.0,
    "recommended_report_wording": "This event could not be confirmed -- [reason]. The 1fps observation stands as the best available evidence."
  }
  ```
  Do not invent a technical explanation. Use the exact `reason` and `detail` from extract_segment().

If `success: True`, send this prompt:

```
Confirm this event. This is a short segment extracted at [escalation_target_fps]fps.
Source type: [source_type from source_profile.json]

Event type:     [event_type or result_family]
Analysis scope: [match / opposition / player]
Timestamp:      [timestamp]
Rerun window:   [rerun_window_start] to [rerun_window_end]
Question:       [reason from confirmation_queue -- the specific thing 1fps could not confirm]

If no specific question is recorded, use the default for this event type:
  goal:              "Describe the goal: scorer, zone, shot type, GK position, how possession was won."
  transition:        "How many players were in front of the ball? How fast did the defensive shape recover? Did this lead to a counter-attack?"
  gk_claim/punch:    "Did the goalkeeper cleanly catch or punch? Where did the ball land? Was it contested?"
  local_duels:       "Who won the duel? Did the winner retain possession or win a free kick?"
  box_entries:       "How many players arrived in the box? What was the defensive shape at the point of entry?"
  set_piece_delivery:"Delivery type, target zone, who attacked the ball, outcome."
  pressing:          "How many players pressed simultaneously? Did they cut off the passing lane? Did the press succeed?"
  build_up:          "Who was involved? Central or wide? Did any player break the first line?"

MATCH: [FOCUS TEAM] in [COLOUR], [OPPONENT] in [COLOUR].

View frames in order: [list frame filenames from rerun window]

Answer ONLY the specific question. Do not re-analyse shape, pressing, or
territory -- those are covered by the 1fps scan. Focus only on the fast or
ambiguous event described above.

If the frames are insufficient to answer the question (poor visibility, wrong
angle, too few frames), output status: "inconclusive" -- do not guess.

Output ONLY raw JSON:

{
  "status": "confirmed | inconclusive",
  "timestamp": "[MMmSSs]",
  "event_type": "[event_type or result_family]",
  "analysis_scope": "[match / opposition / player]",
  "confirmed_outcome": "[description -- descriptive, no opinion -- or null if inconclusive]",
  "player_involved": "[#N Name or position label or null]",
  "confidence_after_rerun": 0.0,
  "evidence_tier": "escalated_confirmation",
  "result_family_status": "allowed",
  "recommended_report_wording": "[one sentence -- descriptive language only]",
  "limitations_note": null,
  "frames_used": ["[filenames]"]
}
```

After each confirmation, update the relevant merged window JSON and
`running_summary.json` with the confirmed outcome before Step 4 runs.

**Additional confirmation prompt fields by event type:**

For TRANSITION confirmations (5fps): also answer:
  "pace_comparison": "clear_advantage / evenly_matched / outpaced" -- relative speed of key players
  "first_touch_quality": "clean / bobbled / unclear" -- first touch of the player receiving the ball in transition

For DUEL confirmations (5fps): also answer:
  "first_touch_quality": "clean / bobbled / unclear"
  "goes_to_ground": true/false -- did either player go to ground under light contact?

For GK SHOT confirmations (5fps): also answer:
  "dive_direction": "left / right / stayed / unclear"
  "save_technique": "down_quickly / parry / catch / unclear"

**GK / aerial confidence rules:**

| Evidence available | Wording to use |
|---|---|
| Multiple frames show clean hands contact | "The goalkeeper claimed the cross." |
| Frames show GK in possession after delivery | "The goalkeeper was in possession following the cross." |
| Contact ambiguous -- punch or catch unclear | "The goalkeeper came to the cross." |
| GK near ball but contact frame missing | "The goalkeeper appeared to challenge for the cross." |
| No clean frame of contact | Do not describe the outcome -- note it as unconfirmed |

These rules apply to all aerial interventions, not just crosses. Rebounds,
goalmouth scrambles, and deflections follow the same principle: only state
what the frames can support.

Add new failure mode:

| Confirmation segment too short | Increase window_seconds in extract_segment() |
| Event still ambiguous after 3fps segment | Try 5fps re-extraction; if still unclear, use softest wording tier |

---

---

## Deep Skill Metrics Layer

The deep skill metrics layer translates detected structures, patterns, and behaviours
into performance-level tactical insights. It sits above detection/aggregation and
below report writing. Computed after Step 3i so escalated_confirmation findings
improve metric confidence.

**Design principles:**
- Every metric traces directly to pipeline data -- no metric more than one derivation
  step from a real observation
- No composite-of-composites -- removed control_score, structural_integrity_score,
  dominance_score (signal diluted twice)
- No metrics that require source capabilities the primary footage type cannot provide
  (removed weakside_utilisation_score -- ball-tracking never achieves the required threshold)
- No metrics that trivially score the same for every match (removed shape_stability_score)

**Confidence inheritance rules:**
- Every metric inherits evidence_tier and source limitations from supporting findings
- repeated_pattern evidence: capped at 0.75
- suggestive evidence: capped at 0.4
- Downgraded required family: confidence reduced by 0.2
- Fewer than 3 windows: confidence reduced by 0.15
- Global source cap applies (veo_ball_tracking: 0.6)
- Metrics with value: null are unavailable, not suppressed

---

### System-Level Metrics (5)

| Metric | Required families | Description |
|---|---|---|
| `compactness_score` | shape, pressing | Line height stability (60%) + pressing intensity (40%) |
| `pressing_intensity_score` | pressing | Normalised average pressing score (0-10 -> 0-1) |
| `build_up_effectiveness_score` | build_up | Progressive + threat sequences / total; returns score + progressive_rate + conversion_rate |
| `rest_defence_security_score` | rest_defence | Inverse of line height shifts per window |
| `line_height_range` | shape | Max minus min line height across match in metres and percentage |

---

### Behavioural Metrics (6)

| Metric | Required families | Description |
|---|---|---|
| `width_usage_score` | spacing | Wide zone sequences / total; zone_labels method preferred, cross_outcomes_proxy fallback |
| `halfspace_occupation_score` | phase, territory | Sequences involving half-space zones (requires granular zone encoding in Tier 1 output) |
| `transition_efficiency_score` | transitions | Shot/goal transitions / total; null if fewer than 3 transitions observed |
| `chance_creation_profile` | chance_patterns | Three-layer profile: origin zone -> route -> end point for each chance |
| `pattern_reliability_score` | build_up, phase | Most-common zone route frequency / total sequences |
| `build_up_route_diversity` | build_up | Distinct zone-to-zone route pairs / 12 (recalibrated ceiling) |

---

### Tactical Insight Metrics (4)

| Metric | Required families | Description |
|---|---|---|
| `predictability_score` | build_up, phase | Pattern reliability (60%) + inverse route diversity (40%) |
| `pressing_trigger_consistency` | pressing | How consistent the press trigger type is; 1.0 = always same trigger |
| `set_piece_delivery_profile` | set_pieces | Delivery type, bodies in box, marking system and outcome by zone |
| `attacking_support_score` | chance_patterns, build_up | Average sequence length before shots/crosses |

---

### Player Metrics (5 per player)

Extracted from individual_observations and escalated findings.
Requires player_role and player_positioning families.

| Metric | Required families | Description |
|---|---|---|
| `player_role_consistency` | player_role, player_positioning | Rating variance across observations (1 - normalised variance) |
| `player_positioning_stability` | player_positioning | Average rating normalised to 0-1 |
| `player_decision_profile` | player_decision_making | Requires escalated confirmation findings |
| `player_movement_contribution` | player_movement | Requires escalated confirmation findings |
| `player_duel_effectiveness` | player_duels | Requires escalated duel confirmations at 5fps |

---

### Removed Metrics (do not re-add)

| Metric | Reason removed |
|---|---|
| `shape_stability_score` | Always 1.0 for consistent teams -- no analytical signal |
| `spacing_control_score` | Proxy for line height already captured by compactness and line_height_range |
| `weakside_utilisation_score` | Requires weakside_visibility > 0.5 -- ball-tracking footage never achieves this |
| `control_score` | Composite of composites -- signal diluted twice |
| `dominance_score` | Meaningless without reliable shot data |
| `structural_integrity_score` | Composite of composites -- low value |
| `chaos_index` | Inverse of shape_stability -- redundant |

---

### Metric Schema

Every metric in `deep_skill_metrics.json` carries:

```json
{
  "metric_name":                "pressing_trigger_consistency",
  "analysis_scope":             "match | opposition | player",
  "subject_team":               "home | away | both",
  "value":                      {"consistency_score": 1.0, "dominant_trigger": "gk_in_possession",
                                 "trigger_counts": {"gk_in_possession": 11}, "total_observations": 11},
  "value_type":                 "numeric_0_1 | profile | unavailable",
  "supporting_result_families": ["pressing"],
  "evidence_tier":              "direct",
  "confidence":                 0.60,
  "result_family_status":       "allowed | downgraded",
  "severely_limited":           false,
  "limitation_note":            null,
  "windows_contributing":       11,
  "fps_context":                "1fps observation; 3fps confirmation",
  "source_limitations":         "Ball-follow footage -- zone-limited",
  "calculation_basis":          "Consistency of press trigger type across observed pressing moments",
  "traceable_to":               ["pressing_by_window", "key_moments", "flagged_moments"]
}
```

**Note on profile-type metrics:** Some metrics return a profile dict rather than a single 0-1
score. These include: build_up_effectiveness_score, line_height_range, chance_creation_profile,
pressing_trigger_consistency, set_piece_delivery_profile, attacking_support_score,
build_up_route_diversity. Profile metrics are surfaced in reports as structured insight sections
rather than single table values. Numeric metrics (compactness, pressing_intensity, etc.)
appear in the metrics table.

---

### Metric Degradation Rules

| Condition | Effect |
|---|---|
| All families downgraded | Confidence degraded across all metrics |
| Required family is downgraded | Confidence reduced by 0.2 |
| Evidence tier is suggestive | Confidence capped at 0.4 |
| Evidence tier is repeated_pattern | Confidence capped at 0.75 |
| Windows contributing < 3 | Confidence reduced by 0.15 |
| Source is unknown | Confidence capped at 0.5 |
| Source is veo_ball_tracking | shape/spacing/territory metrics downgraded -- zone-limited |
| Source is broadcast_tv | All structural metrics downgraded -- framing-limited |

---

## Step 3k -- Deep Skill Metrics

Run after Step 3i (escalation and confirmation) so that escalated_confirmation
findings improve metric confidence. Before Step 4.

```python
from deep_skill_metrics import build_deep_skill_metrics

MATCH_DIR  = r"[MATCH_DIR]"
FOCUS_TEAM = "[home / away / both]"

build_deep_skill_metrics(MATCH_DIR, FOCUS_TEAM)
```

Output: `deep_skill_metrics.json`

**What to check after running:**
- Review `unavailable_metrics` list -- these are null and excluded from reports
  (halfspace_occupation and transition_efficiency are commonly unavailable without
  granular zone encoding or sufficient transition observations)
- Review `low_confidence_metrics` list -- these appear with limitation notes in reports
- Review `avg_confidence` -- below 0.5 indicates significant source limitations
- Player metrics require player_role and player_positioning families -- these need
  lineups data to populate individual_observations
- Profile-type metrics (chance_creation_profile, pressing_trigger_consistency etc.)
  appear as structured insight sections in reports, not single table rows

**Metrics and the confidence_reliability_report:**
After Step 3k runs, the confidence_reliability_report (written by build_readiness_check.py)
is updated to include a metric summary section. Severely-limited metrics are listed alongside
downgraded families and limitation notes so the full picture is visible in one place.

---

## Report Standards

These rules apply to every word written in Step 4. They are standing instructions,
not suggestions. All reports -- Tactical, Opposition, and Moments -- follow them.

---


### Language Rule 0 -- Grade-appropriate language

The report level determines how observations are stated. This rule is applied
consistently across the tactical report and opposition report:

  **Grade A** (confirmed, high confidence) -- state as fact at all levels.
  **Grade B** (likely, moderate confidence):
    brief: elevate to fact if preparationally critical, otherwise omit.
    standard: "appeared to", "in observed windows", "consistently in the first half".
    technical: state observation count and window spread inline.
  **Grade C** (single sighting or limited visibility):
    brief: omit.
    standard: "On the one observed occasion..." or "Preliminary indication..."
    technical: state confidence and frequency explicitly.
  **Grade D** (suggestive only, one low-confidence frame):
    brief/standard: omit entirely.
    technical: list player as observed but unprofilable.

### Language Rule 1 -- Descriptive Only, Strengths and Weaknesses Permitted

Match Lens reports describe what was visible and what repeated.
They identify patterns of strength and weakness through observation.
They do not instruct, judge, blame, or offer tactical solutions.

Reports cover:
1. **Observation** -- what the frames show directly
2. **Pattern** -- what repeated across windows
3. **Sequence/context** -- when and where it happened
4. **Strength or weakness** -- a pattern that consistently produced positive or negative outcomes

Reports do not move into:
- coaching instruction ("they should...", "they need to...")
- suggested fixes or tactical solutions
- opinion on player or staff performance
- emotional or motivational interpretation
- criticism or praise

**The strength/weakness boundary:**

A strength or weakness is the observation of a repeating pattern and its outcome.
The coaching response to that pattern belongs to the coach, not the report.

| Permitted (observation of pattern) | Not permitted (coaching response) |
|---|---|
| "The left side was exposed on 5 occasions when the full-back pushed forward -- on each occasion the space in behind was exploited by a runner." | "They should not allow the full-back to advance without cover." |
| "Felixstowe's delivery from right corners consistently targeted the penalty spot with 8 bodies in the box -- all three attempts were cleared from the near post." | "Lowestoft should shift to a near-post zonal block to defend this routine." |
| "Lowestoft's defensive line dropped below 35% on 6 of 8 occasions when Felixstowe held possession in the middle third for more than 10 seconds." | "Lowestoft need to hold a higher line to prevent Felixstowe from establishing possession." |
| "The #7 received the ball with their back to goal in all 14 observed sequences -- 11 resulted in lost possession." | "They should play #7 facing forward to improve retention." |

**Replace judgment with description:**

| Avoid | Use instead |
|---|---|
| "They dropped too deep." | "The defensive line moved from approximately 40m to 25m during the build-up." |
| "They failed to press." | "No coordinated pressure on the ball carrier was observed during this sequence." |
| "They accepted the result." | "Urgency and forward pressure were reduced in the final phase compared with the previous phase." |
| "The left-back was poor." | "Repeated 1v1 and 2v1 situations were observed on the left side across multiple windows." |
| "They needed to press higher." | Do not write this -- it is coaching instruction, not observation. |
| "They should exploit the weakness." | Do not write this -- it is coaching advice. State the weakness; let the coach decide. |

---

### Language Rule 2 -- Evidence Tiers

Every claim is tagged to its evidence tier. The writing tone must match the tier.

| Tier | Definition | Language to use |
|---|---|---|
| **Observed** | Directly visible in frames | "The shape was 4-4-2." / "The line moved to approximately 30m." |
| **Estimated** | Calculated from structured data with known margin | "Approximately [X]% of observed sequences..." / "In roughly [N] of the [N] windows..." |
| **Inferred** | Interpretation based on pattern -- not directly visible | "This appeared to..." / "The pattern suggested..." / "It was not possible to determine whether..." |

Rules:
- Never use a specific number unless it comes from structured data (shot counts, window counts, pass chain lengths)
- Never write "always" or "never" -- use "in the majority of observed windows" or "no instances were observed"
- Mentality, coaching intent, and tactical instructions are always **Inferred** -- phrase them accordingly

---

### Language Rule 3 -- Player Identification

Use the confirmed player ID tier from `match_config.json` to determine wording:

| Tier | When | Wording |
|---|---|---|
| **Confirmed** | Shirt number readable + position consistent + API lineup match | Full name: "J. Smith (#7)" |
| **Probable** | Number readable OR position consistent, not both | "#7 (probable: J. Smith)" |
| **Tentative** | Inferred from position/height/hair only | "the right winger" or "#7 (tentative)" |
| **Unidentified** | Cannot resolve | "an attacking midfielder" |

Never upgrade wording beyond the confidence the evidence supports.

---

### Language Rule 4 -- GK and Aerial Actions

At 1fps, the exact contact frame for aerial interventions is often absent.
Use the following wording tiers based on what the frames support:

| Evidence | Wording |
|---|---|
| Multiple frames show clean hands contact | "The goalkeeper claimed the cross." |
| GK in possession in the frame after delivery | "The goalkeeper was in possession following the cross." |
| Contact frame missing -- outcome ambiguous | "The goalkeeper came to the cross." |
| GK near ball, contact not visible | "The goalkeeper appeared to challenge for the cross." |
| No useful frames of the event | Do not describe outcome -- note as unconfirmed in data |

If the event was sent to confirmation_queue and resolved by Step 3i,
use the `recommended_report_wording` from the confirmation JSON.

---

### Language Rule 5 -- House Style

**Use:**
- Past tense for completed matches: "The shape was...", "The line moved..."
- Calm, confident, specific: "Most attacks were observed through the right channel."
- Football-literate terminology: channel, half-space, unit, line, shape, press trigger
- Concise over elaborate: one clear sentence beats two vague ones

**Avoid:**
- "AI-generated" phrasing: "It is worth noting that...", "Upon analysis...", "The data suggests..."
- Repetition across sections -- state a theme once clearly, then support with evidence
- Hedging every sentence -- one clear caveat per section is enough
- Exact-looking numbers without structured backing: never "67% possession" unless counted

**Tone:** A thoughtful analyst who watched the match and wrote it up. Not a dashboard. Not a model output.

---


### Language Rule 5b -- No pipeline language in reports

Reports are read by coaching staff, not analysts. Never expose internal pipeline
terminology in any output that goes to a coaching staff reader.

NEVER use in reports:
  - Window identifiers: W01, W02, W13, "window 9", "window 11"
  - Agent language: "agent observations", "agent logs", "W09 documents"
  - Technical source language: "1fps", "ball-tracking source", "batch"
    (these belong in the evidence note only, not in findings prose)

REPLACE window references with match time or phase descriptions:
  BAD:  "W01 documents Belmar pressing aggressively in the opening minutes."
  GOOD: "Belmar pressed aggressively in the opening minutes."

  BAD:  "Agent observations from W04, W06, W07 describe Gorleston forcing
         long clearances."
  GOOD: "Gorleston forced long clearances from Tilbury through pressing pressure
         throughout the first half."

  BAD:  "W16 shows a 9-pass Tilbury sequence."
  GOOD: "In the 71st–75th minute, Tilbury produced a 9-pass sequence..."

  BAD:  "From approximately W15 (2H 20:00), agent observations note the home side
         dropping deeper."
  GOOD: "From approximately 65 minutes, Gorleston dropped into a medium block."

Window identifiers are internal reference tools for the analyst.
They should appear only in the evidence note, flagged moments timestamps,
and the pass network source log — never in tactical prose.

### Language Rule 5c -- No cross-match contamination

Players, clubs, events, and statistics from other matches must never appear
in a report. Before finalising any report, verify:
  - Every named player is in this match's lineup (from match_config.json)
  - Every event cited (goal, card, sub) is from this match's match facts
  - No player name, shirt number, or club from a different fixture appears

If a player name or event appears in the report that cannot be traced to
match_config.json for this match: remove it and note the gap.

This rule prevents prior match data leaking into new reports through
accumulated context or prior conversation history.

### Language Rule 6 -- Source and FPS Limitations

Report wording must reflect the `evidence_tier` AND `result_family_status` of
each finding. The Step 4 constraint block enforces this at prompt level.

**When `evidence_tier` is `direct` and `result_family_status` is `allowed`:**
-> State the finding in past tense as fact.

**When `evidence_tier` is `repeated_pattern`:**
-> "Finding supported by repeated structural frames."
-> "Opposition tendency inferred from recurring structural states."
-> "Player role pattern visible across repeated frames."

**When `evidence_tier` is `suggestive`:**
-> "Local visible action suggests..."
-> "Interpretation based on partial pitch visibility."
-> Do not state as confirmed fact.

**When `evidence_tier` is `escalated_confirmation`:**
-> "Moment confirmed by escalated higher-fps review."

**When `result_family_status` is `downgraded`:**
-> Include limitations_note in the finding.
-> "Confidence reduced due to [source_type] framing."
-> "Full-team structure not fully confirmed."
-> "Weak side not visible."
-> "Interpretation based on partial pitch visibility."

**When `result_family_status` is `downgraded` and evidence is thin:**
-> Do not produce the finding.
-> Note the suppression in `confidence_reliability_report.json`.

**Never produce at 1fps only (without escalated_confirmation):**
-> Exact pass events
-> Exact pressing triggers
-> Exact duel outcomes
-> Exact turnover timing
-> Whole-team structural claims from narrow or local-only framing
-> Complete player off-ball behaviour from local footage
-> Opposition-wide conclusions from action-follow-only clips

**When source is `dual_panoramic`:**
-> Do not assert spatial continuity across the two half-pitch views.
-> Qualify findings: "based on [home/away] half view."

**Accepted wording examples:**
- "local visible action suggests..."
- "full-team structure not fully confirmed"
- "weak side not visible"
- "confidence reduced due to ball-follow framing"
- "interpretation based on partial pitch visibility"
- "finding supported by repeated structural frames"
- "moment confirmed by escalated higher-fps review"
- "player role pattern visible across repeated frames"
- "opposition tendency inferred from recurring structural states"

---

## Step 4 -- Report Writing

Read `running_summary.json` and `pass_sequences.json` only.
Never read raw agent or merged window JSONs in this step.

**Pre-condition:** `report_readiness.json` must exist with `report_ready: true`.
Read `player_id_ceiling` from it before writing any player names.
Read `deep_skill_metrics.json` -- it is available for use in reports but is not a hard gate.
All metrics appear in reports. Severely-limited metrics (confidence < 0.2) carry a strong limitations_note.

The following constraint block must appear at the top of every Step 4 prompt
sent to the report-writing agent. Copy it verbatim -- do not summarise it.

```

=== MANDATORY PRE-FLIGHT (output this block verbatim before any prose) ===

Before writing the report, output this block exactly, filling in each line:

MATCH: [copy match field from match_config.json]
HOME TEAM CONFIRMED: [yes/no — did you read match_config.json lineups?]
AWAY TEAM CONFIRMED: [yes/no — did you read match_config.json lineups?]
PLAYER COUNT HOME: [count of home players in lineup]
PLAYER COUNT AWAY: [count of away players in lineup]
PREVIOUS MATCH KNOWLEDGE: EXCLUDED [write this word for word]
WINDOW REFERENCES IN REPORT: [count -- must be 0]
AGENT LANGUAGE IN REPORT: [count -- must be 0]

If WINDOW REFERENCES or AGENT LANGUAGE is not 0: fix those before submitting.
If HOME TEAM CONFIRMED or AWAY TEAM CONFIRMED is "no": read match_config.json now.

=== END MANDATORY PRE-FLIGHT ===


=== MATCH LENS REPORT CONSTRAINTS -- NON-NEGOTIABLE ===

THE VOICE YOU ARE WRITING IN:
You are an analyst who watched the match. You write clearly and specifically
about what was visible and what repeated. You are confident where the evidence
is strong, calibrated where it is partial, and silent where it is absent.
You do not tell anyone what to do. You do not evaluate players or staff.
You describe. You are never vague where you can be specific.

WHAT DESCRIPTIVE MEANS:
Descriptive is NOT cautious or hedged. It is specific and precise.
  WRONG (vague)    -> "Pressing was limited."
  WRONG (opinion)  -> "The pressing was poor."
  RIGHT (specific) -> "No coordinated pressure on the ball carrier was observed
                      across 8 of 9 first-half windows."

  WRONG (vague)    -> "The defence was deep."
  RIGHT (specific) -> "The back four sat at an average of 28m from their own goal
                      across windows 3-9."

  WRONG (coaching) -> "They should have pressed higher."
  RIGHT (stops)    -> "Front players moved toward the ball carrier in 3 of 9 windows.
                      In the remaining 6, no engagement was observed before the
                      ball was played forward."

The report stops at: what happened -> what repeated -> when and where.
It never moves to: what it means -> what should change -> what was wrong.

1. PAST TENSE throughout. "The shape was 4-4-2." Not "the shape is."

2. NO EVALUATIVE LANGUAGE.
   Never: poor, good, failed, accepted, lacked, should have, needed to,
   problem, weakness, wrong, right, dangerous, clinical, passive, lazy.
   These are judgments. Replace every one with description of what was seen.

3. NO COACHING ADVICE OR SUGGESTIONS.
   Never: "this should be reviewed", "consider", "worth noting that",
   "the team would benefit from", "this pattern warrants attention".
   These are instructions. Remove them entirely.

4. EVIDENCE TIERS -- language must match what the data supports:
   OBSERVED  -> directly in frames, counted data -> state as past-tense fact
   ESTIMATED -> pattern across multiple windows  -> "in approximately N of N windows..."
   INFERRED  -> interpretive -> "appeared to...", "the pattern suggested..."
   Never write a specific number without a source. Never write "67% possession"
   unless counted from pass_sequences.json or possession_by_window.

5. PLAYER ID CEILING -- from report_readiness.json:
   confirmed  -> full name "J. Smith (#7)"
   probable   -> "#7 (probable: J. Smith)"
   tentative  -> position label only: "the right winger"

5b. RESULT FAMILY STATUS:
    allowed    -> write normally
    downgraded -> write with one-line limitations_note inline after the claim
    (no suppressed state -- every family is attempted and reported)

    When a family could not be reliably assessed from the source footage,
    the report states this explicitly rather than omitting it:
    "Shape could not be reliably read from this source -- [limitations_note]."
    This is more useful to a coach than an absent section.

5c. SOURCE LIMITATIONS -- if downgraded, add inline note:
    "confidence reduced due to ball-follow framing"
    "full-team structure not fully confirmed at panoramic altitude"
    "weak side not visible from this source angle"

5d. DUAL PANORAMIC -- qualify spatial findings per half-pitch view.
    Do not assert cross-pitch spatial continuity.

6. GK AND AERIAL ACTIONS:
   Clean contact confirmed -> "The goalkeeper claimed the cross."
   GK in possession after -> "The goalkeeper was in possession following the cross."
   Contact ambiguous       -> "The goalkeeper came to the cross."
   GK near ball, no frame -> "The goalkeeper appeared to challenge for the cross."
   No usable frame         -> Do not describe -- note as unconfirmed.
   If resolved via Step 3i -> use recommended_report_wording from confirmation JSON.

7. DEEP SKILL METRICS:
   Include all metrics. For severely_limited metrics, add limitation_note inline.
   Format: "[metric_name]: [value] (confidence: [confidence])"
   Downgraded metrics include one-line limitations_note.
   Derived scores are Inferred tier -- never state as direct observation.

PIPELINE LANGUAGE DOES NOT APPEAR IN REPORTS.
Windows, frame counts, fps, confidence scores, and evidence tiers are the
backbone that makes analysis reliable. They justify conclusions internally.
They never appear as citations in the finished report.

  WRONG -> "In 8 of 9 first-half windows, pressing was observed at below 3/10."
  RIGHT -> "Pressing was minimal throughout the first half."

  WRONG -> "Confirmed at 0.88 confidence by dual agents at 18m47s."
  RIGHT -> "The first goal came after the defensive shape had shifted entirely
           to the right channel, leaving the left side unattended."

  WRONG -> "Average line height 28m from goal across windows 3-9."
  RIGHT -> "The back four sat deep, rarely advancing beyond the halfway line
           even when the team had sustained possession."

  WRONG -> "3fps segment confirmed GK caught the cross."
  RIGHT -> "The goalkeeper claimed the cross."

Write in football time, not pipeline time:
  "Throughout the first half" not "in 8 of 9 windows"
  "From around the hour mark" not "from window 11 onwards"
  "Consistently across the second half" not "in windows 11-21"
  "In the final 20 minutes" not "in the stoppage time windows"

The only exception: Data Quality Notes (section 15) may reference
technical details about source, frame count, and coverage gaps --
because that section exists for transparency, not for the coach.

BEFORE EACH SECTION, ask:
  "Am I describing what happened in the match, or citing my methodology?"
  If the latter: translate to football language.
  "Am I telling someone what to think, or what happened?"
  If the former: rewrite as description.
  "Is this specific enough to be useful, or so vague it adds nothing?"
  If the latter: add the moment, the position, the player, the time.

=== END CONSTRAINTS ===
```

### Confidence Filter Levels

Every report is generated at one of three confidence levels. Pass the level
as `confidence_level` to `build_deep_skill_metrics()` and inject it into the
report prompt.

**Level 1 -- Confirmed** (head coach, match-day preparation, team meeting)
- Metrics with sufficient sample and confidence >= 0.5 only
- Findings with direct or escalated_confirmation evidence tier only
- Individual observations with repeated or consistent frequency and high confidence
- No hedging language. No "Based on a preliminary sample." Confident past tense.
- Player profiles only if 3+ observations meet the threshold
- Typically 10-14 metrics visible depending on source and match

**Level 2 -- Standard** (default: analyst + coaching staff, weekly opposition prep)
- All metrics except context_only (context_only listed in Data Quality only)
- All findings except suggestive
- Individual observations with medium or high confidence
- Preliminary samples included with "Based on a preliminary sample:" prefix
- Context_only metrics listed in Section 15 with explanation
- Typically 14-17 metrics visible

**Level 3 -- Full** (analyst QA, pipeline review, data audit)
- Everything including context_only metrics, suggestive findings, low-confidence observations
- All evidence tiers stated explicitly
- Preliminary and context_only metrics included in main sections with labels
- Single observations included regardless of confidence
- Used for reviewing pipeline completeness, not coaching purposes

The confidence_level is stored in `deep_skill_metrics.json` as `confidence_level_applied`.
The report prompt reads this and applies the language rules accordingly.

**In the report prompt:** inject the confidence level and its rules:
```
CONFIDENCE LEVEL: [1 / 2 / 3]
LEVEL 1 rules: No "preliminary sample" notes. Omit suggestive findings.
               Only repeated/consistent observations. Confident language only.
LEVEL 2 rules: Include preliminary with note. Omit suggestive findings.
               Include single observations at medium+ confidence.
LEVEL 3 rules: Include everything. State evidence tiers explicitly where relevant.
```

---

### Report Level Filter

Before generating any report (4a, 4b, 4c), read `match_config.json` for `report_level`.
Run `report_filter.py` to get the level configuration block and inject it at the top
of the report prompt BEFORE the constraints block.

```
python report_filter.py "[MATCH_DIR]"
```

Three levels:

**brief** -- Pre-match brief (1-2 pages, 5 minutes to read)
Target: manager reading it 30 minutes before kick-off.
Sections: Formation, Key Threats, Pressing Triggers, Set Pieces, Players to Watch (max 3).
Profiles: Grade A observations only. 3-4 sentences per player.
Metrics: not included. Use metric data to inform the brief, not populate it.
Language: plain, direct, no hedging, no qualifiers.

**standard** -- Match analysis report (4-6 pages, 15-20 minutes to read)
Target: coaching staff reviewing the match.

**CONFIDENCE FILTER RULES (read before writing):**

Read `deep_skill_metrics.json` field `confidence_level_applied`.

If `confidence_level_applied = 1` (Confirmed):
  - Write ONLY metrics where confidence_level = 1 in the metrics list
  - Observations: frequency = repeated/consistent AND confidence = high only
  - No hedging. No "appears to". Confident past tense throughout.
  - Nothing in Section 15 -- if it was not confirmed, it is absent entirely
  - Add one line after the report title: "(Confirmed findings only)"

If `confidence_level_applied = 2` (Standard -- default):
  - Write metrics where confidence_level <= 2
  - Observations: confidence = medium or high
  - Preliminary metrics prefixed: "Based on a preliminary sample:"
  - Section 15 lists context_only metrics with one-line explanations

If `confidence_level_applied = 3` (Full -- analyst review):
  - Write everything. State "preliminary" or "context_only" inline where relevant.
  - Add one line after the report title: "(Full pipeline output)"

---

Sections: all tactical report sections 1-15.
Profiles: Grade A and B in main profile. Grade C in preliminary sub-section.
Metrics: active (sufficient + preliminary), prose format.
Language: analytical but accessible. Grade B uses qualified language.

**technical** -- Full technical analysis (8-12 pages, 30-40 minutes to read)
Target: analyst, head of analysis.
Sections: all sections including full data quality.
Profiles: all grades. Grade D listed as observed but unprofilable.
Metrics: all including context_only (with explanations).
Language: full analyst register. Evidence tiers explicit. Confidence scores permitted.

If `report_level` is not set in match_config.json: default to **standard**.

---

### 4a -- Tactical Report prompt

Send this prompt to generate `tactical_report.md`. Inject the constraint block,
source context, player ID ceiling, and data file paths before sending.

**Data files to read before writing:**
  - `running_summary.json` (includes `match_state_by_window`)
  - `pass_sequences.json`
  - `deep_skill_metrics.json`
  - `report_readiness.json`
  - `source_profile.json`
  - `match_config.json`

Use `match_state_by_window` to identify patterns that changed with the scoreline.
State these as observations:
  "The defensive line averaged 43% while the score was level, dropping to
   36% in windows where Lowestoft were losing."
Do not offer coaching responses. Report the pattern; the coach decides what to do.

---

**CRITICAL WRITING INSTRUCTION -- READ BEFORE ANYTHING ELSE:**

The report must read like an analyst who watched the match and wrote it up.
It must be clear, specific, confident, and useful to a coach reading it before
Tuesday training. It is not a data dump. It is not an evidence log. It is not a
cautious academic document.

Descriptive does not mean vague. It means MORE specific, not less.
"The front two maintained a 35-40 yard gap from the back four throughout" is
more descriptive -- and more useful -- than "the team had a passive shape."
"Not a single outfield player closed down the ball carrier during the 34 seconds
leading to the first goal" is more descriptive than "pressing was low."

Be confident in what the frames showed. Hedge only where the evidence genuinely
cannot support the claim. One clear hedge per section is enough -- do not hedge
every sentence.

---

```
=== MATCH LENS REPORT CONSTRAINTS -- NON-NEGOTIABLE ===

=== ABSOLUTE KNOWLEDGE PROHIBITION ===
Everything you write must come from the data files listed above
and the match_config.json for THIS match.

You have processed other matches. That knowledge does not exist for
this report. Players from other matches, results from other fixtures,
tactical patterns from other games -- none of it exists here.

The moment you find yourself writing something you "remember" rather
than something you read from the data files: STOP. Delete it.
If you cannot find it in the data files, it does not go in the report.
=== END PROHIBITION ===

[PASTE FULL CONSTRAINT BLOCK FROM SKILL.md STEP 4]
=== END CONSTRAINTS ===

You are writing a tactical match analysis report for a football coaching staff.
The report must be coach-useful, clearly structured, and readable in one sitting.

DATA FILES (read before writing -- do not write from memory):
- running_summary.json
- pass_sequences.json
- deep_skill_metrics.json (all metrics -- note severely_limited flag)
- source_profile.json
- report_readiness.json

PLAYER ID CEILING: [player_id_ceiling]

=== CONFIRMED PLAYER ROSTER -- THE ONLY NAMES YOU MAY USE ===
Read match_config.json now. Copy both lineups below before writing a single word.

HOME TEAM: [home_team]
  Starting XI: [paste all 11 players as "#N Name (pos)" from match_config.json]
  Substitutes: [paste all subs as "#N Name" from match_config.json]

AWAY TEAM: [away_team]
  Starting XI: [paste all 11 players as "#N Name (pos)" from match_config.json]
  Substitutes: [paste all subs as "#N Name" from match_config.json]

RULE: You may only write a player's name if it appears in the lists above.
If you find yourself writing a name that is not in these lists:
  STOP. Delete it. Do not include it, do not mention it, do not reference it.
This rule has no exceptions. Not even to say a player is "not listed."
If a player is not in the lineup: they do not exist for this report.

=== END ROSTER ===


SOURCE TYPE:       [source_type]
SOURCE LIMITATION: [source_limitations_note]

---

Write these sections in this order. Every section is required.
Do not merge sections. Do not skip sections.
Do not add a "for discussion" or "recommendations" section -- the report ends at
Data Quality Notes.

---

# Tactical Report: [Focus Team]
## [Home] vs [Away] -- [Competition] -- [Date]

---

## 1. Match Overview

Write 2-3 sentences that immediately tell the reader what kind of match this was
and how the result came about. This is the one place where narrative compression
is required -- the reader should know the shape of the game before reading anything
else. Draw from key_moments, goals, and formation_history.

Example register (not to copy -- understand the voice):
"[Team] conceded twice -- the first from an extended defensive hold that allowed
the opposition to reorganise, the second within 30 seconds of the second-half
restart. Between the goals and after them, the shape and pressing approach
remained unchanged. [Opponent] scored twice from seven attempts; [Focus Team]
created four openings and converted none."

Note: shot counts are structured data and can appear as facts.
All other claims must be grounded in what the frames showed.

No evaluation, no judgment. Describe the match plainly and specifically.

---

## 2. Match Phase Timeline

Present the match as a sequence of phases. Use a table:

| Phase | Approx time | Score | Character |
|---|---|---|---|
| [phase name] | [time range] | [score] | [one sentence -- what was happening] |

Draw from: formation_history, pressing_by_window, line_height_by_window,
key_moments (goals, subs, notable shifts). Minimum 5 phases, maximum 10.
One sentence per phase -- no coaching commentary.

---

## 3. Formation and Shape

**[Focus Team]**
Starting shape: [from formation_history window 1]
Shape in possession: [most common across formation_history]
Shape out of possession: [most common]
Changes: [note any formation shift -- which window, what changed]

Name the players in each line using player_id_ceiling for format.
Describe the shape in concrete positional terms -- who plays where, not just the number.

**[Opponent]**
Same structure. Draw from opposition findings[] where result_family is
opposition_structure or opposition_identity.
If all opposition_structure findings are low-confidence: write "Opposition shape could not be reliably read from this source -- [source_limitations_note]."

---

## 4. In Possession -- [Focus Team]

**Build-up style:** Direct or short? Who initiates? Where does it start?
Draw from: pass_sequences.json zone_start distribution, individual_observations for GK/CBs.

**Possession territory:** How much of the match was spent in each third?
Draw from: possession_by_window in running_summary.json (dominant_zone per window).
State plainly: "play was predominantly in the middle third" or
"territory was tilted toward the opponent's half across the second half."
Do NOT cite window counts -- translate to match language.

**Preferred side:** Which side carried more attacking sequences?
Draw from: pass_sequences.json zone_end and attacking field in running_summary.json.
Write in football terms: "attacking play was predominantly through the left channel"
or "the majority of attacking sequences went through the right side".
If the data gives a strong proportion, translate it: "over two-thirds of sequences
progressed through the left" is better than "in 14 of 19 sequences".

**Recurring routes:** What patterns repeated?
Draw from: pattern_reliability_score, build_up_route_diversity in deep_skill_metrics,
and pass_sequences zone_start/zone_end combinations.
Describe the patterns in plain match language: "build-up consistently started from
the goalkeeper and moved through the left centre-back before switching wide" rather
than listing zone codes or counts.
If one route dominated, say so clearly. If the team was varied, say that.

**Width:** Did wide players hold width? Did full-backs overlap?
Draw from: width_usage_score, individual_observations for wide players and full-backs.
Describe in positional terms: who was narrow, who was wide, which side had more
width, whether full-backs got forward. Name the players if player_id_ceiling permits.

**Final third:** How did attacks end? Crosses, cut-backs, shots, set pieces?
Draw from: shots_for, set_pieces, chance_creation_profile.

---

## 5. Out of Possession -- [Focus Team]

**Pressing approach:** Describe the defensive press. Was it high, mid, or low?
Did the front players engage the ball carrier or drop off?
Draw from: pressing_by_window scores. Translate the pattern to match language:
"pressing was consistently active / inconsistent / minimal / absent"
"front players engaged the ball carrier in the first half but dropped off after the second goal"
Do NOT quote the numerical score in the prose -- it belongs in section 14 only.
Note any phases where pressing changed sharply and describe what changed.

**Line height:** Where did the back four sit?
Draw from: line_height_by_window. Describe in football terms:
"deep, rarely advancing past halfway" / "high, squeezing the opposition into their own half"
/ "mid-block, sitting just inside their own half"
Note significant shifts with approximate match time and what triggered them.
Do NOT quote the metric value in metres in the prose -- it belongs in section 14 only.

**Vertical compactness:** What was the gap between the front two and back four?
Draw from: compactness_score, pressing_by_window, line_height_by_window combined.
State the gap in approximate metres where evidence supports it.

**Block shape:** Describe the defensive block -- horizontal width, which players
tucked inside, how the wide areas were covered.

**Rest-defence:** After losing the ball, how quickly did the team recover shape?
Draw from: rest_defence_security_score, notable_shifts in line_height_by_window.

---

## 6. Goals Conceded

For each goal conceded, write a structured description:

**Goal [N] -- [Scorer] ([Team]) [timestamp]**
Build-up: Describe the sequence leading to the goal. What happened, in order.
Use timecodes from confirmation_queue.json where resolved.
Draw from: key_moments, confirmation_queue resolved items, individual_observations.

Contact/finish: Describe the finish using GK/aerial wording tiers from constraints.

Root cause: State what structural condition allowed the goal -- the position of
players, the gap, the line, the sequence. One sentence. No blame language.

If confirmed by dual agents: state confidence.

---

## 7. Goals Scored

For each goal scored by the focus team, write a structured description using the
same format as Goals Conceded:

**Goal [N] -- [Scorer] ([Team]) [timestamp]**
Build-up: Describe the sequence. What happened, in order.
Contact/finish: Use GK/aerial wording tiers.
Contributing pattern: What structural or individual pattern created the opportunity.

If no goals were scored: write one sentence: "[Focus Team] did not score in this match."

---

## 8. Attacking Threat -- [Focus Team]

Draw from: shots_for, chance_creation_profile, attacking_support_score,
build_up_effectiveness_score, pass_sequences outcomes (shot/cross).
State all counts from structured data as facts.

How many attempts? From which zones? How were they created (open play vs set piece)?
What was the typical build-up length before a shot? Were there recurring sequences?
Did chances come from high-danger zones (box/penalty spot) or from distance?

Shot origin table (from running_summary.json shots_for):
| Time | Player | Origin | Type | Outcome | Sequence length |
|---|---|---|---|---|---|

---

## 9. Set Pieces

Draw from: set_pieces in running_summary.json.
Attacking: how many? from which zones? deliveries? outcomes?
Defensive: how many faced? marking system? GK behaviour? outcomes?

State all counts as facts. Note any set_pieces findings that carry limitations_note.

---

## 10. Transitions

**Attack to defence:** When possession was lost, describe the immediate response.
Did front players counter-press or drop? How quickly did the shape recover?
Draw from: transitions findings, rest_defence_security_score, transition_efficiency_score.
If transitions findings are all suggestive or low-confidence: note the source limitation.

**Defence to attack:** When possession was won, describe the forward movement.
How quickly did the team commit players forward?
Draw from: pass_sequences with zone_start in defending_third, transition findings.

---

## 11. Width and Spacing

Describe how the team occupied the pitch horizontally.
Did wide players hold their positions? Were the touchline areas used?
Was there asymmetry between the two sides?
Draw from: width_usage_score, individual_observations for wide players and full-backs,
spacing result family findings.
If spacing findings all carry limitations_note: state the limitation inline.

---

## 12. Tactical Variation and Substitutions

Did the shape or approach change during the match?
Draw from: formation_history -- note any windows where shape changed.
Draw from: pressing_by_window -- note any sharp increases or drops in pressing score.
Draw from: key_moments with type "sub" or "tactical_shift".

For each substitution:
- Timestamp and players (using player_id_ceiling)
- What changed in the shape or approach immediately after
- Whether the change was sustained across subsequent windows

For each non-substitution tactical shift:
- When it occurred (match time, not window number)
- What changed (line height, pressing intensity, shape) and what appears to have triggered it

If no changes were observed: "The shape and approach remained consistent across all observed windows."

---

## 13. Player Observations

Draw from: individual_observations in running_summary.json where team is the focus team.
Write profiles using the same format and 5-step process as the opposition Key Players section.

Minimum 2 observations to write a profile.
Include preferred_foot where logged (right / left / both / unknown).
Include physical_profile impressions where logged (height / pace / build).
Note observation outcome patterns where available (e.g. "won 3 of 4 aerial duels observed").

No ratings. No evaluation. No comparison to ideal performance.
Write what was seen and what repeated -- leave interpretation to the coaching staff.

---

## 14. Performance Metrics Summary

Draw from: deep_skill_metrics.json.

**Reading rules -- follow these exactly:**

1. For every metric where `context_only: true` -- do NOT write it in this section.
   Move it to Section 15 Data Quality Notes as a one-line note:
   "[metric readable name]: insufficient data for a reliable reading in this match."

2. For every metric where `prose_interpretation` is not null -- use that text
   directly as the entry. Do not rephrase it. Do not convert it back to a number.
   The prose_interpretation is the output of a translation step and is ready to use.

3. Where two metrics combine into a stronger observation, write them as one
   compound finding rather than two separate lines. The most important combination:
   pattern_reliability_score + build_up_route_diversity = one finding about
   build-up predictability. For example:
   "62% of sequences used a single dominant route (own third to midfield) across
   only 3 observed route patterns -- the build-up shape is readable and the likely
   options can be covered in preparation."

4. Where sample_status is "preliminary" -- include the finding but open with
   "Based on a preliminary sample:" so the reader knows the confidence is limited.

5. Do not include raw numbers (0.63, 2.5/10) alongside the prose unless they
   add information the prose does not already convey.
   Exception: line height in metres should always be quoted.

**Structure:**

Write this section as short prose paragraphs grouped by theme, not as a table.
A table is appropriate for a dashboard. This is an analyst's document.

Group as:
  Out of possession -- compactness, pressing intensity, pressing triggers,
                      line height range, rest defence
  In possession     -- build-up effectiveness, route patterns, predictability,
                      attacking support
  Creating chances  -- chance creation profile, attacking support score

Each group is 2-4 sentences. If a group has no non-context_only metrics, omit it.

Close the section with one sentence:
"Source: [source_type]. [source_limitations_note]."

**Label map** (snake_case -> readable):
  compactness_score              -> "Defensive compactness"
  pressing_intensity_score       -> "Pressing intensity"
  pressing_trigger_consistency   -> "Press triggers"
  line_height_range              -> "Defensive line height"
  rest_defence_security_score    -> "Rest-defence shape"
  build_up_effectiveness_score   -> "Build-up progression"
  pattern_reliability_score      -> "Build-up route patterns"
  build_up_route_diversity       -> "Route variety"
  predictability_score           -> "Build-up predictability"
  chance_creation_profile        -> "Chance creation"
  attacking_support_score        -> "Attacking support depth"
  transition_efficiency_score    -> "Transition efficiency" -- write as two findings:
                                    "Attack transition: [attack_threat_rate prose]"
                                    "Defensive transition: [defence_exposure_rate prose]"
  set_piece_delivery_profile     -> "Set piece delivery" (context_only if < 2 set pieces)
  width_usage_score              -> "Wide play" (always context_only with ball-follow footage)
  halfspace_occupation_score     -> "Half-space activity" (context_only without lateral zones)

---

## 15. Data Quality Notes

Source: [source_type] -- [source_limitations_note]
Frames analysed: [total from job_log.json]
Windows complete: [windows_complete from running_summary.json]
Data gap windows: [list from running_summary.json]
Unresolvable frames: [from rerun_queue.json]
Downgraded result families: [list from result_family_gates.json where status=downgraded]
Confirmation items: [summary from confirmation_queue.json]

**Metrics with insufficient data (context_only):**
List every metric where context_only: true from deep_skill_metrics.json.
For each one, write one sentence explaining why it could not produce a reliable reading:
  - width_usage_score: "Wide play could not be reliably measured -- ball-follow footage
    does not consistently capture wide-channel activity."
  - set_piece_delivery_profile: "Only [N] set piece(s) observed -- insufficient for a
    delivery pattern. Reference broader season data for set piece preparation."
  - transition_efficiency_score: "Fewer than 5 transitions observed -- not enough for a
    reliable efficiency reading."

**Low-confidence metrics (confidence < 0.4):**
List metric name and one-line reason. Do not describe these as findings.

For each downgraded result family: one line stating what was attempted and what
the source limitation was. Do not leave any family unaddressed.

---
Output as markdown. No preamble before the title. No section after Data Quality Notes.
No "for discussion" section. No recommendations. No coaching advice.
```

**Shot Map & Analysis section** -- compile all shots from `running_summary.json`
`shots_for` and `shots_against` arrays:

**Shot Origin Table** -- one row per shot, sorted chronologically:
```
| Time | Team | For/Against | Player | Origin Zone | Type | Outcome | Target Zone | Seq Length |
```

**Zone Summary -- Shots For vs Shots Against**:
```
| Origin Zone           | Shots For | Shots Against |
| Central / 6yd box     |           |               |
| Central / penalty     |           |               |
| Central / edge        |           |               |
| Left channel          |           |               |
| Right channel         |           |               |
| Outside box           |           |               |
```

**Outcome Summary**:
```
| Outcome    | Shots For | Shots Against |
| Goal       |           |               |
| On target  |           |               |
| Off target |           |               |
| Blocked    |           |               |
| Post/Bar   |           |               |
| TOTAL      |           |               |
```

### 4b -- Opposition Report prompt

Send this prompt to generate `opposition_report_[team].md`.

```
You are writing an opposition scouting report. Follow every rule in the
MATCH LENS REPORT CONSTRAINTS block -- they are not optional.

This report is about the OPPONENT, not the focus team.
Draw ONLY from findings[] where analysis_scope is "opposition".
Include findings from all result families. For downgraded families,
include the limitations_note inline.

DATA FILES:
- running_summary.json    -- individual_observations, key_moments, flagged_moments,
                            set_pieces, shots_against, match_state_by_window
- pass_sequences.json     -- pass chains
- deep_skill_metrics.json -- only metrics with analysis_scope "opposition" or "match"
- source_profile.json     -- source limitations

PLAYER ID CEILING: [player_id_ceiling]
SOURCE TYPE:       [source_type]
SOURCE LIMITATION: [source_limitations_note]
OPPONENT:          [opponent team name]

Write the following sections. Past tense throughout.
No coaching advice. No suggested solutions. No evaluation of quality.
Strengths, weaknesses, and traits are permitted as observations -- state
the pattern, do not prescribe a response to it.

---
# Opposition Report: [Opponent] -- Observed [Date]

## Executive Summary

Three to four sentences only. One sentence each on:
- How they organised defensively
- How they built play and where attacks came from
- One or two key players who were most prominent
- Anything that changed materially during the match

## Formation and Shape

Draw from: findings where result_family is opposition_structure or opposition_identity.
State: starting shape, shape in possession, shape out of possession.
If all opposition_structure findings are suggestive, write:
"Opposition structure could not be reliably read from this source -- [source_limitations_note]."

## Defensive Behaviour

Draw from: findings where result_family is opposition_pressing or opposition_structure.
Describe: block type (high/mid/low), press triggers if any, line height if available.
Use line_height_range from deep_skill_metrics if available (give the metres figures, not
the percentage, and state the categorical band: very_high / high / medium / low / very_low).
Use prose_interpretation from pressing_intensity_score and pressing_trigger_consistency
if available and not context_only.

**Pressing structure:** beyond intensity and triggers, describe the SHAPE of the press:
  - How many players press simultaneously (first press line count)?
  - Do they press with the forwards, with midfield support, or in isolation?
  - Do they try to force play wide (funnel to the touchline) or force play central?
  - When the press is broken, how quickly do they recover into their block?
Draw from opposition_pressing findings and pressing_by_window observations.
If only suggestive evidence exists, use qualified language accordingly.

**Defensive width:** does the team defend with a narrow shape (protecting central
channels) or a wide shape (defending from touchline to touchline)? This is distinct
from line height. Draw from spacing and territory findings.

## Build-Up and Attacking Patterns

Draw from: findings where result_family is opposition_build_up or opposition_patterns.
Draw from: chance_creation_profile and pattern_reliability_score in deep_skill_metrics
if available and not context_only.
Describe: distribution style, preferred zones, route patterns, how chances were created.
Reference match_state_by_window if patterns changed with the scoreline.

**Aerial threat:** explicitly address whether this team uses aerial play as a primary
attacking method. Draw from:
  - GK distribution observations (short vs long bias)
  - individual_observations with aerial_ability or hold_up_play action_category
  - shot_attempts where shot_type is "header"
  - Any long-ball patterns in pass_sequences (very short sequences ending in threat)

At Step 7 non-league level, aerial threat via long distribution is often the
primary attacking weapon. Note: the dominant target player for long balls if
identifiable, the zone they target (left/central/right channel), and whether
they win the first or second ball more consistently.

## Shot Map

Draw from: shots_against in running_summary.json.
Present origin zone table and outcome distribution.
If shots_against is empty, note the gap and reference key_moments for goal descriptions.

## Transitions

Draw from: findings where result_family is opposition_transitions.
State direction (defending-to-attacking or attacking-to-defending), trigger type if
logged, number of players in front of ball.
If fewer than 3 transitions were observed: note this as a thin sample.

## Match State Behaviour

Draw from: match_state_by_window in running_summary.json.
Cross-reference with line_height_by_window, pressing_by_window, and formation_history
to identify patterns that changed with the scoreline.

This section answers: did the team behave differently when winning, losing, or level?

Structure as observations only -- no coaching advice:
  - Line height when winning vs when level (quote metres, not percentages)
  - Pressing intensity when losing vs when level (quote 0-10 avg)
  - Any formation or shape changes after goals
  - Whether build-up style changed (more direct when losing?)

If match_state_by_window shows only one state throughout (e.g. the team was never
leading), state that explicitly: "Felixstowe were level or losing throughout -- "
no comparison of winning behaviour is available."

If fewer than 3 windows in any state: note as insufficient to establish a pattern.

## Tactical Variations

Draw from: key_moments where type is "sub", "tactical_shift", or "formation_change".
Cross-reference with formation_history before and after the event.

Cover:
  - Any substitution and its immediate effect on shape or line height
    (e.g. "a striker replaced a midfielder at 68m -- the shape shifted from 4-4-2
    to 4-3-3 and the defensive line pushed 6m higher in the following window")
  - Any visible tactical shift not related to a substitution
  - Whether the team became more direct, more compact, or more aggressive
    after a goal (cross-reference match_state_by_window)

If no substitutions or tactical shifts were observed: write one sentence:
"No substitutions or visible tactical shifts were recorded in this match."

## Set Pieces

Draw from: set_pieces in running_summary.json where team is the opponent.

**Attacking set pieces** (when the opponent takes corners, free kicks, throw-ins):
For each delivery type and zone, state: delivery type, target zone, bodies in box,
runners observed (zones and assignments if captured), outcome.
Note any consistent pattern across multiple deliveries.
If the set_piece_delivery_profile is context_only (fewer than 2 set pieces observed):
write one sentence: "Insufficient set piece observations in this match for a delivery
profile -- [N] observed."

**Defensive set pieces** (when the opponent defends corners and free kicks against them):
State: marking system (zonal/man/mixed), number of bodies in the box, who clears
near post and far post if identifiable, goalkeeper behaviour (claims vs punches,
command of box).
If the opponent's defensive set piece organisation was not clearly visible
(ball-follow footage away from the defensive zone): note the limitation.

## Key Players

Draw from: individual_observations in running_summary.json where team is the opponent.
Filter to opponent players only. Exclude focus team observations from this section.

**How to build each profile:**

Step 1 -- Gather all observations for each player. Group by observation_type:
  strength, weakness, trait, neutral.

Step 2 -- For each player, identify the story. A profile should read as a coherent
  picture of a player -- their role, their recurring patterns, where they are
  most active, and what happens consistently. It should not be a list of isolated
  actions.

Step 3 -- Where the same action_category appears multiple times (e.g. three entries
  for "aerial_ability"), consolidate into one observation with the highest frequency
  label and quote specific timestamps:
  "Won aerial duels consistently in the right channel -- observed at 12m05s,
   27m40s, and 34m15s."
  Not three separate bullet points.

Step 4 -- Lead the profile with the most significant observation (highest frequency,
  highest confidence, or most tactically relevant), not necessarily the first
  timestamp.

Step 5 -- Write the opening sentence as a positional + behavioural description:
  "The right winger (#11) operated predominantly in the left channel on the
   attack, cutting inside repeatedly and linking with the central striker."
  Not: "A player on the right wing who was active."

**Format for each player:**

### [Player identifier -- use player_id_ceiling format]

**Role:** [position] -- [one phrase describing their function in the team shape, e.g.
  "target striker holding up play", "ball-carrying right back pushing into midfield",
  "defensive midfielder screening the back four"]

[Opening sentence: most prominent observable pattern -- what they did most, where,
and what it led to]

**Profile grade: [A/B/C/D]** -- [one sentence from player_profile_grade summary]

For Grade A and B profiles, write the full profile:

**Strengths:**
- [consolidated observation] ([frequency]) [Grade A/B obs only -- do not include C/D here]

**Vulnerabilities:**
- [consolidated observation] ([frequency]) [Grade A/B obs only]

**Traits and tendencies:**
- [consolidated observation] ([frequency]) [Grade A/B obs only]

Omit any sub-heading where that observation_type has no Grade A/B entries.

If any Grade C observations exist, add:
**Preliminary observations (single sighting or limited visibility):**
- [observation] -- [obs_grade: C, confidence: X, frequency: X]

Do not include Grade D observations in the profile at all.
Write each bullet as a complete descriptive sentence, not a phrase.

For Grade C profiles: write only the preliminary observations block.
For Grade D profiles: do not write a profile. List the player under:
"Players observed but insufficient data for a profile: #N [position label]"

**Action category context** -- use these to write more specific observations:

| action_category | What to describe |
|---|---|
| ball_carrying | Where they carried, direction, distance, outcome |
| distribution | Target type (short/long/switch), accuracy, frequency |
| hold_up_play | Body position, ability to retain under pressure, link partners |
| movement_off_ball | Run type (in behind / dropping deep / wide), timing |
| finishing | Shot zone, type, outcome, approach before shot |
| set_piece_delivery | Delivery type, target zone, consistency |
| pressing_behaviour | Trigger used, intensity, whether cover shadow was established |
| defensive_positioning | Depth, cover provided, spacing to nearest player |
| aerial_ability | Zone, win/loss rate, technique observed |
| duels | Zone, outcome, whether pressure was sustained |
| recovery_runs | Speed of recovery, distance covered, direction |
| gk_distribution | Short/long split, targets, accuracy |
| gk_positioning | Line height, sweeping behaviour, command of box |
| gk_shot_stopping | Technique, direction, saves/concessions |
| positional_tendency | Where the player consistently appears on the pitch |
| body_orientation | Which direction they face when receiving, implications |
| link_up_partner | Who they combine with, combination type, zone |

**Frequency label map:**
  single     -> "observed once"
  repeated   -> "repeated across multiple windows"
  consistent -> "consistent throughout"

**Ordering:** highest observations count first. Where counts are equal, lead with
the player most central to the team's attacking or defensive pattern.

**Pass network cross-reference:**
Before writing Key Players, check pass_sequences.json for the most frequent
player-to-player combinations (by shirt number in the sequence chains).
The most common receiving player in the sequence chains is usually the most
central player tactically. Note this player's combination partners in their profile.
Example: "The most frequent combination in observed sequences was #8 -> #10,
appearing in [N] of the sequences where #10 received."

**Minimum threshold:** 2 or more observations to write a profile. Single observations
are included only for directly observed key moments (goal, assist, disciplinary,
direct duel win/loss confirmed by escalated_confirmation evidence tier).

**Profile count:** write profiles for every player meeting the threshold.
Do not cap at a fixed number. If the match yielded only observations for 3 players,
write 3 profiles. If it yielded 9, write 9.

If individual_observations is empty or all observations are low-confidence:
"Individual player profiles could not be produced from this match -- source coverage
and player identification were insufficient. Opposition player data available from
match record only (goals, cards, substitutions)."

## Goalkeeper -- Observed Behaviour

The goalkeeper gets a dedicated section regardless of observation count.
Draw from:
  - individual_observations where position is "gk" and team is the opponent
  - set_pieces outcomes (claims, punches, distribution)
  - confirmation_queue items resolved for gk_claim / gk_punch / gk_parry
  - key_moments referencing the goalkeeper

Write using the same profile format as Key Players.
Use GK/aerial wording tiers from Report Standards (Language Rule 4) for all
contact descriptions -- never describe a claim or punch with certainty unless
directly confirmed by escalated_confirmation evidence tier.

**GK-specific observations to cover if data is available:**
  - Distribution: short or long bias, typical target, setup time on goal kicks.
    For every goal kick observed, note: short or long, which direction, which
    player was targeted. Log as individual_observations with action_category
    "gk_distribution". After 3+ observations, a pattern emerges.
    Example: "Goal kicks: long to #9 on right channel -- observed at 07m30s,
    18m15s, 34m50s (3 of 3 observed goal kicks)."
  - Positioning: line height, sweeping tendency, command of box on crosses
  - Shot stopping: direction, technique, where saves were made
  - Aerial: how they handled inswinging deliveries, corner claim rate
  - Pressing trigger: if the goalkeeper being in possession is the opponent's
    press trigger (from pressing_trigger_consistency) -- note this explicitly
    here as well as in Defensive Behaviour. If the opponent always presses on
    GK possession, this GK's distribution habits become a tactical priority.

If no GK observations were captured beyond match record:
"No goalkeeper-specific observations were recorded in this match beyond
confirmed match events."

## Data Quality Notes

List:
- Context_only metrics and why each could not produce a reliable reading
- Downgraded result families and what that meant for observation confidence
- Any windows with data gaps that affect the opposition read
State concisely. One line per item.

---
Output as markdown. No preamble before the first heading. Past tense throughout.
```

### 4c -- Flagged Moments prompt

Send this prompt to generate `flagged_moments.md`.

```
You are compiling the flagged moments report from a football match analysis.
Follow every rule in the MATCH LENS REPORT CONSTRAINTS block.

DATA FILES:
- running_summary.json    -- flagged_moments, key_moments, confirmation_queue results

SOURCE TYPE:       [source_type]
SOURCE LIMITATION: [source_limitations_note]
PLAYER ID CEILING: [player_id_ceiling]

Instructions:
1. Pull all entries from flagged_moments and key_moments in running_summary.json
2. Deduplicate by timestamp ± 30 seconds
3. Keep only moments where confidence is "high" or "medium"
4. Sort chronologically
5. For moments from downgraded families: include them with their limitations_note
6. If a moment was in confirmation_queue and resolved: use evidence_tier "escalated_confirmation"
7. Target 10-20 moments. If fewer high/medium confidence moments exist, report what is there.

For each moment, state:
- What happened (description only, past tense)
- Result family it belongs to
- Evidence tier (direct / repeated_pattern / escalated_confirmation)
- Confidence level
- Source limitation if result_family was downgraded
- Do NOT state why it is tactically significant -- that is coaching interpretation

Output format (markdown):

## Moment [N]: [Title]
**Timestamp:** [MMmSSs]
**Team:** [home team name / away team name]
**Result family:** [result_family]
**Evidence tier:** [direct / repeated_pattern / escalated_confirmation]
**Confidence:** [high / medium]
[If downgraded: **Source note:** [limitations_note]]
**What happened:** [description in past tense -- no opinion, no evaluation]
**Key frames:** frame_XXmYYs.jpg[, frame_XXmYYs.jpg]

---
Output as markdown. No preamble before the first heading.
```

### 4d -- Pass Network Report prompt

Send this prompt to generate `pass_network.md`.

```
You are writing a pass network report from football match analysis.
Follow every rule in the MATCH LENS REPORT CONSTRAINTS block.

DATA FILES:
- pass_sequences.json     -- all pass chains from the match
- deep_skill_metrics.json -- pattern_reliability_score and build_up_route_diversity

SOURCE TYPE:       [source_type]
SOURCE LIMITATION: [source_limitations_note]
PLAYER ID CEILING: [player_id_ceiling]

If all build_up findings are low-confidence from source limitations:
  state "Pass network analysis was limited -- [source_limitations_note]"
  and produce what is available with appropriate caveats.

If build_up is downgraded: produce the report but prefix with:
  "Pass network findings are approximate -- [source_limitations_note]."

Instructions:
Read pass_sequences.json. Group sequences by player involvement.
Produce all seven sections below in order.
State all counts as facts (they are direct structured data).
Do not interpret why routes are used -- describe what was used and how often.

---
# Pass Network: [Home] vs [Away]
[If build_up downgraded: include source caveat here]

## 1. Node Map
For each player appearing in pass chains:
- Total involvements (passes sent + received)
- Top 3 players they received from (with count)
- Top 3 players they passed to (with count)
Sort by total involvements descending.
Use player_id_ceiling for name format.

## 2. Most Used Routes
Top 10 player-to-player combinations by frequency.
Format: [Player A] -> [Player B]: [N] times

## 3. Progressive Routes
All sequences where progressive: true.
Group by zone channel: left (zone_start or zone_end includes left) /
central / right.
State total count per channel.

## 4. Dead End Patterns
All sequences ending in lost_possession or clearance.
Identify which player-to-player links most commonly precede dead ends.
State counts only -- no interpretation.

## 5. Threat Sequences
All sequences ending in shot or cross.
For each: state the full chain, zone_start, zone_end, sequence length.
State average sequence length before threat.

## 6. Bypassed Players
Players in the lineup (from match_config.json) who appear in fewer than
3 pass sequences total. List them with their involvement count.

## 7. First Half vs Second Half
Compare pass network shape for windows in first half vs second half.
Use pattern_reliability_score and build_up_route_diversity from
deep_skill_metrics.json if not suppressed.
State whether the most common route changed between halves.

---
Output as markdown. No preamble before the first heading.
```

---

## Step 5 -- Word Conversion

```bash
PYTHON="C:\Users\dbmux\AppData\Local\Programs\Python\Python313\python.exe"
$PYTHON "C:\Users\dbmux\.claude\skills\match-analysis\scripts\md_to_docx.py" "[MATCH_DIR]"
```

Converts all four `.md` files to `.docx`.

---

## Pre-Match Checklist

**Before starting:**
- [ ] Video file path confirmed
- [ ] `frames/` cleared from previous match
- [ ] `agent_logs/` directory created and empty
- [ ] `pass_sequences.json` initialised: `{"match":"","focus_team":"","total_sequences":0,"sequences":[]}`
- [ ] `running_summary.json` initialised: `{"windows_complete":0,"formation_history":[],"pressing_by_window":[],"line_height_by_window":[],"shots_for":[],"shots_against":[],"flagged_moments":[],"key_moments":[],"individual_observations":[],"set_pieces":[],"possession_by_window":[],"data_gap_windows":[]}`
- [ ] `job_log.json` started via JobLogger

**After Step 1d-1e (before Step 1f):**
- [ ] `match_config.json` verified = true
- [ ] enrichment_level and player_id_ceiling noted
- [ ] kit colours confirmed
- [ ] attack directions confirmed (auto-detected)

**After Step 1f (before analysis):**
- [ ] `source_profile.json` exists and classification_confidence ≥ 0.6
- [ ] `result_family_gates.json` exists
- [ ] Suppressed and downgraded families reviewed -- understand what the footage cannot support
- [ ] If `dual_panoramic`: split_aware confirmed before proceeding

**After Step 3h (before Step 3j):**
- [ ] `ground_truth_check.json` -- missed = 0

**After Step 3j (before Step 3i):**
- [ ] `report_readiness.json` -- report_ready = true

**After Step 3k (before Step 4):**
- [ ] `deep_skill_metrics.json` -- exists
- [ ] `low_confidence_metrics` list reviewed -- noted in Data Quality section
- [ ] `avg_confidence` reviewed -- if < 0.5, note source limitations prominently

**After Step 4:**
- [ ] All four reports reviewed against the four quality axes:
  - Clarity -- each section makes one clear point without repetition
  - Neutrality -- no evaluative language, no opinion, no coaching advice
  - Usefulness -- readable and actionable before Tuesday training
  - Evidence discipline -- every specific claim traces to structured data or is labelled Estimated/Inferred
- [ ] `confidence_reliability_report.json` reviewed -- downgraded families and limitation notes reviewed
- [ ] All downgraded family findings carry limitations_note inline
- [ ] Downgraded findings carry limitations_note inline
- [ ] No whole-team structural claims from sources that cannot support them

---

## Quick Reference -- Failure Modes

| Problem | Fix |
|---|---|
| Agent outputs prose instead of JSON | Re-prompt: "Raw JSON only. Nothing before or after." |
| Agent loses kit colour mid-window | Re-state kit colours at top of every prompt |
| Frame confidence consistently low across window | Check lighting/camera -- flag as data_gap_window |
| Confidence aggregator finds >30% low-conf in window | Flag window; note in data_gap_windows; do not force re-run unless key event |
| Re-run frame still unresolvable | Set unresolvable: true; exclude from metric aggregation |
| Ground truth event missed by both agents | Mandatory deep scan re-run for that window before Step 4 |
| Pass chain breaks mid-sequence | Log partial chain with "incomplete": true |
| Line height calibration drifts | Re-anchor to penalty box edge (16%) in prompt |
| Formation inconsistent across windows | Use agent_01 merged as baseline; flag only if 2+ consecutive windows agree on change |
| running_summary.json missing a window | Re-run update_running_summary for that merged file |
| Duplicate flagged moments | Deduplicate by timestamp ± 30 seconds before writing flagged_moments.md |
| confirmation_queue empty but GK/cross events present | Check agent prompt includes confirmation_queue in schema |
| Confirmation segment too short to resolve event | Increase window_seconds in extract_segment() |
| Event still ambiguous after 3fps re-extraction | Re-extract at 5fps; if still unclear, use softest wording tier |
| Report uses exact % without structured backing | Replace with estimated/approximate language per evidence tier rules |
| Opposition report contains coaching advice | Remove -- report stops at observation and pattern, never instruction |
| report_readiness.json blocking issues | Resolve each blocking issue, re-run build_readiness_check.py |
| Step 4 runs before report_readiness.json exists | Hard stop -- build_readiness_check.py must run first |
| Player names appear despite tentative ceiling | Review -- player_id_ceiling from report_readiness.json must cap all names |
| confirmation_queue has ineligible event types | Check ELIGIBLE_TYPES list in Step 3i -- agents may be over-queuing |
| job_log.json missing a step | Re-run the step; JobLogger.step_end() writes on completion |
| enrichment_level identity_only causes player ID gaps | Expected -- use position labels; note in Data Quality section |
| source_profile.json missing at Step 3j | Run Step 1f (source_profiler.py) before continuing |
| Source classification_confidence < 0.6 | Re-run Step 1f with more sample frames or manually set source_type |
| dual_panoramic: spatial claims produced | Remove -- each half view must be interpreted independently |
| Downgraded finding missing limitations_note | Add inline note from source_profile.json |
| Downgraded finding has no limitations_note | Add source note from source_profile.json source_limitations_note field |
| escalation cap hit with goals remaining | Expected if goals > 10 -- escalation_router.py handles this automatically |
| result_family_gates.json shows unexpected suppressions | Review source_profiles_config.json rules for this source type |
| Whole-team shape claim from veo_ball_tracking | Qualify -- add 'visible zone only' limitations note |
| Player role claim from broadcast_tv | Qualify -- add broadcast framing limitation note |
| Opposition structure claim from ball-follow source | Qualify -- 'visible zone only; full structure not confirmed' |
| evidence_tier suggestive in final report | Replace with qualified wording or escalate the finding |
| deep_skill_metrics.json missing | Run Step 3k before Step 4 |
| Severely limited metric missing limitation_note | Add limitation_note from deep_skill_metrics.json |
| Metric confidence unexpectedly low | Check source global cap and evidence tier of supporting findings |
| Derived score very low confidence | Component metrics are severely limited -- note source limitations |
| Player metrics all severely limited | Source coverage insufficient for player analysis -- note in Data Quality |
| avg_confidence below 0.5 | Source limitations are significant -- note in Data Quality section |
