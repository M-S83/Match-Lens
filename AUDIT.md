# Pipeline File I/O Audit

Generated: 2026-05-14  
Scope: All Python scripts in `C:\Users\dbmux\.claude\skills\match-analysis\scripts\`  
Instruction: Source code only — no match directories inspected on disk.

---

## Findings

### F1 — Window ID key inconsistency (reversed fallback order)

**Severity:** HIGH — causes silent data corruption when both keys are present  
**Files affected:** `pipeline_runner_v2.py`, `pipeline_runner.py` (via `shots_log_addition.py`), `window_plan.py`

`window_plan.py` writes `agent_id` as the canonical window identifier (the two-digit sequence number, e.g. `"07"`). It does NOT write a `window_id` key.

`pipeline_runner_v2.py::get_window_frames()` reads the window with:

```python
w.get("agent_id", w.get("window_id", ""))
```

`pipeline_runner_v2.py::build_structural_prompt()` reads the same window dict with:

```python
w.get("window_id", w.get("agent_id", ""))
```

The fallback order is reversed between the two callers. If a window dict ever acquires both keys (e.g. from a merge or manual edit), `build_structural_prompt` will silently use `window_id` while `get_window_frames` uses `agent_id`, potentially pairing the wrong prompt with the wrong frames.

`shots_log_addition.py::_window_for_minute()` also does `window.get("window_id", window.get("agent_id", ""))` — same non-canonical order.

**Suggested fix:** Standardise on `agent_id` as the only window identifier key. Remove all fallbacks to `window_id` or rename the canonical key to `window_id` consistently throughout.

---

### F2 — Merged filename naming drift (Pattern A vs Pattern B)

**Severity:** HIGH — causes merge file lookup failures across pipeline steps  
**Files affected:** `merge_utils.py`, `setpiece_writeback.py`, `escalation_router.py`, `pipeline_runner_v2.py`

**Pattern A** (written by `merge_utils.py`):
```
agent_logs/agent_{agent_id}_{safe_label}_merged.json
```
e.g. `agent_07_20m03s–25m03s_merged.json`

**Pattern B** (looked up by `setpiece_writeback.py` direct path):
```
agent_logs/agent_{window_id}_merged.json
```
e.g. `agent_07_merged.json`

`setpiece_writeback.py::writeback_all_bursts()` first attempts the exact path `agent_{window_id}_merged.json`, which will never exist because `merge_utils.py` always appends `_{safe_label}`. It only succeeds via the glob fallback. This means every single writeback call silently degrades to the glob fallback path.

`escalation_router.py` reads from THREE glob patterns:
1. `agent_logs/*_merged.json`
2. `merged_windows/merged_*.json`
3. `agent_logs/agent_*_merged.json`

Patterns 1 and 3 overlap completely — every file matching pattern 3 also matches pattern 1. The dedup logic prevents double-reading, but the redundancy signals historical naming drift that was patched over rather than resolved.

`pipeline_runner_v2.py` (Phase 2 structural file lookup) uses the exact path `agent_{wid}_structural.json` then falls back to glob `agent_{wid}_*_structural.json` — same first-attempt-will-always-miss pattern as `setpiece_writeback.py`.

**Suggested fix:** Remove the exact-path first-attempt in `setpiece_writeback.py` and `pipeline_runner_v2.py` Phase 2. Use only the glob pattern. Remove the redundant third glob pattern in `escalation_router.py`.

---

### F3 — Source profile key polymorphism (four different key names for the same field)

**Severity:** HIGH — some scripts will silently read an empty string for the limitations note  
**Files affected:** `source_profiler.py`, `pipeline_runner.py`, `merge_utils.py`, `build_readiness_check.py`, `generate_flagged_moments.py`, `generate_pass_network.py`

`source_profiler.py` writes `source_profile.json` with the key `source_limitations_note`.

Reading scripts use four different key names:

| Script | Key(s) read |
|---|---|
| `pipeline_runner.py::_build_source_injection()` | `source_limitations_note` (correct) |
| `merge_utils.py` | `source_prof.get("source_limitations_note") or source_prof.get("notes", "")` |
| `build_readiness_check.py` | `source_prof.get("source_limitations_note") or source_prof.get("notes", "")` |
| `generate_flagged_moments.py` | `sp.get("source_limitations") or sp.get("limitations_note") or ""` |
| `generate_pass_network.py` | `sp.get("source_limitations") or sp.get("limitations_note") or ""` |

`generate_flagged_moments.py` and `generate_pass_network.py` will always get an empty string because they read keys (`source_limitations`, `limitations_note`) that `source_profiler.py` never writes.

**Suggested fix:** Standardise all reads to `source_limitations_note`. Remove the fallback key aliases.

---

### F4 — Set piece field name drift (`marking` vs `marking_system`)

**Severity:** MEDIUM — causes data loss in the deep metrics calculation  
**Files affected:** `accumulator.py`, `setpiece_writeback.py`, `deep_skill_metrics.py`

`accumulator.py::validate_set_piece()` normalises incoming data:
```python
"marking": sp.get("marking") or sp.get("marking_system")
```
It writes the normalised value under the key `marking`.

`setpiece_writeback.py::BURST_CONFIRMS` contains `"marking_system"` — so burst writeback patches the record with key `marking_system`.

`deep_skill_metrics.py::calc_set_piece_delivery_profile()` reads only `sp.get("marking_system", "unknown")` — never reads `marking`.

Result: records normalised by `accumulator.py` to `marking` will read back as `"unknown"` in `deep_skill_metrics.py`. Records patched by `setpiece_writeback.py` with `marking_system` are readable by `deep_skill_metrics.py` but may be missed by anything reading `marking`.

**Suggested fix:** Pick one canonical key (`marking_system`). Update `accumulator.py::validate_set_piece()` to write `marking_system`. Update `deep_skill_metrics.py` to also try `sp.get("marking") or sp.get("marking_system", "unknown")` as a transitional fallback.

---

### F5 — Formation schema drift (`shape_in_possession` vs `formation.home`/`formation.away`)

**Severity:** MEDIUM — dual-agent merge loses formation data silently  
**Files affected:** `merge_utils.py`

`merge_utils.py::merge_dual_agents()` reads the structural agent output using the legacy key:
```python
fm_a.get("shape_in_possession")
```

Current structural agent prompts (as evidenced by `accumulator.py` which reads `w.get("formation", {}).get("home")` and `w.get("formation", {}).get("away")`) write formation data under the nested `formation` key. The legacy `shape_in_possession` key is no longer written, so `merge_dual_agents` silently drops formation data on every dual-agent merge pass.

**Suggested fix:** Update `merge_utils.py::merge_dual_agents()` to read `fm_a.get("formation", {})` (and separately `.get("home")` / `.get("away")`). Add a fallback to `shape_in_possession` for any historical files.

---

### F6 — Hardcoded absolute paths in generator scripts

**Severity:** MEDIUM — scripts cannot run on any machine except the original developer's  
**Files affected:** `generate_flagged_moments.py`, `generate_pass_network.py`

Both scripts contain hardcoded absolute paths:

`generate_flagged_moments.py`:
```python
SCRIPTS_DIR = r"C:\Users\dbmux\.claude\skills\match-analysis\scripts"
EXEMPLAR    = r"C:\Users\dbmux\Desktop\Match Lens Jobs\2026-04-11_gorleston_vs_tilbury\flagged_moments.md"
```

`generate_pass_network.py` contains the same pattern with a different exemplar path.

These paths are baked into the top-level module scope, not inside functions, so they cannot be overridden at call time. Any deployment to a different machine, CI system, or worktree will fail at import.

**Suggested fix:** Replace with `os.path.dirname(os.path.abspath(__file__))` for `SCRIPTS_DIR`. Replace the hardcoded exemplar path with a config-driven or match-dir-relative path, or make it optional with a `None` fallback.

---

### F7 — `match_id` fallback to `os.path.basename(match_dir)`

**Status:** RESOLVED — all 5 fallback sites routed through get_match_id() accessor. WARNING logged at the 4 standard sites where match_config.json is present but missing the 'match' key; INFO logged at window_plan.py:225 where match_config.json doesn't exist yet (expected on cold runs).

**Severity:** LOW-MEDIUM — produces valid but unexpected values when `match_config.json` is absent or the `match` key is missing  
**Files affected:** `accumulator.py`, `shots_log_addition.py`, `window_plan.py` (TWO fallback sites with different semantics: line ~224 where match_config.json is present but missing 'match' key → WARNING; line ~228 where match_config.json doesn't exist yet → INFO), `cost_estimator.py`

Multiple scripts contain:
```python
match_id = mc.get("match", os.path.basename(match_dir))
```

This silently embeds the directory name (e.g. `2026-04-11_gorleston_vs_tilbury`) as `match_id` in `running_summary.json`, `shots_log.json`, and `window_plan.json`. If `match_config.json` is present but its `match` key is blank or differently named, the fallback activates without any warning.

`window_plan.py` also falls back to `os.path.basename(match_dir)` when resolving the `match` key from `match_boundaries.json`, compounding the issue across two files.

**Suggested fix:** Add an explicit warning log line when the fallback activates. Consider a separate `match_id` key in `match_config.json` distinct from the display-name `match` field, to prevent the basename fallback from appearing in analytical output.

Note: `pipeline_runner.py` (v1) also contained this pattern but was removed from the codebase. F7 scope is now three scripts: `accumulator.py`, `shots_log_addition.py`, `window_plan.py`.

---

### F8 — `confirmation_queue` exists in two places with divergent schemas

**Status:** RESOLVED-AS-DOCUMENTED — the divergence is real in the code
but does not fire under the project's one-run-per-match workflow. No
consumer reads embedded confirmation queues after `setpiece_writeback.py`
runs. The standalone `confirmation_queue.json` is canonical; embedded
queues are write-once consolidation input. Architectural invariant
recorded in `SKILL.md` § Pipeline Invariants and inline comment in
`escalation_router.py` at the consolidation read site. Becomes a real
bug if any re-run path is added — see SKILL.md for the conditions.

**Severity:** MEDIUM — downstream consumers may read stale or incomplete data  
**Files affected:** `accumulator.py`, `escalation_router.py`, `setpiece_writeback.py`, `deep_skill_metrics.py`, `build_readiness_check.py`

`accumulator.py` appends `confirmation_queue` items to each individual `agent_{id}_merged.json` file (embedded queue, per-window).

`escalation_router.py` collects all embedded per-window queues, applies routing logic, and writes a single `confirmation_queue.json` at the match directory root (standalone queue).

`setpiece_writeback.py` marks entries as resolved in the standalone `confirmation_queue.json` but does NOT update the embedded per-window confirmation_queue in the merged files.

`deep_skill_metrics.py` and `build_readiness_check.py` both read the standalone `confirmation_queue.json` — never the embedded per-window queues.

Result: after `setpiece_writeback.py` runs, the standalone queue has `resolved: true` entries, but the embedded per-window queues still show those same items as unresolved. Any tool that re-reads from merged files (e.g. a future re-run of `escalation_router.py`) will re-queue already-resolved set piece bursts.

**Suggested fix:** Document which queue is canonical. Either (a) remove the embedded queue from merged files and treat the standalone `confirmation_queue.json` as the only source, or (b) have `setpiece_writeback.py` also mark the embedded merged-file entries as resolved.

---

### F9 — `ground_truth.py` direct key access without fallback for `start_s`/`end_s`

**Severity:** MEDIUM — raises `KeyError` on windows written with old schema  
**Files affected:** `ground_truth.py`

`ground_truth.py` accesses window fields with direct dict access:
```python
w["start_s"]
w["end_s"]
```

`window_plan.py` currently writes `start_s` and `end_s`. However, `cost_estimator.py` and `shots_log_addition.py` both contain fallback code:
```python
w.get("start_s", w.get("start_seconds", 0))
w.get("end_s",   w.get("end_seconds",   0))
```

This implies there are (or were) window_plan files using the keys `start_seconds`/`end_seconds`. `ground_truth.py` will raise `KeyError` on such files because it does not use `.get()` with a fallback.

**Suggested fix:** Change `ground_truth.py` to use `w.get("start_s", w.get("start_seconds", 0))` and `w.get("end_s", w.get("end_seconds", 0))`, consistent with the other scripts.

---

### F10 — `deep_skill_metrics.py` metric key mismatch (`avg_backward_shifts_per_phase` vs `avg_backward_shifts_per_window`)

**Severity:** LOW — metric appears in output under two different keys depending on path  
**Files affected:** `deep_skill_metrics.py`

In the `rest_defence_security_score` metric calculation, the `make_metric()` call uses:
```python
"avg_backward_shifts_per_phase": ...
```
but the value dict being passed in contains:
```python
"avg_backward_shifts_per_window": ...
```

The key name in the metric definition does not match the key name in the computed value. Consumers reading `avg_backward_shifts_per_phase` from `deep_skill_metrics.json` will get `None`/missing while `avg_backward_shifts_per_window` exists alongside it.

**Suggested fix:** Standardise to one key name throughout — `avg_backward_shifts_per_window` is the more accurate label since windows (not phases) are the unit of analysis.

---

### F11 — `accumulator.py` reset block omits `set_pieces_rejected` key

**Status:** RESOLVED — already fixed in the codebase prior to this audit work; verified during F10 fix. set_pieces_rejected is present in accumulator.py's reset block at the canonical location.

**Severity:** LOW — `setdefault` compensates but creates an asymmetric schema on first vs. subsequent runs  
**Files affected:** `accumulator.py`

`accumulate_all_windows()` resets `running_summary.json` with an explicit structure that includes `set_pieces: []`, `transitions: []`, etc., but does NOT include `set_pieces_rejected: []`. The key is added later via:
```python
summary.setdefault("set_pieces_rejected", [])
```

On the first run this produces a `running_summary.json` where `set_pieces_rejected` appears at the end of the object rather than in the initialisation block. This inconsistency is harmless in Python JSON (dicts are unordered for logic purposes) but makes schema inspection inconsistent and could confuse schema validation tools.

**Suggested fix:** Add `"set_pieces_rejected": []` to the reset block in `accumulate_all_windows()`.

---

### F12 — No `schema_version` field in any pipeline-written JSON file

**Status:** RESOLVED — schema_version stamped on every live pipeline
JSON output in commit [SHA-pending]. 32 write sites across 19 scripts
route through pipeline_schemas.stamp_schema_version(), which reads from
a per-file-type version registry (27 entries, all at "1.0"). Two writers
are explicitly excluded with documented rationale: the JSONL writer in
setpiece_writeback.py (no top-level object to stamp) and
accumulator.aggregate_shots (writes a top-level JSON list, not a dict).
Future schema migrations bump the per-file version in the registry;
readers gain branch logic on schema_version at that time, not
speculatively now.

**Severity:** LOW (structural) — absence makes future schema migration undetectable  
**Files affected:** All pipeline output JSON files

None of the 20+ distinct JSON output files written by the pipeline include a `schema_version` field. The only version-adjacent field is `generated_at` (present in most files) and `config_version` (present only in `source_profile.json`, sourced from `source_profiles_config.json`).

As schemas evolve (as evidenced by findings F4, F5, F9 above, which are all the result of past schema changes), there is no programmatic way to detect that a file was written by an older pipeline version and may use deprecated field names.

**Suggested fix:** Add `"schema_version": "1.0"` (or a meaningful semver) to the top-level of each file written by the pipeline. Increment on any breaking field rename or structural change. Readers can then gate on the version field before deciding which key aliases to apply.

---

### F13 — Dead `"queue"` key fallback in `step_3i_escalation` (v1)

**Status:** RESOLVED — v1 removed. See `git show de93081:pipeline_runner.py` for historical reference.

**Severity:** LOW (dead code, no runtime effect)  
**Files affected:** `pipeline_runner.py`

`pipeline_runner.py::step_3i_escalation` reads the confirmation queue with:

```python
items = [i for i in cq_doc.get("items", cq_doc.get("queue", []))
         if i.get("status") == "queued"]
```

`escalation_router.py` writes the standalone `confirmation_queue.json` with the top-level key `"items"`. No script writes `"queue"` as a top-level array key in this file. The `cq_doc.get("queue", [])` fallback is dead code. Its presence documents that `confirmation_queue.json` previously used `"queue"` as the array key before being renamed to `"items"` — an unresolved schema migration artifact.

**Suggested fix:** Remove the dead `cq_doc.get("queue", [])` fallback. If historical files with the old key must be supported, add an explicit migration comment.

---

### F14 — `apply_confirmation_to_summary()` is a second write path to `running_summary.json` in v1

**Status:** RESOLVED — v1 removed. See `git show de93081:pipeline_runner.py` for historical reference.

**Severity:** MEDIUM — v1 and v2 write overlapping fields to the same file via different mechanisms  
**Files affected:** `pipeline_runner.py`, `accumulator.py`, `setpiece_writeback.py`

`pipeline_runner.py::step_3i_escalation` processes confirmation segments inline and calls:

```python
apply_confirmation_to_summary(result, summary_path)
```

This is a direct patch write to `running_summary.json` from confirmation results, bypassing `accumulator.py`. In v2, set piece confirmation data instead flows through `setpiece_writeback.py`, which patches the merged window files and relies on a subsequent re-run of `accumulate_all_windows` to rebuild `running_summary.json`.

These two mechanisms produce overlapping edits to the same canonical file. If both pipelines have ever touched the same match directory (e.g. a v1 initial run followed by a v2 continuation), the `apply_confirmation_to_summary` patches may be partially overwritten by the v2 `accumulate_all_windows` pass, or v2 writeback data may be missing from the `running_summary.json` that `apply_confirmation_to_summary` last wrote to.

**Suggested fix:** Consolidate all writes to `running_summary.json` through a single path (`accumulator.py::accumulate_all_windows`). Deprecate `apply_confirmation_to_summary` or convert it to a patch-only helper that calls into `accumulate_all_windows`.

---

### F15 — v1 fallback merge produces Pattern C filenames (source suffix embedded in merged name)

**Status:** RESOLVED — v1 removed. See `git show de93081:pipeline_runner.py` for historical reference.

**Severity:** LOW (only triggers when `window_plan.json` is absent)  
**Files affected:** `pipeline_runner.py`

When `window_plan.json` is absent, `pipeline_runner.py::step_3e_merge` falls back to filename pattern detection. It constructs the merged output path as:

```python
base = fname.replace(".json", "")                          # no suffix stripping
out_path = os.path.join(self.logs_dir, f"{base}_merged.json")
```

For a v2-written structural file `agent_07_20m03s-25m03s_structural.json`, this produces:

```
agent_07_20m03s-25m03s_structural_merged.json  ← Pattern C
```

Pattern A (written by `merge_utils.py`) would be:

```
agent_07_20m03s-25m03s_merged.json
```

Pattern C names are matched by `escalation_router.py`'s broad `*_merged.json` glob, so the queue will be built correctly. However `setpiece_writeback.py`'s direct-path lookup and `pipeline_runner_v2.py`'s Phase 2 structural file lookup both construct Pattern B names (`agent_07_merged.json`) before falling back to glob — so they will degrade to the glob path and may match the wrong file if multiple windows share a window ID prefix.

Additionally, the same fallback in v1 pattern-detects agent suffixes using `_agentA`/`_a` (agent A) and `_agentB`/`_b` (agent B). v2 structural agents write `_structural` not `_agentA`, so the fallback would treat every v2 structural file as a "single" agent rather than pairing it with its corresponding `_agentB` file. Dual-agent merges would never fire on v2 output via this fallback.

**Suggested fix:** If `window_plan.json` is absent and the fallback must run, strip the known v2 suffixes (`_structural`, `_agentB`) from `base` before constructing the merged output path. Alternatively, require `window_plan.json` to be present and fail fast if it is missing.

---

## File I/O Table

| Path Pattern | Dir | Producer | Consumers | Top-level Keys (selected) |
|---|---|---|---|---|
| `match_config.json` | match_dir | `extract_match_details.py` (draft→final: human) | All scripts | `match`, `home_team`, `away_team`, `goals`, `lineups`, `substitutions`, `cards`, `home_kit`, `away_kit`, `attack_direction_1h`, `attack_direction_2h`, `report_level` |
| `match_config_draft.json` | match_dir | `extract_match_details.py` | Human reviewer | Same as match_config.json + `verified: false` |
| `teamsheet_image_raw.json` | match_dir | `extract_match_details.py` | None (audit record) | `screenshot`, `raw_text`, `usage`, `model` |
| `container_profile.json` | match_dir | `container_analyser.py` | `window_plan.py` | `seek_reliable`, `boundary_timestamps_s`, `discontinuity_count`, `source_pattern`, `remux_recommended` |
| `match_boundaries.json` | match_dir | `detect_boundaries.py` | `window_plan.py`, `ground_truth.py`, `shots_log_addition.py` | `boundaries.ko_1h.seconds`, `boundaries.ht_whistle.seconds`, `boundaries.ko_2h.seconds`, `boundaries.ft_whistle.seconds`, `video_duration_seconds` |
| `source_profile.json` | match_dir | `source_profiler.py` | `pipeline_runner.py`, `pipeline_runner_v2.py`, `merge_utils.py`, `build_readiness_check.py`, `jersey_ocr.py`, `generate_flagged_moments.py`, `generate_pass_network.py` | `source_type`, `classification_confidence`, `visibility_scores`, `source_limitations_note`, `split_aware` |
| `result_family_gates.json` | match_dir | `source_profiler.py` | `merge_utils.py`, `build_readiness_check.py`, `pipeline_runner.py`, `deep_skill_metrics.py` | `source_type`, `gates` (per-family: allowed/downgraded/suppressed) |
| `window_plan.json` | match_dir | `window_plan.py` | `pipeline_runner.py`, `pipeline_runner_v2.py`, `ground_truth.py`, `build_readiness_check.py`, `cost_estimator.py`, `shots_log_addition.py`, `synthesis_agent.py` | `windows[].agent_id`, `windows[].start_s`, `windows[].end_s`, `windows[].start_frame`, `windows[].end_frame`, `windows[].half`, `windows[].event_window` |
| `pipeline_state.json` | match_dir | `pipeline_state.py` | `pipeline_runner.py`, `pipeline_runner_v2.py` | `steps` (per-step status/timestamp) |
| `jersey_number_map.json` | match_dir | `jersey_ocr.py` | `pipeline_runner.py`, `pipeline_runner_v2.py` | `source_type`, `player_summary`, `frame_detail` |
| `cost_estimate.json` | match_dir | `cost_estimator.py` | Human (reference only) | `match`, `estimates` |
| `job_log.json` | match_dir | `job_logger.py` (via `JobLogger`) | Human (reference only) | `steps`, `counts`, `source_type`, `reports_generated` |
| `frame_metadata/{window_id}_metadata.json` | match_dir | `frame_preprocessor.py` | `pipeline_runner.py` | `window_id`, `stats`, `flagged_events`, `metadata` |
| `agent_logs/agent_{agent_id}_{safe_label}_structural.json` | match_dir | `batch_runner.py` (step 3a) | `merge_utils.py`, `accumulator.py`, `pipeline_runner.py` | `agent_id`, `formation`, `findings`, `confirmation_queue` |
| `agent_logs/agent_{agent_id}_{safe_label}_player.json` | match_dir | `batch_runner.py` (step 3b) | `merge_utils.py`, `accumulator.py` | `agent_id`, `players`, `findings` |
| `agent_logs/agent_{agent_id}_{safe_label}_event.json` | match_dir | `batch_runner.py` (step 3d_event) | `merge_utils.py`, `shots_log_addition.py` | `agent_id`, `events`, `confirmation_queue` |
| `agent_logs/agent_{agent_id}_{safe_label}_setpiece.json` | match_dir | `batch_runner.py` (step 3d_setpiece) | `setpiece_writeback.py`, `shots_log_addition.py` | `anchor_timestamp`, `team`, `window`, `confirmed_fields`, `burst_fields` |
| `agent_logs/agent_{agent_id}_{safe_label}_recovery.json` | match_dir | `batch_runner.py` (step 3d_recovery) | `merge_utils.py` | `agent_id`, `findings` |
| `agent_logs/agent_{agent_id}_{safe_label}_merged.json` | match_dir | `merge_utils.py` | `accumulator.py`, `escalation_router.py`, `setpiece_writeback.py` | `window`, `agent_id`, `findings`, `set_pieces`, `confirmation_queue`, `formation` |
| `agent_logs/{merged_base}_orphan_setpieces.json` | match_dir | `setpiece_writeback.py` (orphan path) | None (audit record) | newline-delimited JSON objects: `anchor_timestamp`, `team`, `burst_path` |
| `running_summary.json` | match_dir | `accumulator.py`; `apply_confirmation_to_summary()` (v1 inline path — F14) | `ground_truth.py`, `build_readiness_check.py`, `deep_skill_metrics.py`, `synthesis_agent.py`, `setpiece_writeback.py` | `match_id`, `set_pieces`, `transitions`, `pass_sequences_raw`, `findings`, `set_pieces_rejected` |
| `pass_sequences.json` | match_dir | `accumulator.py` | `deep_skill_metrics.py`, `synthesis_agent.py`, `generate_pass_network.py` | `match_id`, `sequences` |
| `shots_log.json` | match_dir | `shots_log_addition.py` | `synthesis_agent.py`, `build_readiness_check.py` | `match_id`, `goals[]` (with `origin_zone`, `shot_foot`, `evidence_grade`) |
| `confirmation_queue.json` | match_dir | `escalation_router.py` | `setpiece_writeback.py`, `deep_skill_metrics.py`, `build_readiness_check.py` | `total`, `goals`, `items[]` (with `status`, `escalation_target_fps`, `rerun_window_start/end`) |
| `ground_truth_check.json` | match_dir | `ground_truth.py` | `build_readiness_check.py`, `synthesis_agent.py` | `goals_checked`, `mismatches`, `windows_covered` |
| `rerun_queue.json` | match_dir | `pipeline_runner.py`/`pipeline_runner_v2.py` | `merge_utils.py`, `build_readiness_check.py` | `items[]` (per-window patches) |
| `report_readiness.json` | match_dir | `build_readiness_check.py` | `synthesis_agent.py`, `pipeline_runner.py`/`pipeline_runner_v2.py` | `ready`, `blocking_issues`, `warnings` |
| `confidence_reliability_report.json` | match_dir | `build_readiness_check.py` | Human (reference) | `per_family_scores`, `overall_confidence` |
| `deep_skill_metrics.json` | match_dir | `deep_skill_metrics.py` | `synthesis_agent.py`, `build_readiness_check.py`, `pipeline_runner.py`/`pipeline_runner_v2.py` | `metrics[]` (with `metric_name`, `value`, `sample_status`, `confidence`) |
| `tactical_report.md` | match_dir | `synthesis_agent.py` | `md_to_docx.py` | Markdown text |
| `opposition_report_{slug}.md` | match_dir | `synthesis_agent.py` | `md_to_docx.py` | Markdown text |
| `advanced_tactical_report.md` | match_dir | `synthesis_agent.py` | `md_to_docx.py` | Markdown text |
| `advanced_opposition_report_{slug}.md` | match_dir | `synthesis_agent.py` | `md_to_docx.py` | Markdown text |
| `flagged_moments.md` | match_dir | `generate_flagged_moments.py` | `synthesis_agent.py` | Markdown text |
| `pass_network.md` | match_dir | `generate_pass_network.py` | `md_to_docx.py` | Markdown text |
| `tactical_report.docx` | match_dir | `md_to_docx.py` | End user | Word document |
| `opposition_report_{slug}.docx` | match_dir | `md_to_docx.py` | End user | Word document |
| `frames/source_samples/source_sample_{MM}m{SS}s.jpg` | match_dir | `source_profiler.py` | LLM vision call | JPEG image |
| `frames/frame_{MM}m{SS}s.jpg` | match_dir | Step 1 (ffmpeg/cv2, external) | All frame-reading steps | JPEG image |
| `frames_burst/{window_id}_{anchor_ts}/frame_{MM}m{SS}s_{ms}ms.jpg` | match_dir | `frame_extraction.py` | `batch_runner.py`, `pipeline_runner.py` | JPEG image |
| `frame_metadata/{window_id}_metadata.json` | match_dir | `frame_preprocessor.py` | `pipeline_runner.py` | `stats`, `flagged_events`, `metadata` |

---

## Schema Catalogue

### `match_config.json`

| Field | Type | Notes |
|---|---|---|
| `match` | string | Display name, doubles as match_id fallback (F7) |
| `home_team` | string | Full club name |
| `away_team` | string | Full club name |
| `home_kit` | string | Colour label |
| `away_kit` | string | Colour label |
| `home_gk_kit` | string | Colour label |
| `away_gk_kit` | string | Colour label |
| `attack_direction_1h` | string | "left" or "right" |
| `attack_direction_2h` | string | "left" or "right" |
| `goals[]` | array | `{time.elapsed, team.name, player.name, type}` |
| `substitutions[]` | array | `{time.elapsed, team.name, player.name, assist.name}` |
| `cards[]` | array | `{time.elapsed, team.name, player.name, detail}` |
| `lineups[]` | array | `{team.name, startXI[], substitutes[]}` each player: `{player.name, player.number}` |
| `report_level` | string | `brief`, `standard`, or `technical`; optional |
| `verified` | bool | Set true after human review |
| `boundaries_override` | object | Optional manual boundary values |
| **`schema_version`** | — | **ABSENT — F12** |

### `window_plan.json`

| Field | Type | Notes |
|---|---|---|
| `match` | string | |
| `total_windows` | int | |
| `windows[]` | array | |
| `windows[].agent_id` | string | Two-digit zero-padded sequence number (canonical window ID) |
| `windows[].start_s` | float | Video seconds (cf. `start_seconds` alias — F9) |
| `windows[].end_s` | float | Video seconds (cf. `end_seconds` alias — F9) |
| `windows[].start_frame` | string | `frame_MMmSSs.jpg` |
| `windows[].end_frame` | string | `frame_MMmSSs.jpg` |
| `windows[].half` | string | `"1H"` or `"2H"` |
| `windows[].event_window` | bool | |
| `windows[].match_state` | object | `{score_home, score_away, match_state}` |
| `windows[].boundary_nearby` | bool | True if container segment boundary within 15s |
| **`windows[].window_id`** | — | **ABSENT — never written, but some scripts read it (F1)** |
| **`schema_version`** | — | **ABSENT — F12** |

### `source_profile.json`

| Field | Type | Notes |
|---|---|---|
| `source_type` | string | One of the VALID_SOURCE_TYPES |
| `classification_confidence` | float | 0.0–1.0 |
| `split_aware` | bool | True for dual_panoramic |
| `visibility_scores` | object | 9 numeric scores |
| `source_limitations_note` | string | **Canonical key** (F3) |
| `visibility_based_limitations` | array of strings | |
| `config_version` | string | From `source_profiles_config.json._version` |
| `generated_at` | ISO datetime string | |
| **`notes`** | — | **Not written here; only as fallback key in merge_utils (F3)** |
| **`source_limitations`** | — | **Not written; read by generate_flagged_moments (F3)** |
| **`limitations_note`** | — | **Not written; read by generate_flagged_moments (F3)** |
| **`schema_version`** | — | **ABSENT — F12** |

### `agent_logs/agent_{id}_{label}_merged.json`

| Field | Type | Notes |
|---|---|---|
| `window` | string | Window ID (set by dual-merge path) |
| `agent_id` | string | Window ID (set by single-agent merge path) |
| `findings[]` | array | Merged tactical findings |
| `set_pieces[]` | array | Each: `{timestamp, team, type, burst_resolved, marking_system, delivery_zone, outcome, ...}` |
| `confirmation_queue[]` | array | Per-window escalation items (embedded) |
| `formation` | object | `{home, away}` — **new schema** |
| **`shape_in_possession`** | — | **Legacy key read by merge_utils (F5)** |
| **`schema_version`** | — | **ABSENT — F12** |

### `running_summary.json`

| Field | Type | Notes |
|---|---|---|
| `match_id` | string | From `match_config.match` with dir-basename fallback (F7) |
| `set_pieces[]` | array | `{timestamp, team, type, marking_system, ...burst_resolved fields}` |
| `set_pieces_rejected[]` | array | Added via `setdefault`, not in reset block (F11) |
| `transitions[]` | array | |
| `pass_sequences_raw[]` | array | |
| `findings[]` | array | |
| **`schema_version`** | — | **ABSENT — F12** |

### `confirmation_queue.json` (standalone)

| Field | Type | Notes |
|---|---|---|
| `total` | int | |
| `goals` | int | |
| `high_priority_other` | int | |
| `medium_priority` | int | |
| `skipped_ineligible` | int | |
| `skipped_by_cap` | int | |
| `cap_note` | string | |
| `match_goal_count` | int | |
| `goals_uncapped` | bool | |
| `items[]` | array | Each: `{event_type, timestamp, priority, escalation_target_fps, rerun_window_start, rerun_window_end, status, window, source}` |
| `generated_at` | ISO datetime string | |
| **`schema_version`** | — | **ABSENT — F12** |

Items gain `resolved: true`, `resolved_at`, `resolved_by` after `setpiece_writeback.py` runs. These fields are absent from items not yet resolved, creating a partially-updated schema within the same array.

### `deep_skill_metrics.json`

| Field | Type | Notes |
|---|---|---|
| `metrics[]` | array | Each: `{metric_name, value, sample_status, confidence, analysis_scope, context_only}` |
| `generated_at` | ISO datetime string | |
| **`avg_backward_shifts_per_phase`** vs **`avg_backward_shifts_per_window`** | — | Key mismatch in F10 |
| **`schema_version`** | — | **ABSENT — F12** |

---

## Filename Construction Sites

| Location | Pattern Produced | Variables Used | Risk |
|---|---|---|---|
| `window_plan.py:138` | `frame_{m:02d}m{s:02d}s.jpg` | `m, s` from `divmod(int(s), 60)` | Authoritative 1fps frame filename format |
| `window_plan.py:263` | `windows[].agent_id = f"{i+1:02d}"` | Sequential integer | **Canonical window ID** — all lookups must use this format |
| `batch_runner.py` | `agent_{window_id}_{suffix}.json` | `window_id` (from `custom_id`), `suffix` from suffix_map | **NAMING DRIFT**: `window_id` extracted by `custom_id.replace(f"_{step}", "")` — fragile if step string appears in window_id |
| `merge_utils.py` | `agent_{agent_id}_{safe_label}_merged.json` | `agent_id` = `"01"`–`"18"` format; `safe_label` = label with unsafe chars replaced | **NAMING DRIFT**: Pattern A. `safe_label` includes the time range string. Exact name is unpredictable without knowing the label |
| `pipeline_runner.py` (fallback) | `{base}_merged.json` where `base = fname.replace(".json","")` | Source filename without `.json`; no suffix stripping | **NAMING DRIFT**: Pattern C. Embeds `_structural` in the merged name (`agent_07_20m03s-25m03s_structural_merged.json`). Only triggers when window_plan.json is absent (F15) |
| `setpiece_writeback.py:237` | `agent_{window_id}_merged.json` (lookup) | `window_id` from burst JSON | **NAMING DRIFT**: Pattern B. This exact path NEVER exists (see F2). Always falls back to glob |
| `pipeline_runner_v2.py` (Phase 2) | `agent_{wid}_structural.json` (lookup) | `wid` | **NAMING DRIFT**: Pattern B for structural. Same miss-then-glob problem as setpiece_writeback |
| `escalation_router.py:168-170` | Three overlapping globs (see F2) | `logs_dir`, `match_dir` | Redundant patterns with dedup; patterns 1 and 3 are subsets of each other |
| `setpiece_writeback.py:129` | `{merged_base}_orphan_setpieces.json` | `merged_window_path.replace(".json", ...)` | Orphan file is appended (not overwritten); grows unboundedly across re-runs |
| `frame_extraction.py:116` | `frame_{m:02d}m{s:02d}s_{ms:03d}ms.jpg` | Millisecond-precision timestamp | Burst frame format — distinct from 1fps format (has `_ms` suffix) |
| `source_profiler.py:73` | `source_sample_{m:02d}m{s:02d}s.jpg` | `m, s` from video timestamp | Source profiler frames — different directory (`frames/source_samples/`) |
| `frame_preprocessor.py:396` | `frame_metadata/{window_id}_metadata.json` | `window_id` as passed by caller | Written to `frame_metadata/` subdirectory |
| `synthesis_agent.py` | `opposition_report_{slug}.md` | `slug = re.sub(r"[^\w\-]", "_", team_name.lower())` | Slug depends on team name string; special characters silently replaced |
| `generate_flagged_moments.py` | `flagged_moments.md` | Hardcoded filename | Single file per match dir; no timestamp or version suffix |

---

## Canonical Directory Layout

The following shows all files a complete pipeline run produces, in step order.

```
{match_dir}/
│
│  ── Step 1a ─────────────────────────────────────────────
│  container_profile.json
│
│  ── Step 1b ─────────────────────────────────────────────
│  match_boundaries.json
│
│  ── Step 1c ─────────────────────────────────────────────
│  window_plan.json
│
│  ── Step 1d ─────────────────────────────────────────────
│  match_config_draft.json
│  teamsheet_image_raw.json
│  match_config.json                    ← human-verified copy
│
│  ── Step 1f ─────────────────────────────────────────────
│  source_profile.json
│  result_family_gates.json
│  frames/
│  └── source_samples/
│      └── source_sample_{MM}m{SS}s.jpg
│
│  ── Step 1 (frame extraction) ──────────────────────────
│  frames/
│  └── frame_{MM}m{SS}s.jpg             ← one per second of video
│
│  ── Step 1 (jersey OCR, optional) ─────────────────────
│  jersey_number_map.json
│
│  ── Step 1 (frame preprocessor, optional) ─────────────
│  frame_metadata/
│  └── {agent_id}_metadata.json
│
│  ── Step 1 (cost estimator, optional) ─────────────────
│  cost_estimate.json
│
│  ── Pipeline state ─────────────────────────────────────
│  pipeline_state.json
│  job_log.json
│
│  ── Steps 3a, 3b, 3d (batch agent outputs) ─────────────
│  agent_logs/
│  ├── agent_{id}_{label}_structural.json
│  ├── agent_{id}_{label}_player.json
│  ├── agent_{id}_{label}_event.json     ← event windows only
│  ├── agent_{id}_{label}_setpiece.json  ← set piece burst outputs
│  └── agent_{id}_{label}_recovery.json  ← boundary-adjacent windows
│
│  ── Step 3e (merge) ─────────────────────────────────────
│  agent_logs/
│  └── agent_{id}_{label}_merged.json    ← one per window
│
│  ── Step 3d-SP-WB (set piece writeback) ────────────────
│  agent_logs/
│  └── agent_{id}_{label}_orphan_setpieces.json  ← only if orphan detected
│
│  ── Step 3f/3g (accumulator) ───────────────────────────
│  running_summary.json
│  pass_sequences.json
│  shots_log.json
│
│  ── Step 3h (ground truth) ─────────────────────────────
│  ground_truth_check.json
│
│  ── Step 3i (escalation router) ────────────────────────
│  confirmation_queue.json
│
│  ── Step 3j (readiness check) ──────────────────────────
│  report_readiness.json
│  confidence_reliability_report.json
│
│  ── Step 3k (deep skill metrics) ───────────────────────
│  deep_skill_metrics.json
│
│  ── Step 3l (rerun queue, produced by pipeline runner) ─
│  rerun_queue.json
│
│  ── Step 4a/4b (synthesis agent) ───────────────────────
│  tactical_report.md
│  opposition_report_{slug}.md
│  advanced_tactical_report.md           ← optional
│  advanced_opposition_report_{slug}.md  ← optional
│
│  ── Step 4c (flagged moments) ──────────────────────────
│  flagged_moments.md
│
│  ── Step 4d (pass network) ─────────────────────────────
│  pass_network.md
│
│  ── Step 5 (Word conversion) ───────────────────────────
│  tactical_report.docx
│  opposition_report_{slug}.docx
│  flagged_moments.docx
│  pass_network.docx
│
│  ── Burst frame extraction (Step 3d-SP) ────────────────
│  frames_burst/
│  └── {window_id}_{anchor_ts}/
│      └── frame_{MM}m{SS}s_{ms}ms.jpg   ← 5fps burst frames
```

### Key variables

- `{id}` — two-digit zero-padded agent/window sequence number, e.g. `07`
- `{label}` — window time-range label with unsafe chars replaced, e.g. `20m03s-25m03s`
- `{slug}` — team name lowercased with non-word characters replaced by `_`
- `{MM}m{SS}s` — match timestamp in `MMmSSs` format
- `{anchor_ts}` — timestamp string from the escalation item, e.g. `20m03s`

---

*End of audit.*
