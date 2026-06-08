"""
merge_utils.py -- Match Lens Step 3e

Merges agent window JSON outputs into a single merged file per window.
Event windows (dual agents): merge_dual_agents
Routine windows (single agent): merge_single_agent

Both functions apply re-run patches from rerun_queue.json before merging,
so the merged file always contains the best available frame data.

Usage:
    from merge_utils import merge_dual_agents, merge_single_agent, merge_all_windows

    # Merge a single dual-agent window:
    merge_dual_agents(a_path, b_path, out_path, agent_id, match_dir)

    # Merge a single routine window:
    merge_single_agent(a_path, out_path, agent_id, match_dir)

    # Merge all windows in the plan:
    merge_all_windows(match_dir)
"""

import json
import os
import re as _re
import sys
import glob
from pipeline_paths import (
    find_agent_output as _find_agent_file,
    derive_window_label as _derive_window_label,
)
from pipeline_accessors import (
    get_source_limitations_note,
    get_formation_home,
    get_formation_away,
)
from pipeline_schemas import stamp_schema_version
from datetime import datetime


# v3.0.1 polish Task 144: fields the player agent canonically owns.
# These are loaded from agent_NN_player.json and folded into the merged
# window output ALONGSIDE the structural agent's fields. Pre-fix the
# player file was treated as a fallback for the structural input rather
# than as an additive third input, so these fields were silently
# absent from every merged window on modern v3 matches — which made
# running_summary.individual_observations stay [] and player_summary_cards
# render with empty content despite Step 17 file-existence checks passing.
PLAYER_AGENT_FIELDS = (
    "individual_observations",
    "duels",
    "player_escalation_queue",
    "watch_list_confirmations",
    "top_observations_for_next_window",
)


def _load_player_file(logs_dir: str, wid) -> dict:
    """v3.0.1 polish Task 144: load the player-agent output for a window
    and return its parsed dict, or {} if missing/unreadable. Reads via
    find_agent_output's canonical multi-pattern lookup, so it tolerates
    every batch_runner / merge_utils filename variant the runner has used
    historically.

    Returns {} (not None) so call sites can safely .get(field, []) on the
    result without a guarding conditional.
    """
    path = _find_agent_file(logs_dir, wid, "player")
    if not path:
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f) or {}
    except (OSError, json.JSONDecodeError):
        return {}


def _fold_player_fields(merged: dict, player: dict) -> None:
    """v3.0.1 polish Task 144: fold the player-agent output's canonical
    fields into the merged-window dict in place.

    Semantics per field:
      - individual_observations: APPENDED to whatever the structural side
        emitted (structural rarely emits any). The accumulator's player-
        tendency loop reads from merged.individual_observations.
      - duels: APPENDED. Same rationale.
      - player_escalation_queue: OVERWRITTEN if the player file emits a
        non-empty list (player-agent canonical; structural doesn't emit it).
      - watch_list_confirmations: OVERWRITTEN (same).
      - top_observations_for_next_window: OVERWRITTEN (same).

    A player file with empty lists is preserved as empty (not skipped) so
    downstream consumers see a uniform shape.
    """
    if not player:
        return
    # Append-only fields: combine with whatever structural side already had.
    for fld in ("individual_observations", "duels"):
        src_list = player.get(fld) or []
        if not isinstance(src_list, list):
            continue
        existing = merged.get(fld) or []
        if not isinstance(existing, list):
            existing = []
        merged[fld] = existing + src_list
    # Overwrite fields: player agent is the canonical owner.
    for fld in ("player_escalation_queue",
                "watch_list_confirmations",
                "top_observations_for_next_window"):
        if fld in player:
            merged[fld] = player[fld]


# -- Numeric and categorical merge helpers -------------------------------------

def merge_numeric(a, b):
    """Average two numeric values. Returns whichever is non-None if only one."""
    if a is None: return b
    if b is None: return a
    return round((a + b) / 2, 1)


def merge_categorical(a, b):
    """
    Resolve two categorical values.
    If they agree: return value.
    If they differ: prefer longer (more specific) string; log disagreement.
    """
    if a == b: return a
    if a is None: return b
    if b is None: return a
    return a if len(str(a)) >= len(str(b)) else b


# -- Re-run patch loading ------------------------------------------------------

def load_rerun_patches(agent_id: str, match_dir: str) -> dict:
    """
    Load resolved re-run data for a given agent_id.
    Returns dict of {frame_filename -> resolved patch data}.
    """
    rerun_path = os.path.join(match_dir, "rerun_queue.json")
    if not os.path.exists(rerun_path):
        return {}

    with open(rerun_path, encoding="utf-8") as f:
        rq = json.load(f)

    logs_dir = os.path.join(match_dir, "agent_logs")
    patches  = {}

    for item in rq.get("rerun_queue", []):
        if item.get("agent_id") != agent_id:
            continue
        if item.get("status") != "resolved":
            continue

        rerun_file = os.path.join(
            logs_dir,
            item["source_file"].replace(".json", "_rerun.json")
        )
        if os.path.exists(rerun_file):
            with open(rerun_file, encoding="utf-8") as f:
                patches[item["frame"]] = json.load(f)

    return patches


def patch_frames(frames: list, patches: dict) -> list:
    """Replace low-confidence frame observations with re-run resolutions."""
    patched = []
    for frame in frames:
        name = frame.get("frame", "")
        if name in patches:
            rerun = patches[name]
            frame["confidence"]    = rerun.get("confidence", frame["confidence"])
            frame["observations"]  = rerun.get("resolved_observations",
                                                frame.get("observations", {}))
            frame["rerun_applied"] = True
        patched.append(frame)
    return patched


# -- findings gate stamping ----------------------------------------------------

def stamp_gates_on_findings(findings: list, gates: dict,
                             source_note: str) -> list:
    """
    Apply result_family gate states to findings.
    allowed    -> write normally
    downgraded -> add limitations_note from source profile
    """
    for finding in findings:
        family = finding.get("result_family", "")
        status = gates.get(family, "downgraded")
        finding["result_family_status"] = status
        if status == "downgraded" and not finding.get("limitations_note"):
            finding["limitations_note"] = source_note
    return findings


# -- Single-agent merge --------------------------------------------------------

def merge_single_agent(a_path: str, out_path: str,
                       agent_id: str, match_dir: str,
                       player_path: str = None) -> dict:
    """
    Routine window merge: apply re-run patches and write as merged.
    No consensus layer -- single agent output is the source of truth.

    v3.0.1 polish Task 144: optional player_path. When provided, the
    player file's canonical fields (individual_observations, duels,
    player_escalation_queue, watch_list_confirmations,
    top_observations_for_next_window) are folded into the merged output
    alongside the structural agent's fields. Pre-fix the merge step only
    ever consumed the structural file, so player-agent fields were
    silently absent from every merged window and downstream consumers
    (accumulator, player_aggregator) saw empty inputs.
    """
    with open(a_path, encoding="utf-8") as f:
        a = json.load(f)

    # v3.0.1 polish Task 144: load player file if available.
    player = {}
    if player_path and os.path.exists(player_path):
        try:
            with open(player_path, encoding="utf-8") as _f:
                player = json.load(_f) or {}
        except (OSError, json.JSONDecodeError):
            player = {}

    patches = load_rerun_patches(a.get("agent_id", agent_id), match_dir)
    if patches:
        a["frames"] = patch_frames(a.get("frames", []), patches)
        a["rerun_patches_applied"] = len(patches)

    # Apply gate states if available
    gates_path   = os.path.join(match_dir, "result_family_gates.json")
    profile_path = os.path.join(match_dir, "source_profile.json")
    if os.path.exists(gates_path) and os.path.exists(profile_path):
        with open(gates_path, encoding="utf-8") as f:
            gates_doc = json.load(f)
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        gates       = gates_doc.get("gates", {})
        # Fix 39 (same root cause as Fix 38): broadcast source_profile.json
        # stores the limitation under "notes". Either-key fallback so merged
        # findings carry the correct source-note stamp on broadcast runs.
        source_note = get_source_limitations_note(profile)
        a["findings"] = stamp_gates_on_findings(
            a.get("findings", []), gates, source_note
        )
        a["source_gates_applied"] = True


    a["agent_id"]     = agent_id          # ensure accumulator window_id resolves
    a["merge_type"]   = "single_agent"
    a["merged_at"]    = datetime.now().isoformat()
    # Bug A fix (defence-in-depth, single-agent variant): the structural
    # agent does not emit a window field, so without this assignment the
    # single-agent merged file inherits no `window` and any downstream
    # reader hits the same None-propagation pattern dual_agent had.
    a["window"]       = _derive_window_label(out_path)

    # v3.0.1 polish Task 144: fold the player agent's canonical fields
    # into the merged dict before writing.
    _fold_player_fields(a, player)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(a, "agent_merged"), f, indent=2)

    n_obs = len(a.get("individual_observations") or [])
    n_duels = len(a.get("duels") or [])
    print(f"  [{agent_id}] Single -> {os.path.basename(out_path)}"
          f" | patches: {len(patches)}"
          f" | player-fold: obs={n_obs} duels={n_duels}")
    return a


# -- Dual-agent merge ----------------------------------------------------------

def merge_dual_agents(a_path: str, b_path: str, out_path: str,
                      agent_id: str, match_dir: str,
                      player_path: str = None) -> dict:
    """
    Event window merge: two independent agents, consensus layer, disagreement log.
    The merged output reflects the best available reading from both agents.

    v3.0.1 polish Task 144: optional player_path. Same semantics as
    merge_single_agent — player file is an additive third input whose
    canonical fields (individual_observations, duels, etc.) get folded
    into the merged output. The explicit allowlist that builds the merged
    dict (around line 400) was missing gk_distribution + far_side_shape;
    those are now added so structural-only fields survive the event-window
    merge path. Pre-fix, event windows (~45% of matches) silently dropped
    every GK distribution event the structural agent recorded.
    """
    with open(a_path, encoding="utf-8") as f: a = json.load(f)
    with open(b_path, encoding="utf-8") as f: b = json.load(f)

    # v3.0.1 polish Task 144: load player file if available.
    player = {}
    if player_path and os.path.exists(player_path):
        try:
            with open(player_path, encoding="utf-8") as _f:
                player = json.load(_f) or {}
        except (OSError, json.JSONDecodeError):
            player = {}

    # Apply re-run patches to both
    patches_a = load_rerun_patches(a.get("agent_id", agent_id + "a"), match_dir)
    patches_b = load_rerun_patches(b.get("agent_id", agent_id + "b"), match_dir)
    a["frames"] = patch_frames(a.get("frames", []), patches_a)
    b["frames"] = patch_frames(b.get("frames", []), patches_b)

    review_required = []

    # -- Merge defensive line --------------------------------------------------
    dl_a = a.get("defensive_line", {})
    dl_b = b.get("defensive_line", {})
    line_avg = merge_numeric(dl_a.get("avg_pct"), dl_b.get("avg_pct"))
    line_status = "agreed" if dl_a.get("avg_pct") == dl_b.get("avg_pct") else "averaged"

    # -- Merge formation -------------------------------------------------------
    # Fix 32a schema: formation is a nested dict with `home` and `away`
    # strings. F5: merge each side separately so agent B's reading isn't
    # silently lost via a single-field merge on the legacy schema.
    fm_a = a.get("formation", {})
    fm_b = b.get("formation", {})

    fm_a_home = get_formation_home(a)
    fm_b_home = get_formation_home(b)
    fm_a_away = get_formation_away(a)
    fm_b_away = get_formation_away(b)

    home_merged = merge_categorical(fm_a_home, fm_b_home)
    away_merged = merge_categorical(fm_a_away, fm_b_away)

    if fm_a_home != fm_b_home:
        review_required.append(
            f"formation home: A={fm_a_home}, B={fm_b_home} -> used {home_merged}"
        )
    if fm_a_away != fm_b_away:
        review_required.append(
            f"formation away: A={fm_a_away}, B={fm_b_away} -> used {away_merged}"
        )

    home_status = "agreed" if fm_a_home == fm_b_home else "resolved"
    away_status = "agreed" if fm_a_away == fm_b_away else "resolved"
    form_status = "agreed" if (home_status == "agreed" and away_status == "agreed") else "resolved"

    shape_dispute = None
    disputes = []
    if fm_a_home != fm_b_home and fm_a_home and fm_b_home:
        disputes.append({
            "side": "home",
            "agent_a": fm_a_home,
            "agent_b": fm_b_home,
            "resolved_to": home_merged,
        })
    if fm_a_away != fm_b_away and fm_a_away and fm_b_away:
        disputes.append({
            "side": "away",
            "agent_a": fm_a_away,
            "agent_b": fm_b_away,
            "resolved_to": away_merged,
        })
    if disputes:
        shape_dispute = {
            "type": "formation_disagreement",
            "disputes": disputes,
            "note": "Disagreements on formation often indicate tactical transitions, "
                    "formation changes, or difficult-to-read framing. "
                    "Review the window frames manually if formation is key to the analysis.",
        }

    # -- Line height dispute check -----------------------------------------------
    lh_a = a.get("defensive_line", {}).get("avg_height_pct")
    lh_b = b.get("defensive_line", {}).get("avg_height_pct")
    line_height_dispute = None
    if lh_a is not None and lh_b is not None and abs(lh_a - lh_b) > 10:
        line_height_dispute = {
            "agent_a_pct": lh_a,
            "agent_b_pct": lh_b,
            "delta":       round(abs(lh_a - lh_b), 1),
            "note": f"Line height readings differ by {round(abs(lh_a-lh_b),1)}% (>{round(abs(lh_a-lh_b)*1.05,1)}m). "
                    "Consider whether a press activation or transition mid-window explains the split.",
        }
        review_required.append(
            f"line_height: A={lh_a}%, B={lh_b}% -- delta {round(abs(lh_a-lh_b),1)}%"
        )

    # -- Merge pressing --------------------------------------------------------
    pr_a = a.get("pressing", {})
    pr_b = b.get("pressing", {})
    press_avg = merge_numeric(pr_a.get("avg_score"), pr_b.get("avg_score"))

    # -- Merge key moments -- consensus check -----------------------------------
    # Accept both "timestamp" (legacy) and "minute" (Fix 32a schema) keys.
    # Skip entries with neither rather than crashing on bracket access.
    def _moment_key(m):
        return m.get("timestamp") or m.get("minute") or ""
    a_moments = {_moment_key(m): m for m in a.get("key_moments", []) if _moment_key(m)}
    b_moments = {_moment_key(m): m for m in b.get("key_moments", []) if _moment_key(m)}
    merged_moments = []

    for ts, moment in a_moments.items():
        if ts in b_moments:
            moment["consensus"] = "confirmed"
            moment["confidence_a"] = moment.get("confidence_before_rerun", 0)
            moment["confidence_b"] = b_moments[ts].get("confidence_before_rerun", 0)
        else:
            moment["consensus"] = "partial_a_only"
            review_required.append(
                f"key_moment at {ts} seen by Agent A only -- "
                f"type: {moment.get('type','?')}, conf: {moment.get('confidence_before_rerun','?')}"
            )
        merged_moments.append(moment)

    for ts, moment in b_moments.items():
        if ts not in a_moments:
            moment["consensus"] = "partial_b_only"
            review_required.append(
                f"key_moment at {ts} seen by Agent B only -- "
                f"type: {moment.get('type','?')}, conf: {moment.get('confidence_before_rerun','?')}"
            )
            merged_moments.append(moment)

    # -- Merge findings -- combine both, prefer higher evidence tier -------------
    def tier_weight(t):
        return {"escalated_confirmation": 4, "direct": 3,
                "repeated_pattern": 2, "suggestive": 1}.get(t, 0)

    combined_findings = {}
    for finding in a.get("findings", []) + b.get("findings", []):
        key = (finding.get("result_family"), finding.get("time_start"))
        if key not in combined_findings:
            combined_findings[key] = finding
        else:
            existing = combined_findings[key]
            if tier_weight(finding.get("evidence_tier")) > \
               tier_weight(existing.get("evidence_tier")):
                combined_findings[key] = finding

    merged_findings = list(combined_findings.values())

    # Apply gate states
    gates_path   = os.path.join(match_dir, "result_family_gates.json")
    profile_path = os.path.join(match_dir, "source_profile.json")
    gates        = {}
    source_note  = ""
    if os.path.exists(gates_path) and os.path.exists(profile_path):
        with open(gates_path, encoding="utf-8") as f:
            gates_doc = json.load(f)
        with open(profile_path, encoding="utf-8") as f:
            profile = json.load(f)
        gates       = gates_doc.get("gates", {})
        # Fix 39 (same root cause as Fix 38): broadcast source_profile.json
        # stores the limitation under "notes". Either-key fallback so merged
        # findings carry the correct source-note stamp on broadcast runs.
        source_note = get_source_limitations_note(profile)
        merged_findings = stamp_gates_on_findings(merged_findings, gates, source_note)

    # -- Merge confirmation queue items (union) --------------------------------
    seen_cq = set()
    merged_cq = []
    for item in a.get("confirmation_queue", []) + b.get("confirmation_queue", []):
        key = (item.get("timestamp", ""), item.get("result_family", ""), item.get("event_type", ""))
        if key not in seen_cq:
            seen_cq.add(key)
            merged_cq.append(item)

    # -- Merge shot attempts (deduplicate by timestamp) ------------------------
    seen_shots = set()
    merged_shots = []
    for shot in a.get("shot_attempts", []) + b.get("shot_attempts", []):
        key = shot.get("timestamp", "")
        if key not in seen_shots:
            seen_shots.add(key)
            merged_shots.append(shot)

    # -- Build merged output ---------------------------------------------------
    # Bug A fix (defence-in-depth): set the top-level `window` from the
    # output filename rather than `a.get("window")`. Structural agents
    # do not emit a window field, so the previous code always wrote None
    # and propagated that to every downstream reader. derive_window_label
    # parses the canonical label out of the basename and never returns
    # None.
    merged = {
        "agent_id":         agent_id,
        "window":           _derive_window_label(out_path),
        "merge_type":       "dual_agent",
        "frames_reviewed":  max(a.get("frames_reviewed", 0), b.get("frames_reviewed", 0)),
        "review_required":  review_required,
        "source_gates_applied": bool(gates),

        "defensive_line": {
            **dl_a,
            "avg_pct":    line_avg,
            "merge_note": line_status,
        },
        "formation": {
            **fm_a,
            "home":                home_merged,
            "away":                away_merged,
            # Legacy key preserved for back-compat with old readers; reflects
            # the home-side merge (closest semantic match to "in-possession").
            "shape_in_possession": home_merged,
            "merge_note":          form_status,
        },
        "pressing": {
            **pr_a,
            "avg_score":  press_avg,
            "merge_note": "averaged",
        },

        "pass_sequences":          a.get("pass_sequences", []) +
                                   b.get("pass_sequences", []),
        "set_pieces":              a.get("set_pieces", []),
        "shot_attempts":           merged_shots,
        "attacking":               a.get("attacking", {}),
        "defensive":               a.get("defensive", {}),
        "key_moments":             merged_moments,
        # v3.0.1 polish Task 144: individual_observations is appended-to
        # by _fold_player_fields below. Starting empty here means a
        # structural-only event window with no player file still produces
        # a valid empty-list field; running both inputs (the common case)
        # produces the player agent's observations.
        "individual_observations": [],
        "duels":                   [],
        # v3.0.1 polish Task 144: structural-only fields that the pre-fix
        # allowlist silently dropped. gk_distribution drove the ~45% GK
        # event drop on event windows (Task 122 diagnosis); far_side_shape
        # similarly absent for broadcast / dual-camera matches.
        "gk_distribution":         a.get("gk_distribution", []) +
                                   b.get("gk_distribution", []),
        "far_side_shape":          a.get("far_side_shape", b.get("far_side_shape")),
        "flaggable_moments":       a.get("flaggable_moments", []) +
                                   b.get("flaggable_moments", []),
        "possession_summary":      a.get("possession_summary", {}),
        "window_summary":          a.get("window_summary", ""),
        "findings":                merged_findings,
        "confirmation_queue":      merged_cq,

        "window_confidence": {
            "overall_score": merge_numeric(
                a.get("window_confidence", {}).get("overall_score"),
                b.get("window_confidence", {}).get("overall_score"),
            ),
            "low_confidence_frame_count": max(
                a.get("window_confidence", {}).get("low_confidence_frame_count", 0),
                b.get("window_confidence", {}).get("low_confidence_frame_count", 0),
            ),
            "data_gap_warning": (
                a.get("window_confidence", {}).get("data_gap_warning", False) or
                b.get("window_confidence", {}).get("data_gap_warning", False)
            ),
        },
        "merged_at": datetime.now().isoformat(),
    }

    # v3.0.1 polish Task 144: fold the player agent's canonical fields
    # into the merged dict. Done after dict construction so the helper
    # can append to individual_observations/duels (already seeded as []
    # above) and overwrite player-canonical fields cleanly.
    _fold_player_fields(merged, player)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(merged, "agent_merged"), f, indent=2)

    issues = len(review_required)
    n_obs = len(merged.get("individual_observations") or [])
    n_gk  = len(merged.get("gk_distribution") or [])
    print(f"  [{agent_id}] Dual   -> {os.path.basename(out_path)}"
          f" | issues: {issues}"
          f" | player-fold: obs={n_obs} | gk_dist: {n_gk}"
          + (f" [!] review needed" if issues else ""))

    if review_required:
        for issue in review_required:
            print(f"    [!] {issue}")

    return merged


# -- Merge all windows ---------------------------------------------------------

def merge_all_windows(match_dir: str) -> dict:
    """
    Merge all windows in window_plan.json.
    Event windows: dual agent merge.
    Routine windows: single agent merge.
    Returns summary of merge results.
    """
    logs_dir  = os.path.join(match_dir, "agent_logs")
    plan_path = os.path.join(match_dir, "window_plan.json")

    if not os.path.exists(plan_path):
        raise FileNotFoundError(f"window_plan.json not found in {match_dir}")

    with open(plan_path, encoding="utf-8") as f:
        plan = json.load(f)

    total    = plan["total_windows"]
    merged   = 0
    skipped  = 0
    errors   = []

    print(f"\n{'-'*55}")
    print(f"  Step 3e -- Merging {total} windows")
    print(f"{'-'*55}")

    for w in plan["windows"]:
        agent_id = w["agent_id"]
        label    = w.get("label", agent_id)
        wid      = w.get("window_id", agent_id)

        # Safe filename: replace characters invalid on Windows
        safe_label = label.replace(' ', '_').replace(':', '-')
        out_file   = os.path.join(logs_dir, f"agent_{agent_id}_{safe_label}_merged.json")

        # Locate Agent A (structural) using multi-pattern finder so any
        # writer naming convention works.
        a_file = _find_agent_file(logs_dir, wid, "structural")
        if not a_file:
            # Fall back to any non-event/non-merged file for this window.
            for sfx in ("player",):
                a_file = _find_agent_file(logs_dir, wid, sfx)
                if a_file:
                    break

        if not a_file:
            errors.append(f"[{agent_id}] Agent A file not found")
            skipped += 1
            continue

        # Agent B (event) lookup uses the same finder; falls back to legacy name.
        b_file = _find_agent_file(logs_dir, wid, "event")
        if not b_file:
            legacy_b = os.path.join(logs_dir, f"agent_{agent_id}_agentB.json")
            if os.path.exists(legacy_b):
                b_file = legacy_b

        # v3.0.1 polish Task 144: player file is an ADDITIVE third input,
        # not a fallback for Agent A. Pre-fix the lookup above only used
        # the player file when no structural file existed, which is almost
        # never on modern v3 matches — so individual_observations / duels /
        # player_escalation_queue never reached merged windows and
        # player_summary_cards rendered empty content.
        # Only treat as third input if it's not already being used as
        # Agent A's fallback (i.e. structural was found).
        p_file = None
        if a_file and "structural" in os.path.basename(a_file):
            p_file = _find_agent_file(logs_dir, wid, "player")

        if w.get("event_window") and b_file and os.path.exists(b_file):
            try:
                merge_dual_agents(a_file, b_file, out_file, agent_id,
                                  match_dir, player_path=p_file)
                merged += 1
            except Exception as e:
                errors.append(f"[{agent_id}] Dual merge failed: {e}")
                skipped += 1
        else:
            if w.get("event_window") and (not b_file or not os.path.exists(b_file)):
                print(f"  [{agent_id}] [!] Event window but no Agent B -- "
                      f"falling back to single agent")
            try:
                merge_single_agent(a_file, out_file, agent_id,
                                   match_dir, player_path=p_file)
                merged += 1
            except Exception as e:
                errors.append(f"[{agent_id}] Single merge failed: {e}")
                skipped += 1

    print(f"{'-'*55}")
    print(f"  Merged: {merged}/{total} | Skipped: {skipped}")
    if errors:
        print(f"  Errors:")
        for e in errors: print(f"    [FAIL] {e}")
    print(f"{'-'*55}")

    return {
        "total": total, "merged": merged,
        "skipped": skipped, "errors": errors,
    }


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    merge_all_windows(match_dir)
