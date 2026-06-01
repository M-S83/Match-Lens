"""
accumulator.py -- Match Lens Steps 3f and 3g

Step 3f: Appends pass sequences from each merged window to pass_sequences.json
Step 3g: Updates running_summary.json from each merged window

These run after each window is merged (Steps 3f/3g) and again after
confirmation items are resolved (Step 3i updates running_summary).

Usage:
    from accumulator import accumulate_pass_sequences, update_running_summary
    from accumulator import accumulate_all_windows

    # After each window merge:
    accumulate_pass_sequences(merged_path, pass_sequences_path)
    update_running_summary(merged_path, summary_path)

    # Or process all merged windows in one pass:
    accumulate_all_windows(match_dir)
"""

import json
import os
import sys
import glob
from datetime import datetime
from pipeline_accessors import get_marking, get_match_id
from pipeline_schemas import stamp_schema_version


def _normalise_team_ref(value: str, mc: dict) -> str:
    """
    Normalise a team reference string to canonical 'home' or 'away'.

    Agents may write any of the following for the home team:
      - 'home_kit'
      - 'home'
      - the actual team name  (e.g. 'Gorleston')
      - the kit colour        (e.g. 'red')

    Returns 'home', 'away', or 'unknown'.
    """
    if not value:
        return "unknown"

    v = str(value).lower().strip()

    # Canonical strings
    if v in ("home", "home_kit", "home_team"):
        return "home"
    if v in ("away", "away_kit", "away_team"):
        return "away"

    # Team name matching
    home_name = str(mc.get("home_team", "")).lower().strip()
    away_name = str(mc.get("away_team", "")).lower().strip()

    if home_name and v == home_name:
        return "home"
    if away_name and v == away_name:
        return "away"

    # Partial match (handles "Gorleston FC" vs "Gorleston")
    if home_name and home_name in v:
        return "home"
    if away_name and away_name in v:
        return "away"

    # Kit colour matching from match_config
    kits = mc.get("kits", {})
    home_kit = str(kits.get("home", mc.get("home_kit", ""))).lower()
    away_kit = str(kits.get("away", mc.get("away_kit", ""))).lower()

    if home_kit and v == home_kit:
        return "home"
    if away_kit and v == away_kit:
        return "away"

    return "unknown"


# -- Pass sequence accumulation ------------------------------------------------

def accumulate_pass_sequences(merged_path: str,
                               pass_sequences_path: str) -> int:
    """
    Append pass sequences from a merged window JSON to pass_sequences.json.
    Returns number of sequences appended.
    """
    with open(merged_path, encoding="utf-8") as f:
        window = json.load(f)

    new_sequences = window.get("pass_sequences", [])
    if not new_sequences:
        return 0

    # Tag each sequence with its window
    window_label = window.get("window", "unknown")
    for seq in new_sequences:
        seq.setdefault("window", window_label)

    if os.path.exists(pass_sequences_path):
        with open(pass_sequences_path, encoding="utf-8") as f:
            ps = json.load(f)
        ps.setdefault("total_sequences", len(ps.get("sequences", [])))
        ps.setdefault("sequences", [])
    else:
        ps = {
            "match":           window.get("match", ""),
            "total_sequences": 0,
            "sequences":       [],
        }

    ps["sequences"].extend(new_sequences)
    ps["total_sequences"] = len(ps["sequences"])
    ps["last_updated"]    = datetime.now().isoformat()

    with open(pass_sequences_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(ps, "pass_sequences"), f, indent=2)

    return len(new_sequences)


# -- Running summary update ----------------------------------------------------

def _normalise_trigger(text: str) -> str:
    """Map free-text trigger descriptions to structured codes."""
    t = text.lower()
    if any(k in t for k in ("gk", "goalkeeper", "keeper", "goal kick",
                             "in possession", "in hands", "on ball")):
        return "gk_in_possession"
    if any(k in t for k in ("back pass", "back-pass", "played back")):
        return "back_pass"
    if any(k in t for k in ("facing own goal", "back to goal", "turned",
                             "facing away", "back to pressure")):
        return "defender_facing_own_goal"
    if any(k in t for k in ("free kick", "free-kick", "restart")):
        return "free_kick_restart"
    if any(k in t for k in ("throw", "throw-in")):
        return "throw_in_restart"
    # Already a code
    known = ("gk_in_possession", "back_pass", "defender_facing_own_goal",
             "free_kick_restart", "throw_in_restart", "other")
    if any(text == k for k in known):
        return text
    return "other"


VALID_SET_PIECE_TYPES = {
    "corner_left", "corner_right",
    "direct_fk", "indirect_fk",
    "throw_final_third", "kickoff",
    # legacy compatibility -- accept but log
    "corner", "free_kick", "throw_in",
}


def validate_set_piece(sp: dict, window_id) -> tuple:
    """
    Returns (is_valid, normalized_entry, rejection_reason).
    A set piece without a valid timestamp cannot be escalated to 5fps -- reject
    it and route to data_gap_windows for visibility.
    """
    ts = sp.get("timestamp")
    if not ts or not isinstance(ts, str) or "m" not in ts or "s" not in ts:
        return False, None, f"missing_or_malformed_timestamp:{ts!r}"

    sp_type = sp.get("type")
    if sp_type not in VALID_SET_PIECE_TYPES:
        return False, None, f"invalid_type:{sp_type!r}"

    team = sp.get("team")
    if team not in ("home_kit", "away_kit"):
        return False, None, f"invalid_team:{team!r}"

    normalized = {
        "timestamp":          ts,
        "timestamp_inferred": sp.get("timestamp_inferred", False),
        "type":               sp_type,
        "team":               team,
        "taker_position":     sp.get("taker_position"),
        "delivery_zone":      sp.get("delivery_zone"),
        "bodies_in_box":      sp.get("bodies_in_box"),
        "marking_system":     get_marking(sp),
        "rest_defence":       sp.get("rest_defence") or sp.get("rest_defence_count"),
        "outcome":            sp.get("outcome"),
        "delivery_observed":  sp.get("delivery_observed", True),
        "window":             window_id,
        # delivery/runners alias (metric reads 'delivery')
        "delivery":           sp.get("delivery") or sp.get("delivery_type"),
        "delivery_type":      sp.get("delivery_type"),
        "runners":            sp.get("runners", []),
        "wall_size":          sp.get("wall_size"),
        "wall_position":      sp.get("wall_position"),
        # Preserve burst write-back fields if already present in the merged record.
        # On first accumulation these will be None/False (the 1fps defaults).
        # On re-accumulation after 5fps burst they carry the enriched values.
        "source":               sp.get("source", "1fps_initial"),
        "burst_resolved":       sp.get("burst_resolved", False),
        "burst_fps":            sp.get("burst_fps"),
        "burst_notes":          sp.get("burst_notes"),
        "resolved_at":          sp.get("resolved_at"),
        "corrections":          sp.get("corrections", []),
        "delivery_target_zone": sp.get("delivery_target_zone"),
        "second_phase":         sp.get("second_phase"),
    }
    return True, normalized, None


def build_set_piece_queue_entry(sp: dict, window_id) -> dict:
    """
    Auto-generate a confirmation_queue entry from a validated set piece.
    The escalation_router (Step 3i) picks this up and routes to 3d-SP at 5fps.
    """
    return {
        "event_type":     "set_piece_delivery",
        "timestamp":      sp["timestamp"],
        "window":         window_id,
        "priority":       "high",
        "set_piece_type": sp["type"],
        "team":           sp["team"],
        "reason":         "set piece delivery requires 5fps burst for runner and delivery_type capture",
        "auto_queued":    True,
        "source_record":  {
            "delivery_zone": sp.get("delivery_zone"),
            "outcome":       sp.get("outcome"),
        },
    }


def append_to_confirmation_queue(entries: list, queue_path: str) -> int:
    """
    Append entries to confirmation_queue.json.
    Dedupes by (event_type, timestamp, window) so manual + auto queuing of
    the same set piece collapses to one entry.
    Returns number of new entries appended.
    """
    if os.path.exists(queue_path):
        with open(queue_path, encoding="utf-8") as f:
            queue = json.load(f)
    else:
        queue = {"total": 0, "skipped": 0, "items": []}

    existing_keys = {
        (item.get("event_type"), item.get("timestamp"), item.get("window"))
        for item in queue.get("items", [])
    }

    appended = 0
    for entry in entries:
        key = (entry.get("event_type"), entry.get("timestamp"), entry.get("window"))
        if key in existing_keys:
            continue
        queue.setdefault("items", []).append(entry)
        existing_keys.add(key)
        appended += 1

    queue["total"] = len(queue.get("items", []))
    with open(queue_path, "w", encoding="utf-8") as f:
        # This is the minimal-shape initialisation write; the full
        # confirmation_queue schema is rebuilt by escalation_router.py
        # later in the pipeline. Both writers stamp the same schema
        # version; consumers should not read confirmation_queue.json
        # between these two writes.
        json.dump(stamp_schema_version(queue, "confirmation_queue"), f, indent=2)

    return appended


def update_running_summary(merged_path: str,
                            summary_path: str,
                            queue_path: "str | None" = None) -> dict:
    """
    Update running_summary.json from a merged window JSON.
    Creates the summary file if it doesn't exist.
    Returns the updated summary.
    """
    with open(merged_path, encoding="utf-8") as f:
        w = json.load(f)

    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        # Ensure all expected keys exist (handles partial or legacy files)
        for key in ["formation_history", "pressing_by_window", "line_height_by_window",
                    "line_height_m_by_window",
                    "match_state_by_window", "possession_by_window",
                    "shots_for", "shots_against", "flagged_moments", "key_moments",
                    "individual_observations", "set_pieces", "set_pieces_rejected",
                    "transitions", "gk_kicks",
                    "duels", "findings", "confirmation_queue", "data_gap_windows"]:
            summary.setdefault(key, [])
        summary.setdefault("windows_complete", 0)
        summary.setdefault("match", "")
        summary.setdefault("home_team", "")
        summary.setdefault("away_team", "")
    else:
        summary = {
            "match":                  "",
            "windows_complete":       0,
            "formation_history":      [],
            "pressing_by_window":     [],
            "line_height_by_window":  [],
            "line_height_m_by_window":[],
            "match_state_by_window":  [],
            "shots_for":              [],
            "shots_against":          [],
            "flagged_moments":        [],
            "key_moments":            [],
            "individual_observations":[],
            "set_pieces":             [],
            "set_pieces_rejected":    [],
            "transitions":            [],
            "possession_by_window":   [],
            "gk_kicks":               [],
            "duels":                  [],
            "data_gap_windows":       [],
            "findings":               [],
            "confirmation_queue":     [],
        }

    summary["windows_complete"] += 1
    if not summary.get("match"):
        summary["match"] = w.get("match", "")

    # Formation
    # Fix 32a introduced the new structural schema:
    #   formation.{home, away, home_shape, away_shape}
    # Legacy schema (pre-Fix-32a, still in use for Veo runs on Gorleston etc.):
    #   formation.{shape_in_possession, shape_out_of_possession}
    # We populate both new and legacy keys so consumers of either work.
    _formation = w.get("formation", {}) or {}
    _home_form  = _formation.get("home")
    _away_form  = _formation.get("away")
    _home_shape = _formation.get("home_shape") or _formation.get("shape_in_possession")
    _away_shape = _formation.get("away_shape") or _formation.get("shape_out_of_possession")

    # Derive window_id from agent_id when the merged file's "window" field is null
    # (the merge step doesn't always preserve window timing).
    _agent_id = w.get("agent_id")
    _wid = None
    if _agent_id:
        import re as _re_aid
        _m = _re_aid.search(r'(\d+)', str(_agent_id))
        if _m:
            _wid = int(_m.group(1))

    summary["formation_history"].append({
        "window_id":      _wid,
        "window":         w.get("window"),
        "agent_id":       _agent_id,
        "home_formation": _home_form,
        "away_formation": _away_form,
        "home_shape":     _home_shape,
        "away_shape":     _away_shape,
        # Legacy keys preserved for any downstream consumer that reads them
        "shape":          _home_shape,
        "shape_oop":      _away_shape,
    })

    # Match state (score and winning/losing at window start)
    if "match_state_by_window" not in summary:
        summary["match_state_by_window"] = []
    ms = w.get("match_state")
    if ms:
        if isinstance(ms, dict):
            summary["match_state_by_window"].append({
                "window":        w.get("window"),
                "agent_id":      w.get("agent_id"),
                "score_home":    ms.get("score_home",    0),
                "score_away":    ms.get("score_away",    0),
                "score_focus":   ms.get("score_focus",   0),
                "score_opponent":ms.get("score_opponent", 0),
                "match_state":   ms.get("match_state",   "unknown"),
            })
        else:
            summary["match_state_by_window"].append({
                "window":      w.get("window"),
                "agent_id":    w.get("agent_id"),
                "match_state": str(ms),
            })

    # Pressing -- extract trigger codes from scores array into observations
    pressing_data = w.get("pressing", {}) if isinstance(w.get("pressing"), dict) else {}
    trigger_observations = []
    for score_entry in pressing_data.get("scores", []):
        trigger = score_entry.get("trigger")
        if trigger and str(trigger).lower() not in ("null", "none", ""):
            trigger_code = _normalise_trigger(str(trigger))
            trigger_observations.append({
                "trigger":   trigger_code,
                "timestamp": score_entry.get("frame_group", ""),
                "score":     score_entry.get("score", 0),
            })
    summary["pressing_by_window"].append({
        "window":    w.get("window"),
        "agent_id":  w.get("agent_id"),
        "avg_score": pressing_data.get("avg_score"),
        "peak":      pressing_data.get("peak_score"),
        "peak_ts":   pressing_data.get("peak_timestamp"),
        "observations": trigger_observations,
    })

    # Defensive line -- store both pct and metres
    def _pct_to_m(pct, pitch=105.0):
        return round(pct / 100.0 * pitch, 1) if pct is not None else None

    dl = w.get("defensive_line", {})
    if not isinstance(dl, dict):
        dl = {}

    avg_pct   = dl.get("avg_pct")
    start_pct = dl.get("start_pct")
    end_pct   = dl.get("end_pct")

    summary["line_height_by_window"].append({
        "window":   w.get("window"),
        "agent_id": w.get("agent_id"),
        "avg_pct":  avg_pct,
        "avg_m":    _pct_to_m(avg_pct),
        "start_pct":start_pct,
        "start_m":  _pct_to_m(start_pct),
        "end_pct":  end_pct,
        "end_m":    _pct_to_m(end_pct),
        "shifts":   dl.get("notable_shifts", []),
    })

    # v3 line_height_m_by_window -- fixes the home/away vs avg_pct field-name
    # mismatch. The structural agent schema emits home_height_pct and
    # away_height_pct (not avg_pct). The old line_height_by_window above keeps
    # reading dl.get("avg_pct") for backward compatibility, which is None on
    # current structural outputs. Downstream deep_skill_metrics reads
    # line_height_m_by_window looking for avg_m_approx; this block populates it.
    home_height_pct = dl.get("home_height_pct")
    away_height_pct = dl.get("away_height_pct")

    # Derive avg_pct: prefer home/away mean if either is present,
    # fall back to legacy dl.get("avg_pct") if not.
    if home_height_pct is not None and away_height_pct is not None:
        derived_avg_pct = round((home_height_pct + away_height_pct) / 2.0, 1)
    elif home_height_pct is not None:
        derived_avg_pct = home_height_pct
    elif away_height_pct is not None:
        derived_avg_pct = away_height_pct
    else:
        derived_avg_pct = avg_pct  # legacy fallback

    # avg_m_approx: prefer direct field if present, derive from pct otherwise
    avg_m_approx = dl.get("avg_m_approx")
    if avg_m_approx is None:
        avg_m_approx = _pct_to_m(derived_avg_pct)

    summary["line_height_m_by_window"].append({
        "window":              w.get("window"),
        "agent_id":            w.get("agent_id"),
        "avg_pct":             derived_avg_pct,
        "avg_m_approx":        avg_m_approx,
        "line_width_m_approx": dl.get("line_width_m_approx"),
        "space_behind_m":      dl.get("space_behind_m"),
        "home_height_pct":     home_height_pct,
        "away_height_pct":     away_height_pct,
        "shifts":              dl.get("notable_shifts", []),
    })

    # Shots -- separate by home/away. Fix 33b: pipeline produces
    # reports for both teams independently; "for" == home, "against" == away.
    home_team = summary.get("home_team", "")
    for shot in w.get("shot_attempts", []):
        if shot.get("team") == home_team:
            summary["shots_for"].append(shot)
        else:
            summary["shots_against"].append(shot)

    # Moments and observations
    summary["flagged_moments"].extend(w.get("flaggable_moments", []))
    summary["key_moments"].extend(w.get("key_moments", []))
    # Grade each observation as it is accumulated
    raw_obs = w.get("individual_observations", [])
    try:
        import deep_skill_metrics as _dsm
        for obs in raw_obs:
            if "obs_grade" not in obs:
                obs["obs_grade"] = _dsm.observation_grade(
                    obs.get("confidence", "low"),
                    obs.get("frequency", "single")
                )
    except Exception:
        pass
    summary["individual_observations"].extend(raw_obs)

    # Set pieces: validate timestamps + types, reject invalid entries, auto-queue valid ones
    _window_id = w.get("window") or w.get("agent_id")
    _valid_sps   = []
    _rejected_sps = []
    _queue_entries = []
    for sp in (w.get("set_pieces") or []):
        is_valid, normalized, reason = validate_set_piece(sp, _window_id)
        if is_valid:
            _valid_sps.append(normalized)
            _queue_entries.append(build_set_piece_queue_entry(normalized, _window_id))
        else:
            _rejected_sps.append({
                "window":     _window_id,
                "reason":     reason,
                "raw_entry":  sp,
                "rejected_at": datetime.now().isoformat(),
            })

    summary["set_pieces"].extend(_valid_sps)
    summary.setdefault("set_pieces_rejected", []).extend(_rejected_sps)

    if _rejected_sps and _window_id not in summary.get("data_gap_windows", []):
        summary.setdefault("data_gap_windows", []).append(_window_id)

    if _queue_entries and queue_path:
        appended = append_to_confirmation_queue(_queue_entries, queue_path)
        print(f"  [SP] Auto-queued {appended} set piece(s) from window {_window_id}")

    # GK kicks — both teams [B]
    for kick in (w.get("gk_kicks") or []):
        summary["gk_kicks"].append(kick)

    # Duels — both teams [I]
    for duel in (w.get("duels") or []):
        summary["duels"].append(duel)
    # Transitions (added in schema update)
    if "transitions" not in summary:
        summary["transitions"] = []
    summary["transitions"].extend(w.get("transitions", []))

    # Possession
    # Possession from both-teams sequence counts [A+G]
    _ps_raw = w.get("pass_sequences") or []
    if isinstance(_ps_raw, dict):
        _ps_raw = _ps_raw.get("sequences", [])
    _win_seqs = [s for s in _ps_raw if s.get("window") == w.get("window")]
    if not _win_seqs:
        _win_seqs = _ps_raw
    _match_dir  = os.path.dirname(os.path.dirname(merged_path))
    _cfg_path   = os.path.join(_match_dir, "match_config.json")
    _cfg        = {}
    _home_name  = ""
    _away_name  = ""
    if os.path.exists(_cfg_path):
        with open(_cfg_path, encoding="utf-8") as _cf:
            _cfg = json.load(_cf)
        _home_name = _cfg.get("home_team", "")
        _away_name = _cfg.get("away_team", "")
    _home_seqs = sum(1 for s in _win_seqs
                     if _normalise_team_ref(s.get("team", ""), _cfg) == "home")
    _away_seqs = sum(1 for s in _win_seqs
                     if _normalise_team_ref(s.get("team", ""), _cfg) == "away")
    _total_both = _home_seqs + _away_seqs
    _focus_pct  = round(_home_seqs / max(_total_both, 1) * 100, 1) if _total_both > 0 else None
    summary["possession_by_window"].append({
        "window":     w.get("window"),
        "summary":    w.get("possession_summary", {}),
        "focus_pct":  _focus_pct,
        "focus_seqs": _home_seqs,
        "opp_seqs":   _away_seqs,
        "basis":      "sequence_count" if _total_both > 0 else "estimated",
    })

    # Data gap flag
    if w.get("window_confidence", {}).get("data_gap_warning"):
        summary["data_gap_windows"].append(w.get("window"))

    # Findings -- accumulate with gate states already applied
    summary["findings"].extend(w.get("findings", []))

    # Confirmation queue -- accumulate items for Step 3i
    for item in w.get("confirmation_queue", []):
        # Avoid duplicates by timestamp
        existing_ts = {i.get("timestamp") for i in summary["confirmation_queue"]}
        if item.get("timestamp") not in existing_ts:
            summary["confirmation_queue"].append(item)

    summary["last_updated"] = datetime.now().isoformat()

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(summary, "running_summary"), f, indent=2)

    return summary



# -- Batch accumulation --------------------------------------------------------

def _majority_formation(logs_dir: str, team_key: str, mc: dict | None = None) -> "str | None":
    """
    Walk every structural agent JSON in logs_dir, find every
    formation-shaped string (e.g. "4-2-3-1", "4-4-1-1"), classify
    each by the path it lives under (home / away / kit-color), and
    return the most frequent formation for the requested side.

    team_key:
        "home_formation" -> tally HOME side
        "away_formation" -> tally AWAY side

    Structural agents improvise the JSON shape per window, so paths
    range from "team_observations.home.formation_shape" to
    "defensive_shapes.away_kit.formation" to "tactical_overview.
    home_kit_structure" etc. Path-based classification handles them
    without listing every variant.

    mc (optional): match_config dict, used to map kit colours to
    sides (e.g. blue->home, white->away for Wingate vs Cray).
    """
    import re as _re
    from collections import Counter

    form_re = _re.compile(r'\b(\d-\d-\d(?:-\d)?)\b')
    counts = {"home": Counter(), "away": Counter()}

    # Build kit-colour -> side map from match_config
    kit_to_side = {}
    if isinstance(mc, dict):
        kits = mc.get("kits", {}) or {}
        home_kit = (kits.get("home") or mc.get("home_kit") or "").lower()
        away_kit = (kits.get("away") or mc.get("away_kit") or "").lower()
        if home_kit:
            kit_to_side[home_kit] = "home"
        if away_kit:
            kit_to_side[away_kit] = "away"
        # Common alternative kit-colour readings (e.g. "yellow" for white-kit GK confusion)
        # are NOT mapped here -- if an agent says "yellow" but config says "white",
        # we don't know which side they meant, so skip rather than guess wrong.

    HOME_HINTS = {"home", "focus", "home_team", "home_kit", "blue_team",
                  "blue_kit", "blue_defensive_shape"}
    AWAY_HINTS = {"away", "opponent", "away_team", "away_kit", "white_team",
                  "white_kit", "yellow_team", "yellow_kit",
                  "yellow_defensive_shape", "white_defensive_shape"}

    def _classify(path_segments):
        """Decide if path indicates home or away. Returns 'home', 'away', or None."""
        for seg in path_segments:
            seg_l = str(seg).lower()
            if seg_l in HOME_HINTS or any(h in seg_l for h in ("home", "focus")):
                return "home"
            if seg_l in AWAY_HINTS or any(a in seg_l for a in ("away", "opponent")):
                return "away"
            # Kit-colour based hints
            for kit, side in kit_to_side.items():
                if kit and kit in seg_l:
                    return side
        return None

    def _walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                _walk(v, path + [k])
        elif isinstance(node, list):
            for v in node:
                _walk(v, path + ["[]"])
        elif isinstance(node, str):
            m = form_re.search(node)
            if m:
                side = _classify(path)
                if side:
                    counts[side][m.group(1)] += 1

    for fpath in glob.glob(os.path.join(logs_dir, "*structural*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        _walk(d, [])

    side = "home" if "home" in team_key or "focus" in team_key else "away"
    if not counts[side]:
        return None
    return counts[side].most_common(1)[0][0]


def _formation_distribution(logs_dir: str, mc: "dict | None" = None) -> dict:
    """
    Return per-side formation count distribution across all structural windows.

    Reads only formation.home and formation.away — deliberately excludes
    home_variation / away_variation strings, which contain narrative text
    like "pushes to 3-2-5 in possession" that would otherwise pollute
    the distribution with phase-specific shape descriptions.
    """
    import re as _re
    from collections import Counter

    form_re = _re.compile(r'\b(\d-\d-\d(?:-\d)?)\b')
    counts = {"home": Counter(), "away": Counter()}

    for fpath in glob.glob(os.path.join(logs_dir, "*structural*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        f_data = d.get("formation", {})
        if not isinstance(f_data, dict):
            continue
        # Only count the primary home/away formation keys — not variation text
        for side in ("home", "away"):
            val = f_data.get(side)
            if isinstance(val, str):
                m = form_re.search(val)
                if m:
                    counts[side][m.group(1)] += 1

    return {"home": dict(counts["home"]), "away": dict(counts["away"])}


def _ip_shape_summary(logs_dir: str) -> dict:
    """
    Collect home_variation and away_variation strings from structural logs.
    Returns frequency counts of observed IP/OOP shape changes, which are
    deliberately excluded from _formation_distribution.
    """
    from collections import Counter

    TRIVIAL = {None, "", "null", "shape consistent in both phases",
               "shape consistent in both phases.",
               "only in possession phase observed",
               "only out of possession phase observed"}

    ip_home = Counter()
    ip_away = Counter()

    for fpath in glob.glob(os.path.join(logs_dir, "*structural*.json")):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue
        f_data = d.get("formation", {})
        if not isinstance(f_data, dict):
            continue
        hv = f_data.get("home_variation")
        av = f_data.get("away_variation")
        if hv not in TRIVIAL and isinstance(hv, str):
            ip_home[hv.strip()] += 1
        if av not in TRIVIAL and isinstance(av, str):
            ip_away[av.strip()] += 1

    return {
        "available":               bool(ip_home or ip_away),
        "home_variations":         dict(ip_home.most_common(10)),
        "away_variations":         dict(ip_away.most_common(10)),
        "home_variation_windows":  sum(ip_home.values()),
        "away_variation_windows":  sum(ip_away.values()),
    }


def _accumulate_far_side_shape(logs_dir: str, mc: dict) -> dict:
    """
    Aggregate far_side_shape observations across all structural windows.
    Returns a summary of dominant fullback positions per side and a
    handful of sample observations. Returns {available: False} if the
    field is absent across all windows (e.g. veo_ball_tracking source).
    """
    from collections import Counter

    home_pos = Counter()
    away_pos = Counter()
    observations = []
    windows_with_data = 0

    for fpath in sorted(glob.glob(os.path.join(logs_dir, "*structural*.json"))):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            continue

        far = d.get("far_side_shape")
        if not isinstance(far, dict) or not far.get("observable"):
            continue

        windows_with_data += 1
        h = far.get("home_fullback_position")
        a = far.get("away_fullback_position")
        if h and h != "not_visible":
            home_pos[h] += 1
        if a and a != "not_visible":
            away_pos[a] += 1
        observations.append({
            "window":         os.path.basename(fpath),
            "home_far_side":  far.get("home_far_side", ""),
            "away_far_side":  far.get("away_far_side", ""),
        })

    if windows_with_data == 0:
        return {
            "available": False,
            "note": "No far-side shape data -- source may not support it",
        }

    return {
        "available":                       True,
        "windows_with_data":               windows_with_data,
        "home_fullback_dominant_position": (home_pos.most_common(1)[0][0]
                                            if home_pos else "unknown"),
        "home_fullback_distribution":      dict(home_pos),
        "away_fullback_dominant_position": (away_pos.most_common(1)[0][0]
                                            if away_pos else "unknown"),
        "away_fullback_distribution":      dict(away_pos),
        "sample_observations":             observations[:5],
    }


def _accumulate_press_triggers(logs_dir: str, mc: dict) -> dict:
    """
    Aggregate press trigger observations across all structural windows.
    Returns the most common press triggers per side and windows covered.

    Fix 32e: agents write `pressing.home_trigger` / `pressing.away_trigger`
    on every 1fps structural window. Previously discarded by the accumulator;
    aggregated they reveal whether a team's pressing is back-pass triggered,
    GK-distribution triggered, etc.
    """
    import glob
    from collections import Counter

    home_triggers      = Counter()
    away_triggers      = Counter()
    windows_with_data  = 0

    for fpath in sorted(glob.glob(os.path.join(logs_dir, '*structural*.json'))):
        try:
            with open(fpath, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue

        pressing = d.get('pressing', {})
        if not isinstance(pressing, dict):
            continue

        home_trigger = pressing.get('home_trigger')
        away_trigger = pressing.get('away_trigger')

        captured_in_window = False
        if home_trigger and str(home_trigger).lower() not in ('null', 'none', '?', ''):
            home_triggers[str(home_trigger)] += 1
            captured_in_window = True
        if away_trigger and str(away_trigger).lower() not in ('null', 'none', '?', ''):
            away_triggers[str(away_trigger)] += 1
            captured_in_window = True

        # Some agents use a separate `press_trigger` / `pressing_triggers`
        # / `press_triggers_identified` field at top level. Bucket those
        # into away_triggers as a fallback.
        for field in ('press_trigger', 'pressing_triggers',
                      'press_triggers_identified'):
            val = d.get(field)
            if not val:
                continue
            captured_in_window = True
            if isinstance(val, list):
                for v in val:
                    away_triggers[str(v)] += 1
            else:
                away_triggers[str(val)] += 1

        if captured_in_window:
            windows_with_data += 1

    if windows_with_data == 0:
        return {
            'available': False,
            'note': 'No press trigger data found in structural outputs',
        }

    def _top(counter, n=3):
        return [{'trigger': t, 'windows': c} for t, c in counter.most_common(n)]

    return {
        'available':            True,
        'windows_with_data':    windows_with_data,
        'home_top_triggers':    _top(home_triggers),
        'away_top_triggers':    _top(away_triggers),
        'home_trigger_counts':  dict(home_triggers.most_common(10)),
        'away_trigger_counts':  dict(away_triggers.most_common(10)),
    }


def _accumulate_runner_patterns(logs_dir: str, mc: dict) -> dict:
    """
    Aggregate runner tracking observations across structural and merged windows.
    Surfaces repeated run patterns, most common run zones and types.

    Fix 32e: 1fps structural agents typically don't emit runners[] (it's a
    5fps signal). This aggregator activates when 3d_setpiece re-runs supply
    runner detail, or when richer set_pieces[].runners data is present.
    """
    import glob
    from collections import Counter

    run_zones         = Counter()
    run_types         = Counter()
    unmarked_routes   = Counter()
    windows_with_data = 0

    # Look in both raw structural files and merged files (3d_setpiece runs
    # contribute via the merge step).
    files = sorted(set(
        glob.glob(os.path.join(logs_dir, '*structural*.json')) +
        glob.glob(os.path.join(logs_dir, '*_merged.json')) +
        glob.glob(os.path.join(logs_dir, '*setpiece*.json'))
    ))

    for fpath in files:
        try:
            with open(fpath, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue

        any_runner_in_file = False

        # Top-level runners array
        runners = d.get('runners') or d.get('runner_tracking') or []
        if not isinstance(runners, list):
            runners = [runners]
        # Also walk set_pieces[].runners
        for sp in (d.get('set_pieces') or []):
            if isinstance(sp, dict):
                sp_runners = sp.get('runners') or []
                if isinstance(sp_runners, list):
                    runners.extend(sp_runners)

        for runner in runners:
            if not isinstance(runner, dict):
                continue
            any_runner_in_file = True
            zone     = runner.get('run_zone') or runner.get('zone')
            rtype    = runner.get('run_type') or runner.get('type')
            unmarked = runner.get('unmarked_runner', runner.get('unmarked', False))

            if zone:
                run_zones[str(zone)] += 1
            if rtype:
                run_types[str(rtype)] += 1
            if unmarked:
                route = f"{zone or '?'} ({rtype or '?'})"
                unmarked_routes[route] += 1

        if any_runner_in_file:
            windows_with_data += 1

    if windows_with_data == 0:
        return {
            'available': False,
            'note': ('No runner tracking data found. 1fps structural agents '
                     'typically do not emit runners[]; populated when 3d_setpiece '
                     'or richer set-piece extraction has run.'),
        }

    return {
        'available':                  True,
        'windows_with_data':          windows_with_data,
        'most_common_run_zones':      dict(run_zones.most_common(5)),
        'most_common_run_types':      dict(run_types.most_common(5)),
        'most_common_unmarked_routes': dict(unmarked_routes.most_common(5)),
    }


def _accumulate_player_tendencies(logs_dir: str, mc: dict) -> dict:
    """
    Fix 46A: build a per-player action profile across all player-agent
    windows. Tracks run patterns, positioning, defensive contribution,
    and (from structural gk_distribution arrays) goalkeeper distribution
    direction and length. Output drives the synthesis PLAYER TENDENCY,
    GK DISTRIBUTION, and EXPLOITABLE PATTERN rules.
    """
    import glob
    import re as _re
    from collections import Counter, defaultdict

    # Fix 55: Build TWO lookups so we can resolve agent-written names
    # like "Charlie Stallard (#11)" (with the (#N) suffix) against
    # match_config keys like "Charlie Stallard" (no suffix). The pre-Fix-55
    # code keyed all_players by clean name only and the direct
    # `if player_name in all_players` check always failed when the agent
    # appended (#N), so team/side/pos never populated.
    all_players = {}        # clean name → {team, side, pos, name}
    players_by_number = {}  # (side, number_int) → {team, side, pos, name}

    for lineup in mc.get("lineups", []):
        team_name = lineup.get("team", {}).get("name", "")
        side      = lineup.get("team_side", "")
        for p in lineup.get("startXI", []) + lineup.get("substitutes", []):
            player = p.get("player", p)
            name   = player.get("name", "")
            number = player.get("number", None)
            pos    = player.get("pos", "")
            if name:
                entry = {"team": team_name, "side": side, "pos": pos, "name": name}
                all_players[name] = entry
                if number is not None:
                    try:
                        players_by_number[(side, int(number))] = entry
                    except (ValueError, TypeError):
                        pass

    def _resolve_player(obs_name: str) -> dict:
        """
        Match an agent-written name like 'Charlie Stallard (#11)' to a
        verified match_config entry. Returns {team, side, pos, name} or {}.
        Strategy:
          1. Strip (#N) suffix; direct lookup
          2. Last-name partial match
          3. (#N)-based fallback, gated on a name overlap so we don't
             cross-attribute to the opposite team's #N.
        """
        if not obs_name:
            return {}
        clean = _re.sub(r'\s*\(#\d+\)\s*$', '', obs_name).strip()
        if clean in all_players:
            return all_players[clean]
        clean_lower = clean.lower()
        for key, val in all_players.items():
            kl = key.lower()
            if kl and (kl in clean_lower or clean_lower in kl):
                return val
        m = _re.search(r'\(#(\d+)\)', obs_name)
        if m:
            num = int(m.group(1))
            for side_try in ('home', 'away'):
                entry = players_by_number.get((side_try, num))
                if not entry:
                    continue
                entry_name_lower = entry['name'].lower()
                clean_words = [w for w in clean_lower.split() if len(w) > 2]
                if any(w in entry_name_lower for w in clean_words):
                    return entry
        return {}

    tendencies = defaultdict(lambda: {
        "windows_appeared":       0,
        "team":                   "",
        "side":                   "",
        "pos":                    "",
        "run_type":               Counter(),
        "positioning":            Counter(),
        "defensive_contribution": Counter(),
        "direct_play":            Counter(),
        "aerial":                 Counter(),
        "gk_distribution_zones":  Counter(),
        "gk_distribution_lengths": Counter(),
    })

    for fpath in sorted(glob.glob(os.path.join(logs_dir, '*player*.json'))):
        try:
            with open(fpath, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue

        observations = d.get("individual_observations", [])
        if not isinstance(observations, list):
            continue

        for obs in observations:
            if not isinstance(obs, dict):
                continue

            player_name = obs.get("player", "")
            if not player_name or player_name.startswith("home_") \
               or player_name.startswith("away_"):
                # Positional identifier (tactical_wide_static convention) —
                # cannot accumulate across windows by name.
                continue

            t = tendencies[player_name]
            t["windows_appeared"] += 1

            # Fix 55: resolve via _resolve_player to handle the
            # "Charlie Stallard (#11)" vs "Charlie Stallard" key mismatch
            # and to populate side (home/away) for synthesis attribution.
            resolved = _resolve_player(player_name)
            if resolved:
                t["team"] = resolved.get("team", "")
                t["side"] = resolved.get("side", "")
                t["pos"]  = resolved.get("pos", "")

            # Fix 51: scan action + observation + context + description
            # together. The agent writes most descriptive content into
            # `observation` (Bayern 86%, Wingate 68%); the `action` field
            # alone catches only ~15-19% of observations. Vocabulary map
            # broadened per real agent output (snake_case tags from
            # Bayern + long descriptive sentences from Wingate).
            text_parts = [
                str(obs.get("action", "")),
                str(obs.get("observation", "")),
                str(obs.get("context", "")),
                str(obs.get("description", "")),
                str(obs.get("action_type", "")),
                str(obs.get("tactical_context", "")),
            ]
            text = " ".join(p for p in text_parts if p).lower()

            # Order matters — most specific first. `if/elif` keeps the
            # one-category-per-observation semantics of the original.
            if any(p in text for p in (
                "run in behind", "runs in behind", "runs behind",
                "run behind", "in behind", "runs beyond", "run beyond",
                "through ball run", "stay onside", "attacking run behind",
            )):
                t["run_type"]["run_in_behind"] += 1
            elif any(p in text for p in (
                "overlapping run", "overlapping",
                "overlaps", "wide overlap",
            )) and "underlap" not in text:
                t["run_type"]["overlap"] += 1
            elif any(p in text for p in (
                "underlapping", "underlap", "inside run",
            )):
                t["run_type"]["underlap"] += 1
            elif any(p in text for p in (
                "channel run", "runs into channel", "into the channel",
                "diagonal run", "runs into the channel",
            )):
                t["run_type"]["channel_run"] += 1
            elif any(p in text for p in (
                "drops to receive", "drops into midfield", "drops deep",
                "dropping deep", "dropping into midfield",
                "dropping_deep", "drops back", "pulls back", "drops off",
                "dropping slightly", "dropping to link",
                "collects deep", "receives deep",
            )):
                t["run_type"]["drop_deep"] += 1
            elif any(p in text for p in (
                "drives at", "drives forward", "driving run",
                "runs at defender", "runs at the defender", "takes on",
                "carries ball", "carrying ball", "carrying_ball",
                "ball-carrying", "running with ball", "running_with_ball",
                "carry forward", "dribbles", "dribbling",
                " run at ", "dribble",
            )):
                t["run_type"]["run_at_defender"] += 1
            elif any(p in text for p in (
                "arrives in box", "arrives late", "late run",
                "late box", "runs into box", "movement in box",
                "arriving in box", "box-to-box",
                "arrives in the box", "late box arrival",
            )):
                t["run_type"]["late_run"] += 1
            elif any(p in text for p in (
                "hold-up play", "hold up play", "holds up",
                "holding up play", "holding_up_play",
                "back to goal", "shield possession", "shields possession",
                "lays off", "lay off", "chest down",
            )):
                t["run_type"]["hold_up"] += 1
            elif any(p in text for p in (
                "closes down", "presses ", "pressing center",
                "pressing centre", "pressing_center", "pressing_centre",
                "press the ", "presses opponent", "engages the ball",
                "winning the ball", "intercepts", "stepping out to intercept",
                "tackles ", "tracking_back", "tracks back",
            )):
                t["run_type"]["pressing"] += 1

            zone = str(obs.get("zone", "")).lower()
            if "attacking" in zone:
                t["positioning"]["high"] += 1
            elif "defending" in zone:
                t["positioning"]["deep"] += 1
            elif zone:
                t["positioning"]["standard"] += 1

            context = str(obs.get("context", "")).lower()
            if "track" in context or "press" in context or "cover" in context:
                t["defensive_contribution"]["tracks_back"] += 1
            elif "hold" in context or "position" in context:
                t["defensive_contribution"]["holds_position"] += 1

    # GK distribution from structural gk_distribution arrays
    home_team = mc.get("home_team", "")
    away_team = mc.get("away_team", "")
    gk_home = None
    gk_away = None
    for lineup in mc.get("lineups", []):
        for p in lineup.get("startXI", []):
            player = p.get("player", p)
            if player.get("pos") == "GK":
                name = player.get("name", "")
                if lineup.get("team_side") == "home":
                    gk_home = name
                else:
                    gk_away = name

    for fpath in sorted(glob.glob(os.path.join(logs_dir, '*structural*.json'))):
        try:
            with open(fpath, encoding='utf-8') as f:
                d = json.load(f)
        except Exception:
            continue

        for kick in d.get("gk_distribution", []) or []:
            if not isinstance(kick, dict):
                continue
            team = str(kick.get("team", ""))
            gk_name = gk_home if "home" in team.lower() else gk_away
            if not gk_name:
                continue
            t = tendencies[gk_name]
            t["windows_appeared"] = t.get("windows_appeared", 0)
            t["team"] = home_team if "home" in team.lower() else away_team
            t["side"] = "home" if "home" in team.lower() else "away"  # Fix 55
            t["pos"] = "GK"
            zone   = kick.get("target_zone", "")
            length = kick.get("length", "")
            if zone:
                t["gk_distribution_zones"][zone] += 1
            if length:
                t["gk_distribution_lengths"][length] += 1

    PATTERN_THRESHOLD     = 3
    CONSISTENT_THRESHOLD  = 5

    def _dominant(counter, threshold=PATTERN_THRESHOLD, min_share=0.35):
        """
        Returns (top_key, top_count, confidence).
        Fix 46C: confidence is "consistent" only if BOTH count >=
        CONSISTENT_THRESHOLD AND count/total >= min_share. This stops
        wide distributions (e.g. GK kicks across 5 zones with 5 each)
        from being labelled "consistent" just because the absolute count
        clears the floor while no zone genuinely dominates.
        """
        if not counter:
            return None, 0, "insufficient"
        top_key, top_count = counter.most_common(1)[0]
        total = sum(counter.values())
        share = top_count / total if total else 0
        if top_count >= CONSISTENT_THRESHOLD and share >= min_share:
            confidence = "consistent"
        elif top_count >= threshold:
            confidence = "tendency"
        else:
            confidence = "insufficient"
        return top_key, top_count, confidence

    result = {}
    for name, data in tendencies.items():
        if data["windows_appeared"] == 0 and not data["gk_distribution_zones"]:
            continue

        entry = {
            "team":             data["team"],
            "side":             data.get("side", ""),  # Fix 55: home/away
            "pos":              data["pos"],
            "windows_appeared": data["windows_appeared"],
        }

        for field in ("run_type", "positioning", "defensive_contribution"):
            counter = data[field]
            if counter:
                dom, count, conf = _dominant(counter)
                entry[f"dominant_{field}"]   = dom
                entry[f"{field}_count"]      = count
                entry[f"{field}_confidence"] = conf
                entry[f"{field}_distribution"] = dict(counter.most_common(5))

        if data["gk_distribution_zones"]:
            dom_zone, zone_count, zone_conf = _dominant(
                data["gk_distribution_zones"], threshold=2)
            dom_len,  len_count,  len_conf  = _dominant(
                data["gk_distribution_lengths"], threshold=2)
            entry["gk_dominant_zone"]      = dom_zone
            entry["gk_dominant_length"]    = dom_len
            entry["gk_zone_confidence"]    = zone_conf
            entry["gk_length_confidence"]  = len_conf
            entry["gk_zone_distribution"]  = dict(
                data["gk_distribution_zones"].most_common(5))
            entry["gk_length_distribution"] = dict(
                data["gk_distribution_lengths"].most_common(3))
            entry["gk_total_kicks"]        = sum(data["gk_distribution_zones"].values())

        result[name] = entry

    return {
        "available":            bool(result),
        "player_count":         len(result),
        "pattern_threshold":    PATTERN_THRESHOLD,
        "consistent_threshold": CONSISTENT_THRESHOLD,
        "players":              result,
    }


def accumulate_all_windows(match_dir: str) -> dict:
    """
    Process all merged window JSONs in agent_logs/.
    Updates both pass_sequences.json and running_summary.json.
    Run once after all merges are complete (or incrementally after each merge).
    """
    logs_dir = os.path.join(match_dir, "agent_logs")
    merged_files = sorted(glob.glob(os.path.join(logs_dir, "*_merged.json")))

    if not merged_files:
        print(f"  No merged files found in {logs_dir}")
        return {"windows_processed": 0}

    pass_sequences_path = os.path.join(match_dir, "pass_sequences.json")
    summary_path        = os.path.join(match_dir, "running_summary.json")

    # Fix 33b: load home/away from match config.
    config_path = os.path.join(match_dir, "match_config.json")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        home_team = config.get("home_team", "")
        away_team = config.get("away_team", "")
    else:
        config    = {}
        home_team = ""
        away_team = ""

    # Always reset summary and pass_sequences before accumulating all windows
    # (function processes ALL merged files each call, so must build fresh)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version({
            "match":                  config.get("match", "") if os.path.exists(config_path) else "",
            "home_team":              home_team,
            "away_team":              away_team,
            "windows_complete":       0,
            "formation_history":      [],
            "pressing_by_window":     [],
            "line_height_by_window":  [],
            "line_height_m_by_window":[],
            "match_state_by_window":  [],
            "shots_for":              [],
            "shots_against":          [],
            "flagged_moments":        [],
            "key_moments":            [],
            "individual_observations":[],
            "set_pieces":             [],
            "transitions":            [],
            "gk_kicks":               [],
            "duels":                  [],
            "possession_by_window":   [],
            "data_gap_windows":       [],
            "findings":               [],
            "confirmation_queue":     [],
        }, "running_summary"), f, indent=2)
    with open(pass_sequences_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version({
            "match":           config.get("match", "") if os.path.exists(config_path) else "",
            "home_team":       home_team,
            "away_team":       away_team,
            "total_sequences": 0,
            "sequences":       [],
        }, "pass_sequences"), f, indent=2)

    print(f"\n{'-'*55}")
    print(f"  Accumulating {len(merged_files)} merged windows")
    print(f"{'-'*55}")

    total_sequences = 0
    for path in merged_files:
        n = accumulate_pass_sequences(path, pass_sequences_path)
        total_sequences += n
        update_running_summary(path, summary_path)

    # Calculate canonical formation via majority vote across structural windows.
    # Structural agents improvise schema; the walker handles path variants.
    mc = config if os.path.exists(config_path) else {}
    home_canonical = _majority_formation(logs_dir, "home_formation", mc)
    away_canonical = _majority_formation(logs_dir, "away_formation", mc)
    formation_dist = _formation_distribution(logs_dir, mc)
    ip_shape       = _ip_shape_summary(logs_dir)
    far_side_summary = _accumulate_far_side_shape(logs_dir, mc)

    # Fix 32e: press trigger and runner pattern aggregation.
    press_summary  = _accumulate_press_triggers(logs_dir, mc)
    runner_summary = _accumulate_runner_patterns(logs_dir, mc)

    # Fix 46A: per-player tendency aggregation (run patterns, positioning,
    # GK distribution direction). Drives the synthesis PLAYER TENDENCY,
    # GK DISTRIBUTION, and EXPLOITABLE PATTERN rules.
    player_tendencies = _accumulate_player_tendencies(logs_dir, mc)

    if press_summary.get('available'):
        print(f"  Press triggers : {press_summary['windows_with_data']} windows with data")
    else:
        print(f"  Press triggers : {press_summary.get('note', 'not available')}")
    if runner_summary.get('available'):
        print(f"  Runner patterns: {runner_summary['windows_with_data']} windows with data")
    else:
        print(f"  Runner patterns: not available -- {runner_summary.get('note', '')[:60]}")
    if player_tendencies.get('available'):
        print(f"  Player tendencies: {player_tendencies['player_count']} players profiled")
    else:
        print(f"  Player tendencies: not available (positional IDs only or no observations)")

    if (home_canonical or away_canonical or far_side_summary
            or press_summary or runner_summary):
        with open(summary_path, encoding="utf-8") as f:
            summary_for_update = json.load(f)
        if home_canonical or away_canonical:
            # Count formation_basis values across all structural windows
            # to surface how many were confirmed vs. unverified vs. overridden.
            _basis_counts = {"home": {}, "away": {}}
            for _fp in glob.glob(os.path.join(logs_dir, "*structural*.json")):
                try:
                    with open(_fp, encoding="utf-8") as _f:
                        _d = json.load(_f)
                    for _side in ("home", "away"):
                        _b = _d.get("formation", {}).get(f"{_side}_formation_basis")
                        if _b:
                            _basis_counts[_side][_b] = (
                                _basis_counts[_side].get(_b, 0) + 1)
                except Exception:
                    pass
            summary_for_update["canonical_formation"] = {
                "home":         home_canonical,
                "away":         away_canonical,
                "method":       "majority_vote_across_windows",
                "basis_counts": _basis_counts,
            }
        # Per-side distribution lets synthesis apply the FORMATION RULE's
        # variation narrative when the secondary formation is significant.
        # ip_shape_summary captures variation text excluded from distribution.
        summary_for_update["formation_distribution"]  = formation_dist
        summary_for_update["ip_shape_summary"]        = ip_shape
        summary_for_update["far_side_shape_summary"]  = far_side_summary
        summary_for_update["press_trigger_summary"]   = press_summary
        summary_for_update["runner_pattern_summary"]  = runner_summary
        summary_for_update["player_tendencies"]       = player_tendencies
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(stamp_schema_version(summary_for_update, "running_summary"), f, indent=2)
        print(f"  Canonical formations: home={home_canonical}, away={away_canonical}")
        if far_side_summary.get("available"):
            print(f"  Far-side shape: {far_side_summary['windows_with_data']} windows with data")
        else:
            print(f"  Far-side shape: not available for this source type")

    # Final counts
    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(pass_sequences_path, encoding="utf-8") as f:
        ps = json.load(f)
    ps.setdefault("total_sequences", len(ps.get("sequences", [])))

    print(f"  Windows complete:   {summary['windows_complete']}")
    print(f"  Pass sequences:     {ps['total_sequences']}")
    print(f"  Key moments:        {len(summary.get('key_moments', []))}")
    print(f"  Flagged moments:    {len(summary.get('flagged_moments', []))}")
    print(f"  Shots for/against:  {len(summary.get('shots_for', []))} / "
          f"{len(summary.get('shots_against', []))}")
    print(f"  Data gap windows:   {summary.get('data_gap_windows', [])}")
    print(f"{'-'*55}")

    return {
        "windows_processed": len(merged_files),
        "total_sequences":   total_sequences,
        "windows_complete":  summary["windows_complete"],
    }


# ── Addition to accumulator.py ─────────────────────────────────────────────
# Place this function after accumulate_all_windows()
# Called from pipeline_runner_v2.py as step 3f_shots (after 3e merge)
# No changes to existing functions required.


def build_shots_log(match_dir: str) -> dict:
    """
    Aggregate goal shot data from match_config.json and event agent outputs.
    Writes shots_log.json to match_dir.
    Returns the shots log dict.

    Data sources (in priority order):
      1. match_config.json  -- confirmed goal events (scorer, minute, team)
      2. event agent output -- shot detail (origin zone, foot, target zone,
                               build-up chain, defensive shape)
      3. set piece agent    -- if the goal came from a set piece

    Goals without event agent output are still recorded from match facts
    with evidence_grade C and source 'match_facts_only'.
    """
    from pathlib import Path

    mc_path = os.path.join(match_dir, "match_config.json")
    wp_path = os.path.join(match_dir, "window_plan.json")

    if not os.path.exists(mc_path):
        print("  shots_log: match_config.json not found -- skipping")
        return {}

    with open(mc_path, encoding="utf-8") as f:
        mc = json.load(f)
    with open(wp_path, encoding="utf-8") as f:
        wp = json.load(f)

    windows   = wp.get("windows", [])
    match_id  = get_match_id(match_dir, mc)
    if not mc.get("match"):
        print(f"  [!] WARNING: match_config.json missing 'match' key -- using dirname '{match_id}'", file=sys.stderr)
    home_team = mc.get("home_team", "home")
    away_team = mc.get("away_team", "away")

    # ── Helper: find which window contains a given minute ─────────────────
    def _window_for_minute(minute: int) -> dict:
        for w in windows:
            start = w.get("start_s", w.get("start_seconds", 0)) / 60
            end   = w.get("end_s",   w.get("end_seconds",   0)) / 60
            if start <= minute <= end:
                return w
        return {}

    # ── Helper: load event agent output for a window ──────────────────────
    def _load_event_output(window_id: str) -> dict:
        logs_dir = os.path.join(match_dir, "agent_logs")
        patterns = [
            os.path.join(logs_dir, f"*{window_id}*event*.json"),
            os.path.join(logs_dir, f"agent_{window_id}_event.json"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], encoding="utf-8") as f:
                    return json.load(f)
        return {}

    # ── Helper: load set piece output for a window ────────────────────────
    def _load_setpiece_output(window_id: str) -> dict:
        logs_dir = os.path.join(match_dir, "agent_logs")
        patterns = [
            os.path.join(logs_dir, f"*{window_id}*setpiece*.json"),
            os.path.join(logs_dir, f"agent_{window_id}_setpiece.json"),
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                with open(matches[0], encoding="utf-8") as f:
                    return json.load(f)
        return {}

    # ── Helper: resolve team name from match_config event format ──────────
    def _team_name(ev: dict) -> str:
        team = ev.get("team", {})
        if isinstance(team, dict):
            return team.get("name", "unknown")
        return str(team)

    def _team_side(team_name: str) -> str:
        if team_name == home_team:
            return "home"
        if team_name == away_team:
            return "away"
        return "unknown"

    def _player_name(ev: dict) -> str:
        player = ev.get("player", {})
        if isinstance(player, dict):
            return player.get("name", "unknown")
        return str(player)

    def _player_number(ev: dict, team_side: str) -> int | None:
        """Try to find shirt number from lineups."""
        pname = _player_name(ev)
        target = home_team if team_side == "home" else away_team
        for lineup in mc.get("lineups", []):
            t = lineup.get("team", "")
            lineup_team = t.get("name", "") if isinstance(t, dict) else str(t)
            if lineup_team != target:
                continue
            for p in lineup.get("startXI", []) + lineup.get("substitutes", []):
                pdata = p.get("player", p)
                if isinstance(pdata, dict) and pdata.get("name") == pname:
                    return pdata.get("number")
        return None

    def _minute(ev: dict) -> int:
        t = ev.get("time", {})
        if isinstance(t, dict):
            return t.get("elapsed", ev.get("minute", 0))
        return ev.get("minute", 0)

    def _score_at_minute(minute: int, goals: list) -> str:
        h, a = 0, 0
        for g in goals:
            if _minute(g) <= minute:
                if _team_side(_team_name(g)) == "home":
                    h += 1
                else:
                    a += 1
        return f"{h}-{a}"

    def _match_state(team_side: str, score: str) -> str:
        h, a = map(int, score.split("-"))
        if team_side == "home":
            diff = h - a
        else:
            diff = a - h
        if diff > 0:
            return "winning"
        if diff < 0:
            return "losing"
        return "level"

    # ── Build shots log ───────────────────────────────────────────────────
    shots = []
    goals = mc.get("goals", [])

    for ev in goals:
        minute    = _minute(ev)
        team_name = _team_name(ev)
        team_side = _team_side(team_name)
        score     = _score_at_minute(minute - 1, goals)  # score before this goal
        state     = _match_state(team_side, score)
        player    = _player_name(ev)
        number    = _player_number(ev, team_side)

        # Base shot record from match facts
        shot = {
            "match_id":        match_id,
            "competition":     mc.get("competition", "unknown"),
            "match_date":      mc.get("date", "unknown"),
            "home_team":       home_team,
            "away_team":       away_team,
            "team_side":       team_side,
            "team_name":       team_name,
            "player":          player,
            "shirt_number":    number,
            "minute":          minute,
            "match_state":     state,
            "score_at_shot":   score,

            # Shot detail fields -- populated from event agent below
            "origin_zone":     None,
            "shot_foot":       None,
            "target_zone":     None,
            "outcome":         "goal",   # confirmed from match facts
            "shot_quality":    None,
            "set_piece":       False,
            "set_piece_type":  None,
            "build_up_length": None,
            "build_up_chain":  None,

            # Provenance
            "evidence_grade":  "C",      # upgraded to A/B if event agent confirms
            "source":          "match_facts_only",
        }

        # ── Try to enrich from event agent output ─────────────────────────
        window     = _window_for_minute(minute)
        window_id  = window.get("window_id", window.get("agent_id", ""))

        if window_id:
            ev_output = _load_event_output(window_id)

            if ev_output and ev_output.get("events"):
                for ev_detail in ev_output["events"]:
                    if ev_detail.get("type") != "goal":
                        continue
                    ev_min = ev_detail.get("minute", ev_detail.get("timestamp", ""))
                    try:
                        ev_min_int = int(str(ev_min).replace("m", "").split("m")[0])
                    except (ValueError, AttributeError):
                        ev_min_int = minute

                    if abs(ev_min_int - minute) <= 2:
                        shot["origin_zone"]     = ev_detail.get("shot_origin_zone")
                        shot["shot_foot"]       = ev_detail.get("shot_foot")
                        shot["target_zone"]     = ev_detail.get("target_zone")
                        shot["shot_quality"]    = ev_detail.get("shot_quality")
                        shot["evidence_grade"]  = "A"
                        shot["source"]          = "event_agent"

                        chain = ev_detail.get("build_up_sequence", "")
                        if chain:
                            shot["build_up_chain"]  = chain
                            shot["build_up_length"] = len(
                                [x for x in chain.split("->") if x.strip()]
                            )
                        break

            # ── Try to enrich from set piece agent ────────────────────────
            sp_output = _load_setpiece_output(window_id)
            if sp_output and sp_output.get("set_piece_agent"):
                sp_min = sp_output.get("timestamp", "")
                try:
                    sp_min_int = int(str(sp_min).replace("m", "").split("m")[0])
                except (ValueError, AttributeError):
                    sp_min_int = minute

                if abs(sp_min_int - minute) <= 2 and sp_output.get("outcome") == "goal":
                    shot["set_piece"]     = True
                    shot["set_piece_type"] = sp_output.get("type", "corner")
                    if shot["evidence_grade"] != "A":
                        shot["evidence_grade"] = "B"

        shots.append(shot)

    # ── Write shots_log.json ──────────────────────────────────────────────
    shots_log = {
        "match_id":     match_id,
        "match_date":   mc.get("date", "unknown"),
        "home_team":    home_team,
        "away_team":    away_team,
        "total_goals":  len(shots),
        "goals":        shots,
        "data_note":    (
            "Goals confirmed from match facts. Shot detail (origin_zone, "
            "shot_foot, target_zone) from event agent where available. "
            "evidence_grade A = event agent confirmed. "
            "evidence_grade C = match facts only, no event agent output "
            "for this window."
        )
    }

    out_path = os.path.join(match_dir, "shots_log.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(shots_log, "shots_log"), f, indent=2, ensure_ascii=False)

    confirmed  = sum(1 for s in shots if s["evidence_grade"] == "A")
    facts_only = sum(1 for s in shots if s["evidence_grade"] == "C")
    print(f"  shots_log: {len(shots)} goals written "
          f"({confirmed} event agent confirmed, {facts_only} match facts only)")

    return shots_log


# ── Season aggregator ──────────────────────────────────────────────────────
# Run separately against a directory of match directories to combine
# all shots_log.json files into a single dataset for visualisation.

def aggregate_shots(matches_root: str, output_path: str = None) -> list:
    """
    Reads all shots_log.json files from subdirectories of matches_root.
    Returns a flat list of shot records suitable for pandas or JSON export.
    Writes to output_path if provided.
    """
    from pathlib import Path as _Path
    all_shots = []
    for match_dir in sorted(_Path(matches_root).iterdir()):
        log_path = match_dir / "shots_log.json"
        if log_path.exists():
            with open(log_path, encoding="utf-8") as f:
                log = json.load(f)
            all_shots.extend(log.get("goals", []))

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            # F12 exclusion: writes a top-level JSON list (not a dict), so
            # stamp_schema_version cannot apply without changing the output
            # contract. See pipeline_schemas.py "Exclusions" section.
            json.dump(all_shots, f, indent=2, ensure_ascii=False)
        print(f"Aggregated {len(all_shots)} goals from "
              f"{matches_root} -> {output_path}")

    return all_shots


if __name__ == "__main__":
    match_dir = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    accumulate_all_windows(match_dir)
