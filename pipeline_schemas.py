"""Per-file schema versions for pipeline JSON outputs.

Every JSON file written by the pipeline carries a top-level
``schema_version`` field stamped by ``stamp_schema_version()``. The
registry below maps a short ``schema_name`` (one per distinct output
file type) to the current version string for that file's schema.

Versioning policy
-----------------
- Versions are independent per schema_name. Bumping ``running_summary``
  does not bump ``window_plan``.
- Start every schema at "1.0".
- Bump the minor version ("1.0" -> "1.1") for additive changes that
  remain backward-readable (new key added, new optional field).
- Bump the major version ("1.x" -> "2.0") for breaking changes (renamed
  key, removed key, changed value semantics, restructured nesting).
- When a major bump lands, the reader side gains a branch on
  ``schema_version`` to handle both shapes until the migration is
  complete.

Adding a new schema: add an entry to SCHEMA_VERSIONS, call
``stamp_schema_version(data, "new_name")`` at the writer, document the
shape in AUDIT.md's Schema Catalogue.

Exclusions
----------
Three live writes are deliberately not stamped:

1. JSONL files (one JSON object per line). The pipeline has one:
   ``agent_logs/{merged_base}_orphan_setpieces.json``, written by
   setpiece_writeback.py. Per-line stamping would be wasteful and
   there is no top-level object to stamp once.

2. ``accumulator.aggregate_shots`` season-level export. Writes a
   top-level JSON list (not a dict), so the canonical
   ``{"schema_version": ...}`` stamping pattern cannot apply without
   changing the output contract. Treated as an intentional exclusion;
   if this file ever needs versioning, wrap the list in a dict shape
   at that point and add a ``shots_aggregate`` registry entry.

3. shots_aggregate from the now-deleted ``shots_log_addition.py`` —
   removed in commit 54f3f4d. The accumulator.aggregate_shots
   exclusion above covers the surviving copy.
"""


SCHEMA_VERSIONS = {
    "match_config": "1.0",
    "match_config_draft": "1.0",
    "teamsheet_image_raw": "1.0",
    "container_profile": "1.0",
    "match_boundaries": "1.0",
    "source_profile": "1.0",
    "result_family_gates": "1.0",
    "window_plan": "1.0",
    "pipeline_state": "1.0",
    "jersey_number_map": "1.0",
    "cost_estimate": "1.0",
    "job_log": "1.0",
    "frame_metadata": "1.0",
    "agent_structural": "1.0",
    "agent_player": "1.0",
    "agent_event": "1.0",
    "agent_setpiece": "1.0",
    # v3 port Step 14: schema-version stamp for the player-action
    # confirmation agent responses (Phase 3b-player block in
    # pipeline_runner_v2.py). collect_results stamps the saved
    # response file as agent_3i_player_action via this name.
    "agent_3i_player_action": "1.0",
    "agent_merged": "1.0",
    "running_summary": "1.0",
    "pass_sequences": "1.0",
    "shots_log": "1.0",
    "confirmation_queue": "1.0",
    "ground_truth_check": "1.1",  # Step 19: added fact_only_context_available /
                                    # fact_only_no_context; "missed" now goals-only.

    "report_readiness": "1.0",
    "confidence_reliability_report": "1.0",
    "deep_skill_metrics": "1.1",  # Step 20: momentum_score's value shape
                                     # changed from one blended {avg,
                                     # by_window, peak, low} to {home:
                                     # {...}, away: {...}} -- two
                                     # independent per-team series.
}


def stamp_schema_version(data: dict, schema_name: str) -> dict:
    """Stamp ``data`` with the current schema version for ``schema_name``.

    Mutates ``data`` in place and also returns it, so the call site can
    chain naturally:

        json.dump(stamp_schema_version(result, "window_plan"), f)

    Raises ``KeyError`` if ``schema_name`` is not in the registry — fail
    loud at write time rather than silently produce an unstamped file.
    """
    data["schema_version"] = SCHEMA_VERSIONS[schema_name]
    return data
