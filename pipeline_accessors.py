"""Canonical accessors for pipeline data structures.

Single source of truth for reading fields that have historical name drift.
Every reader in the pipeline should route through these accessors instead
of calling `.get("foo", .get("bar"))` inline. The point is to make field
aliases a one-line change in this file rather than a hunt across the
codebase.

Conventions
-----------
- Accessor names describe the concept (e.g. ``window_id``).
- The first key tried is whatever the live producer actually writes;
  fallbacks exist only to tolerate legacy or hand-edited files.
- Each accessor documents who writes the canonical key and why other
  aliases exist.

Adding a new accessor: keep it short, declarative, and pure. No I/O,
no logging, no side effects. If you find yourself wanting to log a
warning when the fallback fires, do it once at the call site that
matters, not inside the accessor.
"""


def get_window_id(w: dict) -> str:
    """Return the canonical window identifier.

    `window_plan.py` (line ~262) writes this as ``agent_id`` — a two-digit
    zero-padded sequence number, e.g. ``"07"``. It does NOT write a
    ``window_id`` key. The ``window_id`` fallback exists only for legacy
    window dicts or hand-edited plans; no live writer in the current
    pipeline produces it.

    Returns "" if neither key is present (caller must handle).
    """
    return w.get("agent_id") or w.get("window_id") or ""


def get_source_limitations_note(profile: dict) -> str:
    """Return the source limitations note from a source_profile dict.

    `source_profiler.py` writes this as ``source_limitations_note`` —
    a one-sentence description of the main limitation of the footage type.

    Three legacy aliases are tolerated for hand-edited or older profile
    files (``notes``, ``source_limitations``, ``limitations_note``);
    no live writer in the current pipeline produces them. Two scripts
    (generate_flagged_moments, generate_pass_network) historically read
    only the legacy aliases, so their reports shipped without the source
    caveat — see AUDIT.md F3.

    Returns "" if no limitation key is present.
    """
    return (profile.get("source_limitations_note")
            or profile.get("notes")
            or profile.get("source_limitations")
            or profile.get("limitations_note")
            or "")


def get_marking(record: dict):
    """Return the set piece marking value from a set piece record.

    Canonical key is ``marking_system`` — written by both
    ``setpiece_writeback.apply_burst_to_record`` (via BURST_CONFIRMS) and,
    after F4, by ``accumulator.validate_set_piece``.

    ``marking`` is the legacy alias written by older accumulator versions
    and by some 1fps structural agents. Tolerated as a fallback so
    historical running_summary.json files still read correctly.

    Returns None if neither key is present. Callers that need a categorical
    default should do ``get_marking(sp) or "unknown"`` — this keeps the
    distinction between "field absent" and "field present, value empty".

    See AUDIT.md F4.
    """
    return record.get("marking_system") or record.get("marking") or None


def get_formation_home(record: dict):
    """Return the home team's formation string from a structural agent dict.

    Canonical: ``formation.home`` (Fix 32a schema, written by current
    structural agents). Legacy fallback: ``formation.shape_in_possession``
    — pre-Fix-32a Veo runs wrote a single shape string here, usually
    describing the focus/home team's in-possession shape. This fallback
    is approximate on legacy single-side data.

    Returns None if neither key is present.
    """
    f = record.get("formation") or {}
    return f.get("home") or f.get("shape_in_possession") or None


def get_formation_away(record: dict):
    """Return the away team's formation string from a structural agent dict.

    Canonical: ``formation.away``. Legacy fallback:
    ``formation.shape_out_of_possession`` — pre-Fix-32a Veo runs sometimes
    wrote the opposition shape here. The mapping IP→home / OOP→away is
    approximate; legacy files often won't have OOP populated at all.

    Returns None if neither key is present.
    """
    f = record.get("formation") or {}
    return f.get("away") or f.get("shape_out_of_possession") or None


def get_window_start_seconds(w: dict) -> float:
    """Return the canonical window start time in seconds.

    `window_plan.py` writes this as ``start_s``. The legacy ``start_seconds``
    alias is tolerated for any historical window_plan.json files written
    under the old schema; no live writer in the current pipeline produces it.

    Returns 0.0 if neither key is present (caller must handle).
    """
    v = w.get("start_s")
    if v is None:
        v = w.get("start_seconds", 0.0)
    return float(v)


def get_window_end_seconds(w: dict) -> float:
    """Return the canonical window end time in seconds.

    `window_plan.py` writes this as ``end_s``. The legacy ``end_seconds``
    alias is tolerated for any historical window_plan.json files written
    under the old schema; no live writer in the current pipeline produces it.

    Returns 0.0 if neither key is present (caller must handle).
    """
    v = w.get("end_s")
    if v is None:
        v = w.get("end_seconds", 0.0)
    return float(v)
