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
