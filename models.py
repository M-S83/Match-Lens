from pydantic import BaseModel, Field
from typing import Literal

# This line says: "the ONLY acceptable values for source_type are these
# 10 exact words -- anything else is rejected." Literal is the Python
# word for "must be exactly one of this fixed list."
SourceType = Literal[
    "tactical_wide_static",
    "tactical_wide_auto_tracking",
    "veo_ball_tracking",
    "drone_high_wide",
    "drone_follow",
    "dual_panoramic",
    "behind_goal",
    "broadcast_tv",
    "tracking_overlay",
    "unknown",
]


# WHY a separate class for this instead of 9 loose variables: these 9
# scores are always used together (build_gates() reads them as one unit).
# Grouping them means any function that needs "the visibility picture"
# takes one argument, not 9, and if a 10th score gets added later, only
# this one class changes -- nothing else needs to know.
class VisibilityScores(BaseModel):
    # WHY Field(..., ge=0.0, le=1.0) instead of plain `float`: the prompt
    # asks the LLM for a score from 0.0 to 1.0, but nothing currently
    # stops it returning 1.4 or -0.2. `ge`/`le` make that impossible --
    # Pydantic rejects it the instant the object is created, not several
    # steps later when a report shows a nonsensical number.
    full_pitch_visibility_score: float = Field(..., ge=0.0, le=1.0)
    weakside_visibility_score: float = Field(..., ge=0.0, le=1.0)
    off_ball_coverage_score: float = Field(..., ge=0.0, le=1.0)
    camera_motion_score: float = Field(..., ge=0.0, le=1.0)
    zoom_variability_score: float = Field(..., ge=0.0, le=1.0)
    stability_score: float = Field(..., ge=0.0, le=1.0)
    orientation_consistency_score: float = Field(..., ge=0.0, le=1.0)
    occlusion_score: float = Field(..., ge=0.0, le=1.0)
    ball_follow_bias: float = Field(..., ge=0.0, le=1.0)


# The full profile. Notice visibility_scores is typed as VisibilityScores
# (the class above), not `dict`. That's the key idea of "nesting": Pydantic
# automatically validates every one of the 9 sub-fields too, just by
# putting the class name as the type -- you don't have to check them
# separately.
class SourceProfile(BaseModel):
    source_type: SourceType
    classification_confidence: float = Field(..., ge=0.0, le=1.0)
    split_aware: bool
    visibility_scores: VisibilityScores
    # Always present in a well-formed response from source_profiler.py's
    # prompt (it's asked for on every call, not conditionally) -- so we
    # type it as a plain required `str`, not `str | None`.
    source_limitations_note: str


# --- SetPiece -----------------------------------------------------------
# WHY this schema looks different from SourceProfile: a SetPiece record
# is built progressively. A 1fps first pass captures the basics; if the
# window later gets escalated to a slower 5fps "burst" re-watch, fields
# like runners/wall_size/marking_system get filled in *afterwards*. So a
# brand-new record legitimately has many empty fields -- that's not bad
# data, it's an earlier stage of the same record. `Optional[...] = None`
# is Pydantic's way of saying "allowed to be missing, defaults to None"
# -- different from SourceProfile, where every field was always required.
from typing import Optional, List
from pydantic import field_validator

# Kept exactly as your code has it -- including the 3 legacy values
# ("corner", "free_kick", "throw_in"). WHY keep values you consider
# outdated in the allowed list: old files already on disk were written
# with these values. If we excluded them, loading historical data through
# this model would crash. The comment in accumulator.py itself says
# "legacy compatibility -- accept but log" -- we're doing the same thing,
# just with Pydantic instead of an if-statement.
SetPieceType = Literal[
    "corner_left", "corner_right",
    "direct_fk", "indirect_fk",
    "throw_final_third", "kickoff",
    "corner", "free_kick", "throw_in",
]

Team = Literal["home_kit", "away_kit"]


class SetPiece(BaseModel):
    # Required from the very first pass -- a set piece can't exist
    # without knowing when it happened, what it was, and whose it was.
    timestamp: str
    type: SetPieceType
    team: Team
    window: str

    timestamp_inferred: bool = False
    delivery_observed: bool = True

    # WHY marking_system only, with no "marking" fallback field at all:
    # this is the actual fix for F4. There is now exactly one name this
    # value can be written under. get_marking()'s fallback logic in
    # pipeline_accessors.py becomes unnecessary for anything new -- it's
    # only still needed to read OLD files written before this model existed.
    marking_system: Optional[str] = None

    # Everything below is genuinely optional -- only filled in once a
    # burst re-watch happens, or sometimes never, depending on the record.
    taker_position: Optional[str] = None
    delivery_zone: Optional[str] = None
    bodies_in_box: Optional[int] = None
    rest_defence: Optional[int] = None
    outcome: Optional[str] = None
    delivery_type: Optional[str] = None
    runners: List[str] = []
    wall_size: Optional[int] = None
    wall_position: Optional[str] = None
    delivery_target_zone: Optional[str] = None
    second_phase: Optional[str] = None
    corrections: List[str] = []

    # Progressive-enrichment tracking fields -- these tell you WHERE a
    # value came from, which matters for confidence: a burst-resolved
    # field is more reliable than an unresolved 1fps guess.
    source: str = "1fps_initial"
    burst_resolved: bool = False
    burst_fps: Optional[int] = None
    burst_notes: Optional[str] = None
    resolved_at: Optional[str] = None

    # WHY a custom validator instead of just `str`: Literal and Field(ge=..)
    # only check fixed values or numeric ranges. A *pattern* check needs
    # actual code, so Pydantic lets you write a function decorated with
    # @field_validator that runs automatically whenever this field is set.
    #
    # WHY regex instead of the "m" in v / "s" in v check that
    # accumulator.py currently uses: we tested that exact check with the
    # string "not-a-timestamp" and it INCORRECTLY passed, because the
    # word "timestamp" itself contains the letters m and s. A regex
    # (\d+m \d+s) instead requires actual digits followed by the literal
    # letters m and s in the right positions -- e.g. "23m 14s" matches,
    # "not-a-timestamp" does not. This is a real improvement discovered
    # by testing a deliberately-bad case, not a hypothetical one.
    @field_validator("timestamp")
    @classmethod
    def timestamp_must_look_like_a_timestamp(cls, v: str) -> str:
        import re
        if not v or not re.match(r"^\d+m \d+s$", v):
            raise ValueError(f"timestamp must look like '23m 14s', got {v!r}")
        return v
