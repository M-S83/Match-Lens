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
