from models import SourceProfile
from pydantic import ValidationError

VALID_SCORES = {
    "full_pitch_visibility_score": 0.9,
    "weakside_visibility_score": 0.6,
    "off_ball_coverage_score": 0.7,
    "camera_motion_score": 0.1,
    "zoom_variability_score": 0.05,
    "stability_score": 0.95,
    "orientation_consistency_score": 1.0,
    "occlusion_score": 0.15,
    "ball_follow_bias": 0.2,
}

print("--- test 1: a fully valid SourceProfile, should succeed ---")
good = SourceProfile(
    source_type="tactical_wide_static",
    classification_confidence=0.87,
    split_aware=False,
    visibility_scores=VALID_SCORES,
    source_limitations_note="Static wide angle misses close-up duel detail.",
)
print(good.model_dump_json(indent=2))

print("\n--- test 2: invalid source_type + out-of-range score, should FAIL ---")
bad_scores = dict(VALID_SCORES)
bad_scores["full_pitch_visibility_score"] = 1.4  # invalid: over 1.0
try:
    bad = SourceProfile(
        source_type="drone_zoomed_in",  # not in the 10 valid values
        classification_confidence=0.5,
        split_aware=False,
        visibility_scores=bad_scores,
        source_limitations_note="test",
    )
    # WHY raise instead of print here -- see test_step2.py's comment on the
    # same pattern: a print-only "problem" branch is a fail-silent bug in
    # the test, not a fail-loud one.
    raise AssertionError(f"this should NOT have succeeded, but got: {bad}")
except ValidationError as e:
    print("Correctly rejected. Error message:")
    print(e)
