from models import VisibilityScores
from pydantic import ValidationError

print("--- test 1: valid scores, should succeed ---")
good = VisibilityScores(
    full_pitch_visibility_score=0.9,
    weakside_visibility_score=0.6,
    off_ball_coverage_score=0.7,
    camera_motion_score=0.1,
    zoom_variability_score=0.05,
    stability_score=0.95,
    orientation_consistency_score=1.0,
    occlusion_score=0.15,
    ball_follow_bias=0.2,
)
print(good.model_dump_json(indent=2))

print("\n--- test 2: one score out of range (1.4), should FAIL ---")
try:
    bad = VisibilityScores(
        full_pitch_visibility_score=1.4,   # invalid: over 1.0
        weakside_visibility_score=0.5,
        off_ball_coverage_score=0.5,
        camera_motion_score=0.5,
        zoom_variability_score=0.5,
        stability_score=0.5,
        orientation_consistency_score=0.5,
        occlusion_score=0.5,
        ball_follow_bias=0.5,
    )
    # WHY this is `raise AssertionError(...)` now, not `print("PROBLEM: ...")`:
    # a `print()` here is a "fail silent" bug living inside the TEST ITSELF --
    # the exact thing Concept 1 in the glossary warns about. If a future
    # change accidentally loosened the `le=1.0` constraint on this field,
    # this branch would run, print a warning nobody is watching for, and
    # the script would still exit 0 -- a green test suite hiding a real
    # regression. Raising makes the test fail loudly and visibly instead,
    # which is the entire point of writing the test in the first place.
    raise AssertionError(f"this should NOT have succeeded, but got: {bad}")
except ValidationError as e:
    print("Correctly rejected. Error message:")
    print(e)
