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
    print("PROBLEM: this should NOT have succeeded:", bad)
except ValidationError as e:
    print("Correctly rejected. Error message:")
    print(e)
