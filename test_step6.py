from models import WindowResult, SourceProfile, VisibilityScores, Formation, TeamFormation, SetPiece
from pydantic import ValidationError

visibility = VisibilityScores(
    full_pitch_visibility_score=0.9, weakside_visibility_score=0.7,
    off_ball_coverage_score=0.8, camera_motion_score=0.2,
    zoom_variability_score=0.1, stability_score=0.95,
    orientation_consistency_score=0.9, occlusion_score=0.1,
    ball_follow_bias=0.3,
)

print("--- test 1: a fully-populated window, all three models nested together ---")
w = WindowResult(
    window_id="07",
    start_s=420.0,
    end_s=450.0,
    half="1H",
    event_window=True,
    source_profile=SourceProfile(
        source_type="tactical_wide_static",
        classification_confidence=0.95,
        split_aware=False,
        visibility_scores=visibility,
        source_limitations_note="Slight weak-side occlusion during counter-attacks.",
    ),
    formation=Formation(
        home=TeamFormation(in_possession="4-3-3", out_of_possession="4-4-2"),
        away=TeamFormation(in_possession="3-4-3", out_of_possession="5-3-2"),
    ),
    set_pieces=[
        SetPiece(timestamp="7m 02s", type="corner_left", team="home_kit", window="07"),
    ],
)
print(w.model_dump_json(indent=2))

print("\n--- test 2: a bare-minimum window (no structural data yet, no set pieces) ---")
bare = WindowResult(window_id="12", start_s=720.0, end_s=750.0, half="2H")
print(bare.model_dump_json(indent=2))

print("\n--- test 3: bad window_id format (not two-digit zero-padded), should FAIL ---")
try:
    WindowResult(window_id="7", start_s=0.0, end_s=30.0, half="1H")
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)

print("\n--- test 4: invalid half value, should FAIL ---")
try:
    WindowResult(window_id="07", start_s=0.0, end_s=30.0, half="halftime")
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)
