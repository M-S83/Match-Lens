from models import SetPiece
from pydantic import ValidationError

print("--- test 1: a minimal, first-pass (1fps) SetPiece, should succeed ---")
print("(most fields empty on purpose -- this is what a brand-new record looks like)")
minimal = SetPiece(
    timestamp="23m 14s",
    type="corner_left",
    team="home_kit",
    window="w014",
)
print(minimal.model_dump_json(indent=2))

print("\n--- test 2: a fully burst-resolved SetPiece, should succeed ---")
full = SetPiece(
    timestamp="23m 14s",
    type="corner_left",
    team="home_kit",
    window="w014",
    marking_system="zonal",
    taker_position="right_wing",
    delivery_zone="near_post",
    bodies_in_box=6,
    outcome="cleared",
    delivery_type="inswinger",
    runners=["CB1 near post", "ST1 far post"],
    source="5fps_burst",
    burst_resolved=True,
    burst_fps=5,
    resolved_at="2026-07-13T17:00:00",
)
print(full.model_dump_json(indent=2))

print("\n--- test 3: bad timestamp (no 'm'/'s'), should FAIL ---")
try:
    SetPiece(timestamp="not-a-timestamp", type="corner_left", team="home_kit", window="w014")
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)

print("\n--- test 4: invalid type AND invalid team together, should FAIL ---")
try:
    SetPiece(timestamp="23m 14s", type="penalty", team="referee", window="w014")
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)
