from models import Formation, TeamFormation
from pydantic import ValidationError

print("--- test 1: home and away each have DIFFERENT attacking vs defensive shapes ---")
print("(this is the whole point: home's shape doesn't have to equal its own")
print(" out-of-possession shape, and away isn't just 'whatever home isn't')")
f = Formation(
    home=TeamFormation(in_possession="4-3-3", out_of_possession="4-4-2", basis="player_positions"),
    away=TeamFormation(in_possession="3-4-3", out_of_possession="5-3-2", basis="inferred"),
)
print(f.model_dump_json(indent=2))

print("\n--- test 2: only out_of_possession known yet (in_possession not observed this window) ---")
partial = Formation(
    home=TeamFormation(out_of_possession="4-4-2"),
    away=TeamFormation(in_possession="4-2-3-1"),
)
print(partial.model_dump_json(indent=2))

print("\n--- test 3: formation that doesn't sum to 10 outfield players, should FAIL ---")
try:
    TeamFormation(in_possession="4-3-2")  # 4+3+2 = 9, not 10
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)

print("\n--- test 4: non-numeric formation string, should FAIL ---")
try:
    TeamFormation(out_of_possession="four-four-two")
    print("PROBLEM: this should NOT have succeeded")
except ValidationError as e:
    print("Correctly rejected:")
    print(e)
