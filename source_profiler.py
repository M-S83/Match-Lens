"""
source_profiler.py -- Match Lens Step 1f

Samples frames from the match video, classifies the footage source type,
scores visibility dimensions, and produces:
  - source_profile.json    (source type + visibility scores)
  - result_family_gates.json  (per-family gate states for this job)

The classification uses an LLM agent to inspect sample frames.
If classification_confidence is below MIN_CONFIDENCE, source_type defaults
to 'unknown' and all structural result families are downgraded.

Usage:
    python source_profiler.py [MATCH_DIR] [VIDEO_FILE]
"""

import json
import os
import sys
import cv2
from datetime import datetime


MIN_CONFIDENCE          = 0.6   # below this -> default to unknown
SAMPLE_INTERVAL_SECONDS = 60    # one frame per minute for classification
MAX_SAMPLES             = 10    # cap to avoid token overrun


VALID_SOURCE_TYPES = [
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

ALL_RESULT_FAMILIES = [
    "shape", "spacing", "territory", "phase", "pressing", "build_up",
    "rest_defence", "transitions", "chance_patterns", "set_pieces",
    "local_duels", "line_breaking_actions", "box_entries",
    "opposition_identity", "opposition_structure", "opposition_patterns",
    "opposition_build_up", "opposition_pressing", "opposition_transitions",
    "opposition_strengths", "opposition_weaknesses",
    "player_role", "player_positioning", "player_movement",
    "player_decision_making", "player_duels", "player_technical_actions",
    "player_line_breaking_contributions",
]


# -- Frame sampling ------------------------------------------------------------

def sample_frames(video_path: str, out_dir: str, interval_s: int = SAMPLE_INTERVAL_SECONDS) -> list:
    """Extract one frame per interval_s from the video for classification."""
    os.makedirs(out_dir, exist_ok=True)
    cap   = cv2.VideoCapture(video_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    dur   = total / src_fps if src_fps > 0 else 0

    samples = []
    sec = 0
    while sec < dur and len(samples) < MAX_SAMPLES:
        frame_idx = int(sec * src_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            m, s = divmod(int(sec), 60)
            fname = os.path.join(out_dir, f"source_sample_{m:02d}m{s:02d}s.jpg")
            cv2.imwrite(fname, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            samples.append(fname)
        sec += interval_s

    cap.release()
    print(f"  Sampled {len(samples)} frames for source classification")
    return samples


# -- Classification prompt -----------------------------------------------------

CLASSIFICATION_PROMPT = """
You are classifying football match footage to determine the camera source type
and visibility characteristics. Examine the sample frames provided.

Classify the source type as exactly ONE of:
  tactical_wide_static         -- fixed wide-angle covering full/most of pitch
  tactical_wide_auto_tracking  -- wide angle but auto-tracking, crop may tighten
  veo_ball_tracking            -- Veo or similar system following the ball closely
  drone_high_wide              -- high-altitude drone with wide pitch coverage
  drone_follow                 -- lower drone following action
  dual_panoramic               -- two stacked half-pitch views (top + bottom)
  behind_goal                  -- fixed camera behind/near one goal
  broadcast_tv                 -- television broadcast feed
  tracking_overlay             -- tracking data overlay on video
  unknown                      -- cannot determine

Then score each visibility dimension from 0.0 (none) to 1.0 (excellent):
  full_pitch_visibility_score    -- how much of the pitch is visible in typical frames
  weakside_visibility_score      -- how often the far/weak side is visible
  off_ball_coverage_score        -- how much off-ball player behaviour is visible
  camera_motion_score            -- degree of camera motion (0=static, 1=constant motion)
  zoom_variability_score         -- how much zoom changes across frames (0=constant, 1=frequent)
  stability_score                -- overall frame stability (0=unstable, 1=perfectly stable)
  orientation_consistency_score  -- how consistent the pitch orientation is (0=inconsistent, 1=always consistent)
  occlusion_score                -- degree of player/pitch occlusion (0=none, 1=severe)
  ball_follow_bias               -- how strongly camera follows the ball (0=none, 1=always follows)

For dual_panoramic footage, set split_aware: true.

Output ONLY raw JSON -- no preamble, no markdown:

{
  "source_type": "...",
  "classification_confidence": 0.0,
  "split_aware": false,
  "visibility_scores": {
    "full_pitch_visibility_score": 0.0,
    "weakside_visibility_score": 0.0,
    "off_ball_coverage_score": 0.0,
    "camera_motion_score": 0.0,
    "zoom_variability_score": 0.0,
    "stability_score": 0.0,
    "orientation_consistency_score": 0.0,
    "occlusion_score": 0.0,
    "ball_follow_bias": 0.0
  },
  "source_limitations_note": "[one sentence describing the main limitation of this footage type]"
}
"""


# -- Result family gating ------------------------------------------------------

def build_gates(source_type: str, visibility_scores: dict,
                split_aware: bool, config: dict) -> dict:
    """
    Apply base rules + visibility override rules to produce per-family gate states.
    Returns dict of {family: "allowed"|"downgraded"|"suppressed"}.
    """
    if source_type not in config:
        source_type = "unknown"

    profile = config[source_type]
    gates   = dict(profile["result_family_rules"])  # copy base rules

    # Apply visibility override rules
    for rule_key, overrides in profile.get("visibility_override_rules", {}).items():
        triggered = False

        if rule_key.startswith("if_") and "_below_" in rule_key:
            _, rest  = rule_key.split("if_", 1)
            dim, thr = rest.rsplit("_below_", 1)
            score    = visibility_scores.get(dim, 1.0)
            triggered = float(score) < float(thr)

        elif rule_key.startswith("if_") and "_above_" in rule_key:
            _, rest  = rule_key.split("if_", 1)
            dim, thr = rest.rsplit("_above_", 1)
            score    = visibility_scores.get(dim, 0.0)
            triggered = float(score) > float(thr)

        elif rule_key == "if_split_confirmed_false":
            triggered = not split_aware

        if triggered:
            for family, new_state in overrides.items():
                current = gates.get(family, "allowed")
                # Only downgrade, never upgrade via visibility rules
                order = {"allowed": 0, "downgraded": 1, "suppressed": 2}
                if order.get(new_state, 0) > order.get(current, 0):
                    gates[family] = new_state

    # Ensure all families are present
    for family in ALL_RESULT_FAMILIES:
        if family not in gates:
            gates[family] = "downgraded"  # conservative default

    return gates


def compute_visibility_limitations(gates: dict, visibility_scores: dict) -> list:
    """Produce human-readable limitation notes based on gate states and scores."""
    notes = []

    suppressed = [f for f, s in gates.items() if s == "suppressed"]
    downgraded = [f for f, s in gates.items() if s == "downgraded"]

    if suppressed:
        notes.append(f"Suppressed result families: {', '.join(suppressed)}")
    if downgraded:
        notes.append(f"Downgraded result families: {', '.join(downgraded)}")

    vs = visibility_scores
    if vs.get("full_pitch_visibility_score", 1.0) < 0.4:
        notes.append("Low full-pitch visibility -- whole-team structural claims not supported")
    if vs.get("weakside_visibility_score", 1.0) < 0.3:
        notes.append("Low weak-side visibility -- weak-side occupation and spacing not reliably observable")
    if vs.get("ball_follow_bias", 0.0) > 0.7:
        notes.append("High ball-follow bias -- near-ball findings only; full-team structure not supported")
    if vs.get("stability_score", 1.0) < 0.4:
        notes.append("Low stability -- camera motion may reduce structural claim reliability")

    return notes


# -- Main build function -------------------------------------------------------

def build_source_profile(match_dir: str, classification_result: dict,
                         config_path: str = None) -> bool:
    """
    Takes the LLM classification result dict, applies gates, writes both files.
    Returns True if source type was confidently classified.
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "source_profiles_config.json"
        )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    # Remove metadata keys
    source_config = {k: v for k, v in config.items() if not k.startswith("_")}

    source_type   = classification_result.get("source_type", "unknown")
    confidence    = classification_result.get("classification_confidence", 0.0)
    split_aware   = classification_result.get("split_aware", False)
    vis_scores    = classification_result.get("visibility_scores", {})
    source_note   = classification_result.get("source_limitations_note", "")

    # Fall back to unknown if confidence is too low
    confident = True
    if source_type not in VALID_SOURCE_TYPES or confidence < MIN_CONFIDENCE:
        print(f"  [!]  Classification confidence {confidence:.2f} below threshold {MIN_CONFIDENCE}")
        print(f"     Defaulting to 'unknown' -- all structural families will be downgraded")
        source_type = "unknown"
        confident = False

    # Build gates
    gates = build_gates(source_type, vis_scores, split_aware, source_config)
    visibility_limitations = compute_visibility_limitations(gates, vis_scores)

    # Write source_profile.json
    profile = {
        "source_type":              source_type,
        "classification_confidence": confidence,
        "split_aware":               split_aware,
        "visibility_scores":         vis_scores,
        "source_limitations_note":   source_note,
        "visibility_based_limitations": visibility_limitations,
        "frames_sampled":            classification_result.get("frames_sampled", []),
        "config_version":            config.get("_version", "unknown"),
        "generated_at":              datetime.now().isoformat(),
    }

    profile_path = os.path.join(match_dir, "source_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    # Write result_family_gates.json
    gates_doc = {
        "source_type":                source_type,
        "analysis_scope":             "all",
        "gates":                      gates,
        "visibility_based_limitations": visibility_limitations,
        "generated_at":               datetime.now().isoformat(),
    }

    gates_path = os.path.join(match_dir, "result_family_gates.json")
    with open(gates_path, "w", encoding="utf-8") as f:
        json.dump(gates_doc, f, indent=2)

    suppressed = [f for f, s in gates.items() if s == "suppressed"]
    downgraded = [f for f, s in gates.items() if s == "downgraded"]

    print(f"\n  Source type:    {source_type} (confidence: {confidence:.2f})")
    print(f"  Split-aware:    {split_aware}")
    # All families are attempted -- downgraded = zone-limited or partial signal
    print(f"  Downgraded:     {len(downgraded)} families")
    if visibility_limitations:
        for note in visibility_limitations:
            print(f"  [!]  {note}")
    print(f"  source_profile.json      written")
    print(f"  result_family_gates.json written")

    return confident


def apply_gate_to_finding(finding: dict, gates: dict) -> dict:
    """
    Stamp result_family_status onto a finding dict.
    finding must have a 'result_family' field.
    """
    family = finding.get("result_family")
    if family:
        finding["result_family_status"] = gates.get(family, "downgraded")
    else:
        finding["result_family_status"] = "downgraded"
    return finding


if __name__ == "__main__":
    """
    CLI usage:
        python source_profiler.py [MATCH_DIR] [VIDEO_FILE]

    Extracts sample frames and prints the classification prompt.
    You then run the prompt manually (or via Claude Code) and call
    build_source_profile() with the returned JSON.
    """
    match_dir  = sys.argv[1] if len(sys.argv) > 1 else input("Match directory: ").strip()
    video_file = sys.argv[2] if len(sys.argv) > 2 else input("Video file path: ").strip()

    sample_dir = os.path.join(match_dir, "frames", "source_samples")
    samples    = sample_frames(video_file, sample_dir)

    print("\n" + "="*60)
    print("SOURCE CLASSIFICATION PROMPT")
    print("="*60)
    print(CLASSIFICATION_PROMPT)
    print("\nFrames to include:")
    for s in samples:
        print(f"  {s}")
    print("\nRun this prompt with the above frames.")
    print("Then call build_source_profile(match_dir, result_json) with the output.")
