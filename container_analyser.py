"""
container_analyser.py — Match Lens Step 1a

Reads container structure via ffprobe to detect:
- timestamp discontinuities (segment boundaries)
- keyframe intervals (seek reliability)
- fragmented/edit-list containers (phone footage)
- remux requirements

Writes container_profile.json to match_dir.
Takes 2-5 seconds regardless of video length.

Usage:
    python container_analyser.py [MATCH_DIR] [VIDEO_PATH]
"""

import subprocess
import json
import os
import sys
from datetime import datetime
from pipeline_schemas import stamp_schema_version


def analyse_container(video_path: str) -> dict:
    """
    Read container structure directly via ffprobe.
    Returns a unified profile regardless of source type.
    """

    # ── 1. Format and stream metadata ────────────────────────────────────────
    fmt_cmd = [
        "ffprobe", "-v", "quiet",
        "-show_format",
        "-show_streams",
        "-of", "json",
        video_path
    ]
    fmt_result = subprocess.run(fmt_cmd, capture_output=True, text=True)

    if fmt_result.returncode != 0:
        return {
            "error": f"ffprobe failed: {fmt_result.stderr}",
            "video_path": video_path,
            "seek_reliable": False,
            "boundaries": []
        }

    fmt_data     = json.loads(fmt_result.stdout)
    fmt          = fmt_data.get("format", {})
    streams      = fmt_data.get("streams", [])
    video_stream = next(
        (s for s in streams if s.get("codec_type") == "video"), {}
    )

    # Check for fragmented container via major_brand and compatible_brands
    # iso5/iso6 = ISO Base Media v5/v6 (fragmented MP4 / fMP4)
    # dash/cmfc/msdh = DASH fragmented formats
    FRAG_BRANDS   = {"iso5","iso6","dash","cmfc","msdh","msix"}
    fmt_tags      = fmt.get("tags",{})
    major_brand   = fmt_tags.get("major_brand","").lower()
    compat_brands = fmt_tags.get("compatible_brands","").lower()
    raw_out       = fmt_result.stdout.lower()

    is_fmp4 = (
        major_brand in FRAG_BRANDS
        or any(b in compat_brands for b in FRAG_BRANDS)
        or "fmp4" in raw_out
    )
    has_edit_list = "elst" in raw_out

    # ── 2. Packet stream — timestamp discontinuities ──────────────────────────
    pkt_cmd = [
        "ffprobe", "-v", "quiet",
        "-select_streams", "v:0",
        "-show_packets",
        "-show_entries", "packet=pts_time,dts_time,duration_time,flags",
        "-of", "json",
        video_path
    ]
    pkt_result = subprocess.run(pkt_cmd, capture_output=True, text=True)

    packets = []
    if pkt_result.returncode == 0:
        pkt_data = json.loads(pkt_result.stdout)
        packets  = pkt_data.get("packets", [])

    discontinuities    = []
    keyframe_intervals = []
    prev_dts = prev_key_pts = None

    for pkt in packets:
        # Use DTS (decode timestamp) — always monotonically increasing.
        # PTS jumps around due to B-frame reordering and must NOT be used
        # for discontinuity detection (120-160ms alternating gaps/overlaps
        # are normal B-frame behaviour, not segment boundaries).
        raw_dts = pkt.get("dts_time")
        raw_pts = pkt.get("pts_time")
        raw_dur = pkt.get("duration_time")

        if raw_dts is None or raw_dts == "N/A":
            continue

        dts    = float(raw_dts)
        pts    = float(raw_pts) if (raw_pts and raw_pts != "N/A") else dts
        dur    = float(raw_dur) if (raw_dur and raw_dur != "N/A") else 0.0
        is_key = "K" in pkt.get("flags", "")

        if prev_dts is not None:
            gap_ms = (dts - prev_dts) * 1000 - (dur * 1000)

            # 500ms threshold: B-frame reorder is max ~2 frame durations
            # (~80ms at 25fps). Real segment joins produce gaps of hundreds
            # to thousands of ms. 500ms safely separates the two.
            if gap_ms > 500:
                discontinuities.append({
                    "timestamp_s":  round(dts, 3),
                    "gap_ms":       round(gap_ms, 1),
                    "type":         "gap",
                    "prev_dts_s":   round(prev_dts, 3),
                })
            elif gap_ms < -500:
                discontinuities.append({
                    "timestamp_s":  round(dts, 3),
                    "gap_ms":       round(gap_ms, 1),
                    "type":         "overlap",
                    "prev_dts_s":   round(prev_dts, 3),
                })

        if is_key:
            if prev_key_pts is not None:
                keyframe_intervals.append(round(pts - prev_key_pts, 3))
            prev_key_pts = pts

        prev_dts = dts

    # ── 3. Keyframe stats ─────────────────────────────────────────────────────
    avg_kfi = round(sum(keyframe_intervals) / len(keyframe_intervals), 2) \
              if keyframe_intervals else None
    max_kfi = round(max(keyframe_intervals), 2) if keyframe_intervals else None

    # ── 4. fps from declared r_frame_rate ─────────────────────────────────────
    fps_raw = video_stream.get("r_frame_rate", "0/1")
    try:
        num, den = map(int, fps_raw.split("/"))
        fps_declared = round(num / den, 3) if den else 0
    except Exception:
        fps_declared = 0

    # ── 5. Build profile ──────────────────────────────────────────────────────
    n_disc   = len(discontinuities)
    is_clean = n_disc == 0 and not has_edit_list
    remux_ok = not is_clean   # remux recommended for any non-clean container

    profile = {
        # File info
        "video_path":        video_path,
        "format_name":       fmt.get("format_name", "unknown"),
        "duration_s":        round(float(fmt.get("duration", 0)), 2),
        "file_size_mb":      round(int(fmt.get("size", 0)) / 1_048_576, 1),

        # Video stream
        "codec":             video_stream.get("codec_name", "unknown"),
        "width":             video_stream.get("width"),
        "height":            video_stream.get("height"),
        "fps_declared":      fps_declared,
        "total_frames_declared": int(video_stream.get("nb_frames", 0) or 0),
        "total_packets_read": len(packets),

        # Container structure
        "is_fragmented":     is_fmp4,
        "has_edit_list":     has_edit_list,

        # Seeking reliability
        "seek_reliable":          is_clean and not is_fmp4,
        "discontinuity_count":    n_disc,
        "discontinuities":        discontinuities,
        "boundary_timestamps_s":  [d["timestamp_s"] for d in discontinuities],

        # Keyframe structure
        "avg_keyframe_interval_s": avg_kfi,
        "max_keyframe_interval_s": max_kfi,
        "keyframe_count":          len(keyframe_intervals) + (1 if keyframe_intervals else 0),

        # Recommendations
        "remux_recommended":         remux_ok,
        "align_windows_to_boundaries": n_disc > 0,
        "higher_fps_extraction_safe":  is_clean and (max_kfi is None or max_kfi <= 4.0),

        # Interpretation
        "source_pattern":  _classify_pattern(discontinuities, has_edit_list,
                                               fps_declared, max_kfi, n_disc),
        "notes":           [],
        "generated_at":    datetime.now().isoformat(),
    }

    # ── 6. Human-readable notes ───────────────────────────────────────────────
    profile["notes"] = _build_notes(profile)

    return profile


def _classify_pattern(discontinuities, has_edit_list,
                       fps_declared, max_kfi, n_disc) -> str:
    """
    Classify the container pattern to help identify source type.
    Does NOT name a source — it describes what it sees.
    """
    if has_edit_list:
        return "fragmented_with_edit_list"   # phone-style

    if n_disc == 0 and max_kfi and max_kfi > 4.0:
        return "long_gop"                    # DJI / high-compression drone

    if n_disc == 0:
        return "clean_continuous"

    if n_disc > 0:
        # Check periodicity
        timestamps = [d["timestamp_s"] for d in discontinuities]
        if len(timestamps) >= 2:
            intervals = [
                round(timestamps[i+1] - timestamps[i], 1)
                for i in range(len(timestamps)-1)
            ]
            avg_interval = sum(intervals) / len(intervals)
            variance     = sum((x - avg_interval)**2 for x in intervals) / len(intervals)
            is_periodic  = variance < 5.0  # low variance = regular interval

            if is_periodic and 250 < avg_interval < 350:
                return "periodic_5min_segments"     # Veo-style
            elif is_periodic and 900 < avg_interval < 1100:
                return "periodic_15min_segments"    # GoPro chapter-style
            elif is_periodic:
                return f"periodic_segments_{round(avg_interval)}s"
        else:
            return "single_join"    # two files concatenated

    if max_kfi and max_kfi > 4.0:
        return "long_gop"           # DJI / aggressive compression / high-quality drone

    return "irregular_discontinuities"  # broadcast capture


def _build_notes(p: dict) -> list:
    notes = []
    pattern = p["source_pattern"]
    n_disc  = p["discontinuity_count"]

    if pattern == "clean_continuous":
        notes.append(
            "Clean container. No timestamp discontinuities. "
            "Seeking is reliable throughout."
        )

    elif pattern == "periodic_5min_segments":
        notes.append(
            f"Periodic timestamp discontinuities at approximately 5-minute "
            f"intervals ({n_disc} found at: "
            f"{[d['timestamp_s'] for d in p['discontinuities']]}s). "
            "Consistent with segmented panoramic recording (e.g. Veo). "
            "Align analysis windows to boundary_timestamps_s to avoid "
            "seek failures near joins."
        )

    elif pattern in ("periodic_15min_segments", "single_join"):
        notes.append(
            f"{n_disc} segment join(s) detected at: "
            f"{[d['timestamp_s'] for d in p['discontinuities']]}s. "
            "Consistent with joined recording files. "
            "Align windows to boundary_timestamps_s."
        )

    elif pattern == "fragmented_with_edit_list":
        notes.append(
            "Fragmented container with edit list detected. "
            "Consistent with phone/mobile recording. "
            "pts values may be offset. "
            "Verify timestamp accuracy after first 1fps extraction."
        )

    if p.get("is_fragmented") and pattern != "fragmented_with_edit_list":
        notes.append(
            "Fragmented MP4 container (fMP4) detected. "
            "Consistent with phone or mobile recording format. "
            "Random-access seeking may be unreliable. "
            "Remux with ffmpeg -i input.mp4 -c copy -movflags faststart output.mp4"
        )

    elif pattern == "long_gop":
        notes.append(
            f"Long GOP detected (max keyframe interval: "
            f"{p['max_keyframe_interval_s']}s). "
            "Higher-fps segment extraction may land on the wrong frame "
            "if the target is between keyframes. "
            "Remux with ffmpeg -c copy to fix keyframe alignment."
        )

    elif pattern == "irregular_discontinuities":
        notes.append(
            f"{n_disc} irregular timestamp discontinuity(ies) detected. "
            "Seeking may be unreliable at affected timestamps. "
            "Review discontinuities list and remux before higher-fps extraction."
        )

    if p["remux_recommended"]:
        notes.append(
            "Remux recommended before higher-fps extraction: "
            "ffmpeg -i input.mp4 -c copy -movflags faststart output.mp4"
        )

    max_kfi = p.get("max_keyframe_interval_s")
    if max_kfi and max_kfi > 2.0:
        notes.append(
            f"Max keyframe interval {max_kfi}s. For higher-fps segment extraction, "
            "use ffmpeg -ss [time] -i rather than OpenCV frame seek. "
            "OpenCV must decode from the nearest keyframe forward, which may "
            "land on wrong frames when the keyframe interval is long."
        )

    return notes


def run_step_1a(match_dir: str, video_path: str) -> dict:
    """
    Step 1a entry point. Analyses container and writes container_profile.json.
    """
    print(f"\n{'-'*55}")
    print(f"  Step 1a — Container Analysis")
    print(f"  File: {os.path.basename(video_path)}")
    print(f"{'-'*55}")

    profile  = analyse_container(video_path)

    out_path = os.path.join(match_dir, "container_profile.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(profile, "container_profile"), f, indent=2)

    # Print summary
    print(f"  Format:      {profile['format_name']}")
    print(f"  Duration:    {profile['duration_s']}s")
    print(f"  Resolution:  {profile['width']}x{profile['height']} @ {profile['fps_declared']}fps")
    print(f"  Codec:       {profile['codec']}")
    print(f"  Packets:     {profile['total_packets_read']}")
    print(f"  Keyframe interval: avg {profile['avg_keyframe_interval_s']}s / "
          f"max {profile['max_keyframe_interval_s']}s")
    print(f"")
    print(f"  Seek reliable:     {profile['seek_reliable']}")
    print(f"  Discontinuities:   {profile['discontinuity_count']}")
    print(f"  Source pattern:    {profile['source_pattern']}")
    print(f"  Remux recommended: {profile['remux_recommended']}")
    print(f"  Higher-fps safe:   {profile['higher_fps_extraction_safe']}")
    print(f"  Boundary timestamps: {profile['boundary_timestamps_s']}")
    print(f"")
    for note in profile["notes"]:
        print(f"  ℹ  {note}")
    print(f"{'-'*55}")

    return profile


if __name__ == "__main__":
    if len(sys.argv) == 3:
        result = run_step_1a(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        # Test mode — just analyse without writing
        profile = analyse_container(sys.argv[1])
        print(json.dumps(profile, indent=2))
    else:
        print("Usage: python container_analyser.py [MATCH_DIR] [VIDEO_PATH]")
        print("       python container_analyser.py [VIDEO_PATH]  (test mode)")
