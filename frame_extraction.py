"""
frame_extraction.py -- higher-fps frame extraction from source video

Used by Step 3d-SP (and future goal/event escalation) to extract a short
burst of frames at a target fps around an anchor timestamp. Distinct from
the 1fps extraction in Step 1, which extracts the whole match once.

Burst extractions go to a per-anchor temp folder so multiple bursts in the
same window do not collide. Frames are written at JPEG quality 90 (higher
than Step 1's 80) because spatial detail matters more at this stage --
runner positions, foot orientation, ball trajectory.

Usage:
    from frame_extraction import extract_segment

    frames = extract_segment(
        video_path     = r"[MATCH_DIR]/match.mp4",
        anchor_seconds = 1203.0,         # 20m03s
        out_dir        = r"[MATCH_DIR]/frames_burst/sp_20m03s",
        target_fps     = 5,
        padding_s      = 3,
    )
    # frames -> list of file paths, ordered chronologically
"""

import cv2
import os
import glob


def find_source_video(match_dir: str) -> str:
    """
    Locate the source video file in match_dir. Expects a single .mp4 at
    the top level. Returns the absolute path, or raises if missing/ambiguous.
    """
    candidates = sorted(glob.glob(os.path.join(match_dir, "*.mp4")))
    if not candidates:
        raise FileNotFoundError(f"No .mp4 found in {match_dir}")
    if len(candidates) > 1:
        # Multiple videos -- pick the largest (typically the full match) and
        # warn rather than failing. Common case: clip files alongside main video.
        candidates.sort(key=lambda p: os.path.getsize(p), reverse=True)
        print(f"  [WARN] Multiple .mp4 in {match_dir}; "
              f"using {os.path.basename(candidates[0])}")
    return candidates[0]


def extract_segment(video_path: str,
                    anchor_seconds: float,
                    out_dir: str,
                    target_fps: int = 5,
                    padding_s: float = 3.0,
                    jpeg_quality: int = 90) -> list:
    """
    Extract frames at target_fps in a window of [anchor - padding, anchor + padding].
    Returns chronologically ordered list of frame paths.

    Idempotent: if out_dir already contains frames, skips extraction and
    returns the existing paths. To force re-extraction, delete out_dir first.
    """
    os.makedirs(out_dir, exist_ok=True)

    existing = sorted(glob.glob(os.path.join(out_dir, "frame_*.jpg")))
    if existing:
        return existing

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    video_fps    = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / video_fps if video_fps > 0 else 0

    if video_fps <= 0:
        cap.release()
        raise ValueError(f"Could not read fps from {video_path}")

    # Compute burst window, clamped to video duration
    start_s = max(0.0, anchor_seconds - padding_s)
    end_s   = min(duration_s, anchor_seconds + padding_s)

    if start_s >= end_s:
        cap.release()
        raise ValueError(
            f"Empty burst window for anchor {anchor_seconds}s "
            f"(video duration {duration_s:.1f}s)"
        )

    # Sample at target_fps within the burst window
    interval_s   = 1.0 / target_fps
    sample_times = []
    t = start_s
    while t <= end_s + 1e-6:
        sample_times.append(t)
        t += interval_s

    extracted = []
    for t in sample_times:
        frame_num = int(round(t * video_fps))
        if frame_num >= total_frames:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            print(f"  [WARN] Could not read frame at {t:.2f}s "
                  f"(frame {frame_num}); skipping")
            continue

        # Filename includes millisecond precision so 5fps samples are
        # sortable chronologically without ambiguity.
        whole_s = int(t)
        ms      = int(round((t - whole_s) * 1000))
        m, s    = divmod(whole_s, 60)
        fname   = f"frame_{m:02d}m{s:02d}s_{ms:03d}ms.jpg"
        out_path = os.path.join(out_dir, fname)

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        extracted.append(out_path)

    cap.release()
    return sorted(extracted)


def timestamp_to_seconds(ts: str) -> float:
    """Convert 'MMmSSs' to seconds. Mirrors escalation_router's parser."""
    try:
        ts_clean = ts.replace("m", ":").replace("s", "")
        parts    = ts_clean.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError, AttributeError):
        return 0.0


if __name__ == "__main__":
    print("frame_extraction.py -- import to use extract_segment()")
