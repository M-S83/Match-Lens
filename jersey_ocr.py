"""
jersey_ocr.py — EasyOCR jersey number recognition for Match Lens.

Scans sampled frames from a match directory, finds 1-3 digit numbers
at player-height positions, and builds a jersey_number_map.json
mapping frame filenames to lists of detected numbers with positions.

Usage:
    python jersey_ocr.py <match_dir> [--confidence 0.82] [--sample-rate 30]

Output:
    <match_dir>/jersey_number_map.json

Source compatibility:
    broadcast_fixed_wide:  runs (numbers visible in ~20-40% of frames)
    veo_ball_tracking:     runs (numbers visible when camera is close)
    tactical_wide_static:  SKIPPED (players too small, returns empty map)
"""

import argparse, json, os, sys
from pathlib import Path
from collections import defaultdict


def _should_run_ocr(source_profile_path: str) -> bool:
    """Return False for tactical_wide_static — numbers unreadable."""
    if not os.path.exists(source_profile_path):
        return True  # Default: try
    with open(source_profile_path, encoding='utf-8') as f:
        profile = json.load(f)
    source_type = profile.get('source_type', '')
    if source_type == 'tactical_wide_static':
        print(f"  [OCR] Source type {source_type} — jersey OCR skipped "
              f"(players too distant for reliable number reading)")
        return False
    return True


def _is_jersey_number(text: str) -> bool:
    """Return True if text looks like a jersey number (1-99)."""
    text = text.strip()
    if not text.isdigit():
        return False
    n = int(text)
    return 1 <= n <= 99


def _is_player_height(bbox, frame_height: int) -> bool:
    """
    Return True if bounding box is in the player zone of the frame.
    Excludes scoreboard overlays (top 12%) and pitch surrounds (bottom 8%).
    bbox format from PaddleOCR: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
    """
    ys = [pt[1] for pt in bbox]
    centre_y = sum(ys) / len(ys)
    relative_y = centre_y / frame_height
    return 0.12 < relative_y < 0.92


def run_ocr(match_dir: str,
            confidence_threshold: float = 0.82,
            sample_rate: int = 30) -> tuple:
    """
    Run EasyOCR on sampled frames and return (number_map, frames_scanned).

    sample_rate: scan every Nth frame (30 = 1 frame per 30s at 1fps)
    Returns: ({frame_filename: [{"number": int, "confidence": float,
                                 "bbox": [[x,y],...]}]}, int)
    """
    import easyocr
    from PIL import Image

    frames_dir = os.path.join(match_dir, 'frames')
    profile_path = os.path.join(match_dir, 'source_profile.json')

    if not _should_run_ocr(profile_path):
        return {}, 0

    if not os.path.isdir(frames_dir):
        print(f"  [OCR] Frames directory not found: {frames_dir}")
        return {}, 0

    all_frames = sorted(Path(frames_dir).glob('*.jpg'))
    sampled = all_frames[::sample_rate]
    print(f"  [OCR] Scanning {len(sampled)} frames "
          f"(1 per {sample_rate}s from {len(all_frames)} total)")

    # EasyOCR: English only, CPU, no GPU
    print("  [OCR] Loading EasyOCR model (first run downloads ~100MB)...")
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)

    number_map = {}
    hits = 0

    for i, frame_path in enumerate(sampled):
        fname = frame_path.name
        try:
            img = Image.open(frame_path)
            frame_h = img.height
            # EasyOCR returns list of (bbox, text, confidence)
            # bbox format: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            result = reader.readtext(str(frame_path), detail=1)
        except Exception:
            continue

        if not result:
            continue

        frame_hits = []
        for (bbox, text, conf) in result:
            if (conf >= confidence_threshold
                    and _is_jersey_number(text)
                    and _is_player_height(bbox, frame_h)):
                frame_hits.append({
                    "number":     int(text.strip()),
                    "confidence": round(conf, 3),
                    "bbox":       [[int(x), int(y)] for x, y in bbox]
                })
                hits += 1

        if frame_hits:
            number_map[fname] = frame_hits

        if (i + 1) % 20 == 0:
            print(f"  [OCR] {i+1}/{len(sampled)} frames scanned, "
                  f"{hits} number sightings so far")

    print(f"  [OCR] Complete: {hits} sightings across "
          f"{len(number_map)} frames")
    return number_map, len(sampled)


def resolve_to_players(number_map: dict, mc: dict) -> dict:
    """
    Cross-reference sighted numbers with match_config lineups.
    Returns {frame: [{"number": N, "name": "Player Name",
                      "team_side": "home"|"away", "confidence": F}]}
    """
    # Build (side, number) → name lookup
    lookup = {}
    for lineup in mc.get('lineups', []):
        side = lineup.get('team_side', '')
        for p in lineup.get('startXI', []) + lineup.get('substitutes', []):
            player = p.get('player', p)
            name   = player.get('name', '') if isinstance(player, dict) else ''
            number = player.get('number') if isinstance(player, dict) else None
            if name and number is not None:
                lookup[(side, int(number))] = name

    resolved = {}
    for fname, sightings in number_map.items():
        frame_resolved = []
        for s in sightings:
            n = s['number']
            matched = None
            for side in ('home', 'away'):
                if (side, n) in lookup:
                    matched = {
                        "number":    n,
                        "name":      lookup[(side, n)],
                        "team_side": side,
                        "confidence": s['confidence'],
                        "bbox":      s['bbox'],
                    }
                    break
            if matched:
                frame_resolved.append(matched)
            else:
                frame_resolved.append({
                    "number":    n,
                    "name":      None,
                    "team_side": None,
                    "confidence": s['confidence'],
                    "bbox":      s['bbox'],
                })
        if frame_resolved:
            resolved[fname] = frame_resolved
    return resolved


def summarise_by_player(resolved: dict) -> dict:
    """
    Aggregate resolved map: for each player, how many times
    were they sighted and in which frames?
    Returns {player_name: {"count": N, "frames": [...]}}
    """
    summary = defaultdict(lambda: {"count": 0, "frames": []})
    for fname, sightings in resolved.items():
        for s in sightings:
            name = s.get('name')
            if name:
                summary[name]['count'] += 1
                summary[name]['frames'].append(fname)
    return dict(summary)


def main():
    parser = argparse.ArgumentParser(
        description="Jersey number OCR for Match Lens")
    parser.add_argument('match_dir')
    parser.add_argument('--confidence', type=float, default=0.82)
    parser.add_argument('--sample-rate', type=int, default=30)
    args = parser.parse_args()

    mc_path = os.path.join(args.match_dir, 'match_config.json')
    if not os.path.exists(mc_path):
        print(f"match_config.json not found: {mc_path}")
        sys.exit(1)
    with open(mc_path, encoding='utf-8') as f:
        mc = json.load(f)

    # Run OCR
    number_map, frames_scanned = run_ocr(
        args.match_dir, args.confidence, args.sample_rate)

    # Resolve to players
    resolved = resolve_to_players(number_map, mc)

    # Build summary
    summary = summarise_by_player(resolved)

    # Determine source type
    sp_path = os.path.join(args.match_dir, 'source_profile.json')
    source_type = 'unknown'
    if os.path.exists(sp_path):
        with open(sp_path, encoding='utf-8') as f:
            source_type = json.load(f).get('source_type', 'unknown')

    output = {
        "source_type":         source_type,
        "frames_scanned":      frames_scanned,
        "total_sightings":     sum(len(v) for v in number_map.values()),
        "resolved_sightings":  sum(len(v) for v in resolved.values()),
        "player_summary":      summary,
        "frame_detail":        resolved,
    }

    out_path = os.path.join(args.match_dir, 'jersey_number_map.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    print(f"\n  [OCR] jersey_number_map.json written")
    print(f"  Players sighted: {len(summary)}")
    for name, data in sorted(summary.items(),
                              key=lambda x: -x[1]['count'])[:10]:
        print(f"    {name}: {data['count']} sightings")


if __name__ == '__main__':
    main()
