"""
cost_estimator.py — Match Lens pipeline cost estimator

Run BEFORE starting any pipeline step to:
  1. Calculate expected token usage and cost
  2. Check whether the run can complete without interruption
  3. Recommend sampling strategy based on available budget

Usage:
    python cost_estimator.py [MATCH_DIR] [--budget 20.00] [--quality standard]

Quality levels:
    economy   -- 10 frames/window,  75 event frames  (~$6/match)
    standard  -- 30 frames/window, 150 event frames  (~$11/match)
    full      -- 60 frames/window, 300 event frames  (~$20/match)
"""

import json, os, sys, argparse

# ── Token costs (Anthropic API, claude-sonnet-4-5, April 2026) ──────────────
TOKENS_PER_FRAME = 1568      # standard resolution image

PROMPT_TOKENS = {
    "structural": 3000,
    "player":     2500,
    "event":      2000,
    "setpiece":   1200,
    "recovery":   800,
    "report":     8000,
}
OUTPUT_TOKENS = {
    "structural": 4000,
    "player":     2000,
    "event":      1500,
    "setpiece":   800,
    "recovery":   400,
    "report":     6000,
}

# claude-sonnet-4-5 pricing (per token)
INPUT_COST_PER_TOKEN  = 3.00  / 1_000_000
OUTPUT_COST_PER_TOKEN = 15.00 / 1_000_000

QUALITY_PROFILES = {
    "economy":   {"frames_per_window": 10,  "event_frames": 75,  "label": "Economy (~$6)"},
    "standard":  {"frames_per_window": 30,  "event_frames": 150, "label": "Standard (~$11)"},
    "full":      {"frames_per_window": 60,  "event_frames": 300, "label": "Full quality (~$20)"},
    "full_1fps": {"frames_per_window": 300, "event_frames": 300, "label": "Full 1fps 512×288",
                  "tokens_per_frame": 170},  # single tile — cheaper than standard
}

# ── Load match data ──────────────────────────────────────────────────────────

def load_match_data(match_dir: str) -> dict:
    mc_path = os.path.join(match_dir, "match_config.json")
    wp_path = os.path.join(match_dir, "window_plan.json")
    mb_path = os.path.join(match_dir, "match_boundaries.json")

    mc = json.load(open(mc_path, encoding="utf-8")) if os.path.exists(mc_path) else {}
    wp = json.load(open(wp_path, encoding="utf-8")) if os.path.exists(wp_path) else {}
    mb = json.load(open(mb_path, encoding="utf-8")) if os.path.exists(mb_path) else {}

    windows  = wp.get("windows", [])
    goals    = mc.get("goals", [])
    subs     = mc.get("substitutions", [])

    # Count event windows (windows containing a goal or sub)
    goal_minutes = [g.get("time", {}).get("elapsed", 0) for g in goals]
    sub_minutes  = [s.get("time", {}).get("elapsed", 0) for s in subs]
    event_minutes = set(goal_minutes + sub_minutes)

    event_windows = 0
    null_windows  = 0
    for w in windows:
        t_start = w.get("start_s", w.get("start_seconds", 0)) / 60
        t_end   = w.get("end_s",   w.get("end_seconds", 0)) / 60
        if w.get("event_window") or any(t_start <= m <= t_end for m in event_minutes):
            event_windows += 1
        # null windows: not event_window and boundary_nearby (data gap proxy)
        if not w.get("event_window") and w.get("boundary_nearby"):
            null_windows += 1

    set_pieces_estimated = max(2, round(len(windows) * 0.15))  # ~15% windows have a set piece

    return {
        "match":           mc.get("match", os.path.basename(match_dir)),
        "total_windows":   len(windows),
        "event_windows":   event_windows,
        "null_windows":    null_windows,
        "set_pieces_est":  set_pieces_estimated,
        "goals":           len(goals),
        "subs":            len(subs),
    }


# ── Cost calculation ─────────────────────────────────────────────────────────

def calculate_cost(match_data: dict, quality: str = "standard") -> dict:
    profile = QUALITY_PROFILES[quality]
    fpw     = profile["frames_per_window"]
    evf     = profile["event_frames"]
    tpf     = profile.get("tokens_per_frame", TOKENS_PER_FRAME)

    W   = match_data["total_windows"]
    EW  = match_data["event_windows"]
    NW  = match_data["null_windows"]
    SP  = match_data["set_pieces_est"]

    steps = {}

    def step_cost(name, windows, frames_each):
        img_tok  = frames_each * tpf
        in_tok   = (PROMPT_TOKENS[name] + img_tok) * windows
        out_tok  = OUTPUT_TOKENS[name] * windows
        cost     = in_tok * INPUT_COST_PER_TOKEN + out_tok * OUTPUT_COST_PER_TOKEN
        return {"windows": windows, "input_tokens": in_tok,
                "output_tokens": out_tok, "cost_usd": round(cost, 2)}

    steps["3a_structural"] = step_cost("structural", W,   fpw)
    steps["3b_player"]     = step_cost("player",     W,   fpw)
    steps["3d_event"]      = step_cost("event",      EW,  evf)
    steps["3d_setpiece"]   = step_cost("setpiece",   SP,  evf // 3)
    steps["3d_recovery"]   = step_cost("recovery",   NW,  20)
    steps["reports"]       = {"windows": 2, "cost_usd": round(
        (PROMPT_TOKENS["report"] * 2 * INPUT_COST_PER_TOKEN +
         OUTPUT_TOKENS["report"] * 2 * OUTPUT_COST_PER_TOKEN), 2)}

    total_cost  = round(sum(s.get("cost_usd", 0) for s in steps.values()), 2)
    total_input = sum(s.get("input_tokens", 0) for s in steps.values())

    return {
        "quality":      quality,
        "label":        profile["label"],
        "steps":        steps,
        "total_cost_usd":   total_cost,
        "total_input_tokens": total_input,
        "frames_per_window":  fpw,
        "event_frames":       evf,
        "api_calls_total":    W + W + EW + SP + NW + 2,
    }


# ── Credit check ─────────────────────────────────────────────────────────────

def check_credits_available(required_usd: float) -> dict:
    """
    Attempt to read remaining credits from Anthropic API.
    Returns best-effort estimate.
    """
    try:
        import anthropic
        client = anthropic.Anthropic()
        # The API doesn't expose balance directly -- this is a placeholder
        # that would call the billing endpoint if Anthropic exposes it
        return {
            "checked": False,
            "message": "Anthropic API does not expose remaining balance via SDK. "
                       "Check console.anthropic.com before running.",
            "required_usd": required_usd,
        }
    except Exception as e:
        return {"checked": False, "error": str(e), "required_usd": required_usd}


# ── Print estimate ────────────────────────────────────────────────────────────

def print_estimate(match_data: dict, estimates: list, budget: float = None):
    print(f"\n{'='*60}")
    print(f"  Match Lens Cost Estimate")
    print(f"  {match_data['match']}")
    print(f"  {match_data['total_windows']} windows | {match_data['goals']} goals | "
          f"{match_data['event_windows']} event windows")
    print(f"{'='*60}\n")

    for est in estimates:
        print(f"  {est['label']} ({est['quality']})")
        print(f"    Total API calls: {est['api_calls_total']}")
        for step, data in est["steps"].items():
            print(f"    {step:<22} ${data.get('cost_usd', 0):.2f}")
        print(f"    {'TOTAL':<22} ${est['total_cost_usd']:.2f}")
        if budget:
            if est["total_cost_usd"] <= budget:
                print(f"    ✓ Within budget of ${budget:.2f}")
            else:
                print(f"    ✗ Exceeds budget of ${budget:.2f} "
                      f"(over by ${est['total_cost_usd']-budget:.2f})")
        print()

    # Recommendation
    best = min(estimates, key=lambda e: e["total_cost_usd"]
               if not budget or e["total_cost_usd"] <= budget
               else float('inf'))
    print(f"  Recommendation: {best['label']}")
    print(f"  Frames per window: {best['frames_per_window']}")
    print(f"  Event frames:      {best['event_frames']}")
    print(f"\n  To run:  python pipeline_runner_v2.py {sys.argv[1] if len(sys.argv)>1 else '[MATCH_DIR]'} "
          f"--quality {best['quality']}")
    print()

    credit_info = check_credits_available(best["total_cost_usd"])
    print(f"  Credit check: {credit_info['message']}")
    print(f"  https://console.anthropic.com/settings/billing")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("match_dir", nargs="?", default=".")
    parser.add_argument("--budget",  type=float, default=None)
    parser.add_argument("--quality", choices=["economy","standard","full","all"],
                        default="all")
    args = parser.parse_args()

    match_data = load_match_data(args.match_dir)

    if args.quality == "all":
        estimates = [calculate_cost(match_data, q)
                     for q in ["economy","standard","full"]]
    else:
        estimates = [calculate_cost(match_data, args.quality)]

    print_estimate(match_data, estimates, args.budget)

    # Write estimate to file
    out_path = os.path.join(args.match_dir, "cost_estimate.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"match": match_data, "estimates": estimates}, f, indent=2)
    print(f"  Estimate written to: {out_path}\n")
