"""
extract_match_details.py — Match Lens Step 1d (image variant)

Extracts match details from one or more screenshots of the Full-Time
website (or similar structured match results page, e.g. footballwebpages.
co.uk) using Claude vision. Multiple screenshots are needed when a single
page requires scrolling to capture in full.

Outputs:
  match_config_draft.json  — structured match data, ready for verification
  teamsheet_image_raw.json — raw Claude response (permanent record)

Usage:
    python extract_match_details.py [MATCH_DIR] [SCREENSHOT_PATH]

    Or interactively:
    python extract_match_details.py
"""

import anthropic
import base64
import json
import os
import sys
from datetime import datetime
from pipeline_schemas import stamp_schema_version

# Load API key from .env in the same directory as this script.
# Use dotenv_values + explicit os.environ assignment so that an already-set
# but empty ANTHROPIC_API_KEY in the process environment is always overridden.
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    from dotenv import dotenv_values as _dotenv_values
    for _k, _v in _dotenv_values(_env_path).items():
        if _v:
            os.environ[_k] = _v


EXTRACTION_PROMPT = """
You are extracting structured match data from a football results page screenshot.
Extract every piece of data visible in the image exactly as shown — do not infer,
guess, or fill in anything not explicitly visible.

Output ONLY a single raw JSON object with this exact structure. No commentary,
no markdown, no code fences — just the JSON.

{
  "match": "[Home Team] vs [Away Team]",
  "competition": "[Competition name as shown]",
  "date": "[YYYY-MM-DD — convert the displayed date to ISO format]",
  "venue": "[Venue name as shown]",
  "home_team": "[Home team full name]",
  "away_team": "[Away team full name]",
  "ft_score": "[home_goals]-[away_goals]",
  "ht_score": "[home_goals]-[away_goals]",
  "attendance": [number or null if not shown],
  "kickoff": "[time as shown, e.g. 3pm or 15:00, or null]",
  "referee": "[name or null]",
  "goals": [
    {
      "time": {"elapsed": [minute]},
      "team": {"name": "[team name]"},
      "player": {"name": "[full name]"},
      "type": "normal"
    }
  ],
  "cards": [
    {
      "time": {"elapsed": [minute]},
      "team": {"name": "[team name — match carefully to which team the player plays for]"},
      "player": {"name": "[full name]"},
      "detail": "Yellow Card"
    }
  ],
  "substitutions": [
    {
      "time": {"elapsed": [minute]},
      "team": {"name": "[team name]"},
      "player": {"name": "[player coming OFF]"},
      "assist": {"name": "[player coming ON]"}
    }
  ],
  "lineups": [
    {
      "team": {"name": "[home team]"},
      "startXI": [
        {"player": {"name": "[name]", "number": [shirt number as integer or null], "pos": null}}
      ],
      "substitutes": [
        {"player": {"name": "[name]", "number": null, "pos": null}}
      ]
    },
    {
      "team": {"name": "[away team]"},
      "startXI": [
        {"player": {"name": "[name]", "number": [shirt number as integer or null], "pos": null}}
      ],
      "substitutes": [
        {"player": {"name": "[name]", "number": null, "pos": null}}
      ]
    }
  ],
  "kit_colours": {
    "home": null,
    "away": null,
    "home_gk": null,
    "away_gk": null,
    "_note": "Kit colours are not shown on this page — fill in manually"
  },
  "source": "full_time_screenshot",
  "extraction_confidence": "[high/medium/low — based on image clarity]",
  "extraction_notes": "[any ambiguity, partial names, or uncertain items]"
}

Important rules:
- Shirt numbers: use the numbered position shown next to each player name if visible.
  If those numbers are lineup positions (1–11), list them as shirt numbers only if
  the page clearly labels them as shirt numbers. If uncertain, use null.
- Cards: check carefully which team each carded player belongs to by cross-referencing
  the lineup. Do not assume — a player's name appears in one team's column.
- Substitutions: "player" = coming OFF, "assist" = coming ON.
- Dates: convert displayed date (e.g. "Monday 6th April 2026") to ISO format "2026-04-06".
- Partial data: if a field is not visible in the image, use null — do not guess.
"""


def encode_image(image_path: str) -> tuple[str, str]:
    """Return (base64_data, media_type) for a local image file."""
    ext = os.path.splitext(image_path)[1].lower()
    media_types = {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }
    media_type = media_types.get(ext, "image/jpeg")
    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, media_type


def extract_match_details(match_dir: str, screenshot_path) -> dict:
    """
    Run vision extraction on one or more screenshots of the SAME match page
    and write output files. Returns the extracted data dict.

    screenshot_path: a single path (str), OR a list of paths when the source
    page needed multiple screenshots to capture in full (e.g. a page that
    required scrolling -- footballwebpages.co.uk match reports commonly do,
    since the goals/cards/subs timeline sits below the fold). WHY accept
    both shapes instead of forcing callers to always pass a list: every
    existing caller (and the CLI usage in this file) only ever had one
    screenshot, so keeping that the simple case avoids breaking anything
    while still covering the real multi-screenshot case.
    """
    # Normalise to a list up front so the rest of the function only has to
    # handle ONE shape, not two -- this is the same reasoning we've used
    # everywhere else in this codebase for Optional/Union inputs: normalise
    # once at the boundary, keep the body simple.
    paths = [screenshot_path] if isinstance(screenshot_path, str) else list(screenshot_path)

    for p in paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Screenshot not found: {p}")

    print(f"\n{'─'*55}")
    print(f"  Step 1d — Image extraction")
    print(f"  Screenshot(s): {', '.join(os.path.basename(p) for p in paths)}")
    print(f"{'─'*55}")
    print("  Sending to Claude vision...")

    # WHY one image block per screenshot, in order, ahead of the single text
    # block: Anthropic's vision API accepts multiple images in one message
    # and reads them in the order given. Sending them together (rather than
    # as separate API calls per screenshot) lets the model cross-reference
    # data that spans screenshots -- e.g. screenshot 1 has the teamsheets
    # (which player plays for which team), screenshot 2 has the cards/subs
    # timeline (which only lists player names) -- Claude needs BOTH in view
    # at once to correctly attribute a carded player to their team.
    image_blocks = []
    for p in paths:
        image_data, media_type = encode_image(p)
        image_blocks.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_data,
            },
        })

    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": image_blocks + [
                    {
                        "type": "text",
                        "text": EXTRACTION_PROMPT,
                    },
                ],
            }
        ],
    )

    raw_text = response.content[0].text.strip()

    # Save raw response as permanent record. WHY store the normalised
    # `paths` list (not the original screenshot_path arg): the arg could be
    # a single string OR a list -- the raw record should always show what
    # was ACTUALLY sent to the model, in a consistent shape, regardless of
    # how the caller invoked the function.
    raw_record = {
        "screenshots": paths,
        "extracted_at": datetime.now().isoformat(),
        "model": response.model,
        "usage": {
            "input_tokens":  response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "raw_text": raw_text,
    }
    raw_path = os.path.join(match_dir, "teamsheet_image_raw.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(raw_record, "teamsheet_image_raw"), f, indent=2)
    print(f"  Raw response saved → teamsheet_image_raw.json")
    print(f"  Tokens: {response.usage.input_tokens} in / {response.usage.output_tokens} out")

    # Parse the JSON from Claude's response
    # Strip any accidental markdown fences
    text = raw_text
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.rsplit("```", 1)[0]
    text = text.strip()

    try:
        extracted = json.loads(text)
    except json.JSONDecodeError as e:
        print(f"\n  [ERROR] Could not parse JSON from Claude response: {e}")
        print("  Raw output saved to teamsheet_image_raw.json for inspection.")
        raise

    # Enrich with pipeline fields that are not in the screenshot
    extracted.setdefault("home_kit",     None)
    extracted.setdefault("away_kit",     None)
    extracted.setdefault("home_gk_kit",  None)
    extracted.setdefault("away_gk_kit",  None)
    # Fix 33b: legacy focus field removed -- pipeline produces both team reports.
    extracted.setdefault("attack_direction_1h", None)
    extracted.setdefault("attack_direction_2h", None)
    extracted.setdefault("enrichment_level",    "full")
    extracted.setdefault("player_id_ceiling",   "confirmed")
    # v3 port Step 2: seed pitch_dimensions_assumed and watch_list with
    # safe defaults so downstream v3 readers (pitch_validation,
    # watch_list_aggregator, build_deep_skill_metrics_v2's metre-based
    # metric fns) never hit a missing-key path on fresh drafts. The
    # setdefault pattern preserves any operator-edited values across
    # re-runs against the same draft. SKILL.md:679-684 documents the
    # same default shape (executable code in extract_match_details.py
    # is the source of truth; the SKILL.md block is illustrative).
    extracted.setdefault("pitch_dimensions_assumed", {
        "length_m": 105,
        "width_m":  68,
        "source":   "default",
    })
    extracted.setdefault("watch_list", [])
    extracted["verified"] = False  # always False until human signs off

    # Write draft config
    draft_path = os.path.join(match_dir, "match_config_draft.json")
    with open(draft_path, "w", encoding="utf-8") as f:
        json.dump(stamp_schema_version(extracted, "match_config_draft"), f, indent=2)

    print(f"  Draft config saved → match_config_draft.json")
    print(f"{'─'*55}\n")

    return extracted


def print_verification_block(data: dict) -> None:
    """Print the same-style verification block as the API step."""
    home = data.get("home_team", "?")
    away = data.get("away_team", "?")

    def fmt_lineup(team_lineup):
        lines = []
        for p in team_lineup.get("startXI", []):
            pl = p["player"]
            num = pl.get("number", "?") or "?"
            lines.append(f"  #{num:>2}  {pl.get('name', 'Unknown')}")
        subs = [p["player"] for p in team_lineup.get("substitutes", [])]
        lines.append(f"  Bench: {', '.join(p.get('name','') for p in subs) or 'none'}")
        return "\n".join(lines)

    lineups = data.get("lineups", [])
    home_lineup = next((l for l in lineups if l["team"]["name"] == home), {})
    away_lineup = next((l for l in lineups if l["team"]["name"] == away), {})

    goals = data.get("goals", [])
    cards = data.get("cards", [])
    subs  = data.get("substitutions", [])

    null_fields = [k for k in ["home_kit", "away_kit",
                                "attack_direction_1h", "attack_direction_2h"]
                   if not data.get(k)]

    block = f"""
MATCH DETAILS VERIFICATION — MANUAL CHECK REQUIRED
{'─' * 60}
Match:        {data.get('match', '?')}
Date:         {data.get('date', '?')}
Venue:        {data.get('venue', '?')}
Competition:  {data.get('competition', '?')}
Attendance:   {data.get('attendance', '?')}
Kick-off:     {data.get('kickoff', '?')}
Referee:      {data.get('referee', '?')}
HT Score:     {data.get('ht_score', '?')}
FT Score:     {data.get('ft_score', '?')}
Confidence:   {data.get('extraction_confidence', '?')}

Goals:
{chr(10).join(f"  {g['time']['elapsed']}' {g['team']['name']} — {g['player']['name']}" for g in goals) or '  None recorded'}

HOME — {home}:
{fmt_lineup(home_lineup) if home_lineup else '  Not extracted'}

AWAY — {away}:
{fmt_lineup(away_lineup) if away_lineup else '  Not extracted'}

Substitutions:
{chr(10).join(f"  {s['time']['elapsed']}' {s['team']['name']}: {s['assist']['name']} ON for {s['player']['name']}" for s in subs) or '  None recorded'}

Cards:
{chr(10).join(f"  {c['time']['elapsed']}' {c['team']['name']} {c['detail']} — {c['player']['name']}" for c in cards) or '  None recorded'}

Extraction notes: {data.get('extraction_notes', 'none')}
{'─' * 60}
SOURCE: image extraction (claude-opus-4-6)
{'⚠  FILL IN BEFORE CONFIRMING: ' + ', '.join(null_fields) if null_fields else '✓  All standard fields extracted'}

Kit colours (fill in from video):
  Home kit:    {data.get('home_kit') or '[FILL IN]'}
  Away kit:    {data.get('away_kit') or '[FILL IN]'}
  Home GK:     {data.get('home_gk_kit') or '[FILL IN]'}
  Away GK:     {data.get('away_gk_kit') or '[FILL IN]'}

Attack direction (fill in from video):
  1H: [team] attack [left/right]
  2H: [team] attack [left/right]
{'─' * 60}
CHECK EACH ITEM:
  [ ] Correct fixture (not a different match between same teams)
  [ ] Starting XI matches actual lineup
  [ ] Cards assigned to the correct team
  [ ] Substitutions complete with correct timings
  [ ] Goals and scorers correct
  [ ] Kit colours filled in above
  [ ] Focus team set
  [ ] Attack directions filled in above

Edit match_config_draft.json, then copy to match_config.json and set verified: true.
"""
    print(block)


if __name__ == "__main__":
    # WHY args[1:] (plural) rather than args[1]: the CLI now mirrors the
    # function signature -- any number of trailing args are all screenshot
    # paths for the SAME match page, e.g.
    #   python extract_match_details.py MATCH_DIR shot1.jpg shot2.jpg
    args = sys.argv[1:]
    if len(args) >= 2:
        match_dir       = args[0]
        screenshot_path = args[1:]
    elif len(args) == 1:
        match_dir       = args[0]
        screenshot_path = input("Screenshot path: ").strip().strip('"')
    else:
        match_dir       = input("Match directory: ").strip().strip('"')
        screenshot_path = input("Screenshot path: ").strip().strip('"')

    if not os.path.isdir(match_dir):
        print(f"Error: '{match_dir}' is not a valid directory.")
        sys.exit(1)

    data = extract_match_details(match_dir, screenshot_path)
    print_verification_block(data)
