"""
test_step17.py -- tests for extract_match_details.py's multi-image hardening

WHY this step exists: extract_match_details() was originally written assuming
exactly one screenshot per match page. Real footballwebpages.co.uk match reports
often need TWO screenshots because the page requires scrolling -- one screenshot
shows the teamsheets (which player plays for which team), a second shows the
goals/cards/subs timeline (which lists player names but NOT their team). Claude
needs BOTH images in view at once in the SAME message to correctly attribute a
carded or subbed player to their team -- this was confirmed for real when the
live Waltham Abbey extraction correctly attributed all 8 cards to the right
team by cross-referencing exactly this way.

These tests do NOT call the real Anthropic API (that would cost real money and
be non-deterministic to assert against on every run). Instead they replace
anthropic.Anthropic with a small spy class that records exactly what content
blocks extract_match_details() built and sent, then returns a fixed fake
response -- so we can assert the REAL hardening logic (path normalisation,
image block construction/order, validation, raw_record shape) with total
determinism, while keeping the real network call out of the test suite
entirely.

The one thing these tests do NOT re-verify with a live call is extraction
QUALITY (would Claude actually attribute cards correctly?) -- that was already
proven once with a real API call against the real Waltham Abbey screenshots.
TestWalthamAbbeyRealExtractionRegression below locks in that real, already-paid-for
result as a fixture, so we keep permanent proof of it without re-hitting the API
on every test run.
"""

import base64
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import extract_match_details as emd


# -- A minimal, deterministic JSON body matching the real extraction schema.
# WHY not the real Claude output: these tests are about the HARDENING (path
# handling, image block construction) not about extraction quality -- a small
# fixed payload keeps every assertion about the surrounding plumbing exact and
# fast, with no dependence on model output at all.
FAKE_EXTRACTED_JSON = {
    "match": "Fake Town vs Fake City",
    "competition": "Fake League",
    "date": "2026-01-01",
    "venue": "Fake Ground",
    "home_team": "Fake Town",
    "away_team": "Fake City",
    "ft_score": "1-0",
    "ht_score": "0-0",
    "attendance": 100,
    "kickoff": "3pm",
    "referee": "A. Referee",
    "goals": [{"time": {"elapsed": 10}, "team": {"name": "Fake Town"},
               "player": {"name": "Fake Scorer"}, "type": "normal"}],
    "cards": [],
    "substitutions": [],
    "lineups": [],
    "kit_colours": {"home": None, "away": None, "home_gk": None, "away_gk": None,
                    "_note": "n/a"},
    "source": "full_time_screenshot",
    "extraction_confidence": "high",
    "extraction_notes": "",
}


class _FakeUsage:
    def __init__(self, input_tokens=111, output_tokens=22):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    """Mimics the shape of an anthropic.types.Message the real code reads:
    response.content[0].text, response.model, response.usage.*"""
    def __init__(self, text, model="claude-opus-4-6-fake"):
        self.content = [SimpleNamespace(text=text)]
        self.model = model
        self.usage = _FakeUsage()


class _FakeMessages:
    def __init__(self, captured_calls):
        self._captured_calls = captured_calls

    def create(self, **kwargs):
        # Record exactly what was sent so tests can inspect the real content
        # blocks the hardening logic built, without touching the network.
        self._captured_calls.append(kwargs)
        return _FakeResponse(json.dumps(FAKE_EXTRACTED_JSON))


class _FakeAnthropicClient:
    """Spy standing in for anthropic.Anthropic(). Every instance shares the
    same captured_calls list passed in, so the test can inspect calls made
    by whichever code constructed the client."""
    def __init__(self, captured_calls):
        self.messages = _FakeMessages(captured_calls)


def _make_image_file(directory, name, payload: bytes) -> str:
    """Write a small fake 'screenshot' file with distinguishable byte content
    so tests can prove image blocks preserve ORDER and CONTENT, not just count."""
    path = os.path.join(directory, name)
    with open(path, "wb") as f:
        f.write(payload)
    return path


class TestEncodeImage(unittest.TestCase):
    """Unit tests for encode_image() -- the building block multi-image
    hardening relies on being correct per-file before it's called in a loop."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_media_type_by_extension(self):
        cases = {
            "shot.jpg":  "image/jpeg",
            "shot.jpeg": "image/jpeg",
            "shot.png":  "image/png",
            "shot.webp": "image/webp",
            "shot.gif":  "image/gif",
            "shot.bmp":  "image/jpeg",  # unknown extension falls back to jpeg
        }
        for filename, expected_type in cases.items():
            path = _make_image_file(self.tmp, filename, b"fake-bytes")
            _, media_type = emd.encode_image(path)
            self.assertEqual(media_type, expected_type, filename)

    def test_base64_data_matches_file_contents(self):
        payload = b"distinguishable-payload-12345"
        path = _make_image_file(self.tmp, "shot.jpg", payload)
        data, _ = emd.encode_image(path)
        self.assertEqual(base64.standard_b64decode(data), payload)


class TestExtractMatchDetailsMultiImageHardening(unittest.TestCase):
    """Deterministic tests of the hardening itself, using the spy client --
    no live API call anywhere in this class."""

    def setUp(self):
        self.match_dir = tempfile.mkdtemp()
        self.shots_dir = tempfile.mkdtemp()
        self.captured_calls = []
        self._patcher = patch(
            "extract_match_details.anthropic.Anthropic",
            side_effect=lambda: _FakeAnthropicClient(self.captured_calls),
        )
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()

    def test_single_string_path_backward_compatible(self):
        # The original, pre-hardening call shape: a bare string, not a list.
        # Must still work exactly as before, producing exactly one image block.
        path = _make_image_file(self.shots_dir, "one.jpg", b"shot-1-bytes")

        result = emd.extract_match_details(self.match_dir, path)

        self.assertEqual(result["match"], "Fake Town vs Fake City")
        self.assertEqual(len(self.captured_calls), 1)
        content = self.captured_calls[0]["messages"][0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        self.assertEqual(len(image_blocks), 1)

    def test_raw_record_screenshots_field_is_always_a_list(self):
        # WHY this matters: raw_record["screenshots"] must be a consistent
        # shape (always a list) regardless of whether the CALLER passed a
        # bare string or a list -- callers reading teamsheet_image_raw.json
        # later should never have to check "is this a string or a list?".
        path = _make_image_file(self.shots_dir, "one.jpg", b"shot-1-bytes")
        emd.extract_match_details(self.match_dir, path)

        with open(os.path.join(self.match_dir, "teamsheet_image_raw.json"), encoding="utf-8") as f:
            raw = json.load(f)
        self.assertIsInstance(raw["screenshots"], list)
        self.assertEqual(raw["screenshots"], [path])

    def test_list_of_paths_sends_one_block_per_screenshot_in_order(self):
        # The real multi-screenshot case: three distinct fake screenshots,
        # each with different byte content, so we can prove both COUNT and
        # ORDER are preserved -- not just that "some" blocks were sent.
        p1 = _make_image_file(self.shots_dir, "a.jpg", b"AAA-first")
        p2 = _make_image_file(self.shots_dir, "b.png", b"BBB-second")
        p3 = _make_image_file(self.shots_dir, "c.jpg", b"CCC-third")

        emd.extract_match_details(self.match_dir, [p1, p2, p3])

        content = self.captured_calls[0]["messages"][0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        self.assertEqual(len(image_blocks), 3)

        expected_order = [b"AAA-first", b"BBB-second", b"CCC-third"]
        for block, expected_bytes in zip(image_blocks, expected_order):
            decoded = base64.standard_b64decode(block["source"]["data"])
            self.assertEqual(decoded, expected_bytes)

    def test_media_type_detected_correctly_per_screenshot_in_a_list(self):
        p1 = _make_image_file(self.shots_dir, "a.png", b"png-bytes")
        p2 = _make_image_file(self.shots_dir, "b.jpg", b"jpg-bytes")

        emd.extract_match_details(self.match_dir, [p1, p2])

        content = self.captured_calls[0]["messages"][0]["content"]
        image_blocks = [c for c in content if c["type"] == "image"]
        self.assertEqual(image_blocks[0]["source"]["media_type"], "image/png")
        self.assertEqual(image_blocks[1]["source"]["media_type"], "image/jpeg")

    def test_image_blocks_precede_the_text_block(self):
        # Claude reads content in the order given -- the text prompt (which
        # asks it to cross-reference "the image(s)") must come AFTER every
        # image block, never interleaved or first.
        p1 = _make_image_file(self.shots_dir, "a.jpg", b"one")
        p2 = _make_image_file(self.shots_dir, "b.jpg", b"two")

        emd.extract_match_details(self.match_dir, [p1, p2])

        content = self.captured_calls[0]["messages"][0]["content"]
        types_in_order = [c["type"] for c in content]
        self.assertEqual(types_in_order, ["image", "image", "text"])

    def test_missing_screenshot_single_string_raises_before_any_api_call(self):
        missing_path = os.path.join(self.shots_dir, "does_not_exist.jpg")

        with self.assertRaises(FileNotFoundError):
            emd.extract_match_details(self.match_dir, missing_path)

        # Fail fast, before spending any API call -- validate every path
        # up front rather than discovering a missing file mid-request.
        self.assertEqual(len(self.captured_calls), 0)

    def test_missing_screenshot_in_list_raises_before_any_api_call(self):
        # The missing file is the SECOND of three -- proves every path in
        # the list is checked, not just the first.
        p1 = _make_image_file(self.shots_dir, "a.jpg", b"exists")
        missing_path = os.path.join(self.shots_dir, "missing.jpg")
        p3 = _make_image_file(self.shots_dir, "c.jpg", b"also-exists")

        with self.assertRaises(FileNotFoundError):
            emd.extract_match_details(self.match_dir, [p1, missing_path, p3])

        self.assertEqual(len(self.captured_calls), 0)

    def test_output_files_written_for_multi_screenshot_call(self):
        p1 = _make_image_file(self.shots_dir, "a.jpg", b"one")
        p2 = _make_image_file(self.shots_dir, "b.jpg", b"two")

        emd.extract_match_details(self.match_dir, [p1, p2])

        draft_path = os.path.join(self.match_dir, "match_config_draft.json")
        raw_path = os.path.join(self.match_dir, "teamsheet_image_raw.json")
        self.assertTrue(os.path.exists(draft_path))
        self.assertTrue(os.path.exists(raw_path))

        with open(draft_path, encoding="utf-8") as f:
            draft = json.load(f)
        # verified must always start False -- a human has to sign off before
        # this draft can ever be promoted to the real match_config.json.
        self.assertFalse(draft["verified"])


class TestWalthamAbbeyRealExtractionRegression(unittest.TestCase):
    """
    Locks in the result of a REAL, already-executed live Claude vision call
    (claude-opus-4-6, 3781 input / 2397 output tokens) against the real
    Waltham Abbey vs Stanway Rovers footballwebpages.co.uk screenshots, given
    to the model as TWO images in one message -- the real-world case this
    whole hardening step exists for. This test does not call the API again;
    it re-reads the saved fixture so this real result stays verified as the
    codebase changes, without spending another API call every test run.
    """

    FIXTURE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "tests", "fixtures", "waltham_abbey_extraction",
                                "match_config_draft.json")

    @classmethod
    def setUpClass(cls):
        with open(cls.FIXTURE_PATH, encoding="utf-8") as f:
            cls.data = json.load(f)

    def test_match_identity_and_score(self):
        self.assertEqual(self.data["match"], "Waltham Abbey vs Stanway Rovers")
        self.assertEqual(self.data["ft_score"], "1-2")
        self.assertFalse(self.data["verified"],
                          "A draft must never be auto-marked verified.")

    def test_three_goals_extracted(self):
        self.assertEqual(len(self.data["goals"]), 3)

    def test_seven_substitutions_extracted(self):
        self.assertEqual(len(self.data["substitutions"]), 7)

    def test_eight_cards_all_attributed_to_the_correct_team(self):
        # This is the specific claim the multi-image hardening exists to
        # support: a single-screenshot extraction could not see team
        # affiliation for carded players (the timeline screenshot alone has
        # no team column) -- only cross-referencing BOTH screenshots in one
        # message lets every card get attributed correctly. Verified here
        # against the real, manually-checked roster from that match.
        cards = self.data["cards"]
        self.assertEqual(len(cards), 8)

        stanway_players = {
            "Zack Zola Littlejohn", "Reece Harris", "Alexandruleonard Bolovan",
            "Alfie Cutbush", "George Okoye", "Callum Robinson",
        }
        waltham_players = {"Kieron Cathline", "Jesse Akubuine"}

        for card in cards:
            player = card["player"]["name"]
            team = card["team"]["name"]
            if player in stanway_players:
                self.assertEqual(team, "Stanway Rovers", player)
            elif player in waltham_players:
                self.assertEqual(team, "Waltham Abbey", player)
            else:
                self.fail(f"Unexpected carded player in fixture: {player}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
