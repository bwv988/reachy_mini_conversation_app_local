"""Activation by name: matching rules and the conversation window."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.wake_word import WakeWordGate, find_wake_word
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


@pytest.mark.parametrize(
    "transcript",
    ["Tom, what do you see?", "Hey tom!", "What do you think, Tom?", "TOM. Are you there?", "Tom's turn now"],
)
def test_name_anywhere_matches(transcript: str) -> None:
    """The name matches at any position, in any case, next to punctuation."""
    assert find_wake_word(transcript, ["tom"]) == "tom"


@pytest.mark.parametrize(
    "transcript",
    ["See you tomorrow.", "An atom is small.", "A custom build.", "Go to the top.", "Tim, hello", "Ha-ha-ha-ha."],
)
def test_short_name_is_not_fuzzy(transcript: str) -> None:
    """Short names match whole words only: no 'tomorrow', 'atom', 'top' or 'Tim'."""
    assert find_wake_word(transcript, ["tom"]) is None


def test_long_names_match_close_misspellings() -> None:
    """Names of 5+ letters tolerate ASR misspellings."""
    assert find_wake_word("Hey Reechy, look here", ["reachy"]) == "reachy"
    assert find_wake_word("Hey Richard, look here", ["reachy"]) is None


def test_multi_word_phrase_needs_consecutive_words() -> None:
    """A phrase like 'hey tom' must appear in order."""
    assert find_wake_word("Hey Tom, hi", ["hey tom"]) == "hey tom"
    assert find_wake_word("Hey there Tom", ["hey tom"]) is None


def test_conversation_window() -> None:
    """After the name, follow-ups are accepted until the window runs out."""
    gate = WakeWordGate(["tom"], window_s=25)
    assert not gate.check("what time is it", now=0).accepted
    assert gate.check("Tom, what time is it", now=10).accepted
    assert gate.check("and tomorrow?", now=30).accepted  # within 25 s of the last accepted utterance
    assert gate.check("thanks", now=50).accepted  # window extended by the previous follow-up
    assert not gate.check("unrelated chatter", now=100).accepted


def test_keep_active_from_playback_end() -> None:
    """The window runs from when the robot stops speaking, not when it started."""
    gate = WakeWordGate(["tom"], window_s=25)
    gate.check("Tom, tell me a story", now=0)
    gate.keep_active(60)  # the reply finished playing at t=60
    assert gate.check("and then?", now=80).accepted
    assert not gate.check("and then?", now=200).accepted


def test_disabled_gate_accepts_everything() -> None:
    """WAKE_MODE=off (or no wake words) answers everything."""
    assert WakeWordGate(["tom"], window_s=25, enabled=False).check("anything", now=0).accepted
    assert WakeWordGate([], window_s=25, enabled=True).check("anything", now=0).accepted


def test_every_transcript_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Ignored utterances are logged too, for tuning."""
    gate = WakeWordGate(["tom"], window_s=25)
    with caplog.at_level(logging.INFO, logger="reachy_mini_conversation_app.wake_word"):
        gate.check("background chatter", now=0)
        gate.check("Tom, hello", now=1)
    assert "Heard (ignored, no wake word): background chatter" in caplog.text
    assert "Heard (wake word 'tom'): Tom, hello" in caplog.text


def _handler(monkeypatch: pytest.MonkeyPatch, transcript: str) -> rt_mod.OpenaiRealtimeHandler:
    monkeypatch.setattr(rt_mod.config, "WAKE_MODE", "name")
    monkeypatch.setattr(rt_mod.config, "WAKE_WORDS", ["tom"])
    movement_manager = MagicMock()
    movement_manager.is_playing_move.return_value = False
    handler = rt_mod.OpenaiRealtimeHandler(ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager))
    handler._local_llm_client = MagicMock()
    monkeypatch.setattr(handler._local_asr, "transcribe", AsyncMock(return_value=transcript))
    monkeypatch.setattr(handler, "_generate_local_response", AsyncMock())
    return handler


@pytest.mark.asyncio
async def test_handler_ignores_speech_without_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the name, the transcript never reaches the LLM."""
    handler = _handler(monkeypatch, "Ha-ha-ha-ha.")
    await handler._process_local_speech(b"\x00\x00" * 100)
    handler._generate_local_response.assert_not_called()
    assert not handler._turn_in_progress


@pytest.mark.asyncio
async def test_handler_answers_when_named(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the name, the transcript goes to the LLM."""
    handler = _handler(monkeypatch, "Tom, how many fingers?")
    await handler._process_local_speech(b"\x00\x00" * 100)
    handler._generate_local_response.assert_awaited_once_with("Tom, how many fingers?")
