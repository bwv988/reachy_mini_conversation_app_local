import time
import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


SR = rt_mod.INPUT_SAMPLE_RATE
FRAME = SR // 10  # 100 ms frames


def _handler() -> rt_mod.OpenaiRealtimeHandler:
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return rt_mod.OpenaiRealtimeHandler(deps)


def _loud() -> tuple[int, np.ndarray]:
    return SR, np.full(FRAME, 8000, dtype=np.int16)


def _silent() -> tuple[int, np.ndarray]:
    return SR, np.zeros(FRAME, dtype=np.int16)


async def _speak(handler: rt_mod.OpenaiRealtimeHandler, loud_frames: int = 5, silent_frames: int = 10) -> None:
    for _ in range(loud_frames):
        await handler.receive(_loud())
    for _ in range(silent_frames):
        await handler.receive(_silent())


@pytest.mark.asyncio
async def test_start_up_runs_until_shutdown() -> None:
    """start_up idles without any external connection and returns on shutdown."""
    handler = _handler()
    task = asyncio.create_task(handler.start_up())
    await asyncio.sleep(0.05)
    assert not task.done()

    await handler.shutdown()
    await asyncio.wait_for(task, timeout=5.0)


@pytest.mark.asyncio
async def test_utterance_triggers_one_turn(monkeypatch: pytest.MonkeyPatch) -> None:
    """A complete utterance is handed to ASR -> LLM -> TTS exactly once."""
    handler = _handler()
    process = AsyncMock()
    monkeypatch.setattr(handler, "_process_local_speech", process)

    await _speak(handler)
    await asyncio.sleep(0)

    process.assert_called_once()
    assert len(process.call_args.args[0]) > 0
    assert handler._turn_in_progress


@pytest.mark.asyncio
async def test_mic_ignored_while_turn_in_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """Speech arriving while a reply is being generated is ignored, not buffered."""
    handler = _handler()
    process = AsyncMock()
    monkeypatch.setattr(handler, "_process_local_speech", process)
    handler._turn_in_progress = True

    await _speak(handler)
    await asyncio.sleep(0)

    process.assert_not_called()
    assert handler._audio_buffer == []


@pytest.mark.asyncio
async def test_mic_muted_during_playback_and_unmute_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    """The robot's own speech (and the unmute delay after it) never reaches the VAD."""
    handler = _handler()
    process = AsyncMock()
    monkeypatch.setattr(handler, "_process_local_speech", process)
    monkeypatch.setattr(rt_mod.config, "MIC_UNMUTE_DELAY", 0.5)

    handler._extend_playback_window(1.0)
    await _speak(handler)
    await asyncio.sleep(0)
    process.assert_not_called()

    # Playback finished, but still inside the unmute delay
    handler._playback_end_time = time.monotonic() - 0.2
    assert handler._mic_muted()

    # Past the delay: the mic is live again
    handler._playback_end_time = time.monotonic() - 1.0
    assert not handler._mic_muted()
    await _speak(handler)
    await asyncio.sleep(0)
    process.assert_called_once()


@pytest.mark.asyncio
async def test_muting_mid_utterance_discards_partial_speech() -> None:
    """If the mic mutes while the user is mid-utterance, the partial audio is dropped."""
    handler = _handler()
    for _ in range(3):
        await handler.receive(_loud())
    assert handler._is_speech_active and handler._audio_buffer

    handler._turn_in_progress = True
    await handler.receive(_loud())

    assert not handler._is_speech_active
    assert handler._audio_buffer == []
    assert not handler._local_vad.is_speaking


def test_playback_window_accumulates() -> None:
    """Queued replies extend the playback window back-to-back."""
    handler = _handler()
    before = time.monotonic()
    handler._extend_playback_window(2.0)
    handler._extend_playback_window(1.5)
    assert handler._playback_end_time >= before + 3.5


@pytest.mark.asyncio
async def test_turn_flag_cleared_even_if_asr_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """A failing turn must not leave the mic muted forever."""
    handler = _handler()
    monkeypatch.setattr(handler._local_asr, "transcribe", AsyncMock(side_effect=RuntimeError("boom")))
    handler._turn_in_progress = True

    with pytest.raises(RuntimeError):
        await handler._process_local_speech(b"\x00\x00" * 100)
    assert not handler._turn_in_progress
