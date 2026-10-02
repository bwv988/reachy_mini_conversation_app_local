"""Tests for Whisper device/dtype selection."""

from unittest.mock import MagicMock

import pytest

from reachy_mini_conversation_app.local_audio import resolve_asr_device


def _fake_torch(monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool) -> None:
    torch = MagicMock()
    torch.cuda.is_available.return_value = cuda
    torch.backends.mps.is_available.return_value = mps
    monkeypatch.setitem(__import__("sys").modules, "torch", torch)


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [
        (True, True, ("cuda", "float16")),
        (False, True, ("mps", "float32")),
        (False, False, ("cpu", "float32")),
    ],
)
def test_auto_prefers_cuda_then_mps_then_cpu(
    monkeypatch: pytest.MonkeyPatch, cuda: bool, mps: bool, expected: tuple[str, str]
) -> None:
    """Auto picks the fastest available device, with float16 only on CUDA."""
    _fake_torch(monkeypatch, cuda=cuda, mps=mps)
    assert resolve_asr_device("auto", "auto") == expected


def test_mps_never_uses_float16() -> None:
    """Whisper on MPS in float16 produces garbage, so it is forced to float32."""
    assert resolve_asr_device("mps", "float16") == ("mps", "float32")


def test_explicit_settings_are_kept() -> None:
    """An explicit device/dtype is respected (except the MPS float16 guard)."""
    assert resolve_asr_device("cpu", "float32") == ("cpu", "float32")
    assert resolve_asr_device("cuda", "float32") == ("cuda", "float32")


def test_english_only_models_get_no_language_args() -> None:
    """*.en checkpoints reject language/task arguments."""
    from reachy_mini_conversation_app.local_audio import LocalASR

    assert LocalASR(model_name="distil-whisper/distil-medium.en", language="fr")._generate_kwargs() == {}


def test_multilingual_models_get_language_and_task() -> None:
    """Multilingual checkpoints transcribe in the configured language, or auto-detect."""
    from reachy_mini_conversation_app.local_audio import LocalASR

    asr = LocalASR(model_name="distil-whisper/distil-large-v3", language="en")
    assert asr._generate_kwargs() == {"task": "transcribe", "language": "en"}
    asr.language = "auto"
    assert asr._generate_kwargs() == {"task": "transcribe"}


def test_whisper_warning_filter_is_narrow() -> None:
    """Only the upstream suppress_tokens warning is dropped."""
    import logging

    from reachy_mini_conversation_app.local_audio import _DropWhisperSuppressTokensWarning

    f = _DropWhisperSuppressTokensWarning()

    def rec(msg: str) -> logging.LogRecord:
        return logging.LogRecord("transformers.generation.utils", logging.WARNING, "", 0, msg, None, None)

    noisy = "Passing `generation_config` together with generation-related arguments=({'suppress_tokens'}) is deprecated"
    assert f.filter(rec(noisy)) is False
    assert f.filter(rec("Passing `generation_config` together with generation-related arguments=({'max_new_tokens'})"))
    assert f.filter(rec("something else entirely"))
