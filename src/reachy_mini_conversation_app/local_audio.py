"""Local audio processing: VAD, ASR, and TTS without external dependencies.

This module provides built-in local alternatives to cloud services:
- VAD: Energy-based voice activity detection
- ASR: Whisper (distil-whisper by default) for speech-to-text, via transformers
- TTS: Kokoro-82M for text-to-speech (lightweight, edge-optimized)
"""

import re
import asyncio
import logging
from math import gcd
from typing import Any, Optional

import numpy as np
from scipy.signal import resample_poly


logger = logging.getLogger(__name__)

# Default sample rate for audio output
DEFAULT_SAMPLE_RATE = 24000
# Sample rate Whisper models are trained on
WHISPER_SAMPLE_RATE = 16000
# Whisper's native window; longer audio uses long-form decoding
WHISPER_MAX_SHORT_FORM_S = 30


def clean_text_for_speech(text: str) -> str:
    """Clean text for TTS by removing special characters and formatting.

    Args:
        text: Raw text that may contain brackets, asterisks, etc.

    Returns:
        Cleaned text suitable for speech synthesis

    """
    # Remove content in parentheses like (Pauses, then softly)
    text = re.sub(r'\([^)]*\)', '', text)

    # Remove content in square brackets like [thinking]
    text = re.sub(r'\[[^\]]*\]', '', text)

    # Remove content in curly braces like {stage direction}
    text = re.sub(r'\{[^}]*\}', '', text)

    # Remove asterisks used for emphasis like *italic* or **bold**
    text = re.sub(r'\*+', '', text)

    # Remove underscores used for emphasis
    text = re.sub(r'_+', '', text)

    # Remove markdown headers
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)

    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)

    # Trim leading/trailing whitespace
    text = text.strip()

    return text


class LocalVAD:
    """Energy-based Voice Activity Detection."""

    def __init__(
        self,
        energy_threshold: float = 0.01,
        silence_duration: float = 0.8,
        min_speech_duration: float = 0.3,
        sample_rate: int = 24000,
    ):
        """Initialize the VAD.

        Args:
            energy_threshold: RMS threshold for speech detection (0.0-1.0)
            silence_duration: Seconds of silence before considering speech ended
            min_speech_duration: Minimum seconds of speech to be valid
            sample_rate: Audio sample rate in Hz

        """
        self.energy_threshold = energy_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.sample_rate = sample_rate

        # State
        self.is_speaking = False
        self.speech_start_time: Optional[float] = None
        self.last_speech_time: Optional[float] = None
        self._current_time = 0.0

    def reset(self) -> None:
        """Reset VAD state."""
        self.is_speaking = False
        self.speech_start_time = None
        self.last_speech_time = None
        self._current_time = 0.0

    def process(self, audio_frame: np.ndarray) -> tuple[bool, bool]:
        """Process an audio frame and detect speech.

        Args:
            audio_frame: Audio samples as int16 numpy array

        Returns:
            Tuple of (speech_started, speech_ended)

        """
        # Calculate frame duration
        frame_duration = len(audio_frame) / self.sample_rate
        self._current_time += frame_duration

        # Calculate RMS energy
        audio_float = audio_frame.astype(np.float32) / 32768.0
        rms = np.sqrt(np.mean(audio_float ** 2))

        speech_started = False
        speech_ended = False

        if rms > self.energy_threshold:
            # Speech detected
            self.last_speech_time = self._current_time

            if not self.is_speaking:
                self.is_speaking = True
                self.speech_start_time = self._current_time
                speech_started = True
                logger.debug("VAD: Speech started (RMS=%.4f)", rms)

        elif self.is_speaking:
            # Silence detected while speaking
            silence_time = self._current_time - (self.last_speech_time or self._current_time)

            if silence_time >= self.silence_duration:
                # Check if speech was long enough
                speech_duration = (self.last_speech_time or self._current_time) - (self.speech_start_time or 0)

                if speech_duration >= self.min_speech_duration:
                    speech_ended = True
                    logger.debug("VAD: Speech ended (duration=%.2fs)", speech_duration)
                else:
                    logger.debug("VAD: Speech too short (%.2fs), ignoring", speech_duration)

                self.is_speaking = False
                self.speech_start_time = None

        return speech_started, speech_ended


def resolve_asr_device(device: str = "auto", dtype: str = "auto") -> tuple[str, str]:
    """Pick the Whisper device and dtype.

    ``auto`` prefers CUDA, then Apple Silicon (MPS), then CPU. float16 is only used
    on CUDA: Whisper on MPS in float16 produces garbage transcripts, so MPS always
    runs float32 (still faster than CPU on an M4: ~1.1 s vs ~1.5 s for 5 s of speech
    with distil-large-v3).
    """
    if device == "auto":
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"

    if dtype == "auto" or (device == "mps" and dtype == "float16"):
        dtype = "float16" if device == "cuda" else "float32"

    return device, dtype


class _DropWhisperSuppressTokensWarning(logging.Filter):
    """Drop a warning transformers' own Whisper generate() triggers on every call.

    WhisperGenerationMixin moves suppress_tokens into logits processors and clears them on
    its generation config copy; the generic generate() then sees the difference and warns
    about "passing generation_config together with generation-related arguments". Nothing
    on our side causes it, so only this exact message is filtered.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not ("together with generation-related arguments" in message and "suppress_tokens" in message)


class LocalASR:
    """Local speech recognition with a Whisper model, loaded directly through transformers.

    Works with any Whisper checkpoint on the Hugging Face hub: the distil-whisper
    family (``distil-whisper/distil-small.en``, ``distil-medium.en``, ``distil-large-v3``)
    or OpenAI's (e.g. ``openai/whisper-large-v3-turbo``).
    """

    def __init__(
        self,
        model_name: str = "distil-whisper/distil-small.en",
        device: str = "auto",
        dtype: str = "auto",
        language: str = "en",
    ):
        """Initialize the ASR.

        Args:
            model_name: Hugging Face id of a Whisper checkpoint
            device: Device to use (auto, cpu, cuda, mps)
            dtype: Data type (auto, float16, float32)
            language: Language code for multilingual models ("en", "fr", ...), or "auto" to
                detect it. English-only (``*.en``) models ignore it.

        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.language = language
        self._model: Any = None
        self._processor: Any = None
        self._torch_dtype: Any = None
        self._initialized = False

    @property
    def _english_only(self) -> bool:
        return self.model_name.endswith(".en")

    def _ensure_initialized(self) -> bool:
        """Lazy-load the Whisper model and processor."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        try:
            import torch
            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            device, dtype = resolve_asr_device(self.device, self.dtype)
            self.device = device
            self._torch_dtype = torch.float16 if dtype == "float16" else torch.float32

            logging.getLogger("transformers.generation.utils").addFilter(_DropWhisperSuppressTokensWarning())

            logger.info("Loading Whisper model '%s' on %s with %s...", self.model_name, device, dtype)
            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_name, dtype=self._torch_dtype, low_cpu_mem_usage=True
            ).to(device)
            self._model.eval()
            # Older checkpoints (distil-whisper) also carry generation settings in the *model*
            # config (forced_decoder_ids, suppress tokens), which transformers 5 flags as
            # deprecated on every call. The generation config holds the same values, so drop
            # the stale copies and let it be the single source.
            for attr in ("forced_decoder_ids", "suppress_tokens", "begin_suppress_tokens"):
                if getattr(self._model.config, attr, None) is not None:
                    setattr(self._model.config, attr, None)

            logger.info("Whisper model loaded successfully")
            return True

        except Exception as e:
            logger.error("Failed to load Whisper model '%s': %s", self.model_name, e)
            self._model = None
            return False

    def _generate_kwargs(self) -> dict[str, Any]:
        """Language/task arguments for generate(); English-only models take none."""
        if self._english_only:
            return {}
        kwargs: dict[str, Any] = {"task": "transcribe"}
        if self.language and self.language.lower() != "auto":
            kwargs["language"] = self.language
        return kwargs

    async def transcribe(self, audio_data: bytes, sample_rate: int = 24000) -> Optional[str]:
        """Transcribe audio to text.

        Args:
            audio_data: Raw PCM audio bytes (16-bit signed)
            sample_rate: Sample rate of the audio

        Returns:
            Transcribed text or None if failed

        """
        if not self._ensure_initialized():
            return None

        try:
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._transcribe_array, sample_rate, audio_array)
        except Exception as e:
            logger.error("ASR transcription failed: %s", e)
            return None

    def _transcribe_array(self, sample_rate: int, audio_array: np.ndarray) -> Optional[str]:
        """Transcribe an audio array (runs in executor)."""
        try:
            import torch

            # Whisper expects float32 at 16 kHz; resample with scipy (the transformers
            # pipeline would need torchaudio for this).
            audio = audio_array.astype(np.float32) / 32768.0 if audio_array.dtype == np.int16 else audio_array
            if sample_rate != WHISPER_SAMPLE_RATE:
                g = gcd(sample_rate, WHISPER_SAMPLE_RATE)
                audio = resample_poly(audio, WHISPER_SAMPLE_RATE // g, sample_rate // g).astype(np.float32)

            long_form = len(audio) > WHISPER_MAX_SHORT_FORM_S * WHISPER_SAMPLE_RATE
            if long_form:
                # > 30 s: keep all audio and let generate() do Whisper's sequential long-form decoding
                inputs = self._processor(
                    audio,
                    sampling_rate=WHISPER_SAMPLE_RATE,
                    return_tensors="pt",
                    truncation=False,
                    padding="longest",
                    return_attention_mask=True,
                )
            else:
                inputs = self._processor(audio, sampling_rate=WHISPER_SAMPLE_RATE, return_tensors="pt")

            features = inputs.input_features.to(self.device, dtype=self._torch_dtype)
            gen_kwargs = self._generate_kwargs()
            if long_form:
                # Long-form decoding chains 30 s windows and needs timestamps to do so
                gen_kwargs["attention_mask"] = inputs.attention_mask.to(self.device)
                gen_kwargs["return_timestamps"] = True

            with torch.inference_mode():
                ids = self._model.generate(features, **gen_kwargs)
            text = self._processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            text = text.strip()
            if text:
                logger.info("ASR transcription: %s", text[:100])
                return text
            return None

        except Exception as e:
            logger.error("Whisper transcription error: %s", e)
            return None


class LocalTTS:
    """Local Text-to-Speech using Kokoro via FastRTC.

    Uses FastRTC's built-in Kokoro support - a lightweight 82M parameter TTS
    optimized for edge devices with high-quality voice synthesis.
    """

    def __init__(
        self,
        output_sample_rate: int = 24000,
        voice: str = "af_sarah",
        speed: float = 1.0,
    ):
        """Initialize the TTS.

        Args:
            output_sample_rate: Target sample rate for output audio
            voice: Voice to use (af_sarah, am_michael, bf_emma, etc.)
            speed: Speech speed multiplier (0.5-2.0)

        """
        self.output_sample_rate = output_sample_rate
        self.voice = voice
        self.speed = speed
        self._model = None
        self._initialized = False

    def _ensure_initialized(self) -> bool:
        """Lazy-load the Kokoro TTS model using FastRTC."""
        if self._initialized:
            return self._model is not None

        self._initialized = True

        try:
            from fastrtc import get_tts_model

            logger.info("Loading Kokoro TTS model (voice: %s)...", self.voice)

            # Use FastRTC's built-in Kokoro support
            self._model = get_tts_model(model="kokoro", voice=self.voice)

            logger.info("Kokoro TTS model loaded successfully")
            return True

        except ImportError:
            logger.error("FastRTC not installed properly. Install with: pip install fastrtc")
            return False
        except Exception as e:
            logger.error("Failed to load Kokoro TTS model: %s", e)
            return False

    async def synthesize(self, text: str) -> Optional[np.ndarray]:
        """Synthesize text to audio.

        Args:
            text: Text to synthesize

        Returns:
            Audio samples as int16 numpy array, or None if failed

        """
        if not text or not text.strip():
            return None

        # Clean text for speech (remove brackets, asterisks, stage directions, etc.)
        cleaned_text = clean_text_for_speech(text)

        if not cleaned_text or not cleaned_text.strip():
            logger.warning("Text became empty after cleaning: %s", text[:100])
            return None

        if not self._ensure_initialized():
            return None

        try:
            # Run synthesis in executor to not block
            loop = asyncio.get_event_loop()
            audio = await loop.run_in_executor(None, self._synthesize_sync, cleaned_text)
            return audio

        except Exception as e:
            logger.error("TTS synthesis failed: %s", e)
            return None

    def _synthesize_sync(self, text: str) -> Optional[np.ndarray]:
        """Synthesize synchronously (runs in executor)."""
        try:
            from scipy.signal import resample

            # Collect all audio chunks from the streaming TTS
            audio_chunks = []
            for audio_chunk in self._model.stream_tts_sync(text):
                audio_chunks.append(audio_chunk)

            if not audio_chunks:
                logger.warning("TTS produced no audio")
                return None

            # Concatenate all chunks
            # Assuming audio_chunks are (sample_rate, audio_data) tuples
            if isinstance(audio_chunks[0], tuple):
                # Extract audio data from tuples
                audio_arrays = [chunk[1] if isinstance(chunk, tuple) else chunk for chunk in audio_chunks]
                audio = np.concatenate(audio_arrays)
            else:
                audio = np.concatenate(audio_chunks)

            # Ensure int16 format
            if audio.dtype != np.int16:
                # Assume float32 in [-1, 1] range
                if audio.dtype in (np.float32, np.float64):
                    max_val = np.abs(audio).max()
                    if max_val > 1.0:
                        audio = audio / max_val
                    audio = (audio * 32767).astype(np.int16)
                else:
                    audio = audio.astype(np.int16)

            # Kokoro outputs at 24kHz by default
            model_sample_rate = 24000

            # Resample if needed
            if model_sample_rate != self.output_sample_rate:
                num_samples = int(len(audio) * self.output_sample_rate / model_sample_rate)
                audio = resample(audio, num_samples).astype(np.int16)

            # Apply speed adjustment if needed
            if self.speed != 1.0:
                num_samples = int(len(audio) / self.speed)
                audio = resample(audio, num_samples).astype(np.int16)

            logger.debug("TTS generated %d samples", len(audio))
            return audio

        except Exception as e:
            logger.error("Kokoro TTS synthesis error: %s", e)
            return None
