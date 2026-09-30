import re
import time
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional

import numpy as np
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.local_audio import LocalASR, LocalTTS, LocalVAD
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """Fully local conversation handler for fastrtc Stream: VAD -> ASR -> LLM -> TTS.

    The LLM is reached through an OpenAI-compatible API (Ollama, LM Studio,
    llama.cpp, vLLM); nothing is sent to OpenAI.

    Turn-taking is half-duplex: the microphone is ignored from the end of the
    user's utterance until the robot has finished speaking (plus
    ``config.MIC_UNMUTE_DELAY``), so the robot does not hear and answer itself.
    """

    def __init__(self, deps: ToolDependencies, gradio_mode: bool = False, instance_path: Optional[str] = None):
        """Initialize the handler."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=OUTPUT_SAMPLE_RATE,
            input_sample_rate=INPUT_SAMPLE_RATE,
        )
        self.output_sample_rate: Literal[24000] = OUTPUT_SAMPLE_RATE
        self.input_sample_rate: Literal[24000] = INPUT_SAMPLE_RATE

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()
        self._shutdown_event = asyncio.Event()

        # Speech capture
        self._local_vad = LocalVAD(
            energy_threshold=config.VAD_ENERGY_THRESHOLD,
            silence_duration=config.VAD_SILENCE_DURATION,
            min_speech_duration=config.VAD_MIN_SPEECH_DURATION,
            sample_rate=self.input_sample_rate,
        )
        self._audio_buffer: list[bytes] = []
        self._is_speech_active: bool = False

        # Half-duplex mic gating: set while a turn is being processed (ASR -> LLM -> TTS),
        # and a monotonic timestamp of when the queued robot speech will have finished playing.
        self._turn_in_progress: bool = False
        self._playback_end_time: float = 0.0

        self._local_asr = LocalASR(model_name=config.DISTIL_WHISPER_MODEL, language=config.WHISPER_LANGUAGE)
        logger.info("ASR: Distil-Whisper (%s)", config.DISTIL_WHISPER_MODEL)

        self._local_tts = LocalTTS(
            output_sample_rate=self.output_sample_rate,
            voice=config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
        )
        logger.info("TTS: Kokoro via FastRTC (voice: %s)", config.KOKORO_VOICE)

        # Local LLM client (any OpenAI-compatible server)
        self._local_llm_client: AsyncOpenAI | None = None
        self._local_llm_model: str = config.LOCAL_LLM_MODEL or "local-model"
        self._conversation_history: list[dict[str, Any]] = []

        if config.LOCAL_LLM_ENDPOINT:
            try:
                self._local_llm_client = AsyncOpenAI(
                    base_url=config.LOCAL_LLM_ENDPOINT,
                    api_key="not-needed",  # local servers don't check it, but the client requires one
                )
                logger.info(
                    "%s client initialized at %s with model %s",
                    (config.LLM_PROVIDER or "Local LLM").upper(),
                    config.LOCAL_LLM_ENDPOINT,
                    self._local_llm_model,
                )
            except Exception as e:
                logger.error("Failed to initialize local LLM client: %s", e)
        else:
            logger.warning("No local LLM endpoint configured (set LLM_PROVIDER in .env)")

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        """Keep the session alive; audio is processed as it arrives in receive()."""
        logger.info("Local session started - VAD, ASR, LLM and TTS all running locally")
        await self._shutdown_event.wait()
        logger.info("Local session ended")

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        self._shutdown_event.set()
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def apply_personality(self, profile: str | None) -> str:
        """Select a new personality (profile); it applies from the next turn.

        Returns a short status message for UI feedback.
        """
        from reachy_mini_conversation_app.config import set_custom_profile

        try:
            set_custom_profile(profile)
            try:
                get_session_instructions()  # validate the profile resolves
            except BaseException as e:  # the prompt loader raises SystemExit on missing files
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"
            logger.info("Applied personality: %s", profile or "built-in default")
            return "Applied personality. Takes effect from the next reply."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def get_available_voices(self) -> list[str]:
        """Voice names for the personality UI (none: Kokoro's voice is set via KOKORO_VOICE)."""
        return []

    # ------------------------------------------------------------------
    # Audio in
    # ------------------------------------------------------------------

    def _mic_muted(self) -> bool:
        """Whether mic input should be ignored (a turn is in progress or the robot is speaking)."""
        return self._turn_in_progress or time.monotonic() < self._playback_end_time + config.MIC_UNMUTE_DELAY

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive an audio frame from the microphone and run it through the local VAD.

        Handles mono and stereo input and resamples to the expected rate. When the
        VAD detects the end of an utterance, ASR -> LLM -> TTS runs in the background.
        """
        if self._mic_muted():
            if self._is_speech_active or self._local_vad.is_speaking:
                self._is_speech_active = False
                self._audio_buffer.clear()
                self._local_vad.reset()
            return

        input_sample_rate, audio_frame = frame

        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        audio_frame = audio_to_int16(audio_frame)

        speech_started, speech_ended = self._local_vad.process(audio_frame)

        if speech_started:
            self._is_speech_active = True
            self._audio_buffer.clear()
            self.deps.movement_manager.set_listening(True)
            logger.info("VAD: speech started")

        if self._is_speech_active:
            self._audio_buffer.append(audio_frame.tobytes())

        if speech_ended:
            self._is_speech_active = False
            self._turn_in_progress = True
            self.deps.movement_manager.set_listening(False)

            audio_data = b"".join(self._audio_buffer)
            self._audio_buffer.clear()
            logger.info("VAD: speech ended (%d bytes)", len(audio_data))

            asyncio.create_task(self._process_local_speech(audio_data))

    async def _process_local_speech(self, audio_data: bytes) -> None:
        """Process one utterance: ASR -> LLM -> TTS."""
        try:
            transcript = await self._local_asr.transcribe(audio_data, self.input_sample_rate)
            if not transcript:
                logger.warning("ASR returned no transcription")
                return

            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": transcript}))

            if self._local_llm_client:
                await self._generate_local_response(transcript)
            else:
                logger.warning("Local LLM not available, cannot generate response")
        finally:
            self._turn_in_progress = False

    # ------------------------------------------------------------------
    # LLM
    # ------------------------------------------------------------------

    async def _generate_local_response(self, user_message: str) -> None:
        """Generate a reply with the local LLM and speak it."""
        if not self._local_llm_client:
            return

        try:
            self._conversation_history.append({"role": "user", "content": user_message})
            messages = [
                {"role": "system", "content": get_session_instructions()},
                *self._conversation_history[-20:],  # keep the last 20 messages for context
            ]
            logger.debug("Calling local LLM with %d messages", len(messages))

            create_kwargs: dict[str, Any] = {}
            if config.LLM_DISABLE_THINKING:
                # llama.cpp: keep Qwen3-style thinking out of the token budget
                # (unknown templates ignore the kwarg, so this is safe to always send)
                create_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            response = await self._local_llm_client.chat.completions.create(
                model=self._local_llm_model,
                messages=messages,
                max_tokens=512,
                temperature=0.7,
                **create_kwargs,
            )

            text_response = response.choices[0].message.content
            if not text_response:
                logger.warning("Local LLM returned an empty reply")
                return

            # Strip reasoning blocks some models emit anyway (Qwen, DeepSeek, ...)
            text_response = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text_response, flags=re.DOTALL).strip()
            logger.info("Local LLM response: %s", text_response[:100])

            self._conversation_history.append({"role": "assistant", "content": text_response})
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text_response}))

            await self._synthesize_locally(text_response)

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] LLM failed: {e}"}))

    # ------------------------------------------------------------------
    # Audio out
    # ------------------------------------------------------------------

    async def _synthesize_locally(self, text: str) -> None:
        """Synthesize text with Kokoro and queue it for playback."""
        if not text or not text.strip():
            return

        try:
            audio_data = await self._local_tts.synthesize(text)
        except Exception as e:
            logger.error("Local TTS synthesis failed: %s", e)
            return
        if audio_data is None:
            logger.warning("Local TTS returned no audio")
            return

        if self.deps.head_wobbler is not None:
            self.deps.head_wobbler.feed(base64.b64encode(audio_data.tobytes()).decode("utf-8"))

        self._extend_playback_window(len(audio_data) / self.output_sample_rate)

        chunk_size = 4800  # 200ms at 24kHz
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i : i + chunk_size]
            await self.output_queue.put((self.output_sample_rate, chunk.reshape(1, -1)))
        logger.debug("Local TTS synthesis complete")

    def _extend_playback_window(self, duration_s: float) -> None:
        """Record that ``duration_s`` more seconds of speech were queued for playback."""
        self._playback_end_time = max(time.monotonic(), self._playback_end_time) + duration_s

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit the next audio frame or UI message (called periodically by the fastrtc Stream)."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]
