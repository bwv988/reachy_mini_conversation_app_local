import re
import json
import time
import base64
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional

import numpy as np
from openai import AsyncOpenAI, APIStatusError
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import config
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.wake_word import WakeWordGate
from reachy_mini_conversation_app.local_audio import LocalASR, LocalTTS, LocalVAD
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs, dispatch_tool_call


logger = logging.getLogger(__name__)

INPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000
OUTPUT_SAMPLE_RATE: Final[Literal[24000]] = 24000

# Max LLM <-> tool round trips per user turn; the last round forces a spoken answer.
MAX_TOOL_ROUNDS: Final = 3
# Conversation turns (user message + tool exchanges + reply) kept as LLM context
MAX_HISTORY_TURNS: Final = 10

CAMERA_IMAGE_PROMPT = "Here is the camera image you just took. Answer from what you actually see in it."
NO_VISION_NOTE = (
    "(The camera image could not be shown to you: the language model server has no vision "
    "support. Tell the user briefly that you can't see right now.)"
)
IMAGE_ATTACHED = "attached in the next message"
IMAGE_EXPIRED = "shown to you at the time; no longer available, call the camera again to look"


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
        # The mic also stays muted until the ASR/TTS models are loaded (see start_up).
        self._models_ready: bool = False

        self._local_asr = LocalASR(model_name=config.DISTIL_WHISPER_MODEL, language=config.WHISPER_LANGUAGE)
        logger.info("ASR: Whisper (%s)", config.DISTIL_WHISPER_MODEL)

        # Activation by name (WAKE_MODE=name): only transcripts that address the robot reach the LLM
        self._wake_gate = WakeWordGate(
            config.WAKE_WORDS, window_s=config.WAKE_WINDOW_S, enabled=config.WAKE_MODE == "name"
        )
        if self._wake_gate.enabled:
            logger.info(
                "Activation by name: %s (conversation window %.0fs)",
                ", ".join(self._wake_gate.wake_words),
                config.WAKE_WINDOW_S,
            )

        self._local_tts = LocalTTS(
            output_sample_rate=self.output_sample_rate,
            voice=config.KOKORO_VOICE,
            speed=config.KOKORO_SPEED,
        )
        logger.info("TTS: Kokoro via FastRTC (voice: %s)", config.KOKORO_VOICE)

        # Local LLM client (any OpenAI-compatible server)
        self._local_llm_client: AsyncOpenAI | None = None
        self._local_llm_model: str = config.LOCAL_LLM_MODEL or "local-model"
        # One entry per turn: [user message, *tool exchanges (images removed), assistant reply].
        # Tool calls are kept on purpose: with only the replies in context, the model sees
        # itself answering visual questions without the camera and stops calling it.
        self._conversation_history: list[list[dict[str, Any]]] = []
        self._tools: list[dict[str, Any]] = self._build_tool_specs()
        logger.info("LLM tools: %s", ", ".join(t["function"]["name"] for t in self._tools) or "none")

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

    def _build_tool_specs(self) -> list[dict[str, Any]]:
        """Convert the profile's tools to Chat Completions format, skipping ones that can't work."""
        exclude: list[str] = []
        if self.deps.camera_worker is None:
            exclude += ["camera", "head_tracking"]
        elif getattr(self.deps.camera_worker, "head_tracker", None) is None:
            exclude.append("head_tracking")
        return [
            {
                "type": "function",
                "function": {
                    "name": spec["name"],
                    "description": spec["description"],
                    "parameters": spec["parameters"],
                },
            }
            for spec in get_tool_specs(exclude)
        ]

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(self.deps, self.gradio_mode, self.instance_path)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start_up(self) -> None:
        """Load the models, then keep the session alive; audio is processed as it arrives in receive()."""
        logger.info("Local session started - loading ASR and TTS models (mic muted until ready)...")
        await self._preload_models()
        self._models_ready = True
        logger.info("Models ready - listening")
        await self._shutdown_event.wait()
        logger.info("Local session ended")

    async def _preload_models(self) -> None:
        """Load ASR and TTS in a worker thread so the event loop (mic capture, playback) keeps running.

        Loading lazily on the first turn blocked the event loop for up to a minute on
        the robot's CM4, which made GStreamer drop mic samples.
        """
        loop = asyncio.get_running_loop()
        t0 = time.monotonic()
        await loop.run_in_executor(None, self._local_asr._ensure_initialized)
        await loop.run_in_executor(None, self._local_tts._ensure_initialized)
        logger.info("ASR/TTS models loaded in %.1fs", time.monotonic() - t0)

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
        """Whether mic input should be ignored (models loading, a turn in progress, or the robot speaking)."""
        return (
            not self._models_ready
            or self._turn_in_progress
            or time.monotonic() < self._playback_end_time + config.MIC_UNMUTE_DELAY
            or self.deps.movement_manager.is_playing_move()  # motor noise from dances/emotions
        )

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

            if not self._wake_gate.check(transcript).accepted:
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
        """Generate a reply with the local LLM (running any tool calls it makes) and speak it."""
        if not self._local_llm_client:
            return

        try:
            # Whole turns only, so a tool call is never separated from its result
            prior = [msg for turn in self._conversation_history[-MAX_HISTORY_TURNS:] for msg in turn]
            turn: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
            self._conversation_history.append(turn)
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": get_session_instructions()},
                *prior,
                *turn,
            ]
            turn_start = len(messages) - 1
            logger.debug("Calling local LLM with %d messages", len(messages))

            text_response = await self._run_llm_with_tools(messages)
            turn[:] = _for_history(messages[turn_start:])
            if not text_response:
                logger.warning("Local LLM returned an empty reply")
                return

            # Strip reasoning blocks some models emit anyway (Qwen, DeepSeek, ...)
            text_response = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text_response, flags=re.DOTALL).strip()
            logger.info("Local LLM response: %s", text_response[:100])

            turn.append({"role": "assistant", "content": text_response})
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": text_response}))

            await self._synthesize_locally(text_response)

        except Exception as e:
            logger.error("Local LLM generation failed: %s", e)
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": f"[error] LLM failed: {e}"}))

    async def _run_llm_with_tools(self, messages: list[dict[str, Any]]) -> str | None:
        """Call the LLM, execute any tool calls it makes, and loop until it answers in text.

        A camera tool result carries a JPEG; it is sent back to the LLM as an image in a
        follow-up user message (tool messages can't carry images).
        """
        for round_idx in range(MAX_TOOL_ROUNDS + 1):
            message = await self._chat(messages, use_tools=round_idx < MAX_TOOL_ROUNDS)
            tool_calls = message.tool_calls or []
            if not tool_calls:
                return message.content  # type: ignore[no-any-return]

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": call.id,
                            "type": "function",
                            "function": {"name": call.function.name, "arguments": call.function.arguments or "{}"},
                        }
                        for call in tool_calls
                    ],
                },
            )

            images: list[str] = []
            for call in tool_calls:
                name = call.function.name
                logger.info("Tool call: %s(%s)", name, call.function.arguments)
                result = await dispatch_tool_call(name, call.function.arguments or "{}", self.deps)
                image_b64 = result.pop("b64_im", None)
                if image_b64:
                    images.append(image_b64)
                    result["image"] = IMAGE_ATTACHED
                logger.info("Tool result: %s -> %s", name, json.dumps(result)[:200])
                await self.output_queue.put(
                    AdditionalOutputs(
                        {
                            "role": "assistant",
                            "content": json.dumps(result),
                            "metadata": {"title": f"🛠️ Used tool {name}", "status": "done"},
                        },
                    ),
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})

            if images:
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAMERA_IMAGE_PROMPT},
                            *({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images),
                        ],
                    },
                )
        return None

    async def _chat(self, messages: list[dict[str, Any]], use_tools: bool) -> Any:
        """Make one chat completion request, degrading gracefully if the server lacks features.

        - Image rejected (llama-server without --mmproj answers HTTP 500 "image input is not
          supported"): replace images with a note and retry.
        - Tools rejected (llama-server without --jinja): disable tools for the session and retry.
          Only when the error mentions tools/jinja, so a transient failure doesn't disable them.
        """
        assert self._local_llm_client is not None
        for _ in range(3):
            kwargs: dict[str, Any] = {}
            if config.LLM_DISABLE_THINKING:
                # llama.cpp: keep Qwen3-style thinking out of the token budget
                # (unknown templates ignore the kwarg, so this is safe to always send)
                kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
            tools = self._tools if use_tools else []
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            try:
                response = await self._local_llm_client.chat.completions.create(
                    model=self._local_llm_model,
                    messages=messages,  # type: ignore[arg-type]
                    max_tokens=512,
                    temperature=0.7,
                    **kwargs,
                )
                return response.choices[0].message
            except APIStatusError as e:
                if _replace_images_with_note(messages):
                    logger.warning(
                        "LLM rejected the camera image (%s). Start llama-server with --mmproj to enable vision.", e
                    )
                    continue
                if tools and any(word in str(e).lower() for word in ("tool", "jinja")):
                    logger.warning(
                        "LLM rejected tool calling (%s). Start llama-server with --jinja; continuing without tools.", e
                    )
                    self._tools = []
                    continue
                raise
        raise RuntimeError("LLM request failed after fallbacks")

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
        # The conversation window runs from when the robot stops talking
        self._wake_gate.keep_active(self._playback_end_time)

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


def _replace_images_with_note(messages: list[dict[str, Any]]) -> bool:
    """Replace image content parts with a text note in place. Return True if any were replaced."""
    replaced = False
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) and any(part.get("type") == "image_url" for part in content):
            msg["content"] = NO_VISION_NOTE
            replaced = True
    return replaced


def _for_history(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Copy a turn's messages for the conversation history, without the camera images.

    Image messages (and their no-vision replacement note) are dropped, and tool results
    that pointed at an attached image say it is no longer available, so later turns
    still see *that* the camera was used without carrying the large images along.
    """
    kept: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list) or content == NO_VISION_NOTE:
            continue
        if msg.get("role") == "tool" and isinstance(content, str) and IMAGE_ATTACHED in content:
            msg = {**msg, "content": content.replace(IMAGE_ATTACHED, IMAGE_EXPIRED)}
        kept.append(msg)
    return kept
