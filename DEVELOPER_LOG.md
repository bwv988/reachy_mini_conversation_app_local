# Developer Log

## Session 2026-09-29 → 2026-09-30 (late night)

Goal: make the app run fully local with the LLM on an external GB10, and get it
talking on the real (wireless) robot.

### Environment facts (verify before relying on them)

| Thing | Detail |
|---|---|
| Fork | `github.com/bwv988/reachy_mini_conversation_app_local` — `origin`; `upstream` = `dwain-barnes` repo |
| Work branch | `reachy_tests` (pushed, tracks origin) |
| Mac | M4 Pro, 24 GB. uv-managed Python 3.12 in `.venv`. Sim works (`mjpython -m reachy_mini.daemon.app.main --sim`; needs the `.venv/libpython3.12.dylib` symlink — see README Quick Start) |
| GB10 (GPU server) | `192.168.0.25:8085`, llama.cpp serving **Qwen3.8-27B UD-Q4_K_XL** (unsloth GGUF, model id = full file path, `…/Qwen3.8-27B-UD-Q4_K_XL.gguf`). ~20 tok/s generation, ~90 tok/s prompt (was BF16 at ~4 tok/s until 2026-09-30; swapped for latency). llama.cpp with a single loaded model ignores the requested `model` name, so a stale `LMSTUDIO_MODEL` still works; keep it accurate anyway. **Also serves this Pi harness** — expect shared latency |
| Robot (wireless) | `192.168.0.19` = `reachy-mini.local`, user `pollen` (pw `root`). SSH key of the Mac (`sral@andromeda.local`) installed in `~/.ssh/authorized_keys`. CM4, 4 GB RAM, 14 GB disk (77 % used after app install, ~3 GB free) |
| Robot SDK | `/venvs/mini_daemon` (daemon) and `/venvs/apps_venv` (apps) both **reachy_mini 1.10.0**. App venv was manually aligned to 1.10.0 after pip initially resolved 1.8.0 |
| Robot app install | `~/reachy_mini_conversation_app_local` (clone of `reachy_tests`), editable install in `apps_venv`, **CPU-only torch 2.14.0+cpu** (PyPI's aarch64 torch pulls ~2 GB of nvidia CUDA packages — avoid) |
| Robot `.env` | LLM → GB10 (`LLM_PROVIDER=lmstudio`, `LMSTUDIO_ENDPOINT=http://192.168.0.25:8085/v1`), `distil-small.en`, console mode. ⚠ `LMSTUDIO_MODEL` still names the old BF16 file; update it (robot was offline when the Mac `.env` was switched) |

### Changes committed on `reachy_tests` (oldest → newest)

1. `ef12526` — Add AGENTS.md
2. `aaebc44` — Regenerate uv.lock to match pyproject (stale lock carried `openai`, lacked local-audio deps)
3. `f5c7183` — Add missing `openai` dependency (used as the OpenAI-compatible client for Ollama/LM Studio/llama.cpp)
4. `7bd0b35` — Replace stale upstream `start_up` test (hung the suite: local mode never returns)
5. `4173c6b` — Preflight check: daemon reachability + USB detection + actionable checklist; 4 new tests
6. `7fe1756` — README: simulator/wired/wireless quickstart, llama.cpp LLM example, serial-port troubleshooting
7. `66c9889` — Disable Qwen3 thinking mode in local LLM calls (`LLM_DISABLE_THINKING`, default on). **Critical**: without it Qwen3 burns the 512-token budget on hidden reasoning and returns empty content → robot never speaks
8. `f9a02c0` — 30 s timeout for the robot daemon handshake (SDK default 5 s is too tight on the CM4)
9. `82078c9` — Tolerate `DaemonStatus` object from `client.get_status()` (SDK 1.9+ no longer returns a dict)

## Session 2026-09-30 (cleanup before robot testing)

Changes (see git log on `reachy_tests` for the commit):

- **Deps:** `reachy_mini~=1.10.0` (match the robot daemon), `gradio==5.23.1` (the only gradio that works with reachy_mini 1.10's pydantic>=2.12.5 *and* fastrtc's gradio<6), `requires-python>=3.11`, `fastrtc[tts]` (kokoro-onnx was never declared, so Kokoro TTS couldn't load from a clean install). Knock-on: huggingface-hub 1.x, transformers 5.x. The Mac venv is now synced to SDK 1.10.0 (`uv sync --inexact` keeps the manually installed MuJoCo extra).
- **ASR:** resample to 16 kHz with scipy before Whisper. The transformers pipeline otherwise needs torchaudio, which isn't installed.
- **OpenAI key removed everywhere.** Console mode used to crash on `config.OPENAI_API_KEY` (an attribute that doesn't exist); it also tried to claim a key from a HuggingFace Space and could block forever waiting for one. Gradio telemetry is now off by default (`GRADIO_ANALYTICS_ENABLED=False`).
- **Half-duplex mic mute:** mic input is ignored from the end of an utterance until TTS playback finishes plus `MIC_UNMUTE_DELAY` (default 0.6 s), so the robot doesn't answer itself.
- **Dead code removed:** OpenAI realtime session, Chatterbox TTS, external Gradio ASR, external smart-turn VAD (`LOCAL_VAD_ENDPOINT`, `vad_server_layout.md`), `FULL_LOCAL_MODE`, `ONNX_PROVIDERS`. `openai_realtime.py` went from 1428 to ~300 lines; the class name is kept.
- **Verified on the Mac:** 14 tests pass, ruff clean, Kokoro→Whisper round trip, the app in `--gradio` mode against the 1.10.0 sim daemon (UI served), and a GB10 reply through `_generate_local_response` (2.7 s, in persona, thinking disabled). **Not verified:** console mode on real hardware.

### What is left to do

1. **Robot bring-up:** on the robot, first point `.env` at the new model: `sed -i 's|^LMSTUDIO_MODEL=.*|LMSTUDIO_MODEL=/home/sral/models/hf/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/4ca720788d1e01f1bff70c033e0d0028fd02e502/Qwen3.8-27B-UD-Q4_K_XL.gguf|' ~/reachy_mini_conversation_app_local/.env`. Then `git pull`, then reinstall so the new deps land: `/venvs/apps_venv/bin/pip install -e .` (expect transformers 5 / hf-hub 1.x / kokoro-onnx; keep CPU-only torch; watch the ~3 GB free disk). Then `/venvs/apps_venv/bin/reachy-mini-conversation-app --no-camera`. The OpenAI-key crash is fixed, so the next failure points are `robot.media` on 1.10.0 and model download/RAM on the CM4.
2. **First real conversation test** (mic/speaker via GStreamer `.asoundrc`). Tune `VAD_ENERGY_THRESHOLD` against fan/motor noise and `MIC_UNMUTE_DELAY` if the robot still hears itself. Then try the camera (`media.get_frame()` in `camera_worker.py` is unverified on 1.10.0).
3. **Latency (less urgent now):** at ~20 tok/s a typical short reply takes ~3 s, but the reply still isn't streamed and `max_tokens=512` allows ~25 s worst case. Next: stream the LLM and speak sentence by sentence, lower `max_tokens`, and set an explicit client timeout (the OpenAI client defaults to 600 s with 2 retries).
4. **Remaining debt:** no tool calling on the local path (dance/emotion tools are unused; `core_tools.get_tool_specs`/`dispatch_tool_call` are kept for it). The personality UIs still offer OpenAI voice names and write `voice.txt`, which nothing reads (could map to `KOKORO_VOICE`). No barge-in (interrupting the robot). `docs/scheme.mmd` is stale. Consider renaming `OpenaiRealtimeHandler`/`openai_realtime.py`. 10 mypy errors remain (none are new kinds).
5. **Optional later (Path 2):** run the app on the Mac against the robot over WiFi. SDK 1.10 has `connection_mode="network"`/auto-detect + mDNS (`reachy-mini.local`); the Mac venv is now on 1.10.0 too.

### Gotchas learned (don't rediscover)

- `pkill -f "pip install"` over ssh kills your own session (pattern matches the sshd command line) — use `ps`/pids.
- Backgrounding over ssh: use `nohup ... > log 2>&1 < /dev/null &`; sessions otherwise linger.
- The robot daemon's `--no-localhost-only` flag **does not exist** in 1.10.0 (connection modes are client-side: `connection_mode="auto"|"localhost_only"|"network"`).
- Daemon health: `http://192.168.0.19:8000/` (dashboard) + `ws://…/ws/sdk` streams ~150 msg/s (imu/joint/head_pose).
- Qwen3: always send `chat_template_kwargs: {"enable_thinking": false}` for chat use.
