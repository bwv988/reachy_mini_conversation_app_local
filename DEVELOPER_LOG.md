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
| Robot app install | `~/reachy_mini_conversation_app_local` (clone of `reachy_tests`), editable install in `apps_venv`, **CPU-only torch 2.14.0+cpu** (PyPI's aarch64 torch pulls ~2 GB of nvidia CUDA packages — avoid). The clone has `git lfs install --local --skip-smudge`, see Gotchas |
| Robot `.env` | LLM → GB10 (`LLM_PROVIDER=lmstudio`, `LMSTUDIO_ENDPOINT=http://192.168.0.25:8085/v1`, `LMSTUDIO_MODEL=…/Qwen3.8-27B-UD-Q4_K_XL.gguf`), `distil-small.en`, `MIC_UNMUTE_DELAY=0.6`, console mode. Backup of the pre-Q4 file: `~/env_backup_20260930` |

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

### Robot bring-up, part 1 (2026-09-30 afternoon)

- The robot repo was broken: `.git/index` was missing and the worktree was a half-checkout of `ef12526`. **Root cause: the fork's GitHub LFS store has no objects** (404), so every checkout died on the first LFS file. Fixed with `git lfs install --local --skip-smudge` plus a reset to `origin/reachy_tests` (worktree backup: `~/conv_app_worktree_backup_20260930.tgz`). The LFS files (README gif/svg, gradio avatar PNGs) are pointer stubs on the robot; the robot doesn't need them.
- `pip install -e .` in `apps_venv` added gradio 5.23.1 and kokoro-onnx; `pip check` is clean (the pydantic conflict is gone). Versions: reachy_mini 1.10.0, transformers 5.3.0, torch 2.14.0+cpu. Disk 2.4 GB free after the model downloads.
- Verified on the robot: GB10 reachable (reply in 2.4 s); GStreamer media backend records 16 kHz stereo (ambient peak ~0.02).
- **CM4 is too slow for on-robot ASR/TTS** (4 cores, not throttled, torch 4 threads). Warm Kokoro: 2.8 s of audio in 18.9 s (~6× slower than real time). Warm distil-small.en: 2.8 s clip in 49.9 s (~18×). One turn ≈ 75–80 s. First load: TTS 76 s, ASR 117 s.

### First on-robot run (2026-09-30 ~16:10): findings and fixes

What the run showed:
- **VAD triggered on motor noise.** `speech started` fired 27 ms after session start; the "utterance" transcribed as "Oh." and the robot replied to it. The source was the idle `BreathingMove` (head bob + antenna sway every time the robot was idle for 0.3 s) plus the energy VAD at 0.01.
- **Model loading blocked the event loop.** ASR/TTS loaded lazily on the first turn *on the event loop*: a 43 s gap after `speech ended` and a 32 s Kokoro load, producing GStreamer `Can't record audio fast enough … Dropped samples` warnings.
- **HF 404/307/302 lines are harmless.** They are cache revalidation (optional files 404, CDN redirects), not re-downloads; `HF_HUB_OFFLINE=1` silences them once models are cached.

Fixes:
- `IDLE_BREATHING` setting, **default off** (`MovementManager._manage_breathing` is guarded). The head wobble while speaking is unchanged (the mic is muted then).
- ASR/TTS are preloaded in `start_up()` via `run_in_executor`; the mic stays muted until `_models_ready`.
- Running from the Mac: new `--robot-host` flag (default `reachy-mini.local`; mDNS doesn't resolve from the Mac, so use `192.168.0.19`). Remote mode uses `ReachyMini(host=…, connection_mode="network", media_backend="webrtc")`, and preflight now checks `http://<robot-host>:8000`. On-device mode uses `media_backend="local"` (`"gstreamer"` is deprecated in 1.10).
- **Verified from the Mac:** a network connection to the robot daemon in 0.5 s, and WebRTC mic audio from the robot at 16 kHz stereo (peak 0.27 in that sample, possibly speech). Not yet verified: a full conversation from the Mac.

### ASR upgrade on the Mac (2026-09-30 evening)

- Kokoro has only one size (82M), so there is no bigger TTS; "large" applies to Whisper. Benchmark on the M4 Pro with a 5.5 s utterance (warm): distil-medium.en CPU 0.87 s; distil-large-v3 CPU 1.45 s; **distil-large-v3 MPS float32 1.07 s**; MPS **float16 produces garbage** (repeated `____` or `!`).
- `LocalASR` device `auto` now tries CUDA → MPS → CPU (`resolve_asr_device`), with MPS forced to float32. The Mac `.env` now uses `DISTIL_WHISPER_MODEL=distil-whisper/distil-large-v3`.
- Note: `WHISPER_LANGUAGE` is passed to `LocalASR`, but distil-whisper-fastrtc ignores it. large-v3 is multilingual and auto-detects the language, while Kokoro voices are per-language.

### Tool calling + camera (2026-09-30 night)

- **Tool calling on the local path:** `_run_llm_with_tools` sends the profile's tools (converted to Chat Completions format; `camera`/`head_tracking` are hidden when unusable), runs calls through `dispatch_tool_call`, and loops for at most `MAX_TOOL_ROUNDS=3` (the last round offers no tools, forcing a spoken answer). Only the spoken reply is kept in history.
- **Camera:** the tool result's JPEG (now downscaled to 640 px wide, quality 85; SDK frames are BGR) goes back to the LLM as an `image_url` part in a follow-up user message. Frames arrive on the Mac over WebRTC at 1280×720, ~31 fps.
- **Fallbacks:** llama-server without `--mmproj` answers **HTTP 500** "image input is not supported", so any API error while an image is attached is replaced with a note and retried (the robot says it can't see). Tool rejection (without `--jinja`) disables tools for the session, but only when the error mentions tools/jinja.
- The mic is also muted while a dance/emotion plays (`MovementManager.is_playing_move()`, breathing excluded).
- **Live-tested against the GB10:** a dance tool call works on the current server (reply in 8.5 s, two LLM rounds). Vision is **pending**: the GB10 must load `mmproj-F16.gguf` (the repo has it; the model is multimodal). Download: `HF_HOME=/home/sral/models/hf hf download unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf --revision 4ca720788d1e01f1bff70c033e0d0028fd02e502`, then `--mmproj <snapshot>/mmproj-F16.gguf`.

### Whisper loader, activation by name (2026-09-30 late)

- **Vision "sometimes works":** the camera pipeline was fine; the model sometimes didn't *call* the tool (it called `play_emotion` instead and said "give me a second to check" without looking). Fixed by the user's prompt edit plus a much more directive `camera` tool description ("call it again for every new visual question…").
- **Replaced `distil-whisper-fastrtc`** with a direct transformers loader in `LocalASR` (the dependency is removed). Any HF Whisper checkpoint works; `WHISPER_LANGUAGE` is now honoured for multilingual models (`auto` detects); >30 s audio uses long-form decoding (`return_timestamps=True`). The deprecation warnings are gone: stale `forced_decoder_ids`/suppress tokens are cleared from the model config at load, and one warning that transformers' own Whisper `generate()` causes is filtered narrowly. Faster on the Mac (MPS): medium.en 0.42 s, large-v3 0.76 s for a 3 s clip.
- **Activation by name** (`wake_word.py`, `WAKE_MODE=name`, `WAKE_WORDS=tom`, `WAKE_WINDOW_S=25`). Transcript-based gate between ASR and LLM. Whole-word matching; fuzzy only for names of 5+ letters. The window restarts at each accepted utterance and at the end of each reply's playback. Every transcript is logged with its verdict. Mac `.env`: `WAKE_MODE=name`, `WAKE_WORDS=tom`. Robot renamed **Tom**.
- **Live end-to-end test** (synthesized speech through VAD → Whisper large-v3 → gate → Qwen + camera, mocked motion): no name → ignored; "Tom, how many fingers…" → camera called; a follow-up without the name → accepted and the camera called again.
- Follow-ups for activation: body language for engaged/dormant (the antennas currently react to *any* speech via `set_listening`, even when it is then ignored); optional Whisper name hint (risk: Whisper echoing the prompt on noise); openWakeWord if transcript gating proves too slow.

### Follow-up camera questions (2026-09-30 night)

- **Symptom:** after one successful camera answer, "And how many fingers is now, Tom?" got "I don't have a camera tool". Reproduced 8/8 against the GB10 with tools sent in every request.
- **Cause:** the history kept only spoken replies, so the model saw its earlier visual answer with *no* camera call behind it, imitated that, and rationalized it. Keeping the earlier tool call in context → camera called 4/4; asked directly, the model lists all its tools.
- **Fix:** history is now per turn: `[user, *tool exchanges, reply]`, with images dropped (the tool result says the picture is "no longer available, call the camera again to look"). Trimmed to `MAX_HISTORY_TURNS=10` whole turns, so a tool result never loses its call.
- **Verified live:** follow-up camera calls 8/8 (two phrasings × 4 fresh sessions), up from 0/8.
- Noise seen in the user's log: `phonemizer: words count mismatch` comes from Kokoro's espeak phonemizer and is harmless.

### Offline startup (2026-09-30 night)

- The HF requests at startup were cache revalidation (307→200, 404 for optional files) plus the new huggingface_hub 1.x anonymous-request warning.
- `HF_HUB_OFFLINE=1` in `.env` had **no effect**: huggingface_hub reads it once at import, and gradio imports it before `config.py` loaded `.env`. **Fix:** the package `__init__.py` now loads `.env` first, so any import-time library setting in `.env` works (`HF_HUB_OFFLINE`, `HF_HOME`, `HF_TOKEN`, `GRADIO_ANALYTICS_ENABLED`).
- The Mac `.env` has `HF_HUB_OFFLINE=1`. Verified: Whisper large-v3 + Kokoro load with zero HTTP requests, in 4.4 s (was 7–8 s). To try a new model, set it to 0 for one run.

### What is left to do

1. **Run from the Mac (preferred):** `.venv/bin/reachy-mini-conversation-app --wireless-version --robot-host 192.168.0.19 --no-camera`, with the robot's daemon running and the on-robot app stopped. On-robot fallback: `/venvs/apps_venv/bin/reachy-mini-conversation-app --wireless-version --on-device --no-camera` (~75–80 s per turn on the CM4). The robot clone needs `git pull` for today's fixes.
2. **First real conversation test** (mic/speaker via GStreamer `.asoundrc`). Tune `VAD_ENERGY_THRESHOLD` against fan/motor noise and `MIC_UNMUTE_DELAY` if the robot still hears itself. Then try the camera (`media.get_frame()` in `camera_worker.py` is unverified on 1.10.0).
3. **Latency:** with the app on the Mac, ASR/TTS take seconds (Mac: TTS ~3 s, ASR ~4.5 s for a short clip) and the LLM ~3 s. Next: stream the LLM reply and speak sentence by sentence, lower `max_tokens`, set an explicit client timeout. If the app must run on the robot: move ASR/TTS to the GB10 or use edge models (Moonshine, faster-whisper int8).
4. **Remaining debt:** tools are loaded once at startup, so switching personality at runtime doesn't change the tool set. The personality UIs still offer OpenAI voice names and write `voice.txt`, which nothing reads (could map to `KOKORO_VOICE`). No barge-in (interrupting the robot). `docs/scheme.mmd` is stale. Consider renaming `OpenaiRealtimeHandler`/`openai_realtime.py`. 10 mypy errors remain (none are new kinds).
5. **Path 2 (now the main path, see item 1):** run the app on the Mac against the robot over WiFi. SDK 1.10 has `connection_mode="network"`/auto-detect + mDNS (`reachy-mini.local`); the Mac venv is now on 1.10.0 too.

### Gotchas learned (don't rediscover)

- **Fork has no LFS objects on GitHub** (404 on smudge). Any new clone needs `GIT_LFS_SKIP_SMUDGE=1 git clone …` or `git lfs install --local --skip-smudge`. Real fix: from a machine with git-lfs, run `git lfs fetch upstream --all && git lfs push origin --all`, if upstream still has them.

- `pkill -f "pip install"` over ssh kills your own session (pattern matches the sshd command line) — use `ps`/pids.
- Backgrounding over ssh: use `nohup ... > log 2>&1 < /dev/null &`; sessions otherwise linger.
- The robot daemon's `--no-localhost-only` flag **does not exist** in 1.10.0 (connection modes are client-side: `connection_mode="auto"|"localhost_only"|"network"`).
- Daemon health: `http://192.168.0.19:8000/` (dashboard) + `ws://…/ws/sdk` streams ~150 msg/s (imu/joint/head_pose).
- Qwen3: always send `chat_template_kwargs: {"enable_thinking": false}` for chat use.
