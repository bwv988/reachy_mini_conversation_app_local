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
| GB10 (GPU server) | `192.168.0.25:8085`, llama.cpp serving **Qwen3.8-27B BF16** (unsloth GGUF, model id = full file path). ~4 tok/s. **Also serves this Pi harness** — expect shared latency |
| Robot (wireless) | `192.168.0.19` = `reachy-mini.local`, user `pollen` (pw `root`). SSH key of the Mac (`sral@andromeda.local`) installed in `~/.ssh/authorized_keys`. CM4, 4 GB RAM, 14 GB disk (77 % used after app install, ~3 GB free) |
| Robot SDK | `/venvs/mini_daemon` (daemon) and `/venvs/apps_venv` (apps) both **reachy_mini 1.10.0**. App venv was manually aligned to 1.10.0 after pip initially resolved 1.8.0 |
| Robot app install | `~/reachy_mini_conversation_app_local` (clone of `reachy_tests`), editable install in `apps_venv`, **CPU-only torch 2.14.0+cpu** (PyPI's aarch64 torch pulls ~2 GB of nvidia CUDA packages — avoid) |
| Robot `.env` | LLM → GB10 (`LLM_PROVIDER=lmstudio`, `LMSTUDIO_ENDPOINT=http://192.168.0.25:8085/v1`), `distil-small.en`, console mode |

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

### What is left to do

1. **Finish the robot bring-up (where we stopped):** on the robot, `cd ~/reachy_mini_conversation_app_local && git pull` (last launch raced the pull and ran pre-fix code) then relaunch: `/venvs/apps_venv/bin/reachy-mini-conversation-app --no-camera`. Next expected failure points: `robot.media` API usage in `console.py`/shutdown paths on SDK 1.10.0 — check the log, patch, repeat.
2. **First real conversation test** on the robot (mic/speaker via GStreamer `.asoundrc`); then try camera (`media.get_frame()` in `camera_worker.py` unverified on 1.10.0).
3. **Pin/resolve the SDK version skew**: pip resolved `reachy_mini` 1.8.0 for the app install; we force-aligned the robot venv to 1.10.0. Decide: pin `reachy_mini` in `pyproject.toml` or document "match the robot's daemon version".
4. **Housekeeping debt** (known, not started): 29 pre-existing ruff errors in `local_audio.py`/`openai_realtime.py` (CI lint fails); `gradio==5.50.1.dev1` dev-pin causes a pydantic conflict warning with reachy_mini 1.10.0; `docs/scheme.mmd` + README "OPENAI API Key" UI are stale cloud-era artifacts; local LLM path is non-streaming (full-reply latency) and has no tool calling (dance/emotion tools dead on the local path).
5. **Optional later (Path 2):** run the app on the Mac against the robot over WiFi — SDK 1.10 has `connection_mode="network"`/auto-detect + mDNS (`reachy-mini.local`); the Mac's venv still has SDK 1.2.3rc1 and would need upgrading to match.

### Gotchas learned (don't rediscover)

- `pkill -f "pip install"` over ssh kills your own session (pattern matches the sshd command line) — use `ps`/pids.
- Backgrounding over ssh: use `nohup ... > log 2>&1 < /dev/null &`; sessions otherwise linger.
- The robot daemon's `--no-localhost-only` flag **does not exist** in 1.10.0 (connection modes are client-side: `connection_mode="auto"|"localhost_only"|"network"`).
- Daemon health: `http://192.168.0.19:8000/` (dashboard) + `ws://…/ws/sdk` streams ~150 msg/s (imu/joint/head_pose).
- Qwen3: always send `chat_template_kwargs: {"enable_thinking": false}` for chat use.
