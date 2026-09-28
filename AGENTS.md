# Reachy Mini Conversation App — Agent Guide

This is a fully local, on-robot **Python app** for Reachy Mini (speech-to-text, local LLM, TTS, and choreographed motion). It is exactly the kind of "developer tool / on-robot control loop / offline LAN" app for which the upstream [pollen-robotics/reachy_mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md) recommends the **Python path** over a JS/web app. Read that file for the full Reachy Mini SDK and app-development conventions; this file only covers what's specific to this repo.

## Quick orientation

- Entry points: `reachy-mini-conversation-app` (console) and `--gradio` (web UI, required when using the simulator).
- Source lives in `src/reachy_mini_conversation_app/`:
  - `main.py` / `console.py` — app bootstrap and CLI
  - `local_audio.py`, `openai_realtime.py` — STT/TTS and LLM audio pipeline
  - `moves.py`, `dance_emotion_moves.py` — motion/choreography
  - `camera_worker.py`, `vision/` — face tracking / vision
  - `tools/` — LLM-callable tools
  - `profiles/` — personality profiles (prompts + identity per character)
  - `prompts/` — behavior and identity prompt fragments
- Config is via `.env` (see `.env.example`, `.env.jetson` for Jetson Nano tuning) and `config.py`.
- Tests live in `tests/`.

## Setup

```bash
pip install -e "."          # base install
pip install -e ".[jetson]"  # Jetson Nano CUDA optimization
cp .env.example .env
```

Requires the [Reachy Mini SDK](https://github.com/pollen-robotics/reachy_mini) to be installed separately (works with real hardware or the simulator). Requires a local LLM backend: Ollama (recommended) or LM Studio.

## Running

```bash
reachy-mini-conversation-app            # console mode (headless)
reachy-mini-conversation-app --gradio   # web UI, required for the simulator
```

Useful flags: `--head-tracker {yolo,mediapipe}`, `--local-vision`, `--no-camera`, `--wireless-version` (GStreamer, for wireless robots).

## Conventions to follow

- **Motion**: use `goto_target()` for smooth gestures (≥0.5s), `set_target()` only for real-time control loops (10Hz+) — per the upstream SDK guide.
- **Safety limits**: head pitch/roll ±40°, head yaw ±180°, body yaw ±160°, max 65° yaw delta between head and body. The SDK clamps automatically but don't fight it.
- **Personality profiles**: new characters/behaviors go under `src/reachy_mini_conversation_app/profiles/<name>/` following the existing profile layout (see `profiles/example` or `profiles/default`), with prompt fragments in `prompts/`.
- **Local-only**: this app is designed to run with no cloud dependencies. Avoid introducing hard dependencies on external/cloud APIs; keep STT/TTS/LLM swappable via `.env` config (`LLM_PROVIDER`, `OLLAMA_MODEL`, `DISTIL_WHISPER_MODEL`, `KOKORO_VOICE`, etc.).
- **Jetson support**: keep Jetson-specific tuning in `.env.jetson` / the `jetson` extra rather than hardcoding device-specific behavior into core logic.
- Lint with `ruff` (pinned in `pyproject.toml` dev group); run `pytest` for `tests/`.

## Before making SDK-level changes

If a change touches robot motion, the daemon REST API, or the JS/host-shell surface (not currently used by this app), consult the upstream [reachy_mini AGENTS.md](https://github.com/pollen-robotics/reachy_mini/blob/main/AGENTS.md) and its linked `skills/` docs (`motion-philosophy.md`, `safe-torque.md`, `ai-integration.md`, `control-loops.md`) rather than guessing at conventions.
