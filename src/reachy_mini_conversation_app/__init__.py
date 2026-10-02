"""Nothing (for ruff)."""

import os

from dotenv import find_dotenv, load_dotenv


# Load .env before anything imports gradio / huggingface_hub / transformers: those read
# settings such as HF_HUB_OFFLINE, HF_HOME or GRADIO_ANALYTICS_ENABLED once, at import
# time, so loading .env later (in config.py) would come too late for them.
_dotenv_path = find_dotenv(usecwd=True)
if _dotenv_path:
    load_dotenv(dotenv_path=_dotenv_path, override=True)

# Local-only app: keep Gradio from phoning home (telemetry, version checks).
# Must be set before gradio is imported; an explicit env/.env setting still wins.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
