"""Nothing (for ruff)."""

import os


# Local-only app: keep Gradio from phoning home (telemetry, version checks).
# Must be set before gradio is imported; an explicit env setting still wins.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
