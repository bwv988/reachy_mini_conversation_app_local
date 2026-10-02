import base64
import asyncio
import logging
from typing import Any, Dict

import cv2

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

MAX_IMAGE_WIDTH = 640


class Camera(Tool):
    """Take a picture with the camera and ask a question about it."""

    name = "camera"
    description = (
        "Take a fresh picture with your camera (your eyes) and look at it. Call this whenever the user "
        "asks about anything currently visible: what you see, who is there, counting things or fingers, "
        "colors, objects, gestures, or how someone looks. You cannot see without calling this tool, and "
        "earlier pictures are not kept, so call it again for every new visual question instead of "
        "answering from memory. Never say you will look without calling it."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "What to look for in the picture, e.g. 'How many fingers is the person holding up?'",
            },
        },
        "required": ["question"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Take a picture with the camera and ask a question about it."""
        image_query = (kwargs.get("question") or "").strip()
        if not image_query:
            logger.warning("camera: empty question")
            return {"error": "question must be a non-empty string"}

        logger.info("Tool call: camera question=%s", image_query[:120])

        # Get frame from camera worker buffer (like main_works.py)
        if deps.camera_worker is not None:
            frame = deps.camera_worker.get_latest_frame()
            if frame is None:
                logger.error("No frame available from camera worker")
                return {"error": "No frame available"}
        else:
            logger.error("Camera worker not available")
            return {"error": "Camera worker not available"}

        # Use vision manager for processing if available
        if deps.vision_manager is not None:
            vision_result = await asyncio.to_thread(
                deps.vision_manager.processor.process_image, frame, image_query,
            )
            if isinstance(vision_result, dict) and "error" in vision_result:
                return vision_result
            return (
                {"image_description": vision_result}
                if isinstance(vision_result, str)
                else {"error": "vision returned non-string"}
            )

        # Downscale before sending to the LLM: a full 720p frame costs far more image
        # tokens (and latency) than it adds for conversational questions.
        h, w = frame.shape[:2]
        if w > MAX_IMAGE_WIDTH:
            frame = cv2.resize(frame, (MAX_IMAGE_WIDTH, round(h * MAX_IMAGE_WIDTH / w)), interpolation=cv2.INTER_AREA)

        # Encode image directly to JPEG bytes without writing to file (frames are BGR, as cv2 expects)
        success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("Failed to encode frame as JPEG")

        b64_encoded = base64.b64encode(buffer.tobytes()).decode("utf-8")
        return {"b64_im": b64_encoded}
