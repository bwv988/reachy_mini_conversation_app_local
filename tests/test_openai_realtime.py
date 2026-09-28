import asyncio
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.openai_realtime import OpenaiRealtimeHandler
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _build_handler(loop: asyncio.AbstractEventLoop) -> OpenaiRealtimeHandler:
    asyncio.set_event_loop(loop)
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    return OpenaiRealtimeHandler(deps)


def test_format_timestamp_uses_wall_clock() -> None:
    """Test that format_timestamp uses wall clock time."""
    loop = asyncio.new_event_loop()
    try:
        print("Testing format_timestamp...")
        handler = _build_handler(loop)
        formatted = handler.format_timestamp()
        print(f"Formatted timestamp: {formatted}")
    finally:
        asyncio.set_event_loop(None)
        loop.close()

    # Extract year from "[YYYY-MM-DD ...]"
    year = int(formatted[1:5])
    assert year == datetime.now(timezone.utc).year

@pytest.mark.asyncio
async def test_start_up_local_mode_runs_until_shutdown() -> None:
    """In full local mode start_up enters the local-only session and exits on shutdown.

    The local-only session signals readiness via _connected_event, then idles
    until _shutdown_requested is set. This replaced the upstream test for the
    OpenAI realtime retry loop, which is unreachable in the fully-local fork.
    """
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock())
    handler = rt_mod.OpenaiRealtimeHandler(deps)

    task = asyncio.create_task(handler.start_up())
    try:
        # Session should become ready without any external connection
        await asyncio.wait_for(handler._connected_event.wait(), timeout=5.0)
        assert handler.connection is None

        # Shutdown flag should terminate the session cleanly
        handler._shutdown_requested = True
        await asyncio.wait_for(task, timeout=5.0)
    finally:
        handler._shutdown_requested = True
        if not task.done():
            task.cancel()
            await asyncio.wait([task])
