"""Tool calling on the local LLM path, including camera images and server-capability fallbacks."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import numpy as np
import pytest
from openai import BadRequestError, InternalServerError

import reachy_mini_conversation_app.openai_realtime as rt_mod
from reachy_mini_conversation_app.tools.camera import Camera
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


def _handler(camera_worker: object | None = None) -> rt_mod.OpenaiRealtimeHandler:
    movement_manager = MagicMock()
    movement_manager.is_playing_move.return_value = False
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=movement_manager, camera_worker=camera_worker)
    handler = rt_mod.OpenaiRealtimeHandler(deps)
    handler._models_ready = True
    handler._local_llm_client = MagicMock()
    return handler


def _msg(content: str | None = None, calls: list[tuple[str, str, dict]] | None = None) -> SimpleNamespace:
    tool_calls = [
        SimpleNamespace(id=cid, function=SimpleNamespace(name=name, arguments=json.dumps(args)))
        for cid, name, args in (calls or [])
    ]
    return SimpleNamespace(content=content, tool_calls=tool_calls or None)


def _resp(message: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _bad_request(text: str) -> BadRequestError:
    request = httpx.Request("POST", "http://llm/v1/chat/completions")
    return BadRequestError(text, response=httpx.Response(400, request=request), body=None)


def _server_error(text: str) -> InternalServerError:
    request = httpx.Request("POST", "http://llm/v1/chat/completions")
    return InternalServerError(text, response=httpx.Response(500, request=request), body=None)


def _set_llm(handler: rt_mod.OpenaiRealtimeHandler, *results: object) -> AsyncMock:
    create = AsyncMock(side_effect=list(results))
    handler._local_llm_client.chat.completions.create = create
    return create


def test_tool_specs_use_chat_completions_format() -> None:
    """Specs are nested under 'function' as Chat Completions expects."""
    handler = _handler(camera_worker=MagicMock(head_tracker=None))
    names = {t["function"]["name"] for t in handler._tools}
    assert all(t["type"] == "function" and "parameters" in t["function"] for t in handler._tools)
    assert "camera" in names and "dance" in names
    assert "head_tracking" not in names  # no head tracker configured


def test_camera_tools_hidden_without_camera() -> None:
    """With --no-camera the LLM isn't offered camera/head tracking."""
    names = {t["function"]["name"] for t in _handler(camera_worker=None)._tools}
    assert "camera" not in names and "head_tracking" not in names


@pytest.mark.asyncio
async def test_plain_reply_without_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    """A normal text reply is returned as-is, with tools offered."""
    handler = _handler()
    create = _set_llm(handler, _resp(_msg("Hello there.")))
    assert await handler._run_llm_with_tools([{"role": "user", "content": "hi"}]) == "Hello there."
    assert create.call_args.kwargs["tools"] == handler._tools


@pytest.mark.asyncio
async def test_tool_call_is_dispatched_and_result_fed_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """The LLM's tool call runs, its result goes back as a tool message, then the LLM answers."""
    handler = _handler()
    dispatch = AsyncMock(return_value={"status": "queued", "move": "simple_nod"})
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", dispatch)
    create = _set_llm(handler, _resp(_msg(calls=[("c1", "dance", {"move": "simple_nod"})])), _resp(_msg("Nodding!")))

    messages = [{"role": "user", "content": "nod please"}]
    assert await handler._run_llm_with_tools(messages) == "Nodding!"

    dispatch.assert_awaited_once()
    assert dispatch.call_args.args[0] == "dance"
    second_messages = create.call_args_list[1].kwargs["messages"]
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "dance"
    assert second_messages[-1] == {"role": "tool", "tool_call_id": "c1", "content": json.dumps(dispatch.return_value)}


@pytest.mark.asyncio
async def test_camera_image_sent_to_llm_as_image(monkeypatch: pytest.MonkeyPatch) -> None:
    """A camera result's JPEG goes back as an image_url part, not as text in the tool message."""
    handler = _handler()
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", AsyncMock(return_value={"b64_im": "QUJD"}))
    create = _set_llm(
        handler, _resp(_msg(calls=[("c1", "camera", {"question": "what do you see?"})])), _resp(_msg("A cat."))
    )

    assert await handler._run_llm_with_tools([{"role": "user", "content": "look"}]) == "A cat."

    sent = create.call_args_list[1].kwargs["messages"]
    assert "QUJD" not in sent[-2]["content"]  # tool message has no base64 blob
    image_parts = [p for p in sent[-1]["content"] if p["type"] == "image_url"]
    assert image_parts[0]["image_url"]["url"] == "data:image/jpeg;base64,QUJD"


@pytest.mark.asyncio
async def test_image_rejected_falls_back_to_note(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without --mmproj the server rejects images; the image becomes a note and the turn still completes."""
    handler = _handler()
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", AsyncMock(return_value={"b64_im": "QUJD"}))
    create = _set_llm(
        handler,
        _resp(_msg(calls=[("c1", "camera", {"question": "?"})])),
        _server_error("image input is not supported - hint: you may need to provide the mmproj"),
        _resp(_msg("I can't see right now.")),
    )

    assert await handler._run_llm_with_tools([{"role": "user", "content": "look"}]) == "I can't see right now."
    retried = create.call_args_list[2].kwargs["messages"]
    assert retried[-1]["content"] == rt_mod.NO_VISION_NOTE
    assert handler._tools  # tools stay enabled


@pytest.mark.asyncio
async def test_tools_rejected_disables_tools_and_retries() -> None:
    """Without --jinja the server rejects tools; the app continues without them."""
    handler = _handler()
    create = _set_llm(handler, _bad_request("tools param requires --jinja flag"), _resp(_msg("Hi.")))

    assert await handler._run_llm_with_tools([{"role": "user", "content": "hi"}]) == "Hi."
    assert handler._tools == []
    assert "tools" not in create.call_args_list[1].kwargs


@pytest.mark.asyncio
async def test_last_round_forces_text_answer(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps calling tools is cut off: the final round offers no tools."""
    handler = _handler()
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", AsyncMock(return_value={"status": "ok"}))
    loop_call = _resp(_msg(calls=[("c", "do_nothing", {})]))
    create = _set_llm(handler, *([loop_call] * rt_mod.MAX_TOOL_ROUNDS), _resp(_msg("Done.")))

    assert await handler._run_llm_with_tools([{"role": "user", "content": "x"}]) == "Done."
    assert "tools" not in create.call_args_list[-1].kwargs


@pytest.mark.asyncio
async def test_history_keeps_tool_calls_but_not_images(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tool calls stay in history (so the model keeps using the camera); images don't."""
    handler = _handler()
    monkeypatch.setattr(handler, "_synthesize_locally", AsyncMock())
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", AsyncMock(return_value={"b64_im": "QUJD"}))
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "sys")
    _set_llm(handler, _resp(_msg(calls=[("c1", "camera", {"question": "?"})])), _resp(_msg("A mug.")))

    await handler._generate_local_response("what's on the desk?")

    (turn,) = handler._conversation_history
    assert [m["role"] for m in turn] == ["user", "assistant", "tool", "assistant"]
    assert turn[1]["tool_calls"][0]["function"]["name"] == "camera"
    assert rt_mod.IMAGE_EXPIRED in turn[2]["content"]
    assert "QUJD" not in json.dumps(turn)
    assert turn[-1] == {"role": "assistant", "content": "A mug."}


@pytest.mark.asyncio
async def test_next_turn_sees_previous_camera_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """The follow-up request includes the earlier camera call, but no image data."""
    handler = _handler()
    monkeypatch.setattr(handler, "_synthesize_locally", AsyncMock())
    monkeypatch.setattr(rt_mod, "dispatch_tool_call", AsyncMock(return_value={"b64_im": "QUJD"}))
    monkeypatch.setattr(rt_mod, "get_session_instructions", lambda: "sys")
    create = _set_llm(
        handler,
        _resp(_msg(calls=[("c1", "camera", {"question": "?"})])),
        _resp(_msg("Five fingers.")),
        _resp(_msg("Still five.")),
    )

    await handler._generate_local_response("how many fingers?")
    await handler._generate_local_response("and now?")

    followup = create.call_args_list[2].kwargs["messages"]
    assert any(m.get("tool_calls") for m in followup)
    assert "QUJD" not in json.dumps(followup)
    assert followup[-1] == {"role": "user", "content": "and now?"}


def test_history_is_trimmed_to_whole_turns() -> None:
    """Old turns drop out whole: a tool result never loses its tool call."""
    handler = _handler()
    call_turn = [
        {"role": "user", "content": "look"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c", "type": "function", "function": {}}]},
        {"role": "tool", "tool_call_id": "c", "content": "{}"},
        {"role": "assistant", "content": "ok"},
    ]
    handler._conversation_history = [call_turn] * (rt_mod.MAX_HISTORY_TURNS + 5)
    prior = [m for t in handler._conversation_history[-rt_mod.MAX_HISTORY_TURNS :] for m in t]
    assert prior[0]["role"] == "user"
    assert len(prior) == 4 * rt_mod.MAX_HISTORY_TURNS


def test_mic_muted_while_move_plays() -> None:
    """Dance/emotion motor noise can't trigger a turn."""
    handler = _handler()
    assert not handler._mic_muted()
    handler.deps.movement_manager.is_playing_move.return_value = True
    assert handler._mic_muted()


@pytest.mark.asyncio
async def test_camera_tool_downscales_frame() -> None:
    """Frames are shrunk to MAX_IMAGE_WIDTH before being sent to the LLM."""
    import base64

    import cv2

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    worker = MagicMock()
    worker.get_latest_frame.return_value = frame
    deps = ToolDependencies(reachy_mini=MagicMock(), movement_manager=MagicMock(), camera_worker=worker)

    result = await Camera()(deps, question="what do you see?")
    img = cv2.imdecode(np.frombuffer(base64.b64decode(result["b64_im"]), np.uint8), cv2.IMREAD_COLOR)
    assert img.shape == (360, 640, 3)


@pytest.mark.asyncio
async def test_unrelated_error_does_not_disable_tools() -> None:
    """A transient server error propagates instead of silently turning tools off."""
    handler = _handler()
    _set_llm(handler, _server_error("slot unavailable"))
    with pytest.raises(InternalServerError):
        await handler._run_llm_with_tools([{"role": "user", "content": "hi"}])
    assert handler._tools
