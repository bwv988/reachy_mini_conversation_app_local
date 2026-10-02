"""Idle breathing is opt-in (IDLE_BREATHING): motor noise otherwise triggers the VAD."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import reachy_mini_conversation_app.moves as moves_mod
from reachy_mini_conversation_app.moves import BreathingMove, MovementManager


def _manager() -> MovementManager:
    robot = MagicMock()
    robot.get_current_joint_positions.return_value = (None, (0.0, 0.0))
    robot.get_current_head_pose.return_value = np.eye(4)
    manager = MovementManager(current_robot=robot)
    manager.state.last_activity_time -= 10.0  # long idle
    return manager


def test_no_breathing_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With IDLE_BREATHING off, an idle robot stays still."""
    monkeypatch.setattr(moves_mod.config, "IDLE_BREATHING", False)
    manager = _manager()
    manager._manage_breathing(manager._now())
    assert not manager.move_queue
    assert not manager._breathing_active


def test_breathing_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """With IDLE_BREATHING on, an idle robot starts the breathing move."""
    monkeypatch.setattr(moves_mod.config, "IDLE_BREATHING", True)
    manager = _manager()
    manager._manage_breathing(manager._now())
    assert manager._breathing_active
    assert any(isinstance(m, BreathingMove) for m in manager.move_queue)


def test_is_playing_move_ignores_breathing() -> None:
    """Breathing doesn't count as a playing move; a queued dance does."""
    manager = _manager()
    assert not manager.is_playing_move()
    manager.state.current_move = BreathingMove(
        interpolation_start_pose=np.eye(4), interpolation_start_antennas=(0.0, 0.0), interpolation_duration=1.0
    )
    assert not manager.is_playing_move()
    manager.move_queue.append(MagicMock())
    assert manager.is_playing_move()
