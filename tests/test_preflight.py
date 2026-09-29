"""Tests for the robot daemon preflight check."""

import socket
import logging
import argparse
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

from reachy_mini_conversation_app.utils import (
    preflight_check,
    daemon_reachable,
    find_reachy_usb_ports,
)


def _free_port() -> int:
    """Get a free localhost port for a throwaway test server."""
    with socket.socket() as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


class _Handler(BaseHTTPRequestHandler):
    """Minimal 200-OK handler mimicking the daemon's root endpoint."""

    def do_GET(self) -> None:
        """Return 200 for any GET."""
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        """Silence request logging."""


def test_wireless_remote_mode_skips_local_daemon_check() -> None:
    """Remote wireless mode expects the daemon on the robot, not localhost."""
    args = argparse.Namespace(wireless_version=True, on_device=False)
    assert preflight_check(args, logging.getLogger("test"), retries=1, daemon_port=_free_port()) is True


def test_unreachable_daemon_returns_false() -> None:
    """No daemon listening -> preflight fails so the app can explain why."""
    args = argparse.Namespace(wireless_version=False, on_device=False)
    assert preflight_check(args, logging.getLogger("test"), retries=1, daemon_port=_free_port()) is False


def test_reachable_daemon_returns_true() -> None:
    """A listening daemon (any 200) lets startup proceed."""
    port = _free_port()
    server = HTTPServer(("localhost", port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        assert daemon_reachable(port=port) is True
        args = argparse.Namespace(wireless_version=False, on_device=False)
        assert preflight_check(args, logging.getLogger("test"), retries=1, daemon_port=port) is True
    finally:
        server.shutdown()


def test_find_reachy_usb_ports_returns_list() -> None:
    """USB enumeration must not raise, whatever is plugged in."""
    assert isinstance(find_reachy_usb_ports(), list)
