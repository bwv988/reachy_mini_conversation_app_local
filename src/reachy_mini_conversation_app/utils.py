import time
import logging
import argparse
import warnings
from typing import Any, Tuple

from reachy_mini import ReachyMini
from reachy_mini_conversation_app.camera_worker import CameraWorker


def parse_args() -> Tuple[argparse.Namespace, list]:  # type: ignore
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("Reachy Mini Conversation App")
    parser.add_argument(
        "--head-tracker",
        choices=["yolo", "mediapipe", None],
        default=None,
        help="Choose head tracker (default: None)",
    )
    parser.add_argument("--no-camera", default=False, action="store_true", help="Disable camera usage")
    parser.add_argument(
        "--local-vision",
        default=False,
        action="store_true",
        help="Use local vision model instead of gpt-realtime vision",
    )
    parser.add_argument("--gradio", default=False, action="store_true", help="Open gradio interface")
    parser.add_argument("--debug", default=False, action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--wireless-version",
        default=False,
        action="store_true",
        help="Use WebRTC backend for wireless version of the robot",
    )
    parser.add_argument(
        "--on-device",
        default=False,
        action="store_true",
        help="Use when conversation app is running on the same device as Reachy Mini daemon",
    )
    parser.add_argument(
        "--robot-host",
        default="reachy-mini.local",
        help="Hostname or IP of the wireless robot's daemon, used with --wireless-version from another machine "
        "(default: reachy-mini.local; use the IP if mDNS doesn't resolve)",
    )
    return parser.parse_known_args()


def handle_vision_stuff(args: argparse.Namespace, current_robot: ReachyMini) -> Tuple[CameraWorker | None, Any, Any]:
    """Initialize camera, head tracker, camera worker, and vision manager.

    By default, vision is handled by gpt-realtime model when camera tool is used.
    If --local-vision flag is used, a local vision model will process images periodically.
    """
    camera_worker = None
    head_tracker = None
    vision_manager = None

    if not args.no_camera:
        # Initialize head tracker if specified
        if args.head_tracker is not None:
            if args.head_tracker == "yolo":
                from reachy_mini_conversation_app.vision.yolo_head_tracker import HeadTracker

                head_tracker = HeadTracker()
            elif args.head_tracker == "mediapipe":
                from reachy_mini_toolbox.vision import HeadTracker  # type: ignore[no-redef]

                head_tracker = HeadTracker()

        # Initialize camera worker
        camera_worker = CameraWorker(current_robot, head_tracker)

        # Initialize vision manager only if local vision is requested
        if args.local_vision:
            try:
                from reachy_mini_conversation_app.vision.processors import initialize_vision_manager

                vision_manager = initialize_vision_manager(camera_worker)
            except ImportError as e:
                raise ImportError(
                    "To use --local-vision, please install the extra dependencies: pip install '.[local_vision]'",
                ) from e
        else:
            logging.getLogger(__name__).info(
                "Using gpt-realtime for vision (default). Use --local-vision for local processing.",
            )

    return camera_worker, head_tracker, vision_manager


# USB VID:PID of the wired Reachy Mini kit's USB-UART bridge
REACHY_SERIAL_VID_PID = "1A86:55D3"


def find_reachy_usb_ports() -> list[str]:
    """Find wired Reachy Mini serial ports on this machine (VID:PID 1A86:55D3)."""
    try:
        import serial.tools.list_ports
    except ImportError:  # pragma: no cover - pyserial ships with the SDK
        return []
    return [
        port.device
        for port in serial.tools.list_ports.comports()
        if f"USB VID:PID={REACHY_SERIAL_VID_PID}" in (port.hwid or "")
    ]


def daemon_reachable(port: int = 8000, timeout: float = 1.0, host: str = "localhost") -> bool:
    """Check whether a Reachy Mini daemon is serving HTTP on host:port."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/", timeout=timeout):
            return True
    except Exception:
        return False


def preflight_check(
    args: argparse.Namespace,
    logger: logging.Logger,
    retries: int = 5,
    daemon_port: int = 8000,
) -> bool:
    """Verify the robot daemon is reachable and log actionable guidance if not.

    In remote wireless mode the daemon runs on the robot's RPi, so it is
    checked at ``args.robot_host`` instead of localhost.

    Returns:
        True if startup can proceed, False if the daemon is unreachable.

    """
    if args.wireless_version and not args.on_device:
        host = args.robot_host
        if daemon_reachable(port=daemon_port, host=host, timeout=5.0):
            logger.info(f"Reachy Mini daemon reachable on {host}:{daemon_port}")
            return True
        logger.error(
            f"Could not reach the robot's daemon at {host}:{daemon_port}.\n"
            "  - Is the robot powered on and on the same network as this machine?\n"
            "  - Is the daemon running on the robot (reachy-mini-daemon --wireless-version)?\n"
            "  - If the hostname doesn't resolve, pass the robot's IP: --robot-host 192.168.x.y"
        )
        return False

    for attempt in range(1, retries + 1):
        if daemon_reachable(port=daemon_port):
            logger.info(f"Reachy Mini daemon reachable on localhost:{daemon_port}")
            return True
        logger.warning(
            f"Daemon not reachable on localhost:{daemon_port} "
            f"(attempt {attempt}/{retries}), retrying in 1s...",
        )
        time.sleep(1)

    logger.error("=" * 60)
    logger.error(f"Could not reach a Reachy Mini daemon on localhost:{daemon_port}.")
    logger.error("Checklist:")
    usb_ports = find_reachy_usb_ports()
    if usb_ports:
        logger.error(f"  - Reachy Mini USB device found: {usb_ports[0]}")
        logger.error("    Start the daemon: reachy-mini-daemon "
                     "(add --serialport if several devices are connected).")
    else:
        logger.error(f"  - No Reachy Mini USB device ({REACHY_SERIAL_VID_PID}) detected.")
        logger.error("    * Is the robot fully powered on (standby is not enough)?")
        logger.error("    * Is it connected to this machine by USB cable?")
        logger.error("      (a Mac daemon cannot drive the robot over WiFi alone)")
        logger.error("    * Is the Reachy desktop app closed? It holds the serial port.")
        logger.error("  - Wireless kit: run the daemon on the robot's RPi")
        logger.error("    (reachy-mini-daemon --wireless-version) and start this app")
        logger.error("    with --wireless-version.")
        logger.error("  - Simulator: reachy-mini-daemon --sim")
    logger.error("=" * 60)
    return False


def setup_logger(debug: bool) -> logging.Logger:
    """Setups the logger."""
    log_level = "DEBUG" if debug else "INFO"
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s:%(lineno)d | %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Suppress WebRTC warnings
    warnings.filterwarnings("ignore", message=".*AVCaptureDeviceTypeExternal.*")
    warnings.filterwarnings("ignore", category=UserWarning, module="aiortc")

    # Tame third-party noise (looser in DEBUG)
    if log_level == "DEBUG":
        logging.getLogger("aiortc").setLevel(logging.INFO)
        logging.getLogger("fastrtc").setLevel(logging.INFO)
        logging.getLogger("aioice").setLevel(logging.INFO)
        logging.getLogger("openai").setLevel(logging.INFO)
        logging.getLogger("websockets").setLevel(logging.INFO)
    else:
        logging.getLogger("aiortc").setLevel(logging.ERROR)
        logging.getLogger("fastrtc").setLevel(logging.ERROR)
        logging.getLogger("aioice").setLevel(logging.WARNING)
    return logger
