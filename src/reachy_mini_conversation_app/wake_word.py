"""Activation by name: only respond to utterances that address the robot.

Works on the ASR transcript (no separate wake-word model): an utterance is accepted
if it contains one of the wake words, or if it arrives while a conversation is
already active. A conversation stays active for ``window_s`` seconds after the
robot last spoke (or after the last accepted utterance), so follow-ups don't need
the name.
"""

import re
import time
import logging
from difflib import SequenceMatcher
from dataclasses import dataclass


logger = logging.getLogger(__name__)

# Names shorter than this are matched exactly: fuzzy matching a short name like
# "tom" would also accept "to", "top", "tim"...
FUZZY_MIN_LENGTH = 5
FUZZY_THRESHOLD = 0.8


def _normalize(text: str) -> list[str]:
    """Lowercase and split into words; punctuation and apostrophes separate words ("tom's" -> "tom", "s")."""
    return re.findall(r"[a-z0-9]+", text.lower())


def find_wake_word(transcript: str, wake_words: list[str]) -> str | None:
    """Return the wake word found in the transcript, or None.

    Matches whole words only ("tom" doesn't match "tomorrow"); multi-word wake
    phrases ("hey tom") must appear as consecutive words. Names of at least
    FUZZY_MIN_LENGTH letters also match close misspellings ("reachy" ~ "reechy").
    """
    words = _normalize(transcript)
    for wake in wake_words:
        wake_tokens = _normalize(wake)
        if not wake_tokens:
            continue
        n = len(wake_tokens)
        for i in range(len(words) - n + 1):
            candidate = words[i : i + n]
            if candidate == wake_tokens:
                return wake
            joined, target = " ".join(candidate), " ".join(wake_tokens)
            if len(target) >= FUZZY_MIN_LENGTH and SequenceMatcher(None, joined, target).ratio() >= FUZZY_THRESHOLD:
                return wake
    return None


@dataclass
class WakeDecision:
    """Outcome of checking one transcript."""

    accepted: bool
    reason: str  # "wake word 'tom'", "conversation active", "no wake word", "disabled"


class WakeWordGate:
    """Decides which transcripts reach the LLM when activation by name is enabled."""

    def __init__(self, wake_words: list[str], window_s: float, enabled: bool = True):
        """Initialize the gate.

        Args:
            wake_words: Names/phrases that activate the robot (case-insensitive).
            window_s: Seconds a conversation stays active after the robot last spoke.
            enabled: When False, every transcript is accepted (activation off).

        """
        self.wake_words = [w.strip() for w in wake_words if w.strip()]
        self.window_s = window_s
        self.enabled = enabled and bool(self.wake_words)
        self._active_until = float("-inf")

    def is_active(self, now: float | None = None) -> bool:
        """Whether a conversation is currently active (follow-ups need no name)."""
        return (time.monotonic() if now is None else now) < self._active_until

    def check(self, transcript: str, now: float | None = None) -> WakeDecision:
        """Decide whether this transcript should be answered, and log the verdict."""
        now = time.monotonic() if now is None else now
        if not self.enabled:
            return WakeDecision(True, "disabled")

        wake = find_wake_word(transcript, self.wake_words)
        if wake is not None:
            decision = WakeDecision(True, f"wake word '{wake}'")
        elif self.is_active(now):
            decision = WakeDecision(True, "conversation active")
        else:
            decision = WakeDecision(False, "no wake word")

        if decision.accepted:
            self.keep_active(now)
            logger.info("Heard (%s): %s", decision.reason, transcript)
        else:
            logger.info("Heard (ignored, %s): %s", decision.reason, transcript)
        return decision

    def keep_active(self, from_time: float | None = None) -> None:
        """Keep the conversation active for window_s seconds from from_time (default: now)."""
        start = time.monotonic() if from_time is None else from_time
        self._active_until = max(self._active_until, start + self.window_s)
