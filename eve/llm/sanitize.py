"""Input sanitization and prompt-injection guard.

Runs on transcribed (or typed) user text before it reaches the model. Because EVE
has real tools (email, calendar, files), a malicious or garbled instruction can do
damage, so this is the first line of defense: strip junk, reject text containing
known prompt-injection phrases, and leave normal speech untouched.
"""

from __future__ import annotations

import re

# Phrases that signal an attempt to override EVE's instructions. Matched
# case-insensitively as substrings of the user's text.
INJECTION_PATTERNS: list[str] = [
    "ignore previous instructions",
    "exfiltrate secrets",
    "change the system prompt",
    "forget about previous instructions",
    "you are now",
    "disregard previous instructions",
]

# Longest user input forwarded to the model (roughly 40-50 words); anything
# beyond this is truncated.
MAX_CHARS = 250


def injection_detected(text: str) -> bool:
    """Return True if `text` contains a known prompt-injection phrase."""
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in INJECTION_PATTERNS)


def handle_injection(text: str, mode: str = "refuse") -> str:
    """React to detected injection content according to `mode`.

    - ``"refuse"``: replace the input with a rejection notice.
    - ``"flag"``: replace the input with a review notice.
    - ``"redact"``: strip the offending phrases and keep the rest of the text.
    """
    if mode == "redact":
        for pattern in INJECTION_PATTERNS:
            text = re.sub(pattern, "[REDACTED]", text, flags=re.IGNORECASE)
        return text
    if mode == "refuse":
        return "Input rejected due to suspicious content."
    if mode == "flag":
        return "Input flagged for review due to suspicious content."
    raise ValueError("mode must be 'redact', 'refuse', or 'flag'")


def sanitize(text: str, injection_mode: str = "refuse") -> str:
    """Return a cleaned, safe-to-send version of user `text`.

    Strips control characters, collapses whitespace, rejects or redacts known
    injection phrases, and truncates to `MAX_CHARS`.
    """
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)  # strip ASCII control characters
    text = re.sub(r"\s+", " ", text).strip()  # collapse runs of whitespace

    # Detect on the full text before truncating, so an injection phrase that
    # starts past the character limit can't slip through by being sliced off.
    if injection_detected(text):
        return handle_injection(text, mode=injection_mode)

    return text[:MAX_CHARS]
