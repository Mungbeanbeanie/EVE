"""Input sanitization / prompt-injection guard.

Runs on transcribed (or typed) user text BEFORE it reaches the model. Because EVE
has real tools (email, calendar, files), a malicious or garbled instruction can do
damage — this is your first line of defense. Keep it conservative: clean obvious
junk, flag suspicious instructions, but don't mangle normal speech.
"""

from __future__ import annotations


def sanitize(text: str) -> str:
    """Return a cleaned, safe-to-send version of user `text`.

    Currently a near no-op so the loop runs end-to-end. Harden it as you go.
    """
    # TODO(eve): 1. Trim/normalize whitespace and strip control characters.
    # TODO(eve): 2. Enforce a max length to bound token cost / abuse.
    # TODO(eve): 3. Detect/neutralize prompt-injection patterns (e.g. "ignore previous
    #               instructions", attempts to exfiltrate secrets or change the system
    #               prompt). Decide: redact, refuse, or flag for confirmation.
    # TODO(eve): 4. Consider a confirmation gate before destructive tool actions
    #               (deleting email, sending messages) rather than blocking input here.
    # For now, just collapse surrounding whitespace so the pipeline works:
    return text.strip()
