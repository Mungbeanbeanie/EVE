"""Input sanitization / prompt-injection guard.

Runs on transcribed (or typed) user text BEFORE it reaches the model. Because EVE
has real tools (email, calendar, files), a malicious or garbled instruction can do
damage — this is your first line of defense. Keep it conservative: clean obvious
junk, flag suspicious instructions, but don't mangle normal speech.
"""
from __future__ import annotations
import re


def sanitize(text: str) -> str:
    """Return a cleaned, safe-to-send version of user `text`.

    Currently a near no-op so the loop runs end-to-end. Harden it as you go.
    """
    # TODO(eve): 1. Trim/normalize whitespace and strip control characters.
    text = re.sub(r"\s+", " ", text) # Collapses whitespace
    text = re.sub(r"[\x00-\x1F\x7F]", "", text) #Strips the control characters from the ASCII range (0-31 and 127)
    # TODO(eve): 2. Enforce a max length to bound token cost / abuse.
    max_chars = 250 #Around 40 to 50 words
    text = text[:max_chars] #Limit to 250 chars by slicing whats needed from the text (even if a lot was said by the user)
    # TODO(eve): 3. Detect/neutralize prompt-injection patterns (e.g. "ignore previous
    #               instructions", attempts to exfiltrate secrets or change the system
    #               prompt). Decide: redact, refuse, or flag for confirmation.

    #May need to add more
    Injections = [
        "ignore previous instructions",
        "exfiltrate secrets",
        "change the system prompt",
        "forget about previous instructions"
    ]

    #Detect
    def injection_detected(text: str) -> bool:
        for injection in Injections:
            if re.search(injection, text, re.IGNORECASE):
                return True
        return False
    
    #Nuetralize - not really needed unless others use it

    def handle_injection(text: str, mode: str = "refuse") -> str | None:

        # Redact - leave the harmful part out but use the rest of text

        if mode == "redact":
            for injection in Injections:
                text = re.sub(injection, "[REDACTED]", text, flags=re.IGNORECASE)
            return text
        
        # Refuse - return a message indicating the input was rejected

        elif mode == "refuse":
            return "Input rejected due to suspicious content."
        
        # Flag for confirmation - return a message indicating the input was flagged for review

        elif mode == "flag":
            return "Input flagged for review due to suspicious content."
        else:
            raise ValueError("Invalid mode. Choose 'redact', 'refuse', or 'flag'.")
    
        
    # TODO(eve): 4. Consider a confirmation gate before destructive tool actions
    #               (deleting email, sending messages) rather than blocking input here.
    
    # For now, just collapse surrounding whitespace so the pipeline works:
        return text.strip() # Check if you need this