from __future__ import annotations # Has to be first
"""Input sanitization / prompt-injection guard.

Runs on transcribed (or typed) user text BEFORE it reaches the model. Because EVE
has real tools (email, calendar, files), a malicious or garbled instruction can do
damage — this is your first line of defense. Keep it conservative: clean obvious
junk, flag suspicious instructions, but don't mangle normal speech.
"""
import re
from typing import Callable, Any

Injections: list[str] = [
    "ignore previous instructions",
    "exfiltrate secrets",
    "change the system prompt",
    "forget about previous instructions",
    "you are now",
    "disregard previous instructions",
    ]

Destructive_tools: set[str] = {   
    "delete email",
    "delete calendar event",
    "send email",
    "send message",
    "create calendar event", 
    }

#Detect
def injection_detected(text: str) -> bool:
    return any(re.search(injection, text, re.IGNORECASE) 
               for injection in Injections)

def handle_injection(text: str, mode: str = "refuse") -> str:

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
    
def sanitize(text: str, injection_mode: str = "refuse") -> str:
    """Return a cleaned, safe-to-send version of user `text`.

    Currently a near no-op so the loop runs end-to-end. Harden it as you go.
    """
    text = re.sub(r"[\x00-\x1F\x7F]", "", text) #Strips the control characters from the ASCII range (0-31 and 127)
    text = re.sub(r"\s+", " ", text).strip() # Collapses whitespace

    # Detect on the FULL text before truncating, so an injection phrase that begins
    # past the char limit can't slip through by being sliced off.
    if injection_detected(text):
        return handle_injection(text, mode=injection_mode)

    max_chars = 250 #Around 40 to 50 words
    return text[:max_chars] #Limit to 250 chars (even if the user said a lot more)

def summarize(tool_name: str, args: dict) -> str:
    match tool_name:
        case "send email":
            to = args.get("to", "unknown recipient")
            subject = args.get("subject", "no subject")
            return f"Send an email to {to} with subject '{subject}'?"
        
        case "delete email":
            target = args.get("id") or args.get("path", "unknown")
            return f"Delete email with id/path '{target}'?"
        
        case "create calendar event":
            title = args.get("title", "untitled event")
            date = args.get("date", "unspecified date")
            return f"Create a calendar event titled '{title}' on {date}?"
        
        case "delete calendar event":
            event_id = args.get("id", "unknown event")
            return f"Delete calendar event with id '{event_id}'?"
        
        case "send message":
            recipient = args.get("recipient", "unknown recipient")
            return f"Send a message to {recipient}?"
        
        case _:
            return f"Run \"{tool_name}\" with arguments {args}?"
        
def dispatch(tool_name: str, tool_fn: Callable[..., Any], args: dict, confirm_fn: Callable[[str], bool]) -> Any:
    if tool_name in Destructive_tools:
        prompt = summarize(tool_name, args)
        try:
            confirmed = confirm_fn(prompt)
        except Exception as exc:
            return {"status": "error", "reason": f"Confirmation failed: {exc}"}
        
        if not confirmed:
            return {"status": "cancelled", "reason": "User did not confirm the action."}
        
    return tool_fn(**args)
