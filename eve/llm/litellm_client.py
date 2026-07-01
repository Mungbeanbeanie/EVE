"""Default LLM client, backed by LiteLLM.

LiteLLM exposes ONE function — `litellm.completion(model=..., messages=..., tools=...)`
— that speaks to ~all providers using the OpenAI-style schema. That is what makes
EVE provider-agnostic without writing a client per vendor. Set the model string and
key in config and you're done.

Docs: https://docs.litellm.ai/
"""

from __future__ import annotations

import asyncio
import logging

import litellm  # type: ignore

from eve.config import Config
from eve.llm.base import LLMClient, Message

log = logging.getLogger(__name__)

# Some providers occasionally emit a malformed tool call the provider itself
# can't parse and reject the request instead of returning a message. Seen live:
# Groq + Llama (HTTP 400 "tool_use_failed") and Ollama + qwen3.5 ("tool call
# parsing failed" → HTTP 500). It's sampling-dependent, so a retry usually works.
_TOOL_FORMAT_MARKERS = (
    "tool_use_failed",
    "failed to call a function",
    "tool call parsing failed",
    "internal server error",  # Ollama surfaces the parse failure as a bare 500
)
_TOOL_FORMAT_RETRIES = 2  # extra attempts after the first before giving up


class _ToolCallFormatError(Exception):
    """The provider rejected the model's tool-call formatting (e.g. Groq tool_use_failed)."""


class LiteLLMClient(LLMClient):
    """Talks to any provider via LiteLLM's unified completion API."""

    def __init__(self, config: Config) -> None:
        self.model = config.llm_model        # e.g. "anthropic/claude-opus-4-8"
        self.api_key = config.llm_api_key    # generic; LiteLLM also reads vendor env vars
        self.api_base = config.llm_api_base  # for self-hosted / Ollama

    async def respond(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        executor=None,
        max_iterations: int = 10,
    ) -> str:
        """Run the tool-use loop and return the model's final text answer."""
        for iteration in range(max_iterations):
            try:
                resp = await self._completion(messages, tools)
            except _ToolCallFormatError:
                # The model kept producing a tool call the provider can't parse. Rather
                # than fail the turn, answer once WITHOUT tools so the user still gets a
                # coherent reply (e.g. the model explains it can't fetch live data).
                log.warning("Tool call unparseable after retries; answering without tools")
                resp = await self._completion(messages, tools=None)

            msg = resp.choices[0].message
            if not msg.tool_calls:
                # A model that only called tools can return content=None; never
                # hand None back to the caller (it speaks/stores the reply as text).
                return msg.content or ""
            # The assistant message carrying the tool_calls MUST be appended before
            # the tool results — providers reject a 'tool' message that isn't
            # immediately preceded by the matching assistant tool_calls message.
            messages.append(msg.model_dump())
            for call in msg.tool_calls:
                result = await executor.run(call.function.name, call.function.arguments)
                messages.append(Message(role="tool", content=str(result), tool_call_id=call.id))
        # Tool budget exhausted while the model still wants more calls. Withhold
        # the tools and ask once more (same shape as the format-error fallback),
        # so the caller gets the best answer available from what was gathered
        # instead of a crashed turn or a dead self-improvement cycle.
        log.warning("Tool budget (%d calls) exhausted; forcing a tool-free final answer", max_iterations)
        resp = await self._completion(messages, tools=None)
        return resp.choices[0].message.content or ""

    async def _completion(self, messages: list[Message], tools: list[dict] | None):
        """Call the provider, retrying transient malformed-tool-call rejections.

        Raises `_ToolCallFormatError` if the provider keeps rejecting the tool call
        after the retry budget; any other error propagates unchanged.
        """
        attempts = _TOOL_FORMAT_RETRIES + 1
        for attempt in range(attempts):
            try:
                return await asyncio.to_thread(
                    litellm.completion,
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    # Coerce blank config to None: an empty string would make
                    # LiteLLM emit an `Authorization: Bearer ` header, which
                    # keyless local providers (e.g. Ollama) reject outright.
                    api_key=self.api_key or None,
                    api_base=self.api_base or None,
                )
            except Exception as exc:
                if not self._is_tool_format_error(exc):
                    raise  # genuine error (auth, network, rate limit) — let it surface
                if attempt + 1 >= attempts:
                    raise _ToolCallFormatError(str(exc)) from exc
                log.warning(
                    "Provider rejected a malformed tool call (attempt %d/%d); retrying",
                    attempt + 1, attempts,
                )
                await asyncio.sleep(0.4 * (attempt + 1))  # brief backoff before resampling
        raise _ToolCallFormatError("retry loop exhausted")  # unreachable

    @staticmethod
    def _is_tool_format_error(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(marker in text for marker in _TOOL_FORMAT_MARKERS)
