"""ToolExecutor — safely runs a tool call the LLM requested.

The LLM client calls `await executor.run(name, arguments)` for each tool the model
wants to invoke. The executor is the trust boundary: validate before doing
anything with real-world side effects (sending email, deleting files).

Every failure mode here returns a structured `{"error": ...}` dict rather than
raising, so a bad/hallucinated tool call lets the model self-correct on the next
turn instead of crashing the whole conversation.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Awaitable, Callable

from eve.tools.registry import ToolRegistry
from eve.tools.validation import validate_arguments

log = logging.getLogger(__name__)

# async fn(tool_name, arguments) -> True to proceed, False to cancel.
Confirmer = Callable[[str, dict[str, Any]], Awaitable[bool]]


class ToolExecutor:
    """Validates arguments and dispatches a tool call to its handler.

    `confirmer`, if given, is consulted before any tool marked `destructive` runs.
    Returning False cancels the call. Without a confirmer, destructive tools run
    but are logged — wire one in (e.g. a voice "are you sure?") to gate side effects.
    """

    def __init__(self, registry: ToolRegistry, confirmer: Confirmer | None = None) -> None:
        self.registry = registry
        self.confirmer = confirmer

    async def run(self, name: str, arguments: str | dict[str, Any]) -> Any:
        """Execute tool `name` with `arguments` (a JSON string or dict).

        Returns the tool's result, or a structured `{"error": ...}` dict (to be fed
        back to the model as a tool message either way).
        """
        # 1. Parse arguments if the model passed a JSON string (most providers do).
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return self._error(f"arguments were not valid JSON: {exc}")
        if not isinstance(arguments, dict):
            return self._error("arguments must be a JSON object of named parameters")

        # 2. Look up the tool (the model may have hallucinated a name).
        try:
            tool = self.registry.get(name)
        except KeyError:
            available = ", ".join(self.registry.names()) or "(none)"
            return self._error(f"unknown tool '{name}'. Available tools: {available}")

        # 3. Validate arguments against the tool's JSON Schema.
        problems = validate_arguments(arguments, tool.parameters)
        if problems:
            return self._error(f"invalid arguments for '{name}': {'; '.join(problems)}")

        # 4. Guard destructive actions behind explicit confirmation.
        if tool.destructive and self.confirmer is not None:
            try:
                approved = await self.confirmer(name, arguments)
            except Exception as exc:  # a broken confirmer must not run the action
                log.exception("Confirmer for tool '%s' raised", name)
                return self._error(f"could not confirm '{name}': {type(exc).__name__}: {exc}")
            if not approved:
                return self._error(f"user declined to run '{name}'")
        elif tool.destructive:
            log.warning("Running destructive tool '%s' with no confirmer configured", name)

        # 5. Run the tool, turning any failure into a structured error.
        try:
            return await tool.run(**arguments)
        except Exception as exc:
            log.exception("Tool '%s' raised", name)
            return self._error(f"tool '{name}' failed: {type(exc).__name__}: {exc}")

    @staticmethod
    def _error(message: str) -> dict[str, str]:
        """Shape a recoverable error the way tool handlers already do (`{"error": ...}`)."""
        return {"error": message}
