"""ToolExecutor — safely runs a tool call the LLM requested.

The LLM client calls `await executor.run(name, arguments)` for each tool the model
wants to invoke. The executor is the trust boundary: validate before doing
anything with real-world side effects (sending email, deleting files).
"""

from __future__ import annotations

import json
from typing import Any

from eve.tools.registry import ToolRegistry


class ToolExecutor:
    """Validates arguments and dispatches a tool call to its handler."""

    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry

    async def run(self, name: str, arguments: str | dict[str, Any]) -> Any:
        """Execute tool `name` with `arguments` (a JSON string or dict).

        Returns the tool's result (to be fed back to the model as a tool message).
        """
        # Parse arguments if the model passed a JSON string (most providers do).
        if isinstance(arguments, str):
            arguments = json.loads(arguments or "{}")

        tool = self.registry.get(name)  # KeyError if the model hallucinated a tool

        # TODO(eve): 1. Validate `arguments` against tool.parameters (JSON Schema).
        #               Consider the `jsonschema` package, or hand-rolled checks.
        # TODO(eve): 2. Add a confirmation/guard for destructive actions before run
        #               (e.g. require explicit user OK to send/delete).
        # TODO(eve): 3. Wrap tool.run in try/except and return a structured error so
        #               the model can recover instead of the whole turn crashing.
        return await tool.run(**arguments)
