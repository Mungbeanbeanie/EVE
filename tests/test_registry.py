"""Tests for ToolRegistry — registration semantics and spec emission.

The registry is a small pure module with explicit behavioral guarantees:
* ``register()`` deduplicates by name (raises on collision).
* ``specs()`` emits OpenAI-style function specs in insertion order.
* ``get()`` raises ``KeyError`` for missing names — never returns None.

These tests lock down those contracts so adapters and executors can rely on
them without reading the source every time.
"""

from __future__ import annotations

import pytest

from eve.tools.base import Tool, ToolAdapter
from eve.tools.registry import ToolRegistry


# -- helpers -----------------------------------------------------------------

async def _noop_handler(**kwargs):  # type: ignore[type-var]
    return kwargs


def _make_tool(name: str = "test_tool", **overrides) -> Tool:
    """Build a minimal Tool with sane defaults; override anything explicit."""
    params = overrides.pop("parameters", {"type": "object", "properties": {}})
    desc = overrides.pop("description", f"Desc for {name}")
    return Tool(
        name=name,
        description=desc,
        parameters=params,
        handler=_noop_handler,
        **overrides,
    )


# -- registration semantics --------------------------------------------------

class TestRegistration:
    """Core register/get/len behavior."""

    def test_register_exposes_tool_via_get(self) -> None:
        reg = ToolRegistry()
        tool = _make_tool("alpha")
        reg.register(tool)
        assert reg.get("alpha") is tool

    def test_register_returns_none_means_no_side_effect_on_collision(self) -> None:
        """Collision must raise — never silently overwrite or swallow."""
        reg = ToolRegistry()
        a = _make_tool("dup")
        b = _make_tool("dup", description="shadow")
        reg.register(a)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(b)
        # Original tool must remain intact — no silent overwrite.
        assert reg.get("dup").description == "Desc for dup"

    def test_get_missing_raises_keyerror(self) -> None:
        reg = ToolRegistry()
        with pytest.raises(KeyError):
            reg.get("ghost")

    def test_len_reflects_registered_count(self) -> None:
        reg = ToolRegistry()
        assert len(reg) == 0
        reg.register(_make_tool("a"))
        reg.register(_make_tool("b"))
        assert len(reg) == 2

    def test_names_returns_all_registered(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("x"))
        reg.register(_make_tool("y"))
        reg.register(_make_tool("z"))
        # Order matches insertion order (dict preserves it in CPython 3.7+).
        assert reg.names() == ["x", "y", "z"]

    def test_register_empty_name_allowed(self) -> None:
        """Edge case: an empty-string name is a valid key — callers decide."""
        reg = ToolRegistry()
        tool = _make_tool("")
        reg.register(tool)
        assert reg.get("") is tool


# -- spec emission -----------------------------------------------------------

class TestSpecs:
    """specs() must emit provider-neutral OpenAI-style function specs."""

    def test_specs_returns_list_of_dicts(self) -> None:
        reg = ToolRegistry()
        reg.register(_make_tool("t"))
        specs = reg.specs()
        assert isinstance(specs, list)
        assert all(isinstance(s, dict) for s in specs)

    def test_spec_shape_matches_openai_function_schema(self) -> None:
        """Every spec must have type='function' and a function sub-dict."""
        tool = _make_tool("alpha", description="does stuff")
        reg = ToolRegistry()
        reg.register(tool)
        (spec,) = reg.specs()
        assert spec == {
            "type": "function",
            "function": {
                "name": "alpha",
                "description": "does stuff",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_specs_preserves_insertion_order(self) -> None:
        reg = ToolRegistry()
        for n in ("c", "a", "b"):
            reg.register(_make_tool(n))
        names = [s["function"]["name"] for s in reg.specs()]
        assert names == ["c", "a", "b"]

    def test_specs_empty_registry_is_empty_list(self) -> None:
        assert ToolRegistry().specs() == []


# -- adapter contract --------------------------------------------------------

class TestAdapterContract:
    """ToolAdapter.register_into must mutate the registry in place."""

    def test_adapter_mutates_shared_registry(self) -> None:
        class TinyAdapter(ToolAdapter):
            def register_into(self, registry: ToolRegistry) -> None:
                registry.register(_make_tool("adapter_tool"))

        reg = ToolRegistry()
        TinyAdapter().register_into(reg)
        assert len(reg) == 1
        assert "adapter_tool" in reg.names()
