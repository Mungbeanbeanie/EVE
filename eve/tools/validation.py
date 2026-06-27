"""Lightweight JSON-Schema argument validation.

The executor is the trust boundary: before a tool touches the real world we
confirm the model actually sent the arguments the tool declared. We hand-roll a
small subset of JSON Schema (the part our tool specs use — `type`, `properties`,
`required`, array `items`) rather than pull in the full `jsonschema` package; the
tool schemas in this project are simple and a few dozen lines keeps the dependency
surface small. Swap in `jsonschema` here if specs grow richer.
"""

from __future__ import annotations

from typing import Any

# JSON Schema `type` -> the Python type(s) that satisfy it. `bool` is handled
# specially below because Python's `bool` is a subclass of `int`.
_JSON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
    "null": type(None),
}


def _matches_type(value: Any, json_type: str) -> bool:
    """Return whether `value` satisfies a single JSON Schema `type`."""
    expected = _JSON_TYPES.get(json_type)
    if expected is None:
        return True  # unknown/unsupported type → don't block, be permissive

    # `True`/`False` are ints in Python; keep numeric types from accepting bools
    # and vice-versa so e.g. an "integer" field rejects `True`.
    if json_type in ("integer", "number"):
        return isinstance(value, expected) and not isinstance(value, bool)
    return isinstance(value, expected)


def validate_arguments(arguments: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    """Validate `arguments` against a tool's JSON-Schema `parameters`.

    Returns a list of human-readable error strings (empty list means valid), so the
    caller can hand them back to the model to self-correct instead of raising.
    """
    errors: list[str] = []

    # Missing required fields.
    for key in schema.get("required", []):
        if key not in arguments:
            errors.append(f"missing required argument '{key}'")

    # Type-check each provided argument that the schema describes. Arguments not in
    # `properties` are left alone (JSON Schema allows extras by default).
    properties: dict[str, Any] = schema.get("properties", {})
    for key, value in arguments.items():
        prop = properties.get(key)
        if not prop:
            continue
        errors.extend(_check_value(key, value, prop))

    return errors


def _check_value(key: str, value: Any, prop: dict[str, Any]) -> list[str]:
    """Type-check one value against its property schema (incl. array `items`)."""
    json_type = prop.get("type")
    if not json_type or _matches_type(value, json_type):
        # Validate array element types when both the list and an item schema exist.
        if json_type == "array":
            item_schema = prop.get("items")
            if item_schema and isinstance(value, list):
                return _check_items(key, value, item_schema)
        return []
    return [f"argument '{key}' should be {json_type}, got {type(value).__name__}"]


def _check_items(key: str, values: list[Any], item_schema: dict[str, Any]) -> list[str]:
    """Type-check each element of an array argument."""
    item_type = item_schema.get("type")
    if not item_type:
        return []
    return [
        f"argument '{key}[{i}]' should be {item_type}, got {type(item).__name__}"
        for i, item in enumerate(values)
        if not _matches_type(item, item_type)
    ]
