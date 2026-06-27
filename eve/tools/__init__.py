"""Tool subsystem — how EVE acts on the world.

Flow:  LLM asks to call a tool -> ToolExecutor -> the tool's adapter
       (Google / web search) -> result -> back to the LLM.

    Tool          — one callable capability with a JSON schema (e.g. send_email).
    ToolAdapter   — groups related tools for a service and owns its auth/session.
    ToolRegistry  — holds all tools; emits provider-neutral specs for the LLM.
    ToolExecutor  — validates arguments and dispatches a tool call to its Tool.
"""
