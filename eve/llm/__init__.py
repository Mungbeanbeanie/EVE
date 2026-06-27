"""Provider-agnostic LLM layer.

`build_llm(config)` (in factory.py) returns an `LLMClient` for whatever provider
you configured — Anthropic, OpenAI, Gemini, a local Ollama model, etc. The rest of
EVE never imports a vendor SDK directly; swapping models is a config change.
"""
