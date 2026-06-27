"""Service adapters — each wraps one external suite behind ToolAdapter.

Google is scaffolded first (Gmail / Calendar / Drive). Microsoft implements the
same interface and is stubbed for later. Both translate the LLM's intent into real
API calls and normalize results back into plain data the model can read.
"""
