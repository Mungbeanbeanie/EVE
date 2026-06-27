"""Service adapters — each wraps one external suite behind ToolAdapter.

Google (Gmail / Calendar / Drive) translates the LLM's intent into real API calls.
WebSearch adds a single read-only ``web_search`` tool backed by Tavily. Both
normalize results back into plain data the model can read.
"""
