"""EVE — a personal AI agent that lives on your machine.

Package layout (each subpackage is independently swappable; the Agent depends
only on the abstract interfaces, never the concrete implementations):

    eve.config          # typed settings loaded from .env
    eve.agent           # orchestrator wiring everything together
    eve.pipeline        # voice I/O: audio capture, VAD, STT, TTS
    eve.llm             # provider-agnostic LLM client + factory
    eve.memory          # working + procedural + episodic memory
    eve.tools           # tool registry/executor + Google/Microsoft adapters
    eve.utils           # logging and shared helpers
"""

__version__ = "0.0.1"
