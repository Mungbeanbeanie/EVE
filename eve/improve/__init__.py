"""Self-improvement — EVE's sleep-time compute subsystem.

While the user is away, a heavier local model (config.improve_model) researches,
implements, and test-gates small improvements to EVE's own codebase inside a git
worktree sandbox. Conversation always wins: the loop only runs when the
ActivityMonitor says the user has been idle, and it pauses between phases the
moment they return.

Design lineage (see README "Self-improvement loop"):
  - sleep-time compute (Letta): heavy model works offline, light model chats.
  - SICA / Darwin Gödel Machine: self-editing gated by tests, sandboxed, traceable.
  - generative-agents reflection: idle-time distillation of episodic memories.
"""

from eve.improve.activity import ActivityMonitor
from eve.improve.loop import SelfImprovementLoop

__all__ = ["ActivityMonitor", "SelfImprovementLoop"]
