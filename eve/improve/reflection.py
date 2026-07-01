"""Sleep-time memory reflection — distill episodes into durable insights.

Borrowed from generative-agents "reflection" and Letta's sleep-time compute:
while the user is away, the heavy model reads recent episodic memories and
writes up to three durable insights (preferences, routines, standing context)
into procedural memory, so future recalls surface understanding rather than
raw transcript fragments.

Strictly additive: reflection only ever calls `procedural.add()`. It has no
path to delete, rewrite, or reset anything — EVE's memory is never wiped.
"""

from __future__ import annotations

import logging
import re

from eve.improve.subagent import strip_thinking
from eve.llm.base import LLMClient
from eve.memory.manager import MemoryManager

log = logging.getLogger(__name__)

_MIN_EPISODES = 4   # too few events → nothing meaningful to distill
_MAX_INSIGHTS = 3   # keep procedural memory high-signal
_RECENT_K = 25

_PROMPT = """\
You are EVE's sleep-time memory consolidation. Below are recent episodic
memories (conversation events). Distill at most {max_insights} DURABLE insights
worth remembering long-term: stable user preferences, routines, facts about the
user, or standing context. Skip anything transient, uncertain, or already
obvious from a single memory. If nothing qualifies, output exactly NONE.

Output each insight on its own line, formatted exactly:
INSIGHT: <one self-contained sentence>

Recent episodic memories:
{episodes}
"""


async def reflect(memory: MemoryManager, llm: LLMClient) -> list[str]:
    """Distill recent episodes into procedural insights. Returns what was added."""
    episodes = await memory.episodic.recent(_RECENT_K)
    if len(episodes) < _MIN_EPISODES:
        log.info("[improve] reflection skipped: only %d episodes", len(episodes))
        return []

    listing = "\n".join(f"- {record.content}" for record in episodes)
    prompt = _PROMPT.format(max_insights=_MAX_INSIGHTS, episodes=listing)
    reply = strip_thinking(await llm.respond([{"role": "user", "content": prompt}]))

    insights = [
        m.strip()
        for m in re.findall(r"^INSIGHT:\s*(.+)$", reply, re.MULTILINE)
        if len(m.strip()) > 10
    ][:_MAX_INSIGHTS]

    for insight in insights:
        await memory.procedural.add(insight, metadata={"source": "sleep-reflection"})
    log.info("[improve] reflection added %d insight(s)", len(insights))
    return insights
