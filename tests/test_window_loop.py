"""Tests for the UI-driven window loop (Agent.run_window).

These build an Agent from fake subsystems (no audio/LLM/network) and drive it
through the bridge exactly the way the window does — proving a typed prompt and a
mic "listen" both run the full turn and end with EVE speaking the reply, while the
orb is driven through idle → thinking → speaking → idle.
"""

from __future__ import annotations

from eve.agent import Agent
from eve.pipeline.base import TTSEngine
from eve.tools.executor import ToolExecutor
from eve.tools.registry import ToolRegistry
from eve.ui.bridge import InputBridge


class _RecordingTTS(TTSEngine):
    """Fake TTS that records what it was asked to say (no audio)."""

    def __init__(self) -> None:
        self.spoken: list[str] = []

    async def speak(self, text: str) -> None:
        self.spoken.append(text)


class _RecordingViz:
    """Fake visualizer that records the orb states the agent pushes."""

    def __init__(self) -> None:
        self.states: list[str] = []

    def set_state(self, name: str) -> None:
        self.states.append(name)


class _FakeWorking:
    def add_user(self, text: str) -> None:  # noqa: D401 - trivial stub
        self.last = text


class _FakeMemory:
    """Minimal memory: enough surface for _handle_turn, no storage."""

    def __init__(self) -> None:
        self.working = _FakeWorking()

    async def recall(self, text: str) -> list[dict]:
        return [{"role": "user", "content": text}]

    async def remember(self, *, user: str, assistant: str) -> None:
        self.remembered = (user, assistant)

    async def flush(self) -> None:
        pass


class _FakeLLM:
    """Fake LLM that always returns a fixed reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply

    async def respond(self, *, messages, tools, executor) -> str:
        return self.reply


class _FakeAudio:
    def __init__(self, pcm: bytes = b"\x00\x00") -> None:
        self._pcm = pcm

    async def record_utterance(self) -> bytes:
        return self._pcm


class _FakeSTT:
    def __init__(self, text: str) -> None:
        self._text = text

    async def transcribe(self, audio: bytes) -> str:
        return self._text


def _build_agent(config, *, llm_reply="Hi there!", stt_text="hello eve"):
    tts = _RecordingTTS()
    viz = _RecordingViz()
    registry = ToolRegistry()
    agent = Agent(
        config=config,
        audio=_FakeAudio(),
        stt=_FakeSTT(stt_text),
        tts=tts,
        llm=_FakeLLM(llm_reply),
        memory=_FakeMemory(),
        tools=registry,
        executor=ToolExecutor(registry),
        viz=viz,
    )
    agent._running = True
    return agent, tts, viz


async def test_window_text_turn_speaks_reply(config):
    """A typed 'hello eve' runs the full turn and EVE speaks the reply."""
    agent, tts, viz = _build_agent(config, llm_reply="Hi there!")
    bridge = InputBridge()
    agent.set_bridge(bridge)

    bridge.submit_text("hello eve")
    bridge.stop()  # ends the loop after the turn
    await agent.run_window()

    assert tts.spoken == ["Hi there!"]
    # idle on entry, then the turn's thinking → speaking → idle.
    assert viz.states == ["idle", "thinking", "speaking", "idle"]


async def test_window_listen_turn_transcribes_and_speaks(config):
    """A mic 'listen' control captures, transcribes, and speaks the reply."""
    agent, tts, viz = _build_agent(config, llm_reply="Hello!", stt_text="hello eve")
    bridge = InputBridge()
    agent.set_bridge(bridge)

    bridge.submit_control("listen")
    bridge.stop()
    await agent.run_window()

    assert tts.spoken == ["Hello!"]
    assert viz.states == ["idle", "listening", "thinking", "speaking", "idle"]


async def test_window_blank_transcript_stays_idle(config):
    """Silence (empty transcript) returns to idle without a turn."""
    agent, tts, viz = _build_agent(config, stt_text="   ")
    bridge = InputBridge()
    agent.set_bridge(bridge)

    bridge.submit_control("listen")
    bridge.stop()
    await agent.run_window()

    assert tts.spoken == []  # nothing said
    assert viz.states == ["idle", "listening", "idle"]


async def test_window_requires_bridge(config):
    """run_window without a bridge is a clear programming error."""
    agent, _tts, _viz = _build_agent(config)
    try:
        await agent.run_window()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert "bridge" in str(exc)
