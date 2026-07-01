"""SelfImprovementLoop — EVE's sleep-time compute daemon.

Runs on its own daemon thread (with its own asyncio loop) so a long local-model
generation can never block conversation or delay shutdown. The cycle, gated at
every phase on the user still being idle:

    reflect (occasionally) → research → implement → verify (pytest) → review → commit

Everything lands on a `self-improve/<timestamp>` branch inside a worktree
sandbox; every cycle is journaled under `<improve_home>/journal/`; guardrails
(see eve/improve/guardrails.py) mechanically enforce that main and EVE's
persistent memory are untouchable. The human merges branches — the loop never
pushes and never self-deploys.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from eve.config import Config
from eve.improve import codebase, subagent
from eve.improve.activity import ActivityMonitor
from eve.improve.guardrails import GuardrailViolation
from eve.improve.journal import CycleEntry, Journal
from eve.improve.reflection import reflect
from eve.improve.state import load_state, save_state
from eve.improve.subagent import Subagent
from eve.improve.workspace import Workspace
from eve.llm.base import LLMClient
from eve.llm.factory import build_llm
from eve.memory.manager import MemoryManager

log = logging.getLogger(__name__)

# Rotating priorities: every area gets attention, matching the mission —
# memory first (it's EVE's core value), then structure, then polish.
FOCUS_AREAS = (
    "memory quality and recall",
    "architecture and modularity",
    "code quality and readability",
    "test coverage and robustness",
    "documentation accuracy",
    "performance and latency",
)

_REST_SECONDS = 30      # breather between cycles so the machine isn't pegged
_ERROR_BACKOFF = 300    # after an unexpected error, cool off before retrying
_ENGINEER_ITERATIONS = 30  # tool-call budget for the implement phase


class SelfImprovementLoop:
    """Owns the sandbox, the subagents, and the cycle state machine."""

    def __init__(
        self,
        config: Config,
        activity: ActivityMonitor,
        memory: MemoryManager | None = None,
        llm_factory: Callable[[], LLMClient] | None = None,
        web_search=None,  # async fn(query=...) -> results; wired from the agent's adapter
        repo_root: Path | None = None,  # default: discovered from this package's checkout
    ) -> None:
        self.config = config
        self.activity = activity
        self.memory = memory
        self._repo_root = repo_root
        self.home = Path(config.improve_home).expanduser()
        self.journal = Journal(self.home)
        self.state = load_state(self.home / "state.json")
        self._llm_factory = llm_factory or self._default_llm_factory
        self._web_search = web_search
        self._workspace: Workspace | None = None
        self._llm: LLMClient | None = None
        self._running = False
        self._thread: threading.Thread | None = None
        self._session_cycles = 0
        self._status_lock = threading.Lock()
        self._phase = "off"
        # Waking mid-cycle uses a shorter idle bar than starting a cycle: the
        # user stepping away again briefly shouldn't stall a half-done cycle
        # for the full threshold.
        self._resume_idle = min(60.0, config.improve_idle_seconds)

    def _default_llm_factory(self) -> LLMClient:
        """The heavy idle model — same plumbing as chat, different model string."""
        heavy = self.config.model_copy(update={"llm_model": self.config.improve_model})
        return build_llm(heavy)

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def start_in_thread(self) -> None:
        """Run the loop on a daemon thread: quitting EVE never waits on it."""
        self._running = True
        self._thread = threading.Thread(target=self._thread_main, name="eve-improve", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False

    def _thread_main(self) -> None:
        try:
            asyncio.run(self.run())
        except Exception:  # the daemon must never take EVE down with it
            log.exception("[improve] loop crashed")

    def status(self) -> dict:
        """Snapshot for the `self_improvement_status` chat tool."""
        with self._status_lock:
            phase = self._phase
        return {
            "enabled": self._running,
            "phase": phase,
            "model": self.config.improve_model,
            "idle_seconds": round(self.activity.idle_seconds()),
            "branch": self._workspace.branch if self._workspace else None,
            "worktree": str(self._workspace.path) if self._workspace else None,
            "cycles_completed": self.state.cycles_completed,
            "session_cycles": self._session_cycles,
            "recent": self.journal.tail(5),
        }

    def _set_phase(self, phase: str) -> None:
        with self._status_lock:
            self._phase = phase
        log.info("[improve] phase → %s", phase)

    # ── Main loop ─────────────────────────────────────────────────────────────
    async def run(self) -> None:
        """Wait for idle, run a cycle, rest; forever (until stop())."""
        root = self._repo_root or codebase.repo_root()
        if root is None:
            log.warning("[improve] not running from a git checkout — loop disabled")
            self._set_phase("disabled")
            return
        self._repo_root = root
        self.journal.note(
            f"session start — model {self.config.improve_model}, "
            f"idle threshold {self.config.improve_idle_seconds:.0f}s"
        )
        while self._running:
            self._set_phase("sleeping")
            await self.activity.wait_for_idle(self.config.improve_idle_seconds)
            if not self._running:
                break
            if self.config.improve_max_cycles and self._session_cycles >= self.config.improve_max_cycles:
                self.journal.note("session cycle cap reached — resting until restart")
                self._set_phase("capped")
                return
            try:
                await self.run_cycle()
            except GuardrailViolation as exc:
                self._abandon(f"guardrail: {exc}")
            except Exception:
                log.exception("[improve] cycle failed")
                self._abandon("unexpected error (see log)")
                await asyncio.sleep(_ERROR_BACKOFF)
            await asyncio.sleep(_REST_SECONDS)

    def _abandon(self, reason: str) -> None:
        """Revert the sandbox and journal why the cycle died."""
        self.journal.note(f"cycle abandoned — {reason}")
        if self._workspace is not None:
            try:
                self._workspace.reset()
            except Exception:
                log.exception("[improve] sandbox reset failed")

    async def _pause_point(self) -> None:
        """Between phases: if the user came back, wait until they leave again."""
        await self.activity.wait_for_idle(self._resume_idle)

    def _ensure_workspace(self) -> Workspace:
        if self._workspace is None:
            self._workspace = Workspace.create(
                self._repo_root, self.home, self.config.memory_dir
            )
            self.journal.note(
                f"sandbox `{self._workspace.branch}` at {self._workspace.path}"
            )
        return self._workspace

    def _heavy_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = self._llm_factory()
        return self._llm

    # ── One cycle ─────────────────────────────────────────────────────────────
    async def run_cycle(self) -> CycleEntry:
        """One research→implement→verify→review→commit pass. Always journals."""
        number = self.state.cycles_completed + 1
        focus = FOCUS_AREAS[self.state.cycles_completed % len(FOCUS_AREAS)]
        ws = self._ensure_workspace()
        entry = CycleEntry(number=number, focus=focus, branch=ws.branch)
        llm = self._heavy_llm()
        repo_map = codebase.describe(ws.path)

        try:
            await self._maybe_reflect(entry, llm)
            await self._pause_point()

            idea = await self._next_idea(entry, llm, ws, focus, repo_map)
            if idea is None:
                entry.outcome = "skipped"
                entry.change = "SKIP: researcher produced no usable ideas"
                return entry
            await self._pause_point()

            self._set_phase("implementing")
            done = await self._implement(entry, llm, ws, idea, focus, repo_map)
            if not done:
                ws.reset()
                return entry
            await self._pause_point()

            self._set_phase("verifying")
            if not await self._verify(entry, llm, ws):
                ws.reset()
                return entry
            await self._pause_point()

            self._set_phase("reviewing")
            if not await self._review(entry, llm, ws):
                ws.reset()
                return entry

            self._set_phase("committing")
            message = (
                f"self-improve: {entry.change.removeprefix('CHANGE:').strip()}\n\n"
                f"Cycle {number} · focus: {focus}\nIdea: {idea}\n"
                f"Reviewed-by: {self.config.improve_model} (reviewer subagent)"
            )
            entry.commit = await asyncio.to_thread(
                ws.commit, message, self.config.improve_max_files
            )
            entry.outcome = "committed"
            return entry
        finally:
            self.journal.write_cycle(entry)
            self.state.record(
                cycle=number, focus=focus, outcome=entry.outcome,
                change=entry.change or entry.idea or "",
            )
            save_state(self.home / "state.json", self.state)
            self._session_cycles += 1

    # ── Phases ────────────────────────────────────────────────────────────────
    async def _maybe_reflect(self, entry: CycleEntry, llm: LLMClient) -> None:
        """Occasionally consolidate episodic memory (additive-only) while idle."""
        hours = self.config.improve_reflect_hours
        if self.memory is None or hours <= 0:
            return
        last = self.state.last_reflection
        if last:
            try:
                if datetime.now() - datetime.fromisoformat(last) < timedelta(hours=hours):
                    return
            except ValueError:
                pass  # unreadable stamp → just reflect now and rewrite it
        self._set_phase("reflecting")
        try:
            insights = await reflect(self.memory, llm)
        except Exception as exc:  # memory backend down — reflection is optional
            log.warning("[improve] reflection unavailable: %s", exc)
            return
        self.state.last_reflection = datetime.now().isoformat(timespec="seconds")
        if insights:
            entry.reflection = "\n".join(f"- {i}" for i in insights)

    async def _next_idea(
        self, entry: CycleEntry, llm: LLMClient, ws: Workspace, focus: str, repo_map: str
    ) -> str | None:
        """Pop the backlog, running the researcher first when it's empty."""
        if not self.state.backlog:
            self._set_phase("researching")
            tools = subagent.read_tools(ws)
            if self._web_search is not None:
                tools.append(subagent.make_web_search_tool(self._web_search))
            researcher = Subagent("researcher", subagent.RESEARCHER_ROLE, llm, tools)
            reply = await researcher.run(
                f"Focus area for this cycle: {focus}.\n\n"
                f"Codebase map:\n{repo_map}\n\n"
                f"Previous cycles (do not repeat):\n{self.state.history_digest()}",
                require=("IDEA",),
            )
            ideas = subagent.parse_ideas(reply)
            self.state.backlog.extend(ideas)
            entry.research = reply[-3000:]
        if not self.state.backlog:
            return None
        idea = self.state.backlog.pop(0)
        entry.idea = idea
        return idea

    async def _implement(
        self, entry: CycleEntry, llm: LLMClient, ws: Workspace, idea: str, focus: str, repo_map: str
    ) -> bool:
        """Engineer phase: returns True when there is a change worth verifying."""
        engineer = Subagent(
            "engineer",
            subagent.ENGINEER_ROLE,
            llm,
            [*subagent.read_tools(ws), *subagent.write_tools(ws), subagent.test_tool(ws)],
            max_iterations=_ENGINEER_ITERATIONS,
        )
        reply = await engineer.run(
            f"Improvement idea to implement now:\nIDEA: {idea}\n\n"
            f"Cycle focus: {focus}.\n\nCodebase map:\n{repo_map}\n\n"
            f"Previous cycles:\n{self.state.history_digest()}",
            require=("CHANGE", "SKIP"),
        )
        verdict = subagent.final_verdict(reply, "CHANGE", "SKIP")
        if verdict is None or verdict[0] == "SKIP":
            entry.change = "SKIP: " + (verdict[1] if verdict else "engineer gave no verdict")
            entry.outcome = "skipped"
            return False
        entry.change = f"CHANGE: {verdict[1]}"
        if not ws.changed_files():
            entry.change += " (but no files were modified)"
            entry.outcome = "skipped"
            return False
        return True

    async def _verify(self, entry: CycleEntry, llm: LLMClient, ws: Workspace) -> bool:
        """Authoritative test gate, with one engineer repair round on failure."""
        result = await asyncio.to_thread(ws.run_tests)
        if not result.ok:
            repairer = Subagent(
                "engineer",
                subagent.ENGINEER_ROLE,
                llm,
                [*subagent.read_tools(ws), *subagent.write_tools(ws), subagent.test_tool(ws)],
                max_iterations=_ENGINEER_ITERATIONS,
            )
            await repairer.run(
                "Your change broke the test suite. Fix it (or minimize the change "
                f"until tests pass).\n\nTest output:\n{result.output}"
            )
            result = await asyncio.to_thread(ws.run_tests)
        entry.tests = result.output
        if not result.ok:
            entry.outcome = "reverted"
            entry.review = "tests failed after repair round — change reverted"
            return False
        await asyncio.to_thread(ws.stage)
        entry.files = ws.changed_files()
        entry.diff_stat = ws.diff_stat()
        return True

    async def _review(self, entry: CycleEntry, llm: LLMClient, ws: Workspace) -> bool:
        """Reviewer subagent gets the diff + tests; unparseable verdict = reject."""
        diff = ws.staged_diff()
        reviewer = Subagent(
            "reviewer", subagent.REVIEWER_ROLE, llm, subagent.read_tools(ws),
        )
        reply = await reviewer.run(
            f"Proposed change:\n{entry.change}\n\nUnified diff:\n```diff\n{diff[:20000]}\n```\n\n"
            f"Test output (passing):\n{entry.tests}",
            require=("APPROVE", "REJECT"),
        )
        verdict = subagent.final_verdict(reply, "APPROVE", "REJECT")
        entry.review = f"{verdict[0]}: {verdict[1]}" if verdict else "REJECT: no parseable verdict"
        if verdict is None or verdict[0] != "APPROVE":
            entry.outcome = "rejected"
            return False
        return True
