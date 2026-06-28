# EVE — Session Handoff

**Date:** 2026-06-28
**Branch:** `main`
**Focus this session:** Link the native EVE window to the agent backend, strip the
fake window chrome, and make voice output reliable in window mode. Verified
end-to-end with a real LLM reply.

---

## What was done

### 1. Browser → backend input channel (frontend/backend now linked)
Previously the `/events` SSE stream was one-way (agent → browser) and the terminal
`input()` drove every turn. Now the window drives the conversation.

- **New `eve/ui/bridge.py`** — `InputBridge`, a thread-safe `queue.Queue` carrying
  user turns from the HTTP handler thread to the agent's asyncio loop. Events:
  `{"type":"text","text":...}`, `{"type":"control","action":"listen"}`, `{"type":"stop"}`.
  Consumed via `await bridge.next_event()` (`asyncio.to_thread(q.get)`).
- **`eve/ui/server.py`** — `VizServer` takes an optional `bridge=` and handles
  `POST /input` (`{"text"}`) and `POST /control` (`{"action":"listen"}`), returning 204.
- **`eve/agent.py`** — added `set_bridge()` + `run_window()` (and `_listen_once()`).
  Window mode consumes bridge events: `text` → `_handle_turn(text, speak=True)`;
  `control:listen` → `record_utterance()` (VAD, no terminal) → STT → spoken turn.
  `start("window")` dispatches to `run_window()`.
- **`main.py`** — `run_with_window` builds the bridge, wires it to server + agent,
  runs the agent on a worker thread in `"window"` mode, and `on_quit` calls
  `bridge.stop()`. The native window now loads `viz.url + "?embedded=1"`.

### 2. Stripped the window-in-a-window (fake chrome)
The web page drew its own macOS chrome (titlebar + traffic lights + glass panel +
desktop backdrop) inside the real native window.

- **`eve/ui/web/app.js`** — adds `body.embedded` when `?embedded=1` is in the URL.
- **`eve/ui/web/styles.css`** — `.embedded` rules hide `.eve-titlebar`, `.eve-preview`,
  `.eve-blob`, `.eve-grid`; flatten `.eve-panel` (no border/radius/shadow/backdrop, fills
  the window) and `.eve-stage`. **Standalone `python -m eve.ui` keeps its full look** —
  all stripping is gated behind `.embedded`.

### 3. New UI controls (text box + real mic button)
- **`index.html`** — added a "type to EVE" compose form: `#compose-form`,
  `#text-input`, `#text-send` (classes `.eve-compose*`). Visible in both modes.
- **`app.js`** — `submitText()` → `POST ./input`; `requestListen()` (mic button click)
  → `POST ./control {action:'listen'}`; shared `_postJSON()` helper. The old cosmetic
  hold-to-talk timers were removed — **the SSE stream is the authoritative orb state**.

### 4. Voice output fixed (issue: no sound in window mode)
Root cause: pyttsx3's macOS `NSSpeechSynthesizer` is unreliable off the main thread,
and window mode runs the agent on a worker thread.

- **`eve/pipeline/tts.py`** — new `MacSayTTS` engine shelling out to the macOS `say`
  binary via `asyncio.create_subprocess_exec` (thread-safe, no run-loop constraints;
  honors `TTS_VOICE` via `-v`). `build_tts` now picks the local engine per-platform
  (`_build_local_tts`): `MacSayTTS` on macOS, `Pyttsx3TTS` elsewhere.
- With the current `.env` (ELEVENLABS_API_KEY set): **primary = ElevenLabsTTS**
  (PyAudio, thread-safe), **fallback = MacSayTTS**. Both work in window mode.

---

## Verification (all passing)

- `pytest -q` → **57 passed**. New tests: `tests/test_bridge.py`,
  `tests/test_window_loop.py`; updated `tests/test_tts.py` (platform-aware local engine).
- `ruff check` on all changed Python → clean. `node --check app.js` → OK.
- **Headless end-to-end through the real HTTP path** (script in scratchpad,
  `verify_window.py`): `POST /input {"text":"hello eve"}` → real LLM →
  orb states `idle → thinking → speaking → idle` over SSE → reply spoken.
  Real LLM answered *"Hello, it's nice to meet you."* → **PASS ✅**
- `say` synthesis confirmed (wrote a 35 KB AIFF).

---

## Not yet verified — do this next

1. **Native window run on the Mac** (could not render in the headless session):
   ```
   make window-voice      # or: python main.py --window
   ```
   Confirm: (a) no fake titlebar/traffic-lights inside the real window, (b) typing
   "hello eve" in the compose box returns a **spoken** reply, (c) the mic button
   triggers a captured spoken turn, (d) the orb mirrors listening/thinking/speaking.

2. **ElevenLabs audio playback** end-to-end on real hardware (the headless run
   captured TTS text but did not play audio). If ElevenLabs errors, confirm the
   MacSayTTS fallback actually speaks.

3. **Dock-hiding / menu-bar mode** — `--window` uses `hide_dock=not args.dock` +
   `menu_bar=True`. Confirm EVE stays out of the Dock and the menu-bar control works
   (this is `eve/ui/window.py`; pywebview path was not exercised headless).

---

## Open items / future work

- **Destructive-tool confirmation in window mode** falls back to `_ask_voice`
  (records mic) when `speak=True`. There is no in-window confirm UI yet — a
  yes/no prompt routed through the window would be cleaner.
- **`launchd` LaunchAgent** for always-on auto-start/restart was planned but not
  built this session (the original ask before the three bugs took priority).
- The scratchpad `verify_window.py` is a throwaway harness; fold its essence into a
  proper integration test if you want HTTP-level coverage in CI.

## Key files
- Backend: `eve/ui/bridge.py`, `eve/ui/server.py`, `eve/agent.py`, `main.py`,
  `eve/pipeline/tts.py`
- Frontend: `eve/ui/web/{index.html,app.js,styles.css}`
- Tests: `tests/test_bridge.py`, `tests/test_window_loop.py`, `tests/test_tts.py`
- Plan: `~/.claude/plans/a-few-things-1-composed-kahn.md`
