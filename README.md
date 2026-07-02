# EVE — your personal AI agent

EVE is a voice-driven AI agent that lives on your computer, remembers what matters,
and can act on your behalf through Google and web-search tools.

---

## Architecture

```
VOICE   Mic ─▶ STT (faster-whisper, local) ─▶ sanitize ─▶ LLM (any provider) ─▶ TTS ─▶ Speaker

MEMORY  message ─▶ working memory ─▶ { procedural, episodic } stores (mem0 + FAISS, on disk)
                          ▲                      │
                          └──── vector recall ◀──┘

TOOLS   LLM ─▶ tool registry ─▶ tool executor ─▶ Google / web-search adapter ─▶ result
```

Three design choices shape the code:

- **Provider-agnostic LLM.** EVE is not tied to any vendor. `build_llm(config)` returns
  a client driven by a model string (`anthropic/…`, `openai/…`, `gemini/…`, `ollama/…`);
  the default uses [LiteLLM](https://docs.litellm.ai/) so one interface speaks to all.
- **Three memory layers** behind one `MemoryManager`: **working** (volatile context),
  **procedural** (learned preferences/skills), **episodic** (timestamped events).
  Durable layers persist on disk via [mem0](https://docs.mem0.ai/) — a local FAISS
  vector index embedded with [FastEmbed](https://github.com/qdrant/fastembed)
  (ONNX). No database, no network service: EVE is one self-contained process.
- **Swappable everything.** The `Agent` depends only on abstract interfaces, so each
  subsystem can be replaced without touching the orchestrator.


---

## Prerequisites

- **Python 3.11+**
- **System packages** (not pip): `portaudio` (for PyAudio); `ffmpeg` is optional but handy.
  - macOS: `brew install portaudio` (and `brew install ffmpeg` if you want it)
  - Debian/Ubuntu: `apt-get install portaudio19-dev espeak` (+ `ffmpeg`)
- An **LLM API key** for whatever provider you choose (or a local Ollama model).

---

## Quickstart

### TL;DR (with `make`)

A `Makefile` wraps every step below. From a clean clone:

```bash
make setup        # create venv, install deps, scaffold .env (then edit .env)
make run          # text-mode REPL
```

```bash
make install-voice   # add the mic/STT/TTS stack, then: make voice
make test            # run the test suite
make dist            # versioned source tarball in dist/
make install-agent   # run EVE always-on at login (macOS launchd)
```

The rest of this section explains each step manually.

### 1. Configure

```bash
cp .env.example .env          # then fill in LLM_PROVIDER/LLM_MODEL + key
```

Two things to know about `.env`:

- **`LLM_MODEL` format:** it's a LiteLLM model string, `provider/model`
  (e.g. `groq/llama-3.3-70b-versatile`, `anthropic/claude-opus-4-8`). EVE strips
  the `provider/` prefix automatically where mem0's native client needs the bare name.
- **Memory needs nothing.** Long-term memory persists to `MEMORY_DIR`
  (default `~/.eve/memory`) as a FAISS index, and `EMBEDDER_MODEL` runs in-process
  via FastEmbed — it downloads and caches once on first run. To use a different
  embedder, set `EMBEDDER_MODEL` and a matching `EMBEDDING_DIMS` (see `.env.example`).

### 2. Install + run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # core: LLM + memory + tools

python main.py --mode text        # fastest path: run the agent without audio
```

In text mode, each turn responds as soon as the LLM replies — durable memory is
written in the **background**, so a slow mem0 write never blocks the conversation.
On exit, EVE flushes any in-flight writes so the last turn isn't lost (this can add
a short pause when you quit).

The **voice stack is optional and installed separately** (it needs system audio
support):

```bash
brew install portaudio            # macOS system dep for PyAudio
pip install faster-whisper webrtcvad pyttsx3 pyaudio

python main.py --mode voice       # mic → STT → LLM → TTS → speaker
```

**Voice output** uses a local, offline engine by default — `pyttsx3` on Linux/Windows,
the macOS `say` binary on macOS (because pyttsx3's NSSpeechSynthesizer driver hangs off
the main thread). For a higher-quality or custom (cloned) voice, set `ELEVENLABS_API_KEY`
and EVE switches to [ElevenLabs](https://elevenlabs.io) cloud TTS — streamed as PCM for
low latency — falling back to the local voice automatically if the key is missing or a
request fails. Pick a voice with `ELEVENLABS_VOICE_ID` (list them via
`python -m eve.pipeline.tts`) and a model with `ELEVENLABS_MODEL`
(`eleven_flash_v2_5` is the lowest-latency choice). See `.env.example`.

> **PyAudio fails to build?** On Intel macOS with newer Python there's no prebuilt
> wheel, so pip compiles from source and the compiler may not find Homebrew's
> portaudio headers. Point it at them explicitly:
>
> ```bash
> export CFLAGS="-I$(brew --prefix portaudio)/include"
> export LDFLAGS="-L$(brew --prefix portaudio)/lib"
> pip install --no-cache-dir pyaudio
> ```

### Run the tests

```bash
pytest -q                     # hardware-free unit + wiring tests
```

### Swap your LLM

No code change — edit `.env`:

```env
LLM_MODEL=openai/gpt-4o          # or gemini/gemini-1.5-pro, ollama/llama3, …
OPENAI_API_KEY=sk-...
```

### Google Workspace tools (optional)

EVE can search Gmail, list/create Calendar events, and list Drive files. These are
exposed to the LLM as tools (`gmail_search`, `calendar_list_events`,
`calendar_create_event`, `drive_list_files`). Until you configure OAuth they simply
return a structured `{"error": ...}` the model can read — nothing crashes.

To enable them:

1. **Create an OAuth client** in the [Google Cloud Console](https://console.cloud.google.com/):
   - Create (or pick) a project, then **APIs & Services → Enable APIs** and enable
     the **Gmail API**, **Google Calendar API**, and **Google Drive API**.
   - **APIs & Services → OAuth consent screen**: configure it (External is fine for
     personal use), and add your Google account under **Test users**.
   - **APIs & Services → Credentials → Create credentials → OAuth client ID**, type
     **Desktop app**. (Desktop clients allow the `localhost` redirect EVE uses.)
2. **Add the credentials to `.env`:**

   ```env
   GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=xxxx
   GOOGLE_TOKEN_PATH=./.secrets/google_token.json   # where the cached token is stored
   ```
3. **First run consents once.** The first time EVE calls a Google tool it opens a
   browser for you to grant access, then caches the token at `GOOGLE_TOKEN_PATH`;
   later runs reuse and auto-refresh it. Approve the scopes it requests:
   `gmail.readonly`, `calendar.events`, `drive.metadata.readonly` (defined as
   `SCOPES` in `eve/tools/adapters/google.py`).

> **Re-consent after changing scopes.** If you edit `SCOPES`, delete the cached
> token file so EVE runs the consent flow again with the new permissions.

> ⚠️ `calendar_create_event` **writes** to your calendar, so it's flagged
> `destructive=True`. Before any destructive tool runs, EVE pauses and asks you to
> confirm over the current channel — a `[y/N]` prompt in text mode, or a spoken
> "shall I go ahead?" in voice mode (`Agent._confirm_destructive` in `eve/agent.py`).
> Anything that isn't a clear yes is treated as a decline.

### Web search (optional)

EVE can search the live web via [Tavily](https://tavily.com/) — exposed as the
read-only tool `web_search`. It returns ranked results (title, URL, snippet) plus a
short synthesized answer, so the model can look up current facts beyond its training
cutoff. Like the Google tools, it returns a structured `{"error": ...}` until
configured, so nothing crashes.

Get a free API key at [app.tavily.com](https://app.tavily.com) (no card required)
and add it to `.env`:

```env
TAVILY_API_KEY=tvly-xxxxxxxxxxxxxxxx
```

That's all — `web_search` is read-only, so there's no OAuth and no confirmation gate.

### Visualizer window (optional)

EVE ships with a **glass-panel voice visualizer**: a rotating neural-network orb
(amber particle sphere, signal pulses, glowing core, orbital rings + HUD arcs)
that reacts to EVE's conversational state. It's a zero-dependency web frontend
(HTML5 Canvas) served by a small stdlib HTTP server — no Electron, no npm, no
extra pip installs.

```bash
make window          # preview the window standalone (synthetic animation)
make window-voice    # voice mode with the live orb attached
# or directly:
python -m eve.ui                    # standalone preview
python main.py --mode voice --window
```

When the agent runs with `--window`, it pushes its pipeline state to the browser
over **Server-Sent Events**, so the orb mirrors the real loop:
`listening → thinking → speaking → idle`. In the browser, the orb's *loudness*
(radius/brightness pulsing) is driven by the **real microphone** via the Web
Audio API; the *state* comes from the Python agent. Without a backend the window
is fully self-contained — every control animates the orb locally.

The renderer (`eve/ui/web/eve-orb.js`) and its tuning file
(`eve/ui/web/eve.config.json`) come straight from the design handoff; retune
`particleCount`, palettes, depth-of-field, rotation, and per-state energy there.

```
eve/ui/
  server.py            stdlib HTTP + SSE bridge (VizServer); drives the orb from agent state
  __main__.py          `python -m eve.ui` — standalone window preview
  web/
    index.html         glass panel: title bar · canvas · controls
    styles.css         design tokens + window chrome
    app.js             orb wiring · mic energy · controls · SSE client
    eve-orb.js         the neural-orb renderer (from the handoff)
    eve.config.json    palettes / particle count / motion / per-state energy
```

---

## Self-improvement loop (sleep-time compute, optional)

While you're away, EVE can put the idle machine to work on itself: a heavier
local model researches, implements, and test-gates small improvements to EVE's
own codebase, and consolidates conversational memory. When you come back, the
loop yields instantly — conversation always runs on your (lighter) chat model.

"Idle" means **not talking to EVE** — nothing else. The timer is driven only by
EVE conversation turns; EVE never watches your keyboard, mouse, or screen. Use
your computer however you like and EVE keeps improving in the background until
you next speak or type to it.

The design borrows from published work on self-improving agents:

- **Sleep-time compute** ([Letta](https://www.letta.com/blog/sleep-time-compute/)):
  a heavy model works offline during idle time; a light model handles live chat.
- **Test-gated self-editing** ([SICA](https://github.com/MaximeRobeyns/self_improving_coding_agent),
  ICLR '25): an agent edits its own codebase, keeping only changes that pass its
  test gate.
- **Sandbox + traceability** ([Darwin Gödel Machine](https://sakana.ai/dgm/)):
  self-modification happens in isolation, every change is archived, and a human
  supervises what actually ships.
- **Reflection** ([Generative Agents](https://arxiv.org/abs/2304.03442)): idle-time
  distillation of episodic memories into durable insights.

### Enable it

```bash
ollama pull ornith:35b        # or any strong local coding model with tool support
```

```env
SELF_IMPROVE=true
IMPROVE_MODEL=ollama_chat/ornith:35b
IMPROVE_IDLE_SECONDS=180      # how long you must be away before a cycle starts
```

```bash
make improve                  # text mode with the loop on (or add --improve to any run)
make improve-status           # journal + sandbox branches at a glance
```

Ask EVE *"what have you been improving?"* in any conversation — the
`self_improvement_status` tool reads the loop's live state and journal.

### One cycle

```
reflect (occasionally) → research → implement → verify (pytest) → review → commit
```

Three subagents run on `IMPROVE_MODEL` with role-scoped tools: a **researcher**
(web search + read-only code access) turns the cycle's focus area — memory,
architecture, code quality, tests, docs, performance, rotating — into concrete
ideas; an **engineer** (guardrailed file tools + pytest) implements exactly one
idea; a **reviewer** (read-only) judges the diff. The full test suite must pass
— mechanically, not on the model's word — before anything is committed.
Occasionally the loop also "reflects": it distills recent episodic memories
into durable procedural insights (strictly additive — it can only `add`).

### Safety rails (mechanical, not prompt-level)

- **Never main.** All work happens in a git-worktree sandbox under
  `IMPROVE_HOME/worktrees/`, committed to a `self-improve/<timestamp>` branch.
  The commit path re-checks the branch name every time; there is no push.
- **Memory is never wiped.** `MEMORY_DIR` must live outside the sandbox (file
  tools physically can't reach it), and any diff that *adds* deletion code
  (`rmtree`, `os.remove`, …) is rejected before commit.
- **Bounded blast radius.** Per-cycle file budget (`IMPROVE_MAX_FILES`),
  protected paths (`.env`, `.secrets`, `.git`, the guardrails module itself),
  and a per-session cycle cap (`IMPROVE_MAX_CYCLES`).
- **Full audit trail.** Every cycle — committed, skipped, rejected, or reverted
  — gets a markdown record under `IMPROVE_HOME/journal/`.

**You merge; EVE doesn't.** Review a branch and merge it like any PR:

```bash
git log main..self-improve/20260701-183000 --stat
git diff main...self-improve/20260701-183000
git merge --no-ff self-improve/20260701-183000   # onto a branch YOU choose
git worktree remove ~/.eve/improve/worktrees/20260701-183000  # tidy up after
```

```
eve/improve/
  loop.py         orchestrator: idle-gated cycle state machine (daemon thread)
  activity.py     ActivityMonitor — the idle clock; conversation always wins
  subagent.py     researcher / engineer / reviewer roles + the constitution
  workspace.py    git-worktree sandbox: guardrailed file tools, pytest, commit
  guardrails.py   the hard lines: worktree-only writes, branch discipline,
                  memory untouchable, dangerous-diff scan, file budget
  reflection.py   sleep-time memory consolidation (additive-only)
  codebase.py     self-awareness: compact repo map for prompts
  journal.py      per-cycle markdown audit trail
  state.py        backlog + history across sessions (JSON, outside the repo)
```

---

