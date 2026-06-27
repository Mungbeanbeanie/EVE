# EVE — your personal AI agent

EVE is a voice-driven AI agent that lives on your computer, remembers what matters,
and can act on your behalf through Google/Microsoft tools. **This repository is a
learning scaffold**: the architecture, interfaces, and wiring are in place, but the
substantive logic is left for you to implement — every spot is marked with a
`# TODO(eve):` comment and a docstring explaining what to do and why.

Run it at any stage: unimplemented pieces raise a clear `NotImplementedError` (or a
friendly "stub" message in the REPL) telling you exactly which file to open next.

---

## Architecture

```
VOICE   Mic ─▶ STT (faster-whisper, local) ─▶ sanitize ─▶ LLM (any provider) ─▶ TTS ─▶ Speaker

MEMORY  message ─▶ working memory ─▶ { procedural, episodic } stores (mem0 + pgvector)
                          ▲                      │
                          └──── vector recall ◀──┘

TOOLS   LLM ─▶ tool registry ─▶ tool executor ─▶ Google / Microsoft adapter ─▶ result
```

Three design choices shape the code:

- **Provider-agnostic LLM.** EVE is not tied to any vendor. `build_llm(config)` returns
  a client driven by a model string (`anthropic/…`, `openai/…`, `gemini/…`, `ollama/…`);
  the default uses [LiteLLM](https://docs.litellm.ai/) so one interface speaks to all.
- **Three memory layers** behind one `MemoryManager`: **working** (volatile context),
  **procedural** (learned preferences/skills), **episodic** (timestamped events).
  Durable layers persist in Postgres/pgvector via [mem0](https://docs.mem0.ai/).
- **Swappable everything.** The `Agent` depends only on abstract interfaces, so each
  subsystem can be replaced without touching the orchestrator.

### Layout

```
eve/
  config.py            typed settings from .env (implemented)
  agent.py             orchestrator + voice/text loops (wired; calls your stubs)
  pipeline/            audio_io · vad · stt (faster-whisper) · tts (pyttsx3)
  llm/                 base · litellm_client · providers · factory · sanitize
  memory/              manager · working · procedural · episodic · mem0_backend · schema.sql
  tools/               base · registry · executor · adapters/{google,microsoft}
  utils/logging.py
main.py                entrypoint: --mode {voice,text}
tests/                 smoke tests (skeleton wiring only)
```

---

## Prerequisites

- **Python 3.11+**
- **System packages** (not pip): `portaudio` (for PyAudio); `ffmpeg` is optional but handy.
  - macOS: `brew install portaudio` (and `brew install ffmpeg` if you want it)
  - Debian/Ubuntu: `apt-get install portaudio19-dev espeak` (+ `ffmpeg`)
- **Postgres + pgvector** — easiest via Docker (below). Backs long-term memory.
- **[Ollama](https://ollama.com)** running locally — supplies the **embedder** for
  memory. The default config uses `nomic-embed-text` (768-dim, matches `schema.sql`).
- An **LLM API key** for whatever provider you choose (or a local Ollama model).
- **No torch / no GPU needed** — STT uses **faster-whisper** (CTranslate2), which runs on
  CPU and works on Intel macOS where PyTorch no longer ships wheels.

> ⚠️ Voice mode needs host mic/speaker access, so run it on your **host machine**, not
> inside Docker. Docker is best for Postgres and for text mode.

> 💡 **Memory is optional to boot.** If Postgres or Ollama isn't up, EVE detects it
> in ~2s and degrades gracefully — the conversation still runs on working
> (in-session) memory; only durable recall across sessions is skipped. So you can
> start with just an LLM key and add the memory stack later.

---

## Quickstart

### 1. Database (Docker)

```bash
docker compose up -d db       # starts Postgres with pgvector + runs schema.sql
```

`-d` runs it in the background. Use plain `docker compose up db` to watch its logs.
First start runs `eve/memory/schema.sql` (creates the `vector` extension); mem0
then manages its own `mem0` table for the procedural/episodic memories.

### 2. Embedder (Ollama)

Memory embeds text with a local Ollama model. Start Ollama and pull the model once:

```bash
ollama pull nomic-embed-text  # one-time; 768-dim, matches schema.sql
ollama serve                  # (Ollama Desktop starts this for you)
```

### 3. Configure

```bash
cp .env.example .env          # then fill in LLM_PROVIDER/LLM_MODEL + key
```

Two things to know about `.env`:

- **`DATABASE_URL` host:** use `localhost` when running EVE on your host
  (`postgresql://eve:eve@localhost:5432/eve`). The committed example uses `db`,
  which is the Docker Compose service name — only correct when EVE itself runs in
  a container.
- **`LLM_MODEL` format:** it's a LiteLLM model string, `provider/model`
  (e.g. `groq/llama-3.3-70b-versatile`, `anthropic/claude-opus-4-8`). EVE strips
  the `provider/` prefix automatically where mem0's native client needs the bare name.

### 4. Install + run (host)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # core: LLM + memory + tools (no audio stack)

python main.py --mode text        # fastest path: develop the brain without audio
```

In text mode, each turn responds as soon as the LLM replies — durable memory is
written in the **background**, so a slow mem0 write never blocks the conversation.
On exit, EVE flushes any in-flight writes so the last turn isn't lost (this can add
a short pause when you quit).

The **voice stack is optional and installed separately** (it needs system audio
support and is the last MVP piece). When you're ready:

```bash
brew install portaudio            # macOS system dep for PyAudio
pip install faster-whisper webrtcvad pyttsx3 pyaudio

python main.py --mode voice       # once the pipeline TODOs are filled in
```

> **PyAudio fails to build?** On Intel macOS with newer Python there's no prebuilt
> wheel, so pip compiles from source and the compiler may not find Homebrew's
> portaudio headers. Point it at them explicitly:
>
> ```bash
> export CFLAGS="-I$(brew --prefix portaudio)/include"
> export LDFLAGS="-L$(brew --prefix portaudio)/lib"
> pip install --no-cache-dir pyaudio
> ```

### Run the smoke tests

```bash
pytest -q                     # checks the skeleton wires together
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

> ⚠️ `calendar_create_event` **writes** to your calendar. The shared confirmation /
> destructive-action guard in `eve/tools/executor.py` is still a TODO, so today the
> model can create events without an explicit confirmation step.

---

## MVP — what's left to do

Each item maps to `# TODO(eve)` markers in the named file. Suggested order builds the
agent from the inside out (text + brain first, audio last).

**LLM (provider-agnostic)**
- [DONE] Tool-use loop in `eve/llm/litellm_client.py` (`respond`)
- [ ] (optional) Direct vendor clients in `eve/llm/providers.py`
- [DONE] Harden input handling in `eve/llm/sanitize.py`

**Memory (working + procedural + episodic)**
- [DONE] Wire mem0 → pgvector in `eve/memory/mem0_backend.py`
- [DONE] `add`/`search`/`recent` for `eve/memory/procedural.py` and `eve/memory/episodic.py`
- [DONE] Context assembly in `eve/memory/working.py` (`render`)
- [DONE] Recall blending + write policy in `eve/memory/manager.py` (`recall`, `remember`)
- [DONE] Confirm/tune `eve/memory/schema.sql` (embedding dimension, indexes)

**Tools (Google first, Microsoft later)**
- [DONE] Register adapters in `Agent.from_config` (`eve/agent.py`)
- [DONE] OAuth + Gmail/Calendar/Drive handlers in `eve/tools/adapters/google.py`
- [ ] Argument validation + destructive-action guard in `eve/tools/executor.py`
- [ ] Microsoft Graph handlers in `eve/tools/adapters/microsoft.py`

**Voice pipeline (do last)**
- [DONE] VAD frames in `eve/pipeline/vad.py`
- [DONE] Mic capture + playback in `eve/pipeline/audio_io.py`
- [ ] faster-whisper load + transcribe in `eve/pipeline/stt.py`
- [ ] pyttsx3 synthesis in `eve/pipeline/tts.py`

When all boxes are checked, `python main.py --mode voice` gives you a talking,
remembering, tool-using personal agent. 🎉
