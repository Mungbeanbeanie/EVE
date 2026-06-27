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
- **Postgres + pgvector** — easiest via Docker (below).
- An **LLM API key** for whatever provider you choose (or a local Ollama model).
- **No torch / no GPU needed** — STT uses **faster-whisper** (CTranslate2), which runs on
  CPU and works on Intel macOS where PyTorch no longer ships wheels.

> ⚠️ Voice mode needs host mic/speaker access, so run it on your **host machine**, not
> inside Docker. Docker is best for Postgres and for text mode.

---

## Quickstart

### 1. Database (Docker)

```bash
docker compose up db          # starts Postgres with pgvector + runs schema.sql
```

### 2. Configure

```bash
cp .env.example .env          # then fill in LLM_MODEL + key, etc.
```

### 3. Install + run (host)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # core: LLM + memory + tools (no audio stack)

python main.py --mode text        # fastest path: develop the brain without audio
```

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

---

## MVP — what's left to do

Each item maps to `# TODO(eve)` markers in the named file. Suggested order builds the
agent from the inside out (text + brain first, audio last).

**LLM (provider-agnostic)**
- [DONE] Tool-use loop in `eve/llm/litellm_client.py` (`respond`)
- [ ] (optional) Direct vendor clients in `eve/llm/providers.py`
- [ ] Harden input handling in `eve/llm/sanitize.py`

**Memory (working + procedural + episodic)**
- [DONE] Wire mem0 → pgvector in `eve/memory/mem0_backend.py`
- [DONE] `add`/`search`/`recent` for `eve/memory/procedural.py` and `eve/memory/episodic.py`
- [DONE] Context assembly in `eve/memory/working.py` (`render`)
- [DONE] Recall blending + write policy in `eve/memory/manager.py` (`recall`, `remember`)
- [ ] Confirm/tune `eve/memory/schema.sql` (embedding dimension, indexes)

**Tools (Google first, Microsoft later)**
- [ ] Register adapters in `Agent.from_config` (`eve/agent.py`)
- [ ] OAuth + Gmail/Calendar/Drive handlers in `eve/tools/adapters/google.py`
- [ ] Argument validation + destructive-action guard in `eve/tools/executor.py`
- [ ] Microsoft Graph handlers in `eve/tools/adapters/microsoft.py`

**Voice pipeline (do last)**
- [ ] VAD frames in `eve/pipeline/vad.py`
- [ ] Mic capture + playback in `eve/pipeline/audio_io.py`
- [ ] faster-whisper load + transcribe in `eve/pipeline/stt.py`
- [ ] pyttsx3 synthesis in `eve/pipeline/tts.py`

When all boxes are checked, `python main.py --mode voice` gives you a talking,
remembering, tool-using personal agent. 🎉
