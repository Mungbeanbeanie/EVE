# EVE — your personal AI agent

EVE is a voice-driven AI agent that lives on your computer: it listens, remembers
what matters across sessions, and acts through Google and web-search tools. It runs
as a single local process.

## How it works

```
VOICE   Mic ─▶ STT (faster-whisper) ─▶ LLM (any provider) ─▶ TTS ─▶ Speaker
MEMORY  turn ─▶ working memory ─▶ { procedural, episodic }  (mem0 → FAISS on disk)
TOOLS   LLM ─▶ tool registry ─▶ Google / web-search adapters ─▶ result
```

- **Provider-agnostic LLM** via LiteLLM (`anthropic/…`, `openai/…`, `gemini/…`, `ollama/…`).
- **Long-term memory on disk** — a FAISS index embedded with FastEmbed (ONNX).
- **Visualizer window** — a glass-panel orb that reflects EVE's state (listening / thinking / speaking).

## Prerequisites

- Python 3.11+
- An LLM API key (or a local Ollama model)
- For voice: `portaudio` — `brew install portaudio` (macOS) or
  `apt-get install portaudio19-dev espeak` (Debian/Ubuntu)

Memory is a local file; the embedder model (~520MB) downloads and caches on first run.

## Quickstart — always-on, with the visual interface (default)

```bash
make setup            # venv + deps + .env
#  → edit .env: set LLM_MODEL and the matching API key
make install-voice    # mic / STT / TTS stack
make install-window   # native menu-bar window (macOS)
make install-agent    # start EVE at login: voice + window, auto-restart
```

`install-agent` registers a macOS **launchd LaunchAgent** that launches EVE
automatically once you're logged in (voice mode with the visualizer) and restarts it
if it crashes. Logs: `~/Library/Logs/eve.{out,err}.log`. Remove with `make uninstall-agent`.

### Run manually / develop

```bash
make run              # text-mode REPL (no audio)
make voice            # voice mode
make window-voice     # voice + native window
make test             # smoke tests
```

## Configure (`.env`)

| Key | What |
| --- | --- |
| `LLM_MODEL` | LiteLLM string, e.g. `anthropic/claude-opus-4-8`, `openai/gpt-4o` |
| `LLM_API_KEY` | key for that provider (or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / …) |
| `MEMORY_DIR` | where long-term memory persists (default `~/.eve/memory`) |
| `EMBEDDER_MODEL` / `EMBEDDING_DIMS` | in-process FastEmbed model + its dims (default nomic, 768) |
| `ELEVENLABS_API_KEY` | optional cloud TTS; falls back to local pyttsx3 voice |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | optional Gmail/Calendar/Drive tools (OAuth) |
| `TAVILY_API_KEY` | optional live web search |

Everything optional degrades gracefully — a missing key or model never crashes a turn,
it just disables that capability. See `.env.example` for the full annotated list.

## Notes

- **Voice output** uses offline `pyttsx3` by default; set `ELEVENLABS_API_KEY` for cloud
  voices (incl. cloned). Pick one with `ELEVENLABS_VOICE_ID` (`python -m eve.pipeline.tts`).
- **Window** is a zero-dependency HTML5 canvas served over SSE — no Electron, no npm.
  Without the native deps, `--window` just opens the orb in your browser.
- **Google tools** run a one-time browser OAuth consent on first use, caching a token at
  `GOOGLE_TOKEN_PATH`. `calendar_create_event` writes, so EVE asks to confirm before it runs.
- **PyAudio fails to build (Intel macOS)?** Point it at Homebrew's portaudio:
  `export CFLAGS="-I$(brew --prefix portaudio)/include" LDFLAGS="-L$(brew --prefix portaudio)/lib"`

## Layout

```
eve/  config · agent · pipeline (audio · vad · stt · tts) · llm · memory · tools · ui
main.py   entrypoint: --mode {voice,text} [--window]
deploy/   launchd LaunchAgent (always-on) + installer
```

## License

MIT — see [LICENSE](LICENSE).
</content>
</invoke>
