# EVE — task runner. `make` (or `make help`) lists everything.
#
# Fastest path from a clean clone to a running agent:
#   make setup        # venv + deps + .env
#   make db           # Postgres+pgvector in Docker (optional; memory only)
#   make run          # text-mode REPL
#
# Ship it:
#   make package      # build the Docker image
#   make dist         # versioned source tarball in dist/

# ── Config ───────────────────────────────────────────────────────────────────
VENV    := .venv
PY      := $(VENV)/bin/python
PIP     := $(VENV)/bin/pip
IMAGE   := eve
VERSION := $(shell sed -n 's/^__version__ = "\(.*\)"/\1/p' eve/__init__.py)

# Voice stack is installed separately (needs system audio libs).
VOICE_PKGS := faster-whisper webrtcvad pyttsx3 pyaudio

.DEFAULT_GOAL := help
.PHONY: help setup venv install install-voice env db db-down db-logs ollama \
        run text voice window window-voice test lint fmt docker-build package \
        up down logs dist clean clean-all

# ── Help ─────────────────────────────────────────────────────────────────────
help: ## Show this help
	@echo "EVE $(VERSION) — make targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Setup ────────────────────────────────────────────────────────────────────
setup: venv install env ## One-shot: create venv, install deps, scaffold .env
	@echo "✅ Setup complete. Next: 'make db' (optional) then 'make run'."

venv: ## Create the virtualenv if missing
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv ## Install core dependencies into the venv
	@$(PIP) install --upgrade pip
	@$(PIP) install -r requirements.txt

install-voice: venv ## Install the optional voice stack (mic/STT/TTS)
	@$(PIP) install $(VOICE_PKGS)

env: ## Create .env from .env.example if it doesn't exist
	@test -f .env || (cp .env.example .env && echo "📝 Created .env — fill in LLM_MODEL + key.")

# ── Services (Docker / Ollama) ───────────────────────────────────────────────
db: ## Start Postgres+pgvector (memory backend) in the background
	docker compose up -d db

db-down: ## Stop the database container
	docker compose stop db

db-logs: ## Tail the database logs
	docker compose logs -f db

ollama: ## Pull the local embedding model used by memory
	ollama pull nomic-embed-text

# ── Run ──────────────────────────────────────────────────────────────────────
run: text ## Alias for 'make text'

text: ## Run EVE in text mode (no audio hardware needed)
	@$(PY) main.py --mode text

voice: ## Run EVE in voice mode (requires 'make install-voice')
	@$(PY) main.py --mode voice

# ── Visualizer window ─────────────────────────────────────────────────────────
window: ## Preview the EVE visualizer window standalone (no agent, opens browser)
	@$(PY) -m eve.ui

window-voice: ## Run voice mode with the visualizer window attached
	@$(PY) main.py --mode voice --window

# ── Quality ──────────────────────────────────────────────────────────────────
test: ## Run the test suite
	@$(PY) -m pytest -q

lint: ## Lint with ruff
	@$(PY) -m ruff check eve tests main.py

fmt: ## Auto-fix lint issues with ruff
	@$(PY) -m ruff check --fix eve tests main.py

# ── Packaging ────────────────────────────────────────────────────────────────
docker-build: ## Build the Docker image (tagged eve:VERSION and eve:latest)
	docker build -t $(IMAGE):$(VERSION) -t $(IMAGE):latest .

package: docker-build ## Build the shippable Docker image
	@echo "📦 Built image $(IMAGE):$(VERSION) (also tagged :latest)."

up: ## Run the full stack (app + db) via Docker Compose
	docker compose up --build

down: ## Stop and remove the Docker Compose stack
	docker compose down

logs: ## Tail all Docker Compose logs
	docker compose logs -f

dist: ## Build a versioned source tarball in dist/
	@mkdir -p dist
	@git archive --format=tar.gz --prefix=eve-$(VERSION)/ \
		-o dist/eve-$(VERSION).tar.gz HEAD
	@echo "📦 Wrote dist/eve-$(VERSION).tar.gz"

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean: ## Remove caches and build artifacts
	@rm -rf dist .pytest_cache .ruff_cache
	@find . -type d -name __pycache__ -not -path './.venv/*' -exec rm -rf {} + 2>/dev/null || true

clean-all: clean ## Also remove the virtualenv
	@rm -rf $(VENV)
