#!/usr/bin/env bash
# Fetch everything needed to train the "Hey EVE" wake word locally.
# Run from training/hey-eve/:  ./setup.sh
#
# Downloads several GB (feature banks + background audio). Re-running skips files
# that already exist, so it's safe to resume an interrupted setup.
set -euo pipefail
cd "$(dirname "$0")"

DATA="./data"
mkdir -p "$DATA"

have() { command -v "$1" >/dev/null 2>&1; }
fetch() {  # fetch <url> <dest> — skip if dest already present
  local url="$1" dest="$2"
  if [ -f "$dest" ]; then echo "✓ $dest (cached)"; return; fi
  echo "↓ $dest"
  if have wget; then wget -q --show-progress -O "$dest" "$url"
  else curl -fL --progress-bar -o "$dest" "$url"; fi
}

# ── 1. Synthetic speech generator (Piper) + a voice model ────────────────────
if [ ! -d piper-sample-generator ]; then
  git clone --depth 1 https://github.com/rhasspy/piper-sample-generator
fi
mkdir -p piper-sample-generator/models
fetch \
  "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt" \
  "piper-sample-generator/models/en_US-libritts_r-medium.pt"

# ── 2. openWakeWord shared feature models (melspectrogram + embedding) ───────
# Used by the augment/feature stage. download_models() caches them in-package.
python -c "import openwakeword.utils as u; u.download_models()"

# ── 3. Precomputed negative feature banks (the big downloads) ────────────────
BASE="https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main"
fetch "$BASE/openwakeword_features_ACAV100M_2000_hrs_16bit.npy" \
      "$DATA/openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
fetch "$BASE/validation_set_features.npy" \
      "$DATA/validation_set_features.npy"

# ── 4. A slice of background audio (AudioSet balanced subset) ────────────────
fetch "https://huggingface.co/datasets/agkphysics/AudioSet/resolve/main/data/bal_train09.tar" \
      "$DATA/audioset_bal_train09.tar"

# ── 5. Room impulse responses + background-clip conversion ───────────────────
# (MIT RIRs via the datasets lib; extract+resample the AudioSet tar to 16k wav)
python download_data.py

echo
echo "Setup complete. Next:  ./train.sh"
