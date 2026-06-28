# Training a custom "Hey EVE" wake word

openWakeWord ships pretrained keywords (`hey_jarvis`, `alexa`, …) but **not**
"Hey EVE". This folder trains one. The output is a single `hey_eve.onnx` you point
EVE at — no other runtime change needed.

Training is a real ML job: it synthesises tens of thousands of spoken samples of
the phrase, mixes them with room acoustics and background noise, and trains a small
classifier against large negative-audio feature banks. **A CUDA GPU is strongly
recommended.** You have two routes.

---

## Route A — Google Colab (recommended)

The maintainer's notebook does the whole pipeline on a free GPU, so you avoid the
multi-GB local downloads and the TensorFlow/Torch install.

1. Open **[automatic_model_training.ipynb](https://github.com/dscripka/openWakeWord/blob/main/notebooks/automatic_model_training.ipynb)**
   in Colab (Runtime → Change runtime type → **GPU**).
2. When it asks for the target phrase, set it to **`hey eve`**. Mirror the knobs in
   [config.yaml](config.yaml) if you want (sample counts, adversarial negatives,
   `target_false_positives_per_hour`).
3. Run all cells. Download the resulting **`hey_eve.onnx`**.
4. Drop it here (e.g. `training/hey-eve/hey_eve_model/hey_eve.onnx`) and skip to
   [Use it in EVE](#use-it-in-eve).

---

## Route B — Local training

Heavier, but fully offline-reproducible. Expect several GB of downloads and, on
CPU, a long run.

### 1. Isolated env + training deps

```bash
cd training/hey-eve
python -m venv .venv-train && source .venv-train/bin/activate
pip install -r requirements-train.txt
```

> Kept separate from EVE's runtime [requirements.txt](../../requirements.txt) on
> purpose — these deps (Torch, TensorFlow, speechbrain) are only for *training*.

### 2. Fetch data + models

```bash
./setup.sh          # clones piper, downloads feature banks + background audio
```

This pulls: the Piper voice model, openWakeWord's feature models, the precomputed
ACAV100M + validation feature banks, an AudioSet background slice, and (via
[download_data.py](download_data.py)) the MIT room-impulse-response set — all
resampled to 16 kHz mono. Re-running skips what's already there.

### 3. Train

```bash
./train.sh          # generate clips → augment/features → train → export ONNX
```

Stages are split so you can re-run just `--train_model` after tweaking `steps` or
`layer_size` in [config.yaml](config.yaml) without regenerating clips. Output lands
in `./hey_eve_model/hey_eve.onnx`.

### Tuning quality

- **More positives** (`n_samples`) → sturdier model, slower clip generation.
- **`custom_negative_phrases`** → add near-homophones it falsely fires on
  ("hey eva", "heave", …) to suppress them.
- **`target_false_positives_per_hour`** → lower = stricter (fewer accidental
  wakes, more misses). `0.2` is a reasonable start.
- **More/varied background audio** (`background_paths`) → better real-world
  robustness; populate `data/fma/` with the FMA dataset for extra variety.

---

## Use it in EVE

Point the runtime config at the trained model (a path, not a built-in name):

```dotenv
# .env
VOICE_INPUT=wake
WAKE_WORD=./training/hey-eve/hey_eve_model/hey_eve.onnx
WAKE_THRESHOLD=0.5      # raise toward 0.6–0.7 if it wakes too easily
```

```bash
python3 main.py --mode voice      # say "Hey EVE", then your request
```

`WakeWordDetector` already treats a `.onnx`/`.tflite` value as a custom model path
(see [eve/pipeline/wake.py](../../eve/pipeline/wake.py)), so nothing else changes.

---

## Notes

- Generated data, the Piper clone, and `hey_eve_model/` are git-ignored — they're
  large and reproducible. Commit only your tuned `config.yaml` (and, if you like,
  the final `hey_eve.onnx`).
- Phrase choice matters: two-syllable-plus wake words with distinct phonetics
  detect far more reliably than short ones. "Hey EVE" is borderline-short, so lean
  on adversarial negatives and threshold tuning if you see false wakes.
