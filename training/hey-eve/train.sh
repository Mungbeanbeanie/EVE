#!/usr/bin/env bash
# Train the "Hey EVE" wake word end-to-end via openWakeWord's train.py.
# Run from training/hey-eve/ AFTER ./setup.sh:  ./train.sh
#
# Three stages (all driven by config.yaml):
#   1. --generate_clips : TTS the target phrase + adversarial negatives
#   2. --augment_clips  : mix in RIRs + background noise, compute features
#   3. --train_model    : train the DNN, tune threshold, export ONNX
# Splitting them out means you can re-run just training after tweaking
# steps/layer_size without regenerating clips (comment stages out below).
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="config.yaml"
TRAIN_PY="$(python -c 'import openwakeword, os; print(os.path.join(os.path.dirname(openwakeword.__file__), "train.py"))')"

echo "▶ 1/3 generating synthetic clips…"
python "$TRAIN_PY" --training_config "$CONFIG" --generate_clips

echo "▶ 2/3 augmenting + extracting features…"
python "$TRAIN_PY" --training_config "$CONFIG" --augment_clips

echo "▶ 3/3 training model…"
python "$TRAIN_PY" --training_config "$CONFIG" --train_model

echo
echo "Done. Model written under ./hey_eve_model/ (look for hey_eve.onnx)."
echo "Point EVE at it:  WAKE_WORD=./training/hey-eve/hey_eve_model/hey_eve.onnx"
