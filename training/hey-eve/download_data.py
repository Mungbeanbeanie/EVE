"""Prepare the augmentation datasets train.py expects: room impulse responses
and background-noise clips, all as 16 kHz mono WAV.

Two sources:
  • MIT environmental impulse responses — pulled from the Hugging Face `datasets`
    hub (already 16 kHz) into data/mit_rirs/.
  • AudioSet balanced subset — the tar that setup.sh downloaded, extracted and
    resampled to 16 kHz mono into data/audioset/.

Re-running is cheap: directories that already contain WAVs are skipped. An empty
data/fma/ is created too (config.yaml lists it as an optional second background
source you can populate with the FMA dataset for more variety).
"""

from __future__ import annotations

import glob
import os
import tarfile

import numpy as np
import soundfile as sf

DATA = os.path.join(os.path.dirname(__file__), "data")
TARGET_SR = 16_000


def _write_wav_16k_mono(path: str, audio: np.ndarray, sr: int) -> None:
    """Write `audio` to `path` as 16 kHz mono 16-bit PCM, resampling if needed."""
    if audio.ndim > 1:  # mix down to mono
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        import librosa  # imported lazily — only needed when resampling

        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=TARGET_SR)
    audio = np.clip(audio, -1.0, 1.0)
    sf.write(path, (audio * 32767).astype(np.int16), TARGET_SR, subtype="PCM_16")


def fetch_mit_rirs() -> None:
    """Download MIT environmental impulse responses into data/mit_rirs/."""
    out = os.path.join(DATA, "mit_rirs")
    os.makedirs(out, exist_ok=True)
    if glob.glob(os.path.join(out, "*.wav")):
        print(f"✓ {out} (already populated)")
        return

    import datasets  # heavy import; only needed during data prep

    print("↓ MIT impulse responses (Hugging Face)…")
    ds = datasets.load_dataset(
        "davidscripka/MIT_environmental_impulse_responses", split="train", streaming=True
    )
    for i, row in enumerate(ds):
        audio = np.asarray(row["audio"]["array"], dtype=np.float32)
        sr = int(row["audio"]["sampling_rate"])
        _write_wav_16k_mono(os.path.join(out, f"rir_{i:04d}.wav"), audio, sr)
    print(f"✓ {out}")


def extract_audioset() -> None:
    """Extract + resample the AudioSet balanced tar into data/audioset/."""
    out = os.path.join(DATA, "audioset")
    os.makedirs(out, exist_ok=True)
    if glob.glob(os.path.join(out, "*.wav")):
        print(f"✓ {out} (already populated)")
        return

    tar_path = os.path.join(DATA, "audioset_bal_train09.tar")
    if not os.path.exists(tar_path):
        print(f"⚠ {tar_path} missing — run setup.sh first; skipping background clips.")
        return

    print("↓ Extracting + resampling AudioSet clips…")
    staging = os.path.join(DATA, "_audioset_raw")
    os.makedirs(staging, exist_ok=True)
    with tarfile.open(tar_path) as tar:
        tar.extractall(staging)  # noqa: S202 — trusted Hugging Face artifact

    audio_exts = (".flac", ".wav", ".mp3", ".ogg", ".m4a")
    count = 0
    for root, _dirs, files in os.walk(staging):
        for name in files:
            if not name.lower().endswith(audio_exts):
                continue
            try:
                audio, sr = sf.read(os.path.join(root, name))
            except Exception as exc:  # skip the odd unreadable clip, keep going
                print(f"  · skip {name}: {exc}")
                continue
            _write_wav_16k_mono(os.path.join(out, f"bg_{count:05d}.wav"), audio, sr)
            count += 1
    print(f"✓ {out} ({count} clips)")


def main() -> None:
    os.makedirs(os.path.join(DATA, "fma"), exist_ok=True)  # optional 2nd bg source
    fetch_mit_rirs()
    extract_audioset()
    print("Data prep done.")


if __name__ == "__main__":
    main()
