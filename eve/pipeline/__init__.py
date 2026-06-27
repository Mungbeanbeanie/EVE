"""Voice pipeline: audio capture/playback, VAD, speech-to-text, text-to-speech.

Flow:  Mic -> AudioIO (+ VAD segmentation) -> STTEngine -> [Agent/LLM] -> TTSEngine -> Speaker
"""
