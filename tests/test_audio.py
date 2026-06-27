import asyncio
from eve.config import Config
from eve.pipeline.audio_io import PyAudioIO

async def test():
    io = PyAudioIO(Config())
    print("Say something...")
    audio = await io.record_utterance()
    print(f"Captured {len(audio)} bytes")
    print("Playing back...")
    await io.play(audio)
    print("Done!")

asyncio.run(test())