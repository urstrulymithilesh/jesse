"""Real wake/STT/Ollama/memory/approved voice loop with simulated audio hardware."""

import argparse
import asyncio
import json
import tempfile
import time
import wave
from pathlib import Path

from jesse.core.state import ConversationState
from jesse.smoke import Mouth, ScriptedTransport, _build
from jesse.tts.omnivoice import OmniVoiceSynthesizer


async def run(profile, output):
    output.mkdir(parents=True, exist_ok=False)
    mouth = Mouth()
    frames = mouth.frames("hey jarvis") + mouth.frames("say hello in five words")
    transport = ScriptedTransport(frames)
    synth = OmniVoiceSynthesizer(profile)
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        orch, store, _ = _build(transport, Path(directory) / "memory.db")
        orch.synth = synth
        try:
            await orch.run()
            users = [m.content for m in orch._history if m.role == "user"]
            replies = [m.content for m in orch._history if m.role == "assistant"]
            pcm = b"".join(transport.played)
            assert users and "hello" in users[0].lower(), "Real wake/STT did not produce request"
            assert replies and pcm, "No spoken Ollama reply"
            assert orch.state in (ConversationState.IDLE, ConversationState.LISTENING)
            assert not orch._llm_lock.locked()
            assert synth._closed and synth._process is None
            assert any(m.role == "assistant" for m in store.recent()), "Reply not stored"
            with wave.open(str(output / "reply.wav"), "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(synth.sample_rate)
                wav.writeframes(pcm)
            report = {"passed": True, "users": users, "replies": replies,
                      "audio_seconds": len(pcm) / 2 / synth.sample_rate,
                      "total_seconds_including_startup": round(time.perf_counter() - started, 2),
                      "state": orch.state.value, "cleanup_enabled": synth.noise is not None,
                      "steps": synth.steps, "mode": synth.mode, "keep_pauses": synth.keep_pauses,
                      "memory": "temporary database only", "hardware": "simulated"}
            (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report, indent=2), flush=True)
        finally:
            synth.close()
            store.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.profile, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
