# Jesse: full-vision completion gates

Agreed 2026-10-05: finish the full vision, including **yo Jesse**, a custom voice,
and calling a phone number. Do not substitute more unrelated bug fixes for these
milestones. This file tracks acceptance, not estimated percentages.

| Milestone | Current evidence | Done when |
|---|---|---|
| Core conversation and memory | 637 tests pass; 13/14 smoke passed, knowledge passed unchanged rerun after a one-word STT miss | Keep those gates passing; complete a normal headset conversation with memory recall, timer, interruption and restart |
| Custom wake phrase | Asset loading and WAV evaluation tooling; training recipe prepared | A trained `yo_jesse.onnx` meets sampled recall/false-activation gates and works with Mithilesh's real voice, including interruption and echo checks |
| Own-voice clone | Cleaned take 2 accepted; matching voice and generalized noise cleanup integrated | Acceptable latency and a normal headset conversation with the approved voice |
| Phone calls | G.711 codec and existing transport seam | Echo-safe call transport, authenticated Media Streams/TwiML integration, public endpoint, and a real call using Mithilesh's account/number |
| Release usability | Windows launcher and current instructions | Start reliably from a fresh terminal, diagnose missing assets, preserve memory across restart, document actual limitations |

## Voice brief

2026-10-07: the user selected **k2-fsa/OmniVoice** and supplied his own recording.
On 2026-10-09 he accepted cleaned take 2 and authorized integration. The approved
settings are the expressive reference with its pauses retained, 32 decoding steps,
CUDA float32, and noise cleanup. The saved prompt requires the local OmniVoice
base model and audio tokenizer; it is not a standalone Piper voice.

The persistent backend now reproduces that take closely and uses a saved noise
reference to clean new replies without sampling their speech. The private
data/voice-clone/default.json selects it for this PC. An explicit --voice option
selects Piper instead, and installations without a default profile still use Piper.

Quality is accepted; speed is still a limitation. Actual generation/cleanup took
24.32s and 27.09s for two short replies. Cold startup took 55.22s, and the first
reply after interrupting generation took 43.18s including reload. The stop signal
at 0.50s returned at 0.57s with no audio. These are single-run measurements, not
full conversation latency. A normal headset conversation remains unverified.

Use Mithilesh's own voice, with American English, approximately age 21, and a
charismatic, friendly delivery. He authorized cloning his own voice. The recordings
must supply the desired accent and energy; a text label does not guarantee them.
Keep Ryan available as the explicit faster voice option.

## Execution order

Training route chosen 2026-10-06: use the prepared Colab notebook on the **free tier
only**. No paid compute is authorized. The notebook is prepared; its first GPU
run and the trained model are still pending.

1. Keep the accepted voice quality; reduce latency and validate it on the headset.
2. Train `yo Jesse` with the upstream synthetic-data pipeline. Evaluate held-out
   recordings and real background audio locally before selecting it for daily use.
3. Verify the selected wake and voice assets together in normal conversation.
4. Validate a normal local conversation using the selected wake/voice assets.
5. Build phone transport with an explicit echo strategy. Local half-duplex tests
   are not evidence of full-duplex phone safety. Then connect the account/number
   and complete a real call. Do not revisit the paused Tailscale setup as the goal.

Training, microphone acceptance and a live call are **not complete**. The repository
currently contains no trained `yo Jesse` model. Own-voice prompts and audio exist
locally under ignored data; they are not committed or uploaded. GPU voice inference
is selected locally through the private default profile; speaker authentication remains deferred.

See [training/README.md](training/README.md) for collection and training commands.
