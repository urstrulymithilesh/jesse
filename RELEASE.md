# Jesse: full-vision completion gates

Agreed 2026-10-05: finish the full vision, including **yo Jesse**, a custom voice,
and calling a phone number. Do not substitute more unrelated bug fixes for these
milestones. This file tracks acceptance, not estimated percentages.

| Milestone | Current evidence | Done when |
|---|---|---|
| Core conversation and memory | Unit suite and real-stack smoke pass | Keep those gates passing; complete a normal headset conversation with memory recall, timer, interruption and restart |
| Custom wake phrase | Asset loading and WAV evaluation tooling; training recipe prepared | A trained `yo_jesse.onnx` meets sampled recall/false-activation gates and works with Mithilesh's real voice, including interruption and echo checks |
| Own-voice clone | Persistent GPU backend works; user rejected eight-step sample as robotic; 32-step comparisons await review | User-approved likeness and acceptable latency in a normal headset conversation |
| Phone calls | G.711 codec and existing transport seam | Echo-safe call transport, authenticated Media Streams/TwiML integration, public endpoint, and a real call using Mithilesh's account/number |
| Release usability | Windows launcher and current instructions | Start reliably from a fresh terminal, diagnose missing assets, preserve memory across restart, document actual limitations |

## Voice brief

2026-10-07: the user selected **k2-fsa/OmniVoice** and supplied his own recording.
The current path is local zero-shot cloning and audition before integration;
Piper fine-tuning is no longer the first voice experiment. A saved OmniVoice prompt
requires the OmniVoice model and is not a Piper-compatible exported voice.
Local GTX 1050 inference is now measured: roughly 5–6 seconds for a short
eight-step audition, with the audio codec on CPU. This is an audition candidate;
the persistent backend is now available as an explicit launch option. User likeness
approval and live headset acceptance remain pending. In the first runtime check,
warm generation took 4.66–6.14s, cold startup 55.21s, and the first reply after an
interrupted generation took 21.51s including model reload. These are single samples.
On 2026-10-08 Mithilesh rejected that eight-step voice as very robotic. It is not
an accepted voice. Two full-precision, 32-step comparisons now test the original
and expressive reference excerpts. Their naturalness still requires his judgment.

Use Mithilesh's own voice, with American English, approximately age 21, and a
charismatic, friendly delivery. He authorized cloning his own voice. The recordings
must supply the desired accent and energy; a text label does not guarantee them.
Keep the current Ryan voice available until the clone is accepted.

## Execution order

Training route chosen 2026-10-06: use the prepared Colab notebook on the **free tier
only**. No paid compute is authorized. The notebook is prepared; its first GPU
run and the trained model are still pending.

1. Collect and review the voice pilot; prepare a separate GPU training environment.
2. Train `yo Jesse` with the upstream synthetic-data pipeline. Evaluate held-out
   recordings and real background audio locally before selecting it for daily use.
3. Compare OmniVoice auditions from the user's recording, select an accepted voice,
   and measure an inference backend fast enough for conversation before integration.
4. Validate a normal local conversation using the selected wake/voice assets.
5. Build phone transport with an explicit echo strategy. Local half-duplex tests
   are not evidence of full-duplex phone safety. Then connect the account/number
   and complete a real call. Do not revisit the paused Tailscale setup as the goal.

Training, microphone acceptance and a live call are **not complete**. The repository
currently contains no trained `yo Jesse` model. Own-voice prompts and audio exist
locally under ignored data; they are not committed or uploaded. GPU voice inference
is opt-in; speaker authentication remains deferred.

See [training/README.md](training/README.md) for collection and training commands.
