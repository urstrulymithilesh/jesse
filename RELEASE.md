# Jesse: full-vision completion gates

Agreed 2026-10-05: finish the full vision, including **yo Jesse**, a custom voice,
and calling a phone number. Do not substitute more unrelated bug fixes for these
milestones. This file tracks acceptance, not estimated percentages.

| Milestone | Current evidence | Done when |
|---|---|---|
| Core conversation and memory | Unit suite and real-stack smoke pass | Keep those gates passing; complete a normal headset conversation with memory recall, timer, interruption and restart |
| Custom wake phrase | Asset loading and WAV evaluation tooling; training recipe prepared | A trained `yo_jesse.onnx` meets sampled recall/false-activation gates and works with Mithilesh's real voice, including interruption and echo checks |
| Own-voice clone | Voice brief agreed; reviewed recording workflow ready | Accepted recordings, trained/exported Piper model, user-approved voice audition, measured CPU synthesis speed and interruption test |
| Phone calls | G.711 codec and existing transport seam | Echo-safe call transport, authenticated Media Streams/TwiML integration, public endpoint, and a real call using Mithilesh's account/number |
| Release usability | Windows launcher and current instructions | Start reliably from a fresh terminal, diagnose missing assets, preserve memory across restart, document actual limitations |

## Voice brief

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
3. Expand the voice dataset, fine-tune/export Piper, audition it, and measure latency.
4. Validate a normal local conversation using the selected wake/voice assets.
5. Build phone transport with an explicit echo strategy. Local half-duplex tests
   are not evidence of full-duplex phone safety. Then connect the account/number
   and complete a real call. Do not revisit the paused Tailscale setup as the goal.

Training, microphone acceptance and a live call are **not complete**. The repository
currently contains no trained `yo Jesse` model or own-voice clone. No voice recordings
have been uploaded. GPU acceleration and speaker authentication remain optional
unless Mithilesh changes the earlier decision to defer/skip them.

See [training/README.md](training/README.md) for collection and training commands.
