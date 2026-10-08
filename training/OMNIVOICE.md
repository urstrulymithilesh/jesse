# Clone Mithilesh's voice with OmniVoice

User selection: **k2-fsa/OmniVoice**, 2026-10-07. This replaces Piper fine-tuning
as the first voice-clone experiment. The original recording, reference excerpts,
transcripts, voice prompts and auditions stay in ignored **data/voice-clone/**.
Nothing is uploaded. Jesse's running voice remains Ryan until an audition is
accepted and a usable inference backend is integrated.

OmniVoice conditions a shared model on a short recording; this first clone needs
no fine-tuning. Its recommended reference length is 3–10 seconds. Keep complete
sentences with a matching transcript. A saved voice-prompt.pt contains reference
audio tokens, text and level information; it is **not a standalone Piper ONNX
voice**. Generating further speech still requires OmniVoice's base model and audio
tokenizer. See the [upstream documentation](https://github.com/k2-fsa/OmniVoice#voice-cloning).

## Local environment

This PC has a separate Python 3.13 environment in **data/omnivoice-env/** with
Torch/torchaudio 2.8.0+cpu, Transformers 5.3.0 and an editable OmniVoice source
checkout in **data/OmniVoice/** at:

    08be0b4ccbac3e13e374e86fbfead4b4cac343e2

Public model weights and the bundled tokenizer were downloaded to
**data/omnivoice-model/** at Hugging Face revision:

    c5fdb5ccb189668d56333f77ba2629f4cd7535f4

The existing Jesse environment is unchanged. CPU float32 avoids assuming the
older GPU supports the upstream CUDA stack. No paid compute or cloud voice
processing is configured. CPU speed must be measured on actual generated speech.

## Make an audition

First select a complete 3–10 second WAV excerpt and write its exact words to a
UTF-8 text file. Do not use the full paragraph as one reference. Then run:

~~~powershell
data/omnivoice-env/Scripts/python.exe training/clone_omnivoice.py --reference data/voice-clone/conversational.wav --transcript data/voice-clone/conversational.txt --text "Hey, it's Jesse. Good to hear from you. What are we working on today?" --output data/voice-clone/new-audition
~~~

The output directory must be new. The script rejects silent/clipped/non-finite
references, runs with Hugging Face offline mode, saves a reusable voice prompt,
and produces a 24-kHz audition plus timing/reference metadata. It uses 32 diffusion
steps by default and four CPU threads. Reducing steps is a speed/quality experiment,
not proof of an acceptable everyday voice.

The raw output is **audition.wav**. **audition-listen.wav** applies constant gain
to a peak of 0.85 for audible playback; it changes no pitch, timing or dynamics.
Both are kept. The first two CPU auditions took 137s/164s for 4.44s/4.68s of speech,
including reference encoding. That is too slow for live conversation on this CPU.

Listen for identity, pronunciation, accent, pacing and artifacts. Matching a
transcript alone does not establish voice similarity. Compare alternative excerpts
using the same generated text. Do not force an American accent instruction over
the recording: the reference supplies the voice and delivery.

## Remaining integration gate

After the user chooses a candidate, measure multiple natural sentences, memory and
timer responses, cancellation and restart behavior. OmniVoice is not wired into
Jesse's real-time voice loop yet. Keep generated speech separate from microphone
evaluation recordings and never use the audition to claim human wake-word recall.

## GTX 1050 acceleration measured

The existing GTX 1050 (4 GB, compute capability 6.1) and driver 512.78 passed a CUDA
calculation and actual OmniVoice generation. No driver update was necessary.
The separate **data/omnivoice-gpu-env/** uses Torch/torchaudio **2.7.1+cu118** and
Transformers 5.3.0; its freeze is **data/voice-clone/environment-gpu.txt**.
The CPU environment is still available.

Run a benchmark with a saved prompt:

~~~powershell
data/omnivoice-gpu-env/Scripts/python.exe training/benchmark_omnivoice.py --prompt data/voice-clone/audition-conversational/voice-prompt.pt --output data/voice-clone/new-gpu-benchmark --mode cuda-fp16 --steps 8 16
~~~

The benchmark loads the model once and reuses the reference prompt. The language
model, output head and embeddings run on GPU; the audio tokenizer stays on CPU.
This keeps GPU memory lower. A requested CUDA run fails explicitly if unavailable;
it never reports CPU timing as GPU timing. Reports preserve completed runs and
record later errors. Each output directory must be new.

Measured once per setting, four CPU threads, same saved conversational reference:

| Mode | Steps | Generation seconds | Audio seconds | Peak allocated GPU MiB |
|---|---:|---:|---:|---:|
| CPU dynamic int8 | 16 | 56.40 | 4.96 | — |
| CPU dynamic int8 | 32 | 159.55 | 4.46 | — |
| CUDA float32 | 16 | 11.08 | 4.82 | 2537 |
| CUDA float32 | 32 | 17.77 | 4.41 | 2537 |
| CUDA float16 | 8 | 5.95 | 4.58 | 1286 |
| CUDA float16 | 16 | 9.47 | 4.83 | 1286 |
| CUDA float16 | 32 | 18.09 | 4.42 | 1286 |

These timings exclude model loading and reference encoding. The original CPU
32-step audition took 136.87s including reference encoding, so comparison with it
is approximate. Dynamic int8 did not establish a consistent CPU improvement and
is experimental. GPU allocations are PyTorch tensor memory, not total VRAM use.
Cold model setup adds roughly 10 seconds; a live backend should keep it loaded.

A second, unseen sentence on CUDA float16 took **3.95s / 5.27s / 9.12s** at
**4 / 8 / 16** steps, producing about 4.9 seconds of audio. Local Whisper recovered
all intended words in each. Four steps is a more aggressive quality experiment,
not an accepted default. Eight steps is a candidate for user audition; 16 is the
upstream faster-inference recommendation. The 32-step half-precision greeting's
Whisper result also appended slash characters: retain that raw check rather than
treating transcription as a complete quality measurement.

Listen to **data/voice-clone/benchmark-cuda-fp16-heldout/steps-8-listen.wav** and
compare against the 16-step version. Voice likeness, naturalness and live
interruption remain unverified. No benchmark selects a production voice.
