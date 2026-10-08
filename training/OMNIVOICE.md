# Clone Mithilesh's voice with OmniVoice

User selection: **k2-fsa/OmniVoice**, 2026-10-07. This replaces Piper fine-tuning
as the first voice-clone experiment. The original recording, reference excerpts,
transcripts, voice prompts and auditions stay in ignored **data/voice-clone/**.
Nothing is uploaded. Ryan remains the default. An optional inference backend is
integrated, but the eight-step sample was rejected as robotic on 2026-10-08 and
no own-voice candidate is accepted yet.

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
timer responses in a normal headset session. The optional backend has headless
cancellation/restart evidence, but no user-approved voice. Keep generated speech separate from microphone
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
headset interruption remain unverified. No benchmark selects a production voice.

## Optional live backend (2026-10-08)

Jesse can run OmniVoice in a separate, persistent local Python process. It loads
the model and saved prompt once, keeping the codec on CPU and the diffusion model
on GPU. Jesse's existing Python environment needs no new dependencies. The worker
uses offline model loading and has no HTTP listener. Piper stays the default.

The private profile data/voice-clone/runtime-conversational.json reproduces the
rejected speed experiment below. Keep it for diagnosis; do not recommend it as
an accepted everyday voice:

~~~json
{
  "python": "../omnivoice-gpu-env/Scripts/python.exe",
  "model": "../omnivoice-model",
  "prompt": "audition-conversational/voice-prompt.pt",
  "mode": "cuda-fp16",
  "steps": 8
}
~~~

Paths are relative to the profile. The local candidate profile uses eight steps;
omitting steps defaults to 16. Mode cpu is available explicitly but is much slower.
Only use your own locally prepared prompt. A requested GPU backend fails clearly
if unavailable; it does not silently substitute another voice or CPU inference.

~~~powershell
.venv/Scripts/python.exe -m jesse run --ollama --ui --omnivoice-profile data/voice-clone/runtime-conversational.json
~~~

The option also works through Start-Jesse.cmd. It cannot be combined with --voice.
Model loading completes before microphone capture. Audio is generated one sentence
at a time, then played in short PCM chunks; this is not token-level audio streaming.
Constant peak gain matches the audible audition previews, without pitch shifting.

Both desk and remote playback pull synthesis off the microphone event loop.
An interruption during generation kills and reaps the worker and returns no stale
audio; the next request starts a fresh worker. An interruption after generation
stops chunk playback without unloading the model. Shutdown also reaps the worker.
Requests time out after 180 seconds. Phone acoustic echo handling is still separate.

Headless verification, without opening microphone or speakers:

~~~powershell
.venv/Scripts/python.exe -m training.smoke_omnivoice --profile data/voice-clone/runtime-conversational.json --output data/voice-clone/new-runtime-check
~~~

The first real check reused one worker for two replies (6.14s and 4.66s). A stop
signal at 0.50s returned at 0.54s with no audio and the worker reaped. The next
reply succeeded in 21.51s including reload. Cold startup took 55.21s in this run;
earlier standalone benchmark setup was around 10s, so startup is variable.
These single-run numbers are not a daily-use latency guarantee. This candidate
still needs human likeness, naturalness, and live headset acceptance.

## Robotic-voice rejection and quality comparison

Mithilesh rejected runtime-smoke-2026-10-08/second.wav as very robotic on
2026-10-08. Word accuracy and worker correctness did not establish voice quality.
The rejected run used eight steps with the conversational reference and CUDA fp16.
Upstream defaults to 32 steps and documents a quality/speed tradeoff; eight steps
is a plausible contributor, not a proven cause of the reported robotic delivery.

New local comparisons use the same sentence and fixed seed 42:

| Reference | Precision | Steps | Generation | Audio | Purpose |
|---|---|---:|---:|---:|---|
| Conversational | CUDA fp32 | 8 | 5.39s | 4.86s | Controlled step-count baseline |
| Conversational | CUDA fp32 | 32 | 16.55s | 4.94s | A: higher-step version |
| Expressive | CUDA fp32 | 32 | 19.54s | 5.04s | B: change only the reference versus A |

Outputs remain ignored under data/voice-clone/quality-conversational-fp32 and
quality-expressive-fp32. Compare each steps-32-listen.wav. Both are mono 24kHz,
unclipped, and local Whisper recovered the intended words. The preview is constant
gain applied before PCM quantization; its length/rate match the raw output. Reference
excerpts also contain complete phrases according to the saved word timings.

This found no obvious sample-rate/clipping error. It cannot establish whether the
remaining issue is prosody, voice identity, the recording, or model limitations.
The fp32 samples are offline auditions (the runtime currently supports fp16/CPU).
Do not change runtime defaults or claim a quality fix until the user judges them.
If both fail, diagnose the reference/codec and try another clean excerpt before
spending effort on more speed tuning. No new recording or cloud upload is needed
for this comparison.

## Natural-tone work from B

Mithilesh preferred the expressive 32-step B sample as somewhat natural, but wants
to avoid the feeling of talking to a robot. Keep B as the quality baseline; this
is not approval to ship it as the default voice.

The audition tool now accepts --speed (finite 0.5–2, below 1 means slower) and
--keep-pauses (disable output silence removal). Reports record the effective
generation overrides and seed 42. Omitting both preserves the previous behavior.
These are offline audition options; runtime settings have not changed.

Two new 32-step CUDA fp32 samples use the same words and expressive recording:

| Sample | Change | Generation | Audio |
|---|---|---:|---:|
| tone-relaxed-fp32 | Saved B prompt; speed 0.95; output pauses retained | 19.57s | 5.28s |
| tone-original-pauses-fp32 | Re-encode B's full excerpt without reference silence removal; default speed; output pauses retained | 27.54s | 7.00s |

Both live under data/voice-clone; listen to steps-32-listen.wav in each directory.
Local Whisper recovered all intended words and both files are unclipped mono
24kHz PCM. Those checks do not measure naturalness. The second variant changes
the conditioning tokens and estimated duration; it does not copy a measured
pause pattern exactly. Slower output is not automatically more natural.

The untrimmed prompt is data/voice-clone/audition-expressive-untrimmed/voice-prompt.pt,
created with create_voice_clone_prompt(..., preprocess_prompt=False), using the
same expressive.wav and exact transcript. Its report records the reference hash.
The original prompt is preserved. The full recording remains local.

Example to reproduce the first variant into a new directory:

~~~powershell
data/omnivoice-gpu-env/Scripts/python.exe training/benchmark_omnivoice.py --prompt data/voice-clone/audition-expressive/voice-prompt.pt --output data/voice-clone/new-tone-audition --mode cuda-fp32 --steps 32 --speed 0.95 --keep-pauses --text "That sounds exciting. Tell me a little more about what you have in mind."
~~~

Important upstream limitation: this installed revision validates instruct against
a fixed list of age/gender/pitch/accent/whisper tags. Freeform directions such as
warm, relaxed, conversational are rejected; they are not supported emotion controls.
An actual attempt failed before synthesis and its error report is retained under
tone-warm-fp32. No freeform direction option was added. See the installed
omnivoice/models/omnivoice.py::_resolve_instruct for the exact allowed vocabulary;
do not bypass it or claim a natural-tone prompt works without a real generation.

Next: compare the two auditions with B. If they remain stiff, a short spontaneous
voice note in the user's desired speaking style is a useful new reference experiment.
Do not promise that a longer script or additional steps alone will solve prosody.
