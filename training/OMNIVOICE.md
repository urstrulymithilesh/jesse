# Clone Mithilesh's voice with OmniVoice

User selection: **k2-fsa/OmniVoice**, 2026-10-07. This replaces Piper fine-tuning
as the first voice-clone experiment. The original recording, reference excerpts,
transcripts, voice prompts and auditions stay in ignored **data/voice-clone/**.
Nothing is uploaded. **Current status, 2026-10-11:** the user selected faster
sample A, using 16 steps. It is selected on this PC by data/voice-clone/default.json.
It uses the untrimmed expressive prompt, CUDA fp32, preserved output pauses and noise
cleanup. The older auditions below are retained as experimental history. the fallback voice is
available with --voice en_US-pinkman-fallback; installations without a local default
profile continue to use Piper. Quality is accepted; latency and live headset
acceptance remain open.

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

The existing Pinkman environment is unchanged. CPU float32 avoids assuming the
older GPU supports the upstream CUDA stack. No paid compute or cloud voice
processing is configured. CPU speed must be measured on actual generated speech.

## Make an audition

First select a complete 3–10 second WAV excerpt and write its exact words to a
UTF-8 text file. Do not use the full paragraph as one reference. Then run:

~~~powershell
data/omnivoice-env/Scripts/python.exe training/clone_omnivoice.py --reference data/voice-clone/conversational.wav --transcript data/voice-clone/conversational.txt --text "Hey, it's Pinkman. Good to hear from you. What are we working on today?" --output data/voice-clone/new-audition
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

Pinkman can run OmniVoice in a separate, persistent local Python process. It loads
the model and saved prompt once, keeping the codec on CPU and the diffusion model
on GPU. Pinkman's existing Python environment needs no new dependencies. The worker
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
.venv/Scripts/python.exe -m pinkman run --ollama --ui --omnivoice-profile data/voice-clone/runtime-conversational.json
~~~

The option also works through Start-Pinkman.cmd. It cannot be combined with --voice.
Model loading completes before microphone capture. Audio is generated one sentence
at a time, then played in short PCM chunks; this is not token-level audio streaming.
Constant peak gain matches the audible audition previews, without pitch shifting.

Both desk and remote playback pull synthesis off the microphone event loop.
An interruption during generation returns promptly without stale audio. The worker
checks cancellation between diffusion steps and keeps the model loaded. Requests
have IDs, and the cancelled request's final response is discarded before another
request runs. A two-second cancellation deadline kills/reaps an unresponsive
worker even if no new turn arrives; the next request then reloads it. Interruption
during initial loading also stops the worker. An interruption after generation
stops chunk playback without unloading the model. Shutdown reaps the worker.
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

## Preferred tone and static cleanup

The user chose take 2: tone-original-pauses-fp32/steps-32-listen.wav, saying
it is good but has background static. Preserve that take's tone and pacing.
Its reference is the untrimmed expressive prompt; CUDA fp32, 32 steps, default
speed, output silence removal disabled. The old eight-step runtime profile does
not reproduce the preferred voice and must not be described as equivalent.

There is measurable noise in the saved file's quiet sections. The generated raw
file is near -80 dBFS there; the audition's 23.08x gain raises it to around -54 dBFS.
Gain amplifies existing noise along with speech; this does not prove whether the
noise originated in the reference recording or the model/codec. No clipping or
sample-rate error was found.

An offline cleanup now preserves the exact generated take:
data/voice-clone/tone-original-pauses-fp32/steps-32-clean-profiled.wav.
It uses FFmpeg afftdn with 12 dB reduction, a noise profile sampled from the known
quiet 0.2–1.2s interval, and gain smoothing. Padding/trimming compensates the measured
600-sample filter delay. No speech regeneration, time stretching, silence removal,
or additional gain was applied. The original remains available.

Reproduce to a new file (FFmpeg -n refuses overwriting):

~~~powershell
ffmpeg -hide_banner -loglevel error -n -i data/voice-clone/tone-original-pauses-fp32/steps-32-listen.wav -af "apad=pad_len=600,asendcmd=c='0.2 afftdn sn start;1.2 afftdn sn stop',afftdn=nr=12:nf=-40:gs=8,atrim=start_sample=600" -c:a pcm_s16le data/voice-clone/tone-original-pauses-fp32/new-cleaned.wav
~~~

Measured on the 3–4s quiet interval: -53.78 to -65.72 dBFS (11.94 dB reduction).
Speech at 4.5–6.5s changed from -18.40 to -18.57 dBFS (0.17 dB quieter). Duration
remains 7.00s, sample rate 24kHz mono, cross-correlation lag is zero, and output
does not clip. Local Whisper recovered every intended word. These measurements
do not replace the user's judgment of hiss or processing artifacts.

The profile interval belongs to this take only: do not hardcode it into runtime
cleanup, where that interval could contain speech. The first generic filter attempt
only reduced pause noise by 0.92 dB and is retained as steps-32-clean.wav; it is
not the candidate to present. Detailed source hashes, settings and measurements
are in steps-32-clean-profiled-report.json. No runtime filter is enabled yet.
FFmpeg reference: https://ffmpeg.org/ffmpeg-filters.html#afftdn

## Approved voice in the conversation loop (2026-10-09)

The user approved the cleaned take and explicitly asked to proceed. The worker
now supports cuda-fp32, keep_pauses and an optional reproducibility seed. The local
runtime-approved.json contains:

~~~json
{
  "python": "../omnivoice-gpu-env/Scripts/python.exe",
  "model": "../omnivoice-model",
  "prompt": "audition-expressive-untrimmed/voice-prompt.pt",
  "mode": "cuda-fp32",
  "steps": 32,
  "keep_pauses": true,
  "seed": 42,
  "noise_reference": "approved-noise-reference.wav"
}
~~~

The same settings in private data/voice-clone/default.json select this voice for
Start-Pinkman.cmd and pinkman run. An explicit --omnivoice-profile selects another
profile; --voice selects Piper. Missing/corrupt configured assets or missing FFmpeg
fail clearly. There is no silent CPU/voice substitution. Private assets stay local.

Cleanup uses a one-second, 24kHz mono PCM16 noise-only excerpt from the approved
take, saved as approved-noise-reference.wav. For each new reply, it prepends 1.5s
of this calibration audio, profiles that prefix with the accepted afftdn settings,
then removes the prefix and 600-sample filter delay. It never samples a new reply's
opening words as noise. Output length must match the input exactly. No additional
post-cleanup normalization is applied.

FFmpeg pipe I/O runs on a separate thread because Windows communicate(timeout)
can otherwise block while writing stdin. Cancellation and a ten-second deadline
kill/reap the cleanup process. A failed cleanup produces an explicit voice error;
it does not play the uncleaned reply. Generation interruption now keeps a responsive
model worker loaded, as described above. Cancellation during cleanup also keeps it warm.

Processing the approved take through this generalized path gives correlation
0.999992, zero alignment lag, and a speech-level difference of -0.0044 dB versus
the accepted file. Regeneration through the actual worker produced correlation
0.999992 as well, with pause noise -65.83 dBFS. This is numeric parity for one take,
not a guarantee that every future sentence has identical quality.

Actual worker reuse/interrupt/recovery evidence is under
data/voice-clone/runtime-approved-smoke-2026-10-09. Two replies took 24.32s/27.09s;
the 0.50s interruption returned at 0.57s with no PCM; recovery took 43.18s including
reload. Startup took 55.22s. All three WAVs retained their intended words in local
Whisper. These settings prioritize the approved quality and remain slow.

The new full conversation check uses real wake detection, Whisper, Ollama, cloned
speech, and a temporary memory store with simulated hardware:

~~~powershell
.venv/Scripts/python.exe -m training.smoke_omnivoice_loop --profile data/voice-clone/runtime-approved.json --output data/voice-clone/new-approved-loop-check
~~~

Its audio iterator is pulled off the event loop like real playback. The ordinary
14-scenario smoke continues to use Piper; run both when changing this integration.

## Latency profiling and faster auditions (2026-10-11)

The benchmark now accepts --profile-stages to measure _generate_iterative and
_decode_and_post_process separately. CUDA is synchronized around each timed stage;
timings accumulate across chunks and wrappers are restored on success or failure.
This optional instrumentation does not change the live worker or its settings.

Measured the approved seven-second take with 4, 1, 2, then 4 CPU threads in one
loaded process. Totals were 27.13, 27.55, 26.92 and 26.81 seconds. Generation alone
took 25.45–25.90s; decoding took 1.24–2.10s. Four-thread outputs were byte-identical;
the others differed only numerically (correlation above 0.999999999999). The small
timing differences do not establish a thread-tuning improvement. Evidence is in
ignored data/voice-clone/latency-profile-2026-10-11/report.json.

New controlled auditions use the approved untrimmed expressive prompt, fp32,
seed 42, preserved pauses, four CPU threads and identical words:

~~~powershell
data/omnivoice-gpu-env/Scripts/python.exe -m training.benchmark_omnivoice --prompt data/voice-clone/audition-expressive-untrimmed/voice-prompt.pt --output data/voice-clone/new-latency-comparison --mode cuda-fp32 --steps 16 24 32 --keep-pauses --profile-stages --text "That sounds exciting. Tell me a little more about what you have in mind."
~~~

| Steps | Generation including decode | Iterative generation | Decode | Cleanup |
|---|---|---|---|---|
| 16 | 14.62s | 12.991s | 1.627s | 0.10s |
| 24 | 21.09s | 18.789s | 2.296s | 0.14s |
| 32 | 26.42s | 25.036s | 1.373s | 0.14s |

Cleanup uses pinkman.tts.cleanup.clean_pcm on each normalized *-listen.wav with the
existing approved-noise-reference.wav. Present only steps-16-clean.wav and
steps-24-clean.wav under data/voice-clone/latency-auditions-2026-10-11. Every result
is seven seconds long, unclipped, and preserves all words in local Whisper. The
32-step cleaned control matches the accepted take at correlation 0.999992.
report.json and quality-checks.json retain measurements locally.

These single-run timing comparisons exclude startup, LLM and STT. They do not
prove equal naturalness or conversational latency. The user has not judged the
16/24-step versions; default.json remains the approved 32-step profile. All audio
and prompts remain local. The three real generations plus cleanup/ASR checks are
the smoke evidence for this benchmark-only change; no live runtime code changed.

## Sample A selected for daily use (2026-10-11)

The user answered "A", selecting latency-auditions-2026-10-11/steps-16-clean.wav.
Private runtime-approved-16.json and default.json now use 16 steps. All other
settings stay as in the approved 32-step profile: untrimmed expressive prompt,
CUDA fp32, seed 42, output pauses retained and the saved noise reference. The
original runtime-approved.json remains available as the 32-step alternative.

Actual persistent-worker verification under runtime-16-smoke-2026-10-11 passed:
two replies took 12.45/14.61s, worker reused, interruption at 0.50s returned at
0.55s with no audio and the worker reaped, recovery took 26.78s including reload.
Startup was 13.46s in this run; cache state means it cannot be compared directly
with the earlier cold-start measurement. All three replies passed local Whisper
word checks and were unclipped. The regenerated second reply correlates
0.999999984 with approved sample A. The smoke report's user_approved_similarity
flag remains false because its new recordings were not independently reviewed;
the user's "A" judgment refers to the audition, with numeric runtime parity checked.

No inference algorithm changed. These runtime checks exercise the supported
profile setting selected by the user; the live headset acceptance is still open.
The full cloned-voice conversation smoke also passed with real wake/Whisper/Ollama
and temporary memory (simulated hardware): approved-16-loop-2026-10-11, 47.18s total
including startup, 3s reply audio, listening state restored and model closed.

## Keep the approved voice loaded through interruption (2026-10-14)

The local worker now reads cancellation while generating. A temporary PyTorch
forward pre-hook checks between diffusion steps, unwinds that generation and is
removed on every exit. The pinned upstream model files are unchanged. A late
response cannot become the next reply: request IDs and a drain before reuse
discard it. Cancellation during loading, worker errors and a stuck cancellation
still use process teardown; normal generation cancellation keeps the model warm.

Real evidence: data/voice-clone/pinkman-warm-interrupt-2026-10-14/report.json.
Startup took 58.84s; two replies took 17.66s and 17.14s. A stop at 0.50s returned
at 0.56s without audio, then the next reply took 13.80s with the same worker.
Repeating the second sentence took 16.71s and produced byte-identical WAV output,
also identical to the earlier relocation smoke's second.wav. These are single-run
synthesis measurements, not complete conversation latency or a new voice audition.
The smoke now requires reuse after interruption and unchanged seeded output.

The complete manual-wake/Whisper/Ollama/clone/memory loop passed in 72.24s including
startup, returned to listening, and closed its worker. Evidence:
data/voice-clone/pinkman-warm-interrupt-loop-2026-10-14/report.json. Hardware was
simulated and memory temporary. The approved profile, reference and cleanup remain
unchanged; a normal wired-headset session and custom wake training remain pending.
