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
