# Jesse

A **local voice AI friend**. His brain, his memory and every piece of
processing run on this machine and only this machine — no cloud APIs, no cloud costs,
and no conversation or memory ever leaving it. Say a wake word, talk to Jesse, and he
replies in voice — and he *remembers* you across sessions.

He does use the network, deliberately and narrowly: **as a connection, never as a
place he lives.** He fetches the RSS feeds you list, and (once remote access lands)
you can reach him from your own devices. Neither moves an ounce of him off the box.

Built as a portfolio project under a real constraint: a laptop with a **GTX 1050
(4GB VRAM)**. That constraint drives every architecture decision, and the whole
thing is designed so a future GPU upgrade means swapping a bigger model behind a
clean interface — not a rewrite.

> Full design + engineering-review record: **[DESIGN.md](DESIGN.md)**.

## Architecture at a glance

```
  AudioTransport → WakeWord → Transcriber → LLM → Synthesizer → AudioTransport
   (local mic)     (openWW)   (whisper CPU) (Ollama) (Piper CPU)   (headset out)
                         │                    │
                    Orchestrator (asyncio) ── MemoryStore (SQLite + sqlite-vec)
                    preemption state machine   async idle-gap fact extraction
                    Scheduler (SQLite-persisted timers & reminders)
```

**Current hardware:** reasoning runs on CPU because GPU discovery fails on this
GTX 1050. Transcription, synthesis and embeddings also run on CPU.

**Stack (all free / local):** Ollama + llama3.2 · faster-whisper · Piper ·
openWakeWord + energy VAD · SQLite + sqlite-vec · custom asyncio orchestrator.

## Status

Working end to end, on-device:

- **Voice loop** — wake word -> speech-to-text -> local LLM -> speech, on a custom
  asyncio preemption state machine (idle / listening / thinking / speaking) with
  stop-word barge-in and half-duplex mic gating.
- **Memory** — SQLite + `sqlite-vec` semantic recall with CPU embeddings. Facts are
  extracted in the idle gap after a reply and survive restarts; an interrupted
  extraction is retried on next start rather than lost.
- **Personality** — a warm, opinionated persona (not an assistant voice), plus seeded
  identity facts that conversational extraction cannot overwrite.
- **Self-awareness** — he can describe his current build, and his mood tracks
  whether real progress was made since the last version.
- **Timers and reminders** — "set a timer for 10 minutes", "remind me to stretch at
  5pm". Parsed deterministically (no extra model round-trip), stored with an absolute
  wall-clock time, and reconciled on wake — so a reminder survives the laptop sleeping
  or the app restarting, and admits it if it ends up late.

- **Doing things on the computer** — "open Spotify", "next track", "find my tax notes".
  Also deterministic, against a registry of things he is allowed to open (edit
  `CONFIG.actions.apps` to teach his a new one). He reports what actually happened,
  so a failed open is admitted rather than confirmed. Deleting, moving and running
  arbitrary scripts are deliberately excluded.

- **Things he has read** — `python -m jesse learn guitar ./notes/guitar.md`. Documents
  are chunked, embedded, and retrieved when he *names the subject* (in that sentence or
  a recent turn), so a corpus never barges into small talk. If his words merely brush a
  document's own vocabulary, he asks — "Are you asking about your guitar?" — and a yes
  gets the answer. He answers from the passages or says they don't cover it — right
  about five times in six, which is the honest number, not a solved problem.

- **Reading his own sources** (enabled in current config) — `python -m jesse digest --fetch`, or
  on a 6-hourly schedule once `CONFIG.digest.enabled` is on. RSS/Atom feeds only, no
  web scraping. He never brings it up unprompted; ask "anything new?" and he tells
  you what actually came in, or says nothing has. This is the only part of Jesse that
  uses outbound requests to configured public feeds.

- **Reaching him from away** — `python -m jesse run --remote` serves a page your phone
  opens over [Tailscale](https://tailscale.com). It listens continuously, and the audio
  goes into the same pipeline as the desk microphone: same memory, same personality,
  same ability to open things on your computer. Two locks — only devices on your
  tailnet can reach it, and every request carries a token. Side-effect actions ask for
  confirmation when you're away.

Full-vision completion is tracked in **[RELEASE.md](RELEASE.md)**. Custom `yo Jesse`
training and cloning Mithilesh's own voice are active milestones. Local OmniVoice
auditions and an optional GPU voice backend exist; likeness and live headset
acceptance are pending. The custom wake model and phone-number integration remain
unfinished. The Tailscale path is built but paused.

On Windows, **double-click `Start-Jesse.cmd`** after setup, or run it from a terminal.
It selects this project's Python, the real Ollama model, and the local text UI.
It accepts the same extra options as `run`; startup still calibrates the mic unless
you supply a measured gain/threshold pair or `--no-calibrate`.

See **[training/README.md](training/README.md)** for local own-voice recording,
training/export instructions, custom-model selection, and wake-word evaluation.
The approved cleaned voice is selected on this PC by the private
`data/voice-clone/default.json` profile. **Start-Jesse.cmd** and `jesse run` use it
automatically. Its quality setting is slow on the GTX 1050: measured short replies
took 24–27 seconds to generate. For the faster Piper voice, launch
`Start-Jesse.cmd --voice en_US-ryan-high`. Installations without the local default
profile use Piper. See **[training/OMNIVOICE.md](training/OMNIVOICE.md)** for setup.

## Setup

Python **3.11-3.13** (3.13.2 is what this is developed on; every wheel resolves).
You also need [Ollama](https://ollama.com) installed and running.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
# His voice (~60MB, offline after this):
python -m piper.download_voices en_US-ryan-high --download-dir models
# His ears — the wake-word models (~10MB, offline after this):
python -c "import openwakeword.utils as u; u.download_models()"
# Download/cache transcription and embedding models before going offline:
python -c "from jesse.stt.whisper import WhisperTranscriber; from jesse.memory.embedder import FastEmbedEmbedder; WhisperTranscriber()._ensure(); FastEmbedEmbedder().embed(['setup'])"
```

After these model downloads, every
part of him that thinks, listens, speaks or remembers works with the ethernet cable
pulled out. The only things that want a connection are the optional extras — reading
your RSS feeds, and reaching his remotely — and losing it costs you those, not his.

> **Note:** the wake-word models install *inside* the virtualenv, so if you ever
> delete and rebuild `.venv` you need to re-run that last command. `python spike.py`
> checks for them and tells you if they're missing.

Then talk to him:

```bash
python -m jesse run --ollama          # add --device N to pick a specific mic
```

To reuse a measured calibration, supply both numbers (these are example values):

```bash
python -m jesse run --ollama --device 1 --gain 14.8 --threshold 1004
```

Either `--gain` or `--threshold` selects manual audio settings and skips startup
calibration. A value you omit keeps its configured default, so use the measured
pair together. Both must be finite numbers greater than zero. These overrides
last for that run; they do not rewrite configuration.

Useful extras: `python -m jesse memory` (inspect what he remembers),
`python -m jesse devices` (list mics), `python -m jesse say "text"` (test his voice),
`python -m jesse digest` (what he has read from your sources).

To reach him from your phone:

```bash
python -m jesse run --ollama --remote
```

That prints a link and a token. The browser will only give a page the microphone over
https, so serve it with a real certificate:

```bash
tailscale serve https / http://127.0.0.1:8766
```

Then open the MagicDNS name on your phone, paste the token once, and press *start
listening*. Headphones are worth it — on speakerphone the page has to mute your mic
while Jesse talks, so you can't interrupt him.

## Verify your setup (the spike)

Proves the hardware round-trip and clears the Windows install landmines. It runs
even before you've installed everything — it reports what's missing.

```bash
python spike.py                    # probe everything
python spike.py path\to\clip.wav   # also time STT on a real 16kHz mono wav
```

Green = go. A red probe (e.g. sqlite-vec won't load) blocks app code.

To check speaker echo, close the running Jesse, keep the room quiet, and run:

```bash
python diagnose.py echo --device 1 --output-device 4 --gain 1 --threshold 150
```

Pick device indices from `python -m jesse devices` and use the gain and threshold
printed by your calibration. Three short phrases play through the real transport;
the report measures ambient sound, playback, and each three-second tail. Confirm
the phrases are audible. `RISK` means tail audio could form a false user turn,
including speech still waiting for enough silence to finish. No recorded audio is
saved or sent to memory, and the check does not change calibration or system volume.
A quiet result applies only to that setup and those samples; it is not a phone echo
cancellation test.

## Run the tests

Two layers, deliberately:

```bash
.venv\Scripts\python.exe -m pytest        # unit tests; current count in HANDOFF.md
.venv\Scripts\python.exe -m jesse smoke    # real-stack scenarios, roughly 4-5 minutes
```

(If you have run `.venv\Scriptsctivate`, plain `pytest` and `python -m jesse smoke`
work too. Running them with the *system* Python fails on a missing dependency —
`jesse` will tell you so and point at the venv rather than dumping a traceback.)

The unit tests drive the orchestrator with fakes — a stateless wake detector, an
instant LLM — so they pin logic fast. But a fake can only fail in ways you thought
to model, and the serious bugs here were all outside that: a brain failure swallowed
inside a worker thread, a real wake detector going deaf after a long reply because it
needs continuous audio, and the wake word bleeding into the transcript and breaking
fact extraction. None were reachable with fakes.

`jesse smoke` runs the real Ollama, Piper, faster-whisper and SQLite end to end, using
Piper as a mouth feeding the pipeline's ears — so it needs no microphone, no speakers
and no human. It covers a conversation turn, memory stored and recalled across a new
connection, a spoken timer firing, barge-in, the wake word still working after a long
reply, an app he does not have being admitted rather than agreed to, a document being
ingested and answered from, a feed being read and reported honestly, and a phone
joining over HTTP and getting his voice back. Each scenario uses a temporary database;
your real memory is untouched.

Run the unit tests constantly; run the smoke test after anything that touches audio,
threading, or the model boundary.


## Privacy

**The line that does not move:** his brain, his memory and all of his processing
are local, always. The model, the transcription, the voice, the embeddings and the
database all live here. Runtime data (`data/`, `*.db`, `*.wav`, `models/`) is
gitignored, and no conversation, memory or audio is ever sent anywhere.

**What does use the network**, as a connection rather than a home:

- **Reading your sources** (`CONFIG.digest.enabled`, on) — plain GETs for the public
  RSS feeds you list. Outbound only, and they carry no conversation, no memory and no
  identifier beyond a user agent.
- **Remote access** (`--remote`) — reaching *your* Jesse, running on *your* computer,
  from your own devices over Tailscale. The audio travels inside a WireGuard tunnel
  between two machines you own; he does not travel at all.

Neither is a cloud service, and neither moves any part of him off this machine. If
that ever changes, this section is the thing to correct first.

## License note (TTS is GPL-3.0)

Jesse's own code is MIT (see `pyproject.toml`). The TTS engine, **piper-tts**
(OHF-Voice/piper1-gpl), is **GPL-3.0** — the old MIT `rhasspy/piper` is archived.

What that means in practice for this repo:

- We depend on piper-tts via `requirements.txt` and call its Python API; we do
  **not** copy or redistribute its source or the voice model (`models/` is
  gitignored). A source-only project that lists a GPL package as a dependency and
  lets users `pip install` it themselves does not, in common practice, force the
  rest of the repo to become GPL.
- The GPL obligations (offer source, license the combined work under GPL) bite if
  you **distribute a bundled/combined work** — e.g. ship a single installer or
  frozen `.exe` that includes piper-tts. If you go that route, plan to comply or
  swap the TTS backend.
- Because TTS sits behind the `Synthesizer` interface, swapping to a
  permissively-licensed engine later is a one-file change — you're not locked in.

Not legal advice; just the honest lay of the land for a portfolio repo.
