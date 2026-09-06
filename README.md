# Jesse

A **fully-local voice AI partner**. His brain, his memory and every piece of
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
   (WASAPI mic)    (openWW)   (whisper CPU) (llama3  (Piper CPU)    (headset out)
                                            GPU)
                         │                    │
                    Orchestrator (asyncio) ── MemoryStore (SQLite + sqlite-vec)
                    preemption state machine   async idle-gap fact extraction
                    Scheduler (SQLite-persisted timers & reminders)
```

**Compute split:** GPU does reasoning only (the model stays resident). CPU hears (faster-whisper
int8), speaks (Piper), and embeds. Nothing contends for the 4GB.

**Stack (all free / local):** Ollama + llama3.2 · faster-whisper · Piper ·
openWakeWord + Silero VAD · SQLite + sqlite-vec · custom asyncio orchestrator.

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

- **Reading his own sources** (off by default) — `python -m jesse digest --fetch`, or
  on a 6-hourly schedule once `CONFIG.digest.enabled` is on. RSS/Atom feeds only, no
  web scraping. He never brings it up unprompted; ask "anything new?" and he tells
  you what actually came in, or says nothing has. This is the only part of Jesse that
  touches the network, which is why it ships switched off.

- **Reaching him from away** — `python -m jesse run --remote` serves a page your phone
  opens over [Tailscale](https://tailscale.com). It listens continuously, and the audio
  goes into the same pipeline as the desk microphone: same memory, same personality,
  same ability to open things on your computer. Two locks — only devices on your
  tailnet can reach it, and every request carries a token. Side-effect actions ask for
  confirmation when you're away.

Deferred: GPU acceleration (Ollama's Vulkan discovery times out on this GTX 1050, so
the LLM runs ~12 tok/s on CPU), a custom wake word, and voice cloning.

## Setup

Python **3.11-3.13** (3.13.2 is what this is developed on; every wheel resolves).
You also need [Ollama](https://ollama.com) installed and running.

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
ollama pull llama3.2
# His voice (~60MB, offline after this):
python -m piper.download_voices en_US-amy-medium --download-dir models
# His ears — the wake-word models (~10MB, offline after this):
python -c "import openwakeword.utils as u; u.download_models()"
```

Those three downloads are all Jesse needs to *become* himself. After them, every
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

## Run the tests

Two layers, deliberately:

```bash
.venv\Scripts\python.exe -m pytest        # 399 unit tests with fakes, ~2 seconds
.venv\Scripts\python.exe -m jesse smoke    # 9 scenarios on the REAL stack, ~4 minutes
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
