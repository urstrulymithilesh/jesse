# Jesse — Handoff

> **This file is the cold-start prompt for ANY coding agent** — Claude Code, Codex,
> Cursor, whatever is in front of you, on any machine. It assumes nothing about which
> tool you are. Point an agent at this file and it has the full picture: what Jesse is,
> what exists, what was rejected and why, and the failure patterns that cost real time.
> Switch tools freely when you hit usage limits; continuity lives here, not in a
> session.

`c19af33` · 412 tests · 9/9 smoke · Python 3.13 · `D:\New folder\jesse` ·
`github.com/urstrulymithilesh/jesse`

---

## 0. Standing workflow — do this without being asked

**Skills, always on:** `caveman` (terse), `ponytail` (laziest thing that works, stdlib
first), `gstack`, `karpathy-guidelines` (state assumptions, simplest solution, surgical
diffs, verifiable goals).

Every change:

1. Make it.
2. `.venv\Scripts\python.exe -m pytest -q`
3. `.venv\Scripts\python.exe -m jesse smoke` if the real stack is touched.
4. Append one `ProgressEntry` to `jesse/memory/progress.py` — Jesse's own account, his
   voice, plain words. `significant=True` only for a real capability change.
5. Commit explaining **why**, with measurements.
6. **Push. Every commit. No batching.**
7. **Update this file** if a fresh agent would need it.

**Never delete silently.** Stale files, dead code, old artifacts — flag and ask.

**Output style:** no narration of routine steps, no restating what he knows. Never
trade honesty or a real bug for brevity.

---

## 1. Who Jesse is

A **local AI friend** — mate, hype man, the one who pushes him. Lives on one machine,
belongs to one person, sends nothing anywhere.

- **NOT a partner, NOT romantic.** Close friend, hype man, blunt, loyal underneath.
  Banned words in his persona: **companion**, **partner**, **assistant**.
- **He is male, he/him.** Mithilesh is also "he" — when writing docs, name whoever you
  mean; a bare pronoun does not distinguish them.
- **He has no prior identity and no history of being anything else.** Nothing in this
  repo says otherwise; keep it that way.
- **Brain, memory, processing: 100% local, always.** The network is a *connection*, not
  a home — outbound RSS, and remote access to his own machine. Routing a conversation
  through a third party is refused (this is why Twilio was rejected twice).
- **Always listening** once woken, until told to stand down.
- **Voice and text feed one mind** — same memory, same persona, same pipeline.
- **He remembers** facts *and* conversations.
- **HE NEVER MAKES THINGS UP.** The rule that outranks the rest, with its own block
  in the persona. He knows exactly three things: what Mithilesh tells him now, what he
  has actually been given, and the injected clock. No shared history, no tastes of his
  own, no world-state. The other half matters equally: **what he HAS been given, he
  knows — answer from it.** Refusing when the answer is in front of him is a different
  kind of wrong, and the smoke harness caught exactly that regression.

---

## 2. What exists

| Area | State |
|---|---|
| **Voice loop** | wake → STT → LLM → TTS on an asyncio preemption state machine. Streaming TTS: 11.0s → 4.1s to first word. Barge-in ends *listening*, not idle. Half-duplex mic gating. |
| **Continuous conversation** | wake once, talk freely until "go to sleep". Timeouts 8s post-wake / 45s engaged — deliberately not infinite. |
| **Text UI** | `--ui` on `127.0.0.1:8765`. Chat bubbles, charcoal, one amber accent. Same turn pipeline as speech. |
| **Memory** | SQLite + sqlite-vec. Facts = slots (deduped 0.88 on subject). Episodes = events (append-only, never deduped). Temporal queries deterministic. Seeded identity facts protected by `origin`, reseeded on content hash. |
| **Timers/reminders** | create, reschedule, cancel, query. Wall-clock, survives sleep, admits lateness. Refuses to guess when ambiguous. |
| **Actions** | open (23-target registry), media keys, file search. Deterministic parse. Every outcome logged DONE / NOT RUN / REFUSED / FAILED. Delete/move/run-script excluded. |
| **Knowledge (RAG)** | `jesse learn <name> <path>`. Trigger = naming the subject, or a keyword-ask. Not a distance gate — that inverted on the second corpus. |
| **Sources (RSS)** | ON. 6h interval, silent fetch, reactive "what's new". Instruction-shaped items dropped at ingest. |
| **Remote** | **PAUSED — see §4.** Built and left in place. |
| **Smoke harness** | 9 scenarios, real stack, headless, ~4min. Piper is the mouth feeding the pipeline's ears. |

**Honesty guards:** real clock injected every turn; hard rule that he knows only what he is
told/given/timestamped; anchoring blocks for broad and temporal questions; recall-mode
drops few-shot examples for memory questions.

---

## 3. Defaults

| | |
|---|---|
| Model | `llama3.2` via Ollama · ctx 4096 · temp 0.6 · question-keep 0.15 · keep_alive -1 · 90s timeout |
| Voice | Piper **`en_US-ryan-high`** (22050 Hz) — only voice on disk |
| STT | faster-whisper `base.en`, int8, CPU |
| Wake / stop | `hey_jarvis` (placeholder; real word will be "yo Jesse") |
| Mic | `--device 1` in practice · gain 1.0 · auto-calibrate on |
| VAD | threshold 150 · silence 950ms · min-speech 300ms · pre-roll 500ms |
| Timeouts | listen 8s · continuous 45s · continuous **on** |
| Memory | top-3 recall · 12 turns · 2400 chars · min-conf 0.6 · dedupe ≥0.88 |
| Embeddings | `BAAI/bge-small-en-v1.5` (fastembed, CPU) |
| Actions | 23 targets · search depth 4 · top-5 |
| Knowledge | name + keyword trigger · top-2 · 800-char chunks · gate 0.46 |
| Sources | ON · RSS/Atom · 6h · 5/source · 3 told at once |
| Progress log | 28 entries |

Everything sits behind `jesse/core/interfaces.py`. Swapping a model or engine is a
config change.

---

## 4. Parked — do not rebuild or re-litigate

- **Remote access (Tailscale).** Built, tested, works. **Paused 2026-09-03: too much
  friction for the value** — a session went on DNS, certificates and tokens before
  carrying a word of speech. Left in place; `--remote` still runs. **The eventual goal
  is a real phone number (VoIP)**, once everything else is solid. Do not sink more time
  into the Tailscale path.
- **Twilio / VoIP, as of today.** Priced and designed twice. US number $1.15/mo +
  ~$0.013/min. Indian numbers effectively unavailable (registered-address rules).
  Rejected *not* on cost but because Twilio terminates the leg and holds call audio in
  the clear. Landmines for whoever revives it: needs a public `wss` on 443 (a *bigger*
  surface than Tailscale), `audioop` was removed in Python 3.13, and it would be the
  first networking dependency. On a call, skip the wake word — the call *is* the wake.
- **GPU.** Ollama sees the GTX 1050 via Vulkan, its discovery watchdog times out, falls
  back to CPU at ~12 tok/s. Externally blocked. Timebox any attempt.
- **qwen2.5:7b.** Grounds better than 3b, ~15s/reply on CPU. Revisit only with a GPU.
- **nemotron-mini (4B).** Evaluated 2026-08-28, rejected. TTFT identical warm (2.4s),
  but writes 110 words where llama3.2 writes 30, and slips into assistant register
  2/12 vs 0/12. Better on grounding (15/15 vs 14/15) and never false-fires a tool
  (0/7 vs 7/7). Not enough.
- **LLM tool-calling.** Measured, temp 0: llama3.2 6/7 correct but **7/7 false calls on
  plain conversation**; nemotron 0/7 false but 3/7 correct and malformed args. Neither
  is usable. The deterministic registry stands.
- **Custom wake word + voice cloning.** One future session; shares a training pipeline.
  Target word: "yo Jesse". Training is cloud/Colab; integration is ~5 lines.
- **Voice authentication.** Pairs with the voice phase.
- **Multi-turn slot-filling.** "Change the timer" → "to what?" is not supported by
  design. He asks for the missing piece; the request must come in one utterance.
- **Guidance mode** ("walk me through it"). Needs multi-turn position state — the
  dialogue-manager rabbit hole, already declined.
- **Routine/pattern learning.** Needs many episodes over real time.

---

## 5. Roadmap

| # | Step | Status |
|---|---|---|
| 1 | Reliability / smoke harness | **done** |
| 2 | Continuous conversation | **done** |
| 3 | Custom wake word + voice cloning | future ("yo Jesse") |
| 4 | Voice authentication | skipped by choice |
| 5 | Episodic + temporal memory | **done** |
| 6 | GPU | parked, externally blocked |
| 7 | Agentic computer use | **done** |
| 8 | Skill mastery (RAG) | **done** — guidance mode not built |
| 9 | Proactive daily learning | **done** — reactive by default |
| 10 | Remote access | built, **PAUSED**; real goal is VoIP |

Every roadmap item is built, skipped, or parked with a reason. What is left is depth:
the custom wake word, voice auth, GPU if hardware ever cooperates, and the rough edges
in §8.

---

## 6. Hard-won lessons

These cost real debugging time. They are patterns, not trivia.

**A spoken slip is cheap; a stored one is permanent.** Asked whether he likes
pineapple on pizza he answers with a taste he does not have, 8 times in 9, and no
prompt wording fixed it — a one-word answer that happens to assert a preference. The
fix went where it holds: `"Jesse does not like pineapple on pizza"` is third person,
well formed, and the extraction filter *would have stored it forever*. Anything naming
Jesse, and anything under four words, is now refused at the door. **Put the guard where
the damage persists, not where the mistake is made.**

**An honesty rule over-applies in both directions.** Made prominent, it produced "No
idea, man." to *"I burnt the rice"* — refusing ordinary conversation. Scoped to claims,
it then produced "Nah." instead of reading out a headline he had been handed. Both
halves have to be stated: **answer from what you have, admit what you don't.**

**Fakes cannot catch stateless-vs-stateful bugs.** The real wake detector is a
*streaming* model needing ~1s of continuous audio; fed only in its own state it went
deaf after every long reply. A stateless fake can never show this. Verify a regression
test by breaking the fix.

**A prompt rule the model ignores is not a rule.** The extraction prompt said "third
person, not your own replies"; 3 of 9 stored facts were Jesse's own lines. Fixed in
code, not by asking louder. **Ask for the shape, then check the shape.**

**Never quote the bad output in the rule that forbids it.** Happened twice: an anti-tic
rule quoting the tic (0/5 → 2/5), and a rule against empty reactions quoting one — the
model said that exact phrase back to an unrelated turn. `test_no_rule_quotes_the_bad_
output_it_forbids` now enforces this.

**Persona details leak as false claims.** Five times. A taste for rain came back as
"grey and pouring" when asked the actual weather. **No persona detail may name a
perceivable world-state.** Reactive examples are safe; assertions are not.

**Retrieval hands him a topic and he finishes the sentence.** Asked about string gauges
with a document covering only tuning, he invented numbers 3/3 and once attributed them
to the file by name. No threshold fixes this — the passage *is* about the subject. Fixed
with recall-mode + a "this is the complete extent" block. 0/3 → 5/6.

**An embedding threshold is not a trigger.** The knowledge gate measured beautifully on
one document (0.032 clean margin) and **inverted** on the second. Triggers must be his
words, which do not drift as the corpus grows.

**Anchor to real data, or admit not knowing.** Used five times, worked five times:
state the complete list and forbid inventing beyond it. 5/5 fabricated → 0/5.

**A negation is not delegable to a 3B.** "I can open Photoshop" about an app he cannot
open — the model dropped the "not" roughly 1 in 3. **A sentence claiming what he can't
do, or refusing to act, is structural — speak it deterministically.** But this was
tested: event reports ("it failed", "nothing pending") survive as prompt notes 5/5.
Probe before converting.

**One mangled word at the front disables every parser at once.** Whisper rendered the
wake word as "Heeshak,"; every deterministic parser anchors at position 0, so all of
them silently switched off and he *promised* an action that never ran. Anything that
can corrupt the first token is a single point of failure for every feature.

**Silence means two different things.** The remote page stops uploading while he speaks
— that *is* the half-duplex rule. The idle timeout counted it as hanging up. Before
treating absence as a signal, ask whether you caused it.

**An error message that cannot distinguish causes sends people the wrong way.** "Bad
token" for missing/wrong/locked-out sent him re-typing a token that was never wrong.
During lockout the *correct* token is also refused, so the advice pointed away from the
fix.

**My own checks are as likely to be wrong as the thing they check.** Four times: "no
sugar" flagged as invention when the fact says exactly that; "doesn't mention gauge
*recommend*ations" flagged because "recommend" was banned; both models passed 0/8 on
banned phrases while one talked like a helpdesk; a grep escaping bug declared a venv
clean when it wasn't. **A wrong check is worse than no check — it produces a confident
number. Read raw output before trusting any score, especially one that agrees with
you.**

**A green suite says nothing about a module no test imports.** `jesse run --remote`
shipped broken for one commit with an unterminated f-string while 401 tests passed.

**An interface can be right and still not be enough.** `AudioTransport` paid off ~70%:
the data contract needed no change, but the orchestrator binds one transport for the
process lifetime, so a joining/leaving session needed a switching layer. A
data-contract seam is not a lifecycle seam.

**Config that only applies on first run silently rots.** `seed_if_needed` gated on "any
core facts yet", so editing seed.py never reached a live db — he called himself a
banned word for three commits. Gate on a content hash.

**Given material he cannot use, he invents.** Handed an unusable feed item he made up
articles rather than saying nothing. Filter at the door.

**Deterministic parsing for anything structural or destructive.** Every LLM round-trip
is 3–7s here; small models are unreliable at structure; and a wrong guess on a
destructive action fails *silently*. Where genuinely ambiguous, ask.

**Verify live, not just green. Measure with real quotes.** Every serious bug in this
project was found live and reproduced afterwards.

**Dry-run before destroying memory.** `--dedupe` previews by default. That requirement
immediately caught a merge that would have deleted his own name.

**Honest ceilings, stated plainly.** The persona was rated 6.5/10, not "fixed".
Overall progress was called 35/100 when it was. An inflated status report costs more
than a disappointing one.

---

## 7. Hardware, stack, commands

**Machine:** i5-8300H · 32GB · GTX 1050 4GB (unusable, see §4) · Windows 11 · CPU
~12 tok/s.

| Layer | Choice |
|---|---|
| Language | Python 3.13, venv at `.venv` |
| Reasoning | Ollama · `llama3.2` |
| Voice | Piper `en_US-ryan-high` (**GPL-3.0** — matters only if bundling a binary) |
| Hearing | faster-whisper `base.en` int8 CPU |
| Wake | openWakeWord `hey_jarvis` (ONNX) |
| Memory | SQLite + sqlite-vec — one file, `data/jesse.db` |
| Embeddings | fastembed `bge-small-en-v1.5`, CPU |
| Orchestration | custom asyncio loop (deliberately not LangChain) |
| UI | stdlib `http.server` + polling |
| QR | `segno` (pure Python; the only convenience dependency) |

```
jesse/
  orchestrator.py    event loop + preemption state machine
  persona.py         SYSTEM_PROMPT + recall_prompt()   <- tune freely, no logic
  context.py         build_messages + the anchoring blocks
  config.py          every default in one place
  factory.py         wires everything                  <- the swap point
  core/  audio/  stt/  tts/  llm/  memory/  schedule/  actions/  digest/  remote/  ui/
  smoke.py           the live harness
tests/               410 tests
```

```
.venv\Scripts\python.exe -m jesse run --device 1 --ollama --ui
.venv\Scripts\python.exe -m jesse smoke            # 9 scenarios, real stack, ~4min
.venv\Scripts\python.exe -m pytest -q              # 410 tests, ~3s
.venv\Scripts\python.exe -m jesse memory [--forget "..."] [--dedupe [--apply]]
.venv\Scripts\python.exe -m jesse seed             # re-apply protected identity facts
.venv\Scripts\python.exe -m jesse learn <name> <path> | --list | --ask "..."
.venv\Scripts\python.exe -m jesse digest [--fetch]
.venv\Scripts\python.exe -m jesse pair [--ip]      # phone link + QR (remote paused)
.venv\Scripts\python.exe -m jesse say "text" | devices | calibrate
```

> Use the venv Python explicitly. Bare `python` is system Python and lacks the deps;
> `jesse` detects this and says so.

**Landmine:** openWakeWord models are a **runtime download**, not a pip dependency.
Rebuild the venv and they vanish — every smoke scenario fails with "could not load wake
model". Fix: `python -c "import openwakeword.utils as u; u.download_models()"`.

---

## 8. Rough edges

- **Persona: register clean, substance thin.** Measured 0/10 helpdesk, 0/10
  therapist, 0/10 British, 0/10 name tic, 1/10 questions; fabrication 11/12 admit not
  knowing. But reading the replies, "Nah." to burnt rice is not a reaction and "Nah" is
  filler in 4 of 10. **Every strengthening of the honesty rule has cost conversational
  substance** — on a 3B that tension looks structural, not one prompt pass away. Tune
  against real transcripts rather than another prompt rewrite.
- **Only one Jesse at a time.** Two copies fight over the microphone and the second
  hears almost nothing; it presents as calibration failing for no reason.
  `data/jesse.pid` names the clash at startup.
- **Remote never driven from a real handset.** Covered by smoke over real HTTP; iOS
  Safari may reject the self-signed certificate until installed as a profile. Paused
  anyway.
- **Remote barge-in impossible on speakerphone** — the page mutes the mic while he
  talks and cannot tell whether headphones are plugged in.
- **Whisper mangles the wake word unpredictably** ("Heeshak", "Stay Jarvis"). One-word
  commands ("skip", "resume") are the weak spot; two-word forms are reliable.

---

## 9. Working style

- **Reasoning before big decisions.** Explain the trade-off, recommend, don't silently
  pick. Design questions get answered *before* code.
- **He tests live and comes back with real transcripts.** Take those seriously — every
  major bug arrived that way. Reproduce before theorising, and say when a reported
  problem turns out to be a test artefact.
- **Limited time.** Prefer high-impact, low-effort. Flag expensive work as expensive.
- **Own mistakes plainly and move on.** Many bugs here were self-inflicted. Each was
  stated, fixed, and pinned with a test.

---

## 10. Where things stand

Against the full vision: **~60/100**. Voice, personality, memory, episodic memory,
timers, always-listening, text UI, agentic execution, document RAG and proactive
sources all work. Missing: the custom wake word, voice auth, GPU, and phone access.

Against "something worth using every day": **70–80%**. He wakes, listens, remembers,
keeps time, acts on the computer, reads what he is given, admits what he does not know,
and can be talked to or typed at.

**Next:** nothing is blocking. Depth over breadth — tune the persona against real
transcripts, widen the action registry as phrasings miss, and train "yo Jesse" when
there is an appetite for a Colab session.
