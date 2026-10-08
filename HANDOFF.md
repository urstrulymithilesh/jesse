# Jesse — Handoff

> **This file is the cold-start prompt for ANY coding agent** — Claude Code, Codex,
> Cursor, whatever is in front of you, on any machine. It assumes nothing about which
> tool you are. Point an agent at this file and it has the full picture: what Jesse is,
> what exists, what was rejected and why, and the failure patterns that cost real time.
> Switch tools freely when you hit usage limits; continuity lives here, not in a
> session.

2026-10-08 · 630 tests · 14/14 smoke (last runtime change) · Python 3.13 · `D:\New folder\jesse` ·
`github.com/urstrulymithilesh/jesse`

---

## 0. Standing workflow — do this without being asked

**Branch policy: `main` only.** Mithilesh asked to remove all other branches.

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
| **Smoke harness** | 14 scenarios, real stack, headless, ~4min. Piper is the mouth feeding the pipeline's ears. |

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
| Progress log | 46 entries |

Everything sits behind `jesse/core/interfaces.py`. Swapping a model or engine is a
config change.

---

## 4. Parked — do not rebuild or re-litigate

- **Remote access (Tailscale).** Built, tested, works. **Paused 2026-09-03: too much
  friction for the value** — a session went on DNS, certificates and tokens before
  carrying a word of speech. Left in place; `--remote` still runs. **The eventual goal
  is a real phone number (VoIP)**, once everything else is solid. Do not sink more time
  into the Tailscale path.
- **Twilio / VoIP — UNPARKED 2026-09-20 by Mithilesh, now §11.** Rejected twice
  before, *not* on cost but because Twilio terminates the leg and holds call audio in
  the clear. He has since decided a phone number is worth that trade; the "100% local"
  claim in §1 no longer covers a call, and saying so is part of the job. Prices: US
  number $1.15/mo + ~$0.013/min; Indian numbers effectively unavailable
  (registered-address rules), so the number is US and dialling it from India is an
  international call.
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
- **Custom wake word + voice cloning: ACTIVE as of 2026-10-05.** Mithilesh explicitly
  chose the full vision. Target phrase is "yo Jesse"; clone his own voice, American,
  about age 21, charismatic. See `RELEASE.md`, `training/README.md` and §23. Neither
  trained asset exists yet; do not describe this as finished or keep postponing it.
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
| 3 | Custom wake word + voice cloning | **active** — collection, integration and evaluation tools ready; training pending |
| 4 | Voice authentication | skipped by choice |
| 5 | Episodic + temporal memory | **done** |
| 6 | GPU | parked, externally blocked |
| 7 | Agentic computer use | **done** |
| 8 | Skill mastery (RAG) | **done** — guidance mode not built |
| 9 | Proactive daily learning | **done** — reactive by default |
| 10 | Remote access | built, **PAUSED**; real goal is VoIP |
| 11 | Phone number (Twilio) | **in progress** — see §11 |

The completion gates and execution order now live in `RELEASE.md`. Prioritize
trained custom assets and phone access over an unbounded stream of unrelated fixes.

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
tests/               585 tests
```

```
.venv\Scripts\python.exe -m jesse run --device 1 --ollama --ui
.venv\Scripts\python.exe -m jesse smoke            # 14 scenarios, real stack, ~4min
.venv\Scripts\python.exe -m pytest -q              # 585 tests
.venv\Scripts\python.exe diagnose.py echo --device 1 --output-device 4 --gain 1 --threshold 150
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

Mithilesh chose **the full vision** on 2026-10-05, including a custom wake word,
his own voice clone and phone calls. `RELEASE.md` replaces the old subjective
percentage estimates with explicit acceptance gates. Core features work, but
trained custom assets, live local acceptance and phone integration are unfinished.

**Next:** collect the reviewed voice pilot with `jesse voice-record` and provision
the separate GPU training environment for `yo Jesse` and Piper. The user supplied
the voice brief (§23) but has not recorded samples here. Then evaluate/audition the
exports locally, validate normal conversation, and complete phone echo/transport
integration (§11). Voice authentication remains skipped by choice; GPU runtime
acceleration remains optional. Do not quietly reduce the goal to a local-only release.

**Historical verification, 2026-10-01 (superseded by later checks):** 427 unit tests passed in 10.09s; the real-stack smoke
run passed 7/9 scenarios (647s plus model warmup). Conversation, timers, barge-in,
wake-after-reply, actions, knowledge and remote passed. Memory extraction returned
no facts on both attempts despite correctly transcribing the turquoise preference;
conversation turns were persisted. Sources failed because Whisper transcribed the
first "anything new" as "Anything mail?", so the feed parser was never triggered;
the second utterance was transcribed correctly and the headline was read. These
failures led to the later extraction/recall fixes; one-word transcription remains
imperfect. Use the latest header/entry for current verification.

---

## 11. Phone number (Twilio) — in progress

Unparked 2026-09-20. Goal: call a real number and talk to Jesse through the existing
STT -> LLM -> TTS pipeline.

**Done.** `jesse/remote/g711.py` — mu-law <-> PCM and 8k/16k either side of it, which
is the `audioop` landmine cleared (removed in 3.13; written out rather than pulling in
`audioop-lts`). Real speech survives the leg intact for Whisper, SNR 20.3 dB, most of
that the 4 kHz band limit every phone call has. 9 tests. Nothing calls it yet.

**The seam it plugs into already exists.** `SwitchingTransport` + `RemoteSource`
(`jesse/remote/transport.py`) were built for the phone-joins-mid-session case, take
16 kHz frames pushed in from a non-asyncio thread, and need no orchestrator changes.
A Media Streams handler feeds them.

**Not done, and the order matters.**

1. **The self-echo loop (§8) must be fixed FIRST.** Measured 2026-09-20: his own voice
   at raw RMS 13 — inaudible — amplified by a bad calibration's x30 gain, transcribed
   back *accurately* as a user turn. A phone line is full-duplex and carries his voice
   straight back with no `mute_input()` between him and the caller. Shipping the phone
   over this bug means fabricated conversations on a metered line.
   **2026-10-01:** fixed one confirmed path: the orchestrator used to collect
   pre-roll during SPEAKING, then prepend those reply frames to the next user turn
   even after the transport queue was flushed. THINKING/SPEAKING now clear stale
   pre-roll; playback frames are excluded until a stop-word interruption, whose
   subsequent words are preserved. Six regression cases failed before the change
   and pass after it (streamed replies, alerts, playback failure, interruption,
   and a thinking turn without speech). This is not acoustic echo cancellation:
   delayed room echo, over-gain calibration and a full-duplex phone still need live
   validation before this blocker can be closed.
   **2026-10-03:** callbacks captured before unmute but delivered after its queue
   flush are now discarded; five transport regressions cover this second path (§14).
   A live calibrated local test later passed three reply tails and one spoken
   follow-up at gain 14.8 / threshold 1004 (§16). This is half-duplex local evidence,
   not evidence for a full-duplex phone line.
2. Media Streams websocket endpoint + TwiML. **First networking dependency** (stdlib
   has no websocket server), and a public `wss` on 443 — a *bigger* surface than the
   Tailscale path that was paused for being too much friction.
3. Account, number, credentials, and the test call: **Mithilesh's, not an agent's.**
   An agent cannot open the account, buy the number, hold the auth token, or dial.

On a call, skip the wake word — the call *is* the wake.

---

## 12. Memory extraction follow-up — 2026-10-01

The earlier memory smoke failure reproduced with the real model: a user stated the
turquoise preference, Jesse's reply denied remembering it, and extraction of the
combined exchange returned `[]`. Passing just the labelled user statement recovered
the fact in three consecutive probes. A separate fact came back as malformed JSON;
a conditional move was also wrongly accepted as a durable memory.

Changes on `main` (included with the October 2 recall fix):

- Live and catch-up fact extraction now receive only the user's statement. Both
  conversation turns are still persisted, and episodic summaries still see both.
- Extraction uses the same local Ollama model with temperature zero and a fact-array
  JSON schema. Ordinary chat keeps its existing temperature and free-text output;
  the orchestrator's existing lock still serializes all model calls. See
  [Ollama's structured-output contract](https://docs.ollama.com/capabilities/structured-outputs).
- An explicit hypothetical opening (`if`, `what if`, `suppose`, `supposing`,
  `imagine`, `let's pretend`) skips fact extraction in code. The model ignored the
  prompt restriction even with the schema. This intentionally sacrifices real facts
  later in a mixed conditional turn; it does not cover every possible hypothetical.

Limits: JSON shape is not factual accuracy. A bare "yes" lacks enough context for a
new durable fact; the saved conversation remains available. Existing facts and
already-processed exchanges are not rewritten or re-extracted.

Validation: 444 unit tests passed in 3.68s (17 new cases), including source isolation
on both extraction paths, conditional openings, real-plan pass-through, constrained
request serialization and unchanged streaming chat. Local-model probes retained
the color, sister, swimming routine and actual moving plan; the greeting and
preference question returned no facts. The explicit hypothetical is rejected by
the code guard. The previously failing real-stack memory scenario passed on its
first attempt, storing and recalling the correct color through a fresh connection.

## 13. Birthday-plan recall and Ollama startup — 2026-10-02

**Reproduced exactly** from a read-only snapshot of the reported session: asking
"what is my birthday plan?" produced the current date and time. The birthday-plan
fact ranked first, a stale clock fact second, and the corrected activity ninth
(outside top-3). Fact injection omitted subject labels, and specific personal
questions did not enter recall mode. The recent 12-turn context also missed the
earlier explicit transcription correction.

The four fragmented subjects had pairwise cosine similarities 0.503–0.624; this
is **not a reason to lower the 0.88 dedupe threshold**. A plan, an activity and a
generic desire are not interchangeable slots. No facts were consolidated or edited
in the live database.

**Fix:** named personal questions prefer literal subject/text matches over unrelated
semantic neighbours. They retain subject labels, omit the clock and unrelated
assistant history, and use recall mode. A matching retained fact can bring in up to
six nearby user statements (1600 characters, five-minute window) from the latest
matching statement. An explicit "that's not X, it's Y" correction can replace X
in a temporary copy of a conversational fact; protected identity is unchanged.
Bare "I meant Y" is not assigned an antecedent by guesswork. These records are
read at query time, so existing memories work without re-extraction or reindexing.

Prompting alone still picked the old misheard word in two of three probes. Resolving
the explicit correction in code produced the corrected activity in all three final
local-model probes. This is a measured improvement, not a universal grounding claim.

**Forgetting boundary:** after a successful forget, earlier transcript material is
excluded from supplemental recall, even for surviving topics. This conservatively
avoids reconstituting a deleted detail through adjacent statements. Earlier forget
operations made before this boundary existed did not record a transcript cutoff.

**Startup:** `run --ollama` performs one `/api/tags` GET with a two-second timeout
before claiming audio or starting UI servers. An unreachable service prints "Start
Ollama first" and exits. It does not load a model or prove generation will succeed;
the existing honest in-turn error fallback remains necessary. Stub mode is unchanged.

**Regression coverage:** real SQLite restart, deliberately wrong vector ranking,
correction without record mutation, bounded evidence, forgetting, personal/clock
routing and preflight failures. Added `personal-recall` to the smoke suite, which now
has ten scenarios. All live-database investigation used read-only connections;
model replays used temporary copies or synthetic fixtures.

The first smoke run also exposed a harness bug: Whisper's "Photo Shop" correctly
produced a refusal for "photo shop", but the assertion required "photoshop" and its
failure path leaked the SQLite handle. The assertion now checks the exact refusal
for the parsed target, and the store closes in `finally`. Four tests cover both
spellings, a genuine refusal failure and cleanup. Named-document routing also has
a regression test so personal wording does not steal a knowledge question.

**Final verification:** 468 unit tests passed in 4.30s; all 10 real-stack smoke
scenarios passed in 221s plus 7s warmup. The new personal-recall scenario answered
"You want to be sky-diving next month." Duplicate-definition/undefined-name lint
and `git diff --check` also passed. The successful full rerun includes the corrected
action assertion and database cleanup.

## 14. Delayed microphone callbacks — 2026-10-03

Reproduced a second self-echo path with a deterministic mocked PortAudio callback:
audio captured while muted can be scheduled on the event loop but not yet in the
capture queue when `unmute_input()` flushes it. Blocking output shutdown can leave
such callbacks pending. The original transport delivered `echo` as the next user
frame in this test, instead of the freshly captured `user` frame.

Each callback now snapshots a capture generation before copying its audio.
Unmute increments the generation and flushes queued audio; delivery rejects an
older generation before gain processing or queue insertion. Muting still allows
current frames through for wake/stop detectors. No cooldown or microphone delay
was added, and the public transport contract is unchanged.

Five regression cases cover already-queued and pending callback audio, including
another reply starting before the old callback arrives, plus live muted capture
with gain and clipping. The full unit suite passed: 473 tests in 11.09s.
All ten real-stack smoke scenarios passed in 227s plus 18s warmup, including
barge-in, wake-after-reply and remote audio. The progress-entry tests, changed-file
duplicate-definition/undefined-name lint and `git diff --check` also passed.

This closes the reproduced callback scheduling gap only. Audio first captured
after reopening (room reverberation or device buffering), calibration gain and
full-duplex phone echo still require live hardware validation. The headless smoke
suite does not exercise actual PortAudio devices; phone integration remains blocked
on that validation.

## 15. Live echo measurement — 2026-10-03

`diagnose.py echo` now plays three fixed Piper phrases through LocalAudioTransport,
measures two seconds of ambient sound and three seconds after each reply, and runs
the real EnergyVad on each phase independently. It uses configured defaults unless
device, output-device, gain or threshold are supplied. Use the gain/threshold pair
printed by calibration when checking a daily setup. A 60-second asyncio timeout
bounds the capture/play exercise; model loading/synthesis happens beforehand and
native blocking audio calls still depend on the driver returning.

The command takes Jesse's existing instance claim, releases it on exit, closes
capture on failures/cancellation and unmutes after playback. It retains only level
and VAD statistics, never recorded audio, transcripts or memory. It changes neither
calibration nor Windows volume. Missing frames or all-zero capture are inconclusive.
Enough tail speech is flagged as a risk even without a completed endpoint: silence
arriving after the measurement window can finish that turn.

Measured on Realtek MME input 1 and speakers 4 at the existing system volume:

| Gain / threshold | Tail peak RMS, trials 1 / 2 / 3 | Tail speech frames | Completed endpoints |
|---|---|---|---|
| 30 / 150 (stress case) | 779.6 / 741.8 / 36.6 | 17 / 5 / 0 | 0 / 0 / 0 |
| 1 / 150 (configured defaults) | 2.3 / 0.7 / 2.0 | 0 / 0 / 0 | 0 / 0 / 0 |

Mithilesh confirmed the stress-test phrases were audible and the room otherwise
quiet, making speaker echo the likely source. Four 80ms speech frames meet the
configured 300ms speech minimum; two stress tails reached that minimum. The first
report checked only completed endpoints; this measurement exposed the omission,
which is now fixed and tested. Gain 30 with threshold 150 is deliberately sensitive;
auto-calibration normally derives both together. Do not label this as a failed
auto-calibration or silently change its gain from these samples.

The default setup had no observed false-turn risk across these three samples.
This does not establish performance at other volumes, with user speech, or on a
full-duplex phone. Next: verify a real speech-calibrated gain/threshold pair and
normal follow-up speech before deciding on an echo-control change. The phone
blocker remains open; the diagnostic makes that decision measurable.

Validation: 484 unit tests passed in 6.13s, including eleven diagnostic cases;
all ten real-stack smoke scenarios passed in 213s plus 7s warmup. Changed-file
lint and `git diff --check` passed. README documents the command and how to use a
calibrated pair. No production memory, gain settings or volume were changed.

## 16. Speech-calibrated local check — 2026-10-03

Mithilesh was using a **wired headset** and initially missed part of the speaking
window. That first sample was correctly rejected (speech P75 12.6, peak 163.7 RMS).
Windows lists no separate wired-headset input: these measurements used Realtek MME
input 1 and output 4. The endpoint name is Microphone Array; the physical mic path
behind the jack/driver was not independently established. Do not assume AirPods or
DJI merely because their disconnected WDM-KS endpoints appear in the device list.

A second spoken cue and eight-second speech window captured sustained speech,
starting about two seconds into the window. Room P90 was 0.49 RMS; speech P75 was
169.21, peak 820.28. The existing calibration algorithm accepted **gain 14.8,
threshold 1004**. No algorithm, configuration or Windows volume was changed.

Using that pair immediately afterwards:

| Window | Post-gain peak RMS | Speech frames | Endpoint |
|---|---|---|---|
| Tail 1 | 79.5 | 0 | no |
| Tail 2 | 2844.3 | 3 | no |
| Tail 3 | 11.8 | 0 | no |
| Spoken follow-up | 10793.1 | 22 | yes |

None of the three tails met the four-frame speech minimum. The requested follow-up
sentence produced an endpoint during its nine-second window. This checks levels
and endpointing, not the accuracy of a transcript; PCM was discarded without STT,
memory writes or saving audio. Tail 2's brief excursion is not an echo-free claim.
Local half-duplex validation now has one successful calibrated sample; other
volumes, physical mic paths and full-duplex telephone echo remain unverified.

`run` previously accepted gain alone. It now accepts `--threshold` as well, so this
pair can actually be reused without editing config:

```powershell
.venv\Scripts\python.exe -m jesse run --ollama --device 1 --gain 14.8 --threshold 1004
```

Either override selects manual mode, skipping auto-calibration; omitted values keep
configured defaults. Use both measured values together. Invalid, missing, non-finite
or non-positive values exit before Ollama, the instance claim, UI or microphone
startup. No-override auto-calibration and `--no-calibrate` behavior are unchanged.

Validation: 501 tests passed in 6.31s, including seventeen CLI cases for the measured
pair, single overrides, automatic/default behavior and early invalid-value rejection.
All ten real-stack smoke scenarios passed in 216s plus 7s warmup. Changed-file
duplicate-definition/undefined-name lint and `git diff --check` passed.

## 17. Duplicate-instance startup guard — 2026-10-03

`run` printed the existing-instance warning but failed to return, continuing into
UI startup and model/audio construction despite the known microphone conflict.
Two regression cases reproduced this before the fix (plain run and UI/remote run).
It now returns exit code 1 immediately after a failed instance claim. It does not
release the existing owner's claim or start UI, calibration, models or capture.
The warning no longer recommends `--no-mic`, which is not implemented.

This fixes the caller's ignored conflict result, not the best-effort PID mechanism:
simultaneous launches can still race, and inability to read/write the PID file is
still allowed by the existing helper. No stronger locking guarantee is claimed.

The normal live conversation attempt used gain 14.8 / threshold 1004 and was closed
with Ctrl-C. Logs eventually showed a wake followed by a quiet-listening timeout,
but no completed spoken user turn. Background memory catch-up also logged an
Ollama request timeout. This attempt does not validate conversational self-echo or
invalidate the successful calibrated level/endpoint probe in §16. The user asked
to move on; no further microphone session was left running.

Validation: 503 tests passed in 14.53s. A separate real child-process check against
an isolated live-owner PID file exited 1 and preserved that claim. All ten real-stack
smoke scenarios passed in 223s plus 12s warmup. Changed-file duplicate-definition /
undefined-name lint and `git diff --check` passed.

## 18. Explicit memory-request acknowledgement — 2026-10-03

Repeated smoke sessions stored the turquoise preference correctly while replying
"I don't have a favorite color" or "Nope". The current turn was still sent through
the general persona before background extraction. The pre-change replay this time
said "Turquoise", so the bad reply is intermittent, not a deterministic storage
failure. The earlier smoke assertion checked persistence only and accepted either
response. Two routing regressions failed before the fix.

An explicit `remember that ...` declaration (optionally prefixed with `please`)
now receives "Got it. I'll try to remember that." directly in the shared voice/text
handler. Both turns are persisted and the same background extractor runs; no extra
generation or write path was added. This acknowledges receipt, not a completed
durable write. Without a store or extractor, Jesse instead says lasting memory is
unavailable. Extraction failures remain unprocessed for the existing restart retry.

The parser deliberately excludes question-ending utterances, `do you remember`,
`remember my birthday`, `remember to ...`, reminder requests, forgetting and empty
declarations. These retain existing routing; this is not a general paraphrase
classifier. Explicit memory requests take precedence over incidental document words.
The extractor checks a recognized declaration's body for a hypothetical opening so
`remember that if ...` cannot bypass the existing guard. Its model input remains the
original user-only statement, both live and during catch-up.

The memory smoke scenario now requires the honest acknowledgement as well as saved
facts and recall through a fresh database connection. Three harness regressions
verify that a wrong persona reply, premature saved claim or absent reply cannot pass
merely because the fact was stored. This does not guarantee extraction of every
declaration, nor add a later spoken success notification.

Validation: 526 tests passed in 5.96s. The additional UI/continuous-listening
assertions and progress-entry checks passed in the targeted rerun. Changed-file
duplicate-definition/undefined-name lint and `git diff --check` passed.
All ten real-stack smoke scenarios passed in 224s plus 7s warmup. The updated memory
scenario spoke the exact acknowledgement, extracted the preference on its first
attempt and recalled it through a fresh connection. No live memory data was edited.

## 19. Requested memory survives a follow-up — 2026-10-03

Two deterministic regressions reproduced the gap after an acknowledged memory
request: a fresh wake cancelled its extraction, while a continuous follow-up could
build its recall context before the fact existed. If extraction was still waiting
for the model lock, the new THINKING state made it skip entirely. The raw exchange
survived, but the fact could remain unavailable until a later restart retry.

Explicit `remember that ...` jobs are now tracked separately, survive a fresh wake,
and can finish when a turn is active. Multiple queued requests remain tracked even
after `_extract_task` points at a newer exchange. Personal/broad memory questions
wait for those jobs before reading facts. Forget requests wait as well, preventing
an unfinished explicit write from recreating a just-deleted fact. Cancelling the
follow-up does not cancel the underlying memory request. Normal shutdown waits for
tracked requests; forced shutdown still relies on the persisted exchange/retry path.

Incidental background extraction retains its existing cancellation/restart behavior.
Cancelling an asyncio waiter cannot stop a blocking Ollama HTTP thread, so cancelled
extraction now drains its worker while still holding the shared model lock, even
across repeated cancellation. It then stays unprocessed for retry. This prevents a
new generation from overlapping that old worker; it does not abort the HTTP request.
An immediate recall/forget or model reply may therefore wait for queued requests to
finish or time out (currently 90 seconds per request). The explicit-memory wait is
limited to recognized personal/broad memory questions and forget requests.

Extraction can still fail or return no facts; the acknowledgement remains tentative.
No dedupe thresholds, recall ranking, fact parsing or production memory were changed.
The new `memory-followup` smoke scenario holds the model gate, acknowledges a fact,
starts a fresh wake and immediate personal query, then releases the gate. It checks
the real model's answer, the stored fact and the original exchange's processed flags.

Six new tests cover fresh-wake and continuous recall, multiple requests, forget
ordering, cancelling a follow-up and repeated cancellation during a blocking worker.
The full unit suite passed: 532 tests in 7.69s.
Final verification on 2026-10-04: all 11 real-stack smoke scenarios passed in 245s
plus 17s warmup. The new immediate-recall scenario answered "Turquoise, dude." and
confirmed the requested fact and processed exchange in SQLite. The birthday-plan
scenario answered "Sky-diving next month." No live memory data was edited.

## 20. Recover from transcription failure — 2026-10-04

The STT worker ran before `_handle_utterance` and its error recovery. An exception
there left the live loop in THINKING, ignoring both speech and queued text. If the
transport ended, awaiting that failed turn also aborted normal shutdown work.
Seven regression cases failed before this fix.

The spoken-turn boundary now reports "Sorry, I couldn't transcribe that. Please
say it again." in the UI and through speech, then uses the same cleanup as ordinary
turns. Continuous mode resumes listening; otherwise it returns to idle. Pending
alerts are delivered, and the next spoken or typed turn can complete. Playback
failure during this error message is caught as well. No transcript or memory is
fabricated for the failed recording. Cancellation propagates without an apology
and restores the state; it does not forcibly stop the underlying STT thread.

Tests cover idle/continuous recovery, failed apology playback, pending alerts,
spoken/typed follow-ups and cancellation. All 539 unit tests passed in 6.53s;
changed-file duplicate-definition/undefined-name lint and `git diff --check` passed.
The new `transcription-recovery` smoke scenario injects one STT exception, then
uses real Whisper/Ollama/Piper for a spoken follow-up without another wake word.
All 12 real-stack smoke scenarios passed in 249s plus 7s warmup. The new scenario
reported the injected failure, transcribed "Say hello in five words." on the next
utterance, spoke a real model reply and ended in LISTENING. Production memory and
audio-device settings were untouched; this was headless, not a live headset test.

## 21. Reminder audio failures no longer stop the loop — 2026-10-04

Reminder playback had two unguarded paths: an idle announcement ran directly inside
the microphone loop, and queued announcements ran during turn cleanup. Playback or
synthesis errors could therefore abort capture or prevent continuous listening
from resuming, leaving later reminders queued. Successful alerts were also absent
from the text UI. Four regression cases failed before this fix.

Both sites now use one announcement helper. It logs the reminder in the UI before
trying audio and catches synthesis/playback exceptions, logging an explicit audio
failure in the UI and the technical error in the console. Remaining queued alerts
and subsequent turns continue. There is no spoken apology or automatic retry when
the output device may still be broken. Scheduler status still means the reminder
was fired, not that audio was heard; its existing at-most-once behavior is unchanged.
Cancellation still propagates. Alerts stay out of conversation history/extraction.

Five tests cover idle-loop survival, queued synthesis and playback failures,
later reminders and typed input, successful UI delivery, and cancellation. All
544 unit tests passed in 6.21s; changed-file duplicate-definition/undefined-name
lint and `git diff --check` passed. The new `alert-recovery` smoke scenario uses a
real due SQLite timer, injects one output-device failure after real synthesis,
then checks that a real voice exchange completes and the timer does not repeat.

Real-stack verification: 12/13 scenarios passed in 263s plus 6s warmup, including
the new alert recovery. The knowledge scenario failed because Whisper transcribed
the one-word "Yes" follow-up as "Nes?", so the short-affirmation route did not fire.
An unchanged targeted rerun passed in 26s plus 7s warmup: Whisper heard "Yes?" and
Jesse answered "Ferrets sleep 14-18 hours a day, in short bursts." This is not a
fully green single run and does not fix one-word recognition. The failed scenario
also logged a spurious `coffee preference` extraction from the ambiguous exchange;
that was investigated and guarded in §22. All databases here were
temporary. Logs are in ignored `data/smoke-alert-recovery.log` and
`data/smoke-alert-knowledge-rerun.log`; production memory was untouched.

## 22. Questions and isolated words are not durable facts — 2026-10-04

Reproduced §21's extraction failure directly with the real schema-constrained,
temperature-zero model: `Nes?` yielded subject `coffee preference`, text `the user
likes Nes`, confidence 0.8. `Nes` yielded `the user's name is Nes`, and `Do I like
coffee?` yielded `the user does not like coffee`, also at 0.8. These pass the old
third-person/length/confidence checks. This is input interpretation by the extractor,
not recall ranking or the 0.88 dedupe threshold. No prompt change was needed.

FactExtractor now rejects turns consisting only of recognized questions,
single-word fragments or bare acknowledgments before asking the model. It handles
question marks and common unpunctuated interrogative openings, including contractions.
The same gate applies to live jobs, explicit requests and startup catch-up. Rejected
exchanges stay in the transcript and are marked processed, so they do not repeatedly
reach extraction. This does not alter the conversation reply or document routing.

Short declarations such as `I like tea` and `I'm vegetarian` still reach the model,
as do mixed statement/question turns, passed unchanged. This intentionally does not
resolve an isolated answer against a previous question: the user-only extractor
does not have that context. It can miss a fact phrased as a question or a standalone
name; unsupported question forms and mixed turns still depend on the model. It is
not a general grounding verifier or a repair of existing stored facts. No production
memory was inspected or changed, and no fact merging threshold was modified.

Twenty regression/control cases cover the gate, retained declarations, real SQLite
persistence and startup catch-up. Fourteen failed before the fix. All 564 tests
passed in 6.66s; changed-file duplicate-definition/undefined-name lint and diff checks
passed. The new `extraction-guard` smoke scenario checks the reproduced bad inputs
through the orchestrator/store, then verifies a real model still stores an explicit
turquoise preference.
All 14 real-stack smoke scenarios passed in 257s plus 7s warmup, including the
previously intermittent knowledge scenario. The new guard scenario stored no facts
from the four nonstatement probes, marked those exchanges processed, and saved the
explicit turquoise preference. The ignored log is `data/smoke-extraction-guard.log`.

## 23. Full-vision milestone tooling — 2026-10-05

User direction: finish the full vision, including custom wake and voice. Voice
brief: American, about 21, charismatic; he will clone **his own voice**. Runtime
integration is ready for exported assets via `run --wake-model FILE.onnx
--wake-phrase "yo Jesse" --wake-threshold 0.5 --voice en_US-jesse-medium`.
The custom detector handles wake and interruption; the spoken phrase separately
drives transcript cleanup. Custom exports load before capture/memory construction;
bad exports produce a startup error and release the instance claim. Explicit voice
selection requires ONNX/config files and never silently uses a stub. Defaults are
unchanged and no trained custom asset has been installed.

`voice-record` records a resumable, reviewed pilot: countdown, fixed-duration
capture, near-silence/clipping checks, playback, explicit keep/retry/skip/quit.
Accepted clips are unique mono 22050-Hz PCM WAVs plus Piper-compatible metadata
in ignored `data/voice-training`. Existing takes are not overwritten. Forty
original prompts are a pilot, not a production dataset. No real voice samples
were captured or uploaded during implementation; hardware recording remains to be
validated with the user. See `training/README.md` for the separate training workflow.

`wake-check` evaluates labeled 16-kHz WAV folders, reporting per-clip scores/events,
sampled recall, false activations/hour, threshold and model SHA-256. It requires
20 positive clips and one hour negative audio before its sample gate can pass;
human coverage and live echo/interruption acceptance remain separate. A real
stock-asset probe loaded the explicit ONNX path and Ryan voice, detected synthetic
"hey jarvis" at 0.998, did not fire on one negative, and correctly returned failure
for insufficient evaluation coverage. This verifies tooling, not `yo Jesse` quality.

`Start-Jesse.cmd` uses the repo venv, real Ollama and local UI from any working
directory. README now reflects the actual voice, CPU reasoning, energy VAD, model
cache setup and active full-vision milestones. `RELEASE.md` is the finish checklist;
the wake JSON is an overlay for the upstream training notebook, not a trained model.

Validation: 585 unit tests passed in 13.20s, including 21 new asset/recording/evaluation
cases. Undefined-name/duplicate-definition lint and diff checks passed.
All 14 real-stack smoke scenarios passed in 270s plus 16s warmup. The launcher
was invoked from outside the repo and correctly selected the project venv/root,
then rejected a missing custom asset before audio startup. Default-runtime smoke
does not validate the untrained custom assets or the recorder's physical mic path.

## 24. Free-tier Colab wake training prepared — 2026-10-06

Mithilesh chose **Google Colab GPU only if free**. Do not purchase compute, enroll
in a paid plan or upload his own voice recordings. Colab's free GPU availability
and session duration are not guaranteed. This machine has no installed WSL distro,
only Python 3.13, and no usable training environment; no voice pilot files exist.

Open **training/train_yo_jesse.ipynb** through the Colab link in training/README.md.
The notebook prepares a separate Python 3.10 venv, Torch 2.5.1/CUDA 12.1 and ORT
GPU 1.20.2. It pins openWakeWord, the compatible dscripka Piper generator fork and
public dataset revisions. No Drive mount, personal files or credentials are used.
It checks the GPU/disk, probes real synthetic generation and CUDA embeddings, then
downloads separate negative training/validation features, MIT RIRs and one AudioSet
background shard. Budget roughly 25 GB downloads and 60 GiB initial free disk.

The original upstream notebook is stale: its current rhasspy generator URL lacks
the expected generate_samples.py, and the trainer uses five string "False" defaults.
The latter accidentally activates optional TFLite conversion. A separate trainer
copy changes those defaults to booleans; tests ensure unselected stages remain off.
Validation features are a 2-D continuous stream (481345, 96), verified by reading
the actual public NPY header. Training features are 3-D windows.

Each stage stops on subprocess failure. Same-VM generation/downloads can resume;
augmentation explicitly rewrites derived arrays to avoid interrupted partial files.
Classifier training restarts on retry; VM deletion loses work. Download the ZIP
before expiry. It includes ONNX, config, source revisions, package freeze, logs and
an inference/hash report that explicitly leaves human acceptance false. Runtime
defaults are unchanged. No new wake model has been trained or accepted.

Notebook source is build_wake_notebook.py with tested wake_support.py helpers.
Regenerate after changing either helper or yo_jesse.json; helper hashes account for
Windows line endings. All 596 tests passed in 6.75s, including 11 new training cases.
Linux/Python 3.10 dependency resolution passed using uv in an ignored scratch
directory; this is not an installation/import test on Linux. Notebook and embedded
script syntax passed. An actual ORT inference/export-package smoke used the existing
stock Jarvis asset in data/wake-training-source/stock-export-probe; its renamed test
copy is NOT a trained Jesse model. No production asset was changed.

**Next:** first actual free Colab GPU run, repair any environment/runtime failures,
download and evaluate the trained candidate locally. Own-voice recordings and
phone transport remain separate unfinished gates. Progress v2.0.15 is deliberately
not significant: prepared training is not an acquired wake-word capability.

All 14 real-stack smoke scenarios passed in 260s plus 11s warmup; log:
data/smoke-wake-colab.log. This remains stock-asset runtime validation, not evidence
of Colab execution or yo Jesse detection quality. Staged Git blob hashes match the
helper hashes embedded in the notebook.

## 25. OmniVoice own-voice auditions — 2026-10-07

The user chose k2-fsa/OmniVoice and supplied Audio.mp4 after reading the suggested
paragraph. This supersedes Piper fine-tuning as the first cloning path. His prior
authorization covers his own voice; no recording was uploaded to any service.
The supplied file is unchanged. A local WAV extraction is 48.79 seconds long.

OmniVoice source is in ignored data/OmniVoice at
08be0b4ccbac3e13e374e86fbfead4b4cac343e2. A separate data/omnivoice-env contains
Python 3.13, Torch/torchaudio 2.8.0+cpu and Transformers 5.3.0. The full package
freeze is in data/voice-clone/environment.txt. Public model/tokenizer weights are
in data/omnivoice-model at revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4.
Nothing was installed into Jesse's existing runtime environment.

Local Whisper transcription supplied word timestamps. A complete conversational
excerpt (3.10–9.10s) and an expressive excerpt (37.10–46.50s) became separate
reference WAV/transcript pairs. Both produced the same new test sentence:
“Hey, it's Jesse. Good to hear from you. What are we working on today?”
Whisper recovered all intended words in both generated auditions.

Actual CPU float32/32-step/four-thread measurements, including prompt encoding:
conversational 136.87s for 4.44s audio; expressive 163.60s for 4.68s audio.
Model load was 2.02s/3.33s. This is far too slow for live conversation; it is an
audition workflow, not a completed low-latency TTS integration.

Each ignored data/voice-clone/audition-* directory contains voice-prompt.pt,
audition.wav, audition-listen.wav, report.json and whisper-check.txt. The listening
copies use constant gain to peak 0.85 because raw output was quiet. No pitch,
timing or dynamics processing was applied. Original generated WAVs are retained.
Saved prompts are reference tokens, not standalone Piper voices; future synthesis
still needs OmniVoice. See training/OMNIVOICE.md and training/clone_omnivoice.py.

**Pending:** the user must judge likeness and choose an audition/reference.
Then investigate a sufficiently fast local inference backend before changing
Jesse's voice. Ryan remains the default, and the custom wake/phone milestones remain
unfinished. Progress v2.0.16 is not significant because the live voice has not changed.
The two actual offline OmniVoice generations plus transcription checks are this
change's smoke verification; the core voice loop was not changed.

Validation: 607 regression tests passed in 6.59s, including 11 reference/preview
tests. Changed-file lint and diff checks passed. Audio, prompts, transcripts, model
weights, environment files and source checkout remain ignored and uncommitted.

## 26. OmniVoice GPU inference and speed experiments — 2026-10-07

The GTX 1050 works with CUDA despite the older Ollama/Vulkan discovery problem.
Actual nvidia-smi: 4 GB, compute capability 6.1, driver 512.78. A separate
data/omnivoice-gpu-env (Python 3.13, Torch/torchaudio 2.7.1+cu118, Transformers
5.3.0) passed a CUDA matrix operation and full OmniVoice synthesis. Source/model
revisions remain those in §25. No driver/system setting was changed. Environment
freeze: data/voice-clone/environment-gpu.txt. The CPU environment remains usable.

training/benchmark_omnivoice.py accepts a saved local voice prompt, runs entirely
offline, retains the model across its requested step-count tests, keeps the audio
codec on CPU, and moves the diffusion model to CUDA float32/float16. An experimental
CPU mode dynamically quantizes only the language model's linear layers. It does not
modify model files, select a voice, or change Jesse's runtime. Existing result
directories are refused; completed results survive a subsequent inference error.

Measured timings (cached conversational prompt, short greeting, four threads):
CPU int8 16/32 steps = 56.40/159.55s; GPU float32 16/32 = 11.08/17.77s;
GPU float16 8/16/32 = 5.95/9.47/18.09s. GPU float16 peak allocated tensor memory
was 1286 MiB, float32 2537 MiB. These are one-off measurements, exclude setup and
reference encoding, and are not strictly equivalent to §25's 136.87s CPU timing.
CPU quantization is not a demonstrated consistent win. Cold GPU setup adds ~10s.

A second sentence (“That sounds exciting. Tell me a little more about what you
have in mind.”) on GPU float16 took 3.95/5.27/9.12s at 4/8/16 steps for ~4.9s audio.
All intended words were recovered by local Whisper. Eight steps is a candidate,
four is an aggressive quality experiment; neither is likeness-approved. The
32-step fp16 greeting's Whisper result included trailing slash characters despite
recovering the words; do not claim every automated check was entirely clean.

All outputs/reports/transcripts are ignored under data/voice-clone/benchmark-*.
For a fresh audition use benchmark-cuda-fp16-heldout/steps-8-listen.wav and compare
steps-16-listen.wav. Prior A/B choice remains unanswered; the conversational prompt
was used for repeatable benchmarking, not treated as the chosen identity.

**Next:** user likeness/quality judgment, then a persistent local TTS backend with
bounded cancellation and real interruption tests. Keep the model warm rather than
launching a fresh process per sentence. Do not call speech-generation timings
end-to-end conversation latency. Wake training and phone milestones remain open.

Validation: 610 tests passed in 13.11s, including three new failure-preservation,
no-CUDA-fallback and no-overwrite tests. Lint passed. The real model generations
above are the new smoke evidence; the core runtime voice loop was unchanged.
Progress v2.0.17 is not significant because these are still offline auditions.

## 27. Optional persistent OmniVoice runtime — 2026-10-08

The GPU benchmark milestone was committed/pushed as 358ff57. There is now an
explicit live backend option, --omnivoice-profile FILE.json. The local candidate
profile is data/voice-clone/runtime-conversational.json: eight steps, CUDA fp16,
the conversational prompt from §25. This is an experiment, not a user-approved
voice selection. Default Piper/Ryan is unchanged. Do not upload the private
profile, prompt, audio or transcripts. See training/OMNIVOICE.md for setup/launch.

jesse/tts/omnivoice.py launches a persistent JSON-lines worker in the separate
GPU environment, using local model/prompt paths and offline flags. No service
port and no torch dependency added to Jesse's main environment. The worker keeps
the diffusion model on GPU and codec on CPU, returns mono 24kHz PCM with constant
peak gain (same audible level as previews). A sentence finishes generating before
chunk playback starts; this is not low-latency token-level audio streaming.

Both desk and remote transports now pull synchronous audio iterators in a worker
thread. Previously the next() call could block microphone ingestion during TTS.
The orchestrator supplies a thread-safe interruption event to the optional backend.
Stopping during generation kills/reaps the process, discards its unfinished audio,
and allows a fresh worker next turn. Stopping after generation keeps the warm
worker and ends chunk playback. Session shutdown/capture failure closes the worker.
Timeout is 180s; CUDA/model errors are explicit rather than silently choosing CPU
or a different voice. Unheard sentences interrupted before first audio are excluded
from the returned spoken reply/history. Existing partial-sentence accounting still
records the whole sentence once audio starts; word-level playback tracking is not new.

Real local worker smoke (training/smoke_omnivoice.py), no microphone or speakers:

- Same worker reused for two replies: 6.14s/2.88s audio and 4.66s/5.23s audio.
- Stop at 0.50s returned at 0.54s with no PCM and the worker reaped.
- A subsequent reply succeeded in 21.51s including model reload, 3.26s audio.
- Cold startup was 55.21s in this run (earlier benchmark setup ~10s). It varies.
- Worker closed at the end; local Whisper recovered all intended words from the
  three generated WAVs. Similarity/naturalness have NOT been judged by the user.

Evidence: ignored data/voice-clone/runtime-smoke-2026-10-08/{report.json,
word-checks.json,first.wav,second.wav,after-interruption.wav}. Listen to second.wav
for a warm-worker sample. These timings are synthesis only, not LLM-to-audio latency.
Killing the model makes interruption responsive but the next reply slower. Do not
claim this is already comfortable for daily conversation.

Validation: 623 tests passed in 9.02s. New real-subprocess tests cover reuse,
crash/error/timeout recovery, cancellation before audio, local/remote loop
responsiveness, shutdown cleanup, and output-device failure closing its generator.
Full real-stack smoke passed 14/14 in 253s (+12s warmup), including birthday-plan
recall and barge-in. Final barge-in passed again in 10s (+7s warmup) after the
unheard-history adjustment.
New-module and supporting-file lint passed; older orchestrator/main lint debt is
unchanged. Progress v2.0.18 is significant because the clone can join the live loop.

Next: user likeness judgment and a timed live headset conversation with the opt-in
profile. Consider retaining the model through cancellation to reduce the 21.5s
recovery pause. Custom yo Jesse training is still only a prepared free-Colab
notebook, and phone-number integration remains unfinished. No paid compute or
cloud upload of voice recordings has been authorized.

## 28. User rejected the robotic clone — 2026-10-08

After 426b21b, Mithilesh listened to runtime-smoke-2026-10-08/second.wav and said
"no, sounds very robotic". Treat the eight-step fp16 conversational candidate as
REJECTED, not merely unreviewed. Do not promote the existing runtime-conversational
profile or treat ASR success as likeness/naturalness acceptance. Piper is unchanged.

Investigated local upstream generation parameters, reference word boundaries, and
audio formatting. The excerpts contain complete phrases (conversational ends at
8.82s inside the 3.10–9.10s crop; expressive starts at 37.32s inside 37.10–46.50s).
No clipping/sample-rate mismatch found in the new 24kHz mono samples. Preview gain
changes neither sample count nor sample rate. This is not proof of a root cause.
Eight steps is below upstream's 32-step default/16-step speed recommendation.

New actual GPU generations, same text as the rejected sample and seed 42:
conversational fp32 at 8/32 steps = 5.39/16.55s for 4.86/4.94s audio;
expressive fp32 at 32 steps = 19.54s for 5.04s audio. Scripts already support these
experiments; no runtime change was needed. Compare A and B:

- A: data/voice-clone/quality-conversational-fp32/steps-32-listen.wav
- B: data/voice-clone/quality-expressive-fp32/steps-32-listen.wav

The two 32-step samples differ only in reference prompt (same sentence, seed,
precision and generation settings). A matching eight-step fp32 baseline is saved
to isolate step count when listening. Do not attribute improvement to precision or
steps before listening. User has not judged A/B yet. Both passed local Whisper
word checks, as did the baseline, and are unclipped. Details and exact user feedback
are in ignored data/voice-clone/quality-review-2026-10-08.json. All audio stays local.

Only docs and an insignificant v2.0.19 progress entry changed. New real generations
are the relevant smoke evidence; runtime code is unchanged. All 623 tests passed
in 9.37s and diff checks passed. Next is user A/B judgment;
if both remain robotic, inspect reference/codec quality and alternate excerpts.
Naturalness comes before further latency optimization. No accepted own-voice clone yet.

## 29. B preferred; more natural tone still required — 2026-10-08

User verdict: "B is somewhat natural. I don't want to feel like I'm talking to a
robot. I want more natural tone". B = quality-expressive-fp32/steps-32-listen.wav.
This selects the expressive reference as the starting point, not final voice
acceptance. Natural delivery matters more than the earlier speed target.

Actual discovery: the pinned OmniVoice _resolve_instruct accepts only fixed tags
for age/gender/pitch/accent/whisper, not arbitrary prose directions. A warm/relaxed/
conversational instruction was rejected before synthesis. Failure report retained
under ignored tone-warm-fp32. Do not suggest that prompt as a working emotion control.
The temporary CLI instruction option was removed before commit.

training/benchmark_omnivoice.py now exposes supported --speed and --keep-pauses
options, records overrides plus seed 42, and retains old behavior if omitted.
No runtime backend setting changed. Slowing alone does not prove naturalness.

New local real-model auditions, same words as B, CUDA fp32/32 steps/seed 42:

- data/voice-clone/tone-relaxed-fp32/steps-32-listen.wav: B prompt, speed .95,
  output silence removal disabled. Generation 19.57s, audio 5.28s.
- data/voice-clone/tone-original-pauses-fp32/steps-32-listen.wav: same expressive.wav
  encoded again with preprocess_prompt=False; default speed; output silence removal
  disabled. Generation 27.54s, audio 7.00s. This retains reference pauses but also
  changes the estimated duration, so do not claim exact source pacing transfer.

Rebuilt prompt/report: data/voice-clone/audition-expressive-untrimmed. Original prompt
preserved. Both outputs are unclipped 24kHz mono and local Whisper recovered all
intended words. Report with exact preference: data/voice-clone/tone-review-2026-10-08.json.
User must judge whether either sounds more natural than B; no such judgment yet.

Validation: 630 tests passed in 10.33s, new options/defaults/provenance and invalid
speed checks included; changed script/test lint passed. The actual generations
above are this isolated audition-tool change's smoke evidence. Progress v2.0.20
is insignificant. No cloud upload, paid compute, default voice switch, or new branch.

Next: user's tone comparison. If both still feel stiff, use a short spontaneous
recording in his desired everyday tone as another reference experiment, rather
than promising longer scripts or more decoding steps will make it human-like.
The remaining custom wake-word and phone-call milestones are unchanged.
