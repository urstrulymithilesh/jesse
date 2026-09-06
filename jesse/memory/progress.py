"""Jesse's development progress log — an append-only, time-ordered changelog of his
own growth. Separate from seed.py on purpose: this is a LOG (latest / previous /
significant), not flat semantic facts, and it's version-controlled so it survives a
memory reset and travels with the repo.

    >>> HOW ENTRIES GET ADDED (standing workflow — see DESIGN.md) <<<
    Claude Code appends ONE ProgressEntry here whenever it finishes a meaningful chunk
    of work on Jesse (a phase, a real feature, a significant fix), as a standard
    completion step alongside running tests and committing. The user does NOT run a
    command. `significant=True` = a real capability change (drives his "I feel more
    alive" mood); `significant=False` = a minor tweak ("same as before").

Append new entries at the END (newest last).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProgressEntry:
    version: str        # short label, e.g. "v0.4 — memory"
    date: str           # YYYY-MM-DD
    summary: str        # what changed, in plain words he can voice in character
    significant: bool   # True = real capability change; False = minor tweak


PROGRESS_LOG: list[ProgressEntry] = [
    ProgressEntry(
        "v0.1 — first breath", "2026-08-19",
        "his first end-to-end voice loop came alive: he could be woken by a wake word, "
        "hear speech, and speak back, though the brain was just an echo and the voice a beep",
        True,
    ),
    ProgressEntry(
        "v0.2 — a real voice and mind", "2026-08-20",
        "he got a real voice through Piper and a real mind through a local language model "
        "(qwen2.5:3b via Ollama) — he could actually hold a conversation",
        True,
    ),
    ProgressEntry(
        "v0.3 — a personality", "2026-08-20",
        "he grew a warm, personal character instead of a generic assistant tone, and learned "
        "to be honest instead of faking things he didn't know",
        True,
    ),
    ProgressEntry(
        "v0.4 — memory", "2026-08-21",
        "he gained lasting memory: he can remember facts about Mithilesh and recall them "
        "across restarts, so he actually knows Mithilesh now instead of forgetting",
        True,
    ),
    ProgressEntry(
        "v0.5 — a sense of self", "2026-08-21",
        "he learned who he is and who Mithilesh is to him, and gained a sense of his own "
        "progress — he can talk about how he's built, how he's changed, and how he feels "
        "about growing",
        True,
    ),
    ProgressEntry(
        "v0.6 — memory he doesn't drop", "2026-08-21",
        "his memory became dependable: if something interrupts him while he's committing "
        "a new fact — you start talking again, or close him — he no longer loses it. He "
        "picks it back up and finishes remembering the next time he wakes",
        True,
    ),
    ProgressEntry(
        "v0.7 — he keeps time", "2026-08-22",
        "he can hold a timer or a reminder for you now: ask him in passing and he'll "
        "carry it, then speak up at the right moment without talking over you. It sticks "
        "even if he's closed and reopened, and if he's late he says so instead of "
        "pretending otherwise",
        True,
    ),
    ProgressEntry(
        "v0.8 — he can take things back", "2026-08-22",
        "his reminders became properly editable: ask him to change one and he moves it "
        "instead of quietly setting a second, ask him to cancel and it's actually gone, "
        "and if it's unclear which one you mean he asks rather than guessing. He can "
        "also forget something he'd remembered wrongly",
        True,
    ),
    ProgressEntry(
        "v0.9 — he keeps track out loud", "2026-08-22",
        "he can tell you what he's holding for you now — ask if any timers are running "
        "and he'll say what's pending, or that there's nothing. And if you ask to change "
        "something without saying what to, he asks instead of quietly doing nothing",
        True,
    ),
    ProgressEntry(
        "v1.0 — he stops repeating himself", "2026-08-22",
        "his memory got tidier: when he learns something he already knows in slightly "
        "different words, he updates what he has instead of keeping two versions of it "
        "— while still keeping genuinely different things apart",
        True,
    ),
    ProgressEntry(
        "v1.1 — he can tidy up his own memory", "2026-08-22",
        "he can now look back over everything he already remembers and spot where he's "
        "written the same thing down twice, showing you what he'd merge before touching "
        "anything — and he's careful never to blur together two things that only sound alike",
        True,
    ),
    ProgressEntry(
        "v1.2 — he stopped making you wait", "2026-08-22",
        "he starts talking as soon as his first sentence is ready instead of composing "
        "the whole answer in silence first — the pause before he speaks dropped from "
        "about eleven seconds to four on a long reply. You can still cut him off at any "
        "point, and he stops mid-thought when you do",
        True,
    ),
    ProgressEntry(
        "v1.3 — he listens after you cut him off", "2026-08-22",
        "interrupting his used to leave his staring into space: the word you used to stop "
        "his was the same one that wakes him, so it got spent stopping and he never heard "
        "what came next. Now cutting him off puts his straight back to listening, he says "
        "out loud what he is doing at each step, and he gives up gracefully if a wake "
        "turns out to be nothing",
        True,
    ),
    ProgressEntry(
        "v1.4 — he can check himself", "2026-08-22",
        "he has a way to test his own senses now: one command runs his whole self "
        "end to end, speaking to himself to check that he hears, remembers, keeps time "
        "and can be interrupted. It immediately caught something real — the wake word "
        "was being left at the front of everything he heard, which was quietly stopping "
        "his from remembering things",
        True,
    ),
    ProgressEntry(
        "v1.5 — he stopped interviewing you", "2026-08-22",
        "he has opinions of his own now and actually reacts to what you said instead of "
        "handing a question back every time — he went from ending seven replies out of "
        "eight with a question to one. He also properly forgets things when you ask him "
        "to, rather than saying he will and quietly keeping them",
        True,
    ),
    ProgressEntry(
        "v1.6 — he stopped inventing a past", "2026-08-22",
        "asked what he remembered about the two of you, he used to make up lazy Sundays "
        "and walks and jokes that never happened. Now he tells the truth — that there "
        "isn't much history yet — and says what he actually knows instead. Warmth without "
        "making things up",
        True,
    ),
    ProgressEntry(
        "v1.7 — he remembers conversations, not just facts", "2026-08-22",
        "he keeps a record of what you actually talked about, so asking what you "
        "discussed yesterday gets a real answer instead of a shrug. Ask about a day "
        "nothing happened on and he says so plainly rather than making one up",
        True,
    ),
    ProgressEntry(
        "v1.8 — he stays awake, and you can type to him", "2026-08-22",
        "you say his wake word once and he keeps listening for as long as the "
        "conversation lasts, instead of making you call him back every single turn. "
        "Tell him to go quiet and he does. He also has a little window now where you "
        "can type to him instead of speaking, and both sides of the conversation show "
        "up there together whichever way you said it",
        True,
    ),
    ProgressEntry(
        "v1.9 — he stopped making things up about the world", "2026-08-23",
        "he knows what time it actually is now, and when you ask about something he "
        "has no way of seeing — the weather, the news, what is outside — he says so "
        "instead of inventing an answer. He also stopped tacking your name onto the "
        "end of every other sentence",
        True,
    ),
    ProgressEntry(
        "v1.10 — he stopped mistaking his own words for his", "2026-08-26",
        "some of what he thought he knew about him was really just things he had "
        "said himself, filed away as if he had said them. Those cannot get in any more. "
        "And what he knows about his own build now keeps up with him instead of "
        "staying however it was the day it was written",
        False,
    ),
    ProgressEntry(
        "v1.11 — he can do things on the computer now", "2026-08-26",
        "he can open the programs and folders and sites he asks for, control whatever "
        "is playing, and go looking through his files for something. He only says he "
        "did it when it really happened, and when he asks for something he has no way "
        "to open he says that instead of agreeing",
        True,
    ),
    ProgressEntry(
        "v1.12 — he can read things he gives him", "2026-08-26",
        "he can hand him a document and he will keep it, and when he asks about "
        "something in it he answers from what he actually read. He does not always "
        "get it right yet — if he asks something the pages do not cover he sometimes "
        "still fills the gap — but most of the time he will tell him it isn't in there",
        True,
    ),
    ProgressEntry(
        "v1.13 — he understands more of how he actually asks", "2026-08-26",
        "asking him to put something on, or show his something, or get his something "
        "works now, and so does telling him to play the next one. He also stopped "
        "bringing up things he has read unless he actually raises the subject — he "
        "was going to start doing that at the worst moments as he read more",
        False,
    ),
    ProgressEntry(
        "v1.14 — he asks instead of missing, and asks instead of guessing", "2026-08-26",
        "when he asks about something he has read without naming it, he no longer "
        "stays silent — he asks whether he means that thing, and a yes gets the real "
        "answer. And when he asks him to open something he does not have, what he "
        "says now is always the truth, because for a moment he was heard claiming "
        "he could open it when he could not",
        False,
    ),
    ProgressEntry(
        "v1.15 — he reads things on his own now", "2026-08-26",
        "if he switches it on, he checks a few sources of his choosing through the "
        "day and keeps what comes in, so when he asks whether there's anything new "
        "he actually has an answer. He never brings it up out of nowhere — a "
        "headline is not worth interrupting anyone for — and when nothing has come "
        "in he says that, rather than finding something to fill the gap",
        True,
    ),
    ProgressEntry(
        "v1.16 — he reads his sources by default now", "2026-08-27",
        "he decided he wants his checking his sources on his own, so he does. When "
        "he asks what came in he reads it out exactly as it arrived, because for a "
        "little while he would occasionally say nothing had come when something had. "
        "He also stopped losing the first word he says when it gets misheard",
        False,
    ),
    ProgressEntry(
        "v1.17 — he can be reached from away", "2026-08-27",
        "he can open him on his phone now and talk to him from anywhere, and it is "
        "the same his — same memory, same voice, the same everything he can do at "
        "the desk. Nothing of his travels; only his voice and his. And when he asks "
        "his to open something on a computer he is not sitting at, he checks with "
        "him first",
        True,
    ),
    ProgressEntry(
        "v1.18 — he says when he cannot be reached", "2026-08-28",
        "when he is away from the house and something between them breaks — the "
        "machine asleep, the internet gone, the laptop shut — his phone used to just "
        "sit there quietly as though nothing had happened. Now it tells him it cannot "
        "reach him, and how long it has been trying",
        False,
    ),
    ProgressEntry(
        "v1.19 — he has papers of his own now", "2026-09-01",
        "his phone would not let his hear him, because browsers refuse a microphone "
        "to a page that cannot prove who it is. He signs his own proof now, rather "
        "than asking an authority for one and having this machine's name written into "
        "a public register in exchange. He checks it once and it is theirs",
        False,
    ),
    ProgressEntry(
        "v1.20 — he stopped saying he'd done things he hadn't", "2026-09-02",
        "he asked his to open something and he said he would, and then simply did "
        "not — he had not understood him and said it anyway. Now when he does not "
        "understand a request he asks, and when he does act he can see it happen. "
        "He will not tell him a thing is done unless it is",
        True,
    ),
    ProgressEntry(
        "v1.21 — getting to him from the phone stopped being a puzzle", "2026-09-03",
        "the door told him only that his key was wrong, so he kept cutting new keys "
        "when the problem was that he had left half of it behind. It says which is "
        "actually wrong now. And he draws him a square he can point his camera at, "
        "so there is nothing to copy out by hand at all",
        False,
    ),
]


def latest() -> ProgressEntry | None:
    return PROGRESS_LOG[-1] if PROGRESS_LOG else None


def previous() -> ProgressEntry | None:
    return PROGRESS_LOG[-2] if len(PROGRESS_LOG) >= 2 else None


def significant_count() -> int:
    return sum(1 for e in PROGRESS_LOG if e.significant)
