"""Two microphones, one Jesse.

The gap the `AudioTransport` interface did not cover. `capture()`/`play()` map onto a
remote client perfectly well — that seam paid off. What it never anticipated is that
the orchestrator binds ONE transport for the life of the process:

    async for frame in self.transport.capture():

A phone that joins mid-session, takes over, and hands back afterwards is a *session*,
and nothing in the interface or the loop had a notion of one. `SwitchingTransport` is
that missing layer, and it deliberately implements the same interface so the
orchestrator above it needs no changes at all.

**Exclusive, not merged.** While the phone is live the desk microphone is ignored and
his replies go only to the phone. Two live sources feeding one wake detector would
interleave room noise with phone audio into a model that expects one continuous
stream — which is exactly how the detector was made to go deaf once before (§6). And
playing his voice into an empty room is pointless; into an occupied one it is worse.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Iterator

from jesse.audio.frames import CHUNK_SAMPLES, SAMPLE_RATE

SILENCE = b"\x00" * (CHUNK_SAMPLES * 2)


class RemoteSource:
    """Audio arriving from the phone, and audio waiting to go back to it.

    Frames are pushed in from the HTTP thread and pulled out on the asyncio loop, so
    everything crossing that boundary is a plain thread-safe deque under a lock rather
    than an asyncio primitive bound to one loop.
    """

    def __init__(self, *, idle_timeout: float = 12.0, max_queued: int = 400) -> None:
        self._lock = asyncio.Lock()          # only ever taken on the loop side
        self._frames: list[bytes] = []
        self._outbound: list[tuple[bytes, int]] = []
        self._idle_timeout = idle_timeout
        self._max_queued = max_queued
        self.last_seen = 0.0
        self.muted = False                   # he is speaking; the phone stops sending
        self._active = False

    # -- pushed from the HTTP thread ---------------------------------------

    def submit(self, pcm: bytes) -> None:
        """16 kHz mono little-endian Int16, any length. Split into pipeline frames."""
        self.last_seen = time.time()
        self._active = True
        if self.muted or not pcm:
            return                            # half-duplex: drop his own echo
        step = CHUNK_SAMPLES * 2
        for i in range(0, len(pcm) - step + 1, step):
            if len(self._frames) >= self._max_queued:
                # A phone that gets ahead of the pipeline is better trimmed than
                # buffered: stale audio answers a question he has stopped asking.
                del self._frames[0]
            self._frames.append(pcm[i:i + step])

    def take_reply(self) -> tuple[bytes, int] | None:
        """His next spoken reply for the phone to play, if any."""
        return self._outbound.pop(0) if self._outbound else None

    def pending_replies(self) -> int:
        return len(self._outbound)

    # -- used on the asyncio loop ------------------------------------------

    @property
    def active(self) -> bool:
        """True while the phone is still there. Goes false when it stops calling.

        Silence while HE is speaking does not count. The page deliberately stops
        uploading during playback — that is the half-duplex rule, without which his
        own voice comes back through the phone's speaker and trips the stop-word — so
        ageing out on it cannot distinguish "hung up" from "politely quiet". Measured
        the hard way: a reply longer than the timeout handed the floor back to the
        desk mid-conversation and played his answer into an empty room.
        """
        if not self._active:
            return False
        if self.muted:
            return True
        if time.time() - self.last_seen > self._idle_timeout:
            self._active = False
        return self._active

    def end(self) -> None:
        self._active = False
        self._frames.clear()

    def next_frame(self) -> bytes | None:
        return self._frames.pop(0) if self._frames else None

    def queue_reply(self, pcm: bytes, sample_rate: int) -> None:
        self._outbound.append((pcm, sample_rate))


class SwitchingTransport:
    """An `AudioTransport` that is the desk by default and the phone when he calls.

    Implements the interface exactly, so `Orchestrator` is untouched by any of this.
    """

    def __init__(self, local, remote: RemoteSource, *,
                 on_switch=None) -> None:
        self.local = local
        self.remote = remote
        self._on_switch = on_switch
        self._remote_live = False
        self.frames_from_remote = 0
        self.frames_from_local = 0

    @property
    def remote_live(self) -> bool:
        return self._remote_live

    def _set_remote(self, live: bool) -> None:
        if live == self._remote_live:
            return
        self._remote_live = live
        print(f"  [remote] phone {'joined' if live else 'left'} — "
              f"{'desk mic ignored' if live else 'back to the desk mic'}")
        if self._on_switch is not None:
            self._on_switch(live)

    async def capture(self) -> AsyncIterator[bytes]:
        """Desk frames until the phone joins, phone frames until it stops.

        The local generator is drained either way so it never blocks on a full buffer;
        its frames are simply discarded while the phone has the floor.
        """
        local = self.local.capture().__aiter__()
        while True:
            try:
                frame = await local.__anext__()
            except StopAsyncIteration:
                break
            if self.remote.active:
                self._set_remote(True)
                # Hand over every phone frame that has arrived since the last tick,
                # so the wake detector sees one continuous stream rather than a
                # decimated one — it needs about a second of unbroken audio.
                sent = False
                while (pending := self.remote.next_frame()) is not None:
                    self.frames_from_remote += 1
                    sent = True
                    yield pending
                if not sent:
                    yield SILENCE          # keep the loop ticking between chunks
                continue
            self._set_remote(False)
            self.frames_from_local += 1
            yield frame

    async def play(self, frames: Iterator[bytes], *, sample_rate: int = SAMPLE_RATE) -> None:
        """To the phone while it has the floor, otherwise to the desk speakers."""
        if self._remote_live:
            pcm = b"".join(frames)
            if pcm:
                self.remote.queue_reply(pcm, sample_rate)
            return
        await self.local.play(frames, sample_rate=sample_rate)

    # Calibration reaches through to the real microphone: `jesse run` sets a measured
    # gain on the transport, and wrapping one must not hide it.
    @property
    def gain(self) -> float:
        return getattr(self.local, "gain", 1.0)

    @gain.setter
    def gain(self, value: float) -> None:
        if hasattr(self.local, "gain"):
            self.local.gain = value

    def mute_input(self) -> None:
        self.remote.muted = True
        self.local.mute_input()

    def unmute_input(self) -> None:
        # Give the phone a fresh window to be heard from: it has been told not to send
        # for the whole of his reply, so judging it on the last thing it sent before
        # that would age it out the instant he stops talking.
        self.remote.last_seen = time.time()
        self.remote.muted = False
        self.local.unmute_input()
