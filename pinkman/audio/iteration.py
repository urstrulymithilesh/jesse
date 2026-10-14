"""Pull potentially slow synthesis chunks without blocking microphone ingestion."""

import asyncio


async def audio_chunks(frames):
    iterator = iter(frames)
    end = object()
    try:
        while True:
            pending = asyncio.create_task(asyncio.to_thread(next, iterator, end))
            try:
                chunk = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # The caller signals synthesis interruption. Drain the in-flight next
                # before closing its generator (closing a running generator is illegal).
                try:
                    await pending
                finally:
                    raise
            if chunk is end:
                return
            yield chunk
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
