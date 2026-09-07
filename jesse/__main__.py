"""Entry point.

    python -m jesse              # status
    python -m jesse --spike      # hardware + plumbing spike
    python -m jesse run          # run the live walking-skeleton loop (needs mic + models)
    python -m jesse run --ollama # same, but use the real Ollama brain instead of Echo
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path

from jesse import __version__
from jesse.config import CONFIG


def _status() -> int:
    print(f"Jesse v{__version__} — fully-local voice partner")
    print(f"  reasoning : {CONFIG.reasoning.model} via {CONFIG.reasoning.ollama_host}")
    print(f"  stt       : faster-whisper {CONFIG.speech.whisper_model} ({CONFIG.speech.whisper_compute_type}, CPU)")
    print(f"  wake/stop : {CONFIG.wake.model} / {CONFIG.wake.stop_word}")
    print(f"  memory    : {CONFIG.memory.db_path}")
    print()
    print("Commands:  python -m jesse run     (live loop)")
    print("           python -m jesse smoke   (live end-to-end check, ~1-3 min)")
    print("           python -m jesse run --ui  (adds the text UI at 127.0.0.1:8765)")
    print("           python -m jesse run --remote  (adds the phone client, port 8766)")
    print("           python spike.py        (prove the hardware)")
    return 0


def _flag_value(argv: list[str], flag: str) -> str | None:
    if flag in argv:
        i = argv.index(flag)
        if i + 1 < len(argv):
            return argv[i + 1]
    return None


def _device_arg(argv: list[str]) -> int | None:
    v = _flag_value(argv, "--device")
    return int(v) if v is not None else None


def _effective_device(device: int | None) -> int | None:
    return device if device is not None else CONFIG.audio.input_device


def _calibrate_cmd(argv: list[str]) -> int:
    """Standalone: measure the mic and print config values to paste in."""
    from jesse.audio.calibrate import calibrate
    from jesse.audio.devices import DeviceError, validate_input_device

    device = _effective_device(_device_arg(argv))
    try:
        validate_input_device(device)
        result = calibrate(device)
    except DeviceError as e:
        print(f"\n{e}")
        return 1
    print("\n  Put these in jesse/config.py -> AudioConfig to make them permanent:")
    print(f"      capture_gain: float = {result.gain}")
    print(f"      vad_threshold: float = {result.threshold}")
    if not result.ok:
        print("  (calibration was not confident — see the message above)")
    return 0


def _say_cmd(argv: list[str]) -> int:
    """Synthesize text to a wav with a chosen voice — for A/B-ing voices offline.

        python -m jesse say "hello there" --voice en_US-amy-medium
    """
    import wave

    from jesse.tts.piper import PiperSynthesizer, _voice_model_path

    voice = _flag_value(argv, "--voice") or CONFIG.speech.piper_voice
    words, skip = [], False
    for a in argv:
        if a == "--voice":
            skip = True
            continue
        if skip:
            skip = False
            continue
        words.append(a)
    text = " ".join(words) or "Hey you, good to hear your voice. What's going on today?"

    if not _voice_model_path(voice).is_file():
        print(f"Voice '{voice}' isn't downloaded yet. Get it with:")
        print(f"  python -m piper.download_voices {voice} --download-dir models")
        return 1

    synth = PiperSynthesizer(voice=voice)
    pcm = bytearray()
    for chunk in synth.synthesize(text):
        pcm.extend(chunk)
    out = f"say_{voice}.wav"
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(synth.sample_rate)
        w.writeframes(bytes(pcm))
    print(f"Wrote {out} ({len(pcm)/2/synth.sample_rate:.1f}s @ {synth.sample_rate}Hz)")
    print(f"Play it:  Start-Process .\\{out}")
    return 0


def _memory_cmd(argv: list[str]) -> int:
    """Inspect what Jesse has stored (trust/debug).

        python -m jesse memory                      # list every stored fact
        python -m jesse memory "my sister"          # semantic recall for a query
        python -m jesse memory --forget "Pune"      # delete facts matching that text
        python -m jesse memory --dedupe             # PREVIEW near-duplicate merges
        python -m jesse memory --dedupe --apply     # actually merge them
    """
    from jesse.memory.embedder import FastEmbedEmbedder
    from jesse.memory.store import SqliteMemoryStore

    db = CONFIG.memory.db_path
    if not db.exists():
        print(f"No memory yet at {db}.")
        print("Talk to Jesse with `python -m jesse run --ollama` first so he can store things.")
        return 0

    store = SqliteMemoryStore(db, FastEmbedEmbedder(),
                              log_path=db.parent / "memory-log.txt")
    if "--dedupe" in argv:
        groups = store.duplicate_groups()
        if not groups:
            print("No near-duplicate subjects found — nothing to merge.")
            store.close()
            return 0
        total = sum(len(d) for _k, d in groups)
        apply = "--apply" in argv or "--confirm" in argv
        print(f"{'MERGING' if apply else 'WOULD MERGE'} {total} fact(s) into "
              f"{len(groups)} kept fact(s):\n")
        for keeper, dups in groups:
            print(f"  KEEP    ({keeper.origin}) [{keeper.subject}] {keeper.text}")
            for dup, similarity in dups:
                print(f"  MERGE   ({dup.origin}) [{dup.subject}] {dup.text}")
                print(f"          ^ subject similarity {similarity:.3f} "
                      f"(threshold {CONFIG.memory.dedupe_subject_similarity})")
            print()
        if not apply:
            print("Preview only — nothing was changed.")
            print("Re-run with --apply to merge:  python -m jesse memory --dedupe --apply")
        else:
            removed = store.merge_duplicates()
            print(f"Merged away {removed} fact(s). Logged to memory-log.txt.")
        store.close()
        return 0

    if "--forget" in argv:
        needle = " ".join(argv[argv.index("--forget") + 1:]).strip()
        if not needle:
            print('Usage: python -m jesse memory --forget "some text or subject"')
            store.close()
            return 1
        gone = store.forget(needle)
        if not gone:
            print(f"Nothing matched {needle!r} — nothing deleted.")
        else:
            print(f"Forgot {len(gone)} fact(s) matching {needle!r}:")
            for f in gone:
                print(f"  - ({f.origin}) [{f.subject}] {f.text}")
            if any(f.origin in ("core", "self", "self_history") for f in gone):
                print("\nNote: some were seeded facts — `python -m jesse seed` restores those.")
        store.close()
        return 0

    query = " ".join(a for a in argv if not a.startswith("--")).strip()
    if query:
        hits = store.recall(query, k=CONFIG.memory.recall_k)
        print(f"Recall for {query!r} (top {CONFIG.memory.recall_k}):")
        for f in hits:
            print(f"  - ({f.origin}) [{f.subject}] {f.text}  (conf {f.confidence})")
        if not hits:
            print("  (nothing relevant found)")
    else:
        facts = store.all_facts()
        print(f"{len(facts)} fact(s) stored in {db}:")
        for f in facts:
            print(f"  - ({f.origin}) [{f.subject}] {f.text}  (conf {f.confidence})")
    print(f"\nMemory log: {db.parent / 'memory-log.txt'}")
    store.close()
    return 0


def _seed_cmd(argv: list[str]) -> int:
    """(Re)apply the seed facts from jesse/memory/seed.py — core identity/relationship +
    self build facts. Idempotent; run it once, or again after editing seed.py."""
    from jesse.memory.embedder import FastEmbedEmbedder
    from jesse.memory.seed import seed
    from jesse.memory.store import SqliteMemoryStore

    CONFIG.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
    store = SqliteMemoryStore(
        CONFIG.memory.db_path, FastEmbedEmbedder(),
        log_path=CONFIG.memory.db_path.parent / "memory-log.txt",
    )
    n = seed(store)
    print(f"Seeded/updated {n} core + self facts into {CONFIG.memory.db_path}.")
    store.close()
    return 0


def _learn_cmd(argv: list[str]) -> int:
    """Give his something to read, and see what he has read.

        python -m jesse learn guitar ./docs/guitar.md   # ingest a file or a folder
        python -m jesse learn --list                    # what he has learned
        python -m jesse learn --forget guitar           # drop that corpus entirely
        python -m jesse learn --ask "how do I tune it"  # what he'd retrieve, and how close
    """
    from jesse.memory.corpus import CorpusStore
    from jesse.memory.embedder import FastEmbedEmbedder

    CONFIG.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
    store = CorpusStore(CONFIG.memory.db_path, FastEmbedEmbedder())

    def summary() -> None:
        corpora = store.corpora()
        if not corpora:
            print("He hasn't read anything yet.")
            return
        print("He has read:")
        for name, chunks, sources in corpora:
            print(f"  {name:20} {chunks:4} passage(s) from {sources} file(s)")

    if "--list" in argv or not argv:
        summary()
        store.close()
        return 0

    if "--forget" in argv:
        rest = argv[argv.index("--forget") + 1:]
        if not rest:
            print("Which one? e.g. python -m jesse learn --forget guitar")
            store.close()
            return 2
        gone = store.forget(rest[0])
        print(f"Forgot {gone} passage(s) from {rest[0]!r}." if gone
              else f"Nothing stored under {rest[0]!r}.")
        store.close()
        return 0

    if "--ask" in argv:
        rest = argv[argv.index("--ask") + 1:]
        if not rest:
            print('Ask what? e.g. python -m jesse learn --ask "how do I tune it"')
            store.close()
            return 2
        from jesse.memory.corpus import subjects_mentioned
        question = rest[0]
        named = subjects_mentioned(question, store.names())
        if named:
            print(f"Subject named: {', '.join(named)}. He answers from it.\n")
        else:
            maybe = store.keyword_subjects(question)
            hits = store.search(question, k=1,
                                max_distance=CONFIG.knowledge.max_distance,
                                corpora=maybe) if maybe else []
            if hits:
                print(f"No subject named, but the words brush {hits[0].corpus!r} "
                      f"(closest {hits[0].distance:.3f}).")
                print(f"He would ASK — \"Are you asking about your "
                      f"{hits[0].corpus}?\" — and answer after a yes.\n")
            else:
                print(f"No subject named and no trigger word — he treats this as "
                      f"ordinary talk.")
                print(f"He knows these subjects: "
                      f"{', '.join(store.names()) or '(none)'}.\n")
        # No distance gate on the printout: this is the tool for CHOOSING the gate, so
        # it has to show the near misses too. `USED` reflects BOTH gates, so what it
        # prints is what he would actually get.
        for p in store.search(question, k=CONFIG.knowledge.top_k):
            used = p.distance <= CONFIG.knowledge.max_distance and p.corpus in named
            print(f"  [{'USED ' if used else 'below'} d={p.distance:.3f}] "
                  f"({p.corpus}/{p.source}) {p.text[:160]}...")
        print(f"\nGate is {CONFIG.knowledge.max_distance} (config.knowledge.max_distance), "
              "and it only applies inside a subject he named.")
        store.close()
        return 0

    if len(argv) < 2:
        print("Usage: python -m jesse learn <name> <file-or-folder>")
        store.close()
        return 2
    name, path = argv[0], Path(argv[1])
    if not path.exists():
        print(f"No such file or folder: {path}")
        store.close()
        return 2
    stored = store.ingest(name, path, chunk_chars=CONFIG.knowledge.chunk_chars)
    if not stored:
        print(f"Nothing readable in {path} (looking for .txt, .md, .markdown, .rst).")
    else:
        print(f"Read {stored} passage(s) into {name!r}.")
    summary()
    store.close()
    return 0


def _digest_cmd(argv: list[str]) -> int:
    """What he has read from his sources, and a manual fetch.

        python -m jesse digest                  # what's stored, and what's unheard
        python -m jesse digest --fetch          # read the sources right now
        python -m jesse digest --forget bbc     # drop everything from one source
    """
    from jesse.digest.feeds import FeedError, fetch_feed
    from jesse.digest.store import DigestStore

    CONFIG.memory.db_path.parent.mkdir(parents=True, exist_ok=True)
    store = DigestStore(CONFIG.memory.db_path)

    if "--forget" in argv:
        rest = argv[argv.index("--forget") + 1:]
        if not rest:
            print("Which source? e.g. python -m jesse digest --forget bbc")
            store.close()
            return 2
        print(f"Dropped {store.forget_source(rest[0])} item(s) from {rest[0]!r}.")
        store.close()
        return 0

    if "--fetch" in argv:
        if not CONFIG.digest.enabled:
            print("Reading sources is OFF (config.digest.enabled = False).")
            print("Fetching once anyway, since you asked for it explicitly.\n")
        for name, url in CONFIG.digest.sources:
            try:
                items = fetch_feed(url, name, timeout=CONFIG.digest.fetch_timeout,
                                   max_bytes=CONFIG.digest.max_bytes,
                                   limit=CONFIG.digest.items_per_source)
            except FeedError as e:
                print(f"  {name:14} FAILED — {e}")
                continue
            print(f"  {name:14} {len(items)} item(s), {store.add(items)} new")
        store.set_last_fetch(datetime.now())
        print()

    sources = store.sources()
    if not sources:
        print(f"He hasn't read anything yet. Sources configured: "
              f"{', '.join(n for n, _ in CONFIG.digest.sources) or '(none)'}.")
        print("Read them now with:  python -m jesse digest --fetch")
        store.close()
        return 0
    print("He has read:")
    for name, total, untold in sources:
        print(f"  {name:14} {total:4} item(s), {untold} not yet mentioned to you")
    latest = store.last_fetch()
    print(f"\nLast checked: {latest:%A %d %B, %H:%M}" if latest else "\nNever checked.")
    for item in store.recent(limit=5):
        mark = " " if item.told else "*"
        print(f"  {mark} ({item.source}) {item.title[:70]}")
    print('\n"*" = not yet mentioned. Ask him "anything new?" and he\'ll tell you.')
    store.close()
    return 0


def _pair_cmd(argv: list[str]) -> int:
    """Print the phone link and its QR again, without restarting him.

        python -m jesse pair          # link + QR
        python -m jesse pair --url    # just the URL, for copying somewhere

    Exists because the link scrolls off a busy terminal, and re-typing the token by
    hand is the thing that produced "bad token" in the first place.
    """
    from jesse.remote.auth import load_or_create
    from jesse.remote.tls import qr_lines, tailscale_identity

    token = load_or_create(CONFIG.remote.token_path)
    names, ips = tailscale_identity()
    scheme = "https" if CONFIG.remote.tls else "http"
    # --ip skips MagicDNS entirely. Worth having permanently, not just for one
    # diagnosis: the name only resolves while Tailscale is actually connected on the
    # phone, so a dropped tunnel shows up as ERR_NAME_NOT_RESOLVED, which reads like a
    # dead server rather than a disconnected VPN. The IP is in the certificate too.
    host = names[0]
    if "--ip" in argv:
        tailnet_ips = [i for i in ips if i.startswith("100.") ]
        if not tailnet_ips:
            print("  No tailnet IP found — is Tailscale running on this machine?")
            return 1
        host = tailnet_ips[0]
    url = f"{scheme}://{host}:{CONFIG.remote.port}/?t={token}"
    if "--url" in argv:
        print(url)
        return 0
    print(f"\n  {url}\n")
    rows = qr_lines(url)
    if rows:
        for row in rows:
            print("   " + row)
        print("\n  scan that with your phone's camera — nothing to type.")
    else:
        print("  (install segno for a scannable QR: pip install segno)")
    print("  Jesse must be running with --remote for the link to answer.")
    return 0


def _run(argv: list[str]) -> int:
    from jesse.audio.calibrate import calibrate
    from jesse.audio.devices import DeviceError

    from jesse.factory import build_orchestrator

    # Two copies competing for one microphone is not a crash — the second simply hears
    # less, which presents as calibration failing twice for no visible reason. Name the
    # cause rather than leaving the symptom.
    from jesse.core.single_instance import claim, release
    pid_file = CONFIG.memory.db_path.parent / "jesse.pid"
    clash = claim(pid_file)
    if clash:
        print(f"\n  {clash}\n")

    device = _device_arg(argv)
    channel = url = None
    if "--ui" in argv:
        from jesse.ui.channel import TextChannel
        from jesse.ui.server import start as start_ui
        channel = TextChannel()
        url = start_ui(channel, port=int(_flag_value(argv, "--port") or 8765))
    orch, voice_label, brain_label = build_orchestrator(
        use_ollama="--ollama" in argv, input_device=device, text_channel=channel,
    )

    # Remote: his phone, over his own tailnet. Wraps the desk transport rather than
    # replacing it, so the orchestrator above is untouched (see remote/transport.py).
    remote_url = None
    if "--remote" in argv:
        from jesse.remote.auth import RemoteAuth, load_or_create
        from jesse.remote.server import start as start_remote
        from jesse.remote.transport import RemoteSource, SwitchingTransport

        if channel is None:            # the phone shows the same transcript the UI does
            from jesse.ui.channel import TextChannel
            channel = TextChannel()
            orch.text_channel = channel
        token = load_or_create(CONFIG.remote.token_path)
        source = RemoteSource(idle_timeout=CONFIG.remote.idle_timeout_seconds)
        orch.transport = SwitchingTransport(orch.transport, source)
        remote_port = int(_flag_value(argv, "--remote-port") or CONFIG.remote.port)
        tls = fingerprint_line = None
        if CONFIG.remote.tls:
            from jesse.remote.tls import CertError, ensure_cert, fingerprint,                 tailscale_identity
            try:
                names, ips = tailscale_identity()
                tls = ensure_cert(CONFIG.remote.cert_dir, names, ips)
                fingerprint_line = fingerprint(tls[0])
                remote_host = names[0]
            except CertError as e:
                print(f"\n  [remote] no certificate: {e}")
                print("  [remote] serving plain http — your phone will refuse the mic.")
                remote_host = "<this machine>"
        else:
            remote_host = "<this machine>"
        start_remote(RemoteAuth(token), source, channel,
                     host=CONFIG.remote.host, port=remote_port, tls=tls)
        scheme = "https" if tls else "http"
        remote_url = f"{scheme}://{remote_host}:{remote_port}/?t={token}"

    # Gain / threshold: an explicit --gain wins; else auto-calibrate (unless off).
    gain_override = _flag_value(argv, "--gain")
    eff_device = _effective_device(device)
    try:
        if gain_override is not None:
            orch.transport.gain = float(gain_override)
        elif CONFIG.audio.auto_calibrate and "--no-calibrate" not in argv:
            result = calibrate(eff_device)
            if result.ok:
                orch.transport.gain = result.gain
                orch.vad.set_threshold(result.threshold)
    except DeviceError as e:
        print(f"\nAudio device problem:\n{e}")
        return 1

    dev_label = f"index {eff_device}" if eff_device is not None else (
        "OS default (run `python diagnose.py` to pick your mic)"
    )
    print("=" * 60)
    print(" Jesse — walking skeleton (Phase 0)")
    print(f"   brain : {brain_label}")
    print(f"   voice : {voice_label}")
    print(f"   mic   : {dev_label}  (gain x{orch.transport.gain:.1f})")
    print(f"   VAD   : speech > {orch.vad.threshold:.0f} RMS, endpoint after {CONFIG.audio.vad_silence_ms}ms "
          f"silence (min speech {CONFIG.audio.vad_min_speech_ms}ms), pre-roll {CONFIG.audio.preroll_ms}ms")
    print("   wake  : say 'hey jarvis'  (placeholder — 'yo Jesse' gets trained later)")
    print("   stop  : say the stop word while Jesse is speaking to cut him off")
    if url:
        print(f"   ui    : {url}  (type there; it joins the same conversation)")
    if remote_url:
        print(f"   phone : {remote_url}")
        print("           open that on your phone with Tailscale connected.")
        if fingerprint_line:
            print("           Jesse signed his own certificate, so the browser will warn")
            print("           you once. Check this matches what it shows, then accept:")
            print(f"             SHA-256 {fingerprint_line}")
        print(f"           token lives in {CONFIG.remote.token_path}")
        # Scan rather than type. The 43-character token is exactly what went wrong
        # last time — told only "bad token", he re-typed a string that was fine while
        # the link had simply lost its ?t= tail.
        from jesse.remote.tls import qr_lines
        rows = qr_lines(remote_url)
        if rows:
            print()
            for row in rows:
                print("   " + row)
            print("   scan that with your phone's camera — nothing to type.")
    print("   Ctrl-C to quit.")
    print("=" * 60)

    try:
        asyncio.run(orch.run())
    except KeyboardInterrupt:
        print("\nGoodbye.")
    except DeviceError as e:
        print(f"\nAudio device problem:\n{e}")
        return 1
    finally:
        release(pid_file)
    return 0


def _wrong_python(missing: str) -> int:
    """A missing dependency here almost always means the system Python was used
    instead of the project venv, which is an easy copy-paste mistake and a very
    confusing traceback. Say so plainly."""
    print(f"Missing dependency: {missing}")
    print()
    print(f"This is running:  {sys.executable}")
    print("which is probably NOT the project venv.")
    print()
    print("Use the venv Python:")
    print(r"    .\.venv\Scripts\python.exe -m jesse " + " ".join(sys.argv[1:] or ["run"]))
    print()
    print("Or, if the venv itself is incomplete:")
    print(r"    .\.venv\Scripts\python.exe -m pip install -r requirements.txt")
    return 1


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if "--spike" in argv:
        import spike
        return spike.main()
    if argv and argv[0] in ("devices", "--list-devices"):
        import diagnose
        diagnose.list_devices()
        return 0
    if argv and argv[0] == "calibrate":
        return _calibrate_cmd(argv[1:])
    if argv and argv[0] == "say":
        return _say_cmd(argv[1:])
    if argv and argv[0] == "memory":
        return _memory_cmd(argv[1:])
    if argv and argv[0] == "seed":
        return _seed_cmd(argv[1:])
    if argv and argv[0] == "learn":
        return _learn_cmd(argv[1:])
    if argv and argv[0] == "pair":
        return _pair_cmd(argv[1:])
    if argv and argv[0] == "digest":
        return _digest_cmd(argv[1:])
    if argv and argv[0] == "smoke":
        try:
            import jesse.smoke
        except ImportError as e:
            return _wrong_python(e.name or str(e))
        return jesse.smoke.main(argv[1:])
    if argv and argv[0] == "run":
        try:
            return _run(argv[1:])
        except ImportError as e:
            return _wrong_python(e.name or str(e))
    return _status()


if __name__ == "__main__":
    raise SystemExit(main())
