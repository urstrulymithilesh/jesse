"""The page his phone loads. Vanilla JS, no build step, no framework, no websockets.

Continuous listening without a socket: the mic feeds an AudioWorklet that downsamples
to the 16 kHz mono Int16 the pipeline already speaks, and a chunk is POSTed roughly
four times a second. Raw PCM rather than MediaRecorder's WebM/Opus on purpose —
decoding Opus in Python would need a codec dependency, and the pipeline wants exactly
this format anyway.

Half-duplex, same rule as the desk: while his reply is playing the phone stops
uploading, so his own voice cannot come back in and trip the stop-word. On headphones
that is unnecessary, but the page cannot reliably tell, so it always mutes.

The mic needs a secure context. Over Tailscale that means `tailscale serve`, which
issues a real certificate for the machine's MagicDNS name — the page says so plainly
if it finds itself on plain http, because a silent mic failure is the worst outcome.
"""

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Jesse</title>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
  body { margin:0; background:#000; color:#fff; height:100dvh; display:flex;
         flex-direction:column; font:16px/1.6 ui-monospace,Menlo,Consolas,monospace; }
  header { padding:16px 18px; border-bottom:1px solid #1c1c1c; display:flex;
           align-items:center; gap:11px; font-size:12px; letter-spacing:.16em;
           text-transform:uppercase; color:#666; }
  #dot { width:9px; height:9px; border-radius:50%; background:#333; flex:none; }
  #dot.listening { background:#4a7dbd; }
  #dot.speaking { background:#fff; animation:pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:.25;transform:scale(.8)} 50%{opacity:1;transform:scale(1.35)} }
  #log { flex:1; overflow-y:auto; padding:20px 18px; -webkit-overflow-scrolling:touch; }
  .line { margin:0 0 14px; white-space:pre-wrap; word-wrap:break-word; }
  .who { color:#555; margin-right:9px; }
  .you .who { color:#4a7dbd; }
  .isha .who { color:#b06a8f; }
  #empty, #warn { color:#333; }
  #warn { color:#b06a8f; padding:0 18px 14px; font-size:13px; }
  footer { border-top:1px solid #1c1c1c; padding:14px 18px calc(14px + env(safe-area-inset-bottom));
           display:flex; align-items:center; gap:14px; }
  button { background:#000; color:#fff; border:1px solid #333; border-radius:999px;
           padding:13px 26px; font:inherit; cursor:pointer; flex:none; }
  button:disabled { color:#333; border-color:#161616; }
  #status { color:#555; font-size:13px; }
  input { flex:1; background:#000; color:#fff; border:1px solid #333; border-radius:8px;
          padding:12px 14px; font:inherit; outline:none; min-width:0; }
</style></head><body>
<header><span id="dot"></span><span>isha</span><span id="mode" style="margin-left:auto">remote</span></header>
<div id="warn" hidden></div>
<div id="log"><div id="empty" class="line">not listening yet.</div></div>
<footer>
  <button id="go">start listening</button>
  <span id="status">idle</span>
</footer>
<script>
const $ = s => document.querySelector(s);
const RATE = 16000, CHUNK_MS = 250;

// The token arrives once in the address bar, then lives in localStorage and is
// stripped from the URL so it is not sitting in history or a screenshot.
let token = new URLSearchParams(location.search).get('t');
if (token) { localStorage.setItem('isha_token', token);
             history.replaceState({}, '', location.pathname); }
token = token || localStorage.getItem('isha_token') || '';

function headers() { return { 'X-Jesse-Token': token, 'Content-Type': 'application/octet-stream' }; }

if (!window.isSecureContext) {
  const w = $('#warn');
  w.hidden = false;
  w.textContent = 'This page is not on a secure connection, so the browser will not '
    + 'give it the microphone. Restart Jesse with --remote so it serves https with its '
    + 'own certificate, and open the https:// address instead.';
}

let seen = 0, listening = false, speaking = false, ctx = null, node = null, stream = null;
const queue = [];

function paint(lines) {
  const log = $('#log');
  if (lines.length && $('#empty')) $('#empty').remove();
  for (const l of lines) {
    const d = document.createElement('div');
    d.className = 'line ' + (l.role === 'you' ? 'you' : 'jesse');
    d.innerHTML = '<span class="who">' + l.role + '</span>';
    d.appendChild(document.createTextNode(l.text));
    log.appendChild(d);
  }
  if (lines.length) log.scrollTop = log.scrollHeight;
}

// Connection state. A page that silently retries forever looks identical to a page
// where nothing is wrong, so a machine that is asleep, an internet drop or a laptop
// that closed mid-sentence all present as "he just isn't answering". Say it instead.
let offlineSince = 0;

function setDot() {
  const el = $('#dot'), st = $('#status');
  if (offlineSince) {
    el.className = '';
    const secs = Math.round((Date.now() - offlineSince) / 1000);
    st.textContent = "can't reach him — " + (secs < 60 ? secs + 's' : Math.round(secs / 60) + 'm')
      + ' (retrying)';
    st.style.color = '#b06a8f';
    return;
  }
  st.style.color = '';
  el.className = speaking ? 'speaking' : (listening ? 'listening' : '');
  st.textContent = speaking ? 'he is talking' : (listening ? 'listening' : 'idle');
}

function reachable(ok) {
  if (ok) {
    if (offlineSince) { offlineSince = 0; setDot(); }
  } else if (!offlineSince) {
    offlineSince = Date.now();
    setDot();
  } else {
    setDot();                       // keep the elapsed time ticking up
  }
}

// --- his voice back ---------------------------------------------------------
// Played through the same AudioContext so iOS treats it as user-initiated audio.
async function playReply(buf, rate) {
  const pcm = new Int16Array(buf);
  const audio = ctx.createBuffer(1, pcm.length, rate);
  const ch = audio.getChannelData(0);
  for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
  const src = ctx.createBufferSource();
  src.buffer = audio; src.connect(ctx.destination);
  speaking = true; setDot();
  await new Promise(done => { src.onended = done; src.start(); });
  speaking = false; setDot();
}

async function poll() {
  while (true) {
    try {
      const r = await fetch('/remote/state?since=' + seen, { headers: { 'X-Jesse-Token': token } });
      if (r.status === 401) { $('#status').textContent = 'token rejected'; return; }
      const j = await r.json();
      if (j.lines && j.lines.length) { paint(j.lines); seen = j.total; }
      if (j.reply) {
        const a = await fetch('/remote/reply', { headers: { 'X-Jesse-Token': token } });
        if (a.ok) {
          const rate = parseInt(a.headers.get('X-Sample-Rate') || '22050', 10);
          await playReply(await a.arrayBuffer(), rate);
        }
      }
      reachable(true);
    } catch (e) {
      // Machine asleep, home internet down, tunnel dropped, laptop lid closed
      // mid-sentence — all the same from here, and all worth saying out loud.
      reachable(false);
    }
    // Back off while unreachable so a phone in a pocket is not hammering a dead host.
    await new Promise(r => setTimeout(r, offlineSince ? 2000 : (speaking ? 120 : 400)));
  }
}

// --- the mic ----------------------------------------------------------------
const WORKLET = `
class Tap extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; }
  process(inputs) {
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('tap', Tap);`;

function downsample(f32, from, to) {
  if (from === to) return f32;
  const ratio = from / to, out = new Float32Array(Math.floor(f32.length / ratio));
  for (let i = 0; i < out.length; i++) out[i] = f32[Math.floor(i * ratio)];
  return out;
}

async function start() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true } });
  } catch (e) {
    $('#status').textContent = 'no microphone: ' + e.name;
    return;
  }
  ctx = new (window.AudioContext || window.webkitAudioContext)();
  await ctx.resume();
  await ctx.audioWorklet.addModule(
    URL.createObjectURL(new Blob([WORKLET], { type: 'application/javascript' })));
  node = new AudioWorkletNode(ctx, 'tap');
  ctx.createMediaStreamSource(stream).connect(node);
  node.port.onmessage = e => { if (!speaking) queue.push(downsample(e.data, ctx.sampleRate, RATE)); };
  listening = true; setDot();
  $('#go').textContent = 'stop';
  pump();
}

async function pump() {
  while (listening) {
    await new Promise(r => setTimeout(r, CHUNK_MS));
    if (speaking || !queue.length) continue;
    let n = 0; for (const q of queue) n += q.length;
    const pcm = new Int16Array(n);
    let o = 0;
    for (const q of queue) for (let i = 0; i < q.length; i++, o++)
      pcm[o] = Math.max(-1, Math.min(1, q[i])) * 32767;
    queue.length = 0;
    try {
      const r = await fetch('/remote/audio', { method: 'POST', headers: headers(), body: pcm.buffer });
      if (r.status === 401) { $('#status').textContent = 'token rejected'; stop(); return; }
      reachable(true);
    } catch (e) {
      reachable(false);           // he is not hearing this; do not pretend otherwise
    }
  }
}

function stop() {
  listening = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  if (node) node.disconnect();
  $('#go').textContent = 'start listening';
  setDot();
}

$('#go').onclick = () => (listening ? stop() : start());
poll();
</script></body></html>
"""
