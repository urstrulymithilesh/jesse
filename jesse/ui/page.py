"""The desk UI. Chat bubbles, one accent colour, restrained motion.

Still stdlib `http.server` behind it, still polling, still no framework, no build step
and no new dependency — the upgrade is in the design, not the machinery. Everything
here is one file of hand-written CSS and about a hundred lines of vanilla JS, which is
the same weight class as the black-and-white page it replaces.

Design notes, since they are decisions rather than taste:

* **Charcoal, not black.** Pure black against white text vibrates; #0e0f12 with
  #e8e9ed reads as deliberate and is easier to look at for a long session.
* **One accent, used sparingly.** Amber (#f0a850) for Jesse and for the speaking
  state; a muted slate for Mithilesh's own messages. Two colours doing all the work
  keeps the transcript legible at a glance, which is the thing this page is actually
  for — checking what Whisper heard.
* **Motion is feedback, never decoration.** Messages fade up 6px on arrival so a new
  one is noticeable without being loud. The speaking indicator breathes. Nothing else
  moves, and everything respects prefers-reduced-motion.
* **The transcript is the page.** It gets all the space; the composer is a thin bar.
  A waveform centrepiece was considered and rejected for exactly this reason — the
  transcript is what gets read constantly.
"""

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Jesse</title>
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>
  :root {
    --bg:#0e0f12; --panel:#15171c; --line:#22252c;
    --text:#e8e9ed; --dim:#7c8190; --faint:#4a4f5c;
    --accent:#f0a850; --accent-dim:#8a6132; --you:#2b3038;
    --radius:14px;
    color-scheme: dark;
  }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
  html,body { height:100%; }
  body {
    margin:0; background:var(--bg); color:var(--text);
    font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Inter,system-ui,sans-serif;
    display:flex; flex-direction:column; height:100dvh;
  }

  header {
    display:flex; align-items:center; gap:11px;
    padding:15px 22px; border-bottom:1px solid var(--line);
    background:var(--panel);
  }
  #dot { width:8px; height:8px; border-radius:50%; background:var(--faint); flex:none;
         transition:background .25s ease, box-shadow .25s ease; }
  #dot.on { background:var(--accent); box-shadow:0 0 0 4px rgba(240,168,80,.16);
            animation:breathe 1.6s ease-in-out infinite; }
  @keyframes breathe {
    0%,100% { box-shadow:0 0 0 3px rgba(240,168,80,.10); }
    50%     { box-shadow:0 0 0 7px rgba(240,168,80,.22); }
  }
  .name { font-weight:600; letter-spacing:-.01em; }
  #mode { margin-left:auto; font-size:12.5px; color:var(--dim); }

  #log { flex:1; overflow-y:auto; padding:26px 22px 8px;
         display:flex; flex-direction:column; gap:11px;
         scrollbar-width:thin; scrollbar-color:var(--line) transparent; }
  #log::-webkit-scrollbar { width:8px; }
  #log::-webkit-scrollbar-thumb { background:var(--line); border-radius:4px; }

  .row { display:flex; }
  .row.you { justify-content:flex-end; }

  .bubble {
    max-width:min(74%, 62ch); padding:9px 14px; border-radius:var(--radius);
    white-space:pre-wrap; overflow-wrap:anywhere;
    animation:rise .22s cubic-bezier(.2,.7,.3,1) both;
  }
  @keyframes rise { from { opacity:0; transform:translateY(6px); } }
  @media (prefers-reduced-motion:reduce) {
    .bubble { animation:none; }
    #dot.on { animation:none; }
  }

  .jesse .bubble {
    background:var(--panel); border:1px solid var(--line);
    border-left:2px solid var(--accent); border-top-left-radius:5px;
  }
  .you .bubble { background:var(--you); border-top-right-radius:5px; }

  .meta { font-size:11px; color:var(--faint); margin:0 4px 3px;
          letter-spacing:.03em; }
  .row.you .meta { text-align:right; }

  #empty { color:var(--faint); text-align:center; margin:auto 0; font-size:14px; }

  form { display:flex; gap:9px; padding:13px 22px calc(13px + env(safe-area-inset-bottom));
         border-top:1px solid var(--line); background:var(--panel); }
  input {
    flex:1; min-width:0; background:var(--bg); color:var(--text);
    border:1px solid var(--line); border-radius:11px; padding:11px 14px;
    font:inherit; outline:none; transition:border-color .18s ease;
  }
  input:focus { border-color:var(--accent-dim); }
  input::placeholder { color:var(--faint); }
  button {
    background:var(--accent); color:#1a1206; border:0; border-radius:11px;
    padding:0 19px; font:inherit; font-weight:600; cursor:pointer;
    transition:filter .18s ease;
  }
  button:hover { filter:brightness(1.08); }
  button:active { filter:brightness(.94); }
</style></head><body>

<header>
  <span id="dot"></span>
  <span class="name">Jesse</span>
  <span id="mode"></span>
</header>

<div id="log"><div id="empty">nothing yet — say something, or just talk.</div></div>

<form id="f" autocomplete="off">
  <input id="i" placeholder="type to Jesse…" autofocus>
  <button>send</button>
</form>

<script>
let seen = 0;
const log = document.getElementById('log'), dot = document.getElementById('dot'),
      mode = document.getElementById('mode');

document.getElementById('f').onsubmit = async e => {
  e.preventDefault();
  const i = document.getElementById('i'), text = i.value.trim();
  if (!text) return;
  i.value = '';
  await fetch('/send', {method:'POST', headers:{'Content-Type':'application/json'},
                        body: JSON.stringify({text})});
  poll();
};

function add(l) {
  const empty = document.getElementById('empty');
  if (empty) empty.remove();
  const mine = l.role === 'you';
  const row = document.createElement('div');
  row.className = 'row ' + (mine ? 'you' : 'jesse');

  const wrap = document.createElement('div');
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = (mine ? 'you' : 'jesse') + (l.via === 'voice' ? ' · voice' : '');
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = l.text;

  wrap.appendChild(meta);
  wrap.appendChild(bubble);
  row.appendChild(wrap);
  log.appendChild(row);

  // Only follow the transcript if he is already at the bottom, so scrolling back to
  // read something is not yanked away the moment a new line lands.
  const nearBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 120;
  if (nearBottom) log.scrollTop = log.scrollHeight;
}

async function poll() {
  try {
    const r = await fetch('/events?since=' + seen);
    const d = await r.json();
    (d.lines || []).forEach(add);
    seen = d.total;
    dot.className = d.speaking ? 'on' : '';
    mode.textContent = d.speaking ? 'speaking' : '';
  } catch (e) { /* server not up yet */ }
}
setInterval(poll, 400);
poll();
</script></body></html>
"""
