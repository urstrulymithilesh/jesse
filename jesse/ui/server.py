"""A minimal local web UI. Stdlib http.server in a daemon thread — no new dependency,
no websockets, no build step. The page polls; at one request every 400ms against
localhost that is cheaper than the machinery a socket would need.

Binds to 127.0.0.1 only. Nothing about this project should be reachable off the box.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jesse.ui.channel import TextChannel

PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Jesse</title>
<style>
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#000; color:#fff; font:16px/1.6 ui-monospace,Menlo,Consolas,monospace;
         display:flex; flex-direction:column; height:100vh; }
  header { padding:14px 20px; border-bottom:1px solid #1c1c1c; display:flex;
           align-items:center; gap:12px; font-size:13px; letter-spacing:.14em;
           text-transform:uppercase; color:#666; }
  #dot { width:9px; height:9px; border-radius:50%; background:#333; flex:none; }
  #dot.on { background:#fff; animation:pulse 1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:.25;transform:scale(.8)} 50%{opacity:1;transform:scale(1.35)} }
  #log { flex:1; overflow-y:auto; padding:24px 20px; }
  .line { margin:0 0 14px; max-width:70ch; white-space:pre-wrap; word-wrap:break-word; }
  .who { color:#555; margin-right:10px; }
  .you .who { color:#4a7dbd; }
  .isha .who { color:#b06a8f; }
  .via { color:#333; font-size:11px; margin-left:8px; }
  form { display:flex; border-top:1px solid #1c1c1c; }
  input { flex:1; background:#000; color:#fff; border:0; padding:16px 20px;
          font:inherit; outline:none; }
  input::placeholder { color:#333; }
  button { background:#000; color:#666; border:0; border-left:1px solid #1c1c1c;
           padding:0 22px; font:inherit; cursor:pointer; }
  button:hover { color:#fff; }
  #empty { color:#333; }
</style></head><body>
<header><span id="dot"></span><span>isha</span><span id="mode" style="margin-left:auto"></span></header>
<div id="log"><div id="empty" class="line">nothing yet. type below, or just talk.</div></div>
<form id="f" autocomplete="off">
  <input id="i" placeholder="say something…" autofocus>
  <button>send</button>
</form>
<script>
let seen = 0;
const log = document.getElementById('log'), dot = document.getElementById('dot'),
      mode = document.getElementById('mode'), empty = document.getElementById('empty');

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
  if (empty) empty.remove();
  const d = document.createElement('div');
  d.className = 'line ' + (l.role === 'you' ? 'you' : 'jesse');
  const who = document.createElement('span');
  who.className = 'who'; who.textContent = l.role === 'you' ? 'you' : 'jesse';
  d.appendChild(who);
  d.appendChild(document.createTextNode(l.text));
  if (l.via === 'voice') {
    const v = document.createElement('span'); v.className = 'via'; v.textContent = '(voice)';
    d.appendChild(v);
  }
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
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


def _handler(channel: TextChannel):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass                      # the terminal belongs to the conversation

        def _send(self, code, body, content_type):
            payload = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path.startswith("/events"):
                since = 0
                if "since=" in self.path:
                    try:
                        since = int(self.path.split("since=")[1].split("&")[0])
                    except ValueError:
                        since = 0
                self._send(200, json.dumps(channel.snapshot(since)), "application/json")
            else:
                self._send(200, PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            if not self.path.startswith("/send"):
                self._send(404, "{}", "application/json")
                return
            length = int(self.headers.get("Content-Length") or 0)
            try:
                data = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                data = {}
            channel.submit(str(data.get("text", "")))
            self._send(200, '{"ok":true}', "application/json")

    return Handler


def start(channel: TextChannel, *, port: int = 8765) -> str:
    """Start the UI in a daemon thread. Returns the URL."""
    server = ThreadingHTTPServer(("127.0.0.1", port), _handler(channel))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{port}"
