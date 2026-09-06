"""The local web UI server. Stdlib http.server in a daemon thread — no new dependency,
no websockets, no build step. The page polls; at one request every 400ms against
localhost that is cheaper than the machinery a socket would need.

Binds to 127.0.0.1 only. Nothing about this project should be reachable off the box.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jesse.ui.channel import TextChannel

from jesse.ui.page import PAGE


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
