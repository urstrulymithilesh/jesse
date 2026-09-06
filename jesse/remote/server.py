"""The HTTP surface his phone talks to. Stdlib `http.server`, same as the desk UI.

Four endpoints and nothing else:

    GET  /                  the page
    GET  /remote/state      transcript since N, plus whether a reply is waiting
    GET  /remote/reply      his next reply as raw PCM (X-Sample-Rate header)
    POST /remote/audio      a chunk of 16 kHz mono Int16 from the phone's mic

Every one of them requires the token. The bind address is deliberate: `127.0.0.1` is
useless to a phone, so remote listening binds wider — and that is exactly why the
token exists rather than being optional.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from jesse.remote.auth import RemoteAuth, token_from_request
from jesse.remote.page import PAGE

MAX_CHUNK_BYTES = 2_000_000        # ~60s of 16 kHz Int16; a chunk is normally ~8 KB


def _handler(auth: RemoteAuth, source, channel):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args):
            pass                        # the terminal belongs to the conversation

        # -- plumbing ------------------------------------------------------

        @property
        def _who(self) -> str:
            return self.client_address[0] if self.client_address else "?"

        def _send(self, code, body: bytes, content_type, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            # The page is only ever loaded from this machine over the tunnel; there is
            # no reason for another origin to be able to read any of it.
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("X-Content-Type-Options", "nosniff")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode("utf-8"), "application/json")

        # One message per cause. "bad token" sent his re-typing a 43-character string
        # when the URL had simply lost its ?t= tail.
        _WHY = {
            "missing": "No token in the request. Open the full link Jesse printed at "
                       "startup — the part after ?t= is what lets you in.",
            "wrong": "That token is not the one on this machine. Use the link Jesse "
                     "printed at startup, or scan his QR code.",
            "locked": "Too many wrong tokens from this device. Wait five minutes and "
                      "use the link Jesse printed at startup.",
        }

        def _authed(self) -> bool:
            why = auth.verdict(token_from_request(self.headers, self.path), self._who)
            if why == "ok":
                return True
            self._json(401, {"error": why, "detail": self._WHY[why]})
            return False

        # -- routes --------------------------------------------------------

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/remote/state":
                if not self._authed():
                    return
                since = 0
                if "since=" in self.path:
                    try:
                        since = int(self.path.split("since=")[1].split("&")[0])
                    except ValueError:
                        since = 0
                snap = channel.snapshot(since)
                snap["reply"] = source.pending_replies() > 0
                self._json(200, snap)
                return
            if path == "/remote/reply":
                if not self._authed():
                    return
                reply = source.take_reply()
                if reply is None:
                    self._send(204, b"", "application/octet-stream")
                    return
                pcm, rate = reply
                self._send(200, pcm, "application/octet-stream",
                           {"X-Sample-Rate": str(rate)})
                return
            # The page itself. Served on any other path so a bare host name works;
            # the token may ride in the query string on this one request only.
            why = auth.verdict(token_from_request(self.headers, self.path), self._who)
            if why != "ok":
                body = (
                    "<!doctype html><meta charset=utf-8>"
                    "<meta name=viewport content='width=device-width,initial-scale=1'>"
                    "<body style=\"background:#000;color:#fff;font:16px/1.6 "
                    "ui-monospace,monospace;padding:2rem\">"
                    f"<p style='color:#b06a8f'>{self._WHY[why]}</p>"
                    "<p style='color:#555'>Jesse is running and reachable — this is only "
                    "about the token.</p>").encode("utf-8")
                self._send(401, body, "text/html; charset=utf-8")
                return
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")

        def do_POST(self):
            if self.path.split("?", 1)[0] != "/remote/audio":
                self._json(404, {"error": "no"})
                return
            if not self._authed():
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_CHUNK_BYTES:
                self._json(413, {"error": "chunk too large"})
                return
            source.submit(self.rfile.read(length) if length else b"")
            self._json(200, {"ok": True})

    return Handler


class _QuietServer(ThreadingHTTPServer):
    """Keep-alive plus a phone on a moving connection means dropped sockets, often.

    The default handler prints a traceback for each one. That is noise on a terminal
    that belongs to the conversation, and worse, it makes an ordinary hang-up look
    like a fault. Genuine errors still surface.
    """

    daemon_threads = True

    def handle_error(self, request, client_address):
        import sys
        kind = sys.exc_info()[0]
        if kind is not None and issubclass(kind, (ConnectionResetError,
                                                  ConnectionAbortedError,
                                                  BrokenPipeError,
                                                  TimeoutError)):
            return
        super().handle_error(request, client_address)


def start(auth: RemoteAuth, source, channel, *, host: str = "0.0.0.0",
          port: int = 8766, tls: tuple | None = None) -> str:
    """Start the remote server in a daemon thread. Returns the base URL.

    `tls` is (cert, key). Without it the page still loads, but the browser will not
    give it the microphone — so the caller is expected to supply one and the page says
    so on screen if it finds itself insecure.
    """
    server = _QuietServer((host, port), _handler(auth, source, channel))
    scheme = "http"
    if tls is not None:
        from jesse.remote.tls import wrap
        server.socket = wrap(server.socket, *tls)
        scheme = "https"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"{scheme}://{host}:{port}"
