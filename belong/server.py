"""Local web server, loopback only (127.0.0.1 and ::1), never on a network interface.

  /                         landing page (belong/web/landing.html)
  /demo                     two-person demo (belong/web/demo.html); /?phone=A still opens it
  /chat                     mockup: one person using Belong. inside a ChatGPT-style chat, through a connector
                            (belong/web/chat.html), on its own demo session
  /trust                    trust page (belong/web/trust.html)
  /inspector                batch-run inspector (belong/web/index.html + out/run.json)
  /api/...                  demo actions; each POST returns the full demo state

Trust API (for people and their agents; see the trust page for examples):
  GET  /api/trust/attestation?nonce=N   measurements + policy (simulated attestation)
  GET  /api/trust/source                list of open-source files with SHA-256
  GET  /api/trust/source/<file>         the file itself (open-source files only)
  GET  /api/trust/receipts?from=N       the public, hash-chained receipt log
  POST /api/probe/canary                plant secrets, run a week, search all outputs
  POST /api/probe/attacks               try to break in and get data out
"""
from __future__ import annotations

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import attest, probe
from .demo import DemoError, DemoSession

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "belong" / "web"
FILES = {
    "/": (WEB / "landing.html", "text/html; charset=utf-8"),
    "/demo": (WEB / "demo.html", "text/html; charset=utf-8"),
    "/chat": (WEB / "chat.html", "text/html; charset=utf-8"),
    "/trust": (WEB / "trust.html", "text/html; charset=utf-8"),
    "/trust/TRUST.md": (ROOT / "TRUST.md", "text/plain; charset=utf-8"),
    "/inspector": (WEB / "index.html", "text/html; charset=utf-8"),
    "/nav.js": (WEB / "nav.js", "text/javascript; charset=utf-8"),
    "/run.json": (ROOT / "out" / "run.json", "application/json"),
}
MAX_BODY = 64 * 1024

SESSION = DemoSession()
# The chat mockup's own session: one person, matched against the synthetic people, so it never
# touches the two phones. Made on first use.
_CHAT = None
CHAT_ACTIONS = {"reset", "verify", "new_listing", "open_listing", "prefill", "agent", "submit", "match", "answer",
                "decide", "set_prefs", "set_defaults", "message", "react", "update_profile"}


def chat_session() -> DemoSession:
    global _CHAT
    if _CHAT is None:
        _CHAT = DemoSession()
    return _CHAT


def chat_state(session: DemoSession) -> dict:
    return {"me": session.phone_state("A"), "agent_status": session.agent_status}

# action -> function(session, body)
ACTIONS = {
    "reset": lambda s, b: s.reset(),
    "verify": lambda s, b: s.verify(b.get("slot"), b.get("name"), b.get("age")),
    "new_listing": lambda s, b: s.new_listing(b.get("slot"), b.get("prefs")),
    "update_profile": lambda s, b: s.update_profile(b.get("slot"), b.get("name")),
    "set_defaults": lambda s, b: s.set_defaults(b.get("slot"), b.get("defaults")),
    "set_prefs": lambda s, b: s.set_prefs(b.get("slot"), b.get("listing_id"), b.get("prefs")),
    "message": lambda s, b: s.send_message(b.get("slot"), b.get("intro_id"), b.get("text")),
    "react": lambda s, b: s.react(b.get("slot"), b.get("intro_id"), b.get("index"), b.get("emoji")),
    "open_listing": lambda s, b: s.open_listing(b.get("slot"), b.get("listing_id")),
    "prefill": lambda s, b: s.prefill(b.get("slot"), b.get("text"), b.get("listing_id")),
    "agent": lambda s, b: s.agent_message(b.get("slot"), b.get("text"), b.get("listing_id")),
    "submit": lambda s, b: s.submit(b.get("slot"), b.get("listing") or {}, b.get("listing_id")),
    "match": lambda s, b: s.run_matchmaker(),
    "answer": lambda s, b: s.answer_followup(b.get("slot"), b.get("question_id"), b.get("answer") or {},
                                             b.get("listing_id")),
    "decide": lambda s, b: s.decide(b.get("slot"), b.get("intro_id"), b.get("yes")),
    "reveal": lambda s, b: s.reveal(b.get("slot"), b.get("intro_id"), b.get("keys") or []),
    "attack": lambda s, b: s.attack(),
}
UNLOCKED = {ACTIONS["agent"]}
PROBES = {"canary": probe.canary_test, "attacks": probe.attack_suite}


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json")

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/api/state":
            with SESSION.lock:
                return self._json(200, SESSION.state())
        if path == "/api/chat/state":
            session = chat_session()
            with session.lock:
                return self._json(200, chat_state(session))
        if path.startswith("/ds/"):
            return self._design_file(path.removeprefix("/ds/"))
        if path.startswith("/api/trust/"):
            return self._trust_get(path.removeprefix("/api/trust/"))
        if path == "/api/options":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with SESSION.lock:
                return self._json(200, SESSION.options("A", (q.get("kind") or [None])[0], (q.get("city") or [None])[0]))
        if path == "/" and "phone=" in self.path:
            path = "/demo"  # old single-phone links (/?phone=A) still open the demo
        route = FILES.get(path)
        if route is None or not route[0].exists():
            return self.send_error(404)
        self._send(200, route[0].read_bytes(), route[1])

    def _design_file(self, rel: str):
        """Belong. design system files (tokens, components, fonts). Fixed list, no paths from the URL."""
        files = {"tokens.css": "text/css; charset=utf-8", "belong.css": "text/css; charset=utf-8",
                 "theme-dark.css": "text/css; charset=utf-8"}
        files.update({f"fonts/{f.name}": "font/woff2" for f in (WEB / "ds" / "fonts").glob("*.woff2")})
        ctype = files.get(rel)
        if ctype is None:
            return self.send_error(404)
        self._send(200, (WEB / "ds" / rel).read_bytes(), ctype)

    def _query(self) -> dict:
        from urllib.parse import parse_qs, urlparse
        return {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}

    def _trust_get(self, what: str):
        q = self._query()
        if what == "attestation":
            return self._json(200, attest.attestation(q.get("nonce", "")))
        if what == "source":
            m = attest.measurements()
            return self._json(200, {"open_source": m["open_source_files"], "warden_measurement": m["warden"],
                                    "matchmaker_measurement": m["matchmaker"]})
        if what.startswith("source/"):
            text = attest.open_source_text(what.removeprefix("source/"))
            if text is None:
                return self._json(404, {"error": "Not one of the published files. See /api/trust/source."})
            return self._send(200, text.encode(), "text/plain; charset=utf-8")
        if what == "receipts":
            try:
                start = max(0, int(q.get("from", 0)))
            except ValueError:
                start = 0
            with SESSION.lock:
                log = [r.to_dict() for r in SESSION.warden.receipts]
            prev = log[start - 1]["hash"] if 0 < start <= len(log) else "0" * 64
            return self._json(200, {"total": len(log), "from": start, "prev_hash": prev,
                                    "receipts": log[start:start + 5000]})
        return self.send_error(404)

    def do_POST(self):
        path = self.path.split("?")[0]
        if path.startswith("/api/probe/"):
            fn = PROBES.get(path.removeprefix("/api/probe/"))
            if fn is None:
                return self.send_error(404)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_BODY:
                return self._json(413, {"error": "Request too large."})
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
                if not isinstance(body, dict):
                    raise ValueError
                return self._json(200, fn(body))
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": f"Bad request: {e}"})
        session, shape = SESSION, lambda s: s.state()
        if path.startswith("/api/chat/"):
            name = path.removeprefix("/api/chat/")
            if name not in CHAT_ACTIONS:
                return self.send_error(404)
            session, shape, path = chat_session(), chat_state, "/api/" + name
        action = ACTIONS.get(path.removeprefix("/api/"))
        if not path.startswith("/api/") or action is None:
            return self.send_error(404)
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "Request too large."})
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError
        except ValueError:
            return self._json(400, {"error": "Bad request."})
        if session is not SESSION:
            body["slot"] = "A"  # the chat mockup is one person
        try:
            if action in UNLOCKED:
                action(session, body)  # takes the session lock itself, around the slow agent call
            else:
                with session.lock:
                    action(session, body)
        except DemoError as e:
            with session.lock:
                return self._json(400, {"error": str(e), **shape(session)})
        with session.lock:
            return self._json(200, shape(session))

    def log_message(self, *args):
        pass


class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


def serve(port: int = 8000) -> None:
    """Listen on loopback only (IPv4 and IPv6), never on a network interface."""
    httpd = ThreadingHTTPServer(("127.0.0.1", port), _Handler)
    try:
        httpd6 = _IPv6Server(("::1", port), _Handler)
        threading.Thread(target=httpd6.serve_forever, daemon=True).start()
    except OSError:
        pass  # no IPv6 loopback on this machine
    print(f"Demo at http://localhost:{port}   inspector at http://localhost:{port}/inspector   (Ctrl+C to stop)",
          flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
