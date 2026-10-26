"""Web server. By default loopback only (127.0.0.1 and ::1), never on a network interface.

Hosted (see README, "Run it online"): behind HTTPS (Caddy on a VM, or Cloud Run), with
BELONG_PASSWORD set. Then every page and action needs it: /login asks for the password alone and sets
a cookie that proves it without containing it.

  /, /demo                  two-person demo (belong/web/demo.html)
  /welcome                  landing page (belong/web/landing.html)
  /connector (and /chat)    mockup: one person using Belong. inside a ChatGPT-style chat, through a connector
                            (belong/web/chat.html), on its own demo session
  /trust                    trust page (belong/web/trust.html)
  /inspector                batch-run inspector (belong/web/index.html + out/run.json); "Take over" gives a
                            synthetic person a phone in the live demo (/demo?phone=<their id>)
  /api/...                  demo actions; each POST returns the full demo state. A "slot" must be a phone
                            that exists: A, B, or someone taken over (the connector's session: A or B only)

Trust API (for people and their agents; see the trust page for examples):
  GET  /api/trust/attestation?nonce=N   measurements + policy (simulated attestation)
  GET  /api/trust/source                list of open-source files with SHA-256
  GET  /api/trust/source/<file>         the file itself (open-source files only)
  GET  /api/trust/receipts?from=N       the public, hash-chained receipt log
  POST /api/probe/canary                plant secrets, run a week, search all outputs
  POST /api/probe/attacks               try to break in and get data out
"""
from __future__ import annotations

import hashlib
import hmac
import html
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import attest, probe
from .demo import DemoError, DemoSession

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "belong" / "web"
FILES = {
    "/": (WEB / "demo.html", "text/html; charset=utf-8"),
    "/demo": (WEB / "demo.html", "text/html; charset=utf-8"),
    "/welcome": (WEB / "landing.html", "text/html; charset=utf-8"),
    "/connector": (WEB / "chat.html", "text/html; charset=utf-8"),
    "/chat": (WEB / "chat.html", "text/html; charset=utf-8"),
    "/trust": (WEB / "trust.html", "text/html; charset=utf-8"),
    "/trust/TRUST.md": (ROOT / "TRUST.md", "text/plain; charset=utf-8"),
    "/inspector": (WEB / "index.html", "text/html; charset=utf-8"),
    "/nav.js": (WEB / "nav.js", "text/javascript; charset=utf-8"),
    "/run.json": (ROOT / "out" / "run.json", "application/json"),
}
MAX_BODY = 64 * 1024

# How many synthetic people the demo starts with (BELONG_PEOPLE; tests use DemoSession's own 300).
PEOPLE = int(os.environ.get("BELONG_PEOPLE", "5000"))
SESSION = DemoSession(people=PEOPLE)
# The chat mockup's own session: one person, matched against the synthetic people, so it never
# touches the two phones. Made on first use.
_CHAT = None
CHAT_ACTIONS = {"reset", "verify", "new_listing", "open_listing", "prefill", "agent", "submit", "match", "answer", "set_active", "delete_listing",
                "decide", "set_prefs", "message", "react", "update_profile"}


def chat_session() -> DemoSession:
    global _CHAT
    if _CHAT is None:
        _CHAT = DemoSession(people=PEOPLE)
        _CHAT.assistant_slots = {"A"}  # the connector user
    return _CHAT


def chat_state(session: DemoSession) -> dict:
    return {"me": session.phone_state("A"), "agent_status": session.agent_status}

# action -> function(session, body)
ACTIONS = {
    "reset": lambda s, b: s.reset(),
    "verify": lambda s, b: s.verify(b.get("slot"), b.get("name"), b.get("age")),
    "new_listing": lambda s, b: s.new_listing(b.get("slot")),
    "update_profile": lambda s, b: s.update_profile(b.get("slot"), b.get("name")),
    "delete_listing": lambda s, b: s.delete_listing(b.get("slot"), b.get("listing_id")),
    "set_active": lambda s, b: s.set_active(b.get("slot"), b.get("listing_id"), b.get("active")),
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
    # The inspector's "Take over": a phone for a synthetic person (the reply carries its "slot").
    "take_over": lambda s, b: s.take_over(b.get("persona_id")),
    "hand_back": lambda s, b: s.hand_back(b.get("slot")),
}
UNLOCKED = {ACTIONS["agent"]}
PROBES = {"canary": probe.canary_test, "attacks": probe.attack_suite}


# -- The hosted demo's password (BELONG_PASSWORD) ------------------------------------------------

COOKIE = "belong_session"
SIGNED_IN_FOR = 30 * 24 * 3600  # seconds
OPEN_PATHS = {"/login"}  # and the design files (/ds/...), which the sign-in page needs
LOGIN_LIMIT = 20  # failed attempts a minute, from anyone; then sign-in pauses for that minute
_failures: list = []
_failures_lock = threading.Lock()


def _password() -> str:
    return os.environ.get("BELONG_PASSWORD", "")


def _session_token(password: str) -> str:
    """What the cookie holds: derived from the password, so it proves you knew it without containing
    it, survives restarts and redeploys, and stops working when the password changes."""
    return hmac.new(password.encode(), b"belong demo session v1", hashlib.sha256).hexdigest()


def _authorized(cookie_header: str) -> bool:
    """With BELONG_PASSWORD set (the hosted demo), every request needs the sign-in cookie."""
    password = _password()
    if not password:
        return True
    from http.cookies import CookieError, SimpleCookie
    try:
        jar = SimpleCookie(cookie_header or "")
    except CookieError:
        return False
    given = jar[COOKIE].value if COOKIE in jar else ""
    return hmac.compare_digest(given.encode(), _session_token(password).encode())


def _too_many_failures(record: bool = False) -> bool:
    now = time.monotonic()
    with _failures_lock:
        _failures[:] = [t for t in _failures if now - t < 60]
        if record:
            _failures.append(now)
        return len(_failures) >= LOGIN_LIMIT


def _safe_next(target: str) -> str:
    """Only a path on this site (never another site) after signing in."""
    return target if target.startswith("/") and not target.startswith("//") and "\\" not in target else "/"


class _Handler(BaseHTTPRequestHandler):
    def _locked_out(self, path: str) -> bool:
        if path in OPEN_PATHS or path.startswith("/ds/") or _authorized(self.headers.get("Cookie", "")):
            return False
        if self.command == "GET" and not path.startswith("/api/"):
            from urllib.parse import quote
            self.send_response(303)
            self.send_header("Location", "/login?next=" + quote(self.path, safe="/"))
            self.end_headers()
        else:
            # Read (and drop) any body first: answering before it's read can cut the connection
            # before the answer arrives.
            length = int(self.headers.get("Content-Length") or 0)
            if 0 < length <= MAX_BODY:
                self.rfile.read(length)
            self._json(401, {"error": "This demo needs its password. Sign in at /login."})
        return True

    def _login_page(self, code: int = 200, next_path: str = "/", error: str = "") -> None:
        page = (WEB / "login.html").read_text()
        page = page.replace("{{next}}", html.escape(_safe_next(next_path)))
        page = page.replace("{{error_class}}", " error" if error else "")
        page = page.replace("{{error}}", f'<p class="error" role="alert">{html.escape(error, quote=False)}</p>' if error else "")
        self._send(code, page.encode(), "text/html; charset=utf-8")

    def _login(self) -> None:
        from urllib.parse import parse_qs
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            return self._json(413, {"error": "Request too large."})
        form = {k: v[0] for k, v in parse_qs(self.rfile.read(length).decode("utf-8", "replace")).items()}
        next_path = _safe_next(form.get("next", "/"))
        password = _password()
        if not password:
            return self._redirect(next_path)
        if _too_many_failures():
            return self._login_page(429, next_path, "Too many tries. Wait a minute, then try again.")
        if not hmac.compare_digest(form.get("password", "").encode(), password.encode()):
            _too_many_failures(record=True)
            return self._login_page(401, next_path, "That password isn't right.")
        secure = "; Secure" if self.headers.get("X-Forwarded-Proto", "") == "https" else ""
        self.send_response(303)
        self.send_header("Set-Cookie", f"{COOKIE}={_session_token(password)}; Path=/; Max-Age={SIGNED_IN_FOR}; "
                                       f"HttpOnly; SameSite=Lax{secure}")
        self.send_header("Location", next_path)
        self.end_headers()

    def _redirect(self, location: str) -> None:
        self.send_response(303)
        self.send_header("Location", location)
        self.end_headers()

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
        if self._locked_out(path):
            return
        if path == "/login":
            if _authorized(self.headers.get("Cookie", "")):  # signed in already, or no password set
                return self._redirect(_safe_next(self._query().get("next", "/")))
            return self._login_page(next_path=self._query().get("next", "/"))
        if path == "/api/state":
            with SESSION.lock:
                _match_if_due(SESSION)
                return self._json(200, SESSION.state())
        if path == "/api/chat/state":
            session = chat_session()
            with session.lock:
                _match_if_due(session)
                # ?full=1: the test phone beside the chat (person B in the same session) needs the
                # phone-demo state; the chat itself only ever gets person A's.
                return self._json(200, session.state() if "full=1" in self.path else chat_state(session))
        if path.startswith("/ds/"):
            return self._design_file(path.removeprefix("/ds/"))
        if path.startswith("/api/trust/"):
            return self._trust_get(path.removeprefix("/api/trust/"))
        if path == "/api/options":
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(self.path).query)
            with SESSION.lock:
                return self._json(200, SESSION.options("A", (q.get("kind") or [None])[0], (q.get("city") or [None])[0]))
        route = FILES.get(path)
        if path == "/run.json" and route and not route[0].exists():
            return self._json(503, {"preparing": True})  # the inspector's run is still being built
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
        if self._locked_out(path):
            return
        if path == "/login":
            return self._login()
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
            session, path = chat_session(), "/api/" + name
            shape = (lambda x: x.state()) if "full=1" in self.path else chat_state
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
            # The chat is person A; the optional test phone beside it is person B.
            body["slot"] = body.get("slot") if body.get("slot") in ("A", "B") else "A"
        elif "slot" in body:
            # A, B, or a synthetic person taken over in this session; nothing else.
            with session.lock:
                known = isinstance(body["slot"], str) and body["slot"] in session.slots
                if not known:
                    return self._json(400, {"error": "No such phone.", **shape(session)})
        try:
            if action in UNLOCKED:
                result = action(session, body)  # takes the session lock itself, around the slow agent call
            else:
                with session.lock:
                    result = action(session, body)
            with session.lock:
                session.changes += 1  # something happened: the Matchmaker will look again
        except DemoError as e:
            with session.lock:
                return self._json(400, {"error": str(e), **shape(session)})
        with session.lock:
            out = shape(session)
            if action is ACTIONS["take_over"]:
                out["slot"] = result
            return self._json(200, out)

    def log_message(self, *args):
        pass


class _IPv6Server(ThreadingHTTPServer):
    address_family = socket.AF_INET6


MATCH_EVERY = 6  # at most this often (seconds) when something has changed
IDLE_EVERY = 60  # and once a minute anyway, for what only time changes (a daily limit opening up again)
_last_match: dict = {}  # id(session) -> {"at": when it last ran, "seen": what it had seen, "more": it sent something}


def _activity(session: DemoSession) -> tuple:
    return session.board.changes, session.changes


def _match_if_due(session: DemoSession) -> None:
    """The demo's Matchmaker runs on its own, like the real service would, so nobody has to press a
    button. It runs when an open page checks in (they poll every few seconds), so nothing runs while
    nobody is looking; and it only runs when something changed (a listing, an answer, a decision) or
    its last run sent something, plus once a minute anyway. Call with the session's lock held."""
    now, last = time.monotonic(), _last_match.get(id(session))
    if last is not None:
        changed = last["more"] or _activity(session) != last["seen"]
        if not (changed and now - last["at"] >= MATCH_EVERY) and now - last["at"] < IDLE_EVERY:
            return
    try:
        more = session.run_matchmaker(quiet=True)
    except DemoError:
        more = False  # no demo listings yet
    _last_match[id(session)] = {"at": now, "seen": _activity(session), "more": more}


def serve(port: int = 8000, host: str = "127.0.0.1") -> None:
    """Listen on loopback only (IPv4 and IPv6) unless a host is given (the hosted container)."""
    httpd = ThreadingHTTPServer((host, port), _Handler)
    if host == "127.0.0.1":
        try:
            httpd6 = _IPv6Server(("::1", port), _Handler)
            threading.Thread(target=httpd6.serve_forever, daemon=True).start()
        except OSError:
            pass  # no IPv6 loopback on this machine
    where = f"http://localhost:{port}" if host == "127.0.0.1" else f"{host}:{port}"
    lock = "   password required" if os.environ.get("BELONG_PASSWORD") else ""
    print(f"Demo at {where}   inspector at {where}/inspector{lock}   (Ctrl+C to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
