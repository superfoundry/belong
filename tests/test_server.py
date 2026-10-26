"""The web server (belong/server.py): loopback only unless told otherwise, the hosted demo's
password (asked alone, at /login) on every request, and the Matchmaker running when an open page
checks in."""
from __future__ import annotations

import os
import threading
import unittest
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from belong import server
from belong.__main__ import main
from belong.demo import DemoSession


class Password(unittest.TestCase):
    """The hosted demo asks for its password alone (/login), then a cookie proves it."""

    def setUp(self):
        server._failures.clear()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.env = mock.patch.dict(os.environ, {"BELONG_PASSWORD": "tide-42"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.httpd.shutdown()
        self.httpd.server_close()
        server._failures.clear()

    def request(self, path, data=None, cookie=None):
        """(status, headers, body) without following redirects."""
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        req = urllib.request.Request(self.base + path, data=data, headers={"Cookie": cookie} if cookie else {})
        try:
            with urllib.request.build_opener(NoRedirect).open(req) as r:
                return r.status, r.headers, r.read().decode()
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read().decode()

    def sign_in(self, password, next_path="/"):
        return self.request("/login", urllib.parse.urlencode({"password": password, "next": next_path}).encode())

    def test_no_password_set_means_open(self):
        with mock.patch.dict(os.environ, {"BELONG_PASSWORD": ""}):
            self.assertTrue(server._authorized(""))
            self.assertEqual(self.request("/api/state")[0], 200)

    def test_pages_send_you_to_sign_in_and_actions_are_refused(self):
        status, headers, _ = self.request("/connector")
        self.assertEqual((status, headers["Location"]), (303, "/login?next=/connector"))
        for path in ("/api/state", "/api/trust/receipts"):
            self.assertEqual(self.request(path)[0], 401, path)
        self.assertEqual(self.request("/api/reset", b"{}")[0], 401)
        self.assertEqual(self.request("/api/probe/attacks", b"{}")[0], 401)
        self.assertEqual(self.request("/ds/tokens.css")[0], 200)  # the sign-in page's own styles

    def test_the_sign_in_page_asks_for_the_password_alone(self):
        status, _, page = self.request("/login?next=/connector")
        self.assertEqual(status, 200)
        self.assertIn('type="password"', page)
        self.assertNotIn("username", page.lower())
        self.assertIn('value="/connector"', page)

    def test_the_right_password_sets_a_cookie_that_doesnt_contain_it(self):
        status, headers, _ = self.sign_in("tide-42", "/connector")
        self.assertEqual((status, headers["Location"]), (303, "/connector"))
        cookie = headers["Set-Cookie"]
        self.assertNotIn("tide-42", cookie)
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Lax", cookie)
        jar = cookie.split(";")[0]
        self.assertEqual(self.request("/api/state", cookie=jar)[0], 200)
        self.assertEqual(self.request("/trust", cookie=jar)[0], 200)

    def test_a_wrong_password_or_cookie_gets_nothing(self):
        status, headers, page = self.sign_in("tide-4")
        self.assertEqual(status, 401)
        self.assertIsNone(headers["Set-Cookie"])
        self.assertIn("That password isn't right.", page)
        self.assertEqual(self.request("/api/state", cookie=f"{server.COOKIE}=tide-42")[0], 401)
        self.assertEqual(self.request("/api/state", cookie=f"{server.COOKIE}=")[0], 401)

    def test_never_sent_to_another_site_after_signing_in(self):
        for target in ("//evil.example", "https://evil.example", "/\\evil.example"):
            self.assertEqual(self.sign_in("tide-42", target)[1]["Location"], "/", target)

    def test_guessing_is_limited(self):
        for _ in range(server.LOGIN_LIMIT):
            self.sign_in("wrong")
        status, headers, page = self.sign_in("tide-42")  # even the right one waits out the minute
        self.assertEqual(status, 429)
        self.assertIsNone(headers["Set-Cookie"])


@mock.patch("belong.__main__.load_dotenv", lambda: None)  # never read the real .env in tests
class Listening(unittest.TestCase):
    def test_loopback_unless_told_otherwise(self):
        with mock.patch.object(server, "serve") as serve, mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("BELONG_HOST", None)
            os.environ.pop("PORT", None)
            main(["serve"])
            serve.assert_called_once_with(8000, "127.0.0.1")
        with mock.patch.object(server, "serve") as serve, \
                mock.patch.dict(os.environ, {"BELONG_HOST": "0.0.0.0", "PORT": "8080"}):
            main(["serve"])
            serve.assert_called_once_with(8080, "0.0.0.0")


    def test_hosted_never_starts_without_a_password(self):
        with mock.patch.object(server, "serve") as serve, \
                mock.patch.dict(os.environ, {"BELONG_REQUIRE_PASSWORD": "1", "BELONG_PASSWORD": ""}):
            self.assertEqual(main(["serve"]), 2)
            serve.assert_not_called()


class MatchingOnCheckIn(unittest.TestCase):
    """It runs when a page checks in, but only when something changed (or once a minute anyway)."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.run = mock.patch.object(self.s, "run_matchmaker", return_value=False).start()
        self.clock = mock.patch.object(server.time, "monotonic", return_value=1000.0).start()
        self.addCleanup(mock.patch.stopall)
        server._last_match.pop(id(self.s), None)

    def checkin(self, at):
        self.clock.return_value = 1000.0 + at
        server._match_if_due(self.s)
        return self.run.call_count

    def test_the_first_check_in_runs_it(self):
        self.assertEqual(self.checkin(0), 1)
        self.run.assert_called_with(quiet=True)

    def test_nothing_changed_means_it_waits(self):
        self.checkin(0)
        self.assertEqual(self.checkin(server.MATCH_EVERY), 1)
        self.assertEqual(self.checkin(server.IDLE_EVERY - 1), 1)
        self.assertEqual(self.checkin(server.IDLE_EVERY), 2)  # once a minute anyway

    def test_a_change_starts_it_again_but_not_more_than_every_few_seconds(self):
        self.checkin(0)
        self.s.changes += 1
        self.assertEqual(self.checkin(1), 1)
        self.assertEqual(self.checkin(server.MATCH_EVERY), 2)
        self.s.board.changes += 1  # a listing updated (e.g. by the agent, off the request)
        self.assertEqual(self.checkin(2 * server.MATCH_EVERY), 3)

    def test_after_sending_something_it_looks_again(self):
        self.run.return_value = True  # there may be more people waiting
        self.checkin(0)
        self.run.return_value = False
        self.assertEqual(self.checkin(server.MATCH_EVERY), 2)
        self.assertEqual(self.checkin(2 * server.MATCH_EVERY), 2)


if __name__ == "__main__":
    unittest.main()
