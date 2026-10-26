"""The web server (belong/server.py): loopback only unless told otherwise, the hosted demo's
password on every request, and the Matchmaker running when an open page checks in."""
from __future__ import annotations

import base64
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from belong import server
from belong.__main__ import main
from belong.demo import DemoSession


def basic(password: str, user: str = "anyone") -> str:
    return "Basic " + base64.b64encode(f"{user}:{password}".encode()).decode()


class Password(unittest.TestCase):
    def test_no_password_set_means_open(self):
        with mock.patch.dict(os.environ, {"BELONG_PASSWORD": ""}):
            self.assertTrue(server._authorized(""))

    def test_with_a_password_every_request_needs_it(self):
        with mock.patch.dict(os.environ, {"BELONG_PASSWORD": "tide-42"}):
            self.assertTrue(server._authorized(basic("tide-42")))
            for header in ("", basic("wrong"), "Bearer tide-42", "Basic not-base64!", basic("tide-4")):
                self.assertFalse(server._authorized(header), header)

    def test_the_server_refuses_pages_and_actions_without_it(self):
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"

        def status(path, header=None, data=None):
            req = urllib.request.Request(base + path, data=data, headers={"Authorization": header} if header else {})
            try:
                return urllib.request.urlopen(req).status
            except urllib.error.HTTPError as e:
                return e.code
        try:
            with mock.patch.dict(os.environ, {"BELONG_PASSWORD": "tide-42"}):
                for path in ("/", "/api/state", "/api/trust/receipts", "/ds/tokens.css"):
                    self.assertEqual(status(path), 401, path)
                self.assertEqual(status("/api/reset", data=b"{}"), 401)
                self.assertEqual(status("/api/probe/attacks", data=b"{}"), 401)
                self.assertEqual(status("/trust", basic("tide-42")), 200)
        finally:
            httpd.shutdown()
            httpd.server_close()


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


class MatchingOnCheckIn(unittest.TestCase):
    def test_runs_at_most_every_few_seconds(self):
        s = DemoSession(seed=7, agent=None)
        with mock.patch.object(s, "run_matchmaker") as run, mock.patch.object(server.time, "monotonic") as clock:
            clock.return_value = 1000.0
            server._match_if_due(s)
            server._match_if_due(s)  # a second page checking in straight away
            self.assertEqual(run.call_count, 1)
            clock.return_value = 1000.0 + server.MATCH_EVERY
            server._match_if_due(s)
            self.assertEqual(run.call_count, 2)
            run.assert_called_with(quiet=True)


if __name__ == "__main__":
    unittest.main()
