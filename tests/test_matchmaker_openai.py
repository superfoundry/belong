"""The Matchmaker AI's judge over a Chat Completions API (matchmaker_openai.py), and which Matchmaker runs (demo,
`belong run --matcher ai`). Every HTTP call here goes to a stub (no network): fixed instructions with the listings
after a fixed data line, the JSON-schema response format with its plain-JSON fallbacks, the endpoint and key rules,
timeouts and retries, resting after a failure, the rules' verdict when there is no answer, and token usage."""
from __future__ import annotations

import io
import json
import os
import socket
import tempfile
import time
import unittest
import urllib.error
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

from belong import agent_openai, attest, sim
from belong import __main__ as cli
from belong import matchmaker_openai as mm
from belong.demo import DemoSession
from belong.matching.hybrid import JudgeWaiting, LLMMatcher, answer_schema
from belong.matching.matcher import OfflineMatcher
from belong.matchmaker_openai import (API_URL, COOL_OFF, CONFIG_COOL_OFF, DATA_LINE, INSTRUCTIONS, JudgeUnavailable,
                                      OpenAIJudge)
from tests.test_demo import _Page, climbers, fill
from tests.test_hybrid import as_plain, binders, synthetic_board

LOCAL = "http://localhost:8000/v1/chat/completions"
REQUEST = {"focus": "c1", "candidates": ["c2", "c3"],
           "listings": {"c1": {"kind": "other", "exactness": 0.6, "looking_for": ["a sail repair teacher"],
                               "description": "hoping to learn to patch sails"},
                        "c2": {"kind": "other", "exactness": 0.5, "offering": ["a sail repair teacher"],
                               "description": "i mend sails in the harbour loft"},
                        "c3": {"kind": "other", "exactness": 0.5, "offering": ["a choir singer"],
                               "description": "tenor looking for a choir"}}}
ANSWER = {"pairs": [{"candidate": "c2", "focus_wants": "match", "candidate_wants": "match", "fit": 0.8},
                    {"candidate": "c3", "focus_wants": "no", "candidate_wants": "no", "fit": 0.1}],
          "hunts_one_person": []}


def reply(answer=ANSWER, usage=None, content=None, **message):
    """A Chat Completions response holding this answer (or this content as it is)."""
    msg = {"role": "assistant", "content": json.dumps(answer) if content is None else content, **message}
    out = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}]}
    if usage is not None:
        out["usage"] = usage
    return out


def http_error(code, headers=None):
    return urllib.error.HTTPError(LOCAL, code, "error", headers or {}, io.BytesIO(b'{"error": {"message": "x"}}'))


class Stub:
    """The HTTP layer: records each call and gives the scripted outcomes in turn (a dict is a response; an exception
    is raised; a callable gets the payload)."""

    def __init__(self, *outcomes):
        self.outcomes, self.calls = list(outcomes), []

    def __call__(self, url, payload, headers, timeout):
        self.calls.append({"url": url, "payload": json.loads(json.dumps(payload)), "headers": dict(headers),
                           "timeout": timeout})
        got = self.outcomes.pop(0) if len(self.outcomes) > 1 else self.outcomes[0]
        if isinstance(got, BaseException):
            raise got
        return got(payload) if callable(got) else got


class Clock:
    def __init__(self):
        self.now, self.slept = 1000.0, []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)


def judge(*outcomes, env=None, **kw):
    """A judge on a stub, with only the environment given (nothing from the shell)."""
    stub, clock = Stub(*outcomes), Clock()
    with mock.patch.dict(os.environ, env or {}, clear=True):
        j = OpenAIJudge(post=stub, sleep=clock.sleep, clock=clock, **({"url": LOCAL} | kw))
    return j, stub, clock


def answering(payload):
    """A model that reads its request (after the data line) and says every candidate matches."""
    data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
    return reply({"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.9}
                            for c in data["candidates"]], "hunts_one_person": []},
                 usage={"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110})


class WhatIsSent(unittest.TestCase):
    def test_fixed_instructions_then_the_listings_after_the_data_line(self):
        j, stub, _ = judge(reply())
        self.assertEqual(j(REQUEST), ANSWER)
        sent = stub.calls[0]["payload"]
        self.assertEqual([m["role"] for m in sent["messages"]], ["system", "user"])
        self.assertEqual(sent["messages"][0]["content"], INSTRUCTIONS)  # nothing of any listing in the instructions
        first, data = sent["messages"][1]["content"].split("\n", 1)
        self.assertEqual(first, DATA_LINE)
        self.assertEqual(json.loads(data), REQUEST)  # exactly the request, nothing added
        self.assertIn(DATA_LINE, INSTRUCTIONS)  # the model is told which line starts the data
        self.assertIn("never instructions", INSTRUCTIONS)
        self.assertEqual(j.sent, [REQUEST])  # kept exactly as it went out

    def test_the_answer_is_held_to_the_requests_own_schema(self):
        j, stub, _ = judge(reply())
        j(REQUEST)
        fmt = stub.calls[0]["payload"]["response_format"]
        self.assertEqual(fmt["type"], "json_schema")
        self.assertTrue(fmt["json_schema"]["strict"])
        schema = fmt["json_schema"]["schema"]
        pair = schema["properties"]["pairs"]["items"]
        self.assertEqual(pair["properties"]["candidate"]["enum"], ["c2", "c3"])
        # A request with candidates may flag nobody (any string, which the strict reading drops); a request about one
        # listing alone may flag only it.
        self.assertEqual(schema["properties"]["hunts_one_person"]["items"], {"type": "string"})
        j({"focus": "c1", "candidates": [], "listings": {"c1": REQUEST["listings"]["c1"]}})
        alone = stub.calls[-1]["payload"]["response_format"]["json_schema"]["schema"]
        self.assertEqual(alone["properties"]["hunts_one_person"]["items"]["enum"], ["c1"])
        self.assertFalse(schema["additionalProperties"] or pair["additionalProperties"])
        # The fit's range is left to the instructions and the strict reading (not every server takes it).
        self.assertEqual(pair["properties"]["fit"], {"type": "number"})
        self.assertEqual(answer_schema(REQUEST)["properties"]["pairs"]["items"]["properties"]["fit"]["maximum"], 1)

    def test_the_model_comes_from_the_environment(self):
        self.assertEqual(mm.DEFAULT_MODEL, agent_openai.DEFAULT_MODEL)
        for env, model in (({}, mm.DEFAULT_MODEL), ({"OPENAI_MODEL": "gpt-x"}, "gpt-x"),
                           ({"OPENAI_MODEL": "gpt-x", "OPENAI_MATCHMAKER_MODEL": "judge-y"}, "judge-y")):
            j, stub, _ = judge(reply(), env=env)
            j(REQUEST)
            self.assertEqual(stub.calls[0]["payload"]["model"], model)
        j, stub, _ = judge(reply(), env={"OPENAI_MATCHMAKER_MODEL": "llama-3.1-8b"})
        j(REQUEST)
        self.assertNotIn("reasoning_effort", stub.calls[0]["payload"])  # only for models that take it
        j, stub, _ = judge(reply(), env={"OPENAI_MATCHMAKER_MODEL": "gpt-5-mini"})
        j(REQUEST)
        self.assertEqual(stub.calls[0]["payload"]["reasoning_effort"], "low")

    def test_the_instructions_are_in_the_published_file(self):
        self.assertIn("matchmaker_openai.py", attest.MATCHMAKER_FILES)
        self.assertIn(INSTRUCTIONS.split("\n")[0][:60], attest.open_source_text("matchmaker_openai.py").replace("\\\n", ""))


class WhereItGoes(unittest.TestCase):
    def test_the_endpoint_is_a_setting_and_the_key_is_optional(self):
        j, stub, _ = judge(reply(), url=None, env={"BELONG_MATCHMAKER_URL": LOCAL})
        j(REQUEST)
        self.assertEqual(stub.calls[0]["url"], LOCAL)
        self.assertNotIn("Authorization", stub.calls[0]["headers"])  # a self-hosted server that takes no key
        self.assertEqual(j.describe(), f"{mm.DEFAULT_MODEL} at localhost")

    def test_the_openai_key_goes_only_to_openai(self):
        j, stub, _ = judge(reply(), env={"OPENAI_API_KEY": "sk-test-openai"})
        j(REQUEST)
        self.assertNotIn("Authorization", stub.calls[0]["headers"])  # localhost never gets the OpenAI key
        j, stub, _ = judge(reply(), env={"OPENAI_API_KEY": "sk-test-openai", "BELONG_MATCHMAKER_KEY": "local-key"})
        j(REQUEST)
        self.assertEqual(stub.calls[0]["headers"]["Authorization"], "Bearer local-key")
        j, stub, _ = judge(reply(), url=None, env={"OPENAI_API_KEY": "sk-test-openai"})
        j(REQUEST)
        self.assertEqual((stub.calls[0]["url"], stub.calls[0]["headers"]["Authorization"]),
                         (API_URL, "Bearer sk-test-openai"))
        self.assertEqual(j.describe(), f"OpenAI {mm.DEFAULT_MODEL}")
        self.assertNotIn("sk-test", repr(j) + json.dumps(j.state()) + json.dumps(list(j.log)))

    def test_openai_without_a_key_or_plain_http_elsewhere_is_refused(self):
        with self.assertRaises(JudgeUnavailable):
            judge(reply(), url=None, env={})
        for url in ("http://matchmaker.example.org/v1/chat/completions", "ftp://localhost/x", "localhost:8000"):
            with self.assertRaises(JudgeUnavailable, msg=url):
                judge(reply(), url=url)
        for url in ("https://matchmaker.example.org/v1/chat/completions", "http://127.0.0.1:8080/v1/chat/completions",
                    "http://[::1]:11434/v1/chat/completions"):
            self.assertEqual(mm.endpoint_problem(url), "", url)

    def test_a_redirect_is_never_followed(self):
        """urllib would follow a redirect as a GET to any address, carrying the key: the judge's HTTP layer answers it
        as an error instead, so nothing goes past the endpoint set. In process, on a stand-in endpoint (no network)."""
        import email
        import urllib.request
        import urllib.response

        class Endpoint(urllib.request.HTTPSHandler):
            handler_order = 100  # before any handler that could reach the network

            def __init__(self):
                super().__init__()
                self.seen = []

            def https_open(self, req):  # every answer: "it's over there"
                self.seen.append((req.full_url, req.get_header("Authorization")))
                r = urllib.response.addinfourl(io.BytesIO(b""), email.message_from_string(
                    "Location: http://elsewhere.example.org/collect\n\n"), req.full_url, 302)
                r.msg = "Found"
                return r
            http_open = https_open

        endpoint, url = Endpoint(), "https://matchmaker.example.org/v1/chat/completions"
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), mm._NoRedirect, endpoint)
        with self.assertRaises(urllib.error.HTTPError) as caught:
            mm._http_post(url, {"model": "m"}, {"Authorization": "Bearer local-key"}, 5, opener=opener)
        self.assertEqual((caught.exception.code, endpoint.seen), (302, [(url, "Bearer local-key")]))
        # The judge's own opener follows no redirect, and a redirect fails the request like any server error.
        redirects = [h for h in mm._OPENER.handlers if isinstance(h, urllib.request.HTTPRedirectHandler)]
        self.assertTrue(redirects and all(isinstance(h, mm._NoRedirect) for h in redirects))
        j, stub, _ = judge(http_error(302))
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        self.assertEqual(len(stub.calls), 1)

    def test_a_screen_can_keep_a_request_from_going_out(self):
        j, stub, _ = judge(reply(), screen=lambda request: "c3" not in request["candidates"])
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        self.assertEqual((stub.calls, j.sent), ([], []))
        self.assertEqual(j({**REQUEST, "candidates": ["c2"]}), ANSWER)


class FormatsTheServerTakes(unittest.TestCase):
    def test_plain_json_when_the_server_refuses_a_schema(self):
        j, stub, _ = judge(http_error(400), reply())
        self.assertEqual(j(REQUEST), ANSWER)
        self.assertEqual([c["payload"].get("response_format", {}).get("type") for c in stub.calls],
                         ["json_schema", "json_object"])
        self.assertIn("JSON", INSTRUCTIONS)  # what a JSON-object response format asks for
        j(REQUEST)  # the format that worked is used from then on
        self.assertEqual(stub.calls[-1]["payload"]["response_format"], {"type": "json_object"})
        self.assertEqual(j.sent, [REQUEST, REQUEST])  # once per call, whatever the formats tried

    def test_no_format_at_all_and_a_fenced_reply(self):
        j, stub, _ = judge(http_error(400), http_error(422), reply(content="Here it is:\n```json\n"
                                                                            + json.dumps(ANSWER) + "\n```"))
        self.assertEqual(j(REQUEST), ANSWER)
        self.assertNotIn("response_format", stub.calls[-1]["payload"])
        self.assertEqual(j.response_format, "text")

    def test_a_refusal_everywhere_keeps_the_format_and_fails(self):
        j, stub, _ = judge(http_error(400))
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        self.assertEqual(len(stub.calls), 3)
        self.assertEqual(j.response_format, "json_schema")


class WhenThereIsNoAnswer(unittest.TestCase):
    def test_retries_with_backoff_then_answers(self):
        j, stub, clock = judge(http_error(503), socket.timeout("slow"), reply())
        self.assertEqual(j(REQUEST), ANSWER)
        self.assertEqual(len(stub.calls), 3)
        self.assertEqual(clock.slept, [mm.BACKOFF, mm.BACKOFF * 2])
        self.assertEqual(j.usage["retries"], 2)
        self.assertEqual(j.sent, [REQUEST])
        self.assertTrue(all(c["timeout"] == mm.TIMEOUT for c in stub.calls))

    def test_a_rate_limits_retry_after_is_followed_up_to_a_cap(self):
        j, stub, clock = judge(http_error(429, {"Retry-After": "7"}), http_error(429, {"Retry-After": "999"}), reply())
        j(REQUEST)
        self.assertEqual(clock.slept, [7.0, mm.MAX_WAIT])

    def test_giving_up_rests_and_fails_at_once_meanwhile(self):
        j, stub, clock = judge(urllib.error.URLError("down"))
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        self.assertEqual(len(stub.calls), 1 + mm.RETRIES)
        with self.assertRaises(JudgeUnavailable) as again:
            j(REQUEST)
        self.assertEqual(len(stub.calls), 1 + mm.RETRIES)  # not tried while resting
        self.assertIn("resting", str(again.exception))
        self.assertTrue(j.state()["resting"])
        clock.now += COOL_OFF + 1
        stub.outcomes = [reply()]
        self.assertEqual(j(REQUEST), ANSWER)
        self.assertEqual(j.usage | {}, {**j.usage, "calls": 3, "answered": 1, "failed": 2})

    def test_a_rejected_key_or_unknown_model_isnt_retried_and_rests_longer(self):
        for code in (401, 404):
            j, stub, clock = judge(http_error(code))
            with self.assertRaises(JudgeUnavailable):
                j(REQUEST)
            self.assertEqual(len(stub.calls), 1)
            clock.now += COOL_OFF + 1
            with self.assertRaises(JudgeUnavailable):
                j(REQUEST)
            clock.now += CONFIG_COOL_OFF
            stub.outcomes = [reply()]
            self.assertEqual(j(REQUEST), ANSWER)

    def test_an_unreadable_answer_or_a_refusal_fails_this_request_only(self):
        for response in (reply(content="I think c2 is a good match."), reply(content=None, refusal="no"),
                         {"choices": []}, ["not", "an", "object"], reply(content="[1, 2]")):
            j, stub, _ = judge(response, reply())
            with self.assertRaises(JudgeUnavailable):
                j(REQUEST)
            self.assertEqual(j(REQUEST), ANSWER)  # no rest: the next request goes out
            self.assertEqual(len(stub.calls), 2)

    def test_any_other_failure_is_no_answer_never_a_crash(self):
        j, _, _ = judge(RuntimeError("surprise"))
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        with self.assertRaises(JudgeUnavailable):
            judge(reply())[0]("not a request")

    def test_a_reply_nested_too_deep_is_an_unreadable_answer_recorded(self):
        """Whatever a reply holds, no answer is a JudgeUnavailable, counted and logged: a reply nested too deep to
        read, and anything unforeseen while reading one (then the judge rests, as after any failed call)."""
        j, _, _ = judge(reply(content='{"pairs": ' + "[" * 200_000))
        with self.assertRaises(JudgeUnavailable):
            j(REQUEST)
        self.assertEqual((j.usage["failed"], j.usage["answered"], j.last_error), (1, 0, "unreadable answer"))
        self.assertEqual(j.log[-1]["why"], "unreadable answer")
        j, _, _ = judge(reply())
        with mock.patch.object(mm, "json_object", side_effect=TypeError("odd")):
            with self.assertRaises(JudgeUnavailable):
                j(REQUEST)
        self.assertEqual((j.usage["failed"], j.log[-1]["outcome"]), (1, "failed"))
        self.assertTrue(j.state()["resting"])


class WhatIsRecorded(unittest.TestCase):
    def test_token_usage_adds_up_and_odd_counts_are_ignored(self):
        j, _, _ = judge(reply(usage={"prompt_tokens": 1200, "completion_tokens": 300, "total_tokens": 1500,
                                     "completion_tokens_details": {"reasoning_tokens": 200},
                                     "prompt_tokens_details": {"cached_tokens": 1024}}),
                        reply(usage={"prompt_tokens": "lots", "completion_tokens": -5, "total_tokens": True}),
                        reply())
        for _ in range(3):
            j(REQUEST)
        u = j.usage
        self.assertEqual((u["prompt_tokens"], u["completion_tokens"], u["total_tokens"], u["reasoning_tokens"],
                          u["cached_tokens"], u["answered"]), (1200, 300, 1500, 200, 1024, 3))

    def test_the_log_holds_outcomes_never_listing_text(self):
        j, _, _ = judge(http_error(503), reply(usage={"prompt_tokens": 5, "completion_tokens": 2}))
        j(REQUEST)
        entry = j.log[-1]
        self.assertEqual((entry["outcome"], entry["attempts"], entry["format"], entry["prompt_tokens"]),
                         ("answered", 2, "json_schema", 5))
        self.assertNotIn("sail", json.dumps(list(j.log)) + json.dumps(j.state()))


class WithTheHybrid(unittest.TestCase):
    """The judge as hybrid.LLMMatcher calls it: what it sends is what the matcher kept, its answers become proposals
    through the rules, and with no answer the rules decide, recorded."""

    def test_what_goes_out_is_what_the_matcher_kept_and_the_answers_count(self):
        w, board = binders()
        j, stub, _ = judge(answering)
        matcher = LLMMatcher(judge=j)
        result = matcher.match(board, focus={"bk1", "bk2"})
        self.assertEqual(j.sent, matcher.payloads)
        self.assertEqual(matcher.last_run.get("failed", 0), 0)
        self.assertGreater(matcher.last_run["from_judge"], 0)
        self.assertTrue(result.proposals)
        self.assertEqual(j.usage["prompt_tokens"], 100 * len(stub.calls))

    def test_no_answer_means_the_rules_verdict_recorded(self):
        w, board = synthetic_board(120, seed=5)
        some = set(sorted(l.person_id for l in board.listings())[:5])
        expected = as_plain(OfflineMatcher().match(board, focus=some))
        for failure in (urllib.error.URLError("down"), http_error(401), http_error(500)):
            j, _, _ = judge(failure)
            matcher = LLMMatcher(judge=j, max_pairs_per_run=10_000)
            self.assertEqual(as_plain(matcher.match(board, focus=some)), expected)
            self.assertGreater(matcher.last_run["failed"], 0)
            self.assertEqual(matcher.last_run["failed_requests"], j.usage["failed"])
            self.assertEqual({d["how"] for d in matcher.decisions.values()} - {"hard"}, {"failed"})


class WhichMatchmakerRuns(unittest.TestCase):
    """BELONG_MATCHMAKER=ai|rules; by default the Matchmaker AI when OPENAI_API_KEY is set, else the rules. Never a
    call here: sessions are only built, or given a stub judge."""

    def choose(self, env):
        with mock.patch.dict(os.environ, env, clear=True):
            s = DemoSession(seed=7, agent=None, people=40)
        return s

    def test_the_choice_and_what_the_presenter_is_told(self):
        s = self.choose({"OPENAI_API_KEY": "sk-test"})
        self.assertIsInstance(s.matcher, LLMMatcher)
        self.assertIsInstance(s.matcher.judge.judge, OpenAIJudge)
        self.assertEqual(s.matchmaker_status, f"Matchmaker AI: OpenAI {mm.DEFAULT_MODEL} (a mock of the enclosed "
                                              "model; synthetic and test profiles only)")
        self.assertEqual(s.room_state()["matchmaker_status"], s.matchmaker_status)
        for env, label in (({"OPENAI_API_KEY": "sk-test", "BELONG_MATCHMAKER": "rules"},
                            "Matchmaker: rules (BELONG_MATCHMAKER=rules)"),
                           ({}, "Matchmaker: rules"),
                           ({"BELONG_MATCHMAKER": "ai"}, "Matchmaker: rules (the Matchmaker AI is off: OPENAI_API_KEY "
                                                         "isn't set)")):
            s = self.choose(env)
            self.assertIs(type(s.matcher), OfflineMatcher, env)
            self.assertEqual(s.matchmaker_status, label)
            self.assertEqual(s.room_state()["matchmaker"], {"ai": False})
        s = self.choose({"BELONG_MATCHMAKER": "ai", "BELONG_MATCHMAKER_URL": LOCAL,
                         "OPENAI_MATCHMAKER_MODEL": "open-weights-8b"})
        self.assertIsInstance(s.matcher, LLMMatcher)
        self.assertIn("open-weights-8b at localhost", s.matchmaker_status)

    def test_tests_run_on_the_rules_whatever_the_shell_says(self):
        self.assertEqual(os.environ.get("BELONG_MATCHMAKER"), "rules")
        self.assertIs(type(DemoSession(seed=7, agent=None, people=40).matcher), OfflineMatcher)

    def test_the_judge_and_its_answers_outlive_a_reset(self):
        s = self.choose({"OPENAI_API_KEY": "sk-test"})
        first, matcher = s.matcher.judge, s.matcher
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True):
            s.reset()
        self.assertIs(s.matcher.judge, first)  # its usage and answers waiting to be read
        self.assertIsNot(s.matcher, matcher)  # a new board: the matcher's records start again


class TheAgentLineOnThePhone(unittest.TestCase):
    def test_a_phone_knows_what_its_own_agent_runs_on_and_nothing_of_the_matchmaker(self):
        class Model:  # stands in for a model agent; never asked here
            def respond(self, *a, **k):
                raise AssertionError("not called")
        s = DemoSession(seed=7, agent=Model(), people=40, matchmaker=lambda request: {})
        self.assertEqual(s.phone_state("A")["agent_kind"], "model")
        s.llm = None  # a model that fell back to the rule-based agent (DemoSession.agent_status says why)
        self.assertEqual(s.phone_state("A")["agent_kind"], "rules")
        self.assertNotIn("atchmaker", json.dumps(s.phone_state("A")))


class TheDemoWithTheMatchmakerAI(unittest.TestCase):
    def test_a_stub_judges_the_phones_pairs_and_every_check_still_runs(self):
        asked = []

        def yes(request):
            asked.append(request)
            return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.95}
                              for c in request["candidates"]], "hunts_one_person": []}
        s = DemoSession(seed=7, agent=None, matchmaker=yes)
        s.run_inline = True
        climbers(s)
        s.run_matchmaker()
        self.assertTrue(asked)
        self.assertTrue(s.intros)
        state = s.room_state()
        self.assertTrue(state["leak_report"]["ok"], state["leak_report"]["findings"])
        self.assertTrue(state["matchmaker"]["ai"])
        self.assertGreater(state["matchmaker"]["decided"].get("judge", 0), 0)
        self.assertEqual(state["matchmaker_status"], "Matchmaker AI: a stand-in judge (no model)")
        ran = next(e for e in reversed(s.events) if e["title"] == "Matchmaker ran")
        self.assertIn("The Matchmaker AI judged", ran["text"])
        # The log says what goes to it, and that each request passed the Warden's check first.
        self.assertIn("sent only shown details and descriptions, and each request goes only once it passes the "
                      "Warden's check", ran["text"])
        self.assertEqual({r.result for r in s.warden.receipts if r.check == "judge_request"}, {"pass"})
        self.assertEqual(len([r for r in s.warden.receipts if r.check == "judge_request"]), len(asked))
        # Only the phones' listings are matched: every request is about one of them.
        phones = {"demoA-1", "demoB-1"}
        self.assertTrue(all(s.matcher.labels[i][r["focus"]] in phones for i, r in enumerate(s.matcher.payloads)))
        for slot in ("A", "B"):  # phones show nothing of it
            self.assertNotIn("atchmaker", json.dumps(s.phone_state(slot)))

    def test_a_judge_that_fails_leaves_the_rules_matches(self):
        def down(request):
            raise JudgeUnavailable("no connection or no answer in time")
        ruled, failing = DemoSession(seed=7, agent=None), DemoSession(seed=7, agent=None, matchmaker=down)
        for s in (ruled, failing):
            s.run_inline = True
            climbers(s)
            s.run_matchmaker()
        pairs = lambda s: sorted(sorted((i.person_a, i.person_b)) for i in s.intros.values())  # noqa: E731
        self.assertEqual(pairs(failing), pairs(ruled))
        self.assertGreater(failing.matchmaker_state()["decided"]["failed"], 0)

    def test_off_the_lock_its_pairs_wait_then_its_answers_come_in(self):
        """The model answers on a thread of its own: the run doesn't wait, its pairs wait (nobody is asked about
        close matches meanwhile), the answer counts as a change, and the next run uses it."""
        import threading
        release, calls = threading.Event(), []

        def slow(request):
            calls.append(request["focus"])
            release.wait(10)
            return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.95}
                              for c in request["candidates"]], "hunts_one_person": []}
        s = DemoSession(seed=7, agent=None, matchmaker=slow)
        s.run_inline = True
        climbers(s)
        s.run_inline = False  # from here on, as served
        before = s.changes
        started = time.monotonic()
        s.run_matchmaker()
        self.assertLess(time.monotonic() - started, 5)  # it didn't wait for the model
        judgeable = s.matchmaker_state()["decided"].get("waiting", 0)
        self.assertGreater(judgeable, 0)
        self.assertEqual([i for i in s.intros.values() if {i.person_a, i.person_b} & {"demoA-1", "demoB-1"}
                          and s.matcher.decisions.get(tuple(sorted((i.person_a, i.person_b))), {}).get("how") ==
                          "waiting"], [])
        self.assertFalse([q for st in s.slots.values() for l in st.listings for q in l.followups
                          if q["question_id"] == "broaden"])
        self.assertTrue(any(e["title"] == "Waiting for the Matchmaker AI" for e in s.events)
                        or s.intros)  # a run that sent nothing says it waits
        release.set()
        deadline = time.monotonic() + 10
        while s.matcher.judge.busy() and time.monotonic() < deadline:
            time.sleep(0.02)
        while s.changes == before and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertGreater(s.changes, before)  # the answer counts as a change: the Matchmaker runs again
        s.run_matchmaker()
        self.assertEqual(s.matchmaker_state()["decided"].get("waiting", 0), 0)
        self.assertGreater(s.matchmaker_state()["decided"].get("judge", 0), 0)
        self.assertTrue(s.room_state()["leak_report"]["ok"])


class ThePresentersPanel(_Page, unittest.TestCase):
    """demo.html, run with node: the panel says which Matchmaker runs and, with the AI, what its last run did."""

    def panel(self, state) -> str:
        out = self.run_page(f"""S = {json.dumps(state)}; renderRoom();
console.log(JSON.stringify({{ room: document.getElementById("room").innerHTML }}));""")
        return out["room"]

    def test_it_names_the_matchmaker_and_what_the_ai_did(self):
        def yes(request):
            return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.95}
                              for c in request["candidates"]], "hunts_one_person": []}
        s = DemoSession(seed=7, agent=None, matchmaker=yes)
        s.run_inline = True
        climbers(s)
        s.run_matchmaker()
        state = json.loads(json.dumps(s.state()))
        state["room"]["matchmaker"].update(usage={"answered": 2, "failed": 0, "prompt_tokens": 12345,
                                                  "completion_tokens": 678})
        room = self.panel(state)
        self.assertIn("<b>Matchmaker AI: a stand-in judge (no model)</b>.", room)
        self.assertIn("Last run: the Matchmaker AI judged", room)
        self.assertIn("Requests: 2 answered, 0 failed.", room)
        ruled = DemoSession(seed=7, agent=None, people=40, matchmaker=None)
        room = self.panel(json.loads(json.dumps(ruled.state())))
        self.assertIn("<b>Matchmaker: rules</b>.", room)
        self.assertNotIn("Last run: the Matchmaker AI", room)


class OffTheLockWhenTheModelFails(unittest.TestCase):
    def test_a_failure_off_the_lock_leaves_those_pairs_to_the_rules_for_a_while(self):
        from belong.demo import _JudgeOffLock
        calls, changes = [], []

        def down(request):
            calls.append(request["focus"])
            raise JudgeUnavailable("no connection or no answer in time")
        off = _JudgeOffLock(down, lambda: changes.append(1), lambda: False)
        request = {"focus": "c1", "candidates": ["c2"], "listings": {}}
        with self.assertRaises(JudgeWaiting):  # first: its pairs wait
            off(request)
        deadline = time.monotonic() + 10
        while (off.busy() or not changes) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertEqual((calls, changes), (["c1"], [1]))
        for _ in range(3):  # then the rules decide, without asking again, for FAILED_FOR seconds
            with self.assertRaises(JudgeUnavailable):
                off(request)
        self.assertEqual(calls, ["c1"])
        with mock.patch("belong.demo.time.monotonic", lambda: time.time() + 10 ** 9):
            with self.assertRaises(JudgeWaiting):  # asked again after that
                off(request)


class OffTheLockTheBoardAsItIsThen(unittest.TestCase):
    """A request waits in the queue while the model answers another: it is checked by the Warden against the board as
    it is when it is handed over, not as it was when it was made, and a reset drops the queue."""

    def test_a_request_that_waited_never_goes_once_it_is_no_longer_true(self):
        import threading
        for change in ("hide", "pause", "delete", "reset"):
            release, received = threading.Event(), []

            def slow(request):
                received.append((time.monotonic(), json.dumps(request, ensure_ascii=False)))
                release.wait(10)
                return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.95}
                                  for c in request["candidates"]], "hunts_one_person": []}
            s = DemoSession(seed=7, agent=None, matchmaker=slow)
            s.run_inline = True
            climbers(s)
            s.run_inline = False  # as served
            s.run_matchmaker()
            deadline = time.monotonic() + 10
            while not received and time.monotonic() < deadline:
                time.sleep(0.01)
            matcher = s.matcher
            b_said = {r["listings"][label]["description"] for r, labels in zip(matcher.payloads, matcher.labels)
                      for label, pid in labels.items() if pid == "demoB-1"}
            self.assertTrue(b_said, change)
            self.assertGreater(s._judge.busy(), 1, change)  # requests with B's listing in them, waiting their turn
            if change == "hide":
                s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD",
                                                               "me:interests": ["jazz", "ramen"],
                                                               "me:level": ["intermediate"]}))
            elif change == "pause":
                s.set_active("B", "demoB-1", False)
            elif change == "delete":
                s.delete_listing("B", "demoB-1")
            else:
                s.reset(7)
            acted = time.monotonic()
            release.set()
            while s._judge.busy() and time.monotonic() < deadline:
                time.sleep(0.01)
            late = [text for t, text in received if t > acted]
            self.assertFalse([text for text in late if any(said in text for said in b_said)], change)
            if change != "reset":
                self.assertGreater(s.matchmaker_state()["refused"], 0, change)
                self.assertIn("fail", {r.result for r in s.warden.receipts if r.check == "judge_request"})
            report = s.room_state()["leak_report"]
            self.assertTrue(report["ok"], (change, report["findings"]))

    def test_whatever_the_judge_raises_later_requests_are_still_asked(self):
        from belong.demo import _JudgeOffLock
        calls = []

        def judge(request):
            calls.append(request["focus"])
            if len(calls) == 1:
                raise SystemExit("gone")  # would end the thread
            return {"pairs": [], "hunts_one_person": []}
        off = _JudgeOffLock(judge, lambda: None, lambda: False)
        first, second = ({"focus": f, "candidates": [], "listings": {f: {}}} for f in ("c1", "c2"))
        deadline = time.monotonic() + 10
        for request in (first, second):
            with self.assertRaises(JudgeWaiting):
                off(request)
            while off.busy() and time.monotonic() < deadline:
                time.sleep(0.01)
        self.assertEqual(calls, ["c1", "c2"])
        with self.assertRaises(JudgeUnavailable):  # the first one's failure stands, for a while
            off(first)
        self.assertEqual(off(second), {"pairs": [], "hunts_one_person": []})


class BelongRun(unittest.TestCase):
    """`python3 -m belong run` stays on the rules, network-free; `--matcher ai` is the opt-in."""

    def run_cli(self, *args, judge_factory=None):
        out, err = io.StringIO(), io.StringIO()
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(cli, "OUT", Path(d) / "run.json"), mock.patch.object(cli, "ROOT", Path(d)), \
                mock.patch.object(cli, "load_dotenv"), \
                mock.patch.object(cli, "OpenAIJudge", judge_factory or self.never), \
                redirect_stdout(out), redirect_stderr(err):
            code = cli.main(["run", "--n", "60", *args])
            written = json.loads((Path(d) / "run.json").read_text()) if (Path(d) / "run.json").exists() else None
        return code, out.getvalue(), err.getvalue(), written

    @staticmethod
    def never(*a, **k):
        raise AssertionError("the Matchmaker AI was set up without --matcher ai")

    def test_the_default_is_the_rules(self):
        code, out, _, written = self.run_cli()
        self.assertEqual(code, 0)
        self.assertNotIn("Matchmaker AI", out)
        self.assertEqual(written["meta"]["matcher"], "OfflineMatcher")

    def test_matcher_ai_is_the_opt_in(self):
        j, stub, _ = judge(answering)
        code, out, _, written = self.run_cli("--matcher", "ai", "--ai-pairs", "12", judge_factory=lambda: j)
        self.assertEqual(code, 0)  # every leak audit still passes
        self.assertEqual(written["meta"]["matcher"], "LLMMatcher")
        self.assertLessEqual(written["meta"]["matchmaker_ai"]["sent"], 12)
        self.assertIn("Matchmaker AI: ", out)
        self.assertIn(f"requests: {j.usage['calls']}", out)
        self.assertGreater(len(stub.calls), 0)

    def test_matcher_ai_that_cant_be_set_up_says_why(self):
        def unavailable():
            raise JudgeUnavailable("OPENAI_API_KEY isn't set")
        code, _, err, _ = self.run_cli("--matcher", "ai", judge_factory=unavailable)
        self.assertEqual(code, 2)
        self.assertIn("OPENAI_API_KEY isn't set", err)

    def test_a_judge_that_says_yes_to_everything_leaks_nothing(self):
        def yes(request):
            return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 1.0}
                              for c in request["candidates"]], "hunts_one_person": request["candidates"][:0]}
        result = sim.run(n=120, seed=4, judge=yes, max_pairs=10_000)
        self.assertTrue(result["leak_report"]["ok"], result["leak_report"]["findings"])
        self.assertGreater(result["meta"]["matchmaker_ai"]["from_judge"], 0)


if __name__ == "__main__":
    unittest.main()
