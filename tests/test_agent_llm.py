"""The Claude-powered personal agent, tested with a scripted stand-in for the API
(no network, no credentials needed)."""
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace as NS

from belong.agent_llm import AgentUnavailable, ClaudeAgent, build_draft
from belong.demo import DemoSession
from belong.schema import HIDDEN, SHOWN


def text(t):
    return NS(type="text", text=t)


def tool(args, id_="t1"):
    return NS(type="tool_use", id=id_, name="update_listing", input=args)


def reply(*blocks, stop=None):
    stop = stop or ("tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn")
    return NS(stop_reason=stop, content=list(blocks))


CLIMBER = {
    "kind": "climbing", "exactness": 0.6,
    "intent_text": "Looking for a climbing partner in Osaka on weekends.",
    "fields": [
        {"key": "city", "values": ["Osaka"], "visibility": SHOWN},
        {"key": "schedule", "values": ["weekends"], "visibility": SHOWN},
        {"key": "level", "values": ["intermediate"], "visibility": SHOWN},
        {"key": "interests", "values": ["bouldering", "ramen"], "visibility": SHOWN},
        {"key": "religion", "values": ["Buddhist"], "visibility": SHOWN},  # sensitive: must end up hidden
        {"key": "seeking", "values": ["a climbing partner"], "visibility": SHOWN},
        {"key": "offering", "values": ["a climbing partner"], "visibility": SHOWN},
    ],
}
NO_PHRASES = {**CLIMBER, "fields": [f for f in CLIMBER["fields"] if f["key"] not in ("seeking", "offering")]}


class ScriptedClient:
    """Plays back responses in order and records every request it receives."""

    def __init__(self, *responses, error=None):
        self.responses = list(responses)
        self.requests = []
        self.error = error
        self.beta = NS(messages=NS(create=self._create))

    def _create(self, **request):
        self.requests.append(copy.deepcopy(request))
        if self.error:
            raise self.error
        return self.responses.pop(0)


class FailingAgent:
    def respond(self, conversation, *a):
        conversation.append({"role": "user", "content": "x"})
        raise AgentUnavailable("network down")


class Drafting(unittest.TestCase):
    def test_a_valid_draft_is_saved_and_sensitive_details_stay_hidden(self):
        draft, problems, notes = build_draft(CLIMBER, "Yuki", "25-29")
        self.assertEqual(problems, [])
        by_key = {f["key"]: f for f in draft["fields"]}
        self.assertEqual(by_key["level"]["value"], "intermediate")
        self.assertEqual(by_key["interests"]["value"], ["bouldering", "ramen"])
        self.assertEqual((by_key["religion"]["visibility"], by_key["religion"]["enabled"]), (HIDDEN, True))
        self.assertEqual(by_key["first_name"]["value"], "Yuki")  # from the ID check, not the agent
        self.assertTrue(any("religion" in n for n in notes))

    def test_problems_are_reported_instead_of_saved(self):
        cases = {
            "reads like dating": {**CLIMBER, "intent_text": "Looking for a relationship in Osaka."},
            "needs a 'seeking' phrase": NO_PHRASES,
            "specific person": {**CLIMBER, "intent_text": "Looking for a climbing partner, ideally @sakura_climbs."},
            "Don't put religion": {**CLIMBER, "intent_text": "Looking for a Buddhist climbing partner in Osaka."},
            "isn't an allowed value": {**CLIMBER, "fields": CLIMBER["fields"][:1] + [{"key": "level", "values": ["expert"], "visibility": SHOWN}]},
            "country is required": {**CLIMBER, "fields": []},
        }
        for expected, args in cases.items():
            draft, problems, _ = build_draft(args, "Yuki", "25-29")
            self.assertIsNone(draft, expected)
            self.assertTrue(any(expected in p for p in problems), (expected, problems))

    def test_open_ended_requests_get_their_own_kind(self):
        ren = {"kind": "other", "exactness": 0.6, "intent_text": "Looking for a guitarist for my ska band in Tokyo.",
               "fields": [{"key": "city", "values": ["Tokyo"], "visibility": SHOWN},
                          {"key": "seeking", "values": ["a ska guitarist"], "visibility": HIDDEN},
                          {"key": "offering", "values": ["a ska band"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(ren, "Ren", "25-29")
        self.assertEqual(problems, [])
        seeking = next(f for f in draft["fields"] if f["key"] == "seeking")
        self.assertEqual((seeking["value"], seeking["visibility"]), ("a ska guitarist", SHOWN))  # matching needs it
        # Phrases are never used for dating.
        _, problems, _ = build_draft({**ren, "kind": "dating"}, "Ren", "25-29")
        self.assertTrue(problems)
        # Missing the offer is refused.
        _, problems, _ = build_draft({**ren, "fields": ren["fields"][:2]}, "Ren", "25-29")
        self.assertTrue(any("offering" in p for p in problems))

    def test_any_place_in_the_world_works(self):
        fishing = {"kind": "other", "exactness": 0.6, "intent_text": "Looking for someone to teach me to fish in Ireland.",
                   "fields": [{"key": "country", "values": ["Ireland"], "visibility": SHOWN},
                              {"key": "seeking", "values": ["a fishing teacher"], "visibility": SHOWN},
                              {"key": "offering", "values": ["a fishing student"], "visibility": SHOWN},
                              {"key": "interests", "values": ["fishing", "hiking"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(fishing, "Aoife", "25-29")
        self.assertEqual(problems, [])
        by_key = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((by_key["country"], by_key["city"]), ("Ireland", None))
        self.assertEqual(by_key["interests"], ["fishing", "hiking"])  # free text, not a fixed list

    def test_nothing_the_agent_sets_is_dropped(self):
        """Ren's 'fluent French speaker' must survive on an open-ended listing."""
        ren = {"kind": "other", "exactness": 0.6, "intent_text": "Looking to meet new people in Malaga for chess.",
               "fields": [{"key": "city", "values": ["Malaga"], "visibility": SHOWN},
                          {"key": "country", "values": ["Spain"], "visibility": SHOWN},
                          {"key": "native_language", "values": ["French"], "visibility": SHOWN},
                          {"key": "seeking", "values": ["a chess and French conversation partner"], "visibility": SHOWN},
                          {"key": "offering", "values": ["a chess and French conversation partner"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(ren, "Ren", "25-29")
        self.assertEqual(problems, [])
        self.assertEqual({f["key"]: f["value"] for f in draft["fields"]}["native_language"], "French")
        tutoring = {**CLIMBER, "kind": "tutoring", "intent_text": "Need a chess tutor in Osaka."}
        draft, problems, _ = build_draft(tutoring, "Yuki", "25-29")
        self.assertEqual(problems, [])
        self.assertIn("seeking", {f["key"] for f in draft["fields"] if f["value"]})

    def test_dating_needs_who_they_want_to_meet(self):
        args = {"kind": "dating", "exactness": 0.6, "intent_text": "Looking for a relationship in Lisbon, aged 30 to 39.",
                "fields": [{"key": "city", "values": ["Lisbon"], "visibility": SHOWN}]}
        _, problems, _ = build_draft(args, "Inês", "30-34")
        self.assertTrue(any("who they want to meet" in p for p in problems))


class Conversation(unittest.TestCase):
    def test_agent_drafts_via_the_tool_and_replies(self):
        client = ScriptedClient(reply(text("Got it!"), tool(CLIMBER)), reply(text("Your draft is ready to review.")))
        conv = []
        message, draft = ClaudeAgent(client=client).respond(conv, "climbing partner in Osaka", "Yuki", "25-29")
        self.assertIn("Your draft is ready", message)
        self.assertEqual(draft["kind"], "climbing")
        # The tool result went back to the model, and the request used the expected settings.
        second = client.requests[1]
        self.assertEqual(second["messages"][-1]["content"][0]["type"], "tool_result")
        self.assertEqual(second["model"], "claude-opus-5-5")
        self.assertEqual(second["fallbacks"], "default")

    def test_a_rejected_draft_goes_back_to_the_model_to_fix(self):
        client = ScriptedClient(reply(tool(NO_PHRASES, "t1")), reply(tool(CLIMBER, "t2")), reply(text("Done.")))
        _, draft = ClaudeAgent(client=client).respond([], "climbing in Osaka", "Yuki", "25-29")
        error = client.requests[1]["messages"][-1]["content"][0]
        self.assertTrue(error["is_error"])
        self.assertIn("needs a 'seeking' phrase", error["content"])
        self.assertEqual(draft["intent_text"], CLIMBER["intent_text"])

    def test_never_claims_a_change_that_was_not_saved(self):
        """The model says it updated the listing, but every save was refused."""
        client = ScriptedClient(*[reply(tool(NO_PHRASES, f"t{i}")) for i in range(5)])
        message, draft = ClaudeAgent(client=client).respond([], "add that I'm a fluent French speaker", "Ren", "25-29")
        self.assertIsNone(draft)
        self.assertIn("wasn't able to save", message)

    def test_a_failed_turn_is_rolled_back(self):
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        client = ScriptedClient(error=AgentUnavailable("down"))
        client.beta.messages.create = lambda **kw: (_ for _ in ()).throw(AgentUnavailable("down"))
        with self.assertRaises(AgentUnavailable):
            ClaudeAgent(client=client).respond(conv, "anything", "Yuki", "25-29")
        self.assertEqual(len(conv), 2)


class InTheDemo(unittest.TestCase):
    def test_each_phone_has_its_own_private_conversation(self):
        """An agent only ever sees its own person's messages."""
        client = ScriptedClient(*[reply(text("Tell me more?")) for _ in range(4)])
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Yuki", 27)
        s.verify("B", "Ren", 29)
        s.agent_message("A", "I want a climbing partner. My secret is A-ONLY-123.")
        s.agent_message("B", "I want a roommate.")
        b_request = client.requests[1]
        self.assertNotIn("A-ONLY-123", repr(b_request["messages"]) + b_request["system"])
        self.assertIn("Ren", b_request["system"])
        self.assertEqual([m["from"] for m in s.phone_state("B")["chat"]], ["agent", "me", "agent"])
        self.assertNotIn("A-ONLY-123", repr(s.phone_state("B")))

    def test_falls_back_to_the_rule_based_agent_when_claude_is_unreachable(self):
        s = DemoSession(seed=7, agent=FailingAgent())
        s.verify("A", "Yuki", 27)
        s.agent_message("A", "Looking for a climbing partner in Osaka on weekends.")
        self.assertIsNotNone(s.slots["A"].draft)
        self.assertIn("network down", s.room_state()["agent_status"])
        self.assertFalse(s.slots["A"].agent_busy)


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------------------------
# OpenAI-powered agent (same checks, different API)
# ---------------------------------------------------------------------------------

import io
import json
import os
import urllib.error
from unittest import mock

from belong.agent_openai import OpenAIAgent


def oa_text(t):
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": t}}]}


def oa_tool(args, id_="c1", t=None):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": t,
        "tool_calls": [{"id": id_, "type": "function",
                        "function": {"name": "update_listing", "arguments": json.dumps(args)}}]}}]}


class ScriptedPost:
    def __init__(self, *responses, error=None):
        self.responses, self.requests, self.error = list(responses), [], error

    def __call__(self, url, payload, headers):
        self.requests.append(copy.deepcopy(payload))
        if self.error:
            raise self.error
        return self.responses.pop(0)


class OpenAIConversation(unittest.TestCase):
    def test_agent_drafts_via_the_tool_and_replies(self):
        post = ScriptedPost(oa_tool(CLIMBER, t="On it."), oa_text("Your draft is ready to review."))
        conv = []
        message, draft = OpenAIAgent(api_key="test", model="gpt-5", post=post).respond(conv, "climbing", "Yuki", "25-29")
        self.assertIn("Your draft is ready", message)
        self.assertEqual(draft["kind"], "climbing")
        tool_msg = post.requests[1]["messages"][-1]
        self.assertEqual((tool_msg["role"], tool_msg["tool_call_id"]), ("tool", "c1"))
        self.assertEqual(post.requests[0]["messages"][0]["role"], "system")
        self.assertTrue(post.requests[0]["tools"][0]["function"]["strict"])

    def test_a_rejected_draft_goes_back_to_the_model_to_fix(self):
        post = ScriptedPost(oa_tool(NO_PHRASES, "c1"), oa_tool(CLIMBER, "c2"), oa_text("Done."))
        _, draft = OpenAIAgent(api_key="test", post=post).respond([], "climbing", "Yuki", "25-29")
        self.assertTrue(post.requests[1]["messages"][-1]["content"].startswith("ERROR"))
        self.assertEqual(draft["intent_text"], CLIMBER["intent_text"])

    def test_a_rejected_key_rolls_back_and_explains(self):
        err = urllib.error.HTTPError("u", 401, "Unauthorized", {}, io.BytesIO(b'{"error": {"message": "bad key"}}'))
        conv = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        with self.assertRaises(AgentUnavailable) as ctx:
            OpenAIAgent(api_key="test", post=ScriptedPost(error=err)).respond(conv, "x", "Yuki", "25-29")
        self.assertIn("key was rejected", str(ctx.exception))
        self.assertEqual(len(conv), 2)

    def test_no_key_means_unavailable(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": ""}):
            with self.assertRaises(AgentUnavailable):
                OpenAIAgent()

    def test_demo_prefers_openai_when_a_key_is_set(self):
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BELONG_AGENT": "auto", "OPENAI_MODEL": ""}):
            s = DemoSession(seed=7)
        self.assertIsInstance(s.llm, OpenAIAgent)
        self.assertIn("OpenAI agent", s.room_state()["agent_status"])
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "BELONG_AGENT": "rules"}):
            self.assertIsNone(DemoSession(seed=7).llm)


class DotEnv(unittest.TestCase):
    def test_env_file_is_loaded_without_overriding(self):
        import tempfile
        from pathlib import Path
        from belong.__main__ import load_dotenv
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / ".env"
            path.write_text('# comment\nexport BELONG_TEST_A="from-file"\nBELONG_TEST_B=from-file\n')
            with mock.patch.dict(os.environ, {"BELONG_TEST_B": "already-set"}):
                load_dotenv(path)
                self.assertEqual(os.environ["BELONG_TEST_A"], "from-file")
                self.assertEqual(os.environ["BELONG_TEST_B"], "already-set")
            os.environ.pop("BELONG_TEST_A", None)
