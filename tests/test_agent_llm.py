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
# A value that isn't allowed: never saved.
INVALID = {**CLIMBER, "fields": CLIMBER["fields"] + [{"key": "level", "values": ["expert"], "visibility": SHOWN}]}


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

    def test_suggestions_are_kept_short_and_capped(self):
        args = dict(CLIMBER, suggestions=[{"label": "The problem", "ask": "The problem is"},
                                          {"label": "the problem", "ask": "dup"}, {"label": "x" * 99, "ask": "y"},
                                          {"label": "", "ask": "no label"}, "junk", {"label": "When", "ask": "I'm free (day and time):"},
                                          {"label": "Budget", "ask": "I can pay"}, {"label": "Fifth", "ask": "z"}])
        draft, _, _ = build_draft(args, "Yuki", "25-29")
        self.assertEqual([x["label"] for x in draft["suggestions"]], ["The problem", "x" * 40, "When", "Budget"])
        self.assertEqual(draft["suggestions"][0]["ask"], "The problem is ")
        self.assertEqual(draft["suggestions"][2]["ask"], "I'm free ")
        self.assertEqual(build_draft(CLIMBER, "Yuki", "25-29")[0]["suggestions"], [])

    def test_problems_are_reported_instead_of_saved(self):
        cases = {
            "reads like dating": {**CLIMBER, "intent_text": "Looking for a relationship in Osaka."},
            "specific person": {**CLIMBER, "intent_text": "Looking for a climbing partner, ideally @sakura_climbs."},
            "Don't put religion": {**CLIMBER, "intent_text": "Looking for a Buddhist climbing partner in Osaka."},
            "isn't an allowed value": INVALID,
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
        # Missing the offer: saved, with the offer on the to-ask list.
        draft, problems, _ = build_draft({**ren, "fields": ren["fields"][:2]}, "Ren", "25-29")
        self.assertTrue(any("offering" in m for m in draft["missing"]))

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

    def test_missing_details_are_saved_as_a_work_in_progress(self):
        """Save early: what isn't known yet goes on a to-ask list instead of blocking the save."""
        draft, problems, _ = build_draft(NO_PHRASES, "Yuki", "25-29")
        self.assertEqual(problems, [])
        self.assertFalse(draft["complete"])
        self.assertTrue(any("seeking" in m for m in draft["missing"]))
        draft, problems, _ = build_draft({**CLIMBER, "fields": []}, "Yuki", "25-29")
        self.assertEqual(problems, [])
        self.assertTrue(any("where they are" in m for m in draft["missing"]))

    def test_format_and_travel_take_their_fixed_choices(self):
        """Matching reads both, so the agent's wording is put into the fixed wording, and anything else
        goes back to the agent to fix."""
        from belong.agent_llm import UPDATE_LISTING_TOOL
        keys = UPDATE_LISTING_TOOL["input_schema"]["properties"]["fields"]["items"]["properties"]["key"]["enum"]
        self.assertIn("format", keys)
        self.assertIn("travel", keys)
        args = {**CLIMBER, "fields": CLIMBER["fields"] + [{"key": "format", "values": ["in-person"], "visibility": SHOWN},
                                                          {"key": "travel", "values": ["up to 30km"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(args, "Yuki", "25-29")
        self.assertEqual(problems, [])
        by_key = {f["key"]: f for f in draft["fields"]}
        self.assertEqual((by_key["format"]["value"], by_key["travel"]["value"]), ("In person", "Up to 30 km"))
        self.assertEqual(by_key["travel"]["options"], ["Same city", "Up to 30 km", "Up to 100 km"])
        bad = {**CLIMBER, "fields": CLIMBER["fields"] + [{"key": "travel", "values": ["Up to 50 km"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(bad, "Yuki", "25-29")
        self.assertIsNone(draft)
        self.assertTrue(any("travel" in p for p in problems))

    def test_a_format_in_other_words_is_put_into_a_fixed_choice_the_warden_accepts(self):
        """The Warden takes a format only from the fixed choices, so the agent's wording is read into one
        first (a negated online word means in person), and the reviewed draft passes the entry check."""
        from tests.test_demo import fill
        for wording, fixed in (("online only", "Online"), ("not online", "In person"),
                               ("in-person or remote", "In person or online"), ("only in person", "In person")):
            args = {**CLIMBER, "fields": CLIMBER["fields"] + [{"key": "format", "values": [wording], "visibility": SHOWN}]}
            draft, problems, _ = build_draft(args, "Yuki", "25-29")
            self.assertEqual(problems, [], wording)
            self.assertEqual({f["key"]: f["value"] for f in draft["fields"]}["format"], fixed, wording)
            s = DemoSession(seed=7, agent=None, people=50)
            s.verify("A", "Yuki", 27)
            lid = s.new_listing("A")
            s.submit("A", fill(draft), lid)
            self.assertTrue(s.slots["A"].listing(lid).entry["accepted"], (wording, s.slots["A"].listing(lid).entry))

    def test_a_listing_that_meets_online_needs_no_place(self):
        cv = {"kind": "other", "exactness": 0.6, "intent_text": "I review CVs and I'm looking for new clients.",
              "fields": [{"key": "seeking", "values": ["a CV review client"], "visibility": SHOWN},
                         {"key": "offering", "values": ["a CV reviewer"], "visibility": SHOWN},
                         {"key": "headline", "values": ["Offering CV reviews"], "visibility": SHOWN}]}
        online = {**cv, "fields": cv["fields"] + [{"key": "format", "values": ["Online"], "visibility": SHOWN}]}
        draft, _, _ = build_draft(online, "Kai", "25-29")
        self.assertFalse(any("where they are" in m for m in draft["missing"]))
        in_person = {**cv, "fields": cv["fields"] + [{"key": "format", "values": ["In person"], "visibility": SHOWN}]}
        draft, _, _ = build_draft(in_person, "Kai", "25-29")
        self.assertTrue(any("where they are" in m for m in draft["missing"]))

    def test_the_instructions_explain_how_and_where_people_meet(self):
        from belong.agent_llm import SYSTEM_PROMPT
        self.assertIn("travel", SYSTEM_PROMPT)
        self.assertIn("- travel (default shown): Same city, Up to 30 km, Up to 100 km", SYSTEM_PROMPT)
        self.assertIn("- format (default shown): In person, Online, In person or online", SYSTEM_PROMPT)

    def test_dating_needs_who_they_want_to_meet(self):
        args = {"kind": "dating", "exactness": 0.6, "intent_text": "Looking for a relationship in Lisbon, aged 30 to 39.",
                "fields": [{"key": "city", "values": ["Lisbon"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(args, "Inês", "30-34")
        self.assertTrue(any("who they want to meet" in m for m in draft["missing"]))


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
        client = ScriptedClient(reply(tool(INVALID, "t1")), reply(tool(CLIMBER, "t2")), reply(text("Done.")))
        _, draft = ClaudeAgent(client=client).respond([], "climbing in Osaka", "Yuki", "25-29")
        error = client.requests[1]["messages"][-1]["content"][0]
        self.assertTrue(error["is_error"])
        self.assertIn("isn't an allowed value", error["content"])
        self.assertEqual(draft["intent_text"], CLIMBER["intent_text"])

    def test_never_claims_a_change_that_was_not_saved(self):
        """The model says it updated the listing, but every save was refused."""
        client = ScriptedClient(*[reply(tool(INVALID, f"t{i}")) for i in range(5)])
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
        # Each turn: a question, then "nothing to save" when the app checks (SAVE_CHECK).
        client = ScriptedClient(*[reply(text(t)) for t in ("Tell me more?", "NO_SAVE") * 2])
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Yuki", 27)
        s.verify("B", "Ren", 29)
        s.agent_message("A", "I want a climbing partner. My secret is A-ONLY-123.")
        s.agent_message("B", "I want a roommate.")
        for b_request in client.requests[2:]:
            self.assertNotIn("A-ONLY-123", repr(b_request["messages"]) + b_request["system"])
        b_request = client.requests[2]
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
        post = ScriptedPost(oa_tool(INVALID, "c1"), oa_tool(CLIMBER, "c2"), oa_text("Done."))
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


class InstantProfile(unittest.TestCase):
    """The profile updates the moment a message is sent, before the agent replies."""

    def test_plain_facts_are_picked_up(self):
        """Only in words the agent recognises (TRUST.md #3): a place it knows, the fixed choices, known
        activities. A place or interest it doesn't know is left for the agent and the person to set."""
        from belong.agent import quick_facts
        self.assertEqual(quick_facts("I live in Osaka and I'm free on weekends. I'm a beginner."),
                         {"city": "Osaka", "country": "Japan", "schedule": "weekends", "level": "beginner"})
        self.assertEqual(quick_facts("I live in Galway and I'm free on weekends. I'm a beginner."),
                         {"schedule": "weekends", "level": "beginner"})
        self.assertEqual(quick_facts("i live in malaga, spain, looking to meet people"), {"country": "Spain"})
        self.assertEqual(quick_facts("based in Porto, looking to meet people")["city"], "Porto")
        self.assertEqual(quick_facts("yes, i enjoy chess and languages")["interests"], ["chess"])

    def test_how_and_how_far_they_meet_are_picked_up(self):
        from belong.agent import quick_facts
        self.assertEqual(quick_facts("Language exchange in Osaka, online is fine")["format"], "In person or online")
        self.assertEqual(quick_facts("I need a tutor in Lisbon or online")["format"], "In person or online")
        self.assertEqual(quick_facts("I review CVs, online only")["format"], "Online")
        self.assertEqual(quick_facts("happy to meet in person")["format"], "In person")
        self.assertEqual(quick_facts("I can travel up to 50 km")["travel"], "Up to 30 km")  # never further than they said
        self.assertEqual(quick_facts("anywhere within 20 miles")["travel"], "Up to 30 km")
        self.assertEqual(quick_facts("within 150km is fine")["travel"], "Up to 100 km")
        self.assertEqual(quick_facts("within 10 km")["travel"], "Same city")

    def test_online_only_takes_only_or_just(self):
        """Any other mention of online keeps meeting in person open; a negated one means in person."""
        from belong.agent import format_from_text
        cases = {
            "online works for me": "In person or online", "remote is fine": "In person or online",
            "happy to meet online": "In person or online", "in person or online": "In person or online",
            "online only": "Online", "only online": "Online", "just online": "Online", "remote only, please": "Online",
            "not online": "In person", "no online": "In person", "without online": "In person",
            "i don't want to meet online": "In person", "online doesn't work for me": "In person",
            "only in person": "In person", "in person only, online if i must": "In person",
            "not just online": "In person or online", "not only online": "In person or online",
            "i'm not sure yet, but online is fine": "In person or online", "weekends": None,
        }
        for said, fmt in cases.items():
            self.assertEqual(format_from_text(said), fmt, said)

    def test_kinds_without_a_format_never_get_one(self):
        from belong.agent import fill_gaps, quick_facts
        self.assertNotIn("format", quick_facts("Climbing partner in Osaka on weekends, online works for me"))
        self.assertEqual(quick_facts("Language exchange in Osaka, online works for me")["format"], "In person or online")
        for kind, added in (("climbing", False), ("running", False), ("roommate", False), ("dating", False),
                            ("language", True), ("tutoring", True), ("cofounder", True), ("other", True), (None, True)):
            draft = {"kind": kind, "fields": []}
            fill_gaps(draft, {"format": "In person or online", "schedule": "weekends"})
            keys = {f["key"] for f in draft["fields"]}
            self.assertEqual("format" in keys, added, kind)
            self.assertIn("schedule", keys)
        draft = {"kind": "climbing", "fields": [{"key": "format", "value": None}]}  # one the agent set is still filled
        self.assertEqual(fill_gaps(draft, {"format": "Online"}), ["format"])

    def test_a_format_noted_early_doesnt_stick_to_a_kind_without_one(self):
        """"Online works for me" before the kind is known is noted, but a climbing draft doesn't take it (the
        instant draft is the rule-based agent's: with a model agent there's none)."""
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Yuki", 27)
        s.prefill("A", "Online works for me")
        self.assertEqual({f["key"]: f["value"] for f in s.slots["A"].draft["fields"]}["format"], "In person or online")
        s.agent_message("A", "climbing partner in Osaka")
        self.assertEqual(s.slots["A"].draft["kind"], "climbing")
        self.assertNotIn("format", {f["key"] for f in s.slots["A"].draft["fields"]})

    def test_vague_messages_fill_nothing(self):
        from belong.agent import quick_facts
        for text in ["I would love to meet someone kind", "i live in the city", "hello"]:
            self.assertEqual(quick_facts(text), {}, text)

    def test_prefill_never_overwrites(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Yuki", 27)
        s.prefill("A", "I'm free on weekends and I'm into tennis")
        fields = {f["key"]: f["value"] for f in s.slots["A"].draft["fields"]}
        self.assertEqual((fields["schedule"], fields["interests"]), ("weekends", ["tennis"]))
        s.prefill("A", "evenings work too")  # already set: not overwritten
        self.assertEqual({f["key"]: f["value"] for f in s.slots["A"].draft["fields"]}["schedule"], "weekends")

    def test_with_a_model_agent_its_save_fills_the_profile(self):
        """No instant draft to hand it: the profile it builds on is only what it (or the person) saved."""
        client = ScriptedClient(reply(tool(CLIMBER)), reply(text("Saved.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Yuki", 27)
        s.prefill("A", "I'm free on weekends and I'm into tennis")
        self.assertIsNone(s.slots["A"].draft)
        s.agent_message("A", "I'm free on weekends and I'm into tennis. Climbing partner in Osaka.")
        fields = {f["key"]: f["value"] for f in s.slots["A"].draft["fields"]}
        self.assertEqual((fields["schedule"], fields["interests"]), ("weekends", ["bouldering", "ramen"]))

class SaveCheck(unittest.TestCase):
    """A turn that ends without the model trying to save is checked once (SAVE_CHECK): the profile only changes
    when it saves, so it saves what it learned, or answers that there's nothing to save, which leaves no trace."""

    def test_asked_once_more_it_saves_and_the_person_sees_its_reply(self):
        client = ScriptedClient(reply(text("Lovely. Which city are you in?")), reply(tool(CLIMBER)),
                                reply(text("Saved.")))
        conv = []
        message, draft = ClaudeAgent(client=client).respond(conv, "I'd like a climbing partner", "Yuki", "25-29")
        self.assertEqual(draft["kind"], "climbing")
        self.assertEqual(message, "Lovely. Which city are you in?")  # what it wrote after the check isn't shown
        self.assertIn("A check from the app", client.requests[1]["messages"][-1]["content"])

    def test_nothing_to_save_leaves_no_trace(self):
        client = ScriptedClient(reply(text("Hi! What are you looking for?")), reply(text("NO_SAVE")))
        conv = []
        message, draft = ClaudeAgent(client=client).respond(conv, "hi", "Yuki", "25-29")
        self.assertIsNone(draft)
        self.assertEqual(message, "Hi! What are you looking for?")
        self.assertEqual([m["role"] for m in conv], ["user", "assistant"])
        self.assertNotIn("NO_SAVE", message)

    def test_the_openai_agent_checks_the_same_way(self):
        post = ScriptedPost(oa_text("Nice. What's your budget?"), oa_tool(CLIMBER), oa_text("Saved."))
        message, draft = OpenAIAgent(api_key="test", post=post).respond([], "climbing partner please", "Yuki", "25-29")
        self.assertEqual((draft["kind"], message), ("climbing", "Nice. What's your budget?"))
        post = ScriptedPost(oa_text("Hello! What are you after?"), oa_text("NO_SAVE"))
        conv = []
        message, draft = OpenAIAgent(api_key="test", post=post).respond(conv, "hello", "Yuki", "25-29")
        self.assertEqual((draft, message, len(conv)), (None, "Hello! What are you after?", 2))
