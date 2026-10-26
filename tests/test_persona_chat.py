"""Simulated people reply in the demo (belong/persona_chat.py, DemoSession._persona_turn): a hello
when a real person matches with them, then answers to that person's messages, written only from what
they'd know. Never anyone's hidden details. Tests use a stub model (no network)."""
from __future__ import annotations

import unittest

from belong.demo import PERSONA_REPLIES, REACTIONS, DemoSession
from belong.schema import HIDDEN
from tests.test_demo import climbers


class StubModel:
    def __init__(self, reply="Sounds fun! Saturday?", react=""):
        self.calls, self.reply, self.react = [], reply, react

    def chat_json(self, system, messages, schema):
        self.calls.append((system, messages))
        return {"reply": self.reply, "react": self.react}


def connected_to_a_synthetic_person(s: DemoSession, model=None):
    """Phone A says yes to someone synthetic, and they say yes too. The model plays simulated people
    from then on (the listings are drafted by the rule-based agent first)."""
    climbers(s)
    s.run_matchmaker(quiet=True)
    s.llm = model
    for st in s.phone_state("A")["intros"]:
        intro = s.intros[st["intro_id"]]
        other = intro.person_b if intro.person_a == "demoA-1" else intro.person_a
        if s._slot_of(other) is None:
            s.decisions[(intro.intro_id, other)] = True
            s.warden.record_decision(intro.intro_id, other, True)  # they already said yes
            s.decide("A", intro.intro_id, True)
            return intro.intro_id, other
    raise AssertionError("A has no synthetic match")


class SimulatedReplies(unittest.TestCase):
    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.s.run_inline = True
        self.model = StubModel()

    def thread(self, intro_id):
        return next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["thread"]

    def test_they_say_hello_when_you_match(self):
        intro_id, _ = connected_to_a_synthetic_person(self.s, self.model)
        t = self.thread(intro_id)
        self.assertEqual([(m["mine"], m["text"]) for m in t], [(False, "Sounds fun! Saturday?")])
        self.assertEqual(len(self.model.calls), 1)

    def test_they_answer_your_message_and_may_react(self):
        intro_id, _ = connected_to_a_synthetic_person(self.s, self.model)
        self.model.reply, self.model.react = "Haha, me too.", "😂"
        self.s.send_message("A", intro_id, "I only climb when there's ramen after, haha")
        t = self.thread(intro_id)
        self.assertEqual(t[-1]["text"], "Haha, me too.")
        self.assertEqual(t[-2]["reactions"], [{"emoji": "😂", "mine": False}])
        state = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)
        self.assertFalse(state["typing"])
        self.assertTrue(state["simulated"])

    def test_the_model_never_sees_anyones_hidden_details(self):
        intro_id, other = connected_to_a_synthetic_person(self.s, self.model)
        self.s.send_message("A", intro_id, "Hi!")
        both = (self.s.board.listing("demoA-1"), self.s.board.listing(other))
        hidden = {v for listing in both for f in listing.fields if f.visibility == HIDDEN for v in f.strings()}
        shown = {v for listing in both for f in listing.fields if f.visibility != HIDDEN for v in f.strings()}
        hidden -= shown  # a value one of them shows openly (e.g. both don't smoke) may appear
        self.assertIn("Buddhist", hidden)
        for system, messages in self.model.calls:
            sent = system + " ".join(m["content"] for m in messages)
            for value in hidden:
                self.assertNotIn(value, sent)
        self.assertIn("Yuki", self.model.calls[-1][0])  # they know the name A shows

    def test_two_real_people_get_no_simulated_replies(self):
        climbers(self.s)
        self.s.run_matchmaker(quiet=True)
        self.s.llm = self.model
        intro_id = next(i["intro_id"] for i in self.s.phone_state("A")["intros"]
                        if {self.s.intros[i["intro_id"]].person_a, self.s.intros[i["intro_id"]].person_b} == {"demoA-1", "demoB-1"})
        self.s.decide("A", intro_id, True)
        self.s.decide("B", intro_id, True)
        self.s.send_message("A", intro_id, "Hi Ren!")
        self.assertEqual(self.model.calls, [])

    def test_without_a_model_a_short_built_in_reply(self):
        intro_id, _ = connected_to_a_synthetic_person(self.s, None)
        self.s.send_message("A", intro_id, "Thanks for saying yes!")
        t = self.thread(intro_id)
        self.assertFalse(t[-1]["mine"])
        self.assertEqual(t[-2]["reactions"], [{"emoji": "🙏", "mine": False}])
        self.assertTrue(set(r["emoji"] for m in t for r in m["reactions"]) <= set(REACTIONS))

    def test_a_model_error_still_gets_a_reply(self):
        class Broken:
            def chat_json(self, *a):
                raise RuntimeError("down")
        intro_id, _ = connected_to_a_synthetic_person(self.s, Broken())
        self.assertEqual(len(self.thread(intro_id)), 1)

    def test_replies_stop_after_a_while(self):
        intro_id, _ = connected_to_a_synthetic_person(self.s, self.model)
        for n in range(PERSONA_REPLIES + 3):
            self.s.send_message("A", intro_id, f"message {n}")
        self.assertEqual(len(self.model.calls), PERSONA_REPLIES)
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])


if __name__ == "__main__":
    unittest.main()


class OpenAIChatModel(unittest.TestCase):
    def test_a_smaller_model_writes_replies_and_the_main_one_steps_in(self):
        import io
        import json
        import urllib.error
        from belong.agent_openai import OpenAIAgent
        asked = []

        def post(url, payload, headers):
            asked.append(payload["model"])
            if payload["model"] == "gpt-5-mini":
                raise urllib.error.HTTPError(url, 404, "no", {}, io.BytesIO(b"{}"))
            return {"choices": [{"message": {"content": json.dumps({"reply": "Hi!", "react": ""})}}]}
        llm = OpenAIAgent(api_key="test", model="gpt-5", post=post)
        self.assertEqual(llm.chat_json("system", [], {}), {"reply": "Hi!", "react": ""})
        self.assertEqual(asked, ["gpt-5-mini", "gpt-5"])
