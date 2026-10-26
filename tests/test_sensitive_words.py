"""A sensitive word is never written into anything shown (TRUST.md #3, #33). Neither agent puts a
religion, a sexual orientation, a health condition, politics or intimacy into an activity, a "looking
for" / "you are" phrase, a headline, topics, interests or a place: those are shown to matches. When
the word is one of a sensitive detail's choices, it is recorded there, hidden; a partner named only by
a sensitive word ("a Muslim partner") is asked about, never read as an activity (SPEC §4, §5)."""
from __future__ import annotations

import re
import unittest

from belong import agent
from belong.agent_llm import build_draft
from belong.demo import DemoSession
from belong.schema import FIELD_REGISTRY, HIDDEN, SHOWN
from tests.test_demo import fill

# (message, the sensitive word in it, the hidden detail it should become, if any)
CASES = [
    ("Looking for a Muslim partner in Berlin.", "muslim", ("religion", "Muslim")),
    ("Looking for a catholic partner in Lisbon.", "catholic", ("religion", "Christian")),
    ("Looking for a Christian hiking buddy in Tokyo.", "christian", ("religion", "Christian")),
    ("Looking for a gay tennis partner in London.", "gay", None),
    ("Looking for a queer partner in Berlin.", "queer", None),
    ("Looking for a diabetic walking buddy in Osaka.", "diabetic", None),
    ("Looking for a progressive book club buddy in Berlin.", "progressive", ("politics", "politically progressive")),
    ("I'm into Buddhism, BDSM and hiking. Looking for a chess partner in Osaka.", "bdsm", ("religion", "Buddhist")),
    ("I'm in recovery and looking for a running buddy in Osaka.", "recovery", None),
]


def shown_words(fields) -> str:
    """Everything a draft would show to a match, as one lower-case string."""
    out = []
    for f in fields:
        if f.get("value") in (None, "", []) or not f.get("enabled", True) or f.get("visibility") != SHOWN:
            continue
        if f["key"] in ("first_name", "age_band"):
            continue
        out += f["value"] if isinstance(f["value"], list) else [str(f["value"])]
    return " ".join(out).lower()


class TheOfflineAgent(unittest.TestCase):
    def test_a_sensitive_word_never_reaches_a_shown_detail(self):
        for message, word, detail in CASES:
            draft = agent.draft_from_message(message, "Sam", "30-34")
            self.assertNotIn(word, shown_words(draft["fields"]), message)
            self.assertFalse(agent.mentions_sensitive(shown_words(draft["fields"])), message)
            by = {f["key"]: f for f in draft["fields"]}
            if detail:
                key, value = detail
                f = by[key]
                self.assertEqual((f["value"] if not FIELD_REGISTRY[key].multi else f["value"][0], f["visibility"],
                                  f["enabled"]), (value, HIDDEN, True), message)
            self.assertTrue(any("sensitive" in n for n in draft["notes"]), message)  # and the person is told

    def test_a_partner_named_only_by_a_sensitive_word_is_asked_about(self):
        for message in ("Looking for a Muslim partner in Berlin.", "Looking for a gay partner in London.",
                        "Looking for a catholic partner in Lisbon."):
            self.assertIsNone(agent.activity_partner(message), message)
            question = agent.kind_question(message)
            self.assertIn("a partner for an activity (which one)", question["text"], message)
            self.assertIsNone(agent.draft_from_message(message, "Sam", "30-34")["kind"], message)  # no guessed kind
        # With an activity too, only the activity is kept.
        self.assertEqual(agent.activity_partner("a Christian hiking buddy"), "hiking")
        self.assertEqual(agent.activity_partner("a gay tennis partner"), "tennis")
        self.assertEqual(agent.kind_question("Hoping to meet a woman in Lisbon, and looking for a Jewish hiking buddy "
                                             "too.")["options"][1]["label"], "a hiking partner")

    def test_a_sensitive_word_in_an_answer_is_never_the_activity(self):
        options = agent.kind_question("Looking for a partner in Berlin.")["options"]
        for answer in ("both, I'm Muslim", "both, I'm gay", "both, I'm diabetic"):
            self.assertEqual(agent.chosen_kinds(answer, options), [{"kind": "dating", "activity": None},
                                                                   {"kind": None, "activity": None}], answer)
        self.assertIsNone(agent.chosen_kinds("a Christian partner", options))  # not an answer: asked again

    def test_in_the_demo_a_both_answer_with_a_sensitive_fact(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Sam", 33)
        s.agent_message("A", "Looking for a partner in Berlin.")
        s.agent_message("A", "both, I'm Muslim")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([(l["draft"] or {}).get("kind") for l in listings], ["dating", None])
        self.assertIn("What would you like a partner for?", listings[1]["chat"][-1]["text"])
        dating = {f["key"]: f for f in listings[0]["draft"]["fields"]}
        self.assertEqual((dating["religion"]["value"], dating["religion"]["visibility"]), ("Muslim", HIDDEN))
        self.assertNotIn("muslim", shown_words(listings[0]["draft"]["fields"]))
        # Saved, the live listing shows nothing sensitive, and the Warden's leak audit agrees.
        s.submit("A", fill(listings[0]["draft"], values={"gender": "man"}), listings[0]["id"])
        live = s.board.listing(listings[0]["id"])
        self.assertFalse(agent.mentions_sensitive(" ".join(" ".join(f.strings()) for f in live.fields
                                                           if f.visibility == SHOWN and f.key != "first_name")))
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_quick_facts_never_put_a_sensitive_word_in_a_shown_detail(self):
        self.assertEqual(agent.quick_facts("I'm into church choir, BDSM and hiking")["interests"], ["hiking"])
        facts = agent.quick_facts("I'm in recovery and I live in Osaka")
        self.assertEqual((facts.get("city"), facts.get("country")), ("Osaka", "Japan"))  # "recovery" is no place
        self.assertNotIn("offering", agent.quick_facts("I'm offering bible study lessons"))
        self.assertEqual(agent.quick_facts("I'm offering piano lessons")["offering"], "a piano teacher")

    def test_ordinary_words_are_not_sensitive(self):
        for text in ("a tennis partner", "climbing on weekends", "jazz and ramen", "a running buddy", "Osaka",
                     "a piano teacher", "board games", "a dating coach"):
            self.assertFalse(agent.mentions_sensitive(text), text)


class TheModelAgent(unittest.TestCase):
    def draft(self, extra):
        args = {"kind": "other", "exactness": 0.6, "intent_text": "Looking for a hiking partner in Tokyo.",
                "fields": [{"key": "city", "values": ["Tokyo"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a hiking partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a hiking partner"], "visibility": SHOWN},
                           {"key": "headline", "values": ["Looking for a hiking partner"], "visibility": SHOWN}]}
        keys = {f["key"] for f in extra}
        args["fields"] = [f for f in args["fields"] if f["key"] not in keys] + extra
        return build_draft(args, "Sam", "30-34")

    def test_a_shown_detail_with_a_sensitive_word_is_not_saved(self):
        for f in ({"key": "seeking", "values": ["a Muslim hiking partner"], "visibility": SHOWN},
                  {"key": "headline", "values": ["Looking for a gay hiking buddy"], "visibility": SHOWN},
                  {"key": "interests", "values": ["hiking", "church choir"], "visibility": SHOWN},
                  {"key": "topics", "values": ["diabetes"], "visibility": SHOWN}):
            draft, problems, _ = self.draft([f])
            self.assertIsNone(draft, f)
            self.assertTrue(any(p.startswith(f"'{f['key']}' is shown") for p in problems), problems)
        # The same words in a hidden detail are the person's choice, and the agent's draft is saved.
        draft, problems, _ = self.draft([{"key": "neighborhood", "values": ["near the mosque"], "visibility": HIDDEN}])
        self.assertEqual(problems, [])
        draft, problems, _ = self.draft([{"key": "religion", "values": ["Muslim"], "visibility": HIDDEN}])
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
