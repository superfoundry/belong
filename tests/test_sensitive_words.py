"""A sensitive word is never written into anything shown (TRUST.md #3, #33). The offline agent writes
a shown detail (an activity, a "looking for" / "you are" phrase, a headline, topics, interests or a
place) only from words it recognises, so nothing else a person types (sensitive or not) is ever shown;
when it doesn't recognise what they asked for, it asks. When the word is one of a sensitive detail's
choices, it is recorded there, hidden. The model agent's save is refused when a shown detail it wrote
has a sensitive word, a safeguard that can miss other wordings (SPEC §4, §5)."""
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


# The shown details whose values aren't one of a fixed list of choices.
FREE = ("seeking", "offering", "headline", "topics", "interests", "city", "country", "neighborhood", "employer",
        "subject", "pay")


def shown_words(fields, keys=None) -> str:
    """Everything a draft would show to a match (or only these keys), as one lower-case string."""
    out = []
    for f in fields:
        if f.get("value") in (None, "", []) or not f.get("enabled", True) or f.get("visibility") != SHOWN:
            continue
        if f["key"] in ("first_name", "age_band") or (keys and f["key"] not in keys):
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
                     "a piano teacher", "board games", "a dating coach", "a blind date", "easy recovery runs",
                     "straight after work", "a bi-lingual language exchange", "a straight-A maths tutor",
                     "a bilingual tutor", "recovery runs on Sundays", "going straight there"):
            self.assertFalse(agent.mentions_sensitive(text), text)
        # The same words where they say who someone is.
        for text in ("I'm bi", "I am straight", "I’m blind", "in recovery", "a straight man", "bi-curious",
                     "who is bi", "legally blind", "recovering alcoholic"):
            self.assertTrue(agent.mentions_sensitive(text), text)

    def test_every_word_it_recognises_is_ordinary(self):
        for activity in agent.KNOWN_ACTIVITIES:
            self.assertFalse(agent.mentions_sensitive(activity), activity)


# Words the offline agent has no rule for: sensitive ones (religion, gender identity, sexuality, health)
# and plain ones it simply doesn't know. Whatever they are, they never reach anything shown.
UNKNOWN = ("orthodox", "baptist", "sunni", "shia", "nonbinary", "genderqueer", "demisexual", "dyslexic", "hsv",
           "halal", "kosher", "ramadan", "chemo", "aa", "therapy", "addiction recovery", "pottery", "zorbling",
           "quilting circle", "spiritual healing")
TEMPLATES = ("Looking for a {} partner in Osaka.", "Looking for a {} buddy in Tokyo, weekends.",
             "I'm into {} and hiking. Looking for a chess partner in Berlin.",
             "I enjoy {}. Looking for a running buddy in Osaka.", "I'm in {} and looking for a tennis partner.",
             "I live in {} and want a climbing partner.", "I'm offering {} lessons in Lisbon.",
             "Looking for a {} coach in London.", "Hoping to meet a woman in Kyoto. I play {} most weekends.",
             "Looking for a relationship in Osaka. I'm into {}.", "I teach {} in Osaka, Japan.")


class OnlyWordsItRecognises(unittest.TestCase):
    """The offline agent never copies a person's own words into a shown detail: it writes only vocabulary
    it recognises, and asks when it doesn't recognise what they asked for (TRUST.md #3)."""

    def test_no_unknown_word_ever_reaches_a_shown_detail(self):
        for word in UNKNOWN:
            for template in TEMPLATES:
                message = template.format(word)
                said = re.compile(rf"\b{re.escape(word)}\b")
                draft = agent.draft_from_message(message, "Sam", "30-34")
                self.assertIsNone(said.search(shown_words(draft["fields"], FREE)), message)
                facts = agent.quick_facts(message)
                self.assertIsNone(said.search(" ".join(str(facts[k]) for k in FREE if k in facts).lower()), message)
                # Asked, the options it offers, and the listings "both" makes, say nothing it doesn't know.
                options = (agent.kind_question(message) or {}).get("options", [])
                for option in options:
                    self.assertIsNone(said.search(option["label"].lower()), message)
                for c in (agent.chosen_kinds("both", options) if options else None) or []:
                    if c.get("kind"):
                        again = agent.draft_from_message(message, "Sam", "30-34", kind=c["kind"],
                                                         activity=c.get("activity"))
                        self.assertIsNone(said.search(shown_words(again["fields"], FREE)), message)

    def test_a_partner_for_an_activity_it_doesnt_know_is_asked_about(self):
        for word in UNKNOWN:
            message = f"Looking for a {word} partner in Osaka."
            question = agent.kind_question(message)
            self.assertIsNotNone(question, message)
            self.assertTrue(any(o["kind"] is None for o in question["options"]), message)  # which activity?
            draft = agent.draft_from_message(message, "Sam", "30-34")
            self.assertIsNone(draft["kind"], message)
            self.assertFalse({"seeking", "offering", "headline", "topics"} & {f["key"] for f in draft["fields"]})
        self.assertTrue(agent.draft_from_message("Looking for a pottery partner in Osaka.", "Sam", "30-34")["notes"][0]
                        .startswith("What would you like a partner for?"))
        # A known activity is written as itself, and only that.
        draft = agent.draft_from_message("Looking for an Orthodox hiking buddy in Tokyo.", "Sam", "30-34")
        self.assertEqual({f["key"]: f["value"] for f in draft["fields"]}["seeking"], "a hiking partner")

    def test_the_answer_must_be_an_activity_it_knows(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Sam", 33)
        s.agent_message("A", "Looking for a pottery partner in Osaka.")
        chat = lambda: [m["text"] for m in s.phone_state("A")["listings"][0]["chat"]]
        self.assertTrue(chat()[-1].startswith("What would you like a partner for?"))
        self.assertIsNone(s.phone_state("A")["listings"][0]["draft"] and s.phone_state("A")["listings"][0]["draft"]
                          .get("kind"))
        for answer in ("pottery", "a pottery partner", "zorbling", "both", "yes"):
            s.agent_message("A", answer)
            self.assertIn("I only put activities I know on your listing", chat()[-1], answer)
            self.assertIsNone((s.phone_state("A")["listings"][0]["draft"] or {}).get("kind"), answer)
        s.agent_message("A", "board games")
        draft = s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["seeking"], values["city"]), ("other", "a board games partner", "Osaka"))
        self.assertNotIn("pottery", shown_words(draft["fields"]))

    def test_romantic_words_answer_which_activity_with_a_date(self):
        options = agent.kind_question("Looking for a pottery partner in Osaka.")["options"]
        partner = agent.kind_question("Looking for a partner in Berlin.")["options"]
        for answer in ("love", "a lover", "companionship", "to settle down", "someone to spend my life with",
                       "someone to love"):
            for opts in (options, partner):
                self.assertEqual(agent.chosen_kinds(answer, opts), [{"kind": "dating", "activity": None}], answer)
        self.assertIsNone(agent.chosen_kinds("not love", options))
        self.assertEqual(agent.chosen_kinds("I'd love a tennis partner", partner), [{"kind": "other", "activity": "tennis"}])

    def test_a_diet_that_says_something_about_religion_is_kept_hidden(self):
        for diet in ("halal", "kosher"):
            draft = agent.draft_from_message(f"I only eat {diet}. Looking for a cooking partner in Tokyo.", "Sam", "30-34")
            by = {f["key"]: f for f in draft["fields"]}
            self.assertEqual((by["diet"]["value"], by["diet"]["visibility"]), (diet, HIDDEN))
            self.assertNotIn(diet, shown_words(draft["fields"]))
        draft = agent.draft_from_message("I'm vegan. Looking for a cooking partner in Tokyo.", "Sam", "30-34")
        self.assertEqual({f["key"]: f["visibility"] for f in draft["fields"]}["diet"], SHOWN)


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

    def test_ordinary_words_are_saved(self):
        for f in ({"key": "headline", "values": ["Looking for a bi-lingual hiking buddy"], "visibility": SHOWN},
                  {"key": "seeking", "values": ["a hiking partner for easy recovery hikes"], "visibility": SHOWN},
                  {"key": "headline", "values": ["Hiking straight after work"], "visibility": SHOWN},
                  {"key": "interests", "values": ["blind wine tasting"], "visibility": SHOWN}):
            draft, problems, _ = self.draft([f])
            self.assertEqual(problems, [], f)
            self.assertIsNotNone(draft, f)


if __name__ == "__main__":
    unittest.main()
