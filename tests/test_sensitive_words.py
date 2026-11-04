"""A sensitive word is never written into anything shown (TRUST.md #3, #33). The fixed sensitive words are
published in schema.py (SENSITIVE_WORDS), one list for the Warden, the agents and the audit. Sensitive
things go only in the hidden built-in categories (religion, politics, health, income_band, and kinks on
dating listings): the Warden refuses an open detail that says one, and points to the category. The
offline agent writes a shown detail only from vocabulary it recognises, so nothing else a person types
(sensitive or not) is ever shown; a sensitive word it does recognise is recorded, as said, hidden under its
category. The model agent's save is refused when a shown detail it wrote has a sensitive word, a safeguard
that can miss other wordings (SPEC §4, §5)."""
from __future__ import annotations

import re
import unittest

from belong import agent, schema
from belong.agent_llm import build_draft
from belong.demo import DemoSession
from belong.matching.marketboard import Marketboard
from belong.schema import HIDDEN, SHOWN, canonical
from tests.helpers import detail, fld, listing, person, warden
from tests.test_demo import fill

# (message, the sensitive word in it, the hidden built-in it is recorded under, if any)
CASES = [
    ("Looking for a Muslim partner in Berlin.", "muslim", "religion"),
    ("Looking for a catholic partner in Lisbon.", "catholic", "religion"),
    ("Looking for a Christian hiking buddy in Tokyo.", "christian", "religion"),
    ("Looking for a gay tennis partner in London.", "gay", None),  # who someone is: no detail of its own
    ("Looking for a queer partner in Berlin.", "queer", None),
    ("Looking for a diabetic walking buddy in Osaka.", "diabetic", "health"),
    ("Looking for a progressive book club buddy in Berlin.", "progressive", "politics"),
    ("I'm into Buddhism, BDSM and hiking. Looking for a chess partner in Osaka.", "bdsm", None),  # not dating
    ("I'm in recovery and looking for a running buddy in Osaka.", "recovery", "health"),
]
# The offline agent's vocabulary, the shared list of details in use: here, the activities the messages ask for,
# so it does write phrases and details from them (and still nothing sensitive).
WORDS = agent.words_in_use({
    "phrases": [{"key": "topics", "value": t} for t in ("chess", "climbing", "hiking", "running", "tennis", "walking")],
    "details": [{"about": "me", "name": "interests", "compare": "same"}],
    "values": [{"about": "me", "name": "interests", "value": v} for v in ("hiking", "chess")]})


def shown_words(fields, free_only=False) -> str:
    """Everything a draft would show to a match, as one lower-case string: built-in values and open
    details' names and values (free_only: leave out the fields with a fixed list of choices)."""
    fixed = {"first_name", "age_band", "kind", "format", "travel", "gender", "wants_gender", "wants_age"}
    out = []
    for f in fields:
        if f.get("value") in (None, "", []) and not f.get("values") or not f.get("enabled", True) \
                or f.get("visibility") != SHOWN:
            continue
        key = f.get("key") or ""
        if key in ("first_name", "age_band") or (free_only and key in fixed):
            continue
        value = f.get("values", f.get("value"))
        out += value if isinstance(value, list) else [str(value)]
        out += [f["name"]] if f.get("name") else [key.partition(":")[2]] if ":" in key else []
    return " ".join(out).lower()


class OneListForEveryone(unittest.TestCase):
    def test_the_agents_use_the_published_words(self):
        self.assertIs(agent.mentions_sensitive, schema.mentions_sensitive)

    def test_ordinary_words_are_not_sensitive(self):
        for text in ("a tennis partner", "climbing on weekends", "jazz and ramen", "a running buddy", "Osaka",
                     "a piano teacher", "board games", "a dating coach", "a blind date", "easy recovery runs",
                     "straight after work", "a bi-lingual language exchange", "a straight-A maths tutor",
                     "a bilingual tutor", "recovery runs on Sundays", "going straight there"):
            self.assertFalse(schema.mentions_sensitive(text), text)
        # The same words where they say who someone is.
        for text in ("I'm bi", "I am straight", "I’m blind", "in recovery", "a straight man", "bi-curious",
                     "who is bi", "legally blind", "recovering alcoholic"):
            self.assertTrue(schema.mentions_sensitive(text), text)

    def test_the_sensitive_categories_take_any_words_hidden(self):
        """Not only the synthetic people's choices: whatever a person says can be saved, hidden."""
        for key, value in (("religion", "Sikh"), ("religion", "Zoroastrian"), ("health", "asthma"),
                           ("health", "long covid"), ("politics", "green party member")):
            self.assertTrue(warden().admit(person(), listing(fields=[fld(key, value, HIDDEN)]), Marketboard()).accepted)


class TheWarden(unittest.TestCase):
    def admit(self, *details, kind="other"):
        l = listing(fields=[fld("kind", kind), *details])
        return warden().admit(person(), l, Marketboard())

    def test_a_diet_that_says_something_about_religion_goes_in_the_religion_detail(self):
        """halal and kosher once stayed hidden only by the offline agent's own rule; now no open detail can
        hold them, shown or hidden, and the Warden says where they belong."""
        for diet in ("halal", "Kosher"):
            for visibility in (SHOWN, HIDDEN):
                r = self.admit(detail("diet", diet, visibility=visibility))
                self.assertIn("listing_schema", r.failed_checks, diet)
                self.assertIn("hidden religion detail (religion)", r.problems[0])
        self.assertTrue(self.admit(detail("diet", "vegan")).accepted)

    def test_a_sensitive_word_in_an_open_detail_is_refused_with_its_category(self):
        cases = {"about me": ("Sikh", "religion"), "condition": ("asthma", "health"), "lean": ("socialist", "politics"),
                 "pay band": ("high salary", "income_band")}
        for name, (value, key) in cases.items():
            r = self.admit(detail(name, value))
            self.assertIn(f"({key})", r.problems[0], name)
        self.assertIn("only for dating", self.admit(detail("after dark", "bondage")).problems[0])
        self.assertIn("description only", self.admit(detail("about me", "bisexual")).problems[0])


class TheOfflineAgent(unittest.TestCase):
    def test_a_sensitive_word_never_reaches_a_shown_detail(self):
        for (message, word, key), words in [(case, w) for case in CASES for w in (None, WORDS)]:
            draft = agent.draft_from_message(message, "Sam", "30-34", words=words)
            self.assertNotIn(word, shown_words(draft["fields"]), message)
            self.assertFalse(schema.mentions_sensitive(shown_words(draft["fields"])), message)
            by = {f["key"]: f for f in draft["fields"]}
            if key:  # recorded as said, hidden, under its built-in category
                f = by[key]
                value = f["value"] if isinstance(f["value"], str) else " ".join(f["value"])
                self.assertIn(word, canonical(value), message)
                self.assertEqual((f["visibility"], f["enabled"]), (HIDDEN, True), message)
            self.assertNotIn("kinks", by, message)  # intimacy only on dating listings
            self.assertTrue(any("sensitive" in n for n in draft["notes"]), message)  # and the person is told

    def test_a_partner_named_only_by_a_sensitive_word_is_asked_about(self):
        for message in ("Looking for a Muslim partner in Berlin.", "Looking for a gay partner in London.",
                        "Looking for a catholic partner in Lisbon."):
            for words in (None, WORDS):
                self.assertIsNone(agent.activity_partner(message, words), message)
                question = agent.kind_question(message, words)
                self.assertIn("a partner for an activity (which one)", question["text"], message)
                self.assertIsNone(agent.draft_from_message(message, "Sam", "30-34", words=words)["kind"], message)
        # With a listed activity too, only the activity is kept.
        self.assertEqual(agent.activity_partner("a Christian hiking buddy", WORDS), "hiking")
        self.assertEqual(agent.activity_partner("a gay tennis partner", WORDS), "tennis")
        self.assertEqual(agent.kind_question("Hoping to meet a woman in Lisbon, and looking for a Jewish hiking buddy "
                                             "too.", WORDS)["options"][1]["label"], "a hiking partner")

    def test_a_sensitive_word_in_an_answer_is_never_the_activity(self):
        for words in (None, WORDS):
            options = agent.kind_question("Looking for a partner in Berlin.", words)["options"]
            for answer in ("both, I'm Muslim", "both, I'm gay", "both, I'm diabetic"):
                self.assertEqual(agent.chosen_kinds(answer, options, words), [{"kind": "dating", "activity": None},
                                                                              {"kind": None, "activity": None}], answer)
            self.assertIsNone(agent.chosen_kinds("a Christian partner", options, words))  # not an answer: asked again

    def test_in_the_demo_a_both_answer_with_a_sensitive_fact(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Sam", 33)
        s.agent_message("A", "Looking for a partner in Berlin.")
        s.agent_message("A", "both, I'm Muslim")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([(l["draft"] or {}).get("kind") for l in listings], ["dating", None])
        dating = {f["key"]: f for f in listings[0]["draft"]["fields"]}
        self.assertEqual((canonical(dating["religion"]["value"]), dating["religion"]["visibility"]), ("muslim", HIDDEN))
        self.assertNotIn("muslim", shown_words(listings[0]["draft"]["fields"]))
        # Saved, the live listing shows nothing sensitive, and the Warden's leak audit agrees.
        s.submit("A", fill(listings[0]["draft"], values={"gender": "man"}), listings[0]["id"])
        live = s.board.listing(listings[0]["id"])
        self.assertFalse(schema.mentions_sensitive(" ".join(" ".join(f.texts()) for f in live.fields
                                                            if f.visibility == SHOWN and f.key != "first_name")))
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_category_named_is_recorded_only_with_what_they_say_about_themselves(self):
        """A category's own name counts in free text only about someone (schema.SENSITIVE_WORDS): the offline agent
        records what the person says about themselves, without the "my", and nothing for the category's name
        alone or for someone else's. Either way it warns them, and nothing is shown."""
        for message, recorded in (
                ("Training to be a medical interpreter, looking for a study buddy in Graz.", {}),
                ("Hiring a dog groomer, salary negotiable.", {}),
                ("My earnings are private. Looking for a squash partner in Graz.", {}),
                ("Their income should be high. Looking for a date in Graz.", {}),
                ("My long-term health is fine. Looking for a squash partner in Graz.", {"health": "long-term health"}),
                ("Our joint income is modest. Looking for a date in Graz.", {"income_band": "joint income"})):
            self.assertEqual(agent.sensitive_details(message, "dating"), recorded, message)
            draft = agent.draft_from_message(message, "Sam", "30-34", words=WORDS)
            self.assertFalse(schema.mentions_sensitive(shown_words(draft["fields"])), message)
            if schema.mentions_sensitive(message):
                self.assertTrue(any("sensitive" in n for n in draft["notes"]), message)

    def test_a_category_of_someone_named_after_you_is_theirs(self):
        """After "my" or "our", a word that is itself a possessive ("<word>'s", "<word>s'") names someone else's:
        nothing is recorded as the person's own, and they are still warned."""
        for message in ("My landlord's income decides the rent, looking for a flatmate in Graz.",
                        "Our neighbours' health keeps us local. Looking for a chess partner in Graz.",
                        "My stepdad’s salary pays for the boat. Looking for a sailing crew in Graz."):
            self.assertEqual(agent.sensitive_details(message, "other"), {}, message)
            draft = agent.draft_from_message(message, "Sam", "30-34", words=WORDS)
            self.assertFalse([f for f in draft["fields"] if f.get("sensitive") and f.get("value")], message)
            self.assertTrue(any("sensitive" in n for n in draft["notes"]), message)
        # Their own, with what they say about it, is still recorded.
        self.assertEqual(agent.sensitive_details("My yearly income is modest.", "other"), {"income_band": "yearly income"})

    def test_a_sensitive_word_is_never_an_offline_word(self):
        """The Warden never lists one, and the offline agent leaves out any that got onto its list anyway."""
        words = agent.words_in_use({
            "phrases": [{"key": "topics", "value": "bible study"}, {"key": "seeking", "value": "a gay hiking partner"},
                        {"key": "topics", "value": "hiking"}],
            "details": [{"about": "me", "name": "interests", "compare": "same"}],
            "values": [{"about": "me", "name": "interests", "value": "church choir"}],
            "places": [{"city": "Sobriety Springs", "country": "Ireland"}]})
        self.assertEqual((words.activities, words.values), (("hiking",), {}))
        self.assertNotIn("sobriety springs", words.cities)
        draft = agent.draft_from_message("I'm into church choir. Looking for a bible study partner in Osaka.", "Sam",
                                         "30-34", words=words)
        self.assertFalse(schema.mentions_sensitive(shown_words(draft["fields"])))
        self.assertIsNone(draft["kind"])

    def test_quick_facts_never_put_a_sensitive_word_in_a_shown_detail(self):
        for message in ("I'm into church choir, BDSM and hiking", "I'm in recovery and I live in Osaka",
                        "I'm offering bible study lessons"):
            facts = agent.quick_facts(message, WORDS)
            self.assertFalse(schema.mentions_sensitive(" ".join(str(v) for v in facts.values())), message)
        facts = agent.quick_facts("I'm in recovery and I live in Osaka")
        self.assertEqual((facts.get("city"), facts.get("country")), ("Osaka", "Japan"))  # "recovery" is no place


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
    it recognises (the shared list of details in use and its fixed words), and asks when it doesn't
    recognise what they asked for (TRUST.md #3)."""

    def test_no_unknown_word_ever_reaches_a_shown_detail(self):
        for word in UNKNOWN:
            said = re.compile(rf"\b{re.escape(word)}\b")
            for template in TEMPLATES:
                message = template.format(word)
                draft = agent.draft_from_message(message, "Sam", "30-34", words=WORDS)
                self.assertIsNone(said.search(shown_words(draft["fields"], free_only=True)), message)
                facts = agent.quick_facts(message, WORDS)
                self.assertIsNone(said.search(" ".join(str(v) for k, v in facts.items()
                                                       if k not in ("kind", "format", "travel")).lower()), message)

    def test_no_option_or_both_answer_carries_an_unknown_word(self):
        """Asked, the options it offers, and the listings "both" makes, say nothing it doesn't know."""
        for word in UNKNOWN:
            said = re.compile(rf"\b{re.escape(word)}\b")
            for template in TEMPLATES:
                message = template.format(word)
                for words in (None, WORDS):
                    options = (agent.kind_question(message, words) or {}).get("options", [])
                    for option in options:
                        self.assertIsNone(said.search(option["label"].lower()), message)
                    for c in (agent.chosen_kinds("both", options, words) if options else None) or []:
                        if c.get("kind"):
                            again = agent.draft_from_message(message, "Sam", "30-34", kind=c["kind"],
                                                             activity=c.get("activity"), words=words)
                            self.assertIsNone(said.search(shown_words(again["fields"], free_only=True)), message)

    def test_the_answer_must_be_an_activity_on_the_list(self):
        """In the demo: an answer that names nothing listed is asked again and no kind is recorded; a listed
        activity is written as itself, and only that."""
        s = DemoSession(seed=7, agent=None)
        words = agent.words_in_use(s.details_in_use())
        s.verify("A", "Sam", 33)
        s.agent_message("A", "Looking for a pottery partner in Osaka.")
        chat = lambda: [m["text"] for m in s.phone_state("A")["listings"][0]["chat"]]
        kind = lambda: (s.phone_state("A")["listings"][0]["draft"] or {}).get("kind")
        self.assertTrue(chat()[-1].startswith("What would you like a partner for?"))
        self.assertIsNone(kind())
        for answer in ("pottery", "a pottery partner", "zorbling", "both", "yes"):
            s.agent_message("A", answer)
            self.assertIn(agent.activity_hint(words), chat()[-1], answer)
            self.assertIn("I only put activities other people already use on your listing", chat()[-1], answer)
            self.assertIsNone(kind(), answer)
            self.assertEqual(len(s.phone_state("A")["listings"]), 1, answer)
        listed = words.activities[0]
        s.agent_message("A", listed)
        draft = s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["seeking"], values["city"]), ("other", f"{agent._article(listed)} {listed} partner",
                                                                       "Osaka"))
        self.assertNotIn("pottery", shown_words(draft["fields"]))

    def test_a_request_it_doesnt_recognise_is_asked_about_not_written(self):
        for word in UNKNOWN:
            message = f"Looking for a {word} partner in Osaka."
            draft = agent.draft_from_message(message, "Sam", "30-34", words=WORDS)
            self.assertFalse({"seeking", "offering", "headline", "topics"} & {f["key"] for f in draft["fields"]}, message)
            self.assertTrue(draft["notes"], message)  # it asks
        # One it recognises is written, in its listed words only.
        values = {f["key"]: f["value"] for f in agent.draft_from_message("Looking for a hiking partner in Osaka.", "Sam",
                                                                         "30-34", words=WORDS)["fields"]}
        self.assertEqual((values["seeking"], values["topics"]), ("a hiking partner", ["hiking"]))

    def test_romantic_words_answer_which_activity_with_a_date(self):
        options = agent.kind_question("Looking for a pottery partner in Osaka.", WORDS)["options"]
        partner = agent.kind_question("Looking for a partner in Berlin.", WORDS)["options"]
        for answer in ("love", "a lover", "companionship", "to settle down", "someone to spend my life with",
                       "someone to love"):
            for opts in (options, partner):
                self.assertEqual(agent.chosen_kinds(answer, opts, WORDS), [{"kind": "dating", "activity": None}], answer)
        self.assertIsNone(agent.chosen_kinds("not love", options, WORDS))


class TheModelAgent(unittest.TestCase):
    def draft(self, extra, details=()):
        args = {"kind": "other", "exactness": 0.6, "intent_text": "Looking for a hiking partner in Tokyo.",
                "fields": [{"key": "city", "values": ["Tokyo"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a hiking partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a hiking partner"], "visibility": SHOWN},
                           {"key": "headline", "values": ["Looking for a hiking partner"], "visibility": SHOWN}],
                "details": list(details)}
        keys = {f["key"] for f in extra}
        args["fields"] = [f for f in args["fields"] if f["key"] not in keys] + extra
        return build_draft(args, "Sam", "30-34")

    def test_a_shown_detail_with_a_sensitive_word_is_not_saved(self):
        for f in ({"key": "seeking", "values": ["a Muslim hiking partner"], "visibility": SHOWN},
                  {"key": "headline", "values": ["Looking for a gay hiking buddy"], "visibility": SHOWN},
                  {"key": "topics", "values": ["hiking", "church choir"], "visibility": SHOWN},
                  {"key": "topics", "values": ["diabetes"], "visibility": SHOWN}):
            draft, problems, _ = self.draft([f])
            self.assertIsNone(draft, f)
            self.assertTrue(any(p.startswith(f"'{f['key']}' is shown") for p in problems), problems)
        # The same words in a hidden built-in are the person's choice, and the agent's draft is saved.
        for f in ({"key": "neighborhood", "values": ["near the mosque"], "visibility": HIDDEN},
                  {"key": "religion", "values": ["Muslim"], "visibility": HIDDEN},
                  {"key": "religion", "values": ["Sikh"], "visibility": HIDDEN}):
            draft, problems, _ = self.draft([f])
            self.assertEqual(problems, [], f)

    def test_a_category_made_about_someone_never_reaches_a_shown_phrase(self):
        """A category's own name in a shown phrase, made about someone by a possessive (through any number of
        words), a first-person form in the same clause, or as all the phrase says, is refused before saving."""
        from tests.test_open_details import SensitiveWords
        for text in SensitiveWords.ABOUT_SOMEONE:
            draft, problems, _ = self.draft([{"key": "seeking", "values": [text], "visibility": SHOWN}])
            self.assertIsNone(draft, text)
            self.assertTrue(any(p.startswith("'seeking' is shown") for p in problems), (text, problems))
        for text in SensitiveWords.EVERYDAY:
            draft, problems, _ = self.draft([{"key": "seeking", "values": [text], "visibility": SHOWN}])
            self.assertFalse(any(p.startswith("'seeking' is shown") for p in problems), (text, problems))

    def test_an_open_detail_with_a_sensitive_word_is_not_saved(self):
        """Shown or hidden: sensitive things go only in the hidden categories, and the agent is told which."""
        for d, key in (({"name": "diet", "about": "me", "compare": "same", "values": ["halal"], "visibility": SHOWN},
                        "religion"),
                       ({"name": "condition", "about": "me", "compare": "same", "values": ["asthma"],
                         "visibility": HIDDEN}, "health")):
            draft, problems, _ = self.draft([], [d])
            self.assertIsNone(draft, d)
            self.assertTrue(any(f"({key})" in p for p in problems), problems)

    def test_ordinary_words_are_saved(self):
        for f in ({"key": "headline", "values": ["Looking for a bi-lingual hiking buddy"], "visibility": SHOWN},
                  {"key": "seeking", "values": ["a hiking partner for easy recovery hikes"], "visibility": SHOWN},
                  {"key": "headline", "values": ["Hiking straight after work"], "visibility": SHOWN},
                  {"key": "topics", "values": ["blind wine tasting"], "visibility": SHOWN}):
            draft, problems, _ = self.draft([f])
            self.assertEqual(problems, [], f)
            self.assertIsNotNone(draft, f)


if __name__ == "__main__":
    unittest.main()
