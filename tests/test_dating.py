"""Dating, understood (SPEC §5): ordinary ways of asking for a date are read as dating and services
are not, ages written as decades are understood, a request with no age range is open to people
near your own age, and your own gender is asked for and picked up, never forced to be shown."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from belong import agent
from belong.agent_llm import build_draft
from belong.demo import DemoSession
from belong.matching.intent import ages_text, default_age_range, own_gender, parse_intent
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import OfflineMatcher, intent_met
from belong.matching.narrow import ShownView
from belong.schema import HIDDEN, SHOWN, Field, Listing, Person
from belong.sim import run
from belong.synth import generate
from tests.helpers import warden
from tests.test_demo import fill


class DatingWording(unittest.TestCase):
    def test_ordinary_ways_of_asking_for_a_date_are_dating(self):
        cases = {
            "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz.": ("man",),
            "I want to meet a woman in Berlin who loves hiking.": ("woman",),
            "Hoping to find a nice guy in London, late twenties.": ("man",),
            "Looking for a boyfriend in Tokyo.": ("man",),
            "Searching for a girlfriend in Lisbon, mid 30s.": ("woman",),
            "Looking for a husband in Kyoto.": ("man",),
            "Looking for a wife in Kyoto.": ("woman",),
            "Looking for someone special in Osaka.": (),
            "Looking for love in Tokyo, a woman around 30.": ("woman",),
            "Single guy looking for single women in London.": ("woman",),
            "Looking for a romantic connection with men in Berlin.": ("man",),
            "Interested in women in Kyoto, under 35.": ("woman",),
            "Looking for a partner in Osaka, anyone welcome.": ("man", "nonbinary", "woman"),
            "Looking for a woman to share my life with in Tokyo.": ("woman",),
            "Looking to meet a man for a drink in Berlin.": ("man",),
            "Looking for a long-term partner with women in Osaka.": ("woman",),
            "Looking for my future wife in Kyoto.": ("woman",),
        }
        for text, seeking in cases.items():
            p = parse_intent(text)
            self.assertEqual((p.kind, p.seeking), ("dating", seeking), text)

    def test_services_things_and_activities_are_not_dating(self):
        for text in [
            "Looking for a man with a van to help me move in London.",
            "Need a man to fix my sink in Osaka.",
            "Looking for a woman to clean my flat every week.",
            "Looking for a guy who can repair my bike.",
            "Looking for a girl to babysit my kids on weekends.",
            "Looking for a man to paint my living room.",
            "Looking for a woman to teach me piano.",
            "Looking for women to join our book club in Kyoto.",
            "Looking for women for our football team in Berlin.",
            "Selling my husband's golf clubs in Berlin.",
            "Giving away my girlfriend's old desk, free to anyone.",
            "Selling a romantic dinner voucher in Lisbon.",
            "Looking for a dating coach in London.",
            "Looking for a men's barber in Tokyo.",
            "Looking for a chess partner in Tokyo, men or women welcome.",
            "Looking for a business partner for my bakery.",
            "Looking for a partner to play tennis with.",
            "My partner and I are looking for a flat to rent in London.",
            "Looking for a gift for my wife.",
            "My partner and I want to meet new people in Osaka.",
            "Need a room in London, move-in date November.",
            "I'm a man with a van in London, looking for new customers.",
        ]:
            self.assertNotEqual(parse_intent(text).kind, "dating", text)

    def test_functional_kinds_still_come_first(self):
        cases = {
            "Looking for a climbing partner in Osaka, a man or a woman.": "climbing",
            "Want to meet a woman to run with on weekends.": "running",
            "Looking for a female maths tutor in Kyoto.": "tutoring",
            "Looking for a flatmate in London, men or women.": "roommate",
        }
        for text, kind in cases.items():
            self.assertEqual(parse_intent(text).kind, kind, text)

    def test_what_you_say_about_yourself_is_not_who_you_want(self):
        self.assertEqual(parse_intent("I'm a woman looking for a man in Lisbon.").seeking, ("man",))
        self.assertEqual(parse_intent("Single woman, 34, looking for a kind man in Osaka.").seeking, ("man",))
        self.assertEqual(parse_intent("As a gay man I'd like to meet men in Berlin for dating.").seeking, ("man",))
        self.assertEqual(parse_intent("Looking for a relationship in Kyoto, my ex-husband moved away.").seeking, ())

    def test_the_shown_headline_is_read_with_the_description(self):
        text = "I'd like to meet someone kind in Osaka who likes jazz."
        self.assertIsNone(parse_intent(text).kind)
        self.assertEqual(parse_intent(text, "Looking for a long-term partner").kind, "dating")
        # The synthetic people's headlines never change what their descriptions say.
        for p in generate(300, 11):
            if not p.truth.bad_intent:
                expected = None if p.truth.kind in ("other", "service", "item") else p.truth.kind
                self.assertEqual(parse_intent(p.intent_text, p.values.get("headline", "")).kind, expected, p.intent_text)

    def test_the_matcher_reads_the_headline_only_when_it_is_shown(self):
        def pairs(headline_visibility):
            w, board = warden(), Marketboard()
            people = [("pa", "Men in Osaka who like jazz, aged 25 to 34.", "Looking for a long-term partner"),
                      ("pb", "Looking for a relationship with men in Osaka, aged 25 to 34.", "")]
            for pid, intent, headline in people:
                fields = [Field("city", "Osaka", SHOWN, False), Field("country", "Japan", SHOWN, False),
                          Field("gender", "man", SHOWN, False), Field("age_band", "25-29", SHOWN, False),
                          Field("interests", ("jazz",), SHOWN, False), Field("schedule", "weekends", SHOWN, False)]
                if headline:
                    fields.append(Field("headline", headline, headline_visibility, False))
                listing = Listing(pid, intent, 0.6, tuple(fields), "t")
                self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), listing, board).accepted)
            return {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(board).proposals}
        self.assertEqual(pairs(SHOWN), {frozenset(("pa", "pb"))})
        self.assertEqual(pairs(HIDDEN), set())  # a hidden headline is never read

    def test_a_service_is_not_refused_as_dating_by_the_agent_checks(self):
        args = {"kind": "other", "exactness": 0.6, "intent_text": "Need a man with a van to help me move in London.",
                "fields": [{"key": "city", "values": ["London"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a removals van driver"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a removals customer"], "visibility": SHOWN},
                           {"key": "headline", "values": ["Looking for a man with a van"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(args, "Sam", "30-34")
        self.assertEqual(problems, [])
        self.assertIsNotNone(draft)


class DecadeAges(unittest.TestCase):
    def ages(self, text):
        return parse_intent(f"Looking for a man in Osaka, {text}.").age_range

    def test_decades_and_rough_ages_map_onto_age_bands(self):
        # AGE_BANDS: 0 18-24, 1 25-29, 2 30-34, 3 35-39, 4 40-49, 5 50+
        cases = {
            "early 30s": (2, 2), "mid 30s": (2, 3), "late 30s": (3, 3), "mid-thirties": (2, 3),
            "in their 30s": (2, 3), "30s": (2, 3), "thirties": (2, 3), "late twenties": (1, 1),
            "30-something": (2, 3), "around 30": (1, 2), "about 45": (4, 4), "early 20s": (0, 0),
            "in his 40s": (4, 4), "50s": (5, 5), "late 20s to early 30s": (1, 2),
            # The ways already understood still are.
            "aged 25 to 34": (1, 2), "between 30 and 39": (2, 3), "roughly 30-39": (2, 3),
            "aged 40 or older": (4, 5), "30+": (2, 5), "under 35": (0, 2),
        }
        for text, expected in cases.items():
            self.assertEqual(self.ages(text), expected, text)
        # A follow-up answer, added to the end of a description.
        self.assertEqual(parse_intent("I'd like to meet women in Osaka for someone to date. Aged 25 to 39.").age_range,
                         (1, 3))

    def test_your_own_age_is_not_the_age_you_want(self):
        self.assertIsNone(parse_intent("I'm 34 and looking for a man in Osaka.").age_range)
        self.assertIsNone(parse_intent("Single woman, 34, looking for a kind man in Osaka.").age_range)
        self.assertEqual(parse_intent("I'm in my early 30s, looking for a man in his 40s.").age_range, (4, 4))


def dating_listing(pid, intent, gender, band, exactness=0.6, gender_visibility=SHOWN, band_shown=True):
    fields = [Field("first_name", pid, SHOWN, False), Field("city", "Osaka", SHOWN, False),
              Field("country", "Japan", SHOWN, False), Field("gender", gender, gender_visibility, False),
              Field("interests", ("jazz",), SHOWN, False), Field("schedule", "weekends", SHOWN, False)]
    if band_shown:
        fields.append(Field("age_band", band, SHOWN, False))
    return Listing(pid, intent, exactness, tuple(fields), "t")


def met(a: Listing, b: Listing) -> bool:
    """Is a's intent met by b?"""
    return intent_met(parse_intent(a.intent_text), ShownView(a), parse_intent(b.intent_text), ShownView(b))


class NoAgeRangeSaid(unittest.TestCase):
    """Something you didn't say is not a strict preference (the climbing and running rule): with no age
    range, you're open to one age band either side of your own."""

    # Like the synthetic listing in the reported case: no age range at all.
    NO_AGES = "I'd like to meet women in Osaka for someone to date."

    def test_the_default_is_one_band_either_side(self):
        self.assertEqual(default_age_range("30-34"), (1, 3))
        self.assertEqual(default_age_range("18-24"), (0, 1))
        self.assertEqual(default_age_range("50+"), (4, 5))
        self.assertEqual(default_age_range(None), (0, 5))  # no shown age band: any adult
        self.assertEqual((ages_text((1, 3)), ages_text((4, 5))), ("25 to 39", "40 or older"))

    def test_a_listing_with_no_age_range_matches_people_near_its_own_age(self):
        him = dating_listing("him", self.NO_AGES, "man", "30-34")
        for band, ok in [("25-29", True), ("30-34", True), ("35-39", True), ("18-24", False), ("40-49", False)]:
            her = dating_listing("her", "Looking for a relationship with a man in Osaka, aged 18 to 99.", "woman", band)
            self.assertEqual(met(him, her), ok, band)

    def test_both_ways_with_the_reported_request(self):
        him = dating_listing("him", self.NO_AGES, "man", "30-34")
        her = dating_listing("her", "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz.",
                             "woman", "30-34")
        self.assertTrue(met(him, her) and met(her, him))
        # Her request names an age range, so it is strict about it: a man of 40 isn't what she asked for.
        self.assertFalse(met(her, dating_listing("older", self.NO_AGES, "man", "40-49")))

    def test_exactness_slack_still_applies(self):
        her = dating_listing("her", "Looking for a relationship with a man in Osaka, aged 18 to 99.", "woman", "40-49")
        self.assertFalse(met(dating_listing("him", self.NO_AGES, "man", "30-34", exactness=0.6), her))
        self.assertTrue(met(dating_listing("him", self.NO_AGES, "man", "30-34", exactness=0.3), her))

    def test_no_shown_age_band_means_any_adult(self):
        him = dating_listing("him", self.NO_AGES, "man", "30-34", band_shown=False)
        for band in ("18-24", "50+"):
            self.assertTrue(met(him, dating_listing("her", "Looking for a man in Osaka for dating.", "woman", band)))

    def test_people_who_opted_in_are_still_asked_to_narrow_it(self):
        w, board = warden(), Marketboard()
        from dataclasses import replace
        listing = replace(dating_listing("him", self.NO_AGES, "man", "30-34"), notify_questions=True)
        self.assertTrue(w.admit(Person("him", True, "30-34", "t"), listing, board).accepted)
        followups = OfflineMatcher().match(board).followups
        self.assertIn("age_range", [f.question_id for f in followups if f.person_id == "him"])

    def test_the_batch_run_stays_precise(self):
        """Truth keeps each synthetic person's real range; people who didn't state one and weren't
        asked are matched by the default, and proposals stay right by that truth."""
        result = run(n=300, seed=7)
        dating = result["metrics"]["by_kind"]["dating"]
        self.assertGreaterEqual(dating["precision"], 0.95)
        self.assertGreaterEqual(result["metrics"]["precision"], 0.95)
        self.assertGreaterEqual(result["metrics"]["coverage"], 0.85)


class YourGender(unittest.TestCase):
    def test_a_hidden_gender_never_decides_but_anyone_can_still_match_you(self):
        wants_a_man = dating_listing("a", "Looking for a relationship with a man in Osaka, aged 25 to 39.", "woman", "30-34")
        open_to_anyone = dating_listing("b", "Looking for a relationship with anyone in Osaka, aged 25 to 39.",
                                        "woman", "30-34")
        hidden = dating_listing("c", "Looking for a relationship with a woman in Osaka, aged 25 to 39.", "man", "30-34",
                                gender_visibility=HIDDEN)
        self.assertFalse(met(wants_a_man, hidden))  # his hidden gender can't count for her request
        self.assertTrue(met(open_to_anyone, hidden))  # she doesn't need to know it

    def test_own_gender_is_picked_up_from_what_people_type(self):
        cases = {"I'm a woman": "woman", "I am non-binary": "nonbinary", "I'm nonbinary": "nonbinary",
                 "As a 34 year old gay man, I'd like...": "man", "I'm female": "woman", "I'm a straight guy": "man",
                 "I'm looking for a man": None, "I'm into jazz": None}
        for text, gender in cases.items():
            self.assertEqual(own_gender(text), gender, text)
            self.assertEqual(agent.quick_facts(text).get("gender"), gender, text)

    def test_the_rule_based_agent_asks_for_gender_and_says_why(self):
        draft = agent.draft_from_message("Looking to meet a man in Osaka, early 30s, who's into jazz.", "Mika", "30-34")
        self.assertEqual(draft["kind"], "dating")
        self.assertTrue(any("people looking for a woman or a man can find you" in n for n in draft["notes"]))
        draft = agent.draft_from_message("I am non-binary, looking for a relationship with anyone in Kyoto.", "Rei", "25-29")
        self.assertEqual({f["key"]: f["value"] for f in draft["fields"]}["gender"], "nonbinary")
        self.assertTrue(any("you're open to 18 to 34" in n for n in draft["notes"]))  # no age range said

    def test_the_model_agent_is_asked_for_gender_and_ages_and_hidden_stays_hidden(self):
        from belong.agent_llm import SYSTEM_PROMPT
        self.assertIn("their own gender and the age range they want", SYSTEM_PROMPT)
        args = {"kind": "dating", "exactness": 0.6,
                "intent_text": "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz.",
                "fields": [{"key": "city", "values": ["Osaka"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(args, "Mika", "30-34")
        self.assertEqual(problems, [])
        self.assertTrue(any(m.startswith("their gender") for m in draft["missing"]))
        self.assertFalse(any("age range" in m for m in draft["missing"]))  # "early 30s" is one
        args["fields"].append({"key": "gender", "values": ["woman"], "visibility": HIDDEN})
        draft, problems, notes = build_draft(args, "Mika", "30-34")
        self.assertEqual(next(f["visibility"] for f in draft["fields"] if f["key"] == "gender"), HIDDEN)
        self.assertTrue(any("fewer matches" in n for n in notes))
        args["intent_text"] = "Looking to meet a man in Osaka who's into jazz."
        draft, _, _ = build_draft(args, "Mika", "30-34")
        self.assertTrue(any("open to 25 to 39" in m for m in draft["missing"]))


PAGE = (Path(__file__).resolve().parent.parent / "belong" / "web" / "demo.html").read_text()


def _js(name: str) -> str:
    """One top-level function or constant from the phone page's script, by brackets."""
    start = PAGE.index(name)
    i = PAGE.index("{", start) if name.startswith("function") else start
    depth, seen = 0, False
    while True:
        ch = PAGE[i]
        depth += ch in "{[(" and 1 or 0
        depth -= ch in "}])" and 1 or 0
        seen = seen or ch in "{[("
        i += 1
        if seen and depth == 0:
            return PAGE[start:PAGE.find("\n", i) if not name.startswith("function") else i]


class Checklist(unittest.TestCase):
    """The profile's checklist on the phone (demo.html), run with node."""

    def run_checklist(self, cases):
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        code = "\n".join(_js(n) for n in ("const hasValue", "const CLIENT_WORDS", "const FILLER", "const words",
                                          "function sides", "function draftValues", "function layoutFor",
                                          "function checklist"))
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => checklist(c, l))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def case(self, message, **change):
        draft = agent.draft_from_message(message, "Mika", "30-34")
        for f in draft["fields"]:
            if f["key"] in change:
                f["value"], f["visibility"] = change[f["key"]]
        return [{"draft": draft}, {"ages": agent.dating_ages(draft)}]

    def test_dating_needs_your_gender_and_an_age_range_and_says_why(self):
        no_gender = self.case("Looking to meet a man in Osaka, who's into jazz.")
        with_suggestions = self.case("Looking to meet a man in Osaka, who's into jazz.")
        with_suggestions[0]["draft"]["suggestions"] = [{"label": "Your interests", "ask": "I'm into "}]
        shown = self.case("Looking to meet a man in Osaka, early 30s. I'm a woman.")
        hidden = self.case("Looking to meet a man in Osaka, early 30s.", gender=("woman", HIDDEN))
        lists = self.run_checklist([no_gender, with_suggestions, shown, hidden])
        for items in lists[:2]:  # with or without the agent's own suggestions
            by = {i["label"]: i for i in items}
            self.assertFalse(by["Your gender"]["done"])
            self.assertIn("people looking for a woman or a man can find you", by["Your gender"]["why"])
            self.assertFalse(by["An age range"]["done"])
            self.assertIn("25 to 39", by["An age range"]["why"])
        by = {i["label"]: i for i in lists[2]}
        self.assertTrue(by["Your gender"]["done"] and by["An age range"]["done"])
        self.assertFalse(by["Your gender"].get("warn"))
        by = {i["label"]: i for i in lists[3]}
        self.assertTrue(by["Your gender"]["done"] and by["Your gender"]["warn"])  # hidden is allowed, and explained
        self.assertIn("fewer matches", by["Your gender"]["why"])


class TheReportedCase(unittest.TestCase):
    """A person writes "Looking to meet a man in Osaka, around early 30s, who's into photography and
    jazz." A synthetic man in Osaka whose live listing names no age range is looking for women."""

    MESSAGE = "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz."

    def introduced(self, message):
        s = DemoSession(seed=7, agent=None)
        s.verify("B", "Mika", 31)
        s.agent_message("B", message)
        s.submit("B", fill(s.slots["B"].draft))
        self.assertTrue(s.slots["B"].entry["accepted"])
        for _ in range(10):
            s.run_matchmaker(quiet=True)
        others = {i.person_b if i.person_a == "demoB-1" else i.person_a for i in s.intros.values()
                  if "demoB-1" in (i.person_a, i.person_b)}
        no_ages = {pid for pid in others if pid in s.personas and parse_intent(s.board.listing(pid).intent_text).age_range is None}
        return s, others, no_ages

    def test_with_her_gender_shown_she_meets_a_synthetic_man_who_named_no_ages(self):
        s, others, no_ages = self.introduced(self.MESSAGE + " I'm a woman.")
        self.assertTrue(no_ages, others)
        self.assertIn("p003", no_ages)  # the reported listing: "I'd like to meet women in Osaka for someone to date."
        self.assertEqual(s.phone_state("B")["listings"][0]["ages"], {"stated": "30 to 34", "default": "25 to 39"})
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_without_her_gender_men_looking_for_women_cant_find_her(self):
        s, others, _ = self.introduced(self.MESSAGE)
        self.assertNotIn("p003", others)
        self.assertTrue(any("people looking for a woman or a man can find you" in m["text"]
                            for m in s.slots["B"].chat))


class InspectorShowsBatchRunAnswers(unittest.TestCase):
    def test_answers_to_the_runs_questions_are_marked_and_the_rest_is_as_in_the_live_demo(self):
        result = run(n=300, seed=7)
        answered = [r for r in result["people"] if r["answered_in_run"]]
        self.assertTrue(answered)
        demo = DemoSession(seed=7, agent=None)
        for r in result["people"]:
            if r["admitted"]:
                self.assertEqual(r["intent_as_entered"], demo.board.listing(r["id"]).intent_text, r["id"])
        for r in answered:
            self.assertEqual(f"{r['intent_as_entered']} {r['answered_in_run']}", r["intent_text"])
            self.assertTrue(re.fullmatch(r"Aged \d+ (to \d+|or older)\.", r["answered_in_run"]), r)
        page = (Path(__file__).resolve().parent.parent / "belong" / "web" / "index.html").read_text()
        self.assertIn("Batch-run answer", page)
        self.assertIn("p.answered_in_run", page)


if __name__ == "__main__":
    unittest.main()
