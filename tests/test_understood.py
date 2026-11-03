"""What the person's agent understood is on their listing, checked at entry, and is what matching
uses (TRUST.md #33). The kind of match and, for dating, who they'd like to meet and those people's
ages are shown details ("kind", "wants_gender", "wants_age"); the description is read only for what
they don't say, and a hidden one never decides. When a message could mean more than one kind of
match, the agent asks one short question instead of guessing (SPEC §5: AI suggests, the person decides).
"""
from __future__ import annotations

import re
import unittest

from belong import agent
from belong.agent_llm import SYSTEM_PROMPT, UPDATE_LISTING_TOOL, ClaudeAgent, build_draft, turn_drafts
from belong.demo import DemoSession
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import OfflineMatcher, ShownView, intent_met, understood
from belong.schema import AGE_BAND_BOUNDS, GENDERS, HIDDEN, MATCH_KINDS, SHOWN, Field, Listing, Person
from belong.sim import run
from belong.synth import generate
from belong.warden import reason_text
from tests.helpers import warden
from tests.test_agent_llm import ScriptedClient, reply, text, tool
from tests.test_demo import fill

# The two reports, each an instance of the general problem.
VISITING = ("Visiting Osaka and looking to meet a woman around my age for something genuine. I enjoy rock climbing "
            "and tennis.")
EARLY_30S = "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz."


def person_listing(pid, intent, band, kind=None, gender=None, wants_gender=None, wants_age=None, extra=(),
                   visibility=SHOWN, level=None, exactness=0.6):
    """A listing in Osaka; `visibility` applies to kind / wants_gender / wants_age."""
    fields = [Field("first_name", pid, SHOWN, False), Field("city", "Osaka", SHOWN, False),
              Field("country", "Japan", SHOWN, False), Field("age_band", band, SHOWN, False),
              Field("interests", ("tennis",), SHOWN, False), Field("schedule", "weekends", SHOWN, False)]
    if gender:
        fields.append(Field("gender", gender, SHOWN, False))
    if kind:
        fields.append(Field("kind", kind, visibility, False))
    if wants_gender:
        fields.append(Field("wants_gender", tuple(wants_gender), visibility, False))
    if wants_age:
        fields.append(Field("wants_age", wants_age, visibility, False))
    if level:
        fields.append(Field("level", level, SHOWN, False))
    return Listing(pid, intent, exactness, tuple(fields) + tuple(extra), "2026-10-12T09:00:00Z")


def board_of(*listings):
    w, board = warden(), Marketboard()
    for l in listings:
        band = l.get("age_band").value
        result = w.admit(Person(l.person_id, True, band, "t"), l, board)
        assert result.accepted, (l.person_id, result.problems)
    return w, board


def pairs(board):
    return {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(board).proposals if not p.close}


def met(a: Listing, b: Listing) -> bool:
    va, vb = ShownView(a), ShownView(b)
    return intent_met(understood(va), va, understood(vb), vb)


def women_wanting_men(prefix, bands):
    return [person_listing(f"{prefix}{i}", "Looking for a relationship with a man in Osaka.", band, "dating", "woman",
                           ["man"], f"{AGE_BAND_BOUNDS[0][0]}-99") for i, band in enumerate(bands)]


class TheReports(unittest.TestCase):
    """(a) A man, 35-39, mentions climbing while asking to meet a woman around his age; (b) "around early
    30s". Matching follows what the agent recorded, never a hobby in the description."""

    def test_a_dating_listing_that_mentions_an_activity_meets_only_the_people_it_wants(self):
        him = person_listing("him", VISITING, "35-39", "dating", "man", ["woman"], "30-49")
        women = women_wanting_men("w", ["30-34", "35-39", "40-49", "25-29", "50+"])
        man = person_listing("man50", "Looking for a relationship with a man in Osaka.", "50+", "dating", "man",
                             ["man"], "18-99")
        climbers = [person_listing("c1", "Looking for a climbing partner in Osaka on weekends.", "50+", "climbing",
                                   level="intermediate"),
                    person_listing("c2", "Need someone to belay with in Osaka.", "40-49", "climbing", gender="woman")]
        _, board = board_of(him, *women, man, *climbers)
        his = {p - {"him"} for p in pairs(board) if "him" in p}
        self.assertEqual({next(iter(p)) for p in his}, {"w0", "w1", "w2"})  # women aged 30 to 49 only
        # The words alone would have made him a climber; the kind he recorded says dating.
        self.assertEqual(understood(ShownView(him)).kind, "dating")
        self.assertFalse(any(met(him, c) or met(c, him) for c in climbers))

    def test_relative_and_decade_ages_are_understood(self):
        # The rule-based agent works the years out from what they typed, using their age band.
        self.assertEqual(agent.wanted_ages(VISITING, "35-39")[0], "30-49")  # one band either side of 35-39
        self.assertEqual(agent.wanted_ages(EARLY_30S, "30-34")[0], "30-33")
        self.assertEqual(agent.wanted_ages("Looking for a man a bit younger than me", "40-49")[0], "35-49")
        self.assertEqual(agent.wanted_ages("Looking for a woman, a bit older", "25-29")[0], "25-34")
        self.assertEqual(agent.wanted_ages("Looking for a man around 19", "18-24"),  # 16 to 22, as said
                         ("18-22", "Belong. only introduces adults, so the ages start at 18."))
        self.assertIsNone(agent.wanted_ages("Looking for a woman in Osaka", "30-34")[0])
        # Matching reads "around my age" the same way when only the description says it.
        no_field = person_listing("p", VISITING, "35-39", "dating", "man", ["woman"])
        self.assertEqual(understood(ShownView(no_field)).age_range, (2, 4))

    def test_early_30s_is_dating_and_meets_men_in_their_early_30s(self):
        draft = agent.draft_from_message(EARLY_30S, "Mika", "30-34")
        self.assertEqual(draft["kind"], "dating")
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((values["kind"], values["wants_gender"], values["wants_age"]), ("dating", ["man"], "30-33"))
        her = person_listing("her", EARLY_30S, "30-34", "dating", "woman", ["man"], "30-33")
        men = [person_listing(f"m{band}", "Looking for a relationship with a woman in Osaka.", band, "dating", "man",
                              ["woman"], "18-99") for band in ("25-29", "30-34", "35-39")]
        _, board = board_of(her, *men)
        self.assertEqual({next(iter(p - {"her"})) for p in pairs(board) if "her" in p}, {"m30-34"})


class FieldsBeatKeywords(unittest.TestCase):
    def test_the_shown_kind_decides_whatever_the_description_says(self):
        runner = person_listing("r", "Looking for a climbing partner in Osaka.", "30-34", "running")
        runners = person_listing("r2", "Running buddy in Osaka for weekend runs.", "30-34", "running")
        climber = person_listing("c", "Looking for a climbing partner in Osaka.", "30-34", "climbing")
        _, board = board_of(runner, runners, climber)
        self.assertEqual(pairs(board), {frozenset(("r", "r2"))})

    def test_shown_wants_beat_the_description(self):
        # The description says a man; what the agent recorded (and the person saw) says a woman, 40 to 49.
        her = person_listing("her", "Looking for a relationship with a man in Osaka, aged 25 to 29.", "40-49",
                             "dating", "woman", ["woman"], "40-49")
        p = understood(ShownView(her))
        self.assertEqual((p.seeking, p.age_range), (("woman",), (4, 4)))

    def test_narrowing_groups_by_the_same_kind(self):
        from belong.matching.narrow import shortlists
        from belong.matching.matcher import index_words
        him = person_listing("him", VISITING, "35-39", "dating", "man", ["woman"], "30-49")
        climber = person_listing("c", "Looking for a climbing partner in Osaka.", "35-39", "climbing")
        her = women_wanting_men("w", ["35-39"])[0]
        _, board = board_of(him, climber, her)
        views = {l.person_id: ShownView(l) for l in board.listings()}
        lists, _ = shortlists({pid: understood(v) for pid, v in views.items()}, views, 10, words=index_words)
        self.assertEqual(lists["him"], ["w0"])

    def test_a_kind_or_age_range_outside_the_fixed_choices_is_never_read(self):
        self.assertEqual(understood(ShownView(person_listing("p", "Looking for a climbing partner.", "30-34"))).kind,
                         "climbing")


class HiddenNeverDecides(unittest.TestCase):
    def test_a_hidden_kind_or_wants_is_as_if_it_werent_there(self):
        # Hidden, "dating" can't make a climbing description dating; the description decides, as if unstated.
        hidden_kind = person_listing("h", "Looking for a climbing partner in Osaka.", "30-34", "dating",
                                     visibility=HIDDEN)
        self.assertEqual(understood(ShownView(hidden_kind)).kind, "climbing")
        hidden_wants = person_listing("hw", "Looking for a relationship with a woman in Osaka, aged 30 to 34.",
                                      "30-34", "dating", "man", ["man"], "40-49", visibility=HIDDEN)
        p = understood(ShownView(hidden_wants))
        self.assertEqual((p.kind, p.seeking, p.age_range), ("dating", ("woman",), (2, 2)))

    def test_a_dating_listing_with_no_known_wanted_gender_is_never_matched(self):
        nobody = person_listing("n", "Looking for someone special in Osaka.", "30-34", "dating", "man", ["woman"],
                                "25-39", visibility=HIDDEN)
        women = women_wanting_men("w", ["30-34", "35-39"])
        _, board = board_of(nobody, *women)
        self.assertEqual(understood(ShownView(nobody)).seeking, ())
        self.assertFalse(any("n" in p for p in pairs(board)))

    def test_removing_hidden_understanding_never_removes_a_match(self):
        him = person_listing("him", VISITING, "35-39", "dating", "man", ["woman"], "30-49")
        shown_women = women_wanting_men("w", ["30-34", "35-39", "40-49"])
        hidden = [Listing(l.person_id, l.intent_text, l.exactness,
                          tuple(Field(f.key, f.value, HIDDEN, False) if f.key in ("wants_age",) else f
                                for f in l.fields), l.updated_at) for l in shown_women]
        _, with_hidden = board_of(him, *hidden)
        stripped = [Listing(l.person_id, l.intent_text, l.exactness, tuple(f for f in l.fields if f.visibility != HIDDEN),
                            l.updated_at) for l in hidden]
        _, without = board_of(him, *stripped)
        self.assertEqual(pairs(with_hidden), pairs(without))


class TheWardenChecksTheFormats(unittest.TestCase):
    def admit(self, *fields):
        base = person_listing("p", "Looking for a relationship with a woman in Osaka.", "30-34", gender="man")
        listing = Listing("p", base.intent_text, 0.6, base.fields + tuple(fields), "t")
        return warden().admit(Person("p", True, "30-34", "t"), listing, Marketboard())

    def test_only_the_fixed_choices_get_in(self):
        for kind in MATCH_KINDS:
            self.assertTrue(self.admit(Field("kind", kind, SHOWN, False)).accepted, kind)
        self.assertTrue(self.admit(Field("wants_gender", GENDERS, SHOWN, False),
                                   Field("wants_age", "18-99", HIDDEN, False)).accepted)
        bad = [Field("kind", "service", SHOWN, False), Field("kind", "Dating", SHOWN, False),
               Field("wants_gender", ("female",), SHOWN, False), Field("wants_gender", (), SHOWN, False),
               Field("wants_gender", ("woman", "woman"), SHOWN, False), Field("wants_gender", "woman", SHOWN, False),
               Field("wants_age", "16-25", SHOWN, False), Field("wants_age", "40-30", SHOWN, False),
               Field("wants_age", "30-100", SHOWN, False), Field("wants_age", "early 30s", SHOWN, False),
               Field("wants_age", "30 - 39", SHOWN, False), Field("wants_age", "30", SHOWN, False),
               # Only ASCII digits, no leading zero: anything else matching or a reader could take differently.
               Field("wants_age", "٣٠-٣٩", SHOWN, False), Field("wants_age", "３０-３９", SHOWN, False),
               Field("wants_age", "030-039", SHOWN, False), Field("wants_age", "30-039", SHOWN, False),
               Field("wants_age", "030-39", SHOWN, False), Field("wants_age", "30-39\n", SHOWN, False),
               Field("wants_age", "30–39", SHOWN, False), Field("wants_age", "18-100", SHOWN, False)]
        for f in bad:
            result = self.admit(f)
            self.assertFalse(result.accepted, f)
            self.assertIn("listing_schema", result.failed_checks, f)
        for good in ("18-99", "30-39", "99-99"):
            self.assertTrue(self.admit(Field("wants_age", good, SHOWN, False)).accepted, good)

    def test_other_ways_of_writing_ages_are_put_into_the_fixed_form_first(self):
        from belong.schema import age_span, age_span_choice
        for said, fixed in (("33 to 42", "33-42"), ("33–42", "33-42"), ("030-039", "30-39"), ("３０-３９", "30-39"),
                            ("٣٠-٣٩", "30-39"), (" 30 - 39 ", "30-39"), ("16-25", None), ("early 30s", None)):
            self.assertEqual(age_span_choice(said), fixed, said)
            if fixed:
                self.assertIsNotNone(age_span(fixed))
        # The phone's editor and the model agent both convert before the Warden checks.
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "Looking for a relationship with a woman in Osaka. I'm a man.")
        s.submit("A", fill(s.slots["A"].draft, values={"wants_age": "０３０-０３９"}))
        self.assertTrue(s.slots["A"].entry["accepted"], s.slots["A"].entry)
        self.assertEqual(s.board.listing("demoA-1").get("wants_age").value, "30-39")

    def test_the_wardens_wording_never_uses_a_fixed_choice_as_a_word(self):
        """"Near each other" would read as the kind "other" and get a fair introduction refused."""
        japan = ("Japan", "Japan")
        texts = [reason_text([("city", "Osaka", "Kyoto")], japan), reason_text([("city", "Osaka", "Osaka")], japan),
                 reason_text([("country", "Japan", "Japan")]), reason_text([("format", "Online", "Online")]),
                 reason_text([("schedule", "flexible", "flexible")])]
        words = {w for t in texts for w in re.findall(r"[a-z]+", t.lower())}
        self.assertFalse(words & {w for v in MATCH_KINDS + GENDERS for w in v.split()}, texts)

    def test_two_open_ended_listings_in_nearby_cities_are_introduced(self):
        def phrases(pid, city, seeking, offering):
            return Listing(pid, "Hoping to meet people.", 0.6, (
                Field("city", city, SHOWN, False), Field("country", "Japan", SHOWN, False),
                Field("travel", "Up to 100 km", SHOWN, False), Field("kind", "other", SHOWN, False),
                Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False),
                Field("schedule", "weekends", SHOWN, False)), "t")
        w, board = board_of_people(phrases("pa", "Osaka", "a chess partner", "a chess partner"),
                                   phrases("pb", "Kyoto", "a chess partner", "a chess partner"))
        props = OfflineMatcher().match(board).proposals
        self.assertEqual(len(props), 1)
        intro = w.exit_introduction(props[0], board)
        self.assertIsNotNone(intro)
        self.assertIn("near one another", intro.a_view["reason"])


def board_of_people(*listings):
    w, board = warden(), Marketboard()
    for l in listings:
        assert w.admit(Person(l.person_id, True, "30-34", "t"), l, board).accepted
    return w, board


class TheModelAgentRecordsWhatItUnderstood(unittest.TestCase):
    def dating(self, **extra):
        args = {"kind": "dating", "exactness": 0.6, "intent_text": VISITING, "suggestions": [],
                "separate_listing": False,
                "fields": [{"key": "city", "values": ["Osaka"], "visibility": SHOWN},
                           {"key": "gender", "values": ["man"], "visibility": SHOWN},
                           {"key": "interests", "values": ["rock climbing", "tennis"], "visibility": SHOWN}]}
        args["fields"] += extra.pop("fields", [])
        return build_draft({**args, **extra}, "Ken", "35-39")

    def test_the_tool_kind_becomes_the_kind_field(self):
        draft, problems, _ = self.dating(fields=[{"key": "wants_gender", "values": ["woman"], "visibility": SHOWN},
                                                 {"key": "wants_age", "values": ["30 to 49"], "visibility": SHOWN}])
        self.assertEqual(problems, [])
        by = {f["key"]: f for f in draft["fields"]}
        self.assertEqual((by["kind"]["value"], by["kind"]["visibility"]), ("dating", SHOWN))
        self.assertEqual((by["wants_gender"]["value"], by["wants_age"]["value"]), (["woman"], "30-49"))
        self.assertIn("with a woman, 30 to 49", draft["understood"])
        # The description reads as climbing to the keyword rules; that doesn't matter any more.
        self.assertFalse(any("clearly says" in m for m in draft["missing"]))
        # The person's own choice to hide the kind is kept when the agent lists it.
        draft, _, _ = self.dating(fields=[{"key": "kind", "values": ["junk"], "visibility": HIDDEN}])
        self.assertEqual({f["key"]: f["visibility"] for f in draft["fields"]}["kind"], HIDDEN)

    def test_wants_are_validated_strictly(self):
        for f, said in (({"key": "wants_age", "values": ["early 30s"], "visibility": SHOWN}, "wants_age"),
                        ({"key": "wants_age", "values": ["16-25"], "visibility": SHOWN}, "wants_age"),
                        ({"key": "wants_age", "values": ["30-39", "40-49"], "visibility": SHOWN}, "wants_age"),
                        ({"key": "wants_gender", "values": ["female"], "visibility": SHOWN}, "wants_gender")):
            draft, problems, _ = self.dating(fields=[f])
            self.assertIsNone(draft, f)
            self.assertTrue(any(p.startswith(said) for p in problems), problems)
        # Only dating records who and what ages: anything else leaves them out, and says so.
        args = {"kind": "climbing", "exactness": 0.6, "intent_text": "Looking for a climbing partner in Osaka.",
                "fields": [{"key": "city", "values": ["Osaka"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a climbing partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a climbing partner"], "visibility": SHOWN},
                           {"key": "wants_gender", "values": ["woman"], "visibility": SHOWN}]}
        draft, problems, notes = build_draft(args, "Ken", "35-39")
        self.assertEqual(problems, [])
        self.assertNotIn("wants_gender", {f["key"] for f in draft["fields"]})
        self.assertTrue(any("only used for dating" in n for n in notes))

    def test_the_tool_and_instructions_offer_the_new_details(self):
        keys = UPDATE_LISTING_TOOL["input_schema"]["properties"]["fields"]["items"]["properties"]["key"]["enum"]
        self.assertTrue({"wants_gender", "wants_age"} <= set(keys))
        values = UPDATE_LISTING_TOOL["input_schema"]["properties"]["fields"]["items"]["properties"]["values"]
        self.assertIn("wants_age", values["description"])
        self.assertIn("- wants_gender (default shown, only for dating listings): woman, man, nonbinary", SYSTEM_PROMPT)
        self.assertIn("what they want to meet someone FOR, never a hobby", SYSTEM_PROMPT)


class TheRuleAgentRecordsItToo(unittest.TestCase):
    def test_offline_drafts_carry_kind_and_wants(self):
        draft = agent.draft_from_message(VISITING + " I'm a man.", "Ken", "35-39", kind="dating")
        values = {f["key"]: (f["value"], f["visibility"]) for f in draft["fields"] if f["value"]}
        self.assertEqual(values["kind"], ("dating", SHOWN))
        self.assertEqual(values["wants_gender"], (["woman"], SHOWN))
        self.assertEqual(values["wants_age"], ("30-49", SHOWN))
        self.assertEqual(values["gender"], ("man", SHOWN))
        self.assertIn("rock climbing", values["interests"][0])  # the hobby is an interest, not the kind
        for message, kind in (("Looking for a climbing partner in Osaka on weekends.", "climbing"),
                              ("Need a physics tutor in Berlin.", "tutoring"),
                              ("Looking for a tennis partner in Tokyo.", "other")):
            self.assertEqual({f["key"]: f["value"] for f in agent.draft_from_message(message, "A", "25-29")["fields"]}
                             ["kind"], kind, message)


class AskWhenItsUnclear(unittest.TestCase):
    """The user's request: ask whether it's a date or an activity partner when it isn't clear."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.s.verify("A", "Ken", 37)

    def chat(self, listing=0):
        return [m["text"] for m in self.s.phone_state("A")["listings"][listing]["chat"] if m]

    def kinds(self):
        return [(l["draft"] or {}).get("kind") for l in self.s.phone_state("A")["listings"]]

    def test_romantic_wording_with_an_activity_asks_first(self):
        for message, activity in ((VISITING, "a climbing partner"),
                                  ("I'd like to meet a man in Kyoto. I go running most mornings.", "a running buddy"),
                                  ("Hoping to meet a woman in Lisbon, and looking for a hiking buddy too.",
                                   "a hiking partner")):
            question = agent.kind_question(message)
            self.assertIsNotNone(question, message)
            self.assertEqual(question["text"], f"Are you looking for a date, or {activity}? Or both? I can make two "
                                               f"separate listings.")

    def test_partner_alone_asks(self):
        question = agent.kind_question("Looking for a partner in Berlin.")
        self.assertIn("a partner for an activity", question["text"])

    def test_clear_messages_dont_ask(self):
        for message in (EARLY_30S, "Looking for a relationship with a woman in Osaka. I enjoy climbing.",
                        "Looking for a climbing partner in Osaka on weekends.", "Looking for a tennis partner in Tokyo.",
                        "Looking for a life partner in Osaka.", "Need a man with a van to help me move in London."):
            self.assertIsNone(agent.kind_question(message), message)

    def test_the_answer_decides_and_both_makes_two_listings(self):
        self.s.agent_message("A", VISITING)
        self.assertIn("a date, or a climbing partner", self.chat()[-1])
        self.assertIsNone(self.s.phone_state("A")["listings"][0]["draft"])  # nothing saved while it's unclear
        self.s.agent_message("A", "Both, please")
        self.assertEqual(self.kinds(), ["dating", "climbing"])
        for l in self.s.phone_state("A")["listings"]:
            self.assertEqual({f["key"]: f["value"] for f in l["draft"]["fields"]}["kind"], l["draft"]["kind"])
        dating = {f["key"]: f["value"] for f in self.s.phone_state("A")["listings"][0]["draft"]["fields"]}
        self.assertEqual((dating["wants_gender"], dating["wants_age"]), (["woman"], "30-49"))

    def test_one_choice_makes_one_listing_of_that_kind(self):
        self.s.agent_message("A", "I'd like to meet a man in Kyoto. I go running most mornings.")
        self.s.agent_message("A", "just running, not a date")
        self.assertEqual(self.kinds(), ["running"])

    def test_a_partner_for_an_activity_named_in_the_answer(self):
        self.s.agent_message("A", "Looking for a partner in Berlin.")
        self.s.agent_message("A", "tennis")
        draft = self.s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["seeking"], values["offering"]),
                         ("other", "a tennis partner", "a tennis partner"))

    def test_both_with_an_activity_not_named_yet_asks_which(self):
        self.s.agent_message("A", "Looking for a partner in Berlin.")
        self.s.agent_message("A", "both")
        self.assertEqual(self.kinds(), ["dating", None])
        self.assertIn("What would you like a partner for?", self.chat(1)[-1])
        second = self.s.phone_state("A")["listings"][1]["id"]
        self.s.agent_message("A", "chess", second)
        self.assertEqual(self.kinds(), ["dating", "other"])

    def test_not_an_answer_asks_again_and_a_new_request_starts_over(self):
        self.s.agent_message("A", VISITING)
        self.s.agent_message("A", "hmm")
        self.assertIn("didn't catch that", self.chat()[-1])
        self.s.agent_message("A", "Actually I need a roommate in Osaka from November.")
        self.assertEqual(self.kinds(), ["roommate"])

    def test_the_model_asks_before_saving_and_both_saves_two_listings(self):
        """A scripted stand-in for the model: it asks, then saves one listing per kind in one turn."""
        def save(kind, separate, extra):
            return {"kind": kind, "exactness": 0.6, "suggestions": [], "separate_listing": separate,
                    "intent_text": {"dating": "Looking to meet a man in Kyoto.",
                                    "running": "Looking for a running buddy in Kyoto, mornings."}[kind],
                    "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN}] + extra}
        client = ScriptedClient(
            reply(text("Are you looking for a date, or a running buddy? Or both? I can make two separate listings.")),
            reply(text("NO_SAVE")),  # the app's check (SAVE_CHECK): the kind is the open question, so nothing to save
            reply(tool(save("dating", False, [{"key": "gender", "values": ["woman"], "visibility": SHOWN},
                                              {"key": "wants_gender", "values": ["man"], "visibility": SHOWN},
                                              {"key": "wants_age", "values": ["30-49"], "visibility": SHOWN}]), "t1"),
                  tool(save("running", True, [{"key": "pace", "values": ["easy"], "visibility": SHOWN},
                                              {"key": "seeking", "values": ["a running partner"], "visibility": SHOWN},
                                              {"key": "offering", "values": ["a running partner"], "visibility": SHOWN},
                                              {"key": "headline", "values": ["Looking for a running buddy"],
                                               "visibility": SHOWN}]), "t2")),
            reply(text("Done: two listings, one for dating and one for running.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Aiko", 37)
        s.agent_message("A", "I'd like to meet a man in Kyoto. I go running most mornings.")
        self.assertIsNone(s.phone_state("A")["listings"][0]["draft"])  # asked, saved nothing
        self.assertIn("ONE short question", client.requests[0]["system"])
        s.agent_message("A", "Both!")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([l["draft"]["kind"] for l in listings], ["dating", "running"])
        self.assertEqual([{f["key"]: f["value"] for f in l["draft"]["fields"]}["kind"] for l in listings],
                         ["dating", "running"])
        self.assertEqual(s.phone_state("A")["current"], listings[0]["id"])

    def test_turn_drafts_keeps_one_draft_per_listing(self):
        a, b, b2 = {"kind": "dating"}, {"kind": "running", "separate": True}, {"kind": "running", "separate": True, "n": 2}
        self.assertIsNone(turn_drafts([]))
        self.assertEqual(turn_drafts([a]), a)
        both = turn_drafts([a, b, b2])
        self.assertEqual((both["n"], both["earlier"]), (2, [a]))


class SyntheticPeopleCarryIt(unittest.TestCase):
    def test_every_listing_records_its_kind_and_dating_its_wants(self):
        for p in generate(400, 3):
            t, v = p.truth, p.values
            self.assertEqual(v["kind"], t.kind if t.kind in MATCH_KINDS else "other")
            if t.kind == "dating":
                self.assertEqual(set(v["wants_gender"]), set(t.seeking))
                lo, hi = t.age_range
                expected = None if t.omit_age else f"{AGE_BAND_BOUNDS[lo][0]}-{AGE_BAND_BOUNDS[hi][1]}"
                self.assertEqual(v.get("wants_age"), expected)
            else:
                self.assertNotIn("wants_gender", v)
                self.assertNotIn("wants_age", v)

    def test_the_rest_of_each_person_is_unchanged(self):
        """The new details use no randomness: the same people as without them, with three more details."""
        from unittest import mock
        with mock.patch("belong.synth._write_understanding", lambda t, v: None):
            before = generate(300, 7)
        after = generate(300, 7)
        for old, new in zip(before, after):
            extra = {k: v for k, v in new.values.items() if k not in old.values}
            self.assertEqual({k: v for k, v in new.values.items() if k in old.values}, old.values)
            self.assertTrue(set(extra) <= {"kind", "wants_gender", "wants_age"}, extra)
            self.assertEqual((new.intent_text, new.truth, new.person), (old.intent_text, old.truth, old.person))
            self.assertEqual({k: new.visibility[k] for k in old.visibility}, old.visibility)
        self.assertEqual(after[3].intent_text, "I'd like to meet women in Osaka for someone to date.")

    def test_the_batch_run_stays_precise_and_covers(self):
        m = run(n=300, seed=7)["metrics"]
        self.assertGreaterEqual(m["precision"], 0.95)
        self.assertGreaterEqual(m["coverage"], 0.85)


class InTheLiveDemo(unittest.TestCase):
    def test_the_visiting_man_meets_only_women_of_the_right_ages(self):
        """5,000 synthetic people, the rule-based agent, and a message like report (a)."""
        s = DemoSession(seed=7, agent=None, people=5000)
        s.verify("A", "Ken", 37)
        s.agent_message("A", VISITING)
        s.agent_message("A", "a date")
        draft = s.slots["A"].draft
        self.assertEqual(draft["kind"], "dating")
        s.submit("A", fill(draft, values={"gender": "man"}))
        self.assertTrue(s.slots["A"].entry["accepted"])
        for _ in range(6):
            s.run_matchmaker(quiet=True)
        others = [i.person_b if i.person_a == "demoA-1" else i.person_a for i in s.intros.values()
                  if "demoA-1" in (i.person_a, i.person_b)]
        self.assertTrue(others)
        for pid in others:
            listing = s.board.listing(pid)
            self.assertEqual(listing.get("gender").value, "woman", pid)
            self.assertIn(listing.get("age_band").value, {"30-34", "35-39", "40-49"}, pid)
            self.assertEqual(listing.get("kind").value, "dating", pid)  # never a climber
        self.assertTrue(s.room_state()["leak_report"]["ok"])

class HiddenStaysHidden(unittest.TestCase):
    """A detail the person hid is never shown again just because a later agent draft left it out: only a
    visibility the agent chose on purpose replaces theirs."""

    def save(self, fields):
        return {"kind": "dating", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking to meet a woman in Kyoto, aged 30 to 39.",
                "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN}] + fields}

    def visibilities(self, s):
        return {f["key"]: f["visibility"] for f in s.slots["A"].draft["fields"] if f["value"]}

    def test_a_model_save_that_leaves_a_hidden_detail_out_keeps_it_hidden(self):
        first = [{"key": "gender", "values": ["man"], "visibility": SHOWN},
                 {"key": "wants_gender", "values": ["woman"], "visibility": SHOWN},
                 {"key": "wants_age", "values": ["30-39"], "visibility": SHOWN}]
        later = [{"key": "schedule", "values": ["weekends"], "visibility": SHOWN}]  # leaves all of those out
        client = ScriptedClient(reply(tool(self.save(first), "t1")), reply(text("Saved.")),
                                reply(tool(self.save(later), "t2")), reply(text("Added weekends.")),
                                reply(tool(self.save(later + [{"key": "gender", "values": ["man"],
                                                               "visibility": SHOWN}]), "t3")), reply(text("Shown.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'm a man looking to meet a woman in Kyoto, 30 to 39.")
        hide = ("kind", "gender", "wants_gender", "wants_age")
        s.submit("A", fill(s.slots["A"].draft, hidden={k: next(f["value"] for f in s.slots["A"].draft["fields"]
                                                               if f["key"] == k) for k in hide}))
        self.assertTrue(s.slots["A"].entry["accepted"])
        s.agent_message("A", "I'm free on weekends")
        vis = self.visibilities(s)
        self.assertEqual({k: vis[k] for k in hide}, {k: HIDDEN for k in hide})
        self.assertEqual(vis["schedule"], SHOWN)
        values = {f["key"]: f["value"] for f in s.slots["A"].draft["fields"]}
        self.assertEqual((values["gender"], values["wants_gender"], values["wants_age"]), ("man", ["woman"], "30-39"))
        # Saving it keeps them hidden on the live listing too.
        s.submit("A", fill(s.slots["A"].draft))
        live = s.board.listing("demoA-1")
        self.assertEqual({k: live.get(k).visibility for k in hide}, {k: HIDDEN for k in hide})
        # The agent listing one with a visibility of its own (the person asked to show it) is a choice made on purpose.
        s.agent_message("A", "Please show my gender")
        self.assertEqual(self.visibilities(s)["gender"], SHOWN)
        self.assertEqual(self.visibilities(s)["wants_gender"], HIDDEN)

    def test_the_rule_based_agent_keeps_a_hidden_detail_hidden_too(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "Looking for a relationship with a woman in Osaka. I'm a man.")
        s.submit("A", fill(s.slots["A"].draft, hidden={"gender": "man", "wants_gender": ["woman"]}))
        s.agent_message("A", "Looking for a relationship with a woman in Osaka, aged 30 to 39. I'm a man.")
        vis = self.visibilities(s)
        self.assertEqual((vis["gender"], vis["wants_gender"], vis["wants_age"]), (HIDDEN, HIDDEN, SHOWN))

    def test_keep_visibility_never_shows_a_sensitive_detail(self):
        old = {"fields": [{"key": "religion", "visibility": SHOWN, "sensitive": True},
                          {"key": "gender", "visibility": HIDDEN, "sensitive": False}]}
        new = {"listed": [], "fields": [{"key": "religion", "visibility": HIDDEN, "sensitive": True},
                                        {"key": "gender", "visibility": SHOWN, "sensitive": False}]}
        agent.keep_visibility(new, old)
        self.assertEqual([f["visibility"] for f in new["fields"]], [HIDDEN, HIDDEN])
        draft, _, _ = build_draft({"kind": "climbing", "exactness": 0.6, "intent_text": "A climbing partner in Osaka.",
                                   "fields": [{"key": "city", "values": ["Osaka"], "visibility": SHOWN},
                                              {"key": "level", "values": ["beginner"]}]}, "Ken", "35-39")
        self.assertEqual(draft["listed"], ["city"])  # level had no visibility of the agent's own


class HiddenChoicesAreRemembered(unittest.TestCase):
    """The person's choices of shown or hidden are remembered per listing (what they saved, and what they
    set in the editor) and every later draft keeps them, whichever agent drafts it and whatever it leaves
    out: nothing hidden is shown again unless an agent says shown for that detail on purpose (TRUST.md #34)."""

    HIDE = ("kind", "gender", "wants_gender", "wants_age")
    REQUEST = "Looking for a relationship with a woman in Osaka, aged 30 to 39. I'm a man."

    def visibilities(self, s, keys=HIDE):
        vis = {f["key"]: f["visibility"] for f in s.slots["A"].draft["fields"] if f["value"]}
        return {k: vis.get(k) for k in keys}

    def saved_hidden(self, s):
        d = s.slots["A"].draft
        s.submit("A", fill(d, hidden={k: next(f["value"] for f in d["fields"] if f["key"] == k) for k in self.HIDE}))
        self.assertTrue(s.slots["A"].entry["accepted"], s.slots["A"].entry)

    def test_the_rule_based_agent_keeps_them_whatever_comes_between(self):
        for between in (["I'm free on weekends."], ["Actually, a climbing partner in Osaka."],
                        ["I'm free on weekends.", "Actually, a climbing partner in Osaka.", "hello"]):
            s = DemoSession(seed=7, agent=None)
            s.verify("A", "Ken", 37)
            s.agent_message("A", self.REQUEST)
            self.saved_hidden(s)
            for message in between:
                s.agent_message("A", message)
            s.agent_message("A", self.REQUEST)
            self.assertEqual(self.visibilities(s), {k: HIDDEN for k in self.HIDE}, between)
            s.submit("A", fill(s.slots["A"].draft))  # one save: still hidden on the live listing
            live = s.board.listing("demoA-1")
            self.assertEqual({k: live.get(k).visibility for k in self.HIDE}, {k: HIDDEN for k in self.HIDE}, between)

    def test_the_instant_profile_keeps_them_too(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", self.REQUEST)
        self.saved_hidden(s)
        s.agent_message("A", "I'm free on weekends.")  # a draft without gender
        s.prefill("A", "I'm a man")
        self.assertEqual(self.visibilities(s, ("gender",)), {"gender": HIDDEN})

    def test_a_detail_sent_with_no_value_chooses_nothing(self):
        """The tool makes a visibility required, so a key can come with one and no value: that chooses
        nothing, and the old value comes back with the shown or hidden it had."""
        first, _, _ = build_draft(HiddenStaysHidden.save(None, [
            {"key": "gender", "values": ["man"], "visibility": SHOWN},
            {"key": "wants_gender", "values": ["woman"], "visibility": HIDDEN},
            {"key": "wants_age", "values": ["30-39"], "visibility": HIDDEN}]), "Ken", "35-39")
        first["fields"] = [{**f, "visibility": HIDDEN} if f["key"] == "gender" else f for f in first["fields"]]
        for empty in ([], [""], ["  "]):
            later = [{"key": k, "values": empty, "visibility": SHOWN} for k in ("gender", "wants_gender", "wants_age")]
            draft, problems, _ = build_draft(HiddenStaysHidden.save(None, later), "Ken", "35-39", first)
            self.assertEqual(problems, [])
            self.assertFalse({"gender", "wants_gender", "wants_age"} & set(draft["listed"]), empty)
            by = {f["key"]: (f["value"], f["visibility"]) for f in draft["fields"]}
            self.assertEqual((by["gender"], by["wants_gender"], by["wants_age"]),
                             (("man", HIDDEN), (["woman"], HIDDEN), ("30-39", HIDDEN)), empty)
        # carry_over on its own: what it fills in keeps the shown or hidden it had.
        new = {"listed": ["gender"], "kind": "dating",
               "fields": [{"key": "gender", "value": None, "visibility": SHOWN, "sensitive": False}]}
        agent.carry_over(new, first)
        self.assertEqual((new["fields"][0]["value"], new["fields"][0]["visibility"]), ("man", HIDDEN))

    def test_the_checks_and_what_the_agent_is_told_see_the_listing_as_it_will_be(self):
        from belong.agent_llm import _summary, profile_so_far
        first, _, _ = build_draft(HiddenStaysHidden.save(None, [
            {"key": "gender", "values": ["man"], "visibility": SHOWN},
            {"key": "wants_gender", "values": ["woman"], "visibility": SHOWN},
            {"key": "wants_age", "values": ["30-39"], "visibility": SHOWN}]), "Ken", "35-39")
        current = {**first, "choices": {k: HIDDEN for k in self.HIDE}}  # the person hid them, and saved
        later = {"kind": "dating", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                 "intent_text": "Free on weekends in Kyoto.",  # doesn't say dating
                 "fields": [{"key": "schedule", "values": ["weekends"], "visibility": SHOWN}]}
        draft, problems, _ = build_draft(later, "Ken", "35-39", current)
        self.assertEqual(problems, [])
        self.assertEqual({f["key"]: f["visibility"] for f in draft["fields"] if f["key"] in self.HIDE},
                         {k: HIDDEN for k in self.HIDE})
        missing = " ".join(draft["missing"])
        self.assertNotIn("who they want to meet", missing)  # carried over: not asked again
        self.assertNotIn("the ages they'd like", missing)
        self.assertNotIn("their gender", missing)
        self.assertIn("the kind is hidden", missing)  # the hidden kind, as it will be, is checked
        summary = _summary(draft)
        self.assertIn("Hidden (never shown): kind, gender, wants_gender, wants_age", summary)
        self.assertNotIn("gender=man", summary)
        # The agent is told what the person chose to keep hidden, even a detail not on the draft.
        self.assertIn("keep them hidden unless they ask to show one: gender, kind",
                      profile_so_far({"fields": [], "choices": {"gender": HIDDEN, "kind": HIDDEN, "city": SHOWN}}))

    def test_a_model_save_that_leaves_them_out_or_empty_keeps_them_hidden(self):
        save = lambda fields: HiddenStaysHidden.save(None, fields)
        first = [{"key": "gender", "values": ["man"], "visibility": SHOWN},
                 {"key": "wants_gender", "values": ["woman"], "visibility": SHOWN},
                 {"key": "wants_age", "values": ["30-39"], "visibility": SHOWN}]
        empty = [{"key": k, "values": [], "visibility": SHOWN} for k in ("gender", "wants_gender", "wants_age")]
        client = ScriptedClient(reply(tool(save(first), "t1")), reply(text("Saved.")),
                                reply(tool(save(empty), "t2")), reply(text("Saved again.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'm a man looking to meet a woman in Kyoto, 30 to 39.")
        self.saved_hidden(s)
        s.agent_message("A", "I'm free on weekends")
        self.assertEqual(self.visibilities(s), {k: HIDDEN for k in self.HIDE})
        told = client.requests[-1]["messages"][-1]["content"][0]["content"]  # what the save's result said
        self.assertIn("Hidden (never shown): kind, gender, wants_gender, wants_age", told)
        self.assertNotIn("who they want to meet", told)

    def test_a_choice_made_in_the_editor_is_kept_before_it_is_saved(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", self.REQUEST)
        s.set_visibility("A", None, {"gender": HIDDEN, "wants_age": HIDDEN, "religion": SHOWN})
        self.assertEqual(self.visibilities(s, ("gender", "wants_age")), {"gender": HIDDEN, "wants_age": HIDDEN})
        s.agent_message("A", self.REQUEST)
        self.assertEqual(self.visibilities(s, ("gender", "wants_age", "wants_gender")),
                         {"gender": HIDDEN, "wants_age": HIDDEN, "wants_gender": SHOWN})
        self.assertNotIn("religion", s.slots["A"].listing().choices)  # a sensitive detail needs confirming
        s.set_visibility("A", None, {"religion": SHOWN}, ["religion"])
        self.assertEqual(s.slots["A"].listing().choices["religion"], SHOWN)
        from belong import server
        self.assertIn("visibility", server.ACTIONS)
        self.assertIn("visibility", server.CHAT_ACTIONS)

    def test_the_phone_sends_editor_choices_before_a_message(self):
        """demo.html: the changes the phone sends first (run with node)."""
        import json
        import shutil
        import subprocess
        import tempfile
        from tests.test_dating import PAGE, _js
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        self.assertIn('ok = await api("visibility", { slot, listing_id, ...vis });', PAGE)  # send() waits for it
        self.assertLess(PAGE.index('api("visibility"'), PAGE.index('await api("agent"'))  # before the agent redrafts
        server_copy = {"fields": [{"key": "gender", "visibility": SHOWN}, {"key": "city", "visibility": SHOWN},
                                  {"key": "religion", "visibility": HIDDEN}, {"key": "age_band", "visibility": SHOWN}]}
        local = {"fields": [{"key": "gender", "visibility": HIDDEN}, {"key": "city", "visibility": SHOWN},
                            {"key": "religion", "visibility": SHOWN, "confirmed": True},
                            {"key": "age_band", "visibility": HIDDEN, "locked": True},
                            {"key": "wants_age", "visibility": SHOWN}, {"key": "children", "visibility": HIDDEN}]}
        code = _js("function visibilityChanges") + f"""
const from = {json.dumps(json.dumps(server_copy))};
console.log(JSON.stringify([visibilityChanges({json.dumps(local)}, from), visibilityChanges(JSON.parse(from), from)]));"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        result = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        changes, none = json.loads(result.stdout)
        self.assertEqual(changes, {"visibility": {"gender": HIDDEN, "religion": SHOWN, "children": HIDDEN},
                                   "confirmed": ["religion"]})
        self.assertIsNone(none)


class TraitsAreNotChoices(unittest.TestCase):
    """How someone describes the person they'd like to meet ("a girlfriend who climbs", "a date with a
    climber") is a trait of that person, not an activity they want a partner for: no question, no second
    listing."""

    TRAITS = ("Looking for a girlfriend who climbs, in Osaka.", "Looking for a date with a climber in Osaka.",
              "I'd like to meet a man who plays tennis in Kyoto.", "Hoping to find a boyfriend in Tokyo, ideally a runner.",
              "Looking for a relationship with a woman who loves hiking.", "I want to meet a woman that does yoga.")

    def test_no_question_and_dating(self):
        for message in self.TRAITS:
            self.assertIsNone(agent.kind_question(message), message)
            self.assertEqual(agent.draft_from_message(message, "Ken", "35-39")["kind"], "dating", message)
        # Asked for as something to do together, it still asks.
        for message in ("Looking for a girlfriend in Osaka, and a climbing partner too.",
                        "I'd like to meet a man in Kyoto. I play tennis most weekends."):
            self.assertIsNotNone(agent.kind_question(message), message)

    def test_an_answer_that_describes_the_person_chooses_the_date(self):
        for options in (TheAnswerChoosesWhatItNames.OPTIONS, TheAnswerChoosesWhatItNames.PARTNER):
            for answer in ("a girlfriend who climbs", "a date with a climber", "a boyfriend, ideally a climber",
                           "a date with someone who climbs"):
                self.assertEqual(agent.chosen_kinds(answer, options), [{"kind": "dating", "activity": None}], answer)

    def test_in_the_demo_one_listing(self):
        for message in self.TRAITS[:2]:
            s = DemoSession(seed=7, agent=None)
            s.verify("A", "Ken", 37)
            s.agent_message("A", message)
            self.assertEqual([l["draft"]["kind"] for l in s.phone_state("A")["listings"]], ["dating"], message)
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", VISITING)
        s.agent_message("A", "a date with a climber")
        self.assertEqual([l["draft"]["kind"] for l in s.phone_state("A")["listings"]], ["dating"])


class HobbiesFirst(unittest.TestCase):
    """A hobby mentioned before the request is still a hobby (the user's report), and when the offline
    agent isn't sure, it asks instead of guessing."""

    HOBBY_FIRST = "I enjoy rock climbing and I'm looking to meet a woman around my age"

    def test_a_hobby_ends_where_the_request_starts(self):
        from belong.matching.intent import parse_intent
        for message in (self.HOBBY_FIRST, "I love hiking and want to find a girlfriend in Osaka.",
                        "I enjoy rock climbing and I'm looking for someone special in Osaka.",
                        "I like climbing, would like to meet a man in Kyoto."):
            self.assertEqual(parse_intent(message).kind, "dating", message)
            draft = agent.draft_from_message(message, "Ken", "35-39")
            self.assertEqual(draft["kind"], "dating", message)
            self.assertNotEqual({f["key"]: f["value"] for f in draft["fields"]}["kind"], "climbing")
        self.assertEqual(parse_intent("I enjoy running and I want a running buddy.").kind, "running")
        # Asking for something, just not what they enjoy: not read from the hobby.
        self.assertIsNone(parse_intent("I love climbing. Looking for someone fun in Osaka.").kind)
        self.assertEqual(parse_intent("I love climbing. Based in Osaka.").kind, "climbing")  # matching's fallback

    def test_a_hobby_ends_where_they_hope_to_meet_someone(self):
        from belong.matching.intent import parse_intent
        for message in ("I enjoy reading and hope to meet a girlfriend in Osaka.",
                        "I love hiking and I'm keen to find someone special in Osaka.",
                        "I like climbing and I'm trying to meet a boyfriend in Kyoto.",
                        "I enjoy cooking, open to meeting a boyfriend in Tokyo.",
                        "I like climbing, I'd really like a boyfriend in Osaka.",
                        "I love tennis and a girlfriend would be lovely, in Osaka."):
            self.assertEqual(parse_intent(message).kind, "dating", message)
            draft = agent.draft_from_message(message, "Ken", "35-39")
            self.assertEqual(draft["kind"], "dating", message)
            self.assertIsNone(agent.kind_question(message), message)
        # Someone they already have is never who they're looking for.
        self.assertIsNone(parse_intent("I love hiking with my girlfriend. Based in Osaka.").kind)

    def test_a_kind_only_from_a_hobby_is_asked_never_recorded(self):
        for message, label in (("I love climbing. Based in Osaka.", "a climbing partner"),
                               ("I enjoy tennis and chess, I'm in Tokyo.", "a tennis partner"),
                               ("I love climbing. Looking for someone fun in Osaka.", "a climbing partner")):
            question = agent.kind_question(message)
            self.assertTrue(question["text"].startswith(f"Are you looking for {label}? Or something else?"), message)
            draft = agent.draft_from_message(message, "Ken", "35-39")
            self.assertIsNone(draft["kind"], message)
            self.assertNotIn("kind", {f["key"] for f in draft["fields"]})
        self.assertIn("Are you looking for a climbing partner, or something else?",
                      agent.draft_from_message("I love climbing. Based in Osaka.", "Ken", "35-39")["notes"][0])

    def test_in_the_demo_the_answer_decides_and_keeps_the_first_message(self):
        for answer, kind in (("yes", "climbing"), ("a relationship", "dating"), ("a tennis partner", "other")):
            s = DemoSession(seed=7, agent=None)
            s.verify("A", "Ken", 37)
            s.agent_message("A", "I love climbing. Based in Osaka.")
            self.assertIsNone(s.phone_state("A")["listings"][0]["draft"])  # asked, nothing saved
            s.agent_message("A", answer)
            draft = s.phone_state("A")["listings"][0]["draft"]
            values = {f["key"]: f["value"] for f in draft["fields"]}
            self.assertEqual((draft["kind"], values["city"]), (kind, "Osaka"), answer)

    def test_the_hobby_first_report_in_the_demo(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", self.HOBBY_FIRST + " in Osaka.")
        self.assertIn("a date, or a climbing partner", s.phone_state("A")["listings"][0]["chat"][-1]["text"])
        s.agent_message("A", "a date")
        self.assertEqual([l["draft"]["kind"] for l in s.phone_state("A")["listings"]], ["dating"])


class TheAnswerChoosesWhatItNames(unittest.TestCase):
    OPTIONS = agent.kind_question(VISITING)["options"]  # a date, or a climbing partner
    PARTNER = agent.kind_question("Looking for a partner in Berlin.")["options"]  # a date, or a partner for what?

    def kinds(self, answer, options=None):
        return [c["kind"] for c in agent.chosen_kinds(answer, options or self.OPTIONS) or []]

    def test_an_activity_only_mentioned_is_not_chosen(self):
        for answer in ("a date, I just like climbing", "a date, I climb sometimes though", "a date. I really love climbing",
                       "a date, I'm into climbing", "a date, and I go climbing on weekends"):
            self.assertEqual(self.kinds(answer), ["dating"], answer)
            self.assertEqual(self.kinds(answer, self.PARTNER), ["dating"], answer)

    def test_an_activity_named_as_the_choice_is(self):
        for answer in ("climbing", "rock climbing", "just climbing please", "climbing on weekends", "the climbing one",
                       "I'd like a climbing partner", "not a date, climbing"):
            self.assertEqual(self.kinds(answer), ["climbing"], answer)
        for answer in ("both", "a date and a climbing partner", "both, please", "date and climbing"):
            self.assertEqual(self.kinds(answer), ["dating", "climbing"], answer)

    def test_otherwise_it_asks_once_more(self):
        for answer in ("I climb on weekends", "hmm", "yes", "no"):
            self.assertIsNone(agent.chosen_kinds(answer, self.OPTIONS), answer)
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", VISITING)
        s.agent_message("A", "a date, I just like climbing")
        self.assertEqual([l["draft"]["kind"] for l in s.phone_state("A")["listings"]], ["dating"])  # one listing


class RomanticAnswers(unittest.TestCase):
    ANSWERS = ("a life partner", "a long-term partner", "looking for love", "a wife", "someone to marry", "a soulmate",
               "a relationship", "a girlfriend")

    def test_they_choose_the_date(self):
        for options in (TheAnswerChoosesWhatItNames.OPTIONS, TheAnswerChoosesWhatItNames.PARTNER):
            for answer in self.ANSWERS:
                self.assertEqual(agent.chosen_kinds(answer, options), [{"kind": "dating", "activity": None}], answer)
        self.assertIsNone(agent.chosen_kinds("not a relationship", TheAnswerChoosesWhatItNames.OPTIONS))

    def test_the_first_message_is_kept(self):
        for answer in self.ANSWERS:
            s = DemoSession(seed=7, agent=None)
            s.verify("A", "Ken", 37)
            s.agent_message("A", VISITING)
            s.agent_message("A", answer)
            listings = s.phone_state("A")["listings"]
            self.assertEqual(len(listings), 1, answer)
            values = {f["key"]: f["value"] for f in listings[0]["draft"]["fields"]}
            self.assertEqual((listings[0]["draft"]["kind"], values["city"], values["wants_gender"], values["wants_age"]),
                             ("dating", "Osaka", ["woman"], "30-49"), answer)

    def test_a_kind_that_wasnt_offered_starts_over(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", VISITING)
        s.agent_message("A", "Actually I need a roommate in Tokyo from November.")
        draft = s.phone_state("A")["listings"][0]["draft"]
        self.assertEqual((draft["kind"], {f["key"]: f["value"] for f in draft["fields"]}["city"]), ("roommate", "Tokyo"))


class DetailsInTheAnswer(unittest.TestCase):
    def test_what_the_answer_says_besides_the_choice_is_kept(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'd like to meet a woman in Kyoto. I go running most mornings.")
        s.agent_message("A", "a date, I'm a man, ideally 30 to 45")
        draft = s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["gender"], values["wants_age"]), ("dating", "man", "30-45"))
        self.assertIn("I'm a man, ideally 30 to 45.", draft["intent_text"])

    def test_the_choice_itself_adds_nothing(self):
        options = TheAnswerChoosesWhatItNames.OPTIONS
        for answer in ("a date", "Both, please", "just climbing, not a date", "a life partner", "yes"):
            self.assertEqual(agent.answer_details(answer, options, agent.chosen_kinds(answer, options) or []), "", answer)
        self.assertEqual(agent.answer_details("climbing, on weekends", options, [{"kind": "climbing", "activity": None}]),
                         "on weekends.")


class AnyActivityNextToADate(unittest.TestCase):
    def test_the_activity_reader_is_general(self):
        for message, label in (("I'd like to meet a man in Kyoto. I play tennis most weekends.", "a tennis partner"),
                               ("Hoping to meet a woman in Osaka. I go hiking a lot.", "a hiking partner"),
                               ("Want to meet a woman in Osaka, I enjoy chess and jazz.", "a chess partner"),
                               ("Looking to meet a man in Lisbon. We could play golf.", "a golf partner"),
                               ("I'd like to meet a woman in Berlin, I do yoga on Sundays.", "a yoga partner")):
            question = agent.kind_question(message)
            self.assertIsNotNone(question, message)
            self.assertTrue(question["text"].startswith(f"Are you looking for a date, or {label}? Or both?"), message)
        # Romantic words leave no doubt: then what they do or enjoy is a hobby. And nobody else's hobby counts.
        for message in ("Looking for a relationship. I play tennis on weekends.",
                        "Looking for a relationship with a woman in Osaka. I enjoy climbing.",
                        "Looking to meet a man in Osaka, who loves hiking and chess.",
                        "I want to meet a woman in Osaka. I love dogs.", "I'd like to meet a man. I go out a lot."):
            self.assertIsNone(agent.kind_question(message), message)

    def test_both_makes_a_listing_for_that_activity(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'd like to meet a man in Kyoto. I play tennis most weekends.")
        s.agent_message("A", "both")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([l["draft"]["kind"] for l in listings], ["dating", "other"])
        other = {f["key"]: f["value"] for f in listings[1]["draft"]["fields"]}
        self.assertEqual((other["seeking"], other["headline"]), ("a tennis partner", "Looking for a tennis partner"))


class AgesSaidCarefully(unittest.TestCase):
    def test_negation_and_bounds(self):
        from belong.matching.intent import dating_wants
        for text, years in (("no one over 40", (18, 40)), ("nobody older than 45", (18, 45)),
                            ("not younger than 30", (30, 99)), ("no one under 25", (25, 99)), ("at most 35", (18, 35)),
                            ("at least 30 and no one over 40", (30, 40)), ("in his 30s, no one over 35", (30, 35)),
                            ("under 35, 30+", (30, 34)), ("aged 25 to 34", (25, 34)), ("late 20s to early 30s", (27, 33))):
            self.assertEqual(dating_wants(f"Looking for a woman in Osaka, {text}.")[1], years, text)

    def test_both_bounds_under_one_no(self):
        from belong.matching.intent import dating_wants
        for text in ("no one under 30 or over 45", "nobody over 45 or under 30", "not younger than 30 or older than 45",
                     "no one younger than 30, or older than 45", "neither under 30 nor over 45",
                     "nobody below 30 and nobody above 45", "no one over 45 or anyone under 30"):
            self.assertEqual(dating_wants(f"Looking for a woman in Osaka, {text}.")[1], (30, 45), text)
            self.assertEqual(agent.wanted_ages(f"Looking for a woman in Osaka, {text}.", "35-39")[0], "30-45", text)

    def test_a_bare_number_reads_the_same_everywhere(self):
        """The agent doesn't guess an age range from a bare number, and nor does matching: the note, the
        phone and the matching all say the same thing (the no-age default)."""
        from belong.matching.intent import default_age_range, parse_intent, read_listing
        for message in ("Looking for a woman in Osaka, 30.", "Looking for a woman in Osaka aged 30.",
                        "Looking for a woman in Osaka, 30 years old."):
            draft = agent.draft_from_message(message, "Ken", "30-34")
            self.assertTrue(any("for now you're open to 25 to 39" in n for n in draft["notes"]), message)
            wants = agent.dating_wants(draft)  # what the phone shows
            self.assertEqual((wants["ages"], wants["default"]), (None, [25, 39]), message)
            self.assertIsNone(parse_intent(message).age_range, message)
            listing = person_listing("him", message, "30-34", "dating", "man", ["woman"])
            self.assertIsNone(understood(ShownView(listing)).age_range, message)  # matching: the default, 25 to 39
            self.assertEqual(default_age_range("30-34"), (1, 3))
            self.assertIsNone(read_listing(message, {"kind": "dating"}.get).age_range, message)

    def test_numbers_that_arent_ages(self):
        from belong.matching.intent import dating_wants
        for text in ("with 10 years of experience", "I have 2 kids", "within 30 km", "20 minutes from Osaka",
                     "between 20 and 30 minutes away", "who likes 80s music", "budget $40", "about 30 minutes away"):
            self.assertIsNone(dating_wants(f"Looking for a woman, {text}.")[1], text)
        self.assertEqual(dating_wants("Looking for a woman, 30 years old.")[1], (27, 33))  # still an age

    def test_relative_ages_with_not(self):
        from belong.matching.intent import relative_years
        self.assertEqual(relative_years("Looking for a man not older than me", "35-39"), (30, 39))
        self.assertEqual(relative_years("Looking for a man, nobody younger than me", "35-39"), (35, 49))

    def test_the_offline_agent_asks_instead_of_guessing_from_a_number(self):
        for message in ("Looking for a woman in Osaka, 30.", "Looking for a woman in Osaka aged 30."):
            self.assertEqual(agent.wanted_ages(message, "30-34")[0], None, message)
            draft = agent.draft_from_message(message, "Ken", "30-34")
            self.assertNotIn("wants_age", {f["key"] for f in draft["fields"] if f["value"]})
            self.assertTrue(any(n.startswith("I wasn't sure which ages you meant") for n in draft["notes"]), message)
        self.assertEqual(agent.wanted_ages("Looking for a woman in Osaka, no one over 40.", "30-34")[0], "18-40")
        self.assertEqual(agent.wanted_ages("Looking for a woman in Osaka around 30.", "30-34")[0], "27-33")
        draft = agent.draft_from_message("Looking for a woman within 30 km of Osaka, I have 2 kids.", "Ken", "30-34")
        self.assertNotIn("wants_age", {f["key"] for f in draft["fields"] if f["value"]})


class ChangingTheKindInTheEditor(unittest.TestCase):
    def test_the_layout_for_a_kind(self):
        dating, climbing = agent.kind_layout("dating"), agent.kind_layout("climbing")
        self.assertTrue({"gender", "wants_gender", "wants_age"} <= {r["key"] for r in dating["rows"]})
        self.assertTrue({"wants_gender", "wants_age", "gender"} <= set(climbing["drop"]))
        self.assertIn("level", dating["drop"])
        self.assertEqual(agent.kind_layout(None)["drop"], [])  # no kind chosen yet: nothing taken away

    def test_who_and_what_ages_never_go_live_on_a_listing_that_isnt_dating(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "Looking for a relationship with a woman in Osaka, 30 to 39. I'm a man.")
        body = fill(s.slots["A"].draft, values={"kind": "climbing"})
        self.assertTrue({"wants_gender", "wants_age"} <= {f["key"] for f in body["fields"]})
        s.submit("A", body)
        live = s.board.listing("demoA-1")
        self.assertEqual(live.get("kind").value, "climbing")
        self.assertIsNone(live.get("wants_gender"))
        self.assertIsNone(live.get("wants_age"))
        # No kind on the listing, and a description that isn't dating: the same.
        s.agent_message("A", "Looking for a climbing partner in Osaka.")
        body = fill(s.slots["A"].draft)
        body["fields"] = [f for f in body["fields"] if f["key"] != "kind"] + [
            {"key": "wants_gender", "value": ["woman"], "visibility": SHOWN}]
        s.submit("A", body)
        self.assertIsNone(s.board.listing("demoA-1").get("wants_gender"))

    def test_the_phone_rebuilds_the_rows_and_saves_only_what_can_match(self):
        import json
        import shutil
        import subprocess
        import tempfile
        from tests.test_dating import _js
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        dating = agent.draft_from_message("Looking for a relationship with a woman in Osaka. I'm a man.", "Ken", "35-39")
        cases = {"dating": dating, "climbing": agent.kind_layout("climbing"), "to_dating": agent.kind_layout("dating"),
                 "nobody": agent.draft_from_message("Looking for someone special in Osaka.", "Ken", "35-39"),
                 "wants": agent.dating_wants(dating)}
        code = "\n".join(_js(n) for n in ("const hasValue", "const spanOf", "const agesText", "const UNDERSTOOD",
                                          "const GENDER_WORDS", "function wantsText", "function matchFacts",
                                          "const CLIENT_WORDS", "const FILLER", "const words", "function sides",
                                          "function draftValues", "function layoutFor", "function lookingFor",
                                          "function canSave", "function checklist", "function relayout"))
        code += "\nconst val = v => Array.isArray(v) ? v.join(', ') : v;\n"
        code += f"const C = {json.dumps(cases)};\n" + r"""
const keys = fs => fs.map(f => f.key);
const stash = {};
const d = JSON.parse(JSON.stringify(C.dating));
const l = { listed: true, wants: C.wants };
const out = { live_dating: canSave({ draft: d }, l) };
d.kind = "climbing"; d.fields = relayout(d.fields, C.climbing, stash);
out.climbing_keys = keys(d.fields);
d.kind = "dating"; d.fields = relayout(d.fields, C.to_dating, stash);
out.back = d.fields.filter(f => ["gender", "wants_gender"].includes(f.key)).map(f => [f.key, f.value]);
const n = JSON.parse(JSON.stringify(C.nobody));
out.live_nobody = canSave({ draft: n }, { listed: true, wants: null });
const fresh = JSON.parse(JSON.stringify(C.nobody)); fresh.kind = "climbing";
fresh.fields = relayout(fresh.fields, C.climbing, {}); fresh.kind = "dating";
fresh.fields = relayout(fresh.fields, C.to_dating, {});
out.switched_keys = keys(fresh.fields);
out.match_dating = matchFacts({ first_name: "A", wants_gender: ["woman"], kind: "dating" }, true).wants;
out.match_climbing = matchFacts({ first_name: "A", wants_gender: ["woman"], kind: "climbing" }, false).wants;
console.log(JSON.stringify(out));"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        result = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        self.assertTrue(out["live_dating"])
        self.assertFalse({"gender", "wants_gender", "wants_age", "kinks", "children"} & set(out["climbing_keys"]))
        self.assertIn("level", out["climbing_keys"])
        self.assertEqual(out["back"], [["gender", "man"], ["wants_gender", ["woman"]]])  # put aside, and back
        self.assertFalse(out["live_nobody"])  # a live dating listing nobody can match can't be saved
        self.assertTrue({"gender", "wants_gender", "wants_age"} <= set(out["switched_keys"]))
        self.assertNotIn("level", out["switched_keys"])
        self.assertEqual((out["match_dating"], out["match_climbing"]), ("a woman", ""))


if __name__ == "__main__":
    unittest.main()


class FinalReviewFixes(unittest.TestCase):
    """The last review's findings, each pinned."""

    @staticmethod
    def tennis(fields):
        return {"kind": "other", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking for a tennis partner in Kyoto.",
                "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a tennis partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a tennis partner"], "visibility": SHOWN}] + fields}

    def test_a_hidden_headline_or_topics_stay_hidden_on_a_listing_that_isnt_dating(self):
        hidden = [{"key": "headline", "values": ["Looking for a tennis partner"], "visibility": HIDDEN},
                  {"key": "topics", "values": ["tennis"], "visibility": HIDDEN}]
        draft, problems, _ = build_draft(self.tennis(hidden), "Mia", "30-34")
        by = {f["key"]: f["visibility"] for f in draft["fields"]}
        self.assertEqual((by["headline"], by["topics"]), (HIDDEN, HIDDEN))
        self.assertEqual((by["seeking"], by["offering"]), (SHOWN, SHOWN))  # matching needs these shown
        # The person hid them before, and a later save that rewrites them doesn't say: still hidden.
        rewrite = [{"key": "headline", "values": ["Playing tennis in Kyoto"], "visibility": SHOWN}]
        current = {**draft, "choices": {"headline": HIDDEN, "topics": HIDDEN}}
        again, _, _ = build_draft(self.tennis(rewrite), "Mia", "30-34", current)
        by = {f["key"]: f["visibility"] for f in again["fields"]}
        self.assertEqual(by["topics"], HIDDEN)

    def test_a_country_merely_mentioned_isnt_where_they_are(self):
        for text in ("My parents came from Nigeria. Looking for a partner.", "I grew up in Ghana and love chess.",
                     "Originally from Brazil, now looking for a tennis partner."):
            self.assertNotIn("country", agent.quick_facts(text), text)
        self.assertEqual(agent.quick_facts("Looking for a maths tutor in Germany.").get("country"), "Germany")
        self.assertEqual(agent.quick_facts("i live in malaga, spain").get("country"), "Spain")

    def draft_kind(self, *messages):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        for m in messages:
            s.agent_message("A", m)
        return s, (s.slots["A"].listing().draft or {}).get("kind")

    def test_what_they_do_beside_a_romantic_request_is_a_hobby(self):
        for text in ("I climb a lot. Looking for a girlfriend in Osaka.",
                     "Looking for a relationship. I go running every morning, Osaka.",
                     "Looking for a girlfriend in Osaka. I go climbing at weekends."):
            self.assertEqual(self.draft_kind(text)[1], "dating", text)

    def test_a_partner_named_only_by_a_sensitive_word_is_asked_and_the_answer_taken(self):
        s, kind = self.draft_kind("Looking for a chemo buddy in Lisbon.")
        self.assertIsNone(kind)
        self.assertIn("What would you like a partner for?", s.slots["A"].listing().chat[-1]["text"])
        s.agent_message("A", "tennis")
        d = s.slots["A"].listing().draft
        shown = {f["key"]: f["value"] for f in d["fields"] if f["visibility"] == SHOWN and f["value"] not in (None, "", [])}
        self.assertEqual((d["kind"], shown.get("city"), shown.get("seeking")), ("other", "Lisbon", "a tennis partner"))
        self.assertNotIn("chemo", str(shown).lower())

    def test_the_phone_sends_a_shown_or_hidden_choice_as_soon_as_it_is_made(self):
        from pathlib import Path
        page = (Path(__file__).resolve().parent.parent / "belong" / "web" / "demo.html").read_text()
        for fn in ("function setVis", "function enable"):
            self.assertIn("pushVisibility(slot)", page[page.index(fn):page.index(fn) + 600], fn)


class WhereAndWhileReplying(unittest.TestCase):
    """Where someone is, read only when said about them; and choices made while the model agent replies."""

    def test_a_place_said_about_someone_else_or_the_past_isnt_where_they_are(self):
        # Only a few explicit forms about the person, now; anything else stays blank and the agent asks.
        nowhere = ["My parents live in Nigeria. Looking for a chess partner.", "My family lives in Mexico",
                   "My sister lives in Canada", "I used to live in Israel, now looking for a tennis partner",
                   "My parents came from Nigeria. Looking for a partner.", "My girlfriend lives in Osaka. Looking for a tennis partner.",
                   "My boyfriend is in Tokyo. Looking for a chess partner.", "My daughter is lovely. She lives in Osaka.",
                   "I'm in my thirties, Berlin is where my parents live", "I was in Tokyo last month and want a chess partner",
                   "I escaped from North Korea", "I'm making turkey sandwiches", "I'd love to visit Japan someday",
                   "I come from a village in Nigeria", "I'm from northern Spain", "Deported to Mexico last year", "Big Chile fan",
                   "I grew up in Munich but live in Osaka now", "My roommate lives in Berlin, looking for a chess partner"]
        for text in nowhere:
            facts = agent.quick_facts(text)
            self.assertFalse(facts.get("city") or facts.get("country"), text)
        here = {"Born in Kyoto, based in Berlin": ("Berlin", "Germany"),
                "Originally from London, now living in Tokyo": ("Tokyo", "Japan"),
                "I'm from Brazil but now I'm in Portugal": (None, "Portugal"),
                "I moved to Spain last year and want a tennis partner": (None, "Spain"),
                "i live in malaga, spain, looking to meet people": (None, "Spain"),
                "Actually I need a roommate in Tokyo from November.": ("Tokyo", "Japan"),
                "Visiting Osaka and looking to meet a woman around my age": ("Osaka", "Japan"),
                "Need someone to belay with at Osaka gyms, weekends.": ("Osaka", "Japan"),
                "Looking for a girlfriend in Osaka": ("Osaka", "Japan"),
                "Looking for friends in Lisbon": ("Lisbon", "Portugal"),
                "Looking for a maths tutor in Germany.": (None, "Germany")}
        for text, (city, country) in here.items():
            facts = agent.quick_facts(text)
            self.assertEqual((facts.get("city"), facts.get("country")), (city, country), text)

    def test_with_a_model_agent_nothing_is_guessed_first(self):
        """No keyword guess goes up while the model replies (a place, or "online preferred" read as either way):
        the model reads their words itself, and its save fills the profile."""
        message = "Could someone teach me pottery? Online preferred, weekends, I can pay 30 an hour in Osaka."
        save = {"kind": "other", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking for someone to teach me pottery online.",
                "fields": [{"key": "format", "values": ["Online"], "visibility": SHOWN},
                           {"key": "headline", "values": ["Looking for a pottery teacher"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a pottery teacher"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a pottery student"], "visibility": SHOWN}]}
        client = ScriptedClient(reply(tool(save, "t1")), reply(text("Saved.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Ada", 30)
        s.prefill("A", message)  # as the phone does, before the agent replies
        self.assertIsNone(s.slots["A"].draft)
        s.agent_message("A", message)
        values = {f["key"]: f["value"] for f in s.slots["A"].draft["fields"] if f["value"] not in (None, "", [])}
        self.assertEqual(values["format"], "Online")  # the model's reading
        self.assertNotIn("schedule", values)  # nothing it didn't say

    def test_a_guessed_place_is_dropped_unless_the_model_agent_says_it(self):
        save = {"kind": "other", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking for a chess partner.",
                "fields": [{"key": "seeking", "values": ["a chess partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a chess partner"], "visibility": SHOWN}]}
        client = ScriptedClient(reply(tool(save, "t1")), reply(text("Saved.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Ada", 30)
        listing = s.slots["A"].listing(s.new_listing("A"))
        listing.draft = agent.empty_draft("Ada", "30-34")
        listing.draft["fields"].append({"key": "country", "label": "Country", "value": "Nigeria", "visibility": SHOWN,
                                        "sensitive": False, "multi": False, "enabled": True, "locked": False, "options": []})
        listing.provisional = {"country"}  # as if the instant first draft had guessed it
        s.agent_message("A", "My parents live in Nigeria. Looking for a chess partner.", listing.id)
        self.assertNotIn("Nigeria", client.requests[0]["system"])  # never handed to the model
        values = {f["key"]: f["value"] for f in listing.draft["fields"] if f["value"] not in (None, "", [])}
        self.assertNotIn("country", values)  # and dropped, since its draft didn't say it

    def test_a_choice_made_while_the_model_agent_replies_wins(self):
        save = {"kind": "other", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking for a tennis partner in Kyoto.",
                "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN},
                           {"key": "headline", "values": ["Looking for a tennis partner"], "visibility": SHOWN},
                           {"key": "seeking", "values": ["a tennis partner"], "visibility": SHOWN},
                           {"key": "offering", "values": ["a tennis partner"], "visibility": SHOWN}]}
        s = None

        class Midway(ScriptedClient):
            def _create(self, **request):
                if len(self.requests) == 0:  # the person hides two details while the model is thinking
                    s.set_visibility("A", None, {"city": HIDDEN, "headline": HIDDEN})
                return super()._create(**request)
        client = Midway(reply(tool(save, "t1")), reply(text("Saved.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Ada", 30)
        s.agent_message("A", "Looking for a tennis partner in Kyoto.")
        vis = {f["key"]: f["visibility"] for f in s.slots["A"].listing().draft["fields"]}
        self.assertEqual((vis["city"], vis["headline"]), (HIDDEN, HIDDEN))
        self.assertEqual(vis["seeking"], SHOWN)
