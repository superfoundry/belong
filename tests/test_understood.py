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
               Field("wants_age", "30 - 39", SHOWN, False), Field("wants_age", "30", SHOWN, False)]
        for f in bad:
            result = self.admit(f)
            self.assertFalse(result.accepted, f)
            self.assertIn("listing_schema", result.failed_checks, f)

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
        self.assertIn("- wants_gender (default shown): woman, man, nonbinary", SYSTEM_PROMPT)
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


if __name__ == "__main__":
    unittest.main()
