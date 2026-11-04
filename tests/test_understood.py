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
from tests.helpers import detail, warden
from tests.test_agent_llm import ScriptedClient, det, reply, text, tool
from tests.test_demo import fill

# The two reports, each an instance of the general problem.
VISITING = ("Visiting Osaka and looking to meet a woman around my age for something genuine. I enjoy rock climbing "
            "and tennis.")
EARLY_30S = "Looking to meet a man in Osaka, around early 30s, who's into photography and jazz."
# The shared list of details in use (Warden.details_in_use) the offline agent reads in these tests: it writes only
# listed words. In the demo it is the Warden's own list for the synthetic people, with "a climbing partner" and
# the topic "bouldering" on it.
IN_USE = {"phrases": [{"key": "seeking", "value": "a climbing partner"}, {"key": "offering", "value": "a climbing partner"}]
          + [{"key": "topics", "value": t} for t in ("chess", "golf", "hiking", "running", "tennis", "yoga")],
          "details": [{"about": "me", "name": "interests", "compare": "same"}],
          "values": [{"about": "me", "name": "interests", "value": v} for v in ("rock climbing", "tennis")]}
WORDS = agent.words_in_use(IN_USE)


def person_listing(pid, intent, band, kind=None, gender=None, wants_gender=None, wants_age=None, extra=(),
                   visibility=SHOWN, exactness=0.6):
    """A listing in Osaka; `visibility` applies to kind / wants_gender / wants_age."""
    fields = [Field("first_name", pid, SHOWN, False), Field("city", "Osaka", SHOWN, False),
              Field("country", "Japan", SHOWN, False), Field("age_band", band, SHOWN, False),
              detail("interests", "tennis"), detail("free time", "weekends")]
    if gender:
        fields.append(Field("gender", gender, SHOWN, False))
    if kind:
        fields.append(Field("kind", kind, visibility, False))
    if wants_gender:
        fields.append(Field("wants_gender", tuple(wants_gender), visibility, False))
    if wants_age:
        fields.append(Field("wants_age", wants_age, visibility, False))
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


def phrases(seeking, offering=None):
    """What a listing that isn't dating is for: its "looking for" and "you are" phrases."""
    return (Field("seeking", seeking, SHOWN, False), Field("offering", offering or seeking, SHOWN, False))


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
        level = detail("level", "intermediate", compare="steps", steps=("beginner", "intermediate", "advanced"))
        climbers = [person_listing("c1", "Looking for a climbing partner in Osaka on weekends.", "50+", "other",
                                   extra=(*phrases("a climbing partner"), level)),
                    person_listing("c2", "Need someone to belay with in Osaka.", "40-49", gender="woman",
                                   extra=phrases("a climbing partner"))]
        _, board = board_of(him, *women, man, *climbers)
        his = {p - {"him"} for p in pairs(board) if "him" in p}
        self.assertEqual({next(iter(p)) for p in his}, {"w0", "w1", "w2"})  # women aged 30 to 49 only
        # The kind he recorded says dating; the climbers' phrases say what they want.
        self.assertEqual(understood(ShownView(him)).kind, "dating")
        self.assertEqual({understood(ShownView(c)).kind for c in climbers}, {"other"})
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
        """Shown "other": words that read like dating don't make it dating, and its phrases decide. Shown
        "dating": an activity in the description doesn't make it anything else."""
        paddler = person_listing("k", "Want to meet women for weekend kayaking trips in Osaka.", "30-34", "other",
                                 extra=phrases("a kayaking partner"))
        paddlers = person_listing("k2", "Kayaking buddy in Osaka for weekend trips.", "30-34", "other",
                                  extra=phrases("a kayaking partner"))
        dater = person_listing("d", "Looking for a kayaking partner in Osaka.", "30-34", "dating", "man", ["woman"],
                               "25-39")
        _, board = board_of(paddler, paddlers, dater, *women_wanting_men("w", ["30-34"]))
        self.assertEqual(pairs(board), {frozenset(("k", "k2")), frozenset(("d", "w0"))})
        self.assertEqual((understood(ShownView(paddler)).kind, understood(ShownView(dater)).kind), ("other", "dating"))

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
        climber = person_listing("c", "Looking for a climbing partner in Osaka.", "35-39", "other",
                                 extra=phrases("a climbing partner"))
        her = women_wanting_men("w", ["35-39"])[0]
        _, board = board_of(him, climber, her)
        views = {l.person_id: ShownView(l) for l in board.listings()}
        lists, _ = shortlists({pid: understood(v) for pid, v in views.items()}, views, 10, words=index_words)
        self.assertEqual(lists["him"], ["w0"])

    def test_a_kind_or_age_range_outside_the_fixed_choices_is_never_read(self):
        """A kind that isn't one of MATCH_KINDS is as if unstated: the description says only whether it is
        dating, and phrases make it "other". An age range not in the fixed form is never read either."""
        for kind in (None, "climbing", "Dating"):
            listing = person_listing("p", "Looking for a climbing partner.", "30-34", kind)
            self.assertIsNone(understood(ShownView(listing)).kind, kind)
            listing = person_listing("p", "Looking for a climbing partner.", "30-34", kind,
                                     extra=phrases("a climbing partner"))
            self.assertEqual(understood(ShownView(listing)).kind, "other", kind)
        loose = person_listing("p", "Looking for a relationship with a woman in Osaka.", "30-34", "dating", "man",
                               ["woman"], "early 30s")
        self.assertIsNone(understood(ShownView(loose)).age_range)


class HiddenNeverDecides(unittest.TestCase):
    def test_a_hidden_kind_or_wants_is_as_if_it_werent_there(self):
        # Hidden, "dating" can't make a description that asks for no date dating; the description decides, as
        # if unstated, and phrases make it "other".
        hidden_kind = person_listing("h", "Looking for a climbing partner in Osaka.", "30-34", "dating",
                                     visibility=HIDDEN)
        self.assertIsNone(understood(ShownView(hidden_kind)).kind)
        with_phrases = person_listing("h", "Looking for a climbing partner in Osaka.", "30-34", "dating",
                                      extra=phrases("a climbing partner"), visibility=HIDDEN)
        self.assertEqual(understood(ShownView(with_phrases)).kind, "other")
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
                 reason_text([("country", "Japan", "Japan")]), reason_text([("format", "Online", "Online")])]
        # The open-detail templates too: agreeing on a same detail, agreeing on a scale, and a wish met each
        # way. Names and values here are slot words with no fixed choice in them, so any hit is the template's.
        scale = ("low", "middle", "high")
        mine = {f.key: f for f in (detail("tempo", "slow"), detail("skill", "middle", compare="steps", steps=scale),
                                   detail("skill", ("middle", "high"), about="them", compare="steps", steps=scale))}
        theirs = {f.key: f for f in (detail("tempo", "slow"), detail("skill", "high", compare="steps", steps=scale),
                                     detail("skill", "middle", about="them", compare="steps", steps=scale))}
        for ref in ("agree:tempo", "agree:skill", "wants:skill"):
            texts.append(reason_text([(ref, mine, theirs)]))
            self.assertTrue(texts[-1], ref)  # each template is worded, not left out
        self.assertIn("They'd like your skill", texts[-1])  # the wish both ways
        self.assertIn("You'd like their skill", texts[-1])
        words = {w for t in texts for w in re.findall(r"[a-z]+", t.lower())}
        self.assertFalse(words & {w for v in MATCH_KINDS + GENDERS for w in v.split()}, texts)

    def test_two_open_ended_listings_in_nearby_cities_are_introduced(self):
        def phrases(pid, city, seeking, offering):
            return Listing(pid, "Hoping to meet people.", 0.6, (
                Field("city", city, SHOWN, False), Field("country", "Japan", SHOWN, False),
                Field("travel", "Up to 100 km", SHOWN, False), Field("kind", "other", SHOWN, False),
                Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False),
                detail("free time", "weekends")), "t")
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
                           {"key": "gender", "values": ["man"], "visibility": SHOWN}],
                "details": [det("interests", "rock climbing", "tennis")]}
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
        args = {"kind": "other", "exactness": 0.6, "intent_text": "Looking for a climbing partner in Osaka.",
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
        draft = agent.draft_from_message(VISITING + " I'm a man.", "Ken", "35-39", kind="dating", words=WORDS)
        values = {f["key"]: (f["value"], f["visibility"]) for f in draft["fields"] if f["value"]}
        self.assertEqual(values["kind"], ("dating", SHOWN))
        self.assertEqual(values["wants_gender"], (["woman"], SHOWN))
        self.assertEqual(values["wants_age"], ("30-49", SHOWN))
        self.assertEqual(values["gender"], ("man", SHOWN))
        self.assertEqual(values["me:interests"], (["rock climbing", "tennis"], SHOWN))  # the hobby, in listed words
        # Every request that isn't dating is "other", with phrases in listed words.
        for message, phrases in (("Looking for a climbing partner in Osaka on weekends.", ("a climbing partner",) * 2),
                                 ("Need a running coach in Berlin.", ("a running coach", "a running student")),
                                 ("I teach golf in Lisbon.", ("a golf student", "a golf teacher")),
                                 ("Looking for a tennis partner in Tokyo.", ("a tennis partner",) * 2)):
            values = {f["key"]: f["value"] for f in agent.draft_from_message(message, "A", "25-29", words=WORDS)["fields"]}
            self.assertEqual((values["kind"], values["seeking"], values["offering"]), ("other", *phrases), message)
        # Nothing listed: no kind it could be sure of, no phrase, and it asks.
        draft = agent.draft_from_message("Looking for a tennis partner in Tokyo.", "A", "25-29")
        self.assertIsNone(draft["kind"])
        self.assertIn("What would you like a partner for?", draft["notes"][0])


class TheRuleAgentWritesOnlyListedWords(unittest.TestCase):
    """The offline agent's only vocabulary is the shared list of details in use: a listed value under the one
    name it's listed under, about the person (or the request, on a listing with a kind), never a wish."""

    LISTED = {"phrases": IN_USE["phrases"],
              "details": [{"about": "me", "name": "free time", "compare": "same"},
                          {"about": "me", "name": "level", "compare": "steps"},
                          {"about": "me", "name": "noise", "compare": "same"},
                          {"about": "me", "name": "voice", "compare": "same"},
                          {"about": "both", "name": "genre", "compare": "same"},
                          {"about": "them", "name": "speed", "compare": "same"}],
              "scales": [{"about": "me", "name": "level", "steps": ["beginner", "intermediate", "advanced"]}],
              "values": [{"about": "me", "name": "free time", "value": v} for v in ("weekends", "any")]
              + [{"about": "me", "name": "level", "value": v} for v in ("intermediate", "advanced", "expert")]
              + [{"about": "me", "name": "noise", "value": "quiet"}, {"about": "me", "name": "voice", "value": "quiet"},
                 {"about": "both", "name": "genre", "value": "ska"}, {"about": "them", "name": "speed", "value": "gentle"}]}

    def details(self, message, **kw):
        draft = agent.draft_from_message(message, "Ken", "35-39", words=agent.words_in_use(self.LISTED), **kw)
        return {f["key"]: (f["value"], f["compare"], f["steps"], f["visibility"]) for f in draft["fields"] if ":" in f["key"]}

    def test_a_listed_value_under_its_one_name(self):
        steps = ["beginner", "intermediate", "advanced"]
        self.assertEqual(self.details("Looking for a climbing partner in Osaka. I'm free on weekends, anytime "
                                      "really. I'm Intermediate. Quiet person, into ska, gentle."),
                         {"me:free time": (["weekends"], "same", [], SHOWN),  # never "any" from "anytime"
                          "me:level": (["intermediate"], "steps", steps, SHOWN),  # the step in its own form
                          "both:genre": (["ska"], "same", [], SHOWN)})  # "quiet": two names; "gentle": only a wish
        self.assertNotIn("me:level", self.details("Looking for a climbing partner. Intermediate, maybe advanced."))
        self.assertNotIn("me:level", self.details("Looking for a climbing partner. I'm an expert."))  # not a step
        # No request yet: nothing about one ("both"), only what's about the person.
        self.assertEqual(set(self.details("I'm free on weekends, into ska.")), {"me:free time"})

    def test_never_a_wish_or_a_denial_as_a_fact(self):
        """A value about the person counts only where they say it of themselves; one about the person they'd like
        to meet, or one they deny, is never written as theirs, and two steps of a scale are neither. Every step of
        a listed scale counts, listed as a value or not."""
        listed = {"phrases": [],
                  "details": [{"about": "me", "name": "spice tolerance", "compare": "steps"},
                              {"about": "me", "name": "chronotype", "compare": "same"}],
                  "scales": [{"about": "me", "name": "spice tolerance", "steps": ["mild", "medium", "hot"]}],
                  "values": [{"about": "me", "name": "spice tolerance", "value": "medium"}]
                  + [{"about": "me", "name": "chronotype", "value": v} for v in ("early riser", "night owl")]}
        words = agent.words_in_use(listed)

        def mine(message):
            draft = agent.draft_from_message(message, "Ines", "30-34", words=words)
            return {f["key"]: f["value"] for f in draft["fields"] if ":" in f["key"]}

        for message, expected in (
                ("I'm mild, hoping to share a table with someone medium.", {"me:spice tolerance": ["mild"]}),
                ("Looking for a dinner companion who is medium.", {}),
                ("Looking for a dinner companion with a night owl schedule.", {}),
                ("Looking for a dinner companion, medium spice.", {}),
                ("Looking for a dinner companion. I'm not medium yet.", {}),
                ("I'm fine with medium, looking for a dinner companion.", {}),
                ("Looking for a gym buddy, someone who's an early riser; I'm a night owl.", {"me:chronotype": ["night owl"]}),
                ("I'm somewhere between mild and medium. Looking for a dinner companion.", {}),
                ("My spice tolerance is hot. Looking for a dinner companion.", {"me:spice tolerance": ["hot"]}),
                ("I'm usually an early riser and looking for a gym buddy.", {"me:chronotype": ["early riser"]}),
                ("I'm new here and my sister is a night owl.", {})):
            self.assertEqual(mine(message), expected, message)

    def test_a_value_about_you_is_a_predicate_of_you(self):
        """Between "I'm" and a value only an article and adverbs or degree words may stand (or the words of the
        detail's own name, as in "my {name} is"), never a verb: a verb makes the value its object, so it says
        nothing about the person. The whole negation family denies, read as sentence structure ("hardly", "far
        from", "neither ... nor", "no longer", "anything but", "except"). What they like or do keeps working, never
        as the object of another -ing verb."""
        listed = {"phrases": [],
                  "details": [{"about": "me", "name": "vocal range", "compare": "same"},
                              {"about": "me", "name": "sight reading", "compare": "steps"},
                              {"about": "me", "name": "available days", "compare": "same"}],
                  "scales": [{"about": "me", "name": "sight reading", "steps": ["shaky", "steady", "fluent"]}],
                  "values": [{"about": "me", "name": "vocal range", "value": v} for v in ("alto", "tenor", "soprano")]
                  + [{"about": "me", "name": "available days", "value": v} for v in ("mondays", "thursdays")]}
        words = agent.words_in_use(listed)

        def mine(message):
            return {r["key"]: r["value"] for r in agent.listed_details(message, words, "other")}

        for message in ("I'm hardly fluent, looking for a choir buddy.", "Honestly I'm barely steady.",
                        "I'm far from fluent but keen to sing.", "I'm neither alto nor soprano, just keen.",
                        "I'm no longer a tenor.", "I'm anything but fluent at it.", "I like every part except alto.",
                        "I'm rarely available on mondays.", "I'm conducting an alto section this term.",
                        "I'm accompanying fluent singers on piano.", "I'm married to a tenor.",
                        "I love coaching shaky sight readers.", "I'm busy on thursdays.",
                        "I'm unavailable mondays.", "I'm nearly fluent now."):
            self.assertEqual(mine(message), {}, message)
        for message, expected in (
                ("I'm a fairly steady sight reader and I'm an alto.",
                 {"me:sight reading": ["steady"], "me:vocal range": ["alto"]}),
                ("I'm mostly a tenor, looking for a choir buddy.", {"me:vocal range": ["tenor"]}),
                ("I'm kind of a shaky sight reader.", {"me:sight reading": ["shaky"]}),
                ("I'm not a soprano, I'm a total alto.", {"me:vocal range": ["alto"]}),
                ("I'm usually available on thursdays.", {"me:available days": ["thursdays"]}),
                ("My sight reading is fluent.", {"me:sight reading": ["fluent"]}),
                ("I'm a retired nurse into alto parts and soprano duets.", {"me:vocal range": ["alto", "soprano"]}),
                ("I enjoy singing and alto harmonies.", {"me:vocal range": ["alto"]})):
            self.assertEqual(mine(message), expected, message)

    def test_a_list_goes_on_only_while_no_clause_starts_and_a_contraction_is_no_possessive(self):
        """A value after a comma or "and" continues a list of the same detail's values only where no other clause
        starts with it: a verb of being or seeming, a modal, the negation family or a request after it makes it what
        that clause is about (a wish, a denial, someone else's), never one more of theirs. And a contraction of "is",
        "has" or "us" after a value ("it's", "that's", "let's", "she's") is no possessive, so the value stays theirs."""
        listed = {"phrases": [],
                  "details": [{"about": "me", "name": "studio role", "compare": "same"},
                              {"about": "me", "name": "free evenings", "compare": "same"},
                              {"about": "me", "name": "glaze skill", "compare": "steps"}],
                  "scales": [{"about": "me", "name": "glaze skill", "steps": ["unsure", "capable", "masterful"]}],
                  "values": [{"about": "me", "name": "studio role", "value": v} for v in ("thrower", "glazer", "kiln tender")]
                  + [{"about": "me", "name": "free evenings", "value": v} for v in ("tuesdays", "saturdays")]}
        words = agent.words_in_use(listed)

        def mine(message):
            return {r["key"]: r["value"] for r in agent.listed_details(message, words, "other")}

        for message, expected in (
                ("I'm a thrower, a glazer would round out our bench nicely.", {"me:studio role": ["thrower"]}),
                ("I'm a kiln tender, a glazer is who I'm looking for.", {"me:studio role": ["kiln tender"]}),
                ("I'm free tuesdays, saturdays seem hopeless though.", {"me:free evenings": ["tuesdays"]}),
                ("I'm free tuesdays, saturdays hardly ever happen.", {"me:free evenings": ["tuesdays"]}),
                ("I'm a thrower and a glazer, after a studio mate.", {"me:studio role": ["thrower", "glazer"]}),
                ("I'm free tuesdays, saturdays and some sundays.", {"me:free evenings": ["tuesdays", "saturdays"]}),
                ("I'm a glazer it's my favourite part", {"me:studio role": ["glazer"]}),
                ("I'm free saturdays that's when the studio is quiet", {"me:free evenings": ["saturdays"]}),
                ("I'm a thrower let's share a wheel", {"me:studio role": ["thrower"]}),
                ("I'm capable she's the masterful one", {"me:glaze skill": ["capable"]}),
                ("I'm a glazer's apprentice.", {})):
            self.assertEqual(mine(message), expected, message)

    def test_a_denied_verb_reaches_the_clause_under_it_and_someones_value_is_theirs(self):
        """A denial before a verb reaches a clause that depends on it ("I don't believe I'm..."), and so do the
        verbs of denial ("I doubt I'm..."); only a form that starts a clause of its own ("..., and I'm", "but I'm")
        starts afresh. A value followed by a possessive is someone else's ("a fluent choir's ..."). A list of one
        detail's values after "I'm" is read whole ("I'm an alto and a soprano")."""
        listed = {"phrases": [],
                  "details": [{"about": "me", "name": "vocal range", "compare": "same"},
                              {"about": "me", "name": "sight reading", "compare": "steps"},
                              {"about": "me", "name": "available days", "compare": "same"}],
                  "scales": [{"about": "me", "name": "sight reading", "steps": ["shaky", "steady", "fluent"]}],
                  "values": [{"about": "me", "name": "vocal range", "value": v} for v in ("alto", "tenor", "soprano")]
                  + [{"about": "me", "name": "available days", "value": v} for v in ("mondays", "thursdays", "fridays")]}
        words = agent.words_in_use(listed)

        def mine(message):
            return {r["key"]: r["value"] for r in agent.listed_details(message, words, "other")}

        for message in ("I don't believe I'm a fluent sight reader yet.", "I never said I'm an alto.",
                        "I wouldn't call myself fluent, and I wouldn't go as far as to say I'm steady.",
                        "I can't honestly claim I'm a tenor.", "I doubt I'm fluent enough for this.",
                        "I'm a fluent choir's accompanist.", "I'm the alto section's page turner.",
                        "I'm an alto's rehearsal pianist."):
            self.assertEqual(mine(message), {}, message)
        for message, expected in (
                ("I'm not a soprano but I'm an alto.", {"me:vocal range": ["alto"]}),
                ("I don't sing much, and I'm a shaky sight reader.", {"me:sight reading": ["shaky"]}),
                ("I think I'm a steady sight reader.", {"me:sight reading": ["steady"]}),
                ("I'm an alto and a soprano, looking for a choir.", {"me:vocal range": ["alto", "soprano"]}),
                ("I'm usually available mondays, thursdays and fridays.",
                 {"me:available days": ["mondays", "thursdays", "fridays"]}),
                ("I'm available mondays or fridays.", {"me:available days": ["mondays", "fridays"]})):
            self.assertEqual(mine(message), expected, message)

    def test_the_details_it_cant_write_stay_from_before(self):
        old = {"kind": "other", "fields": [
            {"key": "seeking", "value": "a climbing partner"},
            agent.detail_row("me", "instrument", "same", (), ["drums"], visibility=HIDDEN),
            agent.detail_row("them", "level", "steps", ("beginner", "advanced"), ["advanced"]),
            agent.detail_row("both", "genre", "same", (), ["ska"]),
            agent.detail_row("me", "free time", "same", (), ["evenings"])]}
        same = agent.draft_from_message("Looking for a climbing partner on weekends.", "Ken", "35-39", words=WORDS)
        same["fields"].append(agent.detail_row("me", "free time", "same", (), ["weekends"]))  # one it wrote now
        self.assertEqual(agent.keep_details(same, old), ["me:instrument", "them:level", "both:genre"])
        by = {f["key"]: f for f in same["fields"]}
        self.assertEqual((by["me:instrument"]["visibility"], by["me:free time"]["value"]), (HIDDEN, ["weekends"]))
        other = agent.draft_from_message("Looking for a tennis partner.", "Ken", "35-39", words=WORDS)
        self.assertEqual(agent.keep_details(other, old), ["me:instrument", "me:free time"])  # a new request
        unsure = agent.draft_from_message("hello", "Ken", "35-39", words=WORDS)
        self.assertEqual(len(agent.keep_details(unsure, old)), 4)
        self.assertEqual(agent.keep_details(unsure, None), [])


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
                                  ("I'd like to meet a man in Kyoto. I go running most mornings.", "a running partner"),
                                  ("Hoping to meet a woman in Lisbon, and looking for a hiking buddy too.",
                                   "a hiking partner")):
            question = agent.kind_question(message, WORDS)
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
            self.assertIsNone(agent.kind_question(message, WORDS), message)

    def test_the_answer_decides_and_both_makes_two_listings(self):
        self.s.agent_message("A", VISITING)
        self.assertIn("a date, or a climbing partner", self.chat()[-1])
        self.assertIsNone(self.s.phone_state("A")["listings"][0]["draft"])  # nothing saved while it's unclear
        self.s.agent_message("A", "Both, please")
        self.assertEqual(self.kinds(), ["dating", "other"])
        for l in self.s.phone_state("A")["listings"]:
            self.assertEqual({f["key"]: f["value"] for f in l["draft"]["fields"]}["kind"], l["draft"]["kind"])
        dating = {f["key"]: f["value"] for f in self.s.phone_state("A")["listings"][0]["draft"]["fields"]}
        self.assertEqual((dating["wants_gender"], dating["wants_age"]), (["woman"], "30-49"))

    def test_one_choice_makes_one_listing_of_that_kind(self):
        self.s.agent_message("A", "I'd like to meet a man in Kyoto. I go climbing most mornings.")
        self.s.agent_message("A", "just climbing, not a date")
        self.assertEqual(self.kinds(), ["other"])

    def test_a_partner_for_an_activity_named_in_the_answer(self):
        self.s.agent_message("A", "Looking for a partner in Berlin.")
        self.s.agent_message("A", "bouldering")
        draft = self.s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["seeking"], values["offering"]),
                         ("other", "a bouldering partner", "a bouldering partner"))

    def test_both_with_an_activity_not_named_yet_asks_which(self):
        self.s.agent_message("A", "Looking for a partner in Berlin.")
        self.s.agent_message("A", "both")
        self.assertEqual(self.kinds(), ["dating", None])
        self.assertIn("What would you like a partner for?", self.chat(1)[-1])
        second = self.s.phone_state("A")["listings"][1]["id"]
        self.s.agent_message("A", "climbing", second)
        self.assertEqual(self.kinds(), ["dating", "other"])

    def test_not_an_answer_asks_again_and_a_new_request_starts_over(self):
        self.s.agent_message("A", VISITING)
        self.s.agent_message("A", "hmm")
        self.assertIn("didn't catch that", self.chat()[-1])
        self.s.agent_message("A", "Actually I'm looking for a bouldering partner in Osaka.")
        self.assertEqual(self.kinds(), ["other"])

    def test_the_model_asks_before_saving_and_both_saves_two_listings(self):
        """A scripted stand-in for the model: it asks, then saves one listing per kind in one turn."""
        def save(kind, separate, extra, details=()):
            return {"kind": kind, "exactness": 0.6, "suggestions": [], "separate_listing": separate,
                    "intent_text": {"dating": "Looking to meet a man in Kyoto.",
                                    "other": "Looking for a running buddy in Kyoto, mornings."}[kind],
                    "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN}] + extra,
                    "details": list(details)}
        client = ScriptedClient(
            reply(text("Are you looking for a date, or a running buddy? Or both? I can make two separate listings.")),
            reply(text("NO_SAVE")),  # the app's check (SAVE_CHECK): the kind is the open question, so nothing to save
            reply(tool(save("dating", False, [{"key": "gender", "values": ["woman"], "visibility": SHOWN},
                                              {"key": "wants_gender", "values": ["man"], "visibility": SHOWN},
                                              {"key": "wants_age", "values": ["30-49"], "visibility": SHOWN}]), "t1"),
                  tool(save("other", True, [{"key": "seeking", "values": ["a running partner"], "visibility": SHOWN},
                                            {"key": "offering", "values": ["a running partner"], "visibility": SHOWN},
                                            {"key": "headline", "values": ["Looking for a running buddy"],
                                             "visibility": SHOWN}],
                            [det("speed", "gentle", compare="steps", steps=("gentle", "steady", "brisk"))]), "t2")),
            reply(text("Done: two listings, one for dating and one for running.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.verify("A", "Aiko", 37)
        s.agent_message("A", "I'd like to meet a man in Kyoto. I go running most mornings.")
        self.assertIsNone(s.phone_state("A")["listings"][0]["draft"])  # asked, saved nothing
        self.assertIn("ONE short question", client.requests[0]["system"])
        s.agent_message("A", "Both!")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([l["draft"]["kind"] for l in listings], ["dating", "other"])
        self.assertEqual([{f["key"]: f["value"] for f in l["draft"]["fields"]}["kind"] for l in listings],
                         ["dating", "other"])
        self.assertEqual({f["key"]: f["value"] for f in listings[1]["draft"]["fields"]}["me:speed"], ["gentle"])
        self.assertEqual(s.phone_state("A")["current"], listings[0]["id"])

    def test_turn_drafts_keeps_one_draft_per_listing(self):
        """One draft per search: the current listing's latest, then each separate search's latest (its kind and
        what it looks for), so two separate searches of the same kind both become listings."""
        seeking = lambda phrase: [{"key": "seeking", "value": phrase}]
        a = {"kind": "dating"}
        b, b2 = ({"kind": "other", "separate": True, "fields": seeking("a kayaking partner"), "n": n} for n in (1, 2))
        c = {"kind": "other", "separate": True, "fields": seeking("a pub quiz team")}
        self.assertIsNone(turn_drafts([]))
        self.assertEqual(turn_drafts([a]), a)
        both = turn_drafts([a, b, b2])
        self.assertEqual((both["n"], both["earlier"]), (2, [a]))
        three = turn_drafts([a, b, c, b2])  # each search keeps its place, with its latest draft
        self.assertEqual(three, {**c, "earlier": [a, b2]})


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

    def save(self, fields, weekends=False):
        return {"kind": "dating", "exactness": 0.6, "suggestions": [], "separate_listing": False,
                "intent_text": "Looking to meet a woman in Kyoto, aged 30 to 39.",
                "fields": [{"key": "city", "values": ["Kyoto"], "visibility": SHOWN}] + fields,
                "details": [det("free time", "weekends")] if weekends else []}

    def visibilities(self, s):
        return {f["key"]: f["visibility"] for f in s.slots["A"].draft["fields"] if f["value"]}

    def test_a_model_save_that_leaves_a_hidden_detail_out_keeps_it_hidden(self):
        first = [{"key": "gender", "values": ["man"], "visibility": SHOWN},
                 {"key": "wants_gender", "values": ["woman"], "visibility": SHOWN},
                 {"key": "wants_age", "values": ["30-39"], "visibility": SHOWN}]
        later = []  # leaves all of those out, and writes a detail of its own (self.save)
        client = ScriptedClient(reply(tool(self.save(first), "t1")), reply(text("Saved.")),
                                reply(tool(self.save(later, weekends=True), "t2")), reply(text("Added weekends.")),
                                reply(tool(self.save(later + [{"key": "gender", "values": ["man"], "visibility": SHOWN}],
                                                     weekends=True), "t3")), reply(text("Shown.")))
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
        self.assertEqual(vis["me:free time"], SHOWN)
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
        draft, _, _ = build_draft({"kind": "other", "exactness": 0.6, "intent_text": "A climbing partner in Osaka.",
                                   "fields": [{"key": "city", "values": ["Osaka"], "visibility": SHOWN},
                                              {"key": "pay", "values": ["¥1,000 a session"]}]}, "Ken", "35-39")
        self.assertEqual(draft["listed"], ["city"])  # pay had no visibility of the agent's own


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
                 "fields": [], "details": [det("free time", "weekends")]}
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
            self.assertIsNone(agent.kind_question(message, WORDS), message)
            self.assertEqual(agent.draft_from_message(message, "Ken", "35-39", words=WORDS)["kind"], "dating", message)
        # Asked for as something to do together, it still asks.
        for message in ("Looking for a girlfriend in Osaka, and a climbing partner too.",
                        "I'd like to meet a man in Kyoto. I play tennis most weekends."):
            self.assertIsNotNone(agent.kind_question(message, WORDS), message)

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
            draft = agent.draft_from_message(message, "Ken", "35-39", words=WORDS)
            self.assertEqual(draft["kind"], "dating", message)
            self.assertEqual({f["key"]: f["value"] for f in draft["fields"]}["kind"], "dating")
        # Matching reads only dating from words; every other request is read from its phrases.
        self.assertIsNone(parse_intent("I enjoy running and I want a running buddy.").kind)
        self.assertIsNone(parse_intent("I love climbing. Looking for someone fun in Osaka.").kind)
        self.assertIsNone(parse_intent("I love climbing. Based in Osaka.").kind)

    def test_a_hobby_ends_where_they_hope_to_meet_someone(self):
        from belong.matching.intent import parse_intent
        for message in ("I enjoy reading and hope to meet a girlfriend in Osaka.",
                        "I love hiking and I'm keen to find someone special in Osaka.",
                        "I like climbing and I'm trying to meet a boyfriend in Kyoto.",
                        "I enjoy cooking, open to meeting a boyfriend in Tokyo.",
                        "I like climbing, I'd really like a boyfriend in Osaka.",
                        "I love tennis and a girlfriend would be lovely, in Osaka."):
            self.assertEqual(parse_intent(message).kind, "dating", message)
            draft = agent.draft_from_message(message, "Ken", "35-39", words=WORDS)
            self.assertEqual(draft["kind"], "dating", message)
            self.assertIsNone(agent.kind_question(message, WORDS), message)
        # Someone they already have is never who they're looking for.
        self.assertIsNone(parse_intent("I love hiking with my girlfriend. Based in Osaka.").kind)

    def test_a_kind_only_from_a_hobby_is_asked_never_recorded(self):
        for message, label in (("I love climbing. Based in Osaka.", "a climbing partner"),
                               ("I enjoy tennis and chess, I'm in Tokyo.", "a tennis partner"),
                               ("I love climbing. Looking for someone fun in Osaka.", "a climbing partner")):
            question = agent.kind_question(message, WORDS)
            self.assertTrue(question["text"].startswith(f"Are you looking for {label}? Or something else?"), message)
            draft = agent.draft_from_message(message, "Ken", "35-39", words=WORDS)
            self.assertIsNone(draft["kind"], message)
            self.assertNotIn("kind", {f["key"] for f in draft["fields"]})
        self.assertIn("Are you looking for a climbing partner, or something else?",
                      agent.draft_from_message("I love climbing. Based in Osaka.", "Ken", "35-39", words=WORDS)["notes"][0])

    def test_in_the_demo_the_answer_decides_and_keeps_the_first_message(self):
        for answer, kind in (("yes", "other"), ("a relationship", "dating"), ("a bouldering partner", "other")):
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
    OPTIONS = agent.kind_question(VISITING, WORDS)["options"]  # a date, or a climbing partner
    PARTNER = agent.kind_question("Looking for a partner in Berlin.", WORDS)["options"]  # a date, or a partner for what?

    def kinds(self, answer, options=None):
        return [c["kind"] for c in agent.chosen_kinds(answer, options or self.OPTIONS, WORDS) or []]

    def test_an_activity_only_mentioned_is_not_chosen(self):
        for answer in ("a date, I just like climbing", "a date, I climb sometimes though", "a date. I really love climbing",
                       "a date, I'm into climbing", "a date, and I go climbing on weekends"):
            self.assertEqual(self.kinds(answer), ["dating"], answer)
            self.assertEqual(self.kinds(answer, self.PARTNER), ["dating"], answer)

    def test_an_activity_named_as_the_choice_is(self):
        for answer in ("climbing", "rock climbing", "just climbing please", "climbing on weekends", "the climbing one",
                       "I'd like a climbing partner", "not a date, climbing"):
            self.assertEqual(self.kinds(answer), ["other"], answer)
            self.assertEqual(agent.chosen_kinds(answer, self.OPTIONS, WORDS), [{"kind": "other", "activity": "climbing"}])
        for answer in ("both", "a date and a climbing partner", "both, please", "date and climbing"):
            self.assertEqual(self.kinds(answer), ["dating", "other"], answer)
        # A partner for what: only a listed activity is an answer.
        self.assertEqual(agent.chosen_kinds("golf", self.PARTNER, WORDS), [{"kind": "other", "activity": "golf"}])
        self.assertIsNone(agent.chosen_kinds("pottery", self.PARTNER, WORDS))

    def test_otherwise_it_asks_once_more(self):
        for answer in ("I climb on weekends", "hmm", "yes", "no"):
            self.assertIsNone(agent.chosen_kinds(answer, self.OPTIONS, WORDS), answer)
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
                self.assertEqual(agent.chosen_kinds(answer, options, WORDS), [{"kind": "dating", "activity": None}],
                                 answer)
        self.assertIsNone(agent.chosen_kinds("not a relationship", TheAnswerChoosesWhatItNames.OPTIONS, WORDS))

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
        s.agent_message("A", "Actually I'm looking for a bouldering partner in Tokyo.")
        draft = s.phone_state("A")["listings"][0]["draft"]
        self.assertEqual((draft["kind"], {f["key"]: f["value"] for f in draft["fields"]}["city"]), ("other", "Tokyo"))


class DetailsInTheAnswer(unittest.TestCase):
    def test_what_the_answer_says_besides_the_choice_is_kept(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'd like to meet a woman in Kyoto. I go climbing most mornings.")
        s.agent_message("A", "a date, I'm a man, ideally 30 to 45")
        draft = s.phone_state("A")["listings"][0]["draft"]
        values = {f["key"]: f["value"] for f in draft["fields"]}
        self.assertEqual((draft["kind"], values["gender"], values["wants_age"]), ("dating", "man", "30-45"))
        self.assertIn("I'm a man, ideally 30 to 45.", draft["intent_text"])

    def test_the_choice_itself_adds_nothing(self):
        options = TheAnswerChoosesWhatItNames.OPTIONS
        for answer in ("a date", "Both, please", "just climbing, not a date", "a life partner", "yes"):
            self.assertEqual(agent.answer_details(answer, options, agent.chosen_kinds(answer, options, WORDS) or []), "",
                             answer)
        self.assertEqual(agent.answer_details("climbing, on weekends", options, [{"kind": "other", "activity": "climbing"}]),
                         "on weekends.")


class AnyActivityNextToADate(unittest.TestCase):
    def test_the_activity_reader_is_general(self):
        for message, label in (("I'd like to meet a man in Kyoto. I play tennis most weekends.", "a tennis partner"),
                               ("Hoping to meet a woman in Osaka. I go hiking a lot.", "a hiking partner"),
                               ("Want to meet a woman in Osaka, I enjoy chess and jazz.", "a chess partner"),
                               ("Looking to meet a man in Lisbon. We could play golf.", "a golf partner"),
                               ("I'd like to meet a woman in Berlin, I do yoga on Sundays.", "a yoga partner")):
            question = agent.kind_question(message, WORDS)
            self.assertIsNotNone(question, message)
            self.assertTrue(question["text"].startswith(f"Are you looking for a date, or {label}? Or both?"), message)
        # Romantic words leave no doubt: then what they do or enjoy is a hobby. And nobody else's hobby counts.
        for message in ("Looking for a relationship. I play tennis on weekends.",
                        "Looking for a relationship with a woman in Osaka. I enjoy climbing.",
                        "Looking to meet a man in Osaka, who loves hiking and chess.",
                        "I want to meet a woman in Osaka. I love dogs.", "I'd like to meet a man. I go out a lot."):
            self.assertIsNone(agent.kind_question(message, WORDS), message)
        # Only what's listed is an activity: with nothing listed, a hobby is just a hobby.
        self.assertIsNone(agent.kind_question("I'd like to meet a man in Kyoto. I play tennis most weekends."))

    def test_both_makes_a_listing_for_that_activity(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "I'd like to meet a man in Kyoto. I go climbing most weekends.")
        s.agent_message("A", "both")
        listings = s.phone_state("A")["listings"]
        self.assertEqual([l["draft"]["kind"] for l in listings], ["dating", "other"])
        other = {f["key"]: f["value"] for f in listings[1]["draft"]["fields"]}
        self.assertEqual((other["seeking"], other["headline"]), ("a climbing partner", "Looking for a climbing partner"))


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
        dating, other = agent.kind_layout("dating"), agent.kind_layout("other")
        self.assertTrue({"gender", "wants_gender", "wants_age"} <= {r["key"] for r in dating["rows"]})
        self.assertTrue({"wants_gender", "wants_age", "gender", "kinks"} <= set(other["drop"]))
        self.assertTrue({"seeking", "offering"} <= set(dating["drop"]))
        self.assertFalse([k for k in dating["drop"] + other["drop"] if ":" in k])  # open details go on either kind
        self.assertEqual(agent.kind_layout(None)["drop"], [])  # no kind chosen yet: nothing taken away
        self.assertEqual(agent.kind_layout("climbing"), agent.kind_layout(None))  # not a kind of match

    def test_who_and_what_ages_never_go_live_on_a_listing_that_isnt_dating(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Ken", 37)
        s.agent_message("A", "Looking for a relationship with a woman in Osaka, 30 to 39. I'm a man.")
        body = fill(s.slots["A"].draft, values={"kind": "other"})
        self.assertTrue({"wants_gender", "wants_age"} <= {f["key"] for f in body["fields"]})
        s.submit("A", body)
        live = s.board.listing("demoA-1")
        self.assertEqual(live.get("kind").value, "other")
        self.assertIsNone(live.get("wants_gender"))
        self.assertIsNone(live.get("wants_age"))
        # No kind on the listing, and a description that isn't dating: the same.
        s.agent_message("A", "Looking for a climbing partner in Osaka.")
        body = {**fill(s.slots["A"].draft), "intent_text": "Looking for a climbing partner in Osaka."}
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
        dating["fields"].append(agent.detail_row("me", "interests", "same", (), ["sailing"]))
        cases = {"dating": dating, "climbing": agent.kind_layout("other"), "to_dating": agent.kind_layout("dating"),
                 "nobody": agent.draft_from_message("Looking for someone special in Osaka.", "Ken", "35-39"),
                 "wants": agent.dating_wants(dating)}
        code = "\n".join(_js(n) for n in ("const hasValue", "const cap", "const spanOf", "const agesText", "const UNDERSTOOD",
                                          "const detailParts", "const isDetail",
                                          "const GENDER_WORDS", "const orList", "const detailValues", "function wantsText", "function matchFacts",
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
d.kind = "other"; d.fields = relayout(d.fields, C.climbing, stash);
out.climbing_keys = keys(d.fields);
d.kind = "dating"; d.fields = relayout(d.fields, C.to_dating, stash);
out.back = d.fields.filter(f => ["gender", "wants_gender"].includes(f.key)).map(f => [f.key, f.value]);
const n = JSON.parse(JSON.stringify(C.nobody));
out.live_nobody = canSave({ draft: n }, { listed: true, wants: null });
const fresh = JSON.parse(JSON.stringify(C.nobody)); fresh.kind = "other";
fresh.fields = relayout(fresh.fields, C.climbing, {}); fresh.kind = "dating";
fresh.fields = relayout(fresh.fields, C.to_dating, {});
out.switched_keys = keys(fresh.fields);
out.match_dating = matchFacts({ first_name: "A", wants_gender: ["woman"], kind: "dating" }, true).wants;
out.match_climbing = matchFacts({ first_name: "A", wants_gender: ["woman"], kind: "other" }, false).wants;
console.log(JSON.stringify(out));"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        result = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        out = json.loads(result.stdout)
        self.assertTrue(out["live_dating"])
        self.assertFalse({"gender", "wants_gender", "wants_age", "kinks", "children"} & set(out["climbing_keys"]))
        self.assertIn("me:interests", out["climbing_keys"])  # an open detail stays on either kind
        self.assertEqual(out["back"], [["gender", "man"], ["wants_gender", ["woman"]]])  # put aside, and back
        self.assertFalse(out["live_nobody"])  # a live dating listing nobody can match can't be saved
        self.assertTrue({"gender", "wants_gender", "wants_age"} <= set(out["switched_keys"]))
        self.assertTrue({"seeking", "offering"}.isdisjoint(out["switched_keys"]))
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
        s.agent_message("A", "climbing")
        d = s.slots["A"].listing().draft
        shown = {f["key"]: f["value"] for f in d["fields"] if f["visibility"] == SHOWN and f["value"] not in (None, "", [])}
        self.assertEqual((d["kind"], shown.get("city"), shown.get("seeking")), ("other", "Lisbon", "a climbing partner"))
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
        self.assertFalse([k for k in values if ":" in k])  # nothing it didn't say ("weekends")

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
