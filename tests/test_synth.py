"""The synthetic people (belong/synth.py): names from their region, and listings in the format a
person's agent writes now (headline, topics, phrases on every listing but dating's, open details, the
kind dating or other), dated some days back, the agent dialects of the batch run's second pass, the
invented wishes of its next passes, with and without the rule that a must comes from the person's own
words (agent.ground_wishes, TRUST.md #45), and the musts said in other words of its last pass, with the
person's yes to their agent's question and by values alone."""
from __future__ import annotations

import re
import unittest
from collections import Counter
from datetime import datetime, timedelta

from unittest import mock

from belong import agent
from belong.agent import said_words, wish_said
from belong.evaluate import can_meet, truth_intent_met
from belong.matching.marketboard import Marketboard
from belong.schema import (FIELD_REGISTRY, SHOWN, STRICT_EXACTNESS, canonical, comparable, detail_problems,
                           is_detail_key, own, satisfies, split_key)
from belong.sim import run, said_listing
from belong.synth import (ANSWER_NO, ANSWER_YES, CITIES, COUNTRY_NAMES, DIALECT_SHARE, DIALECTS, INVENTED,
                          INVENTED_SHARE, ITEMS, LISTED_DAYS, LOCAL_SHARE, NAMES, OTHER_WORDS_SHARE, SERVICES, Detail,
                          answer, draft_listing, generate)
from belong.warden import Clock, Warden

PEOPLE = generate(3000, 7)
NOW = "2026-10-12T09:00:00Z"
# The stories whose open details decide who suits whom (cofounders' and tutors' roles are in their phrases).
DETAIL_STORIES = ("climbing", "running", "language", "tutoring", "roommate")


def details_ok(a, b) -> bool:
    """Are A's shown wishes (them: and both: details) met by B's shown details? The rule as SPEC §5 states
    it, written out here so the synthetic details are checked against the truth apart from matching: B's
    own value must be comparable and satisfy the wish (one step either way for a wide net, on the same
    scale); a wish with nothing to compare fails unless it is only "ideally" and A isn't strict."""
    a_shown, b_shown = a.shown_fields(), b.shown_fields()
    for want in a_shown.values():
        if want.about not in ("them", "both"):
            continue
        mine = own(b_shown, want.name)
        if mine is None or not comparable(want, mine):
            if want.required or a.exactness >= STRICT_EXACTNESS:
                return False
            continue
        same_scale = tuple(map(canonical, want.steps)) == tuple(map(canonical, mine.steps))
        if not satisfies(want, mine, 1 if a.exactness < 0.5 and same_scale else 0):
            return False
    return True


def detail_pairs(people, saved=lambda p: draft_listing(p, NOW)):
    """(truth says A's intent is met by B, A's details are met by B's) for every pair the details decide. saved: each
    person's listing as their agent saved it (as written, by default)."""
    people = [p for p in people if p.truth.kind in DETAIL_STORIES]
    listings = {p.person.id: saved(p) for p in people}
    for a in people:
        for b in people:
            if a is b or a.truth.kind != b.truth.kind or not can_meet(a.truth, b.truth):
                continue
            if a.truth.kind == "tutoring" and a.truth.role == b.truth.role:
                continue  # two tutors, or two students: their phrases tell them apart
            yield truth_intent_met(a.truth, b.truth), details_ok(listings[a.person.id], listings[b.person.id])


class Names(unittest.TestCase):
    def test_most_people_have_a_name_from_their_region(self):
        for code, country in COUNTRY_NAMES.items():
            here = [p for p in PEOPLE if p.values["country"] == country]
            local = {n for pool in NAMES[code].values() for n in pool}
            share = sum(p.values["first_name"] in local for p in here) / len(here)
            self.assertGreater(share, LOCAL_SHARE - 0.05, country)
            self.assertLess(share, 1.0, country)  # and some names from elsewhere, as in any city


class Listings(unittest.TestCase):
    def test_every_listing_has_a_headline_like_an_agent_writes(self):
        cities = {c.lower() for c in CITIES}
        for p in PEOPLE:
            h = p.values["headline"]
            self.assertRegex(h, r"^[A-Z][a-z]+ing\b", h)
            self.assertLessEqual(len(h), 60, h)
            self.assertFalse(set(re.findall(r"[a-z]+", h.lower())) & cities, h)  # no place in it
            self.assertNotRegex(h, r"[¥€£$]|\d", h)  # no price in it
            self.assertTrue(p.values["topics"], h)

    def test_services_and_things_are_matched_by_phrase(self):
        kinds = Counter(p.truth.kind for p in PEOPLE)
        self.assertTrue(kinds["service"] and kinds["item"])
        for p in PEOPLE:
            if p.truth.kind in ("service", "item"):
                self.assertTrue(p.values["seeking"] and p.values["offering"], p.intent_text)
        self.assertTrue(any(SERVICES[p.truth.sub][3] and p.truth.remote_ok for p in PEOPLE if p.truth.kind == "service"))
        givers = [p for p in PEOPLE if p.truth.kind == "item" and p.truth.role == "giver"]
        self.assertTrue(givers and all(p.values["pay"] == "Free" for p in givers))
        self.assertEqual({p.truth.sub for p in PEOPLE if p.truth.kind == "item"}, set(ITEMS))

    def test_how_far_people_travel_and_who_works_online_only(self):
        travel = Counter(p.values.get("travel") for p in PEOPLE)
        self.assertGreater(travel["Same city"], 0.65 * len(PEOPLE))  # most stay in their city
        self.assertGreater(travel["Up to 30 km"], travel["Up to 100 km"])  # some go 30 km, a few 100
        self.assertGreater(travel["Up to 100 km"], 0)
        online_only = [p for p in PEOPLE if p.values.get("format") == "Online"]
        self.assertTrue(online_only)
        for p in online_only:  # web designers and CV reviewers, who have no use for a travel distance
            self.assertEqual((p.truth.kind, p.truth.role, SERVICES[p.truth.sub][3]), ("service", "provider", True))
            self.assertNotIn("travel", p.values)
            self.assertFalse(p.truth.in_person)

    def test_the_new_details_leave_everyone_else_as_they_were(self):
        """Travel and online-only come from their own random stream: the 300 people the tests and the
        batch run use still have the same names, cities and descriptions."""
        first = generate(300, 7)
        self.assertEqual([(p.values["first_name"], p.intent_text) for p in first[:3]], [
            ("Hana", "Running buddy in Kyoto for weekends runs."),
            ("Chihiro", "Looking for a technical cofounder for a education startup in Kyoto. I handle sales and fundraising."),
            ("Ayumi", "Language exchange in Osaka: I speak Japanese natively and want to practice German.")])


class OpenDetails(unittest.TestCase):
    def test_every_listing_is_dating_or_other_and_the_rest_have_both_phrases(self):
        """Stories are ground truth only: on the listing every one but dating is "other", and every
        admitted non-dating listing shows what it's looking for and what it offers."""
        self.assertTrue(Counter(p.truth.kind for p in PEOPLE)["climbing"])  # the stories are still there
        w, board = Warden(Clock(), key=b"test-key"), Marketboard()
        for p in PEOPLE[:1200]:
            w.admit(p.person, draft_listing(p, NOW), board)
        self.assertGreater(len(board.listings()), 1100)
        for l in board.listings():
            kind = l.get("kind")
            self.assertIn(kind.value, ("dating", "other"), l.person_id)
            self.assertEqual(kind.visibility, SHOWN, l.person_id)
            if kind.value != "dating":
                shown = l.shown()
                self.assertTrue(shown.get("seeking") and shown.get("offering"), l.person_id)

    def test_every_open_detail_passes_the_entry_rule(self):
        """With and without dialects: one rule for open details (schema.detail_problems), wishes always
        shown, never a sensitive open detail (a halal or kosher diet stays out), and every other key a
        built-in."""
        for people in (PEOPLE, generate(1500, 3, dialects=1.0)):
            for p in people:
                listing = draft_listing(p, NOW)
                self.assertEqual(detail_problems(listing.fields, listing.get("kind").value), [], p.person.id)
                for f in listing.fields:
                    self.assertTrue(f.key in FIELD_REGISTRY or is_detail_key(f.key), f.key)
                    if f.about:
                        self.assertFalse(f.sensitive, f.key)
                        self.assertNotIn(canonical(f.value[0]), ("halal", "kosher"), p.person.id)
                        if f.about == "them":
                            self.assertEqual(f.visibility, SHOWN, (p.person.id, f.key))
        names = Counter(k for p in PEOPLE for k in p.values if is_detail_key(k))
        for key in ("me:interests", "me:free time", "me:level", "them:level", "me:pace", "me:language",
                    "them:language", "both:subject", "both:move in month", "them:free time"):
            self.assertTrue(names[key], key)

    def test_listings_last_changed_8_to_90_days_before_the_start(self):
        """So the shared list of details in use, which counts listings unchanged for a week, has entries from
        the first day."""
        start = datetime.fromisoformat(NOW.rstrip("Z"))
        ages = Counter((start - datetime.fromisoformat(draft_listing(p, NOW).updated_at.rstrip("Z"))).days
                       for p in PEOPLE)
        self.assertEqual((min(ages), max(ages)), LISTED_DAYS)
        self.assertGreater(len(ages), 60)  # spread out, not all on one day
        self.assertEqual(draft_listing(PEOPLE[0], NOW).updated_at,
                         (start - timedelta(days=PEOPLE[0].listed_days)).isoformat() + "Z")

    def test_the_details_say_exactly_what_the_truth_needs(self):
        """For every pair whose open details decide it, A's details are met by B's exactly when the truth
        says A's intent is met by B: no wrong match and no missed one comes from how the details are
        written."""
        pairs = Counter(detail_pairs(PEOPLE[:1500]))
        self.assertEqual(set(pairs), {(True, True), (False, False)}, pairs)
        self.assertGreater(pairs[(True, True)], 1000)


class Dialects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain, cls.dialect = generate(3000, 7), generate(3000, 7, dialects=DIALECT_SHARE)

    def test_a_share_of_people_write_one_detail_differently_and_nothing_else_changes(self):
        spoken = [p for p in self.dialect if p.dialect]
        self.assertAlmostEqual(len(spoken) / len(self.dialect), DIALECT_SHARE, delta=0.03)
        self.assertEqual({p.dialect.partition(":")[0] for p in spoken}, set(DIALECTS))

        def written(p, details: bool) -> dict:
            return {k: (v, p.visibility[k]) for k, v in p.values.items() if is_detail_key(k) == details}

        for before, after in zip(self.plain, self.dialect):
            self.assertEqual((after.person, after.truth, after.intent_text, after.listed_days),
                             (before.person, before.truth, before.intent_text, before.listed_days))
            self.assertEqual(written(after, False), written(before, False))  # the built-ins
            if after.dialect:
                self.assertNotEqual(written(after, True), written(before, True), after.dialect)
                self.assertTrue(all(isinstance(d, Detail) for k, d in after.values.items() if is_detail_key(k)))
            else:
                self.assertEqual(written(after, True), written(before, True))

    def test_a_detail_written_differently_never_makes_a_wrong_match(self):
        """It may cost a match (that's the price of not reusing the shared list), never make a wrong one."""
        pairs = Counter(detail_pairs(self.dialect[:1500]))
        self.assertEqual(pairs[(False, True)], 0, pairs)
        self.assertGreater(pairs[(True, False)], 0)  # and it does cost some

    def test_the_batch_run_stays_precise_and_leaks_nothing_with_dialects(self):
        """The batch run's two passes (python3 -m belong run): exact matches all correct by the truth and
        a clean leak report, the shared lists included, with and without dialects."""
        for share in (0.0, DIALECT_SHARE):
            with self.subTest(dialects=share):
                result = run(n=300, seed=7, dialects=share)
                self.assertEqual(result["metrics"]["precision"], 1.0)
                self.assertTrue(result["leak_report"]["ok"], result["leak_report"]["findings"][:5])
                self.assertEqual(result["leak_report"]["checked"]["details_lists"], 2)
                self.assertTrue(result["details_in_use"]["details"] and result["details_in_use"]["phrases"])


class InventedWishes(unittest.TestCase):
    """The invented-wish dialect: some agents add a must the person never had, quoting the first words of their
    description for it (not an answer to anything). It costs matches, never makes a wrong one, and the rule that a
    must comes from the person's own words (agent.ground_wishes, and agent.ground_exactness on a listing that asks for
    exactly what it says, which sim.run applies to every listing as a model agent's saves are in the demo) wins most
    of them back, whatever the agent quotes before the person's answer; none if it quotes their no."""

    @classmethod
    def setUpClass(cls):
        cls.plain, cls.invented = generate(3000, 7), generate(3000, 7, invented=INVENTED_SHARE)

    def test_a_share_of_listings_that_arent_dating_get_one_made_up_must_and_nothing_else_changes(self):
        made_up = [p for p in self.invented if p.invented]
        others = [p for p in self.invented if p.truth.kind != "dating"]
        self.assertAlmostEqual(len(made_up) / len(others), INVENTED_SHARE, delta=0.04)  # a few already had that wish
        self.assertFalse([p for p in made_up if p.truth.kind == "dating"])
        self.assertEqual({p.invented.partition(":")[2] for p in made_up}, {name for name, *_ in INVENTED.values()})
        for before, after in zip(self.plain, self.invented):
            self.assertEqual((after.person, after.truth, after.intent_text, after.listed_days, after.dialect),
                             (before.person, before.truth, before.intent_text, before.listed_days, before.dialect))
            added = {k: d for k, d in after.values.items() if k not in before.values}
            self.assertEqual({k: d for k, d in after.values.items() if k in before.values}, before.values)
            if after.invented:
                self.assertEqual(list(added), [after.invented])
                self.assertTrue(after.invented.startswith("them:"))
                self.assertTrue(added[after.invented].required)
                self.assertEqual(after.visibility[after.invented], SHOWN)
                # Its agent quotes their own first words for it; asked, they say it doesn't matter.
                self.assertEqual(list(after.their_words), [after.invented])
                self.assertTrue(after.intent_text.startswith(after.their_words[after.invented]))
                self.assertEqual(answer(after, after.invented), ANSWER_NO)
                listing = draft_listing(after, NOW)
                self.assertEqual(detail_problems(listing.fields, listing.get("kind").value), [], after.person.id)
            else:
                self.assertEqual(added, {})

    def test_it_costs_matches_never_makes_a_wrong_one_and_the_rule_wins_them_back(self):
        """The batch run's invented-wish passes (python3 -m belong run): precision 1.0 and a clean leak report with
        the rule and without it, and more matches with it, with and without dialects. On the batch run's first two
        passes the rule costs no matches either."""
        for share in (0.0, DIALECT_SHARE):
            with self.subTest(dialects=share):
                ruled = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE)
                unruled = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE, ground=False)
                for result in (ruled, unruled):
                    self.assertEqual(result["metrics"]["precision"], 1.0)
                    self.assertTrue(result["leak_report"]["ok"], result["leak_report"]["findings"][:5])
                self.assertGreater(ruled["metrics"]["coverage"], unruled["metrics"]["coverage"] + 0.03)
                self.assertGreater(ruled["metrics"]["wishes_kept_ideally"], 0)
                self.assertEqual(unruled["metrics"]["wishes_kept_ideally"], 0)
                # Remove the rule (every wish counts as said) and the matches it won back are gone again.
                with mock.patch.object(agent, "wish_said", lambda row, words: True):
                    removed = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE)
                self.assertEqual(removed["metrics"]["coverage"], unruled["metrics"]["coverage"])
                # Count any quote as theirs, wherever it came from, and the made-up musts their agents quote
                # something for stay musts: so do the matches they cost.
                with mock.patch.object(agent, "answered", lambda their_words, texts: bool(their_words)):
                    quoted = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE)
                self.assertEqual(quoted["metrics"]["coverage"], unruled["metrics"]["coverage"])
                # An agent that quotes the person's no as their answer keeps every must it made up: a word rule
                # can't tell a yes from a no, so it wins nothing back from such an agent (TRUST.md limit 28).
                over = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE, quote_any=True)
                self.assertEqual(over["metrics"]["precision"], 1.0)
                self.assertEqual(over["metrics"]["coverage"], unruled["metrics"]["coverage"])
                self.assertEqual(over["metrics"]["wishes_kept_ideally"], 0)
                # Without making a listing less exact over a made-up wish, the wish, kept as ideally, still shuts
                # out everyone who doesn't show it on the listings that ask for exactly what they say.
                with mock.patch.object(agent, "ground_exactness", lambda exactness, *a, **kw: (exactness, [])):
                    loose = run(n=300, seed=7, dialects=share, invented=INVENTED_SHARE)
                self.assertEqual(loose["metrics"]["precision"], 1.0)
                self.assertLess(loose["metrics"]["coverage"], ruled["metrics"]["coverage"])
                self.assertGreater(loose["metrics"]["coverage"], unruled["metrics"]["coverage"])
                # Without invented wishes, the rule costs nothing.
                self.assertGreaterEqual(run(n=300, seed=7, dialects=share)["metrics"]["coverage"],
                                        run(n=300, seed=7, dialects=share, ground=False)["metrics"]["coverage"])


class MustsInOtherWords(unittest.TestCase):
    """Every person says each must they have, with its value, as a person telling their agent would
    (synth.say_musts); where their agent wrote it otherwise (a dialect's scale) its save keeps it as ideally, it asks,
    and they answer yes (synth.answer), which it quotes. So the rule that a must comes from the person's own words keeps
    every must they really have: a run with it matches exactly as one without it, and a detail written differently
    still never makes a wrong match. Said in other words than the value their agent writes (OTHER_WORDS_SHARE), only
    their answer shows it is theirs: read by its values alone, a must they said would be kept as ideally, and with
    dialects that makes wrong matches."""

    @classmethod
    def setUpClass(cls):
        cls.dialect = generate(3000, 7, dialects=DIALECT_SHARE)
        cls.other = generate(3000, 7, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE)

    def test_each_must_is_said_with_its_value_or_answered_and_nothing_else_changes(self):
        for before, after in zip(self.dialect, self.other):
            musts = {k for k, d in before.values.items() if isinstance(d, Detail) and d.required
                     and split_key(k)[0] in ("them", "both")}
            self.assertEqual((after.person, after.truth, after.intent_text, after.values, after.visibility, after.dialect),
                             (before.person, before.truth, before.intent_text, before.values, before.visibility,
                              before.dialect))
            self.assertEqual((before.their_words, after.their_words), ({}, {}))  # nobody's agent quotes a request
            self.assertEqual(after.other_words, bool(musts))
            self.assertEqual(after.said, ())  # no value of a must typed
            self.assertEqual(len(before.said), len(musts))  # a message for each
            typed = said_words([before.intent_text, *before.said])
            for f in draft_listing(before, NOW).fields:
                # Each must they typed with its values, unless their agent wrote it on another scale (a dialect).
                if f.key in musts and not before.dialect.startswith("other scale"):
                    self.assertTrue(wish_said(f, typed), (before.person.id, f.key))
            for p in (before, after):
                for key in musts:
                    self.assertEqual(answer(p, key), ANSWER_YES)
                # Asked about each must its save kept as ideally, they say yes: every must stays one.
                self.assertEqual(said_listing(draft_listing(p, NOW), p)[1], [], p.person.id)
            # Without their answers, every must they said in other words is kept as ideally.
            self.assertEqual(set(said_listing(draft_listing(after, NOW), after, False)[1]),
                             {f.key for f in draft_listing(after, NOW).fields if f.key in musts and not wish_said(
                                 f, said_words([after.intent_text]))})

    def test_saved_listings_never_make_a_wrong_match_and_values_alone_would(self):
        """With the rule, as every run saves listings, a detail written differently still costs matches and never
        makes a wrong one: every must a person has stays one. Read by values alone, the musts said in other words are
        kept as ideally, and where the other listing doesn't show that detail they let through pairs the truth
        rejects."""
        answered = Counter(detail_pairs(self.other[:1500], lambda p: said_listing(draft_listing(p, NOW), p)[0]))
        self.assertEqual(answered[(False, True)], 0, answered)
        self.assertEqual(answered, Counter(detail_pairs(self.dialect[:1500])))
        alone = Counter(detail_pairs(self.other[:1500], lambda p: said_listing(draft_listing(p, NOW), p, False)[0]))
        self.assertGreater(alone[(False, True)], 0, alone)

    def test_the_batch_run_with_musts_in_other_words(self):
        """The batch run's last pass (python3 -m belong run): with their answers, it matches exactly as the dialects'
        week and keeps every must; by values alone it keeps some as ideally (and at another seed makes wrong matches).
        The rule costs nothing where nobody's agent made a must up."""
        week = run(n=300, seed=7, dialects=DIALECT_SHARE)
        answered = run(n=300, seed=7, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE)
        alone = run(n=300, seed=7, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE, answers=False)
        for result in (answered, alone):
            self.assertTrue(result["leak_report"]["ok"], result["leak_report"]["findings"][:5])
        self.assertEqual({k: answered["metrics"][k] for k in ("proposals", "proposals_correct", "precision", "coverage")},
                         {k: week["metrics"][k] for k in ("proposals", "proposals_correct", "precision", "coverage")})
        self.assertEqual(answered["metrics"]["wishes_kept_ideally"], 0)
        self.assertGreater(alone["metrics"]["wishes_kept_ideally"], 0)
        self.assertGreater(alone["metrics"]["musts_in_other_words"], 0)
        self.assertEqual(week["metrics"]["coverage"], run(n=300, seed=7, dialects=DIALECT_SHARE,
                                                          ground=False)["metrics"]["coverage"])
        self.assertEqual(run(n=300, seed=3, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE)["metrics"]["precision"],
                         1.0)
        self.assertLess(run(n=300, seed=3, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE,
                            answers=False)["metrics"]["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
