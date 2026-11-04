"""The synthetic people (belong/synth.py): names from their region, and listings in the format a
person's agent writes now (headline, topics, phrases on every listing but dating's, open details, the
kind dating or other), dated some days back, and the agent dialects of the batch run's second pass."""
from __future__ import annotations

import re
import unittest
from collections import Counter
from datetime import datetime, timedelta

from belong.evaluate import can_meet, truth_intent_met
from belong.matching.marketboard import Marketboard
from belong.schema import (FIELD_REGISTRY, SHOWN, STRICT_EXACTNESS, canonical, comparable, detail_problems,
                           is_detail_key, own, satisfies)
from belong.sim import run
from belong.synth import (CITIES, COUNTRY_NAMES, DIALECT_SHARE, DIALECTS, ITEMS, LISTED_DAYS, LOCAL_SHARE, NAMES,
                          SERVICES, Detail, draft_listing, generate)
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


def detail_pairs(people):
    """(truth says A's intent is met by B, A's details are met by B's) for every pair the details decide."""
    people = [p for p in people if p.truth.kind in DETAIL_STORIES]
    listings = {p.person.id: draft_listing(p, NOW) for p in people}
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


if __name__ == "__main__":
    unittest.main()
