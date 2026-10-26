"""The synthetic people (belong/synth.py): names from their region, and listings in the format a
person's agent writes now (headline, topics, phrases where they're matched by phrase)."""
from __future__ import annotations

import re
import unittest
from collections import Counter

from belong.synth import CITIES, COUNTRY_NAMES, ITEMS, LOCAL_SHARE, NAMES, SERVICES, generate

PEOPLE = generate(3000, 7)


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


if __name__ == "__main__":
    unittest.main()
