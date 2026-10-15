"""Warden entry checks (SPEC §5, §7): who and what gets into the room."""
from __future__ import annotations

import unittest

from belong import agent
from belong.matching.marketboard import Marketboard
from belong.schema import SHOWN, Field
from belong.synth import generate
from belong.warden import specific_person_problems
from tests.helpers import fld, listing, person, warden


class Entry(unittest.TestCase):
    def test_unverified_person_is_rejected(self):
        r = warden().admit(person(verified=False), listing(), Marketboard())
        self.assertFalse(r.accepted)
        self.assertIn("kyc_verified", r.failed_checks)

    def test_sensitive_field_shown_without_confirmation_is_rejected(self):
        l = listing(fields=[fld("religion", "Buddhist", SHOWN)])
        self.assertIn("sensitive_defaults", warden().admit(person(), l, Marketboard()).failed_checks)

    def test_sensitive_field_shown_with_confirmation_is_accepted(self):
        l = listing(fields=[fld("religion", "Buddhist", SHOWN)], confirmed=("religion",))
        self.assertTrue(warden().admit(person(), l, Marketboard()).accepted)

    def test_mislabelled_sensitivity_is_rejected(self):
        """A listing can't relabel a sensitive category as non-sensitive to dodge the default."""
        l = listing(fields=[Field("health", "ADHD", SHOWN, sensitive=False)])
        self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks)

    def test_unknown_field_is_rejected(self):
        l = listing(fields=[Field("home_address", "1-2-3 Namba", SHOWN, False)])
        self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks)

    def test_age_band_must_match_kyc(self):
        l = listing(fields=[fld("age_band", "18-24")])
        self.assertIn("listing_schema", warden().admit(person(age_band="40-49"), l, Marketboard()).failed_checks)

    def test_free_text_fields_get_the_specific_person_check_too(self):
        l = listing(fields=[fld("seeking", "Mika Sato from the Blue Note"), fld("offering", "a jazz band")])
        self.assertIn("specific_person", warden().admit(person(), l, Marketboard()).failed_checks)

    def test_free_text_fields_are_length_limited(self):
        l = listing(fields=[fld("seeking", "a " + "very " * 30 + "long phrase"), fld("offering", "a jazz band")])
        self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks)

    def test_listing_update_rate_limit(self):
        w, board = warden(updates_per_day=5), Marketboard()
        results = [w.admit(person(), listing(), board).accepted for _ in range(6)]
        self.assertEqual(results, [True] * 5 + [False])


class NeverASpecificPerson(unittest.TestCase):
    BAD = [
        "Looking for Mika Sato who works at Nintendo, tall with short red hair.",
        "I want to meet @sakura_climbs from the climbing gym in Osaka.",
        "The barista at the Blue Bottle in Tokyo with a nose ring, please find her.",
        "Looking for my ex, Daniel Brooks, he lives in London.",
        "Find the woman named Hana who works at Siemens in Berlin.",
        "Text me if you're the guy on +44 7700 900123.",
        "Is this you? https://instagram.com/someone",
        "The cashier at the Lawson near my place, short with glasses.",
    ]

    def test_identifying_intents_are_rejected(self):
        for text in self.BAD:
            self.assertTrue(specific_person_problems(text), text)

    def test_every_ordinary_synthetic_intent_passes(self):
        """No false positives on the 2,000 normal intents across several seeds."""
        for seed in range(5):
            for p in generate(400, seed):
                if not p.truth.bad_intent:
                    self.assertEqual(specific_person_problems(p.intent_text), [], p.intent_text)

    def test_drafted_listings_pass_entry_for_verified_people(self):
        w, board = warden(), Marketboard()
        for p in generate(300, 7):
            r = w.admit(p.person, agent.draft_listing(p, "2026-10-12T09:00:00Z"), board)
            self.assertEqual(r.accepted, p.person.verified and not p.truth.bad_intent, (p.person.id, r.problems))


if __name__ == "__main__":
    unittest.main()
