"""The offline agent builds on what was already said (agent.draft_from_message's `earlier`, DemoSession._rules_draft).

A request is often given in pieces: what they're looking for in one message, where they are in the next, a free
time later. A message that adds to the request, or answers what the agent asked, builds on the draft as it stands;
one that plainly starts a different request replaces it, keeping only what's about the person. Every hidden choice
and every rule for reading a message stays. The vocabulary here is made up for these tests (the offline agent writes
only words on the shared list it is given), so nothing depends on the demo's words.
"""
from __future__ import annotations

import unittest

from belong import agent
from belong.agent_llm import description_save_problems
from belong.schema import HIDDEN, SHOWN
from belong.warden import full_name_pairs

# A shared list of details in use, as the Warden would build it, for a board of its own.
IN_USE = {"phrases": [{"key": k, "value": v} for v in ("a rowing partner", "a pottery partner")
                      for k in ("seeking", "offering")] + [{"key": "topics", "value": "woodwork"}],
          "details": [{"about": "me", "name": "free evenings", "compare": "same"},
                      {"about": "me", "name": "hobbies", "compare": "same"},
                      {"about": "me", "name": "experience", "compare": "steps"}],
          "scales": [{"about": "me", "name": "experience", "steps": ["novice", "regular", "expert"]}],
          "values": [{"about": "me", "name": "hobbies", "value": v} for v in ("baking", "birdwatching", "crosswords",
                                                                              "darts")]
          + [{"about": "me", "name": "free evenings", "value": v} for v in ("tuesdays", "thursdays")]
          + [{"about": "me", "name": "experience", "value": v} for v in ("novice", "regular")],
          "places": [{"city": "Galway", "country": "Ireland"}, {"city": "Tromso", "country": "Norway"}]}
WORDS = agent.words_in_use(IN_USE)


def draft(message, earlier=None, **kw):
    """What the offline agent saves for this message (DemoSession._rules_draft): its draft on top of the earlier one,
    the open details it can't write kept, and every shown or hidden from before."""
    d = agent.draft_from_message(message, "Orla", "40-44", words=WORDS, earlier=earlier, **kw)
    agent.keep_details(d, earlier)
    agent.keep_visibility(d, earlier)
    return d


def values(d):
    return {f["key"]: f["value"] for f in d["fields"]
            if f.get("enabled", True) and f["value"] not in (None, "", []) and f["key"] not in ("first_name", "age_band")}


def pieces(*messages, start=None):
    d = start
    for m in messages:
        d = draft(m, d)
    return d


def asks(d) -> str:
    return " ".join(d["notes"])


class WhatAndWhereInPieces(unittest.TestCase):
    def test_a_place_said_later_joins_the_request(self):
        first = draft("I'd like a rowing partner.")
        self.assertFalse(agent.ready(first))
        self.assertIn("Which city?", asks(first))
        d = draft("I live in Galway.", first)
        self.assertEqual(values(d), {**values(first), "city": "Galway", "country": "Ireland"})
        self.assertTrue(agent.ready(d))
        self.assertEqual(d["notes"], [])  # nothing still needed: no question about the kind or the place
        self.assertEqual(d["intent_text"], "I'd like a rowing partner. I live in Galway.")
        self.assertEqual(d["understood"], "Looking for a rowing partner in Galway")

    def test_the_request_said_after_the_place(self):
        d = pieces("I live in Tromso.", "I teach woodwork.")
        self.assertEqual((d["kind"], values(d)["seeking"], values(d)["offering"], values(d)["city"]),
                         ("other", "a woodwork student", "a woodwork teacher", "Tromso"))
        self.assertTrue(agent.ready(d))

    def test_a_date_in_three_pieces(self):
        d = pieces("I'm hoping to find a relationship.", "I'm a woman and I'd like to meet a man, 45 to 55.",
                   "I'm based in Galway.")
        self.assertEqual(values(d), {"kind": "dating", "gender": "woman", "wants_gender": ["man"], "wants_age": "45-55",
                                     "city": "Galway", "country": "Ireland"})
        self.assertTrue(agent.ready(d))
        self.assertNotIn("Who would you like to meet", asks(d))
        self.assertNotIn("your gender", asks(d))
        # Who they'd like to meet, said on its own, is read as the request it answers.
        d = pieces("I'm hoping to find a relationship.", "Someone who is a woman, 40 to 50 please.")
        self.assertEqual((values(d)["wants_gender"], values(d)["wants_age"]), (["woman"], "40-50"))

    def test_a_listing_for_something_else_with_nothing_its_for_is_asked_about(self):
        """Its kind changed on their phone, with nothing yet for what it's for: a message that adds a place keeps the
        kind, and the agent asks what they'd like a partner for."""
        d = draft("I'm hoping to find a relationship.")
        d["kind"] = "other"
        next(f for f in d["fields"] if f["key"] == "kind")["value"] = "other"
        d = draft("I live in Galway.", d)
        self.assertEqual((d["kind"], values(d)["city"]), ("other", "Galway"))
        self.assertIn(agent.WHICH_ACTIVITY, asks(d))
        self.assertFalse(agent.ready(d))

    def test_online_said_earlier_still_counts(self):
        d = pieces("I'd like a pottery partner, online is fine.", "I'm free on tuesdays.")
        self.assertTrue(agent.ready(d))
        self.assertNotIn("Which city?", asks(d))
        self.assertIn("(online/remote is fine)", d["understood"])


class AddingToAReadyDraft(unittest.TestCase):
    READY = ("I'd like a rowing partner.", "I live in Galway.")

    def test_a_message_that_only_adds_something_keeps_it_ready(self):
        before = pieces(*self.READY)
        d = draft("I'm free on thursdays.", before)
        self.assertEqual(values(d), {**values(before), "me:free evenings": ["thursdays"]})
        self.assertTrue(agent.ready(d))
        d = draft("I'm into darts.", d)
        self.assertEqual(values(d)["me:free evenings"], ["thursdays"])
        self.assertEqual(values(d)["me:hobbies"], ["darts"])

    def test_values_added_to_a_detail_join_those_there_and_a_step_moves(self):
        d = pieces(*self.READY, "I'm into baking. I'm a novice.", "I'm also into crosswords and I'm a regular now.")
        self.assertEqual(values(d)["me:hobbies"], ["baking", "crosswords"])
        self.assertEqual(values(d)["me:experience"], ["regular"])  # one place on its scale
        d = pieces("I'm into baking, darts and birdwatching.", "I'm into crosswords.", start=d)
        self.assertEqual(values(d)["me:hobbies"], ["darts", "birdwatching", "crosswords"])  # the most it may have

    def test_saying_the_request_again_keeps_the_persons_own_wording(self):
        d = pieces(*self.READY)
        for f in d["fields"]:
            if f["key"] == "headline":
                f["value"] = "Rowing at dawn on the Corrib"  # changed on their phone
        d = draft("I'd like a rowing partner, I'm free on tuesdays.", d)
        self.assertEqual(values(d)["headline"], "Rowing at dawn on the Corrib")
        self.assertEqual(values(d)["me:free evenings"], ["tuesdays"])
        # The description says the request once, with what's new.
        self.assertEqual(draft("I'd like a rowing partner in Galway.", draft("I'd like a rowing partner."))
                         ["intent_text"], "I'd like a rowing partner in Galway.")

    def test_the_persons_own_changes_stay(self):
        d = pieces(*self.READY)
        for f in d["fields"]:
            if f["key"] == "city":
                f["value"] = "Tromso"
            if f["key"] == "country":
                f["value"] = "Norway"
        d = draft("I'm free on tuesdays.", d)
        self.assertEqual((values(d)["city"], values(d)["country"]), ("Tromso", "Norway"))

    def test_a_new_place_replaces_the_old_one(self):
        d = pieces(*self.READY)
        next(f for f in d["fields"] if f["key"] == "neighborhood").update(value="Salthill")  # hidden by default
        moved = draft("I live in Tromso now.", d)
        self.assertEqual((values(moved)["city"], values(moved)["country"]), ("Tromso", "Norway"))
        self.assertNotIn("neighborhood", values(moved))  # the old place's, not the new one's
        self.assertEqual(values(draft("I'm free on tuesdays.", d))["neighborhood"], "Salthill")
        country_only = draft("I live in Portugal now.", d)  # somewhere else, without a city the draft records
        self.assertFalse({"city", "country", "neighborhood"} & set(values(country_only)))
        self.assertIn("Which city?", asks(country_only))

    def test_the_description_keeps_the_newest_words_whole(self):
        long = pieces(*self.READY)
        long["intent_text"] = "I row most mornings and want company on the water. " * 9
        d = draft("I'm free on thursdays.", long)
        self.assertLessEqual(len(d["intent_text"]), 500)
        self.assertTrue(d["intent_text"].endswith(" I'm free on thursdays."))
        self.assertTrue(d["intent_text"].startswith("I row most mornings"))


class ADifferentRequest(unittest.TestCase):
    def test_it_replaces_the_request_keeping_only_whats_about_the_person(self):
        before = pieces("I'd like a rowing partner, online is fine.", "I live in Galway.")
        next(f for f in before["fields"] if f["key"] == "religion").update(value="Quaker", enabled=True)  # hidden
        for message, kind in (("I'd like a pottery partner.", "other"), ("I'm hoping to find a relationship.", "dating")):
            with self.subTest(message):
                self.assertTrue(agent.starts_another_request(message, before, WORDS))
                d = draft(message, before)
                self.assertEqual(d["kind"], kind)
                self.assertEqual((values(d)["city"], values(d)["country"]), ("Galway", "Ireland"))  # about them
                self.assertNotIn("format", values(d))  # the old request's own
                self.assertNotIn("religion", values(d))  # never anything sensitive
                self.assertNotIn("rowing", " ".join(map(str, values(d).values())))
                self.assertEqual(d["intent_text"], message)

    def test_what_adds_to_a_request_and_what_starts_another(self):
        before = pieces("I'd like a rowing partner.", "I live in Galway.")
        for message in ("I live in Tromso now.", "I'm free on thursdays.", "I love pottery.", "I'm into darts.",
                        "I'd like a rowing partner in Galway.", "Yes please.", "I'm a regular."):
            self.assertFalse(agent.starts_another_request(message, before, WORDS), message)
        for message in ("I'd like a pottery partner.", "I teach woodwork.", "I'd like a girlfriend.",
                        "I'd also like a juggling partner."):  # the last isn't listed: the agent asks which activity
            self.assertTrue(agent.starts_another_request(message, before, WORDS), message)


class TheRulesStillHold(unittest.TestCase):
    """A later message is read by the same rules as a first one: only what they say of themselves, never a denial,
    and a sensitive word goes into its hidden detail. Every hidden choice stays."""
    READY = ("I'd like a rowing partner.", "I live in Galway.")

    def test_only_what_they_say_of_themselves_and_never_a_denial(self):
        first = draft("I'd like a rowing partner.")
        for message in ("My brother lives in Tromso.", "I don't live in Tromso.", "I'm not into darts.",
                        "I'd like someone who is into crosswords.", "I'm not a novice."):
            with self.subTest(message):
                d = draft(message, first)
                self.assertEqual(values(d), values(first))
                self.assertIn("Which city?", asks(d))

    def test_a_sensitive_word_goes_into_its_hidden_detail_and_stays(self):
        d = draft("I have asthma, so nothing too strenuous.", pieces(*self.READY))
        self.assertEqual(values(d)["health"], "asthma")
        self.assertEqual(next(f for f in d["fields"] if f["key"] == "health")["visibility"], HIDDEN)
        self.assertIn("You mentioned something sensitive (health)", asks(d))
        later = draft("I'm free on tuesdays.", d)
        self.assertEqual(values(later)["health"], "asthma")
        self.assertEqual(next(f for f in later["fields"] if f["key"] == "health")["visibility"], HIDDEN)
        self.assertNotIn("sensitive", asks(later))  # said once, for the message that had it

    def test_every_hidden_choice_stays(self):
        d = pieces(*self.READY, "I'm into baking.")
        for f in d["fields"]:
            if f["key"] in ("city", "me:hobbies"):
                f["visibility"] = HIDDEN
            if f["key"] == "religion":  # a sensitive detail they chose to show, with their confirmation
                f.update(value="Quaker", enabled=True, visibility=SHOWN, confirmed=True)
        d = pieces("I'm free on thursdays.", "I'm also into darts.", start=d)
        vis = {f["key"]: f["visibility"] for f in d["fields"] if f["value"] not in (None, "", [])}
        self.assertEqual((vis["city"], vis["me:hobbies"], vis["me:free evenings"]), (HIDDEN, HIDDEN, SHOWN))
        self.assertEqual(values(d)["me:hobbies"], ["baking", "darts"])
        self.assertEqual((vis["religion"], next(f for f in d["fields"] if f["key"] == "religion").get("confirmed")),
                         (SHOWN, True))  # as they chose it


class TheDescriptionInPieces(unittest.TestCase):
    """Each message joins the description as a sentence of its own (agent._ended), so two messages put together never
    make a problem neither has alone: a capitalised word at the end of one and at the start of the next never reads
    as someone's full name."""

    def where(self, d):
        return [(f["key"], f["value"], f["visibility"]) for f in d["fields"] if f["key"] in ("city", "country")]

    def test_two_clean_messages_make_a_clean_description(self):
        first = ["I'd like a pottery partner who speaks Irish", "I teach woodwork to Norwegians",
                 "I'm hoping to find a relationship with a Catholic", "I'm into crosswords, mostly Guardian",
                 "I'd like a rowing partner near Mt", "I'd like a rowing partner near St.", "Rowing with me, J"]
        then = ["Based in Galway", "Tromso is home", "Thursdays suit me", "Mornings work best", "Galway, mostly"]
        for a in first:
            for b in then:
                with self.subTest(a=a, b=b):
                    self.assertEqual(full_name_pairs(a) + full_name_pairs(b), [])
                    d = draft(b, draft(a))
                    self.assertTrue(d["intent_text"].endswith(b))
                    self.assertEqual(full_name_pairs(d["intent_text"]), [])
                    self.assertEqual(description_save_problems(d["intent_text"], place_fields=self.where(d)), [])

    def test_a_message_already_ending_a_sentence_takes_no_second_stop(self):
        for a, joined in (("I'd like a rowing partner!", "I'd like a rowing partner! Based in Galway"),
                          ("Anyone for a rowing partner?", "Anyone for a rowing partner? Based in Galway"),
                          ("I'd like a rowing partner.", "I'd like a rowing partner. Based in Galway"),
                          ("I'd like a rowing partner,", "I'd like a rowing partner. Based in Galway")):
            self.assertEqual(draft("Based in Galway", draft(a))["intent_text"], joined)


class OnlyTheirOwnRequestStartsAnother(unittest.TestCase):
    """A request the person denies, or someone else's, never replaces theirs (agent.own_requests): it adds to the
    request already there, or changes nothing, and nothing they said before is lost. What they ask for themselves
    still starts a different request."""
    READY = ("I'd like a rowing partner.", "I live in Galway.")

    def ready(self):
        d = pieces(*self.READY, "I'm into darts.")
        next(f for f in d["fields"] if f["key"] == "religion").update(value="Quaker", enabled=True)  # hidden
        return d

    def test_a_denied_request_or_someone_elses_adds_nothing(self):
        before = self.ready()
        for message in ("I'm not after a pottery partner.", "I'd never want a relationship.",
                        "Not a relationship, just to be clear.", "Hardly looking for a relationship.",
                        "My cousin is looking for a pottery partner.", "Her husband would like a relationship.",
                        "They want a pottery partner too.", "I'm not after a pottery partner, I'm a regular."):
            with self.subTest(message):
                self.assertFalse(agent.starts_another_request(message, before, WORDS))
                d = draft(message, before)
                self.assertEqual(d["kind"], "other")
                for key in ("seeking", "offering", "headline", "topics", "city", "country", "religion", "me:hobbies"):
                    self.assertEqual(values(d)[key], values(before)[key], key)
                self.assertEqual(next(f for f in d["fields"] if f["key"] == "religion")["visibility"], HIDDEN)
                self.assertTrue(agent.ready(d))
                self.assertTrue(d["intent_text"].startswith(before["intent_text"]))

    def test_what_they_ask_for_themselves_still_starts_another(self):
        before = self.ready()
        for message, kind in (("I'm not sure about rowing any more, I'd like a pottery partner.", "other"),
                              ("I don't row any more and I'd like a pottery partner.", "other"),
                              ("My friends think I need a relationship.", "dating"),
                              ("I'd like a pottery partner who isn't a novice.", "other")):
            with self.subTest(message):
                self.assertTrue(agent.starts_another_request(message, before, WORDS))
                d = draft(message, before)
                self.assertEqual(d["kind"], kind)
                self.assertNotIn("religion", values(d))
                self.assertEqual(d["intent_text"], message)

    def test_who_and_what_ages_come_only_from_their_own_request(self):
        before = pieces("I'm a woman looking for a man, 45 to 55.", "I'm based in Tromso.")
        for message in ("My sister is looking for a woman, 30 to 40.", "I'm not looking for a woman.",
                        "Her husband wants someone 25 to 35."):
            with self.subTest(message):
                d = draft(message, before)
                self.assertEqual((d["kind"], values(d)["wants_gender"], values(d)["wants_age"]),
                                 ("dating", ["man"], "45-55"))
        d = draft("Actually a woman, 35 to 45.", before)  # their own
        self.assertEqual((values(d)["wants_gender"], values(d)["wants_age"]), (["woman"], "35-45"))


class TheAgesAndHowExact(unittest.TestCase):
    """What a later message doesn't change stays as the profile has it, and nothing said contradicts it: ages it can't
    place leave the ages already set, unremarked; how exact stays as set unless the message says what it's for."""

    def test_ages_it_cant_place_leave_the_ages_set_unremarked(self):
        before = pieces("I'm a woman looking for a man, 45 to 55.", "I'm based in Tromso.")
        for message in ("I'm 41 and my dad is 70.", "My nephews are 12 and 15."):
            with self.subTest(message):
                d = draft(message, before)
                self.assertEqual(values(d)["wants_age"], "45-55")
                self.assertNotIn("ages", asks(d))
        # With no ages set, it still says so.
        d = draft("I'm 41 and my dad is 70.", draft("I'm a woman looking for a man."))
        self.assertNotIn("wants_age", values(d))
        self.assertIn("I wasn't sure which ages you meant", asks(d))

    def test_how_exact_stays_as_set_unless_the_request_is_said(self):
        before = pieces("I'd like a rowing partner.", "I live in Galway.")
        before["exactness"] = 0.25  # as Save draft stores the person's own setting
        for message in ("I can only do tuesdays.", "Anyone is welcome to join on thursdays.", "I'm strictly a novice."):
            with self.subTest(message):
                self.assertEqual(draft(message, before)["exactness"], 0.25)
        self.assertEqual(draft("I'd like a rowing partner, strictly a regular.", before)["exactness"], 0.9)
        self.assertEqual(draft("I'd like a pottery partner, I'm flexible.", before)["exactness"], 0.3)


if __name__ == "__main__":
    unittest.main()
