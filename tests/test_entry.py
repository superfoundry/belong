"""Warden entry checks (SPEC §5, §7): who and what gets into the room."""
from __future__ import annotations

import unittest

from dataclasses import replace

from belong.matching.marketboard import Marketboard
from belong.schema import HIDDEN, K_CROWD, MAX_LISTINGS_PER_OWNER, SHOWN, Field
from belong.synth import draft_listing, generate
from belong.warden import ENTRY_CHECKS, detail_person_problems, specific_person_problems
from tests.helpers import detail, fld, listing, person, warden


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

    def test_sensitive_categories_take_any_words(self):
        """No list of choices: whatever the person says can be saved, hidden (or shown, confirmed)."""
        for key, value in (("religion", "Sikh"), ("health", "asthma"), ("health", "long covid"),
                           ("politics", "green party member"), ("income_band", "about 40k a year")):
            l = listing(fields=[fld(key, value, HIDDEN)])
            self.assertTrue(warden().admit(person(), l, Marketboard()).accepted, (key, value))
            shown = listing(fields=[fld(key, value, SHOWN)], confirmed=(key,))
            self.assertTrue(warden().admit(person(), shown, Marketboard()).accepted, (key, value))

    def test_hidden_built_ins_are_short_plain_text_without_contact_details(self):
        for value in ("x" * 61, "call me on +44 7700 900123", "see www.example.com", "<b>Nintendo</b>", ""):
            l = listing(fields=[fld("employer", value, HIDDEN)])
            self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks, value)

    def test_intimacy_preferences_only_on_dating_listings(self):
        kinks = fld("kinks", ("rope",), HIDDEN)
        self.assertIn("listing_schema", warden().admit(person(), listing(fields=[kinks]), Marketboard()).failed_checks)
        other = listing(fields=[kinks, fld("kind", "other")])
        self.assertIn("listing_schema", warden().admit(person(), other, Marketboard()).failed_checks)
        dating = listing(fields=[kinks, fld("kind", "dating")])
        self.assertTrue(warden().admit(person(), dating, Marketboard()).accepted)

    def test_unknown_field_is_rejected(self):
        for key in ("home_address", "level", "interests", "us:level", "me:", ":level"):
            l = listing(fields=[Field(key, "1-2-3 Namba", SHOWN, False)])
            self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks, key)

    def test_retired_kinds_of_match_are_rejected(self):
        """Only dating has rules of its own; everything else is "other", matched by phrases and details."""
        for kind in ("climbing", "running", "language", "tutoring", "cofounder", "roommate"):
            l = listing(fields=[fld("kind", kind)])
            self.assertIn("listing_schema", warden().admit(person(), l, Marketboard()).failed_checks, kind)

    def test_open_details_go_through_the_one_rule(self):
        good = listing(fields=[fld("kind", "other"), detail("skill", "middle", compare="steps", steps=("low", "middle", "high")),
                               detail("skill", ("middle", "high"), about="them", compare="steps",
                                      steps=("low", "middle", "high")),
                               detail("instrument", "Cello", about="both"), detail("night owl", "certainly", visibility=HIDDEN)])
        self.assertTrue(warden().admit(person(), good, Marketboard()).accepted)
        for bad in (detail("age", "twenty seven"), detail("health", "asthma"), detail("about me", "Sikh"),
                    detail("contact", "insta @janedoe99"), detail("skill", "x", compare="steps", steps=("a", "b")),
                    detail("skill", "middle", about="them", visibility=HIDDEN),
                    replace(detail("skill", "middle"), sensitive=True)):
            r = warden().admit(person(), listing(fields=[fld("kind", "other"), bad]), Marketboard())
            self.assertIn("listing_schema", r.failed_checks, bad)
            self.assertTrue(r.problems, bad)

    def test_built_in_fields_carry_no_compare_or_steps(self):
        l = listing(fields=[replace(fld("city", "Osaka"), compare="steps", steps=("near", "far"))])
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

    def _past_the_limit(self, **switches):
        """A live listing with these switches, its day's updates used up (3 a day here)."""
        w, board = warden(updates_per_day=3), Marketboard()
        start = replace(listing(fields=[fld("city", "Lyon"), fld("religion", "Quaker", HIDDEN)]), **switches)
        self.assertEqual([w.admit(person(), replace(start, version=v), board).accepted for v in (1, 2, 3, 4)],
                         [True, True, True, False])
        versions = iter(range(10, 100))

        def update(**changes):
            return w.admit(person(), replace(board.listing("p1"), version=next(versions), **changes), board)
        return w, board, update

    def test_turning_something_off_is_never_held_back_by_the_update_limit(self):
        """Consent first: past the day's update limit, an update to a live listing that only turns something off
        (questions, close matches, or matching itself) still goes in, one switch at a time or together. It is
        checked and receipted like any update; turning one back on is limited as before."""
        for off in ({"notify_questions": False}, {"notify_close": False}, {"active": False},
                    {"notify_questions": False, "notify_close": False, "active": False}):
            with self.subTest(off=off):
                w, board, update = self._past_the_limit(notify_questions=True, notify_close=True, active=True)
                start = len(w.receipts)
                self.assertTrue(update(**off).accepted)
                done = w.receipts[start:]
                self.assertEqual([r.check for r in done], list(ENTRY_CHECKS))  # every check, with its receipt
                self.assertTrue(all(r.result == "pass" for r in done))
                self.assertTrue(all(getattr(board.listing("p1"), k) is False for k in off))
                back_on = {k: True for k in off}
                r = update(**back_on)
                self.assertEqual(r.failed_checks, ["update_rate_limit"])
                self.assertTrue(all(getattr(board.listing("p1"), k) is False for k in off))  # still off

    def test_turning_something_off_isnt_counted_against_the_limit(self):
        w, board = warden(updates_per_day=3), Marketboard()
        on = replace(listing(fields=[fld("city", "Lyon")]), notify_close=True, notify_questions=True)
        self.assertTrue(w.admit(person(), on, board).accepted)  # 1 of 3
        self.assertTrue(w.admit(person(), replace(on, version=2, notify_close=False), board).accepted)  # not counted
        self.assertTrue(w.admit(person(), replace(on, version=3, notify_questions=False, notify_close=False),
                                board).accepted)  # not counted
        self.assertTrue(w.admit(person(), replace(on, version=4, exactness=0.8, notify_close=False,
                                                  notify_questions=False), board).accepted)  # 2 of 3
        self.assertTrue(w.admit(person(), replace(on, version=5, exactness=0.7, notify_close=False,
                                                  notify_questions=False), board).accepted)  # 3 of 3
        self.assertFalse(w.admit(person(), replace(on, version=6, exactness=0.6, notify_close=False,
                                                   notify_questions=False), board).accepted)

    def test_only_an_update_that_does_nothing_but_turn_something_off_is_let_past(self):
        """Anything else past the limit is refused: turning something off and changing anything else (the description,
        a field or whether it's shown, a confirmation, the owner), turning another switch on, turning nothing off, an
        older time or the same version, or a switch that isn't a plain true or false; and a listing that isn't live
        has nothing to turn off."""
        w, board, update = self._past_the_limit(notify_questions=True, notify_close=False, active=True)
        live = board.listing("p1")
        shown = tuple(replace(f, visibility=SHOWN) if f.key == "religion" else f for f in live.fields)
        for changes in ({"intent_text": live.intent_text + " Quiet evenings."}, {"exactness": 0.9},
                        {"fields": live.fields + (fld("travel", "Same city"),)}, {"fields": live.fields[:1]},
                        {"fields": shown, "confirmed_shown": ("religion",)}, {"owner_id": "p1"},
                        {"notify_close": True}, {"updated_at": "2026-10-01T09:00:00Z"}):
            with self.subTest(changes=changes):
                r = update(notify_questions=False, **changes)
                self.assertFalse(r.accepted)
                self.assertIn("update_rate_limit", r.failed_checks)
        for changes in ({}, {"notify_questions": 0}, {"active": None}):
            with self.subTest(changes=changes):
                self.assertIn("update_rate_limit", update(**changes).failed_checks)
        same_version = w.admit(person(), replace(live, notify_questions=False), board)
        self.assertIn("update_rate_limit", same_version.failed_checks)
        self.assertEqual(board.listing("p1"), live)  # nothing went in
        self.assertTrue(update(notify_questions=False).accepted)  # what it may do, it can
        # A new listing has nothing to turn off: its first entry counts, paused or not.
        w2, board2 = warden(updates_per_day=1), Marketboard()
        paused = replace(listing(), active=False)
        self.assertTrue(w2.admit(person(), paused, board2).accepted)  # 1 of 1
        self.assertFalse(w2.admit(person(), replace(paused, version=2, exactness=0.9), board2).accepted)

    def test_one_person_has_at_most_five_live_listings(self):
        """The crowd rules count people; this keeps one person from filling the board."""
        w, board = warden(), Marketboard()
        own = lambda i: replace(listing(f"p1-{i}"), owner_id="p1")
        results = [w.admit(person(), own(i), board).accepted for i in range(MAX_LISTINGS_PER_OWNER + 1)]
        self.assertEqual(results, [True] * MAX_LISTINGS_PER_OWNER + [False])
        self.assertEqual(w.receipts[-2].check, "listing_cap")
        self.assertTrue(w.admit(person(), replace(own(0), version=2), board).accepted)  # an update isn't another
        w.withdraw(person(), "p1-1", board)
        self.assertTrue(w.admit(person(), own(MAX_LISTINGS_PER_OWNER), board).accepted)  # a deleted one frees a place

    def test_updates_are_also_limited_per_person(self):
        """Five listings with five updates a day each: twenty-five a day for the person, and no more."""
        w, board = warden(updates_per_day=5), Marketboard()
        results = []
        for i in range(MAX_LISTINGS_PER_OWNER):
            for v in range(5):
                results.append(w.admit(person(), replace(listing(f"p1-{i}"), owner_id="p1", version=v + 1), board).accepted)
        self.assertTrue(all(results))
        w.withdraw(person(), "p1-0", board)
        self.assertFalse(w.admit(person(), replace(listing("p1-5"), owner_id="p1"), board).accepted)
        self.assertEqual(w.receipts[-1].check, "update_rate_limit")


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

    def test_wishes_never_single_out_one_person_either(self):
        """The description's rules read the open details too, written out as "name: values": an ex, 'named' or
        'called' + a name, 'works at' + a name; and a workplace in one wish with looks in another count together
        (with the description) as a place to find them with how they look (the identifiers rule)."""
        cases = [
            ([detail("place of work", "cashier at the ferry kiosk", about="them"),
              detail("look", ("beard", "round glasses"), about="them")], "where to find them + how they look"),
            ([detail("who", "my ex from the choir", about="them")], "an ex"),
            ([detail("nickname", "called Tomasz", about="them")], "'named'/'called' + a name"),
            ([detail("job", "works at Orbit Tiles", about="both")], "'works at' + a name"),
        ]
        for details, problem in cases:
            self.assertIn(problem, detail_person_problems(details), problem)
            r = warden().admit(person(), listing(fields=[fld("city", "Lyon"), *details]), Marketboard())
            self.assertIn("specific_person", r.failed_checks, problem)
        # The looks in a wish and the workplace in the description count together too.
        looks = [detail("look", "curly hair", about="them")]
        self.assertIn("where to find them + how they look",
                      detail_person_problems(looks, "Looking for the barista at the corner bakery."))
        r = warden().admit(person(), listing(intent="Looking for the barista at the corner bakery.",
                                             fields=[fld("city", "Lyon"), *looks]), Marketboard())
        self.assertIn("specific_person", r.failed_checks)
        # Ordinary wishes are fine, and looks alone (or a workplace alone) say nothing about one person.
        for details in ([detail("look", "tall", about="them")], [detail("place of work", "employee at a bakery",
                                                                          about="them")],
                        [detail("tempo", "unhurried"), detail("tea", ("green", "black"), about="them")]):
            self.assertEqual(detail_person_problems(details), [], details)
            self.assertTrue(warden().admit(person(), listing(fields=[fld("city", "Lyon"), *details]),
                                           Marketboard()).accepted, details)

    def test_the_problem_says_which_detail_it_is_in(self):
        """When an open detail singles someone out, the Warden says which detail (by its name), not the description;
        the description's own problems keep their wording."""
        def said(intent, details):
            r = warden().admit(person(), listing(intent=intent, fields=[fld("city", "Lyon"), *details]), Marketboard())
            self.assertIn("specific_person", r.failed_checks)
            return r.problems
        plain = "Looking for someone to restore old radios with."
        self.assertEqual(said(plain, [detail("workbench", "my ex from the ferry crew")]),
                         ["detail 'workbench' could identify one person (an ex)"])
        self.assertEqual(said(plain, [detail("collector", "named Augustin", about="them")]),
                         ["detail 'collector' could identify one person ('named'/'called' + a name)"])
        self.assertEqual(said(plain, [detail("post", "employee at the harbour office", about="them"),
                                      detail("style", "silver beard", about="them")]),
                         ["details 'post' and 'style' could identify one person (where to find them + how they look)"])
        self.assertEqual(said("Looking for the clerk who works at the harbour office.",
                              [detail("style", "silver beard", about="them")]),
                         ["detail 'style' (read with the description) could identify one person "
                          "(where to find them + how they look)"])
        # The description on its own: its own wording, and nothing said about the details.
        self.assertEqual(said("Looking for the clerk who works at the harbour office, the one with the silver beard.",
                              [detail("style", "relaxed", about="them")]),
                         ["intent could identify one person (where to find them + how they look)"])

    def test_a_rule_reads_the_same_whichever_side_of_a_detail_its_words_are_on(self):
        """An open detail is read as "name: values" and also as its name before each value, so a wish whose NAME is
        the rule's word ("called", "named", "works at") with a name as its value singles someone out as much as the
        same words in one value do. Ordinary names with capitalised values are fine."""
        for details, problem in (([detail("named", "Leopold", about="them")], "'named'/'called' + a name"),
                                 ([detail("called", "Wilhelmina", about="both")], "'named'/'called' + a name"),
                                 ([detail("works at", "Larkspur Glass", about="them")], "'works at' + a name")):
            self.assertIn(problem, detail_person_problems(details), details)
            r = warden().admit(person(), listing(fields=[fld("city", "Lyon"), *details]), Marketboard())
            self.assertIn("specific_person", r.failed_checks, details)
            self.assertIn(f"detail '{details[0].name}' could identify one person", "; ".join(r.problems))
        for details in ([detail("dialect", "Flemish", about="them")], [detail("month", ("May", "June"))]):
            self.assertEqual(detail_person_problems(details), [], details)

    def test_spacing_never_hides_a_rule(self):
        """The rules read every text with its words one space apart, so a doubled space, a tab, a line break or a
        no-break space between two words is read as one space, in the description and in the phrases alike."""
        for sep in ("  ", "\t", "\n", "\u00a0", " \n "):
            for text, problem in ((f"Looking for my{sep}ex to fix a fence with.", "an ex"),
                                  (f"Hoping for someone who works{sep}at Tidewater Mills.", "'works at' + a name"),
                                  (f"The cashier{sep}at the lido with freckles, to swim with.",
                                   "where to find them + how they look")):
                self.assertIn(problem, specific_person_problems(text), repr(text))
                r = warden().admit(person(), listing(intent=text, fields=[fld("city", "Lyon")]), Marketboard())
                self.assertIn("specific_person", r.failed_checks, repr(text))
            r = warden().admit(person(), listing(fields=[fld("city", "Lyon"), fld("seeking", f"my{sep}ex")]),
                               Marketboard())
            self.assertIn("specific_person", r.failed_checks, repr(sep))

    def test_a_workplace_and_looks_anywhere_in_the_listing_count_together(self):
        """A workplace (a place to find them) with looks is judged over the description, the listing's other free text
        about the person sought (the "looking for" phrase, headline, topics) and every shown wish together, since all
        of them describe that person: the workplace in a phrase and the looks in a wish, or in the description, count
        as one description."""
        cashier = fld("seeking", "the employee at the harbour ferry kiosk")
        for fields, problem in (([cashier, detail("look", ("grey beard",), about="them")],
                                 "detail 'look' (read with the phrases) could identify one person"),
                                ([fld("headline", "Looking for the bartender at the harbour inn"),
                                  detail("style", "red hair", about="both")],
                                 "detail 'style' (read with the phrases) could identify one person")):
            r = warden().admit(person(), listing(fields=[fld("city", "Lyon"), *fields]), Marketboard())
            self.assertIn("specific_person", r.failed_checks)
            self.assertIn(problem, "; ".join(r.problems))
            self.assertIn("where to find them + how they look", detail_person_problems(fields))
        # The description's looks with a phrase's workplace: the listing's own words, together.
        r = warden().admit(person(), listing(intent="Hoping to meet someone tall with glasses to play backgammon.",
                                             fields=[fld("city", "Lyon"), cashier]), Marketboard())
        self.assertEqual(r.problems, ["intent could identify one person (where to find them + how they look)"])
        # Either alone says nothing about one person.
        for fields in ([cashier], [fld("seeking", "a backgammon partner"), detail("look", ("grey beard",), about="them")]):
            self.assertTrue(warden().admit(person(), listing(fields=[fld("city", "Lyon"), *fields]),
                                           Marketboard()).accepted, fields)

    def test_clues_that_narrow_to_one_person_are_refused(self):
        """The identifiers rule (replacing "a workplace + looks"): a place or time where the person sought can be found
        with any detail of how they look, three or more details of how they look, or a distinctive mark (where it is
        on them, or what it shows as their feature). One or two common details with no place, or a place with none,
        are fine."""
        from belong.warden import ANCHORED, TOO_MANY_TRAITS, UNIQUE_MARK
        refused = {
            "Hoping to find the florist who works at the corner stall by the town hall, she wears glasses.": ANCHORED,
            "Looking for the woman who lives on the third floor of my building; she is tall.": ANCHORED,
            "I saw you at the night market last Thursday, short with curly hair.": ANCHORED,
            "To the guy I met at the ceilidh: you had freckles, shall we dance again?": ANCHORED,
            "Looking for the man on the 7:15 tram every morning, he has a grey beard.": ANCHORED,
            "Looking for the girl from the noodle bar by the river, very tall.": ANCHORED,
            "The bartender at the rooftop bar on the square is tall.": ANCHORED,  # said after the anchor
            "Hoping to find the short cashier at the ferry terminal again.": ANCHORED,  # before the anchor's person
            "The florist who works at the corner stall looks quite tall.": ANCHORED,
            "Searching for a short woman with freckles and glasses to go canoeing.": TOO_MANY_TRAITS,
            "Hoping to meet a tall man with a beard and a nose ring.": TOO_MANY_TRAITS,
            "Looking for the man with a scar along his jaw.": UNIQUE_MARK,
            "Looking for the woman with a small birthmark just below her ear.": UNIQUE_MARK,
            "Searching for the guy who has a tattoo of a lighthouse.": UNIQUE_MARK,
            "Looking for the woman with a tattoo that reads 'carpe diem'.": UNIQUE_MARK,
        }
        for text, problem in refused.items():
            self.assertIn(problem, specific_person_problems(text), text)
            r = warden().admit(person(), listing(intent=text, fields=[fld("city", "Lyon")]), Marketboard())
            self.assertEqual(r.problems, [f"intent could identify one person ({problem})"], text)
        fine = ("Hoping to meet someone who works at the bike workshop to learn wheel truing from.",
                "Looking for a tall dance partner with glasses for swing nights.",
                "Hoping to meet a woman with freckles who loves sailing.",
                "Looking for a sketching partner who has a tattoo or two.",
                "Looking for a tattoo artist to design a tattoo of a hummingbird for my shoulder.",
                "Looking for a tall basketball partner at my gym on Tuesday evenings.",
                "Looking for a squash partner who works at weekends, ideally tall.")  # a time, not a place
        for text in fine:
            self.assertEqual(specific_person_problems(text), [], text)
            self.assertTrue(warden().admit(person(), listing(intent=text, fields=[fld("city", "Lyon")]),
                                           Marketboard()).accepted, text)

    def test_a_looks_word_counts_only_where_it_describes_a_person(self):
        """A describing word of looks counts next to a word for a person, said of one, standing alone between commas or
        in a list with another look; never where it describes a thing. A looks noun never counts where it describes
        a thing ("... salon", "... of wine")."""
        anchor = "Hoping to meet someone who works at the old boatyard"
        person_, thing = [], []
        person_ += [f"{anchor}, a short man.", f"{anchor}; someone tall.", f"{anchor}; she's quite short.",
                    f"{anchor}, tall, kind.", f"{anchor}, short with a beard.", f"{anchor}, red hair.",
                    f"{anchor}, glasses and freckles."]
        thing += [f"{anchor} for short sailing trips.", f"{anchor} to help me build a tall bookcase.",
                  f"{anchor} for a season of short paddles.", f"{anchor}, keep it short.",
                  f"{anchor} who also runs a hair salon.", f"{anchor} to share glasses of cider after.",
                  f"{anchor}, tall glasses of lemonade included.", f"{anchor} who knows a tattoo artist."]
        for text in person_:
            self.assertIn("where to find them + how they look", specific_person_problems(text), text)
        for text in thing:
            self.assertEqual(specific_person_problems(text), [], text)

    def test_the_posters_own_job_and_looks_never_count(self):
        """Only clues about the person sought count: the poster's own workplace or home ("I work at", "I'm a cashier
        at", "and work at" in a sentence they start), their own looks ("I'm tall", "I have red hair", "my beard",
        and a list after them) and a mark on their own body. A first-person sentence that points at someone else
        still counts."""
        for text in ("I work at a bakery near the canal at dawn, so I'm after short afternoon strolls.",
                     "I'm a cashier at the central station kiosk and I'm quite tall; looking for a volleyball team.",
                     "I have long red hair and freckles and work at the town library; hoping to join a quiz team.",
                     "My beard is grey now, and I work at the harbour office; seeking a draughts partner.",
                     "I'm short, with glasses and a beard; looking for a partner who works at the hospital.",
                     "We live on a narrowboat and we're both tall; looking for rowing partners.",
                     "I have a scar on my knee, so I'm after gentle canoe trips with someone who works at the lido."):
            self.assertEqual(specific_person_problems(text), [], text)
        for text in ("I'd like the cashier at the ferry terminal, short with glasses.",
                     "I'm into the barista at the lido kiosk, the one with the nose ring.",
                     "I want to meet the bartender at the jazz club; she has short black hair."):
            self.assertIn("where to find them + how they look", specific_person_problems(text), text)

    def test_clues_count_together_across_the_listing(self):
        """The clues are counted over the description, the free-text built-ins about the person sought and the shown
        wishes together. What the poster offers (their "you are" phrase) is their own; a "both" wish is about the
        request, so a looks word there describes a thing; a "them" wish describes the person sought."""
        def problems(intent, *fields):
            r = warden().admit(person(), listing(intent=intent, fields=[fld("city", "Lyon"), *fields]), Marketboard())
            return r.problems
        plain = "Looking for someone to restore old clocks with."
        self.assertEqual(problems(plain, fld("seeking", "the receptionist who works at the dental clinic"),
                                  detail("look", "red hair", about="them")),
                         ["detail 'look' (read with the phrases) could identify one person "
                          "(where to find them + how they look)"])
        self.assertEqual(problems("He was tall and had freckles; I'd love to dance again.",
                                  fld("headline", "Looking for the man I met at the ceilidh")),
                         ["intent could identify one person (where to find them + how they look)"])
        self.assertEqual(problems(plain, detail("height", "tall", about="them"),
                                  detail("look", ("glasses", "beard"), about="them")),
                         ["details 'height' and 'look' could identify one person "
                          "(three or more details of how they look)"])
        self.assertEqual(problems(plain, detail("where", "works at the clock museum", about="them"),
                                  detail("look", "short, with a moustache", about="them")),
                         ["details 'where' and 'look' could identify one person (where to find them + how they look)"])
        self.assertEqual(problems(plain, detail("look", "a tattoo of a swallow on her wrist", about="them")),
                         ["detail 'look' could identify one person (a distinctive mark and where it is or what it "
                          "shows)"])
        # The poster's own offer, a "both" wish about the request, and a them-wish alone: nothing to refuse.
        self.assertEqual(problems("Looking for a tall swing dance partner.",
                                  fld("offering", "a barista at the station cafe"),
                                  fld("seeking", "a swing dance partner")), [])
        self.assertEqual(problems("Looking for a walking group; I work at the hospital.",
                                  fld("seeking", "a walking group who meets at the town hall"),
                                  detail("walk length", "short", about="both")), [])
        self.assertEqual(problems(plain, detail("height", "tall", about="them")), [])

    def test_the_identifiers_rule_reads_in_time_in_proportion_to_the_text(self):
        """Each text is read once, with windows of a few words, so a long hostile text can't stall it: quadratic
        reading would take minutes on these, linear takes milliseconds."""
        import time
        from belong.warden import identifier_problems
        for unit in ("I'm tall and work at the x. ", "and work at the ", "tall, ", "with a tattoo of a ",
                     "the man at the x. is tall, ", "I saw you ", "hair hair ", "my my short ", "the 6:05 ",
                     "works at the x. it is short and "):
            text = unit * (60000 // len(unit))
            start = time.perf_counter()
            identifier_problems([text])
            self.assertLess(time.perf_counter() - start, 5.0, unit)

    def test_the_full_name_rule_has_no_typed_list(self):
        """Two capitalised words in a row in the middle of a sentence look like a full name. The only words never part
        of one are fixed forms (the pronoun I and its contractions, a word written all in capitals, a month or a
        weekday from the standard library) and, at entry, the listing's own shown place and the Warden's shared list:
        no list of places, languages or titles, so a pair the listing doesn't show as its place is refused, whatever it
        is. A name after "in" stays a name, with or without a possessive."""
        import ast
        import re
        from pathlib import Path
        from belong import warden as warden_module
        # Read from the Warden's source, whatever a list would be called: no set, list, tuple or dict of two or more
        # capitalised words, and no pattern that alternates them.
        for node in ast.walk(ast.parse(Path(warden_module.__file__).read_text())):
            elts = node.elts if isinstance(node, (ast.Set, ast.List, ast.Tuple)) else \
                node.keys if isinstance(node, ast.Dict) else None
            if elts is not None:
                words = [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)
                         and re.fullmatch(r"[A-Z][a-z]{2,}", e.value)]
                self.assertLess(len(words), 2, words)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                self.assertIsNone(re.search(r"\b[A-Z][a-z]+\|[A-Z][a-z]+\b", node.value), node.value)
        for text in ("I want to meet Mika Sato again.", "Anyone interested in Mika Sato's music?",
                     "Hoping to reconnect with Tomasz Wójcik from the ferry crew.",
                     "Is anyone else a fan of Amara Okafor's poems?", "Looking for someone interested in Ivo Brandl",
                     "Looking for a kayaking partner around Puerto Varas.",  # a place this listing doesn't show
                     "Looking for a tutor for Old Norse sagas."):  # a language written as two capitalised words
            self.assertIn("what looks like a full name", specific_person_problems(text), text)
            r = warden().admit(person(), listing(intent=text, fields=[fld("city", "Lyon")]), Marketboard())
            self.assertIn("specific_person", r.failed_checks, text)
        for text in ("Since Friday I'm free most evenings, and I'd like a chess partner.",
                     "Free on Saturday Mornings in July for a pottery class.",
                     "Looking for a CAD ENGINEER to check my drawings, and I've got the files.",
                     "Studying for the GMAT AWA essays and hoping for a study partner.",
                     "Mostly free in late October, I'll travel."):
            self.assertEqual(specific_person_problems(text), [], text)
            self.assertTrue(warden().admit(person(), listing(intent=text, fields=[fld("city", "Lyon")]),
                                           Marketboard()).accepted, text)

    def test_the_listings_own_shown_place_is_never_a_name(self):
        """Words of the city and country a listing shows are never part of a name (compared without accents, case or a
        possessive). Only those built-ins, and only shown: an open detail, a hidden city or another hidden field
        with the same words never unlock a name, since anyone could put a name there."""
        def admitted(intent, *fields):
            return warden().admit(person(), listing(intent=intent, fields=list(fields)), Marketboard()).accepted
        lakes = "Looking for a kayaking partner around the Puerto Varas lakes."
        self.assertTrue(admitted(lakes, fld("city", "Puerto Varas"), fld("country", "Chile")))
        self.assertTrue(admitted("Looking for someone to explore Zurich's Altstadt with.", fld("city", "Zürich")))
        self.assertTrue(admitted("Hoping to find a diving buddy along the Sri Lanka coast.", fld("country", "Sri Lanka")))
        for fields in ([], [fld("country", "Chile")], [fld("city", "Puerto Varas", HIDDEN)],
                       [detail("home town", "Puerto Varas")], [fld("neighborhood", "Puerto Varas", HIDDEN)]):
            self.assertFalse(admitted(lakes, *fields), fields)
        # A word of the place unlocks only that word: another capitalised word next to it still counts.
        self.assertFalse(admitted("Walks round the Puerto Varas Old Pier.", fld("city", "Puerto Varas")))
        self.assertEqual(specific_person_problems(lakes, [fld("city", "Puerto Varas")]), [])
        self.assertTrue(specific_person_problems(lakes, [fld("city", "Puerto Varas", HIDDEN)]))

    def test_a_place_on_the_wardens_shared_list_is_never_a_name(self):
        """A place on the last shared list the Warden built (K_CROWD people show it, on listings a week old) is never
        part of a name at entry, wherever the listing itself is. Entry never builds the list: before the first one,
        no place is on it."""
        w, board = warden(), Marketboard()
        text = "Looking for a buddy for gallery days in Santa Fe."
        mine = listing("me1", intent=text, fields=[fld("city", "Albuquerque")])
        self.assertEqual(w.listed_places(), [])
        self.assertFalse(w.admit(person("me1"), mine, Marketboard()).accepted)
        for i in range(K_CROWD):
            other = replace(listing(f"sf{i}", intent="Looking for a sketching partner.",
                                    fields=[fld("city", "Santa Fe"), fld("country", "United States")]),
                            updated_at="2026-10-01T00:00:00Z")
            self.assertTrue(w.admit(person(f"sf{i}"), other, board, history=True).accepted)  # a week old, seeded
        receipts = len(w.receipts)
        self.assertFalse(w.admit(person("me1"), mine, Marketboard()).accepted)  # entry built nothing
        self.assertEqual(len(w.receipts) - receipts, len(ENTRY_CHECKS))  # its own checks' receipts, no more
        w.details_in_use(board)
        self.assertIn({"city": "Santa Fe", "country": "United States"}, w.listed_places())
        self.assertTrue(w.admit(person("me1"), mine, Marketboard()).accepted)
        # Still a name when a person's name is not a listed place.
        self.assertFalse(w.admit(person("me1"), replace(mine, intent_text="Looking for Ivo Brandl in Santa Fe."),
                                 Marketboard()).accepted)

    def test_a_place_unlocks_a_name_only_where_it_stands_whole(self):
        """A place (the listing's own shown city or country, or one on the Warden's shared list) is never part of a
        name only where all its words stand together, in order: one word of a two-word place that is also a given
        name unlocks nothing, so an ordinary full name stays refused wherever the listing is."""
        def admitted(intent, *fields, w=None):
            return (w or warden()).admit(person(), listing(intent=intent, fields=list(fields)), Marketboard()).accepted
        own = fld("city", "Santa Teresa")
        self.assertTrue(admitted("Sketching by the Santa Teresa Tram on weekend mornings.", own))
        self.assertTrue(admitted("Sketching around Santa Teresa's old houses, any weekend.", own))
        for text in ("I want to meet Teresa Okonjo again, we sketched together.",
                     "Hoping to find Teresa Lindgren's sister, she paints."):
            self.assertFalse(admitted(text, own), text)
        self.assertFalse(admitted("Anyone know Louis Ferreira? He used to sail here.", fld("city", "Mahébourg"),
                                  fld("country", "Mauritius")))
        # A listed place, the same way.
        w = warden()
        w.listed_places = lambda: [{"city": "Port Louis", "country": "Mauritius"}]
        self.assertTrue(admitted("Day trips to the Port Louis Market, any Saturday.", fld("city", "Curepipe"), w=w))
        self.assertFalse(admitted("Anyone know Louis Ferreira? He used to sail here.", fld("city", "Curepipe"), w=w))
        self.assertEqual(specific_person_problems("I want to meet Teresa Okonjo again.", [own]),
                         ["what looks like a full name"])

    def test_a_name_with_a_particle_or_after_an_initial_is_still_a_name(self):
        """The lower-case words that stand inside a full name in many languages ("van", "dos", "bin", "de la") don't
        break the pair, and neither does the full stop after an initial or a short title written without vowels."""
        for text in ("Hoping to see Anouk van Leeuwen again, we rowed together.",
                     "Looking for Rui dos Anjos from the old choir.",
                     "Anyone know Yusuf bin Hamad? He coached our team.",
                     "I'd love to meet Inés de la Fuente again.",
                     "I want to meet K. Mensah again, we met at a talk.",
                     "Hoping to find Mrs. Adeyemi from the allotments."):
            self.assertIn("what looks like a full name", specific_person_problems(text), text)
        for text in ("Free from 7 p.m. Anyone up for chess near the station?",
                     "I'm in Bergen. Weekends are free for climbing.",
                     "Looking for a doubles partner, de-stressing after work."):
            self.assertEqual(specific_person_problems(text), [], text)

    def test_a_character_that_shows_nothing_never_hides_a_rule(self):
        """A text is read as it looks: a format character that shows nothing (a zero-width space, a word joiner, a soft
        hyphen, a byte order mark) or a variation selector beside a space, and a letter written with a separate accent
        mark, hide no rule."""
        for joiner in ("\u200b", "\u2060", "\u00ad", "\ufeff", "\ufe0f"):
            for text in (f"Hoping to meet Aino{joiner} Virtanen again.", f"Still hoping to find my{joiner} ex.",
                         f"The short barista{joiner} at the lido on my road."):
                self.assertTrue(specific_person_problems(text), (joiner, text))
                r = warden().admit(person(), listing(intent=text, fields=[fld("city", "Tampere")]), Marketboard())
                self.assertIn("specific_person", r.failed_checks, (joiner, text))
        self.assertIn("what looks like a full name", specific_person_problems("I want to meet Zoe\u0308 Brandt again."))

    def test_a_live_listing_keeps_the_listed_places_it_was_accepted_with(self):
        """A later shared list that drops a place never refuses a live listing's next update (pausing it, say): the
        Warden keeps, for each live listing, the listed places it was accepted with. A new listing reads only today's
        list, and a withdrawn listing's kept places go with it."""
        from belong.warden import Clock, Warden
        clock = Clock("2026-10-20T09:00:00")
        w, board = Warden(clock, key=b"test-key"), Marketboard()
        for i in range(K_CROWD):
            other = replace(listing(f"pl{i}", intent="Looking for a sketching partner.",
                                    fields=[fld("city", "Port Louis"), fld("country", "Mauritius")]),
                            updated_at="2026-10-01T00:00:00Z")
            self.assertTrue(w.admit(person(f"pl{i}"), other, board, history=True).accepted)  # a week old, seeded
        w.details_in_use(board)
        text = "Day trips to the Port Louis Market, any Saturday."
        mine = listing("me1", intent=text, fields=[fld("city", "Curepipe")])
        self.assertTrue(w.admit(person("me1"), mine, board).accepted)
        for i in range(K_CROWD - 5):
            self.assertTrue(w.withdraw(person(f"pl{i}"), f"pl{i}", board))
        clock.advance(days=1)
        w.details_in_use(board)
        self.assertEqual(w.listed_places(), [])
        self.assertTrue(w.admit(person("me1"), replace(mine, active=False, version=2), board).accepted)
        self.assertFalse(w.admit(person("me2"), listing("me2", intent=text, fields=[fld("city", "Curepipe")]),
                                 board).accepted)
        self.assertTrue(w.withdraw(person("me1"), "me1", board))
        self.assertEqual(w.kept_places("me1"), [])

    def test_more_ways_to_say_the_poster_saw_or_met_them(self):
        """Someone the poster saw or met is a place to find them however it's said: with no subject ("Ran into you
        ...": the verb opens the sentence), with "we" or "you and I" and "at" a place, or with "a" + a word for one
        person; plural or general meetings aren't."""
        for text in ("Spotted you at the bus shelter on Elm Road, tall with a grey coat and glasses.",
                     "Met you on the night ferry to Split; you had freckles.",
                     "We met at the allotment open day; you're short with curly hair.",
                     "You and I met at the choir taster. You are tall and wear glasses.",
                     "I noticed a tall man by the fountain on Saturday, with a beard."):
            self.assertIn("where to find them + how they look", specific_person_problems(text), text)
        for text in ("I've met plenty of people at climbing walls; hoping for a tall belay partner.",
                     "We met at university and now we're looking for a third for board games.",
                     "Met lots of runners at the park; looking for someone with glasses to pace with."):
            self.assertEqual(specific_person_problems(text), [], text)

    def test_a_looks_word_with_what_it_is_like_still_counts(self):
        """A looks noun or mark followed by a word that says what it's like ("hair tone", "a tattoo sleeve") is a
        detail of how they look when it is someone's feature: after "with", "has", "his"... or opening a phrase after a
        comma, semicolon, colon, dash or bracket. Where nothing marks it as a feature, a word after it makes a thing
        ("a hair salon")."""
        for text, problem in (
                ("Hoping to find the man who works at the fish market; hair tone silver.", "where to find them + how they look"),
                ("Looking for the woman behind the till at the deli - glasses frame tortoiseshell.", "where to find them + how they look"),
                ("Hoping to meet someone with a tattoo sleeve of waves on his right arm.", "a distinctive mark and where it is or what it shows"),
                ("Looking for someone with hair tone copper, glasses frame round and freckles.", "three or more details of how they look")):
            self.assertIn(problem, specific_person_problems(text), text)
        for text in ("Looking for someone who works at a hair salon to swap haircuts for bike repairs.",
                     "Hoping to find a tattoo studio buddy who works at the market on Saturdays."):
            self.assertEqual(specific_person_problems(text), [], text)

    def test_where_a_mark_is_said_through_the_of_their_body(self):
        """A mark is described with where it is on them through "the ... of his / her / their / your" too ("on the back
        of her neck"), with up to five words of its own between."""
        for text in ("Looking for the woman with a moth tattoo on the side of her neck.",
                     "Hoping to find a man with a scar along the edge of his jaw.",
                     "Looking for someone with a birthmark on the inside of their wrist."):
            self.assertIn("a distinctive mark and where it is or what it shows", specific_person_problems(text), text)
        self.assertEqual(specific_person_problems("Looking for a tattoo artist with a studio on the side of the market."),
                         [])

    def test_a_workplace_or_home_the_poster_shares_with_them_is_where_to_find_them(self):
        """"We both work at", "we all live on" or "the same ... as you" is where the person sought is found too, so it
        counts; the poster's own workplace or home alone never does."""
        for text in ("We both live on the narrowboat moorings by the lock; he has a beard.",
                     "We both work at the hospital pharmacy; she has curly hair and freckles.",
                     "I live on the same street as you, and you're quite tall."):
            self.assertIn("where to find them + how they look", specific_person_problems(text), text)
        for text in ("I work at the hospital pharmacy and I have curly hair; looking for a running buddy.",
                     "I live on the same street every summer and I'm tall; hoping to find a tennis partner."):
            self.assertEqual(specific_person_problems(text), [], text)

    def test_what_the_poster_says_about_themselves_is_read_as_theirs(self):
        """An open detail about the poster and their "what you offer" phrase are read as the poster saying it about
        themselves: their own job and looks never count, but a clue about someone else written there does, with the
        rest of the listing."""
        def problems(*fields, intent="Looking for a chess partner."):
            return detail_person_problems(list(fields), intent)
        self.assertEqual(problems(detail("job", "a nurse at the hospital"), detail("look", "tall, glasses, a beard")),
                         [])
        self.assertEqual(problems(detail("hair", "curly"), detail("height", "tall"), detail("tattoos", "two")), [])
        for fields in ([detail("sundays", "saw her at the deli, she's tall")],
                       [detail("hoping for", "the cashier with a scar on his chin")],
                       [detail("weekends", "i met her at the lido"), detail("look", "freckles", about="them")]):
            self.assertTrue(problems(*fields), fields)
        self.assertEqual(problems(fld("offering", "a nurse at the hospital, tall with glasses")), [])
        self.assertIn("where to find them + how they look",
                      problems(fld("offering", "a friend for the man at the deli with a beard")))
        # At entry, naming the detail.
        r = warden().admit(person(), listing(intent="Looking for a chess partner.",
                                             fields=[fld("city", "Tampere"), detail("sundays", "saw her at the deli, she's tall")]),
                           Marketboard())
        self.assertIn("detail 'sundays' could identify one person (where to find them + how they look)", r.problems)

    def test_a_listing_id_is_one_listings_for_good(self):
        """A listing id is never taken by another person's listing, and never used again once its listing is
        deleted (by anyone, its owner included): the Warden reissues an introduction from what the board holds under
        its two ids, so this keeps that the listing it was sent to. Updates to one's own live listing still go in."""
        w, board = warden(), Marketboard()
        mine = replace(listing("ada-1"), owner_id="ada")
        self.assertTrue(w.admit(person("ada"), mine, board).accepted)
        taken = w.admit(person("ben"), replace(listing("ada-1"), owner_id="ben"), board)
        self.assertFalse(taken.accepted)
        self.assertEqual(taken.failed_checks, ["listing_id_owner"])
        self.assertEqual(board.listing("ada-1").owner, "ada")
        self.assertTrue(w.admit(person("ada"), replace(mine, version=2), board).accepted)  # an update
        self.assertTrue(w.withdraw(person("ada"), "ada-1", board))
        for who in ("ada", "ben"):
            back = w.admit(person(who), replace(listing("ada-1"), owner_id=who, version=3), board)
            self.assertEqual(back.failed_checks, ["listing_id_owner"], who)
            self.assertNotIn("ada-1", board)
        self.assertTrue(w.admit(person("ada"), replace(listing("ada-2"), owner_id="ada"), board).accepted)

    def test_a_shown_headline_or_phrase_never_claims_what_the_id_check_vouches_for(self):
        for key, value in (("headline", "Swapping seeds, KYC verified"), ("seeking", "a verified buyer"),
                           ("topics", ("seeds", "Verified"))):
            r = warden().admit(person(), listing(fields=[fld("city", "Lyon"), fld(key, value)]), Marketboard())
            self.assertIn("listing_schema", r.failed_checks, key)
            self.assertIn("vouches", "; ".join(r.problems), key)
        ok = listing(fields=[fld("city", "Lyon"), fld("headline", "Swapping seeds, same age group welcome")])
        self.assertTrue(warden().admit(person(), ok, Marketboard()).accepted)

    def test_drafted_listings_pass_entry_for_verified_people(self):
        w, board = warden(), Marketboard()
        for p in generate(300, 7):
            r = w.admit(p.person, draft_listing(p, "2026-10-12T09:00:00Z"), board)
            self.assertEqual(r.accepted, p.person.verified and not p.truth.bad_intent, (p.person.id, r.problems))


if __name__ == "__main__":
    unittest.main()
