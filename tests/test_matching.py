"""Matching rules and quality (SPEC §5, §6, §8)."""
from __future__ import annotations

import unittest

from belong import agent
from belong.evaluate import compatible
from belong.matching.intent import parse_intent
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import OfflineMatcher
from belong.sim import run
from belong.synth import generate
from tests.helpers import warden


class Quality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = run(n=300, seed=7)

    def test_proposals_are_correct_by_ground_truth(self):
        self.assertGreaterEqual(self.result["metrics"]["precision"], 0.95)

    def test_most_matchable_people_get_a_proposal(self):
        self.assertGreaterEqual(self.result["metrics"]["coverage"], 0.85)

    def test_every_proposal_is_mutual(self):
        """Both must be looking: nobody is approached who didn't ask. (A close match is the one
        exception, and only for someone who opted in: see the next test.)"""
        exact = [p for p in self.result["proposals_internal"] if not p["close"]]
        self.assertTrue(exact)
        for p in exact:
            self.assertTrue(p["truly_compatible"], p)

    def test_close_matches_go_one_way_to_people_who_opted_in(self):
        people = {r["id"]: r for r in self.result["people"]}
        close = [p for p in self.result["proposals_internal"] if p["close"]]
        self.assertTrue(close)
        intros = {(i["person_a"], i["person_b"]): i for i in self.result["introductions"]}
        for p in close:
            self.assertTrue(people[p["close"]]["notify_close"], p)
            intro = intros.get((p["person_a"], p["person_b"]))
            if intro:
                self.assertEqual(intro["one_way_to"], p["close"])
                me, other = ("a", "b") if p["close"] == p["person_a"] else ("b", "a")
                if not intro[f"{me}_said_yes"]:
                    self.assertIsNone(intro[f"{other}_said_yes"], "the other person saw a close match before a yes")


class HiddenFieldRules(unittest.TestCase):
    def _board(self, n=300, seed=7):
        w, board = warden(), Marketboard()
        personas = generate(n, seed)
        for p in personas:
            w.admit(p.person, agent.draft_listing(p, "2026-10-12T09:00:00Z"), board)
        return board, {p.person.id: p for p in personas}

    def test_hidden_fields_only_raise_scores_and_never_justify_a_match(self):
        board, _ = self._board()
        for p in OfflineMatcher().match(board).proposals:
            self.assertGreaterEqual(p.score, p.shown_score)
            self.assertGreaterEqual(p.shown_score, p.shown_threshold)
            self.assertTrue(p.reason_fields)

    def test_removing_hidden_fields_never_removes_a_match(self):
        """If a pair is proposed, it would also qualify with every hidden field deleted."""
        from dataclasses import replace
        from belong.schema import SHOWN
        board, _ = self._board()
        shown_only = Marketboard()
        for l in board.listings():
            shown_only.put(board.person(l.person_id),
                           replace(l, fields=tuple(f for f in l.fields if f.visibility == SHOWN)))
        m = OfflineMatcher(max_per_person=10_000)
        full = {(p.person_a, p.person_b) for p in m.match(board).proposals}
        without = {(p.person_a, p.person_b) for p in m.match(shown_only).proposals}
        self.assertLessEqual(full, without)

    def test_crowd_threshold_blocks_rare_hidden_values(self):
        board, _ = self._board()
        counts = board.value_counts()
        for p in OfflineMatcher(k_crowd=20).match(board).proposals:
            a, b = board.listing(p.person_a), board.listing(p.person_b)
            for key in p.used_hidden_fields:
                shared = set(a.get(key).strings()) & set(b.get(key).strings())
                self.assertTrue(any(counts[(key, s)] >= 20 for s in shared))
        # With an impossible threshold, no hidden field is ever used.
        self.assertFalse(any(p.used_hidden_fields for p in OfflineMatcher(k_crowd=10_000).match(board).proposals))


class IntentReading(unittest.TestCase):
    def test_kinds_are_read_from_free_text(self):
        cases = {
            "A climbing partner in Osaka on weekends. I'm an intermediate climber.": "climbing",
            "Native German speaker learning English, looking for a tandem partner.": "language",
            "Need a CTO-type co-founder for my fintech idea (remote is fine).": "cofounder",
            "Looking for a serious relationship with a woman in Kyoto, aged 25 to 34.": "dating",
            "Need a physics tutor in Lisbon or online, weekends.": "tutoring",
            "Housemate wanted in Berlin, January move-in.": "roommate",
            "Training for a half marathon in London and want company on weekend runs.": "running",
        }
        for text, kind in cases.items():
            self.assertEqual(parse_intent(text).kind, kind, text)

    def test_dating_details(self):
        p = parse_intent("I'd like to meet men in Tokyo for something casual. Ideally aged 30 to 39.")
        self.assertEqual(p.seeking, ("man",))
        self.assertEqual(p.age_range, (2, 3))

    def test_parser_agrees_with_truth_on_synthetic_people(self):
        for p in generate(300, 11):
            if not p.truth.bad_intent:
                # Open-ended listings, services and things are recognised by their seeking/offering
                # phrases, not the text, so the text must not look like one of the named kinds.
                expected = None if p.truth.kind in ("other", "service", "item") else p.truth.kind
                self.assertEqual(parse_intent(p.intent_text).kind, expected, p.intent_text)


class OpenEnded(unittest.TestCase):
    """'Something else' matches: each person's 'seeking' must fit the other's 'offering'."""

    def _board(self, *people):
        from belong.schema import Field, Listing, Person, SHOWN
        w, board = warden(), Marketboard()
        for pid, intent, seeking, offering in people:
            listing = Listing(pid, intent, 0.6, (Field("city", "Tokyo", SHOWN, False),
                                                 Field("seeking", seeking, SHOWN, False),
                                                 Field("offering", offering, SHOWN, False),
                                                 Field("schedule", "weekends", SHOWN, False)), "t")
            self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), listing, board).accepted, pid)
        return board

    def pairs(self, board):
        return {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(board).proposals}

    def test_band_and_guitarist_are_introduced(self):
        board = self._board(("ren", "Looking for a guitarist for my ska band in Tokyo.", "a ska guitarist", "a ska band"),
                            ("kai", "I'm a guitarist in Tokyo looking for a ska band to join.", "a ska band", "a ska guitarist"))
        self.assertEqual(self.pairs(board), {frozenset(("ren", "kai"))})

    def test_two_bands_wanting_guitarists_are_not(self):
        board = self._board(("ren", "Looking for a guitarist for my ska band in Tokyo.", "a ska guitarist", "a ska band"),
                            ("mio", "Need a guitarist for our ska band in Tokyo.", "a ska guitarist", "a ska band"))
        self.assertEqual(self.pairs(board), set())

    def test_genre_must_fit(self):
        board = self._board(("ren", "Looking for a guitarist for my ska band in Tokyo.", "a ska guitarist", "a ska band"),
                            ("kai", "I'm a guitarist in Tokyo looking for a jazz band.", "a jazz band", "a jazz guitarist"))
        self.assertEqual(self.pairs(board), set())

    def test_teacher_and_learner_match_anywhere_in_a_country(self):
        from belong.schema import Field, Listing, Person, SHOWN
        w, board = warden(), Marketboard()
        people = [
            ("aoife", "Looking for someone to teach me to fish in Ireland.", None, "a fishing teacher", "a beginner fishing student"),
            ("sean", "Happy to teach fishing to a beginner around Galway.", "Galway", "a fishing student", "a fishing mentor"),
            ("hans", "Happy to teach fishing to a beginner around Munich.", "Munich", "a fishing student", "a fishing mentor"),
        ]
        for pid, intent, city, seeking, offering in people:
            fields = [Field("country", "Germany" if pid == "hans" else "Ireland", SHOWN, False),
                      Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False),
                      Field("interests", ("fishing",), SHOWN, False)]
            if city:
                fields.append(Field("city", city, SHOWN, False))
            self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), Listing(pid, intent, 0.6, tuple(fields), "t"), board).accepted)
        props = OfflineMatcher().match(board).proposals
        self.assertEqual({frozenset((p.person_a, p.person_b)) for p in props}, {frozenset(("aoife", "sean"))})
        self.assertIn("country", props[0].reason_fields)

    def test_listings_filed_under_different_kinds_are_still_compared(self):
        """Yuki's agent filed 'tutoring', Ren's filed an open-ended listing. With a wide net,
        a French-speaking chess partner can help someone who wants to learn French and chess."""
        from belong.schema import Field, Listing, Person, SHOWN
        def board(exactness):
            w, b = warden(), Marketboard()
            people = [("yuki", "Looking for a French tutor who can also teach me chess in Malaga.",
                       "a French and chess teacher", "a French and chess student", (("role", "student"), ("subject", "French and chess"))),
                      ("ren", "Looking to meet new people in Malaga for chess games and practising languages.",
                       "a chess and language partner", "a chess and language partner", ())]
            for pid, intent, seeking, offering, extra in people:
                fields = (Field("country", "Spain", SHOWN, False), Field("city", "Malaga", SHOWN, False),
                          Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False),
                          *(Field(k, v, SHOWN, False) for k, v in extra))
                self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), Listing(pid, intent, exactness, fields, "t"), b).accepted)
            return b
        self.assertEqual(self.pairs(board(0.6)), set())  # neither asked for exactly what the other is
        self.assertEqual(self.pairs(board(0.3)), {frozenset(("yuki", "ren"))})  # a wider net brings them together

    def test_someone_who_just_wants_company_is_open_to_anyone_with_a_wide_net(self):
        """The exact phrases two live agents wrote for Yuki and Ren in Malaga."""
        from belong.matching.matcher import phrase_satisfies
        yuki_seeks, yuki_is = "a French teacher who can teach chess in person", "a French learner and chess student"
        ren = "a social hangout partner"
        self.assertFalse(phrase_satisfies(yuki_seeks, ren, 0.6, ["chess", "languages"]))
        self.assertTrue(phrase_satisfies(yuki_seeks, ren, 0.3, ["chess", "languages"]))
        self.assertTrue(phrase_satisfies(ren, yuki_is, 0.3, ["chess", "French"]))
        self.assertFalse(phrase_satisfies(ren, "a ska band", 0.3))  # a band isn't a hangout partner

    def test_open_to_more_than_one_option(self):
        """'A tutor or an exchange partner, either one' must match either kind of person."""
        from belong.matching.matcher import phrase_satisfies, alternatives
        yuki_seeks = "a German tutor or a German language exchange partner"
        yuki_is = "a German student or an English and Japanese language exchange partner"
        tutor = ("a German student", "a German tutor")
        exchange = ("a Japanese language exchange partner", "a German language exchange partner")
        for seeks, is_ in (tutor, exchange):
            self.assertTrue(phrase_satisfies(yuki_seeks, is_, 0.6) and phrase_satisfies(seeks, yuki_is, 0.6), is_)
        # Not a Spanish tutor, and not someone who wants Korean practice Yuki can't give.
        self.assertFalse(phrase_satisfies(yuki_seeks, "a Spanish tutor", 0.6))
        self.assertFalse(phrase_satisfies("a Korean language exchange partner", yuki_is, 0.6))
        # Short options borrow what they leave out.
        self.assertEqual(alternatives("a German or Spanish tutor"),
                         [["german", "language", "teach"], ["spanish", "language", "teach"]])

    def test_places_ignore_accents(self):
        from belong.matching.narrow import _place
        self.assertEqual(_place("Málaga"), _place("malaga "))

    def test_open_ended_people_in_the_population_match_correctly(self):
        result = run(n=300, seed=7)
        other = result["metrics"]["by_kind"]["other"]
        self.assertGreater(other["proposals"], 0)
        self.assertEqual(other["precision"], 1.0)



class UnstatedPreferences(unittest.TestCase):
    """Something you didn't say is a preference you don't have; something the other person
    didn't say only fails a strict request."""

    def board(self, a_fields, b_exactness):
        from belong.schema import Field, Listing, Person, SHOWN
        w, board = warden(), Marketboard()
        people = [("pa", "Looking for a climbing partner in Osaka.", 0.6, a_fields),
                  ("pb", "Looking for a climbing partner in Osaka, intermediate, weekends.", b_exactness,
                   {"level": "intermediate", "schedule": "weekends"})]
        for pid, intent, ex, extra in people:
            fields = [Field("country", "Japan", SHOWN, False), Field("city", "Osaka", SHOWN, False)]
            fields += [Field(k, v, SHOWN, False) for k, v in extra.items()]
            self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), Listing(pid, intent, ex, tuple(fields), "t"), board).accepted)
        return board

    def test_no_level_given_still_matches(self):
        pairs = {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(self.board({}, 0.6)).proposals}
        self.assertIn(frozenset(("pa", "pb")), pairs)

    def test_a_strict_request_needs_the_detail(self):
        pairs = {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(self.board({}, 0.9)).proposals}
        self.assertNotIn(frozenset(("pa", "pb")), pairs)


if __name__ == "__main__":
    unittest.main()
