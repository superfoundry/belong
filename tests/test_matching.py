"""Matching rules and quality (SPEC §5, §6, §8)."""
from __future__ import annotations

import unittest
from dataclasses import replace

from belong.matching.intent import parse_intent
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import (OfflineMatcher, ShownView, details_ok, detail_refs, intent_met, reason_fields,
                                     understood)
from belong.schema import HIDDEN, SHOWN, STRICT_EXACTNESS, Field, Listing, Person, canonical, detail_reasons
from belong.sim import run
from belong.synth import draft_listing, generate
from tests.helpers import detail, warden


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


def synthetic_board(n=300, seed=7, dialects=0.0):
    w, board = warden(), Marketboard()
    personas = generate(n, seed, dialects)
    for p in personas:
        w.admit(p.person, draft_listing(p, "2026-10-12T09:00:00Z"), board)
    return board, {p.person.id: p for p in personas}


class HiddenFieldRules(unittest.TestCase):
    def test_hidden_fields_only_raise_scores_and_never_justify_a_match(self):
        board, _ = synthetic_board()
        for p in OfflineMatcher().match(board).proposals:
            self.assertGreaterEqual(p.score, p.shown_score)
            self.assertGreaterEqual(p.shown_score, p.shown_threshold)
            self.assertTrue(p.reason_fields)

    def test_removing_hidden_fields_never_removes_a_match(self):
        """If a pair is proposed, it would also qualify with every hidden field deleted: hidden open
        details too (the synthetic people hide some of theirs, and some agents hide one more)."""
        for dialects in (0.0, 0.25):
            board, _ = synthetic_board(dialects=dialects)
            self.assertTrue(any(f.visibility == HIDDEN and ":" in f.key for l in board.listings() for f in l.fields))
            shown_only = Marketboard()
            for l in board.listings():
                shown_only.put(board.person(l.person_id),
                               replace(l, fields=tuple(f for f in l.fields if f.visibility == SHOWN)))
            m = OfflineMatcher(max_per_person=10_000)
            full = {(p.person_a, p.person_b) for p in m.match(board).proposals}
            without = {(p.person_a, p.person_b) for p in m.match(shown_only).proposals}
            self.assertLessEqual(full, without, dialects)

    def test_crowd_threshold_blocks_rare_hidden_values(self):
        """A hidden key raises a score only when the pair share a value (compared canonically) that at
        least k OTHER people have: counted here independently, by different people on active listings."""
        board, _ = synthetic_board()
        people = {}
        for l in board.listings():
            for f in l.fields:
                for v in f.strings():
                    people.setdefault((f.key, canonical(v)), set()).add(l.owner)
        used_any = False
        for p in OfflineMatcher(k_crowd=20).match(board).proposals:
            a, b = board.listing(p.person_a), board.listing(p.person_b)
            for key in p.used_hidden_fields:
                used_any = True
                fa, fb = a.get(key), b.get(key)
                self.assertIn(HIDDEN, (fa.visibility, fb.visibility), key)
                self.assertFalse(key.startswith("them:"), key)  # a wish about the other person never counts
                shared = {canonical(v) for v in fa.strings()} & {canonical(v) for v in fb.strings()}
                others = [people[(key, s)] - {a.owner, b.owner} for s in shared]
                self.assertTrue(any(len(o) >= 20 for o in others), (key, shared))
        self.assertTrue(used_any)
        # With an impossible threshold, no hidden field is ever used.
        self.assertFalse(any(p.used_hidden_fields for p in OfflineMatcher(k_crowd=10_000).match(board).proposals))

    def test_the_crowd_is_k_other_people_not_listings(self):
        """Two people who share a hidden value it counts for only when k people besides the two of them
        have it. One person's several listings count once."""
        def board_with(others, listings_each=1):
            board = Marketboard()
            for i in range(2 + others):
                pid = f"q{i:02d}"
                for j in range(listings_each if i >= 2 else 1):
                    lid = pid if j == 0 else f"{pid}-{j}"
                    board.put(Person(pid, True, "25-29", "t"), Listing(lid, "x", 0.5, (
                        detail("night shifts", "every week", visibility=HIDDEN),), "t", owner_id=pid))
            return board
        m = OfflineMatcher(k_crowd=20)
        for others, each, used in ((20, 1, True), (19, 1, False), (4, 5, False)):
            board = board_with(others, each)
            got = m.hidden_overlap(board.listing("q00"), board.listing("q01"), board.value_counts())
            self.assertEqual(got, ("me:night shifts",) if used else (), (others, each))

    def test_a_wish_about_the_other_person_never_counts_as_hidden(self):
        """Only what someone is (me:, both:, the hidden built-ins) may raise a score, never what they'd like
        the other person to be, even on a board the Warden didn't check (the Warden refuses a hidden wish)."""
        board = Marketboard()
        for i in range(30):
            visibility = HIDDEN if i == 0 else SHOWN
            board.put(Person(f"q{i:02d}", True, "25-29", "t"), Listing(f"q{i:02d}", "x", 0.5, (
                detail("night shifts", "every week", about="them", visibility=visibility),
                detail("diet", "vegan", visibility=visibility)), "t"))
        got = OfflineMatcher(k_crowd=20).hidden_overlap(board.listing("q00"), board.listing("q01"), board.value_counts())
        self.assertEqual(got, ("me:diet",))

    def test_reasons_cite_exactly_the_details_in_common(self):
        """The open-detail reasons are exactly the ones the Warden works out from both people's shown
        details (schema.detail_reasons), never a subset or a hidden one."""
        board, _ = synthetic_board(dialects=0.25)
        props = OfflineMatcher().match(board).proposals
        self.assertTrue(any(":" in r for p in props for r in p.reason_fields))
        for p in props:
            a, b = board.listing(p.person_a), board.listing(p.person_b)
            refs = tuple(sorted(r for r in p.reason_fields if r.startswith(("agree:", "wants:"))))
            strict = max(a.exactness, b.exactness) >= STRICT_EXACTNESS
            self.assertEqual(refs, detail_reasons(a.shown_fields(), b.shown_fields(), strict), p)


class IntentReading(unittest.TestCase):
    def test_only_dating_is_read_from_free_text(self):
        """The description is read only to tell dating from the rest: what else a listing is for is in
        its phrases and details, never read from activity words."""
        cases = {
            "Looking for a serious relationship with a woman in Kyoto, aged 25 to 34.": "dating",
            "I'd like to meet men in Tokyo for something casual.": "dating",
            "A climbing partner in Osaka on weekends. I'm an intermediate climber.": None,
            "Native German speaker learning English, looking for a tandem partner.": None,
            "Need a CTO-type co-founder for my fintech idea (remote is fine).": None,
            "Need a physics tutor in Lisbon or online, weekends.": None,
            "Housemate wanted in Berlin, January move-in.": None,
            "Looking for a second violinist for our string quartet in Porto.": None,
            "Want someone to share a beehive allotment with in Leeds.": None,
        }
        for text, kind in cases.items():
            self.assertEqual(parse_intent(text).kind, kind, text)
        self.assertTrue(parse_intent("Need a CTO-type co-founder for my fintech idea (remote is fine).").remote_ok)

    def test_dating_details(self):
        p = parse_intent("I'd like to meet men in Tokyo for something casual. Ideally aged 30 to 39.")
        self.assertEqual(p.seeking, ("man",))
        self.assertEqual(p.age_range, (2, 3))

    def test_parser_agrees_with_truth_on_synthetic_people(self):
        for p in generate(300, 11):
            if not p.truth.bad_intent:
                # Everything but dating is recognised by its seeking/offering phrases, not the text, so
                # the text must not look like dating.
                expected = "dating" if p.truth.kind == "dating" else None
                self.assertEqual(parse_intent(p.intent_text).kind, expected, p.intent_text)

    def test_a_listing_with_phrases_and_no_kind_is_open_ended_whatever_its_words(self):
        """The phrases say what a listing is for: with no kind shown, words that read like dating don't
        make it dating. A kind it shows still decides."""
        def view(*extra, intent="Want to meet women for weekend kayaking trips."):
            return ShownView(Listing("p", intent, 0.5, (Field("city", "Osaka", SHOWN, False), *extra), "t"))
        self.assertEqual(understood(view()).kind, "dating")  # the words alone
        phrases = (Field("seeking", "a kayaking partner", SHOWN, False),
                   Field("offering", "a kayaking partner", SHOWN, False))
        self.assertEqual(understood(view(*phrases)).kind, "other")
        self.assertEqual(understood(view(*phrases, Field("kind", "dating", SHOWN, False))).kind, "dating")
        self.assertEqual(understood(view(*phrases, Field("kind", "dating", HIDDEN, False))).kind, "other")
        online = view(*phrases, intent="Want to meet women for kayaking, online planning is fine.")
        self.assertTrue(understood(online).remote_ok)


class OpenEnded(unittest.TestCase):
    """'Something else' matches: each person's 'seeking' must fit the other's 'offering'."""

    def _board(self, *people):
        w, board = warden(), Marketboard()
        for pid, intent, seeking, offering in people:
            listing = Listing(pid, intent, 0.6, (Field("city", "Tokyo", SHOWN, False),
                                                 Field("seeking", seeking, SHOWN, False),
                                                 Field("offering", offering, SHOWN, False),
                                                 detail("free time", "weekends")), "t")
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

    def test_two_people_who_are_the_same_and_want_the_other_side_are_not(self):
        """What someone looks for and isn't themselves must be what the other person is: two first
        violinists who both want a second violinist are never each other's answer, at any exactness."""
        for exactness in (0.3, 0.6, 0.9):
            w, board = warden(), Marketboard()
            for pid in ("ana", "ben"):
                listing = Listing(pid, "Looking for a second violinist for our quartet in Tokyo.", exactness, (
                    Field("city", "Tokyo", SHOWN, False), Field("seeking", "a second violinist", SHOWN, False),
                    Field("offering", "a first violinist", SHOWN, False)), "t")
                self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), listing, board).accepted)
            self.assertEqual(self.pairs(board), set(), exactness)

    def test_shared_role_words_alone_never_meet_a_request(self):
        """"A partner" in common isn't enough: a request that says what it is about needs that, even
        with a wide net."""
        from belong.matching.matcher import phrase_satisfies
        for exactness in (0.3, 0.6):
            self.assertFalse(phrase_satisfies("a kayaking partner", "a crossword partner", exactness))
            self.assertFalse(phrase_satisfies("a pottery teacher", "a fencing coach", exactness))
            self.assertTrue(phrase_satisfies("a kayaking partner", "a kayaking buddy", exactness))
        # What they show about themselves counts for a wide net: a kayaker who is a crossword partner.
        self.assertTrue(phrase_satisfies("a kayaking partner", "a crossword partner", 0.3, ["kayaking"]))

    def test_teacher_and_learner_match_anywhere_in_a_country(self):
        w, board = warden(), Marketboard()
        people = [
            ("aoife", "Looking for someone to teach me to fish in Ireland.", None, "a fishing teacher", "a beginner fishing student"),
            ("sean", "Happy to teach fishing to a beginner around Galway.", "Galway", "a fishing student", "a fishing mentor"),
            ("hans", "Happy to teach fishing to a beginner around Munich.", "Munich", "a fishing student", "a fishing mentor"),
        ]
        for pid, intent, city, seeking, offering in people:
            fields = [Field("country", "Germany" if pid == "hans" else "Ireland", SHOWN, False),
                      Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False),
                      detail("interests", "fishing")]
            if city:
                fields.append(Field("city", city, SHOWN, False))
            self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), Listing(pid, intent, 0.6, tuple(fields), "t"), board).accepted)
        props = OfflineMatcher().match(board).proposals
        self.assertEqual({frozenset((p.person_a, p.person_b)) for p in props}, {frozenset(("aoife", "sean"))})
        self.assertIn("country", props[0].reason_fields)
        self.assertIn("agree:interests", props[0].reason_fields)

    def test_a_listing_with_a_kind_and_one_without_are_compared_by_phrase(self):
        """Yuki's agent recorded the kind "other", Ren's recorded none. With a wide net, a chess partner
        who shows they're into languages can help someone who wants to learn French and chess."""
        def board(exactness):
            w, b = warden(), Marketboard()
            people = [("yuki", "Looking for a French tutor who can also teach me chess in Malaga.",
                       "a French and chess teacher", "a French and chess student", (Field("kind", "other", SHOWN, False),)),
                      ("ren", "Looking to meet new people in Malaga for chess games and practising languages.",
                       "a chess and language partner", "a chess and language partner",
                       (detail("interests", ("chess", "languages")),))]
            for pid, intent, seeking, offering, extra in people:
                fields = (Field("country", "Spain", SHOWN, False), Field("city", "Malaga", SHOWN, False),
                          Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False), *extra)
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
        """'A tutor or an exchange partner, either one' must match either kind of person. Which language
        each person speaks is an open detail now: a wish no phrase can stand in for."""
        from belong.matching.matcher import alternatives, phrase_satisfies
        yuki_seeks = "a German tutor or a German language exchange partner"
        yuki_is = "a German student or an English and Japanese language exchange partner"
        tutor = ("a German student", "a German tutor")
        exchange = ("a Japanese language exchange partner", "a German language exchange partner")
        for seeks, is_ in (tutor, exchange):
            self.assertTrue(phrase_satisfies(yuki_seeks, is_, 0.6) and phrase_satisfies(seeks, yuki_is, 0.6), is_)
        # Not a Spanish tutor.
        self.assertFalse(phrase_satisfies(yuki_seeks, "a Spanish tutor", 0.6))
        # Short options borrow what they leave out.
        self.assertEqual(alternatives("a German or Spanish tutor"), [["german", "teach"], ["spanish", "teach"]])

        # Not someone who wants Korean practice Yuki can't give: their wish is about her language.
        def exchange_listing(pid, speaks, wants):
            return Listing(pid, "Looking for a language exchange in Tokyo.", 0.6, (
                Field("city", "Tokyo", SHOWN, False), Field("seeking", "a language exchange partner", SHOWN, False),
                Field("offering", "a language exchange partner", SHOWN, False),
                detail("language", speaks), detail("language", wants, about="them")), "t")
        yuki = exchange_listing("yuki", ("English", "Japanese"), "German")
        korean = exchange_listing("kim", "German", "Korean")
        japanese = exchange_listing("aki", "German", "japanese")
        self.assertFalse(met(korean, yuki))
        self.assertTrue(met(yuki, korean))  # she'd be happy, but both must be looking
        self.assertTrue(met(japanese, yuki) and met(yuki, japanese))  # compared ignoring case

    def test_places_ignore_accents(self):
        from belong.matching.narrow import _place
        self.assertEqual(_place("Málaga"), _place("malaga "))

    def test_speaking_is_a_word_like_any_other(self):
        """No word of one kind of request is set aside in phrases: only words with no meaning of their own."""
        from belong.matching.matcher import phrase_tokens
        self.assertIn("speak", phrase_tokens("a Basque speaking hiking guide"))
        self.assertIn("speak", phrase_tokens("a public speaking coach"))


class PhrasesInAnyScript(unittest.TestCase):
    """Phrases are compared the way every value is (schema.canonical), in any script: case, width and accents on
    Latin letters never decide, and a phrase written in another script has words to match on."""

    def test_words_in_any_script(self):
        from belong.matching.matcher import phrase_tokens
        from belong.matching.text import letter_words, tokenize
        self.assertEqual(letter_words("Ｆａçade-Restorer"), ["facade", "restorer"])
        self.assertEqual(letter_words("बागवानी साथी"), ["बागवानी", "साथी"])  # the marks stay with their letters
        self.assertEqual(letter_words("ระดับ กลาง"), ["ระดับ", "กลาง"])
        self.assertEqual(letter_words("напарник для сплава, 2 дня"), ["напарник", "для", "сплава", "дня"])
        self.assertEqual(phrase_tokens("a façade restorer"), phrase_tokens("A FACADE RESTORER"))
        self.assertEqual(phrase_tokens("a façade restorer"), ["facade", "restor"])
        self.assertEqual(phrase_tokens("planners"), ["plan"])  # English endings, on Latin words
        self.assertEqual(phrase_tokens("Ｂｅｅｋｅｅｐｉｎｇ"), phrase_tokens("beekeeping"))
        self.assertEqual(tokenize("Crème brûlée lessons"), tokenize("creme brulee lessons"))
        self.assertTrue(tokenize("Ищу напарника для сплава"))

    def test_the_same_phrase_in_another_script_meets_itself(self):
        from belong.matching.matcher import index_words, phrase_satisfies
        for phrase in ("напарник для сплава", "बागवानी साथी", "ｂｅｅｋｅｅｐｉｎｇ ｐａｒｔｎｅｒ"):
            self.assertTrue(phrase_satisfies(phrase, phrase, 0.6), phrase)
        self.assertTrue(phrase_satisfies("a façade restorer", "a facade restorer", 0.6))
        self.assertTrue(phrase_satisfies("Ein Gärtner", "ein gartner", 0.6))
        self.assertFalse(phrase_satisfies("напарник для сплава", "напарник для шахмат", 0.6))
        # Narrowing indexes the same words, so the two always share a key.
        view = lambda phrase: ShownView(Listing("p", "x", 0.6, (Field("seeking", phrase, SHOWN, False),
                                                                Field("offering", phrase, SHOWN, False)), "t"))
        want, _, _ = index_words(view("напарник для сплава"))
        _, have, _ = index_words(view("напарник для сплава"))
        self.assertTrue(want & have)

    def test_two_people_writing_in_another_script_are_introduced(self):
        for phrase, city, country in (("напарник для сплава", "Kazan", "Russia"),
                                      ("बागवानी साथी", "Pune", "India")):
            w, board = warden(), Marketboard()
            for pid in ("pa", "pb"):
                l = Listing(pid, f"{phrase}, {city}", 0.6, (
                    Field("country", country, SHOWN, False), Field("city", city, SHOWN, False),
                    Field("kind", "other", SHOWN, False), Field("seeking", phrase, SHOWN, False),
                    Field("offering", phrase, SHOWN, False)), "t")
                result = w.admit(Person(pid, True, "25-29", "t"), l, board)
                self.assertTrue(result.accepted, result.problems)
            props = OfflineMatcher().match(board).proposals
            self.assertEqual([(p.person_a, p.person_b) for p in props], [("pa", "pb")], phrase)
            self.assertTrue(w.exit_introduction(props[0], board), phrase)

    def test_open_ended_people_in_the_population_match_correctly(self):
        result = run(n=300, seed=7)
        other = result["metrics"]["by_kind"]["other"]
        self.assertGreater(other["proposals"], 0)
        self.assertEqual(other["precision"], 1.0)


class SameHead(unittest.TestCase):
    """Two options with the same head (the main noun) are compared on what they are about: the head is the
    role, and what they are about is the rest of their content words. A head in common alone never meets an
    option that says what it is about, in any script and at any exactness; at the default exactness and
    above, one describing word in common never does when another differs. No word is named: the rule reads
    which word is the head, not what the words are."""

    def test_one_describing_word_in_common_is_not_enough(self):
        from belong.matching.matcher import phrase_satisfies
        for wanted, offered in (("an evening sketching partner", "an evening baking partner"),
                                ("a Hungarian translating volunteer", "a Finnish translating volunteer"),
                                ("a vintage moped mechanic", "a vintage tractor mechanic"),
                                ("a free futon giver", "a free kettle giver")):
            for exactness in (0.5, 0.6, 0.9):
                self.assertFalse(phrase_satisfies(wanted, offered, exactness), (wanted, offered, exactness))
                self.assertTrue(phrase_satisfies(wanted, wanted, exactness), wanted)
            # A wide net (under 0.5) accepts half of what an option is about: one word of two in common.
            self.assertTrue(phrase_satisfies(wanted, offered, 0.3), (wanted, offered))
        # All of what it is about, with more on the other side, still meets; less does not.
        self.assertTrue(phrase_satisfies("a sketching partner", "an evening sketching partner", 0.9))
        self.assertFalse(phrase_satisfies("an evening sketching partner", "a sketching partner", 0.6))
        self.assertTrue(phrase_satisfies("a dawn chorus birding partner", "a chorus birding partner", 0.6))

    def test_a_head_in_common_alone_never_meets_in_any_script(self):
        from belong.matching.matcher import phrase_satisfies
        for wanted, offered in (("a hockey goalkeeper", "a handball goalkeeper"),
                                ("a bridge player", "a snooker player"),
                                ("шаховий партнер", "тенісний партнер"),
                                ("সাঁতার সঙ্গী", "দাবা সঙ্গী"),
                                ("등산 친구", "요리 친구"),
                                ("写真 仲間", "料理 仲間"),
                                ("a hockey goalkeeper", "a goalkeeper")):
            for exactness in (0.2, 0.3, 0.45, 0.6, 0.9):
                self.assertFalse(phrase_satisfies(wanted, offered, exactness), (wanted, offered, exactness))
            self.assertTrue(phrase_satisfies(wanted, wanted, 0.9), wanted)
        # An option that is only its head is compared on the head, as before.
        for exactness in (0.3, 0.6, 0.9):
            self.assertTrue(phrase_satisfies("a goalkeeper", "a hockey goalkeeper", exactness))
            self.assertTrue(phrase_satisfies("친구", "등산 친구", exactness))
        # For a wide net, what the other shows about themselves counts towards what an option is about.
        self.assertTrue(phrase_satisfies("a bridge player", "a snooker player", 0.3, ["bridge"]))
        self.assertFalse(phrase_satisfies("a bridge player", "a snooker player", 0.6, ["bridge"]))

    def test_each_saying_something_the_other_does_not_never_meets_in_any_word_order(self):
        """At the default exactness and above, two options that each have a content word the other lacks are two
        different things, however many words they share and wherever their role noun is: first, as in many
        languages ("amico di ..."), or last. Only the same option, or one that narrows the other, meets."""
        from belong.matching.matcher import phrase_satisfies
        for wanted, offered in (("compañera de baile tranquila", "compañera de lectura tranquila"),
                                ("amico di pesca serale", "amico di corsa serale"),
                                ("parceira de xadrez online", "parceira de bridge online"),
                                ("a weekday dawn rowing partner", "a weekday dawn cycling partner")):
            for exactness in (0.5, 0.6, 0.9):
                self.assertFalse(phrase_satisfies(wanted, offered, exactness), (wanted, offered, exactness))
                self.assertFalse(phrase_satisfies(offered, wanted, exactness), (offered, wanted, exactness))
                self.assertTrue(phrase_satisfies(wanted, wanted, exactness), wanted)
        self.assertTrue(phrase_satisfies("a dawn rowing partner", "a weekday dawn rowing partner", 0.9))

    def test_end_to_end_only_the_same_thing_is_introduced(self):
        def introduced(phrase_a, phrase_b, exactness):
            w, board = warden(), Marketboard()
            for pid, phrase in (("pa", phrase_a), ("pb", phrase_b)):
                listing = Listing(pid, f"{phrase}, Lviv", exactness, (
                    Field("country", "Ukraine", SHOWN, False), Field("city", "Lviv", SHOWN, False),
                    Field("kind", "other", SHOWN, False), Field("seeking", phrase, SHOWN, False),
                    Field("offering", phrase, SHOWN, False)), "t")
                result = w.admit(Person(pid, True, "25-29", "t"), listing, board)
                self.assertTrue(result.accepted, result.problems)
            props = [p for p in OfflineMatcher().match(board).proposals if not p.close]
            return any(w.exit_introduction(p, board) for p in props)
        self.assertFalse(introduced("an evening sketching partner", "an evening baking partner", 0.6))
        self.assertFalse(introduced("шаховий партнер", "тенісний партнер", 0.3))
        self.assertFalse(introduced("a bridge player", "a snooker player", 0.3))
        self.assertTrue(introduced("шаховий партнер", "шаховий партнер", 0.6))
        self.assertTrue(introduced("an evening sketching partner", "an evening sketching partner", 0.6))
        self.assertFalse(introduced("amico di pesca serale", "amico di corsa serale", 0.6))
        self.assertTrue(introduced("amico di pesca serale", "amico di pesca serale", 0.6))


def met(a: Listing, b: Listing) -> bool:
    """Is a's intent met by b, as the Matchmaker reads it?"""
    va, vb = ShownView(a), ShownView(b)
    return intent_met(understood(va), va, understood(vb), vb)


def crew(pid, *details, exactness=0.6, intent="Looking for crew for weekend sailing in Osaka."):
    """A sailing crew listing in Osaka, with its open details."""
    return Listing(pid, intent, exactness, (
        Field("country", "Japan", SHOWN, False), Field("city", "Osaka", SHOWN, False),
        Field("seeking", "a sailing crew mate", SHOWN, False), Field("offering", "a sailing crew mate", SHOWN, False),
        *details), "2026-10-12T09:00:00Z")


SEA = ("none", "some", "plenty")  # a scale an agent wrote out for sailing experience


def sea(value, about="me", steps=SEA, required=True):
    return detail("sea time", value, about=about, compare="steps", steps=steps, required=required)


class UnstatedPreferences(unittest.TestCase):
    """Something you didn't say is a preference you don't have. A wish you did say, in your own words (TRUST.md #45:
    one your agent made a must that you didn't say is only "ideally"), is required unless you said "ideally"; an
    "ideally" the other person can't be compared with fails only a strict request."""

    def pairs(self, *listings):
        w, board = warden(), Marketboard()
        for l in listings:
            self.assertTrue(w.admit(Person(l.person_id, True, "25-29", "t"), l, board).accepted, l.person_id)
        return {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher().match(board).proposals}

    def test_a_wish_nobody_made_is_no_preference(self):
        self.assertIn(frozenset(("pa", "pb")), self.pairs(crew("pa"), crew("pb", sea("plenty"))))

    def test_an_ideally_with_nothing_to_compare_still_matches(self):
        b = crew("pb", sea("some", "them", required=False))
        self.assertIn(frozenset(("pa", "pb")), self.pairs(crew("pa"), b))

    def test_a_strict_request_needs_the_detail(self):
        b = crew("pb", sea("some", "them", required=False), exactness=0.9)
        self.assertNotIn(frozenset(("pa", "pb")), self.pairs(crew("pa"), b))
        self.assertIn(frozenset(("pa", "pb")), self.pairs(crew("pa", sea("some")), b))

    def test_a_required_wish_needs_the_detail_at_any_exactness(self):
        for exactness in (0.2, 0.6):
            b = crew("pb", sea("some", "them"), exactness=exactness)
            self.assertNotIn(frozenset(("pa", "pb")), self.pairs(crew("pa"), b), exactness)
            self.assertIn(frozenset(("pa", "pb")), self.pairs(crew("pa", sea("some")), b), exactness)

    def test_a_hidden_detail_is_as_if_it_werent_there(self):
        b = crew("pb", sea("some", "them"))
        hidden = crew("pa", detail("sea time", "some", compare="steps", steps=SEA, visibility=HIDDEN))
        self.assertNotIn(frozenset(("pa", "pb")), self.pairs(hidden, b))


class OpenDetails(unittest.TestCase):
    """Wishes (them, both) against the other person's own shown details (SPEC §5): by label, by
    distance on the identical scale, "any", and what the request itself is about."""

    def test_steps_by_label_across_different_scales(self):
        """A step is read by its label, whatever scale the other agent wrote; one that isn't on the
        wish's scale can't be compared, so a required wish isn't met."""
        wants_some = crew("pa", sea("some", "them"))
        longer = ("none", "a little", "some", "plenty", "years")
        self.assertTrue(met(wants_some, crew("pb", sea("some", steps=longer))))
        self.assertFalse(met(wants_some, crew("pb", sea("a little", steps=longer))))
        self.assertFalse(met(wants_some, crew("pb", sea("years", steps=longer))))

    def test_a_wide_net_looks_one_step_either_side_on_the_identical_scale_only(self):
        for exactness, ok in ((0.3, True), (0.6, False), (0.9, False)):
            self.assertEqual(met(crew("pa", sea("none", "them"), exactness=exactness), crew("pb", sea("some"))), ok,
                             exactness)
        two_away = crew("pb", sea("plenty"))
        self.assertFalse(met(crew("pa", sea("none", "them"), exactness=0.3), two_away))
        # On another scale only the label counts, never a distance.
        other_scale = ("none", "some", "lots")
        self.assertFalse(met(crew("pa", sea("plenty", "them", steps=SEA), exactness=0.3),
                             crew("pb", sea("some", steps=other_scale))))

    def test_any_meets_any_value(self):
        boat = lambda pid, value, about="me": crew(pid, detail("boat", value, about=about))
        self.assertTrue(met(boat("pa", "dinghy", "them"), boat("pb", "any")))
        self.assertTrue(met(boat("pa", "any", "them"), boat("pb", "keelboat")))
        self.assertFalse(met(boat("pa", "dinghy", "them"), boat("pb", "keelboat")))
        self.assertTrue(met(boat("pa", ("dinghy", "Keelboat"), "them"), boat("pb", "keelboat")))

    def test_what_the_request_is_about_must_be_shared(self):
        """A "both" detail is about the request itself: both people's must agree, and someone who
        doesn't show it can't meet a required one."""
        race = lambda pid, *value: crew(pid, *(detail("race", v, about="both") for v in value))
        self.assertTrue(met(race("pa", "autumn regatta"), race("pb", "Autumn Regatta")))
        self.assertFalse(met(race("pa", "autumn regatta"), race("pb", "spring series")))
        self.assertFalse(met(race("pa", "autumn regatta"), race("pb")))
        self.assertTrue(met(race("pb"), race("pa", "autumn regatta")))  # no wish of their own

    def test_details_decide_in_dating_too(self):
        def date(pid, gender, wants, *extra):
            return Listing(pid, "Looking for a relationship in Osaka, aged 25 to 39.", 0.6, (
                Field("city", "Osaka", SHOWN, False), Field("country", "Japan", SHOWN, False),
                Field("age_band", "30-34", SHOWN, False), Field("gender", gender, SHOWN, False),
                Field("kind", "dating", SHOWN, False), Field("wants_gender", (wants,), SHOWN, False), *extra), "t")
        her = date("her", "woman", "man", detail("smoking", "doesn't smoke", about="them"))
        self.assertTrue(met(her, date("him", "man", "woman", detail("smoking", "doesn't smoke"))))
        self.assertFalse(met(her, date("him", "man", "woman", detail("smoking", "smokes"))))
        self.assertFalse(met(her, date("him", "man", "woman")))

    def test_hidden_details_never_change_a_decision_a_reason_or_a_score(self):
        """Matching reads shown details only: a hidden detail, even a wish the Warden would never let in,
        is as if it weren't there."""
        a = crew("pa", sea("some", "them"), sea("some"), detail("boat", "dinghy"))
        b = crew("pb", sea("some"), sea("some", "them"), detail("boat", "dinghy"))
        hidden = (detail("boat", "keelboat", about="them", visibility=HIDDEN), detail("crew size", "four", visibility=HIDDEN),
                  detail("race", "spring series", about="both", visibility=HIDDEN))
        a2, b2 = replace(a, fields=a.fields + hidden), replace(b, fields=b.fields + hidden[1:])
        for x, y in ((a, b), (b, a)):
            self.assertTrue(met(x, y))
        for x, y in ((a2, b2), (b2, a2)):
            self.assertTrue(met(x, y))
        va, vb, va2, vb2 = map(ShownView, (a, b, a2, b2))
        self.assertEqual(detail_refs(va, vb), detail_refs(va2, vb2))
        self.assertEqual(reason_fields("other", va, vb), reason_fields("other", va2, vb2))
        self.assertTrue(details_ok(va2, vb2) and details_ok(vb2, va2))
        self.assertEqual(detail_refs(va, vb), ("agree:boat", "agree:sea time", "wants:sea time"))

    def test_a_wish_that_says_nothing_never_asks_for_the_detail(self):
        """A wish for any value, or for every step of its scale, is always met, even required and strict: it
        never asks the other person to show that detail (it isn't "open to any" if it turns people away)."""
        for wish in (detail("mooring", "any", about="them"), sea(SEA, "them")):
            for exactness in (0.3, 0.6, STRICT_EXACTNESS):
                self.assertTrue(met(crew("pa", wish, exactness=exactness), crew("pb")), (wish.key, exactness))
                self.assertTrue(met(crew("pa", wish, exactness=exactness), crew("pb", sea("none"))), wish.key)
        # One step short of every step still says something: it is a wish like any other.
        self.assertFalse(met(crew("pa", sea(SEA[1:], "them")), crew("pb")))
        self.assertFalse(met(crew("pa", sea(SEA[1:], "them")), crew("pb", sea("none"))))

    def test_a_strict_request_never_relies_on_an_own_any(self):
        """An own "any" says nothing about the person: it meets a wish of a request that isn't strict (any
        value suits them), but a strict request needs a value it can compare, required or "ideally"."""
        def mooring(pid, value, about="me", required=True, exactness=0.6):
            return crew(pid, detail("mooring", value, about=about, required=required), exactness=exactness)
        for required in (True, False):
            for exactness, ok in ((0.3, True), (0.6, True), (STRICT_EXACTNESS, False), (0.95, False)):
                self.assertEqual(met(mooring("pa", "harbour", "them", required, exactness), mooring("pb", "any")), ok,
                                 (required, exactness))
            # A value it can compare still meets it, however strict.
            self.assertTrue(met(mooring("pa", "harbour", "them", required, 0.95), mooring("pb", "Harbour")))
        # The same through the Warden and the Matchmaker: no proposal for the strict request, and for the
        # one that isn't strict a proposal that never cites the "any" as something they have in common.
        for exactness, proposed in ((0.6, True), (0.9, False)):
            w, board = warden(), Marketboard()
            for pid, l in (("pa", mooring("pa", "harbour", "them", True, exactness)), ("pb", mooring("pb", "any"))):
                self.assertTrue(w.admit(Person(pid, True, "25-29", "t"), l, board).accepted, pid)
            props = OfflineMatcher().match(board).proposals
            self.assertEqual(bool(props), proposed, exactness)
            for p in props:
                self.assertFalse([r for r in p.reason_fields if "mooring" in r], p.reason_fields)
                self.assertTrue(w.exit_introduction(p, board))



def repair(pid, seeking, offering, *details, close=False, questions=False, exactness=0.6):
    return replace(Listing(pid, "Looking for help with repairs in Leeds.", exactness, (
        Field("country", "United Kingdom", SHOWN, False), Field("city", "Leeds", SHOWN, False),
        Field("seeking", seeking, SHOWN, False), Field("offering", offering, SHOWN, False), *details),
        "2026-10-12T09:00:00Z"), notify_close=close, notify_questions=questions)


class NearlyKeepsEveryWish(unittest.TestCase):
    """Close matches and "Would that work for you?" questions look wider than an exact match, but never
    past a wish someone shows: a required one is never ignored, and a question is asked only if a yes
    would make a match both ways (a yes widens only the asked person's phrases)."""

    def run_matcher(self, *listings):
        w, board = warden(), Marketboard()
        for l in listings:
            self.assertTrue(w.admit(Person(l.person_id, True, "25-29", "t"), l, board).accepted, l.person_id)
        return OfflineMatcher().match(board)

    def customer(self, visits="evenings", **kw):
        return repair("pc", "a sewing machine repair technician", "a sewing machine repair customer",
                      detail("visit time", visits, about="them"), close=True, **kw)

    def repairman(self, visits, **kw):
        return repair("pr", "a repair customer", "a repairman", detail("visit time", visits), **kw)

    def test_a_question_only_if_a_yes_could_make_a_match(self):
        for visits, asked in (("evenings", True), ("mornings", False)):
            result = self.run_matcher(self.customer(questions=True), self.repairman(visits, questions=True))
            self.assertEqual(result.proposals, [], visits)
            questions = [(f.person_id, f.about) for f in result.followups if f.question_id == "also_offer"]
            self.assertEqual(questions, [("pr", "pc")] if asked else [], visits)

    def test_a_close_match_from_a_related_request_keeps_every_wish(self):
        for visits, close in (("evenings", True), ("mornings", False)):
            result = self.run_matcher(self.customer(), self.repairman(visits))  # the repairman takes no questions
            self.assertEqual([(p.person_a, p.person_b, p.close) for p in result.proposals],
                             [("pc", "pr", "pc")] if close else [], visits)

    def test_a_close_match_relaxes_one_step_and_never_a_required_wish(self):
        strict = replace(crew("pa", sea("plenty", "them"), exactness=0.9), notify_close=True)
        # Theirs exactly as wished: a match. One step away on the same scale: a close match. Two steps, or
        # nothing to compare: neither.
        for theirs, got in (("plenty", [""]), ("some", ["pa"]), ("none", []), (None, [])):
            other = crew("pb", *([sea(theirs)] if theirs else []))
            self.assertEqual([p.close for p in self.run_matcher(strict, other).proposals], got, theirs)


if __name__ == "__main__":
    unittest.main()
