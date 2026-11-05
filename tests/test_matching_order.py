"""The matching order (SPEC §5, §8.2): how two people can meet first, then where (travel distance),
then what (a word index), with the old rules unchanged after that, and fair ranking at the end."""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from belong.evaluate import can_meet, compare_narrowing
from belong.leaks import leak_report
from belong.matching.contract import Proposal
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import (OfflineMatcher, ShownView, _minutes, index_words, parse_intent,
                                     recently_updated, understood)
from belong.matching.narrow import meeting_modes, shortlists, within_reach
from belong.schema import HIDDEN, km_between, meeting_format, travel_choice
from belong.synth import draft_listing, generate
from belong.warden import reason_text
from tests.helpers import detail, fld, listing, person, warden

SAME_NAME = "Cite two cities with the same name in different countries to make them sound close"
HIDDEN_COUNTRY = "Word where two people are from a country one of them hides"


def board_of(*listings):
    w, board = warden(), Marketboard()
    for l in listings:
        result = w.admit(person(l.person_id), l, board, history=True)  # each fixture's own time, as seeded history
        assert result.accepted, (l.person_id, result.problems)
    return w, board


def web(pid, role, city, country, fmt, intent=None):
    """A web designer or one of their clients."""
    seeking, offering = ("a web design client", "a web designer") if role == "designer" else \
        ("a web designer", "a web design client")
    headline = "Offering web design" if role == "designer" else "Looking for a web designer"
    fields = [fld("first_name", pid.title()), fld("city", city), fld("country", country), detail("free time", "weekends"),
              fld("seeking", seeking), fld("offering", offering), fld("headline", headline),
              fld("topics", ("web design", "websites"))]
    if fmt:
        fields.append(fld("format", fmt))
    text = intent or ("I'm a web designer looking for new clients." if role == "designer" else "Need a web designer.")
    return listing(pid, text, fields)


def roommate(pid, city, travel=None, country="Japan"):
    """Someone looking for a roommate, moving in November (what the request itself is about: "both")."""
    fields = [fld("first_name", pid.title()), fld("city", city), fld("country", country), fld("kind", "other"),
              fld("seeking", "a roommate"), fld("offering", "a roommate"), detail("move in month", "November", "both"),
              detail("free time", "weekends")]
    if travel:
        fields.append(fld("travel", travel))
    return listing(pid, f"Looking for a roommate in {city}, moving in November.", fields)


def pairs(board, **kw):
    return {frozenset((p.person_a, p.person_b)) for p in OfflineMatcher(**kw).match(board).proposals}


class HowTheyMeet(unittest.TestCase):
    """Step 1: online only if both allow it, and then place doesn't matter."""

    def test_meeting_online_skips_place(self):
        w, board = board_of(web("lena", "designer", "London", "United Kingdom", "Online"),
                            web("kenji", "client", "Tokyo", "Japan", "In person or online"))
        props = OfflineMatcher().match(board).proposals
        self.assertEqual({frozenset((p.person_a, p.person_b)) for p in props}, {frozenset(("lena", "kenji"))})
        self.assertIn("format", props[0].reason_fields)  # they aren't near each other: how they meet is the reason
        intro = w.exit_introduction(props[0], board)
        self.assertIn("You can both meet online.", intro.a_view["reason"])
        self.assertNotIn("both in", intro.a_view["reason"])

    def test_online_only_and_in_person_only_never_meet(self):
        for fmts in (("Online", "In person"), ("In person", "Online")):
            _, board = board_of(web("lena", "designer", "Tokyo", "Japan", fmts[0]),
                                web("kenji", "client", "Tokyo", "Japan", fmts[1]))
            self.assertEqual(pairs(board), set(), fmts)
            result = OfflineMatcher().match(board)
            self.assertFalse([f for f in result.followups if f.question_id == "also_offer"], fmts)

    def test_both_in_person_in_the_same_city_meet(self):
        _, board = board_of(web("lena", "designer", "Tokyo", "Japan", "In person"),
                            web("kenji", "client", "Tokyo", "Japan", "In person or online"))
        self.assertEqual(pairs(board), {frozenset(("lena", "kenji"))})

    def test_any_wording_of_the_format(self):
        self.assertEqual(meeting_format("online only"), "Online")
        self.assertEqual(meeting_format("In-person"), "In person")
        self.assertEqual(meeting_format("either is fine"), "In person or online")
        self.assertEqual(meeting_format("Remote or face to face"), "In person or online")
        self.assertIsNone(meeting_format("weekends"))

    def test_a_negated_online_or_only_in_person_means_in_person(self):
        for value in ("not online", "No remote", "in person, not online", "without video calls",
                      "online isn't possible", "only in person", "In person only, online if needed"):
            self.assertEqual(meeting_format(value), "In person", value)
        self.assertEqual(meeting_format("not in person"), "Online")
        self.assertEqual(meeting_format("not only online"), "In person or online")
        # With no format shown, a description that says "not online" means in person too.
        self.assertFalse(parse_intent("Need a web designer, no remote work.").remote_ok)
        self.assertTrue(parse_intent("Need a web designer, remote is fine.").remote_ok)
        not_online = web("kenji", "client", "Tokyo", "Japan", None, intent="Need a web designer in Tokyo, not online.")
        self.assertEqual(pairs(board_of(web("lena", "designer", "London", "United Kingdom", "Online"), not_online)[1]),
                         set())

    def test_the_warden_accepts_only_the_fixed_formats(self):
        for fmt in ("In person", "Online", "In person or online"):
            result = warden().admit(person("lena"), web("lena", "designer", "Tokyo", "Japan", fmt), Marketboard())
            self.assertTrue(result.accepted, (fmt, result.problems))
        for fmt in ("online only", "not online", "Zoom calls"):
            result = warden().admit(person("lena"), web("lena", "designer", "Tokyo", "Japan", fmt), Marketboard())
            self.assertFalse(result.accepted, fmt)
            self.assertIn("listing_schema", result.failed_checks)
            self.assertTrue(any("'format' must be one of" in p for p in result.problems), result.problems)

    def test_without_a_format_the_description_decides(self):
        online_text = web("kenji", "client", "Tokyo", "Japan", None, intent="Need a web designer, online is fine.")
        in_person_text = web("kenji", "client", "Tokyo", "Japan", None, intent="Need a web designer in Tokyo.")
        self.assertEqual(pairs(board_of(web("lena", "designer", "London", "United Kingdom", "Online"), online_text)[1]),
                         {frozenset(("lena", "kenji"))})
        self.assertEqual(pairs(board_of(web("lena", "designer", "London", "United Kingdom", "Online"),
                                        in_person_text)[1]), set())

    def test_any_kind_but_dating_may_meet_online(self):
        def reader(pid, city, country):
            return listing(pid, "Book club buddy for a virtual reading circle, weekend calls online.", [
                fld("city", city), fld("country", country), fld("seeking", "a book club partner"),
                fld("offering", "a book club partner"), detail("free time", "weekends"), fld("format", "Online")])
        self.assertEqual(pairs(board_of(reader("ana", "Berlin", "Germany"), reader("tom", "London", "United Kingdom"))[1]),
                         {frozenset(("ana", "tom"))})
        dating = listing("x", "Looking for a relationship with a woman, aged 25 to 34.", [fld("format", "Online")])
        self.assertEqual(meeting_modes(parse_intent(dating.intent_text), ShownView(dating)), {"in person"})


class TravelDistance(unittest.TestCase):
    """Step 2: in person, two cities are in reach within *both* people's travel distance."""

    def test_both_limits_must_reach(self):
        self.assertAlmostEqual(km_between("Osaka", "Japan", "Kobe", "Japan"), 28, delta=2)
        for a, b, met in (("Up to 30 km", "Up to 30 km", True), ("Up to 30 km", None, False),
                          (None, "Up to 30 km", False), ("Up to 100 km", "Same city", False),
                          ("Same city", "Up to 100 km", False), ("Up to 100 km", "Up to 30 km", True)):
            got = pairs(board_of(roommate("pa", "Osaka", a), roommate("pb", "Kobe", b))[1])
            self.assertEqual(got == {frozenset(("pa", "pb"))}, met, (a, b))

    def test_a_longer_trip_needs_the_longer_choice(self):
        self.assertAlmostEqual(km_between("Osaka", "Japan", "Kyoto", "Japan"), 43, delta=2)
        self.assertEqual(pairs(board_of(roommate("pa", "Osaka", "Up to 30 km"), roommate("pb", "Kyoto", "Up to 30 km"))[1]),
                         set())
        self.assertEqual(pairs(board_of(roommate("pa", "Osaka", "Up to 100 km"), roommate("pb", "Kyoto", "Up to 100 km"))[1]),
                         {frozenset(("pa", "pb"))})

    def test_an_unknown_city_only_matches_itself(self):
        self.assertIsNone(km_between("Galway", "Ireland", "Athenry", "Ireland"))
        far = (roommate("pa", "Galway", "Up to 100 km", "Ireland"), roommate("pb", "Athenry", "Up to 100 km", "Ireland"))
        self.assertEqual(pairs(board_of(*far)[1]), set())
        same = (roommate("pa", "Galway", None, "Ireland"), roommate("pb", "galway ", None, "Ireland"))
        self.assertEqual(pairs(board_of(*same)[1]), {frozenset(("pa", "pb"))})

    def test_same_name_cities_in_different_countries_are_not_in_reach(self):
        """A city's centre is read in the country its listing shows: Cambridge, United States is an ocean
        away from London, and a Cambridge shown without a country (or with a hidden one) is only ever
        "the same city" as another Cambridge."""
        self.assertAlmostEqual(km_between("Cambridge", "United Kingdom", "London", "United Kingdom"), 79, delta=3)
        self.assertGreater(km_between("Cambridge", "United States", "London", "United Kingdom"), 5000)
        self.assertLess(km_between("Cambridge", "United States", "Boston", "United States"), 10)
        self.assertIsNone(km_between("Cambridge", None, "London", "United Kingdom"))
        self.assertIsNone(km_between("Boston", "United Kingdom", "London", "United Kingdom"))  # not known there
        far = "Up to 100 km"

        def uk(pid, city):
            return roommate(pid, city, far, "United Kingdom")

        def us(pid, city):
            return roommate(pid, city, far, "United States")

        def no_country(pid, city, hidden=None):
            l = roommate(pid, city, far)
            fields = tuple(f for f in l.fields if f.key != "country")
            return replace(l, fields=fields + ((fld("country", hidden, HIDDEN),) if hidden else ()))

        cases = [
            (uk("pa", "Cambridge"), uk("pb", "London"), True),  # about 80 km apart
            (us("pa", "Cambridge"), us("pb", "Boston"), True),
            (us("pa", "Cambridge"), uk("pb", "London"), False),  # an ocean apart
            (us("pa", "Cambridge"), uk("pb", "Cambridge"), False),  # the same name in other countries
            (no_country("pa", "Cambridge"), uk("pb", "London"), False),  # no shown country, no known centre
            (no_country("pa", "Cambridge", hidden="United Kingdom"), uk("pb", "London"), False),  # never a hidden one
            (no_country("pa", "Cambridge"), uk("pb", "Cambridge"), True),  # the same name is still the same city
        ]
        for a, b, met in cases:
            where = (a.shown().get("city"), a.shown().get("country"), b.shown().get("city"), b.shown().get("country"))
            self.assertEqual(within_reach(ShownView(a), ShownView(b)) == "city", met, where)
            self.assertEqual(pairs(board_of(a, b)[1]) == {frozenset(("pa", "pb"))}, met, where)
        # Narrowing looks only at cities in reach in the shown country.
        _, board = board_of(us("cam", "Cambridge"), uk("lon", "London"), us("bos", "Boston"), uk("camuk", "Cambridge"))
        views = {l.person_id: ShownView(l) for l in board.listings()}
        parsed = {pid: understood(v) for pid, v in views.items()}
        lists, _ = shortlists(parsed, views, 10, focus={"cam", "camuk"}, words=index_words)
        self.assertEqual((set(lists["cam"]), set(lists["camuk"])), ({"bos"}, {"lon"}))

    def test_no_city_is_still_open_to_anywhere_in_the_country(self):
        anywhere = listing("pa", "Looking for a roommate anywhere in Japan, moving in November.",
                           [fld("country", "Japan"), fld("seeking", "a roommate"), fld("offering", "a roommate"),
                            detail("move in month", "November", "both"), detail("free time", "weekends")])
        self.assertEqual(pairs(board_of(anywhere, roommate("pb", "Kyoto"))[1]), {frozenset(("pa", "pb"))})
        self.assertEqual(pairs(board_of(anywhere, roommate("pb", "Berlin", country="Germany"))[1]), set())

    def test_a_hidden_travel_distance_is_never_used(self):
        """Only shown fields decide: someone who hides how far they travel stays in their city."""
        hidden = roommate("pb", "Kobe", None)
        hidden = replace(hidden, fields=hidden.fields + (fld("travel", "Up to 30 km", HIDDEN),))
        self.assertEqual(pairs(board_of(roommate("pa", "Osaka", "Up to 30 km"), hidden)[1]), set())

    def test_narrowing_only_looks_at_cities_in_reach(self):
        _, board = board_of(roommate("osaka", "Osaka", "Up to 30 km"), roommate("kobe", "Kobe", "Up to 30 km"),
                            roommate("kyoto", "Kyoto", "Up to 100 km"), roommate("tokyo", "Tokyo", "Up to 100 km"))
        views = {l.person_id: ShownView(l) for l in board.listings()}
        parsed = {pid: understood(v) for pid, v in views.items()}
        lists, _ = shortlists(parsed, views, 10, focus={"osaka"}, words=index_words)
        self.assertEqual(set(lists["osaka"]), {"kobe"})

    def test_the_warden_accepts_only_the_fixed_travel_choices(self):
        bad = roommate("pa", "Osaka", "Up to 50 km")
        result = warden().admit(person("pa"), bad, Marketboard())
        self.assertFalse(result.accepted)
        self.assertIn("listing_schema", result.failed_checks)
        self.assertEqual(travel_choice("up to 30km"), "Up to 30 km")
        self.assertIsNone(travel_choice("Up to 50 km"))

    def test_ground_truth_uses_the_same_rule(self):
        from belong.synth import Truth
        osaka = Truth(kind="roommate", city="Osaka", exactness=0.5, schedule="weekends", travel="Up to 100 km")
        kyoto = Truth(kind="roommate", city="Kyoto", exactness=0.5, schedule="weekends", travel="Up to 100 km")
        self.assertTrue(can_meet(osaka, kyoto))
        self.assertFalse(can_meet(osaka, replace(kyoto, travel="Up to 30 km")))
        online = Truth(kind="service", city="London", exactness=0.5, schedule="weekends", remote_ok=True, in_person=False)
        self.assertTrue(can_meet(online, replace(osaka, kind="service", remote_ok=True)))
        self.assertFalse(can_meet(online, replace(osaka, kind="service")))


class WhereInTheReason(unittest.TestCase):
    """The Warden words where people are from both their shown cities, so it is always true."""

    def test_reason_text_for_nearby_and_same_cities(self):
        japan = ("Japan", "Japan")
        self.assertEqual(reason_text([("city", "Osaka", "Kyoto")], japan), "You're near one another: Osaka and Kyoto.")
        self.assertEqual(reason_text([("city", "Osaka", "osaka")], japan), "You're both in osaka.")
        self.assertEqual(reason_text([("city", "Lisbon", "Lisboa")], ("Portugal", "portugal")), "You're both in Lisboa.")
        self.assertEqual(reason_text([("country", "Japan", "Japan")]), "You're both in Japan.")

    def test_a_compromised_matchmaker_cannot_make_far_cities_near(self):
        self.assertEqual(reason_text([("city", "Osaka", "London")]), "You're in Osaka; they're in London.")
        self.assertEqual(reason_text([("city", "Galway", "Athenry")]), "You're in Galway; they're in Athenry.")
        self.assertEqual(reason_text([("country", "Japan", "Germany")]), "You're in Japan; they're in Germany.")
        self.assertEqual(reason_text([("format", "In person", "Online")]), "You'd meet in person; they'd meet online.")

    def test_the_attack_suite_catches_a_reason_that_isnt_true(self):
        """Test the tester: with the old wording ("You're both in ..." whatever the cities), the attack
        that cites far-apart cities gets through."""
        from belong import probe
        name = "Cite where two people are to make far-apart cities sound close"
        result = lambda: next(x for x in probe.attack_suite({"seed": 1})["results"] if x["attack"] == name)
        self.assertTrue(result()["blocked"])
        with mock.patch("belong.warden._where_text", lambda key, mine, theirs, countries=None: f"You're both in {theirs}."):
            self.assertFalse(result()["blocked"])

    def test_same_name_cities_are_worded_by_the_shown_countries(self):
        """"Both in" and "near one another" only when both people show the same country."""
        us_uk, uk, us = ("United States", "United Kingdom"), ("United Kingdom",) * 2, ("United States",) * 2
        self.assertEqual(reason_text([("city", "Cambridge", "Cambridge")], us_uk), "You're in Cambridge; they're in Cambridge.")
        self.assertEqual(reason_text([("city", "Cambridge", "London")], us_uk), "You're in Cambridge; they're in London.")
        self.assertEqual(reason_text([("city", "Cambridge", "London")], uk), "You're near one another: Cambridge and London.")
        self.assertEqual(reason_text([("city", "Cambridge", "Boston")], us), "You're near one another: Cambridge and Boston.")
        self.assertEqual(reason_text([("city", "Boston", "London")], uk), "You're in Boston; they're in London.")
        # Without both countries, nothing is claimed.
        self.assertEqual(reason_text([("city", "Cambridge", "London")]), "You're in Cambridge; they're in London.")
        self.assertEqual(reason_text([("city", "Osaka", "Osaka")]), "You're in Osaka; they're in Osaka.")

    def test_the_warden_words_cities_only_from_countries_both_people_show(self):
        """A compromised Matchmaker cites where two people are. The wording leans on neither a city's
        name alone nor a country someone hides."""
        def placed(pid, city, country, visibility=None):
            return listing(pid, "Looking for a roommate, moving in November.", [
                fld("first_name", pid.title()), fld("city", city), fld("country", country, visibility),
                fld("seeking", "a roommate"), fld("offering", "a roommate")])
        cases = [
            (placed("pa", "Cambridge", "United Kingdom", HIDDEN), placed("pb", "London", "United Kingdom"),
             "You're in Cambridge; they're in London.", "You're in London; they're in Cambridge."),
            (placed("pa", "Cambridge", "United States"), placed("pb", "Cambridge", "United Kingdom"),
             "You're in Cambridge; they're in Cambridge.", "You're in Cambridge; they're in Cambridge."),
            (placed("pa", "Cambridge", "United Kingdom"), placed("pb", "London", "United Kingdom"),
             "You're near one another: Cambridge and London.", "You're near one another: London and Cambridge."),
        ]
        for a, b, a_reads, b_reads in cases:
            w, board = board_of(a, b)
            intro = w.exit_introduction(Proposal("pa", "pb", True, True, 1.0, 1.0, 0.1, ("city",), ()), board)
            self.assertIsNotNone(intro)
            self.assertEqual((intro.a_view["reason"], intro.b_view["reason"]), (a_reads, b_reads))

    def test_the_attack_suite_catches_same_name_cities_and_a_hidden_country(self):
        """Test the tester: a Warden that words two cities as if both people were in one country (by
        their names alone, or from a country one of them hides) lets both attacks through."""
        from belong import probe
        from belong.warden import _where_text

        def blocked():
            got = {x["attack"]: x["blocked"] for x in probe.attack_suite({"seed": 1})["results"]}
            return got[SAME_NAME], got[HIDDEN_COUNTRY]

        self.assertEqual(blocked(), (True, True))
        one_country = lambda key, mine, theirs, countries=None: _where_text(key, mine, theirs, ("United Kingdom",) * 2)
        with mock.patch("belong.warden._where_text", one_country):
            self.assertEqual(blocked(), (False, False))

    def test_an_introduction_between_nearby_cities_passes_every_check(self):
        w, board = board_of(replace(roommate("pa", "Osaka", "Up to 100 km"),
                                    fields=roommate("pa", "Osaka", "Up to 100 km").fields + (fld("religion", "Buddhist", HIDDEN),)),
                            roommate("pb", "Kyoto", "Up to 100 km"))
        props = OfflineMatcher().match(board).proposals
        self.assertEqual(len(props), 1)
        self.assertIn("city", props[0].reason_fields)
        start = len(w.receipts)
        intro = w.exit_introduction(props[0], board)
        self.assertIsNotNone(intro)
        self.assertEqual([r.check for r in w.receipts[start:] if r.result == "fail"], [])
        views = {intro.person_a: intro.a_view["reason"], intro.person_b: intro.b_view["reason"]}
        self.assertIn("You're near one another: Osaka and Kyoto.", views["pa"])
        self.assertIn("You're near one another: Kyoto and Osaka.", views["pb"])
        report = leak_report(board, [intro.to_dict()], [], [], [r.to_dict() for r in w.receipts], props, set(), {},
                             20, 15)
        self.assertTrue(report["ok"], report["findings"])


class WordIndex(unittest.TestCase):
    """Step 3: phrase listings are compared only when they share a word (or its first four
    letters), and that never loses a pair a decision could accept."""

    def test_the_index_keeps_every_pair_a_decision_could_accept(self):
        for n, seed in ((600, 7), (400, 3)):
            r = compare_narrowing(generate(n, seed))
            self.assertTrue(r["subset"], (n, seed))
            self.assertEqual(r["lost"], [], (n, seed))
            self.assertGreater(r["acceptable_pairs_kept"], 0)
            self.assertLess(r["phrase_candidates_after"], r["phrase_candidates_before"] / 2, (n, seed))
            self.assertEqual(r["correct_found_only_before"], 0, (n, seed))

    def _shortlisted(self, a, b, words=index_words):
        _, board = board_of(a, b)
        views = {l.person_id: ShownView(l) for l in board.listings()}
        parsed = {pid: understood(v) for pid, v in views.items()}
        lists, _ = shortlists(parsed, views, 10, words=words)
        return b.person_id in lists[a.person_id] and a.person_id in lists[b.person_id]

    def _phrases(self, pid, seeking, offering, topics=(), interests=(), exactness=0.5):
        fields = [fld("city", "Tokyo"), fld("country", "Japan"), fld("seeking", seeking), fld("offering", offering)]
        if topics:
            fields.append(fld("topics", tuple(topics)))
        if interests:
            fields.append(detail("interests", tuple(interests)))
        # The description names no kind ("partner" would read as dating), so the phrases decide.
        return listing(pid, "Hoping to meet people in Tokyo.", fields, exactness=exactness)

    def test_related_requests_still_meet(self):
        cases = [
            # a shared stem (nearly(): "repair" / "repairman")
            (("an air conditioner repair technician", "an air conditioner repair customer"), ("a repair customer", "a repairman")),
            # only the topics connect them
            (("a beginner ice hockey coach", "an ice hockey student", ("ice skating", "hockey")),
             ("a figure skating student", "a figure skating coach", ("ice skating", "figure skating"))),
            # what one shows about themselves (their interests) meets what the other looks for, with a wide net
            (("a French and chess teacher", "a French and chess student", (), (), 0.3),
             ("a social hangout partner", "a social hangout partner", (), ("chess", "languages"), 0.3)),
            # with a wide net, a partner who shows they kayak is a kayaking partner
            (("a kayaking partner", "a kayaking partner", (), (), 0.3),
             ("a crossword partner", "a crossword partner", (), ("kayaking",), 0.3)),
        ]
        from belong.matching.matcher import nearly, phrases_ok
        for a, b in cases:
            la, lb = self._phrases("pa", *a), self._phrases("pb", *b)
            va, vb = ShownView(la), ShownView(lb)
            # The decisions would accept something here (so the pair must stay a candidate)...
            self.assertTrue(phrases_ok(va, vb) or nearly(va.get("seeking"), vb.get("offering"), va.get("topics"),
                                                         vb.get("topics"), va.get("offering")), (a, b))
            # ...and narrowing keeps it.
            self.assertTrue(self._shortlisted(la, lb), (a, b))

    def test_shared_role_words_alone_are_no_reason_to_compare(self):
        """A role word in common ("partner") never meets a request that says what it is about, so two
        such listings needn't be compared."""
        from belong.matching.matcher import nearly, phrases_ok
        a = self._phrases("pa", "a kayaking partner", "a kayaking partner", exactness=0.3)
        b = self._phrases("pb", "a crossword partner", "a crossword partner", exactness=0.3)
        va, vb = ShownView(a), ShownView(b)
        self.assertFalse(phrases_ok(va, vb) or phrases_ok(vb, va))
        self.assertFalse(nearly(va.get("seeking"), vb.get("offering"), (), (), va.get("offering")))
        self.assertFalse(self._shortlisted(a, b))
        self.assertTrue(self._shortlisted(a, b, words=None))  # the old catch-all group compared them

    def test_someone_who_just_wants_company_is_compared_with_everyone_nearby(self):
        """'A hangout partner' with a wide net can be met by any teacher or student, without a word in common."""
        from belong.matching.matcher import phrase_satisfies
        company = self._phrases("pa", "a hangout partner", "a hangout partner", exactness=0.3)
        teacher = self._phrases("pb", "a fun student", "a fun teacher", exactness=0.3)
        self.assertTrue(phrase_satisfies("a hangout partner", "a fun teacher", 0.3))
        want, _, is_open = index_words(ShownView(company))
        self.assertTrue(is_open)
        self.assertFalse(want & index_words(ShownView(teacher))[1])  # no word in common
        self.assertTrue(self._shortlisted(company, teacher))

    def test_unrelated_requests_are_not_compared(self):
        plumber = self._phrases("pa", "a plumbing customer", "a plumber", ("plumbing", "home repair"))
        band = self._phrases("pb", "a ska guitarist", "a ska band", ("ska music", "bands"))
        self.assertFalse(self._shortlisted(plumber, band))
        self.assertTrue(self._shortlisted(plumber, band, words=None))  # the old catch-all group compared them


class FairRanking(unittest.TestCase):
    """Step 5: ranking only. A busy listing gives way, a freshly updated one comes first, and no
    pair ever skips the threshold or the reason checks."""

    def plumbers(self, z_updated="2026-10-12T09:00:00Z"):
        def phr(pid, seeking, offering, intent):
            return listing(pid, intent, [fld("city", "Tokyo"), fld("country", "Japan"), detail("free time", "weekends"),
                                         fld("seeking", seeking), fld("offering", offering)])
        customer = phr("pc", "a plumber", "a plumbing customer", "Need a plumber in Tokyo.")
        y = phr("py", "a plumbing customer", "a plumber", "I'm a plumber in Tokyo, looking for new customers.")
        z = replace(phr("pz", "a plumbing customer", "a plumber", "I'm a plumber in Tokyo, looking for new customers."),
                    updated_at=z_updated)
        return board_of(customer, y, z)[1]

    def first(self, board, **kw):
        p = OfflineMatcher(max_per_person=1).match(board, **kw).proposals
        return {p[0].person_a, p[0].person_b} - {"pc"}

    def test_a_busy_listing_gives_way(self):
        board = self.plumbers()
        self.assertEqual(self.first(board), {"py"})  # a tie: the order on the board decides
        self.assertEqual(self.first(board, load={"py": 3}), {"pz"})

    def test_a_focused_run_leaves_out_the_focus_listings_own_load(self):
        """The demo matches only its phones (focus). A phone's own load is the same for every one of its
        candidates, so it is left out: two busy phones aren't pushed down for each other."""
        board = self.plumbers()
        self.assertEqual(self.first(board, focus={"pc", "py"}, load={"pc": 5, "py": 5}), {"py"})
        self.assertEqual(self.first(board, focus={"pc"}, load={"pc": 5, "py": 5}), {"pz"})  # the other side's still counts

    def test_a_listing_updated_in_the_last_day_comes_first(self):
        board = self.plumbers(z_updated="2026-10-14T09:00:00Z")
        self.assertEqual(recently_updated(board.listings()), {"pz"})
        self.assertEqual(self.first(board), {"pz"})

    def test_fair_ranking_only_reorders(self):
        """With no limit on matches per person, load changes nothing but the order of exact matches. A
        listing hears about at most one close match (TRUST.md #27), and fair ranking may pick which one,
        but only among close matches that passed every check: the Warden accepts each one picked."""
        w, board = warden(), Marketboard()
        for p in generate(300, 7):
            w.admit(p.person, draft_listing(p, "2026-10-12T09:00:00Z"), board)
        everyone = OfflineMatcher(max_per_person=10_000)
        load = {l.person_id: i % 16 for i, l in enumerate(board.listings())}
        plain = everyone.match(board).proposals
        busy = everyone.match(board, load=load).proposals
        exact = lambda props: sorted((p for p in props if not p.close), key=lambda p: (p.person_a, p.person_b))
        self.assertEqual(exact(plain), exact(busy))  # the same pairs, the same scores
        self.assertNotEqual(plain, busy)  # in another order
        close = lambda props: [p for p in props if p.close]
        self.assertTrue(close(plain))
        self.assertEqual(sorted(p.close for p in close(plain)), sorted(p.close for p in close(busy)))
        for p in close(plain) + close(busy):
            self.assertIsNotNone(warden().exit_introduction(p, board), p)
        for p in OfflineMatcher().match(board, load=load).proposals:  # and with the usual limit, every pick qualifies
            self.assertGreaterEqual(p.shown_score, p.shown_threshold)
            self.assertTrue(p.reason_fields)

    def test_times_are_read_without_a_clock(self):
        self.assertEqual(_minutes("1970-01-01T00:00:00Z"), 0)
        self.assertEqual(_minutes("2026-10-13T09:00:00Z") - _minutes("2026-10-12T09:00:00Z"), 24 * 60)
        self.assertIsNone(_minutes("t"))

    def test_the_demo_ranks_with_its_own_introductions_today(self):
        from belong.demo import DemoSession
        from tests.test_demo import climbers
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        self.assertTrue(s.intros)
        seen = {}
        original = s.matcher.match

        def spy(board, focus=None, load=None):
            seen["load"] = load
            return original(board, focus=focus, load=load)

        with mock.patch.object(s.matcher, "match", spy):
            s.run_matchmaker()
        expected = {}
        for intro in s.intros.values():
            if intro.sent_at < s.last_run:  # sent before this run, all within the day
                for pid in (intro.person_a, intro.person_b):
                    expected[pid] = expected.get(pid, 0) + 1
        self.assertEqual(seen["load"], expected)


if __name__ == "__main__":
    unittest.main()


class NegationThatMeansFine(unittest.TestCase):
    """A negation after "online" that goes on to mean it's fine is not a no (review finding)."""

    def test_fine_phrasings_still_allow_online(self):
        from belong.schema import ONLINE_WORDS, said
        for text in ("Online is not a problem for me", "online isn't an issue", "Online doesn't bother me",
                     "Remote wouldn't be an issue", "remote doesn't matter to me", "remote wouldn't hurt"):
            self.assertIs(said(text, ONLINE_WORDS), True, text)
        for text in ("online isn't for me", "online doesn't work for me", "not online", "online is not an option"):
            self.assertIs(said(text, ONLINE_WORDS), False, text)
