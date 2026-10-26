"""The matching order (SPEC §5, §8.2): how two people can meet first, then where (travel distance),
then what (a word index), with the old rules unchanged after that, and fair ranking at the end."""
from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from belong import agent
from belong.evaluate import can_meet, compare_narrowing
from belong.leaks import leak_report
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import (OfflineMatcher, ShownView, _minutes, effective_kind, index_words,
                                     parse_intent, recently_updated)
from belong.matching.narrow import meeting_modes, shortlists, within_reach
from belong.schema import HIDDEN, km_between, meeting_format, travel_choice
from belong.synth import generate
from belong.warden import reason_text
from tests.helpers import fld, listing, person, warden


def board_of(*listings):
    w, board = warden(), Marketboard()
    for l in listings:
        result = w.admit(person(l.person_id), l, board)
        assert result.accepted, (l.person_id, result.problems)
    return w, board


def web(pid, role, city, country, fmt, intent=None):
    """A web designer or one of their clients."""
    seeking, offering = ("a web design client", "a web designer") if role == "designer" else \
        ("a web designer", "a web design client")
    headline = "Offering web design" if role == "designer" else "Looking for a web designer"
    fields = [fld("first_name", pid.title()), fld("city", city), fld("country", country), fld("schedule", "weekends"),
              fld("seeking", seeking), fld("offering", offering), fld("headline", headline),
              fld("topics", ("web design", "websites"))]
    if fmt:
        fields.append(fld("format", fmt))
    text = intent or ("I'm a web designer looking for new clients." if role == "designer" else "Need a web designer.")
    return listing(pid, text, fields)


def roommate(pid, city, travel=None, country="Japan"):
    fields = [fld("first_name", pid.title()), fld("city", city), fld("country", country), fld("move_in", "November"),
              fld("schedule", "weekends")]
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

    def test_without_a_format_the_description_decides(self):
        online_text = web("kenji", "client", "Tokyo", "Japan", None, intent="Need a web designer, online is fine.")
        in_person_text = web("kenji", "client", "Tokyo", "Japan", None, intent="Need a web designer in Tokyo.")
        self.assertEqual(pairs(board_of(web("lena", "designer", "London", "United Kingdom", "Online"), online_text)[1]),
                         {frozenset(("lena", "kenji"))})
        self.assertEqual(pairs(board_of(web("lena", "designer", "London", "United Kingdom", "Online"),
                                        in_person_text)[1]), set())

    def test_any_kind_but_dating_may_meet_online(self):
        def runner(pid, city, country):
            return listing(pid, f"Running buddy for a virtual running club, easy weekend runs online.", [
                fld("city", city), fld("country", country), fld("pace", "easy"), fld("schedule", "weekends"),
                fld("format", "Online")])
        self.assertEqual(pairs(board_of(runner("ana", "Berlin", "Germany"), runner("tom", "London", "United Kingdom"))[1]),
                         {frozenset(("ana", "tom"))})
        dating = listing("x", "Looking for a relationship with a woman, aged 25 to 34.", [fld("format", "Online")])
        self.assertEqual(meeting_modes(parse_intent(dating.intent_text), ShownView(dating)), {"in person"})


class TravelDistance(unittest.TestCase):
    """Step 2: in person, two cities are in reach within *both* people's travel distance."""

    def test_both_limits_must_reach(self):
        self.assertAlmostEqual(km_between("Osaka", "Kobe"), 28, delta=2)
        for a, b, met in (("Up to 30 km", "Up to 30 km", True), ("Up to 30 km", None, False),
                          (None, "Up to 30 km", False), ("Up to 100 km", "Same city", False),
                          ("Same city", "Up to 100 km", False), ("Up to 100 km", "Up to 30 km", True)):
            got = pairs(board_of(roommate("pa", "Osaka", a), roommate("pb", "Kobe", b))[1])
            self.assertEqual(got == {frozenset(("pa", "pb"))}, met, (a, b))

    def test_a_longer_trip_needs_the_longer_choice(self):
        self.assertAlmostEqual(km_between("Osaka", "Kyoto"), 43, delta=2)
        self.assertEqual(pairs(board_of(roommate("pa", "Osaka", "Up to 30 km"), roommate("pb", "Kyoto", "Up to 30 km"))[1]),
                         set())
        self.assertEqual(pairs(board_of(roommate("pa", "Osaka", "Up to 100 km"), roommate("pb", "Kyoto", "Up to 100 km"))[1]),
                         {frozenset(("pa", "pb"))})

    def test_an_unknown_city_only_matches_itself(self):
        self.assertIsNone(km_between("Galway", "Athenry"))
        far = (roommate("pa", "Galway", "Up to 100 km", "Ireland"), roommate("pb", "Athenry", "Up to 100 km", "Ireland"))
        self.assertEqual(pairs(board_of(*far)[1]), set())
        same = (roommate("pa", "Galway", None, "Ireland"), roommate("pb", "galway ", None, "Ireland"))
        self.assertEqual(pairs(board_of(*same)[1]), {frozenset(("pa", "pb"))})

    def test_no_city_is_still_open_to_anywhere_in_the_country(self):
        anywhere = listing("pa", "Looking for a roommate anywhere in Japan, moving in November.",
                           [fld("country", "Japan"), fld("move_in", "November"), fld("schedule", "weekends")])
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
        parsed = {pid: effective_kind(parse_intent(v.intent_text), v) for pid, v in views.items()}
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
        self.assertEqual(reason_text([("city", "Osaka", "Kyoto")]), "You're near each other: Osaka and Kyoto.")
        self.assertEqual(reason_text([("city", "Osaka", "osaka")]), "You're both in osaka.")
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
        with mock.patch("belong.warden._where_text", lambda key, mine, theirs: f"You're both in {theirs}."):
            self.assertFalse(result()["blocked"])

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
        self.assertIn("You're near each other: Osaka and Kyoto.", views["pa"])
        self.assertIn("You're near each other: Kyoto and Osaka.", views["pb"])
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
        parsed = {pid: effective_kind(parse_intent(v.intent_text), v) for pid, v in views.items()}
        lists, _ = shortlists(parsed, views, 10, words=words)
        return b.person_id in lists[a.person_id] and a.person_id in lists[b.person_id]

    def _phrases(self, pid, seeking, offering, topics=(), interests=(), exactness=0.5):
        fields = [fld("city", "Tokyo"), fld("country", "Japan"), fld("seeking", seeking), fld("offering", offering)]
        if topics:
            fields.append(fld("topics", tuple(topics)))
        if interests:
            fields.append(fld("interests", tuple(interests)))
        # The description names no kind ("partner" would read as dating), so the phrases decide.
        return listing(pid, "Hoping to meet people in Tokyo.", fields, exactness=exactness)

    def test_related_requests_still_meet(self):
        cases = [
            # a shared stem (nearly(): "repair" / "repairman")
            (("an air conditioner repair technician", "an air conditioner repair customer"), ("a repair customer", "a repairman")),
            # only the topics connect them
            (("a beginner ice hockey coach", "an ice hockey student", ("ice skating", "hockey")),
             ("a figure skating student", "a figure skating coach", ("ice skating", "figure skating"))),
            # what one shows about themselves (interests) meets what the other looks for, with a wide net
            (("a French and chess teacher", "a French and chess student", (), (), 0.3),
             ("a social hangout partner", "a social hangout partner", (), ("chess", "languages"), 0.3)),
            # a role word alone carries a short phrase with a wide net
            (("a chess partner", "a chess partner", (), (), 0.3), ("a hiking partner", "a hiking partner", (), (), 0.3)),
        ]
        from belong.matching.matcher import about, nearly, phrase_satisfies
        for a, b in cases:
            la, lb = self._phrases("pa", *a), self._phrases("pb", *b)
            va, vb = ShownView(la), ShownView(lb)
            # The decisions would accept something here (so the pair must stay a candidate)...
            self.assertTrue(phrase_satisfies(va.get("seeking"), vb.get("offering"), va.exactness, about(vb))
                            or nearly(va.get("seeking"), vb.get("offering"), va.get("topics"), vb.get("topics")), (a, b))
            # ...and narrowing keeps it.
            self.assertTrue(self._shortlisted(la, lb), (a, b))

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
            return listing(pid, intent, [fld("city", "Tokyo"), fld("country", "Japan"), fld("schedule", "weekends"),
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

    def test_a_listing_updated_in_the_last_day_comes_first(self):
        board = self.plumbers(z_updated="2026-10-14T09:00:00Z")
        self.assertEqual(recently_updated(board.listings()), {"pz"})
        self.assertEqual(self.first(board), {"pz"})

    def test_fair_ranking_only_reorders(self):
        w, board = warden(), Marketboard()
        for p in generate(300, 7):
            w.admit(p.person, agent.draft_listing(p, "2026-10-12T09:00:00Z"), board)
        everyone = OfflineMatcher(max_per_person=10_000)
        load = {l.person_id: i % 16 for i, l in enumerate(board.listings())}
        plain = everyone.match(board).proposals
        busy = everyone.match(board, load=load).proposals
        self.assertEqual(sorted(plain, key=lambda p: (p.person_a, p.person_b)),
                         sorted(busy, key=lambda p: (p.person_a, p.person_b)))  # the same pairs, the same scores
        self.assertNotEqual(plain, busy)  # in another order
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
