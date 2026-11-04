"""Notification choices (SPEC §5 follow-ups; Belong's close matches): questions only for people
who opted in, close matches only one-way to people who opted in, and an "also offer" question
quotes only a shown phrase of someone who opted in, without saying who. The one question asked
without an opt-in is the opt-in itself ("Want to include close matches?"): once, with no match yet."""
from __future__ import annotations

import unittest
from dataclasses import replace

from belong.leaks import leak_report
from belong.matching.contract import FollowUp, Proposal
from belong.matching.marketboard import Marketboard
from belong.matching.matcher import OfflineMatcher
from belong.schema import HIDDEN, detail_reasons
from tests.helpers import detail, fld, listing, person, warden


def repair_board(customer_close=True, fixer_questions=True):
    """Someone wants an AC repair technician; a repairman nearby wants repair customers."""
    w, board = warden(), Marketboard()
    customer = replace(listing("pc", "Need my air conditioner fixed in Tokyo this weekend.", [
        fld("first_name", "Mia"), fld("country", "Japan"), fld("city", "Tokyo"), detail("free time", "weekends"),
        fld("seeking", "an air conditioner repair technician"), fld("offering", "an air conditioner repair customer"),
        fld("religion", "Buddhist", HIDDEN)]), notify_close=customer_close, notify_questions=True)
    fixer = replace(listing("pf", "I'm a repairman in Tokyo looking for repair jobs.", [
        fld("first_name", "Ken"), fld("country", "Japan"), fld("city", "Tokyo"), detail("free time", "any"),
        fld("seeking", "a repair customer"), fld("offering", "a repairman")]),
        notify_close=False, notify_questions=fixer_questions)
    for l in (customer, fixer):
        assert w.admit(person(l.person_id), l, board).accepted
    return w, board


class AlsoOffer(unittest.TestCase):
    def test_the_matchmaker_asks_the_repairman_about_air_conditioners(self):
        w, board = repair_board()
        result = OfflineMatcher().match(board)
        self.assertEqual(result.proposals, [])  # not a match yet
        q = [f for f in result.followups if f.question_id == "also_offer"]
        self.assertEqual([(f.person_id, f.about) for f in q], [("pf", "pc")])
        out = w.exit_followup(q[0], board)
        self.assertEqual(out.question_text,
                         "Someone is looking for an air conditioner repair technician. Would that work for you?")
        self.assertNotIn("Mia", out.question_text)
        self.assertNotIn("pc", str(out.to_dict()))  # who asked is never sent

    def test_no_question_without_both_opt_ins(self):
        for kw in ({"customer_close": False}, {"fixer_questions": False}):
            w, board = repair_board(**kw)
            self.assertFalse([f for f in OfflineMatcher().match(board).followups if f.question_id == "also_offer"], kw)
            self.assertIsNone(w.exit_followup(FollowUp("pf", "also_offer", about="pc"), board), kw)

    def test_saying_yes_makes_it_a_match(self):
        from belong.agent import also_offer
        w, board = repair_board()
        fixer = also_offer(board.listing("pf"), "an air conditioner repair technician", "2026-10-13T00:00:00Z")
        self.assertTrue(w.admit(person("pf"), fixer, board).accepted)
        pairs = [(p.person_a, p.person_b, p.close) for p in OfflineMatcher().match(board).proposals]
        self.assertEqual(pairs, [("pc", "pf", "")])


def skating_board():
    """A beginner wants an ice hockey coach; a figure skater wants to teach."""
    w, board = warden(), Marketboard()
    hockey = replace(listing("ph", "I'm a beginner and want an ice hockey coach in Toronto.", [
        fld("first_name", "Sam"), fld("country", "Canada"), fld("city", "Toronto"), detail("free time", "weekends"),
        fld("seeking", "a beginner ice hockey coach"), fld("offering", "an ice hockey student"),
        fld("topics", ("ice skating", "hockey"))]),
        notify_close=True, notify_questions=True)
    skater = replace(listing("ps", "Figure skater in Toronto, happy to teach.", [
        fld("first_name", "Lea"), fld("country", "Canada"), fld("city", "Toronto"), detail("free time", "any"),
        fld("seeking", "a figure skating student"), fld("offering", "a figure skating coach"),
        fld("topics", ("ice skating", "figure skating"))]),
        notify_close=False, notify_questions=True)
    for l in (hockey, skater):
        assert w.admit(person(l.person_id), l, board).accepted
    return w, board


class RelatedActivities(unittest.TestCase):
    def test_nearly(self):
        """No built-in list of activities: related requests are found through shared words or
        stems, or the topics each person's agent wrote."""
        from belong.matching.matcher import nearly
        self.assertTrue(nearly("an air conditioner repair technician", "a repairman"))  # a shared stem
        self.assertFalse(nearly("a beginner ice hockey coach", "a figure skating coach"))  # nothing in common...
        self.assertTrue(nearly("a beginner ice hockey coach", "a figure skating coach",
                               ("ice skating", "hockey"), ("ice skating", "figure skating")))  # ...until topics say so
        self.assertTrue(nearly("a bass teacher", "a guitar teacher", ("bass guitar",), ("guitar", "bass guitar")))
        self.assertFalse(nearly("a beginner ice hockey coach", "a figure skating student",
                                ("ice skating",), ("ice skating",)))  # roles must line up
        self.assertFalse(nearly("a piano teacher", "a figure skating coach", ("piano",), ("ice skating",)))
        self.assertTrue(nearly("a snowboarding teacher", "a skiing tutor", ("snowboarding",), ("skiing", "snow sports")))
        # an umbrella word alone ("sports") connects nothing
        self.assertFalse(nearly("an ice hockey coach", "a surfing coach", ("team sports",), ("water sports",)))

    def test_the_figure_skater_is_asked_about_ice_hockey_and_yes_makes_a_match(self):
        from belong.agent import also_offer
        w, board = skating_board()
        result = OfflineMatcher().match(board)
        self.assertEqual(result.proposals, [])
        q = [f for f in result.followups if f.question_id == "also_offer"]
        self.assertEqual([(f.person_id, f.about) for f in q], [("ps", "ph")])
        out = w.exit_followup(q[0], board)
        self.assertEqual(out.question_text, "Someone is looking for a beginner ice hockey coach. Would that work for you?")
        skater = also_offer(board.listing("ps"), out.offer_add, "2026-10-13T00:00:00Z", out.seek_add)
        self.assertEqual(skater.get("seeking").value, "a figure skating student or an ice hockey student")
        self.assertTrue(w.admit(person("ps"), skater, board).accepted)
        pairs = [(p.person_a, p.person_b, p.close) for p in OfflineMatcher().match(board).proposals]
        self.assertEqual(pairs, [("ph", "ps", "")])


class Headlines(unittest.TestCase):
    """People see headlines in plain words; the stiff matching phrases stay behind the scenes."""

    def test_a_question_quotes_the_other_listings_headline(self):
        w, board = warden(), Marketboard()
        seller = replace(listing("px", "Selling 100 cardboard boxes in Tokyo, pickup only.", [
            fld("first_name", "Kai"), fld("country", "Japan"), fld("city", "Tokyo"),
            fld("headline", "Selling 100 cardboard boxes"), fld("seeking", "a cardboard box buyer"),
            fld("offering", "a cardboard box seller"), fld("topics", ("cardboard boxes", "moving supplies"))]),
            notify_close=True, notify_questions=True)
        wanter = replace(listing("py", "Looking for free cardboard boxes in Tokyo.", [
            fld("first_name", "Ren"), fld("country", "Japan"), fld("city", "Tokyo"),
            fld("headline", "Looking for free cardboard boxes"), fld("seeking", "a free cardboard box giver"),
            fld("offering", "a cardboard box recipient"), fld("topics", ("cardboard boxes", "moving"))]),
            notify_close=True, notify_questions=True)
        for l in (seller, wanter):
            self.assertTrue(w.admit(person(l.person_id), l, board).accepted)
        texts = {f.person_id: w.exit_followup(f, board).question_text
                 for f in OfflineMatcher().match(board).followups if f.question_id == "also_offer"}
        self.assertEqual(texts, {"px": "Someone is looking for free cardboard boxes. Would that work for you?",
                                 "py": "Someone is selling 100 cardboard boxes. Would that work for you?"})

    def test_reasons_use_the_headline(self):
        from belong.warden import reason_text
        self.assertEqual(reason_text([("seeking", "x", "y"), ("offering", "a", "a cardboard box seller"),
                                      ("headline", "Looking for free boxes", "Selling 100 cardboard boxes")]),
                         "They're selling 100 cardboard boxes.")


def reasons(board, a="pc", b="pf"):
    """The reasons a proposal for this pair must give: a shown built-in they both have, and exactly the open
    details the Warden finds they have in common (schema.detail_reasons)."""
    return ("city", *detail_reasons(board.listing(a).shown_fields(), board.listing(b).shown_fields(), False))


class CloseMatches(unittest.TestCase):
    def proposal(self, board, **kw):
        return Proposal(**{"person_a": "pc", "person_b": "pf", "a_intent_met": False, "b_intent_met": True,
                           "score": 1.0, "shown_score": 1.0, "shown_threshold": 0.1, "reason_fields": reasons(board),
                           "used_hidden_fields": (), "close": "pc", **kw})

    def test_a_close_match_goes_one_way(self):
        w, board = repair_board()
        self.assertEqual(reasons(board), ("city",))  # an own "any" says nothing in common: no reason
        intro = w.exit_introduction(self.proposal(board), board)
        self.assertEqual(intro.one_way_to, "pc")
        with self.assertRaises(PermissionError):
            w.record_decision(intro.intro_id, "pf", True)  # the repairman can't see it yet
        w.record_decision(intro.intro_id, "pc", True)
        w.record_decision(intro.intro_id, "pf", True)
        self.assertTrue(w.is_connected(intro.intro_id))

    def test_only_for_someone_who_opted_in(self):
        w, board = repair_board(customer_close=False)
        self.assertIsNone(w.exit_introduction(self.proposal(board), board))
        self.assertEqual([r.check for r in w.receipts if r.result == "fail"], ["close_match_opted_in"])


def lonely_board(**prefs):
    """One listing nobody on the board is looking for."""
    w, board = warden(), Marketboard()
    l = replace(listing("pl", "Looking for a theremin teacher in Reykjavik.", [
        fld("first_name", "Mia"), fld("country", "Iceland"), fld("city", "Reykjavik"),
        fld("seeking", "a theremin teacher"), fld("offering", "a theremin student")]), **prefs)
    assert w.admit(person("pl"), l, board).accepted
    return w, board


class IncludeCloseMatches(unittest.TestCase):
    def test_no_match_yet_gets_asked_once(self):
        w, board = lonely_board()
        q = [f for f in OfflineMatcher().match(board).followups if f.question_id == "broaden"]
        self.assertEqual([f.person_id for f in q], ["pl"])
        out = w.exit_followup(q[0], board)
        self.assertEqual(out.question_text, "No matches yet. Want to include close matches?")
        self.assertIsNone(w.exit_followup(q[0], board))  # only ever once
        self.assertEqual(w.receipts[-1].check, "broaden_once")

    def test_not_asked_once_close_matches_are_on(self):
        w, board = lonely_board(notify_close=True, notify_questions=True)
        self.assertFalse([f for f in OfflineMatcher().match(board).followups if f.question_id == "broaden"])
        self.assertIsNone(w.exit_followup(FollowUp("pl", "broaden"), board))

    def test_not_asked_with_a_match(self):
        w, board = repair_board(customer_close=False, fixer_questions=False)
        intro = w.exit_introduction(Proposal("pc", "pf", True, True, 1.0, 1.0, 0.1, reasons(board), ()), board)
        self.assertIsNotNone(intro)
        self.assertIsNone(w.exit_followup(FollowUp("pf", "broaden"), board))
        self.assertEqual(w.receipts[-1].check, "broaden_no_match_yet")

    def test_it_never_carries_another_listing(self):
        w, board = repair_board(customer_close=False, fixer_questions=False)
        self.assertIsNone(w.exit_followup(FollowUp("pf", "broaden", about="pc"), board))

    def test_other_questions_still_need_the_opt_in(self):
        w, board = lonely_board()
        self.assertIsNone(w.exit_followup(FollowUp("pl", "age_range"), board))
        self.assertEqual(w.receipts[-1].check, "followup_opted_in")


class AuditR5(unittest.TestCase):
    BROADEN = {"person_id": "pf", "question_id": "broaden", "phrase": "", "offer_add": "", "seek_add": "", "pay": "",
               "question_text": "No matches yet. Want to include close matches?"}

    def test_the_audit_allows_one_include_close_matches_question(self):
        w, board = repair_board(customer_close=False, fixer_questions=False)
        report = leak_report(board, [], [self.BROADEN], [], [], [], set(), {}, 20, 3)
        self.assertTrue(report["ok"], report["findings"])

    def test_the_audit_catches_it_asked_twice_or_with_close_matches_on(self):
        w, board = repair_board(customer_close=True, fixer_questions=False)
        followups = [self.BROADEN, self.BROADEN, {**self.BROADEN, "person_id": "pc"}]
        r5 = [f for f in leak_report(board, [], followups, [], [], [], set(), {}, 20, 3)["findings"] if f["audit"] == "R5"]
        self.assertEqual(len(r5), 2, r5)  # the second to pf; the one to pc, whose close matches were on

    def test_the_audit_catches_questions_and_close_matches_nobody_opted_in_to(self):
        w, board = repair_board(customer_close=False, fixer_questions=False)
        followups = [{"person_id": "pf", "question_id": "age_range", "phrase": "", "offer_add": "", "seek_add": "", "pay": "",
                      "question_text": "What age range are you open to?"},
                     {"person_id": "pf", "question_id": "also_offer", "phrase": "looking for an air conditioner repair technician",
                      "offer_add": "an air conditioner repair technician", "seek_add": "an air conditioner repair customer", "pay": "",
                      "question_text": "Someone is looking for an air conditioner repair technician. Would that work for you?"}]
        intros = [{"intro_id": "i1", "person_a": "pc", "person_b": "pf", "one_way_to": "pc",
                   "a_view": {"fields": {}, "reason": ""}, "b_view": {"fields": {}, "reason": ""}}]
        report = leak_report(board, intros, followups, [], [], [], set(), {}, 20, 3)
        r5 = [f["detail"] for f in report["findings"] if f["audit"] == "R5"]
        self.assertEqual(len(r5), 4, r5)  # two questions to someone who opted out, a quote, a close match


if __name__ == "__main__":
    unittest.main()
