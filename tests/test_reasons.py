"""The reason an introduction gives (SPEC §6, TRUST.md #12, #13): written by the Warden from fixed sentences and
the shown values both people show, checked by what fills those sentences (never the Warden's own fixed words,
which say nothing about anyone), reissued when a listing changes so nothing since hidden stays on anyone's
screen, and audited independently (leaks A2, which rebuilds the fixed sentences as patterns of its own)."""
from __future__ import annotations

import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest import mock

from belong import warden as warden_module
from belong.demo import DemoSession
from belong.leaks import leak_report
from belong.matching.contract import Proposal
from belong.matching.marketboard import Marketboard
from belong.schema import HIDDEN, SHOWN, detail_reasons
from belong.warden import Warden, _hidden_in_view, _say, reason_slots
from tests.helpers import detail, fld, listing, person, warden

TEMPO = ("slow", "medium", "fast")


def pair(a_extra=(), b_extra=(), a_place=("Osaka", "Japan"), b_place=("Osaka", "Japan"), a_kw=None, b_kw=None):
    """Two people looking for each other, each with their extra fields; a Warden and a board with both."""
    w, board = warden(), Marketboard()
    for pid, place, extra, kw in (("pa", a_place, a_extra, a_kw or {}), ("pb", b_place, b_extra, b_kw or {})):
        l = listing(pid, "Looking for someone to repair bikes with.", [
            fld("first_name", pid.title()), fld("city", place[0]), fld("country", place[1]),
            fld("seeking", "a bike repair partner"), fld("offering", "a bike repair partner"), *extra], **kw)
        result = w.admit(person(pid), l, board)
        assert result.accepted, result.problems
    return w, board


def introduce(w, board, built_in=("city",)):
    """Introduce pa and pb, citing these shown built-ins and exactly the open details they have in common.
    Returns (the introduction or None, the checks that failed)."""
    a, b = board.listing("pa"), board.listing("pb")
    refs = detail_reasons(a.shown_fields(), b.shown_fields(), max(a.exactness, b.exactness) >= 0.8)
    start = len(w.receipts)
    intro = w.exit_introduction(Proposal("pa", "pb", True, True, 1.0, 1.0, 0.1, (*built_in, *refs), ()), board)
    return intro, [r.check for r in w.receipts[start:] if r.result == "fail"]


def report(board, intros):
    return leak_report(board, [i.to_dict() for i in intros], [], [], [], [], set(), {}, 20, 15)


class FixedWordsAreNeverMistakenForValues(unittest.TestCase):
    """Open details are free text, so someone's value can be a word the Warden's fixed sentences use. The checks
    read only what fills the sentences, so such a value never stops an honest introduction."""

    def test_an_uncited_shown_value_that_is_a_template_word(self):
        w, board = pair(a_extra=[detail("spare key", "another")], b_place=("Kyoto", "Japan"))
        intro, failed = introduce(w, board)
        self.assertEqual(failed, [])
        self.assertEqual(intro.a_view["reason"], "You're near one another: Osaka and Kyoto.")
        self.assertTrue(report(board, [intro])["ok"])

    def test_an_uncited_step_that_is_a_template_word(self):
        venue = detail("venue", "out", compare="steps", steps=("in", "out"))
        w, board = pair(a_extra=[venue])
        intro, failed = introduce(w, board)
        self.assertEqual(failed, [])
        self.assertIn("You're both in Osaka.", intro.a_view["reason"])

    def test_a_hidden_value_that_is_template_words(self):
        for hidden, b_extra, built_in, sentence in (
                (detail("work style", "in person", visibility=HIDDEN), [fld("format", "In person")], ("format",),
                 "You can both meet in person."),
                (detail("reaction", "like", visibility=HIDDEN), [detail("tea", "green")], ("city",),
                 "You'd like their tea to be green; theirs is green.")):
            a_extra = [hidden, fld("format", "In person"), detail("tea", "green", about="them")]
            w, board = pair(a_extra=a_extra, b_extra=b_extra)
            intro, failed = introduce(w, board, built_in)
            self.assertEqual(failed, [], hidden)
            self.assertIn(sentence, intro.a_view["reason"])
            self.assertTrue(report(board, [intro])["ok"], hidden)
            # A third person anywhere on the board hiding the same words doesn't trouble the audit either.
            assert w.admit(person("pc"), listing("pc", "Looking for a flat.", [hidden, fld("city", "Lyon")]), board).accepted
            self.assertTrue(report(board, [intro])["ok"], hidden)

    def test_a_hidden_value_inside_what_fills_a_sentence_is_still_caught(self):
        """Test the tester: if a sentence were filled with something hidden, both the Warden and the audit catch it."""
        w, board = pair(a_extra=[fld("employer", "Kestrel Works", HIDDEN)])
        leaky = lambda key, mine, theirs, countries=None: _say("You're both in {}.", "Kestrel Works")
        with mock.patch.object(warden_module, "_where_text", leaky):
            intro, failed = introduce(w, board)
        self.assertIsNone(intro)
        self.assertIn("no_hidden_values", failed)
        # The audit, given such a view, finds it in the slot next to the fixed words.
        intro, _ = introduce(w, board)
        d = intro.to_dict()
        d["b_view"]["reason"] = "You're both in Kestrel Works."
        found = leak_report(board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
        self.assertEqual([(f["audit"], "'Kestrel Works'" in f["detail"]) for f in found], [("A2", True)])

    def test_anything_but_the_fixed_sentences_is_read_whole(self):
        """A sentence that isn't exactly a fixed template filled with shown values is read in full, fixed-looking
        words and all, by the Warden and by the audit."""
        w, board = pair(a_extra=[detail("reaction", "another", visibility=HIDDEN)], b_place=("Kyoto", "Japan"))
        plain = lambda key, mine, theirs, countries=None: f"You're near one another: {mine} and {theirs}."
        with mock.patch.object(warden_module, "_where_text", plain):
            intro, failed = introduce(w, board)
        self.assertIsNone(intro)
        self.assertIn("no_hidden_values", failed)
        intro, failed = introduce(w, board)  # the real sentence: fine
        self.assertEqual(failed, [])
        d = intro.to_dict()
        d["b_view"]["reason"] += " They'd like another."  # no fixed sentence reads like this
        found = leak_report(board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
        self.assertEqual([(f["audit"], "'another'" in f["detail"]) for f in found], [("A2", True)])

    def test_the_slots_are_the_shown_values_as_their_owners_wrote_them(self):
        parts = warden_module.reason_parts([("headline", None, "Selling a kayak."), ("format", "In person", "Online")])
        self.assertEqual(" ".join(parts), "They're selling a kayak. You'd meet in person; they'd meet online.")
        self.assertEqual(reason_slots(parts), ["Selling a kayak.", "In person", "Online"])
        self.assertEqual(reason_slots(["They're selling a kayak."]), ["They're selling a kayak."])  # not a template: whole

    def test_the_view_check_reads_the_slots_in_place_of_the_reason(self):
        a = listing("pa", fields=[detail("reaction", "in", visibility=HIDDEN)])
        b = listing("pb", fields=[fld("city", "Osaka")])
        view = {"fields": {"city": "Osaka"}, "reason": "You're both in Osaka."}
        self.assertEqual(_hidden_in_view(view, a, b, ["city"]), ["in"])  # read whole
        self.assertEqual(_hidden_in_view(view, a, b, ["city"], ["Osaka"]), [])  # read by its slots
        self.assertEqual(_hidden_in_view(view, a, b, ["city"], ["Osaka", "in"]), ["in"])


class TheAuditKnowsTheSameSentences(unittest.TestCase):
    def test_every_fixed_sentence_is_read_by_its_slots_by_the_audit_too(self):
        """leaks.py keeps its own copy of the fixed sentences (it never imports the Warden); they must agree, or
        the audit would read a real reason whole."""
        from belong.leaks import _reason_slots
        for template in warden_module._TEMPLATES:
            fills = [f"value {chr(97 + i)}" for i in range(template.count("{}"))]
            self.assertEqual(_reason_slots(template.format(*fills)), fills, template)
        two = "You're both in Lyon. You'd like their tea to be green, black or oolong; theirs is black."
        self.assertEqual(_reason_slots(two), ["Lyon", "tea", "green, black or oolong", "black"])

    def test_a_shown_headline_that_reads_like_two_fixed_sentences_is_no_false_alarm(self):
        """A headline worded "Looking for ..." fits "They're looking for {}." as well as "They're {}.". The audit
        reads the sentence the way that leaves the least it can't explain (fixed wording and what the recipient may
        see), so a word of the shown headline that someone else hides is no finding; anything else in it still is."""
        lines = lambda h: [fld("headline", h)]
        w, board = pair(a_extra=lines("Looking for a patient bike repair partner"), b_extra=lines("Fixing bikes"))
        third = listing("pc", "Looking for a flat.", [fld("city", "Lyon"),
                                                      detail("manner", "patient", visibility=HIDDEN)])
        assert w.admit(person("pc"), third, board).accepted
        intro, failed = introduce(w, board, ("city", "headline"))
        self.assertEqual(failed, [])
        self.assertIn("They're looking for a patient bike repair partner.", intro.b_view["reason"])
        self.assertTrue(report(board, [intro])["ok"], report(board, [intro])["findings"])
        # Test the tester: a hidden value in that slot that no shown string explains is still found.
        assert w.admit(person("pd"), listing("pd", "Looking for a flat.", [
            fld("city", "Lyon"), fld("employer", "Kestrel Works", HIDDEN)]), board).accepted
        d = intro.to_dict()
        d["b_view"]["reason"] = d["b_view"]["reason"].replace("a patient bike repair partner", "Kestrel Works")
        found = leak_report(board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
        self.assertEqual([(f["audit"], "'Kestrel Works'" in f["detail"]) for f in found], [("A2", True)])
        d["b_view"]["reason"] = d["b_view"]["reason"].replace("Kestrel Works", "a patient Kestrel Works partner")
        found = leak_report(board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
        self.assertIn(("A2", True), [(f["audit"], "'Kestrel Works'" in f["detail"]) for f in found])


class AValueCutAcrossTheFixedWords(unittest.TestCase):
    """A hidden value can hold a fixed sentence's own words (" and ", " is ", ": ", " to be ", "; yours is "). A
    sentence filled with it can then be read with the value cut across a slot boundary or across those fixed words,
    so the audit (A2) and the canary read by character position: a value is let go only if every character of it is
    fixed wording in every reading, or inside a string the recipient may see. Test the tester, for every fixed
    sentence that uses those words and every slot of it."""
    JOINS = (" and ", " is ", ": ", " to be ", "; yours is ", "; theirs is ")

    def test_a_hidden_value_holding_fixed_words_is_still_caught(self):
        from belong import probe
        secrets = {join: f"marram{join}sedge" for join in self.JOINS}
        values = list(secrets.values())
        w, board = pair(b_extra=[detail("drift", values[:3], visibility=HIDDEN),
                                 detail("swell", values[3:], visibility=HIDDEN)], b_place=("Kyoto", "Japan"))
        intro, failed = introduce(w, board)
        self.assertEqual(failed, [])
        self.assertTrue(report(board, [intro])["ok"])
        tried = 0
        for template in sorted(warden_module._TEMPLATES):
            fixed = template.split("{}")
            for join, secret in secrets.items():
                if not any(join in piece for piece in fixed):
                    continue
                for i in range(template.count("{}")):
                    fills = ["Kyoto"] * template.count("{}")  # the counterpart's shown city: the recipient may see it
                    fills[i] = secret
                    d = intro.to_dict()
                    d["a_view"]["reason"] = template.format(*fills)
                    found = leak_report(board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
                    self.assertIn(("A2", True), [(f["audit"], repr(secret) in f["detail"]) for f in found],
                                  d["a_view"]["reason"])
                    canary = probe._secrets_found({"pb": [secret]}, set(), board,
                                                  {"introductions": [d], "reveals": []}, [], "")
                    self.assertEqual([f["secret"] for f in canary], [secret], d["a_view"]["reason"])
                    tried += 1
        self.assertGreaterEqual(tried, 30)  # every sentence that uses one of those words, every slot

    def test_fixed_words_and_shown_strings_around_a_hidden_value_never_explain_it(self):
        """Next to a shown string that spans part of it, or a list of shown strings, a hidden value is caught unless
        every one of its characters is explained."""
        w, board = pair(a_extra=[detail("knot", "reef", visibility=HIDDEN), detail("rope", ("hemp", "sisal"))],
                        b_extra=[detail("rope", ("hemp", "sisal")), detail("cord", "reef and hemp", visibility=HIDDEN)])
        intro, failed = introduce(w, board)
        self.assertEqual(failed, [])
        self.assertIn("You share the same rope: hemp and sisal.", intro.a_view["reason"])
        self.assertTrue(report(board, [intro])["ok"])
        d = intro.to_dict()
        d["a_view"]["reason"] = d["a_view"]["reason"].replace("hemp and sisal", "reef and hemp and sisal")
        found = {f["detail"].split("hidden value ")[-1] for f in leak_report(board, [d], [], [], [], [], set(), {},
                                                                              20, 15)["findings"]}
        self.assertTrue({"'reef' reached pa", "'reef and hemp' reached pa"} <= found, found)
        # The ", ", " and " or " or " the Warden puts between two values both of which the recipient may see is its
        # own list wording: someone else hiding those values joined that way is no false alarm.
        assert w.admit(person("pc"), listing("pc", "Looking for a flat.", [
            fld("city", "Lyon"), detail("cord", "hemp and sisal", visibility=HIDDEN)]), board).accepted
        self.assertTrue(report(board, [intro])["ok"], report(board, [intro])["findings"])


class QuestionsToo(unittest.TestCase):
    def test_a_question_is_checked_by_what_it_quotes_not_its_fixed_words(self):
        """"Someone is looking for a kayak." quotes the other listing's shown phrase; a hidden value that happens
        to be the fixed "looking" never stops it."""
        from belong.matching.contract import FollowUp
        w, board = warden(), Marketboard()
        asked = replace(listing("pa", "Selling a kayak.", [fld("city", "Lyon"), fld("seeking", "a kayak buyer"),
                                                             fld("offering", "a kayak"),
                                                             detail("stance", "looking", visibility=HIDDEN)]),
                        notify_questions=True)
        asking = replace(listing("pb", "Looking for a kayak.", [fld("city", "Lyon"), fld("seeking", "a kayak"),
                                                               fld("offering", "a kayak buyer")]), notify_close=True)
        for l in (asked, asking):
            assert w.admit(person(l.person_id), l, board).accepted
        out = w.exit_followup(FollowUp("pa", "also_offer", about="pb"), board)
        self.assertEqual(out.question_text, "Someone is looking for a kayak. Would that work for you?")


class OpenDetailWording(unittest.TestCase):
    def reasons(self, a_extra, b_extra, **kw):
        w, board = pair(a_extra=a_extra, b_extra=b_extra, **kw)
        intro, failed = introduce(w, board)
        self.assertEqual(failed, [])
        drop = "You're both in Osaka."
        return tuple(v["reason"].replace(drop, "").strip() for v in (intro.a_view, intro.b_view))

    def test_the_same_step_is_said_like_the_same_value(self):
        same = detail("tempo", "medium", compare="steps", steps=TEMPO)
        self.assertEqual(self.reasons([same], [same]), ("You share the same tempo: medium.",) * 2)
        faster = detail("tempo", "fast", compare="steps", steps=TEMPO)
        self.assertEqual(self.reasons([same], [faster]),
                         ("Their tempo is fast; yours is medium.", "Their tempo is medium; yours is fast."))

    def test_a_wish_says_only_what_was_wished_when_the_values_are_already_said(self):
        mine = [detail("tea", "green"), detail("tea", "green", about="them")]
        self.assertEqual(self.reasons(mine, [detail("tea", "green")]),
                         ("You share the same tea: green. You'd like their tea to be green.",
                          "You share the same tea: green. They'd like your tea to be green."))
        # Without a value of their own to agree on, the wish says the other person's value.
        self.assertEqual(self.reasons([detail("tea", "green", about="them")], [detail("tea", "green")]),
                         ("You'd like their tea to be green; theirs is green.",
                          "They'd like your tea to be green; yours is green."))

    def test_lists_read_a_b_or_c(self):
        wish = detail("tea", ("green", "black", "oolong"), about="them")
        a, _ = self.reasons([wish], [detail("tea", "black")])
        self.assertEqual(a, "You'd like their tea to be green, black or oolong; theirs is black.")

    def test_any_is_never_a_reason(self):
        """An own "any", or a wish for any value or every step, says nothing the two have in common."""
        for a_extra, b_extra in (([detail("tea", "any")], [detail("tea", "green")]),
                                 ([detail("tea", "any", about="them")], [detail("tea", "green")]),
                                 ([detail("tea", "green", about="them")], [detail("tea", "any")]),
                                 ([detail("tempo", TEMPO, about="them", compare="steps", steps=TEMPO)],
                                  [detail("tempo", "fast", compare="steps", steps=TEMPO)])):
            self.assertEqual(self.reasons(a_extra, b_extra), ("",) * 2, (a_extra, b_extra))
            w, board = pair(a_extra=a_extra, b_extra=b_extra)
            a, b = board.listing("pa"), board.listing("pb")
            self.assertEqual(detail_reasons(a.shown_fields(), b.shown_fields(), False), (), (a_extra, b_extra))


class NothingSinceHiddenStaysOnScreen(unittest.TestCase):
    def setUp(self):
        self.w, self.board = pair(a_extra=[detail("tea", "green")], b_extra=[detail("tea", "green")])
        self.intro, _ = introduce(self.w, self.board)
        self.assertEqual(self.intro.a_view["fields"]["me:tea"]["values"], ["green"])
        self.assertIn("You share the same tea: green.", self.intro.a_view["reason"])

    def hide_bs_tea(self):
        b = self.board.listing("pb")
        fields = tuple(replace(f, visibility=HIDDEN) if f.key == "me:tea" else f for f in b.fields)
        assert self.w.admit(person("pb"), replace(b, fields=fields, version=b.version + 1), self.board).accepted

    def test_a_reissue_drops_what_was_since_hidden(self):
        self.hide_bs_tea()
        new = self.w.refresh_introduction(self.intro.intro_id, ["city"], self.board)
        self.assertNotIn("me:tea", new.a_view["fields"])
        self.assertNotIn("tea", new.a_view["reason"])
        self.assertTrue(report(self.board, [new])["ok"])

    def test_a_reissue_that_fails_its_checks_is_reduced_never_left_standing(self):
        """If the full reissue fails, the Warden still reissues: no reason, only the fields that pass every
        exit check on their own, with a receipt of its own, and the audit checks it like any introduction."""
        self.hide_bs_tea()
        start = len(self.w.receipts)
        flag_name = lambda view, viewer, other, reasons, slots=None: (
            ["Pb"] if "first_name" in view["fields"] else [])
        with mock.patch.object(Warden, "_checked_views", return_value=None), \
                mock.patch.object(warden_module, "_hidden_in_view", flag_name):
            new = self.w.refresh_introduction(self.intro.intro_id, ["city"], self.board)
        self.assertEqual(new.intro_id, self.intro.intro_id)
        self.assertEqual(new.sent_at, self.intro.sent_at)
        self.assertEqual(new.a_view["reason"], "")
        self.assertNotIn("me:tea", new.a_view["fields"])     # since hidden: gone
        self.assertNotIn("first_name", new.a_view["fields"])  # fails a check on its own: left out
        self.assertEqual(new.a_view["fields"]["city"], "Osaka")
        checks = [r.check for r in self.w.receipts[start:]]
        self.assertIn("reissue_reduced", checks)
        self.assertTrue(report(self.board, [new])["ok"])
        # And a planted field in a reduced reissue is caught by the audit.
        d = new.to_dict()
        d["a_view"]["fields"]["me:tea"] = {"values": ["green"], "compare": "same", "steps": [], "required": True}
        found = leak_report(self.board, [d], [], [], [], [], set(), {}, 20, 15)["findings"]
        self.assertEqual({f["audit"] for f in found}, {"A1", "A2"})


class AfterOneListingIsDeleted(unittest.TestCase):
    """One person deletes their listing and the other then hides a detail both showed: the Warden still reissues
    (reduced, with a receipt of its own), so no stored introduction keeps the since-hidden value, whoever left first."""

    def kite_flyers(self):
        w, board = warden(), Marketboard()
        for pid, name in (("ka", "Odile"), ("kb", "Femi")):
            l = listing(pid, "Looking for someone to fly kites with on Sunday mornings.", [
                fld("first_name", name), fld("city", "Lyon"), fld("country", "France"),
                fld("seeking", "a kite flying partner"), fld("offering", "a kite flying partner"),
                detail("line weight", "light"), detail("field", "open meadow", about="both")])
            assert w.admit(person(pid), l, board).accepted
        a, b = board.listing("ka"), board.listing("kb")
        refs = detail_reasons(a.shown_fields(), b.shown_fields(), False)
        intro = w.exit_introduction(Proposal("ka", "kb", True, True, 1.0, 1.0, 0.1, ("city", *refs), ()), board)
        self.assertIn("You share the same line weight: light.", intro.a_view["reason"])
        return w, board, intro

    def test_whoever_leaves_first(self):
        for leaves, stays in (("ka", "kb"), ("kb", "ka")):
            w, board, intro = self.kite_flyers()
            assert w.withdraw(person(leaves), leaves, board)
            mine = board.listing(stays)
            fields = tuple(replace(f, visibility=HIDDEN) if f.key == "me:line weight" else f for f in mine.fields)
            assert w.admit(person(stays), replace(mine, fields=fields, version=mine.version + 1), board).accepted
            self.assertFalse(report(board, [intro])["ok"])  # the old introduction would keep it
            start = len(w.receipts)
            new = w.refresh_introduction(intro.intro_id, ["city"], board)
            self.assertIn("reissue_reduced", [r.check for r in w.receipts[start:]])
            self.assertEqual((new.intro_id, new.sent_at), (intro.intro_id, intro.sent_at))
            about_stays, about_leaves = (new.a_view, new.b_view) if leaves == "ka" else (new.b_view, new.a_view)
            self.assertNotIn("me:line weight", about_stays["fields"])  # since hidden: gone
            self.assertEqual(about_stays["fields"]["city"], "Lyon")     # still shown: kept
            self.assertEqual(about_leaves["fields"]["me:line weight"]["values"], ["light"])  # as it was when deleted
            self.assertEqual((new.a_view["reason"], new.b_view["reason"]), ("", ""))
            self.assertTrue(report(board, [new])["ok"], report(board, [new])["findings"])
            # Both gone: nothing can change any more, so nothing is reissued.
            assert w.withdraw(person(stays), stays, board)
            self.assertIsNone(w.refresh_introduction(intro.intro_id, ["city"], board))

    def test_nothing_new_reaches_the_one_who_left(self):
        """After one person deletes their listing, their stored view of the other keeps only what it already had and
        the other still shows unchanged: a detail added, a value changed or a name changed afterwards never reaches
        it. The other's view of them is as it was when they left."""
        w, board, intro = self.kite_flyers()
        assert w.withdraw(person("ka"), "ka", board)
        mine = board.listing("kb")
        fields = tuple(f for f in mine.fields if f.key not in ("first_name", "city")) + (
            fld("first_name", "Folake"), fld("city", "Lyon"), detail("spool", "wooden"))
        assert w.admit(person("kb"), replace(mine, fields=fields, version=mine.version + 1), board).accepted
        new = w.refresh_introduction(intro.intro_id, ["city"], board)
        self.assertEqual(set(new.a_view["fields"]), set(intro.a_view["fields"]) - {"first_name"})
        self.assertNotIn("me:spool", new.a_view["fields"])
        self.assertEqual(new.b_view["fields"], intro.b_view["fields"])
        self.assertTrue(report(board, [new])["ok"], report(board, [new])["findings"])
        # A later change keeps it that way: what is gone stays gone.
        mine = board.listing("kb")
        fields = tuple(replace(f, value=("heavy",)) if f.key == "me:line weight" else f for f in mine.fields)
        assert w.admit(person("kb"), replace(mine, fields=fields, version=mine.version + 1), board).accepted
        newer = w.refresh_introduction(intro.intro_id, ["city"], board)
        self.assertEqual(set(newer.a_view["fields"]), set(new.a_view["fields"]) - {"me:line weight"})
        self.assertTrue(report(board, [newer])["ok"], report(board, [newer])["findings"])

    def test_a_listing_that_ever_changed_hands_is_never_reissued_from(self):
        """A listing id is never another person's (the Warden refuses it at entry); were a board ever to hold another
        person's listing under one of an introduction's ids, its reissue would empty both views rather than read it."""
        w, board, intro = self.kite_flyers()
        intruder = replace(board.listing("ka"), owner_id="mallory", fields=(fld("first_name", "Mallory"),
                                                                           fld("city", "Lyon")), version=9)
        board.put(person("mallory"), intruder)  # straight onto the board: no Warden would accept it
        new = w.refresh_introduction(intro.intro_id, ["city"], board)
        self.assertEqual((new.a_view, new.b_view), ({"fields": {}, "reason": ""}, {"fields": {}, "reason": ""}))
        self.assertEqual(w.receipts[-1].check, "reissue_reduced")
        self.assertIn("R6", {f["audit"] for f in report(board, [new])["findings"]})

    def test_the_demo_stores_the_reissue(self):
        from tests.test_demo import climbers, fill
        for leaves, stays in (("A", "B"), ("B", "A")):
            s = DemoSession(seed=7, agent=None)
            climbers(s)
            s.run_matchmaker()
            intro_id = next(i for i, x in s.intros.items() if {x.person_a, x.person_b} == {"demoA-1", "demoB-1"})
            mine = s.board.listing(f"demo{stays}-1")
            key = next(k for k in s.intros[intro_id].a_view["fields"] if k.startswith("me:")
                       and k in s.intros[intro_id].b_view["fields"])
            s.delete_listing(leaves, f"demo{leaves}-1")
            hidden = {"religion": "Buddhist", "me:interests": list(mine.get("me:interests").value),
                      key: list(mine.get(key).value)}
            s.submit(stays, fill(s.slots[stays].draft, hidden=hidden))
            self.assertEqual(s.board.listing(f"demo{stays}-1").get(key).visibility, HIDDEN)
            view = s.intros[intro_id].a_view if stays == "B" else s.intros[intro_id].b_view
            self.assertNotIn(key, view["fields"])
            # What was sent, checked against both listings (the deleted one as it was): nothing since hidden in any
            # stored introduction; and the last run's proposals checked against the board they were made on.
            self.assertEqual(s.room_state()["leak_report"]["findings"], [], (leaves, stays))


class OnThePhone(unittest.TestCase):
    """End to end: one person hides a detail both showed; the other phone no longer shows it, or any sentence
    naming it, whether the Warden's reissue passes in full or has to be reduced."""

    def a_sees_b(self, s):
        for item in s.phone_state("A")["intros"]:
            intro = s.intros[item["intro_id"]]
            if {intro.person_a, intro.person_b} == {"demoA-1", "demoB-1"}:
                return item["view"]
        return None

    def run_case(self, reduced: bool):
        from tests.test_demo import climbers, fill
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        before = self.a_sees_b(s)
        shared = sorted(k for k in before["fields"] if k.startswith("me:"))
        self.assertTrue(shared, before)
        key = shared[0]
        name = key.partition(":")[2]
        self.assertIn(name, before["reason"])
        value = s.board.listing("demoB-1").get(key).value
        patch = mock.patch.object(Warden, "_checked_views", return_value=None) if reduced else nullcontext()
        with patch:
            s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD",
                                                            "me:interests": ["jazz", "ramen"], key: list(value)}))
        self.assertEqual(s.board.listing("demoB-1").get(key).visibility, HIDDEN)
        after = self.a_sees_b(s)
        self.assertNotIn(key, after["fields"])
        self.assertNotIn(name, after["reason"])
        self.assertEqual(after["reason"] == "", reduced)
        # The audit agrees, before the Matchmaker runs again: what was sent, and the proposals against their board.
        self.assertEqual(s.room_state()["leak_report"]["findings"], [])

    def test_a_full_reissue(self):
        self.run_case(reduced=False)

    def test_a_reduced_reissue(self):
        self.run_case(reduced=True)


if __name__ == "__main__":
    unittest.main()
