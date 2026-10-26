"""Privacy promises (SPEC §6, §7). Any leak fails these tests."""
from __future__ import annotations

import json
import unittest

from belong.leaks import leak_report
from belong.matching.contract import Proposal
from belong.schema import HIDDEN
from belong.sim import run
from belong.warden import verify_chain
from tests.helpers import fld, person, roommates_board


def proposal(reasons=("city", "move_in"), used=(), a_met=True, b_met=True):
    return Proposal("pa", "pb", a_met, b_met, 0.9, 0.9, 0.2, tuple(reasons), tuple(used))


class FullRunHasNoLeaks(unittest.TestCase):
    def test_full_300_person_run_leaks_nothing(self):
        report = run(n=300, seed=7)["leak_report"]
        self.assertTrue(report["ok"], report["findings"][:10])
        self.assertGreater(report["checked"]["views"], 100)  # the audit actually looked at things

    def test_other_seeds_leak_nothing(self):
        for seed in (1, 2, 3):
            report = run(n=200, seed=seed)["leak_report"]
            self.assertTrue(report["ok"], (seed, report["findings"][:10]))


class WardenExitBlocksLeaks(unittest.TestCase):
    def test_a_good_proposal_becomes_an_introduction_without_hidden_values(self):
        w, board = roommates_board()
        intro = w.exit_introduction(proposal(), board)
        self.assertIsNotNone(intro)
        text = json.dumps(intro.to_dict(), ensure_ascii=False)
        for secret in ("Buddhist", "Nintendo", "¥4–7M"):
            self.assertNotIn(secret, text)
        self.assertEqual(intro.a_view["fields"]["first_name"], "Ren")

    def test_a_reissued_introduction_shows_current_details_and_nothing_since_hidden(self):
        from dataclasses import replace
        w, board = roommates_board()
        intro = w.exit_introduction(proposal(), board)
        b = board.listing("pb")
        fields = tuple(replace(f, value="Rennie") if f.key == "first_name" else
                       replace(f, visibility=HIDDEN) if f.key == "interests" else f for f in b.fields)
        self.assertTrue(w.admit(person("pb"), replace(b, fields=fields, version=b.version + 1), board).accepted)
        new = w.refresh_introduction(intro.intro_id, ["city", "interests"], board)
        self.assertEqual(new.intro_id, intro.intro_id)
        self.assertEqual(new.a_view["fields"]["first_name"], "Rennie")
        self.assertNotIn("interests", new.a_view["fields"])          # hidden now, so gone
        self.assertNotIn("chess", json.dumps(new.to_dict()))          # and not in the reason either

    def test_hidden_field_as_a_reason_is_rejected(self):
        """Never in the reason: a hidden key can't be a reason field."""
        w, board = roommates_board()
        self.assertIsNone(w.exit_introduction(proposal(reasons=("city", "religion")), board))
        self.assertEqual(w.receipts[-1].check, "reason_fields_shown")
        self.assertEqual(w.receipts[-1].result, "fail")

    def test_proposal_with_no_shown_reason_is_rejected(self):
        """Never the only reason: there must be at least one shared shown reason."""
        w, board = roommates_board()
        self.assertIsNone(w.exit_introduction(proposal(reasons=()), board))

    def test_one_sided_proposal_is_rejected(self):
        """Both must be looking."""
        w, board = roommates_board()
        self.assertIsNone(w.exit_introduction(proposal(b_met=False), board))

    def test_hidden_values_stay_out_even_when_one_side_shows_the_same_key(self):
        """A hides diet=vegan, B shows diet=vegan. A sees B's diet (B chose to show it);
        B must not learn A's."""
        w, board = roommates_board(extra_a=[fld("diet", "vegan", HIDDEN)], extra_b=[fld("diet", "vegan")])
        intro = w.exit_introduction(proposal(), board)
        self.assertEqual(intro.a_view["fields"]["diet"], "vegan")
        self.assertNotIn("diet", intro.b_view["fields"])
        self.assertNotIn("vegan", json.dumps(intro.b_view))

    def test_approach_limit_is_enforced(self):
        w, board = roommates_board()
        w.approach_limit = 1
        self.assertIsNotNone(w.exit_introduction(proposal(), board))
        self.assertIsNone(w.exit_introduction(proposal(), board))

    def test_the_approach_limit_is_per_day(self):
        w, board = roommates_board()
        self.assertEqual(w.approach_limit, 15)  # the default: 15 introductions per listing per day
        w.approach_limit = 2
        self.assertIsNotNone(w.exit_introduction(proposal(), board))
        w.clock.advance(hours=23)
        self.assertIsNotNone(w.exit_introduction(proposal(), board))
        self.assertIsNone(w.exit_introduction(proposal(), board))  # two in the last 24 hours
        w.clock.advance(hours=1, seconds=1)
        self.assertIsNotNone(w.exit_introduction(proposal(), board))  # the first is now over a day old

    def test_an_introduction_carries_when_it_was_sent(self):
        w, board = roommates_board()
        intro = w.exit_introduction(proposal(), board)
        self.assertEqual(intro.to_dict()["sent_at"], w.clock.now())
        w.clock.advance(hours=5)
        again = w.refresh_introduction(intro.intro_id, ["city"], board)
        self.assertEqual(again.sent_at, intro.sent_at)  # a refresh isn't a new introduction


class LeakReportCatchesPlantedLeaks(unittest.TestCase):
    """Test the tester: the independent audit must notice a leak if one gets out."""

    def _report(self, board, intros=(), reveals=(), receipts=(), connections=(), consents=None, proposals=()):
        return leak_report(board, list(intros), [], list(reveals), list(receipts), list(proposals),
                           set(connections), consents or {}, k_crowd=20, approach_limit=3)

    def _intro(self, board, w):
        return w.exit_introduction(proposal(), board).to_dict()

    def test_hidden_value_in_reason_text_is_caught(self):
        w, board = roommates_board()
        d = self._intro(board, w)
        d["b_view"]["reason"] += " You're both Buddhist."
        report = self._report(board, [d])
        self.assertFalse(report["ok"])
        self.assertEqual(report["findings"][0]["audit"], "A2")

    def test_too_many_introductions_in_one_day_are_caught(self):
        w, board = roommates_board()
        d = self._intro(board, w)

        def at(hour):
            day, h = divmod(hour, 24)
            return {**d, "sent_at": f"2026-10-{12 + day:02d}T{h:02d}:00:00Z"}
        r3 = lambda intros: [f for f in self._report(board, intros)["findings"] if f["audit"] == "R3"]
        self.assertEqual(r3([at(0), at(5), at(23)]), [])  # 3 within a day: at the limit
        self.assertEqual(r3([at(0), at(10), at(24), at(30), at(48)]), [])  # 3 a day, spread out
        self.assertEqual(len(r3([at(20), at(25), at(30), at(40)])), 2)  # 4 within 24 hours, for both people
        no_time = {k: v for k, v in d.items() if k != "sent_at"}
        self.assertEqual(len(r3([no_time] * 2 + [at(0), at(48)])), 2)  # no send time: counted as one day

    def test_hidden_field_in_view_is_caught(self):
        w, board = roommates_board()
        d = self._intro(board, w)
        d["a_view"]["fields"]["income_band"] = "¥4–7M"
        audits = {f["audit"] for f in self._report(board, [d])["findings"]}
        self.assertEqual(audits, {"A1", "A2"})

    def test_receipt_with_content_is_caught(self):
        w, board = roommates_board()
        bad = {**w.receipts[0].to_dict(), "note": "Buddhist"}
        self.assertFalse(self._report(board, receipts=[bad])["ok"])

    def test_reveal_without_consent_is_caught(self):
        w, board = roommates_board()
        d = self._intro(board, w)
        rv = {"person_a": "pa", "person_b": "pb", "key": "religion", "values": ["Buddhist"]}
        pair = frozenset(("pa", "pb"))
        consents = {(d["intro_id"], "pa"): {"religion"}, (d["intro_id"], "pb"): set()}
        report = self._report(board, [d], reveals=[rv], connections=[pair], consents=consents)
        self.assertFalse(report["ok"])

    def test_hidden_field_below_crowd_threshold_is_caught(self):
        """Only counts in a crowd: two people sharing 'Buddhist' is far below k = 20."""
        w, board = roommates_board()
        report = self._report(board, proposals=[proposal(used=("religion",))])
        self.assertEqual(report["findings"][0]["audit"], "R2")


class MutualReveal(unittest.TestCase):
    def _connected(self):
        w, board = roommates_board()
        intro = w.exit_introduction(proposal(), board)
        return w, board, intro

    def test_shared_detail_revealed_only_when_both_opt_in(self):
        w, board, intro = self._connected()
        w.record_decision(intro.intro_id, "pa", True)
        w.record_decision(intro.intro_id, "pb", True)
        self.assertEqual(w.exit_reveal(intro.intro_id, {"religion"}, set(), board), [])
        out = w.exit_reveal(intro.intro_id, {"religion", "employer"}, {"religion", "income_band"}, board)
        self.assertEqual([(r.key, r.values) for r in out], [("religion", ["Buddhist"])])

    def test_nothing_revealed_before_both_say_yes(self):
        w, board, intro = self._connected()
        w.record_decision(intro.intro_id, "pa", True)
        w.record_decision(intro.intro_id, "pb", False)
        self.assertEqual(w.exit_reveal(intro.intro_id, {"religion"}, {"religion"}, board), [])


class Receipts(unittest.TestCase):
    def test_receipts_have_fixed_fields_and_no_content(self):
        w, board = roommates_board()
        w.exit_introduction(proposal(), board)
        for r in w.receipts:
            d = r.to_dict()
            self.assertEqual(set(d), {"seq", "check", "result", "timestamp", "commitment", "hash"})
            for secret in ("Buddhist", "Nintendo", "Yuki", "Osaka"):
                self.assertNotIn(secret, json.dumps(d, ensure_ascii=False))

    def test_anyone_can_verify_the_receipt_chain(self):
        w, board = roommates_board()
        w.exit_introduction(proposal(), board)
        log = [r.to_dict() for r in w.receipts]
        self.assertEqual(verify_chain(log), (True, -1))

    def test_deleting_or_editing_a_receipt_breaks_the_chain(self):
        w, board = roommates_board()
        w.exit_introduction(proposal(), board)
        log = [r.to_dict() for r in w.receipts]
        edited = [dict(r) for r in log]
        edited[3]["result"] = "pass" if edited[3]["result"] == "fail" else "fail"
        self.assertEqual(verify_chain(edited), (False, 3))
        self.assertEqual(verify_chain(log[:4] + log[5:]), (False, 4))
        # The independent audit notices too.
        report = leak_report(board, [], [], [], log[:4] + log[5:], [], set(), {}, 20, 3)
        self.assertIn("A6", {f["audit"] for f in report["findings"]})


if __name__ == "__main__":
    unittest.main()


class ReasonWording(unittest.TestCase):
    def test_a_match_is_told_who_they_are_not_what_they_want(self):
        from belong.warden import reason_text
        text = reason_text([("city", "Tokyo", "Tokyo"),
                            ("seeking", "an AC repair technician", "an AC repair customer"),
                            ("offering", "an AC repair customer", "an AC repair technician")], ("Japan", "Japan"))
        self.assertEqual(text, "You're both in Tokyo. They're an AC repair technician.")
