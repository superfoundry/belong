"""Account-wide settings for close matches and questions (TRUST.md #24, #25). Both are off until the person turns one
on. A new listing's questions start from "Always allow questions"; "Always include close matches" turns a listing's
close matches on by itself, only where Belong would ask whether to include them (no match yet, once ever, by the
Warden's own checks for that question), through the usual update the Warden re-checks, and the person is told. The
Warden still enforces each listing's own choices. And the phone (demo.html, run with node): the Settings page's two
switches, the listing menu's two switches, and the menu's delete confirmation."""
from __future__ import annotations

import json
import re
import unittest

from belong import server
from belong.demo import NOTICES, SETTINGS, DemoError, DemoSession
from belong.matching.contract import FollowUp
from belong.schema import SHOWN
from tests.test_demo import _Page, climbers

# Listings of several kinds that nobody on the demo's board is looking for: each has no match at first.
ALONE = {
    "a lesson": ("Looking for a bookbinding mentor in Tromso.",
                 {"country": "Norway", "city": "Tromso", "seeking": "a bookbinding mentor",
                  "offering": "a bookbinding apprentice"}),
    "something to give away": ("Giving away an upright piano in Hobart.",
                               {"country": "Australia", "city": "Hobart", "seeking": "an upright piano taker",
                                "offering": "an upright piano giver"}),
    "a date": ("Hoping to meet a man in Ushuaia.",
               {"country": "Argentina", "city": "Ushuaia", "kind": "dating", "gender": "woman",
                "wants_gender": ["man"], "wants_age": "40-49"}),
}
CLOSE_NOTICE = NOTICES["close"][0]


def session() -> DemoSession:
    s = DemoSession(seed=7, agent=None)
    s.run_inline = True
    return s


def go_live(s: DemoSession, slot: str, name: str, intent: str, fields: dict) -> str:
    lid = s.new_listing(slot)
    s.submit(slot, {"intent_text": intent, "exactness": 0.6,
                    "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in {"first_name": name, **fields}.items()]},
             lid)
    assert s.slots[slot].listing(lid).entry["accepted"], s.slots[slot].listing(lid).entry
    return lid


def alone(s: DemoSession, kind: str, settings=None, slot="A") -> str:
    """Someone with one of the ALONE listings, live, with these account settings set first."""
    s.verify(slot, "Ines", 44)
    if settings is not None:
        s.set_settings(slot, settings)
    return go_live(s, slot, "Ines", *ALONE[kind])


def prefs_of(s: DemoSession, slot: str, lid: str) -> dict:
    return next(l["prefs"] for l in s.phone_state(slot)["listings"] if l["id"] == lid)


def live_prefs(s: DemoSession, lid: str) -> dict:
    live = s.board.listing(lid)
    return {"notify_close": live.notify_close, "notify_questions": live.notify_questions}


def proposing(s: DemoSession, *followups):
    """A Matchmaker that also proposes these questions every run, whatever the board says (a compromised one)."""
    real = s.matcher.match

    def match(board, focus=None, load=None):
        result = real(board, focus=focus, load=load)
        result.followups.extend(followups)
        return result
    s.matcher.match = match


OFF = {"notify_close": False, "notify_questions": False}


class Settings(unittest.TestCase):
    """Both off until the person turns one on; they set a new listing's choices and nothing else."""

    def test_both_start_off(self):
        s = session()
        s.verify("A", "Ines", 44)
        self.assertEqual(s.phone_state("A")["settings"], {"always_close": False, "always_questions": False})
        self.assertEqual(set(SETTINGS), {"always_close", "always_questions"})
        lid = s.new_listing("A")
        self.assertEqual(prefs_of(s, "A", lid), OFF)

    def test_a_new_listing_takes_them(self):
        """Questions start as "Always allow questions" says. Close matches always start off, so a listing looks for
        exact matches first (the automatic switch is for later: AlwaysIncludeCloseMatches)."""
        s = session()
        s.verify("A", "Ines", 44)
        s.set_settings("A", {"always_questions": True})
        first = go_live(s, "A", "Ines", *ALONE["a lesson"])
        self.assertEqual(prefs_of(s, "A", first), {"notify_close": False, "notify_questions": True})
        self.assertEqual(live_prefs(s, first), {"notify_close": False, "notify_questions": True})
        s.set_settings("A", {"always_close": True})
        second = go_live(s, "A", "Ines", *ALONE["something to give away"])
        self.assertEqual(prefs_of(s, "A", second), OFF)
        self.assertEqual(prefs_of(s, "A", first), {"notify_close": False, "notify_questions": True})  # its own choice

    def test_a_listing_just_started_takes_them_as_they_are_now(self):
        """A listing started but not yet written to has no choices of its own yet: it follows the settings as they
        change. Once the person has said something in it, it keeps its own."""
        s = session()
        s.verify("A", "Ines", 44)
        empty = s.new_listing("A")
        s.set_settings("A", {"always_questions": True})
        self.assertEqual(prefs_of(s, "A", empty), {"notify_close": False, "notify_questions": True})
        s.agent_message("A", "Looking for someone to share a rowing boat with in Hobart.")
        s.set_settings("A", {})
        self.assertEqual(prefs_of(s, "A", empty), {"notify_close": False, "notify_questions": True})
        self.assertEqual(prefs_of(s, "A", s.new_listing("A")), OFF)

    def test_they_never_change_a_listing_already_made(self):
        s = session()
        climbers(s)
        before = live_prefs(s, "demoA-1"), s.board.listing("demoA-1").version
        s.set_settings("A", {"always_close": True, "always_questions": True})
        self.assertEqual((live_prefs(s, "demoA-1"), s.board.listing("demoA-1").version), before)
        self.assertEqual(prefs_of(s, "A", "demoA-1"), OFF)

    def test_a_listing_with_choices_of_its_own_keeps_them(self):
        """Once a listing's own choices are set on it, the account settings never reset them, even while nothing has
        been said in it yet, and starting a new listing that reuses it keeps them too."""
        s = session()
        s.verify("A", "Ines", 44)
        empty = s.new_listing("A")
        s.set_prefs("A", empty, {"notify_close": False, "notify_questions": True})
        s.set_settings("A", {"always_close": True})
        self.assertEqual(prefs_of(s, "A", empty), {"notify_close": False, "notify_questions": True})
        s.set_settings("A", {"always_questions": True})
        s.set_prefs("A", empty, OFF)
        s.set_settings("A", {"always_questions": True, "always_close": True})
        self.assertEqual(prefs_of(s, "A", empty), OFF)
        self.assertEqual(s.new_listing("A"), empty)  # reused, as an untouched listing is
        self.assertEqual(prefs_of(s, "A", empty), OFF)

    def test_only_the_two_switches_and_only_a_plain_yes_turns_one_on(self):
        s = session()
        s.verify("A", "Ines", 44)
        for sent in ({"always_close": "yes", "always_questions": 1, "notify_close": True}, "on", None, ["always_close"]):
            with self.subTest(sent=sent):
                s.set_settings("A", sent)
                self.assertEqual(s.phone_state("A")["settings"], {"always_close": False, "always_questions": False})
        with self.assertRaises(DemoError):
            s.set_settings("B", {"always_close": True})  # nobody verified on that phone

    def test_each_phone_sees_only_its_own(self):
        s = session()
        s.verify("A", "Ines", 44)
        s.verify("B", "Tariq", 37)
        server.ACTIONS["settings"](s, {"slot": "A", "settings": {"always_close": True, "always_questions": True}})
        self.assertEqual(s.phone_state("A")["settings"], {"always_close": True, "always_questions": True})
        self.assertEqual(s.phone_state("B")["settings"], {"always_close": False, "always_questions": False})
        self.assertNotIn("settings", s.room_state())


class AlwaysIncludeCloseMatches(unittest.TestCase):
    """Off: no match yet asks once, and a yes turns on close matches only. On: close matches turn on by themselves at
    that moment instead, once, by the Warden's checks for the question, and the person is told."""

    def test_off_the_question_is_asked_and_a_yes_turns_on_close_matches_only(self):
        for kind in ALONE:
            for questions in (False, True):
                with self.subTest(kind, questions=questions):
                    s = session()
                    lid = alone(s, kind, {"always_questions": questions})
                    s.run_matchmaker()
                    l = s.phone_state("A")["listings"][0]
                    self.assertEqual([q["question_id"] for q in l["followups"]], ["broaden"])
                    self.assertNotIn(CLOSE_NOTICE, [m["text"] for m in l["chat"]])
                    s.answer_followup("A", "broaden", {"yes": True}, lid)
                    want = {"notify_close": True, "notify_questions": questions}  # questions follow their own switch
                    self.assertEqual((prefs_of(s, "A", lid), live_prefs(s, lid)), (want, want))
                    said = s.phone_state("A")["listings"][0]["chat"][-1]["text"]
                    self.assertIn("close matches", said)
                    self.assertNotIn("question", said)
                    self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_on_close_matches_turn_on_by_themselves_once_there_is_no_match(self):
        for kind in ALONE:
            with self.subTest(kind):
                s = session()
                lid = alone(s, kind, {"always_close": True})
                self.assertEqual(live_prefs(s, lid), OFF)  # exact matches first
                s.run_matchmaker()
                l = s.phone_state("A")["listings"][0]
                want = {"notify_close": True, "notify_questions": False}
                self.assertEqual((l["prefs"], live_prefs(s, lid)), (want, want))
                self.assertEqual(l["followups"], [])  # not asked: turned on instead
                self.assertEqual(l["intros"], [])
                self.assertEqual(l["chat"][-1], {"from": "agent", "text": CLOSE_NOTICE, "notice": "close", "count": 1})
                self.assertIn("gear on this listing", CLOSE_NOTICE)  # and how to turn them off
                # Only once the Warden's checks for the question passed, then the update was re-checked like any.
                event = next(e for e in s.events if e["title"] == "Close matches for Ines, by their settings")
                checks = {r["check"]: r["result"] for r in event["receipts"]}
                for check in ("broaden_still_off", "broaden_no_match_yet", "broaden_once", "update_rate_limit"):
                    self.assertEqual(checks.get(check), "pass", check)
                self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_not_while_there_is_a_match(self):
        """Two people who meet each other exactly: their close matches stay off, and nobody is told anything about
        close matches. A Matchmaker that proposes the switch anyway is refused by the Warden."""
        s = session()
        for slot, name, age in (("A", "Yuki", 27), ("B", "Ren", 29)):
            s.verify(slot, name, age)
            s.set_settings(slot, {"always_close": True})
        climbers(s)
        s.run_matchmaker()
        self.assertTrue(s.phone_state("A")["listings"][0]["intros"])
        proposing(s, FollowUp("demoA-1", "broaden"))
        s.clock.advance(hours=1)
        s.run_matchmaker()
        for slot, lid in (("A", "demoA-1"), ("B", "demoB-1")):
            self.assertEqual((prefs_of(s, slot, lid), live_prefs(s, lid)), (OFF, OFF))
            self.assertNotIn(CLOSE_NOTICE, [m["text"] for m in s.phone_state(slot)["listings"][0]["chat"]])
        refused = [r for r in s.warden.receipts if r.check == "broaden_no_match_yet"]
        self.assertEqual(refused[-1].result, "fail")
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_not_in_the_same_run_as_a_match(self):
        """A Matchmaker that proposes "include close matches?" in the same run as an exact match (handing them over in
        any order) is refused: the question, or the switch by the settings, waits until the run's introductions are
        out, so the Warden's "no match yet" check sees them. With the setting on or off, close matches stay off."""
        for always in (False, True):
            with self.subTest(always=always):
                s = session()
                s.verify("A", "Lucia", 39)
                s.verify("B", "Mateo", 31)
                s.set_settings("A", {"always_close": always})
                a = go_live(s, "A", "Lucia", "Teaching wheel pottery to a beginner in Valparaiso.",
                            {"country": "Chile", "city": "Valparaiso", "seeking": "a pottery student",
                             "offering": "a pottery teacher"})
                go_live(s, "B", "Mateo", "Looking for a pottery teacher in Valparaiso, I'm a beginner.",
                        {"country": "Chile", "city": "Valparaiso", "seeking": "a pottery teacher",
                         "offering": "a pottery student"})
                proposing(s, FollowUp(a, "broaden"))
                s.run_matchmaker()
                l = s.phone_state("A")["listings"][0]
                self.assertEqual([i["status"] for i in l["intros"]], ["pending"])
                self.assertEqual((l["followups"], prefs_of(s, "A", a), live_prefs(s, a)), ([], OFF, OFF))
                self.assertNotIn(CLOSE_NOTICE, [m["text"] for m in l["chat"]])
                last = [r for r in s.warden.receipts if r.check == "broaden_no_match_yet"][-1]
                self.assertEqual(last.result, "fail")
                self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_yes_the_warden_refuses_leaves_the_question_to_answer_again(self):
        """Past the day's updates a yes can't be put through: the question stays on the card (nothing changed, and it
        isn't used up), and a yes the next day turns close matches on."""
        s = session()
        lid = alone(s, "something to give away", {})
        for _ in range(20):
            try:
                s.set_prefs("A", lid, {"notify_close": False, "notify_questions": True})
            except DemoError:
                break
        else:
            self.fail("the day's update limit never came")
        s.run_matchmaker()
        with self.assertRaises(DemoError):
            s.answer_followup("A", "broaden", {"yes": True}, lid)
        l = s.phone_state("A")["listings"][0]
        self.assertEqual([q["question_id"] for q in l["followups"]], ["broaden"])
        self.assertEqual(live_prefs(s, lid), {"notify_close": False, "notify_questions": True})
        s.clock.advance(days=1)
        s.answer_followup("A", "broaden", {"yes": True}, lid)
        self.assertEqual(live_prefs(s, lid), {"notify_close": True, "notify_questions": True})
        self.assertEqual(s.phone_state("A")["listings"][0]["followups"], [])
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_never_twice_even_after_they_turn_them_off(self):
        s = session()
        lid = alone(s, "a lesson", {"always_close": True})
        s.run_matchmaker()
        s.set_prefs("A", lid, OFF)  # the gear on the listing
        proposing(s, FollowUp(lid, "broaden"))  # even a Matchmaker that keeps proposing it
        for _ in range(2):
            s.clock.advance(hours=1)
            s.run_matchmaker()
        l = s.phone_state("A")["listings"][0]
        self.assertEqual((l["prefs"], live_prefs(s, lid), l["followups"]), (OFF, OFF, []))
        self.assertEqual([m["text"] for m in l["chat"]].count(CLOSE_NOTICE), 1)
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_paused_listing_is_not_switched(self):
        s = session()
        lid = alone(s, "something to give away", {"always_close": True})
        s.set_active("A", lid, False)
        proposing(s, FollowUp(lid, "broaden"))
        s.run_matchmaker()
        self.assertEqual((prefs_of(s, "A", lid), live_prefs(s, lid)), (OFF, OFF))
        self.assertNotIn(CLOSE_NOTICE, [m["text"] for m in s.phone_state("A")["listings"][0]["chat"]])

    def test_questions_keep_their_own_switch_and_the_warden_still_refuses_them(self):
        s = session()
        lid = alone(s, "a date", {"always_close": True})
        s.run_matchmaker()
        self.assertEqual(live_prefs(s, lid), {"notify_close": True, "notify_questions": False})
        self.assertIsNone(s.warden.exit_followup(FollowUp(lid, "age_range"), s.board))
        self.assertEqual((s.warden.receipts[-1].check, s.warden.receipts[-1].result), ("followup_opted_in", "fail"))
        s.set_prefs("A", lid, {"notify_close": True, "notify_questions": True})  # their own switch
        self.assertIsNotNone(s.warden.exit_followup(FollowUp(lid, "age_range"), s.board))

    def test_one_way_close_matches_are_unchanged(self):
        """Someone whose close matches turned on by their settings hears about a close match, one-way: the other
        person isn't told, and isn't asked anything either (their questions are off)."""
        s = session()
        s.verify("B", "Ken", 42)
        b = go_live(s, "B", "Ken", "I'm a repairman in Tokyo looking for repair jobs.",
                    {"country": "Japan", "city": "Tokyo", "seeking": "a repair customer", "offering": "a repairman"})
        s.verify("A", "Mia", 30)
        s.set_settings("A", {"always_close": True})
        a = go_live(s, "A", "Mia", "Need my air conditioner fixed in Tokyo this weekend.",
                    {"country": "Japan", "city": "Tokyo", "seeking": "an air conditioner repair technician",
                     "offering": "an air conditioner repair customer"})
        s.run_matchmaker()
        self.assertEqual(live_prefs(s, a), {"notify_close": True, "notify_questions": False})
        self.assertFalse(s.intros)  # no exact match came first
        s.clock.advance(hours=1)
        before = json.dumps(s.phone_state("B"))
        s.run_matchmaker()
        intro = next(i for i in s.intros.values() if {i.person_a, i.person_b} == {a, b})
        self.assertEqual(intro.one_way_to, a)
        self.assertTrue(s.phone_state("A")["listings"][0]["intros"])
        # Their phone is as it was before the close match: no introduction, no news, and the one question they were
        # asked (their own "No matches yet", their settings being off) still there, never one about the other person.
        theirs = s.phone_state("B")["listings"][0]
        self.assertEqual(theirs["intros"], [])
        self.assertEqual([q["question_id"] for q in theirs["followups"]], ["broaden"])
        self.assertEqual([m["notice"] for m in theirs["chat"] if m.get("notice")], ["question"])
        # Nothing at all changed on it, not even the receipts of the Warden's checks, nor when the recipient's
        # listing changes and the Warden reissues the introduction.
        self.assertEqual(json.dumps(s.phone_state("B")), before)
        s.set_prefs("A", s.slots["A"].listing().id, {"notify_close": True, "notify_questions": True})
        self.assertEqual(json.dumps(s.phone_state("B")), before)
        self.assertTrue(s.room_state()["leak_report"]["ok"])
        # Once the recipient says yes, the other person is told, and from then on gets its receipts too.
        s.decide("A", intro.intro_id, True)
        self.assertEqual([i["status"] for i in s.phone_state("B")["listings"][0]["intros"]], ["pending"])
        seen = s.phone_state("B")["receipts"][-1]["seq"]
        s.set_prefs("A", s.slots["A"].listing().id, {"notify_close": True, "notify_questions": False})
        self.assertGreater(s.phone_state("B")["receipts"][-1]["seq"], seen)

    def test_if_the_warden_refuses_the_update_the_question_is_asked_instead(self):
        """Past the day's updates the Warden refuses turning close matches on: nothing changed, so the person is
        asked as they would be without the setting, and told it's there."""
        s = session()
        lid = alone(s, "a lesson", {"always_close": True})
        for _ in range(20):
            try:
                s.set_prefs("A", lid, {"notify_close": False, "notify_questions": True})
            except DemoError:
                break
        else:
            self.fail("the day's update limit never came")
        s.run_matchmaker()
        l = s.phone_state("A")["listings"][0]
        self.assertEqual((l["prefs"], live_prefs(s, lid)), ({"notify_close": False, "notify_questions": True},) * 2)
        self.assertEqual([q["question_id"] for q in l["followups"]], ["broaden"])
        self.assertEqual(l["chat"][-1]["notice"], "question")
        self.assertNotIn(CLOSE_NOTICE, [m["text"] for m in l["chat"]])
        self.assertTrue(s.room_state()["leak_report"]["ok"])


class InTheSimulation(unittest.TestCase):
    """The simulation's synthetic people answer the same question the same way: a yes turns on close matches only, and
    questions keep the choice the listing had (agent.answer_followup)."""

    def test_a_synthetic_yes_turns_on_close_matches_only(self):
        import random
        from dataclasses import replace
        from belong import agent
        s = session()
        lid = alone(s, "a lesson", {})
        seen = set()
        for questions in (False, True):
            listing = replace(s.board.listing(lid), notify_questions=questions)
            for seed in range(12):
                after = agent.answer_followup(None, listing, "broaden", s.clock.now(), random.Random(seed))
                seen.add(after.notify_close)
                self.assertEqual(after.notify_questions, questions)
        self.assertEqual(seen, {False, True})  # some said yes, some no


class OnThePhone(_Page, unittest.TestCase):
    """demo.html, run with node on real phone states: the Settings page's two switches (both off by default), the
    listing menu's own two, and its delete confirmation (asks first, Cancel first and focused, a draft's note never
    says it was being matched)."""

    @classmethod
    def setUpClass(cls):
        s = session()
        s.verify("A", "Ines", 44)
        cls.live_id = go_live(s, "A", "Ines", *ALONE["a lesson"])
        cls.draft_id = s.new_listing("A")
        s.prefill("A", "Looking for someone to share a rowing boat with in Hobart.")
        s.agent_message("A", "Looking for someone to share a rowing boat with in Hobart.")
        both = lambda: json.loads(json.dumps(s.state()))
        cls.two = both()
        assert [l["listed"] for l in cls.two["phones"]["A"]["listings"]] == [True, False]
        s.set_settings("A", {"always_questions": True})
        s.set_prefs("A", cls.live_id, {"notify_close": True, "notify_questions": False})
        cls.on = both()
        s.set_active("A", cls.live_id, False)
        cls.paused = both()
        s.delete_listing("A", cls.live_id)
        cls.deleted = both()

    def phone(self, state, then: str, answer=None):
        """Phone A on its Belong tab with this state, then the steps in `then` (they may await); every request the
        page makes is recorded, and answered with `answer` when given."""
        return self.run_page(f"""
const calls = [], focusedBy = [], reply = {json.dumps(answer)};
fetch = (url, o) => {{ calls.push([url, o ? JSON.parse(o.body) : null]);
  return reply ? Promise.resolve({{ json: () => Promise.resolve(reply) }}) : new Promise(() => {{}}); }};
const realQuery = document.querySelector;
document.querySelector = sel => /\\[data-focus\\]$/.test(sel) ? {{ focus() {{ focusedBy.push(sel); }} }} : realQuery(sel);
S = {json.dumps(state)}; go("A", "belong");
(async () => {{
  {then}
  console.log(JSON.stringify({{ calls, focused: focusedBy, after: phone("A"), out: typeof out === "undefined" ? null : out }}));
}})().catch(e => console.log(JSON.stringify({{ error: String(e && e.stack || e) }})));""")

    @staticmethod
    def menu(html: str) -> str:
        start = html.index('class="gear-pop')
        return html[start:html.index('class="listing-open"', start)]

    def test_the_menu_has_active_close_matches_questions_and_delete(self):
        out = self.phone(self.two, f'toggleGear("A", "{self.live_id}"); var out = phone("A");')
        menu = self.menu(out["out"])
        self.assertEqual(re.findall(r'<span class="pop-label">([^<]*)</span>', menu), ["Active", "Close matches", "Questions"])
        self.assertIn(f"askDelete('A','{self.live_id}')", menu)
        self.assertIn("Delete listing", menu)
        self.assertIn(f"setPref('A','{self.live_id}','notify_close',this.checked)", menu)
        self.assertIn(f"setPref('A','{self.live_id}','notify_questions',this.checked)", menu)
        self.assertEqual(menu.count("checked"), menu.count("this.checked") + 1)  # only Active is on

    def test_each_switch_changes_only_its_own_choice(self):
        """Each tap changes its own switch and sends the other as the screen shows it: a second switch tapped before
        the first is answered carries the first one's new value, never the older one the server last sent (which
        would undo the first tap, an opt-out turned back on)."""
        out = self.phone(self.on, f"""
setPref("A", "{self.live_id}", "notify_questions", true); setPref("A", "{self.live_id}", "notify_close", false);
toggleGear("A", "{self.live_id}"); var out = phone("A");""")
        self.assertEqual(out["calls"], [
            ["/api/set_prefs", {"slot": "A", "listing_id": self.live_id,
                                "prefs": {"notify_close": True, "notify_questions": True}}],
            ["/api/set_prefs", {"slot": "A", "listing_id": self.live_id,
                                "prefs": {"notify_close": False, "notify_questions": True}}]])
        menu = self.menu(out["out"])  # while they're on their way, the switches show what was tapped
        self.assertRegex(menu, r'data-switch="[^"]*:notify_questions" class="switch" checked')
        self.assertNotRegex(menu, r'data-switch="[^"]*:notify_close" class="switch" checked')

    def test_delete_asks_first_with_cancel_focused(self):
        out = self.phone(self.two, f"""
toggleGear("A", "{self.live_id}"); askDelete("A", "{self.live_id}"); var out = phone("A");""")
        ask = self.menu(out["out"])
        self.assertIn('role="alertdialog"', ask)
        self.assertIn("Delete this listing?", ask)
        self.assertIn("This can't be undone.", ask)
        self.assertEqual(re.findall(r"<button[^>]*data-focus[^>]*>([^<]*)</button>", ask), ["Cancel"])
        self.assertLess(ask.index(">Cancel<"), ask.index(">Delete<"))  # the safe choice first
        self.assertEqual(out["focused"], ["#phone-A .gear-pop [data-focus]"])
        self.assertEqual(out["calls"], [])  # nothing deleted by asking

    def test_cancel_goes_back_to_the_menu_and_deletes_nothing(self):
        out = self.phone(self.two, f"""
toggleGear("A", "{self.live_id}"); askDelete("A", "{self.live_id}"); cancelDelete("A");""")
        menu = self.menu(out["after"])
        self.assertNotIn("Delete this listing?", menu)
        self.assertIn('role="dialog"', menu)
        self.assertIn("Delete listing", menu)
        self.assertEqual(out["calls"], [])
        self.assertIn(f"openListing('A','{self.live_id}')", out["after"])

    def test_confirming_deletes_it_and_the_card_goes(self):
        out = self.phone(self.two, f"""
toggleGear("A", "{self.live_id}"); askDelete("A", "{self.live_id}"); await deleteListing("A", "{self.live_id}");""",
                         answer=self.deleted)
        self.assertNotIn("error", out)
        self.assertEqual(out["calls"], [["/api/delete_listing", {"slot": "A", "listing_id": self.live_id}]])
        self.assertNotIn(f"openListing('A','{self.live_id}')", out["after"])
        self.assertIn(f"openListing('A','{self.draft_id}')", out["after"])  # the other listing stays
        self.assertNotIn("gear-pop", out["after"])

    def test_a_drafts_note_never_says_it_was_being_matched(self):
        notes = {}
        for lid in (self.live_id, self.draft_id):
            out = self.phone(self.two, f'toggleGear("A", "{lid}"); askDelete("A", "{lid}"); var out = phone("A");')
            notes[lid] = re.search(r'<p class="pop-note" id="del-note-A">(.*?)</p>', self.menu(out["out"]), re.S).group(1)
        self.assertIn("It stops being matched", notes[self.live_id])
        self.assertIn("The draft and your chat with your agent about it go.", notes[self.draft_id])
        self.assertNotRegex(notes[self.draft_id].lower(), r"match")

    def test_a_paused_listings_note_never_says_it_is_being_matched(self):
        out = self.phone(self.paused, f'toggleGear("A", "{self.live_id}"); askDelete("A", "{self.live_id}"); var out = phone("A");')
        note = re.search(r'<p class="pop-note" id="del-note-A">(.*?)</p>', self.menu(out["out"]), re.S).group(1)
        self.assertEqual(note, "Its matches and chats leave your phone. This can't be undone.")
        self.assertNotIn("being matched", note)

    def test_one_tap_deletes_and_a_second_sends_nothing(self):
        """The confirmation goes at once on the first tap, so a quick second tap never sends another delete (which
        the server would answer with an error, after a delete that worked)."""
        out = self.phone(self.two, f"""
toggleGear("A", "{self.live_id}"); askDelete("A", "{self.live_id}");
deleteListing("A", "{self.live_id}"); var out = phone("A"); deleteListing("A", "{self.live_id}");""")
        self.assertEqual(out["calls"], [["/api/delete_listing", {"slot": "A", "listing_id": self.live_id}]])
        self.assertNotIn("Delete this listing?", out["out"])

    def test_escape_steps_back_and_focus_returns_where_it_was(self):
        """Escape on the confirmation goes back to the menu, and Cancel or Escape puts focus on its Delete listing row;
        Escape on the menu closes it and puts focus on its gear. Opening the menu puts focus on its first switch."""
        out = self.run_page(f"""
const focused = [], calls = [], realQuery = document.querySelector;
fetch = (url, o) => {{ calls.push(url); return new Promise(() => {{}}); }};
document.querySelector = sel => /(pop-row\\.danger|\\.gear\\[data-id=[^\\]]*\\]|input\\.switch:not\\(\\[disabled\\]\\))$/.test(sel)
  ? {{ focus() {{ focused.push(sel); }} }} : realQuery(sel);
const escape = {{ key: "Escape", preventDefault() {{}}, stopPropagation() {{}} }};
S = {json.dumps(self.two)}; go("A", "belong");
const id = {json.dumps(self.live_id)};
toggleGear("A", id); askDelete("A", id);
const asked = phone("A");
gearKey(escape, "A", id); const back = phone("A");
gearKey(escape, "A", id); const closed = phone("A");
console.log(JSON.stringify({{ asked, back, closed, focused, calls }}));""")
        self.assertIn(f'onkeydown="gearKey(event,\'A\',\'{self.live_id}\')"', self.menu(out["asked"]))
        self.assertIn("Delete listing", self.menu(out["back"]))
        self.assertNotIn("Delete this listing?", out["back"])
        self.assertNotIn("gear-pop", out["closed"])
        self.assertEqual(out["focused"], ["#phone-A .gear-pop input.switch:not([disabled])",
                                          "#phone-A .gear-pop .pop-row.danger", f'#phone-A .gear[data-id="{self.live_id}"]'])
        self.assertEqual(out["calls"], [])

    def test_two_settings_tapped_quickly_both_keep_their_tap(self):
        out = self.phone(self.on, 'menuGo("A", "settings"); setSetting("A", "always_close", true); '
                                  'setSetting("A", "always_questions", false); renderPhone("A"); var out = phone("A");')
        self.assertEqual([c[1]["settings"] for c in out["calls"]], [{"always_close": True, "always_questions": True},
                                                                   {"always_close": True, "always_questions": False}])
        self.assertRegex(out["out"], r'data-switch="always_close" class="switch" checked')
        self.assertNotRegex(out["out"], r'data-switch="always_questions" class="switch" checked')

    def test_the_settings_page_has_both_switches_off_by_default(self):
        out = self.phone(self.two, 'menuGo("A", "settings"); var out = phone("A");')
        page = out["out"]
        self.assertEqual(re.findall(r'<span class="pop-label">([^<]*)</span>', page),
                         ["Always include close matches", "Always allow questions"])
        self.assertEqual(len(re.findall(r'<span class="pop-note">[^<]+</span>', page)), 2)
        self.assertEqual(len(re.findall(r'class="switch"', page)), 2)
        self.assertNotRegex(page, r'class="switch" checked')
        self.assertIn("tap the gear on it", page)
        on = self.phone(self.on, 'menuGo("A", "settings"); var out = phone("A");')["out"]
        self.assertEqual(len(re.findall(r'class="switch" checked', on)), 1)
        self.assertRegex(on, r'class="switch" checked\s*onchange="setSetting\(\'A\',\'always_questions\'')

    def test_the_agent_line_says_what_this_persons_agent_runs_on(self):
        """A model, or rules (no key, BELONG_AGENT=rules, or a model that fell back): the phone says which, and never
        which Matchmaker runs."""
        self.assertEqual(self.two["phones"]["A"]["agent_kind"], "rules")  # this session's agent is the rule-based one
        rules = self.phone(self.two, 'menuGo("A", "settings"); var out = phone("A");')["out"]
        self.assertIn("In this demo your agent runs on fixed rules, not an AI model.", rules)
        self.assertNotIn("hosted AI model", rules)
        model = json.loads(json.dumps(self.two))
        model["phones"]["A"]["agent_kind"] = "model"
        hosted = self.phone(model, 'menuGo("A", "settings"); var out = phone("A");')["out"]
        self.assertIn("In this demo your agent runs on a hosted AI model.", hosted)
        self.assertNotIn("fixed rules", hosted)
        for page in (rules, hosted):
            self.assertNotIn("atchmaker", page)

    def test_a_settings_switch_sends_both_settings(self):
        out = self.phone(self.on, 'menuGo("A", "settings"); setSetting("A", "always_close", true);')
        self.assertEqual(out["calls"], [["/api/settings", {"slot": "A", "settings": {"always_close": True,
                                                                                   "always_questions": True}}]])

    def test_the_question_no_longer_promises_questions(self):
        state = json.loads(json.dumps(self.two))
        live = state["phones"]["A"]["listings"][0]
        live["followups"] = [{"question_id": "broaden", "question_text": "No matches yet. Want to include close matches?"}]
        out = self.phone(state, 'var out = phone("A");')["out"]
        self.assertIn("No matches yet. Want to include close matches?", out)
        self.assertIn("It's one-way: they aren't told about you.", out)
        self.assertNotIn("short question", out)


if __name__ == "__main__":
    unittest.main()
