"""Taking over a synthetic person (the inspector's "Take over"): they get a phone held to the same
rules as A and B. Their phone shows only what they may know, their listing is matched like a phone's,
their answers come from the phone and are never simulated, and other phones still see them only
through the Warden's introductions."""
from __future__ import annotations

import json
import os
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from unittest import mock

from belong import server
from belong.demo import DemoError, DemoSession
from belong.schema import HIDDEN, SHOWN
from belong.sim import run
from belong.synth import generate
from tests.test_demo import fill


def climber_a(s: DemoSession) -> None:
    """Person A looks for a climbing partner in Osaka; the Matchmaker introduces synthetic climbers."""
    s.verify("A", "Yuki", 27)
    s.agent_message("A", "Looking for a climbing partner in Osaka on weekends. I'm an intermediate climber, "
                         "into bouldering and ramen.")
    s.submit("A", fill(s.slots["A"].draft, hidden={"religion": "Buddhist", "me:interests": ["bouldering", "ramen"]}))


def synthetic_match_of_a(s: DemoSession) -> tuple:
    """(intro_id, persona id) of A's introduction to the synthetic person with the most hidden details."""
    pairs = [(i.intro_id, i.person_b if i.person_a == "demoA-1" else i.person_a) for i in s.intros.values()
             if "demoA-1" in (i.person_a, i.person_b) and not i.one_way_to]
    pairs = [(iid, pid) for iid, pid in pairs if pid in s.personas]
    return max(pairs, key=lambda x: len(s.board.listing(x[1]).hidden()))


def strings(listing, visibility=None) -> set:
    return {v for f in listing.fields if visibility in (None, f.visibility) for v in f.strings()}


def without_drafts(phone: dict) -> str:
    """Everything on a phone except its own person's drafts (whose drop-downs list every possible value)."""
    phone = dict(phone)
    phone.pop("draft")
    phone["listings"] = [{k: v for k, v in l.items() if k != "draft"} for l in phone["listings"]]
    return json.dumps(phone, ensure_ascii=False)


class BatchRunPeopleAreInTheDemo(unittest.TestCase):
    """The inspector shows the batch run (300 people, seed 7); the live demo serves 5,000 with the same seed.
    Synthetic people are generated one after another, so the run's people are the demo's first 300."""

    def test_the_first_300_of_5000_are_the_batch_runs_people(self):
        small, large = generate(300, 7), generate(5000, 7)
        for a, b in zip(small, large):
            self.assertEqual((a.person.id, a.intent_text, a.values), (b.person.id, b.intent_text, b.values))

    def test_every_batch_run_person_is_in_the_demo_with_the_same_intent(self):
        result = run(n=300, seed=7)
        s = DemoSession(seed=7, agent=None)
        for row in result["people"]:
            pid = row["id"]
            self.assertIn(pid, s.personas)
            self.assertEqual(row["admitted"], pid in s.board, pid)
            if pid in s.board:
                live = s.board.listing(pid).intent_text
                # The same words, unless the run's follow-up answer added to them (" Aged 30 to 39.").
                self.assertTrue(row["intent_text"] == live or row["intent_text"].startswith(live + " "), pid)


class TakeOver(unittest.TestCase):
    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)

    def test_the_phone_shows_the_persons_own_listing(self):
        pid = next(p for p in self.s.personas if p in self.s.board and self.s.board.listing(p).hidden())
        slot = self.s.take_over(pid)
        self.assertRegex(slot, r"^[A-Za-z0-9]{1,16}$")  # safe in a URL and in HTML
        live = self.s.board.listing(pid)
        phone = self.s.state()["phones"][slot]
        self.assertTrue(phone["verified"] and phone["taken_over"])
        self.assertEqual(phone["name"], live.get("first_name").value)
        self.assertEqual(phone["age_band"], self.s.personas[pid].person.age_band)
        [mine] = phone["listings"]
        self.assertEqual((mine["id"], mine["listed"], mine["unsaved"], phone["current"]), (pid, True, False, pid))
        self.assertEqual(mine["title"], self.s._live_title(live))
        self.assertEqual(mine["prefs"], {"notify_close": live.notify_close, "notify_questions": live.notify_questions})
        self.assertEqual({h["key"] for h in mine["hidden_keys"]}, set(live.hidden()))
        # Their own profile shows them every detail they have, hidden ones marked hidden.
        draft = {f["key"]: f for f in mine["draft"]["fields"] if f["enabled"] and f["value"] not in (None, "", [])}
        for f in live.fields:
            self.assertEqual((draft[f.key]["visibility"], draft[f.key]["value"]),
                             (f.visibility, f.value if isinstance(f.value, str) else list(f.value)))
        self.assertEqual(mine["draft"]["intent_text"], live.intent_text)
        # Their open details as their agent wrote them, each named on their phone (never by its raw key).
        opened = [f for f in live.fields if f.about]
        self.assertTrue(opened)
        for f in opened:
            row = draft[f.key]
            self.assertEqual((row["label"], row["compare"], row["steps"], row["required"]),
                             (f.name[:1].upper() + f.name[1:], f.compare, list(f.steps), f.required))
        self.assertTrue(all(":" not in h["label"] for h in mine["hidden_keys"]))
        self.assertEqual(self.s.phone_state("A")["listings"], [])  # A and B are untouched
        self.assertEqual(self.s.board.listing(pid), live)  # nothing re-entered or changed

    def test_refuses_demo_phones_and_anyone_not_live(self):
        rejected = next(p for p in self.s.personas if p not in self.s.board)
        for bad in ("A", "B", "demoA-1", "nobody", "", None, ["p000"], {"id": "p000"}, rejected, "p000<b>"):
            with self.assertRaises(DemoError, msg=repr(bad)):
                self.s.take_over(bad)
        self.assertEqual(set(self.s.slots), {"A", "B"})

    def test_taking_over_twice_returns_the_same_phone(self):
        pid = next(p for p in self.s.personas if p in self.s.board)
        self.assertEqual(self.s.take_over(pid), self.s.take_over(pid))
        self.assertEqual(len(self.s.slots[pid].listings), 1)

    def test_a_taken_over_person_cant_be_verified_again(self):
        pid = next(p for p in self.s.personas if p in self.s.board)
        slot = self.s.take_over(pid)
        with self.assertRaises(DemoError):
            self.s.verify(slot, "Someone", 60)  # their age band is on their live listing

    def test_introductions_already_sent_to_them_come_with_the_phone(self):
        climber_a(self.s)
        self.s.run_matchmaker()
        intro_id, pid = synthetic_match_of_a(self.s)
        slot = self.s.take_over(pid)
        seen = {i["intro_id"]: i for i in self.s.phone_state(slot)["intros"]}
        self.assertEqual(seen[intro_id]["status"], "pending")
        intro = self.s.intros[intro_id]
        self.assertEqual(seen[intro_id]["view"], intro.a_view if intro.person_a == pid else intro.b_view)


class TakenOverPersonAnswersForThemselves(unittest.TestCase):
    """Once taken over, a person's listing is matched like a phone's and nothing is decided for them."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climber_a(self.s)
        self.s.run_matchmaker()
        self.intro, self.pid = synthetic_match_of_a(self.s)
        self.slot = self.s.take_over(self.pid)

    def test_the_listing_is_matched_like_a_phones(self):
        self.assertIn(self.pid, self.s._demo_listing_ids())
        before = len(self.s.intros)
        self.s.run_matchmaker()
        new = [i for i in list(self.s.intros.values())[before:] if self.pid in (i.person_a, i.person_b)]
        self.assertTrue(new)  # matched in its own right now, not only as someone A was introduced to
        ids = {i["intro_id"] for i in self.s.phone_state(self.slot)["intros"]}
        self.assertTrue({i.intro_id for i in new if not i.one_way_to or i.one_way_to == self.pid} <= ids)
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])

    def test_their_answer_comes_from_the_phone(self):
        self.s.decide("A", self.intro, True)
        self.assertNotIn((self.intro, self.pid), self.s.decisions)  # no simulated answer for them
        status = lambda slot: next(i for i in self.s.phone_state(slot)["intros"] if i["intro_id"] == self.intro)["status"]
        self.assertEqual(status("A"), "waiting")
        self.s.decide(self.slot, self.intro, True)
        self.assertEqual((status("A"), status(self.slot)), ("connected", "connected"))
        found_by_a = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == self.intro)
        self.assertFalse(found_by_a["simulated"])  # they can reply now
        self.s.send_message(self.slot, self.intro, "Saturday at the gym?")
        found_by_a = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == self.intro)
        self.assertEqual([(m["mine"], m["text"]) for m in found_by_a["thread"]], [(False, "Saturday at the gym?")])

    def test_no_simulated_answer_when_comparing_details(self):
        self.s.decide("A", self.intro, True)
        self.s.decide(self.slot, self.intro, True)
        self.s.reveal("A", self.intro, ["religion", "me:interests"])
        self.assertNotIn((self.intro, self.pid), self.s.consents)  # they choose for themselves
        self.s.reveal(self.slot, self.intro, list(self.s.board.listing(self.pid).hidden()))
        self.assertIsNotNone(next(i for i in self.s.phone_state("A")["intros"]
                                  if i["intro_id"] == self.intro)["reveal_result"])
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])

    def test_other_phones_never_see_their_hidden_details(self):
        self.s.decide("A", self.intro, True)
        self.s.decide(self.slot, self.intro, True)
        self.s.run_matchmaker()
        theirs, a = self.s.board.listing(self.pid), self.s.board.listing("demoA-1")
        # Words A may legitimately have: A's own, and anything shown by people A was introduced to.
        a_may_know = strings(a) | {v for i in self.s.intros.values() if "demoA-1" in (i.person_a, i.person_b)
                                   for pid in (i.person_a, i.person_b) for v in strings(self.s.board.listing(pid), SHOWN)}
        secrets = strings(theirs, HIDDEN) - a_may_know
        self.assertTrue(secrets)
        seen_by_a = without_drafts(self.s.phone_state("A")) + without_drafts(self.s.phone_state("B"))
        for secret in secrets:
            self.assertNotIn(secret, seen_by_a)
        view = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == self.intro)["view"]
        self.assertEqual(view["fields"].keys() & theirs.hidden().keys(), set())
        # And the taken-over phone never sees A's hidden details either.
        their_words = strings(theirs) | {v for i in self.s.intros.values() if self.pid in (i.person_a, i.person_b)
                                         for pid in (i.person_a, i.person_b) for v in strings(self.s.board.listing(pid), SHOWN)}
        for secret in strings(a, HIDDEN) - their_words:
            self.assertNotIn(secret, without_drafts(self.s.phone_state(self.slot)))
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])


class CloseMatchToSomeoneTakenOver(unittest.TestCase):
    def test_a_close_match_to_them_waits_for_their_own_answer(self):
        """A one-way close match goes to them alone; nothing answers it for them, and the other person
        sees nothing until they say yes."""
        s = DemoSession(seed=7, agent=None)
        pid = next(p for p in s.personas if p in s.board)
        slot = s.take_over(pid)
        name = s.slots[slot].name
        s.set_prefs(slot, pid, {"notify_close": True, "notify_questions": False})
        s.submit(slot, {"intent_text": "Need my air conditioner fixed in Tokyo this weekend.", "exactness": 0.6,
                        "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in
                                   {"first_name": name, "country": "Japan", "city": "Tokyo",
                                    "seeking": "an air conditioner repair technician",
                                    "offering": "an air conditioner repair customer"}.items()]}, pid)
        s.verify("B", "Ken", 42)
        ken = s.new_listing("B")
        s.submit("B", {"intent_text": "I'm a repairman in Tokyo looking for repair jobs.", "exactness": 0.6,
                       "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in
                                  {"first_name": "Ken", "country": "Japan", "city": "Tokyo",
                                   "seeking": "a repair customer", "offering": "a repairman"}.items()]}, ken)
        s.run_matchmaker()
        intro = next(i for i in s.intros.values() if {i.person_a, i.person_b} == {pid, ken})
        self.assertEqual(intro.one_way_to, pid)
        self.assertNotIn((intro.intro_id, pid), s.decisions)
        mine = next(i for i in s.phone_state(slot)["intros"] if i["intro_id"] == intro.intro_id)
        self.assertEqual((mine["status"], mine["close"]), ("pending", True))
        self.assertNotIn(intro.intro_id, [i["intro_id"] for i in s.phone_state("B")["intros"]])
        s.decide(slot, intro.intro_id, True)
        self.assertNotIn((intro.intro_id, ken), s.decisions)  # Ken is a phone too: he answers himself
        self.assertEqual(next(i for i in s.phone_state("B")["intros"] if i["intro_id"] == intro.intro_id)["status"],
                         "pending")
        self.assertTrue(s.room_state()["leak_report"]["ok"])


class HandBack(unittest.TestCase):
    def test_handing_back_closes_the_phone_and_taking_over_again_restores_it(self):
        s = DemoSession(seed=7, agent=None)
        climber_a(s)
        s.run_matchmaker()
        intro, pid = synthetic_match_of_a(s)
        slot = s.take_over(pid)
        s.slots[slot].listing().chat.append({"from": "me", "text": "Hello agent"})
        s.hand_back(slot)
        self.assertNotIn(slot, s.state()["phones"])
        self.assertNotIn(pid, s._demo_listing_ids())
        with self.assertRaises(DemoError):
            s.decide(slot, intro, True)
        s.decide("A", intro, True)
        self.assertIn((intro, pid), s.decisions)  # simulated again: they answered for themselves
        self.assertEqual(s.take_over(pid), slot)
        self.assertIn("Hello agent", [m["text"] for m in s.phone_state(slot)["chat"]])  # the phone as it was
        for bad in ("A", "B", "nobody"):
            with self.assertRaises(DemoError):
                s.hand_back(bad)

    def test_what_they_chose_while_simulated_comes_back_with_the_phone(self):
        s = DemoSession(seed=7, agent=None)
        climber_a(s)
        s.run_matchmaker()
        intro, pid = synthetic_match_of_a(s)
        slot = s.take_over(pid)
        s.decide("A", intro, True)
        s.decide(slot, intro, True)
        s.hand_back(slot)
        s.reveal("A", intro, ["religion", "me:interests"])  # they choose for themselves while simulated
        self.assertIn((intro, pid), s.consents)
        s.take_over(pid)
        mine = next(i for i in s.phone_state(slot)["intros"] if i["intro_id"] == intro)
        self.assertTrue(mine["reveal_chosen"])
        self.assertIsNotNone(mine["reveal_result"])
        with self.assertRaises(DemoError):
            s.reveal(slot, intro, ["religion"])  # chosen already: never settled twice
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_reset_hands_everyone_back(self):
        s = DemoSession(seed=7, agent=None)
        pid = next(p for p in s.personas if p in s.board)
        s.take_over(pid)
        s.reset()
        self.assertEqual(set(s.state()["phones"]), {"A", "B"})


class Server(unittest.TestCase):
    """POST /api/take_over answers with the new phone; actions accept only phones that exist."""

    def setUp(self):
        self.session = DemoSession(seed=7, agent=None)
        self.patches = [mock.patch.object(server, "SESSION", self.session),
                        mock.patch.dict(os.environ, {"BELONG_PASSWORD": ""})]
        for p in self.patches:
            p.start()
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), server._Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        for p in reversed(self.patches):
            p.stop()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req) as r:
                return r.status, json.loads(r.read() or b"null")
        except urllib.error.HTTPError as e:
            with e:
                raw = e.read()
                return e.code, json.loads(raw) if raw.startswith(b"{") else None

    def test_take_over_answers_with_the_phone_and_only_real_phones_are_accepted(self):
        pid = next(p for p in self.session.personas if p in self.session.board)
        status, out = self.post("/api/take_over", {"persona_id": pid})
        self.assertEqual((status, out["slot"]), (200, pid))
        self.assertIn(pid, out["phones"])
        self.assertEqual(self.post("/api/open_listing", {"slot": pid, "listing_id": pid})[0], 200)
        for slot in ("p999999", "C", "", ["A"], {"x": 1}, "../A", "demoA"):
            status, out = self.post("/api/open_listing", {"slot": slot, "listing_id": pid})
            self.assertEqual(status, 400, slot)
        status, out = self.post("/api/take_over", {"persona_id": "A"})
        self.assertEqual(status, 400)
        self.assertNotIn("slot", out)
        self.assertEqual(self.post("/api/hand_back", {"slot": pid})[0], 200)
        self.assertNotIn(pid, self.session.slots)

    def test_the_connector_session_stays_two_people(self):
        pid = next(p for p in self.session.personas if p in self.session.board)
        with mock.patch.object(server, "_CHAT", DemoSession(seed=7, agent=None)):
            self.assertEqual(self.post("/api/chat/take_over", {"persona_id": pid})[0], 404)
            self.assertEqual(self.post("/api/chat/hand_back", {"slot": pid})[0], 404)
            self.assertEqual(set(server._CHAT.slots), {"A", "B"})


if __name__ == "__main__":
    unittest.main()


class ReviewFixes(unittest.TestCase):
    """Problems the review found, each pinned by a test."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)

    def test_someone_who_shows_a_sensitive_detail_can_save_their_profile(self):
        who = next(pid for pid in self.s.personas if pid in self.s.board
                   and self.s.board.listing(pid).confirmed_shown and self.s.personas[pid].person.verified)
        slot = self.s.take_over(who)
        mine = self.s.slots[slot].listing(who)
        d = mine.draft
        body = {"intent_text": d["intent_text"], "exactness": d["exactness"],
                "fields": [{"key": f["key"], "value": f["value"], "visibility": f["visibility"], "confirmed": bool(f.get("confirmed"))}
                           for f in d["fields"] if f.get("enabled") and f.get("value") not in (None, "", [])]}
        self.s.submit(slot, body, who)
        self.assertTrue(mine.entry["accepted"], mine.entry)

    def test_a_reply_being_written_is_dropped_once_theyre_taken_over(self):
        climber_a(self.s)
        self.s.run_matchmaker(quiet=True)
        intro_id, other = synthetic_match_of_a(self.s)
        before = len(self.s.threads.get(intro_id, []))
        self.s.persona_typing.add(intro_id)
        self.s.take_over(other)
        self.s._persona_reply(intro_id, other, self.s.generation)  # the reply that was in flight lands now
        self.assertEqual(len(self.s.threads.get(intro_id, [])), before)
        self.assertNotIn(intro_id, self.s.persona_typing)

    def test_a_question_already_put_to_the_other_phone_isnt_asked_of_them_too(self):
        climber_a(self.s)
        a = self.s.slots["A"].listing()
        who = next(pid for pid in self.s.personas if pid in self.s.board and self.s.personas[pid].person.verified)
        a.followups.append({"question_id": "also_offer", "question_text": "Someone is …", "about": who})
        slot = self.s.take_over(who)
        self.assertIn(("also_offer", a.id), self.s.slots[slot].listing(who).asked)

    def test_handing_back_settles_what_was_waiting_on_them(self):
        climber_a(self.s)
        self.s.run_matchmaker(quiet=True)
        intro_id, other = synthetic_match_of_a(self.s)
        self.s.take_over(other)
        self.s.decide("A", intro_id, True)  # A said yes; they're on a phone now, so nobody answers for them
        self.assertNotIn((intro_id, other), self.s.decisions)
        self.s.hand_back(other)
        self.assertIn((intro_id, other), self.s.decisions)  # simulated again, so they answer
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])
