"""The two-person demo: the full journey works, phones only learn what they may, and
a compromised Matchmaker is stopped."""
from __future__ import annotations

import json
import unittest

from belong.demo import DemoError, DemoSession
from belong.schema import HIDDEN, SHOWN


def fill(draft, values=None, hidden=None):
    """Take the agent's draft, set some values, and choose visibilities like a person would."""
    values, hidden = values or {}, hidden or {}
    fields = []
    for f in draft["fields"]:
        f = dict(f)
        if f["key"] in values:
            f["value"] = values[f["key"]]
        if f["key"] in hidden:
            f["value"] = hidden[f["key"]]
            f["visibility"] = HIDDEN
            f["enabled"] = True
        if f["value"] and f["enabled"]:
            fields.append(f)
    return {"intent_text": draft["intent_text"], "exactness": draft["exactness"], "fields": fields}


def climbers(s: DemoSession):
    s.verify("A", "Yuki", 27)
    s.agent_message("A", "Looking for a climbing partner in Osaka on weekends. I'm an intermediate climber, "
                         "into bouldering and ramen.")
    s.submit("A", fill(s.slots["A"].draft, hidden={"religion": "Buddhist", "smoking": "doesn't smoke"}))
    s.verify("B", "Ren", 29)
    s.agent_message("B", "Need someone to belay with at Osaka gyms, weekends. Intermediate level, "
                         "I also like jazz and ramen.")
    s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD"}))


class Journey(unittest.TestCase):
    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climbers(self.s)

    def _intro_between_a_and_b(self):
        for st in self.s.state()["phones"]["A"]["intros"]:
            intro = self.s.intros[st["intro_id"]]
            if {intro.person_a, intro.person_b} == {"demoA-1", "demoB-1"}:
                return st["intro_id"]
        return None

    def test_both_enter_and_are_introduced_to_each_other(self):
        self.assertTrue(self.s.slots["A"].entry["accepted"])
        self.assertTrue(self.s.slots["B"].entry["accepted"])
        self.s.run_matchmaker()
        self.assertIsNotNone(self._intro_between_a_and_b())

    def test_phones_never_receive_the_other_persons_hidden_details(self):
        self.s.run_matchmaker()

        def from_others(slot):
            # Everything on the phone except the person's own draft (whose drop-downs
            # list every possible value, e.g. all health options).
            state = dict(self.s.phone_state(slot))
            state.pop("draft")
            state["listings"] = [{k: v for k, v in l.items() if k != "draft"} for l in state["listings"]]
            return json.dumps(state, ensure_ascii=False)

        self.assertNotIn("ADHD", from_others("A"))  # B's hidden health detail (no one else shows it)
        self.assertNotIn("Buddhist", from_others("A") + from_others("B"))  # hidden by both
        # What B sees about A: none of A's hidden keys, only A's shown values.
        intro_id = self._intro_between_a_and_b()
        b_view = next(i["view"] for i in self.s.phone_state("B")["intros"] if i["intro_id"] == intro_id)
        self.assertEqual(b_view["fields"].keys() & {"smoking", "religion"}, set())

    def test_mutual_yes_then_mutual_reveal_of_a_shared_detail(self):
        self.s.run_matchmaker()
        intro_id = self._intro_between_a_and_b()
        self.s.decide("A", intro_id, True)
        self.assertEqual(next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["status"],
                         "waiting")
        self.s.decide("B", intro_id, True)
        self.s.reveal("A", intro_id, ["religion", "smoking"])
        self.assertIsNone(next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["reveal_result"])
        self.s.reveal("B", intro_id, ["religion", "health"])
        result = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["reveal_result"]
        self.assertEqual(result, [{"key": "religion", "values": ["Buddhist"]}])
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])

    def test_a_no_is_never_attributed(self):
        self.s.run_matchmaker()
        intro_id = self._intro_between_a_and_b()
        self.s.decide("A", intro_id, True)
        self.s.decide("B", intro_id, False)
        status = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["status"]
        self.assertEqual(status, "closed")

    def test_saying_no_removes_the_match_from_your_list(self):
        self.s.run_matchmaker()
        intro_id = self._intro_between_a_and_b()
        self.s.decide("B", intro_id, False)
        self.assertNotIn(intro_id, [i["intro_id"] for i in self.s.phone_state("B")["intros"]])
        self.assertIn(intro_id, [i["intro_id"] for i in self.s.phone_state("A")["intros"]])  # A hasn't answered yet

    def test_reveal_requires_a_connection(self):
        self.s.run_matchmaker()
        with self.assertRaises(DemoError):
            self.s.reveal("A", self._intro_between_a_and_b(), ["religion"])

    def test_compromised_matchmaker_is_blocked_three_times(self):
        self.s.attack()
        blocked = [e for e in self.s.events if e["kind"] == "blocked" and e["title"].startswith("Blocked:")]
        self.assertEqual(len(blocked), 3)
        self.assertFalse(any("ATTACK GOT THROUGH" in e["title"] for e in self.s.events))


class Entry(unittest.TestCase):
    def test_identifying_intent_is_refused_at_the_door(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Sam", 30)
        s.agent_message("A", "The barista at the Blue Bottle in Tokyo with a nose ring, please find her.")
        s.submit("A", fill(s.slots["A"].draft, values={"city": "Tokyo"}))
        self.assertFalse(s.slots["A"].entry["accepted"])
        self.assertEqual(s.events[-1]["kind"], "blocked")

    def test_sensitive_detail_typed_into_text_becomes_hidden(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Inês", 31)
        s.agent_message("A", "Looking for a serious relationship with a man in Lisbon, aged 30 to 39. "
                             "I'm a woman, Christian, love hiking.")
        religion = next(f for f in s.slots["A"].draft["fields"] if f["key"] == "religion")
        self.assertEqual((religion["value"], religion["visibility"], religion["enabled"]), ("Christian", HIDDEN, True))

    def test_showing_a_sensitive_detail_needs_confirmation(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Inês", 31)
        s.agent_message("A", "Looking for a roommate in Lisbon, moving in November.")
        body = fill(s.slots["A"].draft, values={"religion": "Christian"})
        for f in s.slots["A"].draft["fields"]:
            if f["key"] == "religion":
                body["fields"].append({**f, "value": "Christian", "visibility": SHOWN})
        s.submit("A", body)
        self.assertFalse(s.slots["A"].entry["accepted"])



class InstantProfile(unittest.TestCase):
    def test_an_instant_profile_can_start_matching_before_the_agent_replies(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        lid = s.new_listing("A", {"acknowledged": True})
        s.prefill("A", "I teach guitar in Osaka, Japan", lid)
        s.submit("A", fill(s.slots["A"].listing(lid).draft), lid)
        self.assertTrue(s.slots["A"].listing(lid).entry["accepted"], s.slots["A"].listing(lid).entry)


class UnsavedChanges(unittest.TestCase):
    """What matching reads is what's live; a draft that isn't saved yet is flagged, not shown as live."""

    def test_an_unsaved_draft_is_flagged_and_the_title_stays_live(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        live_title = s.phone_state("A")["listings"][0]["title"]
        self.assertFalse(s.phone_state("A")["listings"][0]["unsaved"])
        s.agent_message("A", "Actually I'm also into jazz and I'm a beginner.")
        state = s.phone_state("A")["listings"][0]
        self.assertTrue(state["unsaved"])
        self.assertEqual(state["title"], live_title)

    def test_saving_says_what_changed(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        d = s.slots["A"].draft
        body = fill(d, values={"city": "Kyoto"}, hidden={"religion": "Buddhist", "smoking": "doesn't smoke"})
        s.submit("A", body)
        last = s.slots["A"].chat[-1]["text"]
        self.assertIn("What changed: your city", last)
        body["intent_text"] = body["intent_text"] + " Also bouldering."
        s.submit("A", body)
        self.assertIn("your description", s.slots["A"].chat[-1]["text"])
        self.assertNotIn(" is now e", s.slots["A"].chat[-1]["text"])
        self.assertFalse(s.phone_state("A")["listings"][0]["unsaved"])


class Messages(unittest.TestCase):
    """After both say yes, the two people can talk; nobody else sees what they write."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climbers(self.s)
        self.s.run_matchmaker()
        self.intro = next(i["intro_id"] for i in self.s.phone_state("A")["intros"]
                          if {self.s.intros[i["intro_id"]].person_a, self.s.intros[i["intro_id"]].person_b}
                          == {"demoA-1", "demoB-1"})

    def test_no_messages_until_both_say_yes(self):
        with self.assertRaises(DemoError):
            self.s.send_message("A", self.intro, "Hi!")
        self.s.decide("A", self.intro, True)
        with self.assertRaises(DemoError):
            self.s.send_message("A", self.intro, "Hi!")

    def test_messages_reach_only_the_two_people_and_never_matching(self):
        self.s.decide("A", self.intro, True)
        self.s.decide("B", self.intro, True)
        self.s.send_message("A", self.intro, "Hi Ren, Saturday at the bouldering gym?")
        thread_b = next(i for i in self.s.phone_state("B")["intros"] if i["intro_id"] == self.intro)["thread"]
        self.assertEqual([(m["mine"], m["text"]) for m in thread_b], [(False, "Hi Ren, Saturday at the bouldering gym?")])
        self.assertTrue(thread_b[0]["at"])
        room = json.dumps(self.s.room_state())
        self.assertNotIn("Saturday at the bouldering gym", room)  # the presenter sees that, not what
        self.assertNotIn("bouldering gym", json.dumps([l.intent_text for l in self.s.board.listings()]))


class Reactions(unittest.TestCase):
    def test_only_a_few_reactions_and_only_between_connected_people(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        intro = next(i["intro_id"] for i in s.phone_state("A")["intros"]
                     if {s.intros[i["intro_id"]].person_a, s.intros[i["intro_id"]].person_b} == {"demoA-1", "demoB-1"})
        s.decide("A", intro, True)
        s.decide("B", intro, True)
        s.send_message("A", intro, "See you Saturday")
        s.react("B", intro, 0, "👍")
        with self.assertRaises(DemoError):
            s.react("B", intro, 0, "<script>")
        with self.assertRaises(DemoError):
            s.react("A", intro, 0, "👍")  # not your own message
        seen_by_a = next(i for i in s.phone_state("A")["intros"] if i["intro_id"] == intro)["thread"][0]["reactions"]
        self.assertEqual(seen_by_a, [{"emoji": "👍", "mine": False}])
        s.react("B", intro, 0, "👍")  # the same again takes it back
        self.assertEqual(next(i for i in s.phone_state("A")["intros"] if i["intro_id"] == intro)["thread"][0]["reactions"], [])


def repair_people(s: DemoSession, close=True, questions=True):
    """A wants an AC technician; B is a repairman who wants repair customers."""
    def person(slot, name, age, prefs, intent, fields):
        s.verify(slot, name, age)
        lid = s.new_listing(slot, prefs)
        s.submit(slot, {"intent_text": intent, "exactness": 0.6,
                        "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in fields.items()]}, lid)
        return lid
    a = person("A", "Mia", 30, {"acknowledged": True, "notify_close": close, "notify_questions": True},
               "Need my air conditioner fixed in Tokyo this weekend.",
               {"first_name": "Mia", "country": "Japan", "city": "Tokyo", "schedule": "weekends",
                "seeking": "an air conditioner repair technician", "offering": "an air conditioner repair customer"})
    b = person("B", "Ken", 42, {"acknowledged": True, "notify_close": False, "notify_questions": questions},
               "I'm a repairman in Tokyo looking for repair jobs.",
               {"first_name": "Ken", "country": "Japan", "city": "Tokyo", "schedule": "flexible",
                "seeking": "a repair customer", "offering": "a repairman"})
    return a, b


class Notifications(unittest.TestCase):
    def test_the_repairman_is_asked_and_saying_yes_makes_a_match(self):
        s = DemoSession(seed=7, agent=None)
        a, b = repair_people(s)
        s.run_matchmaker()
        q = s.phone_state("B")["listings"][0]["followups"]
        self.assertEqual([x["question_text"] for x in q],
                         ["Someone is looking for an air conditioner repair technician. Would that work for you?"])
        self.assertNotIn("Mia", json.dumps(s.phone_state("B")))
        s.answer_followup("B", "also_offer", {"yes": True}, b)
        s.run_matchmaker()
        self.assertTrue(s.phone_state("A")["listings"][0]["intros"])
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_no_questions_for_someone_who_opted_out(self):
        s = DemoSession(seed=7, agent=None)
        repair_people(s, questions=False)
        s.run_matchmaker()
        self.assertEqual(s.phone_state("B")["listings"][0]["followups"], [])

    def test_changing_your_name_updates_live_listings(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.update_profile("A", "Yu")
        self.assertEqual(s.board.listing("demoA-1").get("first_name").value, "Yu")
        self.assertEqual(s.phone_state("A")["name"], "Yu")

    def test_the_other_person_sees_your_new_name(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        intro = next(i["intro_id"] for i in s.phone_state("B")["intros"]
                     if {s.intros[i["intro_id"]].person_a, s.intros[i["intro_id"]].person_b} == {"demoA-1", "demoB-1"})
        s.decide("A", intro, True)
        s.decide("B", intro, True)
        s.update_profile("A", "Yu")
        seen_by_b = next(i for i in s.phone_state("B")["intros"] if i["intro_id"] == intro)
        self.assertEqual(seen_by_b["view"]["fields"]["first_name"], "Yu")
        self.assertEqual(seen_by_b["status"], "connected")  # same introduction, decisions kept
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_defaults_apply_to_new_listings_once_asking_is_off(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        s.set_defaults("A", {"notify_questions": True, "ask_each_time": False})
        first = s.new_listing("A")  # never seen the privacy step: nothing assumed
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"]["notify_questions"], False)
        s.new_listing("A", {"acknowledged": True, "notify_questions": False})  # seen it once
        s.agent_message("A", "Looking for a roommate in Osaka.", first)
        second = s.new_listing("A")
        prefs = {l["id"]: l["prefs"] for l in s.phone_state("A")["listings"]}[second]
        self.assertEqual((prefs["notify_questions"], prefs["notify_close"]), (True, False))

    def test_a_paused_listing_gets_no_new_matches(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.set_active("A", "demoA-1", False)
        self.assertFalse(s.phone_state("A")["listings"][0]["active"])
        s.run_matchmaker()
        self.assertEqual(s.phone_state("A")["listings"][0]["intros"], [])
        s.set_active("A", "demoA-1", True)
        s.run_matchmaker()
        self.assertTrue(s.phone_state("A")["listings"][0]["intros"])

    def test_choices_are_on_the_listing_and_off_by_default(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        lid = s.new_listing("A")
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"],
                         {"acknowledged": False, "notify_close": False, "notify_questions": False})
        s.set_prefs("A", lid, {"notify_close": True})
        self.assertTrue(s.phone_state("A")["listings"][0]["prefs"]["notify_close"])


if __name__ == "__main__":
    unittest.main()


class SeveralListings(unittest.TestCase):
    """One person can look for a tutor and for love at the same time."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climbers(self.s)

    def test_each_listing_has_its_own_chat_and_draft(self):
        first = self.s.slots["A"].current
        second = self.s.new_listing("A")
        self.assertNotEqual(first, second)
        self.assertEqual(self.s.slots["A"].current, second)
        self.s.agent_message("A", "Looking for a roommate in Osaka, moving in November.", second)
        a = {l["id"]: l for l in self.s.phone_state("A")["listings"]}
        self.assertEqual(a[second]["kind"], "roommate")
        self.assertEqual(a[first]["kind"], "climbing")
        self.assertNotIn("roommate", " ".join(m["text"] for m in a[first]["chat"]))

    def test_both_listings_are_matched_on_their_own(self):
        second = self.s.new_listing("A")
        self.s.agent_message("A", "Looking for a roommate in Osaka, moving in November.", second)
        self.s.submit("A", fill(self.s.slots["A"].listing(second).draft), second)
        self.s.run_matchmaker()
        listings = {l["id"]: l for l in self.s.phone_state("A")["listings"]}
        self.assertTrue(listings["demoA-1"]["intros"])
        self.assertTrue(listings[second]["intros"])

    def test_never_introduced_to_your_own_other_listing(self):
        """Two of A's listings that would suit each other are never paired."""
        second = self.s.new_listing("A")
        self.s.agent_message("A", "Need someone to belay with at Osaka gyms, weekends. Intermediate level.", second)
        self.s.submit("A", fill(self.s.slots["A"].listing(second).draft), second)
        self.s.run_matchmaker()
        for intro in self.s.intros.values():
            owners = {self.s.board.listing(intro.person_a).owner, self.s.board.listing(intro.person_b).owner}
            self.assertEqual(len(owners), 2, intro.intro_id)
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])
