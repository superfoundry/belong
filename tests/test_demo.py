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
