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
    """Two people looking for the same thing, in the words other people already use (the offline agent's only
    vocabulary). Each hides a built-in detail and an open detail its agent wrote (their interests)."""
    s.verify("A", "Yuki", 27)
    s.agent_message("A", "Looking for a climbing partner in Osaka. I'm free on weekends, I'm an intermediate "
                         "climber and I'm into bouldering and ramen.")
    s.submit("A", fill(s.slots["A"].draft, hidden={"religion": "Buddhist", "me:interests": ["bouldering", "ramen"]}))
    s.verify("B", "Ren", 29)
    s.agent_message("B", "Looking for a climbing partner in Osaka. I'm free weekends, I'm an intermediate climber "
                         "and I also like jazz and ramen.")
    s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD", "me:interests": ["jazz", "ramen"]}))


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
        # Their open details went live as their agents wrote them, hidden ones hidden.
        a = self.s.board.listing("demoA-1")
        self.assertEqual(a.get("me:level").value, ("intermediate",))
        self.assertEqual((a.get("me:interests").visibility, a.get("religion").visibility), (HIDDEN, HIDDEN))
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
        # What each sees about the other: none of their hidden keys, and none of their hidden values that the
        # one looking doesn't show themselves (an open detail hidden on one side, shown on none).
        intro_id = self._intro_between_a_and_b()
        for slot, other, theirs_only in (("B", "demoA-1", "bouldering"), ("A", "demoB-1", "jazz")):
            hidden = self.s.board.listing(other).hidden()
            self.assertIn("me:interests", hidden)  # something open is really hidden: never a vacuous pass
            view = next(i["view"] for i in self.s.phone_state(slot)["intros"] if i["intro_id"] == intro_id)
            self.assertEqual(view["fields"].keys() & hidden.keys(), set())
            self.assertNotIn(theirs_only, json.dumps(view, ensure_ascii=False))

    def test_mutual_yes_then_mutual_reveal_of_a_shared_detail(self):
        self.s.run_matchmaker()
        intro_id = self._intro_between_a_and_b()
        self.s.decide("A", intro_id, True)
        self.assertEqual(next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["status"],
                         "waiting")
        self.s.decide("B", intro_id, True)
        self.s.reveal("A", intro_id, ["religion", "me:interests"])
        self.assertIsNone(next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["reveal_result"])
        self.s.reveal("B", intro_id, ["religion", "health", "me:interests"])
        result = next(i for i in self.s.phone_state("A")["intros"] if i["intro_id"] == intro_id)["reveal_result"]
        # Only what both have and both chose: their religion, and of their hidden interests only the one they share.
        self.assertEqual(sorted(result, key=lambda r: r["key"]),
                         [{"key": "me:interests", "values": ["ramen"]}, {"key": "religion", "values": ["Buddhist"]}])
        shown = [e for e in self.s.events if e["title"].endswith("have more in common")][-1]["payload"]
        self.assertIn({"detail": "Interests", "value": "ramen"}, shown)  # its name, never its raw key
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
        s.agent_message("A", "Looking for a language exchange partner in Lisbon.")
        body = fill(s.slots["A"].draft)
        s.submit("A", body)
        self.assertTrue(s.slots["A"].entry["accepted"], s.slots["A"].entry)  # the listing itself is fine
        for f in s.slots["A"].draft["fields"]:
            if f["key"] == "religion":
                body["fields"].append({**f, "value": "Christian", "visibility": SHOWN, "enabled": True})
        s.submit("A", body)
        self.assertFalse(s.slots["A"].entry["accepted"])
        self.assertTrue(any("religion" in p for p in s.slots["A"].entry["problems"]), s.slots["A"].entry)



class InstantProfile(unittest.TestCase):
    def test_an_instant_profile_can_start_matching_before_the_agent_replies(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        lid = s.new_listing("A")
        s.prefill("A", "I teach guitar in Osaka, Japan", lid)
        s.submit("A", fill(s.slots["A"].listing(lid).draft), lid)
        self.assertTrue(s.slots["A"].listing(lid).entry["accepted"], s.slots["A"].listing(lid).entry)


class HowTheyMeet(unittest.TestCase):
    def test_online_works_for_me_still_meets_someone_in_person(self):
        """Saying online works too keeps meeting in person open: only "only" or "just" makes a listing
        online only. Typed as on the phone (the instant profile first, then the agent)."""
        s = DemoSession(seed=7, agent=None)
        for slot, name, age, said in (
                ("A", "Yuki", 27, "Looking for a language exchange partner in Osaka: I speak English natively and "
                                  "want to practice Japanese. Online works for me."),
                ("B", "Ren", 29, "Looking for a language exchange partner in Osaka, in person: I speak Japanese "
                                 "natively and want to practice English.")):
            s.verify(slot, name, age)
            s.prefill(slot, said)
            s.agent_message(slot, said)
            s.submit(slot, fill(s.slots[slot].draft))
            self.assertTrue(s.slots[slot].entry["accepted"], s.slots[slot].entry)
        fmt = {slot: s.board.listing(f"demo{slot}-1").get("format").value for slot in "AB"}
        self.assertEqual(fmt, {"A": "In person or online", "B": "In person"})
        for _ in range(5):
            s.run_matchmaker(quiet=True)
        self.assertIn({"demoA-1", "demoB-1"}, [{i.person_a, i.person_b} for i in s.intros.values()])


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
        body = fill(d, values={"city": "Kyoto"}, hidden={"religion": "Buddhist", "me:interests": ["bouldering", "ramen"]})
        s.submit("A", body)
        last = s.slots["A"].chat[-1]["text"]
        self.assertIn("What changed: your city", last)
        body = fill(s.slots["A"].draft, values={"me:level": ["advanced"]})
        s.submit("A", body)
        self.assertIn("What changed: level", s.slots["A"].chat[-1]["text"])  # an open detail by its name
        body["intent_text"] = body["intent_text"] + " Also bouldering."
        s.submit("A", body)
        self.assertIn("your description", s.slots["A"].chat[-1]["text"])
        self.assertNotIn(" is now e", s.slots["A"].chat[-1]["text"])
        self.assertFalse(s.phone_state("A")["listings"][0]["unsaved"])


class SeparateListingFromChat(unittest.TestCase):
    """Asking the same agent for a different search starts a new listing; the first stays as it is."""

    def test_the_agent_can_start_a_second_listing(self):
        from belong.agent_llm import build_draft
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        first = s.board.listing("demoA-1")

        class Agent:  # stands in for the model: it decides this is a separate search
            def respond(self, conversation, text, name, band, current, **kw):
                conversation.append({"role": "user", "content": text})
                draft, _, _ = build_draft({"kind": "other", "intent_text": "I want guitar lessons in Osaka.",
                                           "exactness": 0.6, "suggestions": [], "separate_listing": True,
                                           "fields": [{"key": "city", "values": ["Osaka"], "visibility": "shown"},
                                                      {"key": "headline", "values": ["Looking for a guitar teacher"], "visibility": "shown"},
                                                      {"key": "seeking", "values": ["a guitar teacher"], "visibility": "shown"},
                                                      {"key": "offering", "values": ["a guitar student"], "visibility": "shown"}]}, name, band)
                return "I've set that up as its own listing.", draft

        s.llm = Agent()
        s.agent_message("A", "I'm also looking for a guitar teacher in Osaka", "demoA-1")
        listings = s.phone_state("A")["listings"]
        self.assertEqual(len(listings), 2)
        self.assertEqual(s.board.listing("demoA-1"), first)  # the climbing listing is untouched
        second = listings[1]
        self.assertEqual(s.phone_state("A")["current"], second["id"])
        self.assertIn("guitar", second["title"].lower())


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
        lid = s.new_listing(slot)
        s.set_prefs(slot, lid, prefs)
        s.submit(slot, {"intent_text": intent, "exactness": 0.6,
                        "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in fields.items()]}, lid)
        return lid
    a = person("A", "Mia", 30, {"notify_close": close, "notify_questions": True},
               "Need my air conditioner fixed in Tokyo this weekend.",
               {"first_name": "Mia", "country": "Japan", "city": "Tokyo",
                "seeking": "an air conditioner repair technician", "offering": "an air conditioner repair customer"})
    b = person("B", "Ken", 42, {"notify_close": False, "notify_questions": questions},
               "I'm a repairman in Tokyo looking for repair jobs.",
               {"first_name": "Ken", "country": "Japan", "city": "Tokyo",
                "seeking": "a repair customer", "offering": "a repairman"})
    return a, b


def lonely(s: DemoSession) -> str:
    """Someone nobody on the board is looking for: no match, so they're asked about close matches."""
    s.verify("A", "Mia", 30)
    lid = s.new_listing("A")
    s.submit("A", {"intent_text": "Looking for a theremin teacher in Reykjavik.", "exactness": 0.6,
                   "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in
                              {"first_name": "Mia", "country": "Iceland", "city": "Reykjavik",
                               "seeking": "a theremin teacher", "offering": "a theremin student"}.items()]}, lid)
    return lid


def box_people(s: DemoSession):
    """A seller of boxes and someone looking for free boxes, both opted in to everything."""
    def person(slot, name, age, fields, intent):
        s.verify(slot, name, age)
        lid = s.new_listing(slot)
        s.set_prefs(slot, lid, {"notify_close": True, "notify_questions": True})
        s.submit(slot, {"intent_text": intent, "exactness": 0.6,
                        "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in fields.items()]}, lid)
        return lid
    seller = person("A", "Kai", 31, {"first_name": "Kai", "country": "Japan", "city": "Nagoya", "headline": "Selling 100 cardboard boxes",
                                     "seeking": "a cardboard box buyer", "offering": "a cardboard box seller", "pay": "$1 each",
                                     "topics": ["cardboard boxes", "moving supplies"]}, "Selling 100 cardboard boxes in Nagoya.")
    seeker = person("B", "Ren", 52, {"first_name": "Ren", "country": "Japan", "city": "Nagoya", "headline": "Looking for free cardboard boxes",
                                     "seeking": "a free cardboard box giver", "offering": "a cardboard box recipient", "pay": "Free",
                                     "topics": ["cardboard boxes", "moving"]}, "Looking for free cardboard boxes in Nagoya.")
    return seller, seeker


class OnePersonAtATime(unittest.TestCase):
    """Whoever is looking for something is asked first; the other only if they say no."""

    def questions(self, s, slot):
        return [q["question_text"] for q in s.phone_state(slot)["listings"][0]["followups"]]

    def test_the_seeker_is_asked_first_and_a_yes_makes_the_match(self):
        s = DemoSession(seed=7, agent=None)
        seller, seeker = box_people(s)
        s.run_matchmaker()
        self.assertEqual(self.questions(s, "B"), ["Someone is selling 100 cardboard boxes. Would that work for you?"])
        self.assertEqual(self.questions(s, "A"), [])  # the seller isn't asked yet
        s.answer_followup("B", "also_offer", {"yes": True}, seeker)
        s.run_matchmaker()
        self.assertTrue(s.phone_state("A")["listings"][0]["intros"])
        self.assertEqual(self.questions(s, "A"), [])

    def test_saying_yes_takes_on_their_price(self):
        s = DemoSession(seed=7, agent=None)
        seller, seeker = box_people(s)
        s.run_matchmaker()
        s.answer_followup("B", "also_offer", {"yes": False}, seeker)  # the seeker won't pay
        s.run_matchmaker()
        s.answer_followup("A", "also_offer", {"yes": True}, seller)    # the seller will give them away
        self.assertEqual(s.board.listing(seller).get("pay").value, "Free")
        self.assertIn("Free", s.slots["A"].listing(seller).chat[-1]["text"])
        s.run_matchmaker()
        # The seeker's earlier question quoted the seller's old listing ($1 each); the audit checks it
        # against the version live when it was asked, so it still passes.
        self.assertTrue(s.room_state()["leak_report"]["ok"], s.room_state()["leak_report"]["findings"])

    def test_a_no_passes_the_question_to_the_other_person_and_it_isnt_asked_again(self):
        s = DemoSession(seed=7, agent=None)
        seller, seeker = box_people(s)
        s.run_matchmaker()
        s.answer_followup("B", "also_offer", {"yes": False}, seeker)
        s.run_matchmaker()
        self.assertEqual(self.questions(s, "A"), ["Someone is looking for free cardboard boxes. Would that work for you?"])
        self.assertEqual(self.questions(s, "B"), [])  # not asked again


class RewordAfterYes(unittest.TestCase):
    """A yes that changes the terms: the person's agent rewords the listing, and the whole update goes
    live before matching sees it, so the other person never sees the old terms."""

    def test_the_new_wording_is_what_the_match_sees(self):
        from belong.agent_llm import build_draft
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        seller, seeker = box_people(s)

        class Agent:  # stands in for the model: it rewords the headline to match the yes
            def respond(self, conversation, text, name, band, current, **kw):
                conversation.append({"role": "user", "content": text})
                fields = [{"key": f["key"], "values": f["value"] if isinstance(f["value"], list) else [f["value"]],
                           "visibility": f["visibility"]} for f in current["fields"]
                          if f.get("value") and f["key"] not in ("first_name", "age_band", "headline")]
                fields.append({"key": "headline", "values": ["Giving away 300 cardboard boxes"], "visibility": "shown"})
                draft, problems, _ = build_draft({"kind": "other", "intent_text": "Giving away cardboard boxes in Tokyo.",
                                                  "exactness": 0.6, "suggestions": [], "separate_listing": False,
                                                  "fields": fields}, name, band)
                assert not problems, problems
                return "Updated.", draft

        s.llm = Agent()
        s.run_matchmaker()
        s.answer_followup("B", "also_offer", {"yes": False}, seeker)
        s.run_matchmaker()
        s.answer_followup("A", "also_offer", {"yes": True}, seller)
        live = s.board.listing(seller)
        self.assertEqual(live.get("headline").value, "Giving away 300 cardboard boxes")
        self.assertEqual(live.get("pay").value, "Free")
        s.run_matchmaker()
        reasons = [i["view"]["reason"] for i in s.phone_state("B")["listings"][0]["intros"]]
        self.assertTrue(any("giving away 300 cardboard boxes" in r for r in reasons), reasons)
        self.assertTrue(s.room_state()["leak_report"]["ok"])


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

    def test_no_match_yet_asks_once_to_include_close_matches(self):
        s = DemoSession(seed=7, agent=None)
        lid = lonely(s)
        s.run_matchmaker()
        q = s.phone_state("A")["listings"][0]["followups"]
        self.assertEqual([x["question_text"] for x in q], ["No matches yet. Want to include close matches?"])
        s.answer_followup("A", "broaden", {"yes": True}, lid)
        # A yes is the opt-in to close matches only: questions keep their own switch (off, as the listing started).
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], {"notify_close": True, "notify_questions": False})
        self.assertTrue(s.board.listing(lid).notify_close)
        self.assertFalse(s.board.listing(lid).notify_questions)
        s.set_prefs("A", lid, {"notify_close": False, "notify_questions": False})  # turned off again later
        s.run_matchmaker()
        self.assertEqual(s.phone_state("A")["listings"][0]["followups"], [])  # never asked twice
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_no_keeps_exact_matches_only_and_isnt_asked_again(self):
        s = DemoSession(seed=7, agent=None)
        lid = lonely(s)
        s.run_matchmaker()
        s.answer_followup("A", "broaden", {"yes": False}, lid)
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], {"notify_close": False, "notify_questions": False})
        s.run_matchmaker()
        self.assertEqual(s.phone_state("A")["listings"][0]["followups"], [])
        self.assertTrue(s.room_state()["leak_report"]["ok"])

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

    def test_a_deleted_listing_is_never_matched_again(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        s.delete_listing("A", "demoA-1")
        self.assertEqual(s.phone_state("A")["listings"], [])
        self.assertNotIn("demoA-1", s.board)
        before = len(s.intros)
        s.run_matchmaker()
        self.assertFalse([i for i in list(s.intros.values())[before:] if "demoA-1" in (i.person_a, i.person_b)])
        self.assertTrue(s.room_state()["leak_report"]["ok"])  # what was already sent is still audited

    def test_a_new_listing_never_takes_a_deleted_ones_id(self):
        """A person who deletes a listing and starts another gets a new id (the Warden would refuse the old one), so
        an introduction to the deleted listing is never reissued from the new one."""
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker()
        intro_id = next(i for i, x in s.intros.items() if {x.person_a, x.person_b} == {"demoA-1", "demoB-1"})
        b_sees = s.intros[intro_id].b_view
        s.delete_listing("A", "demoA-1")
        self.assertEqual(s.new_listing("A"), "demoA-2")
        s.agent_message("A", "Looking for someone to go fossil hunting with in Osaka.")
        s.submit("A", fill(s.slots["A"].draft))
        self.assertIn("demoA-2", s.board)
        self.assertEqual(s.intros[intro_id].b_view, b_sees)
        self.assertTrue(s.room_state()["leak_report"]["ok"], s.room_state()["leak_report"]["findings"])

    def _past_the_days_updates(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        lid = "demoA-1"
        s.set_prefs("A", lid, {"notify_close": True, "notify_questions": True})
        for _ in range(20):  # use up today's updates
            try:
                s.set_prefs("A", lid, {"notify_close": True, "notify_questions": True})
            except DemoError as e:
                self.assertIn("too many listing updates today", str(e))
                break
        else:
            self.fail("an update past the day's limit was taken without a word")
        return s, lid

    def test_the_phone_shows_only_what_the_warden_holds(self):
        """A change to a live listing (notification choices, pausing it, a new name) goes on the phone only when the
        Warden accepts it; refused (here by the daily update limit, which never holds back turning something off), the
        person is told why and the phone keeps showing what the Warden holds."""
        s, lid = self._past_the_days_updates()
        s.set_prefs("A", lid, {"notify_close": False, "notify_questions": True})
        with self.assertRaisesRegex(DemoError, "too many listing updates today"):
            s.set_prefs("A", lid, {"notify_close": True, "notify_questions": True})  # back on: refused
        live = s.board.listing(lid)
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"],
                         {"notify_close": live.notify_close, "notify_questions": live.notify_questions})
        self.assertEqual((live.notify_close, live.notify_questions), (False, True))
        s.set_active("A", lid, False)
        with self.assertRaises(DemoError):
            s.set_active("A", lid, True)
        self.assertFalse(s.board.listing(lid).active or s.phone_state("A")["listings"][0]["active"])
        with self.assertRaisesRegex(DemoError, "still show the one before"):
            s.update_profile("A", "Yu")
        self.assertEqual(s.board.listing(lid).get("first_name").value, "Yuki")

    def test_turning_things_off_never_hits_the_days_update_limit(self):
        """Consent first: past the day's updates, the person can still turn off questions and close matches and pause
        the listing, each checked by the Warden with its receipts, on the phone and on the live listing at once.
        Turning one back on waits for the limit, and the phone keeps showing it off."""
        s, lid = self._past_the_days_updates()
        for prefs in ({"notify_close": True, "notify_questions": False}, {"notify_close": False, "notify_questions": False}):
            before = len(s.warden.receipts)
            s.set_prefs("A", lid, prefs)
            live = s.board.listing(lid)
            self.assertEqual({"notify_close": live.notify_close, "notify_questions": live.notify_questions}, prefs)
            self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], prefs)
            self.assertIn("update_rate_limit", [r.check for r in s.warden.receipts[before:]])
            self.assertTrue(all(r.result == "pass" for r in s.warden.receipts[before:]))
        s.set_active("A", lid, False)
        self.assertFalse(s.board.listing(lid).active or s.phone_state("A")["listings"][0]["active"])
        for turn_on in (lambda: s.set_prefs("A", lid, {"notify_close": False, "notify_questions": True}),
                        lambda: s.set_active("A", lid, True)):
            with self.assertRaisesRegex(DemoError, "too many listing updates today"):
                turn_on()
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], {"notify_close": False, "notify_questions": False})
        self.assertFalse(s.phone_state("A")["listings"][0]["active"])

    def test_a_shared_list_that_drops_a_place_never_locks_a_live_listing(self):
        """A description that passed the full-name rule because a place was on the shared list stays fine for that
        listing after a later list drops the place (the Warden keeps the places it was accepted with): pausing it,
        a notification choice, a new name and Save draft all go on. Someone else's new listing reads today's list."""
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        listed = {p["city"] for p in s.warden.listed_places()}
        self.assertIn("Osaka", listed)
        s.verify("A", "Ines", 36)
        s.agent_message("A", "Looking for a language exchange partner, online is fine. I speak Portuguese natively and "
                             "want to practise Japanese. I live in Porto.")
        d = s.slots["A"].draft
        intent = "Language swaps online; I still miss the Osaka Castle moat walks."
        s.submit("A", fill({**d, "intent_text": intent}))
        self.assertTrue(s.slots["A"].entry["accepted"], s.slots["A"].entry)
        osaka = [l for l in s.board.listings() if l.get("city") and l.get("city").value == "Osaka" and l.owner in s.personas]
        for l in osaka[:len(osaka) - 5]:
            s.warden.withdraw(s.personas[l.owner].person, l.person_id, s.board)
        s.clock.advance(days=1)
        s.details_in_use()
        self.assertNotIn("Osaka", {p["city"] for p in s.warden.listed_places()})
        lid = s.slots["A"].listing().id
        s.set_prefs("A", lid, {"notify_close": True, "notify_questions": False})
        s.set_active("A", lid, False)
        s.update_profile("A", "Inês")
        self.assertEqual((s.board.listing(lid).notify_close, s.board.listing(lid).active,
                          s.board.listing(lid).get("first_name").value), (True, False, "Inês"))
        s.save_draft("A", fill(s.slots["A"].listing().draft), lid)
        s.verify("B", "Rafa", 31)
        s.agent_message("B", "Looking for a language exchange partner, online is fine. I speak Spanish natively and "
                             "want to practise Japanese. I live in Vigo.")
        s.submit("B", fill({**s.slots["B"].draft, "intent_text": intent}))
        self.assertFalse(s.slots["B"].entry["accepted"])

    def test_choices_are_on_the_listing_and_off_by_default(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Mia", 30)
        lid = s.new_listing("A")
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], {"notify_close": False, "notify_questions": False})
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
        self.s.agent_message("A", "Looking for a language exchange partner in Osaka.", second)
        a = {l["id"]: l for l in self.s.phone_state("A")["listings"]}
        seeking = lambda l: next(f["value"] for f in l["draft"]["fields"] if f["key"] == "seeking")
        self.assertEqual((seeking(a[second]), seeking(a[first])), ("a language exchange partner", "a climbing partner"))
        self.assertEqual((a[second]["kind"], a[first]["kind"]), ("other", "other"))
        self.assertNotIn("language exchange", " ".join(m["text"] for m in a[first]["chat"]))
        self.assertNotIn("me:level", {f["key"] for f in a[second]["draft"]["fields"]})  # the first one's detail

    def test_both_listings_are_matched_on_their_own(self):
        second = self.s.new_listing("A")
        self.s.agent_message("A", "Looking to meet a man in Osaka, aged 25 to 35. I'm a woman, into jazz and ramen.", second)
        self.s.submit("A", fill(self.s.slots["A"].listing(second).draft), second)
        self.assertTrue(self.s.slots["A"].listing(second).entry["accepted"], self.s.slots["A"].listing(second).entry)
        self.s.run_matchmaker()
        listings = {l["id"]: l for l in self.s.phone_state("A")["listings"]}
        self.assertTrue(listings["demoA-1"]["intros"])
        self.assertTrue(listings[second]["intros"])

    def test_never_introduced_to_your_own_other_listing(self):
        """Two of A's listings that would suit each other are never paired."""
        second = self.s.new_listing("A")
        self.s.agent_message("A", "Looking for a climbing partner at Osaka gyms, weekends. Intermediate level.", second)
        self.s.submit("A", fill(self.s.slots["A"].listing(second).draft), second)
        self.s.run_matchmaker()
        for intro in self.s.intros.values():
            owners = {self.s.board.listing(intro.person_a).owner, self.s.board.listing(intro.person_b).owner}
            self.assertEqual(len(owners), 2, intro.intro_id)
        self.assertTrue(self.s.room_state()["leak_report"]["ok"])


class ProposalsAreAuditedAgainstTheirBoard(unittest.TestCase):
    """The presenter's leak report audits the last Matchmaker run's proposals against the board they were made
    on (DemoSession._leak_report, Marketboard.as_of): a person who hides a detail that was a reason, or changes
    it, raises no false alarm before the next run, and a bad proposal is still caught after the board moves."""

    def hide_a_reason(self, s):
        from tests.test_demo import climbers
        climbers(s)
        s.run_matchmaker()
        pair = next(p for p in s.last_proposals if {p.person_a, p.person_b} == {"demoA-1", "demoB-1"})
        key = next(k for k in pair.reason_fields if k.startswith("agree:")).replace("agree:", "me:")
        mine = s.board.listing("demoB-1")
        s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD",
                                                        "me:interests": list(mine.get("me:interests").value),
                                                        key: list(mine.get(key).value)}))
        self.assertEqual(s.board.listing("demoB-1").get(key).visibility, HIDDEN)

    def test_hiding_a_reason_raises_no_false_alarm(self):
        s = DemoSession(seed=7, agent=None)
        self.hide_a_reason(s)
        report = s.room_state()["leak_report"]
        self.assertTrue(report["ok"], report["findings"])
        self.assertEqual(report["checked"]["proposals"], len(s.last_proposals))  # still audited, not left out
        s.run_matchmaker(quiet=True)
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_bad_proposal_is_still_caught_after_the_board_moves(self):
        from dataclasses import replace
        s = DemoSession(seed=7, agent=None)
        from tests.test_demo import climbers
        climbers(s)
        s.run_matchmaker()
        good = next(p for p in s.last_proposals if {p.person_a, p.person_b} == {"demoA-1", "demoB-1"})
        s.last_proposals = s.last_proposals + [replace(good, reason_fields=good.reason_fields + ("religion",))]
        self.assertIn("R1", [f["audit"] for f in s.room_state()["leak_report"]["findings"]])
        mine = s.board.listing("demoB-1")
        s.submit("B", fill(s.slots["B"].draft, hidden={"religion": "Buddhist", "health": "ADHD",
                                                        "me:interests": list(mine.get("me:interests").value)},
                           values={"city": "Kobe"}))
        report = s.room_state()["leak_report"]
        self.assertEqual([f for f in report["findings"] if "religion" not in f["detail"]], [], report["findings"])
        self.assertIn("R1", [f["audit"] for f in report["findings"]])


class MatchmakerRunsOnItsOwn(unittest.TestCase):
    """The demo's Matchmaker runs every few seconds (server._match_if_due); there is no button."""

    def test_the_other_phone_is_introduced_even_if_it_joins_later(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Yuki", 27)
        s.agent_message("A", "Looking for a climbing partner in Osaka. I'm free on weekends, I'm an intermediate "
                             "climber and I'm into bouldering and ramen.")
        s.submit("A", fill(s.slots["A"].draft))
        for _ in range(3):
            s.run_matchmaker(quiet=True)  # A meets synthetic climbers first
        s.verify("B", "Ren", 29)
        s.agent_message("B", "Looking for a climbing partner in Osaka. I'm free weekends, I'm an intermediate climber "
                             "and I also like jazz and ramen.")
        s.submit("B", fill(s.slots["B"].draft))
        for _ in range(3):
            s.run_matchmaker(quiet=True)
        pairs = [{i.person_a, i.person_b} for i in s.intros.values()]
        self.assertIn({"demoA-1", "demoB-1"}, pairs)  # people already introduced don't crowd them out
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_two_busy_phones_still_meet_on_a_big_board(self):
        """With 5,000 people, each phone has plenty of other climbers to meet first, and every run adds to
        both phones' introductions today. Fair ranking doesn't hold the two apart: a phone's own load is
        the same for all its candidates, so it isn't counted against meeting the other phone."""
        s = DemoSession(people=5000, agent=None)
        climbers(s)
        pairs = []
        for _ in range(5):
            s.run_matchmaker(quiet=True)
            pairs = [{i.person_a, i.person_b} for i in s.intros.values()]
            if {"demoA-1", "demoB-1"} in pairs:
                break
        self.assertIn({"demoA-1", "demoB-1"}, pairs)
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_a_pair_held_back_isnt_sent_again_until_something_changes(self):
        s = DemoSession(seed=7, agent=None)
        s.warden.approach_limit = 0  # every introduction is held back
        climbers(s)
        s.run_matchmaker(quiet=True)
        receipts, events = len(s.warden.receipts), len(s.events)
        self.assertTrue(any("held back" in e["title"] for e in s.events))  # shown once, even from a quiet run
        s.run_matchmaker(quiet=True)
        self.assertEqual((len(s.warden.receipts), len(s.events)), (receipts, events))  # nothing new, nothing logged
        s.warden.approach_limit = 15
        s.clock.advance(hours=1)  # the daily limit is tried again after an hour of demo time
        s.run_matchmaker(quiet=True)
        self.assertIn({"demoA-1", "demoB-1"}, [{i.person_a, i.person_b} for i in s.intros.values()])

    @staticmethod
    def found(n):
        return ("Belong. found someone for this listing. Go back to Belong. to see them." if n == 1 else
                f"Belong. found {n} people for this listing. Go back to Belong. to see them.")

    def test_the_person_is_told_in_their_listing_chat(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker(quiet=True)
        mine = s.phone_state("A")["listings"][0]
        self.assertTrue(mine["intros"])
        self.assertEqual(mine["chat"][-1]["text"], self.found(len(mine["intros"])))
        self.assertIsNotNone(s.room_state()["matchmaker_last_run"])

    @staticmethod
    def waiting(mine):
        return sum(i["status"] == "pending" for i in mine["intros"])

    def test_many_matches_are_one_line_that_keeps_count(self):
        """No spam: matches since anything else was said in the chat are one line with their number, not a
        line each, and it stays the last line as more arrive."""
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        counts = []
        for _ in range(2):
            s.run_matchmaker(quiet=True)
            mine = s.phone_state("A")["listings"][0]
            news = [m["text"] for m in mine["chat"] if m["text"].startswith("Belong. found")]
            self.assertEqual(news, [self.found(self.waiting(mine))])
            self.assertEqual(mine["chat"][-1]["text"], news[0])
            counts.append(self.waiting(mine))
            s.clock.advance(hours=1)
        self.assertGreater(counts[0], 1)
        self.assertGreater(counts[1], counts[0])  # the second run's matches were added to the same line

    def test_the_number_is_only_what_still_waits_on_you(self):
        """Matches answered on their cards (no chat line) aren't counted again when more arrive."""
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker(quiet=True)
        first = s.phone_state("A")["listings"][0]["intros"]
        self.assertGreater(len(first), 1)
        s.decide("A", first[0]["intro_id"], True)
        s.decide("A", first[1]["intro_id"], False)
        s.clock.advance(hours=1)
        s.run_matchmaker(quiet=True)
        mine = s.phone_state("A")["listings"][0]
        self.assertGreater(len(mine["intros"]), len(first) - 1)  # new matches arrived
        self.assertEqual(mine["chat"][-1]["text"], self.found(self.waiting(mine)))
        self.assertEqual(self.waiting(mine), len(mine["intros"]) - 1)  # all but the yes (the no is gone)

    def test_a_question_answered_on_its_card_isnt_counted_again(self):
        """The repairman is asked something while alone and answers on the card (no chat line); a question
        that comes later is told as one question, not one plus the one already answered."""
        s = DemoSession(seed=7, agent=None)

        def person(slot, name, age, prefs, intent, fields):
            s.verify(slot, name, age)
            lid = s.new_listing(slot)
            s.set_prefs(slot, lid, prefs)
            s.submit(slot, {"intent_text": intent, "exactness": 0.6,
                            "fields": [{"key": k, "value": v, "visibility": SHOWN} for k, v in fields.items()]}, lid)
            return lid
        b = person("B", "Ken", 42, {"notify_close": False, "notify_questions": True},
                   "I'm a repairman in Tokyo looking for repair jobs.",
                   {"first_name": "Ken", "country": "Japan", "city": "Tokyo",
                    "seeking": "a repair customer", "offering": "a repairman"})
        s.run_matchmaker(quiet=True)
        mine = lambda: s.phone_state("B")["listings"][0]
        self.assertTrue(mine()["followups"])
        for q in mine()["followups"]:
            s.answer_followup("B", q["question_id"], {"yes": False}, b)
        self.assertEqual(mine()["followups"], [])
        person("A", "Mia", 30, {"notify_close": True, "notify_questions": True},
               "Need my air conditioner fixed in Tokyo this weekend.",
               {"first_name": "Mia", "country": "Japan", "city": "Tokyo",
                "seeking": "an air conditioner repair technician", "offering": "an air conditioner repair customer"})
        s.clock.advance(hours=1)
        s.run_matchmaker(quiet=True)
        self.assertEqual(len(mine()["followups"]), 1)
        self.assertEqual(mine()["chat"][-1]["text"],
                         "Belong. has a question for this listing. Go back to Belong. to answer it.")

    def test_a_new_line_once_something_else_is_said(self):
        """Notices of one kind collapse only while nothing else is said; other kinds keep their own line,
        and the newest news is always last."""
        listing = type("L", (), {"chat": [{"from": "agent", "text": "Hi"}]})()
        tell = DemoSession._tell
        tell(listing, "match", 1)
        tell(listing, "question", 1)
        tell(listing, "match", 2)
        self.assertEqual([m["text"] for m in listing.chat[1:]], [
            "Belong. has a question for this listing. Go back to Belong. to answer it.",
            "Belong. found 2 people for this listing. Go back to Belong. to see them."])
        tell(listing, "question", 2)
        self.assertEqual(listing.chat[-1]["text"],
                         "Belong. has 2 questions for this listing. Go back to Belong. to answer them.")
        self.assertEqual(len(listing.chat), 3)
        listing.chat.append({"from": "me", "text": "Thanks"})
        tell(listing, "match", 1)
        self.assertEqual([m["text"] for m in listing.chat[-2:]], ["Thanks", self.found(1)])
        self.assertEqual(sum(m["text"].startswith("Belong. found") for m in listing.chat), 2)

    def test_a_one_way_close_match_isnt_news_to_the_other_person(self):
        s = DemoSession(seed=7, agent=None)
        repair_people(s, questions=False)
        s.run_matchmaker(quiet=True)
        intro = next(i for i in s.intros.values() if {i.person_a, i.person_b} == {"demoA-1", "demoB-1"})
        self.assertEqual(intro.one_way_to, "demoA-1")
        mine = s.phone_state("A")["listings"][0]
        self.assertIn(self.found(len(mine["intros"])), [m["text"] for m in mine["chat"]])
        self.assertFalse([m for m in s.phone_state("B")["listings"][0]["chat"] if m["text"].startswith("Belong. found")])


class SaveDraft(unittest.TestCase):
    """Save draft keeps the person's own changes on their draft and nowhere else (DemoSession.save_draft)."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.s.verify("A", "Tomas", 41)
        self.s.agent_message("A", "Looking for a climbing partner in Lisbon. I'm free weekday mornings and I'm an intermediate "
                                  "climber.")

    def phone_copy(self, **values):
        """The phone's copy of the draft with some values changed, as Save draft sends it. me_free_time is the
        open detail me:free time."""
        d = self.s.slots["A"].draft
        values = {("me:" + k[3:].replace("_", " ") if k.startswith("me_") else k): v for k, v in values.items()}
        return {"intent_text": d["intent_text"], "exactness": d["exactness"],
                "fields": [{**{k: f[k] for k in ("about", "name", "compare", "steps", "required") if k in f},
                            "key": f["key"], "value": values.get(f["key"], f["value"]), "enabled": f["enabled"]}
                           for f in d["fields"]]}

    def values(self):
        return {f["key"]: f["value"] for f in self.s.slots["A"].draft["fields"]}

    def test_your_changes_are_kept_on_your_draft_and_go_nowhere_else(self):
        s = self.s
        events, receipts = len(s.events), len(s.warden.receipts)
        self.assertEqual(self.values()["me:free time"], ["weekday mornings"])  # an open detail its agent wrote
        s.save_draft("A", self.phone_copy(me_free_time=["sunday afternoons"], city="Porto"))
        self.assertEqual((self.values()["me:free time"], self.values()["city"]), (["sunday afternoons"], "Porto"))
        self.assertFalse(s.phone_state("A")["listings"][0]["listed"])
        self.assertNotIn(s.slots["A"].listing().id, s.board)  # never matched
        self.assertEqual((len(s.events), len(s.warden.receipts)), (events, receipts))  # no log, no Warden
        self.assertNotIn("sunday afternoons", json.dumps(s.phone_state("B")) + json.dumps(s.room_state()))
        self.assertEqual([f["key"] for f in s.slots["A"].draft["fields"]][:2], ["kind", "first_name"])  # same rows, same order

    def test_your_agent_builds_on_it(self):
        s, seen = self.s, {}
        s.save_draft("A", self.phone_copy(me_free_time=["weekends"]))

        class Agent:  # stands in for the model: notes the profile it's given to build on
            def respond(self, conversation, text, name, band, current, **kw):
                seen.update(current or {})
                return "Noted.", None

        s.llm = Agent()
        s.agent_message("A", "Evenings work too.")
        self.assertEqual([f["value"] for f in seen["fields"] if f["key"] == "me:free time"], [["weekends"]])

    def test_it_changes_values_only(self):
        """Not shown or hidden (its own action, which asks to confirm a sensitive detail), not the name or age, and
        not what an open detail is (its name, whose it is, how it compares, its steps: only its agent writes those)."""
        s = self.s
        s.set_visibility("A", None, {"me:free time": HIDDEN})
        body = self.phone_copy(first_name="Someone else", age_band="18-24", me_free_time=["weekends"])
        for f in body["fields"]:
            f["visibility"] = SHOWN
            if f["key"] == "religion":
                f.update(value="Buddhist", enabled=True)
            if f["key"] == "me:free time":
                f.update(compare="steps", steps=["weekends", "weekdays"], required=False)
        body["fields"].append({"key": "not_a_detail", "value": "x", "enabled": True})
        body["fields"].append({"key": "me:made up", "value": ["x"], "enabled": True, "compare": "same"})
        s.save_draft("A", body)
        by = {f["key"]: f for f in s.slots["A"].draft["fields"]}
        self.assertEqual((by["first_name"]["value"], by["age_band"]["value"]), ("Tomas", "40-49"))
        self.assertEqual((by["me:free time"]["value"], by["me:free time"]["visibility"]), (["weekends"], HIDDEN))
        self.assertEqual((by["me:free time"]["compare"], by["me:free time"]["steps"], by["me:free time"]["required"]),
                         ("same", [], True))
        self.assertEqual((by["religion"]["value"], by["religion"]["enabled"], by["religion"]["visibility"]),
                         ("Buddhist", True, HIDDEN))
        self.assertNotIn("not_a_detail", by)
        self.assertNotIn("me:made up", by)  # a phone can't invent an open detail

    def test_an_open_detail_takes_only_its_own_steps(self):
        s = self.s
        s.save_draft("A", self.phone_copy(me_level=["Advanced "]))  # a step, in another case and spacing
        self.assertEqual(self.values()["me:level"], ["advanced"])
        before = json.dumps(s.slots["A"].draft)
        for wrong in (["expert"], ["beginner", "advanced"]):  # not one of its steps; more than the one that applies
            with self.assertRaises(DemoError):
                s.save_draft("A", self.phone_copy(me_level=wrong))
        self.assertEqual(json.dumps(s.slots["A"].draft), before)  # refused, nothing changed

    def test_open_details_are_held_to_the_same_rules_as_an_agents_save(self):
        """The Warden's own rule (here: contact details in a value) holds for every open detail on the draft."""
        s, before = self.s, json.dumps(self.s.slots["A"].draft)
        with self.assertRaises(DemoError):
            s.save_draft("A", self.phone_copy(me_free_time=["call me on 090 1234 5678"]))
        self.assertEqual(json.dumps(s.slots["A"].draft), before)

    def test_open_details_go_through_the_one_check_an_agents_save_runs(self):
        """Save draft runs the one pre-save check of open details an agent's save runs (agent_llm.
        detail_save_problems): the Warden's specific-person rules read every open detail too, read with the
        description, and the problem names the detail. Nothing changes, so nothing Start matching would refuse
        stays on the draft."""
        from belong import agent_llm, demo
        self.assertIs(demo.detail_save_problems, agent_llm.detail_save_problems)
        s, before = self.s, json.dumps(self.s.slots["A"].draft)
        for value in ("my ex Leopold", "named Gwendolyn", "works at Fennimore Optics"):
            with self.assertRaises(DemoError) as caught:
                s.save_draft("A", self.phone_copy(me_free_time=[value]))
            self.assertIn("specific person", str(caught.exception), value)
            self.assertIn("'free time'", str(caught.exception), value)
            self.assertEqual(json.dumps(s.slots["A"].draft), before, value)
        s.save_draft("A", self.phone_copy(me_free_time=["late evenings"]))  # an ordinary value still saves
        self.assertEqual(self.values()["me:free time"], ["late evenings"])

    def test_a_list_that_changed_since_never_stops_a_value_change(self):
        """One name, one meaning is the agents' rule for what they write. The shared list can change after the
        agent wrote a detail (a name it used comes into use another way), and the phone changes only values,
        so the person's save still goes through: the agent's next save is checked against the list it's shown.
        A definition a save does change is checked against today's list (details_in_use, never an older copy),
        and the problem is worded for the person, never quoting the list (phones never get it)."""
        from unittest import mock
        s = self.s
        listed = {**s.details_in_use(), "details": [{"about": "me", "name": "free time", "compare": "steps"}],
                  "scales": [{"about": "me", "name": "free time", "steps": ["rarely", "often"]}]}
        with mock.patch.object(s, "details_in_use", return_value=listed):
            s.save_draft("A", self.phone_copy(me_free_time=["sunday afternoons"], city="Porto"))
        self.assertEqual((self.values()["me:free time"], self.values()["city"]), (["sunday afternoons"], "Porto"))
        row = next(f for f in s.slots["A"].draft["fields"] if f["key"] == "me:free time")
        self.assertEqual(DemoSession._list_problems([row], {row["key"]: row}, "other", self.fail), [])  # not even built
        rescaled = {row["key"]: {**row, "compare": "steps", "steps": ["mornings", "evenings"]}}
        problems = DemoSession._list_problems([row], rescaled, "other", lambda: listed)
        self.assertEqual(len(problems), 1)
        self.assertTrue(problems[0].startswith("Ask your agent to update the detail"), problems)
        self.assertNotIn("rarely", problems[0])
        self.assertNotIn("steps", problems[0])
        self.assertEqual(DemoSession._list_problems([row], rescaled, "other", dict), [])  # a name nobody lists

    def test_clearing_an_open_detail_takes_it_off_and_back_hidden_if_it_was(self):
        s, listing = self.s, self.s.slots["A"].listing()
        s.set_visibility("A", None, {"me:level": HIDDEN})
        s.save_draft("A", self.phone_copy(me_level=[]))
        self.assertNotIn("me:level", self.values())
        self.assertEqual(listing.choices["me:level"], HIDDEN)  # if its agent writes it again, it starts hidden

    def test_an_open_detail_the_phone_didnt_have_stays(self):
        """Only a built-in of another kind is put aside by the phone; an open detail its copy didn't have yet
        (its agent wrote it since) is kept, never deleted by an old copy."""
        body = self.phone_copy()
        body["fields"] = [f for f in body["fields"] if f["key"] != "me:level"]
        self.s.save_draft("A", body)
        self.assertEqual(self.values()["me:level"], ["intermediate"])

    def test_a_detail_put_aside_comes_back_hidden_if_it_was(self):
        """Changing the kind puts a detail aside on the phone. Saved meanwhile, it comes off the draft (an
        empty row of another kind never lingers to be filled in), and if it was hidden it comes back hidden,
        even with the phone's default for it (a reload loses what the phone put aside)."""
        s, listing = self.s, self.s.slots["A"].listing()
        for f in listing.draft["fields"]:  # hidden by an agent, not by the person: not one of their choices
            if f["key"] == "pay":
                f.update(value="split the gym fees", visibility=HIDDEN)
        aside = self.phone_copy(kind="dating")
        aside["fields"] = [f for f in aside["fields"] if f["key"] != "pay"]  # dating has no pay
        s.save_draft("A", aside)
        self.assertNotIn("pay", {f["key"] for f in listing.draft["fields"]})
        back = self.phone_copy(kind="other")
        back["fields"].append({"key": "pay", "value": "split the gym fees", "enabled": True, "visibility": SHOWN})
        s.save_draft("A", back)
        self.assertEqual({f["key"]: f["visibility"] for f in listing.draft["fields"]}["pay"], HIDDEN)

    def test_a_detail_a_new_kind_adds_takes_the_phones_choice_but_hidden_wins(self):
        s, listing = self.s, self.s.slots["A"].listing()
        s.set_visibility("A", None, {"gender": HIDDEN})  # hidden on this listing before
        body = self.phone_copy(kind="dating")
        body["fields"] += [{"key": "children", "value": "wants children", "enabled": True, "visibility": SHOWN},
                           {"key": "gender", "value": "man", "enabled": True, "visibility": SHOWN},
                           {"key": "kinks", "value": None, "enabled": True, "visibility": SHOWN}]
        s.save_draft("A", body)
        by = {f["key"]: f for f in listing.draft["fields"]}
        self.assertEqual(by["children"]["visibility"], SHOWN)  # as they chose on the phone
        self.assertEqual(by["gender"]["visibility"], HIDDEN)  # hidden before: hidden wins
        self.assertEqual(by["kinks"]["visibility"], HIDDEN)  # sensitive: shown only with their confirmation
        self.assertEqual(by["me:level"]["value"], ["intermediate"])  # open details go on either kind

    def test_only_a_hidden_value_set_aside_is_remembered_hidden(self):
        """A detail empty and hidden only by default, set aside, isn't recorded as their choice."""
        s, listing = self.s, self.s.slots["A"].listing()
        for f in listing.draft["fields"]:
            if f["key"] == "pay":
                f.update(visibility=HIDDEN)  # empty
        aside = self.phone_copy(kind="dating")
        aside["fields"] = [f for f in aside["fields"] if f["key"] != "pay"]
        s.save_draft("A", aside)
        self.assertNotIn("pay", listing.choices)

    def test_a_hidden_first_name_stays_hidden(self):
        """Its value comes from the ID check, but shown or hidden is the person's choice, as at Start matching."""
        s = self.s
        s.set_visibility("A", None, {"first_name": HIDDEN, "age_band": HIDDEN})
        s.save_draft("A", self.phone_copy(me_free_time=["weekends"]))
        by = {f["key"]: f for f in s.slots["A"].draft["fields"]}
        self.assertEqual((by["first_name"]["value"], by["first_name"]["visibility"]), ("Tomas", HIDDEN))
        self.assertEqual(by["age_band"]["visibility"], SHOWN)  # the age band is fixed
        self.assertEqual(s.phone_state("A")["listings"][0]["choices"].get("first_name"), HIDDEN)

    def test_a_malformed_save_is_refused_and_changes_nothing(self):
        before = json.dumps(self.s.slots["A"].draft)
        for body in ({}, {"fields": None}, {"fields": "x"}, None):
            with self.assertRaises(DemoError):
                self.s.save_draft("A", body)
        self.assertEqual(json.dumps(self.s.slots["A"].draft), before)

    def test_not_while_your_agent_is_replying(self):
        """Its reply would replace the draft, so saving waits for it (and the phone's button is off)."""
        self.s.slots["A"].listing().agent_busy = True
        with self.assertRaises(DemoError):
            self.s.save_draft("A", self.phone_copy(me_free_time=["weekends"]))
        self.assertEqual(self.values()["me:free time"], ["weekday mornings"])

    def test_a_guessed_place_you_change_is_yours(self):
        listing = self.s.slots["A"].listing()
        listing.provisional = {"city", "country"}
        self.s.save_draft("A", self.phone_copy(city="Porto"))
        self.assertEqual(listing.provisional, {"country"})  # the one left as it was is still a guess

    def test_the_description_is_held_to_the_wardens_rules_with_the_same_fields(self):
        """Save draft runs the Warden's specific-person rules on the description and the other free text, reading the
        same fields the Warden reads at entry (the draft's shown city and country, the Warden's last shared list), so
        the phone, a model agent's save and the Warden agree; worded for the person, without the list. Nothing is
        logged and the Warden makes no receipt."""
        from belong.agent_llm import build_draft
        from belong.matching.marketboard import Marketboard
        from belong.schema import Field, Listing, Person
        s = self.s
        cases = [("Hoping for a partner for evening walks along the Rio Grande levee.", "Albuquerque", SHOWN),
                 ("Hoping for a partner for evening walks along the Rio Grande levee.", "Rio Grande", SHOWN),
                 ("Hoping for a partner for evening walks along the Rio Grande levee.", "Rio Grande", HIDDEN),
                 ("Looking for a partner for early climbs at Hell Gate, near Vang Vieng.", "Vang Vieng", SHOWN),
                 ("Looking for a partner for early climbs near Vang Vieng.", "Vang Vieng", SHOWN),
                 ("I'd love to meet Oksana Hrytsenko again, a climber I met once.", "Kyiv", SHOWN),
                 ("Climbing partner wanted, ideally free most Sunday Afternoons.", "Lviv", SHOWN)]
        seen = set()
        for i, (intent, city, vis) in enumerate(cases):
            s.set_visibility("A", None, {"city": vis})
            events, receipts = len(s.events), len(s.warden.receipts)
            body = self.phone_copy(city=city)
            body["intent_text"] = intent
            try:
                s.save_draft("A", body)
                phone = True
            except DemoError as e:
                phone = False
                self.assertIn("Belong. never helps find one particular person", str(e))
            self.assertEqual((len(s.events), len(s.warden.receipts)), (events, receipts))
            listing = Listing(f"x{i}", intent, 0.6, (Field("city", city, vis, False),), "2026-10-12T09:00:00Z")
            warden = s.warden.admit(Person(f"x{i}", True, "40-49", "2026-10-01T00:00:00Z"), listing, Marketboard())
            args = {"kind": "other", "intent_text": intent, "exactness": 0.6, "details": [], "suggestions": [],
                    "fields": [{"key": "city", "values": [city], "visibility": vis}]}
            model = not any("specific person" in p for p in build_draft(args, "Tomas", "40-49")[1])
            self.assertEqual((phone, model), (warden.accepted, warden.accepted), intent)
            seen.add(warden.accepted)
        self.assertEqual(seen, {True, False})
        # A place on the Warden's last shared list is read the same way (here, one it lists as if 20 people showed it).
        body = self.phone_copy(city="Kyiv")
        body["intent_text"] = "Looking for a partner for trips to the Vang Vieng crags."
        with self.assertRaises(DemoError):
            s.save_draft("A", body)
        s.warden.listed_places = lambda: [{"city": "Vang Vieng", "country": "Laos"}]
        s.save_draft("A", body)

    def test_what_you_say_about_yourself_is_read_as_yours(self):
        """An open detail about the person is read as theirs on the phone too: their own job and looks save, a clue
        about someone else written there is refused with how to change it (as the Warden would refuse it). A pair
        of capitalised words on two lines is explained as one."""
        from belong.agent_llm import IDENTIFIERS_HINT
        s = self.s
        s.save_draft("A", self.phone_copy(me_free_time=["barista at the bus station café"]))
        with self.assertRaises(DemoError) as caught:
            s.save_draft("A", self.phone_copy(me_free_time=["met him at the bus station café, he's tall"]))
        self.assertIn("'free time'", str(caught.exception))
        self.assertIn(IDENTIFIERS_HINT, str(caught.exception))
        body = self.phone_copy()
        body["intent_text"] = "Looking for a climbing partner\nGranite or Limestone\nWeekday mornings"
        with self.assertRaises(DemoError) as caught:
            s.save_draft("A", body)
        self.assertIn("on two lines", str(caught.exception))
        body["intent_text"] = "Looking for a climbing partner.\nGranite or Limestone.\nWeekday mornings."
        s.save_draft("A", body)

    def test_clues_to_one_person_are_read_the_same_by_the_phone_the_agent_and_the_warden(self):
        """The identifiers rule (warden.identifier_problems) over the description and the phrases about the person
        sought, together: Save draft, a model agent's save and the Warden agree, each way. The phone says how to fix
        it; the model is told how, and that a request for one particular person is refused with singles_out."""
        from belong.agent_llm import IDENTIFIERS_HELP, IDENTIFIERS_HINT, build_draft
        from belong.matching.marketboard import Marketboard
        from belong.schema import Field, Listing, Person
        s = self.s
        cases = [("Looking for a climbing partner; I work at the fish market and I'm short.", {}),
                 ("Looking for a climbing partner with a beard and glasses.", {}),
                 ("Looking for a tall climbing partner with a beard and glasses.", {}),
                 ("She has a grey beard and climbs most mornings.", {"seeking": "the woman who works at the gear shop"}),
                 ("Looking for a tall climbing partner.", {"offering": "a cashier at the gear shop"}),
                 ("Hoping to find the man with a scar across his chin who climbs at dawn.", {})]
        seen = set()
        for i, (intent, phrases) in enumerate(cases):
            body = self.phone_copy(city="Lisbon", **phrases)
            body["intent_text"] = intent
            try:
                s.save_draft("A", body)
                phone = True
            except DemoError as e:
                phone = False
                self.assertIn(IDENTIFIERS_HINT, str(e))
            words = {k: v for k, v in [(f["key"], f["value"]) for f in body["fields"]] if k in ("seeking", "offering")}
            fields = (Field("city", "Lisbon", SHOWN, False), *(Field(k, v, SHOWN, False) for k, v in words.items()))
            warden = s.warden.admit(Person(f"x{i}", True, "40-49", "2026-10-01T00:00:00Z"),
                                    Listing(f"x{i}", intent, 0.6, fields, "2026-10-12T09:00:00Z"), Marketboard())
            args = {"kind": "other", "intent_text": intent, "exactness": 0.6, "details": [], "suggestions": [],
                    "fields": [{"key": "city", "values": ["Lisbon"], "visibility": SHOWN},
                               *({"key": k, "values": [v], "visibility": SHOWN} for k, v in words.items())]}
            said = [p for p in build_draft(args, "Tomas", "40-49")[1] if "specific person" in p]
            self.assertEqual((phone, not said), (warden.accepted, warden.accepted), intent)
            self.assertTrue(all(IDENTIFIERS_HELP in p and "singles_out" in p for p in said), said)
            seen.add(warden.accepted)
        self.assertEqual(seen, {True, False})

    def test_the_phone_has_save_draft_until_the_listing_is_live(self):
        from belong import server
        from tests.test_dating import PAGE
        self.assertIn("save_draft", server.ACTIONS)
        self.assertIn("save_draft", server.CHAT_ACTIONS)
        self.assertIn('api("save_draft"', PAGE)
        self.assertNotIn("Edit details", PAGE)
        # Shown or hidden still on its way goes to the server first, as before a message to the agent.
        save = PAGE[PAGE.index("async function saveDraft"):]
        self.assertLess(save.index('api("visibility"'), save.index('api("save_draft"'))


class Scripted:
    """Stands in for a model agent: each message saves one listing through build_draft (the one way every model
    agent saves), with the details it was given, so tests can give a listing open details of any kind."""

    def __init__(self, fields, details):
        self.fields, self.details, self.given = fields, details, []

    def respond(self, conversation, text, name, band, current, in_use=None):
        from belong.agent_llm import build_draft
        self.given.append(in_use)
        draft, problems, _ = build_draft({"kind": "other", "intent_text": text, "exactness": 0.6, "suggestions": [],
                                          "separate_listing": False, "details": self.details,
                                          "fields": [{"key": k, "values": [v], "visibility": SHOWN} for k, v in self.fields.items()]},
                                         name, band, current, in_use)
        assert not problems, problems
        return "Drafted.", draft


def duet_player(s: DemoSession, slot="A", level=("middle",)):
    """A listing with open details of every kind: about the person (a step on a scale), a wish about the other
    person (a run of steps), and something both must share."""
    scale = ["low", "middle", "high"]
    s.verify(slot, "Noor", 33)
    s.llm = Scripted({"city": "Lyon", "country": "France", "seeking": "a duet partner", "offering": "a duet partner",
                      "headline": "Looking for someone to play duets with"},
                     [{"name": "skill", "about": "me", "compare": "steps", "steps": scale, "values": list(level),
                       "visibility": "shown", "required": True},
                      {"name": "skill", "about": "them", "compare": "steps", "steps": scale, "values": ["middle", "high"],
                       "visibility": "shown", "required": True},
                      {"name": "tempo", "about": "both", "compare": "same", "steps": [], "values": ["slow"],
                       "visibility": "shown", "required": False}])
    s.agent_message(slot, "Looking for someone to play duets with in Lyon.")
    return s.slots[slot].listing()


class OpenDetailsOnThePhone(unittest.TestCase):
    """Open details (SPEC §4): the phone gets them as rows with a label, the person changes only their values and
    shown or hidden, and only what their agent wrote goes live."""

    def test_the_rule_based_agent_runs_the_one_check_on_what_it_keeps(self):
        """The rule-based agent keeps the open details it can't write (agent.keep_details), but every save runs the
        one pre-save check: a kept wish that, read with the new description, would single someone out is taken off,
        and the person is told why. The draft it stores is one Save draft and the Warden accept."""
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Noor", 33)
        s.llm = Scripted({"city": "Lyon", "country": "France", "seeking": "a sketching partner",
                          "offering": "a sketching partner", "headline": "Looking for someone to sketch with"},
                         [{"name": "look", "about": "them", "compare": "same", "steps": [], "values": ["silver hair"],
                           "visibility": "shown", "required": False},
                          {"name": "medium", "about": "me", "compare": "same", "steps": [], "values": ["charcoal"],
                           "visibility": "shown", "required": True}])
        s.agent_message("A", "Looking for someone to sketch with in Lyon.")
        s.llm = None
        s.agent_message("A", "Looking for the employee at the riverside bookshop to sketch with in Lyon.")
        mine = s.slots["A"].listing()
        keys = [f["key"] for f in mine.draft["fields"] if ":" in f["key"]]
        self.assertNotIn("them:look", keys)
        self.assertIn("me:medium", keys)  # nothing wrong with it: kept
        self.assertTrue(any("I took out 'look'" in m["text"] for m in mine.chat if m["from"] == "agent"), mine.chat)
        s.save_draft("A", fill(mine.draft), mine.id)  # what it stored passes the one check

    def test_two_kept_wishes_that_complete_one_problem_are_both_taken_out(self):
        """When no one kept detail alone is the problem (two wishes about looks, each completing one problem with a
        place in the new description), the rule-based agent takes out as many as it takes, tells the person, and
        stores a draft the one check and Save draft accept."""
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Noor", 33)
        wish = lambda name, value: {"name": name, "about": "them", "compare": "same", "steps": [], "values": [value],
                                    "visibility": "shown", "required": False}
        s.llm = Scripted({"city": "Ghent", "country": "Belgium", "seeking": "a kite flying partner",
                          "offering": "a kite flying partner", "headline": "Looking for someone to fly kites with"},
                         [wish("stature", "tall"), wish("eyewear", "glasses"),
                          {"name": "kite type", "about": "me", "compare": "same", "steps": [], "values": ["box kites"],
                           "visibility": "shown", "required": True}])
        s.agent_message("A", "Looking for someone to fly kites with in Ghent.")
        s.llm = None
        s.agent_message("A", "Looking for the cashier at the canal-side bakery to fly kites with in Ghent.")
        mine = s.slots["A"].listing()
        keys = [f["key"] for f in mine.draft["fields"] if ":" in f["key"]]
        self.assertNotIn("them:stature", keys)
        self.assertNotIn("them:eyewear", keys)
        self.assertIn("me:kite type", keys)
        notes = [m["text"] for m in mine.chat if m["from"] == "agent"]
        self.assertTrue(any("I took out" in n and "'stature'" in n and "'eyewear'" in n for n in notes), notes)
        self.assertFalse(any("singles_out" in n for n in notes), notes)  # worded for the person
        s.save_draft("A", fill(mine.draft), mine.id)

    def test_the_phone_gets_each_open_detail_with_its_label_and_definition(self):
        s = DemoSession(seed=7, agent=None)
        duet_player(s)
        rows = {f["key"]: f for f in s.phone_state("A")["draft"]["fields"] if ":" in f["key"]}
        self.assertEqual(set(rows), {"me:skill", "them:skill", "both:tempo"})
        wish = rows["them:skill"]
        self.assertEqual({k: wish[k] for k in ("about", "name", "label", "compare", "steps", "value", "visibility", "required")},
                         {"about": "them", "name": "skill", "label": "Skill", "compare": "steps",
                          "steps": ["low", "middle", "high"], "value": ["middle", "high"], "visibility": SHOWN, "required": True})
        self.assertEqual(rows["both:tempo"]["required"], False)

    def test_what_you_like_in_them_is_always_shown(self):
        """And what the request needs you both to share: both are wishes, which only decide anything shown."""
        s = DemoSession(seed=7, agent=None)
        listing = duet_player(s)
        listing.choices["both:tempo"] = HIDDEN  # remembered from before: dropped
        s.set_visibility("A", listing.id, {"them:skill": HIDDEN, "me:skill": HIDDEN, "both:tempo": HIDDEN})
        rows = {f["key"]: f["visibility"] for f in listing.draft["fields"]}
        self.assertEqual((rows["them:skill"], rows["both:tempo"], rows["me:skill"]), (SHOWN, SHOWN, HIDDEN))
        self.assertNotIn("them:skill", listing.choices)
        self.assertNotIn("both:tempo", listing.choices)

    def test_a_hidden_choice_counts_before_the_agent_writes_the_detail(self):
        """Chosen hidden in the editor (or by key, as the phone sends it), an open detail the agent writes later
        starts hidden. A malformed key is never remembered."""
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Yuki", 27)
        lid = s.new_listing("A")
        s.set_visibility("A", lid, {"me:interests": HIDDEN, "me:<b>": HIDDEN, "nobody:x": HIDDEN, "me:age": HIDDEN})
        self.assertEqual(s.slots["A"].listing(lid).choices, {"me:interests": HIDDEN})
        s.agent_message("A", "Looking for a climbing partner in Osaka. I'm into bouldering and ramen.", lid)
        rows = {f["key"]: f for f in s.slots["A"].listing(lid).draft["fields"]}
        self.assertEqual((rows["me:interests"]["value"], rows["me:interests"]["visibility"]), (["bouldering", "ramen"], HIDDEN))

    def test_only_what_the_agent_wrote_goes_live(self):
        """The phone sends values and shown or hidden; what an open detail is (its scale, whose it is) comes from
        the draft, and a detail that isn't on the draft (an old copy's, or made up) never goes live."""
        s = DemoSession(seed=7, agent=None)
        listing = duet_player(s)
        body = fill(listing.draft)
        for f in body["fields"]:
            if f["key"] == "me:skill":
                f.update(value=["high"], steps=["high", "higher"], compare="same")
        body["fields"].append({"key": "me:secret plan", "value": ["x"], "visibility": SHOWN, "compare": "same"})
        s.submit("A", body)
        self.assertTrue(listing.entry["accepted"], listing.entry)
        live = s.board.listing(listing.id)
        self.assertEqual((live.get("me:skill").value, live.get("me:skill").steps, live.get("me:skill").compare),
                         (("high",), ("low", "middle", "high"), "steps"))
        self.assertIsNone(live.get("me:secret plan"))
        self.assertEqual(live.get("both:tempo").required, False)
        self.assertFalse(s.phone_state("A")["listings"][0]["unsaved"])  # the draft is exactly what went live

    def test_a_wish_decides_who_you_meet(self):
        """Two duet players: one whose own skill is outside the other's wish is never introduced to them."""
        s = DemoSession(seed=7, agent=None)
        duet_player(s, "A", ("middle",))
        s.submit("A", fill(s.slots["A"].draft))
        duet_player(s, "B", ("low",))  # B wants middle or high; A is middle, but A wants middle or high and B is low
        s.submit("B", fill(s.slots["B"].draft))
        for _ in range(3):
            s.run_matchmaker(quiet=True)
        self.assertNotIn({"demoA-1", "demoB-1"}, [{i.person_a, i.person_b} for i in s.intros.values()])
        s.submit("B", fill(s.slots["B"].draft, values={"me:skill": ["high"]}))
        for _ in range(3):
            s.run_matchmaker(quiet=True)
        intro = next(i for i in s.intros.values() if {i.person_a, i.person_b} == {"demoA-1", "demoB-1"})
        # Both skills are said once ("Their skill is high; yours is middle."), so the wish says only what was wished.
        self.assertIn("Their skill is high; yours is middle.", intro.a_view["reason"])
        self.assertIn("They'd like your skill to be middle or high.", intro.a_view["reason"])
        self.assertTrue(s.room_state()["leak_report"]["ok"], s.room_state()["leak_report"]["findings"])

    def test_the_rule_based_agent_keeps_open_details_it_cant_write(self):
        """A model agent becomes unavailable mid-conversation: the offline agent drafts the next message, and
        the open details it can't write (not in words other people use) stay."""
        from belong.agent_llm import AgentUnavailable
        s = DemoSession(seed=7, agent=None)
        listing = duet_player(s)

        class Gone:
            def respond(self, *a, **kw):
                raise AgentUnavailable("no key")

        s.llm = Gone()
        s.agent_message("A", "Looking for a climbing partner in Lyon.")
        self.assertIn("me:skill", {f["key"] for f in listing.draft["fields"]})


class TheSharedList(unittest.TestCase):
    """The Warden's list of details in use (SPEC §6) goes to agents, never to phones; each one handed out is
    in the presenter's log and audited by the leak report."""

    def test_agents_get_it_from_the_warden_and_phones_never_do(self):
        s = DemoSession(seed=7, agent=None)
        listing = duet_player(s)
        given = s.llm.given[-1]
        self.assertEqual(given, s.warden.details_in_use(s.board).vocabulary())
        self.assertTrue(given["phrases"])
        phone = json.dumps(s.phone_state("A"))
        self.assertNotIn("details_in_use", phone)
        for entry in given["phrases"] + given["places"]:
            self.assertNotIn(json.dumps(entry), phone)
        self.assertEqual(s.room_state()["details_in_use"], given)  # the presenter sees what agents get
        self.assertIsNotNone(listing.draft)

    def test_each_list_handed_out_is_logged_once_and_audited(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        logged = [e for e in s.events if e["title"].startswith("Details in use")]
        self.assertEqual(len(logged), 1)  # built once that day, however many agents asked for it
        self.assertEqual([r["check"] for r in logged[0]["receipts"]], ["details_in_use"])
        report = s.room_state()["leak_report"]
        self.assertTrue(report["ok"], report["findings"])
        self.assertEqual(report["checked"]["details_lists"], len(s.outputs["details_in_use"]))
        s.clock.advance(days=1)  # a new day: a new list, audited too
        s.details_in_use()
        self.assertEqual(len(s.outputs["details_in_use"]), 2)
        self.assertTrue(s.room_state()["leak_report"]["ok"])

    def test_the_presenter_counts_a_crowd_as_matching_does(self):
        """Different people with the same value (canonical), and usable only with k others besides the pair."""
        from belong.schema import K_CROWD, canonical
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        counts, rows = s.board.value_counts(), 0
        for l in s.room_state()["inside"]:
            live = s.board.listing(f"demo{l['slot']}-1")
            for r in l["fields"]:
                f = live.get(r["key"])
                if f.visibility != HIDDEN:
                    self.assertNotIn("crowd", r)
                    continue
                rows += 1
                self.assertEqual(r["crowd"], max(counts[(f.key, canonical(v))] for v in f.strings()))
                self.assertEqual(r["usable"], r["crowd"] >= K_CROWD + 2)
        self.assertGreaterEqual(rows, 5)


class _Node:
    """Runs named functions from demo.html's script (or another page's) with node."""

    def run_js(self, names, body, page: str = "demo.html"):
        import shutil
        import subprocess
        import tempfile
        from tests.test_dating import _js
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        from pathlib import Path
        text = (Path(__file__).resolve().parent.parent / "belong" / "web" / page).read_text()
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write("\n".join(_js(n, text) for n in names) + "\n" + body)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)


class PhoneCopy(_Node, unittest.TestCase):
    """demo.html (run with node): the phone's own copy of a draft never brings back an older shown or hidden,
    so what Start matching or Save draft sends is what the person last chose."""

    def test_what_a_copy_set_aside_goes_with_it(self):
        """A new draft from the server replaces the phone's copy, and what that copy set aside on a kind change
        goes too: it can't come back later with a shown (and a confirmation) the person has since changed."""
        server = {"fields": [{"key": "kinks", "value": ["x"], "visibility": HIDDEN, "sensitive": True}]}
        out = self.run_js(("function visibilityChanges", "function sameValues", "function syncDraft"), f"""
const S = {json.dumps(server)}, c = {{ draft: {{ fields: [] }}, draftFrom: "{{}}", keepLocal: 0,
  stash: {{ kinks: {{ key: "kinks", value: ["x"], visibility: "shown", confirmed: true }} }} }};
const curListing = () => ({{ draft: S }}), cur = () => c, SCENARIOS = {{}}, scenario = null;
syncDraft("A");
console.log(JSON.stringify(c.stash));""")
        self.assertEqual(out, {})

    def test_a_detail_a_new_kind_adds_starts_with_the_listings_choice(self):
        from belong import agent
        layout = agent.kind_layout("dating")
        out = self.run_js(("const hasValue", "function relayout"), f"""
const fields = relayout([], {json.dumps(layout)}, {{}}, {{ gender: "hidden", kinks: "shown" }});
console.log(JSON.stringify(Object.fromEntries(fields.map(f => [f.key, [f.visibility, !!f.confirmed]]))));""")
        self.assertEqual(out["gender"], [HIDDEN, False])  # hidden on this listing before: hidden, not the default
        self.assertEqual(out["kinks"], [SHOWN, True])  # shown before, which needed their confirmation
        self.assertEqual(out["wants_gender"][0], next(r for r in layout["rows"] if r["key"] == "wants_gender")["visibility"])

    def test_a_shown_chosen_on_a_new_detail_is_sent(self):
        """A row the server's copy doesn't have yet is sent when the person chose it, shown or hidden; a
        default shown isn't a choice."""
        out = self.run_js(("function visibilityChanges",), """
const from = JSON.stringify({ fields: [{ key: "city", visibility: "shown" }] });
const draft = { fields: [{ key: "city", visibility: "shown" }, { key: "children", visibility: "shown", chosen: true },
                         { key: "gender", visibility: "shown" }] };
console.log(JSON.stringify(visibilityChanges(draft, from)));""")
        self.assertEqual(out["visibility"], {"children": SHOWN})


class OpenDetailsInThePage(_Node, unittest.TestCase):
    """demo.html (run with node): open details on the phone, from their key and their agent's definition."""

    def test_the_key_is_read_as_the_server_reads_it(self):
        from belong.schema import split_key
        keys = ["me:skill", "them:skill level", "both:tempo", "city", "nobody:x", "me:", "wants_age"]
        out = self.run_js(("const detailParts",), f"console.log(JSON.stringify({json.dumps(keys)}.map(detailParts)));")
        for key, parts in zip(keys, out):
            about, name = split_key(key)
            self.assertEqual(parts, {"about": about, "name": name} if about else None, key)

    def test_what_you_like_in_them_stays_a_run_of_neighbouring_steps(self):
        out = self.run_js(("function toggleStep",), """
const f = { steps: ["a", "b", "c", "d"], value: [] }, seen = [];
const cur = () => ({ draft: { fields: [f] } }), renderPhone = () => {};
for (const n of [1, 2, 0, 3, 0, 1, 3, 2, 0]) { toggleStep("A", 0, n); seen.push(f.value.join("")); }
console.log(JSON.stringify(seen));""")
        # Next to the run adds, an end comes off (never the last step: a wish has a value), anything else starts again.
        self.assertEqual(out, ["b", "bc", "abc", "abcd", "bcd", "cd", "c", "c", "a"])

    def test_a_match_card_names_each_open_detail_and_keeps_their_wishes_apart(self):
        fields = {"first_name": "Lea", "city": "Lyon", "me:skill": {"values": ["middle"], "compare": "steps", "steps": ["low", "middle"], "required": True},
                  "them:skill": {"values": ["low", "middle"], "compare": "steps", "steps": ["low", "middle"], "required": False},
                  "both:tempo": {"values": ["slow", "medium"], "compare": "same", "steps": [], "required": True}}
        out = self.run_js(("const val", "const orList", "const cap", "const GENDER_WORDS", "const spanOf", "const agesText", "function wantsText",
                           "const UNDERSTOOD", "const detailParts", "const isDetail", "const detailValues", "function matchFacts"),
                          f"console.log(JSON.stringify(matchFacts({json.dumps(fields)}, false)));")
        self.assertEqual(out["facts"], ["Lyon", "Skill: middle", "Tempo: slow or medium"])
        self.assertEqual(out["wishes"], ["Skill: low or middle (ideally)"])

    def test_a_new_scale_from_the_agent_is_a_new_draft(self):
        """The phone's own visibility push is told apart from an agent's redraft by the values, so a changed
        scale (same value) must count: otherwise the person would pick from steps that no longer exist."""
        out = self.run_js(("function sameValues",), """
const a = { fields: [{ key: "me:skill", value: ["mid"], enabled: true, compare: "steps", steps: ["low", "mid", "high"] }] };
const b = { fields: [{ key: "me:skill", value: ["mid"], enabled: true, compare: "steps", steps: ["mid", "top"] }] };
console.log(JSON.stringify([sameValues(a, a), sameValues(a, b)]));""")
        self.assertEqual(out, [True, False])

    def test_a_new_kind_keeps_the_open_details(self):
        from belong import agent
        layout = agent.kind_layout("dating")
        out = self.run_js(("const hasValue", "function relayout"), f"""
const fields = [{{ key: "seeking", value: "a duet partner" }}, {{ key: "me:skill", value: ["mid"] }}, {{ key: "them:skill", value: ["mid"] }}];
const stash = {{}};
const kept = relayout(fields, {json.dumps(layout)}, stash, {{}});
console.log(JSON.stringify([kept.map(f => f.key).filter(k => k.includes(":")), Object.keys(stash)]));""")
        self.assertEqual(out, [["me:skill", "them:skill"], ["seeking"]])


def page_scenarios(page: str) -> dict:
    """The demo page's SCENARIOS, read from its script (a JavaScript object of plain strings and numbers): keys
    quoted, trailing commas dropped, strings left exactly as written."""
    import re
    src, out, i = re.search(r"const SCENARIOS = (\{.*?\n\});\n", page, flags=re.S).group(1), [], 0
    while i < len(src):
        if src[i] == '"':
            j = i + 1
            while src[j] != '"':
                j += 2 if src[j] == "\\" else 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        key = re.match(r"[A-Za-z_]\w*(?=\s*:)", src[i:])
        if key and not re.match(r"\w", src[i - 1:i] or " "):
            out.append(json.dumps(key.group()))
            i += key.end()
            continue
        out.append(src[i])
        i += 1
    return json.loads(re.sub(r",(\s*[}\]])", r"\1", "".join(out)))


class ThePagesPresets(unittest.TestCase):
    """Every preset on the demo page plays with the offline agent, as its fast-forward plays it (verify, the
    preset's message, up to two nudges to draft, the preset's values and hidden choices, Start matching, then the
    Matchmaker until it sends nothing more): the two people are introduced. A preset that needs a model agent says
    so in its label, and the one that tries to find a specific person is refused at entry. Read from demo.html,
    never copied here."""

    NEEDS_A_MODEL = "(needs the AI agent)"
    REFUSED = {"stalker"}

    def play(self, preset: dict, nudge: str) -> DemoSession:
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        for slot in "AB":
            p = preset[slot]
            s.verify(slot, p["name"], p["age"])
            s.new_listing(slot)
            s.agent_message(slot, p["message"])
            for _ in range(2):
                if s.slots[slot].listing().draft is None:
                    s.agent_message(slot, nudge)
            draft = s.slots[slot].listing().draft
            self.assertIsNotNone(draft, (p["name"], s.slots[slot].listing().chat[-1]["text"]))
            fields = []
            for f in draft["fields"]:  # applyScenario, then what submit() sends
                f = dict(f)
                if f["key"] in p.get("values", {}):
                    f["value"] = p["values"][f["key"]]
                if f["key"] in p.get("hidden", {}):
                    f.update(value=p["hidden"][f["key"]], visibility=HIDDEN, enabled=True)
                if f.get("enabled") and f.get("value") not in (None, "", []):
                    fields.append(f)
            s.submit(slot, {"intent_text": draft["intent_text"], "exactness": draft["exactness"], "fields": fields})
        return s

    def test_every_preset_the_offline_agent_can_play_introduces_its_two_people(self):
        import re
        from tests.test_dating import PAGE
        presets = page_scenarios(PAGE)
        nudge = re.search(r'api\("agent", \{ slot, listing_id, text: "([^"]+)" \}\)', PAGE).group(1)
        self.assertTrue(self.REFUSED <= set(presets))
        played = 0
        for key, preset in presets.items():
            if self.NEEDS_A_MODEL in preset["label"] or not preset["A"]:
                continue
            with self.subTest(preset=key):
                s = self.play(preset, nudge)
                entries = [s.slots[slot].listing().entry for slot in "AB"]
                if key in self.REFUSED:
                    self.assertTrue(all(e and not e["accepted"] and e["problems"] for e in entries), entries)
                    continue
                self.assertTrue(all(e and e["accepted"] for e in entries), entries)
                self.assertTrue(all(s.phone_state(slot)["listings"][0]["ready"] for slot in "AB"))
                s.run_matchmaker()
                for _ in range(10):
                    if not s.run_matchmaker(quiet=True):
                        break
                pair = {s.slots["A"].listing().id, s.slots["B"].listing().id}
                self.assertTrue(any({i.person_a, i.person_b} == pair for i in s.intros.values()), key)
                played += 1
        self.assertGreaterEqual(played, 3)

    def test_fast_forward_and_start_matching_stop_without_a_draft(self):
        """An agent that still has a question has drafted nothing: the page says so, and sends nothing."""
        from tests.test_dating import PAGE
        ff = PAGE[PAGE.index("async function fastForward"):]
        ff = ff[:ff.index("\n}\n")]
        self.assertLess(ff.index("if (!curListing(slot).draft)", ff.index("syncDraft(slot)")), ff.index("await submit(slot)"))
        submit = PAGE[PAGE.index("async function submit"):]
        submit = submit[:submit.index("\n}\n")]
        self.assertLess(submit.index("if (!d) {"), submit.index("d.intent_text"))


class ListsOnThePages(_Node, unittest.TestCase):
    """A list of choices reads "a, b or c" on the phone and in the connector, by one helper on each page."""

    def test_a_match_card_lists_three_choices_as_a_sentence_does(self):
        fields = {"them:tempo": {"values": ["slow", "medium", "quick"], "compare": "steps", "steps": ["slow", "medium", "quick"],
                                 "required": True},
                  "both:venue": {"values": ["hall", "yard", "roof"], "compare": "same", "steps": [], "required": True},
                  "me:tools": {"values": ["saw", "drill"], "compare": "same", "steps": [], "required": True}}
        out = self.run_js(("const val", "const andList", "const orList", "const cap", "const GENDER_WORDS", "const spanOf",
                           "const agesText", "function wantsText", "const UNDERSTOOD", "const detailParts", "const isDetail",
                           "const detailValues", "function matchFacts"),
                          f"console.log(JSON.stringify([matchFacts({json.dumps(fields)}, false), orList(['a']), orList(['a', 'b']),"
                          f" wantsText(['woman', 'nonbinary'], null)]));")
        self.assertEqual(out[0]["wishes"], ["Tempo: slow, medium or quick"])
        self.assertEqual(out[0]["facts"], ["Venue: hall, yard or roof", "Tools: saw, drill"])
        self.assertEqual(out[1:], ["a", "a or b", "a woman or a non-binary person"])

    def test_one_helper_on_each_page(self):
        import re
        from pathlib import Path
        web = Path(__file__).resolve().parent.parent / "belong" / "web"
        for name in ("orList", "detailValues"):
            helper = lambda page: re.search(rf"^const {name} = .*$", (web / page).read_text(), flags=re.M).group(0)
            self.assertEqual(helper("chat.html"), helper("demo.html"), name)
        for page in ("demo.html", "chat.html"):
            self.assertNotIn('join(" or ")', (web / page).read_text(), page)

    def test_the_connector_reads_a_detail_by_whom_it_is_about(self):
        """The connector's own listing card and its match card read a detail both must share, or a wish, as
        "a, b or c", and the person's own values as "a, b": one helper (detailValues), as on the phone."""
        fields = [{"key": "city", "label": "City", "value": "Ghent", "visibility": "shown", "enabled": True},
                  {"key": "both:venue", "label": "Venue", "value": ["hall", "yard", "roof"], "visibility": "shown",
                   "enabled": True, "compare": "same", "steps": [], "required": True},
                  {"key": "them:tempo", "label": "Tempo", "value": ["slow", "medium"], "visibility": "shown",
                   "enabled": True, "compare": "steps", "steps": ["slow", "medium", "quick"], "required": False},
                  {"key": "me:tools", "label": "Tools", "value": ["saw", "drill"], "visibility": "shown", "enabled": True,
                   "compare": "same", "steps": [], "required": True}]
        view = {"first_name": "Mira", "city": "Ghent",
                "both:venue": {"values": ["hall", "yard", "roof"], "compare": "same", "steps": [], "required": True},
                "me:tools": {"values": ["saw", "drill"], "compare": "same", "steps": [], "required": True}}
        out = self.run_js(("const esc", "const cap", "const val", "const orList", "const detailParts", "const isDetail",
                           "const detailValues", "const shownOf", "function listingCard", "function introCard"), f"""
const matchingStatus = () => "";
const ME = {{ listings: [{{ title: "a workshop", intros: [{{ intro_id: "i1", status: "pending",
  view: {{ fields: {json.dumps(view)}, reason: "" }} }}] }}] }};
const card = listingCard({{ id: "x", listed: false, title: "t", ready: false,
  draft: {{ intent_text: "x", fields: {json.dumps(fields)} }} }});
console.log(JSON.stringify([card, introCard("i1")]));""", page="chat.html")
        card, match = out
        self.assertIn("<dt>Venue</dt><dd>hall, yard or roof</dd>", card)
        self.assertIn("<dt>Tools</dt><dd>saw, drill</dd>", card)
        self.assertIn("<dt>Tempo</dt><dd>slow or medium (ideally)</dd>", card)
        self.assertIn("Venue: hall, yard or roof", match)
        self.assertIn("Tools: saw, drill", match)


class AReasonOnlyWhenThereIsOne(_Node, unittest.TestCase):
    """A reissue the Warden had to reduce has no reason (warden._reduced_views): the card shows no empty line."""

    def test_an_introduction_without_a_reason_shows_none(self):
        names = ("const esc", "const badge", "const tags", "const val", "const andList", "const orList", "const cap",
                 "const GENDER_WORDS", "const spanOf", "const agesText", "function wantsText", "const UNDERSTOOD",
                 "const detailParts", "const isDetail", "const detailValues", "function matchFacts", "function introCard")
        out = self.run_js(names, """
const card = reason => introCard("A", { kind: "other" }, { intro_id: "i1", status: "pending", close: false,
  view: { fields: { first_name: "Ana", city: "Porto" }, reason } });
console.log(JSON.stringify([card(""), card("You're both in Porto.")]));""")
        self.assertNotIn('class="reason"', out[0])
        self.assertIn('<p class="reason">You&#39;re both in Porto.</p>', out[1])


class WishesHaveNoHiddenChoice(_Node, unittest.TestCase):
    """The editor offers shown or hidden only for what's about the person: what they'd like in the other person
    and what the request needs them both to share are always shown (schema refuses either hidden)."""

    def test_only_details_about_you_can_be_hidden(self):
        names = ("const esc", "const badge", "const val", "const hasValue", "const KIND_LABELS", "const detailParts",
                 "function fieldRow")
        rows = [{"key": f"{about}:pitch", "label": "Pitch", "value": ["low"], "visibility": "shown", "compare": "same",
                 "steps": [], "enabled": True} for about in ("me", "them", "both")]
        out = self.run_js(names, f"""
const rows = {json.dumps(rows)}, cur = () => ({{ draft: {{ fields: rows }} }});
console.log(JSON.stringify(rows.map(f => fieldRow("A", f))));""")
        me, them, both = out
        self.assertIn("setVis(", me)
        for row in (them, both):
            self.assertNotIn("setVis(", row)
            self.assertIn("Always shown", row)


class _Page:
    """Runs demo.html's whole script with node on a small stand-in for the browser: fetch never answers and timers run
    only when a test runs them, so a test sets the state (S, from a real DemoSession), draws a phone and reads it."""

    STUB = r"""
const vm = require("vm");
const store = {}, timers = [], ids = {}, found = {}, heights = { full: 700, min: 100 };
function el(id, html) {
  return { id, html, style: {}, dataset: {}, value: "", scrollTop: 0, events: [], clientHeight: 720, scrollHeight: 1000,
    classList: { add() {}, remove() {}, toggle() {}, contains: () => false },
    get innerHTML() { return store[id] || ""; }, set innerHTML(v) { store[id] = v; },
    get offsetHeight() { const m = /class="chat-sheet( full| min)?"/.exec(this.html || ""); return m ? (m[1] ? heights[m[1].trim()] : 430) : 0; },
    insertAdjacentHTML() {}, focus() {}, setSelectionRange() {}, scrollIntoView() {}, setPointerCapture() {},
    addEventListener(type, f) { this.events.push(type); (this.on ||= {})[type] = f; }, removeEventListener() {},
    contains: () => false, closest: () => null,
    querySelector: () => null };
}
const document = {
  body: el("body"), documentElement: el("html"), getElementById: id => ids[id] ||= el(id),
  querySelector(sel) {
    const m = /^#phone-(\w+) \.([\w.-]+)$/.exec(sel);
    if (m) {
      const html = store["phone-" + m[1]] || "";
      if (!html.includes(`class="${m[2].replace(/\./g, " ")}`)) return null;
      return found[sel + "\n" + html] ||= el(null, html);  // the same element until the phone is drawn again
    }
    return /^\.(phone-wrap|stage)/.test(sel) ? el(null) : null;
  },
  querySelectorAll: () => [], addEventListener() {}, contains: () => true,
};
document.activeElement = document.body;
const ctx = vm.createContext({ console, document, location: { search: "" }, URLSearchParams, CSS: { escape: s => s },
  matchMedia: q => ({ matches: /reduce/.test(q) ? REDUCED : false }), confirm: () => true,
  fetch: () => new Promise(() => {}), setInterval: () => 0, clearTimeout() {},
  setTimeout: (f, ms) => timers.push({ f, ms }) });
ctx.window = ctx; ctx.top = ctx;
vm.runInContext(SCRIPT, ctx);
Object.assign(ctx, { timers, phone: slot => store["phone-" + slot] || "",
  sheet: slot => document.querySelector(`#phone-${slot} .chat-sheet`),
  runTimers: () => timers.splice(0).map(t => (t.f(), t.ms)) });
vm.runInContext(BODY, ctx);
"""

    def run_page(self, body: str, reduced: bool = False):
        import re
        import shutil
        import subprocess
        import tempfile
        from tests.test_dating import PAGE
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        script = "\n".join(re.findall(r"<script>(.*?)</script>", PAGE, flags=re.S))
        code = f"const SCRIPT = {json.dumps(script)}, BODY = {json.dumps(body)}, REDUCED = {json.dumps(reduced)};\n" + self.STUB
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)


class AListingHasItsProfileForGood(unittest.TestCase):
    """The server keeps whether the phone shows a listing's profile (has_profile): from the first time the listing is
    live or its draft is ready, for good, so every page and a reload agree; a reset starts again. The offline agent
    points to the profile only once it's there."""

    WHAT = "Looking for a language exchange partner."
    WHAT_AND_WHERE = "Looking for a language exchange partner in Osaka."
    ANOTHER = "Actually, I'd rather find a date."  # a different request: who they'd like to meet isn't said yet

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.s.run_inline = True
        self.s.verify("A", "Yuki", 27)
        self.s.new_listing("A")

    def say(self, text):
        self.s.prefill("A", text)
        self.s.agent_message("A", text)
        l = self.s.phone_state("A")["listings"][0]
        return l, [m["text"] for m in l["chat"][l["chat"].index({"from": "me", "text": text}) + 1:]]

    def test_once_ready_it_has_its_profile_for_good(self):
        self.assertIs(self.s.phone_state("A")["listings"][0]["has_profile"], False)
        l, _ = self.say(self.WHAT)
        self.assertEqual((l["ready"], l["has_profile"]), (False, False))
        l, _ = self.say(self.WHAT_AND_WHERE)
        self.assertEqual((l["ready"], l["has_profile"]), (True, True))
        l, _ = self.say(self.ANOTHER)  # it replaces the request, and this one isn't ready yet
        self.assertEqual((l["ready"], l["has_profile"]), (False, True))
        self.s.reset()
        self.s.verify("A", "Yuki", 27)
        self.s.new_listing("A")
        l = self.s.phone_state("A")["listings"][0]
        self.assertEqual((l["id"], l["has_profile"]), ("demoA-1", False))  # the same id, a new listing

    def test_the_offline_agent_points_to_the_profile_only_once_its_there(self):
        _, before = self.say(self.WHAT)
        self.assertTrue(before[0].startswith("Here's what I understood:"))
        self.assertNotIn("Review it", before[0])
        self.assertNotIn("drafted your listing", before[0])
        self.assertTrue(any("?" in t for t in before[1:]))  # it asks for what's still needed
        _, ready = self.say(self.WHAT_AND_WHERE)
        self.assertIn("Review it", ready[0])
        _, later = self.say(self.ANOTHER)  # not ready any more, but the profile is on their screen
        self.assertIn("Review it", later[0])


# Requests given in pieces, each with the person who gives it: what they're looking for and where they are in separate
# messages (either order), a date in three, and one whose kind the agent asks about first. The last message of each is
# the first that leaves the draft ready.
IN_PIECES = {
    "a partner, then where": (("Tomas", 33), ["I'd like a bouldering partner.", "I'm based in Nagoya."]),
    "where, then teaching": (("Kenji", 29), ["I live in Kobe.", "I teach bouldering."]),
    "a date in three": (("Greta", 44), ["I'm hoping to find a relationship.",
                                        "I'm a woman and I'd like to meet a man, 40 to 50.", "I'm based in Hamburg."]),
    "a date asked about first": (("Ana", 38), ["I live in Braga.", "I'd like to meet a man. I love climbing.",
                                               "A date"]),
    "a request of its own instead of an answer": (("Lena", 31), [
        "I live in Hamburg. I'd like to meet a woman. I love bouldering.", "I teach bouldering."]),
}


def in_pieces(name):
    """A session that has said each message of IN_PIECES[name] in turn, and the phone's state after each turn."""
    (who, age), messages = IN_PIECES[name]
    s = DemoSession(seed=7, agent=None)
    s.run_inline = True
    s.verify("A", who, age)
    s.new_listing("A")
    states = [json.loads(json.dumps({"phones": {"A": s.phone_state("A")}}))]
    for text in messages:
        s.prefill("A", text)
        s.agent_message("A", text)
        states.append(json.loads(json.dumps({"phones": {"A": s.phone_state("A")}})))
    return s, states


class ARequestInPieces(unittest.TestCase):
    """The offline agent builds on what was already said: what someone is looking for and where they are, given in
    separate messages, make a ready draft; a message that only adds something keeps it ready; one that plainly starts
    a different request replaces it, keeping what's about the person. Every hidden choice stays."""

    def test_what_and_where_in_separate_messages_make_a_ready_draft(self):
        for name in IN_PIECES:
            with self.subTest(name):
                _, states = in_pieces(name)
                listings = [st["phones"]["A"]["listings"][0] for st in states[1:]]
                self.assertEqual([l["ready"] for l in listings], [False] * (len(listings) - 1) + [True])
                self.assertEqual([l["has_profile"] for l in listings], [False] * (len(listings) - 1) + [True])
                # Until there's a profile on their screen, no reply points to one; each asks for what's still needed.
                chat = listings[-2]["chat"]
                replies = [m["text"] for m in chat if m["from"] == "agent"][1:]
                self.assertTrue(replies)
                for text in replies:
                    self.assertNotIn("Review", text)
                    self.assertNotIn("drafted your listing", text)
                self.assertTrue(any("?" in t for t in replies))
                self.assertIn("Review it", next(m["text"] for m in listings[-1]["chat"][len(chat):]
                                                if m["from"] == "agent"))

    def test_a_message_that_only_adds_something_keeps_it_ready(self):
        s, _ = in_pieces("a partner, then where")
        before = s.slots["A"].listing().draft
        keep = {f["key"]: f["value"] for f in before["fields"] if f["value"] not in (None, "", [])}
        for text in ("I'm free weekday mornings.", "I'm into cycling and board games.", "I'm also into chess."):
            s.prefill("A", text)
            s.agent_message("A", text)
            l = s.phone_state("A")["listings"][0]
            self.assertTrue(l["ready"], text)
            now = {f["key"]: f["value"] for f in l["draft"]["fields"] if f["value"] not in (None, "", [])}
            self.assertEqual({k: now[k] for k in keep}, keep)
        self.assertEqual(now["me:free time"], ["weekday mornings"])
        self.assertEqual(now["me:interests"], ["cycling", "board games", "chess"])
        self.assertTrue(l["draft"]["intent_text"].startswith("I'd like a bouldering partner. I'm based in Nagoya."))

    def test_a_different_request_replaces_it_keeping_whats_about_the_person(self):
        s, _ = in_pieces("where, then teaching")
        for text in ("I'm into karaoke.", "Actually, I want to find a boyfriend."):
            s.prefill("A", text)
            s.agent_message("A", text)
        l = s.phone_state("A")["listings"][0]
        now = {f["key"]: f["value"] for f in l["draft"]["fields"] if f["value"] not in (None, "", [])}
        self.assertEqual(now["kind"], "dating")
        self.assertEqual((now["city"], now["wants_gender"], now["me:interests"]), ("Kobe", ["man"], ["karaoke"]))
        self.assertFalse({"seeking", "offering", "headline", "topics"} & set(now))
        self.assertEqual(l["draft"]["intent_text"], "Actually, I want to find a boyfriend.")
        self.assertTrue(l["ready"])

    def test_every_hidden_choice_stays(self):
        s, _ = in_pieces("a date in three")
        listing = s.slots["A"].listing()
        s.set_visibility("A", listing.id, {"gender": HIDDEN, "wants_age": HIDDEN})
        for text in ("I'm into hiking.", "I'm Buddhist.", "I'm free weekends."):
            s.prefill("A", text)
            s.agent_message("A", text)
        s.set_visibility("A", listing.id, {"me:interests": HIDDEN})
        s.prefill("A", "I'm also into jazz.")
        s.agent_message("A", "I'm also into jazz.")
        fields = {f["key"]: f for f in listing.draft["fields"]}
        for key, value in (("gender", "woman"), ("wants_age", "40-50"), ("religion", "Buddhist"),
                           ("me:interests", ["hiking", "jazz"])):
            self.assertEqual((fields[key]["value"], fields[key]["visibility"]), (value, HIDDEN), key)
        self.assertEqual(fields["me:free time"]["visibility"], SHOWN)

    @staticmethod
    def say(s, *messages):
        for text in messages:
            s.prefill("A", text)
            s.agent_message("A", text)
        return s.phone_state("A")["listings"][0]

    @staticmethod
    def shown(l):
        return {f["key"]: f["value"] for f in l["draft"]["fields"] if f["value"] not in (None, "", [])}

    def test_a_request_of_its_own_instead_of_an_answer_keeps_what_the_first_message_said(self):
        """The agent asked about the first message (a date, or a partner for that activity?) and the person answered
        with a request of its own: what the first message said about them stays (where they are), and its request
        gives way to the new one, none of its words in the new description."""
        s, states = in_pieces("a request of its own instead of an answer")
        l = states[-1]["phones"]["A"]["listings"][0]
        now = self.shown(l)
        self.assertEqual((now["kind"], now["offering"], now["city"], now["country"]),
                         ("other", "a bouldering teacher", "Hamburg", "Germany"))
        self.assertEqual(l["draft"]["intent_text"], "I teach bouldering.")
        self.assertNotIn("Which city?", " ".join(m["text"] for m in l["chat"]))

    def test_two_clean_messages_make_a_draft_matching_admits(self):
        """Each message is a sentence of its own in the description, so a capitalised word ending one and another
        starting the next never look like a full name: Save draft and Start matching both take it."""
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        s.verify("A", "Tomas", 33)
        s.new_listing("A")
        l = self.say(s, "I'd like a bouldering partner, ideally Canadian", "Weekday mornings suit me",
                     "I'm based in Nagoya")
        self.assertTrue(l["ready"])
        self.assertEqual(l["draft"]["intent_text"],
                         "I'd like a bouldering partner, ideally Canadian. Weekday mornings suit me. I'm based in Nagoya")
        s.save_draft("A", fill(l["draft"]))
        s.submit("A", fill(l["draft"]))
        self.assertTrue(s.slots["A"].listing().entry["accepted"], s.slots["A"].listing().entry)
        self.assertNotIn("Before you start matching", " ".join(m["text"] for m in l["chat"]))

    def test_the_agent_says_what_the_warden_would_refuse_once_the_profile_is_on_screen(self):
        """Their own words that the Warden's rules for a description would refuse (here, what looks like a full name)
        are said in the chat once the profile is on their screen, never before, as Save draft words them."""
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        s.verify("A", "Tomas", 33)
        s.new_listing("A")
        first = json.loads(json.dumps(self.say(s, "I'd like a bouldering partner, I train at Granite Works")))
        self.assertFalse(first["has_profile"])
        self.assertNotIn("Before you start matching", " ".join(m["text"] for m in first["chat"]))
        l = self.say(s, "I'm based in Nagoya")
        self.assertTrue(l["has_profile"])
        said = [m["text"] for m in l["chat"][len(first["chat"]):] if m["from"] == "agent"]
        self.assertTrue(any(t.startswith("Before you start matching") and "Granite Works" in t for t in said), said)
        with self.assertRaises(DemoError):
            s.save_draft("A", fill(l["draft"]))

    def test_a_request_they_deny_or_someone_elses_never_replaces_theirs(self):
        s, _ = in_pieces("a partner, then where")
        before = json.loads(json.dumps(self.shown(self.say(s, "I'm Buddhist."))))
        for text in ("I'm definitely not looking for romance.", "My flatmate wants a boyfriend though."):
            with self.subTest(text):
                l = self.say(s, text)
                self.assertNotIn("Or both?", l["chat"][-1]["text"])  # no question about the kind
                self.assertEqual(self.shown(l), before)
                fields = {f["key"]: f for f in l["draft"]["fields"]}
                self.assertEqual((fields["religion"]["value"], fields["religion"]["visibility"]), ("Buddhist", HIDDEN))
                self.assertTrue(l["ready"])

    def test_ages_it_cant_place_leave_the_ages_set_unremarked(self):
        s, _ = in_pieces("a date in three")
        l = self.say(s, "I'm 44 and my mum is 69.")
        self.assertEqual(self.shown(l)["wants_age"], "40-50")
        self.assertNotIn("ages", " ".join(m["text"] for m in l["chat"][-2:]))

    def test_how_exact_stays_as_they_set_it(self):
        s, _ = in_pieces("a partner, then where")
        listing = s.slots["A"].listing()
        s.save_draft("A", {**fill(listing.draft), "exactness": 0.15})
        l = self.say(s, "I can only make it on weekday mornings.")
        self.assertEqual(l["draft"]["exactness"], 0.15)
        self.assertTrue(l["ready"])

    def test_the_rules_for_reading_a_message_still_hold_in_later_ones(self):
        """Only what they say of themselves, never a denial, and the agent asks again for what's still needed."""
        s2 = DemoSession(seed=7, agent=None)
        s2.run_inline = True
        s2.verify("A", "Tomas", 33)
        s2.new_listing("A")
        for text in ("I'd like a bouldering partner.", "My sister lives in Kyoto.", "I'm not in Osaka any more.",
                     "I'd like someone who is into chess."):
            s2.prefill("A", text)
            s2.agent_message("A", text)
            l = s2.phone_state("A")["listings"][0]
            self.assertFalse(l["ready"], text)
            self.assertFalse({"city", "country", "me:interests"} & {f["key"] for f in l["draft"]["fields"]
                                                                     if f["value"] not in (None, "", [])}, text)
            self.assertIn("Which city?", l["chat"][-1]["text"])


class ANewListingStartsAsTheChat(_Page, unittest.TestCase):
    """demo.html, run with node on real phone states (the offline agent): a new listing is the agent's chat, full
    screen, with no profile behind it and no way to minimise it. The first time a reply leaves the draft ready (canSave,
    agent.ready), the phone's own status line shows for a while and until its saves finish, then the chat slides down
    to its header and the profile is there, once. A listing that's live, or whose draft is ready, opens showing it."""

    WHAT = "Looking for a language exchange partner."
    WHAT_AND_WHERE = "Looking for a language exchange partner in Osaka."

    @classmethod
    def setUpClass(cls):
        def session():
            s = DemoSession(seed=7, agent=None)
            s.run_inline = True
            s.verify("A", "Yuki", 27)
            s.new_listing("A")
            return s

        def say(s, text):
            s.prefill("A", text)
            s.agent_message("A", text)

        # Each state as the phone got it at that turn (a copy: phone_state hands out the listing's own chat list).
        state = lambda s: json.loads(json.dumps({"phones": {"A": s.phone_state("A")}}))
        s = session()
        cls.new = state(s)
        for name, text in (("not_ready", cls.WHAT), ("ready", cls.WHAT_AND_WHERE),
                           ("again", AListingHasItsProfileForGood.ANOTHER), ("ready_again", cls.WHAT_AND_WHERE)):
            say(s, text)
            setattr(cls, name, state(s))
        s.submit("A", fill(s.slots["A"].listing().draft))
        cls.live = state(s)
        # Once shown, the person hides where they are (as the detail sheet saves it): the draft isn't ready any more.
        h = session()
        say(h, cls.WHAT_AND_WHERE)
        state(h)
        listing = h.slots["A"].listing()
        h.set_visibility("A", listing.id, {f["key"]: HIDDEN for f in listing.draft["fields"]
                                           if f["key"] in ("city", "country") and f.get("value")})
        cls.place_hidden = state(h)
        first = lambda st: st["phones"]["A"]["listings"][0]
        assert [first(x)["ready"] for x in (cls.new, cls.not_ready, cls.ready, cls.again, cls.ready_again)] == \
            [False, False, True, False, True]
        assert [first(x)["has_profile"] for x in (cls.new, cls.not_ready, cls.ready, cls.again)] == [False, False, True, True]
        assert [len(first(x)["chat"]) for x in (cls.new, cls.not_ready, cls.ready)] == [1, 4, 6]
        assert first(cls.live)["listed"]
        assert not first(cls.place_hidden)["ready"] and first(cls.place_hidden)["has_profile"]

    def show(self, *states, then: str = "", reduced: bool = False):
        """Open the listing on phone A (as Create a listing or Edit does), then draw it with each state in turn."""
        return self.run_page(f"""
const states = {json.dumps(states)}, seen = [];
local.A.view = "profile"; local.A.current = states[0].phones.A.current;
for (const st of states) {{ S = st; renderPhone("A"); seen.push(phone("A")); }}
{then}
console.log(JSON.stringify({{ seen, after: phone("A"), c: cur("A"), delays: timers.map(t => t.ms), out: typeof out === "undefined" ? null : out }}));""", reduced)

    def assertChatOnly(self, html):
        self.assertIn('class="chat-sheet full"', html)
        self.assertNotIn("profile-pane", html)
        self.assertNotIn("sheetDown", html)  # no edge to drag or press: nothing to minimise to
        self.assertIn("Your profile appears once your agent knows what you're looking for and where.", html)
        self.assertNotIn("Putting your profile together", html)

    def assertProfile(self, html):
        self.assertIn('class="profile-pane', html)
        self.assertIn('class="chat-sheet min"', html)
        self.assertNotIn("Putting your profile together", html)

    def test_a_new_listing_is_the_chat_full_screen_with_no_profile(self):
        out = self.show(self.new, self.not_ready)
        for html in out["seen"]:
            self.assertChatOnly(html)
        self.assertIn("Which city?", out["seen"][1])  # the agent asks for what's still needed, in the chat
        self.assertEqual(out["delays"], [])

    def test_a_turn_that_makes_the_draft_ready_shows_the_profile_once_after_the_status_line(self):
        out = self.show(self.new, self.not_ready, self.ready, then="""
const holding = phone("A"), hold = timers.map(t => t.ms);
saving.A = 1; runTimers(); const waiting = phone("A");  // a save still on its way: it waits for it
saving.A = 0; const retry = timers.map(t => t.ms); runTimers(); const shown = phone("A");
S = states[2]; renderPhone("A"); const redrawn = phone("A");
var out = { holding, hold, waiting, retry, shown, redrawn };""")
        status = '<div class="assembling" role="status">'
        holding = out["out"]["holding"]
        self.assertIn(status, holding)
        self.assertIn("Putting your profile together", holding)
        self.assertNotIn("profile-pane", holding)
        self.assertRegex(holding, r'<button class="btn"\s*disabled\s*onclick="send')
        self.assertEqual(len(out["out"]["hold"]), 1)
        self.assertGreaterEqual(out["out"]["hold"][0], 1000)  # held for a while (REVEAL_HOLD), not a flash
        self.assertNotIn("profile-pane", out["out"]["waiting"])
        self.assertTrue(out["out"]["retry"])
        for html in (out["out"]["shown"], out["out"]["redrawn"]):
            self.assertProfile(html)
        self.assertIs(out["c"]["revealed"], True)

    def test_it_happens_once(self):
        """After the profile is first shown, another update steps the chat aside as before: no status line again,
        and a draft that isn't ready any more (a different request that still needs something) still shows it."""
        out = self.show(self.new, self.not_ready, self.ready, then=f"""
runTimers(); S = {json.dumps(self.again)}; renderPhone("A"); var out = phone("A");""")
        self.assertProfile(out["out"])
        self.assertEqual(out["delays"], [])

    def test_not_while_the_agent_is_replying_nor_for_a_reply_that_changed_nothing(self):
        question = json.loads(json.dumps(self.ready))
        question["phones"]["A"]["listings"][0]["chat"] += [{"from": "me", "text": "Is that all?"},
                                                           {"from": "agent", "text": "Anything else?"}]
        out = self.show(self.new, self.not_ready, then=f"""
cur("A").sending = "x"; S = {json.dumps(self.ready)}; renderPhone("A"); const replying = phone("A");
cur("A").sending = null; S = {json.dumps(question)}; renderPhone("A"); const asked = phone("A");
var out = {{ replying, asked, timers: timers.length }};""")
        for html in (out["out"]["replying"], out["out"]["asked"]):
            self.assertChatOnly(html)
        self.assertIn("Your agent is replying", out["out"]["replying"])
        self.assertEqual(out["out"]["timers"], 0)

    def test_a_live_or_ready_listing_opens_showing_its_profile(self):
        for state in (self.ready, self.live):
            with self.subTest(listed=state["phones"]["A"]["listings"][0]["listed"]):
                out = self.show(state)
                self.assertProfile(out["seen"][0])
                self.assertEqual(out["delays"], [])
        self.assertChatOnly(self.show(self.not_ready)["seen"][0])  # a fresh page (a reload) keeps the rule

    def test_the_chat_cant_be_minimised_before_the_profile_is_shown(self):
        out = self.show(self.new, self.not_ready, then="""
const before = cur("A").chatMin; toggleChat("A"); var out = { same: cur("A").chatMin === before };""")
        self.assertChatOnly(out["after"])
        self.assertTrue(out["out"]["same"])
        shown = self.show(self.ready, then='toggleChat("A"); var out = phone("A");')
        self.assertIn('class="chat-sheet"', shown["out"])  # once it's shown, the chat opens and minimises as before

    def test_reduced_motion_skips_the_animation(self):
        then = 'runTimers(); var out = { style: sheet("A").style, events: sheet("A").events };'
        moving = self.show(self.new, self.not_ready, self.ready, then=then)["out"]
        still = self.show(self.new, self.not_ready, self.ready, then=then, reduced=True)["out"]
        self.assertEqual(moving["style"].get("height"), "100px")  # slid from full height to its header (slideSheet)
        self.assertIn("transitionend", moving["events"])
        self.assertEqual(still, {"style": {}, "events": []})  # with reduced motion, it simply appears
        from tests.test_dating import PAGE
        import re
        reduce = re.search(r"@media \(prefers-reduced-motion: reduce\) \{(.*?)\} *\}", PAGE, flags=re.S).group(1)
        self.assertRegex(reduce, r"\.assembling \.lines span \{ animation: none;")
        self.assertIn(".chat-sheet { transition: none; }", reduce)

    def test_the_slide_starts_where_the_chat_was(self):
        """The whole-screen chat is taller than the sheet may rest (its max-height): the slide lifts that while it
        plays, so its first frame is the chat as it was, and puts it back once it's there."""
        out = self.show(self.new, self.not_ready, self.ready, then="""
runTimers(); const s = sheet("A"), during = { ...s.style };
s.on.transitionend({ target: s, propertyName: "height" }); var out = { during, after: { ...s.style } };""")["out"]
        self.assertEqual(out["during"]["maxHeight"], "none")
        self.assertEqual(out["during"]["height"], "100px")
        self.assertEqual(out["after"]["maxHeight"], "")

    def test_once_shown_the_profile_stays_after_a_reload(self):
        """The server keeps that a listing has had its profile (has_profile), so a fresh page (a reload, another
        device) shows it even once its draft has stopped being ready: a redraft that dropped the place, or the person
        hiding where they are, which only the profile's shown or hidden choice can undo. No status line again."""
        for name, state in (("redrafted", self.again), ("place hidden", self.place_hidden)):
            with self.subTest(name):
                out = self.show(state)
                self.assertProfile(out["seen"][0])
                self.assertIn("saveDraft", out["seen"][0])  # the draft can be edited and saved
                self.assertEqual(out["delays"], [])
        out = self.show(self.again, self.ready_again)  # the next ready reply, on that fresh page
        for html in out["seen"]:
            self.assertProfile(html)
        self.assertEqual(out["delays"], [])

    def test_a_listing_that_became_ready_while_away_opens_showing_its_profile(self):
        """The status line is for a reply that arrives while the chat is on screen: a listing opened (again) once it
        has its profile shows it straight away."""
        out = self.show(self.new, self.not_ready, then=f"""
go("A", "belong"); S = {json.dumps(self.ready)}; renderPhone("A");
openView("A", "profile"); var out = phone("A");""")
        self.assertProfile(out["out"])
        self.assertEqual(out["delays"], [])

    def test_a_draft_that_stops_being_ready_during_the_status_line_keeps_the_chat(self):
        """Changed meanwhile (another window on the same phone), so that it isn't ready when the status line ends:
        no profile that can't start matching, and the agent's question stays in front of them. The next reply that
        makes it ready shows it."""
        out = self.show(self.new, self.not_ready, self.ready, then=f"""
const holding = phone("A");
S = {json.dumps(self.again)}; renderPhone("A"); runTimers(); const kept = phone("A"), c = {{ ...cur("A") }};
S = {json.dumps(self.ready_again)}; renderPhone("A"); const again = phone("A"); runTimers();
var out = {{ holding, kept, c, again, shown: phone("A") }};""")["out"]
        self.assertIn("Putting your profile together", out["holding"])
        self.assertChatOnly(out["kept"])
        self.assertIn("Which city?", out["kept"])
        self.assertRegex(out["kept"], r'<button class="btn"\s*onclick="send')  # Send works again
        self.assertIs(out["c"]["revealed"], False)
        self.assertIsNone(out["c"]["revealAt"])
        self.assertIn("Putting your profile together", out["again"])
        self.assertProfile(out["shown"])

    def test_a_reset_in_another_window_never_shows_a_profile_for_a_new_listing(self):
        """Listing ids start again after a reset, and a window that didn't reset keeps its own memory of the old
        listing: a listing the server says has no profile is still the agent's chat, full screen."""
        out = self.show(self.new, self.not_ready, self.ready, then=f"""
runTimers(); const before = phone("A");
go("A", "belong"); S = {json.dumps(self.new)}; openView("A", "profile");
var out = {{ before, after: phone("A") }};""")["out"]
        self.assertProfile(out["before"])
        self.assertChatOnly(out["after"])

    def test_a_request_given_in_pieces_shows_the_profile_once_what_and_where_are_both_said(self):
        """The offline agent builds on what was said, so a request given in pieces (IN_PIECES: several kinds, a date
        among them, either order, one asked about first) shows the profile at the turn that says the last of what and
        where: the chat, full screen, until then, then the status line, then the profile, once."""
        for name in IN_PIECES:
            with self.subTest(name):
                _, states = in_pieces(name)
                out = self.show(*states, then="""
const holding = phone("A"); runTimers(); var out = { holding, shown: phone("A"), c: { ...cur("A") } };""")
                for html in out["seen"][:-1]:
                    self.assertChatOnly(html)
                self.assertIn("Putting your profile together", out["out"]["holding"])
                self.assertNotIn("profile-pane", out["out"]["holding"])
                self.assertProfile(out["out"]["shown"])
                self.assertIs(out["out"]["c"]["revealed"], True)
