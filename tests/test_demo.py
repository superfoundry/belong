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
                ("A", "Yuki", 27, "Language exchange in Osaka: I speak English natively and want to practice "
                                  "Japanese. Online works for me."),
                ("B", "Ren", 29, "Language exchange in Osaka, in person: I speak Japanese natively and want to "
                                 "practice English.")):
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
        body = fill(d, values={"city": "Kyoto"}, hidden={"religion": "Buddhist", "smoking": "doesn't smoke"})
        s.submit("A", body)
        last = s.slots["A"].chat[-1]["text"]
        self.assertIn("What changed: your city", last)
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
            def respond(self, conversation, text, name, band, current):
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
               {"first_name": "Mia", "country": "Japan", "city": "Tokyo", "schedule": "weekends",
                "seeking": "an air conditioner repair technician", "offering": "an air conditioner repair customer"})
    b = person("B", "Ken", 42, {"notify_close": False, "notify_questions": questions},
               "I'm a repairman in Tokyo looking for repair jobs.",
               {"first_name": "Ken", "country": "Japan", "city": "Tokyo", "schedule": "flexible",
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
            def respond(self, conversation, text, name, band, current):
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
        # A yes is the opt-in to both: close matches, and follow-up questions that could find one.
        self.assertEqual(s.phone_state("A")["listings"][0]["prefs"], {"notify_close": True, "notify_questions": True})
        self.assertTrue(s.board.listing(lid).notify_close and s.board.listing(lid).notify_questions)
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
        self.s.agent_message("A", "Looking for a roommate in Osaka, moving in November.", second)
        a = {l["id"]: l for l in self.s.phone_state("A")["listings"]}
        self.assertEqual(a[second]["kind"], "roommate")
        self.assertEqual(a[first]["kind"], "climbing")
        self.assertNotIn("roommate", " ".join(m["text"] for m in a[first]["chat"]))

    def test_both_listings_are_matched_on_their_own(self):
        second = self.s.new_listing("A")
        # Someone among the synthetic people in Osaka is moving in December.
        self.s.agent_message("A", "Looking for a roommate in Osaka, moving in December.", second)
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


class MatchmakerRunsOnItsOwn(unittest.TestCase):
    """The demo's Matchmaker runs every few seconds (server._match_if_due); there is no button."""

    def test_the_other_phone_is_introduced_even_if_it_joins_later(self):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Yuki", 27)
        s.agent_message("A", "Looking for a climbing partner in Osaka on weekends. I'm an intermediate climber, "
                             "into bouldering and ramen.")
        s.submit("A", fill(s.slots["A"].draft))
        for _ in range(3):
            s.run_matchmaker(quiet=True)  # A meets synthetic climbers first
        s.verify("B", "Ren", 29)
        s.agent_message("B", "Need someone to belay with at Osaka gyms, weekends. Intermediate level, "
                             "I also like jazz and ramen.")
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

    def test_the_person_is_told_in_their_listing_chat(self):
        s = DemoSession(seed=7, agent=None)
        climbers(s)
        s.run_matchmaker(quiet=True)
        last = s.phone_state("A")["listings"][0]["chat"][-1]
        self.assertEqual(last["text"], "Belong. found someone for this listing. Go back to Belong. to see them.")
        self.assertIsNotNone(s.room_state()["matchmaker_last_run"])

    def test_a_one_way_close_match_isnt_news_to_the_other_person(self):
        s = DemoSession(seed=7, agent=None)
        repair_people(s, questions=False)
        s.run_matchmaker(quiet=True)
        intro = next(i for i in s.intros.values() if {i.person_a, i.person_b} == {"demoA-1", "demoB-1"})
        self.assertEqual(intro.one_way_to, "demoA-1")
        notice = "Belong. found someone for this listing. Go back to Belong. to see them."
        self.assertIn(notice, [m["text"] for m in s.phone_state("A")["listings"][0]["chat"]])
        self.assertNotIn(notice, [m["text"] for m in s.phone_state("B")["listings"][0]["chat"]])


class SaveDraft(unittest.TestCase):
    """Save draft keeps the person's own changes on their draft and nowhere else (DemoSession.save_draft)."""

    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        self.s.verify("A", "Tomas", 41)
        self.s.agent_message("A", "Looking for a running buddy in Lisbon, weekday mornings.")

    def phone_copy(self, **values):
        """The phone's copy of the draft with some values changed, as Save draft sends it."""
        d = self.s.slots["A"].draft
        return {"intent_text": d["intent_text"], "exactness": d["exactness"],
                "fields": [{"key": f["key"], "value": values.get(f["key"], f["value"]), "enabled": f["enabled"]}
                           for f in d["fields"]]}

    def values(self):
        return {f["key"]: f["value"] for f in self.s.slots["A"].draft["fields"]}

    def test_your_changes_are_kept_on_your_draft_and_go_nowhere_else(self):
        s = self.s
        events, receipts = len(s.events), len(s.warden.receipts)
        s.save_draft("A", self.phone_copy(schedule="weekends", interests=["trail running"]))
        self.assertEqual((self.values()["schedule"], self.values()["interests"]), ("weekends", ["trail running"]))
        self.assertFalse(s.phone_state("A")["listings"][0]["listed"])
        self.assertNotIn(s.slots["A"].listing().id, s.board)  # never matched
        self.assertEqual((len(s.events), len(s.warden.receipts)), (events, receipts))  # no log, no Warden
        self.assertNotIn("trail running", json.dumps(s.phone_state("B")))
        self.assertEqual([f["key"] for f in s.slots["A"].draft["fields"]][:2], ["kind", "first_name"])  # same rows, same order

    def test_your_agent_builds_on_it(self):
        s, seen = self.s, {}
        s.save_draft("A", self.phone_copy(schedule="weekends"))

        class Agent:  # stands in for the model: notes the profile it's given to build on
            def respond(self, conversation, text, name, band, current):
                seen.update(current or {})
                return "Noted.", None

        s.llm = Agent()
        s.agent_message("A", "Evenings work too.")
        self.assertIn(("schedule", "weekends"), {(f["key"], f["value"]) for f in seen["fields"]})

    def test_it_changes_values_only(self):
        """Not shown or hidden (its own action, which asks to confirm a sensitive detail), not the name or age."""
        s = self.s
        s.set_visibility("A", None, {"schedule": HIDDEN})
        body = self.phone_copy(first_name="Someone else", age_band="18-24", schedule="weekends")
        for f in body["fields"]:
            f["visibility"] = SHOWN
            if f["key"] == "religion":
                f.update(value="Buddhist", enabled=True)
        body["fields"].append({"key": "not_a_detail", "value": "x", "enabled": True})
        s.save_draft("A", body)
        by = {f["key"]: f for f in s.slots["A"].draft["fields"]}
        self.assertEqual((by["first_name"]["value"], by["age_band"]["value"]), ("Tomas", "40-49"))
        self.assertEqual((by["schedule"]["value"], by["schedule"]["visibility"]), ("weekends", HIDDEN))
        self.assertEqual((by["religion"]["value"], by["religion"]["enabled"], by["religion"]["visibility"]),
                         ("Buddhist", True, HIDDEN))
        self.assertNotIn("not_a_detail", by)

    def test_a_detail_put_aside_comes_back_hidden_if_it_was(self):
        """Changing the kind puts a detail aside on the phone. Saved meanwhile, it comes off the draft (an
        empty row of another kind never lingers to be filled in), and if it was hidden it comes back hidden,
        even with the phone's default for it (a reload loses what the phone put aside)."""
        s, listing = self.s, self.s.slots["A"].listing()
        for f in listing.draft["fields"]:  # hidden by an agent, not by the person: not one of their choices
            if f["key"] == "pace":
                f.update(value="easy", visibility=HIDDEN)
        aside = self.phone_copy(kind="tutoring")
        aside["fields"] = [f for f in aside["fields"] if f["key"] != "pace"]
        s.save_draft("A", aside)
        self.assertNotIn("pace", {f["key"] for f in listing.draft["fields"]})
        back = self.phone_copy(kind="running")
        back["fields"].append({"key": "pace", "value": "easy", "enabled": True, "visibility": SHOWN})
        s.save_draft("A", back)
        self.assertEqual({f["key"]: f["visibility"] for f in listing.draft["fields"]}["pace"], HIDDEN)

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

    def test_the_choices_follow_the_kind_and_place_but_a_choice_made_is_kept(self):
        """A detail's choices follow a new place, except where what's chosen isn't among them: the place is a
        text box on the phone, which never clears a choice, so neither does Save draft."""
        from belong import agent
        listing = self.s.slots["A"].listing()
        lisbon = agent.field_options("running", "Lisbon")["income_band"]
        for f in listing.draft["fields"]:
            if f["key"] == "income_band":
                f.update(value=lisbon[1], enabled=True, options=lisbon)
        self.s.save_draft("A", self.phone_copy(city="Osaka", country="Japan"))
        by = {f["key"]: f for f in listing.draft["fields"]}
        self.assertEqual((by["income_band"]["value"], by["income_band"]["options"]), (lisbon[1], lisbon))
        self.s.save_draft("A", self.phone_copy(city="Osaka", country="Japan", income_band=None))
        by = {f["key"]: f for f in listing.draft["fields"]}
        self.assertEqual(by["income_band"]["options"], agent.field_options("running", "Osaka")["income_band"])

    def test_only_a_hidden_value_set_aside_is_remembered_hidden(self):
        """A detail empty and hidden only by default, set aside, isn't recorded as their choice."""
        s, listing = self.s, self.s.slots["A"].listing()
        for f in listing.draft["fields"]:
            if f["key"] == "pace":
                f.update(visibility=HIDDEN)  # empty
        aside = self.phone_copy(kind="tutoring")
        aside["fields"] = [f for f in aside["fields"] if f["key"] != "pace"]
        s.save_draft("A", aside)
        self.assertNotIn("pace", listing.choices)

    def test_a_hidden_first_name_stays_hidden(self):
        """Its value comes from the ID check, but shown or hidden is the person's choice, as at Start matching."""
        s = self.s
        s.set_visibility("A", None, {"first_name": HIDDEN, "age_band": HIDDEN})
        s.save_draft("A", self.phone_copy(schedule="weekends"))
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
            self.s.save_draft("A", self.phone_copy(schedule="weekends"))
        self.assertEqual(self.values()["schedule"], "weekday mornings")

    def test_a_guessed_place_you_change_is_yours(self):
        listing = self.s.slots["A"].listing()
        listing.provisional = {"city", "country"}
        self.s.save_draft("A", self.phone_copy(city="Porto"))
        self.assertEqual(listing.provisional, {"country"})  # the one left as it was is still a guess

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


class PhoneCopy(unittest.TestCase):
    """demo.html (run with node): the phone's own copy of a draft never brings back an older shown or hidden,
    so what Start matching or Save draft sends is what the person last chose."""

    def run_js(self, names, body):
        import shutil
        import subprocess
        import tempfile
        from tests.test_dating import _js
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write("\n".join(_js(n) for n in names) + "\n" + body)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

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
