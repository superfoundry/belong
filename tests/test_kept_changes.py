"""A change the person makes on their profile themselves keeps the rest of it consistent (TRUST.md #34).

When a change made directly on the phone is kept (a detail's sheet closed, the editor done, Save draft), the person's
model agent is told what changed, from what to what, in a turn of its own, and updates the other parts that still said
what was replaced, held to one fixed rule (agent.keep_edit). Without a model agent the phone offers the same change in
those parts instead (agent.offers), one tap each or all. Examples here are several kinds of listing, none of them the
case that was reported. No test calls the network: model agents are scripted stand-ins."""
from __future__ import annotations

import copy
import json
import unittest

from belong import agent
from belong.agent_llm import EDIT_TURN, EDIT_UNSAVED_NOTE, ClaudeAgent, build_draft, edit_message
from belong.agent_openai import OpenAIAgent
from belong.demo import DemoError, DemoSession
from belong.schema import HIDDEN, SHOWN
from belong.schema import canonical as schema_canonical
from tests.test_agent_llm import ScriptedClient, ScriptedPost, SavesThenFails, det, oa_text, oa_tool, reply, text, tool
from tests.test_demo import _Page


def args(kind="other", intent="", fields=(), details=()):
    """An update_listing call, as a model agent makes it."""
    return {"kind": kind, "exactness": 0.6, "intent_text": intent, "suggestions": [], "separate_listing": False,
            "fields": [{"key": k, "values": v if isinstance(v, list) else [v], "visibility": vis} for k, v, vis in fields],
            "details": list(details)}


SOFA = args(intent="Selling my green velvet sofa in Utrecht.",
            fields=[("city", "Utrecht", SHOWN), ("country", "Netherlands", SHOWN), ("seeking", "a sofa buyer", SHOWN),
                    ("offering", "a sofa seller", SHOWN), ("headline", "Selling a green velvet sofa", SHOWN),
                    ("topics", ["furniture", "velvet upholstery"], SHOWN)],
            details=[det("colour", "green"), det("seats", "three")])
SIZES = ("small", "medium", "large")
DOG = args(intent="I need someone to walk my large dog in Aarhus on weekday mornings.",
           fields=[("city", "Aarhus", SHOWN), ("country", "Denmark", SHOWN), ("seeking", "a dog walker", SHOWN),
                   ("offering", "a dog owner", SHOWN), ("headline", "Finding a walker for a large dog", SHOWN)],
           details=[det("dog size", "large", compare="steps", steps=SIZES),
                    det("dog size", "large", about="them", compare="steps", steps=SIZES)])
DATE = args(kind="dating", intent="Hoping for a relationship in Porto. I love sea kayaking and chess.",
            fields=[("city", "Porto", SHOWN), ("country", "Portugal", SHOWN), ("gender", "man", SHOWN),
                    ("wants_gender", ["woman"], SHOWN), ("wants_age", "35-45", SHOWN)],
            details=[det("interests", "sea kayaking", "chess"), det("interests", "sea kayaking", about="them")])
TUTOR = args(intent="Tutoring maths for families near Riverside in Leeds.",
             fields=[("city", "Leeds", SHOWN), ("country", "United Kingdom", SHOWN),
                     ("seeking", "a maths student", SHOWN), ("offering", "a maths tutor", SHOWN),
                     ("headline", "Tutoring maths near Riverside", SHOWN), ("neighborhood", "Riverside", HIDDEN)])


def values(draft) -> dict:
    return {k: agent.part_value(draft, k) for k in agent.parts(draft)}


def kept(draft, changes: dict) -> dict:
    """What the phone sends when the person keeps these changes (key -> value; "intent_text" for the description)."""
    return {"intent_text": changes.get("intent_text", draft["intent_text"]), "exactness": draft["exactness"],
            "fields": [{"key": f["key"], "value": changes.get(f["key"], f["value"]), "enabled": f.get("enabled", True),
                        "visibility": f["visibility"], "confirmed": bool(f.get("confirmed"))} for f in draft["fields"]]}


class _Session:
    def session(self, call, name="Noor", age=36, llm=None, slot="A"):
        """A session with one listing whose draft an agent saved (build_draft), as the phone shows it."""
        s = DemoSession(seed=7, agent=llm)
        s.run_inline = True
        s.verify(slot, name, age)
        s.verify("B" if slot == "A" else "A", "Ravi", 41)
        mine = s.slots[slot].listing(s.new_listing(slot))
        mine.draft, problems, _ = build_draft(call, name, s.slots[slot].person.age_band)
        self.assertEqual(problems, [])
        return s, mine


class TheRule(unittest.TestCase):
    """agent.py: which words a change replaced (compared as schema.canonical compares them) and which parts still say
    them. One general rule, no word list."""

    def change(self, key, before, after, private=False):
        return {"key": key, "before": before, "after": after, "private": private}

    def test_the_words_a_change_replaced(self):
        pairs = agent.replacements
        self.assertEqual(pairs(self.change("headline", "Selling a green velvet sofa", "Selling a blue velvet sofa")),
                         [(("green",), "blue")])
        # A run replaced by as many words is read word for word; an article that changed with its word is grammar.
        self.assertEqual(pairs(self.change("seeking", "a oboe teacher", "an alto teacher")), [(("oboe",), "alto")])
        self.assertEqual(pairs(self.change("me:colour", ["green", "grey"], ["Grey", "teal"])), [(("green",), "teal")])
        self.assertEqual(pairs(self.change("pay", "€40 an hour", None)), [(("40", "an", "hour"), None)])
        self.assertEqual(pairs(self.change("city", "Utrecht", "utrecht")), [])  # case alone changes nothing
        self.assertEqual(pairs(self.change("topics", ["furniture"], ["furniture", "antiques"])), [])  # only added
        self.assertEqual(pairs(self.change("kind", "other", "dating")), [])  # a choice, not wording
        self.assertEqual(agent.replaced_words([self.change("seeking", "a cello and harp teacher", "a harp teacher")]),
                         {"cello"})  # the conjunction that went with it is grammar
        self.assertEqual(agent.replaced_words([self.change("me:pet", ["dog"], ["cat"])]), {"dog"})  # a short word alone

    def test_a_part_depends_on_a_change_when_it_has_a_word_it_replaced(self):
        words = agent.replaced_words([self.change("me:dog size", ["large"], ["small"])])
        self.assertTrue(agent.says("Finding a walker for a LARGE dog", words))
        self.assertTrue(agent.says(["medium", "Large"], words))
        self.assertFalse(agent.says("Finding a walker for a dog", words))
        self.assertFalse(agent.says("largely free on weekdays", words))  # whole words only

    def test_a_short_word_replaced_inside_a_list_item_counts(self):
        """A list item replaced by another is read against it word by word, as a text is: the one word that changed is
        replaced alone, however short, and the other parts that say it depend on the change."""
        change = self.change("topics", ["oak shelving", "joinery"], ["ash shelving", "joinery"])
        self.assertEqual(agent.replacements(change), [(("oak",), "ash")])
        self.assertEqual(agent.replaced_words([change]), {"oak"})
        self.assertEqual(agent.replaced_words([self.change("headline", "Building oak shelving", "Building ash shelving")]),
                         {"oak"})  # the same as in a text
        self.assertEqual(agent.replacements(self.change("me:diet", ["no dairy"], ["vegan"])), [(("no", "dairy"), "vegan")])

    def test_one_word_in_two_forms(self):
        """same_word: one word in another form (a plural, another tense), by a fixed length rule, no word list."""
        for a, b in (("tile", "tiles"), ("ferry", "ferries"), ("bake", "baking"), ("fox", "foxes"), ("Lyon", "lyon")):
            self.assertTrue(agent.same_word(schema_canonical(a), schema_canonical(b)), (a, b))
        for a, b in (("cod", "cot"), ("mile", "mild"), ("art", "artisan"), ("an", "and"), ("pine", "pineapple")):
            self.assertFalse(agent.same_word(a, b), (a, b))

    def test_a_change_on_an_older_copy_is_made_on_the_draft_as_it_now_is(self):
        """rebase: what the person changed on a copy their agent has updated since is made on the agent's update;
        where both changed the same words, theirs wins; in a list, item by item."""
        rebase = agent.rebase
        self.assertEqual(rebase("Renting a sunny loft in Graz", "Renting a sunny loft in Graz, pets welcome",
                                "Renting a bright loft in Graz"), "Renting a bright loft in Graz, pets welcome")
        self.assertEqual(rebase("Renting a sunny loft in Graz", "Renting a sunny loft in Linz",
                                "Renting a bright loft in Graz"), "Renting a bright loft in Linz")
        self.assertEqual(rebase("Renting a sunny loft", "Renting a quiet loft", "Renting a bright loft"),
                         "Renting a quiet loft")  # the same word: theirs
        self.assertEqual(rebase(["sunny rooms", "balcony"], ["sunny rooms"], ["bright rooms", "balcony"]), ["bright rooms"])
        self.assertEqual(rebase(["sunny rooms", "balcony"], ["quiet rooms", "balcony"], ["bright rooms", "balcony"]),
                         ["quiet rooms", "balcony"])
        self.assertEqual(rebase(["balcony"], ["balcony", "garden"], ["balcony", "lift"]), ["balcony", "lift", "garden"])
        self.assertEqual(rebase("Renting a loft", "Renting a loft", "Renting a flat"), "Renting a flat")  # not theirs
        self.assertEqual(rebase("monthly", "weekly", "yearly", choices=["weekly", "monthly", "yearly"]), "weekly")

    def test_shown_or_hidden_alone_is_no_change(self):
        before, _, _ = build_draft(SOFA, "Noor", "35-39")
        after = copy.deepcopy(before)
        for f in after["fields"]:
            if f["key"] == "me:seats":
                f["visibility"] = HIDDEN
        self.assertEqual(agent.changes_made(before, after), [])


class WithAModelAgent(_Session, unittest.TestCase):
    """The person's model agent is told about a kept change in a turn of its own and brings the rest of the profile in
    line, held to agent.keep_edit."""

    BLUE = json.loads(json.dumps(SOFA).replace("green", "blue"))

    def test_the_turn_is_sent_and_its_reply_and_save_are_kept(self):
        client = ScriptedClient(reply(tool(self.BLUE)), reply(text("I've made the headline and description say blue too.")))
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        before = values(mine.draft)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        said = client.requests[0]["messages"][-1]["content"]
        self.assertTrue(said.startswith("(From the app, not typed by the person.)"))
        self.assertIn('colour (about me) from "green" to "blue"', said)
        self.assertEqual(said, edit_message([{"key": "me:colour", "before": ["green"], "after": ["blue"], "private": False}]))
        after = values(mine.draft)
        self.assertEqual(after["headline"], "Selling a blue velvet sofa")
        self.assertEqual(after["intent_text"], "Selling my blue velvet sofa in Utrecht.")
        self.assertEqual(after["me:colour"], ["blue"])
        for key in before:  # nothing that didn't say "green" changed
            if key not in ("headline", "intent_text", "me:colour"):
                self.assertEqual(after[key], before[key], key)
        chat = s.phone_state("A")["listings"][0]["chat"]
        self.assertEqual(chat[-2], {"from": "change", "text": "You changed your colour on your profile."})
        self.assertEqual(chat[-1], {"from": "agent", "text": "I've made the headline and description say blue too.",
                                    "draft_updated": True})
        self.assertFalse(mine.agent_busy)

    def test_the_same_turn_and_rule_for_openai(self):
        post = ScriptedPost(oa_tool(self.BLUE), oa_text("Updated the headline and description."))
        s, mine = self.session(SOFA, llm=OpenAIAgent(api_key="test", post=post))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        messages = post.requests[0]["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": edit_message(
            [{"key": "me:colour", "before": ["green"], "after": ["blue"], "private": False}])})
        self.assertEqual(values(mine.draft)["headline"], "Selling a blue velvet sofa")
        self.assertEqual(mine.chat[-1]["text"], "Updated the headline and description.")

    def test_a_save_that_undoes_their_change_adds_a_fact_or_changes_another_part_is_refused(self):
        undo = copy.deepcopy(self.BLUE)
        undo["details"][0]["values"] = ["green"]
        more = copy.deepcopy(self.BLUE)
        more["details"].append(det("fabric care", "washable covers"))
        elsewhere = json.loads(json.dumps(self.BLUE).replace("Utrecht", "Leiden"))
        client = ScriptedClient(reply(tool(undo, "t1")), reply(tool(more, "t2")), reply(tool(elsewhere, "t3")),
                                reply(tool(self.BLUE, "t4")), reply(text("Done.")))
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        results = [r["messages"][-1]["content"][0] for r in client.requests[1:]]
        self.assertTrue(all(r.get("is_error") for r in results[:3]), results)
        self.assertIn('Keep colour (about me) exactly as the person set it: "blue".', results[0]["content"])
        self.assertIn("Don't add fabric care (about me): the person didn't give it.", results[1]["content"])
        self.assertIn("Leave city as it was", results[2]["content"])
        self.assertTrue(results[3]["content"].startswith("Draft saved."))
        after = values(mine.draft)
        self.assertEqual((after["me:colour"], after["city"], after.get("me:fabric care")), (["blue"], "Utrecht", None))

    def test_no_part_is_taken_off_and_one_left_out_that_didnt_say_it_stays(self):
        """Leaving out a part that said what they replaced is refused (it is reworded, never taken off); leaving out
        one that didn't is no claim of anything, so it simply stays as it was."""
        rowing = json.loads(json.dumps(DATE).replace("sea kayaking", "rowing"))
        no_wish = copy.deepcopy(rowing)
        no_wish["details"] = no_wish["details"][:1]
        quiet = copy.deepcopy(SOFA)
        client = ScriptedClient(reply(tool(no_wish, "t1")), reply(tool(rowing, "t2")), reply(text("Done.")))
        s, mine = self.session(DATE, name="Tiago", age=40, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:interests": ["rowing", "chess"]}))
        self.assertIn("Keep interests (about them) on the profile", client.requests[1]["messages"][-1]["content"][0]["content"])
        self.assertEqual(values(mine.draft)["them:interests"], ["rowing"])
        quiet["details"] = [det("colour", "blue")]  # "seats" left out; the headline and description not reworded
        client = ScriptedClient(reply(tool(quiet)), reply(text("Done.")))
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertTrue(client.requests[1]["messages"][-1]["content"][0]["content"].startswith("Draft saved."))
        self.assertEqual(values(mine.draft)["me:seats"], ["three"])
        self.assertEqual(mine.chat[-1]["text"], "Done.")

    def test_a_sensitive_detail_they_chose_to_show_stays_shown(self):
        """An agent can't make a sensitive detail shown (only the person can, with their confirmation), so its save
        restates it hidden: the update keeps the person's choice and isn't refused for it."""
        rowing = json.loads(json.dumps(DATE).replace("sea kayaking", "rowing"))
        rowing["fields"].append({"key": "religion", "values": ["Quaker"], "visibility": SHOWN})
        client = ScriptedClient(reply(tool(rowing)), reply(text("Updated your wish and description.")))
        s, mine = self.session(DATE, name="Tiago", age=40, llm=ClaudeAgent(client=client))
        religion = agent._row(mine.draft, "religion")
        religion.update(value="Quaker", visibility=SHOWN, enabled=True, confirmed=True)
        mine.choices["religion"] = SHOWN
        s.keep_change("A", kept(mine.draft, {"me:interests": ["rowing", "chess"]}))
        self.assertEqual(len(client.requests), 2)  # saved first time
        self.assertEqual(agent._row(mine.draft, "religion")["visibility"], SHOWN)
        self.assertEqual(values(mine.draft)["them:interests"], ["rowing"])

    def test_an_agent_that_doesnt_hold_to_the_rule_is_corrected_by_it(self):
        """Any agent's save is corrected by the same fixed rule (the part the person changed keeps their value, nothing
        new, only parts that said what was replaced, shown or hidden as it was), and the chat then says what actually
        changed, never what the agent claimed."""
        class OwnAgent:  # its own respond, which knows nothing of the rule
            def respond(self, conversation, text, name, band, current, in_use=None):
                d = copy.deepcopy(current)
                for f in d["fields"]:
                    if f["key"] == "me:colour":
                        f["value"] = ["green"]  # undoes their change
                    if f["key"] == "city":
                        f["value"] = "Leiden"  # didn't say what they replaced
                    if f["key"] == "headline":
                        f["value"], f["visibility"] = "Selling a blue velvet sofa", HIDDEN  # rightly reworded, wrongly hidden
                d["fields"].append(agent.detail_row("me", "fabric care", "same", [], ["washable covers"]))  # a new fact
                d["intent_text"] = "Selling my blue velvet sofa in Utrecht."
                return "I changed everything to blue, moved you to Leiden and hid your headline.", d

        s, mine = self.session(SOFA, llm=OwnAgent())
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        after = values(mine.draft)
        self.assertEqual((after["me:colour"], after["city"], after.get("me:fabric care")), (["blue"], "Utrecht", None))
        self.assertEqual(after["headline"], "Selling a blue velvet sofa")
        self.assertEqual(agent._row(mine.draft, "headline")["visibility"], SHOWN)
        self.assertEqual(after["intent_text"], "Selling my blue velvet sofa in Utrecht.")
        self.assertEqual(mine.chat[-1]["text"], "I've updated your headline and your description to match.")

    def test_a_hidden_detail_never_goes_into_a_shown_part(self):
        moved = json.loads(json.dumps(TUTOR).replace("Riverside", "Hillcrest"))
        client = ScriptedClient(reply(tool(moved)), reply(text("Updated.")))
        s, mine = self.session(TUTOR, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"neighborhood": "Hillcrest"}))
        said = client.requests[0]["messages"][-1]["content"]
        self.assertIn("What they changed in neighborhood is hidden: never write its new wording into a part that's shown.",
                      said)
        self.assertIn("Don't write the new wording of a hidden detail into headline, which is shown.",
                      client.requests[1]["messages"][-1]["content"][0]["content"])
        after = values(mine.draft)
        self.assertEqual(after["headline"], "Tutoring maths near Riverside")  # shown: never given the hidden wording
        self.assertNotIn("Hillcrest", repr(s.phone_state("B")))
        self.assertTrue(mine.chat[-1]["text"].endswith(EDIT_UNSAVED_NOTE))  # it never saved, and the chat says so

    def test_a_reply_that_claims_an_update_it_never_saved_isnt_shown(self):
        client = ScriptedClient(reply(text("Done: your headline and description say blue now.")))
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertEqual(len(client.requests), 1)  # no save is asked for after a kept change
        self.assertEqual(mine.chat[-1], {"from": "agent", "text": agent.NOTHING_ELSE, "draft_updated": False})
        self.assertEqual(values(mine.draft)["headline"], "Selling a green velvet sofa")

    def test_the_description_and_hidden_parts_may_take_a_hidden_change(self):
        class Rewords:
            def respond(self, conversation, text, name, band, current, in_use=None):
                d = copy.deepcopy(current)
                d["intent_text"] = d["intent_text"].replace("Riverside", "Hillcrest")
                return "Updated your description.", d

        s, mine = self.session(TUTOR, llm=Rewords())
        s.keep_change("A", kept(mine.draft, {"neighborhood": "Hillcrest"}))
        self.assertEqual(values(mine.draft)["intent_text"], "Tutoring maths for families near Hillcrest in Leeds.")
        self.assertEqual(mine.chat[-1]["text"], "Updated your description.")

    def test_nothing_replaced_means_no_turn(self):
        """Only adding something, or shown or hidden alone, replaces no words: nothing else can still say the old thing,
        so the agent isn't asked (and the chat shows no line)."""
        client = ScriptedClient()
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        chat = list(mine.chat)
        s.keep_change("A", kept(mine.draft, {"topics": ["furniture", "velvet upholstery", "mid-century"]}))
        body = kept(mine.draft, {})
        for f in body["fields"]:
            if f["key"] == "me:seats":
                f["visibility"] = HIDDEN
        s.set_visibility("A", mine.id, {"me:seats": HIDDEN})
        s.keep_change("A", body)
        self.assertEqual(client.requests, [])
        self.assertEqual(mine.chat, chat)
        self.assertEqual(values(mine.draft)["topics"], ["furniture", "velvet upholstery", "mid-century"])

    def test_the_conversation_keeps_no_wording_once_the_turn_is_over(self):
        """The old wording was for that one turn: the conversation the agent is sent from then on says which parts
        changed, never what they said (a value they took off is gone)."""
        moved = json.loads(json.dumps(TUTOR).replace("Riverside", "Hillcrest"))
        moved["fields"] = [f for f in moved["fields"] if f["key"] != "headline"] + \
            [{"key": "headline", "values": ["Tutoring maths near Riverside"], "visibility": SHOWN}]
        client = ScriptedClient(reply(tool(moved)), reply(text("Done.")), reply(text("Hello again.")),
                                reply(text("NO_SAVE")))
        s, mine = self.session(TUTOR, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"neighborhood": "Hillcrest"}))
        users = [m["content"] for m in mine.agent_conversation if m["role"] == "user" and isinstance(m["content"], str)]
        self.assertEqual(users, ["(From the app, not typed by the person.) The person changed neighborhood on their "
                                 "profile themselves."])
        s.agent_message("A", "Thanks.")
        later = repr(client.requests[-1]["messages"])
        self.assertNotIn("Riverside", later.replace("Tutoring maths near Riverside", ""))

    def test_it_never_reaches_anyone_else(self):
        """The change line, the turn and its values stay with the person and their own agent: the other phone, the
        presenter's panel and log, and the Warden's receipts never get them."""
        client = ScriptedClient(reply(tool(self.BLUE)), reply(text("Updated.")))
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        receipts, events = len(s.warden.receipts), len(s.events)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertEqual((len(s.warden.receipts), len(s.events)), (receipts, events))
        others = json.dumps({"B": s.phone_state("B"), "room": s.room_state()})
        for said in ("You changed your colour", "blue", "From the app"):
            self.assertNotIn(said, others)
        self.assertNotIn(mine.id, s.board)

    def test_a_part_brought_in_line_gets_no_word_from_anywhere_else(self):
        """An update may reword a part that said what was replaced only with words it had and the person's new wording
        (in any form, with grammar): a new list item, or new words in the headline or description, add something they
        didn't give, and are refused for both providers. A correct update then goes through."""
        more = copy.deepcopy(self.BLUE)
        for f in more["fields"]:
            if f["key"] == "headline":
                f["values"] = ["Selling a blue velvet sofa with matching cushions"]
        more["intent_text"] = "Selling my blue velvet sofa in Utrecht, free delivery."
        def claude():
            client = ScriptedClient(reply(tool(more, "t1")), reply(tool(self.BLUE, "t2")), reply(text("Updated.")))
            return ClaudeAgent(client=client), lambda: client.requests[1]["messages"][-1]["content"][0]["content"]

        def openai():
            post = ScriptedPost(oa_tool(more, "c1"), oa_tool(self.BLUE, "c2"), oa_text("Updated."))
            return OpenAIAgent(api_key="test", post=post), lambda: post.requests[1]["messages"][-1]["content"]

        for provider in (claude, openai):
            with self.subTest(provider.__name__):
                llm, refused = provider()
                s, mine = self.session(SOFA, llm=llm)
                s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
                refusal = refused()
                self.assertIn("Reword headline only with words it had and the person's new wording", refusal)
                self.assertIn("Reword intent_text only with words it had", refusal)
                after = values(mine.draft)
                self.assertEqual((after["headline"], after["intent_text"]),
                                 ("Selling a blue velvet sofa", "Selling my blue velvet sofa in Utrecht."))
        # A new item in a list that said what they replaced is refused the same way.
        rowing = json.loads(json.dumps(DATE).replace("sea kayaking", "rowing"))
        sailing = copy.deepcopy(rowing)
        sailing["details"][1]["values"] = ["rowing", "sailing"]
        client = ScriptedClient(reply(tool(sailing, "t1")), reply(tool(rowing, "t2")), reply(text("Done.")))
        s, mine = self.session(DATE, name="Tiago", age=40, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:interests": ["rowing", "chess"]}))
        self.assertIn("Reword interests (about them) only with words it had",
                      client.requests[1]["messages"][-1]["content"][0]["content"])
        self.assertEqual(values(mine.draft)["them:interests"], ["rowing"])
        # Another form of a word it had, or of one the person wrote, is the same word.
        kept_draft = copy.deepcopy(self.session(SOFA)[1].draft)
        agent._row(kept_draft, "me:colour")["value"] = ["blue"]
        plural = json.loads(json.dumps(SOFA).replace("green", "blue"))
        plural["fields"] = [{**f, "values": ["Selling blue velvet sofas"]} if f["key"] == "headline" else f
                            for f in plural["fields"]]
        draft, problems, _ = build_draft(plural, "Noor", "35-39", None, None, (), {
            "changes": [{"key": "me:colour", "before": ["green"], "after": ["blue"], "private": False}],
            "saved": kept_draft})
        self.assertEqual(problems, [])

    def test_a_hidden_change_in_another_form_never_goes_into_a_shown_part(self):
        """No word of a hidden detail's new value goes into a shown part in another form either (a plural, say)."""
        days = args(intent="Teaching ceramics glazing in Bruges on Monday mornings.",
                    fields=[("city", "Bruges", SHOWN), ("country", "Belgium", SHOWN),
                            ("seeking", "a glazing student", SHOWN), ("offering", "a glazing teacher", SHOWN),
                            ("headline", "Glazing classes on Monday mornings", SHOWN)],
                    details=[det("class day", "Monday", visibility=HIDDEN)])
        plural = json.loads(json.dumps(days).replace("Monday mornings", "Wednesdays").replace("Monday", "Wednesday"))
        client = ScriptedClient(reply(tool(plural)), reply(text("Updated.")))
        s, mine = self.session(days, name="Lotte", age=50, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:class day": ["Wednesday"]}))
        self.assertIn("Don't write the new wording of a hidden detail into headline, which is shown.",
                      client.requests[1]["messages"][-1]["content"][0]["content"])
        self.assertEqual(values(mine.draft)["headline"], "Glazing classes on Monday mornings")
        self.assertNotIn("Wednesday", repr(s.phone_state("B")))

    def test_the_conversation_keeps_no_wording_from_a_refused_save_either(self):
        """Once the turn is over, its saves (a refused one that wrote the old value too) and the agent's own words are
        gone from the conversation, for both providers: it keeps which parts changed and which were updated."""
        moved = json.loads(json.dumps(TUTOR).replace("Riverside", "Hillcrest"))
        moved["fields"] = [f for f in moved["fields"] if f["key"] != "headline"] + \
            [{"key": "headline", "values": ["Tutoring maths near Riverside"], "visibility": SHOWN}]
        back = copy.deepcopy(TUTOR)  # puts the old hidden value back: refused
        for name, llm in (("claude", lambda: ClaudeAgent(client=ScriptedClient(
                              reply(tool(back, "t1")), reply(tool(moved, "t2")), reply(text("Moved you to Hillcrest from Riverside."))))),
                          ("openai", lambda: OpenAIAgent(api_key="test", post=ScriptedPost(
                              oa_tool(back, "c1"), oa_tool(moved, "c2"), oa_text("Moved you to Hillcrest from Riverside."))))):
            with self.subTest(name):
                s, mine = self.session(TUTOR, llm=llm())
                s.keep_change("A", kept(mine.draft, {"neighborhood": "Hillcrest"}))
                self.assertEqual(values(mine.draft)["intent_text"], "Tutoring maths for families near Hillcrest in Leeds.")
                self.assertEqual(mine.agent_conversation, [
                    {"role": "user", "content": "(From the app, not typed by the person.) The person changed neighborhood "
                                                "on their profile themselves."},
                    {"role": "assistant", "content": "I've updated your description to match."}])
                self.assertEqual(mine.chat[-1]["text"], "Moved you to Hillcrest from Riverside.")  # on their own phone

    def test_a_change_on_an_older_copy_never_undoes_what_the_agent_brought_in_line(self):
        """A change made while the agent was updating the profile goes with the copy it was made on (base): it is made
        on the draft as it now is, and what it replaced is read from that copy, so the agent's update stays and nothing
        is told or offered that would put the old wording back."""
        blue = copy.deepcopy(self.session(SOFA)[1].draft)
        for f in blue["fields"]:
            if f["key"] == "me:colour":
                f["value"] = ["blue"]
            if f["key"] == "headline":
                f["value"] = "Selling a blue velvet sofa"
        blue["intent_text"] = "Selling my blue velvet sofa in Utrecht."
        for llm in (None, ClaudeAgent(client=ScriptedClient())):
            with self.subTest(model=llm is not None):
                s, mine = self.session(SOFA, llm=llm)
                base = kept(mine.draft, {"me:colour": ["blue"]})  # the copy the phone had: colour kept, rest as was
                mine.draft = copy.deepcopy(blue)  # the agent's update arrived meanwhile
                s.keep_change("A", {**kept(blue, {"headline": "Selling a green velvet sofa, like new", "me:colour": ["blue"]}),
                                    "base": base})
                after = values(mine.draft)
                self.assertEqual((after["headline"], after["me:colour"], after["intent_text"]),
                                 ("Selling a blue velvet sofa, like new", ["blue"], "Selling my blue velvet sofa in Utrecht."))
                self.assertEqual(mine.offers, [])
                if llm is not None:
                    self.assertEqual(llm.client.requests, [])  # nothing they replaced: no turn
        # What they did replace on that copy is read from it.
        s, mine = self.session(SOFA)
        base = kept(mine.draft, {"me:colour": ["blue"]})
        mine.draft = copy.deepcopy(blue)
        s.keep_change("A", {**kept(blue, {"me:seats": ["two"], "headline": "Selling a green velvet two-seater"}), "base": base})
        self.assertEqual(values(mine.draft)["headline"], "Selling a blue velvet two-seater")
        # Start matching (and Save changes) makes such a change the same way, and leaves out what the draft has off.
        s, mine = self.session(SOFA)
        base = kept(mine.draft, {"me:colour": ["blue"]})
        mine.draft = copy.deepcopy(blue)
        body = kept(blue, {"headline": "Selling a green velvet sofa, like new", "me:colour": ["blue"]})
        body["fields"] = [f for f in body["fields"] if f.get("enabled", True) and f["value"] not in (None, "", [])]
        s.submit("A", {**body, "base": base}, mine.id)
        self.assertTrue(mine.entry["accepted"], mine.entry)
        self.assertEqual(s.board.listing(mine.id).get("headline").value, "Selling a blue velvet sofa, like new")

    def test_a_change_made_while_it_replies_stays_over_its_update(self):
        """The turn works from the draft as it began: a part changed on the draft meanwhile stays as it is now, its
        other updates are kept, and the chat says what actually changed. Save changes waits until it's done."""
        class Meanwhile:
            def respond(self, conversation, text, name, band, current, in_use=None, edit=None):
                d = copy.deepcopy(current)
                for f in d["fields"]:
                    if f["key"] in ("headline", "topics"):
                        f["value"] = json.loads(json.dumps(f["value"]).replace("green", "blue"))
                d["intent_text"] = d["intent_text"].replace("green", "blue")
                now = copy.deepcopy(mine.draft)  # meanwhile, a part changes on the draft
                for f in now["fields"]:
                    if f["key"] == "headline":
                        f["value"] = "Selling a velvet sofa"
                mine.draft = now
                return "Updated your headline and description.", d

        s, mine = self.session(SOFA, llm=Meanwhile())
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        after = values(mine.draft)
        self.assertEqual((after["headline"], after["intent_text"]), ("Selling a velvet sofa", "Selling my blue velvet sofa in Utrecht."))
        self.assertEqual(mine.chat[-1]["text"], "I've updated your description to match.")
        mine.agent_busy = True
        with self.assertRaises(DemoError):
            s.submit("A", kept(mine.draft, {}), mine.id)

    def test_no_turn_when_nothing_else_says_what_was_replaced(self):
        client = ScriptedClient()
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=client))
        s.keep_change("A", kept(mine.draft, {"me:seats": ["two"]}))
        self.assertEqual((client.requests, mine.chat[-1]["from"]), ([], "agent"))
        self.assertNotIn("change", [m["from"] for m in mine.chat])

    def test_not_while_their_agent_is_replying(self):
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=ScriptedClient()))
        mine.agent_busy = True
        with self.assertRaises(DemoError):
            s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertEqual(values(mine.draft)["me:colour"], ["green"])

    def test_an_agent_that_is_unavailable_leaves_the_phone_to_offer_it(self):
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=SavesThenFails()))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertIsNone(s.llm)
        offered = {o["key"]: o["after"] for o in s.phone_state("A")["listings"][0]["offers"]}
        self.assertEqual(offered, {"headline": "Selling a blue velvet sofa",
                                   "intent_text": "Selling my blue velvet sofa in Utrecht."})
        self.assertEqual(mine.agent_conversation, [])  # the unfinished turn left nothing behind
        self.assertFalse(mine.agent_busy)

    def test_a_turn_that_fails_otherwise_never_leaves_them_waiting(self):
        class Breaks:
            def respond(self, conversation, text, name, band, current, in_use=None, edit=None):
                conversation.append({"role": "user", "content": text})
                raise RuntimeError("something unexpected")

        s, mine = self.session(SOFA, llm=Breaks())
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertFalse(mine.agent_busy)
        self.assertEqual(mine.agent_conversation, [])
        self.assertEqual({o["key"] for o in mine.offers}, {"headline", "intent_text"})
        self.assertIsNotNone(s.llm)  # not switched off for one failure

    def test_one_instruction_text_for_both_providers(self):
        """Both providers send the turn the demo writes (agent_llm.edit_message) and hold every save to the same rule
        through the same build_draft: neither has a text of its own."""
        import inspect
        from pathlib import Path
        from belong import agent_openai
        self.assertIs(agent_openai.build_draft, build_draft)
        self.assertNotIn("From the app", Path(agent_openai.__file__).read_text())
        self.assertIn("{said}", EDIT_TURN)
        for provider in (ClaudeAgent, OpenAIAgent):
            self.assertIn("edit", inspect.signature(provider.respond).parameters)


class WithoutAModelAgent(_Session, unittest.TestCase):
    """The rule-based agent can't reword, so the phone offers the change in the other parts that still say what was
    replaced, one tap each or all, never silently."""

    def offers(self, s, slot="A"):
        return {o["key"]: o["after"] for o in s.phone_state(slot)["listings"][0]["offers"]}

    def test_the_other_parts_that_still_say_it_are_offered_and_nothing_changes_until_a_tap(self):
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        self.assertEqual(self.offers(s), {"headline": "Selling a blue velvet sofa",
                                          "intent_text": "Selling my blue velvet sofa in Utrecht."})
        self.assertEqual(values(mine.draft)["headline"], "Selling a green velvet sofa")
        self.assertEqual(mine.chat[-1]["from"], "agent")  # no line: no agent was told anything
        self.assertNotIn("change", [m["from"] for m in mine.chat])

    def test_one_tap_each_all_or_leave_as_is(self):
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.answer_offers("A", mine.id, ["headline"])
        self.assertEqual(values(mine.draft)["headline"], "Selling a blue velvet sofa")
        self.assertEqual(list(self.offers(s)), ["intent_text"])
        s.answer_offers("A", mine.id, [])
        self.assertEqual(self.offers(s), {})
        self.assertEqual(values(mine.draft)["intent_text"], "Selling my green velvet sofa in Utrecht.")
        s.keep_change("A", kept(mine.draft, {"me:colour": ["teal"]}))
        self.assertEqual(self.offers(s), {"headline": "Selling a teal velvet sofa"})  # what says "blue" now
        s.answer_offers("A", mine.id, list(self.offers(s)))
        self.assertEqual(values(mine.draft)["headline"], "Selling a teal velvet sofa")

    def test_nothing_is_offered_when_nothing_else_says_it(self):
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:seats": ["two"]}))
        self.assertEqual(self.offers(s), {})

    def test_a_part_with_steps_or_choices_is_offered_only_one_of_them(self):
        s, mine = self.session(DOG, name="Mads", age=52)
        s.keep_change("A", kept(mine.draft, {"me:dog size": ["small"]}))
        self.assertEqual(self.offers(s), {"them:dog size": "small", "headline": "Finding a walker for a small dog",
                                          "intent_text": "I need someone to walk my small dog in Aarhus on weekday mornings."})
        s.answer_offers("A", mine.id, [])
        s.keep_change("A", kept(mine.draft, {"headline": "Finding a walker for a giant dog"}))
        offered = self.offers(s)
        self.assertNotIn("them:dog size", offered)  # "giant" isn't one of its steps
        self.assertNotIn("me:dog size", offered)
        self.assertEqual(offered, {"intent_text": "I need someone to walk my giant dog in Aarhus on weekday mornings."})

    def test_a_list_item_replaced_in_one_part_is_offered_in_the_others(self):
        s, mine = self.session(DATE, name="Tiago", age=40)
        s.keep_change("A", kept(mine.draft, {"me:interests": ["rowing", "chess"]}))
        self.assertEqual(self.offers(s), {"them:interests": "rowing",
                                          "intent_text": "Hoping for a relationship in Porto. I love rowing and chess."})
        s.answer_offers("A", mine.id, ["them:interests"])
        self.assertEqual(values(mine.draft)["them:interests"], ["rowing"])

    def test_a_hidden_parts_words_are_offered_only_to_parts_that_are_never_shown(self):
        s, mine = self.session(TUTOR, name="Grace", age=47)
        s.keep_change("A", kept(mine.draft, {"neighborhood": "Hillcrest"}))
        self.assertEqual(self.offers(s), {"intent_text": "Tutoring maths for families near Hillcrest in Leeds."})
        # The other way, a shown part's words may go into a hidden one.
        s, mine = self.session(TUTOR, name="Grace", age=47)
        s.keep_change("A", kept(mine.draft, {"headline": "Tutoring maths near Hillcrest"}))
        self.assertEqual(self.offers(s), {"neighborhood": "Hillcrest",
                                          "intent_text": "Tutoring maths for families near Hillcrest in Leeds."})

    def test_an_offer_that_no_longer_applies_goes(self):
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.keep_change("A", kept(mine.draft, {"headline": "Selling a velvet sofa"}))
        self.assertNotIn("headline", self.offers(s))
        mine.offers = agent.offers(mine.draft, [{"key": "me:colour", "before": ["green"], "after": ["blue"],
                                                 "private": False}])
        mine.draft["intent_text"] = "Selling a sofa in Utrecht."
        self.assertEqual(self.offers(s), {})
        with self.assertRaises(DemoError):
            s.answer_offers("A", mine.id, ["intent_text"])

    def test_offers_stay_after_another_change_and_build_on_each_other(self):
        """What the phone offers stays until it's answered or its part changes: a later change elsewhere, or one that
        only adds something, keeps it, and a later change of the same words is offered on top of it."""
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.keep_change("A", kept(mine.draft, {"me:seats": ["two"]}))
        self.assertEqual(self.offers(s), {"headline": "Selling a blue velvet sofa",
                                          "intent_text": "Selling my blue velvet sofa in Utrecht."})
        s.keep_change("A", kept(mine.draft, {"me:colour": ["teal"]}))
        self.assertEqual(self.offers(s), {"headline": "Selling a teal velvet sofa",
                                          "intent_text": "Selling my teal velvet sofa in Utrecht."})
        s.answer_offers("A", mine.id, ["headline"])
        self.assertEqual(values(mine.draft)["headline"], "Selling a teal velvet sofa")

    def test_offers_from_an_unavailable_agent_stay_when_a_change_that_waited_goes(self):
        s, mine = self.session(SOFA, llm=ClaudeAgent(client=SavesThenFails()))
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.keep_change("A", kept(mine.draft, {"me:seats": ["two"]}))
        self.assertEqual(set(self.offers(s)), {"headline", "intent_text"})

    def test_offers_end_when_they_write_to_their_agent(self):
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.agent_message("A", "I can deliver it within Utrecht.")
        self.assertEqual(mine.offers, [])

    def test_an_offer_is_checked_as_save_draft_checks_it(self):
        """An offer that would put words the Warden's rules refuse into a part isn't made (a tap never fails on its
        own): here two capitalised words in a row in the description, which read as a person's full name."""
        s, mine = self.session(SOFA)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["Ocean Mist"]}))
        self.assertEqual(values(mine.draft)["me:colour"], ["Ocean Mist"])  # their own change is theirs
        change = [{"key": "me:colour", "before": ["green"], "after": ["Ocean Mist"], "private": False}]
        self.assertEqual({o["key"] for o in agent.offers(mine.draft, change)}, {"headline", "intent_text"})
        self.assertEqual(self.offers(s), {})  # both would be refused, so neither is offered

    def test_the_phones_save_keeps_a_change_and_the_answers_go_to_the_server(self):
        """The phone's Save draft (and a closed sheet) reaches keep_change; its answers to offers, answer_offers. Both
        work for the chat page's test phone too."""
        from belong import server
        self.assertIn("offers", server.CHAT_ACTIONS)
        s, mine = self.session(SOFA)
        server.ACTIONS["save_draft"](s, {"slot": "A", "listing_id": mine.id,
                                         "draft": kept(mine.draft, {"me:colour": ["blue"]})})
        self.assertEqual(set(self.offers(s)), {"headline", "intent_text"})
        server.ACTIONS["offers"](s, {"slot": "A", "listing_id": mine.id, "keys": ["intent_text"]})
        self.assertEqual(values(mine.draft)["intent_text"], "Selling my blue velvet sofa in Utrecht.")
        with self.assertRaises(DemoError):
            server.ACTIONS["offers"](s, {"slot": "A", "listing_id": mine.id, "keys": "headline"})

    def test_offers_stay_on_the_phone(self):
        s, mine = self.session(SOFA)
        receipts, events = len(s.warden.receipts), len(s.events)
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        s.answer_offers("A", mine.id, ["headline"])
        self.assertEqual((len(s.warden.receipts), len(s.events)), (receipts, events))
        self.assertNotIn("blue", json.dumps({"B": s.phone_state("B"), "room": s.room_state()}))


class OnThePhone(_Page, _Session, unittest.TestCase):
    """demo.html, run with node on real phone states: a kept change goes to the server as Save draft sends it, the peek
    says the agent is updating the profile, a change made meanwhile waits on top of the agent's draft and goes next, the
    change line is in the agent's colour, and offers take one tap each or all."""

    @classmethod
    def setUpClass(cls):
        s = DemoSession(seed=7, agent=None)
        s.verify("A", "Noor", 36)
        mine = s.slots["A"].listing(s.new_listing("A"))
        mine.draft, _, _ = build_draft(SOFA, "Noor", "35-39")
        state = lambda: json.loads(json.dumps({"phones": {"A": s.phone_state("A")}}))
        cls.ready = state()
        assert cls.ready["phones"]["A"]["listings"][0]["has_profile"]
        busy = state()
        l = busy["phones"]["A"]["listings"][0]
        l["agent_busy"] = True
        l["chat"].append({"from": "change", "text": "You changed your colour on your profile."})
        cls.busy = busy
        agent_draft = copy.deepcopy(mine.draft)
        for f in agent_draft["fields"]:
            if f["key"] == "headline":
                f["value"] = "Selling a blue velvet sofa"
            if f["key"] == "me:colour":
                f["value"] = ["blue"]
        done = copy.deepcopy(busy)
        l = done["phones"]["A"]["listings"][0]
        l.update(agent_busy=False, draft=agent_draft)
        l["chat"].append({"from": "agent", "text": "I've updated your headline to match.", "draft_updated": True})
        cls.done = done
        s.keep_change("A", kept(mine.draft, {"me:colour": ["blue"]}))
        cls.offered = state()
        assert len(cls.offered["phones"]["A"]["listings"][0]["offers"]) == 2

    def phone(self, *states, then=""):
        return self.run_page(f"""
const states = {json.dumps(states)}, calls = [];
fetch = (url, o) => {{ calls.push([url, o ? JSON.parse(o.body) : null]); return new Promise(() => {{}}); }};
local.A.view = "profile"; local.A.current = states[0].phones.A.current;
S = states[0]; renderPhone("A");
const field = key => cur("A").draft.fields.find(f => f.key === key);
{then}
console.log(JSON.stringify({{ calls, after: phone("A"), c: cur("A"), out: typeof out === "undefined" ? null : out }}));""")

    def test_closing_a_sheet_with_a_change_keeps_it(self):
        out = self.phone(self.ready, then="""
openDetail("A", "me:colour"); field("me:colour").value = ["blue"]; closeDetail("A");""")
        self.assertEqual(len(out["calls"]), 1)
        url, body = out["calls"][0]
        self.assertEqual(url, "/api/save_draft")
        sent = {f["key"]: f["value"] for f in body["draft"]["fields"]}
        self.assertEqual(sent["me:colour"], ["blue"])
        self.assertEqual(sent["headline"], "Selling a green velvet sofa")  # only what they changed

    def test_closing_a_sheet_without_a_change_sends_nothing(self):
        out = self.phone(self.ready, then='openDetail("A", "headline"); closeDetail("A"); toggleEdit("A"); toggleEdit("A");')
        self.assertEqual(out["calls"], [])

    def test_done_editing_keeps_the_changes_too(self):
        out = self.phone(self.ready, then='toggleEdit("A"); cur("A").draft.intent_text = "Selling my sofa."; toggleEdit("A");')
        self.assertEqual([c[0] for c in out["calls"]], ["/api/save_draft"])
        self.assertEqual(out["calls"][0][1]["draft"]["intent_text"], "Selling my sofa.")

    def test_while_the_agent_updates_the_profile_the_peek_says_so_and_a_change_waits(self):
        out = self.phone(self.ready, self.busy, self.done, then="""
cur("A").chatMin = true; S = states[1]; renderPhone("A"); const peek = phone("A");
openDetail("A", "me:seats"); field("me:seats").value = ["two"]; closeDetail("A");
const waiting = { calls: calls.length, held: cur("A").held };
S = states[2]; renderPhone("A");
var out = { peek, waiting };""")
        self.assertIn("Updating your profile", out["out"]["peek"])
        self.assertIn('role="status"', out["out"]["peek"])
        self.assertEqual(out["out"]["waiting"], {"calls": 0, "held": True})
        self.assertEqual(len(out["calls"]), 1)  # once the agent is done, the change that waited goes
        sent = {f["key"]: f["value"] for f in out["calls"][0][1]["draft"]["fields"]}
        self.assertEqual(sent["me:seats"], ["two"])  # theirs, on top of the agent's draft
        self.assertEqual(sent["headline"], "Selling a blue velvet sofa")  # the agent's update stays
        seats = next(f for f in out["c"]["draft"]["fields"] if f["key"] == "me:seats")
        self.assertEqual(seats["value"], ["two"])

    def test_a_change_that_waited_goes_with_the_copy_it_was_made_on(self):
        out = self.phone(self.ready, self.busy, self.done, then="""
S = states[1]; renderPhone("A");
openDetail("A", "headline"); field("headline").value = "Selling a green velvet sofa, like new"; closeDetail("A");
S = states[2]; renderPhone("A");""")
        self.assertEqual(len(out["calls"]), 1)
        body = out["calls"][0][1]["draft"]
        sent = {f["key"]: f["value"] for f in body["fields"]}
        base = {f["key"]: f["value"] for f in body["base"]["fields"]}
        self.assertEqual(sent["headline"], "Selling a green velvet sofa, like new")
        self.assertEqual(base["headline"], "Selling a green velvet sofa")  # what they changed it from
        self.assertEqual(base["me:colour"], ["green"])

    def test_save_changes_and_start_matching_wait_for_the_agent(self):
        out = self.phone(self.busy, then='cur("A").chatMin = true; renderPhone("A");')
        import re
        button = re.search(r"<button class=\"btn sm\"[^>]*onclick=\"submit\('A'\)\"", out["after"]).group(0)
        self.assertIn("disabled", button)
        ready = self.phone(self.ready, then='cur("A").chatMin = true; renderPhone("A");')
        self.assertNotIn("disabled", re.search(r"<button class=\"btn sm\"[^>]*onclick=\"submit\('A'\)\"", ready["after"]).group(0))

    def test_once_kept_the_peek_says_the_agent_is_updating_the_profile(self):
        """The server answers a kept change with the agent at work: the chat steps down to its peek, which says so (the
        phone's own words, announced), and the profile stays in front of the person, who can keep editing."""
        out = self.run_page(f"""
const states = {json.dumps([self.ready, self.busy])}, calls = [];
renderRoom = () => {{}};
fetch = (url, o) => {{ calls.push(url); return Promise.resolve({{ json: () => Promise.resolve(states[1]) }}); }};
local.A.view = "profile"; local.A.current = states[0].phones.A.current;
S = states[0]; renderPhone("A"); cur("A").chatMin = false; renderPhone("A");
const open = phone("A");
(async () => {{
  openDetail("A", "me:colour"); cur("A").draft.fields.find(f => f.key === "me:colour").value = ["blue"];
  cur("A").detail = null; renderPhone("A");
  await keepChange("A");
  console.log(JSON.stringify({{ open, calls, after: phone("A"), c: cur("A") }}));
}})();""")
        self.assertNotIn("Updating your profile", out["open"])
        self.assertEqual(out["calls"], ["/api/save_draft"])
        self.assertTrue(out["c"]["chatMin"])
        self.assertIn('class="chat-sheet min"', out["after"])
        self.assertIn('<div class="bubble agent working" role="status">', out["after"])
        self.assertIn("Updating your profile", out["after"])
        self.assertIn('class="profile-pane', out["after"])

    def test_the_change_line_is_in_the_agents_colour(self):
        out = self.phone(self.done, then='cur("A").chatMin = false; renderPhone("A");')
        self.assertIn('<p class="change-line">You changed your colour on your profile.</p>', out["after"])
        from tests.test_dating import PAGE
        import re
        rule = re.search(r"\.change-line \{([^}]*)\}", PAGE).group(1)
        self.assertIn("color: var(--text-ai)", rule)

    def test_offers_take_one_tap_each_or_all(self):
        out = self.phone(self.offered, then="""
const html = phone("A");
answerOffers("A", ["headline"]); answerOffers("A", null); answerOffers("A", []);
var out = html;""")
        html = out["out"]
        self.assertIn("Change these too?", html)
        self.assertIn("Selling a blue velvet sofa", html)
        self.assertIn("Change all", html)
        self.assertIn("Leave as is", html)
        self.assertNotIn('class="btn sm"', html.split('class="offers"')[1].split("</div></div>")[0])  # never coral
        keys = [c[1]["keys"] for c in out["calls"]]
        self.assertEqual(keys, [["headline"], ["headline", "intent_text"], []])
        self.assertTrue(all(c[0] == "/api/offers" for c in out["calls"]))


if __name__ == "__main__":
    unittest.main()
