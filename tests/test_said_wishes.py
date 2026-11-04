"""A must on a listing comes from its person (TRUST.md #45, SPEC §5).

A wish (an open detail about them or both) that an agent saves as one that must be met shuts out every listing that
doesn't show something to meet it. So it stays a must only by one of three fixed rules (agent.ground_wishes, no word
list): what the person typed for the listing says one of its values (agent.wish_said: words compared as same_word
compares them, a number only as the same number, with the word written right after it; the detail's name alone never
counts); or a save kept it as ideally, their agent asked, and the words it quotes as theirs for it are all in the
message they typed next, their answer (agent.answered); or the draft already had it as a must. Otherwise it is kept as
"ideally": only made not required, never taken off, reworded, or made shown or hidden. At exactness 0.8 or more, where
every wish shuts people out, a listing with a wish that passes none of these is made less exact
(agent.ground_exactness). Their own words are what they typed themselves for the
listing the save lands on, never a turn the app wrote. Examples are several kinds of listing, none of them the case that
was reported. No test calls the network: model agents are scripted stand-ins."""
from __future__ import annotations

import copy
import unittest
from unittest import mock

from belong import agent, agent_llm, server
from belong.agent import (THEIR_WORDS_MAX, UNSTRICT, answered, detail_row, ground_exactness, ground_wishes, said_words,
                          wish_said)
from belong.agent_llm import (EXACTNESS_LOWERED, KEPT_IDEALLY, SAVE_CHECK, WISH_IDEALLY, ClaudeAgent, _clean_details,
                              build_draft, profile_so_far)
from belong.agent_openai import OpenAIAgent
from belong.demo import DemoSession
from belong.schema import HIDDEN, SHOWN, STRICT_EXACTNESS, Field, is_detail_key, split_key
from tests.test_agent_llm import ScriptedClient, ScriptedPost, det, oa_text, oa_tool, reply, text, tool
from tests.test_demo import box_people, fill
from tests.test_kept_changes import args, kept

GHENT = "Looking for a sourdough baking partner in Ghent on Sundays."
BAKER = args(intent=GHENT,
             fields=[("city", "Ghent", SHOWN), ("country", "Belgium", SHOWN), ("seeking", "a baking partner", SHOWN),
                     ("offering", "a baking partner", SHOWN), ("headline", "Baking sourdough together", SHOWN)],
             details=[det("bake day", "sunday", about="both"), det("oven type", "wood-fired", about="them")])
CASCAIS = "Looking for crew for my dinghy in Cascais."
SAILOR = args(intent=CASCAIS,
              fields=[("city", "Cascais", SHOWN), ("country", "Portugal", SHOWN), ("seeking", "a sailing crew", SHOWN),
                      ("offering", "a dinghy skipper", SHOWN), ("headline", "Sailing a dinghy off Cascais", SHOWN)],
              details=[det("boat type", "dinghy")])


def wish(about, name, *values, compare="same", steps=()):
    return detail_row(about, name, compare, steps, values)


def must(name, *values, about="them", words=""):
    """A wish the agent saves as a must, with the words it quotes as the person's own for it (their_words)."""
    return {**det(name, *values, about=about), "their_words": words}


INNSBRUCK = "Looking for a hiking buddy around Innsbruck on Saturdays, they must have a car."
# The value in its common form, and in the person's own words next to it (an extra value widens who meets the wish).
HIKER = args(intent="Looking for a hiking buddy around Innsbruck on Saturdays.",
             fields=[("city", "Innsbruck", SHOWN), ("country", "Austria", SHOWN), ("seeking", "a hiking buddy", SHOWN),
                     ("offering", "a hiking buddy", SHOWN), ("headline", "Hiking around Innsbruck", SHOWN)],
             details=[must("transport", "has a car", "have a car")])
PORTO = "Looking for a flatmate in Porto."
FLAT = args(intent=PORTO,
            fields=[("city", "Porto", SHOWN), ("country", "Portugal", SHOWN), ("seeking", "a flatmate", SHOWN),
                    ("offering", "a room", SHOWN), ("headline", "Sharing a flat in Porto", SHOWN)])


def flat(*details):
    return {**FLAT, "details": list(details)}


def rows_of(draft) -> dict:
    return {f["key"]: f for f in draft["fields"] if ":" in f["key"]}


def told(client, n=1) -> str:
    """The tool result the model was sent after its save (the n-th request carries it)."""
    return client.requests[n]["messages"][-1]["content"][0]["content"]


class TheRule(unittest.TestCase):
    """agent.py: whether the person said a wish, by one fixed rule over their own words."""

    def said(self, row, *texts) -> bool:
        return wish_said(row, said_words(texts))

    def test_a_wish_is_said_when_one_of_its_values_is_in_their_words(self):
        self.assertTrue(self.said(wish("both", "bake day", "sunday"), GHENT))  # "Sundays": another form of it
        self.assertFalse(self.said(wish("them", "oven type", "wood-fired"), GHENT))
        self.assertTrue(self.said(wish("them", "oven type", "wood-fired"), "they need a wood-fired oven"))  # a hyphen
        self.assertTrue(self.said(wish("them", "outdoor activity", "hike"), "I like hiking"))  # another tense
        self.assertFalse(self.said(wish("them", "rating", "1800"), "I play chess online most evenings"))
        self.assertTrue(self.said(wish("them", "rating", "1800"), "someone rated around 1800 please"))  # a number
        self.assertFalse(self.said(wish("them", "rating", "1800"), "someone rated around 1600 please"))
        # Every word of a value that says something must be said: half of it is not the value.
        self.assertFalse(self.said(wish("them", "oven type", "wood-fired"), "a wood stove is fine"))
        # Nothing but their words counts: an empty chat says no wish.
        self.assertFalse(self.said(wish("both", "bake day", "sunday")))

    def test_a_value_of_grammar_sized_words_counts_only_with_the_details_name(self):
        """Alone, a word of one or two letters says nothing of what it's about."""
        self.assertFalse(self.said(wish("them", "pets", "no"), "no rush, happy to chat"))
        self.assertTrue(self.said(wish("them", "pets", "no"), "no pets please"))
        self.assertFalse(self.said(wish("them", "pets", "no"), "pets are welcome"))  # the name alone isn't the value

    def test_a_wish_that_says_nothing_is_always_said(self):
        """A lone "any", or every step of its scale, shuts nobody out: nothing to lower."""
        self.assertTrue(self.said(wish("them", "rating", "any"), "hello"))
        self.assertTrue(self.said(wish("them", "noise", "quiet", "lively", compare="steps", steps=("quiet", "lively")),
                                  "hello"))

    def test_one_value_is_enough(self):
        """More values widen who meets a wish, never whom it shuts out for not showing the detail: one said is enough
        for the rest to stay with it."""
        self.assertTrue(self.said(wish("both", "bake day", "saturday", "sunday"), GHENT))
        self.assertTrue(self.said(wish("them", "noise", "quiet", "moderate", compare="steps",
                                       steps=("quiet", "moderate", "lively")), "someone quiet to share a flat with"))

    def test_in_a_script_written_without_spaces_a_value_is_said_inside_a_run(self):
        self.assertTrue(self.said(wish("them", "free day", "週末"), "週末に一緒にパンを焼きたいです"))
        self.assertFalse(self.said(wish("them", "free day", "週末"), "平日の夜に一緒にパンを焼きたいです"))

    def test_the_name_alone_never_counts(self):
        """Words about themselves can share a wish's name ("my stroke is..."): only one of its values says the wish."""
        stroke = wish("them", "swimming stroke", "butterfly")
        self.assertFalse(self.said(stroke, "My swimming stroke is breaststroke; looking for a training partner."))
        self.assertTrue(self.said(stroke, "Looking for a training partner who swims butterfly."))
        self.assertFalse(self.said(wish("both", "start time", "early"), "Which start time suits you, and where?"))

    def test_a_quote_counts_only_as_their_answer_typed_after_a_save_kept_the_wish_as_ideally(self):
        """The words an agent quotes as the person's own for a must count only as their answer to its question whether
        it must be met: every word of them in one message they typed after a save kept that wish as ideally. Words
        typed before that save (their request, anything else they said) are never their answer, however exactly
        quoted."""
        car = wish("them", "transport", "has a car")  # "has" was never typed: the value alone isn't said
        words = said_words([INNSBRUCK])
        self.assertFalse(wish_said(car, words))
        key = car["key"]
        # Quoted from their request, before any save kept it as ideally: nothing to answer yet, so it counts for nothing.
        for quote in ("they must have a car", "Looking for a hiking buddy", "hiking"):
            self.assertEqual(ground_wishes([car], words, {key: quote})[1], [key], quote)
            self.assertEqual(ground_wishes([car], words, {key: quote}, answers={key: []})[1], [key], quote)
        # Their answer, typed after the save that kept it as ideally, quoted: it is a must.
        after = ["Yes, it has to be."]
        self.assertEqual(ground_wishes([car], words, {key: "Yes, it has to be"}, answers={key: after})[1], [])
        self.assertTrue(answered("it has to be", after))
        # Quoted from before that save while an answer came after, in part or in full: not their answer.
        for quote in ("they must have a car", "Yes, they must have a car"):
            self.assertFalse(answered(quote, after), quote)
        # Every word in one message, not pieced together from two.
        self.assertFalse(answered("Yes, it has to be", ["Yes.", "It has to be."]))
        self.assertTrue(answered("Yes", ["Yes.", "It has to be."]))
        # An answer about another wish is no answer about this one.
        self.assertEqual(ground_wishes([car], words, {key: "Yes, it has to be"}, answers={"them:pets": after})[1], [key])
        # Words they never typed, in part or in full, an empty quote, or a whole message rather than the few words
        # that say it.
        for quote in ("they need their own car", "a car please", "a", "", None, 7):
            self.assertFalse(answered(quote, after), quote)
        long = " ".join(["walk"] * (THEIR_WORDS_MAX - 1) + ["car", "they"])
        self.assertFalse(answered(long, [long]))
        self.assertTrue(answered("walk car", [long]))

    def test_a_number_is_said_only_as_the_same_number(self):
        """No form of a number adds or drops a digit: a shorter or longer number that starts the same is another one."""
        rating = wish("them", "rating", "1800 rating")
        self.assertTrue(self.said(rating, "a chess partner with a 1800 rating please"))
        for other in ("I'm 180 cm tall, looking for a chess partner with a rating", "about 18000 games and a rating",
                      "a rating of 1600"):
            self.assertFalse(self.said(rating, other), other)
        self.assertFalse(self.said(wish("them", "budget", "2000 a month"), "about 200 a month"))
        self.assertTrue(self.said(wish("them", "era", "1970"), "records from the 1970s"))  # the same number, a plural
        # Written with or without a thousands separator, or a space before its unit, a number is the same number.
        self.assertTrue(self.said(rating, "a chess partner with a 1,800 rating please"))
        self.assertTrue(self.said(wish("them", "budget", "1,800 a month"), "up to 1800 a month"))
        self.assertTrue(self.said(wish("both", "run distance", "5 km"), "a run of 5km after work"))
        self.assertTrue(self.said(wish("both", "run distance", "5km"), "a 5 km run after work"))
        self.assertFalse(self.said(wish("both", "run distance", "5 km"), "a 15km run after work"))
        self.assertFalse(self.said(rating, "a chess partner with a 1,600 rating please"))

    def test_a_number_and_the_word_right_after_it_are_one_word(self):
        """The word written right after a number says what it counts: with or without a space or a dash, the two are
        one word, said only by that number with that word. A number they typed with a unit still says the number."""
        for value, typed, said in (("10km", "Looking for a running partner, I'm free after 10 on Sundays.", False),
                                   ("5 km", "Let's meet at 5 by the river.", False),
                                   ("5km", "Looking for someone for a 5 km run.", True),
                                   ("2 l", "Two of us, we start at 2 and paddle all afternoon.", False),
                                   ("2 l", "Bring 2 l of water for the ride.", True),
                                   ("40 min", "A session of 40 min, three times a week.", True),
                                   ("40 min", "We're about 40 and play every week.", False),
                                   ("30 minutes", "A 30-minute warm-up before we spar.", True),
                                   ("3pm", "Rehearsals start at 3 on Fridays.", False),
                                   ("12", "Our choir has 12 voices.", True),
                                   ("12", "We rehearse 12km out of town.", True)):
            self.assertEqual(self.said(wish("them", "x", value), typed), said, (value, typed))

    def test_in_a_script_written_without_spaces_a_number_is_never_inside_a_longer_one(self):
        self.assertTrue(self.said(wish("them", "距離", "5キロ"), "毎朝5キロ一緒に走りたい"))
        self.assertFalse(self.said(wish("them", "距離", "5キロ"), "毎朝15キロ一緒に走りたい"))
        self.assertFalse(self.said(wish("them", "距離", "1キロ"), "毎朝10キロ一緒に走りたい"))

    def test_their_answer_is_the_message_they_type_next(self):
        """Only the first message typed after the latest save that kept a wish as ideally is their answer to the
        question: words of any message after it count for nothing, whatever they say."""
        self.assertTrue(answered("Yes, it must", ["Yes, it must.", "Thanks!"]))
        for quote in ("Thanks", "Yes, it must. Thanks"):
            self.assertFalse(answered(quote, ["Yes, it must.", "Thanks!"]), quote)
            self.assertFalse(answered(quote, ["No, it doesn't matter.", "Thanks!"]), quote)
        self.assertFalse(answered("Yes", []))

    def test_at_strict_exactness_every_wish_is_held_to_the_rule(self):
        """At exactness 0.8 or more a wish that isn't required still shuts out who doesn't show it, so there every
        wish must pass the rule as a must does: one that passes none makes the listing less exact, and nothing else
        changes. Said, answered, or on a draft that was as exact already, it stays as exact; it is never raised."""
        words = said_words([GHENT])
        said, unsaid = wish("both", "bake day", "sunday"), wish("them", "oven type", "wood-fired")
        ideally = {**unsaid, "required": False}
        city = {"key": "city", "value": "Ghent", "visibility": SHOWN}
        self.assertEqual(ground_exactness(0.9, [city, said], words), (0.9, []))
        self.assertEqual(ground_exactness(0.9, [city, said, unsaid], words), (UNSTRICT, ["them:oven type"]))
        self.assertEqual(ground_exactness(0.9, [said, ideally], words), (UNSTRICT, ["them:oven type"]))
        self.assertLess(UNSTRICT, STRICT_EXACTNESS)
        for low in (0.6, UNSTRICT, None, "0.9", True):
            self.assertEqual(ground_exactness(low, [said, ideally], words), (low, []), low)
        # Their answer, typed next after the save that made it less exact, keeps it exact.
        key = "them:oven type"
        self.assertEqual(ground_exactness(0.9, [ideally], words, {key: "Yes, it has to be"},
                                          answers={key: ["Yes, it has to be."]}), (0.9, []))
        self.assertEqual(ground_exactness(0.9, [ideally], words, {key: "Looking for a sourdough baking partner"},
                                          answers={key: ["Yes, it has to be."]}), (UNSTRICT, [key]))
        # A draft that was as exact with it already (as wide or wider): it passed the rule then, or went live so.
        self.assertEqual(ground_exactness(0.9, [ideally], words, before=[ideally], before_exactness=0.9), (0.9, []))
        self.assertEqual(ground_exactness(1.0, [ideally], words, before=[ideally], before_exactness=0.8), (1.0, []))
        self.assertEqual(ground_exactness(0.9, [ideally], words, before=[ideally], before_exactness=UNSTRICT),
                         (UNSTRICT, [key]))
        # Fields too (the simulation's), never changed.
        fields = (Field("both:bake day", ("sunday",), SHOWN, False, "same", required=False),
                  Field("them:oven type", ("wood-fired",), SHOWN, False, "same", required=False))
        self.assertEqual(ground_exactness(0.9, fields, words), (UNSTRICT, ["them:oven type"]))
        self.assertEqual([f.required for f in fields], [False, False])

    def test_a_word_keeps_the_marks_of_its_letters(self):
        """A vowel sign in Devanagari or Bengali is part of its word: a value typed as it is counts, and a message
        that only shares some of its letters doesn't."""
        language = wish("them", "भाषा", "मराठी")
        self.assertTrue(self.said(language, "पुणे में साइकिल चलाने का साथी चाहिए, मराठी बोलना जरूरी"))
        self.assertFalse(self.said(language, "पुणे में सुबह साइकिल चलाने का साथी चाहिए, कभी भी"))
        self.assertTrue(self.said(wish("them", "খেলা", "দাবা"), "কলকাতায় দাবা খেলার সঙ্গী চাই"))

    def test_in_hangul_a_word_counts_with_its_particles(self):
        """Hangul writes a noun's particles on to it, and counts a syllable as the letters it is written with: a
        two-syllable value says something, and counts said with a particle after it."""
        day = wish("them", "요일", "주말")
        self.assertTrue(self.said(day, "부산에서 같이 테니스 칠 사람, 꼭 주말에"))
        self.assertTrue(self.said(day, "주말 테니스"))
        self.assertFalse(self.said(day, "부산에서 같이 테니스 칠 사람, 평일 저녁"))
        self.assertTrue(self.said(wish("them", "악기", "바이올린"), "바이올린을 켜는 사람"))
        # A syllable standing alone is a word that says something, said as it is.
        self.assertTrue(self.said(wish("them", "이동 수단", "차"), "대전에서 캠핑 같이 갈 사람, 차 있는 분"))
        self.assertFalse(self.said(wish("them", "이동 수단", "차"), "대전에서 캠핑 같이 갈 사람, 자전거 타는 분"))

    def test_a_must_the_draft_already_had_stays(self):
        """A must on the draft a save replaces passed the rule when it was saved, or came from what went live: saved
        again as it was, or wider, it stays, whatever this turn says. Narrowed or changed, or newly made a must, it is
        held to the rule again."""
        steps = ("short", "medium", "long")
        before = [wish("them", "hike length", "medium", "long", compare="steps", steps=steps)]
        unsaid = said_words(["Mornings suit me."])
        for row, kept in ((before[0], True),
                          (wish("them", "hike length", "short", "medium", "long", compare="steps", steps=steps), True),
                          (wish("them", "hike length", "long", compare="steps", steps=steps), False),
                          (wish("them", "hike length", "medium", "long", compare="steps", steps=("short", "long", "x")),
                           False),
                          (wish("them", "hike length", "medium", "long"), False)):
            out, ideally = ground_wishes([row], unsaid, before=before)
            self.assertEqual(out[0]["required"], kept, row)
        # One that was only ideally, made a must now, needs their words: a value, or their answer when asked, never a
        # quote of what they typed before.
        ideally = [{**before[0], "required": False}]
        self.assertEqual(ground_wishes(before, unsaid, before=ideally)[1], ["them:hike length"])
        self.assertEqual(ground_wishes(before, unsaid, {"them:hike length": "Mornings suit me"}, before=ideally)[1],
                         ["them:hike length"])
        self.assertEqual(ground_wishes(before, unsaid, {"them:hike length": "Yes, it must"}, before=ideally,
                                       answers={"them:hike length": ["Yes, it must."]})[1], [])

    def test_it_only_ever_lowers_required(self):
        """A wish saved as a must that they didn't say is kept as ideally; nothing else about any row changes, nothing is
        added or taken off, and what is about the person, said or not, a wish they said and one already ideally stay
        as they were. The rows given are never changed in place."""
        rows = [{"key": "city", "value": "Ghent", "visibility": SHOWN},
                {**wish("me", "starter age", "ten years"), "visibility": HIDDEN},
                wish("both", "bake day", "sunday"),
                wish("them", "oven type", "wood-fired"),
                {**wish("them", "flour", "rye"), "required": False},
                wish("both", "loaf size", "large", compare="steps", steps=("small", "large"))]
        before = copy.deepcopy(rows)
        out, ideally = ground_wishes(rows, said_words([GHENT]))
        self.assertEqual(rows, before)
        self.assertEqual(ideally, ["them:oven type", "both:loaf size"])
        self.assertEqual([r["key"] for r in out], [r["key"] for r in rows])
        for old, new in zip(rows, out):
            expected = {**old, "required": False} if old["key"] in ideally else old
            self.assertEqual(new, expected)
        # A listing's Fields too (the simulation's), the same way.
        fields = (Field("me:starter age", ("ten years",), HIDDEN, False, "same"),
                  Field("them:oven type", ("wood-fired",), SHOWN, False, "same"),
                  Field("both:bake day", ("sunday",), SHOWN, False, "same"))
        out, ideally = ground_wishes(fields, said_words([GHENT]))
        self.assertEqual(ideally, ["them:oven type"])
        self.assertEqual([f.required for f in out], [True, False, True])
        self.assertEqual([(f.key, f.value, f.visibility) for f in out], [(f.key, f.value, f.visibility) for f in fields])

    def test_a_must_only_when_the_agent_says_so(self):
        """The model agent's save: `required` left out, or anything but true, is ideally."""
        rows, problems = _clean_details([{**det("oven type", "wood-fired", about="them"), "required": r}
                                         for r in (True, False, "yes", None)]
                                        + [{k: v for k, v in det("flour", "rye", about="them").items() if k != "required"}])
        self.assertEqual(problems, [])
        self.assertEqual([r["required"] for r in rows], [True, False, False, False, False])

    def test_the_rule_based_agents_own_wishes_are_their_words(self):
        """The offline agent's behaviour doesn't change: what both must share comes from a listed value said in the
        message, so the rule leaves it a must."""
        vocab = {"details": [{"about": "both", "name": "sail day", "compare": "same"}], "scales": [],
                 "values": [{"about": "both", "name": "sail day", "value": "saturday"}],
                 "phrases": [{"key": "seeking", "value": "a sailing partner"},
                             {"key": "offering", "value": "a sailing partner"}],
                 "places": [{"city": "Cascais", "country": "Portugal"}]}
        message = "Looking for a sailing partner in Cascais, saturday works for me."
        draft = agent.draft_from_message(message, "Noor", "30-34", words=agent.words_in_use(vocab))
        self.assertTrue(rows_of(draft)["both:sail day"]["required"])
        self.assertEqual(ground_wishes(draft["fields"], said_words([message]))[1], [])


class WithAModelAgent(unittest.TestCase):
    """The demo gives a model agent what the person typed for the listing; build_draft keeps a wish they didn't say
    as ideally and tells it; and every agent's draft is held to the same rule before it is kept."""

    def session(self, llm, name="Noor", age=36):
        s = DemoSession(seed=7, agent=llm)
        s.run_inline = True
        s.verify("A", name, age)
        return s

    def test_a_must_they_didnt_say_is_saved_as_ideally_and_the_agent_is_told(self):
        client = ScriptedClient(reply(tool(BAKER)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", GHENT)
        wishes = rows_of(s.slots["A"].listing().draft)
        self.assertEqual((wishes["both:bake day"]["required"], wishes["them:oven type"]["required"]), (True, False))
        self.assertEqual(wishes["them:oven type"]["value"], ["wood-fired"])  # kept, as written
        result = client.requests[1]["messages"][-1]["content"][0]
        self.assertFalse(result.get("is_error"))  # a note, never a refusal: the save went through
        self.assertTrue(result["content"].startswith("Draft saved."))
        self.assertIn(WISH_IDEALLY.format(name="oven type (about them)"), result["content"])
        self.assertNotIn("bake day (about both)' was saved as ideally", result["content"])
        # What a match will see says how much each wish matters, and so does the profile the agent builds on.
        self.assertIn("oven type (about them)=wood-fired (ideally)", result["content"])
        self.assertIn("bake day (about both)=sunday (must)", result["content"])
        profile = profile_so_far(s.slots["A"].listing().draft)
        self.assertIn('detail "oven type" (about them, same, ideally; shown): wood-fired', profile)
        self.assertIn('detail "bake day" (about both, same, must; shown): sunday', profile)
        # Their own words, said later, make it a must on the next save.
        client.responses += [reply(tool(BAKER)), reply(text("Saved."))]
        s.agent_message("A", "They must have a wood-fired oven.")
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:oven type"]["required"])

    def test_the_same_for_openai(self):
        post = ScriptedPost(oa_tool(BAKER), oa_text("Saved."))
        s = self.session(OpenAIAgent(api_key="test", post=post))
        s.agent_message("A", GHENT)
        self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:oven type"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="oven type (about them)"), post.requests[1]["messages"][-1]["content"])

    def test_the_connector_holds_to_it_too(self):
        """The connector's messages go through the same route as the phone's."""
        self.assertIn("agent", server.CHAT_ACTIONS)
        client = ScriptedClient(reply(tool(BAKER)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        server.ACTIONS["agent"](s, {"slot": "A", "text": GHENT})
        self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:oven type"]["required"])

    def test_the_apps_save_check_never_counts(self):
        """A value only in the turn the app writes when the agent replied without saving is not theirs, then or in a
        later turn (the check stays in the agent's conversation once it saved)."""
        call = {**SAILOR, "details": SAILOR["details"] + [det("start", "now", about="both")]}
        self.assertIn(" now ", SAVE_CHECK)
        client = ScriptedClient(reply(text("Tell me more about your boat.")), reply(tool(call)), reply(text("Saved.")),
                                reply(tool(call)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        mine = s.slots["A"].listing(s.new_listing("A"))
        s.agent_message("A", CASCAIS, mine.id)
        self.assertEqual(client.requests[1]["messages"][-1]["content"], SAVE_CHECK)
        self.assertFalse(rows_of(mine.draft)["both:start"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="start (about both)"), told(client, 2))
        s.agent_message("A", "It's a two-person boat.", mine.id)
        self.assertIn({"role": "user", "content": SAVE_CHECK}, mine.agent_conversation)
        self.assertFalse(rows_of(mine.draft)["both:start"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="start (about both)"), told(client, 4))

    def test_a_change_typed_on_the_profile_and_the_apps_turn_about_it_dont_count_yet(self):
        """Values typed by hand on the profile aren't their words in a message (not yet), and nor is the turn the app
        writes to tell the agent about them: a must the agent writes later from them is kept as ideally."""
        renamed = {**SAILOR, "intent_text": "Looking for crew for my catamaran in Cascais.",
                   "fields": [f if f["key"] != "headline" else {**f, "values": ["Sailing a catamaran off Cascais"]}
                              for f in SAILOR["fields"]],
                   "details": [det("boat type", "catamaran")]}
        later = {**renamed, "details": renamed["details"] + [det("boat type", "catamaran", about="them")]}
        client = ScriptedClient(reply(tool(SAILOR)), reply(text("Saved.")),
                                reply(tool(renamed)), reply(text("I've updated your headline and description.")),
                                reply(tool(later)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", CASCAIS)
        mine = s.slots["A"].listing()
        s.keep_change("A", kept(mine.draft, {"me:boat type": ["catamaran"]}))
        self.assertIn('"catamaran"', client.requests[2]["messages"][-1]["content"])  # the app's turn says it
        s.agent_message("A", "Weekends suit me best.")
        self.assertFalse(rows_of(mine.draft)["them:boat type"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="boat type (about them)"), told(client, 5))

    def test_a_yes_to_a_question_doesnt_count(self):
        """The turn that rewords a listing after a yes to "Would that work for you?" is written by the app (with the
        other listing's words), so nothing in it makes a must."""
        s = DemoSession(seed=7, agent=None)
        s.run_inline = True
        seller, seeker = box_people(s)  # nobody typed anything: both listings were made on the phone
        reworded = {"kind": "other", "intent_text": "Giving away cardboard boxes in Nagoya.", "exactness": 0.6,
                    "suggestions": [], "separate_listing": False,
                    "fields": [{"key": "headline", "values": ["Giving away 100 cardboard boxes"], "visibility": SHOWN}],
                    "details": [det("item", "cardboard boxes", about="both")]}
        client = ScriptedClient(reply(tool(reworded)), reply(text("Updated.")),
                                reply(tool(reworded)), reply(text("Saved.")))
        s.llm = ClaudeAgent(client=client)
        s.run_matchmaker()
        s.answer_followup("B", "also_offer", {"yes": False}, seeker)
        s.run_matchmaker()
        s.answer_followup("A", "also_offer", {"yes": True}, seller)
        asked = client.requests[0]["messages"][-1]["content"]
        self.assertIn("cardboard boxes", asked)  # the app's turn says it, in the first person
        self.assertIn(WISH_IDEALLY.format(name="item (about both)"), told(client))
        self.assertEqual(s.board.listing(seller).get("headline").value, "Giving away 100 cardboard boxes")
        # It stays in the agent's conversation, and still isn't theirs in a later turn.
        mine = s.slots["A"].listing(seller)
        self.assertIn({"role": "user", "content": asked}, mine.agent_conversation)
        s.agent_message("A", "Happy to hand them over this week.", seller)
        self.assertFalse(rows_of(mine.draft)["both:item"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="item (about both)"), told(client, 3))

    def test_a_must_in_their_own_words_stays_one_next_to_a_common_form(self):
        """The agent keeps a value as they wrote it next to its common form: it stays a must. With the common form
        alone, the words they typed for it are their request, not an answer to anything, so quoting them keeps
        nothing: it is saved as ideally and the agent is told, so it can ask."""
        client = ScriptedClient(reply(tool(HIKER)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", INNSBRUCK)
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])
        self.assertNotIn("was saved as ideally", told(client))
        self.assertIn("transport (about them)=has a car, have a car (must)", told(client))
        for quote in ("they must have a car", "they need their own car"):
            common = {**HIKER, "details": [must("transport", "has a car", words=quote)]}
            client = ScriptedClient(reply(tool(common)), reply(text("Saved.")))
            s = self.session(ClaudeAgent(client=client))
            s.agent_message("A", INNSBRUCK)
            self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"], quote)
            self.assertIn(WISH_IDEALLY.format(name="transport (about them)"), told(client))

    def test_an_unrelated_quote_never_keeps_a_made_up_must(self):
        """The agent makes up a must they never mentioned and quotes words they did type, about something else: the
        must is kept as ideally all the same, and the agent is told."""
        for quote in (PORTO, "flatmate", "Looking for a flatmate in Porto"):
            client = ScriptedClient(reply(tool(flat(must("pets", "no pets", words=quote)))), reply(text("Saved.")))
            s = self.session(ClaudeAgent(client=client))
            s.agent_message("A", PORTO)
            self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:pets"]["required"], quote)
            self.assertIn(WISH_IDEALLY.format(name="pets (about them)"), told(client))
        post = ScriptedPost(oa_tool(flat(must("pets", "no pets", words="flatmate"))), oa_text("Saved."))
        s = self.session(OpenAIAgent(api_key="test", post=post))
        s.agent_message("A", PORTO)
        self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:pets"]["required"])

    def test_a_quote_from_before_the_save_that_kept_it_as_ideally_never_counts(self):
        """Their answer is the message typed next after the save that kept the wish as ideally: a quote of anything they
        typed before it counts for nothing, in that turn or any later one. A save that keeps it as ideally again has
        their agent told again, so it asks again, and their answer to that counts."""
        ask = flat(must("pets", "no pets"))
        before = flat(must("pets", "no pets", words="Looking for a flatmate"))
        answer = flat(must("pets", "no pets", words="Yes, it has to be"))
        client = ScriptedClient(reply(tool(ask)), reply(text("Must they have no pets?")),
                                reply(tool(before)), reply(text("Saved. Must they have no pets?")),
                                reply(tool(answer)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", PORTO)
        mine = s.slots["A"].listing()
        self.assertEqual(mine.ideally_since, {"them:pets": 1})  # after their first message
        s.agent_message("A", "Yes, it must.")
        self.assertFalse(rows_of(mine.draft)["them:pets"]["required"])  # their request, quoted: not their answer
        self.assertIn(WISH_IDEALLY.format(name="pets (about them)"), told(client, 3))
        self.assertEqual(mine.ideally_since, {"them:pets": 2})  # kept so again: asked again, as of their second message
        s.agent_message("A", "Yes, it has to be.")
        self.assertTrue(rows_of(mine.draft)["them:pets"]["required"])  # their answer to the question asked again
        self.assertEqual(mine.ideally_since, {})  # a must now: nothing is waiting for an answer

    def test_their_answer_is_the_message_they_type_next_after_they_were_asked(self):
        """Their answer, typed next and quoted a turn later, counts. Words of a message typed after their answer (a
        word of thanks) never do, so an agent can't keep a must it made up by quoting whatever they say later; it can still
        quote the answer itself, whatever it says (TRUST.md limit 28)."""
        ask, later = flat(must("pets", "no pets")), flat(must("pets", "no pets", words="Cheers"))
        answer = flat(must("pets", "no pets", words="No, that's fine"))
        for quoted, required in ((later, False), (answer, True)):
            client = ScriptedClient(reply(tool(ask)), reply(text("Must they have no pets?")),
                                    reply(text("Noted.")), reply(text("Not yet.")),
                                    reply(tool(quoted)), reply(text("Saved.")))
            s = self.session(ClaudeAgent(client=client))
            for said in (PORTO, "No, that's fine either way.", "Cheers!"):
                s.agent_message("A", said)
            self.assertEqual(rows_of(s.slots["A"].listing().draft)["them:pets"]["required"], required, quoted)
        self.assertIn(WISH_IDEALLY.format(name="pets (about them)"), told(client))

    def test_a_listing_that_exact_with_a_wish_they_didnt_say_is_made_less_exact_and_the_agent_is_told(self):
        """At exactness 0.8 or more an "ideally" wish still shuts out who doesn't show it, so a save that exact with a
        wish they didn't say, a must or ideally, is made less exact and the agent is told (a note, never a refusal);
        their yes, quoted, makes it a must and the listing can be as exact as they want. With every wish said, it stays
        as exact."""
        exact = lambda call: {**call, "exactness": 0.9}  # noqa: E731
        for details in ([must("oven type", "wood-fired")],
                        [{**det("oven type", "wood-fired", about="them"), "required": False, "their_words": ""}]):
            call = exact({**BAKER, "details": [det("bake day", "sunday", about="both"), *details]})
            yes = exact({**BAKER, "details": [det("bake day", "sunday", about="both"),
                                              must("oven type", "wood-fired", words="Yes, it has to be")]})
            client = ScriptedClient(reply(tool(call)), reply(text("Saved. Must they have a wood-fired oven?")),
                                    reply(tool(yes)), reply(text("Saved.")))
            s = self.session(ClaudeAgent(client=client))
            s.agent_message("A", GHENT)
            mine = s.slots["A"].listing()
            self.assertEqual(mine.draft["exactness"], UNSTRICT)
            self.assertFalse(rows_of(mine.draft)["them:oven type"]["required"])
            self.assertEqual(rows_of(mine.draft)["them:oven type"]["value"], ["wood-fired"])  # kept, as written
            result = client.requests[1]["messages"][-1]["content"][0]
            self.assertFalse(result.get("is_error"))
            self.assertIn(EXACTNESS_LOWERED.format(exactness=UNSTRICT, asked=0.9, strict=STRICT_EXACTNESS,
                                                   names="'oven type (about them)'"), result["content"])
            self.assertEqual(mine.ideally_since, {"them:oven type": 1})  # waiting for their answer
            s.agent_message("A", "Yes, it has to be.")
            self.assertEqual((mine.draft["exactness"], rows_of(mine.draft)["them:oven type"]["required"]), (0.9, True))
        said = exact({**BAKER, "details": [det("bake day", "sunday", about="both")]})
        client = ScriptedClient(reply(tool(said)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", GHENT)
        self.assertEqual(s.slots["A"].listing().draft["exactness"], 0.9)
        self.assertNotIn("Exactness was saved as", told(client))
        post = ScriptedPost(oa_tool(exact(BAKER)), oa_text("Saved."))
        s = self.session(OpenAIAgent(api_key="test", post=post))
        s.agent_message("A", GHENT)
        self.assertEqual(s.slots["A"].listing().draft["exactness"], UNSTRICT)


    def test_a_wish_any_save_of_the_turn_kept_as_ideally_is_remembered(self):
        """Told after its first save, the agent saves again in the same turn with the wish as ideally itself: it was
        still told to ask, so their yes in the next message counts."""
        unsaid, ideally = flat(must("pets", "no pets")), flat({**det("pets", "no pets", about="them"), "required": False})
        yes = flat(must("pets", "no pets", words="Yes, it has to be"))
        client = ScriptedClient(reply(tool(unsaid)), reply(tool(ideally)), reply(text("Must they have no pets?")),
                                reply(tool(yes)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", PORTO)
        mine = s.slots["A"].listing()
        self.assertEqual(mine.ideally_since, {"them:pets": 1})
        s.agent_message("A", "Yes, it has to be.")
        self.assertTrue(rows_of(mine.draft)["them:pets"]["required"])

    def test_which_wishes_were_kept_as_ideally_stays_on_their_own_listing(self):
        """Remembered on the listing only: never on the draft, so never on the phone or anywhere else."""
        client = ScriptedClient(reply(tool(flat(must("pets", "no pets")))), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", PORTO)
        mine = s.slots["A"].listing()
        self.assertEqual(mine.ideally_since, {"them:pets": 1})
        self.assertNotIn(KEPT_IDEALLY, mine.draft)
        for state in (s.phone_state("A"), s.room_state()):
            self.assertNotIn(KEPT_IDEALLY, str(state))
            self.assertNotIn("ideally_since", str(state))

    def test_a_yes_to_the_agents_question_makes_it_a_must(self):
        """Told a wish was kept as ideally, the agent asks whether it must be met; their yes, quoted, makes it one."""
        unsaid = {**HIKER, "details": [must("transport", "has a car")]}
        yes = {**HIKER, "details": [must("transport", "has a car", words="Yes, it has to be")]}
        client = ScriptedClient(reply(tool(unsaid)), reply(text("Saved. Must they have a car?")),
                                reply(tool(yes)), reply(text("Saved.")), reply(tool(unsaid)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", HIKER["intent_text"])
        self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])
        self.assertIn(WISH_IDEALLY.format(name="transport (about them)"), told(client))
        self.assertIn("save it as a must with their_words copied from their answer", WISH_IDEALLY)
        self.assertIn("their yes keeps it", agent_llm.SYSTEM_PROMPT)
        s.agent_message("A", "Yes, it has to be.")
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])
        # From then on it is a must on their draft: a later save that keeps it as it was, quoting nothing, keeps it.
        s.agent_message("A", "Early starts are fine.")
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])
        self.assertNotIn("was saved as ideally", told(client, 5))

    def test_both_counts_the_request_it_answers(self):
        """Asked "a date, or a partner for that? Or both?", their "both" makes two listings, each from the request it
        answered: a must they typed in it stays on both."""
        said = "I'd like to meet a woman in Bergen who paddles; she must have her own kayak."
        place = [("city", "Bergen", SHOWN), ("country", "Norway", SHOWN)]
        date = args(kind="dating", intent=said, fields=place + [("wants_gender", ["woman"], SHOWN)],
                    details=[must("boat", "has a kayak", "own kayak")])
        paddle = {**args(intent="Looking for someone to paddle with around Bergen.",
                         fields=place + [("seeking", "a kayaking partner", SHOWN), ("offering", "a kayaking partner", SHOWN),
                                         ("headline", "Kayaking around Bergen", SHOWN)],
                         details=[must("boat", "has a kayak", "own kayak")]),
                  "separate_listing": True}
        client = ScriptedClient(reply(text("A date, or a kayaking partner? Or both?")), reply(text("Not yet.")),
                                reply(tool(date, "t1"), tool(paddle, "t2")), reply(text("Saved both.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", said)
        s.agent_message("A", "both")
        first, second = s.slots["A"].listings
        self.assertEqual((first.draft["kind"], second.draft["kind"]), ("dating", "other"))
        self.assertTrue(rows_of(first.draft)["them:boat"]["required"])
        self.assertTrue(rows_of(second.draft)["them:boat"]["required"])
        self.assertNotIn("was saved as ideally", told(client, 3))
        self.assertEqual(DemoSession._said(second), [said, "both"])

    def test_a_new_listing_counts_what_they_typed_for_it_from_the_message_that_led_to_the_others_last_save(self):
        """A separate request that takes two messages: both are its own, as is the message that led to the first
        listing's last save (one message can ask for both), and nothing typed for the first listing before that is."""
        sail = {**SAILOR, "intent_text": "Looking for crew for my dinghy in Cascais, Saturdays only.",
                "details": SAILOR["details"] + [det("sail day", "saturday", about="both")]}
        bake = {**BAKER, "intent_text": "Looking for a sourdough baking partner in Cascais.", "separate_listing": True,
                "fields": [f if f["key"] not in ("city", "country") else
                           {**f, "values": ["Cascais" if f["key"] == "city" else "Portugal"]} for f in BAKER["fields"]],
                "details": [det("oven type", "wood-fired", about="them"), det("sail day", "saturday", about="both")]}
        client = ScriptedClient(reply(text("How many crew?")), reply(text("Not yet.")),
                                reply(tool(sail)), reply(text("Saved.")),
                                reply(text("Happy to. Where would you bake?")), reply(text("Not yet.")),
                                reply(tool(bake)), reply(text("Started.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", "Looking for crew for my dinghy in Cascais, Saturdays only.")
        s.agent_message("A", "Two of them.")  # this one led to its save
        s.agent_message("A", "Could you also make another listing: someone to bake with, they need a wood-fired oven?")
        s.agent_message("A", "In Cascais too.")
        first, second = s.slots["A"].listings
        self.assertTrue(rows_of(first.draft)["both:sail day"]["required"])
        self.assertTrue(rows_of(second.draft)["them:oven type"]["required"])  # typed for it, a message before
        self.assertFalse(rows_of(second.draft)["both:sail day"]["required"])  # typed only for the first listing
        self.assertIn(WISH_IDEALLY.format(name="sail day (about both)"), told(client, 7))
        self.assertNotIn("oven type (about them)' was saved as ideally", told(client, 7))
        self.assertEqual(DemoSession._said(second), ["Two of them.", "Could you also make another listing: someone to "
                                                     "bake with, they need a wood-fired oven?", "In Cascais too."])

    def test_a_must_typed_for_a_new_listing_in_the_message_that_saved_the_other_stays(self):
        """One message asks for two things; the agent saves the first listing from it at once and the second a message
        later: the must typed for the second in that first message is still theirs."""
        both = ("Looking for a sourdough baking partner in Ghent on Sundays. And separately, someone to sail with, "
                "they must have their own boat.")
        sail = {**args(intent="Someone to sail with near Ghent who has their own boat.",
                       fields=[("city", "Ghent", SHOWN), ("country", "Belgium", SHOWN),
                               ("seeking", "a sailing partner", SHOWN), ("offering", "a sailing partner", SHOWN),
                               ("headline", "Sailing together off Ghent", SHOWN)],
                       details=[must("boat", "own boat")]), "separate_listing": True}
        client = ScriptedClient(reply(tool(BAKER)), reply(text("Saved the baking one. Where would you like to sail?")),
                                reply(tool(sail)), reply(text("Started.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", both)
        s.agent_message("A", "On the river, near Ghent.")
        first, second = s.slots["A"].listings
        self.assertTrue(rows_of(second.draft)["them:boat"]["required"])
        self.assertNotIn("was saved as ideally", told(client, 3))
        self.assertEqual(DemoSession._said(second), [both, "On the river, near Ghent."])

    def test_a_separate_save_that_lands_on_this_listing_counts_all_of_its_chat(self):
        """A listing with no draft yet: a save marked separate goes on it, so everything typed in its chat counts."""
        call = {**BAKER, "separate_listing": True}
        client = ScriptedClient(reply(text("Which oven do you use?")), reply(text("Not yet.")),
                                reply(tool(call)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", GHENT + " They need a wood-fired oven.")
        s.agent_message("A", "Mornings are best.")
        self.assertEqual(len(s.slots["A"].listings), 1)
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:oven type"]["required"])
        self.assertNotIn("was saved as ideally", told(client, 3))

    def test_a_new_listing_counts_nothing_from_before_the_message_that_led_to_the_others_last_save(self):
        sail = {**SAILOR, "intent_text": "Looking for crew for my dinghy in Cascais, Saturdays only.",
                "details": SAILOR["details"] + [det("sail day", "saturday", about="both")]}
        bake = {**BAKER, "intent_text": "Looking for a sourdough baking partner in Cascais.", "separate_listing": True,
                "fields": [f if f["key"] not in ("city", "country") else
                           {**f, "values": ["Cascais" if f["key"] == "city" else "Portugal"]} for f in BAKER["fields"]],
                "details": [det("bake day", "saturday", about="both")]}
        client = ScriptedClient(reply(tool(sail)), reply(text("Saved.")), reply(tool(sail)), reply(text("Saved.")),
                                reply(tool(bake)), reply(text("Started.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", "Looking for crew for my dinghy in Cascais, Saturdays only.")
        s.agent_message("A", "Mornings are best.")
        s.agent_message("A", "Could you also make another listing: a sourdough baking partner in Cascais?")
        first, second = s.slots["A"].listings
        self.assertTrue(rows_of(first.draft)["both:sail day"]["required"])  # said for this one
        self.assertFalse(rows_of(second.draft)["both:bake day"]["required"])  # said only about the other, before
        self.assertIn(WISH_IDEALLY.format(name="bake day (about both)"), told(client, 5))
        self.assertEqual([m["text"] for m in second.chat if m["from"] == "me"],
                         ["Could you also make another listing: a sourdough baking partner in Cascais?"])
        self.assertEqual(DemoSession._said(second), ["Mornings are best.", "Could you also make another listing: a "
                                                     "sourdough baking partner in Cascais?"])
        self.assertEqual(second.ideally_since, {"both:bake day": 2})  # kept so on it, as of its own two messages

    def test_a_wish_left_out_after_going_live_isnt_pointed_out(self):
        """Wishes are always shown, so nobody chose anything about one: a save that leaves one out isn't asked to put it
        back. A detail about the person that they chose to keep hidden still is."""
        call = {**SAILOR, "details": [det("boat type", "dinghy", visibility=HIDDEN),
                                      det("crew size", "two", about="them")]}
        client = ScriptedClient(reply(tool(call)), reply(text("Saved.")),
                                reply(tool({**SAILOR, "details": []})), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", "Looking for crew for my dinghy in Cascais, two of them.")
        mine = s.slots["A"].listing()
        s.submit("A", fill(mine.draft))
        self.assertTrue(mine.entry["accepted"], mine.entry)
        self.assertNotIn("them:crew size", mine.choices)
        self.assertEqual(mine.choices["me:boat type"], HIDDEN)
        s.agent_message("A", "Mornings are best.")
        result = told(client, 3)
        self.assertIn("'boat type (about me)' was left out, and the person chose to keep it hidden.", result)
        self.assertNotIn("crew size (about them)' was left out", result)

    def test_a_turn_after_a_change_on_the_profile_keeps_every_wish_as_it_was(self):
        """keep_edit keeps every wish as the person kept it, required or not, and adds none, so the rule has nothing to
        do there: a must on the draft stays a must, and the agent's save isn't refused over it."""
        call = {**SAILOR, "details": SAILOR["details"] + [det("boat type", "dinghy", about="them")]}
        renamed = {**SAILOR, "intent_text": "Looking for crew for my catamaran in Cascais.",
                   "fields": [f if f["key"] != "headline" else {**f, "values": ["Sailing a catamaran off Cascais"]}
                              for f in SAILOR["fields"]],
                   "details": [det("boat type", "catamaran"), det("boat type", "catamaran", about="them")]}
        client = ScriptedClient(reply(tool(renamed)), reply(text("I've updated the rest to match.")))
        s = self.session(ClaudeAgent(client=client))
        mine = s.slots["A"].listing(s.new_listing("A"))
        mine.draft, problems, _ = build_draft(call, "Noor", s.slots["A"].person.age_band)
        self.assertEqual(problems, [])
        s.keep_change("A", kept(mine.draft, {"me:boat type": ["catamaran"]}))
        self.assertTrue(told(client).startswith("Draft saved."))  # not refused
        wishes = rows_of(mine.draft)
        self.assertEqual((wishes["them:boat type"]["value"], wishes["them:boat type"]["required"]), (["catamaran"], True))

    def test_an_agent_of_its_own_is_held_to_it_too(self):
        """An agent that doesn't take what they typed: its draft is held to the rule before it is kept."""
        class OwnAgent:
            def respond(self, conversation, text, name, band, current, in_use=None):
                draft, problems, _ = build_draft(BAKER, name, band, current, in_use)
                assert not problems, problems
                return "Saved.", draft

        s = self.session(OwnAgent())
        s.agent_message("A", GHENT)
        wishes = rows_of(s.slots["A"].listing().draft)
        self.assertEqual((wishes["both:bake day"]["required"], wishes["them:oven type"]["required"]), (True, False))

    def test_an_agent_of_its_own_is_made_less_exact_too(self):
        class OwnAgent:
            def respond(self, conversation, text, name, band, current, in_use=None):
                draft, problems, _ = build_draft({**BAKER, "exactness": 0.9}, name, band, current, in_use)
                assert not problems, problems
                return "Saved.", draft

        s = self.session(OwnAgent())
        s.agent_message("A", GHENT)
        self.assertEqual(s.slots["A"].listing().draft["exactness"], UNSTRICT)
        s = self.session(OwnAgent())
        s.agent_message("A", GHENT + " They need a wood-fired oven.")
        self.assertEqual(s.slots["A"].listing().draft["exactness"], 0.9)

    def test_what_the_rule_based_agent_keeps_from_before_stays_as_it_was(self):
        """A model agent's save was held to the rule when it was kept; once it becomes unavailable, the rule-based
        agent keeps its wishes as they are, a must it didn't say in this message included."""
        from belong.agent_llm import AgentUnavailable

        class Gone:
            def respond(self, *a, **kw):
                raise AgentUnavailable("no key")

        call = {**BAKER, "details": [det("bake day", "sunday", about="both"), must("oven type", "wood-fired")]}
        client = ScriptedClient(reply(tool(call)), reply(text("Saved.")))
        s = self.session(ClaudeAgent(client=client))
        s.agent_message("A", GHENT + " We bake in a wood-fired oven, so they need one too.")
        mine = s.slots["A"].listing()
        self.assertTrue(rows_of(mine.draft)["them:oven type"]["required"])
        s.llm = Gone()
        s.agent_message("A", "Mornings suit me.", mine.id)
        wishes = rows_of(mine.draft)
        self.assertEqual((wishes["both:bake day"]["required"], wishes["them:oven type"]["required"]), (True, True))

    def test_a_synthetic_person_taken_over_keeps_their_musts(self):
        """Their listing went live with their musts. Taken over, a message to either agent keeps them, and what they
        typed (their description and their words for each must, as the simulation counts them) is theirs."""
        def taken(llm):
            s = DemoSession(seed=7, agent=llm)
            s.run_inline = True
            pid = next(p for p in s.personas if p in s.board and s.board.listing(p).owner == p
                       and s.personas[p].person.verified and s.personas[p].said
                       and any(f.about in agent.WISHES and f.required for f in s.board.listing(p).fields))
            return s, s.take_over(pid), s.personas[pid]

        s, slot, persona = taken(None)
        mine = s.slots[slot].listing()
        musts = {f["key"] for f in mine.draft["fields"] if is_detail_key(f["key"])
                 and split_key(f["key"])[0] in agent.WISHES and f["required"]}
        self.assertTrue(musts)
        self.assertEqual(s._said(mine), [persona.intent_text, *persona.said])
        s.agent_message(slot, "Evenings suit me best.")
        self.assertEqual({k for k in musts if rows_of(mine.draft)[k]["required"]}, musts)
        s.submit(slot, fill(mine.draft))
        self.assertTrue(all(s.board.listing(mine.id).get(k).required for k in musts))
        # A model agent that saves them again as they are, quoting nothing: they stay musts.
        s, slot, persona = taken(None)
        mine = s.slots[slot].listing()
        again = {"kind": mine.draft.get("kind") or "other", "exactness": mine.draft["exactness"],
                 "intent_text": "Evenings suit me best.", "suggestions": [], "separate_listing": False, "fields": [],
                 "details": [{"name": f["name"], "about": f["about"], "compare": f["compare"], "steps": f["steps"],
                              "values": f["value"], "visibility": f["visibility"], "required": f["required"],
                              "their_words": ""} for f in mine.draft["fields"] if is_detail_key(f["key"])]}
        client = ScriptedClient(reply(tool(again)), reply(text("Saved.")))
        s.llm = ClaudeAgent(client=client)
        s.agent_message(slot, "Evenings suit me best.")
        self.assertNotIn("was saved as ideally", told(client))
        self.assertTrue(all(rows_of(mine.draft)[k]["required"] for k in musts), rows_of(mine.draft))


class EndToEnd(unittest.TestCase):
    """A must the person's agent made up never shuts out a match they'd want; without the rule it does."""

    BAND = args(intent="Our band in Leeds needs a bass player.",
                fields=[("city", "Leeds", SHOWN), ("country", "United Kingdom", SHOWN),
                        ("seeking", "a bass player", SHOWN), ("offering", "a band", SHOWN),
                        ("headline", "Finding a bass player for our band", SHOWN)])
    BASSIST = args(intent="I play bass and want to join a band in Leeds.",
                   fields=[("city", "Leeds", SHOWN), ("country", "United Kingdom", SHOWN),
                           ("seeking", "a band", SHOWN), ("offering", "a bass player", SHOWN),
                           ("headline", "Joining a band as a bass player", SHOWN)],
                   details=[det("rehearsal space", "own studio", about="them")])  # never said: made up

    def pair(self, matchmaker="rules"):
        client = ScriptedClient(reply(tool(self.BAND)), reply(text("Saved.")),
                                reply(tool(self.BASSIST)), reply(text("Saved.")))
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client), matchmaker=matchmaker)
        s.run_inline = True
        for slot, name, age, said in (("A", "Priya", 34, self.BAND["intent_text"]),
                                      ("B", "Callum", 29, self.BASSIST["intent_text"])):
            s.verify(slot, name, age)
            s.agent_message(slot, said)
            s.submit(slot, fill(s.slots[slot].draft))
            self.assertTrue(s.slots[slot].listing().entry["accepted"], s.slots[slot].listing().entry)
        for _ in range(3):
            s.run_matchmaker(quiet=True)
        introduced = {"demoA-1", "demoB-1"} in [{i.person_a, i.person_b} for i in s.intros.values()]
        return s, introduced

    def test_a_must_their_agent_made_up_never_shuts_out_a_match(self):
        s, introduced = self.pair()
        self.assertFalse(s.board.listing("demoB-1").get("them:rehearsal space").required)
        self.assertTrue(introduced)
        self.assertTrue(s.room_state()["leak_report"]["ok"], s.room_state()["leak_report"]["findings"])

    def test_without_the_rule_it_does(self):
        """Remove the rule (every wish counts as said) and the made-up must shuts the pair out."""
        with mock.patch.object(agent, "wish_said", lambda row, words, their_words=None: True):
            s, introduced = self.pair()
        self.assertTrue(s.board.listing("demoB-1").get("them:rehearsal space").required)
        self.assertFalse(introduced)

    def test_a_must_they_confirm_when_asked_still_shuts_out_who_doesnt_meet_it(self):
        """A must the person said in other words than the value their agent writes: its first save keeps it as
        ideally, their agent asks, and their yes, quoted, makes it a must, so a match that doesn't show it isn't
        introduced. Without their answer counting (values alone), it would stay ideally, and they would be."""
        said = "Our band in Leeds needs a bass player. They need their own van to get to gigs."
        asked = {**self.BAND, "intent_text": said, "details": [must("transport", "has a van")]}
        yes = {**asked, "details": [must("transport", "has a van", words="Yes, that's a must")]}
        bassist = {**self.BASSIST, "details": []}

        def pair():
            client = ScriptedClient(reply(tool(asked)), reply(text("Must they have their own van?")),
                                    reply(tool(yes)), reply(text("Saved.")), reply(tool(bassist)), reply(text("Saved.")))
            s = DemoSession(seed=7, agent=ClaudeAgent(client=client), matchmaker="rules")
            s.run_inline = True
            for slot, name, age, texts in (("A", "Priya", 34, (said, "Yes, that's a must.")),
                                           ("B", "Callum", 29, (bassist["intent_text"],))):
                s.verify(slot, name, age)
                for text_ in texts:
                    s.agent_message(slot, text_)
                s.submit(slot, fill(s.slots[slot].draft))
            for _ in range(3):
                s.run_matchmaker(quiet=True)
            return s, {"demoA-1", "demoB-1"} in [{i.person_a, i.person_b} for i in s.intros.values()]

        s, introduced = pair()
        self.assertTrue(s.board.listing("demoA-1").get("them:transport").required)
        self.assertFalse(introduced)
        with mock.patch.object(agent, "answered", lambda their_words, texts: False):  # values alone
            s, introduced = pair()
        self.assertFalse(s.board.listing("demoA-1").get("them:transport").required)
        self.assertTrue(introduced)

    def test_a_wish_their_agent_made_up_never_shuts_out_a_match_on_a_listing_that_exact(self):
        """Their agent also chose exactness 0.9, which they never asked for: the made-up wish, kept as ideally, would
        still shut out everyone who doesn't show it, so the listing is made less exact and the pair is introduced.
        Without that part of the rule it would not be."""
        bassist = {**self.BASSIST, "exactness": 0.9}
        with mock.patch.object(EndToEnd, "BASSIST", bassist):
            s, introduced = self.pair()
            self.assertEqual(s.board.listing("demoB-1").exactness, UNSTRICT)
            self.assertTrue(introduced)
            with mock.patch.object(agent_llm, "ground_exactness", lambda exactness, *a, **kw: (exactness, [])):
                s, introduced = self.pair()
        self.assertEqual(s.board.listing("demoB-1").exactness, 0.9)
        self.assertFalse(s.board.listing("demoB-1").get("them:rehearsal space").required)
        self.assertFalse(introduced)

    def test_a_number_they_typed_never_keeps_a_made_up_must_with_a_unit(self):
        """The number alone was typed, about something else: a must with that number and a unit is theirs only with
        the unit too, so it is kept as ideally and the pair is introduced."""
        band = {**self.BAND, "intent_text": "Our band in Leeds needs a bass player, we've played 50 gigs.",
                "details": [must("travel distance", "50mi")]}  # made up: never said
        with mock.patch.object(EndToEnd, "BAND", band):
            s, introduced = self.pair()
        self.assertFalse(s.board.listing("demoA-1").get("them:travel distance").required)
        self.assertTrue(introduced)

    def test_the_matchmaker_ai_is_sent_it_as_not_required(self):
        sent = []

        def judge(request):
            sent.append(copy.deepcopy(request))
            return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.9}
                              for c in request["candidates"]], "hunts_one_person": []}

        s, introduced = self.pair(matchmaker=judge)
        details = [d for r in sent for e in r["listings"].values() for d in e.get("details", [])
                   if d["name"] == "rehearsal space"]
        self.assertTrue(details)
        self.assertEqual({d["required"] for d in details}, {False})
        self.assertTrue(introduced)



class WithoutEachPart(unittest.TestCase):
    """Remove one part of the rule at a time, and what it does is undone: each part is needed."""

    def session(self, *responses):
        client = ScriptedClient(*responses)
        s = DemoSession(seed=7, agent=ClaudeAgent(client=client))
        s.run_inline = True
        s.verify("A", "Noor", 36)
        return s, client

    def yes_flow(self):
        """The agent asks about a must it saved in the common form alone; they say yes; a later save keeps it as it is."""
        unsaid = {**HIKER, "details": [must("transport", "has a car")]}
        yes = {**HIKER, "details": [must("transport", "has a car", words="Yes, it has to be")]}
        s, _ = self.session(reply(tool(unsaid)), reply(text("Must they have a car?")), reply(tool(yes)),
                            reply(text("Saved.")), reply(tool(unsaid)), reply(text("Saved.")))
        out = []
        for said in (HIKER["intent_text"], "Yes, it has to be.", "Early starts are fine."):
            s.agent_message("A", said)
            out.append(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])
        return out

    def test_the_whole_rule(self):
        self.assertEqual(self.yes_flow(), [False, True, True])

    def test_without_reading_values_a_must_in_their_words_is_lost(self):
        s, _ = self.session(reply(tool(HIKER)), reply(text("Saved.")))
        with mock.patch.object(agent, "wish_said", lambda row, words: False):
            s.agent_message("A", INNSBRUCK)
        self.assertFalse(rows_of(s.slots["A"].listing().draft)["them:transport"]["required"])

    def test_without_their_answer_a_yes_keeps_nothing(self):
        with mock.patch.object(agent, "answered", lambda their_words, texts: False):
            self.assertEqual(self.yes_flow(), [False, False, False])

    def test_without_remembering_which_wish_was_kept_as_ideally_a_yes_keeps_nothing(self):
        with mock.patch.object(DemoSession, "_note_ideally", staticmethod(lambda listing, keys, n: None)):
            self.assertEqual(self.yes_flow(), [False, False, False])

    def test_without_keeping_a_must_the_draft_had_a_later_save_loses_it(self):
        with mock.patch.object(agent, "_as_wide", lambda f, was: False):
            self.assertEqual(self.yes_flow(), [False, True, False])

    def test_without_their_answer_being_typed_after_the_save_any_quote_keeps_a_made_up_must(self):
        """Count a quote of anything they typed, as if every message were an answer: the made-up must stays."""
        def every_message_an_answer(rows, words, their_words=None, before=None, answers=None):
            everything = [" ".join(sorted(words))]
            return ground_wishes(rows, words, their_words, before, {key: everything for key in their_words or {}})

        s, _ = self.session(reply(tool(flat(must("pets", "no pets", words="flatmate")))), reply(text("Saved.")))
        with mock.patch.object(agent_llm, "ground_wishes", every_message_an_answer):
            s.agent_message("A", PORTO)
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:pets"]["required"])

    def test_without_their_answer_being_their_next_message_later_words_keep_a_made_up_must(self):
        """Count every message typed after the save as their answer, and a word typed after their no keeps it."""
        def any_message(their_words, texts):
            if not isinstance(their_words, str) or not 0 < len(their_words.split()) <= THEIR_WORDS_MAX:
                return False
            quote = agent._wish_words(their_words)
            return any(all(agent._word_said(w, said_words([t])) for w in quote) for t in texts or ())

        s, _ = self.session(reply(tool(flat(must("pets", "no pets")))), reply(text("Must they have no pets?")),
                            reply(text("Noted.")), reply(text("Not yet.")),
                            reply(tool(flat(must("pets", "no pets", words="Cheers")))), reply(text("Saved.")))
        with mock.patch.object(agent, "answered", any_message):
            for said in (PORTO, "No, that's fine either way.", "Cheers!"):
                s.agent_message("A", said)
        self.assertTrue(rows_of(s.slots["A"].listing().draft)["them:pets"]["required"])

    def test_without_a_number_and_its_unit_as_one_word_a_bare_number_keeps_a_made_up_must(self):
        """Split the unit off the number (as round 2 did) and a number typed about something else keeps the must."""
        split = lambda text: agent._UNIT.sub(" ", agent._THOUSANDS.sub("", agent._nfc(text)))  # noqa: E731
        call = flat(must("lease", "6mo"))
        for patched, required in ((False, False), (True, True)):
            s, _ = self.session(reply(tool(call)), reply(text("Saved.")))
            with mock.patch.object(agent, "_numbered", split if patched else agent._numbered):
                s.agent_message("A", PORTO + " I can move in from the 6th.")
            self.assertEqual(rows_of(s.slots["A"].listing().draft)["them:lease"]["required"], required)

    def test_without_the_message_that_led_to_the_last_save_a_new_listing_loses_a_must_typed_for_it(self):
        def since_saved(listing):  # only the messages after the other listing's last save
            last = max((i for i, m in enumerate(listing.chat) if m.get("draft_updated")), default=-1)
            return [m["text"] for m in listing.chat[last + 1:] if m.get("from") == "me"]

        sail = {**args(intent="Someone to sail with near Ghent who has their own boat.",
                       fields=[("city", "Ghent", SHOWN), ("country", "Belgium", SHOWN),
                               ("seeking", "a sailing partner", SHOWN), ("offering", "a sailing partner", SHOWN),
                               ("headline", "Sailing together off Ghent", SHOWN)],
                       details=[must("boat", "own boat")]), "separate_listing": True}
        s, _ = self.session(reply(tool(BAKER)), reply(text("Where would you like to sail?")),
                            reply(tool(sail)), reply(text("Started.")))
        with mock.patch.object(DemoSession, "_said_since", staticmethod(since_saved)):
            s.agent_message("A", GHENT + " And separately, someone to sail with, they must have their own boat.")
            s.agent_message("A", "On the river, near Ghent.")
        self.assertFalse(rows_of(s.slots["A"].listings[1].draft)["them:boat"]["required"])


if __name__ == "__main__":
    unittest.main()
