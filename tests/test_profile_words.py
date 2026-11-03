"""The phone's profile in the person's own words (demo.html; SPEC §4 shown and hidden fields, §6): every shown detail reads once,
as a sentence or a labelled fact, and a hidden value never changes it (dating's who and ages follow what
matching reads, lookingFor). Wording that depends on a side is used only when shown details say the side.
Run with node."""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import tempfile
import unittest

from belong import agent
from belong.schema import HIDDEN
from tests.test_dating import _js

FUNCTIONS = ("const hasValue", "const cap", "const val", "const KIND_LABELS", "const KIND_WORDS", "const UNDERSTOOD",
             "const GENDER_WORDS",
             "const spanOf", "const agesText", "function wantsText", "const CLIENT_WORDS", "const FILLER", "const words",
             "function sides", "function draftValues", "function layoutFor", "function lookingFor", "function checklist",
             "function canSave", "function progressOf",
             "const IN_WORDS", "const andList", "function profileWords")
# Shown in the lead sentence (who, what, where; dating's who and ages as matching reads them) or not on the
# profile at all (name and age are in the greeting, the headline is the listing's title, and one side of a
# one-way request mirrors the other).
LEAD = {"first_name", "age_band", "city", "country", "seeking", "offering", "headline", "wants_gender", "wants_age"}


def words_for(cases):
    node = shutil.which("node")
    if not node:
        raise unittest.SkipTest("node isn't installed")
    code = "\n".join(_js(n) for n in FUNCTIONS)
    code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => profileWords(c, l))));"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
        f.write(code)
    out = subprocess.run([node, f.name], capture_output=True, text=True)
    if out.returncode:
        raise AssertionError(out.stderr)
    return json.loads(out.stdout)


def text(sentence):
    return "".join(w if isinstance(w, str) else w["text"] for w in sentence)


def without_hidden_values(case):
    """The same listing with every hidden value taken out (its rows stay: they're the kind's layout)."""
    case = copy.deepcopy(case)
    for f in case[0]["draft"]["fields"]:
        if f.get("visibility") == HIDDEN:
            f["value"] = None
    return case


def made(kind, **values):
    """A hand-made draft: each value is shown, unless given as (value, HIDDEN)."""
    fields = [{"key": "first_name", "value": "Noor", "visibility": "shown", "enabled": True, "label": "First name"},
              {"key": "kind", "value": kind, "visibility": "shown", "enabled": True, "label": "Kind of match"}]
    for k, v in values.items():
        v, vis = v if isinstance(v, tuple) else (v, "shown")
        fields.append({"key": k, "value": v, "visibility": vis, "enabled": True, "label": agent.LABELS.get(k, k)})
    return [{"draft": {"kind": kind, "fields": fields}}, {"wants": None}]


def draft(message, hide=()):
    d = agent.draft_from_message(message, "Noor", "25-29")
    for f in d["fields"]:
        if f["key"] in hide and f.get("value"):
            f["visibility"] = HIDDEN
    return [{"draft": d}, {"wants": agent.dating_wants(d)}]


class InYourWords(unittest.TestCase):
    MESSAGES = [
        "I play bass and I'm looking for a band to join in Kyoto, weekday evenings.",
        "Need someone to fix my washing machine in Lisbon this week, can pay 60 euros.",
        "Looking to meet a woman in Seoul, late 20s, who's into hiking and film. I'm a man.",
        "I'm a chemistry tutor in Toronto, online, 40 dollars an hour.",
        "Looking for a roommate in Berlin from March.",
        "Language exchange in Madrid: I speak Japanese and I'm learning Spanish.",
        "Looking for a running buddy in Nairobi, mornings, beginner pace.",
    ]

    def test_every_shown_detail_reads_once_and_a_hidden_value_never_changes_it(self):
        cases = [draft(m) for m in self.MESSAGES]
        cases += [draft(m, hide={"interests", "schedule", "city", "country", "gender", "pay", "role", "subject",
                                 "wants_gender", "wants_age", "move_in", "learning_language"}) for m in self.MESSAGES]
        cases += [made("tutoring", role=("tutor", HIDDEN), subject="woodwork", pay=("15 an hour", HIDDEN), city="Accra")]
        outs = words_for(cases + [without_hidden_values(c) for c in cases])
        for case, out, bare in zip(cases, outs, outs[len(cases):]):
            # The rule itself: with every hidden value taken out, the profile is exactly the same.
            self.assertEqual(out, bare, case[0]["draft"].get("intent_text"))
        for (c, _), out in zip(cases, outs):
            fields = c["draft"]["fields"]
            shown = {f["key"] for f in fields if f.get("enabled") and f.get("value") not in (None, "", [])
                     and f.get("visibility") != HIDDEN}
            parts = [w for s in out["sentences"] for w in s if isinstance(w, dict)] + out["facts"]
            keys = [w["key"] for w in parts if not w.get("blank")]
            message = c["draft"].get("intent_text")
            for k in shown - LEAD:  # every other shown detail, exactly once
                self.assertEqual(keys.count(k), 1, (message, k, out))
            self.assertLessEqual(set(keys) - {"wants_gender", "wants_age"}, shown, (message, out))  # only shown details

    def test_side_wording_only_when_the_shown_details_say_the_side(self):
        teacher = made("tutoring", role="tutor", subject="violin", pay="20 an hour", city="Accra", country="Ghana")
        learner = made("tutoring", role="student", subject="statistics", pay="up to 25", city="Quito")
        unsaid = made("tutoring", role=None, subject="pottery", pay="10 a session", city="Porto")
        hidden = made("tutoring", role=("tutor", HIDDEN), subject="pottery", pay="10 a session", city="Porto")
        both = made("other", seeking="a drummer", offering="a bass player", pay="we split the costs", city="Oslo")
        teacher, learner, unsaid, hidden, both = words_for([teacher, learner, unsaid, hidden, both])
        said = lambda o: " ".join(text(s) for s in o["sentences"])
        facts = lambda o: {f["key"] for f in o["facts"]}
        self.assertTrue(said(teacher).startswith("I offer tutoring in Accra, Ghana"), said(teacher))
        self.assertIn("I teach violin", said(teacher))
        self.assertIn("My rate is 20 an hour", said(teacher))
        self.assertTrue(said(learner).startswith("I'm looking for tutoring in Quito"), said(learner))
        self.assertIn("I'd like to learn statistics", said(learner))
        self.assertIn("My budget is up to 25", said(learner))
        for o in (unsaid, hidden):  # the side isn't shown: no teach or learn, no rate or budget
            self.assertNotRegex(said(o), r"looking for|offer|teach|learn|rate|budget")
            self.assertLessEqual({"kind", "subject", "pay"}, facts(o))
        self.assertEqual(unsaid, hidden)  # a hidden role chooses nothing
        self.assertNotIn("budget", said(both))
        self.assertIn("pay", facts(both))

    def test_what_is_needed_and_missing_is_a_blank_to_fill_with_the_agent(self):
        nowhere, nobody = draft("Looking for someone to play chess with."), draft("Looking for someone special in Lima.")
        hidden_city = draft("Looking for a cycling partner in Quito.", hide={"city", "country"})
        outs = words_for([nowhere, nobody, hidden_city])
        blanks = [[w["label"] for s in o["sentences"] for w in s if isinstance(w, dict) and w.get("blank")] for o in outs]
        self.assertIn("Where", blanks[0])
        self.assertIn("Who you'd like to meet", blanks[1])  # dating, but who isn't said
        self.assertIn("Where", blanks[2])  # a hidden place isn't on the profile: it never decides a match
        for o in outs:
            for s in o["sentences"]:
                for w in s:
                    if isinstance(w, dict) and w.get("blank"):
                        self.assertGreaterEqual(w["n"], 0, w)  # each blank asks the agent through the checklist

    def test_the_kind_of_match_is_on_the_profile_once(self):
        """As who you're looking for when the kind has no "looking for" of its own, else the first fact (none for
        "something else")."""
        cases = [draft("Looking for a roommate in Berlin from March."), draft("Looking to meet a man in Lima, 30s."),
                 draft("I'm a guitarist looking for a band in Kyoto.")]
        cases[2][0]["draft"]["fields"] += [{"key": "seeking", "value": "a band to join", "visibility": "shown", "enabled": True},
                                           {"key": "kind", "value": "other", "visibility": "shown", "enabled": True}]
        roommate, dating, band = words_for(cases)
        lead = [w for w in roommate["sentences"][0] if isinstance(w, dict)]
        self.assertEqual((lead[0]["key"], lead[0]["text"]), ("kind", "a roommate"))
        self.assertNotIn("kind", [f["key"] for f in roommate["facts"]])
        self.assertEqual(dating["facts"][0]["key"], "kind")
        self.assertEqual([w["key"] for w in band["sentences"][0] if isinstance(w, dict)][0], "seeking")
        self.assertNotIn("kind", [f["key"] for f in band["facts"]])  # "something else" says nothing: the phrases do

    def test_the_lead_is_their_own_words_not_the_matching_phrases(self):
        """A shown headline leads, in the first person when it starts with an -ing word; "looking for" and
        "you are" are written as [activity] [role] for matching and never read as the person's words."""
        boots = made("other", headline="Selling my old hiking boots", seeking="a hiking boots buyer",
                     offering="a hiking boots seller", city="Bergen")
        lessons = made("other", headline="Teaching beginner pottery", seeking="a pottery student",
                       offering="a pottery teacher", city="Cusco")
        wanted = made("other", headline="Photographer wanted for a small wedding", seeking="a wedding photographer",
                      offering="a wedding photography client", city="Hobart")
        hidden = made("other", headline=("Selling my old hiking boots", HIDDEN), seeking="a hiking boots buyer",
                      offering="a hiking boots seller", city="Bergen")
        said = [[text(s) for s in o["sentences"]] for o in words_for([boots, lessons, wanted, hidden])]
        self.assertEqual(said[0][0], "I'm selling my old hiking boots")
        self.assertEqual(said[1][0], "I'm teaching beginner pottery")
        self.assertEqual(said[2][0], "Photographer wanted for a small wedding")
        for s in said[:3]:
            self.assertNotRegex(" ".join(s), r"buyer|seller|student|client")
        self.assertNotIn("Selling my old", " ".join(said[3]))  # a hidden headline never leads

    def test_the_place_is_its_own_sentence_said_once(self):
        """The headline's grammar is the agent's (a phrase, or clauses), so the place is never joined onto it;
        a headline that ends on the shown place drops it there."""
        clauses = made("other", headline="Building a treehouse this spring, could use a carpenter", city="Tartu",
                       country="Estonia")
        kayak = made("other", headline="Selling a sea kayak in Tromsø", city="Tromsø", country="Norway")
        room = made("other", headline="Renting out a room near the Tromsø harbour", city="Tromsø", country="Norway")
        nowhere = made("other", headline="Looking for a chess partner")
        out = words_for([clauses, kayak, room, nowhere])
        said = [[text(s) for s in o["sentences"]] for o in out]
        self.assertEqual(said[0][:2], ["I'm building a treehouse this spring, could use a carpenter", "I'm based in Tartu, Estonia"])
        self.assertEqual(said[1][:2], ["I'm selling a sea kayak", "I'm based in Tromsø, Norway"])
        self.assertEqual(said[2][:2], ["I'm renting out a room near the Tromsø harbour", "I'm based in Tromsø, Norway"])
        self.assertEqual([[w["key"] for w in s if isinstance(w, dict)] for s in out[1]["sentences"][:2]], [["headline"], ["city"]])
        where = [w for w in out[3]["sentences"][1] if isinstance(w, dict)]
        self.assertEqual((where[0].get("blank"), where[0]["label"]), (True, "Where"))  # still asked for

    def test_every_detail_reads_as_a_person_says_it(self):
        """No form-like "Label: value": each fixed choice of when reads as a sentence, and pay follows "is"."""
        cases = [made("running", schedule=w, city="Daegu") for w in ("weekends", "weekday evenings", "flexible")]
        cases.append(made("tutoring", role="tutor", subject="cello", schedule="weekday mornings", city="Lyon"))
        cases.append(made("tutoring", role="student", subject="chemistry", pay="negotiable", city="Lyon"))
        cases += [made("other", format=f, city="Lyon") for f in ("Online", "In person or online")]
        said = [" ".join(text(s) for s in o["sentences"]) for o in words_for(cases)]
        self.assertIn("I'm free on weekends", said[0])
        self.assertIn("I'm free on weekday evenings", said[1])
        self.assertIn("I'm flexible on timing", said[2])
        self.assertIn("I'm available on weekday mornings", said[3])  # someone offering something
        self.assertIn("My budget is negotiable", said[4])
        self.assertIn("I'd prefer to meet online", said[5])
        self.assertIn("I'm happy to meet in person or online", said[6])  # open to both: no preference stated
        for s in said:
            self.assertNotIn(":", s)

    def test_a_model_agent_says_what_to_ask(self):
        """With its suggestions, the sentence has no blanks: it says only what's known, and the profile asks
        only what the agent suggests (what matching needs still counts towards saving)."""
        unknown = made("other")
        unknown[0]["draft"]["suggestions"] = [{"label": "Your instrument", "ask": "I play "}]
        known = made("other", seeking="a sailing teacher", city="Split")
        known[0]["draft"]["suggestions"] = []
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"""
const out = {json.dumps([unknown, known])}.map(([c, l]) => {{
  const w = profileWords(c, l), items = checklist(c, l);
  return {{ blanks: w.sentences.flat().filter(p => p && p.blank).length, said: w.sentences.map(s => s.map(p => typeof p === "string" ? p : p.text).join("")),
           asks: items.filter(i => !i.done && !i.quiet).map(i => i.label), save: !!canSave(c, l) }};
}});
console.log(JSON.stringify(out));"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        unknown, known = json.loads(out.stdout)
        self.assertEqual((unknown["blanks"], unknown["said"]), (0, []))  # nothing known: nothing to say yet
        self.assertEqual(unknown["asks"], ["Your instrument"])  # its suggestion, not a fixed "who" or "where"
        self.assertFalse(unknown["save"])  # matching still needs who
        self.assertEqual(known["said"][0], "I'm looking for a sailing teacher in Split")
        self.assertTrue(known["save"])

    def test_progress_counts_only_what_matching_needs(self):
        """The agent's suggestions are optional: more of them never move the finish line."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        cases = []
        for n in (0, 2, 6):
            case = made("other", seeking="a kite-surfing instructor")
            case[0]["draft"]["suggestions"] = [{"label": f"Thing {i}", "ask": "x"} for i in range(n)]
            cases.append(case)
        cases.append(made("other", seeking="a kite-surfing instructor", city="Tarifa"))  # rule-based: no suggestions
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => progressOf(checklist(c, l)))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        progress = json.loads(out.stdout)
        self.assertEqual(progress[:3], [{"done": 1, "total": 2}] * 3)  # who, not yet where
        self.assertEqual(progress[3], {"done": 2, "total": 2})  # who and where: ready

    def test_a_place_is_needed_only_to_meet_in_person(self):
        """Anyone who can meet online is matched with others online wherever they live (as build_draft counts it)."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        cases = [made("other", seeking="a calligraphy teacher", format=f) for f in ("Online", "In person or online", "In person")]
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => !!canSave(c, l))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), [True, True, False])

    def test_the_kinds_are_worded_as_the_agent_names_them(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        out = subprocess.run([node, "-e", _js("const KIND_WORDS") + "\nconsole.log(JSON.stringify(KIND_WORDS))"],
                             capture_output=True, text=True)
        self.assertEqual(json.loads(out.stdout), agent._KIND_NAMES)


if __name__ == "__main__":
    unittest.main()
