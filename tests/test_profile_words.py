"""The phone's profile in the person's own words (demo.html; SPEC §4 shown and hidden fields, open details, §6): every
shown detail reads once, as a sentence or a labelled fact, and a hidden value never changes it (dating's who and ages
follow what matching reads, lookingFor). Wording that depends on a side is used only when shown details say the side.
An open detail reads as a sentence from its name and whose it is, whatever name its agent chose. Run with node."""
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
             "const detailParts", "const GENDER_WORDS",
             "const spanOf", "const agesText", "function wantsText", "const CLIENT_WORDS", "const FILLER", "const words",
             "function sides", "function draftValues", "function layoutFor", "function lookingFor", "function checklist",
             "function canSave", "function progressOf",
             "const IN_WORDS", "const andList", "const orList", "function detailSentence", "function profileWords")
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


def made(kind, details=(), **values):
    """A hand-made draft: each value is shown, unless given as (value, HIDDEN). details: open details, as
    agent.detail_row makes them."""
    fields = [{"key": "first_name", "value": "Noor", "visibility": "shown", "enabled": True, "label": "First name"},
              {"key": "kind", "value": kind, "visibility": "shown", "enabled": True, "label": "Kind of match"}]
    for k, v in values.items():
        v, vis = v if isinstance(v, tuple) else (v, "shown")
        fields.append({"key": k, "value": v, "visibility": vis, "enabled": True, "label": agent.LABELS.get(k, k)})
    return [{"draft": {"kind": kind, "fields": fields + list(details)}}, {"wants": None}]


def detail(about, name, values, compare="same", steps=(), hidden=False, required=True):
    return agent.detail_row(about, name, compare, steps, [values] if isinstance(values, str) else values,
                            HIDDEN if hidden else "shown", required)


_WORDS = []


def words():
    """The words the rule-based agent may write: the demo's own list of details in use (built once)."""
    if not _WORDS:
        from belong.demo import DemoSession
        _WORDS.append(agent.words_in_use(DemoSession(seed=7, agent=None).details_in_use()))
    return _WORDS[0]


def draft(message, hide=()):
    d = agent.draft_from_message(message, "Noor", "25-29", words=words())
    for f in d["fields"]:
        if f["key"] in hide and f.get("value"):
            f["visibility"] = HIDDEN
    return [{"draft": d}, {"wants": agent.dating_wants(d)}]


class InYourWords(unittest.TestCase):
    MESSAGES = [
        "I play bass and I'm looking for a band to join in Kyoto, weekday evenings.",
        "Need someone to fix my washing machine in Lisbon this week, can pay 60 euros.",
        "Looking to meet a woman in Tokyo, late 20s, who's into hiking and karaoke. I'm a man.",
        "I'm a chemistry tutor in Toronto, online, 40 dollars an hour.",
        "Looking for a climbing partner in Berlin, weekends, I'm a beginner and I don't smoke, into yoga.",
        "Looking for a language exchange partner in London, online works for me.",
    ]

    def test_every_shown_detail_reads_once_and_a_hidden_value_never_changes_it(self):
        cases = [draft(m) for m in self.MESSAGES]
        self.assertTrue(any(":" in f["key"] and f["value"] for c, _ in cases for f in c["draft"]["fields"]))  # open details too
        hide = {"city", "country", "gender", "pay", "wants_gender", "wants_age", "format", "seeking",
                *{f["key"] for c, _ in cases for f in c["draft"]["fields"] if ":" in f["key"]}}
        cases += [draft(m, hide=hide) for m in self.MESSAGES]
        cases += [made("other", pay=("15 an hour", HIDDEN), city="Accra", seeking="a woodwork student",
                       offering="a woodwork teacher",
                       details=[detail("me", "skill", "high", "steps", ("low", "middle", "high"), hidden=True),
                                detail("them", "skill", ["low", "middle"], "steps", ("low", "middle", "high")),
                                detail("both", "tools", "hand tools", hidden=True)])]
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
            unsaid = {"kind"} if c["draft"].get("kind") == "other" else set()  # "something else": the phrases say what
            for k in shown - LEAD - unsaid:  # every other shown detail, exactly once
                self.assertEqual(keys.count(k), 1, (message, k, out))
            self.assertLessEqual(set(keys) - {"wants_gender", "wants_age"}, shown, (message, out))  # only shown details

    def test_side_wording_only_when_the_shown_details_say_the_side(self):
        teacher = made("other", seeking="a violin student", offering="a violin teacher", pay="20 an hour", city="Accra",
                       country="Ghana")
        learner = made("other", seeking="a statistics tutor", offering="a statistics student", pay="up to 25", city="Quito")
        unsaid = made("other", pay="10 a session", city="Porto", details=[detail("both", "subject", "pottery")])
        hidden = made("other", seeking=("a pottery student", HIDDEN), offering=("a pottery teacher", HIDDEN),
                      pay="10 a session", city="Porto", details=[detail("both", "subject", "pottery")])
        both = made("other", seeking="a drummer", offering="a bass player", pay="we split the costs", city="Oslo")
        for case in (unsaid, hidden):
            case[0]["draft"]["suggestions"] = []  # a model agent's draft: only what's known, no blanks
        teacher, learner, unsaid, hidden, both = words_for([teacher, learner, unsaid, hidden, both])
        said = lambda o: " ".join(text(s) for s in o["sentences"])
        facts = lambda o: {f["key"] for f in o["facts"]}
        self.assertTrue(said(teacher).startswith("I'm a violin teacher in Accra, Ghana"), said(teacher))
        self.assertIn("My rate is 20 an hour", said(teacher))
        self.assertTrue(said(learner).startswith("I'm looking for a statistics tutor in Quito"), said(learner))
        self.assertIn("My budget is up to 25", said(learner))
        for o in (unsaid, hidden):  # the side isn't shown: no rate or budget
            self.assertNotRegex(said(o), r"looking for|offer|teach|learn|rate|budget")
            self.assertIn("pay", facts(o))
            self.assertIn("The subject is pottery", said(o))
        self.assertEqual(unsaid, hidden)  # hidden phrases choose nothing
        self.assertNotIn("budget", said(both))
        self.assertIn("pay", facts(both))

    def test_what_is_needed_and_missing_is_a_blank_to_fill_with_the_agent(self):
        nowhere, nobody = draft("Looking for a climbing partner, I'm into chess."), draft("Looking for someone special in Lima.")
        hidden_city = draft("Looking for a climbing partner in Lisbon.", hide={"city", "country"})
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
        """Dating is the first fact; "something else" says nothing (the phrases do), and a kind never stands in
        for who someone is looking for."""
        cases = [draft("Looking to meet a man in Lima, 30s."), draft("I'm a guitarist looking for a band in Kyoto."),
                 made("other", city="Kyoto")]
        cases[1][0]["draft"]["fields"] += [{"key": "seeking", "value": "a band to join", "visibility": "shown", "enabled": True},
                                           {"key": "kind", "value": "other", "visibility": "shown", "enabled": True}]
        dating, band, unsaid = words_for(cases)
        self.assertEqual((dating["facts"][0]["key"], dating["facts"][0]["text"]), ("kind", "Dating"))
        self.assertEqual([w["key"] for w in band["sentences"][0] if isinstance(w, dict)][0], "seeking")
        for o in (band, unsaid):
            self.assertNotIn("kind", [f["key"] for f in o["facts"]] + [w["key"] for s in o["sentences"] for w in s
                                                                        if isinstance(w, dict) and "key" in w])

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
        """No form-like "Label: value": pay follows "is", a way to meet is a preference, and an open detail reads
        from its name and whose it is, for any name its agent chose."""
        scale = ("quiet", "lively", "loud")
        cases = [made("other", seeking="a sailing teacher", offering="a sailing student", pay="negotiable", city="Split"),
                 made("other", format="Online", city="Lyon"), made("other", format="In person or online", city="Lyon"),
                 made("other", city="Lyon", details=[
                     detail("me", "noise tolerance", "lively", "steps", scale),
                     detail("me", "instruments", ["drums", "bass"]),
                     detail("me", "glasses", "reading"),
                     detail("them", "noise tolerance", ["lively", "loud"], "steps", scale),
                     detail("them", "genre", "any", required=False),
                     detail("both", "rehearsal day", ["Tuesday", "Thursday"])])]
        said = [" ".join(text(s) for s in o["sentences"]) for o in words_for(cases)]
        self.assertIn("My budget is negotiable", said[0])
        self.assertIn("I'd prefer to meet online", said[1])
        self.assertIn("I'm happy to meet in person or online", said[2])  # open to both: no preference stated
        for sentence in ("My noise tolerance is lively.", "My instruments are drums and bass.", "My glasses are reading.",
                         "I'd like their noise tolerance to be lively or loud.", "Ideally, I'd like their genre to be any genre.",
                         "The rehearsal day is Tuesday or Thursday."):
            self.assertIn(sentence[:-1], said[3])
        self.assertLess(said[3].index("My instruments"), said[3].index("The rehearsal day"))  # about you first,
        self.assertLess(said[3].index("The rehearsal day"), said[3].index("I'd like their"))  # then both, then them
        for s in said:
            self.assertNotIn(":", s)

    def test_a_model_agent_says_what_to_ask(self):
        """With its suggestions, the sentence has no blanks: it says only what's known, and the profile asks
        only what the agent suggests (what matching needs still counts towards saving)."""
        unknown = made("other")
        unknown[0]["draft"]["suggestions"] = [{"label": "Your instrument", "ask": "I play "}]
        known = made("other", seeking="a sailing teacher", offering="a sailing student", city="Split")
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
            case = made("other", seeking="a kite-surfing instructor", offering="a kite-surfing student")
            case[0]["draft"]["suggestions"] = [{"label": f"Thing {i}", "ask": "x"} for i in range(n)]
            cases.append(case)
        # rule-based: no suggestions
        cases.append(made("other", seeking="a kite-surfing instructor", offering="a kite-surfing student", city="Tarifa"))
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => progressOf(checklist(c, l)))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        progress = json.loads(out.stdout)
        self.assertEqual(progress[:3], [{"done": 1, "total": 2}] * 3)  # what it's for, not yet where
        self.assertEqual(progress[3], {"done": 2, "total": 2})  # who and where: ready

    def test_a_place_is_needed_only_to_meet_in_person(self):
        """Anyone who can meet online is matched with others online wherever they live (as build_draft counts it)."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        cases = [made("other", seeking="a calligraphy teacher", offering="a calligraphy student", format=f)
                 for f in ("Online", "In person or online", "In person")]
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => !!canSave(c, l))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), [True, True, False])

    def test_both_phrases_are_needed_and_the_missing_one_is_asked_for(self):
        """Matching compares who someone is looking for with what the other is to them, so a listing that isn't
        dating needs both: the checklist asks for the one that's missing, as the server counts it."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        cases = [made("other", seeking="a choir accompanist", city="Bergen"), made("other", offering="a pianist", city="Bergen"),
                 made("other", seeking="a choir accompanist", offering="a pianist", city="Bergen")]
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"""
console.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => [checklist(c, l)[0].label, checklist(c, l)[0].done, !!canSave(c, l)])));"""
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(json.loads(out.stdout), [["What you are to them", False, False],
                                                  ["Who you're looking for", False, False],
                                                  ["Who you're looking for", True, True]])
        from belong.demo import DemoSession
        self.assertEqual([DemoSession._ready(c["draft"]) for c, _ in cases], [False, False, True])

    def test_the_kinds_are_worded_as_the_agent_names_them(self):
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        out = subprocess.run([node, "-e", _js("const KIND_WORDS") + "\nconsole.log(JSON.stringify(KIND_WORDS))"],
                             capture_output=True, text=True)
        self.assertEqual(json.loads(out.stdout), agent._KIND_NAMES)


if __name__ == "__main__":
    unittest.main()
