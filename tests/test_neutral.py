"""No early example decides what other requests get (CLAUDE.md, "How we fix things" and "Demo examples never define
the product"): a detail's label, the Warden's reason sentences and a question's answer are true for any listing that
can carry them; a detail goes only on the kinds it belongs to (agent.carries); open details are never filled in from
an earlier request; the phone and the connector offer Start matching by one rule; and no product code holds the
demo's examples: only the simulation and the demo's board of synthetic people read synth.py, no word of its
stories or hobbies is a label, an allowed value, an allow-list or a matching rule (read from the syntax tree, so
f-strings count on every Python version), and no two words of one of its fixed vocabularies (languages, levels,
paces, band roles) are written out together."""
from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from belong import agent, synth
from belong.agent_llm import build_draft
from belong.demo import DemoError, DemoSession
from belong.schema import HIDDEN, SHOWN, detail_key, mentions_sensitive
from belong.warden import reason_text
from tests.helpers import detail
from tests.test_demo import climbers

ROOT = Path(__file__).resolve().parent.parent / "belong"
# The demo's examples, from the synthetic people themselves (test data, never product code): what their stories are
# about, their hobbies, the music they play and the subjects they learn. Role words ("tutor", "partner") and
# relationship words ("roommate") are general and stay out of this list; so do words with an everyday sense the
# product needs ("running" as in a running service, "language" as in any language).
_STORIES = {"climbing", "climber", "bouldering", "belay", "jogging", "marathon", "running buddy", "running partner",
            "language exchange", "tandem"}
ACTIVITY_WORDS = _STORIES | {w.lower() for w in synth.INTERESTS + synth.GENRES + synth.SUBJECTS}
# Product code that may read synth.py, and the only names it may take: the simulation and its grading, the command
# that runs it, the demo's board of synthetic people, and the trust probe's sandbox of synthetic people.
MAY_IMPORT_SYNTH = {"synth.py": None, "sim.py": None, "evaluate.py": None, "__main__.py": {"DIALECT_SHARE"},
                    "demo.py": {"draft_listing", "generate"}, "probe.py": {"draft_listing", "generate"}}
# Words with an everyday use in a product rule, each where it's allowed and why (none today).
EVERYDAY: dict = {}
# The synthetic people's fixed vocabularies whose words each have an everyday sense ("easy", "advanced", "English",
# "singer"), so no single word can be banned: two or more of one written out together is the demo's vocabulary.
SEQUENCES = {"languages": synth.LANGUAGES, "levels": synth.LEVELS, "paces": synth.PACES,
             "band roles": synth.INSTRUMENTS}
# Lists that hold two of them for a reason of their own: where, which words exactly, and why. The owner reviews both.
SEQUENCE_REASONS = {
    ("matcher.py", "_UMBRELLA"): ({"beginner", "advanced"}, "umbrella words that say nothing about an activity (with "
                                  "matching's other word lists, reported to the owner)"),
}


def save(kind, *fields, details=(), current=None):
    """A model agent's save, as build_draft gets it."""
    args = {"kind": kind, "exactness": 0.6, "intent_text": "Looking for someone, in my own words.", "suggestions": [],
            "details": list(details),
            "fields": [{"key": k, "values": [v] if isinstance(v, str) else list(v), "visibility": SHOWN} for k, v in fields]}
    return build_draft(args, "Ines", "30-34", current)


def said(phrase: str, text: str) -> bool:
    return re.search(r"(?<![a-z])" + re.escape(phrase) + r"(?![a-z])", text.lower()) is not None


def product_modules():
    return sorted([*ROOT.glob("*.py"), *(ROOT / "matching").glob("*.py")])


def _docstrings(tree) -> set:
    """The ids of a module's docstring nodes (a module's, class's or function's first statement, a plain string)."""
    docs = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                docs.add(id(first.value))
    return docs


def _names(node) -> list:
    """The identifiers a node writes: a name, an attribute, an argument, a keyword, a definition or an import."""
    if isinstance(node, ast.Name):
        return [node.id]
    if isinstance(node, ast.Attribute):
        return [node.attr]
    if isinstance(node, ast.arg):
        return [node.arg]
    if isinstance(node, ast.keyword):
        return [node.arg] if node.arg else []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.alias):
        return [n for n in (node.name, node.asname) if n]
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return list(node.names)
    if isinstance(node, ast.ImportFrom):
        return [node.module] if node.module else []
    return []


def code_words(path: Path) -> list:
    """(line, text) for each string and name in a module's code: what it can decide with. Read from the syntax
    tree, so the text of an f-string (or a template string) counts like any other string on every Python version:
    the tokenizer splits f-strings into pieces of their own from 3.12 on. Docstrings and comments are
    explanations, not rules, so they're left out."""
    tree = ast.parse(path.read_text())
    docs = _docstrings(tree)
    out = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", 0)
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out.append((line, node.value))  # an f-string's literal parts are Constant nodes inside it
        out += [(line, name) for name in _names(node)]
    return out


def literal_sequences(path: Path) -> list:
    """(line, the name it's assigned to or "", strings) for each literal list, tuple, set or dict (its keys) of
    strings in a module's code, and each string on its own (outside docstrings): where a vocabulary would be
    written out."""
    tree = ast.parse(path.read_text())
    docs = _docstrings(tree)
    named = {id(node.value): target.id for node in ast.walk(tree) if isinstance(node, (ast.Assign, ast.AnnAssign))
             for target in (node.targets if isinstance(node, ast.Assign) else [node.target]) if isinstance(target, ast.Name)}
    out = []
    for node in ast.walk(tree):
        elts = node.keys if isinstance(node, ast.Dict) else node.elts if isinstance(node, (ast.List, ast.Tuple, ast.Set)) \
            else None
        if elts is not None:
            out.append((node.lineno, named.get(id(node), ""),
                        [e.value for e in elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docs:
            out.append((getattr(node, "lineno", 0), named.get(id(node), ""), [node.value]))
    return out


def page_code(name: str) -> str:
    """A web page's own scripts without their comments, and without the demo page's scenario presets (what a
    presenter can have a person type; they never decide anything)."""
    html = (ROOT / "web" / name).read_text()
    code = "\n".join(re.findall(r"<script>(.*?)</script>", html, flags=re.S))
    code = re.sub(r"const SCENARIOS = \{.*?\n\};\n", "", code, flags=re.S)
    return "\n".join(re.sub(r"(^|\s)//.*$", "", line) for line in code.splitlines())


class WordedForAnyListing(unittest.TestCase):
    def test_no_detail_label_names_an_activity(self):
        for key, label in agent.LABELS.items():
            self.assertFalse([w for w in ACTIVITY_WORDS if said(w, label)], (key, label))
        for kind, words in agent._KIND_NAMES.items():
            self.assertFalse([w for w in ACTIVITY_WORDS if said(w, words)], (kind, words))
        # An open detail is labelled by its own name, whatever name its agent chose.
        self.assertEqual(agent.label_for(detail_key("them", "noise tolerance")), "Noise tolerance")

    def test_the_wardens_reasons_name_no_activity(self):
        """The Warden's fixed sentences for open details carry only the name and values both people show."""
        scale = ("gentle", "steady", "brisk")
        mine = {f.key: f for f in (detail("tempo", "steady", compare="steps", steps=scale), detail("rehearsal day", "Tuesday"),
                                   detail("tempo", ("steady", "brisk"), about="them", compare="steps", steps=scale))}
        theirs = {f.key: f for f in (detail("tempo", "brisk", compare="steps", steps=scale),
                                     detail("rehearsal day", ("Tuesday", "Sunday")))}
        text = reason_text([("agree:rehearsal day", mine, theirs), ("agree:tempo", mine, theirs),
                            ("wants:tempo", mine, theirs)])
        self.assertFalse([w for w in ACTIVITY_WORDS if said(w, text)], text)
        for words in ("rehearsal day", "Tuesday", "brisk", "steady"):
            self.assertIn(words, text)
        self.assertNotIn("side", text)  # "They're the tutor side" only fitted one kind's roles


class EachDetailWhereItBelongs(unittest.TestCase):
    def test_who_and_what_ages_only_on_dating(self):
        draft, problems, notes = save("other", ("wants_gender", "woman"), ("seeking", "a sailing crew"),
                                      ("offering", "a sailor"), ("headline", "Looking for sailing crew"))
        self.assertEqual(problems, [])
        self.assertNotIn("wants_gender", {f["key"] for f in draft["fields"] if f["value"]})
        self.assertTrue(any(n.startswith("wants_gender was left out") for n in notes), notes)

    def test_an_open_detail_goes_on_either_kind(self):
        level = {"name": "skill", "about": "me", "compare": "steps", "steps": ["low", "middle", "high"],
                 "values": ["middle"], "visibility": "shown", "required": True}
        for kind, fields in (("other", [("seeking", "a go player"), ("offering", "a go player")]),
                             ("dating", [("wants_gender", "man")])):
            draft, problems, _ = save(kind, *fields, details=[level])
            self.assertEqual(problems, [], kind)
            row = next(f for f in draft["fields"] if f["key"] == "me:skill")
            self.assertEqual((row["value"], row["label"]), (["middle"], "Skill"))

    def test_another_request_keeps_only_whats_about_the_person(self):
        """Every model save states its open details in full, so another request (two "something else" listings,
        one after the other) never inherits the first one's (a level, a subject); built-ins about the person stay.
        The rule-based agent, which can't write every detail, keeps only those about the person."""
        before, problems, _ = save("other", ("city", "Graz"), ("seeking", "a sparring partner"), ("offering", "a sparring partner"),
                                   details=[{"name": "skill", "about": "me", "compare": "steps", "steps": ["low", "high"],
                                             "values": ["high"], "visibility": "shown", "required": True},
                                            {"name": "weight class", "about": "both", "compare": "same", "steps": [],
                                             "values": ["middleweight"], "visibility": "shown", "required": True}])
        self.assertEqual(problems, [])
        after, problems, _ = save("other", ("seeking", "a Welsh tutor"), ("offering", "a Welsh student"), current=before)
        self.assertEqual(problems, [])
        values = {f["key"]: f["value"] for f in after["fields"] if f["value"] not in (None, "", [])}
        self.assertEqual(values["city"], "Graz")
        self.assertFalse([k for k in values if ":" in k], values)  # never filled in from before
        offline = agent.draft_from_message("Looking for someone to learn Welsh with.", "Ines", "30-34")
        offline.update(kind="other", fields=offline["fields"] + [{"key": "seeking", "value": "a Welsh tutor"}])
        self.assertEqual(agent.keep_details(offline, before), ["me:skill"])  # not the old request's weight class


class AnswersHaveTheirValue(unittest.TestCase):
    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climbers(self.s)
        self.mine = self.s.slots["A"].listing()

    def ask(self, question_id):
        self.mine.followups.append({"question_id": question_id, "question_text": "?"})

    def test_a_question_says_what_its_answer_is(self):
        for q in ("age_range", "also_offer", "broaden"):
            self.ask(q)
        types = {q["question_id"]: q["answer"] for q in self.s.phone_state("A")["listings"][0]["followups"]}
        self.assertEqual(types, {"age_range": {"type": "age_range"}, "also_offer": {"type": "yes_no"},
                                 "broaden": {"type": "yes_no"}})

    def test_a_yes_never_stands_in_for_a_value(self):
        self.ask("age_range")
        before = self.s.board.listing(self.mine.id)
        with self.assertRaises(DemoError):
            self.s.answer_followup("A", "age_range", {"yes": True})
        self.assertEqual(self.s.board.listing(self.mine.id), before)  # nothing changed, still asked
        self.assertIn("age_range", [x["question_id"] for x in self.mine.followups])


class StartMatchingByOneRule(unittest.TestCase):
    def test_the_server_and_the_phone_agree(self):
        """The connector shows the server's "ready" (DemoSession._ready); the phone works it out itself
        (canSave). Both must say the same for the same draft."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        from tests.test_dating import _js
        from tests.test_profile_words import FUNCTIONS, words
        drafts = [agent.draft_from_message(m, "Ines", "30-34", words=words()) for m in (
            "Looking for a climbing partner in Osaka.", "Looking for a climbing partner.", "Looking to meet a man in Tokyo.",
            "Looking for someone special in Berlin.", "I review CVs, online only.", "hello")]
        drafts.append(save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("format", "Online"))[0])
        drafts.append(save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("format", "In person"))[0])
        drafts.append(save("other", ("seeking", "a sailing crew"), ("city", "Graz"))[0])  # what they are to them: missing
        drafts.append(save("other", ("offering", "a sailor"), ("city", "Graz"))[0])  # who they're looking for: missing
        hidden = save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("city", "Graz"))[0]
        for f in hidden["fields"]:
            if f["key"] == "city":
                f["visibility"] = HIDDEN  # a hidden place never decides
        drafts.append(hidden)
        drafts.append(save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("city", "Graz"))[0])
        cases = [[{"draft": d}, {"wants": agent.dating_wants(d), "listed": False}] for d in drafts]
        code = "\n".join(_js(n) for n in FUNCTIONS)
        code += f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(([c, l]) => !!canSave(c, l))));"
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(code)
        out = subprocess.run([node, f.name], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        phone = json.loads(out.stdout)
        server = [DemoSession._ready(d) for d in drafts]
        self.assertEqual(server, phone)
        self.assertEqual(server[-5:], [False, False, False, False, True])
        self.assertIn(True, server)
        self.assertIn(False, server)


class NoDemoExamplesInTheProduct(unittest.TestCase):
    def test_only_the_simulation_and_the_demos_board_read_synth(self):
        for path in product_modules():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[-1] == "synth":
                    names = {a.name for a in node.names}
                elif isinstance(node, ast.Import) and any(a.name.split(".")[-1] == "synth" for a in node.names):
                    names = {"*"}
                elif isinstance(node, ast.ImportFrom) and any(a.name == "synth" for a in node.names):
                    names = {"*"}
                else:
                    continue
                rel = path.relative_to(ROOT).as_posix()
                self.assertIn(rel, MAY_IMPORT_SYNTH, f"{rel} imports synth")
                allowed = MAY_IMPORT_SYNTH[rel]
                self.assertTrue(allowed is None or names <= allowed, f"{rel} takes {names - (allowed or set())} from synth")

    def test_no_example_is_a_rule_label_or_allowed_value(self):
        found = []
        for path in product_modules():
            if path.name in ("synth.py", "sim.py", "evaluate.py", "probe.py", "__main__.py"):
                continue  # the synthetic people, the simulation that grades them, and the probe's sandbox of them
            for line, text in code_words(path):
                found += [(path.name, line, w) for w in ACTIVITY_WORDS
                          if said(w, text) and (path.name, w) not in EVERYDAY]
        for page in sorted((ROOT / "web").glob("*.html")):
            if page.name == "index.html":
                continue  # the inspector shows the simulation's own people
            code = page_code(page.name)
            found += [(page.name, w) for w in ACTIVITY_WORDS if said(w, code)]
        self.assertEqual(found, [])

    def test_no_vocabulary_of_the_demo_is_written_out(self):
        """No product module or page holds two or more words of one of the synthetic people's fixed vocabularies
        together (a list, a set, a dict's keys, or one string such as a prompt or a pattern): a language list, the
        demo's level or pace scale, the band roles. Their words have everyday senses, so one alone is fine."""
        found = []
        for path in product_modules():
            if path.name in ("synth.py", "sim.py", "evaluate.py", "probe.py", "__main__.py"):
                continue
            for line, name, texts in literal_sequences(path):
                for vocabulary, words in SEQUENCES.items():
                    hits = sorted({w.lower() for w in words if any(said(w.lower(), t) for t in texts)})
                    allowed = SEQUENCE_REASONS.get((path.name, name), (set(), ""))[0]
                    if len(hits) > 1 and not set(hits) <= allowed:
                        found.append((path.name, line, name, vocabulary, hits))
        for page in sorted((ROOT / "web").glob("*.html")):
            if page.name == "index.html":
                continue  # the inspector shows the simulation's own people
            code = page_code(page.name)
            for vocabulary, words in SEQUENCES.items():
                for literal in re.findall(r"\[[^\[\]]*\]|\{[^{}]*\}|'[^'\n]*'|\"[^\"\n]*\"|`[^`]*`", code):
                    hits = sorted({w.lower() for w in words if said(w.lower(), literal)})
                    if len(hits) > 1:
                        found.append((page.name, vocabulary, hits))
        self.assertEqual(found, [])

    def test_the_guards_read_every_string(self):
        """Both guards read an f-string's text and a plain string's alike (on every Python version), never a
        docstring, and catch a vocabulary written out in a list or in one string."""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "planted.py"
            path.write_text('"""A docstring may say chess and maths."""\n'
                            'X = f"a partner for {1} chess"\nY = "maths"\nZ = ("easy", "fast")\n'
                            'W = f"from {1} beginner to advanced"\nV = "English or French"\n')
            words = [(line, w) for line, text in code_words(path) for w in ACTIVITY_WORDS if said(w, text)]
            self.assertEqual(sorted(words), [(2, "chess"), (3, "maths")])
            seqs = {(name, vocabulary) for _, name, texts in literal_sequences(path) for vocabulary, vocab in SEQUENCES.items()
                    if len({w for w in vocab if any(said(w.lower(), t) for t in texts)}) > 1}
            self.assertEqual(seqs, {("Z", "paces"), ("", "levels"), ("V", "languages")})

    def test_the_fixed_choices_are_the_only_choices(self):
        """Every other detail is written in words; nothing offers the synthetic people's vocabularies."""
        options = agent.field_options()
        self.assertEqual(set(options), {"gender", "format", "travel", "kind", "wants_gender"})
        synthetic = {v.lower() for vocab in (synth.RELIGION, synth.POLITICS, synth.HEALTH, synth.CHILDREN, synth.SMOKING,
                                             synth.DIET, synth.KINKS, synth.INTERESTS, synth.SCHEDULES, synth.LEVELS,
                                             synth.PACES, synth.SUBJECTS) for v in vocab}
        self.assertFalse({str(v).lower() for vs in options.values() for v in vs} & synthetic)

    def test_every_sensitive_word_caught_before_is_still_caught(self):
        """The fixed sensitive words live in schema.py now, with no list from synth.py: everything the agent's
        own list caught before this change still counts as sensitive (tests/test_open_details.py has the list)."""
        from tests.test_open_details import SENSITIVE_BEFORE
        self.assertGreater(len(SENSITIVE_BEFORE), 100)
        self.assertEqual([w for w in SENSITIVE_BEFORE if not mentions_sensitive(w)], [])


if __name__ == "__main__":
    unittest.main()
