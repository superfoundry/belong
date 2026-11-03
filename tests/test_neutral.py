"""No early example decides what other requests get (CLAUDE.md, "How we fix things"): a detail's label, the
Warden's reason sentences and a question's answer are true for any listing that can carry them; a detail goes
only on the kinds it belongs to (agent.carries); a change of kind keeps only what's about the person; and the
phone and the connector offer Start matching by one rule."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import unittest

from belong import agent
from belong.agent_llm import build_draft
from belong.demo import DemoError, DemoSession
from belong.schema import HIDDEN, SHOWN
from belong.warden import reason_text
from tests.test_demo import climbers

# Every activity the offline agent knows by name: a label or a Warden sentence naming one is tied to it.
ACTIVITY_WORDS = {w for a in agent.KNOWN_ACTIVITIES for w in re.findall(r"[a-z]+", a.lower()) if len(w) > 3}


def save(kind, *fields, current=None):
    """A model agent's save, as build_draft gets it."""
    args = {"kind": kind, "exactness": 0.6, "intent_text": "Looking for someone, in my own words.", "suggestions": [],
            "fields": [{"key": k, "values": [v] if isinstance(v, str) else list(v), "visibility": SHOWN} for k, v in fields]}
    return build_draft(args, "Ines", "30-34", current)


class WordedForAnyListing(unittest.TestCase):
    def test_no_detail_label_names_an_activity(self):
        for key, label in agent.LABELS.items():
            self.assertFalse(set(re.findall(r"[a-z]+", label.lower())) & ACTIVITY_WORDS, (key, label))

    def test_the_wardens_reasons_name_no_activity(self):
        text = reason_text([("level", "beginner", "advanced"), ("pace", "easy", "brisk"), ("role", "student", "tutor")])
        self.assertFalse(set(re.findall(r"[a-z]+", text.lower())) & ACTIVITY_WORDS, text)
        for value in ("advanced", "brisk", "tutor"):
            self.assertIn(value, text)
        self.assertNotIn("side", text)  # "They're the tutor side" only fitted one kind's roles


class EachDetailWhereItBelongs(unittest.TestCase):
    def test_a_role_only_on_the_kinds_that_have_its_choices(self):
        _, problems, _ = save("tutoring", ("role", "technical"), ("subject", "Welsh"), ("seeking", "a Welsh tutor"),
                              ("offering", "a Welsh student"))
        self.assertTrue(any("role" in p for p in problems), problems)  # tutoring's roles are tutor or student
        draft, problems, notes = save("other", ("role", "student"), ("seeking", "a sailing crew"),
                                      ("offering", "a sailor"), ("headline", "Looking for sailing crew"))
        self.assertEqual(problems, [])
        self.assertNotIn("role", {f["key"] for f in draft["fields"] if f["value"]})
        self.assertTrue(any(n.startswith("role was left out") for n in notes), notes)

    def test_a_general_detail_goes_wherever_it_applies(self):
        draft, problems, _ = save("other", ("level", "intermediate"), ("seeking", "a go player"),
                                  ("offering", "a go player"), ("headline", "Looking for a go partner"))
        self.assertEqual(problems, [])
        level = next(f for f in draft["fields"] if f["key"] == "level")
        self.assertEqual((level["value"], level["label"]), ("intermediate", "Level"))

    def test_a_change_of_kind_keeps_only_whats_about_the_person(self):
        before, problems, _ = save("climbing", ("level", "advanced"), ("city", "Graz"), ("schedule", "weekends"),
                                   ("interests", ["birdwatching"]))
        self.assertEqual(problems, [])
        after, problems, _ = save("tutoring", ("role", "student"), ("subject", "Welsh"), current=before)
        self.assertEqual(problems, [])
        values = {f["key"]: f["value"] for f in after["fields"] if f["value"] not in (None, "", [])}
        self.assertNotIn("level", values)  # the old request's own detail
        self.assertEqual((values["city"], values["schedule"], values["interests"]), ("Graz", "weekends", ["birdwatching"]))
        again, _, _ = save("climbing", ("city", "Graz"), current=before)  # same kind: nothing it left out is lost
        self.assertEqual({f["key"]: f["value"] for f in again["fields"]}["level"], "advanced")


class AnswersHaveTheirValue(unittest.TestCase):
    def setUp(self):
        self.s = DemoSession(seed=7, agent=None)
        climbers(self.s)
        self.mine = self.s.slots["A"].listing()

    def ask(self, question_id):
        self.mine.followups.append({"question_id": question_id, "question_text": "?"})

    def test_a_question_says_what_its_answer_is(self):
        for q in ("age_range", "schedule", "also_offer"):
            self.ask(q)
        types = {q["question_id"]: q["answer"] for q in self.s.phone_state("A")["listings"][0]["followups"]}
        self.assertEqual(types["age_range"], {"type": "age_range"})
        self.assertEqual(types["schedule"]["type"], "choice")
        self.assertIn("weekends", types["schedule"]["options"])
        self.assertEqual(types["also_offer"], {"type": "yes_no"})

    def test_a_yes_never_stands_in_for_a_value(self):
        for q in ("age_range", "schedule"):
            self.ask(q)
            before = self.s.board.listing(self.mine.id)
            with self.assertRaises(DemoError):
                self.s.answer_followup("A", q, {"yes": True})
            self.assertEqual(self.s.board.listing(self.mine.id), before)  # nothing changed, still asked
            self.assertIn(q, [x["question_id"] for x in self.mine.followups])


class StartMatchingByOneRule(unittest.TestCase):
    def test_the_server_and_the_phone_agree(self):
        """The connector shows the server's "ready" (DemoSession._ready); the phone works it out itself
        (canSave). Both must say the same for the same draft."""
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        from tests.test_dating import _js
        from tests.test_profile_words import FUNCTIONS
        drafts = [agent.draft_from_message(m, "Ines", "30-34") for m in (
            "Looking for a roommate in Graz.", "Looking for a climbing partner.", "Looking to meet a man in Graz.",
            "Looking for someone special in Graz.", "I review CVs, online only.", "hello")]
        drafts.append(save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("format", "Online"))[0])
        drafts.append(save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("format", "In person"))[0])
        hidden = save("other", ("seeking", "a sailing crew"), ("offering", "a sailor"), ("city", "Graz"))[0]
        for f in hidden["fields"]:
            if f["key"] == "city":
                f["visibility"] = HIDDEN  # a hidden place never decides
        drafts.append(hidden)
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
        self.assertIn(True, server)
        self.assertIn(False, server)


if __name__ == "__main__":
    unittest.main()
