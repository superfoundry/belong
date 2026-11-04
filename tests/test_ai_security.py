"""Security checks for the Matchmaker AI, run as if it were enclosed (TRUST.md #39 to #41): the independent audit of
every request its judge is sent (leaks.py, A8), tested on planted leaks; the audit in the demo's presenter leak report;
and whole simulated weeks with stand-in judges that answer as badly as they can (probe.hostile_judge, obedient_judge).
Every judge here is a stub: no network. The examples (telescope making and beekeeping in Graz) are new ones."""
from __future__ import annotations

import copy
import importlib.util
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

from belong import __main__ as cli
from belong import probe, sim
from belong.demo import DemoSession
from belong.leaks import leak_report
from belong.matching.contract import Proposal
from belong.matching.hybrid import REMOVED, LLMMatcher
from belong.matching.marketboard import Marketboard
from belong.matchmaker_openai import OpenAIJudge
from belong.schema import HIDDEN
from belong.warden import checked_judge
from tests.helpers import detail, fld, listing, person, warden
from tests.test_demo import climbers
from tests.test_hybrid import synthetic_board


def graz(pid, intent, seeking, offering, *extra, name="Lena", **kw):
    """A listing that isn't dating, in Graz, with its phrases."""
    return replace(listing(pid, intent, [fld("first_name", name), fld("country", "Austria"), fld("city", "Graz"),
                                         fld("kind", "other"), fld("seeking", seeking), fld("offering", offering),
                                         *extra]), **kw)


def yes(request):
    return {"pairs": [{"candidate": c, "focus_wants": "match", "candidate_wants": "match", "fit": 0.9}
                      for c in request["candidates"]], "hunts_one_person": []}


def mirror_makers():
    """Telescope mirror grinding in Graz: hidden built-ins, a hidden open detail, and a description that names its own
    hidden workplace and another person."""
    w, board = warden(), Marketboard()
    for l in (graz("tm1", "Grinding my first telescope mirror in Graz; I work at sternwerk optics and Jonas sent me.",
                   "a mirror grinding mentor", "a mirror grinding student", fld("employer", "sternwerk optics", HIDDEN),
                   detail("night owl", "always", visibility=HIDDEN), detail("mirror size", "six inch")),
              graz("tm2", "I have ground a dozen telescope mirrors in Graz and like teaching it.",
                   "a mirror grinding student", "a mirror grinding mentor", detail("mirror size", "six inch"),
                   name="Jonas"),
              graz("tm3", "Telescope builder in Graz, happy to help with figuring and testing.",
                   "a mirror grinding student", "a mirror grinding mentor", name="Mira")):
        assert w.admit(person(l.owner), l, board).accepted, l.person_id
    return w, board


def audit(board, sends, wire=()):
    return leak_report(board, [], [], [], [], [], set(), {}, 20, 15, judge_requests=sends, wire_requests=wire)


def a8(report):
    return [f["detail"] for f in report["findings"] if f["audit"] == "A8"]


class TheAuditOfWhatTheAIIsSent(unittest.TestCase):
    def test_an_honest_run_passes_and_every_request_is_counted(self):
        w, board = synthetic_board(200, 9)
        matcher = LLMMatcher(judge=yes, max_pairs_per_run=10_000)
        matcher.match(board)
        report = audit(board, matcher.sent_requests())
        self.assertTrue(report["ok"], report["findings"][:5])
        self.assertEqual(report["checked"]["judge_requests"], len(matcher.payloads))
        self.assertGreater(report["checked"]["judge_entries"], len(matcher.payloads))
        # Without the Matchmaker AI there is nothing of it in the report.
        self.assertNotIn("judge_requests", leak_report(board, [], [], [], [], [], set(), {}, 20, 15)["checked"])

    def test_the_redactor_took_out_what_the_audit_looks_for(self):
        w, board = mirror_makers()
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm1"})
        report = audit(board, matcher.sent_requests())
        self.assertTrue(report["ok"], report["findings"])
        entry = next(e for r in matcher.payloads for e in r["listings"].values() if "first telescope" in e["description"])
        self.assertEqual(entry["description"], f"grinding my first telescope mirror in graz; i work at {REMOVED} and "
                                               f"{REMOVED} sent me.")

    def test_planted_leaks_are_found(self):
        w, board = mirror_makers()
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm1"})
        honest = next(r for r in matcher.sent_requests() if r["request"]["candidates"])  # tm1 with its candidates
        label = next(k for k, v in honest["labels"].items() if v == "tm1")
        other = next(k for k, v in honest["labels"].items() if v != "tm1")

        def planted(change):
            sent = copy.deepcopy(honest)
            change(sent, sent["request"]["listings"][label])
            return a8(audit(board, [sent]))
        self.assertTrue(any("sternwerk optics" in f and "hides" in f for f in planted(
            lambda s, e: e.update(description=e["description"].replace(REMOVED, "sternwerk optics", 1)))))
        self.assertTrue(any("'night owl'" in f for f in planted(lambda s, e: e.setdefault("details", []).append(
            {"about": "me", "name": "night owl", "values": ["always"], "compare": "same"}))))
        self.assertTrue(any("'jonas'" in f and "first name" in f for f in planted(
            lambda s, e: e.update(description=e["description"] + " jonas"))))
        self.assertTrue(any("'tm2'" in f and "id" in f for f in planted(
            lambda s, e: e.update(looking_for=e["looking_for"] + " tm2"))))
        self.assertTrue(any("'city'" in f for f in planted(lambda s, e: e.update(city="graz"))))
        self.assertTrue(any("isn't its owner's" in f for f in planted(
            lambda s, e: e.update(description=s["request"]["listings"][other]["description"]))))
        self.assertTrue(any("exactness" in f for f in planted(lambda s, e: e.update(exactness=0.5000001))))
        self.assertTrue(any("isn't on" in f or "on the board then" in f for f in planted(
            lambda s, e: s["labels"].update({label: "nobody"}))))
        self.assertTrue(any("is an id" in f for f in planted(
            lambda s, e: s["labels"].update({"tm2": s["labels"].pop(label)}) or s["request"]["listings"].update(
                {"tm2": s["request"]["listings"].pop(label)}) or s["request"].update(
                focus="tm2" if s["request"]["focus"] == label else s["request"]["focus"],
                candidates=["tm2" if c == label else c for c in s["request"]["candidates"]]))))
        self.assertTrue(planted(lambda s, e: s["request"].update(note="free text")))
        # A shown phrase the listing hides now, sent as if shown.
        hid = replace(board.listing("tm2"), fields=tuple(replace(f, visibility=HIDDEN) if f.key == "seeking" else f
                                                          for f in board.listing("tm2").fields), version=2)
        self.assertTrue(w.admit(person("tm2"), hid, board).accepted)
        late = copy.deepcopy(honest)
        late["board_change"] = board.changes
        self.assertTrue(any("isn't shown by its owner" in f for f in a8(audit(board, [late]))))

    def test_read_against_the_board_as_it_was_when_sent(self):
        """Someone hiding a detail after a request went out raises no false alarm; the same text in a request built
        after it is a finding."""
        w, board = mirror_makers()
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm2"})
        self.assertTrue(any("six inch" in json.dumps(r) for r in matcher.payloads))
        hidden = tuple(replace(f, visibility=HIDDEN) if f.key == "me:mirror size" else f
                       for f in board.listing("tm2").fields)
        self.assertTrue(w.admit(person("tm2"), replace(board.listing("tm2"), fields=hidden, version=2), board).accepted)
        self.assertTrue(audit(board, matcher.sent_requests())["ok"])
        late = [{**s, "board_change": board.changes} for s in matcher.sent_requests()]
        self.assertTrue(a8(audit(board, late)))
        # And once the Matchmaker runs again, what it sends is fine against the board as it is.
        matcher.match(board, focus={"tm2"})
        self.assertTrue(audit(board, matcher.sent_requests())["ok"])

    def test_a_hidden_text_the_listing_shows_itself_is_its_own(self):
        w, board = warden(), Marketboard()
        for l in (graz("bk1", "Beekeeping partner in Graz; I keep carniolan bees.", "a beekeeping partner",
                       "a beekeeping partner", detail("bees", "carniolan bees")),
                  graz("bk2", "Beekeeping in Graz at weekends.", "a beekeeping partner", "a beekeeping partner",
                       detail("bees", "carniolan bees", visibility=HIDDEN), name="Theo")):
            self.assertTrue(w.admit(person(l.owner), l, board).accepted)
        matcher = LLMMatcher(judge=yes)
        matcher.match(board)
        self.assertIn("carniolan bees", json.dumps(matcher.payloads))  # bk1 shows it itself
        self.assertTrue(audit(board, matcher.sent_requests())["ok"])
        sent = copy.deepcopy(next(r for r in matcher.sent_requests() if "bk2" in r["labels"].values()))
        theirs = next(k for k, v in sent["labels"].items() if v == "bk2")
        sent["request"]["listings"][theirs]["description"] += " carniolan bees"
        self.assertTrue(any("carniolan bees" in f for f in a8(audit(board, [sent]))))  # bk2 hides it

    def test_a_deleted_listings_name_still_counts_and_a_name_in_any_script(self):
        w, board = mirror_makers()
        gone = graz("tm9", "Looking for a telescope club in Graz.", "a telescope club", "a telescope club",
                    name="美咲")
        self.assertTrue(w.admit(person("tm9"), gone, board).accepted)
        self.assertTrue(w.withdraw(person("tm9"), "tm9", board))
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm1"})
        sent = copy.deepcopy(matcher.sent_requests()[0])
        entry = next(iter(sent["request"]["listings"].values()))
        entry["description"] += " 美咲さん"
        self.assertTrue(any("美咲" in f for f in a8(audit(board, [sent]))))

    def test_every_request_the_models_own_record_holds_was_kept(self):
        w, board = mirror_makers()
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm1"})
        sends = matcher.sent_requests()
        wire = [json.loads(json.dumps(s["request"], sort_keys=True)) for s in sends]  # as the model's record keeps it
        self.assertTrue(audit(board, sends, wire)["ok"])
        stray = copy.deepcopy(next(r for r in wire if len(r["candidates"]) > 1))
        stray["candidates"] = stray["candidates"][:1]
        self.assertTrue(any("without being kept" in f for f in a8(audit(board, sends, wire + [stray]))))

    def test_a_watched_text_with_its_marks_written_differently_is_found(self):
        """A hidden text or first name is found in what was sent however its marks are written: a curly or straight
        apostrophe or none, a character that shows nothing inside it or between words, another dash, and with or
        without the punctuation at its ends."""
        w, board = warden(), Marketboard()
        for l in (graz("hv1", "Learning to keep bees in Graz, weekends only.", "a beekeeping mentor",
                       "a beekeeping student", fld("employer", "Hofer's Werkstatt GmbH.", HIDDEN),
                       fld("neighborhood", "Lend-Ost", HIDDEN), name="Zoë-Marie"),
                  graz("hv2", "Beekeeper in Graz with two hives, glad to teach.", "a beekeeping student",
                       "a beekeeping mentor", name="Theo")):
            self.assertTrue(w.admit(person(l.owner), l, board).accepted, l.person_id)
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"hv1"})
        honest = next(r for r in matcher.sent_requests() if r["request"]["candidates"])
        label = next(k for k, v in honest["labels"].items() if v == "hv2")
        self.assertTrue(audit(board, [honest])["ok"])
        for said in ("hofer\u2019s werkstatt gmbh", "hofers werkstatt gmbh", "hofer's werk\u00adstatt gmbh,",
                     "lend\u2013ost", "in\u200blend ost", "zoe\u2011marie", "zo\u00eb marie"):
            sent = copy.deepcopy(honest)
            sent["request"]["listings"][label]["description"] += f" {said}"
            self.assertTrue(a8(audit(board, [sent])), said)

    def test_a_first_name_since_changed_still_counts(self):
        """A first name counts from when it was first on the board, for ever after: a name its person has since
        changed is still theirs."""
        w, board = mirror_makers()
        tm2 = board.listing("tm2")
        renamed = replace(tm2, fields=tuple(fld("first_name", "Joe") if f.key == "first_name" else f
                                            for f in tm2.fields), version=2)
        self.assertTrue(w.admit(person("tm2"), renamed, board).accepted)
        matcher = LLMMatcher(judge=yes)
        matcher.match(board, focus={"tm1"})
        self.assertTrue(audit(board, matcher.sent_requests())["ok"])
        self.assertNotIn("jonas", json.dumps(matcher.payloads))  # the redactor still takes it out
        sent = copy.deepcopy(next(r for r in matcher.sent_requests() if r["request"]["candidates"]))
        next(iter(sent["request"]["listings"].values()))["description"] += " jonas"
        self.assertTrue(any("'jonas'" in f and "first name" in f for f in a8(audit(board, [sent]))))

    def test_a_score_that_isnt_a_number_never_clears_the_threshold(self):
        """Audit R1 with the Matchmaker AI in mind: NaN compares false with everything, so it is read as no score."""
        w, board = mirror_makers()
        p = Proposal("tm1", "tm2", True, True, float("nan"), float("nan"), 0.1, ("city",), ())
        report = leak_report(board, [], [], [], [], [p], set(), {}, 20, 15)
        self.assertIn("R1", {f["audit"] for f in report["findings"]})


class InTheDemo(unittest.TestCase):
    def test_the_presenters_leak_report_audits_every_request(self):
        s = DemoSession(seed=7, agent=None, matchmaker=yes)
        s.run_inline = True
        climbers(s)
        s.run_matchmaker()
        report = s.room_state()["leak_report"]
        self.assertTrue(report["ok"], report["findings"])
        self.assertEqual(report["checked"]["judge_requests"], len(s.matcher.payloads))
        self.assertGreater(report["checked"]["judge_requests"], 0)
        # A leak planted in what was sent shows up in the presenter's report (planted in place, so the report's cache,
        # which counts requests, is cleared by hand).
        entry = next(iter(s.matcher.payloads[0]["listings"].values()))
        name = next(f.value for l in s.board.listings() for f in l.fields if f.key == "first_name")
        entry["description"] += f" {name}"
        s._audit = (None, None)
        report = s.room_state()["leak_report"]
        self.assertFalse(report["ok"])
        self.assertIn("A8", {f["audit"] for f in report["findings"]})
        # With the rules, nothing of it.
        ruled = DemoSession(seed=7, agent=None, people=40)
        self.assertNotIn("judge_requests", ruled.room_state()["leak_report"]["checked"])

    def test_the_models_own_record_is_compared_and_a_reset_leaves_its_old_requests_out(self):
        def post(url, payload, headers, timeout):
            data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            return {"choices": [{"message": {"content": json.dumps(yes(data))}}], "usage": {}}
        class Local(OpenAIJudge):  # a model on this machine, on a stub HTTP layer
            def __init__(self):
                super().__init__(url="http://127.0.0.1:9/v1/chat/completions", post=post)
        s = DemoSession(seed=7, agent=None, matchmaker="rules")
        s.run_inline = True
        with mock.patch("belong.demo.OpenAIJudge", Local):
            s._matchmaker_choice = "ai"
            s.reset(7)
            model = s._judge.judge
            climbers(s)
            s.run_matchmaker()
            self.assertTrue(model.sent)
            report = s._leak_report()
            self.assertTrue(report["ok"], report["findings"])
            self.assertEqual(report["checked"]["wire_requests"], len(model.sent))
            # A request on the wire that the Matchmaker didn't keep is a finding.
            model.sent.append({"focus": "c1", "candidates": [], "listings": {"c1": {"description": "x"}}})
            self.assertIn("A8", {f["audit"] for f in s._leak_report()["findings"]})
            model.sent.pop()
            # After a reset, requests from before it are left out of the comparison (their board is gone).
            s.reset(8)
            report = s._leak_report()
            self.assertTrue(report["ok"], report["findings"])
            self.assertEqual(report["checked"]["wire_requests"], 0)

    def test_the_models_record_is_compared_with_the_wardens_however_many_requests(self):
        """Every request the model's own record says went out must be one the Warden's check let through: the Warden
        keeps the digest of every one, so the comparison goes on after the Matchmaker has let old requests go."""
        def post(url, payload, headers, timeout):
            data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            return {"choices": [{"message": {"content": json.dumps(yes(data))}}], "usage": {}}

        class Local(OpenAIJudge):  # a model on this machine, on a stub HTTP layer
            def __init__(self):
                super().__init__(url="http://127.0.0.1:9/v1/chat/completions", post=post)
        s = DemoSession(seed=7, agent=None, matchmaker="rules")
        s.run_inline = True
        with mock.patch("belong.demo.OpenAIJudge", Local):
            s._matchmaker_choice = "ai"
            s.reset(7)
            model = s._judge.judge
            climbers(s)
            s.run_matchmaker()
            report = s._leak_report()
            self.assertTrue(report["ok"], report["findings"])
            self.assertEqual(report["checked"]["wire_checked"], len(model.sent))
            self.assertEqual(len(s.warden.judge_passed), len(model.sent))
            # Sent past the Warden (straight to the model's client): found, before and after old requests are let go.
            past = {"focus": "c1", "candidates": [], "listings": {"c1": {"kind": "other", "description": "x"}}}
            model(past)
            findings = a8(s._leak_report())
            self.assertTrue(any("without passing the Warden's check" in f for f in findings), findings)
            s.matcher.payloads_dropped = 1  # as if more than PAYLOADS_KEPT were kept: the first comparison stops
            report = s._leak_report()
            self.assertEqual(report["checked"]["wire_requests"], 0)
            self.assertTrue(any("without passing the Warden's check" in f for f in a8(report)))


class TheWardensCheckBeforeARequestGoes(unittest.TestCase):
    """warden.exit_judge_request: every request to the Matchmaker AI is checked against the board as it is just before
    it is handed to the model's client, by the Warden's own rules (not the audit's, not the Matchmaker's)."""

    def honest(self):
        w, board = mirror_makers()
        received = []

        def model(request):
            received.append(request)
            return yes(request)
        matcher = LLMMatcher(judge=None)
        matcher.judge = checked_judge(w, model, lambda: board, lambda: matcher.payloads, lambda: matcher.labels)
        matcher.match(board, focus={"tm1"})
        return w, board, matcher, received

    def test_an_honest_request_passes_and_leaves_a_receipt(self):
        w, board, matcher, received = self.honest()
        self.assertTrue(received)
        self.assertEqual(received, matcher.payloads)  # each one, exactly as kept
        checks = [r for r in w.receipts if r.check == "judge_request"]
        self.assertEqual((len(checks), {r.result for r in checks}), (len(received), {"pass"}))
        self.assertEqual(len(w.judge_passed), len(received))
        self.assertEqual(matcher.last_run.get("failed_requests", 0), 0)

    def test_planted_leaks_never_go(self):
        w, board, matcher, _ = self.honest()
        kept = next(i for i, r in enumerate(matcher.payloads) if r["candidates"])
        request, labels = matcher.payloads[kept], matcher.labels[kept]
        mine = next(k for k, v in labels.items() if v == "tm1")
        theirs = next(k for k, v in labels.items() if v != "tm1")
        self.assertTrue(w.exit_judge_request(request, labels, board))

        def planted(change, mapping=None):
            sent = copy.deepcopy(request)
            change(sent, sent["listings"][mine])
            return w.exit_judge_request(sent, labels if mapping is None else mapping, board)
        for refused in (
                lambda s, e: e.update(description=e["description"].replace(REMOVED, "sternwerk optics", 1)),
                lambda s, e: e.update(description=e["description"] + " stern\u00adwerk optics"),
                lambda s, e: e.setdefault("details", []).append(
                    {"about": "me", "name": "night owl", "values": ["always"], "compare": "same"}),
                lambda s, e: e.update(description=e["description"] + " jonas"),
                lambda s, e: e.update(looking_for=e["looking_for"] + " tm2"),
                lambda s, e: e.update(city="graz"),
                lambda s, e: e.update(description=s["listings"][theirs]["description"]),
                lambda s, e: e.update(exactness=0.9),
                lambda s, e: e.update(description=e["description"] + " and more"),  # words its listing never wrote
        ):
            self.assertFalse(planted(refused))
        self.assertFalse(w.exit_judge_request(request, None, board))  # a request the Matchmaker didn't keep
        self.assertFalse(w.exit_judge_request(request, {**labels, mine: labels[theirs]}, board))  # the wrong listing
        self.assertEqual([r.result for r in w.receipts if r.check == "judge_request"][-11:], ["fail"] * 11)

    def test_it_reads_the_board_as_it_is_when_the_request_goes(self):
        """A request that was true when it was built goes only if it still is: not once a detail in it is hidden, its
        listing is paused or deleted, or its owner changed their name to a word it says."""
        for change in ("hide", "pause", "delete"):
            w, board, matcher, _ = self.honest()
            kept = next(i for i, r in enumerate(matcher.payloads) if r["candidates"])
            request, labels = matcher.payloads[kept], matcher.labels[kept]
            self.assertTrue(w.exit_judge_request(request, labels, board))
            tm1 = board.listing("tm1")
            if change == "hide":
                fields = tuple(replace(f, visibility=HIDDEN) if f.key == "me:mirror size" else f for f in tm1.fields)
                self.assertTrue(w.admit(person("tm1"), replace(tm1, fields=fields, version=2), board).accepted)
            elif change == "pause":
                self.assertTrue(w.admit(person("tm1"), replace(tm1, active=False, version=2), board).accepted)
            else:
                self.assertTrue(w.withdraw(person("tm1"), "tm1", board))
            self.assertFalse(w.exit_judge_request(request, labels, board), change)

    def test_written_differently_it_still_passes_what_the_matchmaker_honestly_sends(self):
        """The redactor and the Warden read marks the same way, each with its own code: what an honest Matchmaker
        sends from listings written with curly apostrophes, soft hyphens and other dashes passes."""
        w, board = warden(), Marketboard()
        for l in (graz("hv1", "Learning to keep bees in Graz, my hives sit behind the baker\u2019s yard in the "
                              "lend\u00adost quarter.", "a beekeeping mentor", "a beekeeping student",
                       fld("employer", "the baker's yard", HIDDEN), fld("neighborhood", "lend-ost", HIDDEN)),
                  graz("hv2", "Beekeeper in Graz \u2013 two hives, glad to teach; a friend of lena\u2019s told me.",
                       "a beekeeping student", "a beekeeping mentor", name="Theo")):
            self.assertTrue(w.admit(person(l.owner), l, board).accepted, l.person_id)
        received = []
        matcher = LLMMatcher(judge=None)
        matcher.judge = checked_judge(w, lambda r: received.append(r) or yes(r), lambda: board,
                                      lambda: matcher.payloads, lambda: matcher.labels)
        matcher.match(board)
        self.assertEqual(len(received), len(matcher.payloads))
        self.assertTrue(received)
        sent = json.dumps(received, ensure_ascii=False)
        for gone in ("baker", "lend", "ost", "lena"):
            self.assertNotRegex(sent, rf"\b{gone}\b")
        self.assertTrue(audit(board, matcher.sent_requests())["ok"])


class AHostileMatchmakerAIOnASyntheticBoard(unittest.TestCase):
    """A whole simulated week (sim.run) with stand-in judges that answer as badly as they can: the full leak report,
    the audit of every request sent among it, still finds nothing; the rules' hard rules decide what they can make."""

    def test_whatever_the_ai_answers_nothing_hidden_gets_out(self):
        flag_everyone = lambda request: {**probe.hostile_judge(request),  # noqa: E731
                                         "hunts_one_person": list(request["listings"])}
        for judge in (probe.hostile_judge, probe.obedient_judge, flag_everyone):
            result = sim.run(n=150, seed=13, judge=judge, max_pairs=10 ** 6)
            report = result["leak_report"]
            self.assertTrue(report["ok"], (judge, report["findings"][:5]))
            self.assertGreater(report["checked"]["judge_requests"], 0)
            if judge is not flag_everyone:  # (a listing flagged when asked about alone isn't asked about its pairs)
                self.assertGreater(report["checked"]["judge_entries"], report["checked"]["judge_requests"])
        self.assertEqual(result["meta"]["matchmaker_ai"].get("from_judge", 0), 0)  # flagged everyone: only drops
        self.assertGreater(result["meta"]["matchmaker_ai"]["hunting"], 0)

    def test_belong_run_hostile_needs_no_network_and_writes_nothing(self):
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d, mock.patch.object(cli, "OUT", Path(d) / "run.json"), \
                mock.patch.object(cli, "ROOT", Path(d)), mock.patch.object(cli, "load_dotenv"), \
                mock.patch.object(cli, "OpenAIJudge", side_effect=AssertionError("no model")), redirect_stdout(out):
            code = cli.main(["run", "--n", "80", "--matcher", "hostile"])
            self.assertFalse((Path(d) / "run.json").exists())
        self.assertEqual(code, 0)
        self.assertIn("hostile stand-in", out.getvalue())
        self.assertEqual(out.getvalue().count("Leak report: PASS"), 2)


def _compare_tool():
    """tools/compare_judges.py, loaded from its file (tools/ isn't a package)."""
    spec = importlib.util.spec_from_file_location("compare_judges", Path(__file__).resolve().parent.parent / "tools"
                                                  / "compare_judges.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TheCompareTool(unittest.TestCase):
    """tools/compare_judges.py: feedback for the rules from the Matchmaker AI. Live it costs money, so here it only ever
    meets stub judges (and a stub HTTP layer); without --yes it sends nothing at all."""

    def setUp(self):
        self.tool = _compare_tool()

    def test_disagreements_are_sorted_into_buckets_with_the_ground_truth(self):
        no = lambda request: {"pairs": [{**p, "focus_wants": "no"} for p in yes(request)["pairs"]],  # noqa: E731
                              "hunts_one_person": []}
        said_yes = self.tool.compare(120, 3, yes, pairs=60, outside=12)
        said_no = self.tool.compare(120, 3, no, pairs=60, outside=12)
        self.assertEqual(said_yes["asked"]["answered"], said_yes["asked"]["shortlisted"] + said_yes["asked"]["outside"])
        self.assertEqual(said_yes["buckets"]["rules_yes_ai_no"], [])
        self.assertEqual(len(said_yes["buckets"]["ai_yes_outside"]), said_yes["asked"]["outside"])
        self.assertEqual(said_no["buckets"]["ai_yes_rules_no"], [])
        self.assertEqual(said_no["buckets"]["ai_yes_outside"], [])
        # What the rules say yes to is the same either way; only who disagrees changes.
        self.assertEqual(len(said_no["buckets"]["rules_yes_ai_no"]), said_yes["agree"]["both_yes"])
        rows = said_yes["buckets"]["ai_yes_rules_no"]
        self.assertTrue(rows)
        self.assertTrue(all(set(r) == {"pair", "truth", "ai", "hard_rules_allow"} for r in rows))
        self.assertIn(True, {r["hard_rules_allow"] for r in rows})
        self.assertIn(False, {r["hard_rules_allow"] for r in rows})
        self.assertTrue(said_yes["audit"]["ok"], said_yes["audit"]["findings"])
        text = self.tool.report(said_yes)
        for title in self.tool.TITLES.values():
            self.assertIn(title, text)
        self.assertIn("the ground truth says", text)

    def test_without_yes_nothing_is_sent(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.tool.main(["--n", "60"], judge_factory=lambda: self.fail("a judge was set up"))
        self.assertEqual(code, 0)
        self.assertIn("Nothing was sent", out.getvalue())

    def test_with_yes_it_reports_the_tokens_used(self):
        def post(url, payload, headers, timeout):
            data = json.loads(payload["messages"][1]["content"].split("\n", 1)[1])
            return {"choices": [{"message": {"content": json.dumps(yes(data))}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 5, "total_tokens": 55}}
        model = OpenAIJudge(url="http://127.0.0.1:9/v1/chat/completions", post=post)
        out = io.StringIO()
        with redirect_stdout(out):
            code = self.tool.main(["--n", "60", "--pairs", "20", "--outside", "4", "--yes"], judge_factory=lambda: model)
        self.assertEqual(code, 0)
        self.assertIn(f"Tokens: {model.usage['prompt_tokens']} in", out.getvalue())
        self.assertIn("Every request passed the leak audit (A8).", out.getvalue())
        self.assertGreater(model.usage["calls"], 0)


if __name__ == "__main__":
    unittest.main()
