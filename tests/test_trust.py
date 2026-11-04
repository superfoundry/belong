"""The trust page's claims: open-source boundary, measurements, and the agent probes."""
from __future__ import annotations

import ast
import hashlib
import re
import unittest
from collections import Counter
from contextlib import ExitStack
from unittest import mock

from belong import attest, probe, schema
from belong import warden as warden_module
from belong.matching import hybrid
from belong.matching.marketboard import Marketboard
from belong.schema import HIDDEN, K_CROWD, MAX_LISTINGS_PER_OWNER, SHOWN
from tests.helpers import detail, fld, listing, person, warden

OPEN_MODULES = {"belong." + p.removesuffix(".py").replace("/", ".").removesuffix(".__init__") for p in attest.OPEN_SOURCE}
MATCHMAKER_MODULES = {"belong." + p.removesuffix(".py").replace("/", ".") for p in attest.MATCHMAKER_FILES}


def _belong_imports(rel: str) -> set:
    """Absolute names of the belong.* modules a file imports."""
    pkg = ["belong"] + rel.split("/")[:-1]
    out = set()
    for node in ast.walk(ast.parse((attest.PKG / rel).read_text())):
        if isinstance(node, ast.ImportFrom) and node.level:
            base = pkg[: len(pkg) - node.level + 1]
            out.add(".".join(base + ([node.module] if node.module else [])))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("belong"):
            out.add(node.module)
    return out


def _lenient_reading(any_label: bool):
    """A reading of the Matchmaker AI's answer without the strict rules: every pair's verdicts and fit as given, for
    any label (any_label) or only this request's candidates."""
    def read(answer, candidates, labels):
        verdicts = {}
        for p in answer.get("pairs", []) if isinstance(answer, dict) else []:
            if isinstance(p, dict) and (any_label or p.get("candidate") in candidates):
                verdicts.setdefault(p.get("candidate"), (p.get("focus_wants"), p.get("candidate_wants"), p.get("fit")))
        return verdicts, set(), Counter()
    return read


class OpenSourceBoundary(unittest.TestCase):
    def test_the_warden_never_depends_on_the_matchmaker(self):
        """Both are open source, but the Warden must stand on its own: it treats the
        Matchmaker as hostile, so none of its code may come from the Matchmaker."""
        for rel in attest.WARDEN_FILES:
            for mod in _belong_imports(rel):
                self.assertNotIn(mod, MATCHMAKER_MODULES, f"{rel} imports Matchmaker code {mod}")

    def test_published_code_only_depends_on_published_code(self):
        for rel in attest.OPEN_SOURCE:
            for mod in _belong_imports(rel):
                self.assertIn(mod, OPEN_MODULES, f"{rel} imports {mod}, which isn't published")

    def test_every_room_file_is_classified(self):
        room = {f"matching/{p.name}" for p in (attest.PKG / "matching").glob("*.py")}
        self.assertEqual(room, {p for p in attest.OPEN_SOURCE if p.startswith("matching/")})

    def test_published_source_hashes_to_the_measurement(self):
        m = attest.measurements()
        for rel in attest.OPEN_SOURCE:
            served = attest.open_source_text(rel).encode()
            self.assertEqual(hashlib.sha256(served).hexdigest(), m["open_source_files"][rel])
        for agent, files in m["groups"].items():
            lines = "\n".join(f"{p}:{m['open_source_files'][p]}" for p in sorted(files))
            self.assertEqual(hashlib.sha256(lines.encode()).hexdigest(), m[agent], agent)

    def test_only_published_files_are_served(self):
        for rel in attest.MATCHMAKER_FILES:
            self.assertIsNotNone(attest.open_source_text(rel), rel)  # the Matchmaker is open source too
        for rel in ["../SPEC.md", "synth.py", "demo.py", "/etc/passwd", "matching/../demo.py"]:
            self.assertIsNone(attest.open_source_text(rel), rel)

    def test_attestation_is_honest_about_being_unsigned(self):
        doc = attest.attestation("abc")
        self.assertEqual(doc["nonce"], "abc")
        self.assertIsNone(doc["signed_by"])
        self.assertIn("SIMULATED", doc["status"])

    def test_attestation_says_the_matchmaker_ai_is_a_mock(self):
        """What an agent verifying Belong. reads says, as the trust page does, that the Matchmaker AI is a mock for
        synthetic and test profiles, and what it may be sent."""
        ai = attest.attestation("abc")["policy"]["matchmaker_ai"]
        self.assertTrue(ai["status"].startswith("MOCK"))
        self.assertIn("synthetic and test profiles only", ai["status"])
        self.assertIn("hidden text taken out", ai["receives"])
        self.assertIn("A8", ai["receives"])
        self.assertIn("warden.exit_judge_request", ai["receives"])  # checked before it goes, not only audited after


class AgentProbes(unittest.TestCase):
    def test_canary_test_finds_no_leaks(self):
        r = probe.canary_test({"seed": 5, "count": 20})
        self.assertTrue(r["ok"], r["secrets_found_in_outputs"] or r["leak_report"])
        self.assertEqual(r["canaries_rejected_at_entry"], [])
        self.assertGreater(r["outputs_scanned"]["introductions"], 100)

    def test_canaries_cover_hidden_open_details_and_the_shared_list(self):
        r = probe.canary_test({"seed": 5, "count": 5})
        self.assertEqual(r["secrets_planted"], 5 * 5)  # two hidden built-ins, a hidden detail's name and value, a note
        self.assertEqual(r["outputs_scanned"]["details_in_use"], 1)

    def test_canary_test_catches_a_sabotaged_warden(self):
        """Test the tester: make the Warden leak each viewer's counterpart's employer."""
        original = warden_module.Warden.exit_introduction

        def leaky(self, proposal, board):
            intro = original(self, proposal, board)
            if intro is not None:
                other = board.listing(intro.person_b).get("employer")
                if other is not None:
                    view = intro.a_view
                    view["reason"] += f" They work at {other.value}."
                    intro._a = __import__("json").dumps(view)
            return intro

        with mock.patch.object(warden_module.Warden, "exit_introduction", leaky):
            r = probe.canary_test({"seed": 5, "count": 30})
        self.assertFalse(r["ok"])
        self.assertTrue(r["secrets_found_in_outputs"])

    def test_canary_test_with_a_hostile_matchmaker_ai_finds_no_leaks(self):
        """The same week with the hybrid Matchmaker and a stand-in AI that says yes to every pair and answers out of
        shape: nothing hidden reaches anyone, and no canary token planted in a hidden detail (one is also written in
        the canary's description) reaches the AI; every request it was sent passes audit A8."""
        r = probe.canary_test({"seed": 5, "count": 20, "matchmaker": "hostile"})
        self.assertTrue(r["ok"], r["secrets_found_in_outputs"] or r["leak_report"])
        self.assertEqual(r["canaries_rejected_at_entry"], [])
        self.assertGreater(r["ai_requests_scanned"], 50)
        self.assertEqual(probe.canary_test({"seed": 5, "count": 2})["ai_requests_scanned"], 0)  # the rules: none

    def test_canary_test_catches_hidden_text_sent_to_the_matchmaker_ai(self):
        """Test the tester: without the redactor, the hidden employer a canary names in its description reaches the
        AI, and both the token search and audit A8 say so."""
        with mock.patch.object(hybrid.Redactor, "clean", lambda self, text, keep=(): schema.canonical(text)):
            r = probe.canary_test({"seed": 5, "count": 20, "matchmaker": "hostile"})
        self.assertFalse(r["ok"])
        self.assertTrue(any(f.get("where") == "sent to the Matchmaker AI" for f in r["secrets_found_in_outputs"]))
        self.assertIn("A8", {f["audit"] for f in r["leak_report"]["findings"]})

    def test_agents_own_listing_and_secret_stay_inside(self):
        secret = "my-private-detail-xyz"
        r = probe.canary_test({"seed": 2, "count": 0, "listings": [{
            "intent_text": "Looking for a running buddy in Kyoto, weekends.",
            "fields": [{"key": "city", "value": "Kyoto"}, {"key": "schedule", "value": "weekends"},
                       {"key": "pace", "value": "easy"}, {"key": "health", "value": secret, "visibility": "hidden"}],
            "secrets": [secret]}]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["custom_listings_rejected_at_entry"], [])

    def test_agents_can_bring_open_details_and_hidden_names(self):
        secret_name, secret_value = "nightowlxyz", "privatevalueqrs"
        r = probe.canary_test({"seed": 2, "count": 0, "listings": [{
            "intent_text": "Looking for someone to play duets with in Lyon.",
            "fields": [{"key": "city", "value": "Lyon"}, {"key": "kind", "value": "other"},
                       {"name": "skill", "compare": "steps", "steps": ["low", "middle", "high"], "values": ["middle"]},
                       {"name": "skill", "about": "them", "compare": "steps", "steps": ["low", "middle", "high"],
                        "values": ["middle", "high"]},
                       {"name": secret_name, "values": [secret_value], "visibility": "hidden"}]}]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["custom_listings_rejected_at_entry"], [])
        self.assertEqual(r["secrets_planted"], 2)  # a hidden detail's name and value are both secrets

    def test_an_agents_open_detail_that_breaks_a_rule_is_refused_not_dropped(self):
        r = probe.canary_test({"seed": 2, "count": 0, "listings": [{
            "intent_text": "Looking for someone to play duets with in Lyon.",
            "fields": [{"key": "city", "value": "Lyon"}, {"name": "health", "values": ["asthma"]}]}]})
        self.assertEqual(len(r["custom_listings_rejected_at_entry"]), 1)

    def test_every_attack_is_blocked(self):
        for seed in (1, 2, 3):
            r = probe.attack_suite({"seed": seed})
            self.assertTrue(r["ok"], [x for x in r["results"] if not x["blocked"]])
            self.assertGreaterEqual(r["attempts"], 20)
            # Stopped by the rule it is about, not by a side effect of how the attack is built.
            by = {x["attack"]: x["stopped_by"] for x in r["results"]}
            self.assertEqual(by["Show a sensitive detail without the person confirming"], "sensitive_defaults")

    def test_each_attack_is_stopped_by_the_rule_it_names(self):
        """Test the tester: take away the rule an attack is about and the attack gets through."""
        def got(patches):
            with ExitStack() as stack:
                for p in patches:
                    stack.enter_context(p)
                return {x["attack"]: x["blocked"] for x in probe.attack_suite({"seed": 1})["results"]}

        def by_listings(board):  # the crowd counted in listings instead of people
            return Counter((f.key, schema.canonical(v)) for l in board.listings() if l.active
                           for f in l.fields for v in f.strings())
        name_problems = schema.name_problems
        one_detail = schema._one_detail_problems
        proper = warden_module._is_proper_noun
        crowd = f"Make a rare hidden value count with {K_CROWD - 2} people and one person's {MAX_LISTINGS_PER_OWNER} listings"
        anchored = "Single out one person with a place to find them and one detail of how they look"
        cases = {
            (crowd,): [mock.patch.object(Marketboard, "value_counts", by_listings)],
            ("Put a sensitive word in an open detail's name, with an ordinary value",): [mock.patch.object(
                schema, "name_problems", lambda n: [p for p in name_problems(n) if "sensitive" not in p])],
            ("Claim to be verified in an open detail's value", "Claim to be verified in a shown headline"): [
                mock.patch.object(schema, "vouching_problems", lambda text: [])],
            ("Hide a detail both people must share (it could never decide anything)",): [mock.patch.object(
                schema, "_one_detail_problems", lambda f, kind: [p for p in one_detail(f, kind) if "always shown" not in p])],
            ("Single out one person with wishes: a workplace in one, looks in another", "Ask for an ex in a wish"): [
                mock.patch.object(warden_module, "detail_person_findings", lambda *a, **k: [])],
            tuple(x for x in got([]) if x.startswith("Put an instruction for agents")): [
                mock.patch.object(warden_module, "_ends_a_clause", lambda c: False),
                mock.patch.object(warden_module, "LIST_PHRASE_WORDS", 99),
                mock.patch.object(warden_module, "LIST_VALUE_WORDS", 99)],
            tuple(x for x in got([]) if x.startswith("Join an instruction for agents")): [
                mock.patch.object(warden_module, "_list_words", lambda text: len(text.split()))],
            tuple(x for x in got([]) if x.startswith("Write an instruction for agents in a script without spaces")): [
                mock.patch.object(warden_module, "LIST_UNSPACED_CHARS", 99)],
            tuple(x for x in got([]) if x.startswith("Run an instruction for agents together")): [
                mock.patch.object(warden_module, "LIST_LONG_RUN", 99)],
            ("Submit a listing under another person's live listing id, to take over its introductions",
             "Bring back a deleted listing's id (someone else's or your own) for a new listing"): [
                mock.patch.object(warden_module, "listing_id_problem", lambda listing, board: "")],
            ("Ask for someone by name with the rule's word as a wish's name",): [
                mock.patch.object(warden_module, "_written", lambda f: [f"{f.name}: {', '.join([*f.strings(), *f.steps])}"])],
            ("Hide an ex behind a doubled space or a line break in the description",): [
                mock.patch.object(warden_module, "_one_line", lambda text: str(text or ""))],
            ("Single out one person with a workplace in a shown phrase and looks in a wish",): [
                mock.patch.object(warden_module, "OWN_WORDS", ())],
            # The words that are never part of a name come only from the listing's own shown city and country: read
            # from every field, whatever its visibility, an open detail or a hidden city unlocks a name.
            ("Unlock a full name by putting its words in an open detail",
             "Unlock a full name by putting its words in a hidden city"): [
                mock.patch.object(warden_module, "never_names", lambda fields=(), places=(): frozenset(
                    warden_module._bare(t.group()) for f in fields for s in f.texts()
                    for t in warden_module._TOKEN.finditer(s)))],
            # The rule the brief rules out: reading capitalised words after "in" as a place lets a name through.
            ("Search for a full name with a possessive after 'interested in'",): [
                mock.patch.object(warden_module, "_is_proper_noun", lambda text, tok, exempt=frozenset(): proper(
                    text, tok, exempt) and not re.search(r"\bin\s+(\S+\s+)?$", text[:tok.start()]))],
            # The identifiers rule, one part at a time: no anchors, any number of traits, no distinctive marks.
            # A place unlocks only as a whole: read word by word, one word of the listing's two-word city unlocks a
            # name.
            ("Unlock a full name with one word of the listing's own two-word city",): [
                mock.patch.object(warden_module, "_in_places", lambda tokens, places: {
                    i for i, t in enumerate(tokens) if any(warden_module._bare(t.group()) in (
                        (p,) if isinstance(p, str) else p) for p in places)})],
            ("Search for a full name joined by a lower-case particle, or after an initial",): [
                mock.patch.object(warden_module, "_NAME_JOIN", frozenset()),
                mock.patch.object(warden_module, "_initial", lambda word: False)],
            ("Hide a name, an ex or a clue behind an invisible character beside a space",): [
                mock.patch.object(warden_module, "_one_line", lambda text: " ".join(str(text or "").split()))],
            ("Say where a distinctive mark is through 'the ... of' their body",): [
                mock.patch.object(warden_module, "_on_someone", lambda toks, i: i < len(toks) and (
                    toks[i] in warden_module._WHOSE_BODY or (toks[i] == "the" and i + 1 < len(toks)
                                                             and toks[i + 1] in {"left", "right"})))],
            ("Single out someone the poster saw or met, without 'I' or with 'a' or 'we'",): [
                mock.patch.object(warden_module, "_encounter", lambda toks, k, place: False)],
            ("Hide a look in a looks word followed by what it is like",): [
                mock.patch.object(warden_module, "_a_feature", lambda toks, k: False)],
            ("Single out one person at a workplace or home the poster shares with them",): [
                mock.patch.object(warden_module, "_shared_place", lambda toks, k, j: False)],
            # What the poster says about themselves, never read: clues about someone else written there go in.
            ("Single out one person with clues in an open detail about yourself",): [
                mock.patch.object(warden_module, "detail_clues", lambda f: warden_module.wish_clues(f)
                                  if f.about != "me" else warden_module.Clues())],
            ("Single out one person with clues in what you offer",): [
                mock.patch.object(warden_module, "poster_words", lambda fields=(): [])],
            ("Write an instruction for agents in letter symbols to get it on the shared list",): [
                mock.patch.object(warden_module, "_LIST_FORM", "NFD"),
                mock.patch.object(warden_module, "_letter_symbol", lambda c: False)],
            (anchored,): [mock.patch.object(warden_module, "_anchors", lambda toks, starts, names=frozenset(): [])],
            ("Single out one person with three details of how they look",): [
                mock.patch.object(warden_module, "MAX_COMMON_TRAITS", 99)],
            ("Single out one person by a distinctive mark and where it is or what it shows",): [
                mock.patch.object(warden_module, "_distinctive", lambda toks, k, featured=False: False)],
            # The Matchmaker AI (stand-in judges): the hard rules it decides inside, the strict reading of its answer,
            # the Warden's daily limit, and what it receives. Without any one, its attack gets through.
            ("A Matchmaker AI says yes to every pair, dating's genders and ages aside",): [
                mock.patch.object(hybrid, "_dating_met", lambda *a: True)],
            ("A Matchmaker AI adds a pair the rules refused: a wish on a detail of the same name unmet",): [
                mock.patch.object(hybrid, "details_hard", lambda *a: True)],
            ("A Matchmaker AI answers out of shape: other labels, odd numbers, extra keys, free text",): [
                mock.patch.object(hybrid, "read_answer", _lenient_reading(any_label=True))],
            ("A listing tells the Matchmaker AI to propose it to everyone, and the AI obeys",): [
                mock.patch.object(warden_module.Warden, "_approaches_today", lambda self, pid: 0)],
            ("A listing tells the Matchmaker AI to copy another person's details to it, and the AI obeys",): [
                mock.patch.object(hybrid.Redactor, "clean", lambda self, text, keep=(): schema.canonical(text))],
            ("A listing tells the Matchmaker AI to say everyone else is after one particular person, and the AI obeys",): [
                mock.patch.object(hybrid.LLMMatcher, "_may_flag", staticmethod(lambda labels: list(labels)))],
            ("A modified Matchmaker sends the Matchmaker AI a hidden detail",): [
                mock.patch.object(warden_module.Warden, "exit_judge_request", lambda self, request, labels, board: True)],
            # Past the update limit, an update counted as only turning something off as soon as it turns one off,
            # whatever else it changes, lets any change through.
            ("Get past the update limit by turning something off and changing something else",): [
                mock.patch.object(warden_module, "only_turns_off", lambda listing, board, owner: (
                    listing.person_id in board and any(getattr(board.listing(listing.person_id), k) is True
                                                       and not getattr(listing, k) for k in warden_module.SWITCHES)))],
        }
        everything = got([])
        for names, patches in cases.items():
            self.assertTrue(names)
            self.assertTrue(all(everything[n] for n in names), names)
            without = got(patches)
            self.assertEqual([without[n] for n in names], [False] * len(names), names)
        # The identifiers attack's controls test the tester too: a rule that read every looks word as describing a
        # person, or the poster's own job and looks as the person sought's, refuses a control, so the attack no
        # longer counts as blocked.
        for patches in ([mock.patch.object(warden_module, "_describes_a_person", lambda *a, **k: True)],
                        [mock.patch.object(warden_module, name, lambda *a, **k: False)
                         for name in ("_own_anchor", "_own_looks", "_own_noun")]):
            self.assertFalse(got(patches)[anchored], patches)
        # A reading that keeps only this request's labels but takes any verdict and any fit lets a fit that isn't a
        # number (NaN, which compares false with every threshold) make a proposal.
        shape = "A Matchmaker AI answers out of shape: other labels, odd numbers, extra keys, free text"
        self.assertFalse(got([mock.patch.object(hybrid, "read_answer", _lenient_reading(any_label=False))])[shape])

    def test_the_matchmaker_ai_attacks_name_their_rule_and_have_controls(self):
        """Each is stopped by the rule it names, and each has a control that must go through, so a rule that refused
        everything wouldn't count as blocking it: a Warden that let no introduction out fails the five whose control is
        an honest introduction; taking out every word of every text fails the one whose control is that the listing's
        own words still reach the AI; a Warden that let no request to the AI go fails the one whose control is that an
        honest request still goes; and never reading a flag fails the one whose control is that a listing flagged
        when asked about alone still loses its pairs."""
        r = probe.attack_suite({"seed": 2})
        ai = {x["attack"]: x for x in r["results"] if x["category"] == "matchmaker AI"}
        self.assertEqual(len(ai), 7)
        self.assertTrue(all(x["blocked"] for x in ai.values()), ai)
        self.assertTrue(all(x["stopped_by"] for x in ai.values()))

        def blocked(patch):
            with patch:
                return {x["attack"] for x in probe.attack_suite({"seed": 2})["results"]
                        if x["category"] == "matchmaker AI" and x["blocked"]}
        copy = "A listing tells the Matchmaker AI to copy another person's details to it, and the AI obeys"
        modified = "A modified Matchmaker sends the Matchmaker AI a hidden detail"
        flags = "A listing tells the Matchmaker AI to say everyone else is after one particular person, and the AI obeys"
        self.assertEqual(blocked(mock.patch.object(warden_module.Warden, "exit_introduction", lambda *a: None)),
                         {copy, modified})
        self.assertNotIn(copy, blocked(mock.patch.object(hybrid.Redactor, "clean",
                                                         lambda self, text, keep=(): hybrid.REMOVED)))
        self.assertNotIn(modified, blocked(mock.patch.object(warden_module.Warden, "exit_judge_request",
                                                             lambda *a: False)))
        self.assertNotIn(flags, blocked(mock.patch.object(hybrid.LLMMatcher, "_may_flag",
                                                          staticmethod(lambda labels: []))))

    def test_a_listing_hiding_a_common_detail_is_no_false_alarm(self):
        """A hidden name or value many other people show isn't found in what is sent about them."""
        _, _, _, board = probe._sandbox(4)
        shown = Counter((f.name, v) for l in board.listings() for f in l.fields
                        if f.about == "me" and f.visibility == SHOWN and f.compare == "same" for v in f.strings())
        (name, value), _ = shown.most_common(1)[0]
        r = probe.canary_test({"seed": 4, "count": 0, "listings": [{
            "intent_text": "Looking for someone to share a quiet allotment with.",
            "fields": [{"key": "city", "value": "Lyon"}, {"key": "kind", "value": "other"},
                       {"key": "seeking", "value": "an allotment partner"},
                       {"key": "offering", "value": "an allotment partner"},
                       {"name": name, "values": [value], "visibility": "hidden"}]}]})
        self.assertEqual(r["custom_listings_rejected_at_entry"], [])
        self.assertEqual(r["secrets_found_in_outputs"], [])
        self.assertTrue(r["ok"], r["leak_report"])

    def test_what_counts_as_a_secret_found(self):
        """Only in what is sent about its owner to someone else, as whole words, less what that person may see."""
        w, board = warden(), Marketboard()
        owner = listing("own", fields=[fld("city", "Lyon"), detail("pace", "brisk"),
                                       detail("locker", "cobalt heron", visibility=HIDDEN)])
        other = listing("oth", fields=[fld("city", "Lyon"), detail("locker", "cobalt heron")])
        for l in (owner, other):
            assert w.admit(person(l.person_id), l, board).accepted
        planted = {"own": ["cobalt heron", "locker", "brisk walk", "brisk"]}

        def found(view_of_owner, view_of_other="", reveals=(), asked=()):
            intro = {"person_a": "oth", "person_b": "own", "a_view": {"fields": {}, "reason": view_of_owner},
                     "b_view": {"fields": {}, "reason": view_of_other}}
            outputs = {"introductions": [intro], "reveals": list(reveals)}
            return [f["secret"] for f in probe._secrets_found(planted, set(), board, outputs, list(asked), "")]
        self.assertEqual(found("They keep a cobalt heron."), ["cobalt heron"])  # about the owner, to someone else
        self.assertEqual(found("", "You'd like a cobalt heron."), [])            # about someone else
        self.assertEqual(found("They walk at a brisk walking pace."), [])       # whole words, and the owner shows "brisk"
        # The viewer's own value, said back about a detail the owner hides: that tells them the owner's.
        self.assertEqual(found("You share the same locker: cobalt heron."), ["cobalt heron", "locker"])
        reveal = {"person_a": "oth", "person_b": "own", "key": "me:locker", "values": ["cobalt heron"]}
        self.assertEqual(found("", reveals=[reveal]), ["cobalt heron", "locker"])
        question = {"person_id": "oth", "question_id": "also_offer", "phrase": "brisk walk", "offer_add": "",
                    "seek_add": "", "pay": "", "question_text": "Someone is brisk walk. Would that work for you?"}
        self.assertEqual(found("", asked=[("", question)]), ["brisk walk", "brisk"])
        quoted = {**question, "phrase": "brisk", "question_text": "Someone is brisk. Would that work for you?"}
        self.assertEqual(found("", asked=[("own", quoted)]), [])  # quoting what the owner shows
        self.assertEqual(found("", asked=[("", quoted)]), ["brisk"])
        # A canary's random token counts anywhere at all.
        self.assertEqual([f["secret"] for f in probe._secrets_found({"own": ["cnryabc"]}, {"cnryabc"}, board,
                                                                       {"introductions": [], "reveals": []}, [],
                                                                       '{"receipts": "cnryabc"}')], ["cnryabc"])

    def test_agents_own_intents_are_judged(self):
        r = probe.attack_suite({"seed": 1, "intents": ["A running buddy in Kyoto", "my coworker Jane Smith"]})
        self.assertEqual([x["refused"] for x in r["your_intents"]], [False, True])


if __name__ == "__main__":
    unittest.main()
