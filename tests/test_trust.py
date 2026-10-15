"""The trust page's claims: open-source boundary, measurements, and the agent probes."""
from __future__ import annotations

import ast
import hashlib
import unittest
from unittest import mock

from belong import attest, probe
from belong import warden as warden_module

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


class OpenSourceBoundary(unittest.TestCase):
    def test_open_source_code_never_depends_on_the_matchmaker(self):
        """The Warden can be published and audited on its own."""
        for rel in attest.OPEN_SOURCE:
            for mod in _belong_imports(rel):
                self.assertNotIn(mod, MATCHMAKER_MODULES, f"{rel} imports Matchmaker code {mod}")
                self.assertIn(mod, OPEN_MODULES, f"{rel} imports {mod}, which isn't published")

    def test_every_room_file_is_classified(self):
        room = {f"matching/{p.name}" for p in (attest.PKG / "matching").glob("*.py")}
        self.assertEqual(room, {p for p in attest.OPEN_SOURCE + attest.MATCHMAKER_FILES if p.startswith("matching/")})

    def test_published_source_hashes_to_the_measurement(self):
        m = attest.measurements()
        for rel in attest.OPEN_SOURCE:
            served = attest.open_source_text(rel).encode()
            self.assertEqual(hashlib.sha256(served).hexdigest(), m["open_source_files"][rel])
        lines = "\n".join(f"{p}:{h}" for p, h in sorted(m["open_source_files"].items()))
        self.assertEqual(hashlib.sha256(lines.encode()).hexdigest(), m["warden"])

    def test_only_published_files_are_served(self):
        for rel in attest.MATCHMAKER_FILES + ["../SPEC.md", "synth.py", "/etc/passwd"]:
            self.assertIsNone(attest.open_source_text(rel), rel)

    def test_attestation_is_honest_about_being_unsigned(self):
        doc = attest.attestation("abc")
        self.assertEqual(doc["nonce"], "abc")
        self.assertIsNone(doc["signed_by"])
        self.assertIn("SIMULATED", doc["status"])


class AgentProbes(unittest.TestCase):
    def test_canary_test_finds_no_leaks(self):
        r = probe.canary_test({"seed": 5, "count": 20})
        self.assertTrue(r["ok"], r["secrets_found_in_outputs"] or r["leak_report"])
        self.assertEqual(r["canaries_rejected_at_entry"], [])
        self.assertGreater(r["outputs_scanned"]["introductions"], 100)

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

    def test_agents_own_listing_and_secret_stay_inside(self):
        secret = "my-private-detail-xyz"
        r = probe.canary_test({"seed": 2, "count": 0, "listings": [{
            "intent_text": "Looking for a running buddy in Kyoto, weekends.",
            "fields": [{"key": "city", "value": "Kyoto"}, {"key": "schedule", "value": "weekends"},
                       {"key": "pace", "value": "easy"}, {"key": "health", "value": secret, "visibility": "hidden"}],
            "secrets": [secret]}]})
        self.assertTrue(r["ok"])
        self.assertEqual(r["custom_listings_rejected_at_entry"], [])

    def test_every_attack_is_blocked(self):
        for seed in (1, 2, 3):
            r = probe.attack_suite({"seed": seed})
            self.assertTrue(r["ok"], [x for x in r["results"] if not x["blocked"]])
            self.assertGreaterEqual(r["attempts"], 20)

    def test_agents_own_intents_are_judged(self):
        r = probe.attack_suite({"seed": 1, "intents": ["A running buddy in Kyoto", "my coworker Jane Smith"]})
        self.assertEqual([x["refused"] for x in r["your_intents"]], [False, True])


if __name__ == "__main__":
    unittest.main()
