"""Build constraints that mirror the locked room (SPEC §8):
  - Matchmaker module has no network access and no logging of raw data.
  - Only the Warden module can produce output objects.
"""
from __future__ import annotations

import ast
import socket
import unittest
from pathlib import Path
from unittest import mock

from belong.matching.matcher import OfflineMatcher
from belong.sim import run
from belong.warden import _SEAL, FollowUpOut, Introduction, Reveal

PKG = Path(__file__).resolve().parent.parent / "belong"
ROOM = PKG / "matching"

# Standard-library modules code inside the room may import. Anything else (socket,
# urllib, http, logging, subprocess, os, pathlib, io, ...) fails the test.
ROOM_ALLOWED_IMPORTS = {"__future__", "collections", "dataclasses", "math", "re", "typing", "unicodedata"}
ROOM_BANNED_CALLS = {"print", "open", "exec", "eval", "__import__", "input", "breakpoint"}


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield a.name, 0
        elif isinstance(node, ast.ImportFrom):
            yield node.module or "", node.level


class RoomHasNoWayOut(unittest.TestCase):
    def test_room_imports_only_allowed_modules(self):
        for path in ROOM.glob("*.py"):
            for module, level in _imports(ast.parse(path.read_text())):
                if level == 1:
                    continue  # another module inside the room
                if level == 2:
                    self.assertEqual(module, "schema", f"{path.name} reaches outside the room: ..{module}")
                    continue
                self.assertIn(module.split(".")[0], ROOM_ALLOWED_IMPORTS, f"{path.name} imports {module}")

    def test_room_never_prints_logs_or_opens_files(self):
        for path in ROOM.glob("*.py"):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ROOM_BANNED_CALLS, f"{path.name} calls {node.func.id}()")

    def test_matching_works_with_the_network_switched_off(self):
        def no_network(*a, **kw):
            raise AssertionError("the room tried to open a network connection")
        with mock.patch.object(socket, "socket", no_network), \
             mock.patch.object(socket, "create_connection", no_network):
            result = run(n=120, seed=4, matcher=OfflineMatcher())
        self.assertTrue(result["leak_report"]["ok"])


class OnlyTheWardenCreatesOutputs(unittest.TestCase):
    def test_output_objects_refuse_construction_without_the_seal(self):
        with self.assertRaises(PermissionError):
            Introduction(object(), "i1", "a", "b", {}, {})
        with self.assertRaises(PermissionError):
            FollowUpOut(None, "a", "age_range")
        with self.assertRaises(PermissionError):
            Reveal(None, "a", "b", "religion", ["x"])

    def test_no_module_but_the_warden_touches_the_seal(self):
        self.assertIsNotNone(_SEAL)
        for path in PKG.rglob("*.py"):
            if path.name != "warden.py":
                self.assertNotIn("_SEAL", path.read_text(), f"{path.relative_to(PKG)} references the Warden's seal")


if __name__ == "__main__":
    unittest.main()
