"""The phone page: every screen it can open exists. (A screen was once deleted by an edit
and the menu silently stopped working.)"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

PAGE = (Path(__file__).resolve().parent.parent / "belong" / "web" / "demo.html").read_text()


class Screens(unittest.TestCase):
    def test_every_screen_the_page_opens_exists(self):
        defined = set(re.findall(r"^  (\w+)\(slot, p, L\) \{", PAGE, flags=re.M))
        opened = set(re.findall(r"""(?:openView|menuGo)\(\s*'\$\{slot\}'\s*,\s*'(\w+)'""", PAGE))
        opened |= set(re.findall(r"""go\(\s*'\$\{slot\}'\s*,\s*'(\w+)'""", PAGE))
        opened |= set(re.findall(r'\[\s*"(\w+)",\s*"(?:Profile|Settings|Receipts)"\s*\]', PAGE))
        opened |= {"verify", "signedout", "privacy", "belong", "found", "thread", "profile"}
        self.assertTrue(opened)
        self.assertEqual(opened - defined, set(), "screens the page opens but doesn't define")


class Script(unittest.TestCase):
    def test_the_page_script_parses(self):
        import shutil, subprocess, tempfile
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        js = PAGE[PAGE.index("<script>\n// ----") + len("<script>"):]
        js = js[:js.index("</script>")]
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
            f.write(js)
        result = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
