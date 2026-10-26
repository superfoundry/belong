"""The web pages: every screen the phone page can open exists (a screen was once deleted by an
edit and the menu silently stopped working), the inspector's "Take over" works, and every script parses."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

PAGE_DIR = Path(__file__).resolve().parent.parent / "belong" / "web"
PAGE = (PAGE_DIR / "demo.html").read_text()


class Screens(unittest.TestCase):
    def test_every_screen_the_page_opens_exists(self):
        defined = set(re.findall(r"^  (\w+)\(slot, p, L\) \{", PAGE, flags=re.M))
        opened = set(re.findall(r"""(?:openView|menuGo)\(\s*'\$\{slot\}'\s*,\s*'(\w+)'""", PAGE))
        opened |= set(re.findall(r"""go\(\s*'\$\{slot\}'\s*,\s*'(\w+)'""", PAGE))
        opened |= set(re.findall(r'\[\s*"(\w+)",\s*"(?:Profile|Settings|Receipts)"\s*\]', PAGE))
        opened |= {"verify", "signedout", "belong", "found", "thread", "profile"}
        self.assertTrue(opened)
        self.assertEqual(opened - defined, set(), "screens the page opens but doesn't define")


class TakeOverLink(unittest.TestCase):
    """The inspector's "Take over" reaches a real action, and the phone page opens any phone the demo makes."""

    def test_the_inspector_takes_over_through_an_action_the_server_has(self):
        from belong import server
        inspector = (PAGE_DIR / "index.html").read_text()
        self.assertIn('fetch("/api/take_over"', inspector)
        self.assertIn("take_over", server.ACTIONS)
        self.assertIn('"/demo?phone=" + encodeURIComponent(d.slot)', inspector)

    def test_the_phone_page_accepts_every_phone_id_the_demo_makes(self):
        from belong.demo import SLOT_ID
        accepted = re.search(r"const VIEW = ONLY && /(.+?)/\.test\(ONLY\)", PAGE).group(1)
        self.assertEqual(accepted, "^" + SLOT_ID.pattern + "$")


class TrustPage(unittest.TestCase):
    def test_the_attack_count_comes_from_the_suite(self):
        """The page states no fixed number of attacks; it shows how many the suite actually tried."""
        trust = (PAGE_DIR / "trust.html").read_text()
        self.assertIsNone(re.search(r"Tries \d+ attacks", trust))
        self.assertIn('id="attack-count"', trust)
        self.assertIn('$("attack-count").textContent = `${r.attempts} attacks`', trust)


class Script(unittest.TestCase):
    """Every page's own script parses (a syntax error blanks the whole page)."""

    def test_every_page_script_parses(self):
        import shutil, subprocess, tempfile
        node = shutil.which("node")
        if not node:
            self.skipTest("node isn't installed")
        web = Path(__file__).resolve().parent.parent / "belong" / "web"
        for page in sorted(web.glob("*.html")):
            html = page.read_text()
            for n, block in enumerate(re.findall(r"<script>(.*?)</script>", html, flags=re.S)):
                with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as f:
                    f.write(block)
                result = subprocess.run([node, "--check", f.name], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, f"{page.name} script {n}: {result.stderr}")


if __name__ == "__main__":
    unittest.main()
