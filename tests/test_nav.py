"""The shared top menu (belong/web/nav.js) sits on every page, so no page's own styles may reach it: each part of
it has a class of its own (site-nav-...), and only nav.js styles those classes. A page once styled its own phone
dropdown as .menu, the same name the menu button had, and on a narrow screen the button left the bar."""
from __future__ import annotations

import re
import unittest
from pathlib import Path

WEB = Path(__file__).resolve().parent.parent / "belong" / "web"
NAV = (WEB / "nav.js").read_text()
# Classes nav.js may use without its prefix: the design system's wordmark, and the states it adds next to its own
# classes (the current page's link, the open menu).
_SHARED = {"wordmark", "on", "open"}


def nav_parts() -> set[str]:
    """The first class of every element nav.js writes."""
    return set(re.findall(r'class="([A-Za-z0-9_-]+)', NAV))


def styled_classes(css: str) -> set[str]:
    """Every class a stylesheet's selectors name (declarations, comments and strings left out)."""
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    selectors = re.findall(r"([^{}]+)\{", css)
    return {c for s in selectors if not s.strip().startswith("@") for c in re.findall(r"\.([A-Za-z_][\w-]*)", s)}


def page_styles() -> dict[str, str]:
    """Each page's own styles (its <style> blocks) and the design system's sheets."""
    out = {}
    for page in WEB.glob("*.html"):
        html = page.read_text()
        out[page.name] = "\n".join(re.findall(r"<style[^>]*>(.*?)</style>", html, flags=re.S | re.I))
    for sheet in (WEB / "ds").glob("*.css"):
        out["ds/" + sheet.name] = sheet.read_text()
    return out


class SharedMenuTest(unittest.TestCase):
    def test_every_part_has_a_class_of_its_own(self):
        parts = nav_parts()
        self.assertIn("site-nav-menu", parts)  # the reader still finds the parts
        for name in parts - _SHARED:
            self.assertTrue(name.startswith("site-nav-"), f"nav.js gives an element the shared class name {name!r}")

    def test_only_nav_js_styles_its_parts(self):
        for name, css in page_styles().items():
            reached = sorted(c for c in styled_classes(css) if c.startswith("site-nav-"))
            self.assertEqual(reached, [], f"{name} styles part of the shared menu: {reached}")

    def test_its_button_and_links_start_from_nothing(self):
        # A page's element rules (nav button { ... }, a { ... }) can't place or size them either.
        for part in ("button.site-nav-menu", "a.site-nav-link"):
            rule = re.search(re.escape(".site-nav " + part) + r"\s*\{([^}]*)\}", NAV)
            self.assertIsNotNone(rule, part)
            self.assertTrue(rule.group(1).strip().startswith("all: unset"), part)

    def test_the_readers_see_what_a_page_styles(self):
        # The demo page's own phone dropdown is a .menu: the reader must see it, or the test above proves nothing.
        self.assertIn("menu", styled_classes(page_styles()["demo.html"]))
        self.assertIn("site-nav-links", styled_classes(NAV))


if __name__ == "__main__":
    unittest.main()
