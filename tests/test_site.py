"""The public website (site/, built by tools/build_site.py): its two pages (the landing page and the Use page, whose
technical part is made from site/use.md) and the plain text published for AI agents (use.md, llms.txt) say Belong
the way every page does, use the design system's words, logo and tokens only, never reuse the demo's examples,
never link to the internal demo or the source repository, never overclaim privacy, and load nothing from any other
host, so visiting it tells no third party anything."""
from __future__ import annotations

import re
import sys
import tempfile
import unittest
from pathlib import Path

from tests.test_copy import _unescape, _wordmarks, name_mid_sentence, shown_strings
from tests.test_neutral import ACTIVITY_WORDS, said

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
sys.path.insert(0, str(ROOT / "tools"))
import build_site  # noqa: E402

_TMP = tempfile.TemporaryDirectory()
OUT = Path(_TMP.name) / "site"


def setUpModule():
    build_site.build(OUT)


def tearDownModule():
    _TMP.cleanup()


def pages() -> list[Path]:
    """Every page a visitor can open: the built site, so the documentation is read as it is shown."""
    return sorted(OUT.rglob("*.html"))


def texts() -> list[Path]:
    """The plain text published for AI agents to read."""
    return sorted([*OUT.rglob("*.md"), *OUT.rglob("*.txt")])


def text_of(page: Path) -> str:
    """Everything a visitor, an agent or a search result reads: shown strings and the page's description, or a
    plain text file whole."""
    if page.suffix in (".md", ".txt"):
        return page.read_text(encoding="utf-8")
    html = page.read_text(encoding="utf-8")
    extra = re.findall(r"""<meta\s+name="description"\s+content="([^"]*)\"""", html)
    return " ".join(_unescape(s) for s in shown_strings(page) + extra)


class TheSiteSpeaksLikeBelong(unittest.TestCase):
    def test_the_pages_are_there(self):
        """Two pages, by the owner's choice: the landing page and the Use page."""
        self.assertEqual({str(p.relative_to(OUT)) for p in pages()}, {"index.html", "use.html"})
        self.assertEqual({str(p.relative_to(OUT)) for p in texts()}, {"use.md", "llms.txt"})

    def test_the_name_has_no_stop_and_the_logo_is_the_design_systems(self):
        """In text the name is Belong; the logo is the design system's own file (ink, or cream on the black footer),
        never typed, linked home."""
        for page in pages():
            self.assertEqual([h for s in shown_strings(page) for h in name_mid_sentence(_unescape(s))], [], page.name)
            html = page.read_text(encoding="utf-8")
            self.assertEqual(_wordmarks(html), [], page.name)
            logos = re.findall(r'<a class="logo" href="\./"><img src="ds/logo/belong-wordmark-small-(?:ink|cream)\.svg" alt="Belong"', html)
            self.assertEqual(len(logos), 2, page.name)  # top bar and footer
        for text in texts():
            self.assertEqual(name_mid_sentence(text.read_text(encoding="utf-8")), [], text.name)

    def test_no_metaphor_no_synonyms_for_shown_and_hidden_no_emoji(self):
        """No room, vault, safe, lock or shield; a detail is "shown" or "hidden", never "secret", "public",
        "private" or "visible" ("tokens are secret", "public infrastructure" and "safe to show" are other things)."""
        for page in pages() + texts():
            text = text_of(page).lower().replace("roommate", "")
            for pattern in (r"\broom", r"\bvaults?\b", r"\b(?:a|the) safe\b", r"\b(?:un)?lock(?:ed|s)?\b", r"\bpadlock",
                            r"\bshields?\b", r"\bvisible\b",
                            r"\b(?:secret|public|private) (?:details?|fields?|profiles?|listings?|data)\b",
                            r"\b(?:is|are|made|make|kept|keep|stays?) public\b"):
                self.assertFalse(re.search(pattern, text), (page.name, pattern))
            self.assertFalse(re.search("[\U0001F300-\U0001FAFF☀-➿]", text), page.name)

    def test_it_never_overclaims_privacy(self):
        """The promise is between people: Belong runs the matcher and can see what it holds, so no page ever says "not
        even we can see it". (By the owner's choice the site reads as the finished product, with no status labels.)"""
        for page in pages() + texts():
            self.assertFalse(re.search(r"not even (we|us|belong)|no one at belong|can't access it|even the people who run",
                                       text_of(page).lower()), page.name)

    def test_no_example_of_the_demo(self):
        """The site's examples are its own, never the synthetic people's stories, hobbies or subjects."""
        for page in pages() + texts():
            text = text_of(page)
            self.assertEqual([w for w in ACTIVITY_WORDS if said(w, text)], [], page.name)

    def test_styled_only_with_tokens(self):
        """No colour of its own: every colour comes from the design system's tokens."""
        for path in [*SITE.rglob("*.css"), *SITE.rglob("*.html")]:
            css = path.read_text(encoding="utf-8")
            if path.suffix == ".html":
                css = " ".join(re.findall(r"<style\b[^>]*>(.*?)</style>", css, flags=re.S) +
                               re.findall(r'style="([^"]*)"', css))
            css = re.sub(r"/\*.*?\*/", " ", css, flags=re.S)
            self.assertFalse(re.search(r"#[0-9a-fA-F]{3,8}\b|\b(?:rgb|rgba|hsl|hsla|oklch)\(", css), path.name)


class TheBuiltSite(unittest.TestCase):
    def test_no_link_to_the_source_repository(self):
        """By the owner's choice the site doesn't point at the code: no GitHub link or address anywhere."""
        for path in pages() + texts():
            self.assertNotIn("github", path.read_text(encoding="utf-8").lower(), path.name)

    def test_no_link_to_the_internal_demo(self):
        """The demo and its pages (/demo, /trust, /connector, /inspector, its host) are internal."""
        for page in pages():
            for ref in re.findall(r'href="([^"]+)"', page.read_text(encoding="utf-8")):
                self.assertFalse(re.search(r"sslip\.io|^/(?:demo|trust|connector|inspector|chat|login)\b", ref),
                                 (page.name, ref))

    def test_motion_can_be_stopped(self):
        """Anything that moves does so only without reduced motion: CSS animation only inside its media query, and a
        script that plays something checks it first, so reduced motion shows the resting markup, the last frame.
        (By the owner's choice the landing page's film has no pause button; it plays only while on screen.)"""
        for path in [*SITE.rglob("*.css"), *SITE.rglob("*.html")]:
            text = path.read_text(encoding="utf-8")
            outside = re.sub(r"@media \(prefers-reduced-motion: no-preference\) \{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\}",
                             "", text)
            self.assertFalse(re.search(r"\banimation\s*:", outside), path.name)
            for script in re.findall(r"<script\b[^>]*>(.*?)</script>", text, flags=re.S):
                if re.search(r"\bset(?:Timeout|Interval)\b|requestAnimationFrame", script):
                    self.assertIn('matchMedia("(prefers-reduced-motion: reduce)")', script, path.name)

    def test_every_page_is_filled_in(self):
        for page in pages():
            self.assertNotIn("{{", page.read_text(encoding="utf-8"), page.name)

    def test_every_markdown_file_becomes_a_page_and_stays_text(self):
        for doc in SITE.rglob("*.md"):
            page = OUT / doc.relative_to(SITE).with_suffix(".html")
            title = doc.read_text(encoding="utf-8").splitlines()[0].removeprefix("# ")
            self.assertIn(f">{title}</h2>", page.read_text(encoding="utf-8"), doc.name)
            self.assertEqual((OUT / doc.relative_to(SITE)).read_text(encoding="utf-8"), doc.read_text(encoding="utf-8"))

    def test_every_local_link_and_file_is_there(self):
        for page in pages():
            html = page.read_text(encoding="utf-8")
            for ref in re.findall(r'(?:href|src)="([^"]+)"', html):
                if re.match(r"https?://", ref):
                    continue
                path, _, anchor = ref.partition("#")
                target = (page.parent / path).resolve() if path else page
                if target.is_dir():
                    target = target / "index.html"
                self.assertTrue(target.exists(), (page.name, ref))
                if anchor:
                    self.assertIn(f'id="{anchor}"', target.read_text(encoding="utf-8"), (page.name, ref))
        for css in OUT.rglob("*.css"):
            for ref in re.findall(r"url\(['\"]?([^'\")]+)", css.read_text(encoding="utf-8")):
                self.assertTrue((css.parent / ref).exists(), (css.name, ref))

    def test_nothing_loads_from_another_host(self):
        """Stylesheets, scripts, fonts, images and icons all come from the site itself: no CDN, font service or
        analytics ever learns who visited. Only plain links (the source code) point elsewhere."""
        for path in OUT.rglob("*"):
            if path.suffix == ".html":
                html = path.read_text(encoding="utf-8")
                self.assertFalse(re.search(r"<script\b[^>]*\bsrc=", html), path.name)
                for tag in re.findall(r"<(?:link|img|source|iframe|video|audio|embed|object)\b[^>]*>", html, flags=re.I):
                    self.assertFalse(re.search(r"""(?:href|src|data)=["']?(?:https?:)?//""", tag), (path.name, tag))
            if path.suffix in (".html", ".css"):
                text = path.read_text(encoding="utf-8")
                self.assertFalse(re.search(r"url\(['\"]?(?:https?:)?//|@import", text), path.name)


class TheMarkdownReader(unittest.TestCase):
    def test_what_the_docs_use(self):
        title, body, toc = build_site.markdown(
            "# A title\n\nSome **bold** and `a <tag>` and [a link](https://x.example).\n\n## 1. First part\n\n"
            "- one\n  continued\n- two\n  - nested\n- three\n\n| a | b |\n|---|---|\n| `x` | y |\n\n"
            "```json\n{\"k\": \"<v>\"}\n```\n")
        self.assertEqual(title, "A title")
        self.assertEqual(toc, [("1-first-part", "1. First part")])
        self.assertIn("<strong>bold</strong> and <code>a &lt;tag&gt;</code>", body)
        self.assertIn('<a href="https://x.example">a link</a>', body)
        self.assertIn("<ul><li>one continued</li><li>two<ul><li>nested</li></ul></li><li>three</li></ul>", body)
        self.assertIn("<td><code>x</code></td><td>y</td>", body)
        self.assertIn("<pre><code>{&quot;k&quot;: &quot;&lt;v&gt;&quot;}</code></pre>", body)


if __name__ == "__main__":
    unittest.main()
