"""How we talk about it (CLAUDE.md): in text the name is Belong, with no full stop; "Belong." with its stop is
the logo only (the wordmark, and the phone's tab labels "Belong." and "Found."), and a sentence that ends on the
name keeps its normal stop. And the internal metaphor of a locked place never reaches TRUST.md or a published
file. Pages are read as a person sees them: their text, the attributes a screen reader or a hover shows, and every
string their scripts can put on screen, never their comments. Python is read as its strings (f-strings and strings
joined across lines included), the docs as their prose (a line break is a space)."""
from __future__ import annotations

import ast
import re
import unittest
from pathlib import Path

from belong import attest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "belong" / "web"
DOCS = ("TRUST.md", "SPEC.md", "README.md", "CLAUDE.md")
_SHOWN_ATTRIBUTES = re.compile(r"""\b(?:aria-label|placeholder|title|alt)\s*=\s*("[^"]*"|'[^']*')""", re.I)
# A slash starts a regular expression, not a division, after one of these (or at the start of the code).
_BEFORE_A_PATTERN = set("(,=:[!&|?{};+-*%<>~^")


def script_strings(code: str) -> list[str]:
    """Every string literal in JavaScript code (the text parts of a template literal too), without its comments
    and regular expressions."""
    out, i, n, stack = [], 0, len(code), []  # stack: brace depths of the template literals we're inside

    def last_code_char(at: int) -> str:
        j = at - 1
        while j >= 0 and code[j].isspace():
            j -= 1
        return code[j] if j >= 0 else ""

    def template(at: int) -> int:
        """Read a template literal's text from `at` (just after a backtick) up to its end or a ${...}."""
        buf = []
        while at < n:
            c = code[at]
            if c == "\\":
                buf.append(code[at:at + 2])
                at += 2
            elif c == "`":
                out.append("".join(buf))
                return at + 1
            elif code.startswith("${", at):
                out.append("".join(buf))
                stack.append(0)
                return at + 2
            else:
                buf.append(c)
                at += 1
        out.append("".join(buf))
        return at

    while i < n:
        c = code[i]
        if code.startswith("//", i):
            i = code.find("\n", i) if "\n" in code[i:] else n
        elif code.startswith("/*", i):
            end = code.find("*/", i + 2)
            i = n if end < 0 else end + 2
        elif c in "'\"":
            j, buf = i + 1, []
            while j < n and code[j] != c and code[j] != "\n":
                buf.append(code[j:j + 2] if code[j] == "\\" else code[j])
                j += 2 if code[j] == "\\" else 1
            out.append("".join(buf))
            i = j + 1
        elif c == "`":
            i = template(i + 1)
        elif c == "/" and (last_code_char(i) in _BEFORE_A_PATTERN or last_code_char(i) == ""):
            j, in_class = i + 1, False
            while j < n and code[j] != "\n":
                if code[j] == "\\":
                    j += 2
                    continue
                if code[j] == "[":
                    in_class = True
                elif code[j] == "]":
                    in_class = False
                elif code[j] == "/" and not in_class:
                    break
                j += 1
            i = j + 1
        elif c == "{" and stack:
            stack[-1] += 1
            i += 1
        elif c == "}" and stack:
            if stack[-1] == 0:
                stack.pop()
                i = template(i + 1)
            else:
                stack[-1] -= 1
                i += 1
        else:
            i += 1
    return out


def shown_strings(page: Path) -> list[str]:
    """What a person can read from a page: its text and shown attributes outside scripts, styles and comments,
    and every string its scripts hold (markup inside them read the same way)."""
    text = page.read_text(encoding="utf-8")
    if page.suffix == ".js":
        return [s for literal in script_strings(text) for s in _markup_strings(literal)]
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.S)
    text = re.sub(r"<style\b.*?</style>", " ", text, flags=re.S | re.I)
    scripts = re.findall(r"<script\b[^>]*>(.*?)</script>", text, flags=re.S | re.I)
    text = re.sub(r"<script\b.*?</script>", " ", text, flags=re.S | re.I)
    return _markup_strings(text) + [s for code in scripts for literal in script_strings(code)
                                    for s in _markup_strings(literal)]


def _markup_strings(markup: str) -> list[str]:
    """The text between tags, and the attributes a person is shown."""
    shown = [m.group(1)[1:-1] for m in _SHOWN_ATTRIBUTES.finditer(markup)]
    return shown + [part for part in re.split(r"<[^<>]*>", markup) if part.strip()]


def _unescape(literal: str) -> str:
    """A script string as it reads on screen: its escapes (\\' \\" \\n ...) as the characters they stand for."""
    return re.sub(r"\\(.)", lambda m: {"n": "\n", "t": "\t", "r": "\r"}.get(m.group(1), m.group(1)), literal,
                  flags=re.S)


def python_strings(code: str) -> list[str]:
    """Every string in Python code: docstrings and other literals, strings joined across lines (the parser joins
    them) and each f-string whole, with {} where a value goes. Never a comment."""
    nodes = list(ast.walk(ast.parse(code)))
    parts = {id(v) for node in nodes if isinstance(node, ast.JoinedStr) for v in node.values}  # read whole
    out = []
    for node in nodes:
        if isinstance(node, ast.JoinedStr):
            out.append("".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
                               for v in node.values))
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in parts:
            out.append(node.value)
    return out


# After a sentence's own full stop, or the logo's when it is quoted, set in bold or code, or bracketed: a closing
# quotation mark, bracket, or a Markdown mark. An apostrophe followed by a letter is a possessive or a contraction.
_CLOSERS = set("\"”'’)]*`")


def name_mid_sentence(text: str) -> list[str]:
    """Where the logo's form, "Belong." with its full stop, stands for the name inside a sentence: the stop followed
    (after spaces or a line break) by a lowercase word, by a lowercase word in brackets or quotation marks, or by a
    middle dot or a dash; or followed straight away by anything but a space or a closing mark (other punctuation,
    a letter, a digit, a possessive). A sentence that ends on the name, and the logo on its own, are fine."""
    hits = []
    for m in re.finditer(r"(?<![\w.])Belong\.", text):
        rest = text[m.end():]
        if not rest:
            continue
        first = rest[0]
        if first.isspace():
            after = rest.lstrip()
            word = re.match(r"[(\[\"“‘']?([^\W\d_])", after)
            bad = bool(after) and (after[0] in "·—–" or bool(word and word.group(1).islower()))
        elif first in "'’":
            bad = len(rest) > 1 and rest[1].isalpha()
        else:
            bad = first not in _CLOSERS
        if bad:
            hits.append(text[max(0, m.start() - 30):m.end() + 12])
    return hits


def page_hits(page: Path) -> list[str]:
    return [hit for s in shown_strings(page) for hit in name_mid_sentence(_unescape(s))]


def python_hits(path: Path) -> list[str]:
    return [hit for s in python_strings(path.read_text(encoding="utf-8")) for hit in name_mid_sentence(s)]


def _wordmarks(text: str) -> list[str]:
    """The text of every element with the class "wordmark" (in markup, or in a script's markup)."""
    return re.findall(r"""<(\w+)\b[^<>]*\bclass\s*=\s*["'](?:[^"']*\s)?wordmark(?:\s[^"']*)?["'][^<>]*>([^<]*)</\1>""",
                      text)


def _pages() -> list[Path]:
    return sorted(WEB.glob("*.html")) + sorted(WEB.glob("*.js"))


class TheName(unittest.TestCase):
    def test_pages_say_belong_without_the_logos_stop(self):
        pages = _pages()
        self.assertGreaterEqual(len(pages), 6)
        for page in pages:
            self.assertEqual(page_hits(page), [], page.name)

    def test_python_strings_say_belong_without_the_logos_stop(self):
        """Agent messages, system prompts, refusals, docstrings: every string in the Python under belong/."""
        files = sorted((ROOT / "belong").rglob("*.py"))
        self.assertGreaterEqual(len(files), 20)
        for path in files:
            self.assertEqual(python_hits(path), [], str(path.relative_to(ROOT)))

    def test_the_docs_say_belong_without_the_logos_stop(self):
        for name in DOCS:
            self.assertEqual(name_mid_sentence((ROOT / name).read_text(encoding="utf-8")), [], name)

    def test_the_logo_keeps_its_full_stop(self):
        """The wordmark on every page that has one, and the phone's two tab labels (and the back bar to Found),
        still say "Belong." and "Found.": the logo isn't renamed with the text."""
        marks = [(page.name, text) for page in _pages() for _, text in _wordmarks(page.read_text(encoding="utf-8"))]
        self.assertGreaterEqual(len(marks), 6, marks)
        self.assertLessEqual({"nav.js", "demo.html", "chat.html", "trust.html"}, {page for page, _ in marks}, marks)
        for page, text in marks:
            self.assertEqual(text, "Belong.", page)
        demo = (WEB / "demo.html").read_text(encoding="utf-8")
        self.assertRegex(demo, r'\[\s*\[\s*"belong"\s*,\s*"Belong\."\s*,[^\]]*\]\s*,\s*\[\s*"found"\s*,\s*"Found\."\s*,')
        self.assertRegex(demo, r'backBar\(slot,\s*"found",\s*"Found\."\)')

    def test_the_readers_find_what_a_person_sees_and_skip_comments(self):
        """Pages: strings, template text, markup and shown attributes (escapes read as they show), never a comment
        or a regular expression. Python: every string, joined across lines, f-strings whole, never a comment."""
        code = ('// Belong.\'s own comment\n/* Belong. asks */ const re = /Belong.,/g;\n'
                'const a = "Ask Belong.?", b = `<p title="About Belong.\'s">x ${n ? `Belong. asks` : ""}</p>`;\n'
                "const c = 'Belong.\\'s', ok = \"Talk to Belong. It listens.\", end = `Ends on Belong.`;\n"
                'const logo = `<span class="wordmark">Belong.</span>`, tabs = [["belong", "Belong."], ["found", "Found."]];')
        found = [hit for literal in script_strings(code) for s in _markup_strings(literal)
                 for hit in name_mid_sentence(_unescape(s))]
        self.assertEqual(len(found), 4, found)
        self.assertFalse(any("comment" in f or "/g" in f or "Talk" in f for f in found), found)
        self.assertEqual(_wordmarks(code), [("span", "Belong.")])

        py = ('"""Belong. prototype."""\n# Belong. asks, in a comment\nn = 2\n'
              'a = ("Go back to Belong. "\n     "to see them.")\nb = f"Ask Belong.\'s agent about {n}"\n'
              'c = f"Found {n} with Belong.{n}"\nok = f"Talk to Belong. It listens to {n}. Ends on Belong."\n')
        found = [hit for s in python_strings(py) for hit in name_mid_sentence(s)]
        self.assertEqual(len(found), 4, found)
        self.assertFalse(any("comment" in f or "Talk" in f for f in found), found)

    def test_what_counts_as_the_name_inside_a_sentence(self):
        for text in ("Belong. asks once", "Belong.'s close matches", "Belong.’s", "Belong., or not", "Ask Belong.?",
                     "Belong.: rules", "Belong.com", "use Belong.\n  asks", "Belong. (the trust page)",
                     "Belong. · connected", "Belong. — at night", 'Belong. "test it" yourself'):
            self.assertEqual(len(name_mid_sentence(text)), 1, text)
        for text in ("Belong.", "Talk to Belong. It listens.", "the people who run Belong.",
                     "the people who run Belong.\nThat's the job", '("the people who run Belong.", "us")',
                     "(even by Belong.)", "**Belong.** (your listings) and **Found.**", '"Belong." is the logo',
                     "`Belong.`", "the 'Belong.' wordmark", "Belong without its stop", "Belonging. asks", "not Belong. |"):
            self.assertEqual(name_mid_sentence(text), [], text)


class NoInternalMetaphor(unittest.TestCase):
    def test_trust_md_and_published_files_never_say_it(self):
        """No "room" in TRUST.md or any published file (attest.OPEN_SOURCE), inside a name too
        ("room_state"); "roommate" is a kind of match, a different word."""
        files = [ROOT / "TRUST.md"] + [attest.PKG / rel for rel in attest.OPEN_SOURCE]
        for path in files:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                self.assertNotRegex(line.lower().replace("roommate", ""), "room", f"{path.name}:{number}")


_CANT_SEE = re.compile(
    r"\bnot even (?:we|us|belong|the people who run)\b"
    r"|\beven from (?:us|belong)\b"
    r"|\b(?:hidden|kept) from (?:us|belong)\b"
    r"|\bnobody at belong\b"
    r"|\b(?:we|belong|the people who run belong)\s+(?:can't|cannot|can not|couldn't|could not|never|don't|do not|"
    r"won't|will not|have no way to|are unable to|is unable to)\s+(?:ever\s+)?(?:see|sees|access|accesses|read|reads|"
    r"look at|looks at|learn|learns)\b")


def claims_belong_cant_see(text: str) -> list[str]:
    """Where a text claims or implies that Belong itself ("we", "us", "the people who run Belong") can't see or
    access people's data. Belong can (TRUST.md, "What Belong can see"); the promises are about other users."""
    flat = " ".join(text.replace("’", "'").split()).lower()
    return [flat[max(0, m.start() - 30):m.end() + 20] for m in _CANT_SEE.finditer(flat)]



if __name__ == "__main__":
    unittest.main()
