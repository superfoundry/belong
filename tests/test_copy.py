"""How we talk about it (CLAUDE.md): the name is Belong., and nothing a person reads puts it before other
punctuation ("Belong.'s", "Belong.?"); and the internal metaphor of a locked place never reaches TRUST.md or a
published file. Pages are read as a person sees them: their text, the attributes a screen reader or a hover shows,
and every string their scripts can put on screen, never their comments."""
from __future__ import annotations

import re
import unicodedata
import unittest
from pathlib import Path

from belong import attest

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "belong" / "web"
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


def _name_before_punctuation(text: str) -> list[str]:
    return [text[max(0, m.start() - 30):m.end() + 3] for m in re.finditer(r"Belong\.", text)
            if m.end() < len(text) and unicodedata.category(text[m.end()]).startswith("P")]


class TheName(unittest.TestCase):
    def test_belong_is_never_followed_by_other_punctuation_on_a_page(self):
        pages = sorted(WEB.glob("*.html")) + sorted(WEB.glob("*.js"))
        self.assertGreaterEqual(len(pages), 6)
        for page in pages:
            for s in shown_strings(page):
                self.assertEqual(_name_before_punctuation(s), [], page.name)

    def test_the_reader_finds_what_a_person_sees_and_skips_comments(self):
        """The guard reads strings, template text, markup and shown attributes, but never a comment or a pattern."""
        code = ('// Belong.\'s own comment\n/* Belong.? */ const re = /["\']/g;\n'
                'const a = "Ask Belong.?", b = `<p title="About Belong.!">x ${n ? `Belong.,` : ""}</p>`;')
        found = [hit for literal in script_strings(code) for s in _markup_strings(literal)
                 for hit in _name_before_punctuation(s)]
        self.assertEqual(len(found), 3, found)
        self.assertFalse(any("comment" in f for f in found))
        self.assertEqual(_name_before_punctuation("Talk to Belong. It listens. Ends on Belong."), [])


class NoInternalMetaphor(unittest.TestCase):
    def test_trust_md_and_published_files_never_say_it(self):
        """No "room" in TRUST.md or any published file (attest.OPEN_SOURCE), inside a name too
        ("room_state"); "roommate" is a kind of match, a different word."""
        files = [ROOT / "TRUST.md"] + [attest.PKG / rel for rel in attest.OPEN_SOURCE]
        for path in files:
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                self.assertNotRegex(line.lower().replace("roommate", ""), "room", f"{path.name}:{number}")


if __name__ == "__main__":
    unittest.main()
