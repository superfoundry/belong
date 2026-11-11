"""Build the public website (site/) into a folder any static host can serve.

    python3 tools/build_site.py                      # into out/site, linking the demo on the same host
    python3 tools/build_site.py --app https://demo.example.org --out /tmp/site

Copies site/ as it is, adds the Belong design system's files from belong/web/ds (tokens.css, belong.css and
the fonts, so the site loads nothing from any other host) and the touch icon, and fills in {{app}}: where the
demo app is served, without a trailing slash ("" means the same host).

The documentation is Markdown (site/docs/*.md, copied in as they are written): each becomes a page beside it
(standard.md -> standard.html) in site/docs/_page.html, read by the small Markdown reader below, which knows
only what the docs use: headings, paragraphs, lists (one level of nesting), tables, fenced code, `code`,
**bold** and [links](url). Pure standard library."""
from __future__ import annotations

import argparse
import html
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
DS = ROOT / "belong" / "web" / "ds"
DS_FILES = ("tokens.css", "belong.css")
ICON = ROOT / "belong" / "web" / "apple-touch-icon.png"
FILLED = (".html", ".css")
REPO = "https://github.com/superfoundry/belong/blob/main/"
# A file the docs name in passing becomes a link: the other docs pages, or the file in the public repository.
DOC_LINKS = {"STANDARD.md": "standard.html", "CONNECTING.md": "connecting.html",
             "TRUST.md": REPO + "TRUST.md", "SPEC.md": REPO + "SPEC.md", "README.md": REPO + "README.md"}


# -- Markdown ---------------------------------------------------------------------------------------------------

def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"[`*]", "", text).lower()).strip("-")


def inline(text: str) -> str:
    """One line's worth of Markdown as HTML: code spans first (their text is never read as anything else)."""
    out = []
    for n, part in enumerate(re.split(r"(`[^`]*`)", text)):
        if n % 2:
            out.append(f"<code>{html.escape(part[1:-1])}</code>")
            continue
        part = html.escape(part, quote=False)
        part = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", part)
        part = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{html.escape(m.group(2))}">{m.group(1)}</a>', part)
        part = re.sub(r"(?<![\w/.-])(" + "|".join(map(re.escape, DOC_LINKS)) + r")(?![\w-])",
                      lambda m: f'<a href="{DOC_LINKS[m.group(1)]}">{m.group(1)}</a>', part)
        out.append(part)
    return "".join(out)


def _list(lines: list[str]) -> str:
    """A list, its items' continuation lines joined, and a more indented list inside an item nested in it."""
    items: list[tuple[int, list[str]]] = []
    for line in lines:
        m = re.match(r"( *)- (.*)", line)
        if m:
            items.append((len(m.group(1)), [m.group(2)]))
        else:
            items[-1][1].append(line.strip())
    out, top = ["<ul>"], items[0][0]
    for i, (indent, words) in enumerate(items):
        text = inline(" ".join(words))
        if indent > top:
            if i == 0 or items[i - 1][0] == top:
                out[-1] = out[-1].removesuffix("</li>") + "<ul>"
            out.append(f"<li>{text}</li>")
            if i + 1 == len(items) or items[i + 1][0] == top:
                out.append("</ul></li>")
        else:
            out.append(f"<li>{text}</li>")
    out.append("</ul>")
    return "".join(out)


def _table(lines: list[str]) -> str:
    rows = [[c.strip() for c in line.strip().strip("|").split("|")] for line in lines]
    head, body = rows[0], [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
    th = "".join(f"<th>{inline(c)}</th>" for c in head)
    trs = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
    return f'<div class="scroll"><table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def markdown(text: str) -> tuple[str, str, list[tuple[str, str]]]:
    """(title, body HTML, [(id, heading)] for each second-level heading)."""
    lines, out, toc, title, i = text.splitlines(), [], [], "", 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("```"):
            end = next((j for j in range(i + 1, len(lines)) if lines[j].startswith("```")), len(lines))
            out.append(f"<pre><code>{html.escape(chr(10).join(lines[i + 1:end]))}</code></pre>")
            i = end + 1
        elif m := re.match(r"(#{1,4}) (.*)", line):
            level, words = len(m.group(1)), m.group(2).strip()
            if level == 1:
                title = words
            else:
                ident = slug(words)
                if level == 2:
                    toc.append((ident, re.sub(r"[`*]", "", words)))
                out.append(f'<h{level} id="{ident}">{inline(words)}</h{level}>')
            i += 1
        else:
            block = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("```", "#")):
                block.append(lines[i])
                i += 1
            if block[0].startswith("|"):
                out.append(_table(block))
            elif re.match(r" *- ", block[0]):
                out.append(_list(block))
            else:
                out.append(f"<p>{inline(' '.join(l.strip() for l in block))}</p>")
    return title, "\n".join(out), toc


def doc_page(source: Path, template: str) -> str:
    title, body, toc = markdown(source.read_text(encoding="utf-8"))
    nav = "".join(f'<a href="#{ident}">{html.escape(words)}</a>' for ident, words in toc)
    here = source.with_suffix(".html").name
    page = template.replace("{{title}}", html.escape(title)).replace("{{toc}}", nav).replace("{{body}}", body)
    # The docs menu marks the page it's on.
    return page.replace(f'href="{here}"', f'href="{here}" aria-current="page"')


# -- Build ------------------------------------------------------------------------------------------------------

def build(out: Path, app: str = "") -> list[Path]:
    """Write the site into `out` (emptied first) and return every file written."""
    app = app.rstrip("/")
    if out.exists():
        shutil.rmtree(out)
    written = []
    for src in sorted(SITE.rglob("*")):
        if src.is_dir() or src.name.startswith((".", "_")):
            continue
        dst = out / src.relative_to(SITE)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.suffix == ".md":
            dst = dst.with_suffix(".html")
            dst.write_text(doc_page(src, (src.parent / "_page.html").read_text(encoding="utf-8")).replace("{{app}}", app),
                           encoding="utf-8")
        elif src.suffix in FILLED:
            dst.write_text(src.read_text(encoding="utf-8").replace("{{app}}", app), encoding="utf-8")
        else:
            shutil.copyfile(src, dst)
        written.append(dst)
    (out / "ds" / "fonts").mkdir(parents=True, exist_ok=True)
    for name in DS_FILES:
        written.append(Path(shutil.copyfile(DS / name, out / "ds" / name)))
    for font in sorted((DS / "fonts").glob("*.woff2")):
        written.append(Path(shutil.copyfile(font, out / "ds" / "fonts" / font.name)))
    written.append(Path(shutil.copyfile(ICON, out / ICON.name)))
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "out" / "site", help="folder to write (emptied first)")
    parser.add_argument("--app", default="", help='where the demo app is served, e.g. https://demo.example.org ("" = same host)')
    args = parser.parse_args()
    files = build(args.out, args.app)
    print(f"Built {len(files)} files into {args.out}")


if __name__ == "__main__":
    main()
