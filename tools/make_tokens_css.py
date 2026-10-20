"""Turn the tokens.json of the Belong. design system into belong/web/ds/tokens.css.

The design system is the "Belong." Design System artifact
(https://claude.ai/artifact/KDPcQmxczT7iQ7Kcgt1et6). Download its project/ folder
(tokens.json, tokens/theme-dark.css, fonts/), then run:

    python3 tools/make_tokens_css.py <path to that project/ folder> belong/web/ds

This copies the fonts and the dark theme, and writes tokens.css with every token as a
CSS variable. belong/web/ds/belong.css (the components) is written by hand from tokens.
"""
import json
import re
import shutil
import sys
from pathlib import Path

src = Path(sys.argv[1])  # .../project
out = Path(sys.argv[2])  # belong/web/ds
out.mkdir(parents=True, exist_ok=True)
(out / "fonts").mkdir(exist_ok=True)

t = json.loads((src / "tokens.json").read_text())


def css_value(v):
    if isinstance(v, dict):
        v = next(iter(v.values()))
    v = str(v)
    m = re.fullmatch(r"\{([A-Za-z0-9_.-]+)\}", v)
    return f"var(--{m.group(1)})" if m else v


lines = ["/* Belong. design system: generated from tokens.json (design system artifact",
         "   https://claude.ai/artifact/KDPcQmxczT7iQ7Kcgt1et6). Do not edit by hand; regenerate. */", ""]

for f in t["type"]["fonts"]:
    name = Path(f["file"]).name
    shutil.copy(src / f["file"], out / "fonts" / name)
    lines.append("@font-face { font-family: '%s'; src: url('fonts/%s') format('woff2'); font-weight: %s; "
                 "font-style: %s; font-display: swap; }" % (f["family"], name, f["weight"], f.get("style", "normal")))

lines += ["", ":root {"]
for fam in ("color", "spacing", "radius", "shadow", "other", "fontWeight", "lineHeight", "letterSpacing", "timing"):
    lines.append(f"  /* {fam} */")
    for tok in t[fam]["tokens"]:
        lines.append(f"  --{tok['name']}: {css_value(tok['value'])};")
lines.append("  /* type */")
fams = t["type"]["families"]
for key in ("display", "body", "mono"):
    lines.append(f"  --font-{key}: {fams[key]};")
for group in t["type"]["groups"]:
    family = f"var(--font-{group['family']})"
    for s in group["styles"]:
        lines.append(f"  --text-{s['name']}: {s['fontWeight']} {s['fontSize']}/{s['lineHeight']} {family};")
lines.append("}")
(out / "tokens.css").write_text("\n".join(lines) + "\n")
shutil.copy(src / "tokens" / "theme-dark.css", out / "theme-dark.css")
print("wrote", out / "tokens.css", len(lines), "lines")
