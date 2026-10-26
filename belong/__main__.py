"""Command line.

  python3 -m belong run    [--n 300] [--seed 7]   simulate, write out/run.json, print summary
  python3 -m belong serve  [--port 8000] [--host 127.0.0.1]   run (if needed) and serve the demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .sim import run

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "run.json"


def _run(n: int, seed: int) -> dict:
    result = run(n=n, seed=seed)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    return result


def _summary(result: dict) -> None:
    m, lr = result["metrics"], result["leak_report"]
    print(f"People: {result['meta']['n']}   admitted: {m['admitted']}   rejected at entry: {m['entry_rejections']}")
    print(f"Proposals: {m['proposals']}   correct by ground truth: {m['proposals_correct']}   "
          f"precision: {m['precision']}   coverage: {m['coverage']}")
    print(f"Introductions: {m['introductions']}   connections: {m['connections']}   reveals: {m['reveals']}")
    status = "PASS - no hidden detail was revealed" if lr["ok"] else f"FAIL - {len(lr['findings'])} finding(s)"
    print(f"Leak report: {status}   checked: {lr['checked']}")
    for f in lr["findings"][:20]:
        print(f"  [{f['audit']}] {f['detail']}")
    print(f"Wrote {OUT.relative_to(ROOT)}")


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=VALUE lines from .env (git-ignored) into the environment, without
    overriding anything already set. Used for OPENAI_API_KEY and friends."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        os.environ.setdefault(key, value.strip().strip('"').strip("'"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="belong")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--n", type=int, default=300)
    r.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    # Loopback only unless told otherwise (the hosted container sets BELONG_HOST=0.0.0.0).
    s.add_argument("--host", default=os.environ.get("BELONG_HOST", "127.0.0.1"))
    args = ap.parse_args(argv)

    if args.cmd == "run":
        result = _run(args.n, args.seed)
        _summary(result)
        return 0 if result["leak_report"]["ok"] else 1
    load_dotenv()
    if os.environ.get("BELONG_REQUIRE_PASSWORD") and not os.environ.get("BELONG_PASSWORD"):
        # Hosted (deploy/belong.service, Dockerfile): never start open to everyone by mistake.
        print("BELONG_PASSWORD isn't set. Add it (e.g. to .env) before starting the hosted demo.", file=sys.stderr)
        return 2
    if not OUT.exists():
        _summary(_run(300, 7))
    from .server import serve
    serve(args.port, args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
