"""Command line (SPEC §8).

  python3 -m belong run    [--n 300] [--seed 7]   simulate, write out/run.json, print summary; then the same
                                                  week with agent dialects (synth.DIALECTS), summary only; then
                                                  both again with invented wishes (synth.INVENTED: a share of agents
                                                  add a must the person never had), with the rule that a must comes
                                                  from the person's own words (agent.ground_wishes, as in every run)
                                                  and without it, to show what it wins back, and with it while their
                                                  agent quotes the person's answer whatever it says (a no too), to show
                                                  what it doesn't; then the dialects' week
                                                  with nobody typing their musts with their values
                                                  (synth.OTHER_WORDS_SHARE): their descriptions still say most of
                                                  them, and the rest (counted in musts: sim.unsaid_musts) only their
                                                  yes to their agent's question keeps; with that yes and by values
                                                  alone, to show what reading values alone would cost on those musts
                                                  (neither with --matcher ai)
                           [--matcher ai] [--ai-pairs 80]   opt-in: the Matchmaker AI judges the fuzzy part of up to
                                                  that many pairs a round (it calls the
                                                  API BELONG_MATCHMAKER_URL names, OpenAI's by default, and costs
                                                  money); without it, the rules alone and no network, as in CI
                           [--matcher hostile]    a stand-in Matchmaker AI that says yes to every pair and answers out
                                                  of shape (probe.hostile_judge; no model, no network, every pair
                                                  unless --ai-pairs): the leak reports, which then audit every request
                                                  it was sent, still pass. Writes nothing.
  python3 -m belong serve  [--port 8000] [--host 127.0.0.1]   run (if needed) and serve the demo; its Matchmaker is
                                                  the AI or the rules by BELONG_MATCHMAKER (demo.DemoSession)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .matching.hybrid import MAX_PAIRS_PER_RUN
from .matchmaker_openai import JudgeUnavailable, OpenAIJudge
from .sim import RUN_FORMAT, run
from .synth import DIALECT_SHARE, INVENTED_SHARE, OTHER_WORDS_SHARE

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "out" / "run.json"


def _current(n: int) -> bool:
    """Is there already a batch run for n people (the demo's own) in today's layout, so the inspector
    matches the demo?"""
    try:
        meta = json.loads(OUT.read_text())["meta"]
        return meta["n"] == n and meta.get("format") == RUN_FORMAT
    except (OSError, ValueError, KeyError, TypeError):
        return False


def _run(n: int, seed: int, **options) -> dict:
    result = run(n=n, seed=seed, **options)
    OUT.parent.mkdir(exist_ok=True)
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False))
    tmp.replace(OUT)  # never a half-written file for the inspector
    return result


def _summary(result: dict, dialects: dict = None, wrote: bool = True, invented=(), other=()) -> None:
    """invented: (title, with the rule, without it, with it while their agent quotes every answer, a no too) for each
    invented-wish pass, printed after the first two; other: (title, with the person's answers, by values alone) for the
    pass where nobody types their musts with their values, printed last, with how many musts the people have and how
    many of them nothing they typed says (sim.unsaid_musts)."""
    m, lr = result["metrics"], result["leak_report"]
    print(f"People: {result['meta']['n']}   admitted: {m['admitted']}   rejected at entry: {m['entry_rejections']}")
    print(f"Proposals: {m['proposals']}   correct by ground truth: {m['proposals_correct']}   "
          f"precision: {m['precision']}   coverage: {m['coverage']}")
    print(f"Introductions: {m['introductions']}   connections: {m['connections']}   reveals: {m['reveals']}")
    used = result["details_in_use"]
    print("Details in use (shared list): " + "   ".join(f"{section}: {len(used[section])}" for section in
                                                       ("details", "scales", "values", "phrases", "places")))
    _leaks(lr)
    if dialects is not None:
        d = dialects["metrics"]
        print(f"With agent dialects ({dialects['meta']['dialects']:.0%} of people): proposals: {d['proposals']}   "
              f"correct by ground truth: {d['proposals_correct']}   precision: {d['precision']}   "
              f"coverage: {d['coverage']}")
        _leaks(dialects["leak_report"], "  ")
    for title, ruled, unruled, quoted in invented:
        for line, run_ in ((title, ruled),
                           ("  Without the rule that a must comes from the person's own words", unruled),
                           ("  With the rule, and their agent quoting the person's answer whatever it says (a no too)",
                            quoted)):
            m = run_["metrics"]
            print(f"{line}: proposals: {m['proposals']}   correct by ground truth: {m['proposals_correct']}   "
                  f"precision: {m['precision']}   coverage: {m['coverage']}   wishes made up: {m['invented_wishes']}   "
                  f"kept as ideally: {m['wishes_kept_ideally']}")
            _leaks(run_["leak_report"], "  ")
    for title, answered, alone in other:
        for line, run_ in ((title, answered), ("  Reading only the values, not the person's yes to their agent's question",
                                               alone)):
            m = run_["metrics"]
            print(f"{line}: proposals: {m['proposals']}   correct by ground truth: {m['proposals_correct']}   "
                  f"precision: {m['precision']}   coverage: {m['coverage']}   musts: {m['musts']}   "
                  f"not in what they typed: {m['musts_unsaid']}   kept as ideally: {m['wishes_kept_ideally']}")
            _leaks(run_["leak_report"], "  ")
    if wrote:
        print(f"Wrote {OUT.relative_to(ROOT)}")


def _leaks(lr: dict, indent: str = "") -> None:
    status = "PASS - no hidden detail was revealed" if lr["ok"] else f"FAIL - {len(lr['findings'])} finding(s)"
    print(f"{indent}Leak report: {status}   checked: {lr['checked']}")
    for f in lr["findings"][:20]:
        print(f"{indent}  [{f['audit']}] {f['detail']}")


def _ai_summary(judge) -> None:
    """What the Matchmaker AI did over both runs: requests, how many failed (their pairs decided by the rules), tokens."""
    u = judge.usage
    print(f"Matchmaker AI: {judge.describe()} (synthetic profiles only)   requests: "
          f"{u['calls']}   answered: {u['answered']}   failed, decided by the rules: {u['failed']}   tokens: "
          f"{u['prompt_tokens']} in, {u['completion_tokens']} out ({u['reasoning_tokens']} reasoning)")


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
    # The rules alone unless asked: the simulation and CI stay network-free.
    r.add_argument("--matcher", choices=("rules", "ai", "hostile"), default="rules")
    r.add_argument("--ai-pairs", type=int, default=None)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    # Loopback only unless told otherwise (the hosted container sets BELONG_HOST=0.0.0.0).
    s.add_argument("--host", default=os.environ.get("BELONG_HOST", "127.0.0.1"))
    args = ap.parse_args(argv)

    if args.cmd == "run":
        options = {}
        if args.matcher == "ai":
            load_dotenv()
            try:
                judge = OpenAIJudge()
            except JudgeUnavailable as e:
                print(f"The Matchmaker AI can't run: {e}", file=sys.stderr)
                return 2
            options = {"judge": judge, "max_pairs": max(0, MAX_PAIRS_PER_RUN if args.ai_pairs is None else args.ai_pairs)}
        elif args.matcher == "hostile":
            from .probe import hostile_judge  # a stand-in, no network: it costs nothing to send it every pair
            options = {"judge": hostile_judge, "max_pairs": 10 ** 9 if args.ai_pairs is None else max(0, args.ai_pairs)}
        hostile = args.matcher == "hostile"
        # A hostile stand-in's week isn't the inspector's: it's printed, never written.
        result = (run if hostile else _run)(n=args.n, seed=args.seed, **options)
        dialects = run(n=args.n, seed=args.seed, dialects=DIALECT_SHARE, **options)
        # The same two weeks with invented wishes, with the rule, without it, and with it while the agent that made a
        # must up quotes the person's answer whatever it says (a word rule can't tell a yes from a no); not with the
        # Matchmaker AI, whose requests cost money (the rules and the hostile stand-in cost nothing).
        invented = [] if args.matcher == "ai" else [
            (title, *(run(n=args.n, seed=args.seed, dialects=share, invented=INVENTED_SHARE, ground=ground,
                          quote_any=quote_any, **options)
                      for ground, quote_any in ((True, False), (False, False), (True, True))))
            for title, share in ((f"With invented wishes ({INVENTED_SHARE:.0%} of listings that aren't dating)", 0.0),
                                 (f"With invented wishes and agent dialects ({DIALECT_SHARE:.0%} of people)",
                                  DIALECT_SHARE))]
        # The dialects' week with nobody typing their musts with their values (their descriptions still say most), with
        # the person's yes when their agent asks whether one their words don't say must be met, and by values alone.
        other = [] if args.matcher == "ai" else [
            (f"With no must typed with its value, and agent dialects ({DIALECT_SHARE:.0%} of people)",
             *(run(n=args.n, seed=args.seed, dialects=DIALECT_SHARE, other_words=OTHER_WORDS_SHARE, answers=answers,
                   **options) for answers in (True, False)))]
        _summary(result, dialects, wrote=not hostile, invented=invented, other=other)
        reports = [result, dialects, *(r for _, *runs in [*invented, *other] for r in runs)]
        if hostile:
            sent = [str(r["leak_report"]["checked"].get("judge_requests", 0)) for r in reports]
            print("Matchmaker AI: a hostile stand-in (no model, no network) that says yes to every pair and answers out "
                  "of shape. Proposals and precision above are what the rules' hard rules let it make; the leak reports "
                  f"audited every request it was sent ({', '.join(sent[:-1])} and {sent[-1]}).")
        elif options:
            _ai_summary(options["judge"])
        return 0 if all(r["leak_report"]["ok"] for r in reports) else 1
    load_dotenv()
    if os.environ.get("BELONG_REQUIRE_PASSWORD") and not os.environ.get("BELONG_PASSWORD"):
        # Hosted (deploy/belong.service, Dockerfile): never start open to everyone by mistake.
        print("BELONG_PASSWORD isn't set. Add it (e.g. to .env) before starting the hosted demo.", file=sys.stderr)
        return 2
    from .server import PEOPLE, serve
    if not _current(PEOPLE):
        # The inspector shows the same people as the demo. Built in the background, so the site
        # starts straight away; the inspector says it's being prepared until then.
        import threading
        OUT.unlink(missing_ok=True)
        threading.Thread(target=lambda: _summary(_run(PEOPLE, 7)), daemon=True).start()
    serve(args.port, args.host)
    return 0


if __name__ == "__main__":
    sys.exit(main())
