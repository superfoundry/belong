"""Compare the rules with the Matchmaker AI on the same pairs, as feedback for the rules (SPEC §8.3).

    python3 tools/compare_judges.py [--n 300] [--seed 7] [--pairs 200] [--outside 40] [--yes]

On a board of synthetic people (synth.generate(n, seed)), the rules (OfflineMatcher) and the Matchmaker AI's judge
(matchmaker_openai.OpenAIJudge, asked through hybrid.LLMMatcher.ask: the same requests, with the same redaction and the
same strict reading of the answer as in matching) judge the same shortlisted pairs, and the AI also judges a sample of
pairs narrowing never shortlisted. It prints where the two disagree, by bucket, with the simulation's ground truth
(evaluate.compatible) saying who is right, and the tokens used:

  rules yes, AI no               the rules propose a two-way match; the AI doesn't say "match" both ways
  AI yes, rules no               the AI says "match" both ways; the rules don't propose it (split by whether the hybrid's
                                 hard rules allow it, that is whether the hybrid could propose it)
  AI yes outside the shortlists  a pair narrowing never shortlisted that the AI says "match" both ways

Opt-in and live: it needs the Matchmaker AI's settings (OPENAI_API_KEY, or BELONG_MATCHMAKER_URL and
BELONG_MATCHMAKER_KEY; .env is loaded as `belong serve` loads it), it costs money, and it sends nothing without --yes
(without it, it says how many pairs and requests it would send). Synthetic profiles only. Tests run compare() and
main() with stub judges, never against a live model.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from belong.evaluate import compatible  # noqa: E402
from belong.leaks import leak_report  # noqa: E402
from belong.matching.hybrid import MATCH, MAX_PER_REQUEST, LLMMatcher  # noqa: E402
from belong.matching.marketboard import Marketboard  # noqa: E402
from belong.matching.matcher import OfflineMatcher  # noqa: E402
from belong.schema import K_CROWD  # noqa: E402
from belong.synth import draft_listing, generate  # noqa: E402
from belong.warden import Clock, Warden, checked_judge  # noqa: E402

BUCKETS = ("rules_yes_ai_no", "ai_yes_rules_no", "ai_yes_outside")
TITLES = {"rules_yes_ai_no": "Rules yes, AI no", "ai_yes_rules_no": "AI yes, rules no",
          "ai_yes_outside": "AI yes outside the shortlists"}
SHOWN_ROWS = 10  # example pairs printed per bucket
OUTSIDE_EACH = 8  # pairs outside the shortlists asked about for one listing


def board_of(n: int, seed: int) -> tuple:
    """(synthetic people by id, a board of every listing the Warden admits, the Warden), as the simulation's first
    day."""
    personas = {p.person.id: p for p in generate(n, seed)}
    clock = Clock()
    warden = Warden(clock, key=hashlib.sha256(f"compare-judges-{seed}".encode()).digest())
    board = Marketboard()
    for p in personas.values():
        warden.admit(p.person, draft_listing(p, clock.now()), board)
    return personas, board, warden


def plan(board, pairs: int, outside: int, seed: int) -> tuple:
    """(the rules, their reading of the board, the shortlisted pairs to compare, the pairs outside every shortlist to
    ask the AI about), each a sample of at most that many, the same every time for the same seed. The shortlisted
    pairs are whole shortlists, of listings taken in a random order, so they go to the AI as few requests."""
    rules = OfflineMatcher()
    run = rules.read(board)
    rng = random.Random(seed)
    focus = sorted(run.lists)
    rng.shuffle(focus)
    chosen, in_lists = set(), set(run.pairs)
    for f in focus:
        for b in run.lists[f]:
            pair = (f, b) if f < b else (b, f)
            if len(chosen) < pairs and pair in in_lists:
                chosen.add(pair)
    chosen = sorted(chosen)
    # Outside the shortlists: for the same listings in the same order, up to OUTSIDE_EACH others narrowing never put
    # with them (never one of their own person's), so they go in the same requests.
    ids, others = sorted(run.listings), set()
    for f in focus:
        if len(others) >= outside:
            break
        pool = [b for b in ids if b != f and ((f, b) if f < b else (b, f)) not in in_lists
                and run.listings[b].owner != run.listings[f].owner]
        for b in rng.sample(pool, min(len(pool), OUTSIDE_EACH, outside - len(others))):
            others.add((f, b) if f < b else (b, f))
    return rules, run, chosen, sorted(others)


def _groups(pairs, per_request: int) -> list:
    """The pairs as few requests as can be: again and again, the listing in the most pairs left is a focus listing
    with all its pairs' other listings as candidates, at most per_request in one request."""
    left, out = set(pairs), []
    while left:
        count = Counter(x for pair in left for x in pair)
        focus = min(count, key=lambda x: (-count[x], x))
        mine = sorted(pair[1] if pair[0] == focus else pair[0] for pair in left if focus in pair)
        left = {pair for pair in left if focus not in pair}
        out += [(focus, mine[i:i + per_request]) for i in range(0, len(mine), per_request)]
    return out


def compare(n: int, seed: int, judge, pairs: int = 200, outside: int = 40, per_request: int = MAX_PER_REQUEST) -> dict:
    """Ask the judge about the same pairs the rules decide, and sort the disagreements into BUCKETS (see the module's
    docstring). Each row: the pair (the two ids in order), the ground truth, the AI's verdicts (what the first wants,
    what the second wants, the fit) and whether the hybrid's hard rules allow the pair."""
    personas, board, warden = board_of(n, seed)
    rules, run, chosen, others = plan(board, pairs, outside, seed)
    matcher = LLMMatcher(judge, rules=rules)
    # Every request goes to the judge only once it passes the Warden's check, as in matching.
    matcher.judge = checked_judge(warden, judge, lambda: board, lambda: matcher.payloads, lambda: matcher.labels)
    verdicts, requests, failed = {}, 0, 0
    for focus, cands in _groups(chosen + others, per_request):
        requests += 1
        got = matcher.ask(board, focus, cands)
        if got is None:
            failed += 1
            continue
        for b, (fw, cw, fit) in got[0].items():  # kept as (the first listing's wants, the second's, fit)
            verdicts[(focus, b) if focus < b else (b, focus)] = (fw, cw, fit) if focus < b else (cw, fw, fit)
    buckets = {name: [] for name in BUCKETS}
    agree = {"both_yes": 0, "both_no": 0}
    outside_set = set(others)
    for pair in chosen + others:
        verdict = verdicts.get(pair)
        if verdict is None:
            continue
        a, b = pair
        ai_yes = verdict[0] == MATCH and verdict[1] == MATCH
        ea, eb, _, _ = matcher._hard(run, a, b)
        row = {"pair": [a, b], "truth": compatible(personas[a], personas[b]), "ai": list(verdict),
               "hard_rules_allow": bool(ea and eb)}
        if pair in outside_set:
            if ai_yes:
                buckets["ai_yes_outside"].append(row)
            continue
        proposal, _ = rules.decide(run, a, b)
        rules_yes = proposal is not None and not proposal.close
        if rules_yes and not ai_yes:
            buckets["rules_yes_ai_no"].append(row)
        elif ai_yes and not rules_yes:
            buckets["ai_yes_rules_no"].append(row)
        else:
            agree["both_yes" if rules_yes else "both_no"] += 1
    usage = getattr(judge, "usage", None)
    # What the AI was sent, through the same independent audit as in matching (leaks.py, A8).
    audit = leak_report(board, [], [], [], [], [], set(), {}, K_CROWD, 15, judge_requests=matcher.sent_requests())
    return {"board": {"n": n, "seed": seed, "listings": len(run.listings), "shortlisted_pairs": len(run.pairs)},
            "asked": {"shortlisted": len(chosen), "outside": len(others), "requests": requests, "failed": failed,
                      "answered": len(verdicts)},
            "agree": agree, "buckets": buckets, "usage": dict(usage) if isinstance(usage, dict) else None,
            "audit": {"ok": audit["ok"], "findings": audit["findings"][:20]}}


def report(result: dict) -> str:
    """The comparison as text: each bucket with who is right by the ground truth, and a few example pairs."""
    b, asked, agree = result["board"], result["asked"], result["agree"]
    lines = [f"Board: {b['listings']} listings (generate({b['n']}, {b['seed']})), {b['shortlisted_pairs']} shortlisted "
             f"pairs. Asked the AI about {asked['shortlisted']} of them and {asked['outside']} outside the shortlists in "
             f"{asked['requests']} requests ({asked['failed']} failed); {asked['answered']} pairs answered.",
             f"Agree: both yes {agree['both_yes']}, both no {agree['both_no']}."]
    for name in BUCKETS:
        rows = result["buckets"][name]
        right = sum(1 for r in rows if r["truth"])  # compatible: whoever said yes is right
        who = "the rules" if name == "rules_yes_ai_no" else "the AI"
        extra = ""
        if name == "ai_yes_rules_no":
            allowed = sum(1 for r in rows if r["hard_rules_allow"])
            extra = f"; the hybrid's hard rules allow {allowed} (it could propose them), refuse {len(rows) - allowed}"
        lines.append(f"{TITLES[name]}: {len(rows)}; the ground truth says {who} {'are' if who == 'the rules' else 'is'} "
                     f"right on {right}, wrong on {len(rows) - right}{extra}.")
        for r in rows[:SHOWN_ROWS]:
            fw, cw, fit = r["ai"]
            lines.append(f"  {r['pair'][0]} + {r['pair'][1]}: truth {'compatible' if r['truth'] else 'not compatible'}; "
                         f"AI {fw}/{cw}, fit {fit:.2f}; hard rules {'allow' if r['hard_rules_allow'] else 'refuse'}")
    a = result["audit"]
    lines.append("Every request passed the leak audit (A8)." if a["ok"] else
                 f"The leak audit found {len(a['findings'])} problem(s) in what was sent: {a['findings'][:3]}")
    u = result["usage"]
    if u:
        lines.append(f"Tokens: {u.get('prompt_tokens', 0)} in, {u.get('completion_tokens', 0)} out "
                     f"({u.get('reasoning_tokens', 0)} reasoning, {u.get('cached_tokens', 0)} cached); calls "
                     f"{u.get('calls', 0)}, failed {u.get('failed', 0)}.")
    return "\n".join(lines)


def main(argv=None, judge_factory=None) -> int:
    ap = argparse.ArgumentParser(prog="compare_judges")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--pairs", type=int, default=200, help="shortlisted pairs to compare (a sample)")
    ap.add_argument("--outside", type=int, default=40, help="pairs outside every shortlist to ask the AI about")
    ap.add_argument("--yes", action="store_true", help="send the requests (live: it costs money)")
    args = ap.parse_args(argv)
    if not args.yes:
        _, board, _ = board_of(args.n, args.seed)
        _, run, chosen, others = plan(board, max(0, args.pairs), max(0, args.outside), args.seed)
        print(f"Would ask the Matchmaker AI about {len(chosen)} shortlisted pairs and {len(others)} outside the "
              f"shortlists, in {len(_groups(chosen + others, MAX_PER_REQUEST))} requests. Nothing was sent: add --yes "
              "to send them (live, it costs money).")
        return 0
    if judge_factory is None:
        from belong.__main__ import load_dotenv
        from belong.matchmaker_openai import OpenAIJudge
        load_dotenv()
        judge_factory = OpenAIJudge
    from belong.matchmaker_openai import JudgeUnavailable
    try:
        judge = judge_factory()
    except JudgeUnavailable as e:
        print(f"The Matchmaker AI can't run: {e}", file=sys.stderr)
        return 2
    print(report(compare(args.n, args.seed, judge, max(0, args.pairs), max(0, args.outside))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
