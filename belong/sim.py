"""End-to-end simulation of one week (SPEC §8).

  Day 0  Each personal agent drafts a listing (last changed some days ago); the Warden's entry check
         admits it or not. The Warden builds the shared list of details in use.
  Day 1  Matchmaker round 1 -> follow-up questions -> Warden exit -> agents answer
         -> updated listings re-enter through the Warden.
  Day 2  The day's shared list. Matchmaker round 2 -> proposals -> Warden exit -> introductions.
  Day 3  People say yes/no. Connected pairs are asked about mutual reveal.
  Then   Leak report (every shared list included) + match-quality evaluation.

With dialects, a share of the synthetic people's agents write a detail differently from most agents
(synth.DIALECTS): the same week, to show that costs matches but never makes a wrong one.

With a judge (`belong run --matcher ai`, or `--matcher hostile` for a stand-in that answers as badly as it can; never by
default), the Matchmaker is the hybrid (matching/hybrid.py): the same rules narrow and decide everything hard, and the
judge decides the fuzzy part of at most max_pairs pairs a round. Each request goes to it only once it passes the Warden's
check (warden.exit_judge_request, a receipt each), and the leak report then also audits every request the judge was
sent (leaks.py, A8).

Returns one JSON-serialisable dict, written to out/run.json for the web inspector.
"""
from __future__ import annotations

import hashlib
import random

from . import agent
from .evaluate import compatible, evaluate
from .leaks import leak_report
from .matching.hybrid import MAX_PAIRS_PER_RUN, LLMMatcher
from .matching.marketboard import Marketboard
from .matching.matcher import OfflineMatcher
from .schema import value_to_json
from .synth import draft_listing, generate
from .warden import Clock, Warden, checked_judge

# The layout of out/run.json; a file in another layout is rebuilt rather than shown (__main__._current).
RUN_FORMAT = 2


def run(n: int = 300, seed: int = 7, matcher=None, k_crowd: int = 20, approach_limit: int = 15,
        per_round: int = 3, dialects: float = 0.0, judge=None, max_pairs: int = MAX_PAIRS_PER_RUN) -> dict:
    """judge: the Matchmaker AI's judge (request -> answer), for the hybrid Matchmaker; None runs the rules alone."""
    rng = random.Random(seed + 1)
    personas = {p.person.id: p for p in generate(n, seed, dialects)}
    # The Warden allows up to approach_limit introductions a day; the Matchmaker proposes at most
    # per_round for each person in one round, so nobody gets them all at once.
    if matcher is None:
        matcher = OfflineMatcher(k_crowd=k_crowd, max_per_person=per_round)
        if judge is not None:
            matcher = LLMMatcher(judge, rules=matcher, max_pairs_per_run=max_pairs)
    clock = Clock()
    # In the real room this key belongs to the auditors; here it's derived from the seed
    # so runs are reproducible.
    warden = Warden(clock, key=hashlib.sha256(f"audit-key-{seed}".encode()).digest(),
                    approach_limit_per_day=approach_limit)
    board = Marketboard()
    if judge is not None and isinstance(matcher, LLMMatcher) and matcher.judge is judge:
        # Every request goes to the judge only once it passes the Warden's check, as in the demo.
        matcher.judge = checked_judge(warden, judge, lambda: board, lambda: matcher.payloads, lambda: matcher.labels)

    # Day 0: entry
    entry_log = {}
    for pid, p in personas.items():
        listing = draft_listing(p, clock.now())
        entry_log[pid] = warden.admit(p.person, listing, board)
    admitted = {pid for pid, r in entry_log.items() if r.accepted}
    # Each description as it went live, which is what the live demo has: answers to this run's
    # follow-up questions are added to it below, and the inspector marks them as such.
    as_entered = {pid: board.listing(pid).intent_text for pid in admitted}
    # The shared list every agent reads (it counts listings unchanged for a week, so it has entries
    # from the first day). Synthetic agents write their details already; each list is audited below.
    details_lists = [warden.details_in_use(board)]

    # Day 1: follow-ups
    clock.advance(days=1)
    round1 = matcher.match(board)
    # Follow-ups that wait for someone else's "no" aren't asked in this one-pass simulation.
    followups_out = [fo for fo in (warden.exit_followup(f, board) for f in round1.followups if not f.after) if fo]
    for fo in followups_out:
        p = personas[fo.person_id]
        updated = agent.answer_followup(p, board.listing(fo.person_id), fo.question_id, clock.now(), rng, fo.phrase)
        warden.admit(p.person, updated, board)

    # Day 2: proposals -> introductions
    clock.advance(days=1)
    details_lists.append(warden.details_in_use(board))
    round2 = matcher.match(board)
    intros = [i for i in (warden.exit_introduction(pr, board) for pr in round2.proposals) if i]

    # Day 3: decisions and mutual reveal
    clock.advance(days=1)
    decisions, connections, consents, reveals = {}, set(), {}, []
    for intro in intros:
        a, b = personas[intro.person_a], personas[intro.person_b]
        good = compatible(a, b)
        # A close match goes to its one recipient first; the other person only sees it after a yes.
        order = sorted((intro.person_a, intro.person_b), key=lambda pid: pid != intro.one_way_to)
        for pid in order:
            yes = rng.random() < (0.85 if good else 0.25)
            decisions[(intro.intro_id, pid)] = yes
            warden.record_decision(intro.intro_id, pid, yes)
            if intro.one_way_to and not yes:
                break
        if warden.is_connected(intro.intro_id):
            connections.add(frozenset((intro.person_a, intro.person_b)))
            ca = agent.reveal_consents(board.listing(intro.person_a), rng)
            cb = agent.reveal_consents(board.listing(intro.person_b), rng)
            consents[(intro.intro_id, intro.person_a)] = ca
            consents[(intro.intro_id, intro.person_b)] = cb
            reveals += warden.exit_reveal(intro.intro_id, ca, cb, board)

    intro_dicts = [i.to_dict() for i in intros]
    followup_dicts = [f.to_dict() for f in followups_out]
    reveal_dicts = [r.to_dict() for r in reveals]
    list_dicts = [d.to_dict() for d in details_lists]
    receipt_dicts = [r.to_dict() for r in warden.receipts]

    # With the Matchmaker AI, every request its judge was sent is audited too (A8), against the board as it was then.
    report = leak_report(board, intro_dicts, followup_dicts, reveal_dicts, receipt_dicts,
                         round2.proposals, connections, consents, k_crowd, approach_limit, details_lists=list_dicts,
                         judge_requests=matcher.sent_requests() if isinstance(matcher, LLMMatcher) else None)
    metrics = evaluate(personas, admitted, round2.proposals, intro_dicts, connections, reveal_dicts)
    metrics["entry_rejections"] = _count_rejections(entry_log)

    return {
        "meta": {"n": n, "seed": seed, "k_crowd": k_crowd, "approach_limit": approach_limit,
                 "matcher": type(matcher).__name__, "dialects": dialects, "format": RUN_FORMAT,
                 # The last round's counts of how each pair was decided, with the Matchmaker AI (hybrid.last_run).
                 **({"matchmaker_ai": dict(matcher.last_run)} if isinstance(matcher, LLMMatcher) else {})},
        "leak_report": report,
        "metrics": metrics,
        "people": [_person_row(p, board, entry_log[pid], as_entered.get(pid)) for pid, p in personas.items()],
        "proposals_internal": [_proposal_row(pr, personas) for pr in round2.proposals],
        "introductions": [{**d, "a_said_yes": decisions.get((d["intro_id"], d["person_a"])),
                           "b_said_yes": decisions.get((d["intro_id"], d["person_b"]))} for d in intro_dicts],
        "followups": followup_dicts,
        "reveals": reveal_dicts,
        # The list the agents read on the last day (vocabulary, built_at, board_change); all were audited (A7).
        "details_in_use": list_dicts[-1],
        # The inspector shows a tally per check and the latest 100; the full chain was audited above (A5, A6).
        "receipts": receipt_dicts[-100:],
        "receipts_total": len(receipt_dicts),
        "receipt_tally": _tally(receipt_dicts),
    }


def _tally(receipts) -> dict:
    out = {}
    for r in receipts:
        out.setdefault(r["check"], {"pass": 0, "fail": 0})[r["result"]] += 1
    return out


def _count_rejections(entry_log) -> dict:
    out = {}
    for r in entry_log.values():
        for c in r.failed_checks:
            out[c] = out.get(c, 0) + 1
    return out


def _person_row(p, board, entry, as_entered=None) -> dict:
    pid = p.person.id
    listing = board.listing(pid) if pid in board else None
    text = listing.intent_text if listing else p.intent_text
    # What this run's follow-up answers added to the description ("Aged 25 to 39."); the live demo's
    # listing is the description as it went live, without them.
    answered = text[len(as_entered):].strip() if as_entered and text != as_entered and text.startswith(as_entered) else ""
    kind = listing.get("kind") if listing else None
    return {
        "id": pid,
        "verified": p.person.verified,
        "admitted": entry.accepted,
        "entry_problems": entry.problems,
        "kind": kind.value if kind else p.values.get("kind"),  # the kind of match on their listing
        "story": p.truth.kind,  # ground truth (synth.KINDS), never on the listing
        "dialect": p.dialect,
        "intent_text": text,  # as matching read it at the end of this run
        "intent_as_entered": as_entered if as_entered is not None else text,  # as in the live demo
        "answered_in_run": answered,
        "exactness": listing.exactness if listing else p.truth.exactness,
        "notify_close": bool(listing and listing.notify_close),
        "notify_questions": bool(listing and listing.notify_questions),
        "updated_at": listing.updated_at if listing else None,
        "fields": [_field_row(f) for f in listing.fields] if listing else [],
    }


def _field_row(f) -> dict:
    """A field for the inspector; an open detail also says what it's about and how it compares."""
    row = {"key": f.key, "value": value_to_json(f.value), "visibility": f.visibility, "sensitive": f.sensitive}
    if f.about:
        row.update(about=f.about, name=f.name, compare=f.compare, steps=list(f.steps), required=f.required)
    return row


def _proposal_row(pr, personas) -> dict:
    return {
        "person_a": pr.person_a, "person_b": pr.person_b,
        "score": pr.score, "shown_score": pr.shown_score, "shown_threshold": pr.shown_threshold,
        "reason_fields": list(pr.reason_fields), "used_hidden_fields": list(pr.used_hidden_fields),
        "truly_compatible": compatible(personas[pr.person_a], personas[pr.person_b]),
        "close": pr.close,
    }
