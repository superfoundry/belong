"""End-to-end simulation of one week (SPEC §8).

  Day 0  Each personal agent drafts a listing; the Warden's entry check admits it or not.
  Day 1  Matchmaker round 1 -> follow-up questions -> Warden exit -> agents answer
         -> updated listings re-enter through the Warden.
  Day 2  Matchmaker round 2 -> proposals -> Warden exit -> introductions.
  Day 3  People say yes/no. Connected pairs are asked about mutual reveal.
  Then   Leak report + match-quality evaluation.

Returns one JSON-serialisable dict, written to out/run.json for the web inspector.
"""
from __future__ import annotations

import hashlib
import random

from . import agent
from .evaluate import compatible, evaluate
from .leaks import leak_report
from .matching.marketboard import Marketboard
from .matching.matcher import OfflineMatcher
from .schema import value_to_json
from .synth import generate
from .warden import Clock, Warden


def run(n: int = 300, seed: int = 7, matcher=None, k_crowd: int = 20, approach_limit: int = 15,
        per_round: int = 3) -> dict:
    rng = random.Random(seed + 1)
    personas = {p.person.id: p for p in generate(n, seed)}
    # The Warden allows up to approach_limit introductions a day; the Matchmaker proposes at most
    # per_round for each person in one round, so nobody gets them all at once.
    matcher = matcher or OfflineMatcher(k_crowd=k_crowd, max_per_person=per_round)
    clock = Clock()
    # In the real room this key belongs to the auditors; here it's derived from the seed
    # so runs are reproducible.
    warden = Warden(clock, key=hashlib.sha256(f"audit-key-{seed}".encode()).digest(),
                    approach_limit_per_day=approach_limit)
    board = Marketboard()

    # Day 0: entry
    entry_log = {}
    for pid, p in personas.items():
        listing = agent.draft_listing(p, clock.now())
        entry_log[pid] = warden.admit(p.person, listing, board)
    admitted = {pid for pid, r in entry_log.items() if r.accepted}
    # Each description as it went live, which is what the live demo has: answers to this run's
    # follow-up questions are added to it below, and the inspector marks them as such.
    as_entered = {pid: board.listing(pid).intent_text for pid in admitted}

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
    receipt_dicts = [r.to_dict() for r in warden.receipts]

    report = leak_report(board, intro_dicts, followup_dicts, reveal_dicts, receipt_dicts,
                         round2.proposals, connections, consents, k_crowd, approach_limit)
    metrics = evaluate(personas, admitted, round2.proposals, intro_dicts, connections, reveal_dicts)
    metrics["entry_rejections"] = _count_rejections(entry_log)

    return {
        "meta": {"n": n, "seed": seed, "k_crowd": k_crowd, "approach_limit": approach_limit,
                 "matcher": type(matcher).__name__},
        "leak_report": report,
        "metrics": metrics,
        "people": [_person_row(p, board, entry_log[pid], as_entered.get(pid)) for pid, p in personas.items()],
        "proposals_internal": [_proposal_row(pr, personas) for pr in round2.proposals],
        "introductions": [{**d, "a_said_yes": decisions.get((d["intro_id"], d["person_a"])),
                           "b_said_yes": decisions.get((d["intro_id"], d["person_b"]))} for d in intro_dicts],
        "followups": followup_dicts,
        "reveals": reveal_dicts,
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
    return {
        "id": pid,
        "verified": p.person.verified,
        "admitted": entry.accepted,
        "entry_problems": entry.problems,
        "kind": p.truth.kind,
        "intent_text": text,  # as matching read it at the end of this run
        "intent_as_entered": as_entered if as_entered is not None else text,  # as in the live demo
        "answered_in_run": answered,
        "exactness": listing.exactness if listing else p.truth.exactness,
        "notify_close": bool(listing and listing.notify_close),
        "notify_questions": bool(listing and listing.notify_questions),
        "fields": [{"key": f.key, "value": value_to_json(f.value), "visibility": f.visibility,
                    "sensitive": f.sensitive} for f in listing.fields] if listing else [],
    }


def _proposal_row(pr, personas) -> dict:
    return {
        "person_a": pr.person_a, "person_b": pr.person_b,
        "score": pr.score, "shown_score": pr.shown_score, "shown_threshold": pr.shown_threshold,
        "reason_fields": list(pr.reason_fields), "used_hidden_fields": list(pr.used_hidden_fields),
        "truly_compatible": compatible(personas[pr.person_a], personas[pr.person_b]),
        "close": pr.close,
    }
