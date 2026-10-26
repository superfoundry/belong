"""Tests an outside agent can run against Belong. (the trust page's "test it yourself").

Each probe builds a fresh SANDBOX: the same Warden code, the same Matchmaker and
300 synthetic people. Probes never touch real listings.

  canary_test(body)   Plant secrets in hidden fields, run a full matching week, then
                      search EVERYTHING that was sent out for them. Any hit = leak.
                      Agents may bring their own listings and secrets.
  attack_suite(body)  Try to get bad things in and private things out: identifying
                      intents, forged listings, a compromised Matchmaker, forged
                      outputs, reveals without consent. Every attempt must be blocked.
"""
from __future__ import annotations

import hashlib
import json
import random
import secrets
from dataclasses import replace

from . import agent
from .leaks import leak_report
from .matching.contract import FollowUp, Proposal
from .matching.marketboard import Marketboard
from .matching.matcher import OfflineMatcher
from .schema import FIELD_REGISTRY, HIDDEN, SHOWN, Field, Listing, Person
from .synth import generate
from .warden import Clock, Introduction, Reveal, Warden, specific_person_problems, verify_chain

MAX_CANARIES = 50
MAX_CUSTOM = 20


def _sandbox(seed: int):
    personas = {p.person.id: p for p in generate(300, seed)}
    clock = Clock()
    warden = Warden(clock, key=secrets.token_bytes(32))
    board = Marketboard()
    for p in personas.values():
        warden.admit(p.person, agent.draft_listing(p, clock.now()), board)
    return personas, clock, warden, board


def _token() -> str:
    # Letters only: long digit runs would (correctly) trip the phone-number rule.
    return "cnry" + "".join(secrets.choice("abcdefghijkmnpqrstuvwxyz") for _ in range(12))


# --------------------------------------------------------------------------------------
# Canary test
# --------------------------------------------------------------------------------------

def _listing_from_json(pid: str, d: dict, now: str) -> Listing:
    fields = []
    for f in d.get("fields", [])[:30]:
        spec = FIELD_REGISTRY.get(f.get("key"))
        if spec is None or f.get("value") in (None, "", []):
            continue
        value = tuple(map(str, f["value"])) if spec.multi and isinstance(f["value"], list) else str(f["value"])
        fields.append(Field(f["key"], value, HIDDEN if f.get("visibility") == HIDDEN else SHOWN, spec.sensitive))
    try:
        exactness = float(d.get("exactness", 0.5))
    except (TypeError, ValueError):
        exactness = 0.5
    return Listing(pid, str(d.get("intent_text", ""))[:500], exactness, tuple(fields), now)


def canary_test(body: dict) -> dict:
    seed = int(body.get("seed", random.randrange(1_000_000)))
    count = max(0, min(int(body.get("count", 10)), MAX_CANARIES))
    custom = (body.get("listings") or [])[:MAX_CUSTOM]
    personas, clock, warden, board = _sandbox(seed)
    rng = random.Random(seed)

    planted = {}  # person id -> list of secret strings
    canary_ids = []
    rejected_canaries = []
    templates = [p for p in personas.values() if p.person.id in board]
    for i in range(count):
        template = rng.choice(templates)
        pid = f"canary{i:02d}"
        base = board.listing(template.person.id)
        tokens = [_token(), _token(), _token()]
        fields = [f for f in base.fields if f.key not in ("employer", "neighborhood")]
        fields += [Field("employer", tokens[0], HIDDEN, False), Field("neighborhood", tokens[1], HIDDEN, False)]
        listing = replace(base, person_id=pid, fields=tuple(fields),
                          intent_text=f"{base.intent_text} Private note {tokens[2]}.")
        person = replace(template.person, id=pid, verified=True)
        entry = warden.admit(person, listing, board)
        if entry.accepted:
            planted[pid] = tokens
            canary_ids.append(pid)
        else:
            rejected_canaries.append({"canary": pid, "problems": entry.problems})

    rejected_custom = []
    for i, d in enumerate(custom):
        pid = f"agent{i:02d}"
        listing = _listing_from_json(pid, d, clock.now())
        band = next((f.value for f in listing.fields if f.key == "age_band"), "30-34")
        result = warden.admit(Person(pid, True, band, clock.now()), listing, board)
        if result.accepted:
            own = [str(s) for s in (d.get("secrets") or [])][:20]
            own += [s for f in listing.fields if f.visibility == HIDDEN for s in f.strings()]
            planted[pid] = sorted(set(own))
            canary_ids.append(pid)
        else:
            rejected_custom.append({"index": i, "problems": result.problems})

    # One full week: questions, matching, introductions, decisions, reveals.
    clock.advance(days=1)
    matcher = OfflineMatcher()
    result = matcher.match(board)
    followups = [f.to_dict() for f in (warden.exit_followup(q, board) for q in result.followups) if f]
    intros = [i for i in (warden.exit_introduction(p, board) for p in result.proposals) if i]
    connections, consents, reveals = set(), {}, []
    for intro in intros:
        # A close match goes to its one recipient first; the other person sees it only after a yes.
        for pid in sorted((intro.person_a, intro.person_b), key=lambda pid: pid != intro.one_way_to):
            yes = pid in planted or rng.random() < 0.8
            warden.record_decision(intro.intro_id, pid, yes)
            if intro.one_way_to and not yes:
                break
        if warden.is_connected(intro.intro_id):
            connections.add(frozenset((intro.person_a, intro.person_b)))
            for pid in (intro.person_a, intro.person_b):
                # Canaries never consent to a reveal, so their secrets must never appear in one.
                consents[(intro.intro_id, pid)] = set() if pid in planted else \
                    agent.reveal_consents(board.listing(pid), rng)
            reveals += warden.exit_reveal(intro.intro_id, consents[(intro.intro_id, intro.person_a)],
                                          consents[(intro.intro_id, intro.person_b)], board)

    outputs = {
        "introductions": [i.to_dict() for i in intros],
        "followups": followups,
        "reveals": [r.to_dict() for r in reveals],
        "receipts": [r.to_dict() for r in warden.receipts],
    }
    blob = json.dumps(outputs, ensure_ascii=False).lower()
    found = [{"owner": pid, "secret": s} for pid, toks in planted.items() for s in toks if s.lower() in blob]
    report = leak_report(board, outputs["introductions"], followups, outputs["reveals"], outputs["receipts"],
                         result.proposals, connections, consents, 20, 3)
    chain_ok, _ = verify_chain(outputs["receipts"])
    received = {pid: [] for pid in canary_ids}
    for d in outputs["introductions"]:
        for pid, key in ((d["person_a"], "a_view"), (d["person_b"], "b_view")):
            if pid in received:
                received[pid].append(d[key])
    return {
        "ok": not found and report["ok"] and chain_ok,
        "sandbox_seed": seed,
        "secrets_planted": sum(len(t) for t in planted.values()),
        "canary_people": len(canary_ids),
        "canaries_rejected_at_entry": rejected_canaries,
        "custom_listings_rejected_at_entry": rejected_custom,
        "outputs_scanned": {k: len(v) for k, v in outputs.items()},
        "bytes_scanned": len(blob),
        "secrets_found_in_outputs": found,
        "leak_report": {"ok": report["ok"], "findings": report["findings"][:20]},
        "receipt_chain_ok": chain_ok,
        "what_canaries_received": {pid: views[:3] for pid, views in received.items() if views},
    }


# --------------------------------------------------------------------------------------
# Attack suite
# --------------------------------------------------------------------------------------

IDENTIFYING_INTENTS = [
    "Looking for Mika Sato who works at Nintendo, tall with short red hair.",
    "I want to meet @sakura_climbs from the climbing gym in Osaka.",
    "The barista at the Blue Bottle in Tokyo with a nose ring, please find her.",
    "Looking for my ex, Daniel Brooks, he lives in London.",
    "Text me if you're the guy on +44 7700 900123.",
    "Is this you? https://instagram.com/someone",
]


def attack_suite(body: dict) -> dict:
    seed = int(body.get("seed", random.randrange(1_000_000)))
    personas, clock, warden, board = _sandbox(seed)
    ids = [l.person_id for l in board.listings()]
    a, b = ids[0], ids[1]
    a_listing = board.listing(a)
    results = []

    def attempt(category: str, name: str, fn) -> None:
        start = len(warden.receipts)
        try:
            out = fn()
            blocked = out is None or out is False or out == [] or getattr(out, "accepted", True) is False
            how = next((r.check for r in warden.receipts[start:] if r.result == "fail"), None)
        except PermissionError as e:
            blocked, how = True, f"refused: {e}"
        results.append({"category": category, "attack": name, "blocked": blocked, "stopped_by": how})

    fresh = iter(range(10_000))

    def admit(person, **changes):
        # A fresh identity per attempt, so one check (e.g. the rate limit) can't mask another.
        pid = f"attacker{next(fresh)}"
        return warden.admit(replace(person, id=pid), replace(a_listing, person_id=pid, **changes), Marketboard())

    me = board.person(a)
    # Getting in
    attempt("entry", "An unverified identity enters", lambda: admit(replace(me, verified=False)))
    for text in IDENTIFYING_INTENTS:
        attempt("entry", f"Search for a specific person: \"{text}\"", lambda t=text: admit(me, intent_text=t))
    attempt("entry", "Show a sensitive detail without the person confirming",
            lambda: admit(me, fields=a_listing.fields + (Field("religion", "Buddhist", SHOWN, True),), confirmed_shown=()))
    attempt("entry", "Relabel a sensitive detail as non-sensitive",
            lambda: admit(me, fields=a_listing.fields + (Field("health", "ADHD", SHOWN, False),)))
    attempt("entry", "Add a field outside the schema (home address)",
            lambda: admit(me, fields=a_listing.fields + (Field("home_address", "1-2-3 Namba", SHOWN, False),)))
    attempt("entry", "Claim a different age than the ID check",
            lambda: admit(replace(me, age_band="18-24" if me.age_band != "18-24" else "50+")))

    def flood():
        scratch, pid = Marketboard(), "flooder"
        person, listing = replace(me, id=pid), replace(a_listing, person_id=pid)
        results_ = [warden.admit(person, listing, scratch).accepted for _ in range(6)]
        return results_[-1] if all(results_[:5]) else None  # first five must pass, sixth must not
    attempt("entry", "Flood Belong. with listing updates (6 in a day)", flood)

    # Getting out: a compromised Matchmaker
    hidden_key = next((f.key for f in a_listing.fields if f.visibility == HIDDEN), "religion")
    P = lambda **kw: Proposal(**{"person_a": a, "person_b": b, "a_intent_met": True, "b_intent_met": True,
                                 "score": 1.0, "shown_score": 1.0, "shown_threshold": 0.1,
                                 "reason_fields": ("city",), "used_hidden_fields": (), **kw})
    attempt("compromised matchmaker", f"Use a hidden detail ({hidden_key}) as the reason",
            lambda: warden.exit_introduction(P(reason_fields=("city", hidden_key)), board))
    attempt("compromised matchmaker", "Propose a match with no shown reason at all",
            lambda: warden.exit_introduction(P(reason_fields=()), board))
    attempt("compromised matchmaker", "Introduce someone to a person who never asked for them",
            lambda: warden.exit_introduction(P(b_intent_met=False), board))
    attempt("compromised matchmaker", "Propose a person who hasn't signed up",
            lambda: warden.exit_introduction(P(person_b="ghost"), board))
    def self_intro():
        # Give person a a second listing, then try to introduce the two to each other.
        second = replace(a_listing, person_id=f"{a}-2", owner_id=a)
        warden.admit(me, second, board)
        return warden.exit_introduction(P(person_b=f"{a}-2", reason_fields=_shared_reasons(board, a, f"{a}-2")), board)
    attempt("compromised matchmaker", "Introduce someone to their own other listing", self_intro)
    attempt("compromised matchmaker", "Ask an off-menu question to fish for details",
            lambda: warden.exit_followup(FollowUp(a, "what_is_your_religion"), board))

    # Notification choices: questions and close matches only for people who opted in.
    def on_board(pid, **prefs):
        warden.admit(replace(me, id=pid), replace(a_listing, person_id=pid, **prefs), board)
        return pid
    quiet = on_board("optedout", notify_questions=False, notify_close=False)
    keen = on_board("optedin", notify_questions=True, notify_close=True)
    attempt("compromised matchmaker", "Ask a question of someone who opted out of questions",
            lambda: warden.exit_followup(FollowUp(quiet, "age_range"), board))
    attempt("compromised matchmaker", "Send a close match to someone who didn't opt in",
            lambda: warden.exit_introduction(P(person_a=quiet, person_b=b, a_intent_met=False, close=quiet,
                                               reason_fields=_shared_reasons(board, quiet, b)), board))
    attempt("compromised matchmaker", "Send a close match about someone who never asked for it",
            lambda: warden.exit_introduction(P(person_a=keen, person_b=b, a_intent_met=False, b_intent_met=False,
                                               close=keen, reason_fields=_shared_reasons(board, keen, b)), board))
    def close_seen_early():
        intro = warden.exit_introduction(P(person_a=keen, person_b=b, a_intent_met=False, close=keen,
                                           reason_fields=_shared_reasons(board, keen, b)), board)
        if intro is None:
            return None
        warden.record_decision(intro.intro_id, b, True)  # the other person acts before the recipient's yes
        return True
    attempt("compromised matchmaker", "Show a close match to the other person before the recipient says yes",
            close_seen_early)
    attempt("compromised matchmaker", "Quote someone's request in a question when they didn't opt in",
            lambda: warden.exit_followup(FollowUp(keen, "also_offer", about=quiet), board))
    paused = on_board("paused", active=False, notify_questions=True)
    attempt("compromised matchmaker", "Introduce someone to a paused listing",
            lambda: warden.exit_introduction(P(person_a=paused, person_b=b, reason_fields=_shared_reasons(board, paused, b)), board))
    attempt("compromised matchmaker", "Ask a paused listing a question",
            lambda: warden.exit_followup(FollowUp(paused, "age_range"), board))
    gone = on_board("deleted", notify_questions=True)
    warden.withdraw(replace(me, id="deleted"), gone, board)
    attempt("compromised matchmaker", "Introduce someone to a deleted listing",
            lambda: warden.exit_introduction(P(person_a=gone, person_b=b, reason_fields=("city",)), board))
    attempt("compromised matchmaker", "Attach another person's listing to an ordinary question",
            lambda: warden.exit_followup(FollowUp(keen, "age_range", about=b), board))

    # "Want to include close matches?" needs no opt-in (it is the opt-in), so it has its own limits.
    def broaden_twice():
        pid = on_board("unasked", notify_questions=False, notify_close=False)
        if warden.exit_followup(FollowUp(pid, "broaden"), board) is None:
            return None  # the first ask must go through
        return warden.exit_followup(FollowUp(pid, "broaden"), board)
    attempt("compromised matchmaker", "Ask \"Want to include close matches?\" twice", broaden_twice)
    attempt("compromised matchmaker", "Ask \"Want to include close matches?\" of someone who already said yes",
            lambda: warden.exit_followup(FollowUp(keen, "broaden"), board))
    def broaden_after_match():
        x = on_board("matched1", notify_questions=False, notify_close=False)
        y = on_board("matched2", notify_questions=False, notify_close=False)
        if warden.exit_introduction(P(person_a=x, person_b=y, reason_fields=_shared_reasons(board, x, y)), board) is None:
            return None  # the introduction must go through
        return warden.exit_followup(FollowUp(x, "broaden"), board)
    attempt("compromised matchmaker", "Ask \"Want to include close matches?\" of someone who already has a match",
            broaden_after_match)
    attempt("compromised matchmaker", "Attach another person's listing to \"Want to include close matches?\"",
            lambda: warden.exit_followup(FollowUp(on_board("unasked2", notify_questions=False, notify_close=False),
                                                  "broaden", about=b), board))

    def approach_flood():
        # Introduce one person up to the daily limit, then once more: that one must not get through.
        sent = 0
        for other in ids[2:]:
            if sent == warden.approach_limit:
                return warden.exit_introduction(P(person_b=other, reason_fields=_shared_reasons(board, a, other)), board)
            sent += warden.exit_introduction(P(person_b=other, reason_fields=_shared_reasons(board, a, other)), board) is not None
        return None  # never reached the limit: nothing to test
    attempt("compromised matchmaker", "Spam one person with more introductions than the daily limit", approach_flood)

    # Forging outputs
    attempt("forgery", "Build an Introduction without going through the Warden",
            lambda: Introduction(object(), "x", a, b, {"fields": {}, "reason": ""}, {"fields": {}, "reason": ""}))
    attempt("forgery", "Build a Reveal without going through the Warden",
            lambda: Reveal(None, a, b, "religion", ["Buddhist"]))
    attempt("reveal", "Reveal a detail to someone you're not connected with",
            lambda: warden.exit_reveal("i9999", {"religion"}, {"religion"}, board))

    custom = []
    for text in [str(t)[:500] for t in (body.get("intents") or [])][:20]:
        problems = specific_person_problems(text)
        custom.append({"intent": text, "refused": bool(problems), "why": problems})

    blocked = sum(r["blocked"] for r in results)
    return {"ok": blocked == len(results), "sandbox_seed": seed, "attempts": len(results), "blocked": blocked,
            "results": results, "your_intents": custom}


def _shared_reasons(board, a: str, b: str) -> tuple:
    sa, sb = board.listing(a).shown(), board.listing(b).shown()
    return tuple(k for k in ("city", "schedule") if k in sa and k in sb) or ("city",)
