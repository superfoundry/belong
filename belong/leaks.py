"""Leak report (SPEC §8.6): an independent audit of everything sent out.

Deliberately does NOT reuse the Warden's code. It reads plain dicts (what was
actually delivered) plus the Marketboard, and asks: did any hidden value reach anyone
who shouldn't have it? Any finding is a test failure.

It also audits the matching rules on the proposals (never the only reason,
only counts in a crowd), which can't be seen from outputs alone.

Audits:
  A1  introduction fields     every field shown in a view is shown by its owner, same value
  A2  introduction text       no hidden value of anyone appears in a view, except values
                              the recipient may legitimately see
  A3  follow-ups              text is exactly the fixed-menu text; no hidden values
  A4  reveals                 only shared details, between connected people, both consenting
  A5  receipts                exactly {seq, check, result, timestamp, commitment, hash};
                              no content
  A6  receipt chain           every receipt's hash links to the one before it (nothing
                              deleted, reordered or edited). Re-implemented here, not
                              imported from the Warden.
  R1  never the only reason   every proposal clears the shown-only threshold and has
                              reason fields shown by both
  R2  only counts in a crowd  every hidden field used had a shared value >= k people
  R3  approach limit          no listing introduced more than the weekly limit
  R4  distinct people         no one is introduced to their own other listing
  R5  opt-ins                 follow-ups only to listings that opted in to questions;
                              close matches only to listings that opted in to them, and
                              each one names one of its two people as its one recipient
"""
from __future__ import annotations

import hashlib
import json
import re

from .schema import FOLLOWUP_MENU, HIDDEN, SHOWN

RECEIPT_KEYS = {"seq", "check", "result", "timestamp", "commitment", "hash"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _shown(listing, key):
    f = listing.get(key)
    return f.value if f is not None and f.visibility == SHOWN else None


def _as_quoted(headline):
    """Re-implemented, not imported from the Warden: a headline as a question quotes it."""
    if not isinstance(headline, str) or not re.match(r"^[A-Za-z]+ing\b", headline):
        return None
    h = headline.strip().rstrip(".")
    return h[:1].lower() + h[1:]


def _contains(text: str, s: str) -> bool:
    return re.search(r"(?<!\w)" + re.escape(s) + r"(?!\w)", text, flags=re.I) is not None


def _scrub(text: str, allowed: set) -> str:
    for s in sorted(allowed, key=len, reverse=True):
        text = re.sub(r"(?<!\w)" + re.escape(s) + r"(?!\w)", " ", text, flags=re.I)
    return text


def leak_report(board, introductions: list, followups: list, reveals: list, receipts: list,
                proposals: list, connections: set, consents: dict, k_crowd: int,
                approach_limit: int) -> dict:
    """All arguments except board/proposals are plain dicts as delivered.

    connections: set of frozenset({a, b}) pairs that both said yes.
    consents: {(intro_id, person_id): set of keys} reveal opt-ins.
    """
    listings = {l.person_id: l for l in board.listings(include_withdrawn=True)}  # what was sent, even if since deleted

    def strings(pid, visibility=None):
        return {s for f in listings[pid].fields
                if visibility is None or f.visibility == visibility for s in f.strings()}

    every_hidden = set().union(*(strings(pid, HIDDEN) for pid in listings)) if listings else set()
    findings = []

    def find(audit, detail):
        findings.append({"audit": audit, "detail": detail})

    # A1 + A2: introductions
    intro_count = {}
    for intro in introductions:
        for viewer, other, key in ((intro["person_a"], intro["person_b"], "a_view"),
                                   (intro["person_b"], intro["person_a"], "b_view")):
            view = intro[key]
            for k, v in view["fields"].items():
                f = listings[other].get(k)
                if f is None or f.visibility != SHOWN:
                    find("A1", f"{intro['intro_id']} {key}: field {k!r} is not shown by its owner {other}")
                elif (list(f.value) if isinstance(f.value, tuple) else f.value) != v:
                    find("A1", f"{intro['intro_id']} {key}: field {k!r} value differs from owner's")
            # The recipient may see: the counterpart's shown values; their own values
            # only for keys BOTH show (those are the only keys a reason may use).
            both_shown = set(listings[viewer].shown()) & set(listings[other].shown())
            allowed = strings(other, SHOWN) | {s for k in both_shown for s in listings[viewer].get(k).strings()}
            text = _scrub(json.dumps(view, ensure_ascii=False), allowed)
            for s in sorted(every_hidden):
                if _contains(text, s):
                    find("A2", f"{intro['intro_id']} {key}: hidden value {s!r} reached {viewer}")
        for p in (intro["person_a"], intro["person_b"]):
            intro_count[p] = intro_count.get(p, 0) + 1

    # A3: follow-ups
    for fu in followups:
        template = FOLLOWUP_MENU.get(fu["question_id"])
        expected = template.format(phrase=fu.get("phrase", "")) if template else None
        if expected != fu["question_text"] or set(fu) != {"person_id", "question_id", "question_text", "phrase",
                                                          "offer_add", "seek_add"} \
                or ((fu.get("phrase") or fu.get("offer_add") or fu.get("seek_add")) and fu["question_id"] != "also_offer"):
            find("A3", f"follow-up to {fu['person_id']} is not a fixed-menu question")
        for s in every_hidden:
            if _contains(fu["question_text"], s):
                find("A3", f"follow-up to {fu['person_id']} contains hidden value {s!r}")

    # A4: reveals
    intro_by_pair = {frozenset((i["person_a"], i["person_b"])): i["intro_id"] for i in introductions}
    for rv in reveals:
        pair = frozenset((rv["person_a"], rv["person_b"]))
        intro_id = intro_by_pair.get(pair)
        label = f"reveal of {rv['key']!r} between {rv['person_a']} and {rv['person_b']}"
        if pair not in connections or intro_id is None:
            find("A4", f"{label}: they are not connected")
            continue
        for p in pair:
            if rv["key"] not in consents.get((intro_id, p), set()):
                find("A4", f"{label}: {p} did not consent")
            f = listings[p].get(rv["key"])
            if f is None or not set(rv["values"]) <= set(f.strings()):
                find("A4", f"{label}: revealed value is not shared by {p}")

    # A5: receipts carry no content
    all_values = set().union(*(strings(pid) for pid in listings)) if listings else set()
    for r in receipts:
        if (set(r) != RECEIPT_KEYS or r["result"] not in ("pass", "fail") or not isinstance(r["seq"], int)
                or not _HEX64.fullmatch(str(r["hash"])) or not _HEX64.fullmatch(str(r["commitment"]))):
            find("A5", f"receipt has unexpected shape: {sorted(r)}")
            continue
        blob = f"{r['check']} {r['timestamp']}"
        for s in all_values:
            if len(s) > 2 and _contains(blob, s):
                find("A5", f"receipt for {r['check']} contains a field value")

    # A6: the chain is unbroken (the first receipt given is taken as the anchor)
    shaped = [r for r in receipts if set(r) == RECEIPT_KEYS]
    for prev, r in zip(shaped, shaped[1:]):
        expected = hashlib.sha256(f"{prev['hash']}|{r['seq']}|{r['check']}|{r['result']}|{r['timestamp']}|"
                                  f"{r['commitment']}".encode()).hexdigest()
        if r["seq"] != prev["seq"] + 1 or r["hash"] != expected:
            find("A6", f"receipt chain broken at seq {r['seq']}")
            break

    # R1, R2: matching rules
    counts = board.value_counts()
    for p in proposals:
        pair = f"{p.person_a}+{p.person_b}"
        a, b = listings[p.person_a], listings[p.person_b]
        if p.shown_score < p.shown_threshold or not p.reason_fields:
            find("R1", f"{pair}: not justified by shown fields alone")
        for k in p.reason_fields:
            if k not in a.shown() or k not in b.shown():
                find("R1", f"{pair}: reason field {k!r} not shown by both")
        for k in p.used_hidden_fields:
            fa, fb = a.get(k), b.get(k)
            shared = set(fa.strings()) & set(fb.strings()) if fa and fb else set()
            if not any(counts[(k, s)] >= k_crowd for s in shared):
                find("R2", f"{pair}: hidden field {k!r} used without a crowd of {k_crowd}")

    # R4: never introduced to yourself
    for intro in introductions:
        a, b = listings.get(intro["person_a"]), listings.get(intro["person_b"])
        if a and b and a.owner == b.owner:
            find("R4", f"{intro['intro_id']}: two listings of the same person were introduced")

    # R5: notification opt-ins
    for fu in followups:
        l = listings.get(fu["person_id"])
        if l is not None and not l.notify_questions:
            find("R5", f"follow-up sent to {fu['person_id']}, who didn't opt in to questions")
    # An "also offer" question quotes the shown phrase of someone who opted in to close matches.
    for fu in followups:
        if fu["question_id"] != "also_offer":
            continue
        sources = [l for l in listings.values() if l.notify_close and l.owner != getattr(listings.get(fu["person_id"]), "owner", None)
                   and _shown(l, "seeking") == fu["offer_add"] and _shown(l, "offering") == fu["seek_add"]
                   and fu["phrase"] in (_as_quoted(_shown(l, "headline")), f"looking for {_shown(l, 'seeking')}")]
        if not sources:
            find("R5", f"question to {fu['person_id']} quotes a phrase no opted-in listing shows")
    for intro in introductions:
        to = intro.get("one_way_to") or ""
        if not to:
            continue
        l = listings.get(to)
        if to not in (intro["person_a"], intro["person_b"]) or l is None or not l.notify_close:
            find("R5", f"{intro['intro_id']}: close match sent to {to}, who didn't opt in to close matches")

    # R3: approach limit
    for person, n in intro_count.items():
        if n > approach_limit:
            find("R3", f"{person} was introduced {n} times (limit {approach_limit})")

    return {
        "ok": not findings,
        "findings": findings,
        "checked": {
            "introductions": len(introductions),
            "views": 2 * len(introductions),
            "followups": len(followups),
            "reveals": len(reveals),
            "receipts": len(receipts),
            "proposals": len(proposals),
            "hidden_values_watched": len(every_hidden),
        },
    }
