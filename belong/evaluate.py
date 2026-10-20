"""Grade match quality against ground truth (SPEC §8: "prove that free-text intent
matching produces good matches").

Truth is what each synthetic person really wants. The Matchmaker never sees it; it
only sees the listing. A proposal is "correct" if, by the truth, both people's
intents are met.
"""
from __future__ import annotations

from .synth import REMOTE_KINDS, Persona, Truth


def _sched_overlap(x, y) -> bool:
    return x == y or "flexible" in (x, y)


def truth_intent_met(a: Truth, b: Truth) -> bool:
    if a.kind != b.kind:
        return False
    if a.city != b.city and not (a.kind in REMOTE_KINDS and a.remote_ok and b.remote_ok):
        return False
    if a.kind == "dating":
        if b.gender not in a.seeking:
            return False
        slack = 1 if a.exactness < 0.5 else 0
        lo, hi = a.age_range
        return lo - slack <= b.age_idx <= hi + slack
    if a.kind == "language":
        return b.native == a.learning
    if a.kind == "cofounder":
        return a.role != b.role
    if a.kind == "tutoring":
        return a.role != b.role and a.subject == b.subject
    if a.kind in ("climbing", "running"):
        tolerance = 0 if a.exactness >= 0.8 else 1
        if abs(a.level - b.level) > tolerance:
            return False
        return a.exactness < 0.8 or _sched_overlap(a.schedule, b.schedule)
    if a.kind == "other":
        if a.sub != b.sub:
            return False
        if a.sub != "music":
            return True  # chess / photo: symmetric
        if (a.role == "band") == (b.role == "band"):
            return False  # a band needs a musician, a musician needs a band
        genre_ok = a.genre == b.genre or a.exactness < 0.5
        return genre_ok and (a.role != "band" or b.role == a.seeks)
    return True  # roommate: same city is enough


def compatible(a: Persona, b: Persona) -> bool:
    return truth_intent_met(a.truth, b.truth) and truth_intent_met(b.truth, a.truth)


def evaluate(personas: dict, admitted: set, proposals: list, introductions: list,
             connections: set, reveals: list) -> dict:
    ids = sorted(admitted)
    compat_partners = {pid: 0 for pid in ids}
    compat_pairs = 0
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if compatible(personas[a], personas[b]):
                compat_pairs += 1
                compat_partners[a] += 1
                compat_partners[b] += 1
    matchable = {pid for pid, n in compat_partners.items() if n > 0}

    # Close matches are one-way by design (only nearly what their recipient asked for), so
    # precision is measured on exact matches, and close matches are counted on their own.
    close = [p for p in proposals if getattr(p, "close", "")]
    proposals = [p for p in proposals if not getattr(p, "close", "")]
    correct = [p for p in proposals if compatible(personas[p.person_a], personas[p.person_b])]
    proposed_people = {x for p in proposals for x in (p.person_a, p.person_b)}
    introduced_people = {x for i in introductions for x in (i["person_a"], i["person_b"])}

    by_kind = {}
    for kind in sorted({personas[p].truth.kind for p in ids}):
        people = {p for p in ids if personas[p].truth.kind == kind}
        props = [p for p in proposals if p.person_a in people]
        good = [p for p in props if compatible(personas[p.person_a], personas[p.person_b])]
        m = people & matchable
        by_kind[kind] = {
            "people": len(people),
            "matchable": len(m),
            "proposals": len(props),
            "precision": round(len(good) / len(props), 3) if props else None,
            "coverage": round(len(m & proposed_people) / len(m), 3) if m else None,
        }

    return {
        "admitted": len(ids),
        "truth_compatible_pairs": compat_pairs,
        "matchable_people": len(matchable),
        "proposals": len(proposals),
        "proposals_correct": len(correct),
        "precision": round(len(correct) / len(proposals), 3) if proposals else None,
        "coverage": round(len(matchable & proposed_people) / len(matchable), 3) if matchable else None,
        "introductions": len(introductions),
        "people_introduced": len(introduced_people),
        "connections": len(connections),
        "reveals": len(reveals),
        "proposals_using_hidden_fields": sum(1 for p in proposals if p.used_hidden_fields),
        "close_matches": len(close),
        "by_kind": by_kind,
    }
