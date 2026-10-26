"""Grade match quality against ground truth (SPEC §8: "prove that free-text intent
matching produces good matches").

Truth is what each synthetic person really wants. The Matchmaker never sees it; it
only sees the listing. A proposal is "correct" if, by the truth, both people's
intents are met.

compare_narrowing() checks narrowing's word index (SPEC §8.2) against the catch-all group it
replaced: it may only drop candidates no decision could accept.
"""
from __future__ import annotations

from dataclasses import replace

from .schema import TRAVEL, km_between
from .synth import Persona, Truth


def _sched_overlap(x, y) -> bool:
    return x == y or "flexible" in (x, y)


def can_meet(a: Truth, b: Truth) -> bool:
    """The matching order's first two steps, on the truth: online if both are open to it (never for
    dating); otherwise in person, with their cities within both people's travel distance."""
    if a.remote_ok and b.remote_ok and "dating" not in (a.kind, b.kind):
        return True
    if not (a.in_person and b.in_person):
        return False
    km = km_between(a.city, b.city)
    return km is not None and km <= min(TRAVEL[a.travel], TRAVEL[b.travel])


def truth_intent_met(a: Truth, b: Truth) -> bool:
    if a.kind != b.kind:
        return False
    if not can_meet(a, b):
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
    if a.kind == "service":
        return a.sub == b.sub and a.role != b.role  # a provider and a customer of the same service
    if a.kind == "item":
        # Selling meets buying; giving away meets looking for it free (not looking to pay).
        return a.sub == b.sub and {a.role, b.role} in ({"seller", "buyer"}, {"giver", "taker"})
    return True  # roommate: being able to meet is enough


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


def compare_narrowing(personas: list, shortlist_size: int = 40) -> dict:
    """Narrow every listing twice with no size limit: with one catch-all group for everything
    compared by phrase (as before the word index), and with the word index. Reports whether the
    index only drops candidates (`subset`), and every dropped pair a decision could still accept
    (`lost`: either listing's intent met by the other, exactly or with the wide net of a close
    match, or nearly met as a question would ask). Then runs the Matchmaker both ways at its usual
    shortlist size and compares the proposals with the ground truth."""
    from . import agent
    from .matching.marketboard import Marketboard
    from .matching.matcher import (CLOSE_EXACTNESS, OfflineMatcher, ShownView, effective_kind, has_phrases,
                                   index_words, intent_met, nearly, parse_intent)
    from .matching.narrow import shortlists

    board, by_id = Marketboard(), {p.person.id: p for p in personas}
    for p in personas:
        board.put(p.person, agent.draft_listing(p, "2026-10-12T09:00:00Z"))
    listings = {l.person_id: l for l in board.listings()}
    views = {pid: ShownView(l) for pid, l in listings.items()}
    parsed = {pid: effective_kind(parse_intent(l.intent_text), views[pid]) for pid, l in listings.items()}
    wide = {pid: ShownView(replace(l, exactness=CLOSE_EXACTNESS)) for pid, l in listings.items()}

    def could_accept(a, b) -> bool:
        pa, pb, va, vb = parsed[a], parsed[b], views[a], views[b]
        if intent_met(pa, va, pb, vb) or intent_met(pb, vb, pa, va):
            return True
        if intent_met(pa, wide[a], pb, vb) or intent_met(pb, wide[b], pa, va):
            return True
        return has_phrases(va) and has_phrases(vb) and (
            nearly(va.get("seeking"), vb.get("offering"), va.get("topics"), vb.get("topics"))
            or nearly(vb.get("seeking"), va.get("offering"), vb.get("topics"), va.get("topics")))

    everyone = len(listings) + 1
    old, _ = shortlists(parsed, views, everyone)
    new, _ = shortlists(parsed, views, everyone, words=index_words)
    new_sets = {a: set(c) for a, c in new.items()}
    subset = all(new_sets[a] <= set(old[a]) for a in old)
    lost = sorted((a, b) for a in old for b in old[a] if b not in new_sets[a] and could_accept(a, b))
    kept = sum(1 for a in old for b in old[a] if b in new_sets[a] and could_accept(a, b))

    def proposals(word_index):
        props = OfflineMatcher(shortlist_size=shortlist_size, word_index=word_index).match(board).proposals
        exact = {(p.person_a, p.person_b) for p in props if not p.close}
        return exact, {pair for pair in exact if compatible(by_id[pair[0]], by_id[pair[1]])}

    (old_props, old_good), (new_props, new_good) = proposals(False), proposals(True)
    by_phrase = [a for a in old if has_phrases(views[a]) and parsed[a].kind != "dating"]
    return {
        "listings": len(listings),
        "candidates_before": sum(len(c) for c in old.values()),
        "candidates_after": sum(len(c) for c in new.values()),
        "phrase_listings": len(by_phrase),
        "phrase_candidates_before": sum(len(old[a]) for a in by_phrase),
        "phrase_candidates_after": sum(len(new[a]) for a in by_phrase),
        "subset": subset,
        "acceptable_pairs_kept": kept,
        "lost": lost,
        "proposals_before": len(old_props), "correct_before": len(old_good),
        "proposals_after": len(new_props), "correct_after": len(new_good),
        "correct_found_only_before": len(old_good - new_good),
        "correct_found_only_after": len(new_good - old_good),
    }
