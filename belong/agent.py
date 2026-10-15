"""Personal agent (SPEC §3): works for one person, outside the room.

It drafts the listing from the person's details and choices, relays answers to
follow-up questions, and asks the person which hidden details they would reveal to
a connection. It only ever sees its own person's data.
"""
from __future__ import annotations

import random
from dataclasses import replace

from .schema import AGE_BAND_BOUNDS, AGE_BANDS, FIELD_REGISTRY, HIDDEN, Field, Listing
from .synth import Persona


def draft_listing(p: Persona, now: str) -> Listing:
    """Turn the person's details into a listing, applying their visibility choices.

    Sensitive categories the person did not turn on are left out entirely.
    """
    fields = []
    for key, value in p.values.items():
        spec = FIELD_REGISTRY[key]
        if spec.sensitive and key not in p.sensitive_on:
            continue
        fields.append(Field(key=key, value=value, visibility=p.visibility[key], sensitive=spec.sensitive))
    return Listing(person_id=p.person.id, intent_text=p.intent_text, exactness=p.truth.exactness,
                   fields=tuple(fields), updated_at=now, confirmed_shown=tuple(sorted(p.confirmed_shown)))


def answer_followup(p: Persona, listing: Listing, question_id: str, now: str, rng: random.Random) -> Listing:
    """The person answers a fixed-menu question; the agent updates the listing."""
    text, exactness = listing.intent_text, listing.exactness
    if question_id == "age_range":
        lo, hi = p.truth.age_range
        lo_age = AGE_BAND_BOUNDS[lo][0]
        if hi == len(AGE_BANDS) - 1:
            text = f"{text} Aged {lo_age} or older."
        else:
            text = f"{text} Aged {lo_age} to {AGE_BAND_BOUNDS[hi][1]}."
        p.truth.omit_age = False
    elif question_id == "schedule":
        pass  # schedule is already a field; the person keeps their choice
    elif question_id == "wider_net":
        if rng.random() < 0.7:
            exactness = 0.3
            p.truth.exactness = 0.3
    p.intent_text = text
    return replace(listing, intent_text=text, exactness=exactness, updated_at=now, version=listing.version + 1)


def reveal_consents(listing: Listing, rng: random.Random) -> set:
    """After connecting, the person is asked about *each* of their hidden details
    ("reveal this if they share it?"). Asking about all of them, shared or not,
    means the question itself gives nothing away (SPEC §6, mutual reveal)."""
    return {f.key for f in listing.fields if f.visibility == HIDDEN and rng.random() < 0.5}


# --------------------------------------------------------------------------------------
# Drafting from a conversation (used by the two-person demo).
#
# Offline rules, no AI: the agent reads what the person wrote, suggests a listing, and
# leaves every choice to the person. An LLM can replace this later behind the same
# function, still running on the person's side, outside the room.
# --------------------------------------------------------------------------------------

import re as _re

from . import synth as _vocab
from .matching.intent import parse_intent as _parse_intent

ALL_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]

LABELS = {
    "first_name": "First name", "city": "City", "country": "Country", "age_band": "Age band (from ID check)",
    "gender": "Gender",
    "interests": "Interests", "schedule": "Free time", "level": "Climbing level", "pace": "Running pace",
    "native_language": "Native language", "learning_language": "Learning", "role": "Your role",
    "subject": "Subject", "move_in": "Moving in", "seeking": "Looking for", "offering": "You are (to them)",
    "diet": "Diet", "smoking": "Smoking",
    "children": "Children", "neighborhood": "Neighborhood", "employer": "Employer",
    "income_band": "Income", "religion": "Religion", "politics": "Politics", "health": "Health",
    "kinks": "Intimacy preferences",
}

_COMMON = ["first_name", "country", "city", "age_band", "interests", "schedule", "diet", "smoking",
           "neighborhood", "employer", "income_band", "religion", "politics", "health"]
_BY_KIND = {
    "dating": ["gender", "children", "kinks"], "climbing": ["level"], "running": ["pace"],
    "language": ["native_language", "learning_language"], "cofounder": ["role"],
    "tutoring": ["role", "subject"], "roommate": ["move_in"], "other": ["seeking", "offering"],
}
_KIND_NAMES = {
    "dating": "a relationship or date", "climbing": "a climbing partner", "running": "a running buddy",
    "language": "a language exchange", "cofounder": "a cofounder", "tutoring": "tutoring",
    "roommate": "a roommate",
    "other": "something else",
}
_SENSITIVE_VOCAB = {
    "religion": [r for r in _vocab.RELIGION if r != "not religious"],
    "health": _vocab.HEALTH,
    "politics": ["progressive", "conservative", "liberal", "left-wing", "right-wing"],
    "kinks": _vocab.KINKS,
}


def field_options(kind, city=None) -> dict:
    """The choices the person can pick from for each field."""
    country = _vocab.CITIES[city][0] if city in _vocab.CITIES else None
    roles = {"cofounder": ["technical", "business"], "tutoring": ["tutor", "student"]}.get(kind, [])
    return {
        "city": [], "country": [], "gender": ["woman", "man", "nonbinary"],
        "interests": [], "schedule": _vocab.SCHEDULES, "level": _vocab.LEVELS,
        "pace": _vocab.PACES, "native_language": _vocab.LANGUAGES, "learning_language": _vocab.LANGUAGES,
        "role": roles, "subject": [], "move_in": ALL_MONTHS, "diet": _vocab.DIET,
        "smoking": _vocab.SMOKING, "children": _vocab.CHILDREN,
        "neighborhood": [], "employer": [],
        "income_band": _vocab.INCOME[country] if country else ["lower income", "middle income", "higher income"],
        "religion": _vocab.RELIGION, "politics": _vocab.POLITICS, "health": _vocab.HEALTH,
        "kinks": _vocab.KINKS,
    }


def _find_language(pattern: str, low: str):
    for m in _re.finditer(pattern, low):
        word = next(g for g in m.groups() if g)
        lang = next((l for l in _vocab.LANGUAGES if l.lower() == word), None)
        if lang:
            return lang
    return None


def draft_from_message(message: str, first_name: str, age_band: str) -> dict:
    """Read the person's own words and suggest a listing. Returns a plain dict the
    app shows for review; nothing is submitted until the person says so."""
    text = " ".join(message.split())[:500]
    low = text.lower()
    parsed = _parse_intent(text)
    kind = parsed.kind
    v = {"first_name": first_name, "age_band": age_band}
    enabled_sensitive = set()
    notes = []

    city = next((c for c in _vocab.CITIES if c.lower() in low), None)
    if city:
        v["city"] = city
        v["country"] = _vocab.COUNTRY_NAMES[_vocab.CITIES[city][0]]
    for pat, sched in [(r"weekend", "weekends"), (r"evening", "weekday evenings"),
                       (r"morning", "weekday mornings"), (r"flexible|whenever|any ?time", "flexible")]:
        if _re.search(pat, low):
            v["schedule"] = sched
            break
    interests = [i for i in _vocab.INTERESTS if i in low]
    if interests:
        v["interests"] = interests
    level = next((l for l in _vocab.LEVELS if l in low), None)
    if kind == "climbing" and level:
        v["level"] = level
    if kind == "running":
        for pat, pace in [(r"\b(easy|slow|relaxed)\b", "easy"), (r"\b(moderate|steady)\b", "moderate"),
                          (r"\b(fast|quick)\b", "fast")]:
            if _re.search(pat, low):
                v["pace"] = pace
                break
    if kind == "language":
        native = _find_language(r"speak (\w+) natively|native (\w+) speaker|i'm a native (\w+)|trade (\w+) for", low)
        learning = _find_language(r"(?:practice|practise|learning|learn|for) (\w+)", low)
        if native:
            v["native_language"] = native
        if learning and learning != native:
            v["learning_language"] = learning
    if kind == "cofounder" and parsed.seeks_role:
        v["role"] = "business" if parsed.seeks_role == "technical" else "technical"
    if kind == "tutoring":
        if parsed.seeks_role:
            v["role"] = "student" if parsed.seeks_role == "tutor" else "tutor"
        subject = next((s for s in _vocab.SUBJECTS if s in low), None)
        if subject:
            v["subject"] = subject
    if kind == "roommate":
        month = next((m for m in ALL_MONTHS if m.lower() in low), None)
        if month:
            v["move_in"] = month
    if kind == "dating":
        m = _re.search(r"\b(?:i'm|i am|as) an? (woman|man)\b|\bi'm (nonbinary|non-binary)\b", low)
        if m:
            v["gender"] = "nonbinary" if m.group(2) else m.group(1)
    diet = next((d for d in _vocab.DIET if _re.search(r"\b" + d + r"\b", low)), None)
    if diet:
        v["diet"] = diet
    if _re.search(r"non-smoker|don't smoke|do not smoke", low):
        v["smoking"] = "doesn't smoke"

    # Sensitive things typed into the free text: move them to a hidden detail and warn.
    for key, words in _SENSITIVE_VOCAB.items():
        if key == "kinks" and kind != "dating":
            continue
        hit = next((w for w in words if _re.search(r"\b" + _re.escape(w.lower()) + r"\b", low)), None)
        if hit:
            value = hit if key != "politics" else next(
                (p for p in _vocab.POLITICS if hit.split("-")[0] in p), "politically progressive")
            v[key] = (value,) if FIELD_REGISTRY[key].multi else value
            enabled_sensitive.add(key)
            notes.append(f"You mentioned something sensitive ({LABELS[key].lower()}). I've added it as a "
                         f"hidden detail: it can help a match but is never shown. Your description is never "
                         f"shown either, but you may prefer to take it out of there.")

    exactness = 0.6
    if _re.search(r"\b(exactly|must|only|strictly)\b", low):
        exactness = 0.9
    elif _re.search(r"\b(open to|flexible|anyone|whatever|wide)\b", low):
        exactness = 0.3

    if kind is None:
        notes.insert(0, "I couldn't tell what kind of match you want. Could you say a bit more, e.g. "
                        "'a climbing partner' or 'a long-term relationship'?")
    if not city and not parsed.remote_ok:
        notes.append("Which city? I couldn't tell, so please pick one below.")
    if kind == "dating" and parsed.age_range is None:
        notes.append("You didn't mention an age range. Add one to your description, or the Matchmaker will ask.")
    if kind == "dating" and "gender" not in v:
        notes.append("Please set your gender below: it has to be shown for dating matches.")

    keys = _COMMON + _BY_KIND.get(kind, [])
    options = field_options(kind, v.get("city"))
    fields = []
    for key in keys:
        spec = FIELD_REGISTRY[key]
        value = v.get(key)
        fields.append({
            "key": key, "label": LABELS[key], "value": list(value) if isinstance(value, (list, tuple)) else value,
            "visibility": spec.default, "sensitive": spec.sensitive, "multi": spec.multi,
            "enabled": (key in enabled_sensitive) if spec.sensitive else True,
            "locked": key == "age_band", "options": options.get(key, []),
        })

    understood = [f"Looking for {_KIND_NAMES[kind]}" if kind else "Kind of match: not sure yet"]
    if city:
        understood.append(f"in {city}")
    if parsed.remote_ok:
        understood.append("(online/remote is fine)")
    return {"intent_text": text, "exactness": exactness, "kind": kind, "fields": fields,
            "understood": " ".join(understood), "notes": notes}
