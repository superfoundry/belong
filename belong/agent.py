"""Personal agent (SPEC §3): works for one person, outside the room.

It drafts the listing from the person's details and choices, relays answers to
follow-up questions, and asks the person which hidden details they would reveal to
a connection. It only ever sees its own person's data.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .schema import AGE_BAND_BOUNDS, AGE_BANDS, FIELD_REGISTRY, FREE_TEXT_MAX, HIDDEN, Field, Listing
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
    # Synthetic people's notification choices, fixed per person: most take questions, about
    # half want close matches.
    pick = int(hashlib.sha256(p.person.id.encode()).hexdigest()[:4], 16)
    return Listing(person_id=p.person.id, intent_text=p.intent_text, exactness=p.truth.exactness,
                   fields=tuple(fields), updated_at=now, confirmed_shown=tuple(sorted(p.confirmed_shown)),
                   notify_questions=pick % 10 < 8, notify_close=pick % 2 == 0)


def answer_followup(p: Persona, listing: Listing, question_id: str, now: str, rng: random.Random,
                    phrase: str = "") -> Listing:
    """The person answers a fixed-menu question; the agent updates the listing."""
    text, exactness = listing.intent_text, listing.exactness
    if question_id == "also_offer":
        # Synthetic people have exactly one role in their ground truth, so they honestly can't
        # take on another: they decline. Real people answer this on their phone (also_offer).
        return listing
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


def also_offer(listing: Listing, phrase: str, now: str) -> Listing:
    """Yes to "Could you help with that too?": the phrase becomes one more thing they offer."""
    fields = []
    for f in listing.fields:
        both = f"{f.value} or {phrase}"
        if f.key == "offering" and phrase.lower() not in str(f.value).lower() and len(both) <= FREE_TEXT_MAX:
            f = replace(f, value=both)
        fields.append(f)
    return replace(listing, fields=tuple(fields), updated_at=now, version=listing.version + 1)


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
    "subject": "Subject", "move_in": "Moving in", "seeking": "Looking for", "offering": "You are (to them)", "pay": "Pay", "format": "Format",
    "diet": "Diet", "smoking": "Smoking",
    "children": "Children", "neighborhood": "Neighborhood", "employer": "Employer",
    "income_band": "Income", "religion": "Religion", "politics": "Politics", "health": "Health",
    "kinks": "Intimacy preferences",
}

_COMMON = ["first_name", "country", "city", "age_band", "interests", "schedule", "diet", "smoking",
           "neighborhood", "employer", "income_band", "religion", "politics", "health"]
_BY_KIND = {
    "dating": ["gender", "children", "kinks"], "climbing": ["level"], "running": ["pace"],
    "language": ["native_language", "learning_language"], "tutoring": ["role", "subject", "pay", "format"], "roommate": ["move_in", "pay"], "cofounder": ["role"],
    "other": ["seeking", "offering", "pay", "format"],
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


# --------------------------------------------------------------------------------------
# Instant profile updates (used by the demo while the agent is still replying).
#
# Picks out only facts stated plainly enough to be sure of ("I live in Galway", "free on
# weekends", "I'm into chess and hiking"). Anything uncertain is left for the agent.
# --------------------------------------------------------------------------------------

COUNTRIES = [
    "Ireland", "United Kingdom", "UK", "England", "Scotland", "Wales", "France", "Germany", "Spain", "Portugal",
    "Italy", "Netherlands", "Belgium", "Switzerland", "Austria", "Sweden", "Norway", "Denmark", "Finland",
    "Poland", "Greece", "Turkey", "Japan", "China", "South Korea", "Korea", "Taiwan", "Singapore", "India",
    "Thailand", "Vietnam", "Indonesia", "Philippines", "Australia", "New Zealand", "Canada", "United States",
    "USA", "US", "Mexico", "Brazil", "Argentina", "Chile", "Colombia", "South Africa", "Nigeria", "Kenya", "Egypt",
    "Morocco", "Israel", "United Arab Emirates", "UAE",
]
_COUNTRY_ALIASES = {"uk": "United Kingdom", "england": "United Kingdom", "scotland": "United Kingdom",
                    "wales": "United Kingdom", "usa": "United States", "us": "United States", "korea": "South Korea",
                    "uae": "United Arab Emirates"}
_PLACE = r"([a-zà-ÿ][a-zà-ÿ'\-]*(?:\s[a-zà-ÿ][a-zà-ÿ'\-]*){0,2}(?:,\s*[a-zà-ÿ][a-zà-ÿ'\-]*(?:\s[a-zà-ÿ][a-zà-ÿ'\-]*)?)?)"
_PLACE_STOP = {"the", "a", "my", "town", "the city", "the area", "person", "love", "need", "search", "here", "there"}


def _country(name: str):
    low = name.strip().lower()
    if low in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[low]
    return next((c for c in COUNTRIES if c.lower() == low), None)


def _title(place: str) -> str:
    return " ".join(w if w.isupper() else w[:1].upper() + w[1:] for w in place.split())


def quick_facts(text: str) -> dict:
    """Facts stated plainly in one message, as field -> value. Only what we're sure of."""
    low = " ".join(text.lower().split())
    out = {}

    # Where: "I live in Galway", "based in Tokyo, Japan", "in malaga, spain"; or a country on its own.
    m = _re.search(r"\b(?:i live in|i'm in|i am in|based in|living in|located in|i'm from|i am from|moved to)\s+"
                   + _PLACE + r"(?=[.!?;,]|$|\s(?:and|but|so|on|for|with|looking)\b)", low)
    place = m.group(1).strip(" ,") if m else None
    if place and place not in _PLACE_STOP:
        if "," in place:
            city, country = (p.strip() for p in place.split(",", 1))
            out["city"], out["country"] = _title(city), (_country(country) or _title(country))
        elif _country(place):
            out["country"] = _country(place)
        else:
            out["city"] = _title(place)
    known_city = next((c for c in _vocab.CITIES if _re.search(r"\b" + c.lower() + r"\b", low)), None)
    if known_city:
        out.setdefault("city", known_city)
        if out["city"] == known_city:
            out.setdefault("country", _vocab.COUNTRY_NAMES[_vocab.CITIES[known_city][0]])
    if "country" not in out:
        found = next((c for c in COUNTRIES if len(c) > 3 and _re.search(r"\b" + _re.escape(c.lower()) + r"\b", low)), None)
        if found:
            out["country"] = _country(found)

    # When they're free.
    for pattern, sched in [(r"\bweekends?\b", "weekends"), (r"\b(?:weekday )?evenings?\b|\bafter work\b", "weekday evenings"),
                           (r"\b(?:weekday )?mornings?\b", "weekday mornings"), (r"\bflexible\b|\bany ?time\b|\bwhenever\b", "flexible")]:
        if _re.search(pattern, low):
            out["schedule"] = sched
            break

    level = next((l for l in _vocab.LEVELS if _re.search(r"\b" + l + r"\b", low)), None)
    if level:
        out["level"] = level
    m = _re.search(r"\b(?:i'm|i am|as) an? (woman|man)\b|\bi'm (nonbinary|non-binary)\b", low)
    if m:
        out["gender"] = "nonbinary" if m.group(2) else m.group(1)

    native = _find_language(r"speak (\w+) natively|native (\w+) speaker|my native language is (\w+)|fluent (\w+) speaker|"
                            r"i'm a native (\w+)|i speak fluent (\w+)", low)
    learning = _find_language(r"(?:learning|learn|practice|practise|improve my) (\w+)", low)
    if native:
        out["native_language"] = native
    if learning and learning != native:
        out["learning_language"] = learning

    # Pay: "4000 yen per hour", "¥4,000/hr", "$30 an hour", "€500 a month".
    m = _re.search(r"(?:(?P<sym>[¥$€£])\s?(?P<a1>\d[\d,]*(?:\.\d+)?)(?P<k1>k)?|(?P<a2>\d[\d,]*(?:\.\d+)?)(?P<k2>k)?\s?"
                   r"(?P<word>yen|dollars?|usd|euros?|eur|pounds?|gbp))"
                   r"(?:\s*(?:per|an|a|/)\s*(?P<unit>hour|hr|h|session|lesson|class|month|week|day))?", low)
    if m and (m.group("unit") or _re.search(r"\b(?:pay|offer|budget|rate|charge|salary)\w*\b", low)):
        symbol = m.group("sym") or {"yen": "¥", "dollar": "$", "dollars": "$", "usd": "$", "euro": "€", "euros": "€",
                                    "eur": "€", "pound": "£", "pounds": "£", "gbp": "£"}[m.group("word")]
        amount = float((m.group("a1") or m.group("a2")).replace(",", ""))
        if m.group("k1") or m.group("k2"):
            amount *= 1000
        unit = {"hr": "hour", "h": "hour"}.get(m.group("unit"), m.group("unit"))
        out["pay"] = f"{symbol}{amount:,.0f}" + (f" per {unit}" if unit else "")
    in_person = bool(_re.search(r"\bin[- ]person\b|\bface to face\b|\bmeet up\b|\bmeetups?\b", low))
    online = bool(_re.search(r"\bonline\b|\bremote(?:ly)?\b|\bover (?:zoom|video)\b|\bvideo call", low))
    if in_person or online:
        out["format"] = "In person or online" if in_person and online else "In person" if in_person else "Online"

    # Interests, only when introduced plainly: "I'm into chess and hiking", "I enjoy jazz".
    m = _re.search(r"\b(?:i'm into|i am into|into|i enjoy|i love|who loves|who enjoys|my hobbies are|i'm interested in|i like)\s+"
                   r"([a-z][a-z ,'\-]{2,80}?)(?=[.!?;]|$)", low)
    if m and not _re.match(r"(?:to |meeting|meet |you|people)", m.group(1)):
        items = [i.strip() for i in _re.split(r",|\band\b|\bor\b", m.group(1))]
        items = [_re.sub(r"^(?:playing|doing|going|some)\s+", "", i) for i in items]
        items = [i for i in items if i and len(i.split()) <= 3 and len(i) <= 30]
        if items:
            out["interests"] = items[:6]
    return out


def empty_draft(first_name: str, age_band: str) -> dict:
    """A blank work-in-progress profile, shown before the agent has drafted anything."""
    fields = []
    for key in _COMMON:
        spec = FIELD_REGISTRY[key]
        value = {"first_name": first_name, "age_band": age_band}.get(key)
        fields.append({"key": key, "label": LABELS[key], "value": value, "visibility": spec.default,
                       "sensitive": spec.sensitive, "multi": spec.multi, "enabled": not spec.sensitive,
                       "locked": key == "age_band", "options": field_options(None, None).get(key, [])})
    return {"intent_text": "", "exactness": 0.6, "kind": None, "fields": fields, "understood": "",
            "notes": [], "missing": [], "complete": False, "by": "quick"}


def fill_gaps(draft: dict, facts: dict) -> list:
    """Put facts into empty fields of a draft (never overwriting anything set). Returns the
    keys filled. Fields the draft doesn't have yet are added."""
    filled = []
    by_key = {f["key"]: f for f in draft["fields"]}
    for key, value in facts.items():
        spec = FIELD_REGISTRY.get(key)
        if spec is None or spec.sensitive:
            continue
        f = by_key.get(key)
        if f is None:
            f = {"key": key, "label": LABELS[key], "value": None, "visibility": spec.default, "sensitive": False,
                 "multi": spec.multi, "enabled": True, "locked": False,
                 "options": field_options(draft.get("kind"), None).get(key, [])}
            draft["fields"].append(f)
            by_key[key] = f
        if f["value"] in (None, "", []):
            f["value"] = list(value) if isinstance(value, (list, tuple)) else value
            filled.append(key)
    return filled
