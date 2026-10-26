"""Personal agent (SPEC §3): works for one person, outside the room.

It drafts the listing from the person's details and choices, relays answers to
follow-up questions, and asks the person which hidden details they would reveal to
a connection. It only ever sees its own person's data.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .schema import (AGE_BAND_BOUNDS, AGE_BANDS, AGE_MAX, AGE_MIN, FIELD_REGISTRY, FORMATS, FREE_TEXT_MAX, GENDERS,
                     HIDDEN, MATCH_KINDS, SHOWN, TRAVEL, Field, Listing, age_span, only_said, said)
from .synth import Persona, ages_in_years


def with_wants_age(fields: tuple, value: str) -> tuple:
    """The listing's fields with "wants_age" set to `value` (years, "lo-hi"): an existing one keeps
    the person's own choice of shown or hidden, a new one is shown, like any field by default."""
    if any(f.key == "wants_age" for f in fields):
        return tuple(replace(f, value=value) if f.key == "wants_age" else f for f in fields)
    return tuple(fields) + (Field("wants_age", value, FIELD_REGISTRY["wants_age"].default, False),)


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
    text, exactness, fields = listing.intent_text, listing.exactness, listing.fields
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
        fields = with_wants_age(fields, ages_in_years(p.truth.age_range))  # and recorded as understood
        p.truth.omit_age = False
    elif question_id == "schedule":
        pass  # schedule is already a field; the person keeps their choice
    elif question_id == "broaden":
        if rng.random() < 0.6:  # most say yes: close matches, and with them follow-up questions
            return replace(listing, notify_close=True, notify_questions=True, updated_at=now,
                           version=listing.version + 1)
        return listing
    p.intent_text = text
    return replace(listing, intent_text=text, exactness=exactness, fields=fields, updated_at=now,
                   version=listing.version + 1)


def also_offer(listing: Listing, phrase: str, now: str, mirror: str = "", pay: str = "") -> Listing:
    """Yes to "Would that work for you?": what they want joins what this person offers, who they
    are joins who this person is looking for, and their terms (pay) replace this person's, so the
    match sees one clear price ("Free", not "$1 each")."""
    add = {"offering": phrase, "seeking": mirror}
    fields, has_pay = [], False
    for f in listing.fields:
        extra = add.get(f.key)
        both = f"{f.value} or {extra}"
        if extra and extra.lower() not in str(f.value).lower() and len(both) <= FREE_TEXT_MAX:
            f = replace(f, value=both)
        if f.key == "pay" and pay:
            f, has_pay = replace(f, value=pay, visibility=SHOWN), True
        fields.append(f)
    if pay and not has_pay:
        fields.append(Field("pay", pay, SHOWN, False))
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
from .matching.intent import (_HOBBY, _KIND_PATTERNS, _MEETING, _ROMANTIC, _TASK_AFTER, _is_dating, _set_aside,
                              ages_text, dating_wants as _dating_wants, default_age_range, own_gender,
                              parse_intent as _parse_intent, read_listing, relative_years)

ALL_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]

LABELS = {
    "first_name": "First name", "city": "City", "country": "Country", "age_band": "Age band (from ID check)",
    "gender": "Gender",
    "interests": "Interests", "schedule": "Free time", "level": "Climbing level", "pace": "Running pace",
    "native_language": "Native language", "learning_language": "Learning", "role": "Your role",
    "subject": "Subject", "move_in": "Moving in", "seeking": "Looking for", "offering": "You are (to them)", "topics": "Related topics", "headline": "Headline", "pay": "Pay", "format": "Format",
    "travel": "How far you'll travel",
    "diet": "Diet", "smoking": "Smoking",
    "children": "Children", "neighborhood": "Neighborhood", "employer": "Employer",
    "income_band": "Income", "religion": "Religion", "politics": "Politics", "health": "Health",
    "kinks": "Intimacy preferences",
    "kind": "Kind of match", "wants_gender": "Looking for", "wants_age": "Their age",
}

_COMMON = ["first_name", "country", "city", "travel", "age_band", "interests", "schedule", "diet", "smoking",
           "neighborhood", "employer", "income_band", "religion", "politics", "health"]
# Only dating records who they'd like to meet and their ages.
DATING_ONLY = ("wants_gender", "wants_age")
_BY_KIND = {
    "dating": ["gender", *DATING_ONLY, "children", "kinks"], "climbing": ["level"], "running": ["pace"],
    "language": ["native_language", "learning_language", "format"], "tutoring": ["role", "subject", "pay", "format"],
    "roommate": ["move_in", "pay"], "cofounder": ["role", "format"],
    "other": ["seeking", "offering", "pay", "format"],
}
_KIND_NAMES = {
    "dating": "a relationship or date", "climbing": "a climbing partner", "running": "a running buddy",
    "language": "a language exchange", "cofounder": "a cofounder", "tutoring": "tutoring",
    "roommate": "a roommate",
    "other": "something else",
}
assert tuple(_KIND_NAMES) == MATCH_KINDS  # the kinds an agent may choose are the ones the Warden accepts
# Why dating needs your gender, and what hiding it means (hidden details never decide a match).
GENDER_NOTE = ("Please set your gender below. It's needed so people looking for a woman or a man can find you. "
               "You can keep it hidden, but a hidden detail never decides a match, so then only people open to "
               "anyone can match you, and you may get fewer matches.")
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
        "format": list(FORMATS), "travel": list(TRAVEL),
        "smoking": _vocab.SMOKING, "children": _vocab.CHILDREN,
        "neighborhood": [], "employer": [],
        "income_band": _vocab.INCOME[country] if country else ["lower income", "middle income", "higher income"],
        "religion": _vocab.RELIGION, "politics": _vocab.POLITICS, "health": _vocab.HEALTH,
        "kinks": _vocab.KINKS,
        "kind": list(MATCH_KINDS), "wants_gender": list(GENDERS), "wants_age": [],
    }


def _find_language(pattern: str, low: str):
    for m in _re.finditer(pattern, low):
        word = next(g for g in m.groups() if g)
        lang = next((l for l in _vocab.LANGUAGES if l.lower() == word), None)
        if lang:
            return lang
    return None


def wanted_ages(text: str, age_band) -> tuple:
    """(ages in years as "lo-hi" or None, a note or None): the ages a dating description asks for,
    absolute ("early 30s", "25 to 34") or relative to their own age band ("around my age": one band
    either side of theirs). Belong. introduces adults only, so nobody under 18 is ever recorded."""
    _, years = _dating_wants(text)
    years = years or relative_years(text, age_band)
    if years is None:
        return None, None
    lo, hi = years[0], min(years[1], AGE_MAX)
    if hi < AGE_MIN:
        return None, "Belong. only introduces adults, so I haven't added an age range. Who would you like to meet?"
    note = None
    if lo < AGE_MIN:
        lo, note = AGE_MIN, f"Belong. only introduces adults, so the ages start at {AGE_MIN}."
    value = f"{lo}-{hi}"
    return (value, note) if age_span(value) else (None, None)


# A partner for an activity, named: "a tennis partner", "a hiking buddy". Romantic words aren't an activity.
_PARTNER_PHRASE = _re.compile(r"\b(?:a|an|some)\s+((?:[a-z]+\s+){0,2}?[a-z]+)\s+(?:partner|buddy|companion)s?\b")
_NOT_AN_ACTIVITY = {"life", "long", "term", "long-term", "lifelong", "romantic", "serious", "loving", "committed",
                    "steady", "new", "nice", "good", "kind", "great", "potential", "future", "real", "special",
                    "perfect", "ideal", "business", "possible", "the", "my", "your", "our"}


def activity_partner(text: str):
    """The activity of a partner they name ("a tennis partner" -> "tennis"), or None."""
    for m in _PARTNER_PHRASE.finditer(" ".join(text.lower().split())):
        said = m.group(1).split()
        if said[-1] not in _NOT_AN_ACTIVITY:  # "a long-term partner" names no activity
            return " ".join(w for w in said if w not in _NOT_AN_ACTIVITY)
    return None


def wants_text(genders, ages) -> str:
    """Who a dating listing would like to meet, in plain words: "a woman, 33 to 42", "anyone, 40 or older"."""
    genders = [g for g in GENDERS if g in (genders or ())]
    who = ("anyone" if set(genders) == set(GENDERS) else
           " or ".join("a non-binary person" if g == "nonbinary" else f"a {g}" for g in genders))
    span = age_span(ages)
    when = (f"{span[0]} or older" if span[1] >= AGE_MAX else f"{span[0]} to {span[1]}") if span else ""
    return ", ".join(p for p in (who, when) if p)


# --- Asking when it's unclear ----------------------------------------------------------------
# A message can mean more than one kind of match: wanting to meet a woman, and mentioning climbing
# (a date, or a climbing partner?); or "a partner" with nothing to say whether it's romantic or for
# an activity. Then the agent asks one short question instead of guessing, and "both" makes two
# listings, each with its own kind. Read generally: the dating reader and an activity reader both
# fire. Romantic words leave no doubt about wanting a date, so an activity they only say they enjoy
# ("I love climbing") is a hobby; one they want to do with someone ("a climbing partner") still asks.
_ASK_NAMES = {"dating": "a date", "climbing": "a climbing partner", "running": "a running buddy",
              "language": "a language exchange", "cofounder": "a cofounder", "tutoring": "tutoring",
              "roommate": "a roommate"}
_PARTNER_ALONE = _re.compile(r"\b(?:a|an|my)\s+partner\b")
ASK_HINT = "Just say which, or \"both\"."


def _activities(text: str) -> list:
    return [k for k, pat in _KIND_PATTERNS if _re.search(pat, text)]


def kind_question(message: str):
    """None when the kind of match is clear; otherwise {"text": one short question, "options": the
    kinds it could be, each {"kind", "label", "activity"}}. An option with kind None is a partner for
    an activity not named yet."""
    low = " ".join(message.lower().split())
    if not _is_dating(low):
        return None
    romantic = bool(_ROMANTIC.search(_set_aside(low)))
    purpose = _HOBBY.sub(" ", low)  # without what they only say they enjoy
    activities = _activities(purpose) if romantic else _activities(low)
    partner_for = activity_partner(purpose if romantic else low)
    if activities:
        other = {"kind": activities[0], "label": _ASK_NAMES[activities[0]], "activity": None}
    elif partner_for:
        other = {"kind": "other", "label": f"{_article(partner_for)} {partner_for} partner", "activity": partner_for}
    elif not romantic and _PARTNER_ALONE.search(low) and not _TASK_AFTER.search(low) and not _MEETING.search(
            _PARTNER_ALONE.sub(" ", low)):
        other = {"kind": None, "label": "a partner for an activity", "activity": None}  # "a partner", nothing more
    else:
        return None
    options = [{"kind": "dating", "label": _ASK_NAMES["dating"], "activity": None}, other]
    which = " (which one)" if other["kind"] is None else ""
    return {"text": f"Are you looking for {options[0]['label']}, or {other['label']}{which}? Or both? I can make "
                    f"two separate listings.", "options": options}


_BOTH = _re.compile(r"\b(?:both|either|all of (?:them|those)|two listings|each)\b")
_DATE_WORDS = r"\b(?:dat(?:e|es|ing)|romance|romantic|relationship|boyfriend|girlfriend|someone special)\b"
_NEGATED = _re.compile(r"(?:\b(?:not|no|never|without)|n't)(?:\s+[a-z']+){0,2}\s*$")  # within the same phrase
_CHOICE_FILLER = {"a", "an", "the", "i", "im", "m", "am", "looking", "for", "want", "just", "only", "please", "yes",
                  "no", "not", "partner", "buddy", "someone", "to", "with", "and", "or", "both", "one", "s", "it",
                  "thanks", "thank", "you", "would", "like", "go", "do", "play", "together", "activity", "in", "on",
                  "at", "of", "some", "that", "this", "is", "be", "me", "my", "really", "actually", "oh", "ok", "okay"}


def says_what_they_want(message: str) -> bool:
    """Does this message ask for a kind of match on its own (a new request, not an answer)?"""
    return _parse_intent(message).kind is not None or activity_partner(message) is not None


def _chose(low: str, pattern: str) -> bool:
    """The answer names this, at least once without "not" / "no" / "never" just before it."""
    return any(not _NEGATED.search(low[:m.start()]) for m in _re.finditer(pattern, low))


def chosen_kinds(answer: str, options: list):
    """What the person chose, in answer to kind_question: a list of {"kind", "activity"} (in the
    order asked), or None if the answer doesn't say. "Both" is every option; a negated choice ("not a
    date") doesn't count. A partner for an activity not named yet takes the activity from the answer
    ("tennis"); with "both" and no activity yet, its kind stays None, to be asked."""
    low = " ".join((answer or "").lower().split())
    both = bool(_BOTH.search(low))
    dating = both or _chose(low, _DATE_WORDS)
    out = []
    for o in options:
        if o["kind"] == "dating":
            if dating:
                out.append({"kind": "dating", "activity": None})
        elif o["kind"] is None:  # which activity: the one they name now
            named, partner_for = _activities(low), activity_partner(low)
            places = {w for place in [*_vocab.CITIES, *COUNTRIES] for w in place.lower().split()}
            content = [w for w in _re.findall(r"[a-z]+", _re.sub(_DATE_WORDS, " ", low))
                       if w not in _CHOICE_FILLER and w not in places]  # a word or two: the activity
            if named:
                out.append({"kind": named[0], "activity": None})
            elif partner_for:
                out.append({"kind": "other", "activity": partner_for})
            elif content and len(content) <= 3 and (both or not dating):
                out.append({"kind": "other", "activity": " ".join(content)})
            elif both:
                out.append({"kind": None, "activity": None})
        else:
            words = [w for w in _re.findall(r"[a-z]+", o["label"]) if w not in ("a", "an", "partner", "buddy")]
            pattern = dict(_KIND_PATTERNS).get(o["kind"]) or r"\b(?:" + "|".join(map(_re.escape, words)) + r")\b"
            if both or _chose(low, pattern):
                out.append({"kind": o["kind"], "activity": o.get("activity")})
    return out or None


def draft_from_message(message: str, first_name: str, age_band: str, kind=None, activity=None) -> dict:
    """Read the person's own words and suggest a listing. Returns a plain dict the
    app shows for review; nothing is submitted until the person says so.

    kind: the kind of match the person chose when asked (kind_question), instead of the one their
    words suggest. activity: for "other", the activity they want a partner for ("tennis")."""
    text = " ".join(message.split())[:500]
    low = text.lower()
    parsed = _parse_intent(text)
    activity = activity or (activity_partner(text) if kind in (None, "other") and parsed.kind is None else None)
    kind = kind or parsed.kind or ("other" if activity else None)
    if kind != parsed.kind:
        parsed = read_listing(text, {"kind": kind}.get)  # read as that kind of listing
    v = {"first_name": first_name, "age_band": age_band}
    if kind:
        v["kind"] = kind  # what the agent understood, recorded on the listing (shown)
    if kind == "other" and activity:
        phrase = f"{_article(activity)} {activity} partner"
        v["seeking"] = v["offering"] = phrase
        v["headline"] = f"Looking for {phrase}"[:60]
        v["topics"] = (activity,)
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
    # Interests: the usual ones, and anything they say they enjoy ("I enjoy rock climbing and tennis").
    interests = list(dict.fromkeys([i for i in _vocab.INTERESTS if i in low] + quick_facts(text).get("interests", [])))
    if interests:
        v["interests"] = interests[:8]
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
        if gender := own_gender(low):
            v["gender"] = gender
        # Who they'd like to meet, and the ages, as the agent understood them (shown, so matching uses
        # them and the person can check them on their profile).
        if parsed.seeking:
            v["wants_gender"] = tuple(g for g in GENDERS if g in parsed.seeking)
        ages, note = wanted_ages(text, age_band)
        if ages:
            v["wants_age"] = ages
        if note:
            notes.append(note)
    diet = next((d for d in _vocab.DIET if _re.search(r"\b" + d + r"\b", low)), None)
    if diet:
        v["diet"] = diet
    if _re.search(r"non-smoker|don't smoke|do not smoke", low):
        v["smoking"] = "doesn't smoke"
    fmt = format_from_text(low)
    if fmt and kind != "dating":  # dating is always in person
        v["format"] = fmt
    travel = travel_from_text(low)
    if travel:
        v["travel"] = travel

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
    if kind == "dating" and "wants_gender" not in v:
        notes.append("Who would you like to meet: a woman, a man, or anyone? Set it under \"Looking for\" below.")
    if kind == "dating" and "wants_age" not in v:
        notes.append(f"You didn't say what ages you'd like, so for now you're open to "
                     f"{ages_text(default_age_range(age_band))} (one age band either side of yours). Tell me, or "
                     f"set \"Their age\" below, to change that.")
    if kind == "dating" and "gender" not in v:
        notes.append(GENDER_NOTE)

    keys = (["kind"] if kind else []) + _COMMON + _BY_KIND.get(kind, [])
    keys += [k for k in ("headline", "topics") if k in v]
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
    if kind == "other" and activity:
        understood = [f"Looking for {v['seeking']}"]
    if city:
        understood.append(f"in {city}")
    if kind == "dating" and (v.get("wants_gender") or v.get("wants_age")):
        understood.append("with " + wants_text(v.get("wants_gender"), v.get("wants_age")))
    if v.get("format") == "Online":
        understood.append("(online only)")
    elif parsed.remote_ok:
        understood.append("(online/remote is fine)")
    if v.get("travel") and TRAVEL[v["travel"]]:
        understood.append(f"(will travel {v['travel'].lower()})")
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
    if gender := own_gender(low):
        out["gender"] = gender  # "I'm a woman", "I am non-binary"

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
    fmt = format_from_text(low)
    if fmt and has_format(_parse_intent(low).kind):  # "climbing in Osaka, online works for me": no format
        out["format"] = fmt
    travel = travel_from_text(low)
    if travel:
        out["travel"] = travel

    # Teaching, stated plainly: "I'm offering piano lessons", "I teach guitar" -> they're the
    # teacher; "looking for a beginner ice hockey coach" -> they're the student.
    m = _re.search(r"\b(?:i'?m |i am )?(?:offering|giving|teaching|i teach|i give|i offer|i can teach)\s+"
                   r"(?:(?:private|group|beginner|online) )?([a-z][a-z ]{1,30}?)\s*(?:lessons?|classes|coaching|tutoring)\b", low) \
        or _re.search(r"\bi (?:teach|coach|tutor)\s+([a-z][a-z ]{1,30}?)(?=[.!?;,]|$|\s(?:to|in|for|on|and)\b)", low)
    if m and m.group(1).strip() not in ("some", "a", "the"):
        activity = m.group(1).strip()
        out["offering"], out["seeking"] = f"{_article(activity)} {activity} teacher", f"{_article(activity)} {activity} student"
    m = _re.search(r"\b(?:looking for|need|want|find|searching for)\s+(?:a |an )?((?:beginner |private |good )?"
                   r"[a-z][a-z ]{1,30}?)\s+(teacher|tutor|coach|instructor|mentor)\b", low)
    if m and "offering" not in out:
        what, role = m.group(1).strip(), m.group(2)
        activity = _re.sub(r"^(?:beginner|private|good) ", "", what)
        out["seeking"] = f"{_article(what)} {what} {role}"
        out["offering"] = f"{_article(activity)} {activity} student"

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


def _article(phrase: str) -> str:
    return "an" if phrase[:1] in "aeiou" else "a"


_ONLINE_WORDS = r"online|remote(?:ly)?|over (?:zoom|video)|video calls?"
_IN_PERSON_WORDS = r"in[- ]person|face to face|meet up|meetups?"


def has_format(kind) -> bool:
    """Does a listing of this kind say how people meet (a "format" field)? Climbing, running,
    roommates and dating don't; a kind not known yet might."""
    return kind not in _BY_KIND or "format" in _BY_KIND[kind]


def format_from_text(low: str):
    """How they want to meet, when they say so plainly (one of FORMATS), else None. Online only
    takes the words "only" or "just" ("online only", "only online", "just online"); any other
    mention of online ("online works for me", "or online") keeps meeting in person open. "Not
    online", "no remote" or "without video calls" means in person, and so does "only in person"."""
    online, in_person = said(low, _ONLINE_WORDS), said(low, _IN_PERSON_WORDS) is True
    if online is None:
        return "In person" if in_person else None
    if online is False or only_said(low, _IN_PERSON_WORDS):
        return "In person"
    if not in_person and only_said(low, _ONLINE_WORDS):
        return "Online"
    return "In person or online"


def travel_from_text(low: str):
    """How far they'll travel to meet, when they say so plainly ("within 30 km", "up to 20 miles"):
    the furthest fixed choice (TRAVEL) that isn't further than what they said. Else None."""
    m = _re.search(r"\b(?:within|up to|upto|under|less than|no more than|travel(?:l?ing)?)\s+(\d+)\s*"
                   r"(km|kilomet(?:er|re)s?|miles?|mi)\b", low)
    if m:
        km = int(m.group(1)) * (1.609 if m.group(2).startswith("mi") else 1)
        return max((c for c, d in TRAVEL.items() if d <= km), key=TRAVEL.get)
    if _re.search(r"\bsame city\b|\b(?:only|just) in (?:my|the) (?:city|town)\b", low):
        return "Same city"
    return None


def dating_wants(draft) -> dict | None:
    """For the person's own dating draft, read exactly the way matching reads it (read_listing, from
    the details they show, then their description): who they'd like to meet (genders), those
    people's ages in years [lo, hi] (None if not said), and the ages they're open to until they say
    (one age band either side of their own). Only ever sent to their own phone."""
    if not draft or draft.get("kind") != "dating":
        return None
    shown = {}
    for f in draft.get("fields", []):
        value = f.get("value")
        if f.get("enabled", True) and value not in (None, "", []) and f.get("visibility", SHOWN) == SHOWN:
            shown[f.get("key")] = tuple(value) if isinstance(value, list) else value
    shown.setdefault("kind", "dating")
    parsed = read_listing(draft.get("intent_text") or "", shown.get)
    span = age_span(shown.get("wants_age"))
    if span is None and parsed.age_range:
        span = (AGE_BAND_BOUNDS[parsed.age_range[0]][0], AGE_BAND_BOUNDS[parsed.age_range[1]][1])
    lo, hi = default_age_range(shown.get("age_band"))
    return {"genders": [g for g in GENDERS if g in parsed.seeking], "ages": list(span) if span else None,
            "default": [AGE_BAND_BOUNDS[lo][0], AGE_BAND_BOUNDS[hi][1]]}


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
    keys filled. Fields the draft doesn't have yet are added, except a format on a kind that has
    none (has_format): a climbing listing isn't made online only by "online works for me". The kind
    of match is the agent's to record, never carried over, and who someone would like to meet and
    their ages belong only on a dating draft."""
    filled = []
    by_key = {f["key"]: f for f in draft["fields"]}
    for key, value in facts.items():
        spec = FIELD_REGISTRY.get(key)
        if spec is None or spec.sensitive or key == "kind":
            continue
        if key in DATING_ONLY and draft.get("kind") != "dating":
            continue
        f = by_key.get(key)
        if f is None and key == "format" and not has_format(draft.get("kind")):
            continue
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
