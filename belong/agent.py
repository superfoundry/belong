"""Personal agent (SPEC §3): works for one person, outside the room.

It drafts the listing from the person's details and choices, relays answers to
follow-up questions, and asks the person which hidden details they would reveal to
a connection. It only ever sees its own person's data.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from functools import lru_cache

from .schema import (AGE_BAND_BOUNDS, AGE_BANDS, AGE_MAX, AGE_MIN, ANY, CITY_COORDS, FIELD_REGISTRY, FORMATS,
                     FREE_TEXT_MAX, GENDERS, HIDDEN, MATCH_KINDS, MAX_VALUES, SENSITIVE_FIELD, SENSITIVE_WORDS, SHOWN,
                     LIST_UNSPACED_SCRIPTS, STRICT_EXACTNESS, TRAVEL, Field, Listing, age_span, canonical, detail_key,
                     display_form, is_detail_key, mentions_sensitive, only_said, plain_place, said, says_nothing,
                     split_key)
from .schema import _CONTRACTED


def with_wants_age(fields: tuple, value: str) -> tuple:
    """The listing's fields with "wants_age" set to `value` (years, "lo-hi"): an existing one keeps
    the person's own choice of shown or hidden, a new one is shown, like any field by default."""
    if any(f.key == "wants_age" for f in fields):
        return tuple(replace(f, value=value) if f.key == "wants_age" else f for f in fields)
    return tuple(fields) + (Field("wants_age", value, FIELD_REGISTRY["wants_age"].default, False),)


def answer_followup(p, listing: Listing, question_id: str, now: str, rng: random.Random,
                    phrase: str = "") -> Listing:
    """A synthetic person (synth.Persona) answers a fixed-menu question; the agent updates the listing."""
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
        fields = with_wants_age(fields, f"{lo_age}-{AGE_BAND_BOUNDS[hi][1]}")  # and recorded as understood
        p.truth.omit_age = False
    elif question_id == "broaden":
        if rng.random() < 0.6:  # most say yes: close matches only (questions keep their own choice)
            return replace(listing, notify_close=True, updated_at=now, version=listing.version + 1)
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

from .matching.intent import (_MEETING, _ROMANTIC, _TASK_AFTER, HOBBY_END, _is_dating, _set_aside,
                              ages_text, dating_wants as _dating_wants, default_age_range, own_gender,
                              parse_intent as _parse_intent, read_listing, relative_years,
                              request_kind as _dating_request, without_hobbies)

LABELS = {
    "first_name": "First name", "city": "City", "country": "Country", "age_band": "Age band (from ID check)",
    "gender": "Gender", "seeking": "Looking for", "offering": "You are (to them)", "topics": "Related topics",
    "headline": "Headline", "pay": "Pay", "format": "Format", "travel": "How far you'll travel",
    "children": "Children", "neighborhood": "Neighborhood", "employer": "Employer",
    "income_band": "Income", "religion": "Religion", "politics": "Politics", "health": "Health",
    "kinks": "Intimacy preferences",
    "kind": "Kind of match", "wants_gender": "Looking for", "wants_age": "Their age",
}
assert set(LABELS) == set(FIELD_REGISTRY)

_COMMON = ["first_name", "country", "city", "travel", "age_band", "neighborhood", "employer", "income_band",
           "religion", "politics", "health"]
# Only dating records who they'd like to meet and their ages.
DATING_ONLY = ("wants_gender", "wants_age")
# What each kind of match has besides the common details: dating its fixed rules, everything else the
# phrases matching compares and its terms. Open details (detail_row) go on either kind.
_BY_KIND = {"dating": ["gender", *DATING_ONLY, "children", "kinks"], "other": ["seeking", "offering", "pay", "format"]}
_KIND_NAMES = {"dating": "a relationship or date", "other": "something else"}
assert tuple(_KIND_NAMES) == MATCH_KINDS and set(_BY_KIND) == set(MATCH_KINDS)
# Why dating needs your gender, and what hiding it means (hidden details never decide a match).
GENDER_NOTE = ("Please tell me your gender. It's needed so people looking for a woman or a man can find you. "
               "You can keep it hidden, but a hidden detail never decides a match, so then only people open to "
               "anyone can match you, and you may get fewer matches.")


def field_options(kind=None, city=None) -> dict:
    """The fixed choices of the built-in details that have them. The same for every kind and place (the
    arguments change nothing); every other detail is written in words, and an open detail's choices are
    its own steps (detail_row)."""
    return {"gender": list(GENDERS), "format": list(FORMATS), "travel": list(TRAVEL), "kind": list(MATCH_KINDS),
            "wants_gender": list(GENDERS)}


def label_for(key: str) -> str:
    """What the phone calls a detail: a built-in's label, or an open detail's name in sentence case."""
    if key in LABELS:
        return LABELS[key]
    name = split_key(key)[1]
    return name[:1].upper() + name[1:]


def detail_row(about: str, name: str, compare: str, steps, values, visibility: str = SHOWN,
               required: bool = True) -> dict:
    """An open detail as a draft carries it (the same row shape as a built-in, plus what an open detail
    has): its key is about:name, its value always a list, and its choices are its own steps."""
    name = canonical(name)
    return {"key": detail_key(about, name), "label": label_for(detail_key(about, name)), "about": about,
            "name": name, "compare": compare, "steps": list(steps), "required": bool(required),
            "value": list(values), "visibility": visibility, "sensitive": False, "multi": True, "enabled": True,
            "locked": False, "options": list(steps)}


def detail_field(row: dict):
    """The Field an open detail's row goes live as, or None if the row isn't an open detail. Its form is
    the Warden's to check (schema.detail_problems), exactly as the person's agent wrote it."""
    key = row.get("key") if isinstance(row, dict) else None
    if not is_detail_key(key):
        return None
    value = row.get("value")
    values = value if isinstance(value, (list, tuple)) else [] if value in (None, "") else [value]
    steps = row.get("steps") if isinstance(row.get("steps"), (list, tuple)) else ()
    return Field(key, tuple(display_form(v) for v in values), row.get("visibility"), False,
                 row.get("compare", ""), tuple(display_form(s) for s in steps), row.get("required", True) is not False)


def wanted_ages(text: str, age_band) -> tuple:
    """(ages in years as "lo-hi" or None, a note or None): the ages a dating description asks for,
    absolute ("early 30s", "25 to 34", "no one over 40") or relative to their own age band ("around my
    age": one band either side of theirs). Belong introduces adults only, so nobody under 18 is ever
    recorded. A bare number ("a woman, 30") is never turned into a range: the note asks instead."""
    _, years = _dating_wants(text, guess=False)
    years = years or relative_years(text, age_band)
    if years is None:
        if _dating_wants(text)[1] is None:
            return None, None
        return None, (f"I wasn't sure which ages you meant, so I haven't set them yet: for now you're open to "
                      f"{ages_text(default_age_range(age_band))} (one age band either side of yours). Tell me a "
                      f"range, e.g. \"30 to 39\".")
    lo, hi = years[0], min(years[1], AGE_MAX)
    if hi < AGE_MIN:
        return None, "Belong only introduces adults, so I haven't added an age range. Who would you like to meet?"
    note = None
    if lo < AGE_MIN:
        lo, note = AGE_MIN, f"Belong only introduces adults, so the ages start at {AGE_MIN}."
    value = f"{lo}-{hi}"
    return (value, note) if age_span(value) else (None, None)


# --- Sensitive words ---------------------------------------------------------------------------
# The fixed sensitive words (schema.SENSITIVE_WORDS, mentions_sensitive) are published in schema.py and
# shared with the Warden and the audit. They are an extra safeguard, not what keeps a person's words out of
# what is shown: the offline agent writes shown details only from words other people already show (below),
# and none of those is sensitive. They are also how a sensitive thing said in a message is recorded,
# hidden, under its built-in category (sensitive_details), and why the model agent's save is refused when
# one is in a shown detail or an open detail (agent_llm.build_draft).

def without_sensitive(text) -> str:
    """The text in lower case, with every sensitive word taken out."""
    low = " ".join(str(text or "").lower().split())
    for rx in SENSITIVE_WORDS.values():
        low = rx.sub(" ", low)
    return " ".join(low.split())


def sensitive_details(text: str, kind) -> dict:
    """The sensitive things a message says, as {hidden built-in: the words as said}, one for each category
    that has a built-in of its own (intimacy only on dating listings). Who someone is (orientation, gender
    identity) has none, so it stays in the description only. These go on the listing hidden, never shown. A
    category named after whose it is counts only as the person's own, with what they say about it ("my long-term
    health" as "long-term health"): never the bare name ("my salary"), nor someone else's ("their income", "my
    sister's health"); those are only warned about (draft_from_message)."""
    said_ = " ".join(str(text or "").split())
    low = said_.lower()
    out = {}
    for category, rx in SENSITIVE_WORDS.items():
        key = SENSITIVE_FIELD[category]
        if not key or (key == "kinks" and kind != "dating"):
            continue
        for m in rx.finditer(low):
            word = said_[m.start():m.end()] if len(said_) == len(low) else m.group(0)  # as they wrote it
            whose = _WHOSE.match(m.group(0))
            if whose:  # a category named after whose it is ("my health", "their income", "my mental health")
                if whose.group(1) not in ("my", "our"):
                    continue  # someone else's: nothing about the person
                word = word[whose.end():]
                if len(word.split()) < 2:
                    continue  # only the category's own name ("my salary"): it says nothing to record
                if _OTHERS_OWN.match(word):
                    continue  # "my uncle's income": someone else's, named after the person (still warned about)
            out[key] = (word,) if FIELD_REGISTRY[key].multi else word
            break
    return out


# A sensitive category named after whose it is, as the free-text reading counts it (schema.SENSITIVE_WORDS).
_WHOSE = _re.compile(r"(my|our|your|his|her|their|whose)\s+")
# A word after "my" / "our" that is itself a possessive ("<word>'s", "<word>s'"): what follows is that person's.
_OTHERS_OWN = _re.compile(r"[\w'’-]*(?:['’]s|s['’])\s")


# --- What the offline agent recognises ---------------------------------------------------------
# The range of what people type is endless, so the offline agent never copies a person's own words into a
# detail that is shown. Its vocabulary is the shared list of details in use (Warden.details_in_use): words at
# least K_CROWD other people already show, so nothing on it is anyone's own. It writes a listed topic, or what
# a listed partner phrase is for, as what they want a partner for ("a ... partner", "a ... teacher"), a listed
# value under its listed name, and a listed place. Everything else stays only in the description, which is
# never shown, and where it matters (which activity?) the agent asks instead.

# A partner for an activity, named: "a kayaking partner", "a quiz buddy". Romantic words aren't an activity.
_PARTNER_PHRASE = _re.compile(r"\b(?:a|an|some)\s+((?:[a-z'-]+\s+){0,2}?[a-z'-]+)\s+(?:partner|buddy|companion)s?\b")
_NOT_AN_ACTIVITY = {"life", "long", "term", "long-term", "lifelong", "romantic", "serious", "loving", "committed",
                    "steady", "new", "nice", "good", "kind", "great", "potential", "future", "real", "special",
                    "perfect", "ideal", "business", "possible", "the", "my", "your", "our"}


def _lower(text) -> str:
    return " ".join(str(text or "").lower().replace("’", "'").split())


def _words_rx(words) -> _re.Pattern:
    """One pattern for any of these words or phrases, longest first ("sea kayaking" before "kayaking")."""
    words = sorted(set(words), key=lambda w: (-len(w), w))
    return _re.compile(r"(?<![\w'-])(" + "|".join(map(_re.escape, words)) + r")(?![\w'-])") if words else None


@dataclass(frozen=True)
class Words:
    """What the offline agent may write, all from the shared list of details in use (words_in_use).

    activities: what each listed partner phrase is for ("a kayaking partner" -> "kayaking"), then the
    listed topics, never one that reads as dating. values: {value: (about, name, compare, steps)} for each
    listed value it may write as the person's own (me) or the request's (both), and every step of a listed
    scale about the person or the request (the scale is listed, so each of its steps is a listed word): only
    one listed under exactly one name, never "any". cities: {plain city: {countries}}, the cities with a known
    centre (schema.CITY_COORDS) and the listed places; a city in two countries ("Cambridge") has no country of
    its own."""
    activities: tuple = ()
    values: dict = field(default_factory=dict)
    cities: dict = field(default_factory=dict)
    _activity_rx: object = field(default=None, compare=False, repr=False)
    _value_rx: object = field(default=None, compare=False, repr=False)
    _city_rx: object = field(default=None, compare=False, repr=False)


def _entries(vocabulary, section: str) -> list:
    entries = (vocabulary or {}).get(section) if isinstance(vocabulary, dict) else None
    return [e for e in entries or [] if isinstance(e, dict)]


def words_in_use(vocabulary=None) -> Words:
    """The offline agent's vocabulary from the Warden's shared list (DetailsInUse.vocabulary(); None or
    empty: nothing listed yet, so it writes no phrase or open detail at all)."""
    phrases = _entries(vocabulary, "phrases")
    partners = {m.group(1) for e in phrases if e.get("key") in ("seeking", "offering")
                for m in [_PARTNER_PHRASE.fullmatch(_lower(e.get("value")))] if m}
    topics = {_lower(e.get("value")) for e in phrases if e.get("key") == "topics"} - partners
    # What people already look for partners for first (the examples activity_hint gives), then the topics.
    activities = [a for a in sorted(partners) + sorted(topics) if a and a.split()[-1] not in _NOT_AN_ACTIVITY
                  and not _dating_request(a) and not mentions_sensitive(a)]

    compare = {(e.get("about"), e.get("name")): e.get("compare") for e in _entries(vocabulary, "details")}
    scales = {(e.get("about"), e.get("name")): tuple(e.get("steps") or ()) for e in _entries(vocabulary, "scales")}
    found: dict = {}  # canonical value -> {name: (about, name, compare, steps, display form)}
    # Each listed value, then each step of a listed scale (so a message naming two steps is seen to, even
    # where only one of them is listed as a value).
    said_ = [(e.get("about"), e.get("name"), e.get("value")) for e in _entries(vocabulary, "values")]
    said_ += [(about, name, step) for (about, name), steps in scales.items()
              if compare.get((about, name)) == "steps" for step in steps]
    for about, name, value in said_:
        how = compare.get((about, name))
        if about not in ("me", "both") or how is None or not isinstance(value, str) or canonical(value) == ANY \
                or mentions_sensitive(value, in_detail=True):
            continue
        steps = scales.get((about, name), ())
        if how == "steps" and canonical(value) not in {canonical(s) for s in steps}:
            continue  # a step it can't place on its scale
        names = found.setdefault(canonical(value), {})
        if name not in names or about == "me":  # about the person, where the same name is listed both ways
            names[name] = (about, name, how, steps, value)
    values = {_lower(v[4]): v[:4] for names in found.values() if len(names) == 1 for v in names.values()}

    cities: dict = {}
    for city, where in CITY_COORDS:
        cities.setdefault(city, set()).add(_country(where) or _title(where))
    for e in _entries(vocabulary, "places"):
        city, country = plain_place(e.get("city")), e.get("country")
        if city and isinstance(country, str) and not mentions_sensitive(city):
            cities.setdefault(city, set()).add(_country(country) or _title(country))
    return Words(tuple(activities), values, cities, _words_rx(activities), _words_rx(values), _words_rx(cities))


@lru_cache(maxsize=1)
def _nothing_listed() -> Words:
    return words_in_use(None)


def _words(words) -> Words:
    """`words` as Words: Words itself, the Warden's vocabulary dict, or None (nothing listed)."""
    return words if isinstance(words, Words) else words_in_use(words) if words else _nothing_listed()


def known_activities(text, words=None) -> list:
    """The listed activities a text names, in order ("I enjoy kayaking, pottery and quizzes" -> kayaking and
    quizzes, if both are listed)."""
    rx = _words(words)._activity_rx
    return list(dict.fromkeys(m.group(1) for m in rx.finditer(_lower(text)))) if rx else []


def known_activity(text, words=None):
    """The first listed activity a text names, or None."""
    return next(iter(known_activities(text, words)), None)


def _partner_phrases(text) -> list:
    """What each "a ... partner" phrase names, unless it's romantic ("a long-term partner")."""
    return [m.group(1) for m in _PARTNER_PHRASE.finditer(_lower(text))
            if m.group(1).split()[-1] not in _NOT_AN_ACTIVITY]


def activity_partner(text: str, words=None):
    """The listed activity of a partner they name ("a kayaking partner" -> "kayaking", "a Christian kayaking
    buddy" -> "kayaking"), or None. Only a listed activity counts: "a pottery partner" (if pottery isn't
    listed) or "a Muslim partner" names none, and nothing else from the phrase is ever used."""
    return next(filter(None, (known_activity(w, words) for w in _partner_phrases(text))), None)


def unknown_partner(text, words=None) -> bool:
    """Do they ask for a partner for something not listed ("a pottery partner")? Then it asks which
    activity, instead of writing their words on the listing."""
    return any(not known_activity(w, words) for w in _partner_phrases(text))


# Teaching a listed activity, stated plainly: "I'm offering ukulele lessons", "I teach ukulele" -> they're the
# teacher; "looking for a ukulele teacher" -> they're the student. Read from a few explicit forms only.
_TEACH = (_re.compile(r"\b(?:i'?m |i am )?(?:offering|giving|teaching|i teach|i give|i offer|i can teach)\s+"
                      r"(?:(?:private|group|beginner|online) )?([a-z][a-z ]{1,30}?)\s*"
                      r"(?:lessons?|classes|coaching|tutoring)\b"),
          _re.compile(r"\bi (?:teach|coach|tutor)\s+([a-z][a-z ]{1,30}?)(?=[.!?;,]|$|\s(?:to|in|for|on|and)\b)"))
_LEARN = _re.compile(r"\b(?:looking for|need|want|find|searching for)\s+(?:a |an )?([a-z][a-z ]{1,30}?)\s+"
                     r"(teacher|tutor|coach|instructor|mentor)\b")


def teaching(text, words=None):
    """(seeking, offering, activity) when they plainly teach a listed activity or want to learn one, else None."""
    low = _lower(text)
    m = next(filter(None, (rx.search(low) for rx in _TEACH)), None)
    activity = known_activity(m.group(1), words) if m else None
    if activity:
        return f"{_article(activity)} {activity} student", f"{_article(activity)} {activity} teacher", activity
    m = _LEARN.search(low)
    activity = known_activity(m.group(1), words) if m else None
    if activity:
        return f"{_article(activity)} {activity} {m.group(2)}", f"{_article(activity)} {activity} student", activity
    return None


# Activities someone says they do or enjoy, in the first person: "I play squash", "I go kayaking", "I do
# pilates", "I enjoy kayaking and quizzes". Only listed activities count from these ("I love dogs" names none).
# Read only to ask a question, never to decide anything.
_I_DO = _re.compile(r"\b(?:i|we)\s+(?:could\s+|can\s+|might\s+|would\s+)?"
                    r"(?:also\s+|really\s+|often\s+|usually\s+|regularly\s+|sometimes\s+|mostly\s+)?"
                    r"(?:play|go|do|practi[cs]e)s?\b"
                    # up to where a new clause starts: "I play squash | and want a girlfriend"
                    r"(?:(?!\s+(?:and|but|so)\s+(?:i\b|i'?m|i'?d|we\b|want|wanna|need|looking|would|am|are|hope|"
                    r"keen|trying))[^.,;!?])*")
_I_ENJOY = _re.compile(r"\b(?:i'?m into|i am into|i (?:really |also |just )?(?:enjoy|love|like|adore)|"
                       r"my hobbies (?:are|include)|i'?m (?:a )?(?:big )?fan of)\s+([a-z][a-z ,'\-]{2,80}?)"
                       r"(?=[.!?;]|$|\s(?:but|so|because)\b|\sand i\b|\s" + HOBBY_END + ")")
# How someone describes the person they'd like to meet: "a girlfriend who skis", "someone that loves
# quizzes", "a date with a sailor", "ideally a squash player". An activity there is that person's, not
# something they want to do together, so it is never a choice of activity, nor anything about them.
_TRAITS = _re.compile(r"\b(?:who|that|whose)\b[^.,;!?]*"
                      r"|\bwith\s+(?:a|an|someone|somebody|people|men|women|guys|girls)\b[^.,;!?]*"
                      r"|\b(?:a|an)\s+(?:[a-z'-]+\s+)?players?\b")


def _first_person_activities(low: str, words=None) -> list:
    spans = [m.group(0) for m in _I_DO.finditer(low)] + [m.group(1) for m in _I_ENJOY.finditer(low)]
    return list(dict.fromkeys(a for span in spans for a in known_activities(span, words)))


def activities_said(text: str, first_person: bool = True, anywhere: bool = False, words=None) -> list:
    """The listed activities a text names, as ("other", activity) pairs: a partner for one ("a kayaking
    partner"), then, with first_person, those they say they do or enjoy ("I play squash"), then, with
    anywhere, any other one it names (an answer to "which activity?": "squash"). Sensitive words are
    taken out first."""
    low = without_sensitive(text)
    named = [activity_partner(low, words)] + (_first_person_activities(low, words) if first_person else [])
    return list(dict.fromkeys(("other", a) for a in filter(None, named + (known_activities(low, words) if anywhere
                                                                            else []))))


def asked_kind(low: str):
    """"dating" when a message asks for a date, as the offline agent reads it: outside what they say they
    enjoy and, when they'd like to meet someone, outside how they describe that person ("a girlfriend who
    skis" and "a date with a sailor" are dating). Else None: any other request is read from its words
    (activity_partner, teaching)."""
    purpose = without_hobbies(low)
    if _is_dating(purpose):
        if _ROMANTIC.search(_set_aside(low)):
            # Romantic wording leaves no doubt: what they say they do ("I sail a lot") is a hobby, exactly
            # as kind_question reads it, so the two never disagree.
            purpose = _I_DO.sub(" ", purpose)
        return _dating_request(_TRAITS.sub(" ", purpose))
    return _dating_request(low)


def wants_text(genders, ages) -> str:
    """Who a dating listing would like to meet, in plain words: "a woman, 33 to 42", "anyone, 40 or older"."""
    genders = [g for g in GENDERS if g in (genders or ())]
    who = ("anyone" if set(genders) == set(GENDERS) else
           " or ".join("a non-binary person" if g == "nonbinary" else f"a {g}" for g in genders))
    span = age_span(ages)
    when = (f"{span[0]} or older" if span[1] >= AGE_MAX else f"{span[0]} to {span[1]}") if span else ""
    return ", ".join(p for p in (who, when) if p)


# --- Asking when it's unclear ----------------------------------------------------------------
# Keywords can't cover everything people type, so when the offline agent isn't sure, it asks instead
# of guessing, and never writes a kind it isn't sure of. It asks one short question when a message
# could mean more than one kind of match: wanting to meet a woman, and an activity they mention (a date,
# or a kayaking partner?); "a partner" with nothing to say whether it's romantic or for an activity; a
# partner for an activity that isn't listed ("a pottery partner": which activity?); or an activity only
# mentioned as something they enjoy, with nothing asking for it ("I love kayaking": a kayaking partner?).
# "Both" makes two listings, each with its own kind. Romantic words leave no doubt about wanting a date,
# so then an activity they only say they do or enjoy ("I love kayaking", "I play squash") is a hobby; one
# they want to do with someone ("a kayaking partner") still asks. How they describe the person they'd
# like to meet ("a girlfriend who skis") is never an activity to choose. The activities it names are listed
# ones (words_in_use).
_DATE_LABEL = "a date"
_PARTNER_ALONE = _re.compile(r"\b(?:a|an|my)\s+partner\b")
ASK_HINT = "Just say which, or \"both\"."
WHICH_ACTIVITY = "What would you like a partner for?"


def activity_hint(words=None) -> str:
    """Said when an answer names no listed activity: it asks again, and says why, with a few that are."""
    listed = list(_words(words).activities[:3])
    examples = f", e.g. {', '.join(listed[:-1])} or {listed[-1]}" if len(listed) > 1 else \
        f", e.g. {listed[0]}" if listed else ""
    return (f"Name the activity in a word or two{examples}. I only put activities other people already use on your "
            f"listing; anything else stays in your description, which is never shown.")


def _option(kind, activity) -> dict:
    label = _DATE_LABEL if kind == "dating" else f"{_article(activity)} {activity} partner"
    return {"kind": kind, "label": label, "activity": activity if kind == "other" else None}


def kind_question(message: str, words=None):
    """None when the kind of match is clear; otherwise {"text": one short question, "options": the
    kinds it could be, each {"kind", "label", "activity"}}. An option with kind None is a partner for
    an activity not named yet (or not a listed one). Sensitive words are read as if they weren't there."""
    words = _words(words)
    low = without_sensitive(message)
    purpose = without_hobbies(low)  # without what they only say they enjoy
    if _is_dating(low):
        romantic = bool(_ROMANTIC.search(_set_aside(low)))
        if romantic:  # only an activity they ask for counts, never one they do, enjoy or want in the person
            found = activities_said(_I_DO.sub(" ", _TRAITS.sub(" ", purpose)), first_person=False, words=words)
        else:
            found = activities_said(_TRAITS.sub(" ", low), words=words)
        if found:
            other = _option(*found[0])
        elif not romantic and _PARTNER_ALONE.search(low) and not _TASK_AFTER.search(low) and not _MEETING.search(
                _PARTNER_ALONE.sub(" ", low)):
            other = {"kind": None, "label": "a partner for an activity", "activity": None}  # "a partner", nothing more
        else:
            return None
        options = [{"kind": "dating", "label": _DATE_LABEL, "activity": None}, other]
        which = " (which one)" if other["kind"] is None else ""
        return {"text": f"Are you looking for {options[0]['label']}, or {other['label']}{which}? Or both? I can make "
                        f"two separate listings.", "options": options}
    if asked_kind(low) is None and activity_partner(purpose, words) is None and teaching(purpose, words) is None:
        # A partner for something not listed ("a pottery partner", or one named only by a sensitive word,
        # "a chemo buddy"): which activity, in listed words. Read on their own wording too, as the note is.
        if unknown_partner(purpose, words) or unknown_partner(without_hobbies(message.lower()), words):
            return {"text": f"{WHICH_ACTIVITY} {activity_hint(words)}",
                    "options": [{"kind": None, "label": "a partner for an activity", "activity": None}]}
        # Nothing outside their hobbies asks for a kind, but a hobby names one: a guess, so ask.
        found = activities_said(low, words=words)
        if found:
            option = {**_option(*found[0]), "open": True}  # "or something else?": any kind they name is an answer
            return {"text": f"Are you looking for {option['label']}? Or something else? Tell me in a few words what "
                            f"you'd like to meet someone for.", "options": [option]}
    return None


_BOTH = _re.compile(r"\b(?:both|either|all of (?:them|those)|two listings|each)\b")
_YES = _re.compile(r"^\W*(?:yes|yeah|yep|yup|sure|ok(?:ay)?|correct|right|exactly|definitely|of course|please|"
                   r"that['’]?s (?:right|it|correct))\b")
# Romantic words in an answer: a date, read as the date choice ("love", "a lover", "companionship", "to
# settle down", "someone to spend my life with").
_DATE_WORDS = (r"\b(?:dat(?:e|es|ing)|romance|romantic|relationship|boyfriend|girlfriend|someone special|lovers?|"
               r"companionship|settl(?:e|ing) down|spend(?:ing)? (?:my|our|the rest of my|the rest of our) li(?:fe|ves)|"
               r"share (?:my|our) li(?:fe|ves))\b")
_LOVE = r"\blove\b"  # a choice only when it's what the answer says ("love", "someone to love"), not "I'd love a..."
_NEGATED = _re.compile(r"(?:\b(?:not|no|never|without)|n't)(?:\s+[a-z']+){0,2}\s*$")  # within the same phrase
_CHOICE_FILLER = {"a", "an", "the", "i", "im", "m", "am", "d", "ll", "ve", "re", "t", "looking", "for", "want",
                  "just", "only", "please", "yes", "no", "not", "partner", "buddy", "someone", "to", "with", "and",
                  "or", "both", "one", "s", "it", "thanks", "thank", "you", "would", "like", "go", "do", "play",
                  "together", "activity", "in", "on", "at", "of", "some", "that", "this", "is", "be", "me", "my",
                  "really", "actually", "oh", "ok", "okay"}
# Said with a choice, not instead of one: "kayaking on weekends" still chooses kayaking.
_WHEN = {"weekends", "weekend", "evenings", "evening", "mornings", "morning", "weekdays", "nights", "afternoons"}
_WANT_WORDS = {"want", "like", "looking", "prefer", "choose", "pick", "rather", "need"}
# A clause of an answer: "a date, I just like kayaking" has two.
_CLAUSE = _re.compile(r"[,.;:!?]|\b(?:but|because|since|though|although|so|as)\b")
_PARTNER_AFTER = _re.compile(r"[a-z]*\s+(?:partners?|buddy|buddies|companions?|mates?|one|listing)\b")


def says_what_they_want(message: str, words=None) -> bool:
    """Does this message ask for a kind of match on its own (a new request, not an answer)? Only
    outside what it says they enjoy: "I just like kayaking" asks for nothing."""
    low = without_sensitive(message)
    purpose = without_hobbies(low)
    return asked_kind(low) is not None or activity_partner(purpose, words) is not None \
        or teaching(purpose, words) is not None


def _chose(low: str, pattern: str) -> bool:
    """The answer names this, at least once without "not" / "no" / "never" just before it."""
    return any(not _NEGATED.search(low[:m.start()]) for m in _re.finditer(pattern, low))


def _chose_dating(low: str) -> bool:
    """The answer asks for a date or a relationship, read the way matching reads romance ("a life
    partner", "looking for love", "a wife", "someone to marry"), or in other romantic words ("love", "a
    lover", "companionship", "to settle down"), and not negated ("not a date")."""
    text = _set_aside(low)
    return _chose(text, _ROMANTIC.pattern) or _chose(text, _DATE_WORDS) or _named_as_choice(text, _LOVE)


def _named_as_choice(low: str, pattern: str) -> bool:
    """Is this activity what the answer chooses: "a kayaking partner", "the kayaking one", or a clause
    that says nothing else ("kayaking", "just kayaking please", "kayaking on weekends")? One they only
    mention ("I kayak sometimes") isn't."""
    for m in _re.finditer(pattern, low):
        if not _NEGATED.search(low[:m.start()]) and _PARTNER_AFTER.match(low[m.end():]):
            return True
    for clause in _CLAUSE.split(_re.sub(_ROMANTIC.pattern, " ", low)):
        if not _chose(clause, pattern):
            continue
        # The activity, with a word that says which kind of it ("sea kayaking", "blitz draughts").
        rest = _re.findall(r"[a-z]+", _re.sub(rf"(?:(?!(?:i|we|im|you|not|no|never)\b)[a-z]+\s+)?(?:{pattern})[a-z]*",
                                              " ", clause))
        if any(w in ("i", "im", "we") for w in rest) and not any(w in _WANT_WORDS for w in rest):
            continue  # "I kayak on weekends" is about them, not a choice ("I'd like kayaking" is)
        if all(w in _CHOICE_FILLER or w in _WHEN for w in rest):
            return True
    return False


def chosen_kinds(answer: str, options: list, words=None):
    """What the person chose, in answer to kind_question: a list of {"kind", "activity"} (in the
    order asked), or None if the answer doesn't say. "Both" (or "yes" to a question with one option) is
    every option; a negated choice ("not a date") doesn't count, and nor does an activity they only
    mention ("a date, I just like kayaking" is a date) or one that describes the person they'd like to
    meet ("a girlfriend who skis" is a date). A partner for an activity not named yet takes the
    activity from the answer, only if it's a listed one ("squash"; "pottery" isn't an answer unless it's
    listed, so it asks again); with "both" and no activity yet, its kind stays None, to be asked. Romantic
    words ("love", "a lover", "companionship") choose a date, even when only an activity was asked about."""
    words = _words(words)
    low = without_hobbies(without_sensitive(answer))  # what they say they enjoy is never a choice
    choice = _TRAITS.sub(" ", low)  # nor how they describe the person they'd like to meet
    which_only = len(options) == 1 and options[0]["kind"] is None  # "what would you like a partner for?"
    both = not which_only and (bool(_BOTH.search(low)) or (len(options) == 1 and bool(_YES.match(low))))
    dating = both or _chose_dating(low)
    out = []
    for o in options:
        if o["kind"] == "dating":
            if dating:
                out.append({"kind": "dating", "activity": None})
        elif o["kind"] is None:  # which activity: a listed one they name now
            named = [(k, a) for k, a in activities_said(choice, anywhere=True, words=words)
                     if both or _named_as_choice(choice, _re.escape(a))]
            if named:
                out.append({"kind": named[0][0], "activity": named[0][1]})
            elif both:
                out.append({"kind": None, "activity": None})
        elif both or _named_as_choice(choice, _option_pattern(o)):
            out.append({"kind": o["kind"], "activity": o.get("activity")})
    if not out and dating:  # romance answers a question that didn't offer a date too
        out.append({"kind": "dating", "activity": None})
    if not out and any(o.get("open") for o in options):  # "or something else?": the kind the answer asks for
        kind, activity = asked_kind(low), activity_partner(choice, words)
        if kind or activity:
            out.append({"kind": kind or "other", "activity": None if kind else activity})
    return out or None


def _option_pattern(option: dict) -> str:
    """How an activity option is named in an answer: its label's own words."""
    words = [w for w in _re.findall(r"[a-z]+", option["label"]) if w not in ("a", "an", "partner", "buddy")]
    return r"\b(?:" + "|".join(map(_re.escape, words)) + r")"


def answer_details(answer: str, options: list, chosen: list) -> str:
    """What an answer to kind_question says besides the choice, to add to the first message so nothing
    they say is lost: "a date, I'm a man, 30 to 40" -> "I'm a man, 30 to 40". The parts that only make
    the choice ("a date", "both please", "just kayaking, not a date") add nothing: the chosen kind goes
    on the listing on its own. This goes only into the description, which is never shown."""
    names = [_ROMANTIC.pattern, _DATE_WORDS, _LOVE, _BOTH.pattern, _YES.pattern]
    names += [_option_pattern(o) for o in options if o["kind"] not in (None, "dating")]
    names += [_re.escape(c["activity"]) for c in chosen if c.get("activity")]
    keep = []
    for part in _re.split(r"[,.;:!?]+", answer or ""):
        rest = part.lower()
        for rx in names:
            rest = _re.sub(rf"(?:{rx})[a-z]*", " ", rest)
        if any(w not in _CHOICE_FILLER for w in _re.findall(r"[a-z0-9]+", rest)):
            keep.append(" ".join(part.split()))
    return ", ".join(keep) + "." if keep else ""


# How someone says something about themselves, each form ending where what they say starts. A value said after one
# of the first forms is a predicate of the person: "I'm (a/an)...", "I am...", "we're...". After the others it is
# what they like or do: "I (really/also...) like / enjoy / love / play / go / do...", "I'm (a retired nurse) into...",
# "I'm a (big) fan of...", "my hobbies are...".
_INTO_ADVERB = r"(?:(?:really|also|just|very|so|totally|quite)\s+)"
_MINE = _re.compile(
    r"\b(?:(?P<likes>(?:i'?m|i am|we'?re|we are)\s+" + _INTO_ADVERB + r"?(?:(?:(?:a|an)\s+(?:[a-z'-]+\s+){1,2})?" + _INTO_ADVERB
    + r"?into|(?:a\s+)?(?:big\s+|huge\s+)?fans?\s+of)|my hobbies (?:are|include)|"
    r"i\s+(?:[a-z]+ly\s+|also\s+|just\s+|often\s+)?(?:enjoy|love|like|adore|play|go|do|practi[cs]e)s?)"
    r"|(?P<is>i'?m|i am|we'?re|we are))\b")
# Between that and the value, nothing that starts another clause (_CLAUSE), asks for something (_REQUEST), or
# turns to someone or something else: another person (_OTHERS, "someone", "a partner") or a preposition that
# points away from them ("I'm fine with...", "I'm open to...", "more X than Y").
_ELSEWHERE = _re.compile(r"\b(?:someone|somebody|anyone|anybody|people|person|partners?|buddy|buddies|companions?|"
                         r"with|for|to|about|than|after|towards?)\b")
_MINE_GAP = 6  # words at most between the form and the value: "I'm usually only free..." has three
# Between "I'm" (or "I am", "we're") and a value that is a predicate of the person, only these may stand: an
# article, and adverbs or degree words that keep what is said true ("I'm usually a...", "I'm a total...", "I'm
# kind of a..."). Closed-class words, never a verb: "I'm leading a weekend group" makes "weekend" the object of
# what they do, so it is nothing about them. The words of the detail's own name may stand there too, as in "my
# {name} is" ("I'm free on weekends" says the "free time" detail by its name), each followed by at most a place
# word (in, on, at), but never an -ing or -ed form of a verb.
_PREDICATE = {"a", "an", "the", "very", "really", "quite", "pretty", "fairly", "rather", "so", "totally", "completely",
              "absolutely", "entirely", "fully", "truly", "definitely", "certainly", "mostly", "mainly", "largely",
              "generally", "usually", "normally", "typically", "always", "often", "sometimes", "also", "just", "only",
              "still", "now", "currently", "already", "actually", "basically", "honestly", "simply", "more", "most",
              "bit", "little", "kind", "sort", "of", "real", "total", "complete", "proper", "true", "pure"}
_NAME_THEN = {"in", "on", "at"}
_VERB_FORM = _re.compile(r"[a-z]{2,}(?:ing|ed)")
# The negation family, read as sentence structure: negation words and negative adverbs or pronouns ("not",
# "never", "neither ... nor", "hardly", "barely", "no longer", "not ... yet"), phrases that deny or leave out
# what follows ("far from", "anything but", "except", "other than", "instead of"), and the verbs of denial
# ("doubt", "deny"). Any of them between where the person's clause starts and the value means the value isn't
# theirs, also when the value is in a clause that depends on the denied verb ("I don't think I'm...").
_NEGATION = (r"n't\b|\b(?:not|no|never|neither|nor|none|nowhere|without|hardly|barely|scarcely|rarely|seldom)\b(?!-)"
             r"|\bfar from\b|\banything but\b|\b(?:except|excluding)\b|\b(?:apart|aside) from\b|\bother than\b"
             r"|\binstead of\b|\b(?:doubt|doubts|doubted|deny|denies|denied)\b")
# A form that says something about the person starts a new clause of its own only at the start of a phrase or
# after a coordinating conjunction ("..., and I'm", "but I'm"); after a verb ("think I'm", "say I'm", "sure I'm")
# it depends on that verb, and a denial of the verb reaches it.
_NEW_CLAUSE = _re.compile(r"\b(?:and|or|but|yet|so|then|also|plus)\s+$")
# Where a phrase ends for a denial: sentence punctuation, a conjunction that starts another clause, or "so" / "as"
# before a new subject ("so I'm", "as we're"), never "so far as to say", where they are degree words.
_PHRASE_END = _re.compile(r"[,.;:!?]|\b(?:but|because|since|though|although)\b"
                          r"|\b(?:so|as)\s+(?=(?:i|i'?m|we|we'?re|you|he|she|it|they)\b)")
_DENIAL = _re.compile(_NEGATION)
_ANYTHING_BUT = _re.compile(r"\banything but\b")  # "but" there is part of the denial, not a new clause
# An -ing word followed by more than a list's "and" / "or": what comes after it is that verb's object ("I like
# coaching beginner climbers"), not what they like.
_ING_OBJECT = _re.compile(r"\b[a-z]{2,}ing\b(?!\s+(?:(?:and|or)\b|&))")


# A value that describes someone else's: directly followed by a possessive ("I'm a green crew's instructor": the
# crew is green, not them), the possessive on the value itself or on the word after it; never a contraction of "is",
# "has" or "us" there ("it's", "that's", "let's", "he's": schema's _CONTRACTED), which starts what comes next.
_OWNED_AFTER = _re.compile(r"(?:['’]s|s['’])(?![\w'’])|\s+(?!" + _CONTRACTED + r")[\w-]+(?:['’]s|s['’])(?![\w'’])")
# What may not follow a value that continues a list of the same detail's values: a word that starts another clause
# about it, a verb of being, having or seeming, a modal ("..., a pastry chef would suit me"; "..., fridays are out"),
# the negation family ("..., fridays never work") or a request ("..., a baker is what I'm looking for"). Closed-class
# words: then the value is what that clause is about, not one more of theirs.
_LIST_STOPS = _re.compile(r"\s*(?:(?:is|are|was|were|be|been|being|am|would|will|could|should|can|may|might|must|"
                          r"shall|do|does|did|has|have|had|seems?|sounds?|looks?|feels?|appears?|becomes?)\b|['’](?:s|re|d|ll)\b|"
                          + _NEGATION + ")")


def _denied(before: str) -> bool:
    """Is a value, whose text before it is `before`, denied: a word of the negation family after the start of its
    phrase and of the last form that says something about the person and starts a clause of its own (_NEW_CLAUSE:
    "... and I'm", never "think I'm", which a "don't" before it still reaches), or one at most two words before it."""
    scan = _ANYTHING_BUT.sub(lambda m: "anythingxbut", before)
    phrase = max([m.end() for m in _PHRASE_END.finditer(scan)] + [0])
    start = max([m.start() for m in _MINE.finditer(scan, phrase)  # only the few words before each form are read
                 if not scan[phrase:m.start()].strip() or _NEW_CLAUSE.search(scan, max(phrase, m.start() - 8), m.start())]
                + [phrase])
    return bool(_DENIAL.search(before[start:])) or bool(_NEGATED.search(before))


def _says_mine(before: str, name: str, earlier=(), after: str = "") -> bool:
    """Is a value, whose text before it is `before` (and after it `after`), said of the person: right after "my
    {name} is", as a predicate of the person (only _PREDICATE words, or the name's own, between "I'm" and the
    value), or as what they like or do (nothing in between that starts another clause, asks for something or turns
    to someone else, and never the object of an -ing verb)? earlier: values of the same detail already said there; a
    list of them, joined by commas, "and" or "or", may stand in between too ("I'm free weekends and weekday
    evenings", "I'm a trimmer and a bowman"), unless a clause starts with this value (_LIST_STOPS, _REQUEST after
    it): a value after a comma that another clause is about ("I'm a baker, a pastry chef would suit me", "I'm free
    mondays, fridays never work") isn't theirs."""
    if _re.search(r"\bmy\s+" + _re.escape(_lower(name)) + r"\s+(?:is|are)\s+(?:a\s+|an\s+)?$", before):
        return True
    opener = None
    for opener in _MINE.finditer(before):
        pass
    if opener is None:
        return False
    gap = before[opener.end():]
    if earlier:  # the values listed before this one, each with the comma, "and" or "or" after it
        listed = "|".join(_re.escape(_lower(v)) for v in sorted(earlier, key=len, reverse=True))
        joined = _re.sub(rf"\b(?:{listed})\b\s*(?:,\s*)?(?:(?:and|or|&)\s+)?", " ", gap)
        if joined != gap and (_LIST_STOPS.match(after) or _REQUEST.match(after.lstrip())):
            return False  # it continues no list: it starts another clause
        gap = joined
    if len(gap.split()) > _MINE_GAP or any(rx.search(gap) for rx in (_CLAUSE, _REQUEST, _OTHERS, _ELSEWHERE)):
        return False
    if opener.group("likes"):
        return not _ING_OBJECT.search(gap)
    own = {w for w in _re.findall(r"[a-z]+", _lower(name)) if not _VERB_FORM.fullmatch(w)}
    words_ = gap.split()
    return all(w in _PREDICATE or w in own or (w in _NAME_THEN and i and words_[i - 1] in own)
               for i, w in enumerate(words_))


def listed_details(text: str, words=None, kind=None) -> list:
    """The open details a message says in listed words, as draft rows (detail_row): each listed value (or step
    of a listed scale) under the one name it's listed under, about the person, or, on a listing with a kind,
    about the request when that's how it's listed ("both").

    A value about the person counts only where they say it of themselves (_says_mine): as a predicate of the
    person ("I'm a beginner", "I'm usually a...", "I'm only free weekday mornings"), as what they like or do ("I
    enjoy croquet", "I'm into..."), or after "my level is...". Nothing else may stand between "I'm" and the value
    (a verb makes the value its object: "I'm leading a weekend group" says nothing about them). None counts where
    it describes someone else ("a partner who...", "with a..."; _TRAITS; or a value followed by a possessive, "a
    tidy crew's ..."; _OWNED_AFTER) or where the negation family reaches it in the same phrase ("not", "hardly",
    "far from", "neither ... nor", "doubt"; _denied), a clause that depends on a denied verb included ("I don't
    think I'm..."), so a wish ("looking for an advanced partner") or a denial ("I'm far from tidy") is never
    written as a fact. A list of one detail's values after "I'm" is read whole ("I'm free mondays and fridays"). On a scale, one step
    only: two steps said is not sure which, so neither. The offline agent never writes a wish about the other
    person; the person checks every value on their profile before anything is shown."""
    words = _words(words)
    if words._value_rx is None:
        return []
    low = _lower(text)
    traits = [m.span() for m in _TRAITS.finditer(low)]
    found: dict = {}  # (about, name) -> [values, in the order said]
    for m in words._value_rx.finditer(low):
        about, name, how, steps = words.values[m.group(1)]
        before = low[:m.start()]
        if about == "both" and kind is None:
            continue
        if any(a <= m.start() < b for a, b in traits) or _denied(before) or _OWNED_AFTER.match(low[m.end():]):
            continue  # how they describe someone else (a value of someone's: "a green crew's"), or a denial
        if about == "me" and not _says_mine(before, name, found.get((about, name, how, steps), []), low[m.end():]):
            continue
        said_ = found.setdefault((about, name, how, steps), [])
        value = next((s for s in steps if canonical(s) == canonical(m.group(1))), m.group(1))
        if canonical(value) not in map(canonical, said_):
            said_.append(value)
    rows = []
    for (about, name, how, steps), values in found.items():
        if how == "steps" and len(values) != 1:
            continue  # two steps said: not sure which
        rows.append(detail_row(about, name, how, steps, values[:MAX_VALUES[about]]))
    return rows


def _request(low: str, words: Words, kind=None, activity=None) -> tuple:
    """What a message asks for, as the offline agent reads it: (kind, activity, taught, guessed). kind and activity:
    what the person chose when asked (kind_question), else what their words ask for; taught: (seeking, offering,
    activity) when they teach a listed activity or want to learn one; guessed: a kind read only from what they say
    they enjoy, to ask about. A kind it isn't sure of is None ("other" needs a listed activity)."""
    asked = asked_kind(low) if kind is None else None  # "dating", or None
    activity = known_activity(activity, words) if activity else None
    taught = None
    if activity is None and kind in (None, "other") and asked is None:
        activity = activity_partner(without_hobbies(low), words)
        taught = None if activity else teaching(without_hobbies(low), words)
    guessed = None  # a kind only from their hobbies: ask
    if kind is None and asked is None and activity is None and taught is None:
        guessed = next(iter(activities_said(low, words=words)), None)
    kind = kind or asked or ("other" if activity or taught else None)
    if kind == "other" and not (activity or taught):
        kind = None  # a partner for an activity that isn't listed: ask
    return kind, activity, taught, guessed


def _phrases(activity, taught) -> tuple:
    """(seeking, offering, topic) for a listing that isn't dating: a partner for the activity, or the teaching pair."""
    if activity:
        phrase = f"{_article(activity)} {activity} partner"
        return phrase, phrase, activity
    return tuple(taught) if taught else (None, None, None)


def _seeking_of(draft) -> str:
    return canonical(next((f.get("value") for f in (draft or {}).get("fields", []) or []
                           if isinstance(f, dict) and f.get("key") == "seeking" and isinstance(f.get("value"), str)),
                          "") or "")


def other_request(new, old) -> bool:
    """Is `new` a different request from `old` (two drafts)? Only when both say what they're for and it isn't the
    same: another kind of match, or, on a listing that isn't dating, another "looking for" phrase. One rule for what
    the offline agent keeps from before (draft_from_message, keep_details)."""
    a, b = (new or {}).get("kind"), (old or {}).get("kind")
    phrase, before = _seeking_of(new), _seeking_of(old)
    return bool(a and b and (a != b or (phrase and before and phrase != before)))


def _as_draft(kind, activity, taught) -> dict:
    """A message's own request in a draft's shape, for other_request."""
    seeking = _phrases(activity, taught)[0] if kind == "other" else None
    return {"kind": kind, "fields": [{"key": "seeking", "value": seeking}] if seeking else []}


# Where a clause starts, for reading whose request it is: where a phrase ends (_PHRASE_END: sentence punctuation, a
# conjunction that starts another clause), or a coordinating conjunction before a new subject ("... and I'd like",
# "or my cousin wants"). First-person words: where the person speaks of themselves.
_SUBJECT = r"(?:i|we|you|he|she|they|my|our|his|her|their)\b"
_CLAUSE_START = _re.compile(_PHRASE_END.pattern + r"|\b(?:and|or|so|then|plus|yet)\s+(?=" + _SUBJECT + ")")
_SELF = _re.compile(r"\b(?:i|we|me|us)\b")
# A verb that asks for someone or something, in any person ("I'm looking for", "she wants", "they'd like"): how a
# request is said, whoever says it.
_WISH = _re.compile(r"\b(?:look(?:s|ing)?\s+(?:for|to)|seek(?:s|ing)?|search(?:es|ing)?\s+for|need(?:s|ed)?|"
                    r"want(?:s|ed)?|would\s+like|hop(?:e|es|ing)\s+to|to\s+(?:meet|find))\b|['’]d\s+like\b")


def _asks(text: str, words) -> bool:
    """Does this text ask for something: a verb that asks (_WISH, so also who and what ages: "she wants someone 30 to
    40"), or a kind of match read from its words (a date, a partner for an activity, listed or not, or teaching)?"""
    return bool(_WISH.search(text.lower())) or says_what_they_want(text, words) or any(
        unknown_partner(without_hobbies(t), words) for t in (without_sensitive(text), text.lower()))


def _not_theirs(clause: str, words) -> bool:
    """Is what a clause asks for one the person denies, or someone else's request? Read in the clause before where it
    is first asked for (the shortest start of the clause that asks for it): a word of the negation family that
    reaches it (_denied: "I'd never want...", "not a..."), or someone else (_OTHERS: "my cousin is after...", "she
    wants...") after the last place the person speaks of themselves ("my friends say I need..." is theirs). What
    comes after it ("someone who doesn't smoke") is how they describe the request, not a denial."""
    for t in _re.finditer(r"\S+", clause):
        if _asks(clause[:t.end()], words):
            before = clause[:t.start()].lower()
            break
    else:
        return False
    if _denied(before):
        return True
    own = [m.end() for m in _SELF.finditer(before)]
    return bool(_OTHERS.search(before[own[-1]:] if own else before))


_NEW_SENTENCE = _re.compile(r"\s*[.!?;]")
_OWN_SUBJECT = _re.compile(r"[\s,;:.!?]*(?:(?:and|or|but|so|then|plus|yet|because|since|though|although|as)\s+)?"
                           + _SUBJECT)


def own_requests(text: str, words=None) -> str:
    """The message as a request is read from it when it follows an earlier request: each clause that asks for
    something the person denies, or that is someone else's (_not_theirs), blanked out, with the clauses after it in
    the same sentence that go on with it (asking nothing of their own and starting with no subject of their own: "her
    husband wants someone, 30 to 40"), so only what they ask for themselves can start a different request or say who
    and what ages they'd like ("I'd never want that", "my aunt wants one" add to the request already
    there, or change nothing). Everything else is as written, in place."""
    words = _words(words)
    text = str(text or "")
    low = text.lower()
    if len(low) != len(text):
        text = low  # the clauses are found in lower case, so the blanks fall in the same places
    cuts = [0, *sorted({m.start() for m in _CLAUSE_START.finditer(low)} - {0}), len(low)]
    out, blank = [], False
    for a, b in zip(cuts, cuts[1:]):
        clause = text[a:b]
        if _NEW_SENTENCE.match(clause):
            blank = False
        if clause.strip(" ,;:.!?") and _asks(clause, words):
            blank = _not_theirs(clause, words)
        elif _OWN_SUBJECT.match(clause.lower()):
            blank = False
        lead = len(clause) - len(clause.lstrip(" ,;:.!?"))
        out.append(clause[:lead] + " " * (len(clause) - lead) if blank else clause)
    return "".join(out)


def starts_another_request(message: str, earlier, words=None) -> bool:
    """Does this message ask for something of its own that isn't the earlier draft's request? A kind of match, a
    partner for a listed activity, or teaching, that differs from it (other_request), or a partner for something
    that isn't listed (which the offline agent asks about), each only where the person asks for it themselves
    (own_requests). Anything else adds to the earlier request: where they are, a free time, something they enjoy,
    an answer to what it asked, the same request said again, or a request they deny or that is someone else's."""
    words = _words(words)
    low = own_requests(" ".join(str(message or "").split())[:500], words).lower()
    kind, activity, taught, _ = _request(low, words)
    if kind is None:  # read as kind_question reads it
        return any(unknown_partner(without_hobbies(t), words) for t in (without_sensitive(low), low))
    return other_request(_as_draft(kind, activity, taught), earlier)


# The last word of a text when a full stop after it doesn't end a sentence, as the Warden's full-name rule reads one
# (warden.full_name_pairs): an initial or a short capitalised word without vowels, a title ("J", "Dr", "St").
_NO_STOP = _re.compile(r"(?<![\w'’])(?:[A-Z]|[A-Z][b-df-hj-np-tv-xz]{1,2})\.?$")


def _ended(text: str) -> str:
    """The text ending a sentence, so a message after it starts one: as it is after "!" or "?", or a full stop that
    ends one; else with a full stop; or, after a word a full stop doesn't end (_NO_STOP: "J. Okafor" reads as one
    name), a semicolon. Two messages put together then never make words of one look like they run on into the
    other's, a name of two capitalised words included ("...speaks French" and "Based in Lisbon")."""
    text = text.rstrip(" ,;:")
    if _NO_STOP.search(text):
        return f"{text};"
    return text if text.endswith(("!", "?", ".")) else f"{text}."


def _description(before, said: str) -> str:
    """The earlier description with a new message after it, within the 500 characters a description may have: the
    new message whole, and as much of the earlier one as fits, cut at a word, with the earlier one ending a sentence
    (_ended). One already in the other isn't repeated (the instant draft's description is the message itself, until
    the agent writes one, and a request said again with more in it replaces the words it repeats)."""
    before = " ".join(str(before or "").split())
    if not before or said in before:
        return before or said
    if before.rstrip(".!?;:, ") in said:
        return said
    room = 500 - len(said) - 2  # the earlier text's own full stop or semicolon, and a space
    if room <= 0:
        return said[:500]
    return f"{_ended(before if len(before) <= room else before[:room].rsplit(' ', 1)[0])} {said}"


def _value(f: dict):
    value = f.get("value")
    return tuple(value) if isinstance(value, list) else value


def _kept_from(old: dict, v: dict, kind, same: bool, here=(None, None)) -> dict:
    """The built-ins on the earlier draft that stay on the new one, by key (their rows as they were, so the person's
    choice of shown or hidden and a sensitive detail's confirmation stay too). Adding to the same request (same):
    everything the message doesn't say, sensitive details included, and the request's own phrases (looking for,
    what they are to them, its headline and topics) whatever the message restates, so the person's own wording stays.
    A different request: only what's about the person and isn't sensitive (never the old request's own details),
    as carry_over keeps for a model agent. Where they are is one place: once the message says they are somewhere
    else (here: where_they_are, a city or a country), none of the earlier place comes over (its city, country or
    neighbourhood). Nothing that kind can't have (carries, and a format only where it has one: has_format), as
    fill_gaps keeps it."""
    rows = {f["key"]: f for f in old.get("fields", []) or [] if isinstance(f, dict) and isinstance(f.get("key"), str)
            and f["key"] in FIELD_REGISTRY and f.get("enabled", True) and f.get("value") not in (None, "", [])}
    said = {"city": v.get("city") or here[0], "country": v.get("country") or here[1]}
    moved = any(said[k] and k in rows and canonical(str(_value(rows[k]))) != canonical(str(said[k])) for k in said)
    phrased = same and bool(_seeking_of(old))
    out = {}
    for key, f in rows.items():
        if key in ("kind", "first_name", "age_band") or not carries(kind, key) \
                or (key == "format" and not has_format(kind)):
            continue
        if moved and key in ("city", "country", "neighborhood"):
            continue
        if not same and (key in _KIND_SPECIFIC or f.get("sensitive") or FIELD_REGISTRY[key].sensitive):
            continue
        if v.get(key) not in (None, "", (), []) and not (phrased and key in _PHRASES):
            continue  # the message says it: the latest word wins
        out[key] = f
    return out


def _with_earlier(rows: list, old: dict, same: bool) -> None:
    """An open detail the message adds values to keeps those it had, before the ones it says (the oldest left out
    past the most it may have), and its shown or hidden, where both mean the same (the same compare and steps); a step
    on a scale is one place, so the new one replaces it. On a different request, only one about the person."""
    before = {f.get("key"): f for f in old.get("fields", []) or []
              if isinstance(f, dict) and is_detail_key(f.get("key")) and f.get("enabled", True)
              and isinstance(f.get("value"), list) and f.get("value")}
    for row in rows:
        f = before.get(row["key"])
        if f is None or row["compare"] == "steps" or (not same and row["about"] != "me") \
                or f.get("compare") != row["compare"] or list(f.get("steps") or []) != row["steps"]:
            continue
        said_now = {canonical(x) for x in row["value"]}
        values = [str(x) for x in f["value"] if canonical(str(x)) not in said_now] + list(row["value"])
        row["value"] = values[-MAX_VALUES[row["about"]]:]
        if f.get("visibility") == HIDDEN and row["about"] == "me":
            row["visibility"] = HIDDEN  # as they chose it (only one about them can be hidden)


def draft_from_message(message: str, first_name: str, age_band: str, kind=None, activity=None,
                       words=None, earlier=None) -> dict:
    """Read the person's own words and suggest a listing. Returns a plain dict the
    app shows for review; nothing is submitted until the person says so.

    kind: the kind of match the person chose when asked (kind_question), instead of the one their
    words suggest. activity: for "other", the activity they want a partner for; only a listed one is
    used. words: the shared list of details in use (words_in_use), the only words it writes.

    earlier: the draft this message follows in the same chat (the profile as it stands, the person's own changes
    included), so a request given in pieces comes together. A message that asks for a different request
    (other_request), itself (own_requests: never one they deny or someone else's), starts a new draft that keeps
    only what's about the person from it (_kept_from). Any other message adds to it: it is read as the earlier
    request (its kind), what it says goes on the profile, everything it doesn't say stays as it was (how exact
    too, unless it says what it's for), a value it adds to an open detail joins those there (_with_earlier), and the
    description is the earlier one with the message after it (_description). Either way the notes ask for what the
    draft as a whole still needs, and the message is read by the same rules as any other.

    It never records a kind it isn't sure of: one read only from what they say they enjoy ("I love
    kayaking", with nothing asking for it) is asked about instead. Every shown detail it writes comes from
    listed words and the fixed choices: nothing else from the message is copied into anything shown. A
    partner for an activity that isn't listed is asked about. Sensitive words become a hidden detail
    under their built-in category, as said."""
    words = _words(words)
    text = " ".join(message.split())[:500]
    low = text.lower()
    old = earlier if isinstance(earlier, dict) and isinstance(earlier.get("fields"), list) else None
    # What it asks for: after an earlier request, only what the person asks for themselves (own_requests), so a
    # request they deny or someone else's never replaces theirs, nor changes who or what ages they'd like to meet.
    asking = own_requests(text, words) if old is not None and kind is None and old.get("kind") in MATCH_KINDS \
        else text
    parsed = _parse_intent(asking)
    kind, activity, taught, guessed = _request(asking.lower(), words, kind, activity)
    states = kind is not None  # the message says what it's for itself (or the person chose it when asked)
    same = old is not None and not other_request(_as_draft(kind, activity, taught), old)
    if same and kind is None and old.get("kind") in MATCH_KINDS:
        kind = old["kind"]  # it adds to the earlier request: read as that kind of listing
    if kind != parsed.kind:
        parsed = read_listing(asking, {"kind": kind}.get)  # read as that kind of listing
    v = {"first_name": first_name, "age_band": age_band}
    if kind:
        v["kind"] = kind  # what the agent understood, recorded on the listing (shown)
    if kind == "other" and (activity or taught):
        seeking, offering, topic = _phrases(activity, taught)
        v["seeking"], v["offering"] = seeking, offering
        v["headline"] = f"Looking for {seeking}"[:60]
        v["topics"] = (topic,)
    enabled_sensitive = set()
    notes = []

    here = where_they_are(low, words)  # where they say they are now: a known city (and its country), or a country
    city, country = here if here[0] else (None, None)  # the draft records a place only with its city (known_city)
    if city:
        v["city"] = city
        if country:
            v["country"] = country
    ages = note = None
    if kind == "dating":
        if gender := own_gender(low):
            v["gender"] = gender
        # Who they'd like to meet, and the ages, as the agent understood them (shown, so matching uses
        # them and the person can check them on their profile).
        if parsed.seeking:
            v["wants_gender"] = tuple(g for g in GENDERS if g in parsed.seeking)
        ages, note = wanted_ages(asking, age_band)
        if ages:
            v["wants_age"] = ages
    age_note = len(notes)  # where a note about the ages goes, once it's known what ages the draft has
    fmt = format_from_text(low)
    if fmt and kind != "dating":  # dating is always in person
        v["format"] = fmt
    travel = travel_from_text(low)
    if travel:
        v["travel"] = travel
    details = listed_details(text, words, kind)

    # Sensitive things typed into the free text: move them to a hidden detail and warn.
    found = sensitive_details(text, kind)
    for key, value in found.items():
        v[key] = value
        enabled_sensitive.add(key)
        notes.append(f"You mentioned something sensitive ({LABELS[key].lower()}). I've added it as a "
                     f"hidden detail: it can help a match but is never shown. Your description is never "
                     f"shown either, but you may prefer to take it out of there.")
    if not found and mentions_sensitive(low):
        notes.append("You mentioned something sensitive. I've kept it out of everything shown on your listing. Your "
                     "description is never shown either, but you may prefer to take it out of there.")
    # What stays from the earlier draft: what this message doesn't say (or, for a different request, what's about
    # the person), each row as it was.
    kept = _kept_from(old, v, kind, same, here) if old is not None else {}
    for key, f in kept.items():
        v[key] = _value(f)
        if f.get("sensitive") or FIELD_REGISTRY[key].sensitive:
            enabled_sensitive.add(key)
    if old is not None:
        _with_earlier(details, old, same)
    if note and (ages or "wants_age" not in v):
        # About the ages this message set, or that none are set: never said when the ages it couldn't place leave
        # those already on the profile as they were.
        notes.insert(age_note, note)
    # An extra safeguard: nothing shown is ever built from a sensitive word (none is listed anyway). What stays from
    # before is as it was.
    for key in [k for k in ("seeking", "offering", "headline", "topics") if k not in kept]:
        value = v.get(key)
        if isinstance(value, (list, tuple)):
            value = [x for x in value if not mentions_sensitive(x)]
        if not value or (isinstance(value, str) and mentions_sensitive(value)):
            v.pop(key, None)
        else:
            v[key] = value
    details = [d for d in details
               if not any(mentions_sensitive(x, in_detail=True) for x in [d["name"], *d["value"], *d["steps"]])]

    # How exact: read from a message that says what it's for (states), as the request is said or said again; a message
    # that only adds to it ("I'm only free...") leaves it as the profile has it, the person's own setting included.
    exactness, own_low = 0.6, asking.lower()
    before = old.get("exactness") if same else None
    before = before if isinstance(before, (int, float)) and not isinstance(before, bool) and 0 <= before <= 1 else None
    if before is not None and not states:
        exactness = before
    elif _re.search(r"\b(exactly|must|only|strictly)\b", own_low):
        exactness = 0.9
    elif _re.search(r"\b(open to|flexible|anyone|whatever|wide)\b", own_low):
        exactness = 0.3
    elif before is not None:
        exactness = before
    city = v.get("city")
    # Meeting online, said now or already on the profile: then no place is needed.
    remote_ok = parsed.remote_ok or _value(kept.get("format") or {}) in ("Online", "In person or online")

    if guessed and kind is None:
        notes.insert(0, f"Are you looking for {_option(*guessed)['label']}, or something else? I haven't set "
                        f"the kind of match yet: tell me in a few words what you'd like to meet someone for.")
    elif kind is None and unknown_partner(without_hobbies(low), words):
        notes.insert(0, f"{WHICH_ACTIVITY} {activity_hint(words)}")
    elif kind is None:
        notes.insert(0, "I couldn't tell what kind of match you want. Could you say a bit more about what you'd "
                        "like to meet someone for?")
    elif kind == "other" and not (v.get("seeking") and v.get("offering")):  # kept from before, with nothing it's for
        notes.insert(0, f"{WHICH_ACTIVITY} {activity_hint(words)}")
    if not city and not remote_ok:
        notes.append("Which city? I couldn't tell, so please tell me where you are.")
    if kind == "dating" and "wants_gender" not in v:
        notes.append("Who would you like to meet: a woman, a man, or anyone?")
    if kind == "dating" and "wants_age" not in v and not note:
        notes.append(f"You didn't say what ages you'd like, so for now you're open to "
                     f"{ages_text(default_age_range(age_band))} (one age band either side of yours). Tell me "
                     f"if you'd like different ages.")
    if kind == "dating" and "gender" not in v:
        notes.append(GENDER_NOTE)

    keys = (["kind"] if kind else []) + _COMMON + _BY_KIND.get(kind, [])
    keys += [k for k in ("headline", "topics") if k in v]
    keys += [k for k in kept if k not in keys]  # nothing that stays is dropped
    options = field_options()
    fields = []
    for key in keys:
        spec = FIELD_REGISTRY[key]
        value = v.get(key)
        if key in kept:
            fields.append({**kept[key], "value": list(value) if isinstance(value, (list, tuple)) else value,
                           "options": list(kept[key].get("options") or options.get(key, []))})
            continue
        fields.append({
            "key": key, "label": LABELS[key], "value": list(value) if isinstance(value, (list, tuple)) else value,
            "visibility": spec.default, "sensitive": spec.sensitive, "multi": spec.multi,
            "enabled": (key in enabled_sensitive) if spec.sensitive else True,
            "locked": key == "age_band", "options": options.get(key, []),
        })
    fields += details

    understood = [f"Looking for {_KIND_NAMES[kind]}" if kind else "Kind of match: not sure yet"]
    if kind == "other" and v.get("seeking"):
        understood = [f"Looking for {v['seeking']}"]
    if city:
        understood.append(f"in {city}")
    if kind == "dating" and (v.get("wants_gender") or v.get("wants_age")):
        understood.append("with " + wants_text(v.get("wants_gender"), v.get("wants_age")))
    if v.get("format") == "Online":
        understood.append("(online only)")
    elif remote_ok:
        understood.append("(online/remote is fine)")
    if v.get("travel") in TRAVEL and TRAVEL[v["travel"]]:
        understood.append(f"(will travel {v['travel'].lower()})")
    # "listed": the details this agent chose shown or hidden for on purpose. It chooses none: every visibility
    # here is a default, so a choice the person already made stays (keep_visibility).
    return {"intent_text": _description(old.get("intent_text"), text) if same else text, "exactness": exactness,
            "kind": kind, "fields": fields, "understood": " ".join(understood), "notes": notes, "listed": []}


# --------------------------------------------------------------------------------------
# Instant profile updates (used by the demo while the agent is still replying).
#
# Picks out only facts stated plainly enough to be sure of ("I live in Osaka", "I'm a woman",
# "within 30 km"), and only in words it recognises: a place it knows, a fixed choice, a listed
# activity they teach or want to learn. Anything else is left for the agent (and the person).
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


def _country(name: str):
    low = name.strip().lower()
    if low in _COUNTRY_ALIASES:
        return _COUNTRY_ALIASES[low]
    return next((c for c in COUNTRIES if c.lower() == low), None)


def _title(place: str) -> str:
    return " ".join(w if w.isupper() else w[:1].upper() + w[1:] for w in place.split())


def city_country(city, words=None):
    """The country of a known city, when it is in only one ("Cambridge" is in two: None)."""
    countries = _words(words).cities.get((plain_place(city) or "").lower(), set()) if city else set()
    return next(iter(countries)) if len(countries) == 1 else None


# Where someone is (the offline agent; the instant first draft only when there's no model agent). People can
# say where they are in endless ways, so this reads only a few explicit forms, all about the person, now, and
# leaves the place blank otherwise (the agent then asks "Where are you based?"):
#   "I live in Osaka", "I'm (now) based/living/staying in Osaka", "I'm in Osaka", "I (just) moved to Spain",
#   "Born in Kyoto, based in Berlin" (a clause starting "based/living in"), "Visiting Osaka",
#   and a request with a place, "Looking for a squash partner in (or at, around, near) Osaka", in a sentence that mentions nobody
#   else and nothing past ("my parents live in Nigeria", "I grew up in Munich" give nothing).
# A country merely named ("turkey sandwiches", "I'd love to visit Japan") is never read. Two places: none.
# Someone else: family (never what people look for here), anyone with "my/our/his/her/their" (a boyfriend,
# wife, roommate or friend can be what they're looking for: "Looking for a girlfriend in Osaka"), or he/she/they.
_OTHERS = _re.compile(r"\b(?:parents?|mum|mom|mother|dad|father|sisters?|brothers?|siblings?|relatives?|"
                      r"grand(?:parents?|ma|mother|pa|father)|cousins?|aunts?|uncles?|nieces?|nephews?|daughters?|"
                      r"sons?|kids?|children|family|families|he|she|they|him|them|"
                      r"(?:my|our|his|her|their)\s+(?:[a-z'-]+\s+)?(?:boyfriend|girlfriend|husband|wife|spouse|"
                      r"partner|fianc[eé]e?|ex|roommate|flatmate|friends?|boss|colleagues?|people))\b")
_PAST = _re.compile(r"\b(?:born|raised|grew up|grown up|originally|used to|lived|was|were|back home|childhood|"
                    r"as a (?:kid|child)|when i|came|come|fled|escaped|deported|emigrated|immigrated|ancestors?|"
                    r"heritage|descent|roots|left|years ago|last (?:year|month|week|summer)|studied|spent|worked|"
                    r"honeymoon|someday|one day|would love to visit|visited|trip to|holiday in|vacation in)\b")
_REQUEST = _re.compile(r"\b(?:looking for|look for|looking to|to meet|to find|seeking|searching for|need|want|wanted|"
                       r"hoping to|hope to|"
                       r"i'?d like|would like|selling|giving away|offering|teach|tutor)\b")
# Explicit forms, each ending right before the place.
_HERE_FORMS = _re.compile(
    r"(?:\b(?:i|we)\s+(?:now\s+|currently\s+|still\s+)?(?:live|reside)\s+in"
    r"|\b(?:i'?m|i am|we'?re|we are)\s+(?:now\s+|currently\s+)?(?:living\s+|based\s+|staying\s+|located\s+)?in"
    r"|\b(?:i|we)\s+(?:just\s+|recently\s+)?(?:moved|relocated)\s+to"
    r"|(?:^|[,;]\s*|\b(?:but|and)\s+)(?:now\s+|currently\s+)?(?:based|living)\s+in"
    r"|(?:^|[,;]\s*|\b(?:i'?m|i am)\s+)visiting)"
    r"\s+(?:the\s+)?$")
_PLACE_THEN_COUNTRY = _re.compile(r"(?:\b(?:i|we)\s+(?:now\s+)?live\s+in|\b(?:i'?m|i am)\s+(?:based\s+|living\s+)?in)"
                                  r"\s+(?!(?:my|our|a|an|the)\b)[a-z'-]+(?:\s+(?!(?:my|our|a|an|the)\b)[a-z'-]+)?,\s+$")
_FROM = _re.compile(r"\bfrom\s+(?:the\s+)?(?:[a-z]+\s+)?$")  # "from Brazil", "from northern Spain": not where they are


def _places_in(sentence: str, words: Words) -> list:
    """(kind, name, where it starts) for the known cities and countries a sentence names."""
    found = [("city", m.group(1), m.start()) for m in words._city_rx.finditer(sentence)] if words._city_rx else []
    for c in COUNTRIES:
        if len(c) > 3:
            found += [("country", c, m.start()) for m in _re.finditer(r"\b" + _re.escape(c.lower()) + r"\b", sentence)]
    return found


def _counts(sentence: str, before: str, kind: str) -> bool:
    """Does this place (whose sentence text before it is `before`) say where the person is?"""
    if _FROM.search(before):
        return False
    if _HERE_FORMS.search(before) or (kind == "country" and _PLACE_THEN_COUNTRY.search(before)):
        return True
    # A request with a place: "Looking for a squash partner in Osaka", nobody else and nothing past in it.
    return bool(_re.search(r"\b(?:in|at|around|near|nearby|across|within)\s+(?:the\s+)?$", before) and _REQUEST.search(sentence)
                and not _OTHERS.search(sentence) and not _PAST.search(sentence))


def where_they_are(text, words=None) -> tuple:
    """(city or None, country or None) for where the person is, only from the explicit forms above, among
    the known cities (schema.CITY_COORDS and the listed places, words_in_use) and countries."""
    words = _words(words)
    low = plain_place(" ".join(str(text or "").lower().split())) or ""
    cities, countries = [], []
    for sentence in _re.split(r"(?<=[.!?;])\s+|\n", low):
        for kind, name, at in _places_in(sentence, words):
            if not _counts(sentence, sentence[:at], kind):
                continue
            if kind == "city":
                cities.append(_title(name))
            else:
                countries.append(_country(name) or _title(name))
    cities, countries = list(dict.fromkeys(cities)), list(dict.fromkeys(countries))
    if len(cities) > 1 or len(countries) > 1:
        return None, None  # two places: not sure which, so none (the agent asks)
    city = cities[0] if cities else None
    country = city_country(city, words)
    if countries:
        if country and countries[0] != country:
            return None, None  # a city in one country, another country named: not sure
        country = countries[0]
    return city, country


def known_city(text, words=None) -> tuple:
    """(city, country or None) where the person is, if the text names a known city for them; see where_they_are.
    A city is never read from anything else the person writes ("I'm in recovery" is not a place)."""
    city, country = where_they_are(text, words)
    return (city, country) if city else (None, None)


def quick_facts(text: str, words=None) -> dict:
    """Built-in facts stated plainly in one message, as field -> value, only in words the agent recognises:
    a known city and its country, a country, their gender, an amount of pay, how they'll meet and how far
    they'll travel (fixed choices), and a listed activity they teach or want to learn. Nothing else from the
    message is copied: "I live in Galway" (not a known city) or "I'm in therapy" sets no city, "I'm offering
    bible study lessons" no teaching phrase."""
    low = " ".join(text.lower().split())
    out = {}

    # Where they are: a known city and its country, or a country, only when said about them now (where_they_are).
    city, country = where_they_are(low, words)
    if city:
        out["city"] = city
    if country:
        out["country"] = country
    if gender := own_gender(low):
        out["gender"] = gender  # "I'm a woman", "I am non-binary"

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
    if fmt and has_format(_parse_intent(low).kind):  # "a date in Osaka, online works for me": no format
        out["format"] = fmt
    travel = travel_from_text(low)
    if travel:
        out["travel"] = travel

    # Teaching a listed activity, or wanting to learn one (teaching): the two phrases, as a pair.
    taught = teaching(low, words)
    if taught:
        out["seeking"], out["offering"] = taught[:2]
    # An extra safeguard: shown details never carry a sensitive word (none is listed anyway).
    for key in ("city", "offering", "seeking"):
        if key in out and mentions_sensitive(out[key]):
            out.pop(key)
    if ("offering" in out) != ("seeking" in out):  # the two phrases come as a pair
        out.pop("offering", None)
        out.pop("seeking", None)
    return out


def _article(phrase: str) -> str:
    return "an" if phrase[:1] in "aeiou" else "a"


_ONLINE_WORDS = r"online|remote(?:ly)?|over (?:zoom|video)|video calls?"
_IN_PERSON_WORDS = r"in[- ]person|face to face|meet up|meetups?"


def has_format(kind) -> bool:
    """Does a listing of this kind say how people meet (a "format" field)? Only dating doesn't: it is
    always in person. A kind not known yet might."""
    return kind != "dating"


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
    people's ages in years [lo, hi] (None if not said), the ages they're open to until they say
    (one age band either side of their own), and "said": what the description alone says, which is
    what matching reads when "Looking for" or "Their age" is hidden or empty (so the phone can follow
    the person's edits). Only ever sent to their own phone."""
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
    said = read_listing(draft.get("intent_text") or "", {k: v for k, v in shown.items() if k not in DATING_ONLY}.get)
    years = lambda bands: [AGE_BAND_BOUNDS[bands[0]][0], AGE_BAND_BOUNDS[bands[1]][1]] if bands else None
    return {"genders": [g for g in GENDERS if g in parsed.seeking], "ages": list(span) if span else None,
            "default": [AGE_BAND_BOUNDS[lo][0], AGE_BAND_BOUNDS[hi][1]],
            "said": {"genders": [g for g in GENDERS if g in said.seeking], "ages": years(said.age_range)}}


def ready(draft) -> bool:
    """Can this draft go live: what it's for (dating: who they'd like to meet; anything else: both phrases, who
    they're looking for and what they are to them, since matching compares the two), and where unless it can meet
    online. One rule for Start matching on every page (demo.html canSave, DemoSession._ready), for when the phone
    first shows a new listing's profile (demo.html profileShown), and for what a model agent is told after a save
    (agent_llm._summary)."""
    if not draft:
        return False
    shown = {f["key"]: f["value"] for f in draft.get("fields", [])
             if f.get("enabled") and f.get("visibility") != HIDDEN and f.get("value") not in (None, "", [])}
    if draft.get("kind") == "dating":
        said = (dating_wants(draft) or {}).get("said") or {}
        who = bool(shown.get("wants_gender") or said.get("genders"))
    else:
        who = bool(shown.get("seeking") and shown.get("offering"))
    online = shown.get("format") in ("Online", "In person or online")
    return who and bool(shown.get("city") or shown.get("country") or online)


def empty_draft(first_name: str, age_band: str) -> dict:
    """A blank work-in-progress profile, shown before the agent has drafted anything."""
    fields = []
    options = field_options()
    for key in _COMMON:
        spec = FIELD_REGISTRY[key]
        value = {"first_name": first_name, "age_band": age_band}.get(key)
        fields.append({"key": key, "label": LABELS[key], "value": value, "visibility": spec.default,
                       "sensitive": spec.sensitive, "multi": spec.multi, "enabled": not spec.sensitive,
                       "locked": key == "age_band", "options": options.get(key, [])})
    return {"intent_text": "", "exactness": 0.6, "kind": None, "fields": fields, "understood": "",
            "notes": [], "missing": [], "complete": False, "by": "quick"}


def fill_gaps(draft: dict, facts: dict, visibilities=None) -> list:
    """Put built-in facts into empty fields of a draft (never overwriting anything set). Returns the
    keys filled. Fields the draft doesn't have yet are added, except a format on a kind that has
    none (has_format): a dating listing isn't made online only by "online works for me". The kind
    of match is the agent's to record, never carried over, and a detail only dating has (carries) never
    goes on another kind. Open details are never filled in: an agent writes them in full each time.
    visibilities: the shown or hidden each filled detail takes, where known (the person's choice, or the
    one it had on the draft it came from)."""
    filled = []
    by_key = {f["key"]: f for f in draft["fields"]}
    options = field_options()
    for key, value in facts.items():
        spec = FIELD_REGISTRY.get(key)
        if spec is None or spec.sensitive or key == "kind" or not carries(draft.get("kind"), key):
            continue
        f = by_key.get(key)
        if f is None and key == "format" and not has_format(draft.get("kind")):
            continue
        if f is None:
            f = {"key": key, "label": LABELS[key], "value": None, "visibility": spec.default, "sensitive": False,
                 "multi": spec.multi, "enabled": True, "locked": False, "options": options.get(key, [])}
            draft["fields"].append(f)
            by_key[key] = f
        if f["value"] in (None, "", []):
            f["value"] = list(value) if isinstance(value, (list, tuple)) else value
            if (visibilities or {}).get(key) in (SHOWN, HIDDEN):
                f["visibility"] = visibilities[key]
            filled.append(key)
    return filled


def _prior_visibility(old, choices) -> dict:
    """The shown or hidden each detail had before a new draft: the person's own choices for this listing
    (`choices`: what they saved, and what they set in the editor) and the previous draft's. Where the two
    differ, hidden wins: nothing hidden is ever shown again without someone choosing to show it."""
    out = {f["key"]: f.get("visibility") for f in (old or {}).get("fields", []) if f.get("visibility") in (SHOWN, HIDDEN)}
    for key, vis in (choices or {}).items():
        if vis == HIDDEN or (vis == SHOWN and key not in out):
            out[key] = vis
    return out


def keep_visibility(new: dict, old, choices=None) -> None:
    """A new draft replaces the one on the profile: every detail keeps its shown or hidden from before
    (_prior_visibility: the person's choices for this listing, and the previous draft) unless the new
    draft chose one on purpose ("listed": the keys its agent gave a value and a visibility of its own;
    build_draft records them). A default never decides, so a detail the person hid is never shown again
    just because a later draft left it out, or brought it back with the default. A sensitive detail is
    never made shown here (showing one needs the person's own confirmation in the editor). Open details
    are keyed the same way (about:name), so the person's choice for one stays too."""
    chose = set(new.get("listed") or ())
    before = _prior_visibility(old, choices)
    for f in new.get("fields", []):
        was = before.get(f["key"])
        if was is None or f["key"] in chose or f.get("locked"):
            continue
        if was == HIDDEN or not f.get("sensitive"):
            f["visibility"] = was


def carry_over(new: dict, old, choices=None) -> list:
    """Keep what's already on the profile when a new agent draft replaces it: any built-in the new draft
    left empty (fill_gaps, never a sensitive detail or the kind, and after a change of kind only what's
    about the person), each with the shown or hidden it had, then the person's choices of shown or hidden
    for the rest (keep_visibility). Open details are not carried over: a model agent restates them in full
    with every save (build_draft tells it about one the person chose shown or hidden that it left out), and
    the offline agent keeps those it can't write (keep_details). Returns the keys filled."""
    if not old and not choices:
        return []
    # A change of kind is a new request: only what's about the person carries over (where they are, how far
    # they'll travel), never the old request's own details (its phrases, its pay, who they'd like to meet).
    changed = bool((old or {}).get("kind") and new.get("kind") and old.get("kind") != new.get("kind"))
    filled = fill_gaps(new, {f["key"]: f["value"] for f in (old or {}).get("fields", [])
                             if f.get("enabled") and f.get("value") not in (None, "", [])
                             and not f.get("sensitive") and f["key"] not in ("first_name", "age_band")
                             and not (changed and f["key"] in _KIND_SPECIFIC)},
                       _prior_visibility(old, choices))
    keep_visibility(new, old, choices)
    return filled


def keep_details(new: dict, old) -> list:
    """The offline agent writes only listed words, so the open details already on the profile that its draft
    doesn't have are kept as they were: ones a model agent wrote before it became unavailable, the person's own,
    or ones read from an earlier message. A new request (other_request: another kind, or other phrases) keeps
    only those about the person: what they'd like in the other person (them) and what the request needs (both)
    belonged to the old one. Returns the keys kept."""
    if not old:
        return []
    have = {f["key"] for f in new.get("fields", [])}
    another = other_request(new, old)
    kept = []
    for f in old.get("fields", []):
        about = split_key(f.get("key"))[0]
        if not about or f["key"] in have or f.get("value") in (None, "", []) or not f.get("enabled", True) \
                or (another and about != "me"):
            continue
        new["fields"].append({**f, "value": list(f["value"]), "steps": list(f.get("steps") or []),
                              "options": list(f.get("options") or [])})
        have.add(f["key"])
        kept.append(f["key"])
    return kept


# The built-ins that belong to a kind of match: its own (_BY_KIND), and for every kind except dating the
# phrases matching compares ("looking for", "you are") with the headline and topics. Dating uses who and
# what ages instead. Everything in _COMMON belongs to both kinds.
_PHRASES = ("headline", "seeking", "offering", "topics")
_KIND_SPECIFIC = sorted({k for keys in _BY_KIND.values() for k in keys} | set(_PHRASES))


def kind_keys(kind) -> list:
    """The built-ins a listing of this kind has, in order (with no kind yet, just the common ones)."""
    own = [k for k in _BY_KIND.get(kind, []) if k not in _COMMON]
    phrases = [] if kind in (None, "dating") else [k for k in _PHRASES if k not in own]
    return ["kind"] + _COMMON + own + phrases


def carries(kind, key) -> bool:
    """Can a listing of this kind have this detail? One rule for every way a draft is built or merged (an
    agent's save, details carried over, the instant draft, a save from the phone). Most details mean the same
    on any listing, so an agent sets them wherever they apply, and every open detail goes on either kind (the
    Warden keeps intimacy to dating, schema.detail_problems). Who someone would like to meet and their ages are
    only for dating; their own gender and intimacy preferences only for dating too, once the kind is known."""
    if key in DATING_ONLY:
        return kind == "dating"
    if key in ("gender", "kinks"):
        return kind in (None, "dating")
    return True


def kind_layout(kind, city=None) -> dict:
    """The profile's rows for a kind of match, for the editor when the person changes the kind: "rows"
    (each built-in of this kind, empty, with its options and default visibility) and "drop" (built-ins
    the other kind has that this one doesn't: a listing that isn't dating has no "Looking for" a woman or
    a man, and a dating listing has no "You are (to them)"). Open details stay on either kind."""
    kind = kind if kind in MATCH_KINDS else None
    keys = kind_keys(kind)
    options = field_options()
    rows = []
    for key in keys:
        spec = FIELD_REGISTRY[key]
        rows.append({"key": key, "label": LABELS[key], "value": kind if key == "kind" else None,
                     "visibility": spec.default, "sensitive": spec.sensitive, "multi": spec.multi,
                     "enabled": not spec.sensitive, "locked": key == "age_band", "options": options.get(key, [])})
    if kind is None:  # not chosen yet: nothing is taken away
        return {"rows": rows, "drop": []}
    allowed = set(keys) | ({"headline"} if kind == "dating" else set())  # a dating headline may stay, hidden or not
    return {"rows": rows, "drop": [k for k in _KIND_SPECIFIC if k not in allowed]}


# --------------------------------------------------------------------------------------
# A change the person makes on their profile themselves (SPEC §3: the person approves, and edits, the listing)
#
# The parts of a profile can say the same thing (the "looking for" phrase, the headline, an open detail, a wish), so
# a change made directly to one can leave the others saying the old thing. A kept change (a detail sheet closed, the
# editor done, Save draft) is read part by part (changes_made) and word by word, compared as schema.canonical compares
# text (replacements). One general rule, no word list, decides which other parts still say what was replaced: a part
# depends on the change when it has a word the person replaced (replaced_words). A model agent may update those, and
# only those, to match (keep_edit, the fixed rule its save is held to), and writes the related topics again when what
# the listing is about changed (rereads: they are its own reading of it); without one the phone offers the same words
# changed in them (offers), one tap each. A change the person made on a copy older than the draft (their agent updated
# it meanwhile) is made on the draft as it now is (rebase), and read from the copy they changed (edited), so it never
# puts back what the agent had just brought in line.
# --------------------------------------------------------------------------------------

import copy as _copy
import difflib as _difflib
import unicodedata as _unicodedata

DESCRIPTION = "intent_text"  # the description is a part too: never shown, and matching reads it
_NOT_PARTS = ("first_name", "age_band")  # from the ID check: nobody changes them on the profile
# A word of this many letters or fewer is grammar (an article that changed with the word after it), and so is one a
# letter longer among several replaced together (the conjunction between two of them).
SHORT_WORD = 2
_WORD = _re.compile(r"[^\W_]+")
# What a part is called in a sentence the person reads; a built-in not here is "your <its label>".
_PART_WORDS = {"seeking": "who you're looking for", "offering": "what you offer", "headline": "your headline",
               "topics": "your related topics", DESCRIPTION: "your description", "kind": "the kind of match",
               "wants_gender": "who you'd like to meet", "wants_age": "their ages", "travel": "how far you'll travel",
               "format": "how you'll meet", "income_band": "your income"}


def _nfc(text) -> str:
    return _unicodedata.normalize("NFC", str(text))


def _text_words(text) -> list:
    """The words of a text, each as (start, end, canonical form), positions in its NFC form (_nfc): runs of letters
    and digits, each keeping the marks that belong to its letters (a vowel sign in Devanagari or Thai is part of its
    word, never a break in it), as schema's shape rule reads words."""
    text = _nfc(text)
    if text.isascii():
        return [(m.start(), m.end(), canonical(m.group())) for m in _WORD.finditer(text)]
    out, start = [], None
    for i, c in enumerate(text):
        if c.isalnum() or start is not None and _unicodedata.category(c)[0] == "M":  # isalnum: as [^\W_] reads
            start = i if start is None else start
        elif start is not None:
            out.append((start, i, canonical(text[start:i])))
            start = None
    if start is not None:
        out.append((start, len(text), canonical(text[start:])))
    return out


def _items(value) -> list:
    return [] if value is None else list(value) if isinstance(value, list) else [value]


def _is_wording(word: str, run: tuple) -> bool:
    """Is this word, replaced in this run, wording rather than grammar (SHORT_WORD)."""
    return len(word) > SHORT_WORD + (len(run) > 1)


def _grammar(word: str) -> bool:
    """A word short enough to be grammar when an agent writes it new (an article that goes with a new word): SHORT_WORD
    letters or fewer, and no digit (a number is never grammar)."""
    return len(word) <= SHORT_WORD and not any(ch.isdigit() for ch in word)


def same_word(a: str, b: str) -> bool:
    """Two canonical words that are one word in two forms (a plural, another tense): the same, or one starts with the
    other (with all but its last letter, when that has more than SHORT_WORD + 1 letters and the other is longer), the
    shorter has more than SHORT_WORD letters, and the longer has at most SHORT_WORD + 1 more. A fixed length rule, no
    word list, so it reads every language the same way."""
    if a == b:
        return True
    short, long = sorted((a, b), key=len)
    if len(short) <= SHORT_WORD or len(long) - len(short) > SHORT_WORD + 1:
        return False
    return long.startswith(short) or (len(short) > SHORT_WORD + 1 and len(long) > len(short)
                                      and long.startswith(short[:-1]))


def _any_form(words, of) -> set:
    """The words that are some form (same_word) of a word in `of`."""
    return {w for w in words if any(same_word(w, x) for x in of)}


def _all_words(value) -> set:
    return {w for x in _items(value) for *_, w in _text_words(x)}


def _same(a, b) -> bool:
    """The same value as schema.canonical compares it: a change of case, accents or spacing is no change."""
    return [canonical(x) for x in _items(a)] == [canonical(x) for x in _items(b)]


def _row(draft, key):
    return next((f for f in (draft or {}).get("fields", []) if f.get("key") == key), None)


def part_value(draft, key):
    """A part as the profile says it: the description's text, or a detail's value while it's on (a list for a multi
    or open detail), else None."""
    if key == DESCRIPTION:
        return str((draft or {}).get("intent_text") or "") or None
    f = _row(draft, key)
    if f is None or not f.get("enabled", True) or f.get("value") in (None, "", []):
        return None
    v = f["value"]
    return [str(x) for x in v] if isinstance(v, (list, tuple)) else str(v)


def parts(draft) -> list:
    """A profile's parts by key, in its order: every detail but those from the ID check, then the description."""
    return [*dict.fromkeys(f["key"] for f in (draft or {}).get("fields", []) if f.get("key") not in _NOT_PARTS),
            DESCRIPTION]


def _private(draft, key) -> bool:
    """A detail that is hidden, or sensitive: its words never go into a part that is shown."""
    f = _row(draft, key) if key != DESCRIPTION else None
    return f is not None and (f.get("visibility") == HIDDEN or bool(f.get("sensitive")))


def _shown(draft, key) -> bool:
    """A part a match can see (the description never is)."""
    return key != DESCRIPTION and not _private(draft, key)


def changes_made(before, after) -> list:
    """What a kept change did, part by part: each part whose value differs (as schema.canonical compares) between the
    draft before and the one after, as {"key", "before", "after", "private"} (None where there was no value; private:
    the part is hidden or sensitive on either draft, so its words never go into a shown part). Shown or hidden alone
    changes nothing the profile says."""
    out = []
    for key in dict.fromkeys(parts(after) + parts(before)):
        old, new = part_value(before, key), part_value(after, key)
        if not _same(old, new):
            out.append({"key": key, "before": old, "after": new,
                        "private": _private(before, key) or _private(after, key)})
    return out


def _text_pairs(old: str, new: str) -> list:
    """What replaced what in a text, word by word (difflib's longest matching blocks over canonical words): a run of
    words replaced by as many is read word for word, any other replaced run whole; words only taken away pair with
    None."""
    a, b, new = _text_words(old), _text_words(new), _nfc(new)
    out = []
    for tag, i1, i2, j1, j2 in _difflib.SequenceMatcher(None, [w for *_, w in a], [w for *_, w in b],
                                                         autojunk=False).get_opcodes():
        if tag == "delete":
            out.append((tuple(w for *_, w in a[i1:i2]), None))
        elif tag == "replace" and i2 - i1 == j2 - j1:
            out += [((a[i][2],), new[b[j][0]:b[j][1]]) for i, j in zip(range(i1, i2), range(j1, j2))]
        elif tag == "replace":
            out.append((tuple(w for *_, w in a[i1:i2]), new[b[j1][0]:b[j2 - 1][1]]))
    return out


def _item_pairs(old: list, new: list) -> list:
    """What replaced what in a list: each item taken away, paired in order with an item put in its place and read
    against it word by word, as a text is (_text_pairs), or whole with None when nothing took its place."""
    was, now = {canonical(x) for x in old}, {canonical(x) for x in new}
    gone, came = [x for x in old if canonical(x) not in now], [x for x in new if canonical(x) not in was]
    out = []
    for n, x in enumerate(gone):
        out += _text_pairs(x, came[n]) if n < len(came) else [(tuple(w for *_, w in _text_words(x)), None)]
    return out


def _edit_of(change: dict) -> tuple:
    """What the person changed in a part, from what to what: the draft's values, or, when they changed a copy older
    than the draft (rebase), that copy's value and theirs ("edited")."""
    return tuple(change.get("edited") or (change["before"], change["after"]))


def replacements(change: dict) -> list:
    """The words one change replaced, as pairs: (the replaced words in a row, canonical; the wording that took their
    place as the person wrote it, or None where words were only taken away). A pair whose words all still appear in
    the new value replaced nothing, nor does one of grammar alone (_is_wording). The kind of match is a choice, not
    wording: it replaces no words."""
    old, new = _edit_of(change)
    if change["key"] == "kind" or old is None:
        return []
    if isinstance(old, list) or isinstance(new, list):
        pairs = _item_pairs(_items(old), _items(new))
    elif new is None:
        pairs = [(tuple(w for *_, w in _text_words(old)), None)]
    else:
        pairs = _text_pairs(old, new)
    kept = _all_words(new)
    return [(run, to) for run, to in pairs if any(w not in kept and _is_wording(w, run) for w in run)]


def replaced_words(changes: list) -> set:
    """Every word the person replaced (replacements), canonical: a part with one of them depends on the change."""
    out = set()
    for c in changes:
        kept = _all_words(_edit_of(c)[1])
        for run, _ in replacements(c):
            out |= {w for w in run if w not in kept and _is_wording(w, run)}
    return out


def says(value, words: set) -> bool:
    """Does a part's value have any of these (canonical) words, exactly: what the phone itself offers to replace word
    for word (offers), where another form would be the wrong word ("large" is no form of "largely")."""
    return bool(_all_words(value) & words)


def says_in_some_form(value, words: set) -> bool:
    """Does a part's value have any of these words in some form (same_word: "baking" for "bake"): what the person's
    agent may bring in line after a change, and is asked to, since it reads what the words mean, so a part that says a
    word they replaced in another form isn't left saying the old thing."""
    return bool(_any_form(_all_words(value), words))


# The parts that say what a listing is about, in the person's words or their agent's, and the one part their agent
# writes from them: related topics are its own reading of what the listing is about (broader areas), so they follow
# what it's about, not only the words they share with it. A detail says something about the listing (a colour, a size,
# a level), not what it is: a change to one rereads the topics only when the headline, seeking or offering, which name
# what it is, still says what it replaced (being brought in line anyway, so no turn is added for it); the description
# says everything (the place and times too), so its saying it doesn't count.
_WHAT = ("headline", "seeking", "offering")
_ABOUT = (*_WHAT, DESCRIPTION)
_READING = "topics"


def rereads(draft, changes: list) -> bool:
    """Is the topics' reading of a listing out of date: a change that isn't to a hidden or sensitive part replaced words
    in a part that says what the listing is about, or words that the headline, seeking or offering, unchanged, still
    says; the draft has topics; and the person didn't set the topics themselves."""
    changed = {c["key"] for c in changes}
    if _READING in changed or part_value(draft, _READING) is None:
        return False
    shown = [c for c in changes if not c["private"]]
    words = replaced_words(shown)
    return (any(c["key"] in _ABOUT and replacements(c) for c in shown)
            or any(says(part_value(draft, k), words) for k in _WHAT if k not in changed))


def reviews(draft, changes: list) -> bool:
    """Did the person change what the listing is about itself (the headline, seeking, offering or the description, not
    hidden), so their agent reviews the whole profile (keep_edit): a change of subject, not a word in a detail."""
    return any(c["key"] in _ABOUT and replacements(c) for c in changes if not c["private"])


def depends(draft, changes: list) -> bool:
    """Is there anything to bring in line: another part of the draft still says a word the person replaced, the
    topics are to be read again (rereads), or the person changed what the listing is about (reviews)."""
    words, changed = replaced_words(changes), {c["key"] for c in changes}
    return (bool(words) and any(says_in_some_form(part_value(draft, k), words) for k in parts(draft) if k not in changed)) \
        or rereads(draft, changes) or reviews(draft, changes)


def _private_texts(draft, changes: list) -> list:
    """Each value of something hidden or sensitive, one item at a time: each hidden or sensitive row of the draft,
    whether it is on or off and whether or not it is a part (a hidden first name too), with an open detail's name
    (hidden, it is never shown either), and each hidden or sensitive part a change replaced, its old values too (on the
    copy the person changed as well, _edit_of)."""
    out = []
    rows = [(f.get("key"), f.get("value")) for f in (draft or {}).get("fields", []) if _private(draft, f.get("key"))]
    rows += [(c["key"], [*_items(c["before"]), *_items(c["after"]), *(v for x in _edit_of(c) for v in _items(x))])
             for c in changes if c["private"]]
    for key, value in rows:
        out += _items(value)
        about, name = split_key(key) if isinstance(key, str) else ("", "")
        if about:
            out.append(name)
    return out


def _private_words(draft, changes: list) -> set:
    """Every word of something hidden or sensitive (_private_texts)."""
    return set().union(*(_all_words(x) for x in _private_texts(draft, changes)))


def _says_all(text, words) -> bool:
    """Does this text say every one of these words, each in some form (same_word), in any order."""
    have = _all_words(text)
    return all(any(same_word(w, x) for x in have) for w in words)


def _hunks(a: list, b: list) -> tuple:
    """What changed from one run of words to another: difflib's changed stretches of a, as (start, end), and where each
    word of a that stayed is in b."""
    hunks, at = [], {}
    for tag, i1, i2, j1, j2 in _difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            at.update({i1 + n: j1 + n for n in range(i2 - i1)})
        else:
            hunks.append((i1, i2))
    return hunks, at


def _swapped(old: list, new: list) -> tuple:
    """A list's items taken away and put in, compared canonically, each taken away paired in order with one put in
    (as _item_pairs reads them): ({taken away: what took its place, or None}, items put in with nothing taken away)."""
    was, now = {canonical(x) for x in old}, {canonical(x) for x in new}
    gone, came = [x for x in old if canonical(x) not in now], [x for x in new if canonical(x) not in was]
    return ({canonical(x): (came[n] if n < len(came) else None) for n, x in enumerate(gone)}, came[len(gone):])


def _rebase_items(base: list, mine: list, theirs: list) -> list:
    """rebase for a list, item by item: an item the person took away comes off the agent's list, an item they put in its
    place takes its place there (or the place of what the agent put in for it), and one they only added is added."""
    mine_swaps, added = _swapped(base, mine)
    their_swaps, _ = _swapped(base, theirs)
    out = [str(x) for x in theirs]
    for gone, put in mine_swaps.items():
        at = next((n for n, x in enumerate(out) if canonical(x) == gone), None)
        if at is None and their_swaps.get(gone) is not None:  # the agent changed it too: the person's wins
            at = next((n for n, x in enumerate(out) if canonical(x) == canonical(their_swaps[gone])), None)
        if at is not None:
            del out[at]
        if put is not None:
            out.insert(len(out) if at is None else at, str(put))
    out += [str(x) for x in added]
    seen = set()
    return [x for x in out if x.strip() and not (canonical(x) in seen or seen.add(canonical(x)))]


def rebase(base, mine, theirs, choices=None):
    """A part the person changed on a copy older than the draft (from base to mine) that their agent changed meanwhile
    (from base to theirs): the person's change made on the draft as it now is, compared as schema.canonical compares.
    In a text, each stretch of words they changed replaces the same stretch of the agent's value, and the agent's
    changes elsewhere in it stay; where both changed the same words, or words side by side, the person's change wins
    there. In a list, item by item (_rebase_items). A part with fixed choices, or one either of them emptied, takes the
    person's value whole. One fixed rule, no word list."""
    if _same(base, mine):
        return theirs
    if _same(base, theirs) or _same(mine, theirs) or None in (base, mine, theirs) or choices is not None:
        return mine
    if any(isinstance(v, list) for v in (base, mine, theirs)):
        return _rebase_items(_items(base), _items(mine), _items(theirs)) or mine
    m, t = _nfc(mine), _nfc(theirs)
    wm, wt = _text_words(m), _text_words(t)
    tb = [w for *_, w in _text_words(base)]
    mine_hunks, mine_at = _hunks(tb, [w for *_, w in wm])
    their_hunks, their_at = _hunks(tb, [w for *_, w in wt])
    # Stretches that share or border a word go together: the words either side of a group stayed in both values, so
    # they place it in each. Only a group with a change of the person's is theirs.
    groups = [[lo, hi, ours] for ours, hunks in ((True, mine_hunks), (False, their_hunks)) for lo, hi in hunks]
    joined = True
    while joined:
        joined = False
        for x, y in ((x, y) for x in range(len(groups)) for y in range(x + 1, len(groups))):
            (a1, a2, a_ours), (b1, b2, b_ours) = groups[x], groups[y]
            if max(a1, b1) <= min(a2, b2):
                groups[x] = [min(a1, b1), max(a2, b2), a_ours or b_ours]
                del groups[y]
                joined = True
                break
    out, n = t, len(tb)
    try:
        for lo, hi, ours in sorted(groups, reverse=True):
            if ours:
                ms, me = (wm[mine_at[lo - 1]][1] if lo else 0), (wm[mine_at[hi]][0] if hi < n else len(m))
                ts, te = (wt[their_at[lo - 1]][1] if lo else 0), (wt[their_at[hi]][0] if hi < n else len(t))
                out = out[:ts] + m[ms:me] + out[te:]
    except KeyError:  # never expected (see above); the person's value, if so
        return mine
    return out if _text_words(out) else mine


def _swap(text: str, pairs: list) -> str:
    """text with every run of words a pair replaced (the same words in a row, compared canonically) written as the
    person wrote what took their place. Words only taken away are left in a sentence: cutting them out could leave
    it broken."""
    text = _nfc(text)
    for run, to in pairs:
        if to is None:
            continue
        words, spans, i = _text_words(text), [], 0
        canon = [w for *_, w in words]
        while i + len(run) <= len(canon):
            if tuple(canon[i:i + len(run)]) == run:
                spans.append((words[i][0], words[i + len(run) - 1][1]))
                i += len(run)
            else:
                i += 1
        for start, end in reversed(spans):
            text = text[:start] + to + text[end:]
    return text


def _choices(draft, key):
    """A part's fixed choices, if it has any: a built-in's (field_options) or an open detail's steps."""
    f = _row(draft, key)
    if f is not None and is_detail_key(key):
        return list(f.get("steps") or []) if f.get("compare") == "steps" else None
    return field_options().get(key)


def _offered(value, pairs: list, choices):
    """A part's value with the pairs' words replaced (_swap); in a list, an item that was exactly what the person took
    away comes off. None when nothing is left, or when a part with fixed choices would get one that isn't one of
    them."""
    items = []
    for x in _items(value):
        if not any(to is None and tuple(w for *_, w in _text_words(x)) == run for run, to in pairs):
            items.append(_swap(x, pairs))
    seen = set()
    items = [x for x in items if x.strip() and not (canonical(x) in seen or seen.add(canonical(x)))]
    if not items:
        return None
    if choices is not None:
        items = [next((c for c in choices if canonical(c) == canonical(x)), None) for x in items]
        if None in items:
            return None
    return items if isinstance(value, list) else items[0]


def offers(draft, changes: list) -> list:
    """Without a model agent, what the phone offers to change too after a kept change (nothing changes without the
    person's tap): every other part that still has words the person replaced, with them replaced the same way
    (replacements, _swap), and an item they took away taken off a list (never its last one). A part with fixed
    choices or steps is offered only one of them; the kind of match and their ages, never. A hidden or sensitive
    part's words are offered only to parts that are never shown (other hidden details, the description). Each offer is
    {"key", "before", "after"}, values as the draft holds them."""
    changed = {c["key"] for c in changes}
    out = []
    for key in parts(draft):
        value = part_value(draft, key)
        if key in changed or key in ("kind", "wants_age") or value is None:
            continue
        pairs = [p for c in changes if not (c["private"] and _shown(draft, key)) for p in replacements(c)]
        new = _offered(value, pairs, _choices(draft, key)) if pairs else None
        if new is not None and not _same(new, value):
            out.append({"key": key, "before": value, "after": new})
    return out


def _definition(row) -> tuple:
    """How an open detail compares, as its agent wrote it (a change of wording never changes it): whom it's about, how
    it compares and its steps. Whether a wish must be met is _required."""
    return (row or {}).get("about"), (row or {}).get("compare"), tuple(canonical(s) for s in (row or {}).get("steps") or ())


def _required(row) -> bool:
    """Whether a wish must be met, as the draft holds it (matching reads it only on a wish)."""
    return (row or {}).get("required", True) is not False


def keep_edit(new: dict, saved: dict, changes: list) -> tuple:
    """The fixed rule a model agent's save after a kept change is held to. From the draft the person kept (saved), the
    only parts taken from the agent's save (new) are those that still said a word the person replaced
    (replaced_words), as the agent reworded them, and only in words the part already had or the person
    just wrote (each in any form, same_word), with grammar (_grammar). So what the person changed keeps their value, a
    part with no value gets none, a part brought in line gets no word from anywhere else (nothing they didn't give),
    every other part stays as it was, no part is taken off, an open detail keeps whom it's about, how it compares and
    its steps, a wish keeps whether it must be met, shown or hidden stays as it was, and no word of a hidden or
    sensitive part's new value, in any form, goes into a part that is shown. When the person changed what the listing
    is about themselves (reviews), the agent reviews the whole profile: any shown part that says what the listing is (the headline, seeking,
    offering, the description) or shown detail about the person, other than what the person just changed, may be
    rewritten in any words, never a new word of anything hidden or sensitive (as below); hidden parts, built-ins with
    fixed choices and wishes stay as the rules above keep them. And the agent's topics are taken in any words a match can
    already see on the profile (its shown parts, the person's new wording among them) or that the topics had, each in
    any form, with grammar; never with a new word, in any form, of anything hidden or sensitive (_private_words) that
    no shown part says word for word; and never with one topic that says every word of a hidden or sensitive value of
    several (grammar aside), each in any form and in any order, unless one shown item, or the topics as they were,
    already says them all. A save with
    another kind of match changes nothing. The agent's save is refused for any of these (agent_llm.build_draft), so it
    can fix it, and whatever comes back is corrected by the same rule. Two are corrected without a word, since neither
    says anything the agent may have claimed: an unrelated part it left out (kept), and a sensitive detail's shown,
    which only the person can choose (it saves one hidden). Returns (draft, fixes): fixes are (key, why) for each part
    of the agent's save that wasn't taken, why one of "theirs", "new", "unrelated", "kept", "compare", "required",
    "hidden", "added", "unseen", "visibility", "kind"."""
    out = _copy.deepcopy(saved)
    if new.get("kind") != saved.get("kind"):
        return out, [("kind", "kind")]
    changed = {c["key"] for c in changes}
    words = replaced_words(changes)
    private = set().union(*(_all_words(c["after"]) for c in changes if c["private"]))
    reread = rereads(saved, changes)
    review = reviews(saved, changes)
    # Read again, the topics are made of what a match can already see: words of the profile's shown parts and the
    # topics' own, each in any form, and grammar. Never a new word, in any form, of anything hidden or sensitive that
    # no shown part says, however short (a word a shown part says word for word tells a match nothing new, whatever it
    # starts like; the topics' own words are theirs, hidden ones too). And never one topic with every word of a hidden
    # or sensitive value of several, in any order or form, which shown words could otherwise put together, unless one
    # shown item (or the topics as they were) already says them all. Other wording for a hidden value (another name
    # for it) is beyond a word rule (TRUST.md limit 26).
    seen = {w for k in parts(saved) if _shown(saved, k) for w in _all_words(part_value(saved, k))}
    topics_were = part_value(saved, _READING)
    secret = _private_words(saved, changes) - seen - _all_words(topics_were) if reread or review else set()
    shown_items = [x for k in parts(saved) if _shown(saved, k) for x in _items(part_value(saved, k))] + _items(topics_were)
    together = [ws for ws in ({w for w in _all_words(x) if not _grammar(w)} for x in _private_texts(saved, changes))
                if len(ws) > 1 and not any(_says_all(x, ws) for x in shown_items)] if reread or review else []
    given = set().union(*(_all_words(c["after"]) for c in changes))  # the person's own new wording
    mine = {f["key"]: f for f in out.get("fields", [])}
    theirs = {f["key"]: f for f in new.get("fields", [])}
    fixes, taken = [], False
    for key in dict.fromkeys(parts(new) + parts(saved)):
        was, now = part_value(saved, key), part_value(new, key)
        old_row, new_row = mine.get(key), theirs.get(key)
        both = old_row is not None and new_row is not None and now is not None
        if both and not old_row.get("sensitive") and new_row.get("visibility") != old_row.get("visibility"):
            fixes.append((key, "visibility"))
        if both and is_detail_key(key) and _definition(new_row) != _definition(old_row):
            fixes.append((key, "compare"))
            continue
        if both and split_key(key)[0] in WISHES and _required(new_row) != _required(old_row):
            fixes.append((key, "required"))  # only a wish's: matching never reads it on a detail about the person
            continue
        if _same(was, now):
            continue
        dependent = was is not None and says_in_some_form(was, words)
        if now is None and key not in changed and not dependent:
            continue  # left out, and it didn't say what they replaced: kept as it was
        extra = _all_words(now) - _all_words(was)  # every word the agent brought into the part
        if review and was is not None and key not in changed and _shown(saved, key) and key != _READING and (
                key in _ABOUT or (is_detail_key(key) and split_key(key)[0] not in WISHES)):
            # What the listing is about changed, so the agent reviews the whole profile: a shown part that says what
            # it is (the headline, seeking, offering, description) or a shown detail about the person may be rewritten
            # in any words but the wording of anything hidden or sensitive, as the topics are. Never a hidden part,
            # one the person just changed, a built-in with fixed choices, or a wish (what they want in the other
            # person stays theirs, so no must is written for them).
            why = ("kept" if now is None
                   else "hidden" if _any_form(extra - seen, secret)
                   or any(_says_all(x, ws) for x in _items(now) for ws in together)
                   else None)
        elif key == _READING and reread and was is not None and key not in changed:
            # Read again for what the listing is about now, from what a match can already see.
            why = ("kept" if now is None
                   else "hidden" if _any_form(extra - seen, secret)
                   or any(_says_all(x, ws) for x in _items(now) for ws in together)
                   else "unseen" if {w for w in extra if not _grammar(w)} - _any_form(extra, seen | _all_words(was))
                   else None)
        else:
            why = ("theirs" if key in changed else "new" if was is None else "unrelated" if not dependent
                   else "kept" if now is None
                   else "hidden" if _shown(saved, key) and _any_form(extra, private)
                   else "added" if {w for w in extra if not _grammar(w)} - _any_form(extra, _all_words(was) | given)
                   else None)
        if why:
            fixes.append((key, why))
            continue
        taken = True
        if key == DESCRIPTION:
            out["intent_text"] = now
        else:
            old_row["value"] = _copy.deepcopy(new_row["value"])
    if taken:
        out["by"] = new.get("by", out.get("by"))
    return out, fixes


def keep_meanwhile(result: dict, start: dict, now: dict) -> tuple:
    """An agent's draft (result), made from the draft as it was when its turn began (start), with every part that
    changed on the draft since (now: a change the person kept, or saved live, while it was replying) as it is now: what
    the person did meanwhile wins over what their agent wrote from an older copy. With another kind of match since, the
    draft as it is now, whole. Returns (draft, keys): keys are the parts the agent had changed whose update gave way."""
    if (start or {}).get("kind") != (now or {}).get("kind"):
        return _copy.deepcopy(now), [c["key"] for c in changes_made(start, result)]
    out, lost = _copy.deepcopy(result), []
    theirs = {c["key"] for c in changes_made(start, result)}
    for c in changes_made(start, now):
        key = c["key"]
        if key in theirs:
            lost.append(key)
        if key == DESCRIPTION:
            out["intent_text"] = (now or {}).get("intent_text", "")
            continue
        row = _row(now, key)
        fields = [f for f in out.get("fields", []) if f.get("key") != key]
        at = next((n for n, f in enumerate(out.get("fields", [])) if f.get("key") == key), len(fields))
        out["fields"] = fields[:at] + ([_copy.deepcopy(row)] if row is not None else []) + fields[at:]
    return out, lost


def part_words(key: str) -> str:
    """A part in plain words, for a sentence the person reads."""
    if key in _PART_WORDS:
        return _PART_WORDS[key]
    about, name = split_key(key)
    if about:
        return {"me": f"your {name}", "them": f"the {name} you'd like", "both": f"the {name}"}[about]
    return "your " + LABELS.get(key, key).lower()


def part_label(key: str) -> str:
    """A part's label on the phone (the description's too)."""
    return "Description" if key == DESCRIPTION else label_for(key)


def _listed(words: list) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def change_line(changes: list) -> str:
    """The short line the person's chat shows when their agent is told about a change they made: which parts, never
    their values (an old hidden value never stays on screen)."""
    return f"You changed {_listed([part_words(c['key']) for c in changes])} on your profile."


NOTHING_ELSE = "Nothing else on your profile needed changing, so the rest is as it was."


def updated_line(saved, result) -> str:
    """What a consistency update did, said from what actually changed (never from what an agent says it did)."""
    done = [part_words(c["key"]) for c in changes_made(saved, result)]
    return f"I've updated {_listed(done)} to match." if done else NOTHING_ELSE


# --------------------------------------------------------------------------------------
# A must comes from the person (SPEC §5, wishes)
#
# A wish (an open detail about them or both) that must be met shuts out every listing that doesn't show something to
# meet it, so only the person can make one. A wish their agent saves as required stays required only by one of three
# fixed rules, with no word list and no judgment of meaning (ground_wishes):
#   - what they typed for the listing says one of its values in full (wish_said). Words are compared as same_word
#     compares them (a plural or another tense, by a fixed length rule), a number only as the same number (written with
#     or without thousands separators; a decimal or a time, "2.5" or "9:15", is one number, never its parts), a number
#     with the word written right after it as one word, with or without a space ("5 km" is "5km", said only by the
#     number with that word, the word compared by its own letters, never the number's digits), except a word of
#     SHORT_WORD letters or fewer between two numbers ("7 to 9"), which connects them and joins neither, in Hangul,
#     which writes a noun's particles on to it, also inside a typed word that starts with it, and in a script written
#     without spaces anywhere inside a run (never inside a longer number, in digits or in numeral characters). A typed
#     run that mixes scripts is also read as its pieces, so a Latin word or a number written inside a Japanese or
#     Chinese run is a word they typed. The detail's name alone never counts: words about themselves can share it. Any
#     one value said is enough: more values widen who meets a wish, never whom it shuts out for not showing the detail,
#     so a value said about anything keeps the wish (TRUST.md limit 28);
#   - a save kept it as ideally, so their agent was told to ask them whether it must be met, and the words it quotes as
#     theirs for it (its their_words) are all in the message they typed next, after the latest such save: their answer
#     (answered). A message typed before that save is never their answer, so a quote from it counts for nothing; one
#     typed after it counts whatever it says (a word rule can't tell a yes from a no);
#   - the draft the save replaces had it as a must, and it is saved again as it was or wider (_as_wide): it passed the
#     rule when it was saved, or came from what went live.
# Otherwise it is kept as "ideally": never taken off, reworded, or made shown or hidden, only no longer required. At
# exactness STRICT_EXACTNESS or more every wish shuts out who doesn't show it, "ideally" ones too, so there every wish
# is held to the same three rules, and a listing with one that passes none is made less exact, just under
# (ground_exactness): only ever lowered. Their own words are what they typed themselves for that listing, never a turn
# the app wrote (the demo passes them on, with which wishes were kept as ideally and when: DemoSession._said,
# _answers). A rule of the person's own agent and session, not a Warden check: the Warden never decides who matches,
# and it never sees anyone's chat.
# --------------------------------------------------------------------------------------

WISHES = ("them", "both")  # whom a wish is about: what the other person's own value should be, or what both share
# The most words an agent may quote as the person's answer for one must (their_words): the few that say it.
THEIR_WORDS_MAX = 12
# Scripts that write a noun's particles on to it, with no space between (the start of their letters' Unicode names).
_PARTICLED_SCRIPTS = ("HANGUL",)
# A thousands separator inside a number ("1,800", "12.500"): a number reads the same written with or without one.
_THOUSANDS = _re.compile(r"(?<=\d)[,.](?=\d{3}(?!\d))")
# Any other point, comma or colon inside a number ("2.5", "4,8", "9:15"): the number is one number, never its parts, so
# it is written with one joiner there (a letter, so a text's words keep it inside the word, and never a digit).
_INNER = _re.compile(r"(?<=\d)[.,:](?=\d)")
_JOIN = "\u02b9"
# A number: digits, with a joiner wherever a point, comma or colon was inside it (_INNER).
_NUMBER = _re.compile(r"\d+(?:\u02b9\d+)*")
# The space or dash between a number and the word written right after it ("5 km", "30-minute"): a number and that word
# are read as one word ("5km"), since the word says what the number counts; but never a word of SHORT_WORD letters or
# fewer followed by another number ("7 to 9"): that word connects the two, and counts nothing. And the place between a
# number and letters written on to it ("5km"), where a number they typed is also read alone.
_GAP = r"[\s\-\u2010-\u2015]+"
_UNIT_GAP = _re.compile(rf"(?<=\d){_GAP}(?=[^\W\d_])(?![^\W\d_]{{1,{SHORT_WORD}}}{_GAP}\d)")
_UNIT = _re.compile(r"(?<=\d)(?=[^\W\d_])(?!\u02b9)")


def _one_way(text) -> str:
    """A text with each number written one way: no thousands separator, and a joiner for any other point, comma or
    colon inside it (_INNER), so "2.5" is one number and never 2 and 5."""
    return _INNER.sub(_JOIN, _THOUSANDS.sub("", _nfc(text)))


def _numbered(text) -> str:
    """A text with each number written one way (_one_way), and with the word written right after it as one word
    ("5 km" is "5km")."""
    return _UNIT_GAP.sub("", _one_way(text))


def _wish_words(value) -> set:
    """The words of a wish's value or name, or of a quote, each in canonical form (_all_words), its numbers written one
    way (_numbered): a number with the word right after it is one word, so "10km" is never said by 10 alone."""
    return {w for x in _items(value) for w in _all_words(_numbered(x))}


def _kind(c: str) -> int:
    """Which kind of writing a character is, for where a typed run splits into its pieces (_pieces): a script written
    without spaces, Hangul, or any other."""
    return 1 if _unspaced(c) else 2 if _particled(c) else 0


def _pieces(word: str) -> set:
    """The pieces of a word that mixes scripts, split wherever a character of a script written without spaces or of
    Hangul meets one of another kind (_kind); a mark stays with its letter. A word of one kind has none."""
    out, start, last = set(), 0, None
    for i, c in enumerate(word):
        if _unicodedata.category(c)[0] == "M":
            continue
        kind = _kind(c)
        if last is not None and kind != last:
            out.add(word[start:i])
            start = i
        last = kind
    return out | {word[start:]} if out else set()


def said_words(texts) -> set:
    """The words of what the person typed themselves, each in its canonical form: as _wish_words reads them (a number
    with the word right after it as one word), and each number and word also alone, so a number they typed with a unit
    still says the number ("10km" says 10, and "after 10 on" says 10, never "10km"). A run that mixes scripts is also
    read as its pieces (_pieces): a Latin word or a number written inside a Japanese or Chinese run is one they
    typed."""
    out = set()
    for t in texts or ():
        out |= _wish_words(t) | _all_words(_UNIT.sub(" ", _one_way(t)))
    return out | {p for w in out for p in _pieces(w)}


def _unspaced(word: str) -> bool:
    """A word in a script written without spaces (schema.LIST_UNSPACED_SCRIPTS): there one run is many words."""
    return any(_unicodedata.name(c, "").startswith(LIST_UNSPACED_SCRIPTS) for c in word)


def _particled(word: str) -> bool:
    """A word in a script that writes a noun's particles on to it (_PARTICLED_SCRIPTS)."""
    return any(_unicodedata.name(c, "").startswith(_PARTICLED_SCRIPTS) for c in word)


def _letters(word: str) -> int:
    """How long a word is, counted after canonical decomposition, as schema's shape rule counts it: a Hangul syllable
    is the two or three letters it is written with, and a mark counts like a letter."""
    return len(_unicodedata.normalize("NFD", word))


def _wording(word: str) -> bool:
    """A word that says something: more than SHORT_WORD letters (_letters), a number, in a script written without
    spaces, where a word of one or two letters is a word, or in Hangul, where a syllable standing alone is one."""
    return _unspaced(word) or _particled(word) or _letters(word) > SHORT_WORD or any(ch.isdigit() for ch in word)


def _numeral(c: str) -> bool:
    """A character that is part of a number: a digit or a numeral character (十, 五), or the joiner inside one."""
    return c.isnumeric() or c == _JOIN


def _inside(word: str, run: str) -> bool:
    """Is this word somewhere inside a run they typed, never as part of a longer number (5 is not inside 15, nor 五
    inside 十五)?"""
    at = run.find(word)
    while at >= 0:
        end = at + len(word)
        if not (_numeral(word[0]) and at > 0 and _numeral(run[at - 1])
                or _numeral(word[-1]) and end < len(run) and _numeral(run[end])):
            return True
        at = run.find(word, at + 1)
    return False


def _word_said(word: str, words: set) -> bool:
    """Is this canonical word one they typed, in some form (same_word)? A number only as the same number (180 never says
    1800: no form of a number adds or drops a digit), and the word joined to it compared by its own letters, never the
    number's digits (so "12cups" is no form of "12cm"). In Hangul a word of more than SHORT_WORD letters also counts
    inside a typed word that starts with it (its particles follow it). In a script written without spaces a run they
    typed is many words, so anywhere inside one counts, never inside a longer number (_inside)."""
    if _unspaced(word):
        return any(_inside(word, w) for w in words)
    number = _NUMBER.findall(word)
    particled = _particled(word) and _letters(word) > SHORT_WORD
    rest = _NUMBER.sub("", word)
    return any(_NUMBER.findall(w) == number and (w == word or particled and w.startswith(word)
                                                  or bool(rest) and same_word(rest, _NUMBER.sub("", w)))
               for w in words)


def _in_words(text, words: set, named: bool) -> bool:
    """Is this text said in full in their words: every word of it that says something (_wording) in some form
    (_word_said)? A text made only of grammar-sized words counts only when named: alone, such a word says nothing of
    what it's about."""
    need = _wish_words(text)
    wording = {w for w in need if _wording(w)}
    return bool(need) and all(_word_said(w, words) for w in wording or need) and bool(wording or named)


def wish_said(row, words: set) -> bool:
    """Did the person say this wish (an open detail's row, or its Field), in their words (said_words): it says nothing
    (schema.says_nothing: "any" alone, or every step), or one of its values is said in full (_in_words), a value made
    only of grammar-sized words only with a word of the detail's name. Any one value is enough: more values widen who
    meets a wish, never whom it shuts out for not showing the detail, so a value said about anything (a word of theirs
    an agent adds as an extra value too) keeps it. The name alone is never enough: someone's words about themselves can
    share it."""
    f = row if isinstance(row, Field) else detail_field(row)
    if f is None or says_nothing(f):
        return True
    name = _wish_words(f.name)
    named = any(_word_said(w, words) for w in ({w for w in name if _wording(w)} or name))
    return any(_in_words(value, words, named) for value in f.strings())


def answered(their_words, texts) -> bool:
    """Are the words an agent quotes as the person's own for a must (their_words) their answer to its question whether
    that wish must be met: a few words (THEIR_WORDS_MAX), every one of them, in some form (_word_said), in the first
    message of these (texts: what they typed after the latest save that kept that wish as ideally, so after their agent
    was last told to ask), the one that answers it. Whatever it says: a word rule can't tell a yes from a no."""
    if not isinstance(their_words, str) or not 0 < len(_text_words(their_words)) <= THEIR_WORDS_MAX:
        return False
    quote, first = _wish_words(their_words), list(texts or ())[:1]
    return bool(first) and all(_word_said(w, said_words(first)) for w in quote)


def _as_wide(f: Field, was) -> bool:
    """Is this wish the must it was (was: the same wish on the draft it replaces, required), or a wider one: the same
    compare and steps, and every value it had still among its values."""
    if was is None or f.compare != was.compare or list(map(canonical, f.steps)) != list(map(canonical, was.steps)):
        return False
    return set(map(canonical, was.strings())) <= set(map(canonical, f.strings()))


def ground_wishes(rows, words: set, their_words=None, before=None, answers=None) -> tuple:
    """Each wish (about them or both) saved as required that stays required by none of the three rules kept as
    "ideally": required False, and nothing else about it changed. The rules: the person's words (said_words) say one of
    its values (wish_said); or the words their agent quotes as theirs for it (their_words: key -> text) are their
    answer, all in one message they typed after a save kept it as ideally (answered; answers: key -> those messages);
    or the draft it replaces had it as a must (before: its rows or Fields), and it is as wide or wider (_as_wide). It
    only ever lowers required: nothing is added, taken off or reworded, and shown or hidden stays as it was. rows: a
    draft's rows (dicts) or a listing's Fields; anything that isn't a wish passes through as it is. Returns (rows,
    keys): a new list, with a changed copy of each wish kept as ideally, and those wishes' keys."""
    musts = {}
    for row in before or ():
        was = row if isinstance(row, Field) else detail_field(row)
        if was is not None and was.about in WISHES and was.required:
            musts[was.key] = was
    their_words, answers = their_words or {}, answers or {}
    out, ideally = [], []
    for row in rows:
        f = row if isinstance(row, Field) else detail_field(row)
        if (f is not None and f.about in WISHES and f.required and not _as_wide(f, musts.get(f.key))
                and not wish_said(f, words) and not answered(their_words.get(f.key), answers.get(f.key))):
            row = replace(row, required=False) if isinstance(row, Field) else {**row, "required": False}
            ideally.append(f.key)
        out.append(row)
    return out, ideally


def asked_about(keys, rows, before=None, answers=None) -> set:
    """Of these wishes' keys (ones a save keeps as ideally), those the person was already asked about: on the draft the
    turn began from (before) as the same wish (the same compare, steps and values), with something typed since a save
    kept it as ideally (answers: key -> those messages, the first their answer). Their agent is never told to ask about
    it again, so their answer stays the first message after it was last told to (the demo's ideally_since stays); a wish
    changed since is another question."""
    answers, was = answers or {}, {}
    for row in before or ():
        f = row if isinstance(row, Field) else detail_field(row)
        if f is not None and f.about in WISHES:
            was[f.key] = f
    out = set()
    for row in rows or ():
        f = row if isinstance(row, Field) else detail_field(row)
        if f is not None and f.key in keys and answers.get(f.key) and _as_wide(f, was.get(f.key)) \
                and _as_wide(was[f.key], f):
            out.add(f.key)
    return out


# The most exact a listing is made while a wish on it passes none of the rules (ground_exactness): just under
# STRICT_EXACTNESS, where an "ideally" wish no longer shuts out who doesn't show it, on the phone's own scale (tenths).
UNSTRICT = round(STRICT_EXACTNESS - 0.1, 1)


def _as_must(row):
    """A wish's row or Field as a must (required), anything else as it is."""
    f = row if isinstance(row, Field) else detail_field(row)
    if f is None or f.about not in WISHES:
        return row
    return replace(row, required=True) if isinstance(row, Field) else {**row, "required": True}


def _exact(exactness) -> bool:
    return isinstance(exactness, (int, float)) and not isinstance(exactness, bool) and exactness >= STRICT_EXACTNESS


def ground_exactness(exactness, rows, words: set, their_words=None, before=None, before_exactness=None,
                     answers=None) -> tuple:
    """At exactness STRICT_EXACTNESS or more a listing fails everyone who doesn't show something to meet each of its
    wishes, "ideally" ones too (matcher.details_ok), so there every wish shuts people out as a must does. So such an
    exactness stays only when every wish on the listing stays a must by the three rules of ground_wishes, each read as
    one (the draft it replaces too, when that was as exact: its wishes were musts then); otherwise it is lowered to
    UNSTRICT, where an "ideally" wish shuts nobody out for not showing the detail. It is never raised, and nothing else
    changes. rows: a draft's rows or a listing's Fields (as ground_wishes takes them, after it). Returns (exactness, the
    keys of the wishes that pass none of the rules: none unless it was that exact)."""
    if not _exact(exactness):
        return exactness, []
    _, keys = ground_wishes([_as_must(r) for r in rows or ()], words, their_words,
                            [_as_must(r) for r in before or ()] if _exact(before_exactness) else before, answers)
    return (UNSTRICT if keys else exactness), keys
