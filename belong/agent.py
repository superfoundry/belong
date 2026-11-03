"""Personal agent (SPEC §3): works for one person, outside the room.

It drafts the listing from the person's details and choices, relays answers to
follow-up questions, and asks the person which hidden details they would reveal to
a connection. It only ever sees its own person's data.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import replace

from .schema import (AGE_BAND_BOUNDS, AGE_BANDS, AGE_MAX, AGE_MIN, CITY_COORDS, FIELD_REGISTRY, FORMATS, FREE_TEXT_MAX,
                     GENDERS, HIDDEN, MATCH_KINDS, SHOWN, TRAVEL, Field, Listing, age_span, only_said, plain_place, said)
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
from .matching.intent import (_KIND_PATTERNS, _MEETING, _ROMANTIC, _TASK_AFTER, HOBBY_END, _is_dating, _set_aside,
                              ages_text, dating_wants as _dating_wants, default_age_range, own_gender,
                              parse_intent as _parse_intent, read_listing, relative_years, request_kind,
                              without_hobbies)

ALL_MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
              "September", "October", "November", "December"]

LABELS = {
    "first_name": "First name", "city": "City", "country": "Country", "age_band": "Age band (from ID check)",
    "gender": "Gender",
    "interests": "Interests", "schedule": "Free time", "level": "Level", "pace": "Pace",
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
GENDER_NOTE = ("Please tell me your gender. It's needed so people looking for a woman or a man can find you. "
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
    absolute ("early 30s", "25 to 34", "no one over 40") or relative to their own age band ("around my
    age": one band either side of theirs). Belong. introduces adults only, so nobody under 18 is ever
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
        return None, "Belong. only introduces adults, so I haven't added an age range. Who would you like to meet?"
    note = None
    if lo < AGE_MIN:
        lo, note = AGE_MIN, f"Belong. only introduces adults, so the ages start at {AGE_MIN}."
    value = f"{lo}-{hi}"
    return (value, note) if age_span(value) else (None, None)


# --- Sensitive words ---------------------------------------------------------------------------
# Words that say something sensitive about a person: the sensitive details' own choices
# (_SENSITIVE_VOCAB), other ways of saying them, and identity words with no detail of their own
# (sexual orientation, gender identity, health conditions). They are an extra safeguard, not what keeps
# a person's words out of what is shown: the offline agent writes shown details only from words it
# recognises (below), and none of those is sensitive. They are also how a sensitive detail mentioned in
# a message is recorded, hidden (sensitive_details), and the model agent's save is refused when one is in
# a shown detail (agent_llm.build_draft). Words with an everyday meaning too ("a blind date", "easy
# recovery runs", "straight after work", "a bi-lingual exchange") count only where they say who someone
# is ("I'm bi", "I'm straight", "in recovery").
_SENSITIVE_OTHER = (
    # religion
    r"buddhism|christianity|catholics?|catholicism|protestants?|evangelicals?|mormons?|islam|islamic|jews?|judaism|"
    r"hinduism|sikhs?|sikhism|atheists?|atheism|agnostics?|religious|religion|faith|church|mosque|synagogue|bible|"
    r"quran|koran|torah|spiritual|prayer|praying|orthodox|baptists?|sunnis?|shias?|shiites?|ramadan|"
    # health
    r"diabetes|diabetic|celiac|coeliac|migraines?|deaf|disabled|disability|disabilities|wheelchair|autism|"
    r"autistic|epilepsy|epileptic|hiv|hsv|herpes|cancer|chemo\w*|depression|depressed|anxiety|bipolar|schizophreni\w*|"
    r"ptsd|ocd|illness|sober|sobriety|rehab|pregnant|pregnancy|asthma|asthmatic|dyslexi\w*|"
    # politics
    r"liberals?|left[- ]wing|right[- ]wing|leftists?|socialists?|socialism|communists?|communism|republicans?|"
    r"democrats?|libertarians?|feminists?|political|politics|activists?|"
    # intimacy
    r"kinks?|kinky|fetish\w*|role[- ]?play\w*|polyamor\w*|poly|voyeur\w*|swingers?|swinging|bondage|submissive|"
    r"dominant|"
    # sexual orientation and gender identity
    r"gay|lesbians?|bisexuals?|bi-?curious|queer|genderqueer|genderfluid|trans|transgender|transsexual|asexual|"
    r"demisexual|pansexual|homosexual|heterosexual|lgbt\w*|intersex|"
    # everyday words, only where they say who someone is
    r"(?:i['’]?m|i am|we['’]?re|we are|who['’]?s|who is|who are|as an?|being)\s+(?:a |an )?(?:bi|straight|blind)(?![\w-])|"
    r"(?:bi|straight|blind)\s+(?:man|men|woman|women|guys?|girls?|lady|ladies|person|people|couple)|"
    r"(?:legally|partially|completely) blind|in recovery|recovering (?:addicts?|alcoholics?)|"
    r"(?:addiction|alcohol|drug) recovery|recovery from (?:addiction|alcohol|drugs)|in therapy|in aa|aa meetings?")
_SENSITIVE_WORDS = _re.compile(r"\b(?:" + "|".join(
    [_re.escape(w.lower()) + "s?" for words in _SENSITIVE_VOCAB.values() for w in words] + [_SENSITIVE_OTHER])
    + r")\b")
# Other ways of saying one of a sensitive detail's choices, so it can be recorded (hidden) as that choice.
_SENSITIVE_SAME = {
    "religion": {"Christian": r"christianity|catholics?|catholicism|protestants?|evangelicals?", "Muslim": r"islam|islamic",
                 "Jewish": r"jews?|judaism", "Hindu": r"hinduism", "Buddhist": r"buddhism"},
    "kinks": {"polyamorous": r"polyamory", "roleplay": r"role[- ]play(?:ing)?"},
}


def mentions_sensitive(text) -> bool:
    """Does this text say something sensitive (see _SENSITIVE_WORDS)?"""
    return bool(_SENSITIVE_WORDS.search(" ".join(str(text or "").lower().split())))


def without_sensitive(text) -> str:
    """The text in lower case, with every sensitive word taken out."""
    return " ".join(_SENSITIVE_WORDS.sub(" ", " ".join(str(text or "").lower().split())).split())


def sensitive_details(low: str, kind) -> dict:
    """The sensitive details a message states plainly, as {key: value} in each detail's fixed choices
    (intimacy only for dating). These go on the listing as hidden details, never shown."""
    out = {}
    for key, words in _SENSITIVE_VOCAB.items():
        if key == "kinks" and kind != "dating":
            continue
        hit = next((w for w in words if _re.search(r"\b" + _re.escape(w.lower()) + r"s?\b", low)), None)
        hit = hit or next((value for value, rx in _SENSITIVE_SAME.get(key, {}).items()
                           if _re.search(rf"\b(?:{rx})\b", low)), None)
        if hit:
            value = hit if key != "politics" else next(
                (p for p in _vocab.POLITICS if hit.split("-")[0] in p), "politically progressive")
            out[key] = (value,) if FIELD_REGISTRY[key].multi else value
    return out


# --- What the offline agent recognises ---------------------------------------------------------
# The range of what people type is endless, so the offline agent never copies a person's own words into
# a detail that is shown. It writes shown details only from vocabulary it recognises: the kinds of match,
# the activities below, each detail's fixed choices, subjects, languages and known places (known_city).
# Everything else stays only in the description, which is never shown, and where it matters (which
# activity?) the agent asks instead. Every activity here is an ordinary pastime (a test checks none is
# sensitive).
_GAMES = {"tennis", "chess", "golf", "yoga", "football", "soccer", "basketball", "volleyball", "badminton", "squash",
          "pilates", "karate", "judo", "padel", "table tennis", "ping pong", "board games", "frisbee", "cricket",
          "rugby", "baseball", "darts", "poker", "tai chi", "kendo", "aikido", "bowling", "hockey", "billiards", "pool"}
KNOWN_ACTIVITIES = tuple(sorted(
    {*_GAMES, *_vocab.INTERESTS, *_vocab.SUBJECTS, "climbing", "rock climbing", "bouldering", "running", "trail running",
     "jogging", "walking", "swimming", "fishing", "photography", "painting", "drawing", "dancing", "singing", "guitar",
     "drums", "violin", "gardening", "reading", "writing", "baking", "skiing", "snowboarding", "skating", "sailing",
     "kayaking", "camping", "gaming", "gym", "boxing", "knitting"},
    key=lambda a: (-len(a), a)))  # longest first, so "table tennis" is read before "tennis"
_ACTIVITY = "|".join(map(_re.escape, KNOWN_ACTIVITIES))
_KNOWN_ACTIVITY = _re.compile(rf"\b({_ACTIVITY})\b")


def known_activities(text) -> list:
    """The known activities a text names, in order ("I enjoy chess, pottery and jazz" -> chess, jazz)."""
    return list(dict.fromkeys(m.group(1) for m in _KNOWN_ACTIVITY.finditer(" ".join(str(text or "").lower().split()))))


def known_activity(text):
    """The first known activity a text names, or None."""
    return next(iter(known_activities(text)), None)


# A partner for an activity, named: "a tennis partner", "a hiking buddy". Romantic words aren't an activity.
_PARTNER_PHRASE = _re.compile(r"\b(?:a|an|some)\s+((?:[a-z'-]+\s+){0,2}?[a-z'-]+)\s+(?:partner|buddy|companion)s?\b")
_NOT_AN_ACTIVITY = {"life", "long", "term", "long-term", "lifelong", "romantic", "serious", "loving", "committed",
                    "steady", "new", "nice", "good", "kind", "great", "potential", "future", "real", "special",
                    "perfect", "ideal", "business", "possible", "the", "my", "your", "our"}


def _partner_phrases(text) -> list:
    """What each "a ... partner" phrase names, unless it's romantic ("a long-term partner")."""
    low = " ".join(str(text or "").lower().split())
    return [m.group(1) for m in _PARTNER_PHRASE.finditer(low) if m.group(1).split()[-1] not in _NOT_AN_ACTIVITY]


def activity_partner(text: str):
    """The known activity of a partner they name ("a tennis partner" -> "tennis", "a Christian hiking
    buddy" -> "hiking"), or None. Only an activity the agent recognises counts: "a pottery partner" or "a
    Muslim partner" names none it knows, and nothing else from the phrase is ever used."""
    return next(filter(None, (known_activity(words) for words in _partner_phrases(text))), None)


def unknown_partner(text) -> bool:
    """Do they ask for a partner for something the agent doesn't recognise ("a pottery partner")? Then it
    asks which activity, instead of writing their words on the listing."""
    return any(not known_activity(words) and not _activities(words) for words in _partner_phrases(text))


# Activities someone says they do or enjoy, in the first person: "I play tennis", "I go hiking", "I do
# yoga", "I climb on weekends", "I enjoy hiking and chess". Only known activities count from these ("I love
# dogs" names none). Read only to ask a question, never to decide anything.
_I_DO = _re.compile(r"\b(?:i|we)\s+(?:could\s+|can\s+|might\s+|would\s+)?"
                    r"(?:also\s+|really\s+|often\s+|usually\s+|regularly\s+|sometimes\s+|mostly\s+)?"
                    r"(?:play|go|do|practi[cs]e|climb|boulder|run|jog)s?\b"
                    # up to where a new clause starts: "I play tennis | and want a girlfriend"
                    r"(?:(?!\s+(?:and|but|so)\s+(?:i\b|i'?m|i'?d|we\b|want|wanna|need|looking|would|am|are|hope|"
                    r"keen|trying))[^.,;!?])*")
_I_ENJOY = _re.compile(r"\b(?:i'?m into|i am into|i (?:really |also |just )?(?:enjoy|love|like|adore)|"
                       r"my hobbies (?:are|include)|i'?m (?:a )?(?:big )?fan of)\s+([a-z][a-z ,'\-]{2,80}?)"
                       r"(?=[.!?;]|$|\s(?:but|so|because)\b|\sand i\b|\s" + HOBBY_END + ")")
# How someone describes the person they'd like to meet: "a girlfriend who climbs", "someone that loves
# hiking", "a date with a climber", "ideally a runner". An activity there is that person's, not something
# they want to do together, so it is never a choice of activity.
_TRAITS = _re.compile(r"\b(?:who|that|whose)\b[^.,;!?]*"
                      r"|\bwith\s+(?:a|an|someone|somebody|people|men|women|guys|girls)\b[^.,;!?]*"
                      r"|\b(?:a|an)\s+(?:[a-z'-]+\s+)?(?:climbers?|boulderers?|runners?|joggers?|players?)\b")


def _first_person_activities(low: str) -> list:
    spans = [m.group(0) for m in _I_DO.finditer(low)] + [m.group(1) for m in _I_ENJOY.finditer(low)]
    return list(dict.fromkeys(a for span in spans for a in known_activities(span)))


def activities_said(text: str, first_person: bool = True, anywhere: bool = False) -> list:
    """The activities a text names, as (kind, activity) pairs, only in words the agent recognises: named
    kinds first ("climbing" -> ("climbing", None)), then a partner for a known activity ("a tennis partner"
    -> ("other", "tennis")), then, with first_person, known activities they say they do or enjoy ("I play
    chess" -> ("other", "chess")), then, with anywhere, any other known activity it names (an answer to
    "which activity?": "tennis"). Sensitive words are taken out first."""
    low = without_sensitive(text)
    out = [(k, None) for k in _activities(low)]
    named = [activity_partner(low)] + (_first_person_activities(low) if first_person else [])
    for activity in filter(None, named + (known_activities(low) if anywhere else [])):
        kind = next((k for k, pat in _KIND_PATTERNS if _re.search(pat, activity)), "other")
        pair = (kind, activity if kind == "other" else None)
        if pair not in out:
            out.append(pair)
    return out


def asked_kind(low: str):
    """The kind of match a message asks for, as the offline agent reads it: outside what they say they
    enjoy (request_kind) and, when they'd like to meet someone, outside how they describe that person ("a
    girlfriend who climbs" and "a date with a climber" are dating). None if it asks for no kind it knows."""
    purpose = without_hobbies(low)
    if _is_dating(purpose):
        if _ROMANTIC.search(_set_aside(low)):
            # Romantic wording leaves no doubt: what they say they do ("I climb a lot") is a hobby, exactly
            # as kind_question reads it, so the two never disagree.
            purpose = _I_DO.sub(" ", purpose)
        return request_kind(_TRAITS.sub(" ", purpose))
    return request_kind(low)


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
# or a climbing partner?); "a partner" with nothing to say whether it's romantic or for an activity; a
# partner for an activity it doesn't know ("a pottery partner": which activity?); or an activity only
# mentioned as something they enjoy, with nothing asking for it ("I love climbing": a climbing partner?).
# "Both" makes two listings, each with its own kind. Romantic words leave no doubt about wanting a date,
# so then an activity they only say they do or enjoy ("I love climbing", "I play tennis") is a hobby; one
# they want to do with someone ("a climbing partner") still asks. How they describe the person they'd
# like to meet ("a girlfriend who climbs") is never an activity to choose.
_ASK_NAMES = {"dating": "a date", "climbing": "a climbing partner", "running": "a running buddy",
              "language": "a language exchange", "cofounder": "a cofounder", "tutoring": "tutoring",
              "roommate": "a roommate"}
_PARTNER_ALONE = _re.compile(r"\b(?:a|an|my)\s+partner\b")
ASK_HINT = "Just say which, or \"both\"."
WHICH_ACTIVITY = "What would you like a partner for?"
# Said when an answer names no activity the agent knows: it asks again, and says why.
ACTIVITY_HINT = ("Name the activity in a word or two, e.g. tennis, hiking or board games. I only put activities I "
                 "know on your listing; anything else stays in your description, which is never shown.")


def _activities(text: str) -> list:
    return [k for k, pat in _KIND_PATTERNS if _re.search(pat, text)]


def _option(kind, activity) -> dict:
    label = _ASK_NAMES.get(kind) or f"{_article(activity)} {activity} partner"
    return {"kind": kind, "label": label, "activity": activity if kind == "other" else None}


def kind_question(message: str):
    """None when the kind of match is clear; otherwise {"text": one short question, "options": the
    kinds it could be, each {"kind", "label", "activity"}}. An option with kind None is a partner for
    an activity not named yet (or not one the agent knows). Sensitive words are read as if they weren't
    there."""
    low = without_sensitive(message)
    purpose = without_hobbies(low)  # without what they only say they enjoy
    if _is_dating(low):
        romantic = bool(_ROMANTIC.search(_set_aside(low)))
        if romantic:  # only an activity they ask for counts, never one they do, enjoy or want in the person
            found = activities_said(_I_DO.sub(" ", _TRAITS.sub(" ", purpose)), first_person=False)
        else:
            found = activities_said(_TRAITS.sub(" ", low))
        if found:
            other = _option(*found[0])
        elif not romantic and _PARTNER_ALONE.search(low) and not _TASK_AFTER.search(low) and not _MEETING.search(
                _PARTNER_ALONE.sub(" ", low)):
            other = {"kind": None, "label": "a partner for an activity", "activity": None}  # "a partner", nothing more
        else:
            return None
        options = [{"kind": "dating", "label": _ASK_NAMES["dating"], "activity": None}, other]
        which = " (which one)" if other["kind"] is None else ""
        return {"text": f"Are you looking for {options[0]['label']}, or {other['label']}{which}? Or both? I can make "
                        f"two separate listings.", "options": options}
    if asked_kind(low) is None and activity_partner(purpose) is None:
        # A partner for something it doesn't know ("a pottery partner", or one named only by a sensitive word,
        # "a chemo buddy"): which activity, in words it knows. Read on their own wording too, as the note is.
        if unknown_partner(purpose) or unknown_partner(without_hobbies(message.lower())):
            return {"text": f"{WHICH_ACTIVITY} {ACTIVITY_HINT}",
                    "options": [{"kind": None, "label": "a partner for an activity", "activity": None}]}
        # Nothing outside their hobbies asks for a kind, but a hobby names one: a guess, so ask.
        found = activities_said(low)
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
# Said with a choice, not instead of one: "climbing on weekends" still chooses climbing.
_WHEN = {"weekends", "weekend", "evenings", "evening", "mornings", "morning", "weekdays", "nights", "afternoons"}
_WANT_WORDS = {"want", "like", "looking", "prefer", "choose", "pick", "rather", "need"}
# A clause of an answer: "a date, I just like climbing" has two.
_CLAUSE = _re.compile(r"[,.;:!?]|\b(?:but|because|since|though|although|so|as)\b")
_PARTNER_AFTER = _re.compile(r"[a-z]*\s+(?:partners?|buddy|buddies|companions?|mates?|one|listing)\b")


def says_what_they_want(message: str) -> bool:
    """Does this message ask for a kind of match on its own (a new request, not an answer)? Only
    outside what it says they enjoy: "I just like climbing" asks for nothing."""
    low = without_sensitive(message)
    return asked_kind(low) is not None or activity_partner(without_hobbies(low)) is not None


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
    """Is this activity what the answer chooses: "a climbing partner", "the climbing one", or a clause
    that says nothing else ("climbing", "just climbing please", "climbing on weekends")? One they only
    mention ("I climb sometimes") isn't."""
    for m in _re.finditer(pattern, low):
        if not _NEGATED.search(low[:m.start()]) and _PARTNER_AFTER.match(low[m.end():]):
            return True
    for clause in _CLAUSE.split(_re.sub(_ROMANTIC.pattern, " ", low)):
        if not _chose(clause, pattern):
            continue
        # The activity, with a word that says which kind of it ("rock climbing", "trail running").
        rest = _re.findall(r"[a-z]+", _re.sub(rf"(?:(?!(?:i|we|im|you|not|no|never)\b)[a-z]+\s+)?(?:{pattern})[a-z]*",
                                              " ", clause))
        if any(w in ("i", "im", "we") for w in rest) and not any(w in _WANT_WORDS for w in rest):
            continue  # "I climb on weekends" is about them, not a choice ("I'd like climbing" is)
        if all(w in _CHOICE_FILLER or w in _WHEN for w in rest):
            return True
    return False


def chosen_kinds(answer: str, options: list):
    """What the person chose, in answer to kind_question: a list of {"kind", "activity"} (in the
    order asked), or None if the answer doesn't say. "Both" (or "yes" to a question with one option) is
    every option; a negated choice ("not a date") doesn't count, and nor does an activity they only
    mention ("a date, I just like climbing" is a date) or one that describes the person they'd like to
    meet ("a girlfriend who climbs" is a date). A partner for an activity not named yet takes the
    activity from the answer, only if it's one the agent knows ("tennis"; "pottery" isn't an answer, so it
    asks again); with "both" and no activity yet, its kind stays None, to be asked. Romantic words ("love",
    "a lover", "companionship") choose a date, even when only an activity was asked about."""
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
        elif o["kind"] is None:  # which activity: a known one they name now
            named = [(k, a) for k, a in activities_said(choice, anywhere=True)
                     if both or _named_as_choice(choice, dict(_KIND_PATTERNS).get(k) or _re.escape(a))]
            if named:
                out.append({"kind": named[0][0], "activity": named[0][1]})
            elif both:
                out.append({"kind": None, "activity": None})
        elif both or _named_as_choice(choice, _option_pattern(o)):
            out.append({"kind": o["kind"], "activity": o.get("activity")})
    if not out and dating:  # romance answers a question that didn't offer a date too
        out.append({"kind": "dating", "activity": None})
    if not out and any(o.get("open") for o in options):  # "or something else?": the kind the answer asks for
        kind, activity = asked_kind(low), activity_partner(choice)
        if kind or activity:
            out.append({"kind": kind or "other", "activity": None if kind else activity})
    return out or None


def _option_pattern(option: dict) -> str:
    """How an activity option is named in an answer: its kind's own words, or its label's."""
    words = [w for w in _re.findall(r"[a-z]+", option["label"]) if w not in ("a", "an", "partner", "buddy")]
    return dict(_KIND_PATTERNS).get(option["kind"]) or r"\b(?:" + "|".join(map(_re.escape, words)) + r")"


def answer_details(answer: str, options: list, chosen: list) -> str:
    """What an answer to kind_question says besides the choice, to add to the first message so nothing
    they say is lost: "a date, I'm a man, 30 to 40" -> "I'm a man, 30 to 40". The parts that only make
    the choice ("a date", "both please", "just running, not a date") add nothing: the chosen kind goes
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


# Diets that say something about religion: recorded when the person says so, but hidden until they choose
# to show them (the other diets are shown by default, like any detail).
_PRIVATE_DIETS = {"halal", "kosher"}


def draft_from_message(message: str, first_name: str, age_band: str, kind=None, activity=None) -> dict:
    """Read the person's own words and suggest a listing. Returns a plain dict the
    app shows for review; nothing is submitted until the person says so.

    kind: the kind of match the person chose when asked (kind_question), instead of the one their
    words suggest. activity: for "other", the activity they want a partner for ("tennis"); only a known
    activity is used.

    It never records a kind it isn't sure of: one read only from what they say they enjoy ("I love
    climbing", with nothing asking for it) is asked about instead. Every shown detail it writes comes from
    words it recognises (KNOWN_ACTIVITIES, the details' fixed choices, known places): nothing else from
    the message is copied into anything shown. A partner for an activity it doesn't know is asked about.
    Sensitive words become a hidden detail where there is one."""
    text = " ".join(message.split())[:500]
    low = text.lower()
    parsed = _parse_intent(text)
    asked = asked_kind(low) if kind is None else None
    guessed = kind is None and asked is None and parsed.kind is not None  # a kind from their hobbies alone: ask
    activity = known_activity(activity) if activity else None
    if activity is None and kind in (None, "other") and asked is None and not guessed:
        activity = activity_partner(without_hobbies(low))
    kind = kind or asked or ("other" if activity else None)
    if kind == "other" and not activity:
        kind = None  # a partner for an activity that isn't known: ask
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

    city, country = known_city(low)
    if city:
        v["city"] = city
        if country:
            v["country"] = country
    for pat, sched in [(r"weekend", "weekends"), (r"evening", "weekday evenings"),
                       (r"morning", "weekday mornings"), (r"flexible|whenever|any ?time", "flexible")]:
        if _re.search(pat, low):
            v["schedule"] = sched
            break
    # Interests: the usual ones, and the known activities they say they enjoy ("I enjoy rock climbing and tennis").
    interests = list(dict.fromkeys([i for i in known_activities(low) if i in _vocab.INTERESTS]
                                   + quick_facts(text).get("interests", [])))
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
    found = sensitive_details(low, kind)
    for key, value in found.items():
        v[key] = value
        enabled_sensitive.add(key)
        notes.append(f"You mentioned something sensitive ({LABELS[key].lower()}). I've added it as a "
                     f"hidden detail: it can help a match but is never shown. Your description is never "
                     f"shown either, but you may prefer to take it out of there.")
    if not found and mentions_sensitive(low):
        notes.append("You mentioned something sensitive. I've kept it out of everything shown on your listing. Your "
                     "description is never shown either, but you may prefer to take it out of there.")
    # An extra safeguard: nothing shown is ever built from a sensitive word (none is a known word anyway).
    for key in ("seeking", "offering", "headline", "topics", "interests"):
        value = v.get(key)
        if isinstance(value, (list, tuple)):
            value = [x for x in value if not mentions_sensitive(x)]
        if not value or (isinstance(value, str) and mentions_sensitive(value)):
            v.pop(key, None)
        else:
            v[key] = value

    exactness = 0.6
    if _re.search(r"\b(exactly|must|only|strictly)\b", low):
        exactness = 0.9
    elif _re.search(r"\b(open to|flexible|anyone|whatever|wide)\b", low):
        exactness = 0.3

    if guessed:
        notes.insert(0, f"Are you looking for {_option(parsed.kind, None)['label']}, or something else? I haven't set "
                        f"the kind of match yet: tell me in a few words what you'd like to meet someone for.")
    elif kind is None and unknown_partner(without_hobbies(low)):
        notes.insert(0, f"{WHICH_ACTIVITY} {ACTIVITY_HINT}")
    elif kind is None:
        notes.insert(0, "I couldn't tell what kind of match you want. Could you say a bit more about what you'd "
                        "like to meet someone for?")
    if not city and not parsed.remote_ok:
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
    options = field_options(kind, v.get("city"))
    fields = []
    for key in keys:
        spec = FIELD_REGISTRY[key]
        value = v.get(key)
        fields.append({
            "key": key, "label": LABELS[key], "value": list(value) if isinstance(value, (list, tuple)) else value,
            "visibility": HIDDEN if key == "diet" and value in _PRIVATE_DIETS else spec.default,
            "sensitive": spec.sensitive, "multi": spec.multi,
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
    # "listed": the details this agent chose shown or hidden for on purpose (only a diet it keeps hidden). Every
    # other visibility here is a default, so a choice the person already made stays (keep_visibility).
    return {"intent_text": text, "exactness": exactness, "kind": kind, "fields": fields,
            "understood": " ".join(understood), "notes": notes,
            "listed": ["diet"] if v.get("diet") in _PRIVATE_DIETS else []}


# --------------------------------------------------------------------------------------
# Instant profile updates (used by the demo while the agent is still replying).
#
# Picks out only facts stated plainly enough to be sure of ("I live in Osaka", "free on
# weekends", "I'm into chess and hiking"), and only in words it recognises: a place it knows, a
# fixed choice, a known activity. Anything else is left for the agent (and the person).
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


# The cities the agent knows: the demo's own and every city with a known centre (schema.CITY_COORDS), each
# with its country when it has only one ("Cambridge" is in two, so its country is left to the person).
_CITY_COUNTRIES: dict = {}
for _city, _where in CITY_COORDS:
    _CITY_COUNTRIES.setdefault(_city, set()).add(_country(_where) or _title(_where))
for _city, (_code, _) in _vocab.CITIES.items():
    _CITY_COUNTRIES.setdefault(plain_place(_city), set()).add(_vocab.COUNTRY_NAMES[_code])
_CITY_NAMES = {plain_place(c): c for c in _vocab.CITIES}  # the demo's own spelling
_KNOWN_CITY = _re.compile(r"\b(" + "|".join(map(_re.escape, sorted(_CITY_COUNTRIES, key=len, reverse=True))) + r")\b")


# Where someone is (the offline agent; the instant first draft only when there's no model agent). People can
# say where they are in endless ways, so this reads only a few explicit forms, all about the person, now, and
# leaves the place blank otherwise (the agent then asks "Where are you based?"):
#   "I live in Osaka", "I'm (now) based/living/staying in Osaka", "I'm in Osaka", "I (just) moved to Spain",
#   "Born in Kyoto, based in Berlin" (a clause starting "based/living in"), "Visiting Osaka",
#   and a request with a place, "Looking for a climbing partner in (or at, around, near) Osaka", in a sentence that mentions nobody
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


def _places_in(sentence: str) -> list:
    """(kind, name, where it starts) for the known cities and countries a sentence names."""
    found = [("city", m.group(1), m.start()) for m in _KNOWN_CITY.finditer(sentence)]
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
    # A request with a place: "Looking for a tennis partner in Osaka", nobody else and nothing past in it.
    return bool(_re.search(r"\b(?:in|at|around|near|nearby|across|within)\s+(?:the\s+)?$", before) and _REQUEST.search(sentence)
                and not _OTHERS.search(sentence) and not _PAST.search(sentence))


def where_they_are(text) -> tuple:
    """(city or None, country or None) for where the person is, only from the explicit forms above."""
    low = plain_place(" ".join(str(text or "").lower().split())) or ""
    cities, countries = [], []
    for sentence in _re.split(r"(?<=[.!?;])\s+|\n", low):
        for kind, name, at in _places_in(sentence):
            if not _counts(sentence, sentence[:at], kind):
                continue
            if kind == "city":
                cities.append(_CITY_NAMES.get(name) or _title(name))
            else:
                countries.append(_country(name) or _title(name))
    cities, countries = list(dict.fromkeys(cities)), list(dict.fromkeys(countries))
    if len(cities) > 1 or len(countries) > 1:
        return None, None  # two places: not sure which, so none (the agent asks)
    city = cities[0] if cities else None
    city_countries = _CITY_COUNTRIES.get(plain_place(city).lower(), set()) if city else set()
    country = next(iter(city_countries)) if len(city_countries) == 1 else None
    if countries:
        if country and countries[0] != country:
            return None, None  # a city in one country, another country named: not sure
        country = countries[0]
    return city, country


def known_city(text) -> tuple:
    """(city, country or None) where the person is, if the text names a known city for them; see where_they_are.
    A city is never read from anything else the person writes ("I'm in recovery" is not a place)."""
    city, country = where_they_are(text)
    return (city, country) if city else (None, None)


def quick_facts(text: str) -> dict:
    """Facts stated plainly in one message, as field -> value, only in words the agent recognises: a
    known city and its country, a country, the fixed choices (schedule, level, gender, languages, format,
    travel), an amount of pay, a known activity they teach or want to learn, and known activities they
    enjoy. Nothing else from the message is copied: "I live in Galway" or "I'm in therapy" sets no city,
    "I'm into pottery and BDSM" adds no interest, "I'm offering bible study lessons" no teaching phrase."""
    low = " ".join(text.lower().split())
    out = {}

    # Where they are: a known city and its country, or a country, only when said about them now (where_they_are).
    city, country = where_they_are(low)
    if city:
        out["city"] = city
    if country:
        out["country"] = country

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

    # Teaching a known activity, stated plainly: "I'm offering piano lessons", "I teach guitar" -> they're
    # the teacher; "looking for a beginner hockey coach" -> they're the student.
    m = _re.search(r"\b(?:i'?m |i am )?(?:offering|giving|teaching|i teach|i give|i offer|i can teach)\s+"
                   r"(?:(?:private|group|beginner|online) )?([a-z][a-z ]{1,30}?)\s*(?:lessons?|classes|coaching|tutoring)\b", low) \
        or _re.search(r"\bi (?:teach|coach|tutor)\s+([a-z][a-z ]{1,30}?)(?=[.!?;,]|$|\s(?:to|in|for|on|and)\b)", low)
    activity = known_activity(m.group(1)) if m else None
    if activity:
        out["offering"], out["seeking"] = f"{_article(activity)} {activity} teacher", f"{_article(activity)} {activity} student"
    m = _re.search(r"\b(?:looking for|need|want|find|searching for)\s+(?:a |an )?([a-z][a-z ]{1,30}?)\s+"
                   r"(teacher|tutor|coach|instructor|mentor)\b", low)
    activity = known_activity(m.group(1)) if m else None
    if activity and "offering" not in out:
        out["seeking"] = f"{_article(activity)} {activity} {m.group(2)}"
        out["offering"] = f"{_article(activity)} {activity} student"

    # Interests, only when introduced plainly, and only the known activities: "I'm into chess and hiking".
    m = _re.search(r"\b(?:i'm into|i am into|into|i enjoy|i love|who loves|who enjoys|my hobbies are|i'm interested in|i like)\s+"
                   r"([a-z][a-z ,'\-]{2,80}?)(?=[.!?;]|$)", low)
    if m and not _re.match(r"(?:to |meeting|meet |you|people)", m.group(1)):
        items = known_activities(m.group(1))
        if items:
            out["interests"] = items[:6]
    # An extra safeguard: shown details never carry a sensitive word (none is a known word anyway).
    for key in ("city", "interests", "offering", "seeking"):
        value = out.get(key)
        if isinstance(value, list):
            value = [x for x in value if not mentions_sensitive(x)]
        if not value or (isinstance(value, str) and mentions_sensitive(value)):
            out.pop(key, None)
        else:
            out[key] = value
    if ("offering" in out) != ("seeking" in out):  # the two phrases come as a pair
        out.pop("offering", None)
        out.pop("seeking", None)
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


def fill_gaps(draft: dict, facts: dict, visibilities=None) -> list:
    """Put facts into empty fields of a draft (never overwriting anything set). Returns the
    keys filled. Fields the draft doesn't have yet are added, except a format on a kind that has
    none (has_format): a climbing listing isn't made online only by "online works for me". The kind
    of match is the agent's to record, never carried over, and a detail only another kind has (carries)
    never goes on: who someone would like to meet only on a dating draft, a level only where it means one. visibilities: the shown or hidden each filled detail
    takes, where known (the person's choice, or the one it had on the draft it came from)."""
    filled = []
    by_key = {f["key"]: f for f in draft["fields"]}
    for key, value in facts.items():
        spec = FIELD_REGISTRY.get(key)
        if spec is None or spec.sensitive or key == "kind" or not carries(draft.get("kind"), key):
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
    never made shown here (showing one needs the person's own confirmation in the editor)."""
    chose = set(new.get("listed") or ())
    before = _prior_visibility(old, choices)
    for f in new.get("fields", []):
        was = before.get(f["key"])
        if was is None or f["key"] in chose or f.get("locked"):
            continue
        if was == HIDDEN or not f.get("sensitive"):
            f["visibility"] = was


def carry_over(new: dict, old, choices=None) -> list:
    """Keep what's already on the profile when a new agent draft replaces it: anything the new draft
    left empty (fill_gaps, never a sensitive detail or the kind, and after a change of kind only what's
    about the person), each with the shown or hidden it had,
    then the person's choices of shown or hidden for the rest (keep_visibility). Returns the keys filled."""
    if not old and not choices:
        return []
    # A change of kind is a new request: only what's about the person carries over (where they are, when they're
    # free, their interests), never the old request's own details (a level, a role, its phrases, its pay).
    changed = bool((old or {}).get("kind") and new.get("kind") and old.get("kind") != new.get("kind"))
    filled = fill_gaps(new, {f["key"]: f["value"] for f in (old or {}).get("fields", [])
                             if f.get("enabled") and f.get("value") not in (None, "", [])
                             and not f.get("sensitive") and f["key"] not in ("first_name", "age_band")
                             and not (changed and f["key"] in _KIND_SPECIFIC)},
                       _prior_visibility(old, choices))
    keep_visibility(new, old, choices)
    return filled


# The details that belong to a kind of match: its own (_BY_KIND), and for every kind except dating the
# phrases matching compares ("looking for", "you are") with the headline and topics. Dating uses who and
# what ages instead. Everything in _COMMON belongs to every kind.
_PHRASES = ("headline", "seeking", "offering", "topics")
_KIND_SPECIFIC = sorted({k for keys in _BY_KIND.values() for k in keys} | set(_PHRASES))
# Details whose choices differ from one kind to another (worked out from field_options, not listed by hand).
_PER_KIND = {k for k in FIELD_REGISTRY if len({tuple(field_options(kind, None).get(k) or ()) for kind in _BY_KIND}) > 1}


def kind_keys(kind) -> list:
    """The details a listing of this kind has, in order (with no kind yet, just the common ones)."""
    own = [k for k in _BY_KIND.get(kind, []) if k not in _COMMON]
    phrases = [] if kind in (None, "dating") else [k for k in _PHRASES if k not in own]
    return ["kind"] + _COMMON + own + phrases


def carries(kind, key) -> bool:
    """Can a listing of this kind have this detail? One rule for every way a draft is built or merged (an
    agent's save, details carried over, the instant draft, a save from the phone). Most details mean the same
    on any listing, so an agent sets them wherever they apply (a level, a native language). Who someone would
    like to meet and their ages are only for dating, and a detail whose choices belong to a kind (_PER_KIND:
    a role is tutor or student, or technical or business) is only for the kinds that have those choices."""
    if key in DATING_ONLY:
        return kind == "dating"
    if key in _PER_KIND:
        return bool(field_options(kind, None).get(key))
    return True


def kind_layout(kind, city=None) -> dict:
    """The profile's rows for a kind of match, for the editor when the person changes the kind: "rows"
    (each detail of this kind, empty, with its options and default visibility) and "drop" (details
    another kind has that this one doesn't: a climbing listing has no "Looking for" a woman or a man,
    and a dating listing has no climbing level)."""
    kind = kind if kind in MATCH_KINDS else None
    keys = kind_keys(kind)
    options = field_options(kind, city)
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
