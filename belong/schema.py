"""Shared data model (SPEC §4).

Plain, immutable dataclasses. Every module may import this one; it has no behaviour
beyond small read helpers. Some of those helpers read a value the same way everywhere it
matters, so matching and the Warden can't disagree about it: a place name, how someone
will meet (in person or online), how far they'll travel, and the distance between two
cities (approximate city centres; SPEC §5 matching rules).
"""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional, Union

SHOWN = "shown"
HIDDEN = "hidden"

# A field value is a single string or a tuple of strings (multi-valued fields).
Value = Union[str, tuple]

AGE_BANDS = ["18-24", "25-29", "30-34", "35-39", "40-49", "50+"]
AGE_BAND_BOUNDS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 49), (50, 99)]


FREE_TEXT_FIELDS = ("first_name", "city", "country", "neighborhood", "employer", "subject", "seeking", "offering",
                    "pay", "format", "headline")
HEADLINE_MAX = 60
FREE_TEXT_MAX = 120


def age_band_index(age: int) -> int:
    for i, (lo, hi) in enumerate(AGE_BAND_BOUNDS):
        if lo <= age <= hi:
            return i
    raise ValueError(f"age out of range: {age}")


@dataclass(frozen=True)
class FieldSpec:
    sensitive: bool
    default: str  # SHOWN or HIDDEN
    multi: bool = False


# Every key a listing may carry. Sensitive categories start hidden and are left out
# of the listing until the person turns them on (SPEC §4).
FIELD_REGISTRY: dict[str, FieldSpec] = {
    "first_name": FieldSpec(False, SHOWN),
    "city": FieldSpec(False, SHOWN),
    "country": FieldSpec(False, SHOWN),
    "age_band": FieldSpec(False, SHOWN),
    "gender": FieldSpec(False, SHOWN),
    "interests": FieldSpec(False, SHOWN, multi=True),
    "schedule": FieldSpec(False, SHOWN),
    "level": FieldSpec(False, SHOWN),
    "pace": FieldSpec(False, SHOWN),
    "native_language": FieldSpec(False, SHOWN),
    "learning_language": FieldSpec(False, SHOWN),
    "role": FieldSpec(False, SHOWN),
    "subject": FieldSpec(False, SHOWN),
    "move_in": FieldSpec(False, SHOWN),
    # Open-ended matches ("something else"): short phrases written with the person.
    "seeking": FieldSpec(False, SHOWN),   # what they're looking for, e.g. "a guitarist"
    "offering": FieldSpec(False, SHOWN),  # what they bring, e.g. "a ska band"
    # The listing in plain words, for people: "Selling 100 cardboard boxes", "Looking for free
    # cardboard boxes". The phrases above are for matching; this is what titles and questions say.
    "headline": FieldSpec(False, SHOWN),
    # A few broader areas the listing is about, written by the person's agent, so related
    # requests can be recognised without a built-in list of activities.
    "topics": FieldSpec(False, SHOWN, multi=True),
    # Paid or practical requests: shown so the other side knows the terms up front.
    "pay": FieldSpec(False, SHOWN),       # e.g. "¥4,000 per hour" (offered or charged), a rent budget
    "format": FieldSpec(False, SHOWN),    # "In person", "Online", "In person or online" (FORMATS)
    "travel": FieldSpec(False, SHOWN),    # how far they'll go to meet in person: one of TRAVEL
    "diet": FieldSpec(False, SHOWN),
    "smoking": FieldSpec(False, SHOWN),
    "children": FieldSpec(False, HIDDEN),
    "neighborhood": FieldSpec(False, HIDDEN),
    "employer": FieldSpec(False, HIDDEN),
    "income_band": FieldSpec(True, HIDDEN),
    "religion": FieldSpec(True, HIDDEN),
    "politics": FieldSpec(True, HIDDEN),
    "health": FieldSpec(True, HIDDEN),
    "kinks": FieldSpec(True, HIDDEN, multi=True),
}

# How two people can meet (the "format" field). Any wording is read into one of these by meeting_format().
FORMATS = ("In person", "Online", "In person or online")

# How far someone will go to meet in person (the "travel" field): fixed choices, in km. Someone who
# doesn't say stays in their own city.
TRAVEL = {"Same city": 0, "Up to 30 km": 30, "Up to 100 km": 100}
TRAVEL_DEFAULT = "Same city"
MAX_TRAVEL_KM = max(TRAVEL.values())

# Approximate city centres (latitude, longitude), by plain city name and plain country (see plain_place):
# a city's centre is known only in its own country, so a "Cambridge" shown with another country, or with
# none, has no known centre and is only ever "the same city" as itself. Data only: nothing here is about
# a person.
CITY_COORDS = {
    **{(city, "japan"): at for city, at in {
        "osaka": (34.694, 135.502), "kyoto": (35.012, 135.768), "kobe": (34.690, 135.196),
        "nara": (34.685, 135.805), "tokyo": (35.676, 139.650), "yokohama": (35.444, 139.638),
        "kawasaki": (35.531, 139.703), "saitama": (35.862, 139.646), "nagoya": (35.182, 136.907)}.items()},
    **{(city, "germany"): at for city, at in {
        "berlin": (52.520, 13.405), "potsdam": (52.391, 13.065), "hamburg": (53.551, 9.994),
        "munich": (48.135, 11.582), "munchen": (48.135, 11.582), "frankfurt": (50.111, 8.682),
        "cologne": (50.938, 6.960), "koln": (50.938, 6.960)}.items()},
    **{(city, "united kingdom"): at for city, at in {
        "london": (51.507, -0.128), "oxford": (51.752, -1.258), "cambridge": (52.205, 0.122),
        "brighton": (50.823, -0.137), "manchester": (53.481, -2.243), "liverpool": (53.408, -2.992)}.items()},
    **{(city, "portugal"): at for city, at in {
        "lisbon": (38.722, -9.139), "lisboa": (38.722, -9.139), "sintra": (38.803, -9.382),
        "cascais": (38.698, -9.422), "porto": (41.158, -8.629), "braga": (41.545, -8.427)}.items()},
    # Another Cambridge, an ocean away from the one above.
    **{(city, "united states"): at for city, at in {
        "cambridge": (42.373, -71.119), "boston": (42.360, -71.058)}.items()},
}
EARTH_RADIUS_KM = 6371.0


def plain_place(value) -> Optional[str]:
    """A place name compared ignoring case, spacing and accents ("Málaga" == "malaga ")."""
    if not value:
        return None
    decomposed = "".join(c for c in unicodedata.normalize("NFKD", str(value)) if not unicodedata.combining(c))
    return " ".join(decomposed.lower().split()) or None


def city_centre(city, country) -> Optional[tuple]:
    """A city's approximate centre, read in the country given with it (by plain names), or None if
    that city isn't known in that country, or no country is given."""
    return CITY_COORDS.get((plain_place(city), plain_place(country)))


def km_between(city_a, country_a, city_b, country_b) -> Optional[float]:
    """Great-circle distance between two cities, each read in its own country: 0 for the same city
    (by plain name) in the same country, None if either country is missing or either city's centre
    isn't known in its country."""
    a, b = plain_place(city_a), plain_place(city_b)
    ca, cb = plain_place(country_a), plain_place(country_b)
    if not (a and b and ca and cb):
        return None
    if (a, ca) == (b, cb):
        return 0.0
    pa, pb = CITY_COORDS.get((a, ca)), CITY_COORDS.get((b, cb))
    if pa is None or pb is None:
        return None
    lat1, lon1, lat2, lon2 = (math.radians(x) for x in (*pa, *pb))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


# Reading how someone will meet from their own words. said() tells a plain mention ("online is fine")
# from a negated one ("not online", "no remote", "online doesn't work for me"); only_said() spots "only
# online", "just online" and "online only". "Not only online" / "not just online" mean "also online".
ONLINE_WORDS = r"on-?line|remote(?:ly)?|video(?: calls?)?|virtual(?:ly)?|zoom"
IN_PERSON_WORDS = r"in[- ]?person|face[- ]to[- ]face|offline|meet(?:ing)?[- ]?ups?|locally"
_NEGATED_BEFORE = re.compile(
    r"(?:\b(?:not|no|never|without|cannot)|n't)"
    # a few words may stand between the negation and the thing: "don't want to meet online"
    r"(?:\s+(?:to|be|do|doing|meet|meeting|want|wanting|interested|in|on|for|keen|available|able|really|any|"
    r"a|an|the|via|over|by|through|go|going|possible|with|comfortable|happy|work|working)){0,3}\s+$")
# "online isn't for me" is a no; "online isn't a problem" / "doesn't bother me" / "wouldn't hurt" is a yes.
_NEGATED_AFTER = re.compile(
    r"(?:(?:\s+(?:is|are|does|do|would|will|really))?\s+(?:not|never)\b|\s+\w*n't\b)"
    r"(?!\s+(?:be\s+)?(?:a\s+(?:problem|big\s+deal|concern|dealbreaker|deal\s+breaker)|an\s+(?:issue|obstacle)"
    r"|matter|bother|hurt|mind|bad)\b)")
_EITHER = re.compile(r"\b(either|both|hybrid)\b")
_NOT_ONLY = re.compile(r"\bnot\s+(?:only|just)\b|\bnot\s+\S+[- ]only\b")


def _plain_words(text, words: str) -> str:
    """Lower case, straight apostrophes, and "not only X" / "not just X" / "not X-only" read as "also X"."""
    low = " ".join(str(text).lower().replace("\u2019", "'").split())
    low = re.sub(r"\bnot\s+(?:only|just)\b", "also", low)
    return re.sub(r"\bnot\s+(" + words + r")[- ]only\b", r"also \1", low)


def said(text, words: str) -> Optional[bool]:
    """Does the text say this (`words`, a regex)? True if at least once without a negation, False if
    only negated ("not online", "without video calls", "online isn't for me"), None if not at all."""
    if not isinstance(text, str):
        return None
    low = _plain_words(text, words)
    found = None
    for m in re.finditer(r"\b(?:" + words + r")\b", low):
        if _NEGATED_BEFORE.search(low[:m.start()]) or _NEGATED_AFTER.match(low, m.end()):
            found = False
        else:
            return True
    return found


def only_said(text, words: str) -> bool:
    """Does the text say *only* this: "only online", "just online", "online only"?"""
    if not isinstance(text, str):
        return False
    low = _plain_words(text, words)
    return bool(re.search(r"\b(?:only|just)\s+(?:" + words + r")\b|\b(?:" + words + r")[- ]only\b", low))


def meeting_format(value) -> Optional[str]:
    """A format in any wording, as one of FORMATS ("online only" -> "Online", "in-person" ->
    "In person", "either" -> "In person or online"), or None if it says neither. A negated online
    word ("not online") or "only" with an in-person word ("in person only") means in person only; a
    negated in-person word ("not in person") means online only; "not only online" means either."""
    if not isinstance(value, str):
        return None
    online, in_person = said(value, ONLINE_WORDS), said(value, IN_PERSON_WORDS)
    if online is False or only_said(value, IN_PERSON_WORDS):
        return "In person"
    if in_person is False or only_said(value, ONLINE_WORDS):
        return "Online"
    if _EITHER.search(value.lower()) or (online and in_person) or _NOT_ONLY.search(value.lower()):
        return "In person or online"
    return "Online" if online else "In person" if in_person else None


def travel_choice(value) -> Optional[str]:
    """A travel distance as one of TRAVEL ("up to 30km" -> "Up to 30 km"), or None if it isn't one."""
    if not isinstance(value, str):
        return None
    key = re.sub(r"[^a-z0-9]", "", value.lower())
    key = {"30km": "upto30km", "100km": "upto100km"}.get(key, key)
    return next((c for c in TRAVEL if re.sub(r"[^a-z0-9]", "", c.lower()) == key), None)


# The only follow-up questions the Matchmaker may ask (SPEC §3, §7). It refers to
# them by id; the Warden attaches the text at exit.
FOLLOWUP_MENU: dict[str, str] = {
    "age_range": "What age range are you open to?",
    "schedule": "When are you usually free?",
    # The one question that needs no opt-in, because it is the opt-in: asked once, to a listing with
    # close matches off and no match yet. A yes turns on close matches and follow-up questions.
    "broaden": "No matches yet. Want to include close matches?",
    # {phrase} is filled in by the Warden, never the Matchmaker: the shown headline of a listing
    # whose person opted in to close matches ("selling 100 cardboard boxes"), or "looking for"
    # plus its shown phrase if it has no headline. It never says who.
    "also_offer": "Someone is {phrase}. Would that work for you?",
}


@dataclass(frozen=True)
class Person:
    id: str
    verified: bool
    age_band: str
    created_at: str


@dataclass(frozen=True)
class Field:
    key: str
    value: Value
    visibility: str  # SHOWN or HIDDEN
    sensitive: bool

    def strings(self) -> list[str]:
        return value_strings(self.value)


@dataclass(frozen=True)
class Listing:
    # The listing's own id in matching. A person may have several listings (say, one looking
    # for a tutor and one looking for love); each is matched on its own. For a person's only
    # listing this is simply their person id.
    person_id: str
    intent_text: str
    exactness: float  # 1 = exact, 0 = wide net
    fields: tuple  # tuple[Field, ...]
    updated_at: str
    version: int = 1
    # Sensitive keys the person explicitly chose to show (overriding the default).
    confirmed_shown: tuple = ()
    # The person who owns this listing ("" = the person whose id is person_id).
    owner_id: str = ""
    # The person's notification choices for this listing, both off unless they opt in:
    # close matches (one-way: the other person isn't told unless this person says yes)
    # and questions from Belong. (fixed-menu follow-ups). The Warden enforces both.
    notify_close: bool = False
    notify_questions: bool = False
    # A paused listing stays on the board but isn't matched: no new introductions or questions.
    active: bool = True

    @property
    def owner(self) -> str:
        return self.owner_id or self.person_id

    def get(self, key: str) -> Optional[Field]:
        for f in self.fields:
            if f.key == key:
                return f
        return None

    def shown(self) -> dict:
        return {f.key: f.value for f in self.fields if f.visibility == SHOWN}

    def hidden(self) -> dict:
        return {f.key: f.value for f in self.fields if f.visibility == HIDDEN}


def value_strings(value: Value) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def value_to_json(value: Value):
    return value if isinstance(value, str) else list(value)
