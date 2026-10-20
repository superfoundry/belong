"""Shared data model (SPEC §4).

Plain, immutable dataclasses. Every module may import this one; it has no behaviour
beyond small read helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

SHOWN = "shown"
HIDDEN = "hidden"

# A field value is a single string or a tuple of strings (multi-valued fields).
Value = Union[str, tuple]

AGE_BANDS = ["18-24", "25-29", "30-34", "35-39", "40-49", "50+"]
AGE_BAND_BOUNDS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 49), (50, 99)]


FREE_TEXT_FIELDS = ("first_name", "city", "country", "neighborhood", "employer", "subject", "seeking", "offering",
                    "pay", "format")
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
    # Paid or practical requests: shown so the other side knows the terms up front.
    "pay": FieldSpec(False, SHOWN),       # e.g. "¥4,000 per hour" (offered or charged), a rent budget
    "format": FieldSpec(False, SHOWN),    # "In person", "Online", "In person or online"
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

# The only follow-up questions the Matchmaker may ask (SPEC §3, §7). It refers to
# them by id; the Warden attaches the text at exit.
FOLLOWUP_MENU: dict[str, str] = {
    "age_range": "What age range are you open to?",
    "schedule": "When are you usually free?",
    "wider_net": "We haven't found a match yet. Should we cast a wider net?",
    # {phrase} is filled in by the Warden, never the Matchmaker: it is the shown "looking for"
    # phrase of a listing whose person opted in to close matches. It never says who.
    "also_offer": "Someone is looking for {phrase}. Could you help with that too?",
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
