"""Read a free-text intent into structure (OfflineMatcher only).

This is the rules-based stand-in for an LLM reading the intent. It looks for the kind
of match, who the person is looking for, and an age range. Anything it can't find is
left as None, which can trigger a fixed-menu follow-up question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schema import AGE_BAND_BOUNDS, said

# Checked in order: functional kinds first, so "climbing partner" isn't read as dating.
_KIND_PATTERNS = [
    ("tutoring", r"\btutor|\blessons?\b"),
    ("language", r"language exchange|tandem|natively|native \w+ speaker|conversation practice"),
    ("cofounder", r"co-?founder"),
    ("climbing", r"climb|boulder|belay"),
    ("running", r"\brun|\bjog|marathon"),
    ("roommate", r"roommate|flatmate|housemate"),
    ("dating", r"\bdate\b|dating|relationship|partner|romance|something casual"),
]

@dataclass(frozen=True)
class ParsedIntent:
    kind: Optional[str]
    # The text says online or remote is fine ("not online" doesn't). Only used when the listing
    # shows no format (narrow.meeting_modes); dating always meets in person, so it is never remote.
    remote_ok: bool = False
    seeking: tuple = ()  # dating: genders acceptable to this person
    age_range: Optional[tuple] = None  # dating: (lo_idx, hi_idx) into AGE_BANDS
    seeks_role: Optional[str] = None  # cofounder: technical/business; tutoring: tutor/student


def _age_range(text: str) -> Optional[tuple]:
    ages = [int(x) for x in re.findall(r"\b(1[89]|[2-9]\d)\b", text)]
    if not ages:
        return None
    lo_age = ages[0]
    hi_age = 99 if re.search(r"or older|\+", text) or len(ages) < 2 else ages[1]
    idx = [i for i, (lo, hi) in enumerate(AGE_BAND_BOUNDS) if hi >= lo_age and lo <= hi_age]
    return (idx[0], idx[-1]) if idx else None


def parse_intent(text: str) -> ParsedIntent:
    low = text.lower()
    kind = next((k for k, pat in _KIND_PATTERNS if re.search(pat, low)), None)
    # Any kind of listing may meet online except dating, which is in person only. "Not online" or
    # "no remote" says the opposite (schema.said).
    remote = said(low, r"online|remote") is True and kind != "dating"

    if kind == "dating":
        seeking = set()
        if re.search(r"\b(anyone|any gender)\b", low):
            seeking = {"woman", "man", "nonbinary"}
        if re.search(r"\b(a woman|women)\b", low):
            seeking.add("woman")
        if re.search(r"\b(a man|men)\b", low):
            seeking.add("man")
        return ParsedIntent(kind, seeking=tuple(sorted(seeking)), age_range=_age_range(low))

    seeks_role = None
    if kind == "cofounder":
        if re.search(r"(technical|engineering|cto)[\w-]*\s+co-?founder", low):
            seeks_role = "technical"
        elif re.search(r"(business|commercial|ceo|non-technical)[\w-]*\s+co-?founder", low):
            seeks_role = "business"
    elif kind == "tutoring":
        if re.search(r"\bi tutor\b|offering \w+ lessons|new students", low):
            seeks_role = "student"
        elif re.search(r"\w+ tutor\b|give me \w+ lessons", low):
            seeks_role = "tutor"
    return ParsedIntent(kind, remote_ok=remote, seeks_role=seeks_role)
