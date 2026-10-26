"""Read a free-text intent into structure (OfflineMatcher only; SPEC §5, §8.3).

This is the rules-based stand-in for an LLM reading the intent. It reads the listing's
description and, after it, the listing's shown headline (the matcher passes shown fields
only), and looks for the kind of match, who the person is looking for, and an age range.
Anything it can't find is left as None, which can trigger a fixed-menu follow-up question.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schema import AGE_BAND_BOUNDS, AGE_BANDS, said

# Checked in order: functional kinds first, so "climbing partner" isn't read as dating.
_KIND_PATTERNS = [
    ("tutoring", r"\btutor|\blessons?\b"),
    ("language", r"language exchange|tandem|natively|native \w+ speaker|conversation practice"),
    ("cofounder", r"co-?founder"),
    ("climbing", r"climb|boulder|belay"),
    ("running", r"\brun|\bjog|marathon"),
    ("roommate", r"roommate|flatmate|housemate"),
]

GENDERS = ("man", "nonbinary", "woman")  # the gender field's values

# --- Dating ---------------------------------------------------------------------------
# Said outright ("a date", "a relationship", "a girlfriend", "looking for love"), or as wanting to
# meet a man, a woman, a guy, a partner. That second way is also how people ask for a service ("a
# man with a van", "a woman to clean my flat"), so it only counts as meeting someone: a meeting word
# before it, nothing to be done after it, and no work or trade anywhere in the request. Selling,
# buying, giving away, hiring or paying is never dating, however it's worded.

# A calendar date, and someone the person already has ("my wife", "my sister's husband"), aren't
# what they're looking for. Read before anything else and set aside.
_CALENDAR = re.compile(r"\b(?:move[- ]?in|start(?:ing)?|end|due|closing|delivery|release|expiry|expiration|birth|"
                       r"up to|out of|later|earlier|exact)\s+dates?\b|\bdates? of\b")
_ALREADY_HAVE = re.compile(
    r"\b(?:(?:my|our|his|her|their|your)\s+(?:(?!(?:future|next|new|dream|perfect|ideal|forever|true|life|lifelong|"
    r"long[- ]?term|special)\b)[a-z'-]+\s+)?|[a-z]+['’]s\s+)"
    r"(?:ex[- ]?)?(?:boyfriends?|girlfriends?|husbands?|wife|wives|partners?|spouses?|fianc[eé]e?s?|ex)\b")
_ROMANTIC = re.compile(
    r"\bdat(?:e|es|ing)\b|\brelationships?\b|\bromance\b|\bromantic|\bsomething casual\b|\bsomeone special\b"
    r"|\b(?:boy|girl)friends?\b|\bhusband\b|\bwife\b|\bspouse\b|\bsoul ?mates?\b|\bfianc[eé]e?\b"
    r"|\b(?:life|long[- ]?term|lifelong|romantic|serious|loving|committed|steady)\s+partners?\b"
    r"|\b(?:find|finding|looking for|seeking|searching for|fall in|be in) love\b|\blove life\b|\bmarry\b|\bmarriage\b"
    r"|\b(?:i'?m|i am) single\b|\bsingle (?:man|men|woman|women|guys?|girls?|lady|ladies|people|person)\b|\bflirt")
_MEETING_WORDS = (r"meet|meeting|find|finding|looking for|look for|seeking|seek|searching for|search for|in search of|"
                  r"want|wanting|would like|hoping for|interested in|into|attracted to")
_GENDERED = r"(?:man|men|woman|women|guys?|girls?|lady|ladies|gentlem[ae]n)\b(?!['’]s\b)"
# "meet a man", "find a nice guy", "interested in women", "looking for a partner" (but "a tennis partner" is
# a tennis partner).
_MEETING = re.compile(rf"\b(?:{_MEETING_WORDS})\s+(?:[a-z'-]+\s+){{0,2}}?{_GENDERED}"
                      rf"|\b(?:{_MEETING_WORDS})\s+(?:a|an)\s+partner\b")
# Something to be done, right after who: "a man to fix my sink", "women to join our book club",
# "women for our football team". Time together is not a task ("a man to share my life with", "a woman
# for a drink").
_TOGETHER = (r"date|meet|love|marry|share|settle|spend|grow old|get to know|be with|see|cuddle|kiss|hold|start|"
             r"build a (?:life|future|family)|have (?:fun|kids|children|a family)|laugh|talk|chat|go out|"
             r"go on dates?|hang out")
_OCCASIONS = r"drinks?|coffee|dinner|meal|walk|chat|night out|good time|fling|long walk|weekend away"
_TASK_AFTER = re.compile(rf"\b(?:{_GENDERED}|partner\b)(?:\s+(?:in|near|around|from)\s+[a-z]+)?\s+"
                         rf"(?:to\s+(?!(?:{_TOGETHER})\b)[a-z]+|for\s+(?:a|an|my|our|the|some)\s+(?!(?:{_OCCASIONS})\b)[a-z]+)")
# Work and trades, anywhere in the request.
_WORK = re.compile(r"\b(?:fix|fixing|repair\w*|mend|mending|install\w*|assembl\w*|clean|cleaning|cleaners?|babysit\w*|"
                   r"nann(?:y|ies)|childcare|removals?|vans?|trucks?|lorry|deliver\w*|plumb\w*|electricians?|"
                   r"handym[ae]n|decorators?|painters?|gardeners?|carpenters?|builders?|mechanics?|couriers?)\b")
_TRADE = re.compile(r"\b(?:sell|sells|selling|sold|for sale|buy|buys|buying|give away|gives away|giving away|for free|"
                    r"free to (?:a good home|anyone|collect)|for rent|to rent|renting out|to let|hire|hiring|for hire|"
                    r"hourly|per (?:hour|session|lesson|day|week|month|night)|coach|coaching|counsel+(?:or|ing)|"
                    r"therapist|therapy|workshop|seminar)\b|[£€$¥]\s?\d")


def _is_dating(low: str) -> bool:
    text = _ALREADY_HAVE.sub(" ", _CALENDAR.sub(" ", low))
    if _TRADE.search(text):
        return False
    if _ROMANTIC.search(text):
        return True
    return bool(_MEETING.search(text)) and not _TASK_AFTER.search(text) and not _WORK.search(text)


# What a person says about themselves ("I'm a woman", "as a 34 year old gay man", "I'm non-binary") is
# their own gender, never who they're looking for.
_SELF_QUALIFIERS = (r"(?:\d{2}(?:[- ]?(?:year[- ]old|yo|y/o))?|straight|gay|bi|bisexual|queer|lesbian|trans|"
                    r"transgender|cis|single|young|divorced|widowed)")
_SELF_GENDER = re.compile(
    rf"\b(?:i'?m|i am|as)\s+(?:{_SELF_QUALIFIERS}\s+)*an?\s+(?:{_SELF_QUALIFIERS}\s+)*(woman|man|guy|girl|lady)\b"
    rf"|\b(?:i'?m|i am)\s+(?:an?\s+)?(?:{_SELF_QUALIFIERS}\s+)*(female|male|non-?binary|enby)\b")
_GENDER_WORDS = {"woman": "woman", "girl": "woman", "lady": "woman", "female": "woman",
                 "man": "man", "guy": "man", "male": "man", "nonbinary": "nonbinary", "non-binary": "nonbinary",
                 "enby": "nonbinary"}
_SEEKS = {
    "woman": re.compile(r"\b(?:woman|women|girls?|lady|ladies|girlfriends?|wife|wives|females?)\b(?!['’]s\b)"),
    "man": re.compile(r"\b(?:man|men|guys?|gentlem[ae]n|boyfriends?|husbands?|males?)\b(?!['’]s\b)"),
    "nonbinary": re.compile(r"\bnon-?binary\b|\benby\b"),
}
_ANYONE = re.compile(r"\b(?:anyone|anybody|any gender|all genders|of any gender|regardless of gender)\b")
# Who they want, and the ages, are read from where they start saying what they want ("... looking
# for a man in his 30s"), so "Single woman, 34, ..." before it describes them, not who they want.
_WANTS = re.compile(rf"\b(?:{_MEETING_WORDS}|hoping|date|dating)\b")


def own_gender(text: str) -> Optional[str]:
    """The person's own gender, when they state it plainly ("I'm a woman", "I am non-binary")."""
    m = _SELF_GENDER.search(" ".join(text.lower().split()))
    return _GENDER_WORDS[m.group(1) or m.group(2)] if m else None


# --- Ages ----------------------------------------------------------------------------------
_N = r"(1[89]|[2-9]\d)"
_SELF_AGE = re.compile(rf"\b(?:i'?m|i am|im)\s+(?:only\s+|just\s+)?{_N}\b|\bin my (?:early |mid |late )?"
                       r"(?:[2-9]0['’]?s|(?:twent|thirt|fort|fift|sixt)ies)\b")
_RANGE = re.compile(rf"\b{_N}\s*(?:-|–|to|and)\s*{_N}\b")
_OLDER = re.compile(rf"\b{_N}\s*(?:\+|or older|or over|and older|and over|and up)|\b(?:over|older than|at least)\s+{_N}\b")
_YOUNGER = re.compile(rf"\b(under|younger than|below|up to)\s+{_N}\b|\b{_N}\s*(?:or younger|or under|and under)\b")
_DECADE = re.compile(r"\b(?:(early|mid|late)[\s-]*)?(?:([2-9])0(?:['’]?s|[\s-]*something)|(twent|thirt|fort|fift|sixt)ies)\b")
_AROUND = re.compile(rf"\b(?:around|about|roughly|approximately|approx\.?|circa|close to)\s+{_N}\b|\b{_N}\s*-?\s*ish\b")
_DECADE_WORDS = {"twent": 2, "thirt": 3, "fort": 4, "fift": 5, "sixt": 6}
_PART_OF_DECADE = {"early": (0, 3), "mid": (4, 6), "late": (7, 9), None: (0, 9)}
AROUND_YEARS = 3  # "around 30" means 27 to 33


def _decade(m) -> tuple:
    tens = 10 * (int(m.group(2)) if m.group(2) else _DECADE_WORDS[m.group(3)])
    lo, hi = _PART_OF_DECADE[m.group(1)]
    return tens + lo, tens + hi


def _ages_said(text: str, plain_numbers: bool) -> list:
    """(lo, hi) ages for every way an age range is written: "25 to 34", "30 or older", "under 40",
    "early 30s", "in their thirties", "30-something", "around 30". With plain_numbers, bare numbers
    count too: two are a range, one is around that age."""
    spans = []

    def take(rx, span):
        nonlocal text
        text = rx.sub(lambda m: spans.append(span(m)) or " ", text)

    take(_RANGE, lambda m: (int(m.group(1)), int(m.group(2))))
    take(_OLDER, lambda m: (int(m.group(1) or m.group(2)), 99))
    take(_YOUNGER, lambda m: (18, int(m.group(2) or m.group(3)) - (1 if m.group(1) in ("under", "younger than", "below")
                                                                    else 0)))
    take(_DECADE, _decade)
    take(_AROUND, lambda m: (int(m.group(1) or m.group(2)) - AROUND_YEARS, int(m.group(1) or m.group(2)) + AROUND_YEARS))
    if not spans and plain_numbers:
        ages = [int(x) for x in re.findall(rf"\b{_N}\b", text)]
        if len(ages) >= 2:
            spans.append((ages[0], ages[1]))
        elif ages:
            spans.append((ages[0] - AROUND_YEARS, ages[0] + AROUND_YEARS))
    return spans


def _bands(lo_age: int, hi_age: int) -> Optional[tuple]:
    idx = [i for i, (lo, hi) in enumerate(AGE_BAND_BOUNDS) if hi >= lo_age and lo <= hi_age]
    return (idx[0], idx[-1]) if idx else None


def _age_range(wanted: str, whole: str = "") -> Optional[tuple]:
    """The age bands someone asks for: read where they say what they want, or failing that anywhere
    (but then only written as an age range, so "Single woman, 34, ..." isn't read as one). Their own
    age ("I'm 34", "in my early 30s") never counts."""
    spans = _ages_said(_SELF_AGE.sub(" ", wanted), plain_numbers=True)
    if not spans and whole:
        spans = _ages_said(_SELF_AGE.sub(" ", whole), plain_numbers=False)
    if not spans:
        return None
    return _bands(min(lo for lo, _ in spans), max(hi for _, hi in spans))


def default_age_range(age_band) -> tuple:
    """Ages for a dating listing that names none. Something you didn't say is not a strict preference
    (as with a climbing level), so you're open to one age band either side of your own shown band, or
    to any adult band if you don't show one."""
    if age_band in AGE_BANDS:
        i = AGE_BANDS.index(age_band)
        return max(0, i - 1), min(len(AGE_BANDS) - 1, i + 1)
    return 0, len(AGE_BANDS) - 1


def ages_text(age_range: tuple) -> str:
    """Plain words for a range of age bands: "25 to 39", "40 or older"."""
    lo, hi = AGE_BAND_BOUNDS[age_range[0]][0], AGE_BAND_BOUNDS[age_range[1]][1]
    return f"{lo} or older" if age_range[1] == len(AGE_BANDS) - 1 else f"{lo} to {hi}"


@dataclass(frozen=True)
class ParsedIntent:
    kind: Optional[str]
    # The text says online or remote is fine ("not online" doesn't). Only used when the listing
    # shows no format (narrow.meeting_modes); dating always meets in person, so it is never remote.
    remote_ok: bool = False
    seeking: tuple = ()  # dating: genders acceptable to this person
    age_range: Optional[tuple] = None  # dating: (lo_idx, hi_idx) into AGE_BANDS, None if not said
    seeks_role: Optional[str] = None  # cofounder: technical/business; tutoring: tutor/student


_PARSED: dict = {}  # (description, headline) -> ParsedIntent, so each listing is read once


def parse_intent(text: str, headline: str = "") -> ParsedIntent:
    """`headline`: the listing's shown headline, read after the description."""
    key = (text, headline or "")
    found = _PARSED.get(key)
    if found is None:
        if len(_PARSED) > 100_000:
            _PARSED.clear()
        found = _PARSED[key] = _parse(text, headline or "")
    return found


def _parse(text: str, headline: str) -> ParsedIntent:
    desc, head = " ".join(text.lower().split()), " ".join(headline.lower().split())
    low = f"{desc}. {head}" if head else desc
    kind = next((k for k, pat in _KIND_PATTERNS if re.search(pat, low)), None)
    if kind is None and _is_dating(low):
        kind = "dating"
    # Any kind of listing may meet online except dating, which is in person only. "Not online" or
    # "no remote" says the opposite (schema.said).
    remote = said(low, r"online|remote") is True and kind != "dating"

    if kind == "dating":
        about_them = lambda t: _ALREADY_HAVE.sub(" ", _SELF_GENDER.sub(" ", t))
        desc, head = about_them(desc), about_them(head)
        start = _WANTS.search(desc)
        wanted = (desc[start.start():] if start else desc) + (f". {head}" if head else "")
        seeking = set(GENDERS) if _ANYONE.search(wanted) else {g for g, rx in _SEEKS.items() if rx.search(wanted)}
        return ParsedIntent(kind, seeking=tuple(sorted(seeking)),
                            age_range=_age_range(wanted, f"{desc}. {head}" if head else desc))

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
