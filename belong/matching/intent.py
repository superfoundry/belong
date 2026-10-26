"""Read what a listing asks for (OfflineMatcher only; SPEC §5, §8.3).

What the person's agent understood comes first: the listing's SHOWN "kind" of match and, for
dating, the shown "wants_gender" and "wants_age" (read_listing). Only what those don't say is read
from the free text, by the rules below: the listing's description and, after it, its shown headline
(the matcher passes shown fields only). They look for the kind of match, who the person is looking
for, and an age range. A hidden field is never read here, so it is as if it weren't there. Anything
still unknown is left as None, which can trigger a fixed-menu follow-up question.

The rules are a fallback, not understanding: the range of what people type is endless. They read
a hobby the person mentions ("I enjoy climbing") as a hobby, not as the kind of match, whenever
something else in the text asks for anything, before or after it ("I enjoy climbing and I'm looking
to meet a woman"). Ages said with "no" or "not" ("no one over 40") mean the opposite of the plain
wording, and numbers that measure or count something ("30 km", "2 kids") are never ages.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schema import AGE_BAND_BOUNDS, AGE_BANDS, GENDERS, MATCH_KINDS, age_bands_for, age_span, said

# Checked in order: functional kinds first, so "climbing partner" isn't read as dating.
_KIND_PATTERNS = [
    ("tutoring", r"\btutor|\blessons?\b"),
    ("language", r"language exchange|tandem|natively|native \w+ speaker|conversation practice"),
    ("cofounder", r"co-?founder"),
    ("climbing", r"climb|boulder|belay"),
    ("running", r"\brun|\bjog|marathon"),
    ("roommate", r"roommate|flatmate|housemate"),
]

# Where someone starts saying what they want ("... and I'm looking to meet a woman", "want a running
# buddy", "I'd love to meet"). A hobby ends there, and text that still has one of these after the
# hobbies are taken out is a request, so the kind is never read from the hobbies instead.
_REQUEST_WORDS = (r"\b(?:looking|seeking|searching|hoping|want(?:s|ed|ing)?|wish(?:ing)?|need(?:s|ing)?|"
                  r"would (?:like|love)|(?:i|we)['’]?d (?:like|love)|in search of|interested in meeting)\b")
REQUEST = re.compile(_REQUEST_WORDS)
_UNTIL_REQUEST = rf"(?:(?!{_REQUEST_WORDS})[^.!?;])*"
# Something the person says they enjoy ("I enjoy rock climbing and tennis", "who loves hiking", "my
# hobbies are..."), up to the end of the sentence or where a request starts. A hobby is not what they
# want to meet someone for, so the kind is read without these first, and from them only if nothing
# else in the text asks for anything.
_HOBBY = re.compile(
    r"\b(?:i|we|who|she|he|they)\s+(?:really\s+|also\s+|both\s+|just\s+|simply\s+)?(?:enjoy|love|like|adore)s?\b"
    + _UNTIL_REQUEST
    + r"|\b(?:i'?m|i am|who'?s|who is|we'?re|also)\s+(?:really\s+|also\s+|big\s+|just\s+)?(?:into|a fan of|passionate about)\b"
    + _UNTIL_REQUEST
    + r"|\bmy (?:hobbies|interests|passions) (?:are|include)\b" + _UNTIL_REQUEST + r"|\bhobbies\b:?" + _UNTIL_REQUEST)

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


def _set_aside(low: str) -> str:
    """The text without calendar dates and people the person already has."""
    return _ALREADY_HAVE.sub(" ", _CALENDAR.sub(" ", low))


def _is_dating(low: str) -> bool:
    text = _set_aside(low)
    if _TRADE.search(text):
        return False
    if _ROMANTIC.search(text):
        return True
    return bool(_MEETING.search(text)) and not _TASK_AFTER.search(text) and not _WORK.search(text)


def _named_kind(text: str) -> Optional[str]:
    """The first functional kind the text names, else dating if it reads as dating, else None."""
    kind = next((k for k, pat in _KIND_PATTERNS if re.search(pat, text)), None)
    if kind is None and _is_dating(text):
        kind = "dating"
    return kind


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
# Their own age: "I'm 34", "I'm a 34 year old ...", "as a 34-year-old", "in my early 30s".
_SELF_AGE = re.compile(rf"\b(?:i'?m|i am|im|as)\s+(?:only\s+|just\s+)?(?:an?\s+)?{_N}(?:[- ]?(?:years?[- ]old|yo|y/o))?\b"
                       r"|\bin my (?:early |mid |late )?(?:[2-9]0['’]?s|(?:twent|thirt|fort|fift|sixt)ies)\b")
_RANGE = re.compile(rf"\b{_N}\s*(?:-|–|to|and)\s*{_N}\b")
# Said with "no", "not" or "at most": the opposite of the plain wording. "No one over 40" is 40 or
# younger, "not younger than 30" is 30 or older. Read before the plain wording below.
_NOBODY = r"(?:no(?:body| one|-one)?|not|never)\s+(?:anyone\s+|anybody\s+|someone\s+|people\s+|one\s+)?"
_AT_MOST = re.compile(rf"\b(?:{_NOBODY}(?:over|older than|above)|at most|max(?:imum)?|no more than)\s+{_N}\b")
_AT_LEAST = re.compile(rf"\b(?:{_NOBODY}(?:under|younger than|below)|min(?:imum)?|no less than)\s+{_N}\b")
_OLDER = re.compile(rf"\b{_N}\s*(?:\+|or older|or over|and older|and over|and up)|\b(?:over|older than|at least)\s+{_N}\b")
_YOUNGER = re.compile(rf"\b(under|younger than|below|up to)\s+{_N}\b|\b{_N}\s*(?:or younger|or under|and under)\b")
# A decade of someone's life, not of music or film ("80s music", "90s films").
_DECADE = re.compile(r"\b(?:(early|mid|late)[\s-]*)?(?:([2-9])0(?:['’]?s|[\s-]*something)|(twent|thirt|fort|fift|sixt)ies)\b"
                     r"(?!\s*(?:music|films?|movies?|songs?|style|fashion|hip[- ]?hop|rock|pop|tv|television|shows?|"
                     r"anime|vibes?|kids?|cars?|bands?|disco|aesthetic))")
# Numbers that aren't ages: a measured or counted thing after them ("30 km", "20 minutes", "10 years
# of experience", "2 kids", "25 people"), or money ("$30"). "30 years old" is still an age.
_UNITS = (r"(?:years?|yrs?)\b(?!\s*(?:old|of age|young))|km\b|kilomet\w*|miles?\b|mi\b|minutes?\b|mins?\b|hours?\b|"
          r"hrs?\b|days?\b|weeks?\b|months?\b|cm\b|kg\b|lbs?\b|%|percent\b|kids?\b|children\b|people\b|persons?\b|"
          r"times\b|euros?\b|dollars?\b|yen\b|pounds?\b|k\b|floors?\b|stops?\b|stations?\b|members?\b|games?\b")
_NOT_AGES = re.compile(rf"[£€$¥]\s?\d[\d,.]*|\b\d+(?:\s*(?:-|–|to|and)\s*\d+)?\s*(?:{_UNITS})")
_AROUND = re.compile(rf"\b(?:around|about|roughly|approximately|approx\.?|circa|close to)\s+{_N}\b|\b{_N}\s*-?\s*ish\b")
_DECADE_WORDS = {"twent": 2, "thirt": 3, "fort": 4, "fift": 5, "sixt": 6}
_PART_OF_DECADE = {"early": (0, 3), "mid": (4, 6), "late": (7, 9), None: (0, 9)}
AROUND_YEARS = 3  # "around 30" means 27 to 33
# Ages said relative to their own: around their age is one age band either side of theirs; a bit
# younger is the band below and theirs; a bit older is theirs and the band above.
_MY_AGE = re.compile(r"\b(?:my|our) (?:own )?age\b|\bsimilar(?:ly)?[- ]aged?\b|\bsame age\b|\bmy age group\b")
_YOUNGER_THAN_ME = re.compile(r"\byounger\b")
_OLDER_THAN_ME = re.compile(r"\bolder\b")
_NEGATED_BEFORE = re.compile(r"(?:\b(?:not|no|never|nobody|without)|n['’]t)(?:\s+[a-z'’-]+){0,2}\s*$")


def _decade(m) -> tuple:
    tens = 10 * (int(m.group(2)) if m.group(2) else _DECADE_WORDS[m.group(3)])
    lo, hi = _PART_OF_DECADE[m.group(1)]
    return tens + lo, tens + hi


def _ages_said(text: str, plain_numbers: bool) -> list:
    """(lo, hi, how) for every way an age range is written: a range ("25 to 34", "early 30s", "in their
    thirties", "30-something", "around 30"; how "range"), a lower bound ("30 or older", "not younger
    than 30"; "min") or an upper bound ("under 40", "no one over 40", "at most 35"; "max"). With
    plain_numbers, bare numbers count too: two are a range, one is around that age."""
    spans = []

    def take(rx, span, how):
        nonlocal text
        text = rx.sub(lambda m: spans.append((*span(m), how)) or " ", text)

    take(_RANGE, lambda m: (int(m.group(1)), int(m.group(2))), "range")
    take(_AT_MOST, lambda m: (18, int(m.group(1))), "max")
    take(_AT_LEAST, lambda m: (int(m.group(1)), 99), "min")
    take(_OLDER, lambda m: (int(m.group(1) or m.group(2)), 99), "min")
    take(_YOUNGER, lambda m: (18, int(m.group(2) or m.group(3)) - (1 if m.group(1) in ("under", "younger than", "below")
                                                                    else 0)), "max")
    take(_DECADE, _decade, "range")
    take(_AROUND, lambda m: (int(m.group(1) or m.group(2)) - AROUND_YEARS, int(m.group(1) or m.group(2)) + AROUND_YEARS),
         "range")
    if not spans and plain_numbers:
        ages = [int(x) for x in re.findall(rf"\b{_N}\b", text)]
        if len(ages) >= 2:
            spans.append((ages[0], ages[1], "range"))
        elif ages:
            spans.append((ages[0] - AROUND_YEARS, ages[0] + AROUND_YEARS, "range"))
    return spans


def _combined(spans: list) -> Optional[tuple]:
    """One age range from everything said: the ranges together ("late 20s to early 30s"), within every
    bound ("30s, no one over 35" is 30 to 35). None if the bounds contradict each other."""
    ranges = [(lo, hi) for lo, hi, how in spans if how == "range"]
    lo = min(r[0] for r in ranges) if ranges else 0
    hi = max(r[1] for r in ranges) if ranges else 99
    lo = max([lo] + [l for l, _, how in spans if how == "min"])
    hi = min([hi] + [h for _, h, how in spans if how == "max"])
    lo = lo if ranges or any(how == "min" for *_, how in spans) else 18
    return (lo, hi) if lo <= hi else None  # as said: below 18 is never a band


def _years(wanted: str, whole: str = "", guess: bool = True) -> Optional[tuple]:
    """The ages, in years, someone asks for: read where they say what they want, or failing that
    anywhere (but then only written as an age range, so "Single woman, 34, ..." isn't read as one).
    Their own age ("I'm 34", "in my early 30s") and measured things ("30 km") never count. Without
    guess, a bare number where they say what they want isn't read as an age either."""
    clean = lambda t: _NOT_AGES.sub(" ", _SELF_AGE.sub(" ", t))
    spans = _ages_said(clean(wanted), plain_numbers=guess)
    if not spans and whole:
        spans = _ages_said(clean(whole), plain_numbers=False)
    return _combined(spans) if spans else None


def _relative_years(wanted: str, own_band) -> Optional[tuple]:
    """Ages said relative to the person's own age band ("around my age", "a bit younger"), in years,
    or None if they say none or their age band isn't known."""
    if own_band not in AGE_BANDS:
        return None
    i, last = AGE_BANDS.index(own_band), len(AGE_BANDS) - 1
    # "Not older than me" / "no one younger" say the opposite of "older" / "younger".
    said = lambda rx: [bool(_NEGATED_BEFORE.search(wanted[:m.start()])) for m in rx.finditer(wanted)]
    younger, older = said(_YOUNGER_THAN_ME), said(_OLDER_THAN_ME)
    bands = []
    if _MY_AGE.search(wanted):
        bands += [max(0, i - 1), min(last, i + 1)]
    if (False in younger) or (True in older):
        bands += [max(0, i - 1), i]
    if (False in older) or (True in younger):
        bands += [i, min(last, i + 1)]
    if not bands:
        return None
    return AGE_BAND_BOUNDS[min(bands)][0], AGE_BAND_BOUNDS[max(bands)][1]


def _bands(lo_age: int, hi_age: int) -> Optional[tuple]:
    return age_bands_for(lo_age, hi_age)


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


def _plain(text) -> str:
    return " ".join(str(text or "").lower().split())


def _wanted_part(desc: str, head: str) -> tuple:
    """(where they say what they want, the whole text), without what they say about themselves."""
    about_them = lambda t: _ALREADY_HAVE.sub(" ", _SELF_GENDER.sub(" ", t))
    desc, head = about_them(desc), about_them(head)
    start = _WANTS.search(desc)
    wanted = (desc[start.start():] if start else desc) + (f". {head}" if head else "")
    return wanted, (f"{desc}. {head}" if head else desc)


def dating_wants(text: str, headline: str = "", guess: bool = True) -> tuple:
    """(genders wanted, ages wanted in years or None), read from a dating description and its shown
    headline, whatever kind the words alone would suggest. Without guess, a bare number ("a woman,
    30") isn't read as around that age: a person's agent asks instead (agent.wanted_ages)."""
    wanted, whole = _wanted_part(_plain(text), _plain(headline))
    seeking = set(GENDERS) if _ANYONE.search(wanted) else {g for g, rx in _SEEKS.items() if rx.search(wanted)}
    return tuple(sorted(seeking)), _years(wanted, whole, guess)


def relative_years(text: str, own_band, headline: str = "") -> Optional[tuple]:
    """Ages a dating description asks for relative to the person's own age band ("around my age", "a
    bit younger"), in years; None if it says none or the band isn't known."""
    wanted, _ = _wanted_part(_plain(text), _plain(headline))
    return _relative_years(_SELF_AGE.sub(" ", wanted), own_band)


def _seeks_role(low: str, kind) -> Optional[str]:
    if kind == "cofounder":
        if re.search(r"(technical|engineering|cto)[\w-]*\s+co-?founder", low):
            return "technical"
        if re.search(r"(business|commercial|ceo|non-technical)[\w-]*\s+co-?founder", low):
            return "business"
    elif kind == "tutoring":
        if re.search(r"\bi tutor\b|offering \w+ lessons|new students", low):
            return "student"
        if re.search(r"\w+ tutor\b|give me \w+ lessons", low):
            return "tutor"
    return None


def _reading(low: str, kind, desc: str, head: str) -> ParsedIntent:
    """The structure of a listing of this kind, read from its text."""
    if kind == "dating":
        seeking, years = dating_wants(desc, head)
        return ParsedIntent(kind, seeking=seeking, age_range=_bands(*years) if years else None)
    # Any kind of listing may meet online except dating, which is in person only. "Not online" or
    # "no remote" says the opposite (schema.said).
    remote = said(low, r"online|remote") is True
    return ParsedIntent(kind, remote_ok=remote, seeks_role=_seeks_role(low, kind))


_PARSED: dict = {}  # (description, headline) -> ParsedIntent, so each listing is read once


def parse_intent(text: str, headline: str = "") -> ParsedIntent:
    """Read the description alone (and the shown `headline`, read after it), by the rules above."""
    key = (text, headline or "")
    found = _PARSED.get(key)
    if found is None:
        if len(_PARSED) > 100_000:
            _PARSED.clear()
        found = _PARSED[key] = _parse(text, headline or "")
    return found


def request_kind(low: str) -> Optional[str]:
    """The kind of match the text asks for outside what it says the person enjoys, or None."""
    return _named_kind(_HOBBY.sub(" ", low))


def _parse(text: str, headline: str) -> ParsedIntent:
    desc, head = _plain(text), _plain(headline)
    low = f"{desc}. {head}" if head else desc
    # What they want, before what they enjoy: "meet a woman ... I enjoy climbing" is dating. Only a text
    # that asks for nothing else is read from its hobbies ("I enjoy climbing and I'm looking for someone
    # kind" asks for something, just not a climbing partner).
    kind = request_kind(low)
    if kind is None and not REQUEST.search(_HOBBY.sub(" ", low)):
        kind = _named_kind(low)
    return _reading(low, kind, desc, head)


_READ: dict = {}  # (description, shown headline, kind, wants_gender, wants_age, age band) -> ParsedIntent


def read_listing(text: str, shown) -> ParsedIntent:
    """What a listing asks for, as matching uses it. `shown(key)` returns one of the listing's SHOWN
    fields (narrow.ShownView.get), never a hidden one.

      kind     the shown "kind" if it is one of MATCH_KINDS, else read from the text;
      dating   genders: the shown "wants_gender" if valid, else read from the text;
               ages: the shown "wants_age" (years, onto the age bands it overlaps), else read from
               the text, absolute ("early 30s") or relative to their shown age band ("around my
               age"); still None if neither says (matching then uses default_age_range).

    The text's activity words never change a kind the listing states."""
    head, kind_f, genders, ages, band = (shown(k) for k in ("headline", "kind", "wants_gender", "wants_age",
                                                            "age_band"))
    key = (text, head, kind_f, genders, ages, band)
    try:
        found = _READ.get(key)
    except TypeError:  # a value that can't be a key: work it out each time
        return _read(text, head, kind_f, genders, ages, band)
    if found is None:
        if len(_READ) > 100_000:
            _READ.clear()
        found = _READ[key] = _read(text, head, kind_f, genders, ages, band)
    return found


def _read(text, head, kind_f, genders, ages, band) -> ParsedIntent:
    headline = head if isinstance(head, str) else ""
    wanted = tuple(sorted(set(genders))) if isinstance(genders, tuple) and genders and set(genders) <= set(GENDERS) \
        else None
    span = age_span(ages)
    if kind_f not in MATCH_KINDS:
        base = parse_intent(text, headline)  # no kind stated: the description says what it can
    elif kind_f == "dating" and wanted and span:
        return ParsedIntent("dating", seeking=wanted, age_range=_bands(*span))  # all stated: nothing to read
    else:
        desc, hd = _plain(text), _plain(headline)
        base = _reading(f"{desc}. {hd}" if hd else desc, kind_f, desc, hd)  # read as the kind it states
    if base.kind != "dating":
        return base
    age_range = _bands(*span) if span else base.age_range
    if age_range is None:
        years = relative_years(text, band, headline)
        age_range = _bands(*years) if years else None
    return ParsedIntent("dating", seeking=wanted or base.seeking, age_range=age_range)
