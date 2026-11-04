"""Read what a listing asks for (OfflineMatcher only; SPEC §5, §8.3).

What the person's agent understood comes first: the listing's SHOWN "kind" of match and, for
dating, the shown "wants_gender" and "wants_age" (read_listing). Only what those don't say is read
from the free text, by the rules below: the listing's description and, after it, its shown headline
(the matcher passes shown fields only). They look for the kind of match, who the person is looking
for, and an age range. A hidden field is never read here, so it is as if it weren't there. Anything
still unknown is left as None, which can trigger a fixed-menu follow-up question.

The kinds of match are dating and "other" (schema.MATCH_KINDS), and the text is read only to tell
dating from the rest: a listing it can't read as dating stays None, and its "looking for" / "you are"
phrases decide (matcher.understood). No activity is named here.

The rules are a fallback, not understanding: the range of what people type is endless. They read
a hobby the person mentions ("I enjoy pottery") as a hobby, not as what they want, whenever something
else in the text asks for anything, before or after it ("I enjoy pottery and I'm looking to meet a
woman"), and a hobby ends where they start saying they hope to meet someone or name who ("I enjoy
reading and hope to meet a girlfriend"). Ages said with "no" or "not" ("no one over 40", "no
one under 30 or over 45") mean the opposite of the plain wording, numbers that measure or count something
("30 km", "2 kids") are never ages, and a bare number ("a woman, 30") is never read as an age range.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

from ..schema import AGE_BAND_BOUNDS, AGE_BANDS, GENDERS, MATCH_KINDS, age_bands_for, age_span, said

# Where someone starts saying what they want ("... and I'm looking to meet a woman", "want a sailing
# buddy", "I'd love to meet", "hope to meet", "I'd really like"). A hobby ends there, and text that still
# has one of these after the hobbies are taken out is a request, so the kind is never read from the
# hobbies instead.
_REQUEST_WORDS = (r"\b(?:looking|seeking|searching|hoping|want(?:s|ed|ing)?|wish(?:ing)?|need(?:s|ing)?|"
                  r"would (?:really |also )?(?:like|love)|(?:i|we)['’]?d (?:really |also |so )?(?:like|love)|"
                  r"in search of|interested in meeting|open to meeting|hopes? to|keen to|eager to|trying to (?:meet|find))\b")
REQUEST = re.compile(_REQUEST_WORDS)
# Who someone hopes to meet, named romantically ("a girlfriend", "someone special", "a long-term partner"):
# a hobby ends there too, so a romantic request is never read as only part of a hobby ("I enjoy croquet and
# a girlfriend would be lovely").
_ROMANTIC_WHO = (r"\b(?:(?:boy|girl)friends?|husband|wife|spouse|soul ?mates?|fianc[eé]e?|someone special|"
                 r"(?:life|long[- ]?term|lifelong|romantic|serious|loving|committed|steady)\s+partners?|relationships?)\b")
HOBBY_END = rf"(?:{_REQUEST_WORDS}|{_ROMANTIC_WHO})"
_UNTIL_REQUEST = rf"(?:(?!{HOBBY_END})[^.!?;])*"
# Something the person says they enjoy ("I enjoy pottery and birdwatching", "who loves baking", "my
# hobbies are..."), up to the end of the sentence or where a request or a romantic "who" starts. A hobby
# is not what they want to meet someone for, so the kind is read without these first, and from them only
# if nothing else in the text asks for anything.
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
# "Dating" describing a noun after it ("a dating app", "my dating profile", "dating coaches", "a dating
# relationship"): only the word "dating" is set aside, and the noun is read for what it means by itself, like any
# other words. A romantic noun ("a relationship") is still dating; a thing or a service ("an app", "a profile",
# "coaches") says nothing romantic. No list of nouns either way. "Dating" before a person ("a dating partner",
# "dating women", "dating someone") asks for that person romantically, so it stays; so does "dating" as a verb or
# on its own, before a grammar word or an -ly word ("dating again", "dating in Leeds", "dating seriously"). Read
# structurally, a verb straight after a sentence's first "Dating" ("Dating feels ...") or another kind of adverb
# ("dating long-term") looks like a noun it describes, so it is set aside: that costs a question, never a match.
_PERSON_NOUN = r"(?:partners?|(?:man|men|woman|women|guys?|girls?|lady|ladies|gentlem[ae]n|people|person)|" \
               r"some(?:one|body)|any(?:one|body))"
_GRAMMAR_WORD = (r"(?:a|an|the|this|that|these|those|my|our|your|his|her|their|some|any|no|in|on|at|for|with|without|"
                 r"near|around|from|to|of|by|about|after|before|again|and|or|but|nor|so|is|was|are|were|be|been|has|"
                 r"have|had|can|could|will|would|should|may|might|must|who|which|when|while|if|as|than)")
# "Dating" as a verb whose object is a person with describing words first ("dating older men", "dating tall, kind
# women") also stays: a person noun after up to three words that aren't grammar words (joined by spaces, commas, "and"
# or "or"), or a describing word made from a verb right after it ("dating divorced ..."), says who is dated.
_DESCRIBED_PERSON = rf"(?:(?!{_GRAMMAR_WORD}\b)[a-z'-]+(?:\s*,\s*|\s+(?:and|or)\s+|\s+)){{1,3}}{_PERSON_NOUN}\b"
_DATING_THING = re.compile(rf"\bdating\s+(?=[a-z])(?!(?:{_PERSON_NOUN}|{_GRAMMAR_WORD})\b)(?![a-z]+ly\b)"
                           rf"(?![a-z][a-z'-]*[a-z]ed\b)(?!{_DESCRIBED_PERSON})")
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
# "meet a man", "find a nice guy", "interested in women", "looking for a partner" (but "a croquet partner" is
# a croquet partner).
_MEETING = re.compile(rf"\b(?:{_MEETING_WORDS})\s+(?:[a-z'-]+\s+){{0,2}}?{_GENDERED}"
                      rf"|\b(?:{_MEETING_WORDS})\s+(?:a|an)\s+partner\b")
# Something to be done, right after who (or after where they are): "a man to fix my sink", "women to join
# our book club", "women for our football team", "a man in Porto Alegre to proofread my thesis". After a
# "with a ..." that says what they have, only a purpose said with "to" is a task ("a woman with a car to
# drive me"): a "for ..." there belongs to the noun before it ("a man with a fondness for the mountains"
# describes him). Time together is not a task ("a man to share my life with", "a woman for a drink").
_TOGETHER = (r"date|meet|love|marry|share|settle|spend|grow old|get to know|be with|see|cuddle|kiss|hold|start|"
             r"build a (?:life|future|family)|have (?:fun|kids|children|a family)|laugh|talk|chat|go out|"
             r"go on dates?|hang out")
_OCCASIONS = r"drinks?|coffee|dinner|meal|walk|chat|night out|good time|fling|long walk|weekend away"
_WHO_SOUGHT = rf"\b(?:{_GENDERED}|partner\b)"
_TO_TASK = rf"to\s+(?!(?:{_TOGETHER})\b)[a-z]+"
_TASK_AFTER = re.compile(
    rf"{_WHO_SOUGHT}(?:\s+(?:in|near|around|from)(?:\s+[a-z]+){{1,2}}?)?\s+"
    rf"(?:{_TO_TASK}|for\s+(?:a|an|my|our|the|some)\s+(?!(?:{_OCCASIONS})\b)[a-z]+)"
    rf"|{_WHO_SOUGHT}\s+with\s+(?:a|an)(?:\s+[a-z]+){{1,2}}?\s+{_TO_TASK}")
# Work to be done, anywhere in the request: general verbs only, never a list of trades (what the work is
# for comes from the listing's shown kind, which is read first).
_WORK = re.compile(r"\b(?:fix|fixing|repair\w*|mend|mending|install\w*|assembl\w*|clean|cleaning|deliver\w*)\b")
# Selling, buying, giving away, renting, hiring or a price: never dating, whatever is traded.
_TRADE = re.compile(r"\b(?:sell|sells|selling|sold|for sale|buy|buys|buying|give away|gives away|giving away|for free|"
                    r"free to (?:a good home|anyone|collect)|for rent|to rent|renting out|to let|hire|hiring|for hire|"
                    r"hourly|per (?:hour|session|lesson|day|week|month|night))\b|[£€$¥]\s?\d")


def _set_aside(low: str) -> str:
    """The text without calendar dates, the word "dating" where it describes a noun ("a dating app" is read
    as "an app", "a dating relationship" as "a relationship") and people the person already has."""
    return _ALREADY_HAVE.sub(" ", _DATING_THING.sub("", _CALENDAR.sub(" ", low)))


def _is_dating(low: str) -> bool:
    text = _set_aside(low)
    if _TRADE.search(text):
        return False
    if _ROMANTIC.search(text):
        return True
    return bool(_MEETING.search(text)) and not _TASK_AFTER.search(text) and not _WORK.search(text)


def _dating_or_none(text: str) -> Optional[str]:
    """"dating" if the text reads as dating, else None: no other kind is read from words."""
    return "dating" if _is_dating(text) else None


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
_NOBODY = r"(?:no(?:body| one|-one)?|not|never|neither)\s+(?:anyone\s+|anybody\s+|someone\s+|people\s+|one\s+)?"
_UNDER, _OVER = r"(?:under|younger than|below)", r"(?:over|older than|above)"
# Both bounds under one "no": "no one under 30 or over 45", "nobody over 45 or younger than 30", "neither
# under 30 nor over 45" are 30 to 45 (read before the single bounds, so the second isn't read plainly).
_JOINED = r"\s*,?\s*(?:or|and|nor)\s+(?:anyone\s+|anybody\s+|someone\s+|people\s+)?"
_NOT_OUTSIDE = re.compile(rf"\b{_NOBODY}(?:{_UNDER}\s+{_N}{_JOINED}{_OVER}\s+{_N}|{_OVER}\s+{_N}{_JOINED}{_UNDER}\s+{_N})\b")
_AT_MOST = re.compile(rf"\b(?:{_NOBODY}{_OVER}|at most|max(?:imum)?|no more than)\s+{_N}\b")
_AT_LEAST = re.compile(rf"\b(?:{_NOBODY}{_UNDER}|min(?:imum)?|no less than)\s+{_N}\b")
_OLDER = re.compile(rf"\b{_N}\s*(?:\+|or older|or over|and older|and over|and up)|\b(?:over|older than|at least)\s+{_N}\b")
_YOUNGER = re.compile(rf"\b(under|younger than|below|up to)\s+{_N}\b|\b{_N}\s*(?:or younger|or under|and under)\b")
_ROUGHLY = r"(?:around|about|roughly|approximately|approx\.?|circa|close to)"
# A decade of someone's life ("early 30s", "thirties", "30-something", "thirty-something") is an age only where
# the sentence makes it one: after "in his / her / their / your / our" or "aged"; as a phrase of its own, from a
# comma (or the start of a sentence) to the next comma, the sentence's end, or a "who", "that" or "with" that
# starts saying more about the person (", late 20s to early 30s, ...", ", early 30s who ..."); after "of" or a word
# for roughly ("of forties or fifties", "about 20s"), as _AROUND reads "around 30"; and, in the sentence where they
# say what they want, next to the person they want (a man, a woman, someone, a partner): directly after them,
# perhaps after a comma, a dash or a bracket, a word for roughly or one that hedges ("a lady late 50s", "someone,
# mid-20s", "a man, ideally 40s", "a guy - early 40s -"), after "in" or "in the" ("a man in late 30s"), or directly
# before them ("a 40-something woman"). After "of", a word for roughly or the person wanted, the decades must end
# the sentence or clause, or be followed by a grammar word (who, that, and, in, with...), never by a word they
# describe: "a fan of 70s cinema", "about 90s pop" and "a fan of 80s and 90s films" are no ages. A run of decades
# ("60s, 70s and 80s") is read whole, so a word after its last decade decides it. A decade that describes
# something else ("who likes 80s films", "a 50s-style kitchen", "the nineties") is never an age, whatever it
# describes: no list of things a decade can describe. A "+" after decades is a lower bound ("40s+": 40 or older).
_DECADE_FORM = (r"(?:(?:early|mid|late)[\s-]*)?(?:[2-9]0(?:['’]?s|[\s-]*something)"
                r"|(?:twent|thirt|fort|fift|sixt)(?:ies|y[\s-]*something))\b")
# Between two decades: a comma (perhaps with "and", "or" or "to" after it), a dash, "to", "or" or "and". Written so
# each join can be read only one way (a space after a comma belongs to the join's end), so a long run of decades is
# read in linear time.
_JOIN = r"\s*(?:,(?:\s*(?:and|or|to)\b)?|-|–|to|or|and)\s*"
_DECADES = rf"\b{_DECADE_FORM}(?:{_JOIN}{_DECADE_FORM})*(?!{_JOIN}{_DECADE_FORM})(?:\s*\+)?"
# What may follow decades read after "of", a word for roughly or the person wanted: the end of a sentence or clause
# (a dash or a closing bracket too), or a grammar word (a relative word, a conjunction or a preposition), never a
# word they describe.
_DECADE_END = re.compile(r"\s*(?:$|[,;:.!?)–—]|-\s)|\s+(?:who|whom|whose|which|that|and|or|but|nor|so|in|near|from|"
                         r"around|with|without|for|to|at|on|of|by|within|across)\b")
# A phrase of its own ends at a comma, the sentence's end, or a "who", "that" or "with" saying more about the person.
_PHRASE_END = re.compile(r"\s*(?:$|[,;:.!?)])|\s+(?:who|that|with)\b")
# Each run of decades is found once, left to right, and read by the words just before and just after it, so a long
# text is read in linear time.
_DECADE_RUN = re.compile(_DECADES)
_AFTER_POSSESSIVE = re.compile(rf"\b(?:in\s+(?:his|her|their|your|our)|aged?)\s+(?:{_ROUGHLY}\s+)?$")
_AFTER_PUNCTUATION = re.compile(rf"[,;:.!?(]\s*(?:{_ROUGHLY}\s+)?$")
_AT_START = re.compile(rf"\s*(?:{_ROUGHLY}\s+)?")
_AFTER_OF = re.compile(rf"\b(?:of|{_ROUGHLY})\s+(?:{_ROUGHLY}\s+)?$")
# Next to the person they want: read only in the sentence where they say what they want (_ages_said, sought), since
# a man or a woman named before that, or in a sentence of its own, is the person writing ("Woman 40s, looking for
# ...", "... Me: woman mid-40s"). A hedge is one of a few words that soften a preference.
_HEDGE = r"(?:ideally|preferably|probably|maybe|perhaps|possibly|likely)"
_AFTER_SOUGHT = re.compile(rf"\b(?:{_GENDERED}|some(?:one|body)\b|(?:a|an)\s+partner\b)(?:\s*[,(]\s*|\s+[-–—]\s+|\s+)"
                           rf"(?:(?:{_ROUGHLY}|{_HEDGE})\s+)*(?:in\s+(?:the\s+)?)?$")
_BEFORE_SOUGHT = re.compile(rf"\s+(?:{_GENDERED}|partners?\b)")
_CONTEXT = 120  # characters before a run of decades in which what makes it an age is read
_ONE_DECADE = re.compile(r"\b(?:(early|mid|late)[\s-]*)?(?:([2-9])0(?:['’]?s|[\s-]*something)"
                         r"|(twent|thirt|fort|fift|sixt)(?:ies|y[\s-]*something))\b")
_SENTENCE_BREAK = re.compile(r"[.!?;](?:\s|$)")
# Numbers that aren't ages: a measured or counted thing after them ("30 km", "20 minutes", "10 years
# of experience", "2 kids", "25 people"), or money ("$30"). "30 years old" is still an age.
_UNITS = (r"(?:years?|yrs?)\b(?!\s*(?:old|of age|young))|km\b|kilomet\w*|miles?\b|mi\b|minutes?\b|mins?\b|hours?\b|"
          r"hrs?\b|days?\b|weeks?\b|months?\b|cm\b|kg\b|lbs?\b|%|percent\b|kids?\b|children\b|people\b|persons?\b|"
          r"times\b|euros?\b|dollars?\b|yen\b|pounds?\b|k\b|floors?\b|stops?\b|stations?\b|members?\b|games?\b")
_NOT_AGES = re.compile(rf"[£€$¥]\s?\d[\d,.]*|\b\d+(?:\s*(?:-|–|to|and)\s*\d+)?\s*(?:{_UNITS})")
_AROUND = re.compile(rf"\b{_ROUGHLY}\s+{_N}\b|\b{_N}\s*-?\s*ish\b")
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


def _decade_ages(text: str, sought: bool) -> list:
    """The runs of decades in the text that are ages (see _DECADE_FORM), in order: after "in his ..." or "aged"; as a
    phrase of its own; after "of" or a word for roughly; and, with sought (the text is where they say what they
    want), next to the person they want in a sentence that says what they want."""
    breaks = [b.end() for b in _SENTENCE_BREAK.finditer(text)] if sought else []
    wants = [(w.start(), w.end()) for w in _WANTS.finditer(text)] if sought else []
    seen = {"break": 0, "want": 0, "sentence": 0}  # runs come in order, so these only move forward

    def in_request(at: int) -> bool:  # a request word in the same sentence, before this point
        while seen["break"] < len(breaks) and breaks[seen["break"]] <= at:
            seen["sentence"] = breaks[seen["break"]]
            seen["break"] += 1
        while seen["want"] < len(wants) and wants[seen["want"]][0] < seen["sentence"]:
            seen["want"] += 1
        return seen["want"] < len(wants) and wants[seen["want"]][1] <= at

    out = []
    for run in _DECADE_RUN.finditer(text):
        start, end, lo = run.start(), run.end(), max(0, run.start() - _CONTEXT)
        before = lambda rx: rx.search(text, lo, start) is not None
        phrase = (before(_AFTER_PUNCTUATION) or _AT_START.fullmatch(text, 0, start) is not None) \
            and _PHRASE_END.match(text, end) is not None
        ended = _DECADE_END.match(text, end) is not None
        if before(_AFTER_POSSESSIVE) or phrase or (ended and before(_AFTER_OF)) or (sought and in_request(start) and (
                (ended and before(_AFTER_SOUGHT)) or _BEFORE_SOUGHT.match(text, end) is not None)):
            out.append(run)
    return out


def _ages_said(text: str, plain_numbers: bool, sought: bool = False) -> list:
    """(lo, hi, how) for every way an age range is written: a range ("25 to 34", "early 30s", "in their
    thirties", "30-something", "around 30"; how "range"), a lower bound ("30 or older", "not younger
    than 30"; "min") or an upper bound ("under 40", "no one over 40", "at most 35"; "max"). With
    plain_numbers, bare numbers count too: two are a range, one is around that age. With sought (the text
    is where they say what they want), a decade directly after the person they want counts too."""
    spans = []

    def take(rx, span, how):
        nonlocal text
        text = rx.sub(lambda m: spans.append((*span(m), how)) or " ", text)

    for m in _NOT_OUTSIDE.finditer(text):  # two bounds at once: a lower and an upper one
        spans += [(int(m.group(1) or m.group(4)), 99, "min"), (18, int(m.group(2) or m.group(3)), "max")]
    text = _NOT_OUTSIDE.sub(" ", text)
    take(_RANGE, lambda m: (int(m.group(1)), int(m.group(2))), "range")
    take(_AT_MOST, lambda m: (18, int(m.group(1))), "max")
    take(_AT_LEAST, lambda m: (int(m.group(1)), 99), "min")
    take(_OLDER, lambda m: (int(m.group(1) or m.group(2)), 99), "min")
    take(_YOUNGER, lambda m: (18, int(m.group(2) or m.group(3)) - (1 if m.group(1) in ("under", "younger than", "below")
                                                                    else 0)), "max")
    said = _decade_ages(text, sought)
    for m in said:  # each decade said there: "late 20s to early 30s" is two; "40s+" is 40 or older
        decades = [_decade(d) for d in _ONE_DECADE.finditer(m.group())]
        spans += [(decades[0][0], 99, "min")] if "+" in m.group() and decades else [(*d, "range") for d in decades]
    if said:
        ends = [0] + [m.end() for m in said]
        text = " ".join(text[a:m.start()] for a, m in zip(ends, said)) + " " + text[ends[-1]:]
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
    spans = _ages_said(clean(wanted), plain_numbers=guess, sought=True)
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
    (as with any detail you didn't say), so you're open to one age band either side of your own shown
    band, or to any adult band if you don't show one."""
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
    30") isn't read as around that age: matching never guesses (_reading), and a person's agent asks
    instead (agent.wanted_ages, which guesses only to notice that there is a number to ask about)."""
    wanted, whole = _wanted_part(_plain(text), _plain(headline))
    seeking = set(GENDERS) if _ANYONE.search(wanted) else {g for g, rx in _SEEKS.items() if rx.search(wanted)}
    return tuple(sorted(seeking)), _years(wanted, whole, guess)


def relative_years(text: str, own_band, headline: str = "") -> Optional[tuple]:
    """Ages a dating description asks for relative to the person's own age band ("around my age", "a
    bit younger"), in years; None if it says none or the band isn't known."""
    wanted, _ = _wanted_part(_plain(text), _plain(headline))
    return _relative_years(_SELF_AGE.sub(" ", wanted), own_band)


def _reading(low: str, kind, desc: str, head: str) -> ParsedIntent:
    """The structure of a listing of this kind (dating, other, or None), read from its text. A bare number
    ("a woman, 30") is never read as an age range here either, so matching, the person's agent and their
    phone agree."""
    if kind == "dating":
        seeking, years = dating_wants(desc, head, guess=False)
        return ParsedIntent(kind, seeking=seeking, age_range=_bands(*years) if years else None)
    # Any kind of listing may meet online except dating, which is in person only. "Not online" or
    # "no remote" says the opposite (schema.said).
    remote = said(low, r"online|remote") is True
    return ParsedIntent(kind, remote_ok=remote)


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


def without_hobbies(low: str) -> str:
    """The text without what it says the person enjoys, calendar dates and people they already have
    ("I love croquet with my girlfriend" asks for nothing)."""
    return _HOBBY.sub(" ", _set_aside(low))


def request_kind(low: str) -> Optional[str]:
    """"dating" if the text asks for a date outside what it says the person enjoys, else None."""
    return _dating_or_none(without_hobbies(low))


def _parse(text: str, headline: str) -> ParsedIntent:
    desc, head = _plain(text), _plain(headline)
    low = f"{desc}. {head}" if head else desc
    # What they want, before what they enjoy: "meet a woman ... I enjoy pottery" is dating. Only a text
    # that asks for nothing else is read from its hobbies ("I enjoy pottery and I'm looking for someone
    # kind" asks for something, and the hobby isn't it).
    kind = request_kind(low)
    if kind is None and not REQUEST.search(without_hobbies(low)):
        kind = _dating_or_none(low)
    return _reading(low, kind, desc, head)


_READ: dict = {}  # (description, shown headline, kind, wants_gender, wants_age, age band) -> ParsedIntent


def read_listing(text: str, shown) -> ParsedIntent:
    """What a listing asks for, as matching uses it. `shown(key)` returns one of the listing's SHOWN
    fields (narrow.ShownView.get), never a hidden one.

      kind     the shown "kind" if it is one of MATCH_KINDS, else read from the text: "dating" or None;
      dating   genders: the shown "wants_gender" if valid, else read from the text;
               ages: the shown "wants_age" (years, onto the age bands it overlaps), else read from
               the text, absolute ("early 30s") or relative to their shown age band ("around my
               age"); still None if neither says (matching then uses default_age_range).

    The text's words never change a kind the listing states."""
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
