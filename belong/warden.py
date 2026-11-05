"""Warden agent (SPEC §3, §5, §6, §7): checks everything that goes into matching and
everything that comes out.

Fixed rules, no AI judgment. The Warden never decides who matches; it only checks.

  ENTRY  admit()                 person + listing  -> onto the Marketboard, or rejected
  EXIT   exit_introduction()     Proposal          -> Introduction, or dropped
         exit_followup()         FollowUp          -> FollowUpOut, or dropped
         exit_reveal()           consents          -> Reveal(s) of shared hidden details
         details_in_use()        the Marketboard   -> DetailsInUse (the shared list every agent sees)
         exit_judge_request()    a request to the  -> handed to the model's client, or refused (SPEC §8.3: with the
                                 Matchmaker AI        Matchmaker AI on; checked just before it goes)

Every check writes a Receipt {check, result, timestamp, hash}. The hash commits to
what was checked (keyed HMAC) without containing it, so receipts can be published
and audited without revealing content.

Output objects (Introduction, FollowUpOut, Reveal, DetailsInUse) can only be built in this module:
their constructors demand a private seal. Anything sent to a person or their agent is one of them. A request to the
Matchmaker AI goes to a model, never to a person: it is the Matchmaker's own plain data, and it goes only if
exit_judge_request passes it, against the board as it is at that moment (checked_judge, or the demo's off-lock caller).
"""
from __future__ import annotations

import calendar
import hashlib
import hmac
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from .matching.marketboard import Marketboard
from .matching.contract import FollowUp, Proposal
from .schema import (AGE_BANDS, AGE_MAX, AGE_MIN, COMPARES, CONTACT_RULES, FIELD_REGISTRY, FOLLOWUP_MENU, FORMATS,
                     FREE_TEXT_FIELDS, FREE_TEXT_MAX, GENDERS, HIDDEN, K_CROWD, LIST_CAPS, LIST_MIN_AGE_DAYS,
                     LIST_LONG_RUN, LIST_LONG_RUN_CHARS, LIST_PHRASE_KEYS, LIST_PHRASE_WORDS, LIST_TEXT_MAX,
                     LIST_UNSPACED_CHARS, LIST_UNSPACED_SCRIPTS, LIST_VALUE_WORDS, MATCH_KINDS,
                     MAX_LISTINGS_PER_OWNER, MAX_TRAVEL_KM, PRIVATE_FIELDS, PRIVATE_VALUE_MAX, SHOWN, STRICT_EXACTNESS,
                     TRAVEL, Field, Listing, Person, age_span, canonical, detail_key, detail_problems, detail_reasons,
                     field_to_json, is_detail_key, is_reason_ref, km_between, meeting_format, meets, mentions_sensitive,
                     name_problems, own, plain_place, shown_claim_problems, split_key, text_problems)

_SEAL = object()  # private to this module; see tests/test_isolation.py


class _Sealed:
    """Base for output objects: refuses construction without the Warden's seal."""

    def __init__(self, seal):
        if seal is not _SEAL:
            raise PermissionError(f"Only the Warden can create {type(self).__name__} objects")


class Introduction(_Sealed):
    """The only thing about a match that is ever sent to anyone (SPEC §4).

    a_view is what person A sees: B's shown fields and a reason built from fields
    both of them show. b_view is the mirror image.
    """

    def __init__(self, seal, intro_id: str, person_a: str, person_b: str, a_view: dict, b_view: dict,
                 one_way_to: str = "", sent_at: str = ""):
        super().__init__(seal)
        self.intro_id, self.person_a, self.person_b = intro_id, person_a, person_b
        # When it was first sent: the independent audit counts each listing's introductions per day.
        self.sent_at = sent_at
        # A close match goes to this listing only; the other person sees nothing unless it says yes.
        self.one_way_to = one_way_to
        self._a, self._b = json.dumps(a_view, ensure_ascii=False), json.dumps(b_view, ensure_ascii=False)

    @property
    def a_view(self) -> dict:
        return json.loads(self._a)

    @property
    def b_view(self) -> dict:
        return json.loads(self._b)

    def to_dict(self) -> dict:
        return {"intro_id": self.intro_id, "person_a": self.person_a, "person_b": self.person_b,
                "a_view": self.a_view, "b_view": self.b_view, "one_way_to": self.one_way_to,
                "sent_at": self.sent_at}


class FollowUpOut(_Sealed):
    def __init__(self, seal, person_id: str, question_id: str, phrase: str = "", offer_add: str = "",
                 seek_add: str = "", pay: str = ""):
        super().__init__(seal)
        self.person_id, self.question_id, self.phrase = person_id, question_id, phrase
        # Their shown terms (pay): saying yes means taking them on, so a match isn't confused by
        # the old price ("$1 each" becomes "Free").
        self.pay = pay
        # For "also_offer": if the person says yes, the asker's shown phrases join theirs (what the
        # asker wants joins what they offer; who the asker is joins who they look for), so a match
        # can follow. Never shown in the question.
        self.offer_add, self.seek_add = offer_add, seek_add
        self.question_text = FOLLOWUP_MENU[question_id].format(phrase=phrase)

    def to_dict(self) -> dict:
        return {"person_id": self.person_id, "question_id": self.question_id, "question_text": self.question_text,
                "phrase": self.phrase, "offer_add": self.offer_add, "seek_add": self.seek_add, "pay": self.pay}


class Reveal(_Sealed):
    """A hidden detail two connected people share, revealed because both opted in."""

    def __init__(self, seal, person_a: str, person_b: str, key: str, values: list):
        super().__init__(seal)
        self.person_a, self.person_b, self.key, self.values = person_a, person_b, key, list(values)

    def to_dict(self) -> dict:
        return {"person_a": self.person_a, "person_b": self.person_b, "key": self.key, "values": self.values}


class DetailsInUse(_Sealed):
    """The shared list of details and words in use (SPEC §6): what every person's agent sees, so agents
    write the same names, scales and values and people can be matched. It is effectively public, so the
    Warden builds it from the board alone (no Matchmaker input) and puts nothing on it until K_CROWD
    different people show it on listings at least LIST_MIN_AGE_DAYS old. No counts, alphabetical order.

    to_dict() is the record, with when it was built and the board change it was built from (so the
    independent audit can rebuild it from the same board); vocabulary() is what agents are given."""

    def __init__(self, seal, built_at: str, board_change: int, details: list, scales: list, values: list,
                 phrases: list, places: list):
        super().__init__(seal)
        self.built_at, self.board_change = built_at, board_change
        self._items = json.dumps({"details": details, "scales": scales, "values": values, "phrases": phrases,
                                  "places": places}, ensure_ascii=False)

    def vocabulary(self) -> dict:
        return json.loads(self._items)

    def to_dict(self) -> dict:
        return {"built_at": self.built_at, "board_change": self.board_change, **self.vocabulary()}


@dataclass(frozen=True)
class Receipt:
    """One entry in the public, hash-chained receipt log. Contains no content.

    commitment = HMAC(audit key, what was checked): auditors holding the key can
                 confirm a receipt refers to a given input; nobody else learns anything.
    hash       = SHA-256(previous hash | seq | check | result | timestamp | commitment):
                 anyone can re-compute the chain, so a deleted, reordered or edited
                 receipt is detectable by everyone (see verify_chain).
    """
    seq: int
    check: str
    result: str  # "pass" | "fail"
    timestamp: str
    commitment: str
    hash: str

    def to_dict(self) -> dict:
        return {"seq": self.seq, "check": self.check, "result": self.result, "timestamp": self.timestamp,
                "commitment": self.commitment, "hash": self.hash}


GENESIS = "0" * 64


def chain_hash(prev: str, seq: int, check: str, result: str, timestamp: str, commitment: str) -> str:
    return hashlib.sha256(f"{prev}|{seq}|{check}|{result}|{timestamp}|{commitment}".encode()).hexdigest()


def verify_chain(receipts: list, prev: str = GENESIS) -> tuple[bool, int]:
    """Re-compute the chain over receipt dicts. Returns (ok, index of the first bad
    entry or -1). Anyone can run this; it needs no keys."""
    for i, r in enumerate(receipts):
        if r["hash"] != chain_hash(prev, r["seq"], r["check"], r["result"], r["timestamp"], r["commitment"]):
            return False, i
        prev = r["hash"]
    return True, -1


ENTRY_CHECKS = ("kyc_verified", "listing_schema", "listing_id_owner", "sensitive_defaults", "specific_person",
                "listing_cap", "update_rate_limit")
EXIT_CHECKS = ("proposal_mutual", "proposal_known_people", "proposal_distinct_people", "reason_fields_shown",
               "reason_details_true", "hidden_crowd", "approach_limit", "introduction_schema",
               "fields_shown", "reason_refs_only_reason_fields", "no_hidden_values",
               "reissue_reduced", "followup_menu", "reveal_connected", "reveal_consent_and_shared", "details_in_use")


class Clock:
    """Simulated time, so runs are reproducible."""

    def __init__(self, start: str = "2026-10-12T09:00:00"):
        self._t = datetime.fromisoformat(start)

    def now(self) -> str:
        return self._t.isoformat() + "Z"

    def advance(self, **kw) -> None:
        self._t += timedelta(**kw)


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts.rstrip("Z"))


# --------------------------------------------------------------------------------------
# Specific-person check (SPEC §5: never a specific person). Plain pattern rules.
# --------------------------------------------------------------------------------------

# The full-name rule (full_name_pairs) is a fixed floor, the same for every listing: two capitalised words in a row in
# the middle of a sentence look like a person's full name, unless either is a word that is never part of a name
# (_never_a_name). No typed list of places, languages or titles: the words it lets through come from fixed forms (the
# pronoun I, a word in capitals, a month or a weekday from the standard library) and from the listing's own shown
# place and the Warden's own shared list (never_names). Judging meaning (a name in lower case, a first name alone, a
# nickname) is left to the person's own model agent, which can only refuse more (agent_llm.SYSTEM_PROMPT,
# singles_out), never let through what this refuses.
FULL_NAME = "what looks like a full name"
_TOKEN = re.compile(r"[^\W\d_][\w'’-]*")
# The pronoun I and its contractions: structure, never part of a name.
_PRONOUN_I = re.compile(r"I(?:['’](?:m|d|ll|ve))?")
_POSSESSIVE = re.compile(r"['’]s?\Z")
# The lower-case grammar words (articles and prepositions) that stand between the words of a full name in many
# languages ("Pieter de Vries", "Maria da Graça", "Khalid bin Rashid", "Femke van der Berg"): structure, not names.
_NAME_JOIN = frozenset({"de", "da", "do", "dos", "das", "del", "della", "dei", "di", "du", "des", "van", "von", "der",
                        "den", "ter", "ten", "zu", "bin", "ibn", "bint", "al", "el", "la", "le"})
# Months and weekdays from the standard library (calendar, in the C locale Python starts in), never a typed list.
_CALENDAR = frozenset(w.casefold() for w in (*calendar.month_name, *calendar.day_name) if w)
# The built-in fields that say where a listing is. Their words, as the listing shows them, are never part of a name.
PLACE_KEYS = ("city", "country")
# Contact details (a handle, email, phone number or link: schema.CONTACT_RULES, which open details'
# values follow too), then the rules that only make sense for a description.
_SPECIFIC_PERSON_RULES = [
    *CONTACT_RULES,
    ("'named'/'called' + a name", re.compile(r"\b(named|called)\s+[A-Z]")),
    ("an ex", re.compile(r"\bmy ex\b", re.I)),
    ("'works at' + a name", re.compile(r"\bworks? at [A-Z]")),
]


# What a listing says in its own words besides its description: its free-text built-ins. The description's rules
# read each of them too.
OWN_WORDS = ("seeking", "offering", "pay", "format", "topics", "headline")
# Of those, the words the listing says about the person who wrote it, not the person sought: what they offer (the
# "you are" phrase). The identifiers rule never counts them (identifier_problems).
POSTER_WORDS = ("offering",)


# --------------------------------------------------------------------------------------
# The identifiers rule (SPEC §5): clues that together narrow to one person. Fixed rules, no judgement.
# --------------------------------------------------------------------------------------
# One or two common details of how someone looks describe a large share of people and identify nobody; clues narrow
# to one person when there are too many of them, or one is very unusual. So the rule counts only clues about the
# person sought (never the poster's own: "I'm a cashier at", "I'm quite short", "my beard"), over everything the
# listing says about that person together (identifier_problems), and refuses (what the poster says about themselves,
# their "what you offer" phrase and their open details about themselves, is read as theirs, poster_clues, so only a
# clue about someone else written there counts):
#   - an ANCHOR, a place or time where that person can be found, with any TRAIT, a detail of how they look;
#   - more than MAX_COMMON_TRAITS distinct traits;
#   - a distinctive MARK: a mark described with where it is on them or what it shows.
# A trait counts only where it describes a person, never a thing: read from how the sentence is built (a person word
# next to it, "who is", a list of looks), never from a list of things. The words it knows are few: the looks words
# and marks below are the traits, and the anchors are the workplace words (_WORKPLACE), "lives at" / "lives on",
# someone the poster saw or met (_ENCOUNTER, _encounter), "the" + a clock time, and "the man" (or another word for one
# person) "at" / "on" / "from" a place. Everything else is grammar. English only; judging meaning is left to the person's own
# model agent, which can only refuse more (agent_llm.SYSTEM_PROMPT, singles_out).
ANCHORED = "where to find them + how they look"
TOO_MANY_TRAITS = "three or more details of how they look"
UNIQUE_MARK = "a distinctive mark and where it is or what it shows"
IDENTIFIERS = (ANCHORED, TOO_MANY_TRAITS, UNIQUE_MARK)
MAX_COMMON_TRAITS = 2
# Anchors: where someone works (the words of the rule this one replaces) and where they live.
_WORKPLACE = re.compile(r"(works?|working|barista|cashier|employee|bartender) at")
_HOME = re.compile(r"(lives?|living) (at|on)")
# Someone the poster saw or met (an anchor only about one person: "who I saw", "the man I met", "I saw you").
_ENCOUNTER = frozenset({"saw", "seen", "met", "noticed", "spotted"})
# Traits: how someone looks (the words of the rule this one replaces), and marks, which are traits too.
_LOOKS = frozenset({"tall", "short"})  # describing words: they count only where they describe a person
_LOOKS_NOUNS = {"hair": "hair", "glasses": "glasses", "beard": "beard", "freckles": "freckles"}
_MARKS = {"tattoo": "tattoo", "tattoos": "tattoo", "piercing": "piercing", "piercings": "piercing",
          "scar": "scar", "scars": "scar", "birthmark": "birthmark", "birthmarks": "birthmark"}
# Words for a person (as matching reads them too, with the words matching reads as "partner": a copy, since the
# Warden never imports the Matchmaker's code), and those for one person, which "the" can point at.
_PERSON = frozenset({"partner", "partners", "man", "men", "woman", "women", "guy", "guys", "girl", "girls", "lady",
                     "ladies", "gentleman", "gentlemen", "people", "person", "someone", "somebody", "anyone",
                     "anybody", "buddy", "buddies", "friend", "friends", "companion", "companions", "pal", "pals"})
_ONE_PERSON = frozenset({"man", "woman", "guy", "girl", "lady", "gentleman", "person"})
# Grammar: who is speaking or spoken of, being and having, degree words, and the closed-class words that may follow a
# looks noun describing a person ("hair and", "glasses who"), unlike a word it describes ("hair salon").
_FIRST = frozenset({"i", "i'm", "i've", "i'd", "i'll", "we", "we're", "we've", "we'd", "we'll"})
_MINE = frozenset({"my", "our"})
_OTHER = frozenset({"he", "she", "they", "who", "you"})
_OTHER_IS = frozenset({"he's", "she's", "they're", "who's", "you're"})
_IS = frozenset({"am", "is", "are", "was", "were", "looks", "looked", "seems", "seemed"})
_HAVE = frozenset({"have", "has", "had", "wear", "wears", "wore", "got"})
_DEGREE = frozenset({"a", "an", "very", "quite", "really", "pretty", "fairly", "rather", "so", "too", "bit", "little",
                     "not", "also", "still", "both", "kind", "sort", "of"})
_JOIN = frozenset({",", "and", "with", "&", "plus"})
_GRAMMAR = frozenset({
    "a", "an", "the", "this", "that", "these", "those", "my", "our", "your", "his", "her", "their", "its", "some",
    "any", "no", "every", "each", "and", "or", "but", "nor", "so", "yet", "with", "without", "who", "whom", "whose",
    "which", "what", "when", "where", "while", "if", "as", "than", "to", "of", "in", "on", "at", "for", "from", "by",
    "near", "under", "over", "across", "above", "below", "behind", "beside", "between", "into", "onto", "through",
    "around", "about", "after", "before", "during", "since", "until", "like", "down", "up", "off", "out", "is",
    "are", "was", "were", "be", "been", "being", "am", "has", "have", "had", "do", "does", "did", "can", "could",
    "will", "would", "should", "may", "might", "must", "i", "me", "we", "us", "you", "he", "him", "she", "they",
    "them", "it", "please", "too", "also", "not", "just", "only", "very", "plus", "again", "now", "then", "though",
    "maybe", "perhaps", "everywhere", *_FIRST, *_OTHER_IS})
# Where a mark is on someone ("across his ...", "on the left ...") and what it shows ("of ...", "saying ...").
_ON = frozenset({"on", "across", "over", "under", "above", "below", "behind", "along", "down", "around", "near",
                 "beside", "inside", "in", "between", "up", "by"})
_WHOSE_BODY = frozenset({"his", "her", "their", "your"})
_SHOWS = frozenset({"of", "saying", "reading", "spelling", "shaped"})
_FEATURE = frozenset({"with", "has", "have", "had", "having", "got", "wears", "wearing", "his", "her", "their",
                      "your"})
_OPENS = frozenset({",", ";", ":", "-", "—", "–", "("})  # punctuation that opens a phrase inside a sentence
_ID_TOKEN = re.compile(r"[^\W\d_][\w'-]*|\d{1,2}[:.]\d{2}|\d+|[^\w\s]")
_CLOCK = re.compile(r"\d{1,2}[:.]\d{2}")
_PLACE_START = frozenset({"a", "an", "the", "my", "our", "your", "his", "her", "their", "this", "that"})
_SENTENCE_END = frozenset({".", "!", "?", ";"})


@dataclass(frozen=True)
class Clues:
    """What one text says about the person sought: anchors (where they can be found), traits (each distinct detail of
    how they look, by its word) and marks described with where they are or what they show. The poster's own never
    counts."""
    anchors: tuple = ()
    traits: frozenset = frozenset()
    marks: tuple = ()

    def __bool__(self) -> bool:
        return bool(self.anchors or self.traits or self.marks)


def identifier_problems(texts=(), wishes=(), poster=()) -> list[str]:
    """Reasons the clues about the person sought, all together, could narrow to one person (the identifiers rule):
    texts are what the listing says about them in its own words (the description, the free-text built-ins but
    POSTER_WORDS), wishes its shown "them" and "both" open details, and poster what the poster says about
    themselves (POSTER_WORDS), read as theirs (poster_clues). Empty list = fine."""
    return _identifiers([*(clues(t) for t in texts), *(wish_clues(f) for f in wishes),
                         *(poster_clues(t) for t in poster)])


def _identifiers(found) -> list[str]:
    """The identifiers rule over clue sets taken together."""
    found = list(found)
    traits = frozenset().union(*(c.traits for c in found))
    out = []
    if traits and any(c.anchors for c in found):
        out.append(ANCHORED)
    if len(traits) > MAX_COMMON_TRAITS:
        out.append(TOO_MANY_TRAITS)
    if any(c.marks for c in found):
        out.append(UNIQUE_MARK)
    return out


def wish_clues(f: Field) -> Clues:
    """A shown wish's clues: its written forms (_written) read like any text, and in a wish about the other person
    ("them"), what describes the person sought by what such a wish is: a looks noun or mark in its name ("hair
    colour"), or else each value read on its own, so a describing word that is all a value says ("tall"), or starts
    it ("short, with ..."), describes them, and a mark in it is theirs ("a tattoo of ...")."""
    found = [clues(t) for t in _written(f)]
    if f.about == "them":
        named = frozenset(k for w in _id_tokens(f.name) for k in [_LOOKS_NOUNS.get(w) or _MARKS.get(w)] if k)
        found += [Clues(traits=named)] if named else [clues(v, featured=True) for v in f.strings()]
    return Clues(tuple(a for c in found for a in c.anchors), frozenset().union(*(c.traits for c in found)),
                 tuple(m for c in found for m in c.marks))


def poster_clues(text) -> Clues:
    """What the poster's own words (their "what you offer" phrase, a value of an open detail about themselves) say
    about someone else: the text read as the poster saying it about themselves ("I'm ..."), so their own job and
    looks never count (the own forms: "I'm a cashier at ...", "I'm tall, with a beard"), while a clue about another
    person still does (where to find "the man" they saw, how "she" looks, a mark on "his" arm)."""
    return clues(f"I'm {_one_line(text)}")


def detail_clues(f: Field) -> Clues:
    """An open detail's clues about the person sought. A wish ("them", "both"): wish_clues. A detail about the poster
    ("me"): its name read as theirs ("my {name}") and each value and step read as theirs (poster_clues), so their own
    job and looks never count but a clue about someone else written there does."""
    if f.about != "me":
        return wish_clues(f)
    found = [clues(f"my {_one_line(f.name)}"), *(poster_clues(s) for s in [*f.strings(), *f.steps])]
    return Clues(tuple(a for c in found for a in c.anchors), frozenset().union(*(c.traits for c in found)),
                 tuple(m for c in found for m in c.marks))


def _id_tokens(text, case: bool = False) -> list[str]:
    """A text as the identifiers rule reads it: its words (with apostrophes and hyphens inside them), clock times,
    numbers and punctuation marks, one space apart (_one_line), in lower case (or, with case, as written)."""
    found = _ID_TOKEN.finditer(_one_line(text).replace("’", "'"))
    return [t.group() if case else t.group().casefold() for t in found]


def _is_word(t: str) -> bool:
    """A word or a number, not a punctuation mark."""
    return bool(t) and (t[0].isalpha() or t[0].isdigit())


def _person(t: str) -> bool:
    """A word for a person, plural or possessive included ("man's")."""
    return re.sub(r"'s?\Z", "", t) in _PERSON


def clues(text, featured: bool = False) -> Clues:
    """The anchors, traits and distinctive marks one text gives about the person sought (see the rule above).
    featured: the text is a value of a wish about them, so a mark in it is their feature (_distinctive)."""
    written = _id_tokens(text, case=True)
    toks = [t.casefold() for t in written]
    starts, start = [], 0  # where each word's sentence starts
    for k, t in enumerate(toks):
        starts.append(start)
        start = k + 1 if t in _SENTENCE_END else start
    anchors = _anchors(toks, starts, {k for k, t in enumerate(written) if t[:1].isupper()})
    return Clues(tuple(a for _, a in anchors), *_traits(toks, starts, {k for k, _ in anchors}, featured))


def _back(toks: list, k: int, n: int) -> list:
    """Up to n words before position k in the same phrase (stopping at a punctuation mark other than "&")."""
    out = []
    for j in range(k - 1, max(-1, k - 1 - n), -1):
        if not _is_word(toks[j]) and toks[j] != "&":
            break
        out.append(j)
    return out


def _skip(toks: list, j: int, words, n: int) -> int:
    """Position j moved back past up to n of these words (and words ending in -ly)."""
    while j >= 0 and n and (toks[j] in words or (toks[j].endswith("ly") and toks[j].isalpha())):
        j, n = j - 1, n - 1
    return j


def _anchors(toks: list, starts: list, names=frozenset()) -> list:
    """Where the person sought can be found: where they work or live, at a place ("at" / "on" + an article, a
    possessive, a number or a capitalised word, the positions in `names`: never "at weekends" or "at night"), and not
    the poster's own ("I work at"); someone the poster saw or met ("who I saw", "the man I met", "I saw you"); "the" +
    a clock time ("the 7:40"); and "the" (or "that", "this") + a word for one person + "at" / "on" / "from" /
    "behind" / "outside" / "near" / "by". As (position, words)."""
    out, n = [], len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    place = lambda i: at(i) in _PLACE_START or at(i).isdigit() or i in names
    for k, t in enumerate(toks):
        pair = f"{t} {at(k + 1)}"
        if (_WORKPLACE.fullmatch(pair) or _HOME.fullmatch(pair)) and place(k + 2) and not _own_anchor(toks, k, starts):
            out.append((k, pair))
        elif t in _ENCOUNTER:
            if _encounter(toks, k, place):
                out.append((k, pair))
            elif t == "seen" and at(k + 1) in {"at", "on"}:
                out.append((k, pair))
        elif t == "the" and (_CLOCK.fullmatch(at(k + 1)) or (at(k + 1).isdigit() and at(k + 2) in {"am", "pm"})):
            out.append((k, pair))
        elif t in {"the", "that", "this"}:
            places = {"at", "on", "behind", "outside", "from", "near", "by"}
            if (at(k + 1) in _ONE_PERSON and at(k + 2) in places) or (
                    _is_word(at(k + 1)) and at(k + 2) in _ONE_PERSON and at(k + 3) in places):
                out.append((k, " ".join(toks[k:k + 4])))
    return out


def _encounter(toks: list, k: int, place) -> bool:
    """Whether a verb of seeing or meeting (_ENCOUNTER, at position k) says the poster saw or met the person sought:
    its subject is the poster ("I", "we", "you and I", with "have", "just"... between) or there is none (the verb
    opens a sentence or a phrase: "Saw you at ..."), and what it says is about one person: "you", "him", "her" or
    "them" after it; "who I saw", "the man I met" (a word for a person, or "the" + a word, before the subject);
    "at" / "on" + a place right after it ("we met at the ...", place: as _anchors reads one); or a word for one
    person after it, alone or after "a" / "the" and up to two words ("I saw a tall woman", "met someone")."""
    n = len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    j = _skip(toks, k - 1, {"'ve", "have", "had", "just", "once", "first"}, 2)
    first = at(j) in _FIRST
    if not (first or j < 0 or not _is_word(at(j))):
        return False
    one = lambda i: at(i) in _ONE_PERSON or at(i) in {"someone", "somebody"}
    return (at(k + 1) in {"you", "him", "her", "them"}
            or (first and (at(j - 1) in {"who", "whom", "that"} or _person(at(j - 1))
                           or (at(j - 2) in {"the", "that", "this"} and _is_word(at(j - 1)))))
            or (at(k + 1) in {"at", "on"} and place(k + 2))
            or one(k + 1)
            or (at(k + 1) in {"a", "an", "the", "this", "that"} and any(
                one(i) for i in range(k + 2, k + 5) if all(_is_word(at(p)) for p in range(k + 2, i + 1)))))


def _own_anchor(toks: list, k: int, starts: list) -> bool:
    """A workplace or home that is the poster's own: right after a first-person subject ("I work at", "I'm a
    cashier at", "we live on"), or "and work at" / "and live on" in a sentence that starts with one (starts: where
    each word's sentence starts). Never one they share with the person sought (_shared_place)."""
    j = _skip(toks, k - 1, {"a", "an", "the", "also", "still", "now", "just", "am", "are", "was", "were"}, 3)
    if _shared_place(toks, k, j):
        return False
    if j >= 0 and toks[j] in _FIRST:
        return True
    return j >= 0 and toks[j] in {"and", "but", "&"} and toks[k] in {"work", "working", "live", "living"} \
        and toks[starts[k]] in _FIRST


def _shared_place(toks: list, k: int, j: int) -> bool:
    """A workplace or home (at position k, its subject at j) the poster shares with someone: "we" with "both" or "all"
    ("we both work at"), or "the same" place, with "we" as its subject or "as" + someone after it in the sentence ("I
    work at the same ... as you"). It is where the person sought can be found too, so it isn't the poster's own."""
    n = len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    if at(j) in {"both", "all"} and at(j - 1) in _FIRST:
        return True
    if not (at(k + 2) == "the" and at(k + 3) == "same"):
        return False
    if at(j).startswith("we"):
        return True
    for i in range(k + 4, min(n, k + 14)):
        if toks[i] in _SENTENCE_END:
            break
        if toks[i] == "as" and (at(i + 1) in {"you", "him", "her", "them", "the", "that", "this"} or _person(at(i + 1))):
            return True
    return False


def _traits(toks: list, starts: list, anchored=frozenset(), featured: bool = False) -> tuple:
    """(the distinct traits that describe the person sought, the distinctive marks): each looks word or mark not the
    poster's own; a describing word ("tall") only where it describes a person, and a noun ("hair") not where it
    describes a thing ("hair salon", "glasses of ..."). starts: where each word's sentence starts; anchored: the
    positions of the anchors; featured: as clues."""
    n = len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    found = []  # (position, key, own), in order
    for k, t in enumerate(toks):
        key = _LOOKS_NOUNS.get(t) or _MARKS.get(t) or ("nose ring" if t == "ring" and at(k - 1) == "nose" else None)
        if key is None and t not in _LOOKS:
            continue
        key = key or t
        own = _own_noun(toks, k) if key not in _LOOKS else _own_looks(toks, k)
        if not own and found:  # a list of the poster's own looks goes on being theirs ("I'm tall, with ...")
            p, _, was_own = found[-1]
            own = was_own and k - p <= 4 and all(
                toks[i] in _JOIN or toks[i] in {"a", "an"}
                or (_is_word(toks[i]) and toks[i] not in _GRAMMAR and not _person(toks[i])) for i in range(p + 1, k))
        found.append((k, key, own))
    nouns = {}  # position -> key, of the counted nouns
    marks = []
    for k, key, own in found:
        if own or key in _LOOKS:
            continue
        nxt = at(k + 1)
        if key in _MARKS.values() and nxt in _ON and at(k + 2) in _MINE:
            continue  # "a tattoo on my wrist": the poster's own
        if _is_word(nxt) and nxt not in _GRAMMAR and nxt not in _ON and not _person(nxt) and nxt not in _LOOKS and \
                not nxt.endswith(("ing", "ed", "ly")) and not (_LOOKS_NOUNS.get(nxt) or _MARKS.get(nxt)) and \
                not _a_feature(toks, k):
            continue  # describing a thing: "hair salon", "tattoo artist"
        if nxt == "of" and key not in _MARKS.values():
            continue  # "glasses of ..."
        nouns[k] = key
        if key in _MARKS.values() and _distinctive(toks, k, featured):
            marks.append(" ".join(toks[k:k + 4]))
    last, latest = [], -1  # the last anchor before each word
    for k in range(n):
        last.append(latest)
        latest = k if k in anchored else latest
    traits = set(nouns.values())
    for k, key, own in found:
        if key in _LOOKS and not own and not _merged(toks, k) and _describes_a_person(
                toks, k, nouns, last[k] >= starts[k], anchored):
            traits.add(key)
    return frozenset(traits), tuple(marks)


def _a_feature(toks: list, k: int) -> bool:
    """A looks noun or mark (at position k) said as someone's feature, so the word after it says what it is like
    ("hair colour ...", "a tattoo sleeve of ...") rather than making it a thing ("hair salon"): a word of having or a
    possessive (_FEATURE: "with", "has", "his"...) up to three words before it in the same phrase, or it opens a phrase
    after a comma, semicolon, colon, dash or bracket ("...; hair colour bright pink")."""
    return any(toks[j] in _FEATURE for j in _back(toks, k, 3)) or (k >= 1 and toks[k - 1] in _OPENS)


def _own_noun(toks: list, k: int) -> bool:
    """A looks noun or mark that is the poster's own: "my" / "our" up to three words before it in the same phrase (not
    when a word for a person follows it), or a first-person "have" ("I have", "I've got", "we wear") up to three words
    before it."""
    back = _back(toks, k, 4)
    if any(toks[j] in _MINE for j in back[:3]) and not _person(toks[k + 1] if k + 1 < len(toks) else ""):
        return True  # but "my ... hair girl": the person it describes
    for j in back:
        if toks[j] in {"i've", "we've"} or (toks[j] in _HAVE and j and toks[j - 1] in {"i", "we"}):
            return all(toks[i] not in _GRAMMAR or toks[i] in {"a", "an", "got", "had"} for i in range(j + 1, k))
    return False


def _own_looks(toks: list, k: int) -> bool:
    """A describing word of looks that is the poster's own: after "I'm" / "we're" (or "I am", "we were"), with up to two
    degree words between ("I'm quite tall", "I'm a short ..."), or after "my" / "our" ("my tall brother")."""
    j = _skip(toks, k - 1, _DEGREE, 2)
    if j >= 0 and (toks[j] in {"i'm", "we're"} or (toks[j] in _IS and j and toks[j - 1] in {"i", "we"})):
        return True
    return any(toks[i] in _MINE for i in _back(toks, k, 3))


def _merged(toks: list, k: int) -> bool:
    """A describing word that is part of a looks noun or mark next to it ("short dark hair", "hair: short"), so it is
    not a trait of its own."""
    noun = lambda t: t in _LOOKS_NOUNS or t in _MARKS
    for i in range(k + 1, min(len(toks), k + 4)):
        if noun(toks[i]):
            return True
        if not _is_word(toks[i]) or toks[i] in _GRAMMAR:
            break
    return k >= 1 and (noun(toks[k - 1]) or (toks[k - 1] == ":" and k >= 2 and noun(toks[k - 2])))


def _describes_a_person(toks: list, k: int, nouns: dict, anchored: bool = False, anchors=frozenset()) -> bool:
    """Whether a describing word of looks describes a person, from how the sentence is built: next to a word for a
    person ("a tall man", "someone tall") or before the person an anchor finds (anchors, their positions: "the short
    cashier at ...", "a tall ... who works at ..."); said of one ("she's tall", "who looks short"; or "is short" ending
    a clause after an anchor in the same sentence, anchored, since an anchor says where a person is found); standing
    alone between commas or at the start of a sentence, degree words aside ("..., ideally tall, ..."); or in a list
    with a looks noun or mark that counts (nouns: "short, with glasses", "freckles and tall")."""
    n = len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    if _person(at(k + 1)) or (_is_word(at(k + 1)) and at(k + 1) not in _GRAMMAR and _person(at(k + 2))) \
            or _person(at(k - 1)):
        return True
    if k + 1 in anchors or (_is_word(at(k + 1)) and at(k + 1) not in _GRAMMAR and (
            k + 2 in anchors or (at(k + 2) in {"who", "that"} and k + 3 in anchors))):
        return True
    j = _skip(toks, k - 1, _DEGREE, 2)  # past degree words
    before, after = at(j), at(k + 1)
    ends = after in ("", ",", ")", "and", "with", "&") or after in _SENTENCE_END  # the word ends a phrase
    if before in _OTHER_IS or (before in _IS and (at(j - 1) in _OTHER or _person(at(j - 1)))):
        return True
    if anchored and before in _IS and at(j - 1) not in _FIRST and ends:
        return True
    if (j < 0 or before in _SENTENCE_END or before in {",", "(", "-", "—", "–"}) and ends:
        return True
    if after in _JOIN:
        for i in range(k + 2, min(n, k + 5)):
            if i in nouns:
                return True
            if not (toks[i] in _JOIN or toks[i] in {"a", "an"} or (_is_word(toks[i]) and toks[i] not in _GRAMMAR)):
                break
    j = _skip(toks, k - 1, _DEGREE, 1)
    return at(j) in {",", "and", "&"} and (j - 1) in nouns


def _distinctive(toks: list, k: int, featured: bool = False) -> bool:
    """A mark described with where it is on someone ("... across her ...", "... on the left ...", "... on the back of
    his ...", with up to five words of its own between: "... of a swallow on her ...", "... sleeve of koi fish on
    her ...") or, as their feature ("with", "has", "his"... up to three words before, or featured: a value of a wish
    about them), what it shows ("... of ...", "... saying ...", "... that reads ...")."""
    n = len(toks)
    at = lambda i: toks[i] if 0 <= i < n else ""
    for p in range(k + 1, min(n, k + 7)):
        if at(p) in _ON and _on_someone(toks, p + 1):
            return True
        if not (at(p) in {"of", "a", "an", "just"} or (_is_word(at(p)) and at(p) not in _GRAMMAR)):
            break
    shows = at(k + 1) in _SHOWS or (at(k + 1) in {"that", "which"} and at(k + 2) in {"says", "reads", "spells",
                                                                                    "shows"})
    return shows and (featured or any(toks[j] in _FEATURE for j in _back(toks, k, 3)))


def _on_someone(toks: list, i: int) -> bool:
    """Whether the words from position i (after "on", "across"...) say where on someone: "his", "her", "their" or
    "your" ("on her ..."), "the left" / "the right", or "the" + up to three words + "of" + one of those ("on the back
    of her ...", "across the bridge of his ...")."""
    n = len(toks)
    at = lambda p: toks[p] if 0 <= p < n else ""
    if at(i) in _WHOSE_BODY:
        return True
    if at(i) != "the":
        return False
    if at(i + 1) in {"left", "right"}:
        return True
    for p in range(i + 1, min(n, i + 5)):
        if at(p) == "of":
            return at(p + 1) in _WHOSE_BODY
        if not _is_word(at(p)) or at(p) in _GRAMMAR:
            return False
    return False


def _one_line(text) -> str:
    """Text as the specific-person rules read it: as it looks, its words with one space between them, so a doubled
    space, a tab or a line break between two words never hides a rule ("my  ex", "works\nat"), and neither does a
    character that shows nothing (_invisible: a zero-width space, a word joiner, a soft hyphen beside or inside a
    word) or a letter written with a separate accent mark (read composed, NFC)."""
    text = unicodedata.normalize("NFC", str(text or ""))
    if not text.isascii():
        text = "".join(c for c in text if not _invisible(c))
    return " ".join(text.split())


def _invisible(c: str) -> bool:
    """A character that shows nothing: a Unicode format character (category Cf: zero-width space and joiners, word
    joiner, soft hyphen, direction marks...) or a variation selector."""
    return unicodedata.category(c) == "Cf" or 0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF


def specific_person_problems(intent: str, fields=(), places=(), identifiers: bool = True,
                             poster: bool = False) -> list[str]:
    """Reasons this text could single out one individual. Empty list = fine. fields: the listing's own fields, and
    places: the places on the Warden's shared list, for the places the full-name rule never reads as part of a name
    (never_names); without them, only its fixed forms are. identifiers: whether the identifiers rule reads this text
    on its own too (identifier_problems), and poster: whether it is what the poster says about themselves
    (POSTER_WORDS), read as theirs (poster_clues)."""
    intent = _one_line(intent)
    problems = [name for name, rx in _SPECIFIC_PERSON_RULES if rx.search(intent)]
    problems += (identifier_problems(poster=[intent]) if poster else identifier_problems([intent])) if identifiers else []
    if full_name_pairs(intent, never_names(fields, places)):
        problems.append(FULL_NAME)
    return problems


def full_name_pairs(text: str, exempt=frozenset()) -> list[str]:
    """Each pair of words in the text that looks like a person's full name ("Mika Sato"): two capitalised words in a
    row, neither at the start of a sentence, and neither a word that is never part of a name (_never_a_name: the
    pronoun I and its contractions, a word written all in capitals, a month or a weekday) or a word of a place named
    whole in the text (exempt: the places of never_names, each as its words in order; _in_places). Between the two
    words: only spaces; or lower-case words that stand inside a name (_NAME_JOIN: "Pieter de Vries", "Femke van der
    Berg"); or, after an initial or a short title without vowels, its full stop ("J. Okafor", "Dr. Okonkwo"). A
    possessive counts as its word ("Mika Sato's"). Each pair as the text writes it. Empty list = fine."""
    text = _one_line(text)
    tokens = list(_TOKEN.finditer(text))
    placed = _in_places(tokens, exempt)
    proper = [i not in placed and _is_proper_noun(text, t) for i, t in enumerate(tokens)]
    gap = lambda i: text[tokens[i].end():tokens[i + 1].start()]  # what stands between token i and the next
    pairs = []
    for i, tok in enumerate(tokens):
        if not proper[i]:
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].group() in _NAME_JOIN and not gap(j - 1).strip():
            j += 1
        if j < len(tokens) and proper[j] and (not gap(j - 1).strip() or (
                j == i + 1 and gap(i).strip() == "." and _initial(tok.group()))):
            pairs.append(text[tok.start():tokens[j].end()])
    return pairs


def _initial(word: str) -> bool:
    """A word that a full stop after doesn't end a sentence: an initial (one capital letter) or a short title written
    without vowels ("Dr", "Mrs", "St")."""
    return (len(word) == 1 and word.isupper()) or (
        word.isascii() and word.isalpha() and word[0].isupper() and len(word) <= 3 and not set(word.casefold()) & set("aeiouy"))


def _in_places(tokens: list, places) -> set:
    """The positions of the words that are part of a place named whole in the text: a place of never_names (a tuple
    of its words, bare; or one word) whose words all stand there, in order. A word that is only part of a place
    ("Diego" without "San") is part of none."""
    bare = [_bare(t.group()) for t in tokens]
    by_first = {}
    for place in places:
        words = (place,) if isinstance(place, str) else tuple(place)
        if words:
            by_first.setdefault(words[0], []).append(words)
    out = set()
    for i, w in enumerate(bare):
        for words in by_first.get(w, ()):
            if tuple(bare[i:i + len(words)]) == words:
                out.update(range(i, i + len(words)))
    return out


def never_names(fields=(), places=()) -> frozenset:
    """The places the full-name rule never reads as part of a name in one listing's own words, besides its fixed forms
    (_never_a_name): the listing's own city and country as it shows them (the built-in fields, PLACE_KEYS, and only
    shown: never an open detail, which could carry a name to unlock it, and never a hidden value), and each place on
    the Warden's own shared list (places: its "places" entries, a city with its country, which K_CROWD different people
    show). Each place as the tuple of its words, compared without a possessive, accents or case (_bare): it unlocks
    only where its words all stand together, in order (_in_places), never one of its words alone."""
    texts = [s for f in fields if f.key in PLACE_KEYS and f.visibility == SHOWN for s in f.strings()]
    texts += [e[k] for e in places if isinstance(e, dict) for k in PLACE_KEYS if isinstance(e.get(k), str)]
    return frozenset(words for text in texts for words in [tuple(_bare(t.group()) for t in _TOKEN.finditer(text))]
                     if words)


def _bare(word: str) -> str:
    """A word as the full-name rule compares it: without a possessive ending, without accents, in one case."""
    word = _POSSESSIVE.sub("", word)
    return "".join(c for c in unicodedata.normalize("NFKD", word) if not unicodedata.combining(c)).casefold()


def _never_a_name(word: str, exempt=frozenset()) -> bool:
    """A word that is never part of a name: the pronoun I and its contractions (structure), a word written all in
    capitals with at least two letters (an abbreviation, a title, a code), a month or a weekday (the standard library's
    calendar), or one of `exempt` (single bare words)."""
    core = _POSSESSIVE.sub("", word)
    return bool(_PRONOUN_I.fullmatch(word)) or (core.isupper() and sum(c.isalpha() for c in core) >= 2) \
        or _bare(word) in _CALENDAR or _bare(word) in exempt


def sought_words(intent_text: str, fields=()) -> list:
    """What a listing says about the person sought in its own words: its description and its free-text built-ins
    (OWN_WORDS), but what the poster says about themselves (POSTER_WORDS), each read with one space between words."""
    return [_one_line(t) for t in [intent_text, *(s for f in fields if f.key in OWN_WORDS and f.key not in POSTER_WORDS
                                                  for s in f.strings())]]


def poster_words(fields=()) -> list:
    """What a listing says about the poster in its own free text (POSTER_WORDS: what they offer), each read with one
    space between words. The identifiers rule reads it as theirs (poster_clues)."""
    return [_one_line(s) for f in fields if f.key in POSTER_WORDS for s in f.strings()]


def listing_person_problems(intent_text: str, fields=(), places=()) -> list[str]:
    """The description's rules over a listing's own words: its description and each free-text built-in (OWN_WORDS)
    read on their own, and the identifiers rule over what they say about the person sought all together
    (sought_words, identifier_problems), since every one of them describes that person, with what they say about the
    poster read as theirs (poster_words, poster_clues). The full-name rule reads them with the listing's own shown
    city and country and the places on the Warden's shared list (places) as places that are never part of a name
    where they are written out whole (never_names). Empty list = fine."""
    texts = [_one_line(t) for t in [intent_text, *(s for f in fields if f.key in OWN_WORDS for s in f.strings())]]
    problems = {p for t in texts for p in specific_person_problems(t, fields, places, identifiers=False)}
    problems.update(identifier_problems(sought_words(intent_text, fields), poster=poster_words(fields)))
    return sorted(problems)


# The rules above that read what a description says about the person sought ('named'/'called' + a name, an
# ex, 'works at' + a name), without the contact rules (open details have them already, CONTACT_RULES).
_SAID_ABOUT_SOMEONE = _SPECIFIC_PERSON_RULES[len(CONTACT_RULES):]


def detail_person_problems(fields, intent_text: str = "") -> list[str]:
    """Reasons a listing's open details could single out one individual, by the description's rules: each open
    detail (_written: "{name}: {values and steps}", and its name before each value or step) against
    'named'/'called' + a name, an ex and 'works at' + a name; and the identifiers rule over the description, the
    listing's other free text about the person sought (sought_words), what it says about the poster (poster_words)
    and all its open details (detail_clues: its shown wishes, "them" and "both" details, and its details about the
    poster, "me", read as theirs) taken together, so a place to find them in one and their looks in another count as
    one description.
    (The full-name rule reads descriptions only.) Empty list = fine. The rules live in one place,
    detail_person_findings, which the Warden checks at entry and the agents before saving (by detail); this is the
    same, by problem, with what the listing's own words have on their own."""
    problems = {problem for _, _, problem in detail_person_findings(fields, intent_text)}
    # The findings leave what the listing's own words have on their own to the listing's own check.
    problems.update(identifier_problems(sought_words(intent_text, fields), poster=poster_words(fields)))
    return sorted(problems)


def _written(f: Field) -> list:
    """An open detail as the description's rules read it: "{name}: {values and steps}", and its name before each
    value or step ("{name} {value}"), so a rule reads the same whichever side of the colon its words are on (a wish
    named "called" with a name as its value)."""
    said = [*f.strings(), *f.steps]
    return [_one_line(t) for t in [f"{f.name}: {', '.join(said)}", *(f"{f.name} {s}" for s in said)]]


def detail_person_findings(fields, intent_text: str = "") -> list[tuple]:
    """The rules of detail_person_problems, saying where each problem is: (the names of the open details it is in,
    what else it was read with: "", or which of the listing's own words, the problem). An identifiers problem the
    listing's own words (sought_words, poster_words) have all on their own is theirs (listing_person_problems), so it
    isn't repeated here; one the open details add (detail_clues) is named by the details that give a clue, and the
    listing's own words that give one. Empty list = fine. The Warden checks it at entry, so a person is told which
    detail to change."""
    details = [f for f in fields if is_detail_key(f.key)]
    out = [((f.name,), "", name) for f in details for name, rx in _SAID_ABOUT_SOMEONE
           if any(rx.search(t) for t in _written(f))]
    read = [(f.name, detail_clues(f)) for f in details if f.about == "me" or f.visibility == SHOWN]
    phrases = sought_words("", fields)[1:]
    own = [("the description", [clues(intent_text)]),
           ("the phrases", [*(clues(t) for t in phrases), *(poster_clues(t) for t in poster_words(fields))])]
    pieces = [c for _, found in own for c in found]
    alone = set(_identifiers(pieces))
    for problem in _identifiers(pieces + [found for _, found in read]):
        if problem not in alone:
            named = tuple(dict.fromkeys(name for name, found in read if _gives(found, problem)))
            read_with = " and ".join(where for where, found in own if any(_gives(c, problem) for c in found))
            out.append((named, read_with, problem))
    return out


def _gives(c: Clues, problem: str) -> bool:
    """Whether these clues are part of this identifiers problem: a mark for a distinctive mark, a trait for too many,
    and an anchor or a trait for both together."""
    return bool(c.marks) if problem == UNIQUE_MARK else bool(c.traits) if problem == TOO_MANY_TRAITS else \
        bool(c.anchors or c.traits)


def _in_details(names: tuple, read_with: str) -> str:
    """Where a problem in open details is, for the person: "detail 'x'", "details 'x' and 'y'", and what else it
    was read with ("the description", "the phrases")."""
    quoted = [f"'{n}'" for n in names]
    where = ("detail " if len(quoted) == 1 else "details ") + (
        quoted[0] if len(quoted) == 1 else f"{', '.join(quoted[:-1])} and {quoted[-1]}")
    return where + (f" (read with {read_with})" if read_with else "")


def _is_proper_noun(text: str, tok: re.Match, exempt=frozenset()) -> bool:
    """A capitalised word that could be part of a name: not at the start of a sentence, and not one that never is
    (_never_a_name)."""
    word = tok.group()
    if not word[0].isupper() or _never_a_name(word, exempt):
        return False
    before = text[:tok.start()].rstrip()
    if before.endswith(".") and _initial(re.split(r"\W", before[-5:-1])[-1]):
        return True  # a full stop after an initial or a short title doesn't start a sentence ("J. Okafor")
    sentence_start = not before or before[-1] in ".!?:;—–(\""
    return not sentence_start


# --------------------------------------------------------------------------------------
# Leak string check, shared by the exit checks.
# --------------------------------------------------------------------------------------

def find_forbidden(text: str, forbidden: set, allowed: set) -> list[str]:
    """Return every forbidden string present in `text` as a whole word, after first
    removing the allowed strings (longest first, so "smokes socially" being allowed
    doesn't make the hidden value "smokes" look like a leak)."""
    for s in sorted(allowed, key=len, reverse=True):
        text = re.sub(_word(s), " ", text, flags=re.I)
    return sorted(s for s in forbidden if re.search(_word(s), text, flags=re.I))


def _word(s: str) -> str:
    return r"(?<!\w)" + re.escape(s) + r"(?!\w)"


def _strings_of(listing: Listing, visibility: str) -> set:
    """Everything the listing's fields with this visibility say: values, and an open detail's name and
    steps too (hidden, those are as private as a value; shown, the person chose to show them)."""
    return {s for f in listing.fields if f.visibility == visibility for s in f.texts()}


def _view_text(view: dict, slots=None) -> str:
    """What a view says, one piece per line so no two pieces read as one phrase: the reason (or, given,
    what fills its sentences: slots, see reason_slots), open details' names, and every value and step. The view's own
    structure (its keys, "compare", true and false) is fixed by _view_schema_ok, and the reason's fixed
    wording by _TEMPLATES; neither says anything about anyone, so a hidden value that happens to be one of
    those words can't stop an introduction."""
    try:
        parts = [view["reason"]] if slots is None else list(slots)
        for k, v in view["fields"].items():
            if is_detail_key(k):
                parts += [split_key(k)[1], *v["values"], *v["steps"]]
            else:
                parts += [v] if isinstance(v, str) else list(v)
        return "\n".join(map(str, parts))
    except (KeyError, TypeError, AttributeError):
        return json.dumps(view, ensure_ascii=False)  # not a view's shape: read all of it


# --------------------------------------------------------------------------------------
# Reason text: written by the Warden from reason_fields only (SPEC §6: never in the reason).
# --------------------------------------------------------------------------------------

def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:]


def quoted_phrase(shown: dict) -> str:
    """How a question describes another listing: its headline ("selling 100 cardboard boxes"),
    or "looking for" plus its phrase when it has no usable headline."""
    headline = shown.get("headline")
    if isinstance(headline, str) and re.match(r"^[A-Za-z]+ing\b", headline):
        return _lower_first(headline.strip().rstrip("."))
    return f"looking for {shown.get('seeking')}"


def _quoted_from(shown: dict) -> str:
    """What quoted_phrase quotes, as its owner shows it (the headline, or the "looking for" phrase): what the
    string check reads, never the fixed "looking for"."""
    headline = shown.get("headline")
    if isinstance(headline, str) and re.match(r"^[A-Za-z]+ing\b", headline):
        return headline
    return str(shown.get("seeking"))


class _Said(str):
    """Part of a reason: its text, and the strings people wrote that fill it (its slots), as they appear on
    their listings. Everything else in it is the Warden's own fixed wording (_TEMPLATES), which says nothing
    about anyone, so the string checks read the slots and never mistake a fixed word ("both", "one", "online")
    for someone's value."""
    template: str = ""
    fills: tuple = ()
    slots: tuple = ()


def _filled(text: str, *fills) -> _Said:
    """Text made of the given fills (each a string someone wrote, or a _Said carrying its own slots)."""
    said = _Said(text)
    said.fills = fills
    said.slots = tuple(s for x in fills for s in (x.slots if isinstance(x, _Said) else (str(x),)))
    return said


def _as(text: str, source):
    """A value written differently in a sentence (another case, without its closing full stop): the slot is the
    value itself, as its owner shows it, so the checks read exactly what was shown. Anything more than that
    is its own slot, read as written."""
    shown = str(source)
    if text.casefold() not in (shown.casefold(), shown.strip().rstrip(".").casefold()):
        return text
    return _filled(text, source)


def _say(template: str, *fills) -> _Said:
    """One sentence: a fixed template (one of _TEMPLATES) with each {} filled in order."""
    said = _filled(template.format(*fills), *fills)
    said.template = template
    return said


# Every fixed sentence a reason is made of. Nothing else goes into a reason but the strings filling them.
_TEMPLATES = frozenset({
    "You're both in {}.", "You're in {}; they're in {}.", "You're near one another: {} and {}.",
    "You can both meet online.", "You can both meet in person.", "You'd meet {}; they'd meet {}.",
    "They're looking for {}.", "They're {}.",
    "You share the same {}: {}.", "Their {} is {}; yours is {}.",
    "You'd like their {} to be {}; theirs is {}.", "You'd like their {} to be {}.",
    "They'd like your {} to be {}; yours is {}.", "They'd like your {} to be {}.",
})


def _where_text(key: str, mine, theirs, countries=None) -> str:
    """Where they are, worded from both values so it is true whatever the Matchmaker cited.

    For a country: "both in" only for the same one. For two cities, `countries` is (my country,
    their country), both *shown*, or None unless both people show one: "both in" and "near one
    another" need the same shown country, then the same city, or two city centres (each read in
    that country) within the longest travel distance anyone can choose. Otherwise, just where each
    of them is: two cities called Cambridge in different countries are not "both in Cambridge".

    The checks read only the cities and countries filling it, never the fixed words around them."""
    if key == "country":
        if plain_place(mine) == plain_place(theirs):
            return _say("You're both in {}.", theirs)
        return _say("You're in {}; they're in {}.", mine, theirs)
    my_country, their_country = countries or (None, None)
    if my_country and their_country and plain_place(my_country) == plain_place(their_country):
        if plain_place(mine) == plain_place(theirs):
            return _say("You're both in {}.", theirs)
        km = km_between(mine, my_country, theirs, their_country)
        if km is not None and km < 1:
            return _say("You're both in {}.", theirs)  # two names for one city (Lisbon, Lisboa)
        if km is not None and km <= MAX_TRAVEL_KM:
            return _say("You're near one another: {} and {}.", mine, theirs)
    return _say("You're in {}; they're in {}.", mine, theirs)


def _format_text(mine, theirs) -> str:
    """How they can meet, worded from both formats so it is true whatever the Matchmaker cited."""
    m, t = meeting_format(mine), meeting_format(theirs)
    if m and t and "online" in m.lower() and "online" in t.lower():
        return _say("You can both meet online.")
    if m and t and "In person" in m and "In person" in t:
        return _say("You can both meet in person.")
    return _say("You'd meet {}; they'd meet {}.", _as(str(mine).lower(), mine), _as(str(theirs).lower(), theirs))


def _listed(values: list, last: str = "and") -> _Said:
    """Values in a sentence: "a", "a and b", "a, b and c" (or "or")."""
    text = values[0] if len(values) == 1 else f"{', '.join(values[:-1])} {last} {values[-1]}"
    return _filled(text, *values)


def _shared_values(mine: Field, theirs: Field) -> list:
    """The values two own details share, in their display form (as the other person wrote them)."""
    m = {canonical(v) for v in mine.strings()}
    return [v for v in theirs.strings() if canonical(v) in m]


def _detail_text(ref: str, mine: dict, theirs: dict, agreed=()) -> list:
    """An open detail's reason, worded from both people's shown details (mine, theirs: shown fields by
    key) with fixed text that names no activity, so it is true for any listing that shows it. Only
    worded when it holds: the Warden accepts a ref only if it equals schema.detail_reasons. agreed: the
    names an "agree:" ref is cited for, whose sentence already says both values, so a wish of that name
    only says what was wished."""
    kind, _, name = ref.partition(":")
    if kind == "agree":
        m, t = own(mine, name), own(theirs, name)
        if m is None or t is None:
            return []
        shared = _shared_values(m, t)
        if shared:  # the same value (for steps, the same step)
            return [_say("You share the same {}: {}.", name, _listed(shared))]
        if m.compare == "steps":
            return [_say("Their {} is {}; yours is {}.", name, _listed(t.strings()), _listed(m.strings()))]
        return []
    parts = []
    want, t = mine.get(detail_key("them", name)), own(theirs, name)
    if want is not None and want.visibility == SHOWN and t is not None and meets(want, t):
        wanted = _listed(want.strings(), "or")
        parts.append(_say("You'd like their {} to be {}.", name, wanted) if name in agreed else
                     _say("You'd like their {} to be {}; theirs is {}.", name, wanted, _listed(t.strings())))
    want, m = theirs.get(detail_key("them", name)), own(mine, name)
    if want is not None and want.visibility == SHOWN and m is not None and meets(want, m):
        wanted = _listed(want.strings(), "or")
        parts.append(_say("They'd like your {} to be {}.", name, wanted) if name in agreed else
                     _say("They'd like your {} to be {}; yours is {}.", name, wanted, _listed(m.strings())))
    return parts


def reason_parts(key_values: list[tuple[str, object, object]], countries=None) -> list:
    """The sentences of a reason, in order (see reason_text). Each is a _Said with its slots, unless something
    other than the templates here wrote it (then it is plain text, and the checks read all of it)."""
    parts = []
    # "They're looking for X" only describes the reader (that's why they were matched), so
    # say who they are instead whenever "offering" is a reason.
    has_offering = any(key == "offering" for key, _, _ in key_values)
    has_headline = any(key == "headline" for key, _, _ in key_values)
    agreed = {key.partition(":")[2] for key, _, _ in key_values if is_reason_ref(key) and key.startswith("agree:")}
    for key, mine, theirs in key_values:
        if is_reason_ref(key):
            parts += _detail_text(key, mine, theirs, agreed)
        elif key in ("city", "country"):
            parts.append(_where_text(key, mine, theirs, countries))
        elif key == "format":
            parts.append(_format_text(mine, theirs))
        elif key == "seeking" and not has_offering:
            parts.append(_say("They're looking for {}.", theirs))
        elif key == "offering" and not has_headline:
            parts.append(_say("They're {}.", theirs))
        elif key == "headline":
            parts.append(_say("They're {}.", _as(_lower_first(str(theirs).strip().rstrip(".")), theirs)))
    return [p for p in parts if p]


def reason_text(key_values: list[tuple[str, object, object]], countries=None) -> str:
    """key_values: (key, my_value, their_value) for each reason field, from the
    recipient's point of view; for an open-detail ref ("agree:{name}", "wants:{name}"), (ref, my shown
    fields, their shown fields), each by key. countries: (my country, their country) when both people
    *show* one, else None; it only decides how two cities are worded (_where_text) and is never written
    out. Uses nothing else."""
    return " ".join(reason_parts(key_values, countries))


def reason_slots(parts: list) -> list:
    """What the string checks read in a reason: the strings that fill its sentences, one each, as their owners
    show them. A sentence is read this way only if it is exactly one of the fixed templates filled with those
    strings; anything else is read whole."""
    out = []
    for p in parts:
        if isinstance(p, _Said) and p.template in _TEMPLATES and str(p) == p.template.format(*p.fills):
            out += p.slots
        else:
            out.append(str(p))
    return out


# --------------------------------------------------------------------------------------
# The Warden
# --------------------------------------------------------------------------------------

def listing_id_problem(listing: Listing, board: Marketboard) -> str:
    """Why this listing can't go on the board under its id ("" if it can): an id is one listing's for good, so it
    is never taken by another person's listing, and once its listing is deleted it is never accepted again (a new
    listing needs a new id). The Warden reissues an introduction from whatever the board holds under its two ids,
    so this is what keeps that the listing it was sent to."""
    lid = listing.person_id
    if not board.ever_held(lid):
        return ""
    if board.withdrawn(lid):
        return "that listing was deleted, so its id can't be used again; a new listing needs a new id"
    if board.listing(lid).owner != listing.owner:
        return "that listing id belongs to someone else's listing"
    return ""


# The switches on a listing that a person turns on to opt in and off to withdraw (SPEC §7): follow-up questions, close
# matches, and matching itself (a paused listing isn't matched).
SWITCHES = ("notify_questions", "notify_close", "active")


def only_turns_off(listing: Listing, board: Marketboard, owner: str) -> bool:
    """Does this update to a live listing only turn something off? Its owner's own listing, already on the board,
    and the update the same in everything (description, exactness, every field and its shown or hidden, confirmed
    sensitive details, owner) but a newer version, a time no earlier than the live one's, and its switches
    (SWITCHES): each one as it was or turned from on to off, at least one turned off, each a plain true or false.
    Such an update is never refused by the daily update limit (Warden.admit): it can only withdraw an opt-in or
    pause matching, never add or show anything, and each one needs something on to turn off, which was limited."""
    if listing.person_id not in board:
        return False
    live = board.listing(listing.person_id)
    whole = lambda n: isinstance(n, int) and not isinstance(n, bool)
    if live.owner != owner or listing.owner != owner or not (whole(listing.version) and whole(live.version)) \
            or listing.version <= live.version:
        return False
    try:
        if _parse(str(listing.updated_at)) < _parse(str(live.updated_at)):
            return False
    except (TypeError, ValueError):
        return False
    if not all(isinstance(getattr(listing, k), bool) for k in SWITCHES):
        return False
    same = {k: getattr(live, k) for k in SWITCHES}
    if replace(listing, updated_at=live.updated_at, version=live.version, **same) != live:
        return False  # something else changed
    turned_off = [k for k in SWITCHES if getattr(live, k) is True and getattr(listing, k) is False]
    return bool(turned_off) and all(k in turned_off or getattr(listing, k) == getattr(live, k) for k in SWITCHES)


@dataclass
class EntryResult:
    accepted: bool
    failed_checks: list = field(default_factory=list)
    problems: list = field(default_factory=list)


class Warden:
    def __init__(self, clock: Clock, key: bytes, approach_limit_per_day: int = 15,
                 updates_per_day: int = 5, max_intent_chars: int = 500, k_crowd: int = K_CROWD):
        self.clock = clock
        self._key = key
        self.approach_limit = approach_limit_per_day
        self.updates_per_day = updates_per_day
        self.max_intent_chars = max_intent_chars
        self.k_crowd = k_crowd
        self.receipts: list[Receipt] = []
        self._updates: dict[str, list[str]] = {}  # per listing
        self._owner_updates: dict[str, list[str]] = {}  # per person, across their listings
        self._details_in_use = None  # today's shared list, once built
        self._kept_places: dict[str, list] = {}  # live listing id -> the listed places it was last accepted with
        self._approaches: dict[str, list[str]] = {}
        self._decisions: dict[str, dict[str, bool]] = {}
        self._intros: dict[str, Introduction] = {}
        self._intro_owners: dict[str, tuple] = {}  # intro id -> the two listings' owners when it was sent
        self._next_intro = 0
        self._broadened: set[str] = set()  # listings already asked whether to include close matches
        # The SHA-256 digest of every request to the Matchmaker AI this Warden let through (exit_judge_request), so an
        # audit can compare the model's own record of what went out with it, however many requests there were.
        self.judge_passed: set[bytes] = set()
        self._watched = None  # (board, its change, _Watched): what no request to the Matchmaker AI may say

    # -- receipts --------------------------------------------------------------

    def _receipt(self, check: str, ok: bool, subject) -> bool:
        """Record a check result in the hash-chained receipt log (see Receipt)."""
        ts = self.clock.now()
        seq = len(self.receipts)
        result = "pass" if ok else "fail"
        body = json.dumps(subject, sort_keys=True, ensure_ascii=False, default=str)
        commitment = hmac.new(self._key, f"{seq}|{check}|{body}".encode(), hashlib.sha256).hexdigest()
        prev = self.receipts[-1].hash if self.receipts else GENESIS
        self.receipts.append(Receipt(seq, check, result, ts, commitment,
                                     chain_hash(prev, seq, check, result, ts, commitment)))
        return ok

    # -- ENTRY -----------------------------------------------------------------

    def admit(self, person: Person, listing: Listing, board: Marketboard, history: bool = False) -> EntryResult:
        """Entry check (SPEC §7). On success the listing goes onto the Marketboard, timed by the Warden's own clock:
        the time an agent writes on a listing is never trusted, so a backdated one can't skip the shared list's wait
        (LIST_MIN_AGE_DAYS). history: the operator's own synthetic history (the demo's and the simulation's seeded
        people, and test fixtures of an aged board), whose times stand; no person's or agent's entry is ever one."""
        subject = {"person": person.id, "version": listing.version}
        failed, problems = [], []

        def check(name, ok, problem=None):
            if not self._receipt(name, ok, subject):
                failed.append(name)
                if problem:
                    problems.append(problem)

        check("kyc_verified", person.verified, "person is not KYC-verified")
        schema_problems = self._schema_problems(person, listing)
        check("listing_schema", not schema_problems, "; ".join(schema_problems))
        # A listing id is one listing's for good (listing_id_problem): never another person's, never back once
        # deleted, so an introduction is only ever reissued from the listing it was sent to.
        taken = listing_id_problem(listing, board)
        check("listing_id_owner", not taken, taken)
        sens = [f.key for f in listing.fields
                if f.sensitive and f.visibility == SHOWN and f.key not in listing.confirmed_shown]
        check("sensitive_defaults", not sens, f"sensitive fields shown without confirmation: {sens}")
        # The rule covers the intent and the free-text fields ("seeking", "offering", ...: OWN_WORDS), each on its
        # own and, for the identifiers rule, what they say about the person sought all together (sought_words). The
        # full-name rule reads the listing's own shown city and country, and the places on the Warden's last shared
        # list, as never part of a name (never_names); a live listing of this person's also keeps the listed places
        # it was last accepted with, so a later list that drops a place never refuses its next update.
        live = listing.person_id in board and board.listing(listing.person_id).owner == person.id
        places = _places([*self.listed_places(), *(self.kept_places(listing.person_id) if live else [])])
        sp = listing_person_problems(listing.intent_text, listing.fields, places)
        said = [f"intent could identify one person ({', '.join(sp)})"] if sp else []
        # And the open details, by the description's rules (a wish could single someone out as well as a
        # description can), each problem naming the detail it is in.
        said += [f"{_in_details(names, read_with)} could identify one person ({problem})"
                 for names, read_with, problem in detail_person_findings(listing.fields, listing.intent_text)]
        check("specific_person", not said, "; ".join(said))
        # The crowd rules count people, so one person can't fill the board: a few live listings each
        # (an update to one of them isn't another).
        others = [l for l in board.listings() if l.owner == person.id and l.person_id != listing.person_id]
        check("listing_cap", len(others) < MAX_LISTINGS_PER_OWNER,
              f"at most {MAX_LISTINGS_PER_OWNER} listings per person")
        # Turning something off is never held back by the limit (consent first: an opt-in can always be withdrawn),
        # and isn't counted against it; it is checked and receipted like any update. Anything else is limited.
        check("update_rate_limit", only_turns_off(listing, board, person.id)
              or self._within_update_limit(listing.person_id, person.id), "too many listing updates today")

        if failed:
            return EntryResult(False, failed, problems)
        if not history:
            listing = replace(listing, updated_at=self.clock.now())
        board.put(person, listing)
        self._kept_places[listing.person_id] = places
        return EntryResult(True)

    def _schema_problems(self, person: Person, listing: Listing) -> list[str]:
        out = []
        if listing.owner != person.id:
            out.append("listing belongs to a different person")
        if not isinstance(listing.intent_text, str) or not 0 < len(listing.intent_text) <= self.max_intent_chars:
            out.append("intent text missing or too long")
        if not (isinstance(listing.exactness, (int, float)) and 0.0 <= listing.exactness <= 1.0):
            out.append("exactness must be between 0 and 1")
        kind = getattr(listing.get("kind"), "value", None)
        seen = set()
        for f in listing.fields:
            if is_detail_key(f.key):
                continue  # open details: all checked together below, by the one rule
            spec = FIELD_REGISTRY.get(f.key)
            if spec is None:
                out.append(f"unknown field {f.key!r}")
                continue
            if f.key in seen:
                out.append(f"duplicate field {f.key!r}")
            seen.add(f.key)
            if f.visibility not in (SHOWN, HIDDEN):
                out.append(f"bad visibility on {f.key!r}")
            if f.sensitive != spec.sensitive:
                out.append(f"{f.key!r} must be marked sensitive={spec.sensitive}")
            if f.compare or f.steps:
                out.append(f"{f.key!r} is a built-in field: it has no compare or steps")
            ok_type = (isinstance(f.value, tuple) and all(isinstance(v, str) for v in f.value)) if spec.multi \
                else isinstance(f.value, str)
            if not ok_type:
                out.append(f"bad value type on {f.key!r}")
            elif f.key in FREE_TEXT_FIELDS and not 0 < len(f.value) <= FREE_TEXT_MAX:
                out.append(f"{f.key!r} must be 1-{FREE_TEXT_MAX} characters")
            # The hidden categories take any words ("Sikh", "asthma"): short plain text, no contact details.
            elif f.key in PRIVATE_FIELDS and (not f.strings() or any(text_problems(s, PRIVATE_VALUE_MAX)
                                                                     for s in f.strings())):
                out.append(f"{f.key!r}: each value must be plain text of 1-{PRIVATE_VALUE_MAX} characters, "
                           f"with no contact details")
            elif f.key == "travel" and f.value not in TRAVEL:
                out.append(f"'travel' must be one of: {', '.join(TRAVEL)}")
            elif f.key == "format" and f.value not in FORMATS:
                out.append(f"'format' must be one of: {', '.join(FORMATS)}")
            # What the person's agent understood: fixed formats only, so matching reads exactly what the
            # person saw on their profile, and no listing can ask for anyone under 18.
            elif f.key == "kind" and f.value not in MATCH_KINDS:
                out.append(f"'kind' must be one of: {', '.join(MATCH_KINDS)}")
            elif f.key == "wants_gender" and not (f.value and set(f.value) <= set(GENDERS)
                                                  and len(set(f.value)) == len(f.value)):
                out.append(f"'wants_gender' must be one or more of: {', '.join(GENDERS)}")
            elif f.key == "wants_age" and age_span(f.value) is None:
                out.append(f"'wants_age' must be an age range in years, written lo-hi, from {AGE_MIN} to {AGE_MAX}")
            # Intimacy preferences only on dating listings, outside dating's rules on who meets whom.
            if f.key == "kinks" and kind != "dating":
                out.append("'kinks' is only for dating listings")
        out += detail_problems(listing.fields, kind)
        out += shown_claim_problems(listing.fields)  # "ID verified" in a shown headline or phrase
        age = listing.get("age_band")
        if age is not None and age.value != person.age_band:
            out.append("age_band must match the KYC age band")
        if age is not None and age.value not in AGE_BANDS:
            out.append("unknown age band")
        return out

    def _within_update_limit(self, listing_id: str, owner: str) -> bool:
        """Updates in the last 24 hours, per listing and per person (as many as all their listings may have)."""
        return all([self._count_update(self._updates, listing_id, self.updates_per_day),
                    self._count_update(self._owner_updates, owner, MAX_LISTINGS_PER_OWNER * self.updates_per_day)])

    def _count_update(self, log: dict, key: str, limit: int) -> bool:
        now = _parse(self.clock.now())
        recent = [t for t in log.get(key, []) if now - _parse(t) < timedelta(days=1)]
        recent.append(self.clock.now())
        log[key] = recent
        return len(recent) <= limit

    # -- EXIT: introductions ---------------------------------------------------

    def exit_introduction(self, proposal: Proposal, board: Marketboard):
        """Turn an internal Proposal into an Introduction, or drop it (returns None).

        The Introduction is built here, by the Warden, from shown fields only; the
        Matchmaker contributes nothing but the pair and the reason field keys (and its open-detail refs
        must be exactly the ones the Warden works out itself, so it has nothing to choose).
        """
        if not self._receipt("proposal_well_formed", _well_formed(proposal),
                             {"a": str(getattr(proposal, "person_a", ""))[:64], "b": str(getattr(proposal, "person_b", ""))[:64]}):
            return None  # the contract's shape, exactly: plain booleans and finite numbers, never "false" or NaN
        a_id, b_id = proposal.person_a, proposal.person_b
        subject = {"a": a_id, "b": b_id}
        close = proposal.close
        if close:
            # A close match: the other person's intent must be fully met, and it goes one-way to a
            # listing that opted in to close matches.
            other_met = proposal.b_intent_met if close == a_id else proposal.a_intent_met
            mutual = close in (a_id, b_id) and other_met
        else:
            mutual = proposal.a_intent_met and proposal.b_intent_met
        if not self._receipt("proposal_mutual", mutual, subject):
            return None
        known = a_id != b_id and a_id in board and b_id in board
        if not self._receipt("proposal_known_people", known, subject):
            return None
        if close and not self._receipt("close_match_opted_in", board.listing(close).notify_close, subject):
            return None
        a, b = board.listing(a_id), board.listing(b_id)
        if not self._receipt("listings_active", a.active and b.active, subject):
            return None  # a paused listing gets no new introductions
        # A person can have several listings; they are never introduced to themselves.
        if not self._receipt("proposal_distinct_people", a.owner != b.owner, subject):
            return None
        a_shown, b_shown = a.shown(), b.shown()

        # Built-in reasons: shown by both. Open-detail refs: exactly the ones both people's shown
        # details give (a subset, or a ref to a hidden detail, could carry a hidden value out).
        reasons = list(proposal.reason_fields)
        both_show = bool(reasons) and all(k in FIELD_REGISTRY and k in a_shown and k in b_shown
                                          for k in reasons if not is_reason_ref(k))
        if not self._receipt("reason_fields_shown", both_show, subject):
            return None
        refs = [k for k in reasons if is_reason_ref(k)]
        if not self._receipt("reason_details_true", tuple(sorted(refs)) == _detail_refs(a, b), subject):
            return None
        # A hidden value may only have raised the score if K_CROWD other people share it.
        counts = board.value_counts() if proposal.used_hidden_fields else None
        crowded = all(_hidden_in_crowd(k, a, b, counts, self.k_crowd) for k in proposal.used_hidden_fields)
        if not self._receipt("hidden_crowd", crowded, subject):
            return None

        under_limit = all(self._approaches_today(p) < self.approach_limit for p in (a_id, b_id))
        if not self._receipt("approach_limit", under_limit, subject):
            return None

        views = self._checked_views(a, b, reasons, subject)
        if views is None:
            return None
        a_view, b_view = views

        intro = Introduction(_SEAL, f"i{self._next_intro:04d}", a_id, b_id, a_view, b_view, one_way_to=close,
                             sent_at=self.clock.now())
        self._next_intro += 1
        self._intros[intro.intro_id] = intro
        self._intro_owners[intro.intro_id] = (a.owner, b.owner)
        for p in (a_id, b_id):
            self._approaches.setdefault(p, []).append(self.clock.now())
        return intro

    def _checked_views(self, a: Listing, b: Listing, reasons: list, subject: dict):
        """Build both people's views from shown fields only, and check them. None if any check fails.

        The reason's string checks read what fills its sentences (reason_slots: names, values, steps, cities,
        phrases, as their owners show them), one per line, against the same allowed and forbidden strings as
        always: the Warden's own fixed wording says nothing about anyone. The fields are read in full."""
        a_shown, b_shown = a.shown(), b.shown()
        a_fields, b_fields = a.shown_fields(), b.shown_fields()
        # Both shown countries, or none: a hidden country never shapes how two cities are worded.
        both = all(isinstance(s.get("country"), str) for s in (a_shown, b_shown))
        a_countries = (a_shown["country"], b_shown["country"]) if both else None
        b_countries = (b_shown["country"], a_shown["country"]) if both else None

        def said(mine, theirs, my_fields, their_fields):
            return [(k, my_fields, their_fields) if is_reason_ref(k) else (k, mine[k], theirs[k]) for k in reasons]
        a_parts = reason_parts(said(a_shown, b_shown, a_fields, b_fields), a_countries)
        b_parts = reason_parts(said(b_shown, a_shown, b_fields, a_fields), b_countries)
        a_view = {"fields": {k: field_to_json(f) for k, f in b_fields.items()}, "reason": " ".join(a_parts)}
        b_view = {"fields": {k: field_to_json(f) for k, f in a_fields.items()}, "reason": " ".join(b_parts)}
        ok = True
        for viewer, other, view, parts in ((a, b, a_view, a_parts), (b, a, b_view, b_parts)):
            slots = reason_slots(parts)
            ok &= self._receipt("introduction_schema", _view_schema_ok(view), subject)
            ok &= self._receipt("fields_shown", _fields_all_shown(view, other), subject)
            ok &= self._receipt("reason_refs_only_reason_fields",
                                _reason_uses_only("\n".join(slots), reasons, viewer, other), subject)
            ok &= self._receipt("no_hidden_values", not _hidden_in_view(view, viewer, other, reasons, slots), subject)
        return (a_view, b_view) if ok else None

    def _reduced_views(self, a: Listing, b: Listing, subject: dict, keep=(None, None)) -> tuple:
        """When a reissue fails its checks, what each person may still see: no reason, and only the other's
        shown fields that each pass every exit check on their own, checked again together. The old views never
        stand, so nothing since hidden stays on anyone's screen (at worst the views are empty).

        keep: for each view (a's, b's), None, or the fields it already had: then it keeps only those the other
        still shows exactly as they were, so nothing new reaches someone whose listing is gone."""
        views = []
        for (viewer, other), kept in zip(((a, b), (b, a)), keep):
            fields = {}
            for k, f in other.shown_fields().items():
                one = {"fields": {k: field_to_json(f)}, "reason": ""}
                if kept is not None and kept.get(k) != one["fields"][k]:
                    continue  # new, or changed, since they left
                if _view_schema_ok(one) and _fields_all_shown(one, other) \
                        and not _hidden_in_view(one, viewer, other, []):
                    fields[k] = one["fields"][k]
            views.append({"fields": fields, "reason": ""})
        ok = True
        for viewer, other, view in ((a, b, views[0]), (b, a, views[1])):
            ok &= self._receipt("introduction_schema", _view_schema_ok(view), subject)
            ok &= self._receipt("fields_shown", _fields_all_shown(view, other), subject)
            ok &= self._receipt("no_hidden_values", not _hidden_in_view(view, viewer, other, []), subject)
        if not ok:
            views = [{"fields": {}, "reason": ""}, {"fields": {}, "reason": ""}]
        self._receipt("reissue_reduced", True, {**subject, "fields": [sorted(v["fields"]) for v in views]})
        return views[0], views[1]

    def refresh_introduction(self, intro_id: str, reasons: list, board: Marketboard):
        """Reissue an introduction after either person updated their listing, so each sees the
        other's current shown details (a new first name, a city they now show) and nothing they've
        since hidden. Same id, decisions and one-way recipient; the same checks as at first. Reason
        fields either person no longer shows are dropped, and the open-detail reasons are the ones
        their shown details give now. If those checks fail, the reissue is reduced (_reduced_views: no
        reason, only fields that pass every check on their own), never the old one left standing.

        If one of the two listings has been deleted, the other can still change, so the reissue is always
        reduced: the deleted listing's view of the other keeps only what it already had and the other still shows
        unchanged (nothing new reaches someone who left), and the other's view keeps the deleted listing's fields
        as they were when it was deleted, each only where it passes every exit check, with no reason. So no stored
        introduction keeps something since hidden, whoever left first. A listing id is never another person's and
        never comes back once deleted (listing_id_problem); were either listing's owner ever not the one it was
        sent from, both views would be emptied. Returns the new Introduction, or None if the introduction is
        unknown or both listings are gone."""
        old = self._intros.get(intro_id)
        gone = [p for p in (old.person_a, old.person_b) if p not in board] if old is not None else []
        if old is None or len(gone) == 2:
            return None
        a, b = board.listing(old.person_a), board.listing(old.person_b)  # a deleted one as it was when deleted
        subject = {"intro": intro_id, "refresh": True}
        if (a.owner, b.owner) != self._intro_owners.get(intro_id, (a.owner, b.owner)):
            empty = ({"fields": {}, "reason": ""}, {"fields": {}, "reason": ""})
            self._receipt("reissue_reduced", True, {**subject, "fields": [[], []]})
            return self._reissued(old, empty)
        a_shown, b_shown = a.shown(), b.shown()
        reasons = [k for k in reasons if not is_reason_ref(k) and k in FIELD_REGISTRY and k in a_shown
                   and k in b_shown] + list(_detail_refs(a, b))
        keep = tuple(view["fields"] if p in gone else None
                     for p, view in ((old.person_a, old.a_view), (old.person_b, old.b_view)))
        views = (None if gone else self._checked_views(a, b, reasons, subject)) \
            or self._reduced_views(a, b, subject, keep)
        return self._reissued(old, views)

    def _reissued(self, old: Introduction, views) -> Introduction:
        intro = Introduction(_SEAL, old.intro_id, old.person_a, old.person_b, views[0], views[1],
                             one_way_to=old.one_way_to, sent_at=old.sent_at)  # a refresh, not a new approach
        self._intros[old.intro_id] = intro
        return intro

    def withdraw(self, person, listing_id: str, board: Marketboard) -> bool:
        """The owner deletes a listing: it leaves the board and is never matched again, and its id is never accepted
        again (listing_id_problem)."""
        ok = listing_id in board and board.listing(listing_id).owner == person.id
        if self._receipt("listing_withdrawn", ok, {"listing": listing_id}):
            board.withdraw(listing_id)
            self._kept_places.pop(listing_id, None)
        return ok

    def _approaches_today(self, person_id: str) -> int:
        """Introductions to this listing in the last 24 hours."""
        now = _parse(self.clock.now())
        return sum(1 for t in self._approaches.get(person_id, []) if now - _parse(t) < timedelta(days=1))

    # -- EXIT: follow-up questions ---------------------------------------------

    def exit_followup(self, followup: FollowUp, board: Marketboard):
        ok = followup.question_id in FOLLOWUP_MENU and followup.person_id in board
        if not self._receipt("followup_menu", ok, {"p": followup.person_id, "q": followup.question_id}):
            return None
        if not self._receipt("listing_active", board.listing(followup.person_id).active, {"p": followup.person_id}):
            return None  # a paused listing gets no questions
        if followup.question_id == "broaden":
            return self._exit_broaden(followup, board)
        # Questions from Belong only reach people who opted in to them.
        if not self._receipt("followup_opted_in", board.listing(followup.person_id).notify_questions,
                             {"p": followup.person_id}):
            return None
        if followup.question_id != "also_offer":
            ok = not followup.about
            return FollowUpOut(_SEAL, followup.person_id, followup.question_id) if self._receipt(
                "followup_no_subject", ok, {"p": followup.person_id}) else None
        # "Could you also help with X?": X is the shown "looking for" phrase of another person's
        # listing, which opted in to close matches. The Warden writes it; the Matchmaker only names
        # the listing, and the recipient is never told whose it is.
        subject = {"p": followup.person_id, "about": followup.about}
        me = board.listing(followup.person_id)
        them = board.listing(followup.about) if followup.about in board else None
        ok = them is not None and them.owner != me.owner and them.notify_close
        if not self._receipt("also_offer_requester_opted_in", ok, subject):
            return None
        shown_them = them.shown()
        seeking, offering = shown_them.get("seeking"), shown_them.get("offering")
        ok = all(isinstance(x, str) and bool(x.strip()) for x in (seeking, offering))
        if not self._receipt("also_offer_phrase_shown", ok, subject):
            return None
        phrase = quoted_phrase(shown_them)
        pay = shown_them.get("pay") if isinstance(shown_them.get("pay"), str) else ""
        forbidden = _strings_of(me, HIDDEN) | _strings_of(them, HIDDEN)
        said = "\n".join([_quoted_from(shown_them), seeking, offering, pay])  # what people wrote, one per line
        clean = not find_forbidden(said, forbidden, _strings_of(them, SHOWN))
        if not self._receipt("no_hidden_values", clean, subject):
            return None
        return FollowUpOut(_SEAL, followup.person_id, "also_offer", phrase, seeking, offering, pay)

    def _exit_broaden(self, followup: FollowUp, board: Marketboard):
        """"Want to include close matches?" is how a person opts in, so it is the one question that
        doesn't need the opt-in to questions. Instead: about nobody, only while close matches are
        off, only to a listing with no match yet, and only ever once per listing."""
        pid = followup.person_id
        subject = {"p": pid}
        if not self._receipt("followup_no_subject", not followup.about, subject):
            return None
        if not self._receipt("broaden_still_off", not board.listing(pid).notify_close, subject):
            return None
        matched = any(pid in (i.person_a, i.person_b) for i in self._intros.values())
        if not self._receipt("broaden_no_match_yet", not matched, subject):
            return None
        if not self._receipt("broaden_once", pid not in self._broadened, subject):
            return None
        self._broadened.add(pid)
        return FollowUpOut(_SEAL, pid, "broaden")

    # -- Decisions and mutual reveal -------------------------------------------

    def record_decision(self, intro_id: str, person_id: str, yes: bool) -> None:
        intro = self._intros[intro_id]
        if person_id not in (intro.person_a, intro.person_b):
            raise ValueError("only the two people in an introduction can answer it")
        if intro.one_way_to and person_id != intro.one_way_to and \
                self._decisions.get(intro_id, {}).get(intro.one_way_to) is not True:
            raise PermissionError("a close match reaches the other person only after its recipient says yes")
        self._decisions.setdefault(intro_id, {})[person_id] = yes

    def is_connected(self, intro_id: str) -> bool:
        d = self._decisions.get(intro_id, {})
        intro = self._intros.get(intro_id)
        if intro is None:
            return False
        return d.get(intro.person_a) is True and d.get(intro.person_b) is True

    def exit_reveal(self, intro_id: str, consent_a: set, consent_b: set, board: Marketboard) -> list:
        """Reveal hidden details the two share, only where BOTH opted in (SPEC §6): a built-in or an
        open detail about the person or the request (never a wish about the other person), hidden by
        at least one of them, with a value both have (compared canonically, shown as A wrote it)."""
        intro = self._intros.get(intro_id)
        subject = {"intro": intro_id}
        if not self._receipt("reveal_connected", self.is_connected(intro_id), subject):
            return []
        a, b = board.listing(intro.person_a), board.listing(intro.person_b)
        out = []
        for key in sorted(set(consent_a) & set(consent_b)):
            fa, fb = a.get(key), b.get(key)
            shared = _shared(fa, fb) if fa and fb and split_key(key)[0] != "them" else []
            hidden = bool(fa and fb) and HIDDEN in (fa.visibility, fb.visibility)
            if self._receipt("reveal_consent_and_shared", bool(shared) and hidden, {**subject, "key": key}):
                out.append(Reveal(_SEAL, intro.person_a, intro.person_b, key, shared))
        return out

    # -- EXIT: the shared list of details in use ---------------------------------

    def details_in_use(self, board: Marketboard) -> DetailsInUse:
        """The shared list every person's agent sees (SPEC §6), so agents reuse the same names, scales,
        values, phrases and places. A pure function of the board and the Warden's clock: nothing from
        the Matchmaker. Built at most once per day (the same list all day, whatever changes), with a
        receipt committing to all of it.

        On it: open details (name, about, compare), their scales and values, the phrases people look
        for and offer and their topics, and places (city with country), each only once K_CROWD different
        people show it, counted only from active listings unchanged for LIST_MIN_AGE_DAYS. Never
        anything hidden or sensitive (and so never the hidden built-ins), never a sensitive word,
        contact details or anything but short plain text of a few words with no sentence punctuation
        (_listable). Alphabetical, no counts; past a section's cap
        (LIST_CAPS), the most used are kept. Each item in the form most of its people write it."""
        now = self.clock.now()
        if self._details_in_use is not None and self._details_in_use.built_at[:10] == now[:10]:
            return self._details_in_use
        lists = _list_items(board.listings(), _parse(now), self.k_crowd)
        made = DetailsInUse(_SEAL, now, board.changes, **lists)
        self._receipt("details_in_use", True, made.to_dict())
        self._details_in_use = made
        return made

    def exit_judge_request(self, request, labels, board: Marketboard) -> bool:
        """EXIT (SPEC §8.3): one request to the Matchmaker AI, just before it is handed to the model's client, read
        against the board as it is now (judge_request_problems). labels: the Matchmaker's own record of which listing
        each label stands for (None if it kept no such request: then it doesn't go). A label that names the wrong
        listing gains nothing: every entry must be that listing's own shown text. Writes a receipt; True if it may go,
        and then its digest is kept (judge_passed)."""
        problems = judge_request_problems(request, labels, board, self._watched_on(board))
        subject = {"request": request, "labels": labels}
        ok = self._receipt("judge_request", not problems, subject)
        if ok:
            self.judge_passed.add(hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode())
                                  .digest())
        return ok

    def _watched_on(self, board: Marketboard) -> "_Watched":
        got = self._watched
        if got is None or got[0] is not board or got[1] != board.changes:
            got = self._watched = (board, board.changes, _Watched(board))
        return got[2]

    def listed_places(self) -> list:
        """The places on the last shared list this Warden built (details_in_use), for the full-name rule at entry: a
        place K_CROWD different people show is never part of a name (never_names). It builds nothing (so entry adds
        no receipt of its own): before the first list, there are none. Never sent anywhere."""
        return self._details_in_use.vocabulary()["places"] if self._details_in_use is not None else []

    def kept_places(self, listing_id: str) -> list:
        """The listed places a live listing was last accepted with (listed_places then, and those it kept before), which
        the full-name rule still reads for that listing's next update: so a later list that drops a place never refuses
        an unchanged description when its owner pauses the listing, changes a notification choice or their name.
        Dropped when the listing is withdrawn. Never sent anywhere."""
        return list(self._kept_places.get(listing_id, []))


# --------------------------------------------------------------------------------------
# Exit-check helpers. Each one is a single rule from SPEC §7.
# --------------------------------------------------------------------------------------

def _places(entries) -> list:
    """Shared-list place entries without repeats (a city with its country), in order."""
    out = {}
    for e in entries:
        if isinstance(e, dict):
            out.setdefault((e.get("city"), e.get("country")), e)
    return list(out.values())


def _view_schema_ok(view) -> bool:
    """Output matches the Introduction schema exactly: built-in fields as values, open details as
    {values, compare, steps, required}, under well-formed keys."""
    if not isinstance(view, dict) or set(view) != {"fields", "reason"}:
        return False
    if not isinstance(view["reason"], str) or not isinstance(view["fields"], dict):
        return False
    strings = lambda v: isinstance(v, list) and all(isinstance(x, str) for x in v)
    for k, v in view["fields"].items():
        if k in FIELD_REGISTRY:
            if not (isinstance(v, str) or strings(v)):
                return False
        elif is_detail_key(k) and not name_problems(split_key(k)[1]):
            if not (isinstance(v, dict) and set(v) == {"values", "compare", "steps", "required"}
                    and strings(v["values"]) and strings(v["steps"]) and v["compare"] in COMPARES
                    and isinstance(v["required"], bool)):
                return False
        else:
            return False
    return True


def _fields_all_shown(view: dict, owner: Listing) -> bool:
    """Every field in the view is marked shown by its owner, with the owner's value (and, for an open
    detail, the owner's compare, steps and required)."""
    for k, v in view["fields"].items():
        f = owner.get(k)
        if f is None or f.visibility != SHOWN or field_to_json(f) != v:
            return False
    return True


def _detail_refs(a: Listing, b: Listing) -> tuple:
    """The open-detail reasons two listings' shown details give (schema.detail_reasons)."""
    return detail_reasons(a.shown_fields(), b.shown_fields(), max(a.exactness, b.exactness) >= STRICT_EXACTNESS)


def _cited_keys(reasons: list, listing: Listing) -> set:
    """The keys of the listing's shown fields that the reasons cite: built-in keys, and for an open-detail
    ref its own value of that name (and, for "wants:", its wish too)."""
    shown = listing.shown_fields()
    keys = set()
    for k in reasons:
        if is_reason_ref(k):
            kind, _, name = k.partition(":")
            keys |= {key for about in ("me", "both", "them") if about != "them" or kind == "wants"
                     for key in [detail_key(about, name)] if key in shown}
        elif k in shown:
            keys.add(k)
    return keys


def _reason_uses_only(reason: str, reasons: list, viewer: Listing, other: Listing) -> bool:
    """Reason text (as the Warden checks it: what fills its sentences, see reason_slots) mentions nothing from
    any field outside the cited (shown) fields."""
    cited = {l.person_id: _cited_keys(reasons, l) for l in (viewer, other)}
    allowed = {s for l in (viewer, other) for k in cited[l.person_id] for s in l.get(k).texts()}
    outside = {s for l in (viewer, other) for f in l.fields if f.key not in cited[l.person_id] for s in f.texts()}
    return not find_forbidden(reason, outside - allowed, allowed)


def _hidden_in_view(view: dict, viewer: Listing, other: Listing, reasons: list, slots=None) -> list:
    """String check: nothing hidden by either person (a value, or an open detail's name or steps)
    appears anywhere in the view. slots: what fills the reason's sentences (see reason_slots), read in
    place of its fixed wording; None reads the whole reason."""
    forbidden = _strings_of(viewer, HIDDEN) | _strings_of(other, HIDDEN)
    allowed = _strings_of(other, SHOWN)  # the counterpart chose to show these
    allowed |= {s for k in _cited_keys(reasons, viewer) for s in viewer.get(k).texts()}
    return find_forbidden(_view_text(view, slots), forbidden, allowed)


def _hidden_in_crowd(key: str, a: Listing, b: Listing, counts, k_crowd: int) -> bool:
    """A hidden field may have raised the score: both listings have it, at least one hides it, it isn't
    a wish about the other person, and a value they share is had by k_crowd people besides the two of
    them (counts: distinct people per (key, canonical value), Marketboard.value_counts)."""
    fa, fb = a.get(key), b.get(key)
    if fa is None or fb is None or split_key(key)[0] == "them" or HIDDEN not in (fa.visibility, fb.visibility):
        return False
    shared = {canonical(v) for v in fa.strings()} & {canonical(v) for v in fb.strings()}
    return any(counts[(key, v)] >= k_crowd + 2 for v in shared)


def _shared(fa: Field, fb: Field) -> list:
    """The values two fields share, compared canonically, as A wrote them."""
    theirs = {canonical(v) for v in fb.strings()}
    mine = {}
    for v in fa.strings():
        if canonical(v) in theirs:
            mine.setdefault(canonical(v), v)
    return sorted(mine.values())


# --------------------------------------------------------------------------------------
# The shared list of details in use: counted from the board by fixed rules (SPEC §6).
# --------------------------------------------------------------------------------------

def _list_items(listings: list, now: datetime, k_crowd: int) -> dict:
    """Every item that goes on the shared list, by section: counted by distinct people, from shown,
    non-sensitive fields of active listings unchanged for LIST_MIN_AGE_DAYS."""
    cutoff = now - timedelta(days=LIST_MIN_AGE_DAYS)
    people: dict = {}  # item -> {owner: the forms they write it in}

    def add(item, owner, form):
        people.setdefault(item, {}).setdefault(owner, set()).add(form)

    for l in listings:
        if not l.active or not _unchanged_since(l.updated_at, cutoff):
            continue
        shown = l.shown_fields()
        for f in shown.values():
            if f.sensitive:
                continue
            if is_detail_key(f.key):
                about, name = split_key(f.key)
                add(("details", about, name, f.compare), l.owner, (about, name, f.compare))
                if f.compare == "steps":
                    add(("scales", about, name, tuple(canonical(s) for s in f.steps)), l.owner, tuple(f.steps))
                for v in f.strings():
                    add(("values", about, name, canonical(v)), l.owner, v)
            elif f.key in LIST_PHRASE_KEYS:
                for v in f.strings():
                    add(("phrases", f.key, canonical(v)), l.owner, v)
        city, country = shown.get("city"), shown.get("country")
        if city and country and isinstance(city.value, str) and isinstance(country.value, str):
            add(("places", canonical(city.value), canonical(country.value)), l.owner, (city.value, country.value))

    chosen = {}
    for item, owners in people.items():
        if len(owners) < k_crowd:
            continue
        forms = {}
        for written in owners.values():
            for form in written:
                forms[form] = forms.get(form, 0) + 1
        form = min(forms, key=lambda f: (-forms[f], f))  # the most common form; ties alphabetical
        words = LIST_PHRASE_WORDS if item[0] == "phrases" else LIST_VALUE_WORDS
        if all(_listable(t, words) for t in _list_texts(item[0], form)):
            chosen[item] = (len(owners), form)

    out = {}
    for section, kinds in (("details", ("details", "scales", "values")), ("phrases", ("phrases",)),
                           ("places", ("places",))):
        kept = sorted((i for i in chosen if i[0] in kinds), key=lambda i: (-chosen[i][0], i))[:LIST_CAPS[section]]
        for kind in kinds:
            out[kind] = [_list_entry(i, chosen[i][1]) for i in sorted(i for i in kept if i[0] == kind)]
    return out


def _well_formed(p) -> bool:
    """Is this Proposal the contract's shape, exactly (security review F08)? Ids and close as text, the intent flags
    plain booleans (a string "false" is truthy), the scores finite numbers that aren't booleans, and the reason and
    hidden keys tuples or lists. Its shape only: whether the pair suits each other is never the Warden's to judge."""
    number = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)
    try:
        return (isinstance(p, Proposal) and all(isinstance(x, str) for x in (p.person_a, p.person_b, p.close))
                and all(type(x) is bool for x in (p.a_intent_met, p.b_intent_met))
                and all(number(x) for x in (p.score, p.shown_score, p.shown_threshold))
                and all(isinstance(x, (tuple, list)) for x in (p.reason_fields, p.used_hidden_fields)))
    except AttributeError:
        return False


def _unchanged_since(updated_at, cutoff: datetime) -> bool:
    try:
        return _parse(str(updated_at)) <= cutoff
    except (ValueError, TypeError):
        return False


def _list_texts(kind: str, form) -> list:
    """The words an item puts on the list (its fixed parts, about and compare, aside)."""
    return [form[1]] if kind == "details" else [form] if isinstance(form, str) else list(form)


def _listable(text: str, words: int) -> bool:
    """Short plain text with no contact details and no sensitive word, shaped like what it stands for: at most
    `words` words (LIST_PHRASE_WORDS for a phrase, LIST_VALUE_WORDS for anything else, counted by _list_words) and
    no sentence punctuation (hyphens and apostrophes are fine). Every agent reads the list, so this, the crowd of
    K_CROWD people on listings a week old, and the data line agents get it after (LIST_PREFIX) are what keep it
    from carrying instructions; no fixed rule can tell every instruction from a phrase."""
    return (isinstance(text, str) and len(text) <= LIST_TEXT_MAX and not text_problems(text, LIST_TEXT_MAX)
            and not mentions_sensitive(text) and _list_words(text) <= words and not any(map(_ends_a_clause, text)))


def _list_words(text: str) -> int:
    """How many words the shape rule counts: runs of letters and digits, read after compatibility decomposition (NFKD:
    a circled, squared or full-width letter is its letter; a mark stays with its letter; anything else separates
    them, and a lower-case letter followed by a capital starts a new run). A symbol that is a letter with no
    decomposition (a negatively squared letter, a regional indicator: its Unicode name says LETTER) counts as a
    letter too. In a script written without spaces (LIST_UNSPACED_SCRIPTS) a run counts one word for every
    LIST_UNSPACED_CHARS letters, rounded up; in any other, one word up to LIST_LONG_RUN letters and one word for
    every LIST_LONG_RUN_CHARS letters past that length, rounded up."""
    words, letters, unspaced, last = 0, 0, False, ""
    for c in unicodedata.normalize(_LIST_FORM, text) + " ":
        kind = "Lo" if _letter_symbol(c) else unicodedata.category(c)
        if kind[0] in "LN" and letters and last == "Ll" and kind in ("Lu", "Lt"):
            words, letters, unspaced = words + _run_words(letters, unspaced), 0, False  # CamelCase
        if kind[0] in "LN":
            letters += 1
            unspaced = unspaced or unicodedata.name(c, "").startswith(LIST_UNSPACED_SCRIPTS)
            last = kind
        elif kind[0] == "M" and letters:
            continue  # a mark on the letter before it
        elif letters:
            words, letters, unspaced, last = words + _run_words(letters, unspaced), 0, False, ""
    return words


_LIST_FORM = "NFKD"  # how the shape rule reads a list text: decomposed for compatibility (a circled letter is its letter)


def _letter_symbol(c: str) -> bool:
    """A letter written as a symbol with no decomposition (a negatively squared letter, a regional indicator): its
    Unicode name says it is a LETTER. The shape rule counts it as a letter."""
    return unicodedata.category(c) == "So" and " LETTER " in unicodedata.name(c, "")


def _run_words(letters: int, unspaced: bool) -> int:
    if unspaced:
        return -(-letters // LIST_UNSPACED_CHARS)
    return 1 if letters <= LIST_LONG_RUN else -(-letters // LIST_LONG_RUN_CHARS)


def _ends_a_clause(c: str) -> bool:
    """Sentence punctuation in any script: a full stop, comma, colon, semicolon, question or exclamation mark."""
    return unicodedata.category(c).startswith("P") and any(
        w in unicodedata.name(c, "") for w in ("FULL STOP", "COMMA", "COLON", "QUESTION MARK", "EXCLAMATION MARK"))


def _list_entry(item: tuple, form) -> dict:
    kind = item[0]
    if kind == "details":
        return {"about": item[1], "name": item[2], "compare": item[3]}
    if kind == "scales":
        return {"about": item[1], "name": item[2], "steps": list(form)}
    if kind == "values":
        return {"about": item[1], "name": item[2], "value": form}
    if kind == "phrases":
        return {"key": item[1], "value": form}
    return {"city": form[0], "country": form[1]}


# --------------------------------------------------------------------------------------
# EXIT: requests to the Matchmaker AI (exit_judge_request). Its own rules, read against the board as it is now: a
# request built a while ago (the demo hands them over off its lock) goes only if it is still true of every listing in
# it, so a detail hidden, a listing paused or deleted, or a reset in the meantime stops it.
# --------------------------------------------------------------------------------------

class RefusedByWarden(Exception):
    """A request to the Matchmaker AI that failed the Warden's check: it never reaches the model (the hybrid
    Matchmaker then decides its pairs by the rules)."""


_JUDGE_REQUEST_KEYS = {"focus", "candidates", "listings"}
# The built-ins a request may carry, by the names they go by there, and the field each is: only what a listing shows.
# Where and how people meet, dating's ages and genders, and a first name are never in one.
_JUDGE_BUILT_INS = {"headline": "headline", "looking_for": "seeking", "offering": "offering", "topics": "topics",
                    "pay": "pay"}
_JUDGE_ENTRY_KEYS = {"kind", "exactness", "description", "details", *_JUDGE_BUILT_INS}
_JUDGE_DETAIL_KEYS = {"about", "name", "values", "compare", "steps", "required"}
_CUT_MARK = "…"  # what stands where a stretch of a listing's own text was taken out
_RUN_OF_WORD = re.compile(r"\w+")
# Quote marks and apostrophes outside the "initial" and "final" quote categories (Pi, Pf), read as one.
_QUOTE_MARKS = frozenset("'\"`´ʹʺʻʼʽʾʿˈˊˋ˝′″‴‵‶‷‚„＂＇")
_WHOLE: dict = {}


def _readings_of(t: str) -> list:
    """A canonical text as it is and read loosely, so a watched text whose marks are written differently is still
    found: a character that shows nothing (_invisible), a quote mark or an apostrophe left out in one reading and a
    space in the other; a dash a space in both; one space between words, none at the ends."""
    out = [t]
    for gap in ("", " "):
        chars = []
        for c in t:
            cat = unicodedata.category(c)
            chars.append(gap if _invisible(c) or cat in ("Pi", "Pf") or c in _QUOTE_MARKS
                         else " " if cat == "Pd" else c)
        reading = " ".join("".join(chars).split())
        if reading not in out:
            out.append(reading)
    return out


def _forms_of(text: str) -> set:
    """A watched text as it is looked for: its canonical form, and each reading of it without the punctuation and
    spaces at its ends ("ltd." in "ltd," too)."""
    out = {canonical(text)} - {""}
    for reading in _readings_of(canonical(text)):
        core = reading.strip(" ")
        while core and unicodedata.category(core[0]).startswith("P"):
            core = core[1:].lstrip(" ")
        while core and unicodedata.category(core[-1]).startswith("P"):
            core = core[:-1].rstrip(" ")
        if core:
            out.add(core)
    return out


def _whole(s: str):
    """Where s stands in a text as whole words: bounded by a non-word character at an end that is a Latin letter or
    digit; at an end in any other script, anywhere (a name in a script written without spaces stands inside a run)."""
    pattern = _WHOLE.get(s)
    if pattern is None:
        if len(_WHOLE) > 200_000:
            _WHOLE.clear()
        left = r"(?<!\w)" if s[:1].isascii() and s[:1].isalnum() else ""
        right = r"(?!\w)" if s[-1:].isascii() and s[-1:].isalnum() else ""
        pattern = _WHOLE[s] = re.compile(left + re.escape(s) + right)
    return pattern


def _spans(s: str, text: str):
    pattern, m = _whole(s), None
    m = pattern.search(text)
    while m:
        yield m.start(), m.end()
        m = pattern.search(text, m.start() + 1)


class _Watched:
    """What no request to the Matchmaker AI may say, from every version of every listing on the board (deleted ones
    too): anyone's first name (one since changed included) and any listing or person id, anywhere; and anyone's hidden
    text (a value, an open detail's name or steps), except inside a text the listing it describes shows itself."""

    def __init__(self, board: Marketboard):
        self.ids = set()
        watched = {}  # form -> "name", "id" or "hidden" (a name or id wins: it may never stand anywhere)
        for v in board.history():
            for i in (v.person_id, v.owner, board.person(v.person_id).id):
                self.ids.add(canonical(i))
                for form in _forms_of(i):
                    watched[form] = "id"
            for f in v.fields:
                if f.key == "first_name":
                    for t in f.strings():
                        for form in _forms_of(t):
                            if watched.get(form) != "id":
                                watched[form] = "name"
                if f.visibility == HIDDEN:
                    for t in f.texts():
                        for form in _forms_of(t):
                            watched.setdefault(form, "hidden")
        self.ids |= {form for form, what in watched.items() if what == "id"}
        self._index, self._rest = {}, []
        for form, what in watched.items():
            first = _RUN_OF_WORD.match(form)
            if first and form[0].isascii() and form[0].isalnum() and (
                    first.end() < len(form) or (form[-1].isascii() and form[-1].isalnum())):
                self._index.setdefault(first.group(), []).append((form, what))
            else:
                self._rest.append((form, what))

    def found(self, reading: str) -> list:
        """(what, form, start, end) for every watched text standing in this reading of a text sent."""
        out = []
        for form, what in [e for w in set(_RUN_OF_WORD.findall(reading)) for e in self._index.get(w, ())] + self._rest:
            if form in reading:
                out += [(what, form, a, b) for a, b in _spans(form, reading)]
        return out


def _is_cut(sent, source: str) -> bool:
    """Is a text sent its listing's own text (canonical) with stretches taken out (each marked _CUT_MARK) and nothing
    added: what is left, in order, all from the source?"""
    if not isinstance(sent, str):
        return False
    pieces = [p.strip() for p in sent.split(_CUT_MARK)]
    if len(pieces) == 1:
        return pieces[0] == source
    if (pieces[0] and not source.startswith(pieces[0])) or (pieces[-1] and not source.endswith(pieces[-1])):
        return False
    at = 0
    for p in pieces:
        if p:
            k = source.find(p, at)
            if k < 0:
                return False
            at = k + len(p)
    return True


def _texts_sent(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _judge_entry_problems(entry, listing: Listing, watched: _Watched) -> list:
    """Why one listing's entry in a request may not go: anything but what the listing shows now (its shown phrases,
    topics, pay and open details, each its own text with stretches taken out) and its own description; a first name
    or id of anyone; a hidden text of anyone outside a text this listing shows itself."""
    if not isinstance(entry, dict):
        return ["an entry isn't an entry"]
    out = [f"it carries {k!r}" for k in entry if k not in _JUDGE_ENTRY_KEYS]
    if entry.get("kind", "") not in (*MATCH_KINDS, ""):
        out.append("its kind isn't one of the fixed kinds")
    x = entry.get("exactness", 0.5)
    try:
        own = round(min(1.0, max(0.0, float(listing.exactness))), 2)
    except (TypeError, ValueError):
        own = 0.5
    if type(x) not in (int, float) or not abs(x - own) <= 0.005 + 1e-9 or x != round(x, 2):
        out.append("its exactness isn't its listing's")
    texts = []
    for name, key in _JUDGE_BUILT_INS.items():
        if name not in entry:
            continue
        f, sent = listing.get(key), _texts_sent(entry[name])
        if f is None or f.visibility != SHOWN or sent is None or \
                not all(any(_is_cut(t, canonical(s)) for s in f.strings()) for t in sent):
            out.append(f"{name} isn't what its listing shows")
        texts += sent or []
    details = entry.get("details", [])
    for d in details if isinstance(details, list) else [None]:
        values = _texts_sent(d.get("values")) if isinstance(d, dict) else None
        steps = _texts_sent(d.get("steps", [])) if isinstance(d, dict) else None
        if values is None or steps is None or not set(d) <= _JUDGE_DETAIL_KEYS or not isinstance(d.get("name"), str):
            out.append("a detail isn't a detail's shape")
            continue
        texts += [d["name"], *values, *steps]
        if not any(f.visibility == SHOWN and f.about and f.about == d.get("about")
                   and _is_cut(d["name"], canonical(f.name))
                   and d.get("compare", "same") == ("steps" if f.compare == "steps" else "same")
                   and ("required" not in d or (type(d["required"]) is bool and d["required"] == bool(f.required)))
                   and all(any(_is_cut(v, canonical(s)) for s in f.strings()) for v in values)
                   and all(any(_is_cut(t, canonical(s)) for s in f.steps) for t in steps)
                   for f in listing.fields):
            out.append(f"detail {d['name']!r} isn't one its listing shows")
    description = entry.get("description", "")
    if not isinstance(description, str) or not _is_cut(description, canonical(listing.intent_text)):
        out.append("its description isn't its listing's")
    else:
        texts.append(description)
    own_words = {form for f in listing.fields if f.visibility == SHOWN and f.key != "first_name" for t in f.texts()
                 for form in _forms_of(t)}
    said = set()
    for text in texts:
        for reading in _readings_of(canonical(text)):
            inside = None
            for what, form, a, b in watched.found(reading):
                if what == "hidden":
                    if inside is None:
                        inside = {i for t in own_words if t in reading for x, y in _spans(t, reading)
                                  for i in range(x, y)}
                    if all(i in inside for i in range(a, b)):
                        continue
                said.add((what, form))
    words = {"name": "someone's first name", "id": "a listing or person id", "hidden": "a text someone hides"}
    return out + [f"it says {form!r}, {words[what]}" for what, form in sorted(said)]


def judge_request_problems(request, labels, board: Marketboard, watched: _Watched = None) -> list:
    """Why a request to the Matchmaker AI may not go now (empty: it may). Only the fixed shape; labels that fit
    together and are never an id; each label a listing on the board and live (not paused) now; each entry only what
    that listing shows now and its own description (_judge_entry_problems), with no first name or id of anyone and no
    hidden text of anyone outside a text that listing shows itself."""
    if not isinstance(request, dict) or set(request) != _JUDGE_REQUEST_KEYS or not isinstance(labels, dict):
        return ["it isn't a request the Matchmaker kept"]
    focus, candidates, listings = request["focus"], request["candidates"], request["listings"]
    if not isinstance(listings, dict) or not isinstance(candidates, list):
        return ["it isn't a request's shape"]
    order = [focus, *candidates]
    if not all(isinstance(x, str) for x in order) or len(set(order)) != len(order) or set(order) != set(listings) \
            or set(labels) != set(listings) or not all(isinstance(v, str) for v in labels.values()):
        return ["its labels don't fit together"]
    watched = watched or _Watched(board)
    out = []
    for label, entry in listings.items():
        if canonical(label) in watched.ids or _forms_of(label) & watched.ids:
            out.append(f"label {label!r} is an id")
            continue
        pid = labels[label]
        if pid not in board or not board.listing(pid).active:
            out.append(f"{label} isn't a live listing now")
            continue
        out += [f"{label}: {p}" for p in _judge_entry_problems(entry, board.listing(pid), watched)]
    return out


def kept_labels(request, requests, labels) -> dict:
    """Which listing each label of a request stands for, from the Matchmaker's own record of the requests it kept
    (parallel lists, newest last): the newest kept request exactly like it, or None if it kept none."""
    try:
        text = json.dumps(request, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return None
    for kept, mapping in zip(reversed(list(requests)), reversed(list(labels))):
        if json.dumps(kept, ensure_ascii=False, sort_keys=True) == text:
            return dict(mapping)
    return None


def checked_judge(warden: "Warden", judge, board, requests, labels):
    """The Matchmaker AI's judge behind the Warden's check, for a caller that hands each request over at once: board()
    is the board now, requests() and labels() the Matchmaker's own record of what it kept. A request that fails never
    reaches the judge (RefusedByWarden)."""
    def checked(request):
        if not warden.exit_judge_request(request, kept_labels(request, requests(), labels()), board()):
            raise RefusedByWarden("refused by the Warden's check")
        return judge(request)
    checked.judge = judge
    return checked
