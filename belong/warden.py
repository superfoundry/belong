"""Warden agent (SPEC §3, §7): checks everything that goes into matching and
everything that comes out.

Fixed rules, no AI judgment. The Warden never decides who matches; it only checks.

  ENTRY  admit()                 person + listing  -> onto the Marketboard, or rejected
  EXIT   exit_introduction()     Proposal          -> Introduction, or dropped
         exit_followup()         FollowUp          -> FollowUpOut, or dropped
         exit_reveal()           consents          -> Reveal(s) of shared hidden details

Every check writes a Receipt {check, result, timestamp, hash}. The hash commits to
what was checked (keyed HMAC) without containing it, so receipts can be published
and audited without revealing content.

Output objects (Introduction, FollowUpOut, Reveal) can only be built in this module:
their constructors demand a private seal. Anything sent to a person is one of them.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from .matching.marketboard import Marketboard
from .matching.contract import FollowUp, Proposal
from .schema import (AGE_BANDS, FIELD_REGISTRY, FOLLOWUP_MENU, FREE_TEXT_FIELDS, FREE_TEXT_MAX, HIDDEN, SHOWN,
                     Listing, Person, value_strings, value_to_json)

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
                 one_way_to: str = ""):
        super().__init__(seal)
        self.intro_id, self.person_a, self.person_b = intro_id, person_a, person_b
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
                "a_view": self.a_view, "b_view": self.b_view, "one_way_to": self.one_way_to}


class FollowUpOut(_Sealed):
    def __init__(self, seal, person_id: str, question_id: str, phrase: str = "", mirror: str = ""):
        super().__init__(seal)
        self.person_id, self.question_id, self.phrase = person_id, question_id, phrase
        # For "also_offer": who the asker is (their shown "you are" phrase). Never shown in the
        # question; if the person says yes, it joins who they're looking for, so a match can follow.
        self.mirror = mirror
        self.question_text = FOLLOWUP_MENU[question_id].format(phrase=phrase)

    def to_dict(self) -> dict:
        return {"person_id": self.person_id, "question_id": self.question_id, "question_text": self.question_text,
                "phrase": self.phrase, "mirror": self.mirror}


class Reveal(_Sealed):
    """A hidden detail two connected people share, revealed because both opted in."""

    def __init__(self, seal, person_a: str, person_b: str, key: str, values: list):
        super().__init__(seal)
        self.person_a, self.person_b, self.key, self.values = person_a, person_b, key, list(values)

    def to_dict(self) -> dict:
        return {"person_a": self.person_a, "person_b": self.person_b, "key": self.key, "values": self.values}


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


ENTRY_CHECKS = ("kyc_verified", "listing_schema", "sensitive_defaults", "specific_person", "update_rate_limit")
EXIT_CHECKS = ("proposal_mutual", "proposal_known_people", "proposal_distinct_people", "reason_fields_shown", "approach_limit", "introduction_schema",
               "fields_shown", "reason_refs_only_reason_fields", "no_hidden_values",
               "followup_menu", "reveal_connected", "reveal_consent_and_shared")


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

_ALLOWED_CAPITALISED = {
    "I", "I'm", "I'd", "I'll", "CTO", "CEO",
    "Osaka", "Tokyo", "Kyoto", "Berlin", "London", "Lisbon",
    "English", "Japanese", "German", "Portuguese", "Spanish", "French",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}
_SPECIFIC_PERSON_RULES = [
    ("social handle", re.compile(r"@\w{2,}")),
    ("email address", re.compile(r"[\w.+-]+@[\w-]+\.\w+")),
    ("phone number", re.compile(r"\+?\d[\d\s().-]{7,}\d")),
    ("web link", re.compile(r"https?://|www\.", re.I)),
    ("'named'/'called' + a name", re.compile(r"\b(named|called)\s+[A-Z]")),
    ("an ex", re.compile(r"\bmy ex\b", re.I)),
    ("'works at' + a name", re.compile(r"\bworks? at [A-Z]")),
]
_WORKPLACE = re.compile(r"\b(works? at|working at|barista at|cashier at|employee at|bartender at)\b", re.I)
_PHYSICAL = re.compile(r"\b(tall|short|hair|tattoos?|glasses|beard|nose ring|piercings?|freckles)\b", re.I)


def specific_person_problems(intent: str) -> list[str]:
    """Reasons this intent could single out one individual. Empty list = fine."""
    problems = [name for name, rx in _SPECIFIC_PERSON_RULES if rx.search(intent)]
    if _WORKPLACE.search(intent) and _PHYSICAL.search(intent):
        problems.append("a workplace + physical description")
    # Two capitalised words in a row mid-sentence look like a full name ("Mika Sato").
    tokens = list(re.finditer(r"[^\W\d_][\w'’-]*", intent))
    for prev, tok in zip(tokens, tokens[1:]):
        if intent[prev.end():tok.start()].strip():
            continue  # not adjacent
        if all(_is_proper_noun(intent, t) for t in (prev, tok)):
            problems.append("what looks like a full name")
            break
    return problems


def _is_proper_noun(text: str, tok: re.Match) -> bool:
    word = tok.group()
    if not word[0].isupper() or word in _ALLOWED_CAPITALISED:
        return False
    before = text[:tok.start()].rstrip()
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
    return {s for f in listing.fields if f.visibility == visibility for s in f.strings()}


# --------------------------------------------------------------------------------------
# Reason text: written by the Warden from reason_fields only (SPEC §6: never in the reason).
# --------------------------------------------------------------------------------------

def reason_text(key_values: list[tuple[str, object, object]]) -> str:
    """key_values: (key, my_value, their_value) for each reason field, from the
    recipient's point of view. Uses nothing else."""
    parts = []
    # "They're looking for X" only describes the reader (that's why they were matched), so
    # say who they are instead whenever "offering" is a reason.
    has_offering = any(key == "offering" for key, _, _ in key_values)
    for key, mine, theirs in key_values:
        if key in ("city", "country"):
            parts.append(f"You're both in {theirs}.")
        elif key == "schedule":
            when = mine if theirs == "flexible" else theirs
            parts.append("You're both flexible on timing." if when == "flexible" else f"You're both free {when}.")
        elif key == "interests":
            common = sorted(set(value_strings(mine)) & set(value_strings(theirs)))
            parts.append(f"You both like {', '.join(common)}.")
        elif key == "level":
            parts.append(f"They climb at {theirs} level; you're {mine}.")
        elif key == "pace":
            article = "an" if theirs[:1] in "aeiou" else "a"
            parts.append(f"They run at {article} {theirs} pace; yours is {mine}.")
        elif key == "native_language":
            parts.append(f"They speak {theirs} natively.")
        elif key == "learning_language":
            parts.append(f"They're learning {theirs}.")
        elif key == "subject":
            parts.append(f"Subject: {theirs}.")
        elif key == "role":
            parts.append(f"They're the {theirs} side.")
        elif key == "move_in":
            parts.append(f"You both plan to move in {theirs}.")
        elif key == "seeking" and not has_offering:
            parts.append(f"They're looking for {theirs}.")
        elif key == "offering":
            parts.append(f"They're {theirs}.")
    return " ".join(parts)


# --------------------------------------------------------------------------------------
# The Warden
# --------------------------------------------------------------------------------------

@dataclass
class EntryResult:
    accepted: bool
    failed_checks: list = field(default_factory=list)
    problems: list = field(default_factory=list)


class Warden:
    def __init__(self, clock: Clock, key: bytes, approach_limit_per_week: int = 3,
                 updates_per_day: int = 5, max_intent_chars: int = 500):
        self.clock = clock
        self._key = key
        self.approach_limit = approach_limit_per_week
        self.updates_per_day = updates_per_day
        self.max_intent_chars = max_intent_chars
        self.receipts: list[Receipt] = []
        self._updates: dict[str, list[str]] = {}
        self._approaches: dict[str, list[str]] = {}
        self._decisions: dict[str, dict[str, bool]] = {}
        self._intros: dict[str, Introduction] = {}
        self._next_intro = 0

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

    def admit(self, person: Person, listing: Listing, board: Marketboard) -> EntryResult:
        """Entry check (SPEC §7). On success the listing goes onto the Marketboard."""
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
        sens = [f.key for f in listing.fields
                if f.sensitive and f.visibility == SHOWN and f.key not in listing.confirmed_shown]
        check("sensitive_defaults", not sens, f"sensitive fields shown without confirmation: {sens}")
        # The rule covers the intent and the free-text fields ("seeking", "offering").
        texts = [listing.intent_text] + [f.value for f in listing.fields
                                         if f.key in ("seeking", "offering", "pay", "format") and isinstance(f.value, str)]
        sp = sorted({p for t in texts for p in specific_person_problems(t)})
        check("specific_person", not sp, f"intent could identify one person ({', '.join(sp)})")
        check("update_rate_limit", self._within_update_limit(listing.person_id), "too many listing updates today")

        if failed:
            return EntryResult(False, failed, problems)
        board.put(person, listing)
        return EntryResult(True)

    def _schema_problems(self, person: Person, listing: Listing) -> list[str]:
        out = []
        if listing.owner != person.id:
            out.append("listing belongs to a different person")
        if not isinstance(listing.intent_text, str) or not 0 < len(listing.intent_text) <= self.max_intent_chars:
            out.append("intent text missing or too long")
        if not (isinstance(listing.exactness, (int, float)) and 0.0 <= listing.exactness <= 1.0):
            out.append("exactness must be between 0 and 1")
        seen = set()
        for f in listing.fields:
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
            ok_type = (isinstance(f.value, tuple) and all(isinstance(v, str) for v in f.value)) if spec.multi \
                else isinstance(f.value, str)
            if not ok_type:
                out.append(f"bad value type on {f.key!r}")
            elif f.key in FREE_TEXT_FIELDS and not 0 < len(f.value) <= FREE_TEXT_MAX:
                out.append(f"{f.key!r} must be 1-{FREE_TEXT_MAX} characters")
        age = listing.get("age_band")
        if age is not None and age.value != person.age_band:
            out.append("age_band must match the KYC age band")
        if age is not None and age.value not in AGE_BANDS:
            out.append("unknown age band")
        return out

    def _within_update_limit(self, person_id: str) -> bool:
        now = _parse(self.clock.now())
        recent = [t for t in self._updates.get(person_id, []) if now - _parse(t) < timedelta(days=1)]
        recent.append(self.clock.now())
        self._updates[person_id] = recent
        return len(recent) <= self.updates_per_day

    # -- EXIT: introductions ---------------------------------------------------

    def exit_introduction(self, proposal: Proposal, board: Marketboard):
        """Turn an internal Proposal into an Introduction, or drop it (returns None).

        The Introduction is built here, by the Warden, from shown fields only; the
        Matchmaker contributes nothing but the pair and the reason field keys.
        """
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
        # A person can have several listings; they are never introduced to themselves.
        if not self._receipt("proposal_distinct_people", a.owner != b.owner, subject):
            return None
        a_shown, b_shown = a.shown(), b.shown()

        reasons = list(proposal.reason_fields)
        both_show = bool(reasons) and all(k in a_shown and k in b_shown for k in reasons)
        if not self._receipt("reason_fields_shown", both_show, subject):
            return None

        under_limit = all(self._approaches_this_week(p) < self.approach_limit for p in (a_id, b_id))
        if not self._receipt("approach_limit", under_limit, subject):
            return None

        a_view = {"fields": {k: value_to_json(v) for k, v in b_shown.items()},
                  "reason": reason_text([(k, a_shown[k], b_shown[k]) for k in reasons])}
        b_view = {"fields": {k: value_to_json(v) for k, v in a_shown.items()},
                  "reason": reason_text([(k, b_shown[k], a_shown[k]) for k in reasons])}

        ok = True
        for viewer, other, view in ((a, b, a_view), (b, a, b_view)):
            ok &= self._receipt("introduction_schema", _view_schema_ok(view), subject)
            ok &= self._receipt("fields_shown", _fields_all_shown(view, other), subject)
            ok &= self._receipt("reason_refs_only_reason_fields",
                                _reason_uses_only(view["reason"], reasons, viewer, other), subject)
            ok &= self._receipt("no_hidden_values", not _hidden_in_view(view, viewer, other, reasons), subject)
        if not ok:
            return None

        intro = Introduction(_SEAL, f"i{self._next_intro:04d}", a_id, b_id, a_view, b_view, one_way_to=close)
        self._next_intro += 1
        self._intros[intro.intro_id] = intro
        for p in (a_id, b_id):
            self._approaches.setdefault(p, []).append(self.clock.now())
        return intro

    def _approaches_this_week(self, person_id: str) -> int:
        now = _parse(self.clock.now())
        return sum(1 for t in self._approaches.get(person_id, []) if now - _parse(t) < timedelta(days=7))

    # -- EXIT: follow-up questions ---------------------------------------------

    def exit_followup(self, followup: FollowUp, board: Marketboard):
        ok = followup.question_id in FOLLOWUP_MENU and followup.person_id in board
        if not self._receipt("followup_menu", ok, {"p": followup.person_id, "q": followup.question_id}):
            return None
        # Questions from Belong. only reach people who opted in to them.
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
        phrase, mirror = them.shown().get("seeking"), them.shown().get("offering")
        shown = all(isinstance(x, str) and bool(x.strip()) for x in (phrase, mirror))
        if not self._receipt("also_offer_phrase_shown", shown, subject):
            return None
        forbidden = _strings_of(me, HIDDEN) | _strings_of(them, HIDDEN)
        clean = not find_forbidden(f"{phrase} {mirror}", forbidden, _strings_of(them, SHOWN))
        if not self._receipt("no_hidden_values", clean, subject):
            return None
        return FollowUpOut(_SEAL, followup.person_id, "also_offer", phrase, mirror)

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
        """Reveal hidden details the two share, only where BOTH opted in (SPEC §6)."""
        intro = self._intros.get(intro_id)
        subject = {"intro": intro_id}
        if not self._receipt("reveal_connected", self.is_connected(intro_id), subject):
            return []
        a, b = board.listing(intro.person_a), board.listing(intro.person_b)
        out = []
        for key in sorted(set(consent_a) & set(consent_b)):
            fa, fb = a.get(key), b.get(key)
            shared = sorted(set(fa.strings()) & set(fb.strings())) if fa and fb else []
            hidden = bool(fa and fb) and HIDDEN in (fa.visibility, fb.visibility)
            if self._receipt("reveal_consent_and_shared", bool(shared) and hidden, {**subject, "key": key}):
                out.append(Reveal(_SEAL, intro.person_a, intro.person_b, key, shared))
        return out


# --------------------------------------------------------------------------------------
# Exit-check helpers. Each one is a single rule from SPEC §7.
# --------------------------------------------------------------------------------------

def _view_schema_ok(view) -> bool:
    """Output matches the Introduction schema exactly."""
    if not isinstance(view, dict) or set(view) != {"fields", "reason"}:
        return False
    if not isinstance(view["reason"], str) or not isinstance(view["fields"], dict):
        return False
    for k, v in view["fields"].items():
        if k not in FIELD_REGISTRY:
            return False
        if not (isinstance(v, str) or (isinstance(v, list) and all(isinstance(x, str) for x in v))):
            return False
    return True


def _fields_all_shown(view: dict, owner: Listing) -> bool:
    """Every field in the view is marked shown by its owner, with the owner's value."""
    for k, v in view["fields"].items():
        f = owner.get(k)
        if f is None or f.visibility != SHOWN or value_to_json(f.value) != v:
            return False
    return True


def _reason_uses_only(reason: str, reasons: list, viewer: Listing, other: Listing) -> bool:
    """Reason text mentions no value from any field outside reason_fields."""
    allowed = {s for l in (viewer, other) for k in reasons if l.get(k) for s in l.get(k).strings()}
    outside = {s for l in (viewer, other) for f in l.fields if f.key not in reasons for s in f.strings()}
    return not find_forbidden(reason, outside - allowed, allowed)


def _hidden_in_view(view: dict, viewer: Listing, other: Listing, reasons: list) -> list:
    """String check: no hidden value of either person appears anywhere in the view."""
    forbidden = _strings_of(viewer, HIDDEN) | _strings_of(other, HIDDEN)
    allowed = _strings_of(other, SHOWN)  # the counterpart chose to show these
    allowed |= {s for k in reasons if viewer.get(k) for s in viewer.get(k).strings()}
    return find_forbidden(json.dumps(view, ensure_ascii=False), forbidden, allowed)
