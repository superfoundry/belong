"""The hybrid Matchmaker (SPEC §3, §5, §6, §8.3): the rules narrow, a judge judges, the Warden checks. OPEN SOURCE.

Open source, and still treated as hostile, like the rules it builds on (matcher.py): it can only hand the Warden the
objects in contract.py, and the Warden checks every one exactly as it checks the rules' own.

  1. The rules keep everything hard and exact (OfflineMatcher.read, hard_met): narrowing (how they meet, where, the
     same kind of match), dating's genders and ages wanted, every wish on a same-named detail both can be compared
     on, a strict listing's every wish, never one person's two listings, and the opt-in to close matches. The judge
     never adds a pair these rules refuse.
  2. The judge, a callable injected from outside this package (plain data in, plain data out), judges the fuzzy part
     of each pair that passes them: does each offer what the other wants, from their shown phrases, topics, open
     details (also ones two agents named or scaled differently) and descriptions. Per direction "match", "close" or
     "no"; a fit in [0, 1] for ranking; and per listing whether it seems to hunt one particular person, which can
     only drop that listing's pairs. One request per focus listing with its candidates; whether a listing hunts one
     person is asked about it alone (a request with no candidates), so no other listing's text can get it flagged,
     and the verdicts on a focus listing's pairs are used only once it has been asked about alone.
  3. What it receives (build_entry, LLMMatcher._send): for each listing its shown fields as a ShownView gives them
     (what it is for, never where or how it meets, dating's ages and genders, or a first name) and its description,
     every text in the canonical form the rules read it in (schema.canonical). Every first name and listing or
     person id on the board is taken out of every text, and every hidden text of anyone (a value, an open detail's
     name or steps) too, unless it stands inside a text the listing shows itself (Redactor). Never a hidden field,
     never a first name, never an id: each listing goes by a label of that request only ("c1", "c2" ...), mapped
     back here. Every request is kept, exactly as sent, with the board change it was built at (payloads,
     payload_changes, sent_requests), and the independent audit checks each one (leaks.py, A8).
  4. Its answer is read strictly (read_answer): only the allowed keys, only labels of that request, only the fixed
     verdicts, only a number in [0, 1]; anything else is dropped and counted. No text of the answer is ever kept: a
     label becomes the id this code gave it, a verdict this code's own constant, a fit a plain float.
  5. The result is the rules' Proposal contract: reason fields and open-detail refs from the rules' own code
     (reason_fields, detail_refs); hidden fields only through the rules' crowd-gated bonus (hidden_overlap); the fit
     as the shown score, which must clear the rules' shown threshold (never the only reason). Questions are the
     rules' own (age range, "also offer", "include close matches?").
  6. Cost: verdicts are kept per pair of listing versions (what the judge saw of each) and asked again only when one
     changes, or any other listing of the request it came from changes or goes (the judge read them all together,
     so a verdict never outlives what it was shown with); at most max_pairs_per_run pairs are sent in a run, shared out fairly between the focus listings, and
     every other pair is decided by the rules, as is every pair of a request the judge fails on. Each run's counts
     are in last_run.

  7. A judge that is still getting an answer elsewhere (it raises JudgeWaiting, as the demo's does while a model
     answers off its lock) leaves that request's pairs waiting: neither proposed nor decided by the rules in this run,
     and their listings aren't asked "include close matches?" meanwhile (they may have a match). A later run reads
     the answer, or, if the judge failed, decides them by the rules.

This file has no network: the judge is whatever the caller passes in (outside this package, e.g. a model reached
over an API). With no judge, or a judge that fails, it decides exactly as OfflineMatcher.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import replace

from ..schema import (HIDDEN, MATCH_KINDS, STRICT_EXACTNESS, canonical, comparable, satisfies, says_nothing,
                      value_strings)
from .contract import MatchResult, Proposal
from .intent import ParsedIntent
from .marketboard import Marketboard
from .matcher import (CLOSE_EXACTNESS, WIDE_NET_STEPS, OfflineMatcher, Reading, _dating_met, detail_refs,
                      reason_fields, shown_threshold, understood)
from .narrow import ONLINE, ShownView, how_they_meet

# A direction's verdict: what one listing wants, the other gives ("match"), nearly gives ("close"), or not ("no").
MATCH, CLOSE, NO = "match", "close", "no"
VERDICTS = (MATCH, CLOSE, NO)
_VERDICT = {v: v for v in VERDICTS}  # the judge's string is never kept: this code's own constant is
# The only shape of an answer read back. Top level: the verdicts, and the labels of listings that seem to hunt one
# particular person. Each verdict: the candidate's label, whether the candidate gives what the focus listing wants,
# whether the focus listing gives what the candidate wants, and the fit.
ANSWER_KEYS = ("pairs", "hunts_one_person")
PAIR_KEYS = ("candidate", "focus_wants", "candidate_wants", "fit")
# Why part of an answer was dropped (counted in last_run["dropped"]), in fixed words.
DROPS = ("answer", "key", "pair", "label", "verdict", "fit", "repeat", "hunt")
# A listing's shown built-ins that go in a request, and the names they go by there. Where and how people meet, the
# ages and genders dating asks for, and a first name never do: the rules decide the first ones, and a name is no
# business of the judge's.
ENTRY_FIELDS = (("headline", "headline"), ("seeking", "looking_for"), ("offering", "offering"), ("topics", "topics"),
                ("pay", "pay"))
REMOVED = "…"  # what stands where a text was taken out
LABEL_LETTERS = "ckmqxz"  # a request's labels: the first of these whose labels are no id on the board

MAX_PAIRS_PER_RUN = 80  # pairs sent to the judge in one run; the rest are decided by the rules
MAX_PER_REQUEST = 40  # candidates in one request; a focus listing with more is sent in several
CACHE_MAX = 50_000  # verdicts kept; all let go when there are more
PAYLOADS_KEPT = 2_000  # requests kept for the audit; older ones are let go and counted


class JudgeWaiting(Exception):
    """Raised by a judge that has no answer yet and is getting one elsewhere: the request's pairs wait for a later run
    (see the module's docstring, 7)."""


_WAITING = object()  # what _send returns for a request whose answer isn't there yet


def details_hard(av: ShownView, bv: ShownView) -> bool:
    """The part of A's shown wishes the rules decide alone: every wish B can be compared on with its own shown detail
    of the same name (schema.comparable) must be met (schema.satisfies, a wide net one step either side on the
    identical scale), required or "ideally"; and a strict A's every wish needs such a detail. A wish that says
    nothing is always met. A wish B has no comparable detail for (named or scaled differently, or not shown) is left
    to the judge, which may read B's other details and description for it. Shown fields only."""
    strict = av.exactness >= STRICT_EXACTNESS
    for want in av.wishes():
        if says_nothing(want):
            continue
        theirs = bv.own(want.name)
        if theirs is None or not comparable(want, theirs, strict):
            if strict:
                return False
            continue
        if not satisfies(want, theirs, WIDE_NET_STEPS if av.exactness < 0.5 else 0):
            return False
    return True


def hard_met(ap: ParsedIntent, av: ShownView, bp: ParsedIntent, bv: ShownView) -> bool:
    """A's hard rules on B: the same kind of match, for dating B's shown gender and age band in what A wants
    (matcher._dating_met), and A's wishes as details_hard reads them. Everything else about whether B gives what A
    wants is the judge's. Every pair matcher.intent_met accepts passes, so the rules' own decisions always do."""
    if ap.kind is None or ap.kind != bp.kind:
        return False
    if ap.kind == "dating" and not _dating_met(ap, av, bv):
        return False
    return details_hard(av, bv)


def _wide(listing) -> ShownView:
    """A listing read as if it cast a wide net (a close match is judged that way, as by the rules)."""
    return ShownView(replace(listing, exactness=CLOSE_EXACTNESS))


# -- what the judge receives ------------------------------------------------------------------------------------------

_WORD = re.compile(r"\w+")
# Quote marks and apostrophes outside the "initial" and "final" quote categories (Pi, Pf): read as one.
_QUOTES = frozenset("'\"`´ʹʺʻʼʽʾʿˈˊˋ˝′″‴‵‶‷‚„＂＇")


def _shows_nothing(c: str) -> bool:
    """A character that shows nothing: a format character (Unicode category Cf: a soft hyphen, a zero-width space or
    joiner, a direction mark...) or a variation selector."""
    return unicodedata.category(c) == "Cf" or 0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF


def _loose(t: str) -> list:
    """A canonical text as it is and its loose readings, [(reading, where each of its characters stands in t, or None
    when they stand where they are)]: a text is found as it is written, however a phone or a paste wrote its marks. A
    character that shows nothing and a quote mark or apostrophe are taken out in one reading ("joined") and read as a
    space in the other ("spaced"); a dash is a space in both (as canonical reads "-"); runs of spaces are one, and
    there are none at the ends."""
    out = [(t, None)]
    if t.isascii() and not any(q in t for q in "'\"`"):
        return out  # nothing to read loosely: both readings are the text itself
    for gap in ("", " "):
        chars, at = [], []
        for i, c in enumerate(t):
            cat = unicodedata.category(c)
            if _shows_nothing(c) or cat in ("Pi", "Pf") or c in _QUOTES:
                c = gap
            elif cat == "Pd":
                c = " "
            if not c or (c == " " and (not chars or chars[-1] == " ")):
                continue
            chars.append(c)
            at.append(i)
        while chars and chars[-1] == " ":
            chars.pop()
            at.pop()
        reading = "".join(chars)
        if all(reading != r for r, _ in out):
            out.append((reading, at))
    return out


def _cores(text: str) -> set:
    """What is looked for when a text is taken out: its canonical form, and each loose reading of it without the
    punctuation and spaces at its ends, so "ltd." is found in "ltd," too."""
    out = {canonical(text)} - {""}
    for reading, _ in _loose(canonical(text)):
        a, b = 0, len(reading)
        while a < b and (reading[a] == " " or unicodedata.category(reading[a]).startswith("P")):
            a += 1
        while b > a and (reading[b - 1] == " " or unicodedata.category(reading[b - 1]).startswith("P")):
            b -= 1
        if a < b:
            out.add(reading[a:b])
    return out


def _pattern(text: str):
    """Where a text stands in another as whole words: bounded by a non-word character at an end that is a Latin letter
    or digit; in any other script, anywhere (a name in a script written without spaces can stand inside a run)."""
    left = r"(?<!\w)" if text[:1].isascii() and text[:1].isalnum() else ""
    right = r"(?!\w)" if text[-1:].isascii() and text[-1:].isalnum() else ""
    return re.compile(left + re.escape(text) + right)


def _occurrences(pattern, text: str):
    m = pattern.search(text)
    while m:
        yield m.start(), m.end()
        m = pattern.search(text, m.start() + 1)


class Redactor:
    """Takes texts out of what the judge receives. Every text is read in canonical form (schema.canonical: case,
    width, accents on Latin letters and spacing don't matter), and so is what it takes out:
      always:  the first names and the listing and person ids of everyone on the board;
      hidden:  every hidden text of everyone (values, and an open detail's name and steps), unless the occurrence
               stands inside an occurrence of a text the listing shows itself (it is in the request anyway).
    Each is found however its marks are written (_loose: a character that shows nothing, a curly or straight quote
    mark or apostrophe, a dash) and with or without the punctuation at its ends (_cores). Each taken-out stretch of
    the canonical text becomes REMOVED; what is sent is the canonical text otherwise."""

    def __init__(self, always=(), hidden=()):
        self._index, self._loose = defaultdict(list), []
        always = {c for s in always for c in _cores(s)}
        for text, is_always in [(s, True) for s in sorted(always)] + \
                [(s, False) for s in sorted({c for s in hidden for c in _cores(s)} - always)]:
            entry = (text, _pattern(text), is_always)
            first = _WORD.match(text)
            # Indexed by its first word when that word, as written, is a whole word wherever the text stands.
            if first and text[0].isascii() and text[0].isalnum() and (
                    first.end() < len(text) or (text[-1].isascii() and text[-1].isalnum())):
                self._index[first.group()].append(entry)
            else:
                self._loose.append(entry)

    def clean(self, text, keep=()) -> str:
        """The text in canonical form with what this redactor takes out replaced by REMOVED. keep: the canonical texts
        the listing shows itself."""
        t = canonical(text)
        if not t:
            return t
        keep = {c for s in keep if s for c in _cores(s)}
        spans = []
        for reading, at in _loose(t):
            kept = set()
            for s in keep:
                if s in reading:
                    for a, b in _occurrences(_pattern(s), reading):
                        kept.update(range(a, b))
            for s, pattern, is_always in [e for w in set(_WORD.findall(reading)) for e in self._index.get(w, ())] \
                    + self._loose:
                if s in reading:
                    spans += [(a, b) if at is None else (at[a], at[b - 1] + 1)
                              for a, b in _occurrences(pattern, reading)
                              if is_always or not all(i in kept for i in range(a, b))]
        if not spans:
            return t
        out, at = [], 0
        for a, b in sorted(spans):
            if a < at:  # overlaps the last one: one stretch
                if b > at:
                    at = b
                continue
            out += [t[at:a], REMOVED]
            at = b
        out.append(t[at:])
        return " ".join("".join(out).split())


def shown_texts(view: ShownView) -> set:
    """The canonical texts a listing shows that go in its entry (ENTRY_FIELDS and its open details' names, values and
    steps): what the hidden part of the Redactor leaves where it stands inside them."""
    out = {canonical(s) for key, _ in ENTRY_FIELDS if view.get(key) is not None for s in value_strings(view.get(key))}
    out |= {canonical(s) for f in view.fields.values() if f.about for s in f.texts()}
    return out - {""}


def build_entry(view: ShownView, kind, redactor: Redactor) -> dict:
    """One listing as the judge receives it: what it is for (kind and exactness, as the rules read them), its shown
    headline, phrases, topics and pay, its shown open details, and its description, each text cleaned by the
    redactor. Built from a ShownView, so nothing hidden is there to send."""
    keep = shown_texts(view)
    clean = lambda text: redactor.clean(text, keep)  # noqa: E731
    try:
        exactness = round(min(1.0, max(0.0, float(view.exactness))), 2)
    except (TypeError, ValueError):
        exactness = 0.5
    entry = {"kind": kind if kind in MATCH_KINDS else "", "exactness": exactness}
    for key, name in ENTRY_FIELDS:
        value = view.get(key)
        if value is None:
            continue
        entry[name] = clean(value) if isinstance(value, str) else [clean(s) for s in value_strings(value)]
    details = []
    for key in sorted(view.fields):
        f = view.fields[key]
        if not f.about:
            continue
        d = {"about": f.about, "name": clean(f.name), "values": [clean(s) for s in f.strings()],
             "compare": "steps" if f.compare == "steps" else "same"}
        if d["compare"] == "steps":
            d["steps"] = [clean(s) for s in f.steps]
        if f.about != "me":
            d["required"] = bool(f.required)
        details.append(d)
    if details:
        entry["details"] = details
    entry["description"] = clean(view.intent_text)
    return entry


def _frozen(x):
    """Plain data as something that can be a key (the verdict cache's)."""
    if isinstance(x, dict):
        return ("d",) + tuple((k, _frozen(v)) for k, v in sorted(x.items()))
    if isinstance(x, list):
        return ("l",) + tuple(_frozen(v) for v in x)
    return x


def _copy(x):
    """A deep copy of plain data, so nothing the judge does to its request changes the one kept."""
    if isinstance(x, dict):
        return {k: _copy(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy(v) for v in x]
    return x


def answer_schema(request: dict) -> dict:
    """The JSON schema of an answer to this request (for a judge that can hold a model to one): exactly the keys
    read_answer reads, this request's candidates as the only labels of a verdict and, in a request about one listing
    alone, its label as the only one that may be said to hunt one person (LLMMatcher._may_flag), the fixed verdicts, a
    fit in [0, 1], and nothing else. An empty list of labels is left as any string (a schema's list of choices can't be
    empty): read_answer drops whatever comes there, as it checks all of it again whatever the judge did."""
    candidates = [c for c in request.get("candidates", ()) if isinstance(c, str)]
    focus = request.get("focus")
    flaggable = [focus] if not candidates and isinstance(focus, str) else []
    label = lambda labels: {"type": "string", "enum": labels} if labels else {"type": "string"}  # noqa: E731
    verdict = {"type": "string", "enum": list(VERDICTS)}
    pair = {"type": "object", "additionalProperties": False, "required": list(PAIR_KEYS),
            "properties": {"candidate": label(candidates), "focus_wants": verdict,
                           "candidate_wants": verdict, "fit": {"type": "number", "minimum": 0, "maximum": 1}}}
    return {"type": "object", "additionalProperties": False, "required": list(ANSWER_KEYS),
            "properties": {"pairs": {"type": "array", "items": pair},
                           "hunts_one_person": {"type": "array", "items": label(flaggable)}}}


def read_answer(answer, candidates, labels) -> tuple:
    """Read a judge's answer strictly: ({candidate label: (focus_wants, candidate_wants, fit)}, {labels said to hunt
    one particular person}, Counter of what was dropped and why (DROPS)), or None if the answer isn't an object at
    all. Only the keys in ANSWER_KEYS and PAIR_KEYS, only labels of this request (a verdict's candidate among
    `candidates`, a hunt among `labels`), only the fixed verdicts and a number in [0, 1], each of exactly its plain
    type. Nothing else is kept: the label is looked up, the verdict is this code's constant, the fit a new float."""
    if type(answer) is not dict:
        return None
    dropped = Counter()
    dropped["key"] += sum(1 for k in answer if k not in ANSWER_KEYS)
    pairs = answer.get("pairs", [])
    if type(pairs) is not list:
        dropped["pair"] += 1
        pairs = []
    if len(pairs) > 2 * len(candidates) + 8:  # more than any honest answer: the rest isn't read
        dropped["pair"] += len(pairs) - (2 * len(candidates) + 8)
        pairs = pairs[:2 * len(candidates) + 8]
    verdicts = {}
    for item in pairs:
        if type(item) is not dict:
            dropped["pair"] += 1
            continue
        dropped["key"] += sum(1 for k in item if k not in PAIR_KEYS)
        if any(k not in item for k in PAIR_KEYS):
            dropped["pair"] += 1
            continue
        label, fw, cw, fit = (item[k] for k in PAIR_KEYS)
        if type(label) is not str or label not in candidates:
            dropped["label"] += 1
            continue
        if type(fw) is not str or fw not in _VERDICT or type(cw) is not str or cw not in _VERDICT:
            dropped["verdict"] += 1
            continue
        if type(fit) not in (int, float) or not 0 <= fit <= 1:  # NaN and infinities compare false
            dropped["fit"] += 1
            continue
        if label in verdicts:
            dropped["repeat"] += 1
            continue
        verdicts[candidates[candidates.index(label)]] = (_VERDICT[fw], _VERDICT[cw], float(fit))
    hunts = answer.get("hunts_one_person", [])
    if type(hunts) is not list:
        dropped["hunt"] += 1
        hunts = []
    if len(hunts) > 2 * len(labels) + 8:
        dropped["hunt"] += len(hunts) - (2 * len(labels) + 8)
        hunts = hunts[:2 * len(labels) + 8]
    hunting = set()
    for label in hunts:
        if type(label) is str and label in labels:
            hunting.add(labels[labels.index(label)])
        else:
            dropped["hunt"] += 1
    return verdicts, hunting, +dropped


class LLMMatcher:
    """The hybrid Matchmaker (see the module's docstring). Same interface as OfflineMatcher.

    judge(request) -> answer: plain data both ways. request = {"focus": label, "candidates": [labels],
    "listings": {label: entry}} (build_entry); answer = {"pairs": [{"candidate", "focus_wants", "candidate_wants",
    "fit"}], "hunts_one_person": [labels]}. It may raise: that request's pairs are then decided by the rules, or, for
    JudgeWaiting, wait for a later run. Whether a listing seems to hunt one particular person is asked about it alone,
    in a request with no candidates (a "hunt request"), so no other listing's text can shape it; a flag in a request
    with candidates is dropped and counted. The judge's verdicts on a focus listing's pairs are used only once it has
    been asked about that listing alone, for its current version.

    Kept for the audit and the presenter panel, never sent anywhere from here: payloads (every request sent, exactly
    as sent, at most PAYLOADS_KEPT, older ones counted in payloads_dropped), labels (each one's labels' listing ids),
    payload_changes (the board change each one was built at, so the audit reads the board as it was then;
    sent_requests() gives all three together, as leaks.leak_report reads them), run_payloads (the last run's
    requests), last_run (the last run's counts: pairs, judgeable, kept (verdicts kept from before), requests (with
    candidates), hunt_requests, sent, over_cap, failed and failed_requests (the judge raised or answered nothing
    readable), waiting and waiting_requests (its answer not there yet), unanswered, from_judge and from_rules
    (proposals), hunting, and dropped by DROPS) and decisions ({pair: how it was decided ("judge", "kept", "cap",
    "failed", "waiting", "unanswered", "hunting" (not asked: its focus listing seems to hunt one person), "hard", or
    "rules" with no judge), whether the rules proposed it, the judge's
    verdict, whether it was proposed, and to whom if a close match})."""

    def __init__(self, judge=None, rules: OfflineMatcher = None, max_pairs_per_run: int = MAX_PAIRS_PER_RUN,
                 max_per_request: int = MAX_PER_REQUEST):
        self.judge = judge
        self.rules = rules or OfflineMatcher()
        self.max_pairs_per_run = max(0, int(max_pairs_per_run))
        self.max_per_request = max(1, int(max_per_request))
        self.payloads, self.labels, self.payloads_dropped, self.run_payloads = [], [], 0, []
        self.payload_changes = []  # the board change each kept request was built at (parallel to payloads)
        self._change = 0  # the board change the current run reads
        self.last_run, self.decisions = {}, {}
        # ((id, entry), (id, entry)) -> (first one's wants, second one's wants, fit, the (id, entry) of every listing
        # in the request it came from): good only while every one of them is on the board unchanged
        self._cache = {}
        self._asked = {}  # listing id -> its (id, entry) when the judge was asked about it alone, and answered
        self._hunting = {}  # listing id -> its (id, entry) when the judge said it hunts one particular person
        self._secrets = None  # (board, its change, version, Redactor, ids, what the Redactor takes out)
        self._entries = {}  # listing id -> (listing, secrets version, entry, key)

    # -- the parts of a run ----------------------------------------------------------------------------------------

    def _redactor(self, board: Marketboard) -> tuple:
        """(version, Redactor, every listing and person id) for the board as it is and has been (every version of
        every listing): rebuilt when the board changes, and its version moves on only when what it takes out does."""
        got = self._secrets
        if got is not None and got[0] is board and got[1] == board.changes:
            return got[2], got[3], got[4]
        always, hidden, ids = set(), set(), set()
        # Every version of every listing, deleted ones too: a first name someone has since changed, and a text they
        # hid in an earlier version, are still theirs.
        for listing in board.history():
            ids.update((listing.person_id, listing.owner, board.person(listing.person_id).id))
            for f in listing.fields:
                if f.key == "first_name":
                    always.update(f.strings())
                if f.visibility == HIDDEN:
                    hidden.update(f.texts())
        always |= ids
        what = (frozenset(always), frozenset(hidden))
        if got is not None and got[5] == what:
            version, redactor = got[2], got[3]
        else:
            version, redactor = (got[2] + 1 if got else 1), Redactor(always, hidden)
            self._entries.clear()
        self._secrets = (board, board.changes, version, redactor, frozenset(ids), what)
        return version, redactor, frozenset(ids)

    def _entry(self, pid: str, listing, view: ShownView, kind, version: int, redactor: Redactor) -> tuple:
        """(entry, key) for one listing, built once per version of it (and of what the redactor takes out)."""
        got = self._entries.get(pid)
        if got is not None and got[0] is listing and got[1] == version:
            return got[2], got[3]
        entry = build_entry(view, kind, redactor)
        key = (pid, _frozen(entry))
        if len(self._entries) > 100_000:
            self._entries.clear()
        self._entries[pid] = (listing, version, entry, key)
        return entry, key

    def _hard(self, run: Reading, a: str, b: str) -> tuple:
        """(a's hard rules met exactly, b's, a's with a wide net when a opted in to close matches, b's)."""
        l, v, p = run.listings, run.views, run.parsed
        if how_they_meet(p[a], v[a], p[b], v[b]) is None:
            return False, False, False, False
        ea, eb = hard_met(p[a], v[a], p[b], v[b]), hard_met(p[b], v[b], p[a], v[a])
        wa = ea or (l[a].notify_close and eb and hard_met(p[a], _wide(l[a]), p[b], v[b]))
        wb = eb or (l[b].notify_close and ea and hard_met(p[b], _wide(l[b]), p[a], v[a]))
        return ea, eb, wa, wb

    @staticmethod
    def _possible(run: Reading, a: str, b: str, hard: tuple) -> bool:
        """Could any verdict make this pair a proposal? Both exactly, or one exactly and the other, opted in to close
        matches, with a wide net."""
        ea, eb, wa, wb = hard
        l = run.listings
        return (ea and eb) or (wa and eb and l[a].notify_close) or (wb and ea and l[b].notify_close)

    def _labels(self, ids: frozenset, count: int) -> list:
        """count labels for one request, none of them an id on the board."""
        for letter in LABEL_LETTERS:
            labels = [f"{letter}{i}" for i in range(1, count + 1)]
            if not ids.intersection(labels):
                return labels
        n = 0
        while True:  # every short letter is taken: a longer one
            n += 1
            labels = [f"{'c' * (n + 1)}{i}" for i in range(1, count + 1)]
            if not ids.intersection(labels):
                return labels

    @staticmethod
    def _may_flag(labels: list) -> list:
        """The labels a request's answer may say hunt one particular person: only its focus, and only in a request that
        holds it alone (no candidates), so what one listing says can never get another flagged."""
        return labels[:1] if len(labels) == 1 else []

    def _send(self, focus: str, cands: list, entries: dict, ids: frozenset, stats: Counter, dropped: Counter):
        """One request: the focus listing and its candidates, or (no candidates) the focus listing alone, to ask
        whether it seems to hunt one particular person. Returns ({candidate id: (focus_wants, candidate_wants, fit)},
        {ids said to hunt one particular person}), None if the judge failed or answered nothing readable, or _WAITING
        if its answer isn't there yet."""
        order = [focus, *cands]
        labels = self._labels(ids, len(order))
        to_id = dict(zip(labels, order))
        request = {"focus": labels[0], "candidates": labels[1:],
                   "listings": {label: _copy(entries[pid]) for label, pid in to_id.items()}}
        self.payloads.append(request)
        self.labels.append(dict(to_id))
        self.payload_changes.append(self._change)
        self.run_payloads.append(request)
        while len(self.payloads) > PAYLOADS_KEPT:
            del self.payloads[0], self.labels[0], self.payload_changes[0]
            self.payloads_dropped += 1
        stats["requests" if cands else "hunt_requests"] += 1
        stats["sent"] += len(cands)
        try:
            answer = self.judge(_copy(request))
        except JudgeWaiting:  # the answer is on its way: these pairs wait for a later run
            stats["waiting_requests"] += 1
            return _WAITING
        except Exception:  # the judge's failure, whatever it is: the rules decide these pairs
            stats["failed_requests"] += 1
            return None
        read = read_answer(answer, labels[1:], self._may_flag(labels))
        if read is None:
            dropped["answer"] += 1
            stats["failed_requests"] += 1
            return None
        verdicts, hunting, why = read
        dropped.update(why)
        return {to_id[label]: v for label, v in verdicts.items()}, {to_id[label] for label in hunting}

    def _proposal(self, run: Reading, a: str, b: str, verdict: tuple, hard: tuple):
        """The judge's verdict on a pair (a's wants, b's wants, fit), within the hard rules, as a Proposal, or None.
        A direction is met if the judge says "match" and the hard rules hold exactly; nearly met (a close match, only
        for a listing that opted in) if it says "match" or "close" and they hold with a wide net. Reasons, refs and
        hidden fields come from the rules' code; the fit is the shown score and must clear the rules' threshold."""
        a_wants, b_wants, fit = verdict
        ea, eb, wa, wb = hard
        l, v, p = run.listings, run.views, run.parsed
        a_met, b_met = a_wants == MATCH and ea, b_wants == MATCH and eb
        a_near = not a_met and a_wants in (MATCH, CLOSE) and wa and l[a].notify_close
        b_near = not b_met and b_wants in (MATCH, CLOSE) and wb and l[b].notify_close
        if a_met and b_met:
            close = ""
        elif a_near and b_met:
            close = a
        elif b_near and a_met:
            close = b
        else:
            return None
        av, bv = v[a], v[b]
        refs = detail_refs(av, bv)
        reasons = reason_fields(p[a].kind, av, bv, how_they_meet(p[a], av, p[b], bv) == ONLINE, refs)
        if not reasons:
            return None  # never the only reason: needs a shown reason
        threshold = shown_threshold(CLOSE_EXACTNESS if close == a else av.exactness,
                                    CLOSE_EXACTNESS if close == b else bv.exactness)
        if fit < threshold:
            return None  # never the only reason: what both show must justify it
        used, bonus = self.rules.hidden_part(run, a, b)
        return Proposal(a, b, a_met, b_met, round(fit + bonus, 4), round(fit, 4), round(threshold, 4), reasons, used,
                        close)

    # -- a run -------------------------------------------------------------------------------------------------------

    def match(self, board: Marketboard, focus=None, load=None) -> MatchResult:
        """As OfflineMatcher.match, with the judge deciding the fuzzy part of each pair the hard rules allow (see the
        module's docstring)."""
        rules = self.rules
        self._change = board.changes
        run = rules.read(board, focus)
        decided = {pair: rules.decide(run, *pair) for pair in run.pairs}
        stats, dropped, self.run_payloads = Counter(pairs=len(run.pairs)), Counter(), []
        verdicts, source, hunting = {}, {}, set()
        if self.judge is not None and run.pairs:
            verdicts, source, hunting = self._judged(board, run, stats, dropped)
        candidates, also_offer, decisions, waiting = [], {}, {}, set()
        for pair in run.pairs:
            a, b = pair
            how = source.get(pair, "rules" if self.judge is None else "hard")
            if pair in verdicts:
                proposal, offers = self._proposal(run, a, b, verdicts[pair][:3], verdicts[pair][3]), decided[pair][1]
            elif how == "waiting":
                proposal, offers = None, []  # decided on a later run, with the judge's answer
                waiting.update(pair)
            else:
                proposal, offers = decided[pair]
            if hunting & {a, b}:
                proposal, offers = None, []  # it seems to hunt one particular person: only ever dropped
            if proposal is not None:
                candidates.append(proposal)
                stats["from_judge" if pair in verdicts else "from_rules"] += 1
            for pid, fu in offers:
                if not hunting & {pid, fu.about}:
                    also_offer.setdefault(pid, fu)
            decisions[pair] = {"how": how, "rules": decided[pair][0] is not None,
                               "judge": verdicts[pair][:3] if pair in verdicts else None,
                               "proposed": proposal is not None, "close": proposal.close if proposal else ""}
        stats["hunting"] = len(hunting & set(run.listings))
        self.decisions = decisions
        self.last_run = {**{k: n for k, n in stats.items()}, "dropped": dict(dropped)}
        result = rules.finish(board, run, candidates, also_offer, focus, load)
        if waiting:  # "no match yet" isn't known while a pair of theirs waits for the judge
            result = MatchResult(proposals=result.proposals, followups=[
                fu for fu in result.followups if not (fu.question_id == "broaden" and fu.person_id in waiting)])
        return result

    def _judged(self, board: Marketboard, run: Reading, stats: Counter, dropped: Counter) -> tuple:
        """The judge's verdicts for this run: ({pair: (a's wants, b's wants, fit, hard)}, {pair: how it was decided},
        {ids that seem to hunt one particular person})."""
        hard = {pair: self._hard(run, *pair) for pair in run.pairs}
        judgeable = {pair for pair, h in hard.items() if self._possible(run, *pair, h)}
        stats["judgeable"] = len(judgeable)
        version, redactor, ids = self._redactor(board)
        entries, keys = {}, {}

        def entry_of(pid):
            if pid not in entries:
                entries[pid], keys[pid] = self._entry(pid, run.listings[pid], run.views[pid], run.parsed[pid].kind,
                                                      version, redactor)
            return entries[pid]

        def unchanged(key) -> bool:
            """Is this listing (id, entry) on the board, read in this run, and what the judge saw of it still?"""
            pid = key[0]
            if pid not in run.listings:
                return False
            entry_of(pid)
            return keys[pid] == key

        # One group per focus listing (the first listing, in board order, whose shortlist holds the pair), its
        # candidates in the order the shortlist ranks them.
        groups, assigned = [], set()
        for f, cands in run.lists.items():
            mine = []
            for b in cands:
                pair = (f, b) if f < b else (b, f)
                if pair in judgeable and pair not in assigned:
                    assigned.add(pair)
                    mine.append(b)
            if mine:
                groups.append((f, mine))

        verdicts, source, hunting = {}, {}, set()

        def keep(f, b, f_wants, b_wants, fit):
            pair = (f, b) if f < b else (b, f)
            a_wants, z_wants = (f_wants, b_wants) if pair[0] == f else (b_wants, f_wants)
            verdicts[pair] = (a_wants, z_wants, fit, hard[pair])

        # A verdict kept from before counts only while every listing of the request it came from is unchanged: what
        # the judge said about a pair rests on everything it was shown with it.
        pending = []
        for f, cands in groups:
            entry_of(f)
            left = []
            for b in cands:
                entry_of(b)
                got = self._cache.get((keys[f], keys[b]))
                if got is not None and all(unchanged(k) for k in got[3]):
                    keep(f, b, *got[:3])
                else:
                    got = self._cache.get((keys[b], keys[f]))
                    if got is not None and all(unchanged(k) for k in got[3]):
                        keep(f, b, got[1], got[0], got[2])
                    else:
                        left.append(b)
                        continue
                source[(f, b) if f < b else (b, f)] = "kept"
                stats["kept"] += 1
            pending.append((f, left))

        # Who is asked about alone this run: each listing the run matches (its focus listings) that is in a pair the
        # judge may decide, once for each version of it, at most max_pairs_per_run of them: those whose pairs the
        # judge is asked about first, in board order, then the rest. The others are asked in a later run, and until
        # then the judge decides none of the pairs it is asked about as their focus.
        heads = [f for f, _ in groups]
        involved = heads + sorted({x for pair in judgeable for x in pair if x in run.lists} - set(heads))
        for x in involved:
            entry_of(x)
        unknown = [x for x in involved if self._asked.get(x) != keys[x]]
        to_ask = unknown[:self.max_pairs_per_run]
        later = set(unknown[len(to_ask):])

        # The run's cap, shared out in turn: each focus listing's best-ranked candidates first (none of a focus
        # listing that won't have been asked about alone).
        for f, left in pending:
            if f in later:
                for b in left:
                    source[(f, b) if f < b else (b, f)] = "cap"
                    stats["over_cap"] += 1
                left.clear()
        chosen, budget, depth = {f: [] for f, _ in pending}, self.max_pairs_per_run, 0
        while budget > 0 and any(depth < len(left) for _, left in pending):
            for f, left in pending:
                if depth < len(left) and budget > 0:
                    chosen[f].append(left[depth])
                    budget -= 1
            depth += 1
        for f, left in pending:
            for b in left[len(chosen[f]):]:
                source[(f, b) if f < b else (b, f)] = "cap"
                stats["over_cap"] += 1

        alone = {x: "asked" for x in involved}  # -> "asked", "waiting", "failed" or "later"
        alone.update({x: "later" for x in later})
        for x in to_ask:
            got = self._send(x, [], entries, ids, stats, dropped)
            if got is _WAITING:
                alone[x] = "waiting"
            elif got is None:
                alone[x] = "failed"
            else:
                self._asked[x] = keys[x]
                if x in got[1]:
                    self._hunting[x] = keys[x]
                else:
                    self._hunting.pop(x, None)

        for f, sent in chosen.items():
            if self._hunting.get(f) == keys[f]:
                for b in sent:  # it seems to hunt one particular person: its pairs go, so they aren't asked about
                    source[(f, b) if f < b else (b, f)] = "hunting"
                continue
            for i in range(0, len(sent), self.max_per_request):
                part = sent[i:i + self.max_per_request]
                got = self._send(f, part, entries, ids, stats, dropped)
                if got is _WAITING:
                    for b in part:
                        source[(f, b) if f < b else (b, f)] = "waiting"
                    stats["waiting"] += len(part)
                    continue
                if got is None:
                    for b in part:
                        source[(f, b) if f < b else (b, f)] = "failed"
                    stats["failed"] += len(part)
                    continue
                answered, flagged = got
                for pid in flagged:  # (none: a request with candidates may flag nobody, _may_flag)
                    self._hunting[pid] = keys[pid]
                group = (keys[f], *(keys[b] for b in part))
                for b in part:
                    pair = (f, b) if f < b else (b, f)
                    if b not in answered:
                        source[pair] = "unanswered"
                        stats["unanswered"] += 1
                        continue
                    f_wants, b_wants, fit = answered[b]
                    if len(self._cache) > CACHE_MAX:
                        self._cache.clear()
                    self._cache[(keys[f], keys[b])] = (f_wants, b_wants, fit, group)
                    keep(f, b, f_wants, b_wants, fit)
                    source[pair] = "judge"

        # The judge's verdict on a pair counts only once its focus listing has been asked about alone (else it waits,
        # or the rules decide it), and never while the other listing's own answer is on its way or failed.
        focus_of = {((f, b) if f < b else (b, f)): f for f, cands in groups for b in cands}
        for pair in list(verdicts):
            states = {alone.get(x, "asked") for x in pair if x == focus_of[pair] or alone.get(x) != "later"}
            why = next((st for st in ("waiting", "failed", "later") if st in states), None)
            if why is not None:
                del verdicts[pair]
                source[pair] = "cap" if why == "later" else why
                stats["over_cap" if why == "later" else why] += 1

        # A listing the judge said hunts one particular person stays so until it changes.
        for pid, key in list(self._hunting.items()):
            if pid not in run.listings:
                continue
            if keys.get(pid, None) is None:
                entry_of(pid)
            if keys[pid] == key:
                hunting.add(pid)
            else:
                del self._hunting[pid]
        return verdicts, source, hunting

    def sent_requests(self) -> list:
        """Every kept request, for the independent audit (leaks.leak_report's judge_requests): {"request": exactly as
        sent, "labels": {label: listing id}, "board_change": the board change it was built at}."""
        return [{"request": request, "labels": dict(labels), "board_change": change}
                for request, labels, change in zip(self.payloads, self.labels, self.payload_changes)]

    # -- for the compare tool ----------------------------------------------------------------------------------------

    def ask(self, board: Marketboard, focus: str, candidates: list) -> tuple:
        """The judge's own verdicts on any pairs, without the hard rules or a proposal: ({candidate id: (focus_wants,
        candidate_wants, fit)}, set(): a request with candidates never flags anyone, see _may_flag), or None if the
        judge failed. The same request, the same strict reading, and the request is kept like every other. For measuring the rules against
        the judge (tools), never for matching."""
        self._change = board.changes
        version, redactor, ids = self._redactor(board)
        entries = {}
        for pid in [focus, *candidates]:
            listing = board.listing(pid)
            view = ShownView(listing)
            entries[pid] = self._entry(pid, listing, view, understood(view).kind, version, redactor)[0]
        stats, dropped = Counter(), Counter()
        self.run_payloads = []
        got = self._send(focus, list(candidates), entries, ids, stats, dropped)
        self.last_run = {**{k: n for k, n in stats.items()}, "dropped": dict(dropped)}
        return None if got is _WAITING else got
