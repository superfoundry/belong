"""Leak report (SPEC §8.6): an independent audit of everything sent out.

Deliberately does NOT reuse the Warden's code. It reads plain dicts (what was
actually delivered) plus the Marketboard, and asks: did any hidden value reach anyone
who shouldn't have it? Any finding is a test failure. (It reads values the way everything
does, through schema.canonical, and the published sensitive words and contact rules.)

It also audits the matching rules on the proposals (never the only reason,
only counts in a crowd), which can't be seen from outputs alone.

Audits:
  A1  introduction fields     every field shown in a view is shown by its owner, same value
                              (for an open detail, also the same compare, steps and required)
  A2  introduction text       nothing hidden by anyone (a value, or an open detail's name or
                              steps) appears in a view, except what the recipient may
                              legitimately see. A reason made only of the Warden's fixed
                              sentences (rebuilt here, not imported) is read by character
                              position: a hidden value is let go only if every character of
                              it is fixed wording in every reading, or inside a string the
                              recipient may see; any other reason is read whole
  A3  follow-ups              text is exactly the fixed-menu text; no hidden values
  A4  reveals                 only shared details, between connected people, both consenting,
                              hidden by at least one of them, never a wish about the other person
  A5  receipts                exactly {seq, check, result, timestamp, commitment, hash};
                              no content
  A6  receipt chain           every receipt's hash links to the one before it (nothing
                              deleted, reordered or edited). Re-implemented here, not
                              imported from the Warden.
  A7  details in use          each shared list is exactly what an independent recount of the
                              board it was built from gives: only what K different people show
                              on active listings old enough, nothing hidden or sensitive, short
                              plain text of a few words with no sentence punctuation,
                              alphabetical, within the caps
  R1  never the only reason   every proposal clears the shown-only threshold and has
                              reason fields shown by both; its open-detail reasons are exactly
                              the ones both people's shown details give
  R2  only counts in a crowd  every hidden field used had a shared value that k different
                              people besides the pair have (recounted here, by people)
  R3  approach limit          no listing introduced more than the daily limit in any 24 hours
                              (an introduction without a send time counts as the same day)
  R4  distinct people         no one is introduced to their own other listing
  R5  opt-ins                 follow-ups only to listings that opted in to questions,
                              except "include close matches?" (the opt-in itself): at most
                              once per listing, and only to one that had close matches off;
                              close matches only to listings that opted in to them, and
                              each one names one of its two people as its one recipient
  R6  one id, one listing     a listing id keeps the owner it was first accepted under and is
                              never accepted again once its listing is deleted, so an
                              introduction and its reissues read only the listing it was sent to
  A8  what the Matchmaker AI  every request sent to the Matchmaker AI's judge (the hybrid Matchmaker's
      was sent                record of it, read against the board as it was when the request was
                              built, from the board's own record of changes): only the fixed shape;
                              each label a listing that was live then and never an id; each entry
                              only what its listing shows (its shown phrases, topics, pay and open
                              details) and its description, each its owner's own text with stretches
                              taken out and nothing added; no first name and no listing or person id
                              of anyone anywhere in it; a hidden text of anyone (a value, an open
                              detail's name or steps) only inside a text that listing shows itself.
                              Each watched text is found however its marks are written (a character
                              that shows nothing, a curly or straight quote mark or apostrophe, a dash)
                              and with or without the punctuation at its ends, and a first name
                              counts from when it was first on the board, for ever after (a name since
                              changed is still that person's). And every request the model's own
                              record says went out is one of those (wire_requests, while the
                              Matchmaker still keeps every request) and one the Warden's check let
                              through (wire_checked, always). Its own code: nothing from the
                              Matchmaker, the judge or the Warden.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from datetime import datetime, timedelta

from .schema import (CONTACT_RULES, FOLLOWUP_MENU, HIDDEN, LIST_CAPS, LIST_LONG_RUN, LIST_LONG_RUN_CHARS,
                     LIST_MIN_AGE_DAYS, LIST_PHRASE_KEYS,
                     LIST_PHRASE_WORDS, LIST_TEXT_MAX, LIST_UNSPACED_CHARS, LIST_UNSPACED_SCRIPTS, LIST_VALUE_WORDS,
                     MATCH_KINDS, SHOWN, STRICT_EXACTNESS, canonical, detail_reasons, mentions_sensitive)

RECEIPT_KEYS = {"seq", "check", "result", "timestamp", "commitment", "hash"}
_HEX64 = re.compile(r"[0-9a-f]{64}")


def _shown(listing, key):
    f = listing.get(key)
    return f.value if f is not None and f.visibility == SHOWN else None


def _about(key) -> str:
    """"me", "them" or "both" for an open detail's key; "" otherwise."""
    about, sep, name = str(key).partition(":")
    return about if sep and name and about in ("me", "them", "both") else ""


def _ref(key):
    """(kind, name) for an open-detail reason ("agree:{name}", "wants:{name}"), else None."""
    kind, sep, name = str(key).partition(":")
    return (kind, name) if sep and name and kind in ("agree", "wants") else None


def _said(f) -> list:
    """Everything a field says: its values, and an open detail's name and steps."""
    return f.strings() + ([f.key.partition(":")[2], *f.steps] if _about(f.key) else [])


def _as_view(f):
    """A field as a view should carry it (re-implemented, not taken from the Warden)."""
    if _about(f.key):
        return {"values": list(f.value), "compare": f.compare, "steps": list(f.steps), "required": f.required}
    return list(f.value) if isinstance(f.value, tuple) else f.value


def _view_text(view, with_reason: bool = True) -> str:
    """Every piece of person data in a view, one per line: the reason (unless with_reason is False, when the
    caller reads it by its slots), open details' names, values and steps. Built-in keys and the view's fixed
    structure are left out; anything of an unexpected shape is read whole."""
    if not isinstance(view, dict):
        return json.dumps(view, ensure_ascii=False)
    parts = []
    for top, v in view.items():
        if top == "reason" and isinstance(v, str):
            if with_reason:
                parts.append(v)
        elif top == "fields" and isinstance(v, dict):
            for k, x in v.items():
                if _about(k):
                    parts.append(k.partition(":")[2])
                if isinstance(x, str):
                    parts.append(x)
                elif isinstance(x, list) and all(isinstance(s, str) for s in x):
                    parts += x
                elif isinstance(x, dict) and set(x) <= {"values", "compare", "steps", "required"} \
                        and all(isinstance(x.get(n, []), list) and all(isinstance(s, str) for s in x.get(n, []))
                                for n in ("values", "steps")):
                    parts += x.get("values", []) + x.get("steps", [])
                else:
                    parts.append(json.dumps({k: x}, ensure_ascii=False))
        else:
            parts.append(json.dumps({top: v}, ensure_ascii=False))
    return "\n".join(parts)


# The fixed sentences the Warden writes reasons with, rebuilt here (its own copy, nothing imported from the
# Warden): each {} is a slot, filled with something someone wrote; everything else is fixed wording.
_REASON_SKELETONS = [tuple(s.split("{}")) for s in (
    "You're both in {}.", "You're in {}; they're in {}.", "You're near one another: {} and {}.",
    "You can both meet online.", "You can both meet in person.", "You'd meet {}; they'd meet {}.",
    "They're looking for {}.", "They're {}.",
    "You share the same {}: {}.", "Their {} is {}; yours is {}.",
    "You'd like their {} to be {}; theirs is {}.", "You'd like their {} to be {}.",
    "They'd like your {} to be {}; yours is {}.", "They'd like your {} to be {}.",
)]
_SPLITS_MAX = 4096  # more ways to read one sentence than any reason the Warden writes: the reason is read whole


def _splits(text: str, start: int, end: int, pieces: tuple):
    """Every way text[start:end] is this one fixed sentence: a list of readings, each the (start, end) of its slots
    (a slot holds at least one character); [] if it isn't the sentence, None if there are too many ways."""
    head, tail = pieces[0], pieces[-1]
    if len(pieces) == 1:
        return [[]] if text[start:end] == head else []
    middle = pieces[1:-1]
    last = end - len(tail)  # where the last slot ends
    if last - start < len(head) + len(pieces) - 1 or not text.startswith(head, start) \
            or not text.endswith(tail, start, end):
        return []
    out = []

    def walk(pos: int, i: int, spans: list) -> None:
        if len(out) > _SPLITS_MAX:
            return
        if i == len(middle):
            if last > pos:
                out.append(spans + [(pos, last)])
            return
        q = text.find(middle[i], pos + 1, last)
        while q != -1 and len(out) <= _SPLITS_MAX:
            walk(q + len(middle[i]), i + 1, spans + [(pos, q)])
            q = text.find(middle[i], q + 1, last)
    walk(start + len(head), 0, [])
    return None if len(out) > _SPLITS_MAX else out


def _readings(reason: str):
    """How a reason reads as the fixed sentences (each ends in a full stop; they are joined by one space): (the
    positions that some reading puts in a slot, the slots of the reading with the least in its slots), or None if
    no reading covers all of it (or it is longer or more ambiguous than any reason the Warden writes), and then
    the caller reads the whole reason. A position no reading puts in a slot is fixed wording in every reading.

    The readings are every way to read it as the most sentences it can be: every skeleton each sentence fits
    ("They're looking for X." is also "They're {}."), and every place each slot can end. A slot may hold a full
    stop and a space (a value "St. Ives"), but never swallows a whole fixed sentence that also reads as one on
    its own ("You're both in X. You'd like..." is two sentences, not one slot "X. You'd like..."): only fixed
    wording, which says nothing about anyone, is ever let go that way, and what fills that sentence is still
    read."""
    n = len(reason)
    ends = [e for e in range(1, n + 1) if reason[e - 1] == "." and (e == n or reason[e] == " ")]
    if n > 2000 or len(ends) > 64:
        return None
    starts = sorted({0} | {e + 1 for e in ends if e < n})
    after = lambda e: e + 1 if e < n else n  # where the next sentence starts
    sentences = {}  # start -> [(end, its readings)]
    for start in starts:
        for end in (e for e in ends if e > start):
            for pieces in _REASON_SKELETONS:
                ways = _splits(reason, start, end, pieces)
                if ways is None:
                    return None
                if ways:
                    sentences.setdefault(start, []).append((end, ways))
    # start -> (the most sentences, then the least in slots, and those slots) for a reading of the rest
    rest = {n: (0, 0, [])}
    for start in reversed(starts):
        best = None
        for end, ways in sentences.get(start, ()):
            then = rest.get(after(end))
            for spans in ways if then is not None else ():
                got = (then[0] + 1, sum(b - a for a, b in spans) + then[1], [reason[a:b] for a, b in spans] + then[2])
                if best is None or (-got[0], got[1]) < (-best[0], best[1]):
                    best = got
        if best is not None:
            rest[start] = best
    if 0 not in rest:
        return None
    most, before, slotted = rest[0][0], {0: 0}, set()  # before: start -> the most sentences up to there
    for start in starts:
        for end, ways in sentences.get(start, ()) if start in before else ():
            nxt = after(end)
            before[nxt] = max(before.get(nxt, -1), before[start] + 1)
            if nxt in rest and before[start] + 1 + rest[nxt][0] == most:  # in a reading with the most sentences
                slotted.update(i for spans in ways for a, b in spans for i in range(a, b))
    return slotted, rest[0][2]


def _reason_slots(reason: str):
    """What fills a reason's sentences in the reading with the least in its slots, if every sentence of it is one
    of the fixed sentences; None if not."""
    got = _readings(reason) if isinstance(reason, str) else None
    return None if got is None else got[1]


def _as_quoted(headline):
    """Re-implemented, not imported from the Warden: a headline as a question quotes it."""
    if not isinstance(headline, str) or not re.match(r"^[A-Za-z]+ing\b", headline):
        return None
    h = headline.strip().rstrip(".")
    return h[:1].lower() + h[1:]


_WORDS: dict = {}  # value -> its compiled whole-word pattern (Python's own cache is too small here)


def _word(s: str):
    pattern = _WORDS.get(s)
    if pattern is None:
        if len(_WORDS) > 200_000:
            _WORDS.clear()
        pattern = _WORDS[s] = re.compile(r"(?<!\w)" + re.escape(s) + r"(?!\w)", re.I)
    return pattern


def _contains(text: str, s: str) -> bool:
    return _word(s).search(text) is not None


_TOKENS = re.compile(r"\w+")


def _finder(values):
    """Which of `values` appear in a text as whole words: the same answer as checking each value with
    _contains, but a value is only checked when its first word is in the text (a value can't appear
    otherwise), so a large population doesn't mean searching every text for every value."""
    by_first, always = {}, []
    for v in values:
        words = _TOKENS.findall(v.lower())
        (by_first.setdefault(words[0], []) if words else always).append((v, frozenset(words)))

    def found(text: str) -> list:
        words = set(_TOKENS.findall(text.lower()))
        hits = [v for w in words for v, need in by_first.get(w, ()) if need <= words and _contains(text, v)]
        return sorted(set(hits + [v for v, _ in always if _contains(text, v)]))
    return found


def _occurrences(pattern, text: str):
    """Every match of a whole-word pattern in the text, overlapping ones included."""
    m = pattern.search(text)
    while m:
        yield m
        m = pattern.search(text, m.start() + 1)


_JOINER = re.compile(r", | and | or ")


def _may_see_at(text: str, allowed: set) -> set:
    """The positions of the text inside an occurrence of a string the recipient may see (also as a sentence writes
    it, without a closing full stop), or in a ", ", " and " or " or " that only joins two of them in a list."""
    allowed = {x for s in allowed for x in (s, s.strip().rstrip(".")) if x}
    spans = [(m.start(), m.end()) for s in allowed for m in _occurrences(_word(s), text)]
    starts, ends = {a for a, _ in spans}, {b for _, b in spans}
    spans += [(m.start(), m.end()) for m in _JOINER.finditer(text) if m.start() in ends and m.end() in starts]
    return {i for a, b in spans for i in range(a, b)}


class Said:
    """What an introduction's view says, as A2 reads it: its text, one piece of person data per line (see
    _view_text), and which of its characters are explained: fixed wording in every reading of the reason, or
    inside a string the recipient may see. A value is said only by an occurrence that isn't all explained."""

    def __init__(self, text: str, explained: set):
        self.text, self._explained = text, explained

    def says(self, value: str) -> bool:
        return any(any(not c.isspace() and i not in self._explained for i, c in enumerate(m.group(), m.start()))
                   for m in _occurrences(_word(value), self.text))

    def __str__(self) -> str:
        return self.text


def view_said(view, allowed: set) -> Said:
    """What an introduction's view says about anyone, as A2 reads it (see Said), given the strings its recipient
    may see (`allowed`). The view's fields are read in full. A reason made only of the Warden's fixed sentences is
    read by character position: a value in it is explained only if every character of it is fixed wording in
    every way the reason can be read as those sentences (_readings), or inside an occurrence of an allowed string
    in the reason. So a value cut across a slot boundary, or across fixed words, is still said. Any other reason
    is read whole."""
    reason = view.get("reason") if isinstance(view, dict) else None
    if not isinstance(reason, str):
        text = _view_text(view)
        return Said(text, _may_see_at(text, allowed))
    fields = _view_text(view, with_reason=False)
    text = f"{fields}\n{reason}" if fields else reason
    offset, read = len(text) - len(reason), _readings(reason)
    fixed = set() if read is None else {offset + i for i in range(len(reason)) if i not in read[0]}
    return Said(text, fixed | _may_see_at(text, allowed))


def leak_report(board, introductions: list, followups: list, reveals: list, receipts: list,
                proposals: list, connections: set, consents: dict, k_crowd: int,
                approach_limit: int, details_lists: list = (), judge_requests=None, wire_requests=(),
                wire_checked=None) -> dict:
    """All arguments except board/proposals are plain dicts as delivered.

    connections: set of frozenset({a, b}) pairs that both said yes.
    consents: {(intro_id, person_id): set of keys} reveal opt-ins.
    details_lists: the shared lists of details in use, as DetailsInUse.to_dict() delivered them.
    judge_requests: with the Matchmaker AI, every request sent to its judge, as {"request", "labels" (label ->
    listing id), "board_change" (the board change it was built at)} (audit A8); None without it.
    wire_requests: the requests the model's own record says went out, each of which must be one of judge_requests.
    wire_checked: (requests the model's own record says went out, the SHA-256 digests of every request the Warden's
    check let through to it): each of those requests must be one the Warden let through. Unlike wire_requests, it
    doesn't depend on how many requests the Matchmaker still keeps.
    """
    listings = {l.person_id: l for l in board.listings(include_withdrawn=True)}  # what was sent, even if since deleted

    def strings(pid, visibility=None):
        return {s for f in listings[pid].fields
                if visibility is None or f.visibility == visibility for s in _said(f)}

    every_hidden = set().union(*(strings(pid, HIDDEN) for pid in listings)) if listings else set()
    hidden_in = _finder(every_hidden)
    findings = []

    def find(audit, detail):
        findings.append({"audit": audit, "detail": detail})

    # A1 + A2: introductions
    for intro in introductions:
        for viewer, other, key in ((intro["person_a"], intro["person_b"], "a_view"),
                                   (intro["person_b"], intro["person_a"], "b_view")):
            view = intro[key]
            for k, v in view["fields"].items():
                f = listings[other].get(k)
                if f is None or f.visibility != SHOWN:
                    find("A1", f"{intro['intro_id']} {key}: field {k!r} is not shown by its owner {other}")
                elif _as_view(f) != v:
                    find("A1", f"{intro['intro_id']} {key}: field {k!r} value differs from owner's")
            # The recipient may see: the counterpart's shown values (and shown open details' names and
            # steps); their own values only for keys BOTH show, and for open details of a name both
            # show (those are the only things a reason may use).
            mine, theirs = listings[viewer], listings[other]
            both_shown = set(mine.shown()) & set(theirs.shown())
            names = {k.partition(":")[2] for k in theirs.shown() if _about(k)}
            allowed = strings(other, SHOWN) | {s for k in both_shown for s in mine.get(k).strings()}
            allowed |= {s for f in mine.fields if f.visibility == SHOWN and _about(f.key)
                        and f.key.partition(":")[2] in names for s in _said(f)}
            said = view_said(view, allowed)
            for s in hidden_in(said.text):
                if said.says(s):
                    find("A2", f"{intro['intro_id']} {key}: hidden value {s!r} reached {viewer}")

    # A3: follow-ups
    for fu in followups:
        template = FOLLOWUP_MENU.get(fu["question_id"])
        expected = template.format(phrase=fu.get("phrase", "")) if template else None
        if expected != fu["question_text"] or set(fu) != {"person_id", "question_id", "question_text", "phrase",
                                                          "offer_add", "seek_add", "pay"} \
                or ((fu.get("phrase") or fu.get("offer_add") or fu.get("seek_add") or fu.get("pay"))
                    and fu["question_id"] != "also_offer"):
            find("A3", f"follow-up to {fu['person_id']} is not a fixed-menu question")
        # Exactly the fixed text: only its phrase can carry anything about anyone (the fixed words can't,
        # whatever someone has hidden, a detail called "work", say). Otherwise, all of it.
        said = (fu.get("phrase") or "") if expected == fu["question_text"] else fu["question_text"]
        for s in hidden_in(said):
            find("A3", f"follow-up to {fu['person_id']} contains hidden value {s!r}")

    # A4: reveals
    intro_by_pair = {frozenset((i["person_a"], i["person_b"])): i["intro_id"] for i in introductions}
    for rv in reveals:
        pair = frozenset((rv["person_a"], rv["person_b"]))
        intro_id = intro_by_pair.get(pair)
        label = f"reveal of {rv['key']!r} between {rv['person_a']} and {rv['person_b']}"
        if pair not in connections or intro_id is None:
            find("A4", f"{label}: they are not connected")
            continue
        if _about(rv["key"]) == "them":
            find("A4", f"{label}: a wish about the other person is never revealed")
        revealed = {canonical(v) for v in rv["values"]}
        held = []
        for p in pair:
            if rv["key"] not in consents.get((intro_id, p), set()):
                find("A4", f"{label}: {p} did not consent")
            f = listings[p].get(rv["key"])
            held.append(f)
            if f is None or not revealed or not revealed <= {canonical(s) for s in f.strings()}:
                find("A4", f"{label}: revealed value is not shared by {p}")
        if all(held) and HIDDEN not in {f.visibility for f in held}:
            find("A4", f"{label}: neither of them hides it")

    # A5: receipts carry no content
    all_values = set().union(*(strings(pid) for pid in listings)) if listings else set()
    value_in = _finder({v for v in all_values if len(v) > 2})
    for r in receipts:
        if (set(r) != RECEIPT_KEYS or r["result"] not in ("pass", "fail") or not isinstance(r["seq"], int)
                or not _HEX64.fullmatch(str(r["hash"])) or not _HEX64.fullmatch(str(r["commitment"]))):
            find("A5", f"receipt has unexpected shape: {sorted(r)}")
            continue
        blob = f"{r['check']} {r['timestamp']}"
        for s in value_in(blob):
            find("A5", f"receipt for {r['check']} contains a field value")

    # A6: the chain is unbroken (the first receipt given is taken as the anchor)
    shaped = [r for r in receipts if set(r) == RECEIPT_KEYS]
    for prev, r in zip(shaped, shaped[1:]):
        expected = hashlib.sha256(f"{prev['hash']}|{r['seq']}|{r['check']}|{r['result']}|{r['timestamp']}|"
                                  f"{r['commitment']}".encode()).hexdigest()
        if r["seq"] != prev["seq"] + 1 or r["hash"] != expected:
            find("A6", f"receipt chain broken at seq {r['seq']}")
            break

    # A7: the shared lists of details in use, each rebuilt here from the board as it was
    entries = 0
    for d in details_lists:
        entries += sum(len(d.get(s) or []) for s in _LIST_SECTIONS)
        for problem in _list_problems(d, board, k_crowd):
            find("A7", f"details in use built {d.get('built_at')}: {problem}")

    # A8: what was sent to the Matchmaker AI
    judged = entries_sent = 0
    if judge_requests is not None:
        watch = _JudgeWatch(board) if judge_requests else None
        kept = set()
        for sent in judge_requests:
            judged += 1
            problems, n = _judge_request_problems(watch, sent)
            entries_sent += n
            for problem in problems:
                find("A8", f"request {judged} to the Matchmaker AI: {problem}")
            try:
                kept.add(json.dumps(sent["request"], ensure_ascii=False, sort_keys=True))
            except (TypeError, ValueError, KeyError):
                pass
        for i, request in enumerate(wire_requests, 1):
            try:
                text = json.dumps(request, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                text = None
            if text not in kept:
                find("A8", f"request {i} in the model's own record went out without being kept for this audit")
        if wire_checked is not None:
            went, passed = wire_checked
            for i, request in enumerate(went, 1):
                try:
                    digest = hashlib.sha256(json.dumps(request, ensure_ascii=False, sort_keys=True).encode()).digest()
                except (TypeError, ValueError):
                    digest = None
                if digest not in passed:
                    find("A8", f"request {i} in the model's own record went out without passing the Warden's check")

    # R1, R2: matching rules
    people = {}  # (key, canonical value) -> the different people who have it on an active listing
    for l in board.listings():
        if l.active:
            for f in l.fields:
                for s in f.strings():
                    people.setdefault((f.key, canonical(s)), set()).add(l.owner)
    for p in proposals:
        pair = f"{p.person_a}+{p.person_b}"
        a, b = listings[p.person_a], listings[p.person_b]
        if not (_number(p.shown_score) and _number(p.shown_threshold) and p.shown_score >= p.shown_threshold) \
                or not p.reason_fields:
            find("R1", f"{pair}: not justified by shown fields alone")
        for k in p.reason_fields:
            if _ref(k) is None and (k not in a.shown() or k not in b.shown()):
                find("R1", f"{pair}: reason field {k!r} not shown by both")
            elif _ref(k) is not None and not _ref_holds(*_ref(k), a, b):
                find("R1", f"{pair}: reason {k!r} isn't in what both people show")
        refs = tuple(sorted(k for k in p.reason_fields if _ref(k) is not None))
        strict = max(a.exactness, b.exactness) >= STRICT_EXACTNESS
        if refs != detail_reasons(a.shown_fields(), b.shown_fields(), strict):
            find("R1", f"{pair}: open-detail reasons aren't exactly the ones both people's shown details give")
        for k in p.used_hidden_fields:
            fa, fb = a.get(k), b.get(k)
            if fa is None or fb is None or _about(k) == "them" or HIDDEN not in (fa.visibility, fb.visibility):
                find("R2", f"{pair}: {k!r} isn't a hidden field both of them have")
                continue
            shared = {canonical(s) for s in fa.strings()} & {canonical(s) for s in fb.strings()}
            if not any(len(people.get((k, s), ())) >= k_crowd + 2 for s in shared):
                find("R2", f"{pair}: hidden field {k!r} used without a crowd of {k_crowd} other people")

    # R4: never introduced to yourself
    for intro in introductions:
        a, b = listings.get(intro["person_a"]), listings.get(intro["person_b"])
        if a and b and a.owner == b.owner:
            find("R4", f"{intro['intro_id']}: two listings of the same person were introduced")

    # R6: one id, one listing (read from the board's own record of changes, in order).
    first_owner, deleted = {}, set()
    for _, lid, version in board.events():
        if version is None:
            deleted.add(lid)
        elif lid in deleted:
            find("R6", f"listing id {lid} was accepted again after its listing was deleted")
        elif first_owner.setdefault(lid, version.owner) != version.owner:
            find("R6", f"listing id {lid} was taken by another person's listing")

    # R5: notification opt-ins
    broadened = {}
    for fu in followups:
        l = listings.get(fu["person_id"])
        if fu["question_id"] == "broaden":
            # The opt-in question itself: once, and only while close matches were off.
            broadened[fu["person_id"]] = broadened.get(fu["person_id"], 0) + 1
            was_off = any(v.person_id == fu["person_id"] and not v.notify_close for v in board.history())
            if broadened[fu["person_id"]] > 1 or not was_off:
                find("R5", f"'include close matches?' sent to {fu['person_id']} again, or with close matches on")
        elif l is not None and not l.notify_questions:
            find("R5", f"follow-up sent to {fu['person_id']}, who didn't opt in to questions")
    # An "also offer" question quotes the shown phrase of someone who opted in to close matches.
    for fu in followups:
        if fu["question_id"] != "also_offer":
            continue
        # Checked against every version a listing has had: the question quotes the one live when it was sent.
        sources = [l for l in board.history() if l.notify_close and l.owner != getattr(listings.get(fu["person_id"]), "owner", None)
                   and _shown(l, "seeking") == fu["offer_add"] and _shown(l, "offering") == fu["seek_add"]
                   and (_shown(l, "pay") or "") == fu["pay"]
                   and fu["phrase"] in (_as_quoted(_shown(l, "headline")), f"looking for {_shown(l, 'seeking')}")]
        if not sources:
            find("R5", f"question to {fu['person_id']} quotes a phrase no opted-in listing shows")
    for intro in introductions:
        to = intro.get("one_way_to") or ""
        if not to:
            continue
        l = listings.get(to)
        if to not in (intro["person_a"], intro["person_b"]) or l is None or not l.notify_close:
            find("R5", f"{intro['intro_id']}: close match sent to {to}, who didn't opt in to close matches")

    # R3: approach limit, per rolling 24 hours. Re-implemented here, not taken from the Warden.
    def sent(intro):
        try:
            return datetime.fromisoformat(str(intro.get("sent_at") or "").rstrip("Z"))
        except ValueError:
            return None
    times: dict = {}
    for intro in introductions:
        for p in (intro["person_a"], intro["person_b"]):
            times.setdefault(p, []).append(sent(intro))
    for person, ts in times.items():
        if None in ts:  # no send time to go on: count them all as one day
            worst = len(ts)
        else:
            ts = sorted(ts)
            worst = max(sum(1 for u in ts if t <= u < t + timedelta(days=1)) for t in ts)
        if worst > approach_limit:
            find("R3", f"{person} was introduced {worst} times in one day (limit {approach_limit})")

    checked = {
        "introductions": len(introductions),
        "views": 2 * len(introductions),
        "followups": len(followups),
        "reveals": len(reveals),
        "receipts": len(receipts),
        "proposals": len(proposals),
        "hidden_values_watched": len(every_hidden),
        "details_lists": len(details_lists),
        "details_entries": entries,
    }
    if judge_requests is not None:  # only with the Matchmaker AI
        checked.update(judge_requests=judged, judge_entries=entries_sent, wire_requests=len(wire_requests))
        if wire_checked is not None:
            checked.update(wire_checked=len(wire_checked[0]))
    return {"ok": not findings, "findings": findings, "checked": checked}


def _number(x) -> bool:
    """A plain finite number (a score that isn't one never clears a threshold)."""
    return type(x) in (int, float) and math.isfinite(x)


def _own_shown(listing, name):
    for about in ("me", "both"):
        f = listing.get(f"{about}:{name}")
        if f is not None and f.visibility == SHOWN:
            return f
    return None


def _ref_holds(kind: str, name: str, a, b) -> bool:
    """Does an open-detail reason rest only on what both people show? "agree": both show a value of their
    own; "wants": one shows a wish and the other shows a value of their own."""
    if kind == "agree":
        return _own_shown(a, name) is not None and _own_shown(b, name) is not None
    wish = lambda l: (l.get(f"them:{name}") is not None and l.get(f"them:{name}").visibility == SHOWN)
    return (wish(a) and _own_shown(b, name) is not None) or (wish(b) and _own_shown(a, name) is not None)


# --------------------------------------------------------------------------------------
# A7: the shared list, recounted here from the board (its own code; nothing from the Warden).
# --------------------------------------------------------------------------------------

_LIST_SECTIONS = ("details", "scales", "values", "phrases", "places")
_LIST_SHAPES = {"details": {"about", "name", "compare"}, "scales": {"about", "name", "steps"},
                "values": {"about", "name", "value"}, "phrases": {"key", "value"}, "places": {"city", "country"}}
_NOT_PLAIN = re.compile(r"[<>{}\[\]\\`|]|://")


def _when(ts):
    """A time as the Warden's clock writes it ("...Z"), or None."""
    try:
        t = datetime.fromisoformat(str(ts).rstrip("Z"))
    except ValueError:
        return None
    return t if t.tzinfo is None else None


_CLAUSE_MARKS = re.compile(r"FULL STOP|COMMA|COLON|QUESTION MARK|EXCLAMATION MARK")


def _short_plain(text, words: int) -> bool:
    """Short plain text (no symbols that could carry markup, no control or invisible characters), no
    contact details, no sensitive word, at most `words` words (_word_count), and no sentence punctuation in any
    script (punctuation named a full stop, comma, colon, semicolon, question or exclamation mark)."""
    return (isinstance(text, str) and 0 < len(text) <= LIST_TEXT_MAX and not _NOT_PLAIN.search(text)
            and all(unicodedata.category(c)[0] != "C" for c in text)
            and not any(rx.search(text) for _, rx in CONTACT_RULES) and not mentions_sensitive(text)
            and _word_count(text) <= words
            and not any(unicodedata.category(c)[0] == "P" and _CLAUSE_MARKS.search(unicodedata.name(c, ""))
                        for c in text))


def _word_count(text: str) -> int:
    """The words of a list text, counted here with code of its own: the text is decomposed for compatibility (NFKD,
    so a circled or full-width letter is its letter), each run of letters and digits (with the marks on them, and
    any symbol whose Unicode name says it is a LETTER, such as a negatively squared one) is a word, split by any other
    character and where a lower-case letter is followed by a capital; a run in a script written without spaces
    (LIST_UNSPACED_SCRIPTS) is one word per LIST_UNSPACED_CHARS letters, rounded up, and any other run longer than
    LIST_LONG_RUN letters one word per LIST_LONG_RUN_CHARS letters, rounded up."""
    runs, current = [], []
    for c in unicodedata.normalize("NFKD", text):
        cat = unicodedata.category(c)
        if cat == "So" and "LETTER" in unicodedata.name(c, "").split():
            cat = "Lo"
        if cat[0] == "M" and current:
            continue
        if cat[0] in ("L", "N"):
            if current and unicodedata.category(current[-1]) == "Ll" and cat in ("Lu", "Lt"):
                runs.append(current)
                current = []
            current.append(c)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)

    def count(run) -> int:
        if any(unicodedata.name(c, "").startswith(LIST_UNSPACED_SCRIPTS) for c in run):
            return math.ceil(len(run) / LIST_UNSPACED_CHARS)
        return 1 if len(run) <= LIST_LONG_RUN else math.ceil(len(run) / LIST_LONG_RUN_CHARS)
    return sum(count(run) for run in runs)


def _words_allowed(section: str) -> int:
    return LIST_PHRASE_WORDS if section == "phrases" else LIST_VALUE_WORDS


def _list_words(section: str, entry: dict) -> list:
    if section == "details":
        return [entry["name"]]
    if section == "scales":
        return list(entry["steps"])
    if section == "places":
        return [entry["city"], entry["country"]]
    return [entry["value"]]


def _recount(listings: list, built_at, k_crowd: int) -> dict:
    """The shared list as the rules give it for these listings at that time."""
    built = _when(built_at)
    oldest_allowed = built - timedelta(days=LIST_MIN_AGE_DAYS)
    tally = {}  # (section, canonical identity) -> {person: {the ways they write it}}
    for l in listings:
        updated = _when(l.updated_at)
        if not l.active or updated is None or updated > oldest_allowed:
            continue
        found = []
        for f in l.fields:
            if f.visibility != SHOWN or f.sensitive:
                continue
            about = _about(f.key)
            if about:
                name = f.key.partition(":")[2]
                found.append((("details", about, name, f.compare),
                              {"about": about, "name": name, "compare": f.compare}))
                if f.compare == "steps":
                    found.append((("scales", about, name, tuple(map(canonical, f.steps))),
                                  {"about": about, "name": name, "steps": list(f.steps)}))
                found += [(("values", about, name, canonical(v)), {"about": about, "name": name, "value": v})
                          for v in f.strings()]
            elif f.key in LIST_PHRASE_KEYS:
                found += [(("phrases", f.key, canonical(v)), {"key": f.key, "value": v}) for v in f.strings()]
        city, country = l.get("city"), l.get("country")
        if city is not None and country is not None and SHOWN == city.visibility == country.visibility \
                and isinstance(city.value, str) and isinstance(country.value, str):
            found.append((("places", canonical(city.value), canonical(country.value)),
                          {"city": city.value, "country": country.value}))
        for identity, entry in found:
            tally.setdefault(identity, {}).setdefault(l.owner, set()).add(json.dumps(entry, sort_keys=True,
                                                                                     ensure_ascii=False))
    eligible = []
    for identity, by_person in tally.items():
        if len(by_person) < k_crowd:
            continue
        ways = [w for written in by_person.values() for w in written]
        # The way most of its people write it; between equally common ways, the first alphabetically.
        best = sorted(set(ways), key=lambda w: (-ways.count(w), _sort_key(identity[0], json.loads(w))))[0]
        entry = json.loads(best)
        if all(_short_plain(t, _words_allowed(identity[0])) for t in _list_words(identity[0], entry)):
            eligible.append((len(by_person), identity, entry))
    out = {s: [] for s in _LIST_SECTIONS}
    for cap_name, sections in (("details", ("details", "scales", "values")), ("phrases", ("phrases",)),
                               ("places", ("places",))):
        group = sorted((e for e in eligible if e[1][0] in sections), key=lambda e: (-e[0], e[1]))
        for _, identity, entry in sorted(group[:LIST_CAPS[cap_name]], key=lambda e: e[1]):
            out[identity[0]].append(entry)
    return out


def _sort_key(section: str, entry: dict):
    """How two ways of writing one item are put in alphabetical order (as the shared list orders forms)."""
    if section == "details":
        return (entry["about"], entry["name"], entry["compare"])
    if section == "scales":
        return tuple(entry["steps"])
    if section == "places":
        return (entry["city"], entry["country"])
    return entry["value"]


def _list_problems(d: dict, board, k_crowd: int) -> list:
    """What's wrong with one delivered shared list (empty: nothing)."""
    if not isinstance(d, dict) or set(d) != {"built_at", "board_change", *_LIST_SECTIONS}:
        return ["it isn't the shared list's shape"]
    out = []
    for section in _LIST_SECTIONS:
        entries = d[section]
        if not isinstance(entries, list) or not all(isinstance(e, dict) and set(e) == _LIST_SHAPES[section]
                                                    for e in entries):
            return [f"its {section} aren't the shared list's shape"]
        out += [f"{section}: {w!r} isn't short plain text (a few words, no sentence punctuation) without contact "
                f"details or sensitive words"
                for e in entries for w in _list_words(section, e) if not _short_plain(w, _words_allowed(section))]
    if sum(len(d[s]) for s in ("details", "scales", "values")) > LIST_CAPS["details"] \
            or len(d["phrases"]) > LIST_CAPS["phrases"] or len(d["places"]) > LIST_CAPS["places"]:
        out.append("it's longer than its caps")
    if _when(d["built_at"]) is None or not isinstance(d["board_change"], int):
        return out + ["it doesn't say when, or from which board, it was built"]
    expected = _recount(board.as_of(d["board_change"]), d["built_at"], k_crowd)
    for section in _LIST_SECTIONS:
        if d[section] == expected[section]:
            continue
        got = [json.dumps(e, sort_keys=True, ensure_ascii=False) for e in d[section]]
        want = [json.dumps(e, sort_keys=True, ensure_ascii=False) for e in expected[section]]
        extra, missing = sorted(set(got) - set(want)), sorted(set(want) - set(got))
        out += [f"{section}: {e} is on it, but not {k_crowd} different people show it that way" for e in extra]
        out += [f"{section}: {e} is missing" for e in missing]
        if not extra and not missing:
            out.append(f"{section}: not in alphabetical order")
    return out


# --------------------------------------------------------------------------------------
# A8: what was sent to the Matchmaker AI, read against the board as it was (its own code: nothing from the Matchmaker,
# its judge or the Warden).
# --------------------------------------------------------------------------------------

_REQUEST_KEYS = {"focus", "candidates", "listings"}
# The built-ins a request may carry, by the names they go by there, and the field each must be (its own copy): only
# what a listing shows. Where and how people meet, dating's ages and genders, and a first name are never in one.
_SENT_BUILT_INS = {"headline": "headline", "looking_for": "seeking", "offering": "offering", "topics": "topics",
                   "pay": "pay"}
_ENTRY_KEYS = {"kind", "exactness", "description", "details", *_SENT_BUILT_INS}
_SENT_DETAIL_KEYS = {"about", "name", "values", "compare", "steps", "required"}
_TAKEN_OUT = "\u2026"  # what stands where a text was taken out (its own copy)
_RUN = re.compile(r"\w+")
# Quote marks and apostrophes outside the "initial" and "final" quote categories (Pi, Pf), read as one (its own copy).
_MARKS = frozenset("'\"`´ʹʺʻʼʽʾʿˈˊˋ˝′″‴‵‶‷‚„＂＇")


def _blank(c: str) -> bool:
    """Shows nothing: a format character (category Cf) or a variation selector."""
    return unicodedata.category(c) == "Cf" or 0xFE00 <= ord(c) <= 0xFE0F or 0xE0100 <= ord(c) <= 0xE01EF


def _two_readings(t: str) -> list:
    """A canonical text as it is, and read loosely, as a text sent may write a watched text's marks differently: a
    character that shows nothing, a quote mark or an apostrophe left out in one reading and read as a space in the
    other; a dash a space in both; one space between words and none at the ends."""
    readings = [t]
    for gap in ("", " "):
        out = []
        for c in t:
            cat = unicodedata.category(c)
            if _blank(c) or cat in ("Pi", "Pf") or c in _MARKS:
                c = gap
            elif cat == "Pd":
                c = " "
            out.append(c)
        reading = " ".join("".join(out).split())
        if reading not in readings:
            readings.append(reading)
    return readings


def _watched_forms(text: str) -> set:
    """What audit A8 looks for of a watched text: its canonical form, and each loose reading of it without the
    punctuation and spaces at its ends ("ltd." is found in "ltd," too)."""
    out = {canonical(text)} - {""}
    for reading in _two_readings(canonical(text)):
        core = reading
        while core and (core[0] == " " or unicodedata.category(core[0]).startswith("P")):
            core = core[1:]
        while core and (core[-1] == " " or unicodedata.category(core[-1]).startswith("P")):
            core = core[:-1]
        if core:
            out.add(core)
    return out


def _watch_pattern(s: str):
    """Where a watched text stands in a text sent (both canonical): bounded by a non-word character at an end that is a
    Latin letter or digit; at an end in any other script, anywhere (a name in a script written without spaces stands
    inside a run of letters)."""
    left = r"(?<!\w)" if s[:1].isascii() and s[:1].isalnum() else ""
    right = r"(?!\w)" if s[-1:].isascii() and s[-1:].isalnum() else ""
    return re.compile(left + re.escape(s) + right)


class _JudgeWatch:
    """The board's own record of changes (Marketboard.events), read for audit A8: which version of each listing was
    live at a change, from when until when each text was hidden by someone (a withdrawn listing's last ones for ever
    after, as for everything sent), and from when each text was someone's first name and each id was held (for ever
    after: a name someone has since changed is still theirs). Each is looked for in every form _watched_forms gives."""

    def __init__(self, board):
        happened = {}
        for at, lid, version in board.events():
            happened.setdefault(lid, []).append((at, version))
        self._live = {}  # listing id -> [(from, until or None, version)]
        self._when = {}  # (what, canonical text) -> [(from, until or None)]
        for lid, steps in happened.items():
            ids = {lid, board.person(lid).id} | {v.owner for _, v in steps if v is not None}
            for i in ids:
                self._when.setdefault(("id", canonical(i)), []).append((steps[0][0], None))
                for form in _watched_forms(i):
                    self._when.setdefault(("id", form), []).append((steps[0][0], None))
            for n, (at, version) in enumerate(steps):
                if version is None:
                    continue
                nxt = steps[n + 1] if n + 1 < len(steps) else None
                self._live.setdefault(lid, []).append((at, nxt[0] if nxt else None, version))
                until = nxt[0] if nxt is not None and nxt[1] is not None else None
                for f in version.fields:
                    if f.key == "first_name":
                        for t in f.strings():
                            for form in _watched_forms(t):
                                self._when.setdefault(("name", form), []).append((at, None))
                    if f.visibility == HIDDEN:
                        for t in _said(f):
                            for form in _watched_forms(t):
                                self._when.setdefault(("hidden", form), []).append((at, until))
        self._index, self._loose = {}, []
        for key in self._when:
            text = key[1]
            if not text:
                continue
            first = _RUN.match(text)
            entry = (key, _watch_pattern(text))
            # Looked up by its first word when that word, as written, is a whole word wherever the text stands.
            if first and text[0].isascii() and text[0].isalnum() and (
                    first.end() < len(text) or (text[-1].isascii() and text[-1].isalnum())):
                self._index.setdefault(first.group(), []).append(entry)
            else:
                self._loose.append(entry)

    def counted(self, key, change: int) -> bool:
        return any(a <= change and (b is None or change < b) for a, b in self._when.get(key, ()))

    def version(self, lid, change: int):
        """The version of a listing live at that change, or None."""
        for a, b, version in self._live.get(lid, ()) if isinstance(lid, str) else ():
            if a <= change and (b is None or change < b):
                return version
        return None

    def found(self, text: str, change: int) -> list:
        """(what, watched text, start, end) for every occurrence of a text that counted at that change, in this text
        (a loose reading of a text sent, _two_readings)."""
        out = []
        for key, pattern in [e for w in set(_RUN.findall(text)) for e in self._index.get(w, ())] + self._loose:
            if key[1] in text and self.counted(key, change):
                out += [(key[0], key[1], m.start(), m.end()) for m in _occurrences(pattern, text)]
        return out


def _cut_of(sent, source: str) -> bool:
    """Is a text sent its source (canonical) with stretches taken out (each marked _TAKEN_OUT) and nothing added: the
    rest, in order, all from the source?"""
    if not isinstance(sent, str):
        return False
    pieces = [p.strip() for p in sent.split(_TAKEN_OUT)]
    if len(pieces) == 1:
        return pieces[0] == source
    if pieces[0] and not source.startswith(pieces[0]):
        return False
    at = 0
    for p in pieces:
        if p:
            k = source.find(p, at)
            if k < 0:
                return False
            at = k + len(p)
    return not pieces[-1] or source.endswith(pieces[-1])


def _text_list(value):
    """A sent value as its texts, or None if it isn't text or a list of texts."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return list(value)
    return None


def _judge_request_problems(watch, sent) -> tuple:
    """(What's wrong with one kept request, how many entries it had)."""
    if not isinstance(sent, dict) or set(sent) != {"request", "labels", "board_change"} \
            or type(sent["board_change"]) is not int or not isinstance(sent["labels"], dict):
        return ["it isn't a kept request's shape"], 0
    request, labels, change = sent["request"], sent["labels"], sent["board_change"]
    if not isinstance(request, dict) or set(request) != _REQUEST_KEYS or not isinstance(request["listings"], dict) \
            or not isinstance(request["candidates"], list):
        return ["it isn't a request's shape"], 0
    listings, focus, candidates = request["listings"], request["focus"], request["candidates"]
    order = [focus, *candidates]
    if not all(isinstance(x, str) for x in order) or len(set(order)) != len(order) or set(order) != set(listings) \
            or set(labels) != set(listings):
        return ["its labels don't fit together"], 0
    out = []
    for label, entry in listings.items():
        if watch.counted(("id", canonical(label)), change):
            out.append(f"label {label!r} is an id on the board")
            continue
        version = watch.version(labels[label], change)
        if version is None:
            out.append(f"{label} isn't a listing that was on the board then")
            continue
        out += [f"{label}: {p}" for p in _entry_problems(watch, entry, version, change)]
    return out, len(listings)


def _entry_problems(watch, entry, version, change: int) -> list:
    """What's wrong with one listing's entry in a request, read against the version of it live then."""
    if not isinstance(entry, dict):
        return ["it isn't an entry"]
    out = [f"it carries {k!r}, which a request never does" for k in entry if k not in _ENTRY_KEYS]
    if entry.get("kind", "") not in (*MATCH_KINDS, ""):
        out.append("its kind isn't one of the fixed kinds")
    x = entry.get("exactness", 0.5)
    try:
        own = min(1.0, max(0.0, float(version.exactness)))
    except (TypeError, ValueError):
        own = 0.5
    if not _number(x) or x != round(x, 2) or abs(x - own) > 0.005 + 1e-9:
        out.append("its exactness isn't its owner's")
    texts = []
    for name, key in _SENT_BUILT_INS.items():
        if name not in entry:
            continue
        f = version.get(key)
        sent = _text_list(entry[name])
        if f is None or f.visibility != SHOWN:
            out.append(f"{name} isn't shown by its owner")
        elif sent is None:
            out.append(f"{name} isn't text")
        elif not all(any(_cut_of(t, canonical(s)) for s in f.strings()) for t in sent):
            out.append(f"{name} isn't what its owner shows")
        texts += sent or []
    details = entry.get("details", [])
    for d in details if isinstance(details, list) else [None]:
        values = _text_list(d.get("values")) if isinstance(d, dict) else None
        steps = _text_list(d.get("steps", [])) if isinstance(d, dict) else None
        if not isinstance(d, dict) or not set(d) <= _SENT_DETAIL_KEYS or not isinstance(d.get("name"), str) \
                or values is None or steps is None or d.get("about") not in ("me", "them", "both"):
            out.append("a detail isn't a detail's shape")
            continue
        texts += [d["name"], *values, *steps]

        def same(f) -> bool:
            return (f.visibility == SHOWN and _about(f.key) == d["about"]
                    and _cut_of(d["name"], canonical(f.key.partition(":")[2]))
                    and d.get("compare", "same") == ("steps" if f.compare == "steps" else "same")
                    and ("required" not in d or (type(d["required"]) is bool and d["required"] == bool(f.required)))
                    and all(any(_cut_of(v, canonical(s)) for s in f.strings()) for v in values)
                    and all(any(_cut_of(t, canonical(s)) for s in f.steps) for t in steps))
        if not any(same(f) for f in version.fields):
            out.append(f"detail {d['name']!r} isn't one its owner shows")
    description = entry.get("description", "")
    if not isinstance(description, str) or not _cut_of(description, canonical(version.intent_text)):
        out.append("its description isn't its owner's")
    if isinstance(description, str):
        texts.append(description)
    # What it says: never a first name or an id of anyone; a hidden text of anyone only inside a text the listing
    # shows itself (its own shown words, which the other person's hiding the same words doesn't make private). Each
    # text is read loosely (_two_readings), so a watched text with its marks written differently is still found.
    shown = {form for f in version.fields if f.visibility == SHOWN and f.key != "first_name"
             for t in _said(f) for form in _watched_forms(t)}
    said = set()
    for text in texts:
        for reading in _two_readings(canonical(text)):
            inside = None
            for what, s, a, b in watch.found(reading, change):
                if what == "hidden":
                    if inside is None:
                        inside = {i for t in shown if t in reading for m in _occurrences(_watch_pattern(t), reading)
                                  for i in range(m.start(), m.end())}
                    if all(i in inside for i in range(a, b)):
                        continue
                said.add((what, s))
    words = {"name": "someone's first name", "id": "a listing or person id", "hidden": "a text someone hides"}
    out += [f"it says {s!r}, {words[what]}" for what, s in sorted(said)]
    return out
