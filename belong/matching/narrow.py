"""Narrowing (SPEC §8.2): a shortlist per listing, found top down in the agreed order.

  1. How they meet. Two listings can meet online only if both allow it, and then place
     doesn't matter. Otherwise both must allow meeting in person, and step 2 applies. The
     shown "format" says which ("In person", "Online", "In person or online", in any
     wording). Without one, a listing whose text says online or remote is fine allows both,
     and any other only in person. Dating is always in person.
  2. Where, by travel distance (in person only). Two cities are in reach when the distance
     between them is within *both* people's shown "travel" (same city = 0 km; no travel
     given = the same city only). A city whose centre isn't known is only in reach of
     itself. In the same country, someone who gives no city is open to anywhere in it.
  3. What, by meaning. Listings of a structured kind (dating, language, ...) are compared
     with their own kind. Listings compared by their "looking for" / "you are" phrases are
     found through an inverted index of the words in their phrases, topics and what they show
     about themselves, inside the place groups of steps 1-2: a guitarist is never compared
     with a plumber, nor anyone with people they could never meet.

The groups always contain every pair the decisions that follow (matcher.intent_met, and
matcher.nearly for close matches and questions) could accept. Then the hard filters apply as
before, in a stable order, ranked by text similarity.

Uses only intent text and *shown* fields. Hidden fields never affect who gets
shortlisted, so they can't become the reason two people meet.
"""
from __future__ import annotations

from collections import defaultdict

from ..schema import (CITY_COORDS, MAX_TRAVEL_KM, TRAVEL, TRAVEL_DEFAULT, Listing, km_between, meeting_format,
                      plain_place, travel_choice, value_strings)
from .intent import ParsedIntent
from .text import TfIdf, cosine

ONLINE, IN_PERSON = "online", "in person"
_MODES = {"In person": frozenset({IN_PERSON}), "Online": frozenset({ONLINE}),
          "In person or online": frozenset({IN_PERSON, ONLINE})}
# Kinds with a rule of their own (matcher.intent_met), compared with their own kind. Everything
# else ("other") is compared by its phrases.
STRUCTURED_KINDS = frozenset({"dating", "language", "cofounder", "tutoring", "climbing", "running", "roommate"})


class ShownView:
    """Read-only view of a listing that exposes shown fields only.

    Matching logic that decides *whether* a pair qualifies receives this, never the
    full listing, so it cannot read a hidden value even by mistake.
    """

    def __init__(self, listing: Listing):
        self._shown = listing.shown()
        self.person_id = listing.person_id
        self.intent_text = listing.intent_text
        self.exactness = listing.exactness

    def get(self, key: str):
        return self._shown.get(key)

    def profile_text(self) -> str:
        parts = [self.intent_text]
        for value in self._shown.values():
            parts.extend(value_strings(value))
        return " ".join(parts)


def _cached(cache: dict, value, read):
    """read(value), remembered: the same few places, formats and distances come up again and again."""
    out = cache.get(value, cache)
    if out is cache:
        if len(cache) > 100_000:
            cache.clear()
        out = cache[value] = read(value)
    return out


_PLACES: dict = {}
_FORMATS: dict = {}
_TRAVEL: dict = {}
_KM: dict = {}
_NEAR: dict = {}


def _place(value):
    """Compare places ignoring case, spacing and accents ("Málaga" == "malaga")."""
    return _cached(_PLACES, str(value), plain_place) if value else None


def meeting_modes(parsed: ParsedIntent, view: ShownView) -> frozenset:
    """How this listing can meet: in person, online, or both (step 1). The shown format decides;
    without a readable one, the intent text does. Dating is always in person."""
    if parsed.kind == "dating":
        return _MODES["In person"]
    shown = view.get("format")
    fmt = _cached(_FORMATS, shown, meeting_format) if isinstance(shown, str) else None
    if fmt is None:
        return _MODES["In person or online"] if parsed.remote_ok else _MODES["In person"]
    return _MODES[fmt]


def travel_km(view: ShownView) -> float:
    """How far this person will go to meet in person, from their shown "travel" (0 = same city)."""
    shown = view.get("travel")
    choice = _cached(_TRAVEL, shown, travel_choice) if isinstance(shown, str) else None
    return TRAVEL[choice or TRAVEL_DEFAULT]


def _km(a: str, b: str):
    return _cached(_KM, (a, b) if a <= b else (b, a), lambda pair: km_between(*pair))


def _near(city: str) -> list:
    """Known cities within the longest travel distance of this one, with their distance: worked out
    once, so a listing only ever looks at the cities it could reach."""
    if not _NEAR:
        for a in CITY_COORDS:
            _NEAR[a] = sorted((km, b) for b in CITY_COORDS if b != a
                              for km in [km_between(a, b)] if km <= MAX_TRAVEL_KM)
    return _NEAR.get(city, [])


def within_reach(av: ShownView, bv: ShownView):
    """Where two people could meet in person (step 2): "city" if their cities are in reach (the same
    city, or within both people's travel distance), "country" if they share a country and at least
    one is open to anywhere in it (no city given), else None. Shown fields only."""
    a_city, b_city = _place(av.get("city")), _place(bv.get("city"))
    a_country, b_country = _place(av.get("country")), _place(bv.get("country"))
    if a_country and b_country and a_country != b_country:
        return None
    if a_city and b_city:
        if a_city == b_city:
            return "city"
        km = _km(a_city, b_city)
        return "city" if km is not None and km <= min(travel_km(av), travel_km(bv)) else None
    if a_country and b_country:
        return "country"
    return None


def how_they_meet(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView):
    """ONLINE if both allow meeting online (then place doesn't matter); otherwise, if both allow
    meeting in person, where they could (within_reach); else None."""
    am, bm = meeting_modes(a, av), meeting_modes(b, bv)
    if ONLINE in am and ONLINE in bm:
        return ONLINE
    if IN_PERSON in am and IN_PERSON in bm:
        return within_reach(av, bv)
    return None


def location_ok(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView) -> bool:
    return how_they_meet(a, av, b, bv) is not None


def _cross_kind_ok(a: ParsedIntent, av: ShownView, b: ParsedIntent, bv: ShownView) -> bool:
    """Listings filed under different kinds can still be compared by their phrases."""
    return _by_phrase(a, av) and _by_phrase(b, bv)


def _by_phrase(p: ParsedIntent, view: ShownView) -> bool:
    """Compared by its "looking for" / "you are" phrases: it has both, and it isn't dating."""
    return view.get("seeking") is not None and view.get("offering") is not None and p.kind != "dating"


_POSTED: dict = {}


def _places_posted(p: ParsedIntent, view: ShownView) -> list:
    """The place groups a listing belongs to (steps 1-2)."""
    key = (p.kind == "dating", p.remote_ok, view.get("format"), view.get("city"), view.get("country"))
    try:
        return _cached(_POSTED, key, lambda _: _places_of(p, view))
    except TypeError:  # a value that can't be a key (not a string): work it out each time
        return _places_of(p, view)


def _places_of(p: ParsedIntent, view: ShownView) -> list:
    modes = meeting_modes(p, view)
    city, country = _place(view.get("city")), _place(view.get("country"))
    out = []
    if IN_PERSON in modes:
        if city:
            out.append(("city", city))
        if country:
            out.append(("country", country))
            if not city:
                out.append(("anywhere in", country))
    if ONLINE in modes:
        out.append((ONLINE,))
    return out


def _places_looked_at(p: ParsedIntent, view: ShownView) -> list:
    """The place groups a listing looks in: its own city and the cities within its own travel
    distance (the other person's distance is checked by the filter), anyone in its country who
    is open to anywhere in it (or, with no city, everyone in its country), and online if it
    meets online. Together they hold everyone it could meet."""
    modes = meeting_modes(p, view)
    city, country = _place(view.get("city")), _place(view.get("country"))
    out = []
    if IN_PERSON in modes:
        if city:
            limit = travel_km(view)
            out.append(("city", city))
            out += [("city", other) for km, other in _near(city) if km <= limit]
            if country:
                out.append(("anywhere in", country))
        elif country:
            out.append(("country", country))
    if ONLINE in modes:
        out.append((ONLINE,))
    return out


OPEN = "open to company"  # listings whose "looking for" may be met without a shared word (see shortlists)


def shortlists(parsed: dict, views: dict, size: int, focus=None, words=None) -> tuple[dict, TfIdf]:
    """For each listing (or only those in `focus`): the `size` most similar candidates that pass
    the hard filters (same kind of match, or both compared by phrase; a way to meet).

    Top down: listings are put in place groups once (how and where they can meet, steps 1-2), and
    each listing only looks in the groups it could meet in. In those groups, structured kinds are
    found by kind, and phrase listings through a word index (step 3), built for a group only when
    a phrase listing first looks in it, so a run for a few listings never indexes the whole board.

    words(view) -> (want, have, open): a listing's index keys, read the way the Matchmaker's own
    decisions read it (matcher.index_words): `want` from what it looks for, `have` from what it is
    and shows, and whether its "looking for" could be met without any shared word. Want keys are
    looked up among others' have keys and the other way round, so two listings meet if either
    one's want shares a key with the other's have. Without `words`, every phrase listing in reach
    is a candidate for every other (one catch-all group, as before the index): slower, and the
    same pairs qualify.

    The groups always contain every listing that could pass the filters and the decisions after
    them; the filters are then applied exactly as written, in board order, so only the ranking
    picks among them."""
    index = TfIdf({pid: v.profile_text() for pid, v in views.items()})
    order = {pid: i for i, pid in enumerate(parsed)}

    by_kind, by_phrase = defaultdict(list), defaultdict(list)  # (place, kind) / place -> listings
    for b, bp in parsed.items():
        if bp.kind is None:
            continue  # a listing nobody can read is never anyone's candidate
        bv = views[b]
        structured, phrase = bp.kind in STRUCTURED_KINDS, _by_phrase(bp, bv)
        if structured or phrase:
            for place in _places_posted(bp, bv):
                if structured:
                    by_kind[(place, bp.kind)].append(b)
                if phrase:
                    by_phrase[place].append(b)

    word_keys: dict = {}

    def keys_of(pid):
        out = word_keys.get(pid)
        if out is None:
            out = word_keys[pid] = words(views[pid])
        return out

    word_groups: dict = {}

    def word_index(place):
        """The phrase listings in one place group, by their want and have keys."""
        idx = word_groups.get(place)
        if idx is None:
            idx = word_groups[place] = defaultdict(list)
            for b in by_phrase.get(place, ()):
                want, have, is_open = keys_of(b)
                for k in want:
                    idx[("want", k)].append(b)
                for k in have:
                    idx[("have", k)].append(b)
                if is_open:
                    idx[OPEN].append(b)
        return idx

    out = {}
    for a, ap in parsed.items():
        if focus is not None and a not in focus:
            continue
        if ap.kind is None:
            out[a] = []
            continue
        av = views[a]
        places = _places_looked_at(ap, av)
        pool = set()
        if ap.kind in STRUCTURED_KINDS:
            for place in places:
                pool.update(by_kind.get((place, ap.kind), ()))
        if _by_phrase(ap, av):
            if words is None:
                for place in places:
                    pool.update(by_phrase.get(place, ()))
            else:
                want, have, is_open = keys_of(a)
                keys = [("have", k) for k in want] + [("want", k) for k in have] + [OPEN]
                for place in places:
                    if is_open:
                        pool.update(by_phrase.get(place, ()))
                        continue
                    idx = word_index(place)
                    for key in keys:
                        pool.update(idx.get(key, ()))
        pool.discard(a)
        cands = [b for b in sorted(pool, key=order.__getitem__)
                 if (parsed[b].kind == ap.kind or _cross_kind_ok(ap, av, parsed[b], views[b]))
                 and location_ok(ap, av, parsed[b], views[b])]
        intent_vec = index.embed(av.intent_text)
        cands.sort(key=lambda b: -cosine(intent_vec, index.vectors[b]))
        out[a] = cands[:size]
    return out, index
