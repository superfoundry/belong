"""Marketboard (SPEC §3): where accepted listings are kept for matching.

Nothing on it acts. Only the Warden's entry check calls put(); the Matchmaker reads.
"""
from __future__ import annotations

from collections import Counter

from ..schema import Listing, Person, canonical


class Counts(Counter):
    """Counts nobody can change: the board shares one copy with every reader until the board changes."""

    def __init__(self, counts=()):
        dict.update(self, counts)

    def _read_only(self, *args, **kwargs):
        raise TypeError("the board's counts are read-only")

    __setitem__ = __delitem__ = update = subtract = clear = pop = popitem = setdefault = _read_only
    __iadd__ = __isub__ = __ior__ = __iand__ = _read_only


class Marketboard:
    def __init__(self):
        self._persons: dict[str, Person] = {}
        self._listings: dict[str, Listing] = {}
        # Deleted listings: off the board and never matched, kept aside only so the independent
        # audit can still check what was already sent (see TRUST.md, honest limits).
        self._withdrawn: dict[str, Listing] = {}
        # Every version each listing has had, so the audit can check an output against the listing
        # as it was when the output was sent (a question quotes the version live at the time).
        self._versions: dict[str, list] = {}
        # What happened to each listing, by change number: (change, the new version, or None when it was
        # withdrawn). The audit rebuilds the board as it was at any change (as_of).
        self._events: dict[str, list] = {}
        self.changes = 0  # goes up whenever a listing is added, updated or withdrawn
        self._counts = None  # (the change they were counted at, value_counts())

    def put(self, person: Person, listing: Listing) -> None:
        """Store a listing under its own id (a person may own several)."""
        self.changes += 1
        self._persons[listing.person_id] = person
        self._listings[listing.person_id] = listing
        self._versions.setdefault(listing.person_id, []).append(listing)
        self._events.setdefault(listing.person_id, []).append((self.changes, listing))

    def listing(self, listing_id: str) -> Listing:
        return self._listings[listing_id] if listing_id in self._listings else self._withdrawn[listing_id]

    def history(self) -> list[Listing]:
        """Every version of every listing ever accepted, deleted ones included (for the audit)."""
        return [v for k in sorted(self._versions) for v in self._versions[k]]

    def as_of(self, change: int) -> list[Listing]:
        """The listings that were live on the board right after change number `change` (for the audit)."""
        out = []
        for k in sorted(self._events):
            then = [listing for at, listing in self._events[k] if at <= change]
            if then and then[-1] is not None:
                out.append(then[-1])
        return out

    def events(self) -> list:
        """Every change to the board in order, as (change, listing id, the new version or None when it was
        withdrawn), for the audit."""
        return sorted((at, k, listing) for k, happened in self._events.items() for at, listing in happened)

    def ever_held(self, listing_id: str) -> bool:
        """Was a listing ever accepted under this id (live now or withdrawn)?"""
        return listing_id in self._events

    def withdrawn(self, listing_id: str) -> bool:
        """Was the listing under this id ever withdrawn?"""
        return listing_id in self._withdrawn

    def withdraw(self, listing_id: str) -> None:
        """Take a listing off the board (only the Warden calls this, when its owner deletes it)."""
        self.changes += 1
        self._withdrawn[listing_id] = self._listings.pop(listing_id)
        self._events.setdefault(listing_id, []).append((self.changes, None))

    def person(self, listing_id: str) -> Person:
        """The verified person who owns the listing."""
        return self._persons[listing_id]

    def listings(self, include_withdrawn: bool = False) -> list[Listing]:
        everything = {**self._withdrawn, **self._listings} if include_withdrawn else self._listings
        return [everything[k] for k in sorted(everything)]

    def __contains__(self, listing_id: str) -> bool:
        return listing_id in self._listings

    def value_counts(self) -> Counts:
        """How many different people have each (key, canonical value), shown or hidden, on active
        listings. Used for the crowd rule: a hidden value counts only if enough other people share it.
        People, not listings, so one person with several listings is counted once.

        Counted once per board change and shared, read-only, until the next one: listings are frozen and
        every change goes through put() or withdraw(), which move `changes` on."""
        if self._counts is not None and self._counts[0] == self.changes:
            return self._counts[1]
        owners: dict = {}
        for listing in self._listings.values():
            if not listing.active:
                continue
            for f in listing.fields:
                for s in f.strings():
                    owners.setdefault((f.key, canonical(s)), set()).add(listing.owner)
        counts = Counts({kv: len(people) for kv, people in owners.items()})
        self._counts = (self.changes, counts)
        return counts
