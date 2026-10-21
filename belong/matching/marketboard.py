"""Marketboard (SPEC §3): where accepted listings are kept for matching.

Nothing on it acts. Only the Warden's entry check calls put(); the Matchmaker reads.
"""
from __future__ import annotations

from collections import Counter

from ..schema import Listing, Person


class Marketboard:
    def __init__(self):
        self._persons: dict[str, Person] = {}
        self._listings: dict[str, Listing] = {}
        # Deleted listings: off the board and never matched, kept aside only so the independent
        # audit can still check what was already sent (see TRUST.md, honest limits).
        self._withdrawn: dict[str, Listing] = {}

    def put(self, person: Person, listing: Listing) -> None:
        """Store a listing under its own id (a person may own several)."""
        self._persons[listing.person_id] = person
        self._listings[listing.person_id] = listing

    def listing(self, listing_id: str) -> Listing:
        return self._listings[listing_id] if listing_id in self._listings else self._withdrawn[listing_id]

    def withdraw(self, listing_id: str) -> None:
        """Take a listing off the board (only the Warden calls this, when its owner deletes it)."""
        self._withdrawn[listing_id] = self._listings.pop(listing_id)

    def person(self, listing_id: str) -> Person:
        """The verified person who owns the listing."""
        return self._persons[listing_id]

    def listings(self, include_withdrawn: bool = False) -> list[Listing]:
        everything = {**self._withdrawn, **self._listings} if include_withdrawn else self._listings
        return [everything[k] for k in sorted(everything)]

    def __contains__(self, listing_id: str) -> bool:
        return listing_id in self._listings

    def value_counts(self) -> Counter:
        """How many people have each (key, value), shown or hidden. Used for the
        crowd rule: a hidden value counts only if at least k people share it."""
        counts: Counter = Counter()
        for listing in self._listings.values():
            for f in listing.fields:
                for s in set(f.strings()):
                    counts[(f.key, s)] += 1
        return counts
