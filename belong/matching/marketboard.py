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

    def put(self, person: Person, listing: Listing) -> None:
        self._persons[person.id] = person
        self._listings[person.id] = listing

    def listing(self, person_id: str) -> Listing:
        return self._listings[person_id]

    def person(self, person_id: str) -> Person:
        return self._persons[person_id]

    def listings(self) -> list[Listing]:
        return [self._listings[k] for k in sorted(self._listings)]

    def __contains__(self, person_id: str) -> bool:
        return person_id in self._listings

    def value_counts(self) -> Counter:
        """How many people have each (key, value), shown or hidden. Used for the
        crowd rule: a hidden value counts only if at least k people share it."""
        counts: Counter = Counter()
        for listing in self._listings.values():
            for f in listing.fields:
                for s in set(f.strings()):
                    counts[(f.key, s)] += 1
        return counts
