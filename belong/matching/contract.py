"""The contract between the Matchmaker and the Warden. OPEN SOURCE.

Both agents are open source, and the Warden still treats the Matchmaker as hostile. This file is the only
thing they share: the shape of what the Matchmaker may hand to the Warden. Because the
Warden treats every Proposal as untrusted and re-checks it against the Marketboard,
the privacy guarantee never depends on what the Matchmaker does.

A Proposal carries no free text: only person ids, booleans, numbers and field keys.
So even a hostile Matchmaker has nowhere to hide a hidden value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class Proposal:
    """Internal to matching. Never sent to anyone as-is: the Warden turns it into an
    Introduction or drops it."""
    person_a: str
    person_b: str
    a_intent_met: bool
    b_intent_met: bool
    score: float  # shown_score + hidden bonus
    shown_score: float  # score from shown fields alone
    shown_threshold: float  # shown_score had to reach this
    reason_fields: tuple  # keys shown by BOTH people
    used_hidden_fields: tuple  # hidden keys that raised the score
    # A close match: the listing whose intent is only nearly met (one-way, to that listing only,
    # which must have opted in). "" for an ordinary match, where both intents are met.
    close: str = ""


@dataclass(frozen=True)
class FollowUp:
    person_id: str
    question_id: str  # must be a key of FOLLOWUP_MENU
    # For "also_offer": the listing the question is about. The Matchmaker names it; the Warden
    # writes the question from that listing's shown phrase, and the recipient is never told who.
    about: str = ""
    # Ask this only after the listing named here said no to the mirror question (one person is
    # asked at a time: whoever is looking for something first). "" = ask now.
    after: str = ""


@dataclass
class MatchResult:
    proposals: list = field(default_factory=list)
    followups: list = field(default_factory=list)


class Matcher(Protocol):
    """Anything with this method can be the Matchmaker."""

    def match(self, board) -> MatchResult: ...
