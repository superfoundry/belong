"""Small builders shared by the tests."""
from __future__ import annotations

from belong.matching.marketboard import Marketboard
from belong.schema import FIELD_REGISTRY, HIDDEN, SHOWN, Field, Listing, Person
from belong.warden import Clock, Warden


def person(pid="p1", verified=True, age_band="25-29"):
    return Person(id=pid, verified=verified, age_band=age_band, created_at="2026-10-01T00:00:00Z")


def fld(key, value, visibility=None):
    spec = FIELD_REGISTRY[key]
    return Field(key=key, value=value, visibility=visibility or spec.default, sensitive=spec.sensitive)


def listing(pid="p1", intent="Looking for a roommate in Osaka, moving in November.", fields=(), exactness=0.5,
            confirmed=()):
    return Listing(person_id=pid, intent_text=intent, exactness=exactness, fields=tuple(fields),
                   updated_at="2026-10-12T09:00:00Z", confirmed_shown=tuple(confirmed))


def warden(**kw):
    return Warden(Clock(), key=b"test-key", **kw)


def roommates_board(w=None, extra_a=(), extra_b=()):
    """Two people who clearly match (roommates in Osaka), each with hidden details."""
    w = w or warden()
    board = Marketboard()
    a = listing("pa", "Looking for a roommate in Osaka, moving in November.", [
        fld("first_name", "Yuki"), fld("city", "Osaka"), fld("age_band", "25-29"),
        fld("interests", ("jazz", "ramen")), fld("schedule", "weekends"), fld("move_in", "November"),
        fld("religion", "Buddhist", HIDDEN), fld("employer", "Nintendo", HIDDEN), *extra_a,
    ], confirmed=())
    b = listing("pb", "Need a flatmate in Osaka from November.", [
        fld("first_name", "Ren"), fld("city", "Osaka"), fld("age_band", "25-29"),
        fld("interests", ("jazz", "chess")), fld("schedule", "flexible"), fld("move_in", "November"),
        fld("religion", "Buddhist", HIDDEN), fld("income_band", "¥4–7M", HIDDEN), *extra_b,
    ])
    assert w.admit(person("pa"), a, board).accepted
    assert w.admit(person("pb"), b, board).accepted
    return w, board
