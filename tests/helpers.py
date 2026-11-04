"""Small builders shared by the tests."""
from __future__ import annotations

from belong.matching.marketboard import Marketboard
from belong.schema import FIELD_REGISTRY, HIDDEN, SHOWN, Field, Listing, Person, detail_key
from belong.warden import Clock, Warden


def person(pid="p1", verified=True, age_band="25-29"):
    return Person(id=pid, verified=verified, age_band=age_band, created_at="2026-10-01T00:00:00Z")


def fld(key, value, visibility=None):
    """A built-in field, with its registry defaults."""
    spec = FIELD_REGISTRY[key]
    return Field(key=key, value=value, visibility=visibility or spec.default, sensitive=spec.sensitive)


def detail(name, values, about="me", visibility=SHOWN, compare="same", steps=(), required=True):
    """An open detail: a name (canonical), about me/them/both, and its values."""
    values = (values,) if isinstance(values, str) else tuple(values)
    return Field(detail_key(about, name), values, visibility, False, compare=compare, steps=tuple(steps),
                 required=required)


def listing(pid="p1", intent="Looking for a roommate in Osaka, moving in November.", fields=(), exactness=0.5,
            confirmed=()):
    return Listing(person_id=pid, intent_text=intent, exactness=exactness, fields=tuple(fields),
                   updated_at="2026-10-12T09:00:00Z", confirmed_shown=tuple(confirmed))


def warden(**kw):
    return Warden(Clock(), key=b"test-key", **kw)


# What the two people in roommates_board have in common, as the Warden works it out from their shown
# details (schema.detail_reasons): a proposal must cite exactly these.
ROOMMATE_REFS = ("agree:free time", "agree:interests", "agree:move in month")


def roommates_board(w=None, extra_a=(), extra_b=()):
    """Two people who clearly match (roommates in Osaka), each with hidden details."""
    w = w or warden()
    board = Marketboard()
    a = listing("pa", "Looking for a roommate in Osaka, moving in November.", [
        fld("first_name", "Yuki"), fld("city", "Osaka"), fld("age_band", "25-29"),
        detail("interests", ("jazz", "ramen")), detail("free time", "weekends"),
        detail("move in month", "November", about="both"),
        fld("religion", "Buddhist", HIDDEN), fld("employer", "Nintendo", HIDDEN), *extra_a,
    ], confirmed=())
    b = listing("pb", "Need a flatmate in Osaka from November.", [
        fld("first_name", "Ren"), fld("city", "Osaka"), fld("age_band", "25-29"),
        detail("interests", ("jazz", "chess")), detail("free time", "weekends"),
        detail("move in month", "November", about="both"),
        fld("religion", "Buddhist", HIDDEN), fld("income_band", "¥4–7M", HIDDEN), *extra_b,
    ])
    assert w.admit(person("pa"), a, board).accepted
    assert w.admit(person("pb"), b, board).accepted
    return w, board
