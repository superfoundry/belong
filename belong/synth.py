"""Synthetic people for the prototype (SPEC §8.1).

Names come from each person's region (most people in Japan have Japanese names), and every listing
is in the format a person's agent writes now: a headline, topics, the format (in person, online, or
both; some web designers and CV reviewers work online only), how far they'll travel to meet in person
(most stay in their city), a price where there is one, "looking for" / "offering" phrases on every
listing but dating's, and open details the agent names on the spot (schema.Field: a name, about the
person, the other person or the request itself, compared as "same" or on a scale of steps it writes
out). Like any agent's listing, each also records what the agent understood: the kind of match
(dating, or "other" for everything else) and, for dating, who they'd like to meet and those people's
ages. The sensitive and default-hidden categories are the built-in ones. The demo serves 5,000 of
them; tests and the batch run use 300.

Each Persona carries:
  - the Person record (as the KYC provider would report it),
  - every value the person *has* (shown or not): built-ins by key, open details by detail key,
  - the person's visibility choices,
  - what they typed besides their description: for each wish they have that must be met, their words
    saying so, its value among them, as a person telling their agent would (see say_musts); asked for,
    a share of people type none of that (see OTHER_WORDS_SHARE: their description still says most of
    their musts), and answer yes when their agent asks whether one it doesn't say must be met (see answer),
  - how long ago their listing last changed, and how their agent's way of writing details differs
    from most agents' (a dialect, for a share of people when asked for: see DIALECTS), and, also
    when asked for, a wish their agent made up that they never had (the invented-wish dialect: see
    INVENTED),
  - a Truth: what they really want, in structured form, as one of the stories below (climbing,
    a language exchange, a roommate...). The stories are test data, never kinds of match. Truth
    never enters the room; it is used only to grade the Matchmaker and to simulate answers and
    decisions.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from .schema import (AGE_BAND_BOUNDS, AGE_BANDS, ANY, FIELD_REGISTRY, GENDERS, HIDDEN, MAX_VALUES, SHOWN,
                     STRICT_EXACTNESS, TRAVEL, Field, Listing, Person, age_band_index, detail_key, is_detail_key,
                     mentions_sensitive, split_key)

CITIES = {
    "Osaka": ("JP", ["Namba", "Umeda", "Tennoji", "Nakazakicho"]),
    "Tokyo": ("JP", ["Shibuya", "Koenji", "Nakameguro", "Shimokitazawa"]),
    "Kyoto": ("JP", ["Gion", "Demachiyanagi", "Arashiyama"]),
    "Berlin": ("DE", ["Kreuzberg", "Neukölln", "Prenzlauer Berg", "Wedding"]),
    "London": ("GB", ["Hackney", "Brixton", "Camden", "Peckham"]),
    "Lisbon": ("PT", ["Alfama", "Graça", "Príncipe Real", "Alcântara"]),
}
LOCAL_LANGUAGE = {"JP": "Japanese", "DE": "German", "GB": "English", "PT": "Portuguese"}
COUNTRY_NAMES = {"JP": "Japan", "DE": "Germany", "GB": "United Kingdom", "PT": "Portugal"}
LANGUAGES = ["English", "Japanese", "German", "Portuguese", "Spanish", "French"]
INCOME = {
    "JP": ["under ¥4M", "¥4–7M", "¥7–10M", "over ¥10M"],
    "DE": ["under €30k", "€30–50k", "€50–80k", "over €80k"],
    "PT": ["under €30k", "€30–50k", "€50–80k", "over €80k"],
    "GB": ["under £30k", "£30–50k", "£50–80k", "over £80k"],
}
EMPLOYERS = {
    "JP": ["Panasonic", "Nintendo", "Rakuten", "Kyocera", "Mercari"],
    "DE": ["Siemens", "Zalando", "Deutsche Bahn", "N26"],
    "GB": ["Monzo", "the NHS", "Deliveroo", "Arup"],
    "PT": ["Farfetch", "Sonae", "Feedzai", "TAP Air Portugal"],
}
RELIGION = ["Buddhist", "Shinto", "Christian", "Muslim", "Jewish", "Hindu", "not religious"]
POLITICS = ["politically progressive", "politically moderate", "politically conservative"]
HEALTH = ["type 1 diabetes", "ADHD", "celiac disease", "chronic migraine", "hearing impairment"]
CHILDREN = ["wants children", "does not want children", "open to children", "already has children"]
SMOKING = ["smokes", "doesn't smoke", "smokes socially"]
DIET = ["vegan", "vegetarian", "pescatarian", "halal", "kosher", "eats everything"]
KINKS = ["BDSM", "roleplay", "polyamorous", "voyeurism"]
INTERESTS = [
    "bouldering", "jazz", "ramen", "hiking", "film photography", "board games", "cycling",
    "anime", "techno", "cooking", "tennis", "museums", "yoga", "craft beer", "sci-fi novels",
    "karaoke", "surfing", "chess",
]
SCHEDULES = ["weekends", "weekday evenings", "weekday mornings", "flexible"]
LEVELS = ["beginner", "intermediate", "advanced"]
PACES = ["easy", "moderate", "fast"]
SUBJECTS = ["maths", "physics", "piano", "coding", "chemistry"]
DOMAINS = ["climate", "healthcare", "fintech", "education", "logistics"]
MONTHS = ["November", "December", "January", "February"]

# First names by country, so most people in Japan have Japanese names and so on. LOCAL_SHARE of
# people get a local name; the rest come from everyone else's names and INTERNATIONAL, as in any city.
NAMES = {
    "JP": {"woman": ["Yuki", "Hana", "Aoi", "Mei", "Sakura", "Rin", "Yui", "Misaki", "Haruka", "Nanami", "Ayaka",
                     "Mio", "Saki", "Riko", "Emi", "Kana", "Natsuki", "Akari", "Chihiro", "Hinata", "Koharu",
                     "Mana", "Shiori", "Tomomi", "Yuna", "Asuka", "Miyu", "Erika", "Ayumi", "Momoka"],
           "man": ["Haruto", "Ren", "Sota", "Kenji", "Daiki", "Yuto", "Takumi", "Riku", "Kaito", "Shota", "Hiroshi",
                   "Takeshi", "Kazuki", "Yusuke", "Ryota", "Naoki", "Kenta", "Sho", "Tatsuya", "Yuma", "Hayato",
                   "Koji", "Taro", "Daisuke", "Shun", "Kota", "Masato", "Ryo", "Tomoya", "Akira"],
           "nonbinary": ["Hikaru", "Makoto", "Nao", "Kaoru", "Haru", "Tsubasa", "Rei", "Aki", "Jun", "Mizuki"]},
    "DE": {"woman": ["Lena", "Leonie", "Anna", "Marie", "Sophie", "Hannah", "Laura", "Lea", "Julia", "Lisa", "Mia",
                     "Emilia", "Clara", "Johanna", "Paula", "Katharina", "Charlotte", "Greta", "Frieda", "Lina",
                     "Nele", "Pia", "Sarah", "Jana", "Franziska"],
           "man": ["Lukas", "Jonas", "Felix", "Leon", "Paul", "Maximilian", "Elias", "Finn", "Tim", "Jan", "Niklas",
                   "Moritz", "Julian", "Tobias", "Florian", "Simon", "Fabian", "Philipp", "Sebastian", "Matthias",
                   "David", "Henrik", "Jakob", "Anton", "Benedikt"],
           "nonbinary": ["Alex", "Kim", "Robin", "Charlie", "Luca", "Mika", "Sascha", "Toni", "Kai", "Eike"]},
    "GB": {"woman": ["Olivia", "Amelia", "Isla", "Ava", "Emily", "Sophie", "Grace", "Freya", "Poppy", "Ella",
                     "Chloe", "Lucy", "Evie", "Daisy", "Holly", "Megan", "Charlotte", "Hannah", "Jessica", "Lily",
                     "Imogen", "Harriet", "Eleanor", "Rosie", "Zara"],
           "man": ["Oliver", "Jack", "Harry", "George", "Noah", "Charlie", "Jacob", "Thomas", "William", "James",
                   "Oscar", "Alfie", "Leo", "Henry", "Joshua", "Daniel", "Samuel", "Max", "Archie", "Freddie",
                   "Lewis", "Callum", "Rhys", "Owen", "Ethan"],
           "nonbinary": ["Alex", "Sam", "Jamie", "Charlie", "Robin", "Frankie", "Jordan", "Taylor", "Morgan", "Rowan"]},
    "PT": {"woman": ["Inês", "Beatriz", "Mariana", "Ana", "Leonor", "Maria", "Carolina", "Matilde", "Sofia",
                     "Francisca", "Joana", "Rita", "Catarina", "Margarida", "Marta", "Daniela", "Sara", "Raquel",
                     "Teresa", "Diana", "Bárbara", "Filipa", "Helena", "Luana", "Mafalda"],
           "man": ["Tiago", "João", "Rafael", "Francisco", "Santiago", "Duarte", "Martim", "Rodrigo", "Afonso",
                   "Tomás", "Gonçalo", "Diogo", "Miguel", "Pedro", "Guilherme", "Lourenço", "Bernardo", "Ricardo",
                   "Nuno", "Rui", "André", "Bruno", "Vasco", "Hugo", "Luís"],
           "nonbinary": ["Alex", "Sam", "Noa", "Ariel", "Kim", "Jules", "Dani", "Sasha", "Rafa", "Cris"]},
}
INTERNATIONAL = {"woman": ["Priya", "Amara", "Fatima", "Aisha", "Lucía", "Valentina", "Seo-yeon", "Mei-lin", "Zainab", "Chiara"],
                 "man": ["Omar", "Arjun", "Mateo", "Diego", "Wei", "Min-jun", "Kwame", "Liam", "Karim", "Luca"],
                 "nonbinary": ["Ari", "Noor", "Sky", "Remy", "Quinn", "Eden", "Tomo", "Jules", "Ash", "River"]}
LOCAL_SHARE = 0.85

# Services and things people offer or look for, in the phrases a person's agent would write.
# key -> (the provider, their customer, topics, can be done online)
SERVICES = {
    "plumber": ("a plumber", "a plumbing customer", ("plumbing", "home repair"), False),
    "electrician": ("an electrician", "an electrical work customer", ("electrical work", "home repair"), False),
    "cleaner": ("a house cleaner", "a house cleaning client", ("house cleaning", "housework"), False),
    "mover": ("a mover", "a moving help customer", ("moving help", "removals"), False),
    "bike mechanic": ("a bike mechanic", "a bike repair customer", ("bike repair", "cycling"), False),
    "dog walker": ("a dog walker", "a dog walking client", ("dog walking", "pets"), False),
    "web designer": ("a web designer", "a web design client", ("web design", "websites"), True),
    "cv reviewer": ("a CV reviewer", "a CV review client", ("CV review", "job applications"), True),
}
# key -> (noun used in phrases, topics, typical prices by country)
ITEMS = {
    "boxes": ("cardboard box", ("cardboard boxes", "moving supplies"), {"JP": "¥100 each", "DE": "€1 each", "GB": "£1 each", "PT": "€1 each"}),
    "sofa": ("sofa", ("furniture", "sofas"), {"JP": "¥15,000", "DE": "€120", "GB": "£100", "PT": "€90"}),
    "desk": ("desk", ("furniture", "desks"), {"JP": "¥8,000", "DE": "€60", "GB": "£50", "PT": "€45"}),
    "bike": ("road bike", ("cycling", "bikes"), {"JP": "¥40,000", "DE": "€300", "GB": "£250", "PT": "€220"}),
    "rice cooker": ("rice cooker", ("kitchen appliances", "rice cookers"), {"JP": "¥3,000", "DE": "€25", "GB": "£20", "PT": "€20"}),
    "bookshelf": ("bookshelf", ("furniture", "bookshelves"), {"JP": "¥5,000", "DE": "€40", "GB": "£35", "PT": "€30"}),
    "baby clothes": ("baby clothes bundle", ("baby clothes", "kids"), {"JP": "¥2,000", "DE": "€15", "GB": "£12", "PT": "€12"}),
}
TUTOR_RATES = {"JP": "¥4,000 per hour", "DE": "€35 per hour", "GB": "£30 per hour", "PT": "€25 per hour"}

# The stories people's truths tell, with their weights. Only dating is also a kind of match; every other
# story is listed as "other" (understood_kind), with phrases and open details.
KINDS = [  # (story, weight)
    ("dating", 0.30), ("climbing", 0.08), ("running", 0.07), ("language", 0.09),
    ("cofounder", 0.05), ("tutoring", 0.08), ("roommate", 0.06),
    ("other", 0.09),  # open-ended: bands and musicians, chess partners, photo-walk buddies
    ("service", 0.10),  # a plumber and their customers, a web designer who works online
    ("item", 0.08),  # selling or buying something, giving it away or looking for it free
]
# How far people will travel to meet in person: most stay in their city, some go up to 30 km, a few 100.
TRAVEL_SHARE = [("Same city", 0.75), ("Up to 30 km", 0.17), ("Up to 100 km", 0.08)]
ONLINE_ONLY_SHARE = 0.4  # of providers of a service done online who are open to online at all
GENRES = ["ska", "punk", "jazz", "indie"]
INSTRUMENTS = ["guitarist", "drummer", "bassist", "singer"]
# Each listing last changed this many days before the start (inclusive), so the shared list of details in
# use (which counts only listings unchanged for a week) has entries from the first day.
LISTED_DAYS = (8, 90)

# Agent dialects. Different people's agents won't all write a detail the same way, however clear the
# shared list is: asked for, a share of people (DIALECT_SHARE) get one difference each, so the batch run
# shows that a detail written differently costs a match but never makes a wrong one. The ways: another
# name for a detail, another scale (overlapping the usual one), a detail about the person left out or
# hidden, or someone free whenever written as "flexible" instead of "any".
DIALECT_SHARE = 0.25
DIALECTS = ("other name", "other scale", "left out", "hidden", "no any")
OTHER_NAMES = {"level": "ability", "pace": "speed", "language": "spoken language", "subject": "lesson subject",
               "move in month": "moving month", "free time": "availability", "interests": "hobbies",
               "smoking": "smoker", "diet": "eating"}
OTHER_SCALES = {"level": ("novice", "intermediate", "advanced", "expert"),
                "pace": ("very easy", "easy", "moderate", "fast")}
# The details a dialect changes, the ones matching depends on most first.
_DIALECT_ORDER = ("level", "pace", "language", "subject", "move in month", "free time", "interests", "smoking",
                  "diet")
# The invented-wish dialect. Some agents add a wish the person never had, as one that must be met: asked for, on a share
# of the listings that aren't dating (INVENTED_SHARE), one wish each, made up in one of these ways: about a detail
# nobody shows, about one many people show of themselves, or the person's own free time made into what the other person
# must have. Truth is unchanged, so the batch run measures what such a guess costs, and what the rule that a must comes
# from the person's own words (agent.ground_wishes, which sim.run applies) wins back. The agent that makes a must up
# quotes the first words of their description as theirs for it (their_words, as a model asked for them might): a quote
# that isn't their answer to anything, which the rule never counts.
INVENTED_SHARE = 0.25
INVENTED_QUOTE_WORDS = 5
# Musts said in other words. Each person says every wish they have that must be met, in a message of their own with
# its value (Persona.said), as a person telling their agent would. Asked for, a share of people (OTHER_WORDS_SHARE: in
# the batch run, everyone) type none of those messages. Their description is unchanged and still says most of their
# musts; for each it doesn't say, their agent's save keeps it as ideally, asks them whether it must be met, and quotes
# their yes (answer): the batch run measures what the rule that a must comes from the person's own words would cost on
# those musts if it read the values alone, without their answers (sim.unsaid_musts counts them).
OTHER_WORDS_SHARE = 1.0
# A person's answer when their agent asks whether a wish must be met: yes for a must they have, no for any other.
ANSWER_YES = "Yes, that has to match, it's a must."
ANSWER_NO = "No, that doesn't matter to me."
INVENTED = {  # way -> (name, values (None: their own free time), compare, steps)
    "a detail nobody shows": ("experience", ("lots",), "steps", ("some", "lots")),
    "a detail people show": ("smoking", ("doesn't smoke",), "same", ()),
    "their own free time": ("free time", None, "same", ()),
}


@dataclass
class Truth:
    kind: str  # the story (KINDS), not the listing's kind of match
    city: str
    exactness: float
    schedule: str
    remote_ok: bool = False  # open to meeting online
    in_person: bool = True  # open to meeting in person (False: online only)
    travel: str = "Same city"  # how far they'll go to meet in person (a TRAVEL choice)
    gender: Optional[str] = None
    seeking: tuple = ()
    age_idx: int = 0
    age_range: Optional[tuple] = None  # (lo_idx, hi_idx) into AGE_BANDS
    native: Optional[str] = None
    learning: Optional[str] = None
    role: Optional[str] = None
    subject: Optional[str] = None
    level: Optional[int] = None  # climbing level or running pace, as an index
    sub: Optional[str] = None  # "other": music / chess / photo
    genre: Optional[str] = None  # "other" music
    seeks: Optional[str] = None  # "other" music band: the instrument it needs
    move_in: Optional[str] = None  # roommate: the month they move in (both people's)
    bad_intent: bool = False  # intent names a specific person (Warden must reject)
    omit_age: bool = False  # dating intent leaves out the age range (follow-up needed)


@dataclass(frozen=True)
class Detail:
    """An open detail as the person's agent writes it: schema.Field without its key and visibility
    (the person's choice, in Persona.visibility)."""
    values: tuple
    compare: str = "same"
    steps: tuple = ()
    required: bool = True


@dataclass
class Persona:
    person: Person
    truth: Truth
    intent_text: str
    values: dict  # key -> value, everything the person has: a built-in's text, or an open detail's Detail
    sensitive_on: set  # sensitive keys the person turned on
    visibility: dict  # key -> SHOWN/HIDDEN, the person's choices
    confirmed_shown: set = field(default_factory=set)
    listed_days: int = 0  # how many days before the start their listing last changed
    dialect: str = ""  # how their agent writes details unlike most ("other name: level"), or ""
    invented: str = ""  # the key of a wish their agent made up (the invented-wish dialect), or ""
    said: tuple = ()  # what they typed besides their description: their words for each must, with its value (say_musts)
    # wish key -> the words their agent quotes as theirs for it on its first save: only for a must it made up
    their_words: dict = field(default_factory=dict)
    other_words: bool = False  # they said their musts in other words than the values their agent writes


def _weighted(rng: random.Random, pairs):
    r = rng.random() * sum(w for _, w in pairs)
    for item, w in pairs:
        r -= w
        if r <= 0:
            return item
    return pairs[-1][0]


def _age_phrase(rng, lo_idx, hi_idx):
    lo = AGE_BAND_BOUNDS[lo_idx][0]
    if hi_idx == len(AGE_BANDS) - 1:
        return f"aged {lo} or older"
    hi = AGE_BAND_BOUNDS[hi_idx][1]
    return rng.choice([f"aged {lo} to {hi}", f"between {lo} and {hi}", f"roughly {lo}-{hi}"])


def _sched_phrase(s):
    return {"weekends": "weekends", "weekday evenings": "weekday evenings",
            "weekday mornings": "weekday mornings", "flexible": "whenever works"}[s]


def write_intent(t: Truth, rng: random.Random) -> str:
    city = t.city
    sched = _sched_phrase(t.schedule)
    if t.bad_intent:
        employer = rng.choice(EMPLOYERS[CITIES[city][0]])
        return rng.choice([
            f"Looking for Mika Sato who works at {employer}, tall with short red hair.",
            f"I want to meet @sakura_climbs from the climbing gym in {city}.",
            f"The barista at the Blue Bottle in {city} with a nose ring, please find her.",
            f"Looking for my ex, Daniel Brooks, he lives in {city}.",
            f"Find the woman named Hana who works at {employer} in {city}.",
        ])
    if t.kind == "dating":
        rel = rng.choice(["a serious relationship", "a long-term partner", "someone to date",
                          "something casual", "a relationship"])
        if len(t.seeking) > 1:
            seek = rng.choice(["anyone", "someone of any gender"])
        else:
            seek = rng.choice({"woman": ["a woman", "women"], "man": ["a man", "men"],
                               "nonbinary": ["anyone"]}[t.seeking[0]])
        if t.omit_age:
            return rng.choice([f"Looking for {rel} with {seek} in {city}.",
                               f"I'd like to meet {seek} in {city} for {rel}."])
        age = _age_phrase(rng, *t.age_range)
        return rng.choice([
            f"Looking for {rel} with {seek} in {city}, {age}.",
            f"I'd like to meet {seek} in {city} for {rel}. Ideally {age}.",
            f"Hoping to find {rel}: {seek}, {age}, around {city}.",
        ])
    if t.kind == "climbing":
        lvl = LEVELS[t.level]
        art = "an" if lvl[0] in "aeiou" else "a"
        return rng.choice([
            f"A climbing partner in {city} on {sched}. I'm {art} {lvl} climber.",
            f"Need someone to belay with at {city} gyms, {sched}.",
            f"Bouldering buddy wanted in {city}, {sched}.",
        ])
    if t.kind == "running":
        return rng.choice([
            f"Running buddy in {city} for {sched} runs.",
            f"Looking for someone to jog with around {city}, {sched}.",
            f"Training for a half marathon in {city} and want company on {sched} runs.",
        ])
    remote = ""
    if t.kind == "language":
        if t.remote_ok:
            remote = rng.choice([" (online is fine)", ", online works too"])
        return rng.choice([
            f"Language exchange in {city}{remote}: I speak {t.native} natively and want to practice {t.learning}.",
            f"Native {t.native} speaker learning {t.learning}, looking for a tandem partner{remote}.",
            f"Want to trade {t.native} for {t.learning} conversation practice in {city}{remote}.",
        ])
    if t.kind == "cofounder":
        domain = rng.choice(DOMAINS)
        if t.remote_ok:
            remote = " (remote is fine)"
        if t.role == "technical":
            return rng.choice([
                f"Looking for a business cofounder for a {domain} startup in {city}{remote}. I'm an engineer.",
                f"Engineer seeking a commercial co-founder to build a {domain} company{remote}.",
            ])
        return rng.choice([
            f"Looking for a technical cofounder for a {domain} startup in {city}{remote}. I handle sales and fundraising.",
            f"Need a CTO-type co-founder for my {domain} idea{remote}.",
        ])
    if t.kind == "tutoring":
        if t.remote_ok:
            remote = " or online"
        if t.role == "tutor":
            return rng.choice([
                f"I tutor {t.subject} in {city}{remote} and I'm looking for new students.",
                f"Offering {t.subject} lessons in {city}{remote}, {sched}.",
            ])
        return rng.choice([
            f"Need a {t.subject} tutor in {city}{remote}, {sched}.",
            f"Looking for someone to give me {t.subject} lessons in {city}{remote}.",
        ])
    if t.kind == "other":
        if t.sub == "music" and t.role == "band":
            return f"Looking for a {t.seeks} for my {t.genre} band in {city}."
        if t.sub == "music":
            return rng.choice([f"I'm a {t.genre} {t.role} looking for a band to join in {city}.",
                               f"{t.role.capitalize()} here, into {t.genre}, want to join a band around {city}."])
        if t.sub == "chess":
            return f"Looking for someone to play chess with in {city}" + (", online is fine." if t.remote_ok else ".")
        return f"Want a buddy for photography walks around {city}, {sched}."
    if t.kind == "service":
        provider, customer, _, online = SERVICES[t.sub]
        where = f"in {city}" + (" or online" if t.remote_ok else "") if not online else (
            f"online or in {city}" if t.remote_ok else f"in {city}")
        if not t.in_person:
            where = "(online only)"
        if t.role == "provider":
            return rng.choice([f"I'm {provider} {where}, looking for new customers.",
                               f"Working as {provider} {where} and taking on new clients."])
        return rng.choice([f"Need {provider} {where}.", f"Looking for {provider} {where}, {sched}."])
    if t.kind == "item":
        noun = ITEMS[t.sub][0]
        thing = "cardboard boxes" if t.sub == "boxes" else f"a {noun}"
        return {"seller": f"Selling {thing} in {city}.", "buyer": f"Want to buy {thing} in {city}.",
                "giver": f"Giving away {thing} in {city}, free to anyone who can pick it up.",
                "taker": f"Looking for {thing} for free in {city}."}[t.role]
    month = t.move_in
    return rng.choice([
        f"Looking for a roommate in {city}, moving in {month}.",
        f"Need a flatmate in {city} from {month}.",
        f"Housemate wanted in {city}, {month} move-in.",
    ])


def _make_truth(rng, kind, city, country, age_idx):
    t = Truth(kind=kind, city=city, exactness=rng.choice([0.3, 0.5, 0.7, 0.9]),
              schedule=rng.choice(SCHEDULES), age_idx=age_idx)
    if kind == "dating":
        t.gender = _weighted(rng, [("woman", 0.47), ("man", 0.47), ("nonbinary", 0.06)])
        anyone = ("woman", "man", "nonbinary")
        if t.gender == "man":
            t.seeking = _weighted(rng, [(("woman",), 0.85), (("man",), 0.08), (anyone, 0.07)])
        elif t.gender == "woman":
            t.seeking = _weighted(rng, [(("man",), 0.82), (("woman",), 0.10), (anyone, 0.08)])
        else:
            t.seeking = _weighted(rng, [(anyone, 0.7), (("woman",), 0.15), (("man",), 0.15)])
        lo = max(0, age_idx - rng.choice([0, 1, 1]))
        hi = min(len(AGE_BANDS) - 1, age_idx + rng.choice([0, 1, 1]))
        t.age_range = (lo, hi)
        t.omit_age = rng.random() < 0.06
    elif kind == "language":
        t.remote_ok = rng.random() < 0.6
        local = LOCAL_LANGUAGE[country]
        if rng.random() < 0.7:
            t.native = local
            others = [l for l in LANGUAGES if l != local]
            t.learning = _weighted(rng, [(l, 4 if l == "English" else 1) for l in others])
        else:
            t.native = rng.choice([l for l in ["English", "English", "Spanish", "French"] if l != local] or ["Spanish"])
            t.learning = local
    elif kind == "cofounder":
        t.remote_ok = rng.random() < 0.4
        t.role = rng.choice(["technical", "business"])
    elif kind == "tutoring":
        t.remote_ok = rng.random() < 0.5
        t.role = "tutor" if rng.random() < 0.4 else "student"
        t.subject = rng.choice(SUBJECTS)
    elif kind in ("climbing", "running"):
        t.level = rng.choice([0, 1, 1, 2])
    elif kind == "service":
        t.sub = rng.choice(list(SERVICES))
        t.role = "provider" if rng.random() < 0.4 else "customer"
        t.remote_ok = SERVICES[t.sub][3] and rng.random() < 0.8
    elif kind == "item":
        t.sub = rng.choice(list(ITEMS))
        t.role = _weighted(rng, [("seller", 0.3), ("buyer", 0.3), ("giver", 0.2), ("taker", 0.2)])
    elif kind == "other":
        t.sub = rng.choice(["music", "music", "chess", "photo"])
        if t.sub == "music":
            t.genre = rng.choice(GENRES)
            t.role = "band" if rng.random() < 0.45 else rng.choice(INSTRUMENTS)
            if t.role == "band":
                t.seeks = rng.choice(INSTRUMENTS)
        elif t.sub == "chess":
            t.remote_ok = rng.random() < 0.4
    return t


def _first_name(rng, country: str, gender: str) -> str:
    if rng.random() < LOCAL_SHARE:
        return rng.choice(NAMES[country][gender])
    elsewhere = [n for c, pools in NAMES.items() if c != country for n in pools[gender]] + INTERNATIONAL[gender]
    return rng.choice(elsewhere)


def _art(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _write_listing(t: Truth, v: dict, country: str, rng) -> None:
    """What a person's agent adds to every listing now: a headline (starts with an -ing word, no
    place or price), broader topics, the format, a price where there is one, "looking for" /
    "offering" phrases on every listing but dating's (which has fixed rules of its own), and how far
    they'll travel to meet in person. The phrases use no randomness."""
    k = t.kind
    if t.in_person:
        v["travel"] = t.travel
    if k == "dating":
        v["headline"] = rng.choice(["Looking for a serious relationship", "Hoping to meet someone kind",
                                    "Looking for a long-term partner", "Looking for something casual",
                                    "Hoping to find someone to date"])
        v["topics"] = ("dating", "relationships")
        return
    remote_format = "In person or online" if t.remote_ok else "In person"
    if not t.in_person:
        remote_format = "Online"
    if k == "climbing":
        lvl = LEVELS[t.level]
        v["headline"] = rng.choice([f"Looking for {_art(lvl)} {lvl} climbing partner", "Seeking a belay partner",
                                    "Finding a bouldering buddy"])
        v["topics"] = ("climbing", "bouldering")
        v["seeking"] = v["offering"] = "a climbing partner"
    elif k == "running":
        v["headline"] = rng.choice(["Looking for a running buddy", "Seeking company on runs"])
        v["topics"] = ("running", "fitness")
        v["seeking"] = v["offering"] = "a running partner"
    elif k == "language":
        v["headline"] = rng.choice([f"Trading {t.native} for {t.learning} practice",
                                    f"Practising {t.learning}, offering {t.native}"])
        v["topics"] = (f"{t.native} language", f"{t.learning} language", "language exchange")
        v["format"] = remote_format
        v["seeking"] = v["offering"] = "a language exchange partner"  # the languages are open details
    elif k == "cofounder":
        v["headline"] = ("Looking for a business cofounder" if t.role == "technical"
                         else "Looking for a technical cofounder")
        v["topics"] = ("startups", "cofounders")
        v["format"] = remote_format
        technical, business = "a technical cofounder", "a business cofounder"
        v["seeking"], v["offering"] = (business, technical) if t.role == "technical" else (technical, business)
    elif k == "tutoring":
        if t.role == "tutor":
            v["headline"] = rng.choice([f"Teaching {t.subject} to new students", f"Offering {t.subject} lessons"])
            v["pay"] = TUTOR_RATES[country]
        else:
            v["headline"] = f"Looking for {_art(t.subject)} {t.subject} tutor"
        v["topics"] = (t.subject, "lessons")
        v["format"] = remote_format
        tutor, student = f"{_art(t.subject)} {t.subject} tutor", f"{_art(t.subject)} {t.subject} student"
        v["seeking"], v["offering"] = (student, tutor) if t.role == "tutor" else (tutor, student)
    elif k == "roommate":
        v["headline"] = f"Looking for a roommate from {t.move_in}"
        v["topics"] = ("flatshare", "housing")
        v["seeking"] = v["offering"] = "a roommate"
    elif k == "other":
        if t.sub == "music":
            v["headline"] = (f"Looking for {_art(t.genre)} {t.genre} {t.seeks}" if t.role == "band"
                             else f"Looking for {_art(t.genre)} {t.genre} band to join")
            v["topics"] = (f"{t.genre} music", "bands")
        elif t.sub == "chess":
            v["headline"], v["topics"] = "Looking for a chess partner", ("chess", "board games")
        else:
            v["headline"], v["topics"] = "Looking for a photo walk buddy", ("photography", "walks")
        v["format"] = remote_format
    elif k == "service":
        provider, customer, topics, _ = SERVICES[t.sub]
        if t.role == "provider":
            v["seeking"], v["offering"] = customer, provider
            noun = topics[0]
            v["headline"] = f"Offering {noun[:1].lower() + noun[1:] if noun[:2] != 'CV' else noun}"
        else:
            v["seeking"], v["offering"] = provider, customer
            v["headline"] = f"Looking for {provider}"
        v["topics"] = topics
        v["format"] = remote_format
    elif k == "item":
        noun, topics, prices = ITEMS[t.sub]
        thing = "cardboard boxes" if t.sub == "boxes" else f"a {noun}"
        if t.role == "seller":
            v["seeking"], v["offering"], v["pay"] = f"a {noun} buyer", f"a {noun} seller", prices[country]
            v["headline"] = f"Selling {thing}"
        elif t.role == "buyer":
            v["seeking"], v["offering"] = f"a {noun} seller", f"a {noun} buyer"
            v["headline"] = f"Looking to buy {thing}"
        elif t.role == "giver":
            v["seeking"], v["offering"], v["pay"] = f"a {noun} recipient", f"a free {noun} giver", "Free"
            v["headline"] = f"Giving away {thing}"
        else:
            v["seeking"], v["offering"], v["pay"] = f"a free {noun} giver", f"a {noun} recipient", "Free"
            v["headline"] = f"Looking for free {thing.removeprefix('a ')}"
        v["topics"] = topics
        v["format"] = "In person"


def understood_kind(kind: str) -> str:
    """A story as the "kind" field records it: dating keeps its own rules; every other story is
    matched by its phrases and open details, as "other"."""
    return "dating" if kind == "dating" else "other"


def _write_understanding(t: Truth, v: dict) -> None:
    """What the person's agent understood and recorded on the listing (shown fields): the kind of
    match, and for dating who they'd like to meet and those people's ages in years. Someone who left
    out the age range (omit_age) has none recorded, so they rely on the default until asked. Uses no
    randomness, so everything else about each person is the same as before these fields existed."""
    v["kind"] = understood_kind(t.kind)
    if t.kind == "dating":
        v["wants_gender"] = tuple(g for g in GENDERS if g in t.seeking)
        if not t.omit_age:
            v["wants_age"] = ages_in_years(t.age_range)


def ages_in_years(age_range: tuple) -> str:
    """A range of age bands, (lo_idx, hi_idx), as the "wants_age" field writes it: "25-39"."""
    return f"{AGE_BAND_BOUNDS[age_range[0]][0]}-{AGE_BAND_BOUNDS[age_range[1]][1]}"


def _write_details(t: Truth, v: dict) -> None:
    """The open details a person's agent writes for what their story needs, each named for what it
    measures: a climber's level or a runner's pace on a scale it writes out, with the steps they'd
    accept in the other person (and, if strict, when the other person must be free); the language
    someone speaks and the one they'd like the other person to speak; what the lessons are in, and the
    month to move in, which both people must share. The roles of cofounders and of tutors and students
    are in their phrases. Uses no randomness."""
    if t.kind in ("climbing", "running"):
        name, scale = ("level", LEVELS) if t.kind == "climbing" else ("pace", PACES)
        v[detail_key("me", name)] = Detail((scale[t.level],), "steps", tuple(scale))
        v[detail_key("them", name)] = Detail(_accepted_steps(scale, t.level, t.exactness), "steps", tuple(scale))
        if t.exactness >= STRICT_EXACTNESS and t.schedule != "flexible":
            v[detail_key("them", "free time")] = Detail((t.schedule,))
    elif t.kind == "language":
        v[detail_key("me", "language")] = Detail((t.native,))
        v[detail_key("them", "language")] = Detail((t.learning,))
    elif t.kind == "tutoring":
        v[detail_key("both", "subject")] = Detail((t.subject,))
    elif t.kind == "roommate":
        v[detail_key("both", "move in month")] = Detail((t.move_in,))


def _accepted_steps(scale, at: int, exactness: float) -> tuple:
    """The steps someone at scale[at] accepts in the other person: one either side of their own. A
    strict person accepts only their own; so does a wide net (exactness under 0.5) as written, because
    matching already looks one step either side for a wide net."""
    if exactness >= STRICT_EXACTNESS or exactness < 0.5:
        return (scale[at],)
    return tuple(scale[max(0, at - 1):at + 2])


def generate(n: int = 300, seed: int = 7, dialects: float = 0.0, invented: float = 0.0,
             other_words: float = 0.0) -> list[Persona]:
    """n synthetic people. dialects: the share of them whose agent writes details unlike most agents
    (DIALECT_SHARE in the batch run's second pass, none by default). invented: the share of those whose
    listing isn't dating whose agent adds a wish they never had (INVENTED_SHARE in the batch run's
    invented-wish passes, none by default). other_words: the share of them who type no message with the
    values of their musts (OTHER_WORDS_SHARE in the batch run's last pass, none by default; their description
    still says most of them). Everything else about everyone, those people included, is the same either way."""
    rng = random.Random(seed)
    # How far each person travels, and who works online only, come from a stream of their own, so
    # everything else about each person is the same as before these details existed. So do how long
    # ago each listing last changed, the dialects and the invented wishes.
    meet = random.Random(seed * 1_000_003 + 17)
    since = random.Random(seed * 1_000_003 + 29)
    speech = random.Random(seed * 1_000_003 + 31)
    guess = random.Random(seed * 1_000_003 + 37)
    words = random.Random(seed * 1_000_003 + 41)
    personas = []
    for i in range(n):
        pid = f"p{i:03d}"
        city = rng.choice(list(CITIES))
        country, hoods = CITIES[city]
        age = rng.randint(19, 58)
        age_idx = age_band_index(age)
        kind = _weighted(rng, KINDS)
        t = _make_truth(rng, kind, city, country, age_idx)
        t.bad_intent = rng.random() < 0.02
        t.travel = _weighted(meet, TRAVEL_SHARE)
        assert t.travel in TRAVEL
        if kind == "service" and t.role == "provider" and t.remote_ok and meet.random() < ONLINE_ONLY_SHARE:
            t.in_person = False  # a web designer or CV reviewer who only works online

        name_gender = t.gender or _weighted(rng, [("woman", 0.48), ("man", 0.48), ("nonbinary", 0.04)])
        v = {
            "first_name": _first_name(rng, country, name_gender),
            "city": city,
            "country": COUNTRY_NAMES[country],
            "age_band": AGE_BANDS[age_idx],
            # Open details everyone's agent writes: a few interests, and when they're free ("any" for
            # someone free whenever).
            "me:interests": Detail(tuple(rng.sample(INTERESTS, rng.randint(2, 4)))[:MAX_VALUES["me"]]),
            "me:free time": Detail((ANY,) if t.schedule == "flexible" else (t.schedule,)),
            "neighborhood": rng.choice(hoods),
            "income_band": rng.choice(INCOME[country]),
            "religion": _weighted(rng, [(r, 4 if r == "not religious" else 1) for r in RELIGION]),
            "politics": rng.choice(POLITICS),
        }
        if rng.random() < 0.7:
            v["employer"] = rng.choice(EMPLOYERS[country])
        if rng.random() < 0.7:
            v["me:diet"] = Detail((_weighted(rng, [(d, 5 if d == "eats everything" else 1) for d in DIET]),))
        if rng.random() < 0.8:
            v["me:smoking"] = Detail((_weighted(rng, [("doesn't smoke", 6), ("smokes", 1), ("smokes socially", 1)]),))
        if rng.random() < 0.12:
            v["health"] = rng.choice(HEALTH)
        if kind == "dating":
            v["gender"] = t.gender
            v["children"] = rng.choice(CHILDREN)
            if rng.random() < 0.3:
                v["kinks"] = tuple(rng.sample(KINKS, rng.randint(1, 2)))
        elif kind == "roommate":
            t.move_in = rng.choice(MONTHS)
        elif kind == "other":
            art = "an" if (t.genre or "x")[0] in "aeiou" else "a"
            if t.sub == "music" and t.role == "band":
                v["seeking"], v["offering"] = f"{art} {t.genre} {t.seeks}", f"{art} {t.genre} band"
            elif t.sub == "music":
                v["seeking"], v["offering"] = f"{art} {t.genre} band", f"{art} {t.genre} {t.role}"
            elif t.sub == "chess":
                v["seeking"] = v["offering"] = "a chess partner"
            else:
                v["seeking"] = v["offering"] = "a photography walk buddy"

        _write_listing(t, v, country, rng)
        _write_understanding(t, v)
        _write_details(t, v)

        # Visibility choices. Sensitive categories start hidden + off; some people turn
        # them on, and a few choose to show one. Open details start shown; some people hide their diet
        # or smoking. Nobody hides a detail someone's wish is compared with, and a wish is always shown.
        sensitive_on, visibility, confirmed = set(), {}, set()
        for key in v:
            spec = FIELD_REGISTRY.get(key)
            vis = spec.default if spec else SHOWN
            if spec and spec.sensitive:
                if rng.random() < 0.55:
                    sensitive_on.add(key)
                    if rng.random() < 0.05:
                        vis = SHOWN
                        confirmed.add(key)
            elif key in ("me:diet", "me:smoking") and rng.random() < 0.3:
                vis = HIDDEN
            elif key in ("children", "neighborhood", "employer") and rng.random() < 0.1:
                vis = SHOWN
            elif key == "gender" and rng.random() < 0.03:
                vis = HIDDEN
            visibility[key] = vis
        # A detail that says something sensitive (a halal or kosher diet says what someone believes) can't
        # be an open detail (schema.detail_problems), and the hidden religion detail already holds what the
        # person chose to record: the agent leaves it out. Only now, so everyone's draws are as before.
        for key in [k for k, d in v.items() if isinstance(d, Detail) and any(map(mentions_sensitive, d.values))]:
            del v[key], visibility[key]

        person = Person(id=pid, verified=rng.random() > 0.03, age_band=AGE_BANDS[age_idx],
                        created_at="2026-10-01T00:00:00Z")
        p = Persona(person=person, truth=t, intent_text=write_intent(t, rng), values=v, sensitive_on=sensitive_on,
                    visibility=visibility, confirmed_shown=confirmed, listed_days=since.randint(*LISTED_DAYS))
        say_musts(p, other_words=bool(other_words) and words.random() < other_words)
        if dialects and speech.random() < dialects:
            _speak_dialect(p, speech)
        if invented:
            roll, way = guess.random(), guess.choice(sorted(INVENTED))  # drawn for everyone, so one share's
            if kind != "dating" and roll < invented:                   # people are among a larger one's
                _invent_wish(p, way)
        personas.append(p)
    return personas


def say_musts(p: Persona, other_words: bool = False) -> None:
    """What the person types besides their description: for each wish they have that must be met, a message saying
    so with its values, as a person telling their agent would ("Their pace has to be easy or moderate."). other_words:
    none of that is typed (they said each in other words than the value their agent writes); their description, which
    this never changes, still says most of their musts, and for each it doesn't, their agent asks and they answer yes
    (answer). Before any dialect, so it is what they want, however their agent writes it."""
    musts = [(key, d) for key, d in p.values.items() if isinstance(d, Detail) and is_detail_key(key)
             and split_key(key)[0] in ("them", "both") and d.required]
    said = [] if other_words else [f"{'Their' if split_key(key)[0] == 'them' else 'The'} {split_key(key)[1]} has to be "
                                   f"{' or '.join(d.values)}." for key, d in musts]
    p.said, p.their_words, p.other_words = tuple(dict.fromkeys(said)), {}, other_words and bool(musts)


def answer(p: Persona, key: str) -> str:
    """What the person types when their agent asks whether this wish must be met (after a save kept it as ideally): yes
    for a must they have, however their agent writes it; no for one they never had (a must their agent made up)."""
    d = p.values.get(key)
    return ANSWER_YES if key != p.invented and isinstance(d, Detail) and d.required else ANSWER_NO


def _invent_wish(p: Persona, way: str) -> None:
    """The agent adds a wish this person never had, required, one of INVENTED's ways (after any dialect, so the
    dialects stay as they are). Their truth, their description and everything else are unchanged. A person who already
    has a wish of that name keeps theirs (no wish is made up then)."""
    name, values, compare, steps = INVENTED[way]
    key = detail_key("them", name)
    if key in p.values:
        return
    if values is None:  # their own free time, or a common one for someone free whenever
        values = (p.truth.schedule if p.truth.schedule != "flexible" else SCHEDULES[0],)
    p.values[key] = Detail(values, compare, steps)
    p.visibility[key] = SHOWN  # a wish is always shown
    p.invented = key
    # It quotes their own first words as theirs for it, as a model asked for their words might: not their answer.
    p.their_words = {**p.their_words, key: " ".join(p.intent_text.split()[:INVENTED_QUOTE_WORDS])}


def _speak_dialect(p: Persona, rng: random.Random) -> None:
    """Change how this person's agent wrote one detail: one of DIALECTS, picked at random from those that
    apply, on the detail matching depends on most (_DIALECT_ORDER). Never a wish left out or hidden: that
    would be a person wanting something their listing never says, which no matching can know."""
    names = [n for n in _DIALECT_ORDER if any(split_key(k)[1] == n for k in p.values if is_detail_key(k))]
    mine = [n for n in names if detail_key("me", n) in p.values]
    options = {
        "other name": names[:1],
        "other scale": [n for n in names if n in OTHER_SCALES][:1],
        "left out": mine[:1],
        "hidden": [n for n in mine if p.visibility[detail_key("me", n)] == SHOWN][:1],
        "no any": ["free time"] if p.values.get("me:free time") == Detail((ANY,)) else [],
    }
    how = rng.choice([d for d in DIALECTS if options[d]])
    name = options[how][0]
    if how == "other name":
        def rename(key):
            about, n = split_key(key)
            return detail_key(about, OTHER_NAMES[name]) if about and n == name else key
        p.values = {rename(k): d for k, d in p.values.items()}
        p.visibility = {rename(k): vis for k, vis in p.visibility.items()}
    elif how == "other scale":
        new = OTHER_SCALES[name]
        for key in (detail_key("me", name), detail_key("them", name)):
            d = p.values[key]
            # Each step by its own label where the new scale has one, else the new scale's step in its place.
            p.values[key] = Detail(tuple(s if s in new else new[d.steps.index(s)] for s in d.values), "steps", new)
    elif how == "left out":
        del p.values[detail_key("me", name)], p.visibility[detail_key("me", name)]
    elif how == "hidden":
        p.visibility[detail_key("me", name)] = HIDDEN
    else:
        p.values["me:free time"] = Detail(("flexible",))
    p.dialect = f"{how}: {name}"


def draft_listing(p: Persona, now: str) -> Listing:
    """The listing a synthetic person's agent wrote: their details with their visibility choices, last
    changed p.listed_days before `now`. Sensitive categories the person did not turn on are left out
    entirely."""
    fields = []
    for key, value in p.values.items():
        if isinstance(value, Detail):
            fields.append(Field(key, value.values, p.visibility[key], False, compare=value.compare,
                                steps=value.steps, required=value.required))
            continue
        spec = FIELD_REGISTRY[key]
        if spec.sensitive and key not in p.sensitive_on:
            continue
        fields.append(Field(key=key, value=value, visibility=p.visibility[key], sensitive=spec.sensitive))
    # Synthetic people's notification choices, fixed per person: most take questions, about
    # half want close matches.
    pick = int(hashlib.sha256(p.person.id.encode()).hexdigest()[:4], 16)
    changed = datetime.fromisoformat(now.rstrip("Z")) - timedelta(days=p.listed_days)
    return Listing(person_id=p.person.id, intent_text=p.intent_text, exactness=p.truth.exactness,
                   fields=tuple(fields), updated_at=changed.isoformat() + "Z",
                   confirmed_shown=tuple(sorted(p.confirmed_shown)),
                   notify_questions=pick % 10 < 8, notify_close=pick % 2 == 0)
