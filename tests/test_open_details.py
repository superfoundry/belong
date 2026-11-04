"""Open details (SPEC §4, §5): the one normaliser, the entry rules every detail passes, the comparisons
matching and the Warden share, and the fixed sensitive words, which must still catch every word they
caught before details were opened up."""
from __future__ import annotations

import unittest
import unittest.mock
from dataclasses import replace

from belong import schema
from belong.schema import (HIDDEN, MAX_DETAILS, SHOWN, Field, agrees, canonical, city_centre, comparable, detail_key,
                           detail_problems, detail_reasons, display_form, is_detail_key, km_between, meets,
                           mentions_sensitive, name_problems, own, plain_place, satisfies, says_nothing,
                           sensitive_categories, shown_claim_problems, split_key, vouching_problems)
from tests.helpers import detail

SKILL = ("low", "middle", "high", "top")


def problems(*details, kind="other"):
    return detail_problems(details, kind)


class OneNormaliser(unittest.TestCase):
    def test_case_width_accents_and_separators_are_one_value(self):
        for a, b in (("Same-Day", "same day"), ("ｓａｍｅ　ｄａｙ", "same day"), ("Café", "cafe"), ("MÜNCHEN", "munchen"),
                     ("night_owl", "night owl"), ("either/or", "either or"), ("  two   spaces ", "two spaces"),
                     ("Straße", "strasse")):
            self.assertEqual(canonical(a), canonical(b), (a, b))

    def test_marks_that_are_part_of_a_letter_elsewhere_are_kept(self):
        self.assertNotEqual(canonical("でんき"), canonical("てんき"))
        self.assertEqual(canonical("한국어"), "한국어")
        self.assertEqual(canonical("हिंदी"), "हिंदी")

    def test_it_is_idempotent(self):
        for text in ("Same-Day", "Café au lait", "でんき", "ＡＢＣ_def/ghi"):
            self.assertEqual(canonical(canonical(text)), canonical(text))

    def test_it_remembers_what_it_has_read_without_growing_for_ever(self):
        schema._CANONICAL.clear()
        self.assertEqual(canonical("Night-Owl"), "night owl")
        self.assertEqual(schema._CANONICAL["Night-Owl"], "night owl")
        self.assertEqual(canonical(3), "3")  # anything, read as text
        with unittest.mock.patch.object(schema, "_CANONICAL_MAX", 3):
            for text in ("a", "b", "c", "d"):
                canonical(text)
            self.assertLessEqual(len(schema._CANONICAL), 3)

    def test_places_are_compared_the_same_way_with_every_accent_off(self):
        """One normaliser: a place written with hyphens, spaces, another width or a sharp s is one place, in
        matching as on the shared list."""
        for a, b in (("Saint-Étienne", "saint etienne"), ("Hann. Münden", "hann. munden"), ("Groß-Gerau", "gross gerau"),
                     ("Ｌｉｍｅｒｉｃｋ", "limerick"), ("Bad_Ischl", "Bad Ischl"), ("Ólafsvík", "olafsvik")):
            self.assertEqual(plain_place(a), plain_place(b), (a, b))
        self.assertEqual(km_between("Saint-Étienne", "France", "saint etienne", "france"), 0.0)
        # Known city centres are found however their names are written.
        self.assertEqual(city_centre("KÖLN", "Germany"), city_centre("koln", "germany"))
        self.assertIsNotNone(city_centre("Lisboa", "PORTUGAL"))

    def test_values_are_stored_as_written_with_single_spaces(self):
        self.assertEqual(display_form("  Cello   Suite "), "Cello Suite")
        self.assertEqual(display_form("Ｃｅｌｌｏ"), "Cello")


class DetailKeys(unittest.TestCase):
    def test_keys_say_who_a_detail_is_about(self):
        self.assertEqual(detail_key("them", "skill"), "them:skill")
        self.assertEqual(split_key("both:move in month"), ("both", "move in month"))
        self.assertEqual(split_key("city"), ("", "city"))
        for key in ("city", "us:skill", "me:", ":skill", None, 3):
            self.assertFalse(is_detail_key(key), key)
        f = detail("skill", "low")
        self.assertEqual((f.about, f.name, f.strings(), f.texts()), ("me", "skill", ["low"], ["low", "skill"]))
        built_in = Field("city", "Lyon", SHOWN, False)
        self.assertEqual((built_in.about, built_in.name, built_in.texts()), ("", "city", ["Lyon"]))


class EntryRules(unittest.TestCase):
    def test_well_formed_details_pass(self):
        self.assertEqual(problems(detail("skill", "middle", compare="steps", steps=SKILL),
                                  detail("skill", ("middle", "high"), about="them", compare="steps", steps=SKILL),
                                  detail("instrument", ("Cello", "Viola"), about="both"),
                                  detail("night owl", "certainly", visibility=HIDDEN),
                                  detail("pets", "any", about="them", required=False)), [])

    def test_names_are_short_plain_canonical_words(self):
        for name in ("Skill", "skill!", "a b c d", "x" * 25, "", "skill_level", "two  spaces", " skill"):
            self.assertTrue(name_problems(name), name)
        for name in ("skill", "x" * 24, "a b c", "5k time", "한국어"):
            self.assertEqual(name_problems(name), [], name)

    def test_names_in_every_script_with_the_marks_their_letters_carry(self):
        """Names are letters (with the marks that belong to them), digits and single spaces, by Unicode category,
        so a name in Devanagari, Thai or Tamil is as good as one in Latin letters."""
        for name in ("स्तर", "भाषा", "ระดับ", "ภาษา", "நிலை", "அளவு", "ระดับ ภาษา", "स्तर 2"):
            self.assertEqual(name_problems(name), [], name)
            self.assertEqual(problems(detail(name, "plain words")), [], name)
        for name in ("\u0301abc", "2\u0301", "स्तर!", "स्तर  भाषा", "நிலை-"):  # a mark on nothing, symbols, spacing
            self.assertTrue(name_problems(name), name)

    def test_names_never_pass_for_a_built_in_or_an_id_checked_fact(self):
        for name in ("city", "first name", "age band", "wants gender", "topics", "topic", "kinks", "kink", "citys",
                     "headline", "income band", "employer", "children"):
            self.assertTrue(any("built-in" in p for p in name_problems(name)), name)
        for name in ("age", "ages", "real name", "id", "verified", "verified status", "official title", "kyc",
                     "belong", "warden", "gender", "identity"):
            self.assertTrue(any("vouches" in p for p in name_problems(name)), name)

    def test_a_name_with_a_sensitive_word_points_to_its_hidden_detail(self):
        self.assertIn("use the hidden health detail (health)", name_problems("medical history")[0])
        self.assertTrue(any("use the hidden health detail" in p for p in name_problems("health")))  # built-in too
        self.assertIn("use the hidden religion detail (religion)", name_problems("faith")[0])
        self.assertIn("use the hidden income detail (income_band)", name_problems("salary")[0])
        self.assertIn("keep it in the description only", name_problems("pronouns")[0])

    def test_compare_and_steps(self):
        for bad in (detail("skill", "low", compare="rank"), detail("skill", "low", steps=SKILL),
                    detail("skill", "low", compare="steps", steps=("low",)),
                    detail("skill", "a1", compare="steps", steps=tuple(f"a{i}" for i in range(8))),
                    detail("skill", "low", compare="steps", steps=("low", "Low")),
                    detail("skill", "low", compare="steps", steps=("low", "x" * 25)),
                    detail("skill", "expert", compare="steps", steps=SKILL),
                    detail("skill", ("low", "high"), compare="steps", steps=SKILL),
                    detail("skill", ("low", "high"), about="them", compare="steps", steps=SKILL),
                    detail("skill", ("high", "middle"), about="them", compare="steps", steps=SKILL),
                    # On a scale, your own value (or the request's) is a step: "any" only says what you'd like.
                    detail("skill", "any", compare="steps", steps=SKILL),
                    detail("skill", "any", about="both", compare="steps", steps=SKILL)):
            self.assertTrue(problems(bad), bad)
        for good in (detail("skill", "any", about="them", compare="steps", steps=SKILL),
                     detail("skill", ("low", "middle", "high"), about="them", compare="steps", steps=SKILL)):
            self.assertEqual(problems(good), [], good)

    def test_values(self):
        for bad in (detail("pets", ("cats", "dogs", "fish", "birds")), detail("pets", tuple("abcdef"), about="them"),
                    detail("pets", ()), detail("pets", ("cats", "Cats")), detail("pets", "x" * 41), detail("pets", "5k"),
                    detail("pets", "12"), detail("pets", "Yes"), detail("pets", "TRUE"), detail("pets", "<b>cats</b>"),
                    detail("pets", "cats | dogs"), detail("pets", "see http://cats"), detail("pets", "cats​"),
                    detail("pets", "cats  dogs"), detail("pets", " cats"), detail("pets", ("any", "cats"))):
            self.assertTrue(problems(bad), bad)
        self.assertEqual(problems(detail("pets", ("cats", "dogs", "fish", "birds", "mice"), about="them")), [])

    def test_a_claim_the_id_check_or_belong_vouches_for_never_goes_in_a_value(self):
        """Names have TRUST_WORDS; values, steps and shown built-in free text can't claim it either. Only the vouching
        words: everyday ones stay usable."""
        for value in ("ID verified", "Verified seller", "kyc done", "ID-verified"):
            self.assertTrue(any("vouches" in p for p in problems(detail("account", value))), value)
        self.assertTrue(vouching_problems("ＫＹＣ cleared"))  # any width
        steps = detail("standing", "new", compare="steps", steps=("new", "verified"))
        self.assertTrue(any("vouches" in p for p in problems(steps)))
        for value in ("same age", "brand identity", "official rules", "my name day", "gender neutral", "unverified rumours"):
            self.assertEqual(vouching_problems(value), [], value)
        shown = [Field("headline", "Lending a canoe, Verified owner", SHOWN, False),
                 Field("topics", ("canoes", "kyc"), SHOWN, False), Field("pay", "Free", SHOWN, False),
                 Field("employer", "Verified Ltd", HIDDEN, False)]  # hidden: never shown, so not a claim
        self.assertEqual([p.split(":")[0] for p in shown_claim_problems(shown)], ["'headline'", "'topics'"])

    def test_contact_details_never_go_in_a_value(self):
        for value in ("insta @janedoe99", "jane@example.com", "call +44 7700 900123", "www.janedoe.com",
                      "https://janedoe.com"):
            self.assertTrue(any("contact details" in p for p in problems(detail("contact", value))), value)

    def test_contact_details_never_go_in_a_name_either(self):
        """A name is shown wherever its detail is: a phone number as the name is contact details too."""
        for name in ("07700 900123", "call 0612345678", "44 7700 900123"):
            self.assertTrue(any("contact details" in p for p in name_problems(name)), name)
            self.assertTrue(any("contact details" in p for p in problems(detail(name, "plain words"))), name)
        for name in ("top 10", "class of 2024", "size 42"):  # a number in a name is still fine
            self.assertEqual(name_problems(name), [], name)

    def test_a_wish_is_always_shown(self):
        """What someone would like from the other person (them) and what the request needs them both to share
        (both) are wishes on the other person's own value: matching reads only shown ones, so a hidden one
        could never decide anything."""
        self.assertTrue(problems(detail("pets", "cats", about="them", visibility=HIDDEN)))
        self.assertTrue(problems(detail("place", "north", about="both", visibility=HIDDEN)))
        self.assertEqual(problems(detail("pets", "cats", visibility=HIDDEN), detail("place", "north", about="both")), [])

    def test_open_details_are_never_sensitive(self):
        self.assertTrue(problems(replace(detail("pets", "cats"), sensitive=True)))

    def test_a_sensitive_word_anywhere_sends_it_to_its_hidden_detail(self):
        cases = {("about me", ("Sikh",)): "religion", ("diet", ("halal",)): "religion", ("diet", ("kosher",)): "religion",
                 ("condition", ("asthma",)): "health", ("lean", ("politically moderate",)): "politics",
                 ("pay band", ("high earnings",)): "income_band"}
        for (name, values), key in cases.items():
            self.assertTrue(any(f"({key})" in p for p in problems(detail(name, values))), name)
        steps = detail("outlook", "secular", compare="steps", steps=("secular", "religious"))
        self.assertTrue(any("(religion)" in p for p in problems(steps)))
        self.assertTrue(any("description only" in p for p in problems(detail("about me", "queer"))))

    def test_intimacy_only_on_dating_listings(self):
        for kind in ("other", None, "nonsense"):
            self.assertTrue(any("only for dating" in p for p in problems(detail("after dark", "bondage"), kind=kind)), kind)
        self.assertTrue(any("(kinks)" in p for p in problems(detail("after dark", "bondage"), kind="dating")))

    def test_per_listing_rules(self):
        many = [detail(f"note {chr(97 + i)}", "plain words") for i in range(MAX_DETAILS + 1)]
        self.assertTrue(problems(*many))
        self.assertEqual(problems(*many[:MAX_DETAILS]), [])
        self.assertTrue(problems(detail("pets", "cats"), detail("pets", "dogs")))  # one key once
        self.assertTrue(problems(detail("place", "north"), detail("place", "north", about="both")))  # me or both
        self.assertTrue(problems(detail("place", "north", about="both"), detail("place", "north", about="them")))
        self.assertEqual(problems(detail("tongue", "Korean"), detail("tongue", "English", about="them")), [])  # an exchange


class Comparisons(unittest.TestCase):
    def test_own_is_the_shown_me_or_both_value(self):
        me, both, wish = detail("pets", "cats"), detail("place", "north", about="both"), detail("size", "big", about="them")
        fields = {f.key: f for f in (me, both, wish, detail("hidden one", "x y", visibility=HIDDEN))}
        self.assertEqual((own(fields, "pets"), own(fields, "place"), own(fields, "size"), own(fields, "hidden one")),
                         (me, both, None, None))

    def test_same(self):
        want = detail("tongue", ("Korean", "Japanese"), about="them")
        self.assertTrue(satisfies(want, detail("tongue", "korean")))
        self.assertFalse(satisfies(want, detail("tongue", "English")))
        self.assertTrue(satisfies(want, detail("tongue", "any")))
        self.assertTrue(satisfies(detail("tongue", "any", about="them"), detail("tongue", "English")))
        self.assertTrue(agrees(detail("tongue", ("English", "French")), detail("tongue", "french"), strict=True))
        self.assertFalse(agrees(detail("tongue", "English"), detail("other", "English"), strict=False))

    def test_steps_by_label(self):
        want = detail("skill", ("middle", "high"), about="them", compare="steps", steps=SKILL)
        self.assertTrue(satisfies(want, detail("skill", "High", compare="steps", steps=SKILL)))
        self.assertFalse(satisfies(want, detail("skill", "low", compare="steps", steps=SKILL)))
        other_scale = ("beginner", "middle", "expert")
        self.assertTrue(comparable(want, detail("skill", "middle", compare="steps", steps=other_scale)))
        self.assertTrue(satisfies(want, detail("skill", "middle", compare="steps", steps=other_scale)))
        self.assertFalse(comparable(want, detail("skill", "expert", compare="steps", steps=other_scale)))
        self.assertFalse(comparable(want, detail("skill", "middle")))  # a different compare

    def test_steps_by_distance_only_on_one_scale(self):
        want = detail("skill", "high", about="them", compare="steps", steps=SKILL)
        near = detail("skill", "middle", compare="steps", steps=SKILL)
        self.assertFalse(satisfies(want, near))
        self.assertTrue(satisfies(want, near, tolerance=1))
        self.assertFalse(satisfies(want, detail("skill", "low", compare="steps", steps=SKILL), tolerance=1))
        moved = detail("skill", "middle", compare="steps", steps=("low", "middle", "high"))
        self.assertFalse(satisfies(want, moved, tolerance=1))  # not the same scale: labels only

    def test_two_own_values_agree(self):
        a, b = (detail("skill", v, compare="steps", steps=SKILL) for v in ("middle", "high"))
        self.assertTrue(agrees(a, b, strict=False))
        self.assertFalse(agrees(a, b, strict=True))
        self.assertFalse(agrees(a, detail("skill", "top", compare="steps", steps=SKILL), strict=False))
        c = detail("skill", "middle", compare="steps", steps=("middle", "expert"))
        self.assertTrue(agrees(a, c, strict=True))  # different scales: the same label
        self.assertFalse(agrees(b, c, strict=False))

    def test_any_says_nothing_in_common(self):
        """An own "any" never agrees or meets a wish exactly (no reason, no score for it), and a strict request
        can't rely on it at all; a wish for any value, or for every step of its scale, says nothing either."""
        anything, green = detail("tea", "any"), detail("tea", "green")
        wish = detail("tea", "green", about="them")
        self.assertTrue(says_nothing(anything))
        self.assertTrue(says_nothing(detail("skill", SKILL, about="them", compare="steps", steps=SKILL)))
        self.assertFalse(says_nothing(detail("skill", SKILL[:3], about="them", compare="steps", steps=SKILL)))
        self.assertFalse(agrees(anything, green, strict=False))
        self.assertTrue(comparable(wish, anything))              # an open request may take it
        self.assertFalse(comparable(wish, anything, strict=True))  # a strict one never does
        self.assertTrue(comparable(wish, green, strict=True))
        self.assertFalse(meets(wish, anything))
        self.assertFalse(meets(detail("tea", "any", about="them"), green))
        self.assertTrue(meets(wish, green))
        a = {f.key: f for f in (anything, detail("tea", "green", about="them"))}
        b = {f.key: f for f in (green,)}
        self.assertEqual(detail_reasons(a, b, strict=False), ("wants:tea",))  # their green meets the wish; any agrees with nothing
        self.assertEqual(detail_reasons(b, {"me:tea": anything}, strict=False), ())

    def test_reasons_are_worked_out_from_shown_details_only(self):
        a = {f.key: f for f in (detail("tongue", "Korean"), detail("tongue", "English", about="them"),
                                detail("pets", "cats"), detail("place", "north", about="both"),
                                detail("secret", "x y", visibility=HIDDEN))}
        b = {f.key: f for f in (detail("tongue", "English"), detail("tongue", "Korean", about="them"),
                                detail("pets", "dogs"), detail("place", "North", about="both"), detail("secret", "x y"))}
        self.assertEqual(detail_reasons(a, b, strict=False), ("agree:place", "wants:tongue"))
        self.assertEqual(detail_reasons(b, a, strict=True), ("agree:place", "wants:tongue"))
        one_way = dict(b)
        del one_way["them:tongue"]
        self.assertEqual(detail_reasons(a, one_way, strict=False), ("agree:place", "wants:tongue"))
        # A wish is met only exactly: one step away is close, not a reason.
        want = {"them:skill": detail("skill", "high", about="them", compare="steps", steps=SKILL)}
        near = {"me:skill": detail("skill", "middle", compare="steps", steps=SKILL)}
        self.assertEqual(detail_reasons(want, near, strict=False), ())


# Every word (or wording) the agent's sensitive-word check caught before open details (agent.py at
# 2ffb1c7, including the words it took from the synthetic data's choices). Each must still be caught.
SENSITIVE_BEFORE = [
    # religion (from the old synthetic choices, then other words)
    "buddhist", "buddhists", "shinto", "shintos", "christian", "christians", "muslim", "muslims", "jewish",
    "hindu", "hindus", "buddhism", "christianity", "catholic", "catholics", "catholicism", "protestant",
    "protestants", "evangelical", "evangelicals", "mormon", "mormons", "islam", "islamic", "jew", "jews",
    "judaism", "hinduism", "sikh", "sikhs", "sikhism", "atheist", "atheists", "atheism", "agnostic", "agnostics",
    "religious", "religion", "faith", "church", "mosque", "synagogue", "bible", "quran", "koran", "torah",
    "spiritual", "prayer", "praying", "orthodox", "baptist", "baptists", "sunni", "sunnis", "shia", "shias",
    "shiite", "shiites", "ramadan",
    # health
    "type 1 diabetes", "adhd", "celiac disease", "celiac diseases", "chronic migraine", "chronic migraines",
    "hearing impairment", "hearing impairments", "diabetes", "diabetic", "celiac", "coeliac", "migraine",
    "migraines", "deaf", "disabled", "disability", "disabilities", "wheelchair", "autism", "autistic", "epilepsy",
    "epileptic", "hiv", "hsv", "herpes", "cancer", "chemo", "chemotherapy", "depression", "depressed", "anxiety",
    "bipolar", "schizophrenia", "schizophrenic", "ptsd", "ocd", "illness", "sober", "sobriety", "rehab",
    "pregnant", "pregnancy", "asthma", "asthmatic", "dyslexia", "dyslexic",
    # politics
    "progressive", "progressives", "conservative", "conservatives", "liberal", "liberals", "left-wing",
    "left wing", "right-wing", "right wing", "leftist", "leftists", "socialist", "socialists", "socialism",
    "communist", "communists", "communism", "republican", "republicans", "democrat", "democrats", "libertarian",
    "libertarians", "feminist", "feminists", "political", "politics", "activist", "activists",
    # intimacy
    "bdsm", "roleplay", "roleplays", "role play", "role-play", "roleplaying", "polyamorous", "polyamory",
    "voyeurism", "voyeur", "kink", "kinks", "kinky", "fetish", "fetishes", "poly", "swinger", "swingers",
    "swinging", "bondage", "submissive", "dominant",
    # sexual orientation and gender identity
    "gay", "lesbian", "lesbians", "bisexual", "bisexuals", "bicurious", "bi-curious", "queer", "genderqueer",
    "genderfluid", "trans", "transgender", "transsexual", "asexual", "demisexual", "pansexual", "homosexual",
    "heterosexual", "lgbt", "lgbtq", "lgbtqia", "intersex",
    # everyday words, only where they say who someone is
    "i'm bi", "im bi", "i am bi", "i’m straight", "we're straight", "we are bi", "who's blind", "who is bi",
    "who are straight", "as a bi", "as an straight", "being blind", "i'm a bi", "i am an straight", "bi man",
    "bi men", "straight woman", "straight women", "blind guy", "bi guys", "straight girl", "blind girls",
    "bi lady", "straight ladies", "blind person", "straight people", "bi couple", "legally blind",
    "partially blind", "completely blind", "in recovery", "recovering addict", "recovering addicts",
    "recovering alcoholic", "recovering alcoholics", "addiction recovery", "alcohol recovery", "drug recovery",
    "recovery from addiction", "recovery from alcohol", "recovery from drugs", "in therapy", "in aa",
    "aa meeting", "aa meetings",
]
# Words that were sensitive only because of the key they were saved under, or only to one agent rule
# (a hidden diet), now caught wherever they are written.
SENSITIVE_NOW_TOO = {
    "religion": ["halal", "kosher", "religion", "faith", "beliefs", "belief"],
    "health": ["health", "mental health", "medical"],
    "politics": ["politics", "political", "politically moderate"],
    "income": ["income", "salary", "earnings"],
    "identity": ["sexuality", "orientation", "pronouns", "gender identity", "ethnicity", "caste", "immigration status"],
}
ORDINARY = ("a tennis partner", "jazz and ramen", "a piano teacher", "board games", "a dating coach", "a blind date",
            "easy recovery runs", "straight after work", "a bi-lingual language exchange", "a straight-A maths tutor",
            "a bilingual tutor", "recovery runs on Sundays", "going straight there", "healthy cooking", "cello",
            "night owl", "every week", "middle", "pescatarian")


class SensitiveWords(unittest.TestCase):
    def test_every_word_caught_before_is_still_caught(self):
        for text in SENSITIVE_BEFORE:
            self.assertTrue(mentions_sensitive(text), text)
            self.assertTrue(mentions_sensitive(f"Looking for someone {text} to talk to."), text)
            self.assertTrue(mentions_sensitive(text.upper()), text)

    def test_the_category_words_and_hidden_diets_are_caught_in_their_category(self):
        for category, words in SENSITIVE_NOW_TOO.items():
            for text in words:
                self.assertIn(category, sensitive_categories(text), text)

    def test_other_ways_of_writing_them_are_caught(self):
        for text in ("ＭＵＳＬＩＭ", "Muslím", "TYPE 1 DIABETES", "I’M BI"):
            self.assertTrue(mentions_sensitive(text), text)

    def test_ordinary_words_are_not_sensitive(self):
        for text in ORDINARY:
            self.assertFalse(mentions_sensitive(text), text)

    def test_a_categorys_own_name_counts_in_free_text_only_about_someone(self):
        """"a nutrition coach" is no disclosure; "my health" is. In free text a category's own name counts only
        where it's about someone, or is all the text says; in an open detail's name, value or step, anywhere."""
        for text in ("a health food co-op volunteer", "a medical illustrator", "salary on request",
                     "an earnings call note taker", "an orientation day guide", "a grammar tutor for pronouns"):
            self.assertFalse(mentions_sensitive(text), text)
            self.assertTrue(mentions_sensitive(text, in_detail=True), text)
        for text in ("my health comes first", "their salary must be high", "our annual income", "whose beliefs match",
                     "salary", "high earnings"):
            self.assertTrue(mentions_sensitive(text), text)
        self.assertTrue(any("(health)" in p for p in problems(detail("job", "medical illustrator"))))
        self.assertTrue(any("(income_band)" in p for p in name_problems("salary range")))

    # A category's own name in free text, made about someone by the sentence's structure (never by its topic).
    ABOUT_SOMEONE = {
        # a possessive governs it, through any number of words of its phrase (hyphenated ones too), or a name's
        "someone easy about my irregular freelance earnings": "income",
        "a walking group kind to our ageing parents' health": "health",
        "a lift to Ines's weekly medical check": "health",
        "patient with my long-standing family beliefs": "religion",
        # a hyphenated word whatever its parts, and a degree or negation adverb before another word of the phrase
        "fine with our touch-and-go seasonal earnings": "income",
        "my on-and-off physical health": "health",
        "our so-called family beliefs": "religion",
        "kind about my not exactly lavish income": "income",
        "my never very steady medical history": "health",
        # a first-person form reaches it in the same clause
        "we live on one small pension income": "income",
        "I've had medical leave since spring": "health",
        "I prefer neutral pronouns at meetups": "identity",
        "as a single-income household we budget hard": "income",
        # all the text says, once the punctuation and brackets around it are left out
        "Earnings?": "income", "[medical]": "health", "— pronouns —": "identity", "'Beliefs'": "religion",
        "xe/xem pronouns": "identity", "he/they pronouns": "identity",
    }
    # The same words in their everyday senses, where nothing makes them about anyone.
    EVERYDAY = ("a salary negotiation workshop buddy", "an orientation hike leader", "someone who teaches health economics",
                "I like hiking, and a health club nearby", "I know a coach who offers medical massage",
                "beliefs and doubts reading circle", "my not so secret love of medical dramas", "a so-so health food cafe")

    def test_a_categorys_own_name_counts_where_the_sentence_makes_it_about_someone(self):
        for text, category in self.ABOUT_SOMEONE.items():
            self.assertEqual(sensitive_categories(text), {category}, text)
        for text in self.EVERYDAY:
            self.assertFalse(mentions_sensitive(text), text)
            self.assertTrue(mentions_sensitive(text, in_detail=True), text)  # in an open detail, anywhere

    def test_at_entry_an_open_detail_never_carries_one(self):
        """The Warden reads an open detail's values with the category nouns counting anywhere, so these forms are
        refused there too, with the hidden category to use."""
        from belong.matching.marketboard import Marketboard
        from tests.helpers import fld, listing, person, warden
        w = warden()
        for text, category in self.ABOUT_SOMEONE.items():
            if len(text) > 40:
                continue
            l = listing("pa", "Looking for a walking group in Lyon.", [fld("city", "Lyon"), detail("about", text)])
            result = w.admit(person("pa"), l, Marketboard())
            self.assertFalse(result.accepted, text)
            self.assertIn("listing_schema", result.failed_checks, text)

    def test_reading_them_takes_time_in_proportion_to_the_text(self):
        """Each form reads a text once, however many possessives, first-person forms or marks it repeats, so a long
        hostile value (the Warden reads an open detail's values before refusing one that is too long) can't stall
        it: quadratic reading would take minutes on these, linear takes milliseconds. A noun after them is still
        caught."""
        import time
        for unit, noun in (("i ", "health"), ("we x ", "medical leave"), ("my ", "salary"), ("sam's ", "income"),
                           ("x'my ", "beliefs"), ("as a ", "(orientation)"), ("-", "pronouns"), ("_", "health"),
                           ("'", "salary"), ("my not-", "health"), ("our so-so ", "income"), ("my a-b ", "salary"),
                           ("x-my ", "beliefs"), ("my so ", "earnings"), ("a/", " pronouns")):
            text = unit * (40000 // len(unit))
            start = time.perf_counter()
            self.assertEqual(sensitive_categories(text, in_detail=True), set(), unit)
            self.assertFalse(mentions_sensitive(text), unit)
            self.assertLess(time.perf_counter() - start, 5.0, unit)
            self.assertTrue(mentions_sensitive(text + noun), unit)


if __name__ == "__main__":
    unittest.main()
