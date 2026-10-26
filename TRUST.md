# How Belong. keeps its privacy promises

This document is for anyone who wants to **check** the privacy claims of Belong. rather
than take them on faith. Every promise below links to the code that enforces it and
the test that proves it. If a promise has no test, the promise isn't kept yet.

It covers the **Step 2 prototype** (SPEC.md §8): synthetic people, no secure hardware
yet, but built the way the production service will run. Section 5 lists what is *not* yet
guaranteed.

---

## 1. Verify it yourself (2 minutes)

You need Python 3.10+ and nothing else. No packages, no API keys, no network.

```bash
python3 -m unittest -v          # all tests; any leak fails them
python3 -m belong run           # simulate 300 people; prints the leak report
python3 -m belong serve         # demo, trust page and inspector on localhost:8000
```

`python3 -m belong run` exits non-zero if the leak report finds anything.

With the server running:

- **http://localhost:8000/trust** is the trust page for the public and their agents. It
  verifies the Warden's source and the receipt chain *in your browser*, and runs the
  canary test and attack suite against a sandbox.
- **http://localhost:8000/** is the two-person demo: two phones, with a presenter
  panel in between that logs everything Belong. receives and sends.
- **http://localhost:8000/inspector** is the batch-run inspector. Its people are the live demo's
  first synthetic people, and "Take over" opens one person's phone in the demo, so you can answer as them.

---

## 2. The shape of the system

```text
             PEOPLE                     WARDEN (open source)         PRIVATE MATCHING (open source)
                                                                    (no way out)
  Person ── Personal agent ──▶  Warden: ENTRY check  ──▶  Marketboard (all listings,
            drafts listing       kyc, schema,              shown + hidden)
            (shown / hidden)     sensitive defaults,              │
                                 specific person,                 ▼
                                 rate limit               Matchmaker agent
                                                          reads everything,
  Person ◀── Introduction ◀──  Warden: EXIT check  ◀──  outputs only Proposals
             (shown fields +     builds the output         and fixed-menu FollowUps
              reason from        itself, from shown
              shared shown       fields only, then
              fields)            string-checks it
                                        │
                                        ▼
                                 Receipts {check, result, timestamp, hash}: no content

  Independent audit (leaks.py) re-reads every output sent and fails on any leak.
```

**The separation rule (SPEC §3):** the agent that sees everything (Matchmaker) has
no path out; the agent that controls the exit (Warden) never decides who matches.

### Both agents are open source, and the Matchmaker is still treated as hostile

The **Warden** and the **Matchmaker** are both open source, so anyone can read every check
and exactly how matching works. Each has its own measurement, so a verifier can check each
one separately. The Warden still treats the Matchmaker as **hostile**: the privacy guarantee
never depends on the Matchmaker behaving well, so it holds even if someone runs a modified
Matchmaker.

- The Matchmaker can hand the Warden only the objects in [matching/contract.py](belong/matching/contract.py):
  person ids, yes/no flags, numbers and field *names*. No free text, so there's nowhere to
  smuggle a hidden value.
- The Warden writes every output itself, from shown fields, and string-checks it.
- The trust page's attack suite plays a compromised Matchmaker and checks that every
  attempt is blocked.

The published files are listed in [attest.py](belong/attest.py) (`WARDEN_FILES`,
`MATCHMAKER_FILES`). `test_the_warden_never_depends_on_the_matchmaker` makes sure no Warden
code comes from the Matchmaker, and `test_published_code_only_depends_on_published_code`
that nothing published relies on unpublished code.

---

## 3. The promises, and where to check each one

### Who can get in

| # | Promise | Enforced by | Proved by |
|---|---|---|---|
| 1 | Only KYC-verified people enter. | `Warden.admit` → `kyc_verified` in [warden.py](belong/warden.py) | `test_unverified_person_is_rejected` in [test_entry.py](tests/test_entry.py) |
| 2 | Nobody can search for one specific person (names, handles, emails, phone numbers, links, "my ex", workplace + looks). The rule covers the description and the free-text "looking for" / "what you offer" phrases. | `specific_person_problems` in [warden.py](belong/warden.py) | `test_identifying_intents_are_rejected`, `test_free_text_fields_get_the_specific_person_check_too`; `test_every_ordinary_synthetic_intent_passes` checks for false positives on 2,000 normal intents |
| 3 | Sensitive categories (religion, politics, health, income, kinks) start hidden and are left out until the person turns them on. Showing one needs explicit confirmation. What goes from a person's own words into a shown detail (an activity, a "looking for" / "you are" phrase, a headline, topics, interests, a place) depends on the agent. **The rule-based (offline) agent shows only words it recognises:** the kinds of match, a fixed list of ordinary activities (in its own phrases, such as "a tennis partner" or "a piano teacher"), each detail's fixed choices, subjects, languages, known cities and countries, and an amount of pay written in a fixed form. It never copies anything else from a message into a shown detail, so nothing else the person types, sensitive or not, is shown by it; what it doesn't recognise stays only in the description (never shown), and it asks instead ("What would you like a partner for?" for "a pottery partner"), taking only an activity it knows as the answer. It records a sensitive detail it recognises as that hidden detail and tells the person, and keeps a halal or kosher diet hidden until they show it. **The model agent's shown details are written by the model**, and the person reviews them before saving; its save is refused while a shown detail it wrote contains a word from a list of sensitive words (religion, politics, health, sexual orientation or gender identity, intimacy; everyday words such as "blind", "recovery", "straight" or "bi" count only where they say who someone is, as in "I'm bi" or "in recovery"). That list is a safeguard: other wordings can get past it (limit 4). This is the agents' rule, on the person's side: the Warden doesn't read free text for sensitive words. | `draft_listing`, `draft_from_message`, `KNOWN_ACTIVITIES`, `known_city`, `activity_partner`, `unknown_partner`, `chosen_kinds`, `quick_facts`, `sensitive_details`, `mentions_sensitive` in [agent.py](belong/agent.py); `DemoSession._rules_reply` in [demo.py](belong/demo.py); `build_draft` in [agent_llm.py](belong/agent_llm.py); `sensitive_defaults` check | `test_sensitive_field_shown_without_confirmation_is_rejected`, `..._with_confirmation_is_accepted`; [test_sensitive_words.py](tests/test_sensitive_words.py): `test_no_unknown_word_ever_reaches_a_shown_detail`, `test_a_partner_for_an_activity_it_doesnt_know_is_asked_about`, `test_the_answer_must_be_an_activity_it_knows`, `test_romantic_words_answer_which_activity_with_a_date`, `test_a_diet_that_says_something_about_religion_is_kept_hidden`, `test_every_word_it_recognises_is_ordinary`, `test_a_sensitive_word_never_reaches_a_shown_detail`, `test_a_partner_named_only_by_a_sensitive_word_is_asked_about`, `test_a_sensitive_word_in_an_answer_is_never_the_activity`, `test_in_the_demo_a_both_answer_with_a_sensitive_fact`, `test_quick_facts_never_put_a_sensitive_word_in_a_shown_detail`, `test_ordinary_words_are_not_sensitive`, `test_a_shown_detail_with_a_sensitive_word_is_not_saved`, `test_ordinary_words_are_saved`; `test_plain_facts_are_picked_up` ([test_agent_llm.py](tests/test_agent_llm.py)) |
| 4 | A listing can't relabel a sensitive field as non-sensitive, add unknown fields, claim a different age than KYC reported, or give a travel distance or a way to meet other than the fixed choices (same city, up to 30 km, up to 100 km; in person, online, in person or online). What a personal agent understood is held to fixed choices too (#33): the kind of match (dating, climbing, running, language, cofounder, tutoring, roommate, other), who a dating listing would like to meet (woman, man, nonbinary) and their ages in years (exactly "30-39": ASCII digits, no leading zero, from 18 to 99, so no listing can ask for anyone under 18). A personal agent reads other wording into a fixed choice first ("not online" means in person; "33 to 42" means 33-42, and "030-039" or "３０-３９" means 30-39). | `_schema_problems` in [warden.py](belong/warden.py); `meeting_format`, `age_span`, `age_span_choice` in [schema.py](belong/schema.py) | `test_mislabelled_sensitivity_is_rejected`, `test_unknown_field_is_rejected`, `test_age_band_must_match_kyc`, `test_the_warden_accepts_only_the_fixed_travel_choices`, `test_the_warden_accepts_only_the_fixed_formats`, `test_a_format_in_other_words_is_put_into_a_fixed_choice_the_warden_accepts`, `test_only_the_fixed_choices_get_in`, `test_other_ways_of_writing_ages_are_put_into_the_fixed_form_first` ([test_understood.py](tests/test_understood.py)); attack suite: "Give a kind of match outside the fixed choices", "Ask to meet people under 18", "Write an age range in other digits or with a leading zero", "Ask for a gender outside the fixed choices" |
| 5 | Listing updates are rate-limited (5 per day). | `_within_update_limit` | `test_listing_update_rate_limit` |

### What happens during matching

| # | Promise | Enforced by | Proved by |
|---|---|---|---|
| 6 | **Both must be looking.** A pair is proposed only if each person's intent is met by the other. The one exception is a close match (#27), and only for someone who asked for them. | `intent_met` (called both ways) in [matcher.py](belong/matching/matcher.py); Warden's `proposal_mutual` check | `test_every_proposal_is_mutual`, `test_one_sided_proposal_is_rejected` |
| 7 | **Never the only reason.** Whether a pair qualifies is decided from shown fields alone, including how they can meet (format) and how far each will travel: someone who hides their travel distance is treated as staying in their own city. Hidden fields can only raise the score of a pair that already qualifies. | Qualifying code receives a `ShownView`, which can't return hidden fields ([narrow.py](belong/matching/narrow.py)). Hidden values are read in exactly one function, `hidden_overlap`. | `test_removing_hidden_fields_never_removes_a_match`: delete every hidden field and every proposal still qualifies. Also `test_hidden_fields_only_raise_scores_and_never_justify_a_match`, `test_a_hidden_travel_distance_is_never_used`, audit R1 |
| 8 | **Only counts in a crowd.** A hidden value can raise a score only if at least *k* = 20 people share it. | `hidden_overlap` + `Marketboard.value_counts` | `test_crowd_threshold_blocks_rare_hidden_values`, audit R2 |
| 9 | Hidden fields don't influence who gets shortlisted. Shortlists are found in a fixed order, all from shown details: how two people can meet (online only if both allow it), where (cities within both people's travel distance, in the country both people show), then what (a word in common between what one looks for and what the other is, shows or lists as topics). | `shortlists`, `how_they_meet`, `within_reach` read only the `ShownView`; the word index (`index_words` in [matcher.py](belong/matching/matcher.py)) reads only shown phrases, topics and details; text embeddings use intent + shown fields only (`ShownView.profile_text`) | Covered by #7: the shown-only board yields a superset of matches. `test_the_index_keeps_every_pair_a_decision_could_accept` shows the word index never drops a pair the decisions could accept ([tools/compare_narrowing.py](tools/compare_narrowing.py) runs it on 3,000 people); `test_same_name_cities_in_different_countries_are_not_in_reach` |
| 10 | The Matchmaker has no way out: no network, files, logging, printing, or subprocesses. | Matching code may import only `re`, `math`, `collections`, `dataclasses`, `typing`, `unicodedata` and `belong.schema` ([matching/\_\_init\_\_.py](belong/matching/__init__.py)) | [test_isolation.py](tests/test_isolation.py): import and call scan of every matching file, plus a full run with sockets disabled |

### What can come out

| # | Promise | Enforced by | Proved by |
|---|---|---|---|
| 11 | **Only the Warden creates outputs.** Introductions, follow-ups and reveals can't be constructed anywhere else. | Constructors require a private seal held only in [warden.py](belong/warden.py) | `test_output_objects_refuse_construction_without_the_seal`, `test_no_module_but_the_warden_touches_the_seal` |
| 12 | An introduction contains only the other person's **shown** fields, and stays current: when either person updates their listing, the Warden reissues it from their current shown fields (a new name appears; anything since hidden disappears), with the same checks. | The Warden builds views from `listing.shown()` itself, then re-checks (`fields_shown`); `refresh_introduction` | `test_a_good_proposal_becomes_an_introduction_without_hidden_values`, `test_a_reissued_introduction_shows_current_details_and_nothing_since_hidden`, `test_the_other_person_sees_your_new_name`, audit A1 |
| 13 | **Never in the reason.** Reason text is written by the Warden from fixed templates, using only fields *both* people show. The Matchmaker supplies field names, never text. Where people are is worded from both shown cities and, for two cities, both people's shown countries (never a hidden one), so it is true whatever the Matchmaker cites: "You're both in Osaka" only for the same city in the same shown country, "You're near one another: Osaka and Kyoto" only for two cities in the same shown country within 100 km of each other (approximate city centres in [schema.py](belong/schema.py), each known only in its own country), otherwise just where each one is: two cities called Cambridge in different countries are not "both in Cambridge", and if either person doesn't show a country, nothing more is claimed. How they meet is worded from both formats ("You can both meet online"). The fixed wording never uses a word that is also a fixed choice of some field ("other" is a kind of match), so the check that a reason uses only its reason fields can't mistake the template for a value. | `reason_text` (`_where_text`, `_format_text`), `Warden._checked_views` (passes the countries only when both people show one), `reason_fields_shown`, `reason_refs_only_reason_fields` | `test_hidden_field_as_a_reason_is_rejected`, `test_proposal_with_no_shown_reason_is_rejected`, `test_reason_text_for_nearby_and_same_cities`, `test_a_compromised_matchmaker_cannot_make_far_cities_near`, `test_same_name_cities_are_worded_by_the_shown_countries`, `test_the_warden_words_cities_only_from_countries_both_people_show`, `test_an_introduction_between_nearby_cities_passes_every_check`, `test_the_wardens_wording_never_uses_a_fixed_choice_as_a_word`, `test_two_open_ended_listings_in_nearby_cities_are_introduced`; attack suite: "Cite where two people are to make far-apart cities sound close" (`test_the_attack_suite_catches_a_reason_that_isnt_true`), "Cite two cities with the same name in different countries to make them sound close", "Word where two people are from a country one of them hides" (`test_the_attack_suite_catches_same_name_cities_and_a_hidden_country`) |
| 14 | No hidden value of either person appears anywhere in an introduction (string check). | `no_hidden_values` → `find_forbidden` | `test_hidden_values_stay_out_even_when_one_side_shows_the_same_key`, audit A2 |
| 15 | Follow-up questions come only from the fixed menu, and the Warden attaches the wording. | `FOLLOWUP_MENU` in [schema.py](belong/schema.py); `exit_followup` | Audit A3 |
| 16 | **Mutual reveal.** A hidden detail is revealed only if both people share it, are connected (both said yes), and both opted in. Each person is asked about *all* their hidden details, so the question itself reveals nothing. | `exit_reveal`; `reveal_consents` in [agent.py](belong/agent.py) | `test_shared_detail_revealed_only_when_both_opt_in`, `test_nothing_revealed_before_both_say_yes`, audit A4 |
| 17 | **No listing is introduced more than 15 times a day** (any 24 hours), so nobody is flooded and a fake account can only reach a few people. The Matchmaker also proposes at most 3 per listing in one run, so they arrive a few at a time. The Warden stamps each introduction with when it was sent, and the independent audit re-counts every listing's introductions in each 24 hours (one with no send time counts as the same day). | `approach_limit` check (counted per listing, last 24 hours) in [warden.py](belong/warden.py); `Introduction.sent_at`; audit R3 in [leaks.py](belong/leaks.py) | `test_approach_limit_is_enforced`, `test_the_approach_limit_is_per_day`, `test_an_introduction_carries_when_it_was_sent`, `test_too_many_introductions_in_one_day_are_caught`; attack suite: "Spam one person with more introductions than the daily limit" |
| 18 | **Receipts carry no content:** exactly `{seq, check, result, timestamp, commitment, hash}`. The commitment is a keyed HMAC of what was checked: auditors holding the key can confirm it, and nobody can read content out of it. | `Warden._receipt` | `test_receipts_have_fixed_fields_and_no_content`, audit A5 |
| 19 | **Receipts can't be quietly deleted, reordered or edited, even by Belong.** Each receipt's hash covers the previous one, and anyone can re-compute the chain without keys. | `chain_hash`, `verify_chain` in [warden.py](belong/warden.py); re-implemented independently in [leaks.py](belong/leaks.py) and in the trust page's JavaScript | `test_anyone_can_verify_the_receipt_chain`, `test_deleting_or_editing_a_receipt_breaks_the_chain`, audit A6 |
| 20 | A proposal naming someone who hasn't signed up (or pairing a person with themselves) is refused cleanly. | `proposal_known_people` check | Attack suite: "Propose a person who hasn't signed up" |
| 20a | **Never matched with yourself.** A person can have several listings (say, a tutor and a date), each matched on its own. Two listings by the same person are never paired, and a listing can only be added by its owner. | Matcher skips same-owner pairs; Warden's `proposal_distinct_people` check at exit and owner check at entry | `test_never_introduced_to_your_own_other_listing`, `test_both_listings_are_matched_on_their_own`, attack suite: "Introduce someone to their own other listing", audit R4 |
| 21 | **The published source of both agents is the code that was measured** (one measurement per agent). | [attest.py](belong/attest.py) `measurements()` | `test_published_source_hashes_to_the_measurement`; the trust page re-hashes every file in the browser |
| 22 | The trust API serves the published files of both agents, and nothing else. | `open_source_text` | `test_only_published_files_are_served` |
| 23 | **A phone only receives what its person may know.** Nobody is told who said no; a person who opted into a reveal isn't told whether the other side shares the detail or simply declined. The same holds for a synthetic person the presenter takes over from the inspector: their phone gets their own listing (their hidden details appear only there, as on any person's own profile) and only what the Warden sent them; every other phone still sees them only through the Warden's introductions. Once taken over, nothing is decided for them (no simulated yes or no, and no simulated choice of details to compare) until they're handed back, and the server accepts only phones that exist: A, B and people taken over (the connector: A and B). | `DemoSession.phone_state`, `DemoSession.take_over` and `_slot_of` (which ends the simulated answers) in [demo.py](belong/demo.py); the slot check in `_Handler.do_POST` in [server.py](belong/server.py) | [test_demo.py](tests/test_demo.py): `test_phones_never_receive_the_other_persons_hidden_details`, `test_a_no_is_never_attributed`; [test_takeover.py](tests/test_takeover.py): `test_other_phones_never_see_their_hidden_details`, `test_the_phone_shows_the_persons_own_listing`, `test_their_answer_comes_from_the_phone`, `test_a_close_match_to_them_waits_for_their_own_answer`, `test_take_over_answers_with_the_phone_and_only_real_phones_are_accepted` |
| 24 | **Close matches and questions are off until you say yes.** A new listing starts with both off. If a live listing has no match yet, Belong. asks once, on the listing itself: "No matches yet. Want to include close matches?" A yes turns on close matches (one-way: the other person isn't told) and, with them, follow-up questions that could find a match; the card says so before the person answers. A no keeps the listing to exact matches, and the question is never asked again. Either way the person can change it later with the gear on the listing (one switch, "Close matches", for both). | `Listing.notify_close` / `notify_questions` in [schema.py](belong/schema.py) (default off); `DemoSession.new_listing` / `answer_followup` / `set_prefs` in [demo.py](belong/demo.py) | `test_choices_are_on_the_listing_and_off_by_default`, `test_no_match_yet_asks_once_to_include_close_matches`, `test_a_no_keeps_exact_matches_only_and_isnt_asked_again` |
| 25 | **Questions only for people who opted in.** The Warden refuses any follow-up question to a listing that didn't opt in, with one exception: the opt-in question itself. "Want to include close matches?" is a fixed sentence about nobody, and the Warden sends it only while the listing's close matches are off, only if the listing has no introduction yet, and only once per listing, ever. The independent audit re-checks "once" and "close matches were off"; "no introduction yet" is checked by the Warden alone, and its receipt is in the public log. | Warden's `followup_opted_in` check; for the opt-in question, `_exit_broaden` (`followup_no_subject`, `broaden_still_off`, `broaden_no_match_yet`, `broaden_once`) in [warden.py](belong/warden.py) | `test_no_questions_for_someone_who_opted_out`, `test_no_question_without_both_opt_ins`, [test_optins.py](tests/test_optins.py) `IncludeCloseMatches`; attack suite: "Ask a question of someone who opted out of questions", "Ask "Want to include close matches?" twice", "... of someone who already said yes", "... of someone who already has a match", "Attach another person's listing to "Want to include close matches?""; audit R5 |
| 26 | **"Could you help with that too?" never says who's asking.** If someone who opted in to close matches is looking for something that a person they could meet (nearby, or online when both are happy online) nearly offers (an air conditioner technician, from a repairman; a beginner ice hockey coach, from a figure skating coach), the person who opted in to questions may be asked. "Nearly" means roles that line up and a shared word or stem, either in the two phrases or in the broader *topics* each person's agent wrote for their listing (a shown field, checked at entry like the phrases). There is no built-in list of activities, and umbrella words like "sports" or "lessons" never count (`nearly` in [matcher.py](belong/matching/matcher.py)). The Matchmaker only names the listing; the Warden writes the question itself, from a fixed template ("Someone is … Would that work for you?") and that listing's *shown* headline in plain words (or its "looking for" phrase if it has none), string-checked for hidden values. No name or id is sent. One person is asked at a time: whoever is looking for something (from their headline) first, and the other only if they say no. A yes adds the other person's phrases to theirs and takes on their shown price, their own agent rewords the headline to match, and the whole update goes live (through the Warden) before matching sees it, so nobody is shown the old terms; a match follows without anyone else agreeing to anything; every question is asked only once. The audit checks each question against the version of the quoted listing that was live when it was asked; that phrase is carried by the Warden's output, checked the same way, and never shown in the question. | `exit_followup` (`also_offer_requester_opted_in`, `also_offer_phrase_shown`, `no_hidden_values`, `followup_no_subject`); `FOLLOWUP_MENU["also_offer"]` | [test_optins.py](tests/test_optins.py): `test_the_matchmaker_asks_the_repairman_about_air_conditioners`, `test_saying_yes_makes_it_a_match`, `test_the_figure_skater_is_asked_about_ice_hockey_and_yes_makes_a_match`, `test_nearly`; `test_the_repairman_is_asked_and_saying_yes_makes_a_match`; attack suite: "Quote someone's request in a question when they didn't opt in", "Attach another person's listing to an ordinary question"; audit R5 |
| 27 | **Close matches are one-way.** A person who opted in may hear about someone who fully meets that person's own criteria but is only nearly what they asked for. The other person isn't told: the Warden refuses their answer until the recipient has said yes, and their phone shows nothing until then. Close matches are ranked on shown details, never a hidden one (fair ranking, #32, may reorder them), and never take a slot from an exact match. | Warden's `proposal_mutual` (other side must be met), `close_match_opted_in`, and `record_decision` (one-way order); `Introduction.one_way_to`; `OfflineMatcher.match` ranking; `_listing_state` in [demo.py](belong/demo.py) | `test_a_close_match_goes_one_way`, `test_only_for_someone_who_opted_in`, `test_close_matches_go_one_way_to_people_who_opted_in`, attack suite: "Send a close match to someone who didn't opt in", "Show a close match to the other person before the recipient says yes"; audit R5 |
| 28 | **Messages only between two people who both said yes**, delivered only to them, never read by matching, and never logged: the presenter panel shows that a message was sent, not what it says. Reactions are one of a fixed few, on that same conversation only. If someone uses Belong. through their own AI assistant (the connector), their match is told on their side of the chat, because what they write passes through that assistant's provider. | `DemoSession.send_message` (requires the Warden's `is_connected`); threads are kept outside the Marketboard | `test_no_messages_until_both_say_yes`, `test_messages_reach_only_the_two_people_and_never_matching`, `test_only_a_few_reactions_and_only_between_connected_people` |
| 29 | **A paused listing isn't matched.** The person can switch a listing off (the gear on it): it gets no new introductions or questions until they switch it back on. Existing matches and chats stay. | `Listing.active`; `OfflineMatcher.match` skips paused listings; Warden's `listings_active` and `listing_active` checks | `test_a_paused_listing_gets_no_new_matches`; attack suite: "Introduce someone to a paused listing", "Ask a paused listing a question" |
| 30 | **A deleted listing is never matched again.** Deleting takes it off the Marketboard through the Warden, which checks the person owns it and writes a receipt. | `Warden.withdraw` (`listing_withdrawn`), `Marketboard.withdraw`; `DemoSession.delete_listing` | `test_a_deleted_listing_is_never_matched_again`; attack suite: "Introduce someone to a deleted listing" |
| 31 | **A simulated person's replies are written only from what they'd know.** In the demo, an AI model plays a synthetic person you're connected with: it says hello when you match, then answers only your messages (never a chat between two synthetic people), and may react with one of the same fixed few reactions. It is given that person's own shown details and request, what their introduction showed them about you (your shown details and the Warden's reason) and the conversation, never anyone's hidden details. At most 30 replies a conversation; a synthetic person someone has taken over doesn't auto-reply. | [persona_chat.py](belong/persona_chat.py) `context`, `reply`; `DemoSession._persona_turn` / `_persona_reply` in [demo.py](belong/demo.py) | [test_persona_chat.py](tests/test_persona_chat.py): `test_the_model_never_sees_anyones_hidden_details`, `test_two_real_people_get_no_simulated_replies`, `test_replies_stop_after_a_while`, `test_they_say_hello_when_you_match` |
| 32 | **Fair ranking only changes the order.** Among pairs that already passed every check, a listing that had introductions in the last 24 hours ranks a little lower and one updated in the last day a little higher, so new and quiet listings get seen. It uses only how many introductions each listing had (the demo counts the Warden's own send times) and when listings were updated, never a hidden detail, and it can't let a pair skip the threshold or the reason checks. When only some listings are matched (the demo's phones), their own count is left out: it is the same for all their candidates, so it would only push two busy phones down for each other. | `OfflineMatcher.match` (`load`, `recently_updated`, applied after the checks; the focus listings' own load left out) in [matcher.py](belong/matching/matcher.py); `DemoSession._introductions_today` | `test_fair_ranking_only_reorders`, `test_a_busy_listing_gives_way`, `test_a_listing_updated_in_the_last_day_comes_first`, `test_a_focused_run_leaves_out_the_focus_listings_own_load`, `test_the_demo_ranks_with_its_own_introductions_today`, `test_two_busy_phones_still_meet_on_a_big_board` |
| 33 | **What your agent understood is on your profile, checked at entry, and is what matching uses.** Your personal agent records the kind of match you want (what you want to meet someone for, never a hobby you mention) and, for dating, who you'd like to meet and their ages in years, as details on your listing: shown by default, so you can see and change exactly what matching will use ("Looking for: a woman, 30–49"). The Warden accepts only the fixed choices (#4). Matching uses these shown details first, and narrowing groups listings by the same kind; it reads your description only for what they don't say. A hidden one is ignored like any hidden detail: it never decides a match, and matching reads the description as if it weren't there. A dating listing with no known gender wanted is never matched, and the phone won't save one; who and what ages are recorded only on a dating listing. When your words could mean more than one kind of match (meeting a woman, and climbing; or just "a partner"), the agent should ask one short question before saving, and "both" makes two listings, each with its own kind. How well that works differs by agent: the model agent is *instructed* to ask (its instructions say so; nothing checks that it does, see limit 5). The rule-based agent asks when its own readers see two kinds or a partner for an unnamed activity, or find a kind only in something you say you enjoy, and then records no kind until you answer; anything its readers miss, it can still get wrong. It reads an activity it only sees mentioned ("a date, I just like climbing"), or one that describes the person you'd like to meet ("a girlfriend who climbs", "a date with a climber"), as no choice; a hobby ends where you say you hope to meet someone ("I enjoy reading and hope to meet a girlfriend" is dating); "no one under 30 or over 45" is 30 to 45; and it never turns a bare number into an age range, and nor does matching, so its note, your phone and matching say the same ages. | `read_listing` in [intent.py](belong/matching/intent.py) and `understood` in [matcher.py](belong/matching/matcher.py) (shown fields only, via `ShownView`); `_schema_problems` in [warden.py](belong/warden.py); `build_draft`, `turn_drafts` and `SYSTEM_PROMPT` in [agent_llm.py](belong/agent_llm.py); `draft_from_message`, `kind_question`, `chosen_kinds`, `answer_details`, `wanted_ages` in [agent.py](belong/agent.py); `DemoSession._rules_reply` and `_listing_from_json` in [demo.py](belong/demo.py); `canSave` in [demo.html](belong/web/demo.html) | [test_understood.py](tests/test_understood.py): `test_a_dating_listing_that_mentions_an_activity_meets_only_the_people_it_wants`, `test_the_shown_kind_decides_whatever_the_description_says`, `test_shown_wants_beat_the_description`, `test_narrowing_groups_by_the_same_kind`, `test_a_hidden_kind_or_wants_is_as_if_it_werent_there`, `test_removing_hidden_understanding_never_removes_a_match`, `test_a_dating_listing_with_no_known_wanted_gender_is_never_matched`, `test_the_tool_kind_becomes_the_kind_field`, `test_wants_are_validated_strictly`, `test_offline_drafts_carry_kind_and_wants`, `test_the_answer_decides_and_both_makes_two_listings`, `test_the_model_asks_before_saving_and_both_saves_two_listings`, `test_the_visiting_man_meets_only_women_of_the_right_ages`, `test_a_hobby_ends_where_the_request_starts`, `test_a_kind_only_from_a_hobby_is_asked_never_recorded`, `test_an_activity_only_mentioned_is_not_chosen`, `test_they_choose_the_date`, `test_the_first_message_is_kept`, `test_what_the_answer_says_besides_the_choice_is_kept`, `test_the_activity_reader_is_general`, `test_the_offline_agent_asks_instead_of_guessing_from_a_number`, `test_no_question_and_dating`, `test_an_answer_that_describes_the_person_chooses_the_date`, `test_in_the_demo_one_listing`, `test_a_hobby_ends_where_they_hope_to_meet_someone`, `test_both_bounds_under_one_no`, `test_a_bare_number_reads_the_same_everywhere`, `test_who_and_what_ages_never_go_live_on_a_listing_that_isnt_dating`, `test_the_phone_rebuilds_the_rows_and_saves_only_what_can_match` |
| 34 | **A detail you hid stays hidden when your agent updates your listing.** Your choices of shown or hidden are remembered for each listing: what you saved, and what you set in the editor (the phone sends those before each message to your agent, and doesn't send the message if that fails). Every new draft keeps them, from either agent and from the instant profile: a detail the draft leaves out, sends with no value, or brings back with its default keeps the shown or hidden it had, and where your choice and the previous draft differ, hidden wins. Only an agent that gives a detail both a value and a visibility of its own changes it: the rule-based agent never does (except to keep a halal or kosher diet hidden, #3), and the model agent does for each detail it sends, after being told which details you chose to keep hidden, so it *can* still show one; you review the draft before saving. The model agent's save is merged with your profile before it is checked and before the agent is told what a match will see, so what it is told is the listing you will review. A sensitive detail is never made shown this way: showing one needs your own confirmation (#3). | `fill_gaps`, `keep_visibility`, `carry_over` in [agent.py](belong/agent.py); `build_draft` (`listed`, `current`), `builds_on` and `profile_so_far` in [agent_llm.py](belong/agent_llm.py) and [agent_openai.py](belong/agent_openai.py); `_Listing.choices`, `DemoSession._sync_draft`, `set_visibility`, `prefill`, `agent_message` and `_rules_draft` in [demo.py](belong/demo.py); `visibilityChanges` and `send` in [demo.html](belong/web/demo.html) | [test_understood.py](tests/test_understood.py): `test_a_model_save_that_leaves_a_hidden_detail_out_keeps_it_hidden`, `test_the_rule_based_agent_keeps_a_hidden_detail_hidden_too`, `test_keep_visibility_never_shows_a_sensitive_detail`, `test_the_rule_based_agent_keeps_them_whatever_comes_between`, `test_the_instant_profile_keeps_them_too`, `test_a_detail_sent_with_no_value_chooses_nothing`, `test_the_checks_and_what_the_agent_is_told_see_the_listing_as_it_will_be`, `test_a_model_save_that_leaves_them_out_or_empty_keeps_them_hidden`, `test_a_choice_made_in_the_editor_is_kept_before_it_is_saved`, `test_the_phone_sends_editor_choices_before_a_message` |

### Testing Belong. with your own agent

The trust page and its API (in [probe.py](belong/probe.py)) let anyone, or their agent,
test a sandbox that runs the same Warden and Matchmaker:

- **Canary test** (`POST /api/probe/canary`): plants unique secret words in hidden
  details and private descriptions, runs a full week, then searches every byte that
  was sent out. Canaries never consent to a reveal, so any hit is a leak. Agents can
  bring their own listings and secrets.
  `test_canary_test_catches_a_sabotaged_warden` proves the test detects a real leak.
- **Attack suite** (`POST /api/probe/attacks`): 42 attempts to get bad data in or
  private data out. Each result names the rule that stopped it. Agents can add their
  own intents.

### The independent audit

[leaks.py](belong/leaks.py) re-reads every output that was actually sent out, as
plain data, **without reusing the Warden's code**, so a bug in the Warden can't hide
itself. It watches every hidden value on the Marketboard and fails if any reaches
someone who shouldn't see it.

**The tests check the auditor as well** ([test_privacy.py](tests/test_privacy.py),
`LeakReportCatchesPlantedLeaks`): each test plants a leak (a hidden value in a
reason, a hidden field in a view, content in a receipt, a reveal without consent, a
hidden field used below the crowd threshold) and confirms the audit catches it.

During the build we also sabotaged the Warden on purpose by disabling its string
checks and injecting hidden values into reasons. The audit reported 428 findings.
Adding `import logging` to a matching file failed the isolation test. Both guards work.

---

## 4. Who can see what

| Component | Sees | Can output |
|---|---|---|
| Person | Their own listings; introductions addressed to them; reveals they agreed to | Listing, yes/no, reveal opt-ins |
| Personal agent | Its own person's conversation and data only. In the demo it is powered by a hosted model (OpenAI or Claude), so that person's chat is sent to that provider's API (synthetic or test data only). Each person has a separate conversation (`test_each_phone_has_its_own_private_conversation`). | A drafted listing, checked before it's saved (`build_draft` in [agent_llm.py](belong/agent_llm.py)); the person reviews it, and the Warden checks it again at entry |
| KYC provider | ID documents | `verified`, `age_band` only |
| Warden | Listings at entry; proposals and the board at exit (it must know the hidden values to string-check for them) | Accept/reject, sealed outputs, receipts. It never ranks or picks pairs. |
| Matchmaker | Everything on the Marketboard | Proposals and fixed-menu follow-up IDs, only to the Warden |
| Marketboard | n/a (it stores listings; nothing on it acts) | Nothing |

---

## 5. What is NOT guaranteed yet (honest limits)

These need to be true before real people use Belong. They are deliberately visible
here so nobody mistakes the prototype for the finished service.

1. **No secure hardware, so "not even us" is not yet true.** Matching is a Python
   package whose isolation is enforced by tests, not by hardware, and it runs on a
   machine its operators control. The strongest promise needs all of these (Step 3):
   - matching inside a hardware enclave (Nitro Enclaves / Confidential Space /
     confidential VMs), with no debug or operator access;
   - **hardware-signed** attestation (today's document is unsigned and says so);
   - listings encrypted on the person's device to the enclave's attested key, so our
     servers only ever relay ciphertext;
   - decryption and audit keys released only to attested code;
   - reproducible builds, so anyone can rebuild the Warden and get the same measurement;
   - an independent audit of the Warden's rules.
   The trust page's roadmap table tracks each of these.
2. **The string check catches exact values, not paraphrases.** If a reason said
   "you're both spiritual" instead of "Buddhist", the string check wouldn't catch it.
   This can't happen today because the Warden writes reasons from fixed templates. If
   an LLM ever writes reason text, it needs an additional semantic check, and
   templates should remain the default.
3. **Inference from who you're matched with.** Hidden fields can raise a score, which
   can change *which* of several qualifying people you're introduced to. The crowd
   rule (k = 20) limits what this reveals, but doesn't make it zero. The value of *k*
   is an open question (SPEC §10); 20 is a placeholder.
4. **The intent text itself is read during matching.** It is never sent to anyone
   (introductions don't include it), but a person could type something sensitive into
   it. The personal agent spots common sensitive words (religion, health, politics,
   intimacy, sexual orientation and gender identity), adds the ones that are a sensitive
   detail's choice as hidden details and warns the person; that list can't catch
   everything. It matters most for the model agent: what it writes into shown details is
   checked only against that list (and reviewed by the person before saving), so other
   wordings can get through. The rule-based agent doesn't rely on the list for that: it
   writes only words it recognises into anything shown (#3).
5. **Match-quality numbers are optimistic.** The rule-based intent reader was written
   alongside the synthetic phrasings, so its 100% precision proves the pipeline
   works, not that it understands language. Synthetic listings now record what their
   agent understood (#33), as a model-powered agent's listings do, so the batch run
   mostly tests matching on those details; the keyword reader is only the fallback for
   what a listing doesn't record. How well a model-powered agent understands people is
   tested only with scripted stand-ins, never a live model. That it asks when the kind of
   match is unclear is an instruction in its prompt, not something the code checks: a
   model can still save a guess. The rule-based agent asks only when its own keyword
   readers notice a second reading, so wording they don't know can still be misread. The
   real test is real phrasings it has never seen.
6. **Receipt key management** is a placeholder: the HMAC key is derived from the run
   seed. In production it belongs to independent auditors.
7. **Rate and approach limits** are simple in-memory counters, reset when the server restarts. 15 a day is
   the user's choice for the prototype; whether it should differ by kind of listing (dating vs selling) is open.
8. **The Warden's rules haven't been audited by an outside party** (planned for Step 3).
9. **The Warden checks that reasons are private, not always that they're true.** Where people
   are and how they meet are now worded from both people's shown values (and the distance
   between their cities), so those parts of a reason stay true even if a compromised Matchmaker
   cites them for two people far apart (#13). Other cited fields are only checked for privacy: a
   compromised Matchmaker could cite "schedule" for two people who aren't free at the same time,
   and the reason would be wrong, though nothing would leak. Next step: check that every cited
   reason field actually agrees. The city distances are approximate (city centres, a short fixed
   list, each known only in its own country). A distance is used only when both people show the same
   country, so any other city, or a city shown without a country, is only ever "near" a city of the
   same name.
10. **Receipts were extended beyond SPEC §7** (which lists `{check, result, timestamp,
    hash}`) with `seq` and `commitment`, to make the log publicly verifiable. Neither
    field contains content.
11. **The sandbox is not the live service.** Probes test a fresh copy built from the
    same code. In production, attestation must show the sandbox and the live service run
    the same measured image.
12. **The demo's personal agent uses a hosted model.** A person's chat with their own
    agent goes to the OpenAI or Claude API. That's the person's agent, not matching, and nothing
    the agent writes skips the person's review or the Warden. Real people need a
    decision here: an agent of their choice that Belong. has certified, or one that Belong. runs under
    its own privacy terms.
13. **Messages are held by the demo server in memory.** Matching never reads them, but the
    service could. Before real people use it, messages need end-to-end encryption between the
    two phones, so only the two of them can read them.
16. **Through the connector, messages pass through the person's AI assistant.** When someone uses
    Belong. inside their own assistant, their conversations with matches are shown there, so the
    assistant's provider handles both sides' messages. The match is told, but can't opt out of
    it in this prototype. A per-person choice is a possible next step.
15. **A deleted listing is kept aside in this prototype.** It's off the board and never matched,
    but a copy stays so the independent audit can still check what was already sent. A real
    service needs a deletion policy: what is erased, when, and what an auditor may keep.
14. **A close match or an "also offer" question tells someone a little.** A close match shows the
    recipient the other person's shown details before that person has chosen them, which is why
    it needs the other person's criteria to be fully met and is one-way. An "also offer" question
    tells someone that a person somewhere is looking for a thing, without who or where; it only
    quotes a phrase that person chose to show.
17. **The hosted demo is still a demo.** Run online (README, "Run it online"), it is one shared
    demo session for everyone who has its password: anyone using it at the same time sees the same
    two phones, and what they type stays in the server's memory until it restarts. It runs on Google
    Cloud, so Google, and anyone who can manage that project, could in principle read that memory.
    The password keeps strangers out (and off the AI key): it's asked for on its own sign-in page and
    sent over HTTPS; the browser then keeps a cookie that proves it without containing it (30 days, or
    until the password changes), guessing pauses after 20 wrong tries a minute, and the hosted demo
    refuses to start without a password (`_authorized`, `_login` in [server.py](belong/server.py);
    `tests/test_server.py`). On a VM the demo still listens only on the VM itself, behind Caddy;
    run locally, only on your own machine. Use made-up details only.
18. **What you write to a simulated person goes to the AI provider** (OpenAI or Anthropic), because a
    model writes their replies. Only in the demo, and only to synthetic people: messages between two real
    people are never sent to a model.
19. **Taking over is a presenter's tool.** The inspector shows every synthetic person's hidden details,
    and taking one over lets whoever runs the demo act as them and see their hidden details on their
    phone. A real service would never let anyone act as someone else; it's there only because these
    people are synthetic. Their phone is still held to promise 23, so nothing about them reaches
    anyone else except through the Warden.
20. **Who you'd like to meet is shown to your matches by default** (#33), like your gender. A match
    already implies most of it (you're only introduced to people you'd like to meet who'd like to
    meet you), but not all: someone open to anyone shows that to a match who only asked for one
    gender. You can hide it; matching then reads your description instead, so hiding it costs
    nothing as long as your description says who you'd like to meet. Whether it should start hidden
    for people open to more than one gender is open.

---

## 6. Glossary

- **Intent:** what the person is looking for, in their own words.
- **Listing:** the person's details, each marked *shown* (others may see it) or *hidden*
  (used only for matching, never shown).
- **Exactness:** 1 = exactly what I said; 0 = cast a wide net.
- **Proposal:** the Matchmaker's internal suggestion. It is never sent to anyone as-is.
- **Introduction:** the only output about a match: shown fields plus a reason.
- **Reason field:** a field both people show that connects them (same city, shared
  interest, etc.).
- **k (crowd threshold):** a hidden value counts only if at least k people share it.
- **Receipt:** proof that a check ran and passed or failed, with no content.
