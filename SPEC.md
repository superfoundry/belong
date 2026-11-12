# Belong — prototype spec (v0.1)

Belong connects two people (dating or functional needs) through AI agents and a
**locked room**. The product promise is privacy between users: nobody can see your profile,
find you or learn anything about you until you're a two-way match, and private details can
improve a match without ever being shown to another person. Belong itself can access the data
people give it, to run and improve the service (TRUST.md, "What Belong can see"); that is stated
plainly, never a selling point. Both agents are open source so anyone can check the promise.

This file is the handoff from the design canvas to the build. Put it in the repo
root (and/or copy the key rules into `CLAUDE.md`) so every coding session starts
from the same understanding.

---

## 1. Priorities (from Jad, in order)

1. **Privacy between users** (nobody learns anything about anyone, or finds them, without a two-way match)
2. **No spam, no fakes**
3. **Efficiency**

When two goals conflict, the higher one wins.

## 2. The system in one paragraph

Each person is KYC-verified by a third party (e.g. Stripe Identity) and has an AI
agent (their own if Belong-certified, or one Belong provides). The agent helps the
person write an **intent** in plain text and a **listing** of details, each marked
hidden or shown. The listing enters the locked room through the **Warden agent's**
entry check. Inside, the **Matchmaker agent** reads everything on the
**Marketboard**, narrows the field, and proposes pairs where *both* intents are met.
Nothing the Matchmaker agent sees ever leaves the room. Proposals go out only
through the Warden agent's exit check, which strips anything not marked shown.
Both people must say yes before they are introduced.

## 3. Roles

| Role | Where it runs | Can read | Can write / output |
|---|---|---|---|
| Person | App | Their own data, their matches | Intent, listing, yes/no |
| Personal agent (A, B…) | Outside the room | Its person's data; the shared list of details in use (§6) | Drafts intent + listing (person approves); relays follow-up answers |
| KYC provider | Third party | ID documents | `verified: bool`, `age_band` only |
| Warden agent | Boundary (entry + exit) | Inputs at entry, proposals at exit, the Marketboard (to build the shared list) | Accept/reject; sanitized outputs; the shared list; tamper-proof check receipts (no content) |
| Matchmaker agent | Inside the room | Everything on the Marketboard | Proposals (internal only); fixed-menu follow-up questions; with the Matchmaker AI, requests to its judge |
| Matchmaker AI (judge) | OpenAI's API (the demo's profiles are synthetic), treated as hostile | Per pair the hard rules allow: both listings' shown details and descriptions, names, ids and hidden texts taken out, under request-local labels | Verdicts to the Matchmaker only (match / close / no each way, a fit, listings that seem to hunt one person), read strictly |
| Marketboard | Inside the room | — | Stores listings; nothing on it acts |

**Separation rule:** the agent that sees everything (Matchmaker) has no path out;
the agent that controls the exit (Warden) never decides who matches.

**The Matchmaker AI (hybrid, `matching/hybrid.py`).** The rules narrow and keep everything hard and exact
(how and where, the same kind, dating's genders and ages, wishes on same-named details, a strict listing's
every wish, a person's own listings never paired, daily limits, the close-match and question opt-ins); a judge
injected from outside `matching/` (plain data in, plain data out) judges the fuzzy part of each pair that passes
them; the result is the same Proposal contract, reasons and refs from the rules' code, hidden fields only
through the crowd-gated bonus, and the Warden checks it as before. What the judge receives is a fixed rule: shown
fields (a ShownView) and the description, with every first name, id and hidden text of anyone taken out, never
a hidden field, under labels of that request; every request goes only once the Warden's check passes it against
the board as it is then (`warden.exit_judge_request`), and is kept and audited again (`leaks.py`, A8). Its answer is
read strictly. The judge is a model over OpenAI's API (`matchmaker_openai.py`, the Chat Completions format, so a
self-hosted OpenAI-compatible server can replace it by settings only), treated as hostile (TRUST.md #39 to #44).
The demo runs it on synthetic and test profiles.

**The person's own changes:** a change the person makes to their listing themselves is theirs. Their
agent is told what changed, from what to what, and may update only the other parts that still say a
word they replaced, by one fixed rule (`agent.keep_edit`: their change keeps its value, a part brought in
line uses only words it had and their new wording, shown or hidden stays as it was, a hidden detail's words
never go into a shown part, in any form); its update is made on the draft as it is when it finishes, and
checked against what is shown then. The related topics are the agent's own reading of what the listing is
about, so when what it is about changes (the headline, seeking, offering or description, nothing hidden),
the agent writes the topics again, only with words a match can already see on the profile, in any form,
and never with all the words of a hidden value together in one topic (`agent.rereads`). A
change made on an older copy (while the agent was working) is made on the draft as it now is
(`agent.rebase`). Without a model agent the phone offers the same change in those parts, one tap each
(TRUST.md #34).

**Close matches and questions** are each listing's own choices (`notify_close`, `notify_questions`), both off
by default and enforced by the Warden. The person's account settings, both off until they turn one on, only
set those choices: "always allow questions" is where a new listing's questions start, and "always include
close matches" turns a listing's close matches on, through the usual Warden-checked update, only where the
Warden would let out "No matches yet. Want to include close matches?" (no match yet, once ever, asked only
after the run's introductions); the person is told in that listing's chat. They never change a listing's
choices once it has its own. A yes to that question turns on close matches only (TRUST.md #24, #25).

## 4. Data model (v0.1)

```text
Person
  id, verified: bool, age_band, created_at

Listing                      # several per person (at most 5 live), versioned
  person_id, owner_id
  intent_text: str           # "Looking for a second violinist for a string quartet…"
  exactness: float 0..1      # 1 = exact, 0 = wide net
  fields: [Field]            # built-in fields and open details
  confirmed_shown: [key]     # sensitive built-ins the person chose to show
  notify_close, notify_questions: bool   # close matches, questions: off by default (§3)
  updated_at

Field
  key: str                   # a built-in key ("first_name", "religion", …) or an open
                             #   detail's "about:name" ("me:instrument", "them:skill")
  value: str | [str]         # an open detail's value is always a list (1-3; a wish 1-5)
  visibility: "shown" | "hidden"
  sensitive: bool            # sensitive built-ins start hidden + off; open details never
  compare: "same" | "steps"  # open details only
  steps: [str]               # open details compared on a scale: 2-7 ordered steps
  required: bool             # wishes only: false = "ideally"; true only from the person's own words (§5)

Proposal  (internal to the room, never leaves as-is)
  person_a, person_b
  a_intent_met: bool, b_intent_met: bool
  score: float
  reason_fields: [key]       # built-in keys shown by BOTH people, plus exactly the
                             #   open-detail refs ("agree:{name}", "wants:{name}") both
                             #   people's shown details give (schema.detail_reasons)
  used_hidden_fields: [key]

Introduction  (the only thing about a match that leaves, via Warden exit)
  person_a_view: { shown fields of B, reason text from shown fields only }
  person_b_view: { shown fields of A, reason text from shown fields only }
                             # an open detail travels as {values, compare, steps, required}

DetailsInUse  (the shared list every personal agent sees, built by the Warden; §6)
  details, scales, values, phrases, places   # no counts, alphabetical
```

**Built-in fields** (`schema.FIELD_REGISTRY`, a fixed set):

- from the ID check: `first_name`, `age_band`;
- where and how to meet: `city`, `country` (free text), `format` (in person, online, or either),
  `travel` (same city, up to 30 km, up to 100 km);
- what the listing is for: `seeking` / `offering` ("looking for" / "you are" phrases), `headline`,
  `topics`, `pay`;
- what the agent understood: `kind` (`dating` or `other`), and for dating who they'd like to meet
  (`wants_gender`) and those people's ages (`wants_age`); dating also needs the person's own `gender`;
- hidden by default, values in the person's own words (short plain text, no list of choices):
  `children`, `neighborhood`, `employer`, and the sensitive categories `religion`, `politics`,
  `health`, `income_band` and `kinks` (intimacy, on dating listings only). A sensitive one is shown
  only if the person confirms it.

**Open details** are everything else a listing says, written by the person's agent on the spot:
nothing in the product lists their names or values. Each has a name ("instrument", "skill",
"noise tolerance"), says whom it is about (`me`: the person; `them`: what they'd like the other
person's own value to be; `both`: something about the request itself that both must share), and
how two values compare (`same`: they share a value; `steps`: places on an ordered scale the agent
writes out). The value "any" means open to any value of that name; on a scale it goes only in a wish
about the other person (their own value, or the request's, is one step). It says nothing two people
have in common, so it is never a reason. One entry rule (`schema.detail_problems`) holds for every
detail, checked by the Warden at entry and by the agents before saving: a name of 1 to 3 plain words
in any script (letters, the marks on them, digits) that isn't a built-in's name or a word the ID check or Belong
vouches for; short plain values and steps with no contact details and no "verified" or "KYC" (nor
in the shown headline, phrases, topics, pay or format, `schema.shown_claim_problems`); at most 12
details; a wish (`them`) and what the request needs both to share (`both`) are always shown. Open
details are never sensitive: a name, value or step with a sensitive word is refused and sent to the
hidden built-in category it belongs in, and intimacy is refused outright on a listing that isn't for
dating. Nor may they single out one person: the description's specific-person rules (an ex, "named"
+ a name, "works at" + a name) are read over each open detail too, written as "name: values" and as
its name before each value (so "named": a name counts), every text with its words one space apart,
and the identifiers rule (§5: a place to find them with how they look, three or more details of
looks, a distinctive mark) over the description, the listing's other free text about the person
sought and every shown wish together, with what the poster says about themselves (their "you are"
phrase and every open detail about them) read as theirs (`warden.detail_person_problems`; at entry the Warden names the
detail a problem is in, `warden.detail_person_findings`). Every save of open details (the model agent's,
the rule-based agent's, and the phone's Save draft) runs these rules first through one check
(`agent_llm.detail_save_problems`), so none can drift from the Warden's; the rule-based agent takes
off a detail it kept from before that the check refuses under the new description, and says so. The
model agent's save and the phone's Save draft also run the description's own rules, the full-name floor
included, with the same fields the Warden reads (`agent_llm.description_save_problems`, §5).

**One normaliser** (`schema.canonical`): every comparison, count, crowd, reveal and the shared list
compare text ignoring case, width, accents on Latin letters, spacing and "-", "_" or "/" between
words. Phrase matching and text ranking read words the same way, in any script (`text.letter_words`),
and places also drop every accent (`schema.plain_place`). Values are stored as written.

## 5. Matching rules

- **One kind of match.** No tiers. The person's intent text + exactness define it.
- **Dating, or anything else.** Dating has fixed rules of its own (the gender each wants to meet,
  an age in the other's range). Everything else is "other": what one is looking for must be what
  the other is, by their "looking for" / "you are" phrases. There is no list of activities. Shared
  role or social words alone ("partner", "teacher", "meetup") never meet a request that says what
  it is about, and what someone looks for but isn't themselves must be in what the other is (two
  first violinists both looking for a second violinist are not each other's answer). Two phrases
  with the same main noun are compared on what they are about besides it, in any script: the same
  noun alone is never enough, and more than half of those words must be in common (at least half
  for a wide net, exactness under 0.5, where what the other shows about themselves counts too); at
  the default exactness and above, two phrases that each have a word the other lacks never meet,
  whatever they share and wherever the role noun is (first, as in "amico di ...", or last), so only
  the same phrase, or one that narrows the other, meets. The
  description is never shown and never in a reason. Matching reads it to tell dating from the rest;
  for who and what ages a dating listing wants, and how a listing meets, when its shown details
  don't say; and as text similarity, which is part of the score and orders shortlists (TRUST §5,
  limit 4).
- **Wishes.** Every wish a listing shows (its `them` and `both` details) must be met by the other
  listing's own shown detail of the same name, compared by `schema.satisfies`: a shared value, or
  one of the wished steps (a wide net, exactness under 0.5, also accepts the step next to one, on
  the identical scale). A wish that says nothing ("any", or every step of its scale,
  `schema.says_nothing`) is always met and never asks for the detail. An own "any" meets a wish
  only for a listing that isn't strict (exactness under 0.8, `schema.comparable`), and is never a
  reason (`schema.meets`). A wish the other listing can't be compared on (no such detail, another
  way of comparing, a step not on the scale) fails a required wish, and an "ideally" one only at
  exactness 0.8 or more. So a detail named or scaled differently by two agents costs a match, never
  makes a wrong one while the wish is required; the shared list (§6) keeps agents' names the same.
  Matching and the Warden share these comparisons, so they can't disagree.
- **A wish is required only from the person's own words.** A required wish shuts out every listing
  that doesn't show something to meet it, so an agent's guess must never do that. Whatever agent
  drafts a listing, a wish it saves as required stays required only by one of three fixed rules
  (`agent.ground_wishes`):
  1. What the person typed themselves for that listing says one of its values in full
     (`agent.wish_said`). Each word that says something counts in some form by the length rule
     `agent.same_word`, a number only as the same number (thousands separators don't matter; a decimal
     or a time, "2.5" or "9:15", is one number, never its parts) and with the word written right after it
     as one word, with or without a space ("5 km" is "5km", never said by 5 alone; 5 is said by "5km"),
     that word compared by its own letters, never the number's digits ("12 cups" never says "12 cm"),
     except a one- or two-letter word between two numbers ("7 to 9"), which joins neither, a word with
     the marks of its letters, in Hangul also with its particles after it (a syllable standing alone is
     a word), and anywhere inside a run in a script written without spaces (never inside a longer
     number, in digits or numeral characters); a run that mixes scripts is also read as its pieces, so
     a Latin word or a number written inside a Japanese or Chinese run counts; a value of only one- or two-letter words
     counts only with a word of the detail's name; the name alone never counts; a wish that says nothing
     always counts. Any one value said is enough: more values widen who meets a wish, never whom it
     shuts out for not showing the detail, so a value said about anything (one an agent adds next to a
     made-up one too) keeps it (TRUST.md limit 28).
  2. A save kept it as "ideally" earlier, so the agent was told to ask whether it must be met, and the
     few words the agent quotes as theirs for it (its `their_words`, at most `agent.THEIR_WORDS_MAX`)
     are all in the message they typed next, after the latest such save that told the agent to ask: their answer
     (`agent.answered`). A quote of anything typed before that save, or of a later message, never
     counts; the answer counts whatever it says (a word rule can't tell a yes from a no). The demo
     remembers on the person's own listing which wishes were kept as ideally and when, as of the latest
     save that told the agent to ask about each (`_Listing.ideally_since`, `DemoSession._note_ideally`;
     the draft carries that list only from the agent to the demo, `agent_llm.KEPT_IDEALLY`, while the
     notes that told the agent, naming those wishes, stay on the person's own draft like its other notes
     until the next save, so their own phone is sent them, unshown, and nobody else), and passes the
     messages typed since (`DemoSession._answers`); it forgets one when an agent's save no longer has
     the wish as ideally (a must now, or gone). A wish the person was already asked about (the same
     wish, kept as ideally, with a message typed since: `agent.asked_about`) is never asked about again:
     the agent is told so, and what to do on a no (keep it as ideally, and the exactness under 0.8),
     and when it was asked stays, so their answer stays the message after that question
     (`agent_llm.WISH_ASKED`, `EXACTNESS_ASKED`). The profile the agent builds on says the saved
     exactness (`agent_llm.profile_so_far`).
  3. The draft its agent's turn began from already had it as a must, and it is saved as it was or
     wider: it passed these rules when it was saved, or came from what went live. A turn's earlier save
     never stands in for that draft (`agent_llm.turn_start`), so a must put back as it was after a save
     that left it out stays.

  Otherwise it is kept as "ideally": only made not required, never taken off, reworded, or made shown
  or hidden. At exactness 0.8 or more (`schema.STRICT_EXACTNESS`) every wish shuts out who doesn't show
  it, "ideally" ones too, so there every wish is held to the same three rules (a draft as exact counts
  its wishes as musts for rule 3), and a save with one that passes none is made less exact
  (`agent.ground_exactness`: `agent.UNSTRICT`, 0.7), never raised, nothing else changed; a model agent
  is told, and the wish is remembered as one waiting for an answer. What counts is the person's own
  messages in the chat of the listing a save lands on (a new separate listing: their messages from the
  one that led to the other listing's last save, `DemoSession._said_since`; a synthetic person taken
  over: also their description and messages, as the simulation counts them), never a turn the app writes
  for their agent (a change they made on their profile, the check when the agent replied without saving,
  a yes to a question); values typed on the profile don't count yet. A model agent's every save is checked
  against those words, for where it lands (they go to the check, not into the model's prompt: it has the person's messages
  only as its own conversation), and the agent is told
  which wish was kept as ideally, so it can ask, and its instructions ask it to keep at least one value of
  a must as the person wrote it for what must be met, next to any common form (`agent_llm.build_draft`,
  `typed_for`, `SYSTEM_PROMPT`); the draft of an agent that doesn't take them is held to rules 1 and 3,
  and on exactness by them, without being told (it gets no note), before it is kept (`demo.DemoSession.agent_message`), its exactness read as a model
  agent's is (`demo.DemoSession._grounded`, `agent_llm.read_exactness`). An exactness the person left as
  an older copy of their draft had it is the draft's as it now is (`demo.DemoSession._rebased`), so a
  change kept from that copy never undoes one made less exact meanwhile (when the phone sends that copy; one that sends none can: TRUST.md limit 28). In a turn after a change the person made themselves,
  `agent.keep_edit` already keeps every wish as they kept it. It is a rule of the person's own agent and
  session, not a Warden check: the Warden never decides who matches and never sees anyone's chat. It
  reads whether words are theirs and when they were typed, not what they mean (TRUST.md #45, limit 28).
  The Matchmaker AI is told that a wish that isn't required, and that the other listing says nothing
  about, never makes its answer "no" or "close" on its own.
- **Both must be looking.** A pair is proposed only if A's intent is met by B *and*
  B's intent is met by A. Nobody is approached who didn't ask.
- **Never a specific person.** The Warden agent rejects intents narrow enough to identify one
  individual (names, handles, clues that together narrow to one person…), in the description, the
  shown free text and the open details. **Clues** follow one fixed rule
  (`warden.identifier_problems`): one or two common details of how someone looks describe many
  people and identify nobody, so only clues about the person sought count (never the poster's own:
  "I'm a cashier at", "I'm quite tall", "my beard", a mark on their own body; but a workplace or home
  they share with the person sought, "we both work at", "the same ... as you", is where to find them),
  over the description, the other free text and the shown wishes together, with what the poster says
  about themselves (what they offer, their "you are" phrase, and their open details about themselves)
  read as theirs, as if it began "I'm" (so a clue about someone else written there still counts), and
  the Warden refuses a place or time where that person can be found (an anchor: where they work or
  live, someone the poster saw or met, with "I", "we", "you and I" or no subject at all, "the" + a
  clock time, "the man" or another word for one person at or from a place) with any detail of how
  they look; three or more distinct details of how they look; or a distinctive mark (a scar,
  birthmark, tattoo or piercing described with where it is on them, "on her ...", "on the back of his
  ...", or what it shows). A looks word counts only where it describes a person (next to a word for
  one or before the person an anchor finds, said of one, standing alone between commas, in a list of
  looks, or the value of a wish about them), never a thing ("short sailing trips", "tall hedges", "a
  hair salon", "glasses of cider"), and a looks noun followed by a word that says what it is like
  counts where it is someone's feature ("with hair tone ...", "...; beard trim ..."); a workplace or
  home counts only at a place ("at" + an article, a possessive, a number or a name), never "works at
  night". Every text is read as it looks: a character that shows nothing never hides a rule. It reads how the sentence
  is built, with the few looks and workplace words of the rule it replaced, the marks, a few words for
  where someone lives or was seen or met, and words for a person; English only (TRUST.md limit 26). The model agent's save and the phone's Save draft run
  it too, with the same fields (`agent_llm.description_save_problems`, `detail_save_problems`); the
  rule-based agent's save runs it over open details read with the description (`detail_save_problems`),
  and the description it writes is checked at entry (and by Save draft, when the person saves). Names are
  kept out in three layers, the AI only ever stricter:
  1. The Warden's fixed floor (`warden.full_name_pairs`): two capitalised words in a row in the
     middle of a sentence look like a full name, in the description and the shown free text, also
     with lower-case name particles between them ("van", "dos", "de la") or after an initial's full
     stop ("K. Mensah"). No typed list of places, languages or titles: a word is never part of a name
     only as the pronoun I or its contractions, a word written all in capitals, a month or a weekday
     (the standard library's calendar), or part of a place written out whole: the city or country the
     listing itself shows (built-ins only, shown only: never an open detail or a hidden value), or a
     place on the last shared list the Warden built (§6; `warden.never_names`, `Warden.listed_places`).
     One word of a place with more words unlocks nothing alone. A live listing keeps the listed places
     it was accepted with (`Warden.kept_places`), so a later list never refuses its next update. The
     model agent's save and the phone's Save draft run the same floor with the same fields
     (`agent_llm.description_save_problems`), so they agree with entry.
  2. The person's own model agent judges words and context (a name in lower case, a first name or
     nickname alone, clues that fit one person). Both providers share its instructions and its tool,
     where it sets `singles_out` when it judges a request singles someone out: `build_draft` then saves
     nothing and the person is told why (in fixed words, even if it saves something else or its provider
     refuses to go on), and the same request sent again in that turn is refused too. It can only refuse,
     never let through what the floor refuses;
     when the floor refuses two capitalised words that are really a place, a title or a thing, it is
     told which two, rewords them and tells the person, and the rules check the new text. The
     rule-based agent has no model: it relies on the floor.
  3. The Matchmaker AI (§3, §8.3) says per listing whether it seems to hunt one particular person, asked
     about that listing alone (so no other listing's text can get it flagged), and then drops all that
     listing's pairs until what it sees of the listing changes. It can only leave pairs
     out, never add one, and can't explain itself to the person (`hunts_one_person` in `LLMMatcher`;
     instructions to a model, tested with stand-in judges).
- **AI suggests, the person decides.** Defaults for exactness and hidden/shown come
  from the intent; the person can override everything.

## 6. Privacy rules for hidden fields (always on, not user-configurable)

- **Never the only reason.** A proposal must be justified by shown fields alone; a
  hidden field may only *raise* a score.
- **Only counts in a crowd.** A hidden field value is used only if at least *k*
  other people share it besides the two being matched (k = 20, `schema.K_CROWD`). The crowd is
  counted in people, not listings, on active listings, with values compared canonically. A wish
  about the other person (`them`) is never a hidden value: it is always shown.
- **Never in the reason.** Reason text is generated only from fields shown by both. For open
  details the Warden works out itself which ones the two have in common (`schema.detail_reasons`)
  and accepts a proposal only if it cites exactly those.
- **Mutual reveal.** A hidden detail both people share is revealed only if both opt in
  after connecting. Never a wish about the other person.
- **The shared list of details in use.** So that agents writing details on the spot name them the
  same way, every personal agent sees one list of the open details (name, whom it is about, how it
  compares), scales, values, phrases and places already in use. It is effectively public, so: the
  Warden alone builds it, from the Marketboard alone (no Matchmaker input), at most once a day; an
  item goes on it only when *k* = 20 different people show it, counted only from active listings
  unchanged for at least 7 days; never from anything hidden or sensitive, never a sensitive word,
  contact details or anything but short plain text shaped like what it stands for (a phrase at most
  6 words, anything else at most 5, no sentence punctuation; a word is a run of letters and digits
  whatever joins them, a capital after a lower-case letter starts one, in a script written without
  spaces every 3 letters count as a word, and in any other a run longer than 12 letters counts one
  word for every 5); no
  counts, alphabetical, capped (200 detail items, 300 phrases, 100 places). Agents get it as data
  after a fixed line, never as instructions, and check every entry against the same shape with
  their own code. The crowd, the shape and that line are what keep instructions off it; no fixed
  rule can tell every instruction from a phrase. The independent audit rebuilds it from the same
  board and requires the same list.

## 7. Warden agent checks (fixed rules, no AI judgment)

Entry:
- Person is KYC-verified.
- Listing matches the schema: built-ins take their fixed forms (kind, format, travel, who and what
  ages for dating) or short plain text; open details pass the one entry rule (§4); intimacy only on
  dating listings; sensitive fields default to hidden.
- Intent, shown free text and open details pass the specific-person check; nothing shown claims
  what the ID check or Belong vouches for ("verified", "KYC").
- At most 5 live listings per person.
- Rate limits on listing updates (per listing, and per person across their listings). An update to a live
  listing that only turns something off (questions, close matches, or matching itself, by pausing it) and
  changes nothing else is never refused by them, nor counted; it is checked and receipted like any update.

Exit:
- Output matches the Introduction schema exactly.
- Every field in each view is marked shown by its owner.
- Reason fields are shown by both; open-detail reasons are exactly the ones both people's shown
  details give.
- A hidden field the Matchmaker says it used is hidden by one of the two, shared, and had by *k*
  other people (§6).
- Reason text references only reason_fields; no hidden values appear (string check), nor a hidden
  open detail's name or steps. Reasons are made of fixed sentences, so the checks read only the
  strings that fill them (what people wrote); anything else is read whole.
- A reissue after either listing changes gets the same checks; if it fails them, it is reduced (no
  reason, only fields that pass every check on their own), never left standing with something since
  hidden. Once one of the two listings is deleted, every reissue is reduced, and the deleted
  listing's view keeps only what it already had and the other still shows unchanged.
- A listing id is one listing's for good (entry check `listing_id_owner`): never another person's,
  and never accepted again once its listing is deleted, so a reissue reads only the listing the
  introduction was sent to.
- Follow-up questions come from the fixed menu only.
- Approach limits per listing per day (15 in the prototype).
- The shared list of details in use (§6) is built here, once a day, with a receipt.
- With the Matchmaker AI on (§3, §8.3), each request to it is checked just before it goes, against the board
  as it is then: only what each listing in it shows now and its own description, under labels the Matchmaker
  kept, with no first name, id or hidden text of anyone; a request that fails never goes (`exit_judge_request`).
- Write a receipt: `{check, result, timestamp, hash}`; never the content.

## 8. Prototype scope (Step 2)

Goal: prove that free-text intent matching produces good matches, and that nothing
hidden leaks, built the way the service will run.

1. Generate 300 synthetic people (mix of dating + functional intents, realistic
   hidden fields), written as phrases and open details. A second run gives a quarter of them
   agents that write one detail differently (another name, another scale, one left out or
   hidden), to show that costs matches but never makes a wrong one. Every run holds each listing's
   wishes to the person's own words (§5: each synthetic person types their description and each must
   they have with its value; where their agent writes it otherwise, it asks and they answer yes, so
   every real must stays one); two more sets of three runs give a quarter of the listings that aren't
   dating an agent that adds a must the person never had (quoting their own first words for it; asked,
   they say no), with that rule and without it, to show what it wins back, and with it while that agent
   quotes their no as their answer, to show what a word rule can't win back; a last pair has nobody type
   their musts with their values: their descriptions still say most of them, and the run counts, in
   musts, how many nothing they typed says (`sim.unsaid_musts`), which only their yes to its question
   keeps; with that yes and by values alone, to show what reading values alone would cost on those.
2. Narrow: embeddings + hard filters → shortlist per person.
3. Matchmaker (pluggable interface):
   - `OfflineMatcher`: embeddings + rules, no API key.
   - `LLMMatcher` (`matching/hybrid.py`): the hybrid. The rules narrow and decide everything hard;
     an injected judge decides the fuzzy part (§3). The judge is a model over OpenAI's API
     (`matchmaker_openai.py`; the demo's profiles are synthetic; `BELONG_MATCHMAKER`,
     `belong run --matcher ai`), or a self-hosted OpenAI-compatible server by settings only.
     It is also where a model may decline to match a listing it judges is hunting one person (§5): it
     can only drop pairs, never add one, and can't explain itself to the person. Cost: one request per
     focus listing, verdicts kept per pair of listing versions, at most 80 pairs a run (the rest by the
     rules). `tools/compare_judges.py` compares it with the rules against the ground truth (live, opt-in).
4. Warden exit check as plain code.
5. Mutual yes simulation.
6. Web view to inspect: people, proposals, introductions, and a **leak report**
   (any hidden value that reached an output = test failure; with the Matchmaker AI, every request
   its judge was sent is audited too).

Build constraints that mirror the room:
- Matchmaker module has no network access and no logging of raw data (the Matchmaker AI's judge is
  injected from outside it; what it is sent is a fixed rule and audited).
- Only the Warden module can produce output objects.
- Tests fail on any leak.

## 9. Later steps

- **Step 3 — Verifiable code:** a signed attestation that the running code is the published
  code, reproducible builds, an outside audit of the Warden's rules.
- **Step 4 — Trust and safety + app:** KYC integration, approach limits, consent
  screens, a privacy policy reviewed by a lawyer (what Belong can see, what goes to AI
  providers), closed pilot.

## 10. Open questions

- Which details start hidden: the built-in ones are fixed (§4); an open detail starts shown unless
  the person or their agent hides it.
- Value of *k* for the crowd threshold and the shared list (20 is a placeholder), the list's caps
  and its 7-day wait.
- Certification process for outside agents.
- Model choice for the Matchmaker AI (OpenAI's API today).
