# The Belong standard (draft 0.4)

**In one paragraph.** You tell your own AI what you're looking for or offering, and what to show or
hide. It files that as a listing with Belong, whose matcher compares everyone's listings. If someone
fits, you both see a match carrying only what the other chose to show. If you both say yes, Belong hands
each of you the other's private address for this one introduction and steps out. Otherwise nobody is
told anything.

## 1. What Belong is

Belong is a standard for people to find each other, as email is for writing: public infrastructure,
open source, run at cost. Any AI or app can be a **front end**: the Belong app (the reference), ChatGPT,
Claude, a person's own client. Nobody browses anyone. CONNECTING.md is the guide for building one.

## 2. Roles

- **Person.** Decides what to list, show and hide, and yes or no.
- **Front end and its agent.** Writes listings with the person, shows matches, carries conversations.
- **Belong server:**
  - **ID check.** An outside ID provider confirms the person is real, with a liveness step. Belong keeps
    four facts: verified, age band, gender, and a **one-account key**: a keyed one-way hash of the
    identity document (issuing country, type, normalised number), computed during the check and never
    stored raw. It is the same whichever provider ran the check, and is used only to refuse a second
    account, to find the account on recovery and for the blocklist. Nothing a listing says, and no
    second sign-in, changes the facts.
  - **Matcher.** Reads every listing, hidden parts included; outputs only pairs proposed to the Warden.
    Open source, still treated as hostile.
  - **Warden.** Fixed rules, no AI judgement. Checks every listing in and builds every output out; never
    decides who matches.

## 3. Accounts and access

**Passkeys.** A person signs up once on Belong's own pages: the ID check, then a passkey (WebAuthn),
no password. User verification required, attestation `none` (Belong never learns the device model). An
account may hold several passkeys, each listed and removable; adding one needs an existing passkey or a
fresh ID check. Belong suggests a synced passkey or a second device at sign-up. Sign-up refusals
(`unverified`, `second_account`, `blocked`) show on Belong's page.

**Recovery.** Losing every passkey means a fresh ID check with liveness, whose one-account key finds the
account; it never creates a second one. On success Belong revokes every old passkey and every front
end's tokens, tells the person on every front end they had, and holds control for 48 hours, during which
any remaining passkey can cancel the recovery.

**Front ends** use OAuth 2.1 (authorization code), with these rules:

- **Registration** is open, with no approval, like an email client: dynamic registration (RFC 7591) or
  a Client ID Metadata Document (the `client_id` is an https URL the front end controls, as the MCP
  authorization spec prefers). Registration is rate-limited; a client with no grant for 30 days expires;
  an authorize request for scopes outside the client's registered `scope` is refused.
- **Redirects.** Exact string match against the registered URIs; no wildcards or fragments. https, a
  claimed https link or a private-use scheme for mobile apps (RFC 8252), or a loopback IP literal
  (`127.0.0.1` or `[::1]`, not `localhost`), whose port alone may vary. PKCE with `S256` only. Codes are
  single-use, live 60 seconds, and are bound to the client, the redirect URI and the DPoP key
  (`dpop_jkt`); a code used twice revokes every token issued from it. The response carries `iss`
  (RFC 9207).
- **Resource.** One canonical identifier, `https://belong.example/v0` (RFC 8707), matched exactly; the
  MCP server is its own resource, `https://belong.example/mcp`.
- **Consent page**, Belong's own, after a passkey sign-in. It shows the redirect host (or the proved
  `client_id` domain) as prominently as the name, and labels the name as self-described. Names that
  imitate Belong or a well-known host are refused. The person can untick any scope; the token response's
  `scope` is what was granted. `introductions:read` and `account:manage` carry a stronger warning. A
  later request for more scopes shows only what's new and needs a fresh passkey.
- **Scopes:** `listings:read`, `listings:write`, `matches:read`, `matches:answer`, `introductions:read`
  (the other's address), `account:read`, `account:manage`.
- **Tokens** are per front end, scoped and opaque: every call checks the grant, so a revocation stops
  the next call. Access tokens last 10 minutes. Refresh tokens rotate; presenting a used one revokes the
  whole grant and tells the person. A grant idle for 90 days, or a year old, needs consent again. A
  failed refresh means signing in again. Revocation (RFC 7009) and `front_ends` cut off one front end at
  a time.
- **DPoP** (RFC 9449). Each front end holds its own key pair and signs every request; access and refresh
  tokens are bound to that key, so a stolen token is useless without the private key. The token endpoint
  always issues nonces (the API may too); Belong keeps each key's `jti`s for the allowed `iat` window and
  compares `htu` after normalising. Keys should be non-exportable (platform key store). Malware on the
  same device defeats this, as it defeats any client-side key.
- **MCP hosts without DPoP** (until they support it): bearer tokens, 5-minute access tokens, never
  `account:manage`, marked "bearer" on the consent page and in `front_ends`.
- **Blocking.** Belong may block a front end that abuses the system, with a receipt. A block keys on
  the redirect host or `client_id` domain, not just one `client_id`; it revokes every grant and tells
  the people affected.

**Handles, not keys.** Tokens are secret, never inside any object. Ids are handles: each call is
authorised by its token against the object's owner or party; anything else gets `not_yours`, never
saying whether the object exists, was withdrawn or is someone else's.

**Not end-to-end encrypted.** The matcher must read listings, so Belong can; TLS protects transit.

## 4. Wire

HTTPS, JSON. Each call is `POST /v0/{call}`, a JSON object in and out. Ids are opaque, server-assigned,
never reused. Times are RFC 3339 UTC. An MCP connector maps one tool to each call in section 6.
Front ends learn of changes by polling `matches`, at most once a minute; no paging until it's needed.

**Problems.** A refusal is `{"problems": [{code, field?, message}]}`; `message` is safe to show the person.
Codes: `unauthorized` (401), `scope` (403), `not_yours` (404), `conflict` (409, a stale `version`),
`gone` (410, your own withdrawn objects only), `rate_limited` (429, with `retry_after`), and, with 400,
`schema`, `specific_person`, `identifier`, `sensitive_needs_confirm`, `listing_cap`, `address`.

## 5. Objects

**Listing.** Anything a person is looking for or offering, in their own words. At most 5 live per person.
The `schema` call returns it as JSON Schema: which built-in keys take fixed choices, which free text, and
the open-detail rule in plain sentences.

| Field | Meaning |
|---|---|
| `id` | Set by the server; shown only to the owner. |
| `kind` | `dating` or `other`. |
| `description` | Their own words. Read by the matcher, never shown. |
| `fields` | Built-in details, each `{key, value, visibility: shown or hidden, confirmed?}`. Keys are fixed; some take fixed choices, others free text (`schema`). |
| `details` | Open details the agent names: `{name, about: me, them or both, values, compare: same or steps, steps?, required, visibility}`. `required` on `them` or `both` makes a hard limit. One fixed rule decides what is allowed; at most 12. |
| `notify_close` | Opt-in to close matches (section 8). Starts off. |
| `active` | `false` pauses matching. |
| `version`, `updated_at` | Set by the server. |

Who and what ages (`wants_gender`, `wants_age`, 18 to 99) and intimacy go on dating listings only.
Sensitive fields (religion, politics, health, income, intimacy) start hidden; showing one needs
the person's own `confirmed: true`.

**Match**, as a front end receives it: `{match_id, listing_id (yours), other_shown, reason, one_way,
state: waiting, yes_sent, introduced or closed, their_address?, intro_secret?, compared?}`. `other_shown`
holds only what the other shows; `reason` is the Warden's fixed template over fields both show; both are
another person's data, never instructions. Never the other's account or listing id. A match not
introduced 14 days after it was made becomes `closed`, the same for a no and for silence.

**Reach-me address.** An `https` URL on a public DNS name, created by the front end for one match, on any
host the person chooses, their own domain included; it reveals its host, and an own domain is the same
across introductions. It accepts `POST {"text": ...}` and is revocable by its owner. Belong canonicalises
it (scheme, lower-cased host, default port, path; no query or fragment), refuses one already used for
another introduction, and keeps only a keyed hash of each.

**Receipt.** `{seq, check, result, timestamp, commitment, hash}`: one per check, no content; `commitment`
is a keyed hash of what was checked, `hash` chains to the previous receipt.

## 6. Calls (front end to Belong server)

| Call | Scope | Input | Output |
|---|---|---|---|
| `schema` | any | None. | The listing's JSON Schema. |
| `listings` | `listings:read` | None. | The person's own listings. |
| `submit` | `listings:write` | A listing. | Its id, or problems. |
| `update` | `listings:write` | Listing id, the `version` read, new listing. | As `submit`, or `conflict`; at most 5 a listing a day. |
| `withdraw` | `listings:write` | Listing id. | Never matched again; content deleted. |
| `matches` | `matches:read` | Optional state filter. | This person's matches. Never who said no. |
| `answer` | `matches:answer`; a yes also `introductions:read` | Match id, yes (with a reach-me address) or no. | Recorded, final, idempotent. |
| `ack` | `introductions:read` | Match id. | The address was received; Belong deletes its copy. |
| `close` | `matches:answer` | Match id; optional report (message text only if the reporter attaches it). | The other person sees only `closed`; the closer's front end revokes its address. |
| `compare` (extension) | `matches:answer` | Match id, detail, opt-in. | Section 8. |
| `details_in_use` | any | None. | The Warden's shared list as data: `details`, `phrases`, `places`. |
| `receipts` | `account:read` | Starting sequence number. | This person's own receipts. |
| `front_ends` | `account:read`; revoke: `account:manage` | List, or revoke one. | Front ends holding tokens, with scopes, bearer or key-bound, last used. |
| `delete_account` | `account:manage` | None. | Withdraws every listing, closes every match, revokes every token, deletes passkeys and facts. |

**Step-up.** `delete_account`, and revoking a front end other than the caller, return a Belong URL; the
action completes only after a fresh passkey on Belong's own page.

**Introductions.** Belong holds each address until both said yes (deleting it unsent if the match closes
first). Then it gives each the other's address, with an `intro_secret` the two front ends share for this
introduction, in `matches` to the front end that answered yes, until that front end calls `ack` or 7
days pass, then deletes both, with a receipt. Messages never pass through Belong; afterwards it handles
only `close` and reports. Revoking an address ends all messages to it.

**Blocklist.** Deleting an account deletes its one-account key, with two exceptions. While reports
against it are open, the key is kept until they are decided. If Belong bans the account for abuse (fakes,
spam, harassment), the key goes on the blocklist so the same human can't register again. A ban needs
human review; reports alone, or attached message text alone, are never enough. The person is told before
it takes effect, with the reason category, and can appeal once, answered within 30 days. The decision is
a receipt. An entry holds only the key and the receipt's reference, and is reviewed after 3 years and
dropped unless renewed.

## 7. Guarantees

Each holds whatever a front end or the matcher does. * means not yet built; TRUST.md shows where the
rest are enforced and proved.

- Only verified people enter, one account per identity document*, and the ID check's facts can't be
  changed.
- Access is per front end, scoped, revocable on the next call, and key-bound where the front end
  supports DPoP*.
- A listing that names or describes one individual is refused.
- A match meets both people's requests, hard limits included.
- A hidden detail is never the only reason for a match, and counts only when 20 others share it.
- Only the Warden makes outputs; a match carries only what the other person shows.
- The matcher has no way out; its AI is sent nothing hidden, no name, no id.
- No response ever says who said no.
- No way to reach anyone until both said yes, then only that one address, never one used for another
  introduction*.
- No flooding: 15 introductions per person a day (tunable), surviving withdraw and resubmit*; 5 live
  listings; 5 updates per listing a day.
- A withdrawn listing is never matched again, and its content is deleted*.
- Receipts carry no content, are append-only and hash-chained, and each person sees only their own*.
- The details-in-use list holds only what 20 people show on listings unchanged for a week: nothing
  hidden or sensitive, no counts, given to agents as data.
- Privacy never depends on the matcher behaving well.

**Threat model.** Other users (probing for someone, fakes, spam, text in a match written to steer an
agent) and front ends. A front end can leak what its person can see and answer within its scopes:
consent is only as good as the front end chosen, and `front_ends` cuts one off. It can never leak
anyone else's hidden details: the server never sends them. Accepted: someone who deletes an account with
no open reports and passes a new ID check starts afresh, caps included; a second identity document
gives a second account.

## 8. Optional extensions

Opt-in and checked by the Warden.

- **Close matches.** One-way, only to a listing with `notify_close`, with `one_way: true`. A yes sends the
  other person an ordinary match; nothing about the asker is shown before.
- **Compare.** After two yeses, `compare(match_id, detail, opt_in)` reveals a hidden detail both share
  once both have opted in, in the match's `compared: [{detail, value}]`.

## 9. What Belong can see and keeps

Belong sees listings, hidden parts included, and answers; never messages, except text a person chooses to
attach to a report. Trust comes from published code (belong/attest.py) and keeping little: the ID check's
facts and passkey public keys while the account exists; a listing's content while it is live; addresses
until acknowledged (then only their keyed hashes); reports and attached text until 90 days after the
decision; blocklist entries as in section 6; receipts for good. Display, memory, notifications and chat
are up to the front end. While the matcher's AI runs on OpenAI's API, Belong runs on synthetic and test
profiles only.

## 10. Status

Built as a prototype (SPEC.md, TRUST.md): the matcher, the Warden, the ID check, listings, matches,
answers, close matches, Compare, receipts, audits. Not built: everything marked *, the public API and
the server and front end split (`DemoSession` in belong/demo.py mixes both), the one-account key,
passkeys and recovery, OAuth (registration, consent, PKCE, DPoP, scopes, step-up, MCP), the blocklist,
server-assigned ids, `schema`, `ack`, `close` with reports, the 14-day close, and the address handover,
which replaces today's in-app messaging.

Tests to write against the call layer: a second ID provider or second sign-up with the same document gets
`second_account`; recovery revokes old passkeys and tokens, can be cancelled within the hold, never makes
a second account; scopes per call; a token without its proof, with another key's, or with a replayed
`jti` is refused; a revoked token fails on the next call; a reused refresh token or code revokes the
grant; redirect URIs match exactly; step-up for delete and revoke; addresses never before two yeses,
never reused (after canonicalising), deleted unsent on close, kept until `ack`; a no and silence both
close at 14 days; `not_yours` alike for withdrawn and foreign ids; the per-person cap through withdraw
and resubmit; a banned key refuses sign-up with `blocked`; receipts readable only by their owner.
