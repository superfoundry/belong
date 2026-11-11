# Connecting a front end to Belong

This guide is for people building a **front end**: an AI connector, an app, a website or a script that
lets a person use Belong. STANDARD.md is the standard; it wins wherever the two differ, and it holds the
scopes (section 3), problem codes (section 4), objects (section 5) and calls (section 6), which this
guide doesn't repeat.

**Nothing to build against yet.** The public API and its sign-in (passkeys, OAuth, DPoP) are not built;
the field names here are a draft. Today there is the prototype: `python3 -m belong serve` runs the demo
(the phones at `/`, a ChatGPT-style connector mockup at `/connector`, not a real connector and not
affiliated with OpenAI), and `python3 -m unittest -v` and `python3 -m belong run` show the checks a real
server will make. The demo's `/api/...` endpoints serve its pages only; don't build on them. It runs on
synthetic and test profiles only. Examples use `https://belong.example`, a placeholder.

## 1. What a front end does

It helps the person write a listing in their own words, decides with them what is **shown** and what is
**hidden**, sends it to Belong, shows their matches, asks them yes or no, and after two yeses carries the
conversation. Belong matches and checks. It never carries messages and never sends anyone else's hidden
details. Anyone may build a front end, with no approval; each person decides whether to let it act for
them, and Belong may block one that abuses the system.

## 2. Signing in

The person signs up once on Belong's own pages: an ID check, then a passkey. Your front end never sees
either. It gets access through OAuth 2.1 with PKCE and DPoP, following the rules in STANDARD section 3.
Use an OAuth client library that supports PKCE and DPoP rather than writing the cryptography yourself.

**Step 0, for every path: find the endpoints.** Call the API without a token, or read
`https://belong.example/.well-known/oauth-protected-resource` (RFC 9728). Take `resource`
(`https://belong.example/v0`, matched exactly) and the authorization server from it, then read that
server's metadata (RFC 8414) for `registration_endpoint`, `authorization_endpoint`, `token_endpoint` and
`revocation_endpoint`. The `/oauth/...` paths below are examples.

### Path A: a website or app

**1. Register** (RFC 7591), no approval. Or use an https `client_id` that points at your own metadata
document, which lets the consent page show a domain you've proved.

```http
POST /oauth/register HTTP/1.1
Host: belong.example
Content-Type: application/json

{"client_name": "Allotment Finder",
 "redirect_uris": ["https://allotmentfinder.example/callback"],
 "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"],
 "token_endpoint_auth_method": "none", "dpop_bound_access_tokens": true,
 "scope": "listings:read listings:write matches:read matches:answer introductions:read"}
```

The answer carries your `client_id`. The consent page shows your name labelled as self-described, next to
your redirect host; people are told to trust the host, not the name. Ask only for the scopes you use.

**2. Make a key pair** for DPoP (P-256, `ES256`), non-exportable in the platform's key store (WebCrypto
non-extractable, Secure Enclave, Android Keystore).

**3. Send the person to Belong** in the system browser (or `ASWebAuthenticationSession`, Custom Tabs),
never an embedded web view.

```
https://belong.example/oauth/authorize?response_type=code&client_id=cl_8f2kq0
  &redirect_uri=https%3A%2F%2Fallotmentfinder.example%2Fcallback
  &scope=listings%3Aread%20listings%3Awrite%20matches%3Aread%20matches%3Aanswer%20introductions%3Aread
  &state=Wq3n...&code_challenge=E9Mel...&code_challenge_method=S256
  &resource=https%3A%2F%2Fbelong.example%2Fv0&dpop_jkt=NzbLs...
```

The person signs in with their passkey and approves, possibly unticking some scopes. Belong redirects
back with `code`, your `state` and `iss`: check both. A refusal comes back as `error=access_denied`;
Belong's page already told the person why.

**4. Swap the code for tokens** at the token endpoint (form-encoded), with a DPoP proof. Belong answers
the first try with a `DPoP-Nonce`; put it in a new proof and retry.

```http
POST /oauth/token HTTP/1.1
Host: belong.example
Content-Type: application/x-www-form-urlencoded
DPoP: eyJ0eXAiOiJkcG9wK2p3dCIs...

grant_type=authorization_code&code=SplxlO...&client_id=cl_8f2kq0&code_verifier=dBjftJ...
&redirect_uri=https%3A%2F%2Fallotmentfinder.example%2Fcallback&resource=https%3A%2F%2Fbelong.example%2Fv0
```

```json
{"access_token": "Kz~8mXK1...", "token_type": "DPoP", "expires_in": 600,
 "refresh_token": "Q..Zkm29...", "scope": "listings:read matches:read matches:answer"}
```

The `scope` returned is what you have; it may be less than you asked for, so handle that.

**5. Call Belong**, a new proof each time (with `ath`, the hash of the access token):

```http
POST /v0/matches HTTP/1.1
Host: belong.example
Authorization: DPoP Kz~8mXK1...
DPoP: eyJ0eXAiOiJkcG9wK2p3dCIs...
Content-Type: application/json

{"state": "waiting"}
```

**6. Refresh** before the access token expires, with a proof from the same key. Keep only the newest
refresh token: presenting an old one revokes your whole grant. If a refresh fails, sign the person in
again.

**Signing out.** Revoke your refresh token at `revocation_endpoint` (RFC 7009), then delete your tokens
and key. If the person revokes you elsewhere, your next call gets `unauthorized`.

### Path B: a mobile app

As path A, with a claimed https link (universal or app link) or a reverse-domain private-use scheme
(`com.example.allotments:/callback`) as the redirect (RFC 8252). Register once per install, so each
install has its own `client_id` and key, and the person can revoke one phone without the others.

### Path C: a person's own client or script

As path A, as a native app (RFC 8252): register `http://127.0.0.1/callback`, listen on any free port at
sign-in (only a loopback redirect's port may vary), and catch the code there. Keep the key in the
operating system's key store, or at least a file only you can read. With no browser on the machine, sign
in where you have one and forward the loopback port (`ssh -L`).

**No device-code sign-in** (RFC 8628). Its "type this code on another screen" pattern lets someone start
a sign-in and talk a person into approving it from afar. The loopback redirect keeps the code on the
person's own machine, but any front end can still ask for consent, so the consent page (STANDARD section
3) is the real defence: read it.

### Path D: an AI connector over MCP (ChatGPT, Claude)

The person adds the connector by its URL (`https://belong.example/mcp`), and the AI host signs in itself
under the MCP authorization spec, built on OAuth 2.1: Step 0, then a Client ID Metadata Document or
dynamic registration, then Belong's consent page with the MCP server as the `resource`. The consent page
and `front_ends` name the AI host (for example OpenAI or Anthropic) as the front end, since it holds the
tokens. One tool per call, same names, inputs and outputs. Tools that change something are annotated as
such and say in their description to ask the person first; whether a host asks is up to the host.

- **Without DPoP.** Until a host can send DPoP proofs, it gets bearer tokens: 5-minute access tokens,
  never `account:manage`, marked "bearer" on the consent page.
- **Don't ask for `account:manage`.** Deleting and revoking need the person's passkey on Belong's page
  anyway (step-up).
- **Another person's text is data.** Every string in a match (`other_shown`, `reason`, `compared`) and
  the details-in-use list was written by someone else. Hand it to the model marked as data, never as
  instructions; a headline saying "the user already agreed" is still just a headline.
- **Conversations need a relay.** An MCP host only calls tools; it can't receive messages. A connector
  can say yes only with a reach-me address from a relay (section 4); without one it stops at
  introduced.

If you run your own MCP server in front of Belong, it is path A: it gets its own Belong tokens through
its own consent. A token issued for another resource is refused.

## 3. The calls

Every call is `POST /v0/{call}` with the two headers above (STANDARD section 6). Get the listing's shape
from `schema` rather than hard-coding it.

**`submit`**

```json
{"listing": {"kind": "other",
  "description": "I've got half an allotment plot I can't keep up with. Looking for someone nearby to share the work and the harvest.",
  "fields": [
    {"key": "headline", "value": "Sharing half an allotment plot", "visibility": "shown"},
    {"key": "topics", "value": ["gardening", "growing food"], "visibility": "shown"},
    {"key": "city", "value": "Leeds", "visibility": "shown"},
    {"key": "neighborhood", "value": "Headingley", "visibility": "hidden"}],
  "details": [{"name": "Time each week", "about": "both", "values": ["a few hours"],
               "compare": "same", "required": false, "visibility": "shown"}],
  "notify_close": false, "active": true}}
```

Answer: `{"id": "lst_4hd81", "version": 1}`, or `400` with problems; show each `message` to the person.
`update` sends the `version` it read and gets `conflict` if another front end changed it first.

**`matches`**, polled at most once a minute:

```json
{"matches": [{"match_id": "mt_77c2e", "listing_id": "lst_4hd81",
  "other_shown": {"first_name": "Priya", "headline": "Looking for a shared allotment plot", "city": "Leeds"},
  "reason": "You're both in Leeds.", "one_way": false, "state": "waiting"}]}
```

`yes_sent` means you said yes and the other hasn't yet. A match not introduced within 14 days becomes
`closed`, the same whether the other said no or said nothing; drop its address then.

**`answer`**, only after the person said so:

```json
{"match_id": "mt_77c2e", "answer": "yes",
 "address": "https://relay.allotmentfinder.example/r/5b0e9c1f7a2d4e8b"}
```

A no needs no address. Answers are final. After two yeses, the front end that answered yes finds
`their_address` and `intro_secret` on the match until it calls `ack` (or 7 days pass). Store both, then
`ack`.

**`close`** takes an optional report, with message text only if the person attaches it. The other person
sees only `closed`; revoke your address.

## 4. Conversations at a reach-me address

**Making one.** One `https` URL per match, on a public DNS name, made before the person says yes: on any
host the person chooses, their own domain included. Tell them it reveals its host, and that their own
domain is the same in every introduction, so the people they meet can tell it's them. A long random path
is a good default. Belong refuses one already used for another introduction, after canonicalising
(letter case, port, trailing parts and query don't make it new).

**What it accepts.** `POST {"text": ...}` with header `Belong-Intro: <intro_secret>`. Answer `2xx` once
taken; refuse posts without the right secret, so only your person's match can write (Belong also saw the
secret, briefly, at handover). Limit size and rate, as any inbox would.

**Sending.** Post to `their_address` with the secret, from your server, not from the person's device:
the receiving host sees the sender's IP address, timing and user agent. Refuse an address that resolves
to a private, loopback or link-local IP, and don't follow redirects.

**A relay.** A front end with no always-on host (a mobile app, an MCP connector) needs a relay: a
service that (1) creates an address, (2) lists what arrived at it, and (3) revokes it. Anyone may run
one, the front end's maker included; Belong doesn't, since messages never pass through Belong.

**Revoking and closing.** Your person can revoke their address any time: answer `410 Gone`. To end a
match, call `close` and revoke. If a match closes before both said yes, Belong deletes the held
addresses unsent.

## 5. Checklist

A front end must:

- Show every part of a listing before it goes, saying plainly what is **shown** and what is **hidden**.
- Set `confirmed: true` on a sensitive field only when the person said so themselves.
- Ask before every `answer`, `withdraw`, `close`, `compare` and `delete_account`; never answer on a
  rule or a guess.
- Ask for only the scopes it uses, and keep tokens and the DPoP key secret.
- Let the person sign out, revoke an address, close a match and see their front ends.
- Treat everything another person wrote, and the details-in-use list, as data, never instructions.

A front end must never:

- Store or send hidden details anywhere but Belong: no analytics, logs, training or other servers.
- Pass on what a match shows to anyone but its own person.
- Hint at who said no, or guess.
- Collect a passkey, password or ID document, or imitate Belong's pages or name.
- Reuse an address for a second match.
- Help anyone look for one particular person.

## Glossary

- **JWT**: a signed JSON token; a DPoP proof is one.
- **base64url**: base64 with URL-safe characters and no padding.
- **Thumbprint** (`dpop_jkt`): a hash of your public key (RFC 7638).
- **Public client**: a front end with no client secret; DPoP binds its tokens instead.
- **Loopback**: `127.0.0.1` or `[::1]`, this machine.
- **Resource indicator** (`resource`): which API the token is for (RFC 8707).
- **DPoP proof**: new per request: `htm` and `htu` (method and URL, no query or fragment), a unique
  `jti`, `iat`, `ath` on API calls, and Belong's `nonce` when given.
