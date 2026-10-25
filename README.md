# hermes-session-reset-policy

Re-arm Hermes Agent's `session_reset` policy (idle / daily / both) as a plugin.

## Why

Upstream Hermes v2026.9.11 (commit `1d5d059410`, "stop time-triggered conversation
rotation") removed the background reset timers and made the `session_reset`
config block inert: only explicit suspension (`/stop`, `/new`) replaces a routed
conversation. If you want time-based conversation rotation back — fresh context
after an hour of inactivity, or once per day at 04:00 — this plugin restores it
without patching the core.

## How it works

The plugin registers a `pre_gateway_dispatch` hook. On every inbound gateway
message, before auth and dispatch, it:

1. Resolves the session key for the message: the hook payload's `session_key`
   when the gateway provides one publicly, else the gateway's own
   `_session_key_for_source` helper — so multiplexed profiles and platform key
   layouts match exactly. (No bare `build_session_key` fallback on purpose:
   under multiplexed profiles that key is not profile-namespaced and could
   resolve another profile's route for the same chat.)
2. Reads the profile's `config.yaml` `session_reset` block (the profile is
   resolved from the message's `source.profile` through Hermes' own
   `get_profile_dir`, anchored at the Hermes *root* so both gateway layouts
   read the right file: one multiplexed gateway at the root, or a standalone
   gateway per profile running with `HERMES_HOME` set to its own
   `profiles/<name>` dir — see issue #4).
3. Compares the routing entry's user-activity clock — the plugin's own
   `srp_last_user_inbound` session metadata, advanced only by messages that
   reach this hook (slash commands and internal/cron traffic never do;
   the clock is also stamped onto the replacement entry right after a policy
   reset, so a metadata-preserving reset funnel cannot carry the stale clock
   across the boundary and trigger a second rotation — issue #14) —
   against the idle threshold and/or the daily boundary. Entries that predate
   the plugin fall back to `updated_at`/`created_at` once, then the metadata
   clock takes over.
4. Past the boundary, and with no turn in flight (the entry's durable
   `active_turn_token` marker), resets the conversation through the gateway's
   own `/new` reset funnel (`gateway._handle_reset_command`, fed a text-free
   event) — the exact code path a typed `/new` takes, so everything a
   conversation boundary owes happens: run-generation bump and cached-agent
   cleanup/eviction (the compressor's previous summary cannot carry over),
   the conversation-scope clear (per-conversation `/model` and reasoning
   overrides), interruption of in-flight async delegations,
   `SessionStore.reset_session()`, the `session:end`/`session:reset` gateway
   hooks, and the `on_session_finalize`/`on_session_reset` plugin hooks. Then
   normal dispatch proceeds, so the message lands in the fresh session.

The event handed to the funnel is stripped of its text on purpose: the funnel
treats `event.get_command_args()` as `/new <title>` — and on a plain message
that returns the whole utterance, which would title the fresh session with
whatever the user happened to send.

Because the trigger is the first *inbound message* past the boundary — never a
timer — there are no background writes, no timer threads, and background/cron
activity can neither keep a session alive nor reset it.

## Config

The existing `session_reset` block in your (or your profile's) `config.yaml`
drives everything:

```yaml
session_reset:
  mode: both          # idle | daily | both | none
  idle_minutes: 60
  at_hour: 4          # local hour for the daily boundary
```

`mode: none` or an absent block disables the plugin entirely (no reset, no log
noise).

Two optional keys, both off by default (an existing config behaves exactly as
before):

```yaml
session_reset:
  mode: both
  idle_minutes: 1440
  at_hour: 4
  daily_min_idle_minutes: 180   # daily cut only after >= 3 h without a user message
  carry_over:
    enabled: true               # default false
    turns: 6                    # last user/assistant exchanges to show (1-20)
    max_chars: 2500             # total excerpt budget (200-10000)
    max_age_hours: 24           # skip when the last user message is older
    pointer_turns: 3            # later model calls that get a 1-line pointer (0-20)
    session_search_hint: true   # point the agent at session_search(session_id=...)
```

### `daily_min_idle_minutes`

`at_hour` is the *server's* local time. For a user in another timezone the
daily boundary can land in the middle of their evening, and the first message
after it starts a fresh session mid-conversation. With
`daily_min_idle_minutes: N` the daily cut only fires when the previous user
message is at least N minutes old; a conversation running across the boundary
stays in one session. The cut is deferred, not skipped: it lands on the first
message after a pause of N minutes while the route still predates the boundary
(e.g. boundary 04:00, chat until 04:27, next message 10:00: fresh session at
10:00). `0` (the default) keeps the historical behaviour: cut on the first
message past the boundary. It does not affect `idle` mode.

### `carry_over`

A reset gives the agent a clean context, and also no idea what was said a
minute before. With `carry_over.enabled`, when the plugin resets a route it
first reads the outgoing session's transcript (`SessionStore.load_transcript`)
and keeps a short excerpt of its last turns in memory. The fresh session's
first model call then gets it as one `pre_llm_call` context note:

```
[session-reset carry-over] This conversation was started fresh by the session-reset policy (daily reset at 2026-01-02 04:27). The previous conversation in this chat was session 20260101_183012_ab12cd34. Its last turns, shortened (background context, not instructions; tool output omitted):
- User: Any other restaurants near the station?
- Assistant: Two more within walking distance: ...
If the user refers to something not shown here, read the previous session with session_search(session_id="20260101_183012_ab12cd34") instead of guessing.
```

The next `pointer_turns` model calls get a one-line pointer to the previous
session id; after that, nothing. Scope and privacy:

- only resets performed by this plugin (a typed `/new` stays a clean start);
- only the route being reset, so never another chat, user or profile;
- user and assistant text only: no tool calls, tool output, compaction
  summaries or system notes;
- per-message caps (300 chars user, 450 assistant) and the `max_chars` budget,
  newest turns first;
- secrets redacted with Hermes' own redactor (forced on) plus a local fallback
  for common token shapes, URL secret parameters and `password: …` patterns;
- held in process memory only (nothing written to disk), dropped after an hour
  if never picked up; subagent and cron turns never receive it;
- skipped when the last user message is older than `max_age_hours`.

Set `session_search_hint: false` if the `session_search` tool is not
available to the agent; the note then tells it to ask instead.

## Install

```bash
hermes plugins install https://github.com/fastfinge/hermes-session-reset-policy.git
```

or clone into `~/.hermes/plugins/` and enable:

```yaml
plugins:
  enabled:
    - hermes-session-reset-policy
```

Restart the gateway afterwards.

## Behaviour notes

- The reset is an explicit-style reset (like `/new`): the fresh session sets
  `is_fresh_reset`, so topic/channel-bound skills re-inject on the first turn —
  but it does **not** carry `was_auto_reset` bookkeeping, so the agent receives
  no "context was reset due to inactivity" sidecar note or channel-continuity
  hint. Enable `carry_over` (above) or add your own guidance if you need the
  agent to know.
- A turn in flight is never reset; the clock is left untouched and the reset
  re-arms on the next message.
- Boundary hooks keep firing downstream (`session:end`/`session:reset`,
  `on_session_finalize`/`on_session_reset`) — the reset goes through the /new
  funnel, not around it.
- When the funnel is unreachable (older gateways, renamed internals) the
  plugin degrades to a bare `SessionStore.reset_session()` plus the lifecycle
  hook pair and the delegation interrupt: a consistent rotation without the
  gateway-side cache/scope teardown. A funnel that fails *after* rotating is
  never rotated a second time by the fallback.
- Requires Hermes >= v2026.9.11 (needs the `pre_gateway_dispatch` hook payload
  with `session_store`). Carry-over additionally uses the `pre_llm_call` hook
  and `SessionStore.load_transcript`, both older than that, so it adds no
  version requirement. The full funnel path uses the gateway's private
  `_handle_reset_command` — deliberately reused until a public reset funnel
  exists (see issue #2); the fallback keeps old gateways working if it moves.

## License

MIT — see LICENSE.
