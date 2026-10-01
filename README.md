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
   reach this hook (slash commands and internal/cron traffic never do) —
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
drives everything; the plugin adds no keys:

```yaml
session_reset:
  mode: both          # idle | daily | both | none
  idle_minutes: 60
  at_hour: 4          # local hour for the daily boundary
```

`mode: none` or an absent block disables the plugin entirely (no reset, no log
noise).

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
  hint. Add your own guidance if you need the agent to know.
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
  with `session_store`). The full funnel path uses the gateway's private
  `_handle_reset_command` — deliberately reused until a public reset funnel
  exists (see issue #2); the fallback keeps old gateways working if it moves.

## License

MIT — see LICENSE.
