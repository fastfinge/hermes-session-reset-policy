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

1. Computes the session key for the message (the gateway's own helper, so
   multiplexed profiles and platform key layouts match exactly).
2. Reads the profile's `config.yaml` `session_reset` block (under
   `multiplex_profiles`, the profile is resolved from the message's
   `source.profile` and read from `profiles/<name>/config.yaml`).
3. Compares the routing entry's user-activity clock (`updated_at`, advanced
   only by real user turns — internal events pass `touch_activity=False`)
   against the idle threshold and/or the daily boundary.
4. Past the boundary, and with no turn in flight (the entry's durable
   `active_turn_token` marker), calls `SessionStore.reset_session()` — the same
   path `/new` uses — then allows normal dispatch, so the message lands in the
   fresh session.

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
- A turn in flight is never reset; the reset re-arms on the next message.
- Requires Hermes >= v2026.9.11 (needs the `pre_gateway_dispatch` hook payload
  with `session_store`).

## License

MIT — see LICENSE.
