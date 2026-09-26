"""Hermes session-reset-policy plugin.

Re-arms the config-declared ``session_reset`` policy (mode: idle|daily|both)
that upstream Hermes v2026.9.11 (commit 1d5d059410) made inert by removing the
time-triggered conversation-rotation timers. Upstream's stance is that only
explicit suspension should replace a routed conversation; this plugin restores
the time-based policy for operators who want it, as an opt-in plugin instead of
a core patch.

How it works
------------
On every inbound gateway message (``pre_gateway_dispatch``), before auth and
dispatch, the plugin:

1. Computes the session key for the message source (same helper the gateway
   uses, so multiplexed profiles and platform key layouts match exactly).
2. Reads that profile's ``config.yaml`` ``session_reset`` block. Under
   multiplexing the hook fires before the per-profile runtime scope is
   installed, so the profile home is resolved from ``event.source.profile``
   and the config is read directly from ``profiles/<name>/config.yaml``.
3. Compares the route's OWN user-activity clock — ``srp_last_user_inbound``
   session metadata, advanced only by real user messages that reach this hook
   — against the idle threshold and/or daily boundary. (Why not the entry's
   ``updated_at``: turn-start stamps it for EVERY turn, including internal
   background-review turns, so on an idle gateway where reviews chain the
   clock never goes stale and the idle reset can never fire. Slash commands
   such as ``/status`` also stamp ``updated_at`` while never reaching this
   hook — they are bookkeeping, not conversation.)
4. If past the boundary and no turn is currently in flight for that session,
   calls ``SessionStore.reset_session(session_key)`` — the same store path the
   ``/new`` command uses — so the new conversation starts on a fresh
   session id with the route, transcript end, and state.db row handled by the
   store's own transition logic.
5. Otherwise records this message's arrival time as the new clock value
   (via ``set_session_metadata``, which deliberately does not touch
   ``updated_at``).

Migration: routes that predate the plugin have no metadata clock; their first
hook-seen message falls back to ``updated_at``/``created_at`` once, then the
metadata clock takes over.

Because the reset happens lazily on the first *inbound message* past the
boundary (not on a timer), a session that goes idle is simply continued when
the user comes back within the threshold, and reset exactly once when they
come back after it. Internal/cron traffic and slash commands never pass
through this hook, so background activity can never keep a session alive nor
reset it — and can no longer keep one *un-reset* either.

Deferral semantics: when the boundary is past but a turn is in flight, the
clock is NOT advanced, so the next message re-evaluates the same boundary
against the same last-user-inbound time.

Compatibility notes
-------------------
- Requires Hermes >= v2026.9.11 (needs ``pre_gateway_dispatch`` hook payload
  with ``session_store``, and ``SessionStore.reset_session``).
- The reset is an *explicit-style* reset (like ``/new``): the fresh session
  does not carry ``was_auto_reset``/``auto_reset_reason`` bookkeeping, so the
  agent does not receive the auto-reset context note or channel-continuity
  hint that the old core patch produced. First-turn topic/channel skill
  re-injection DOES apply (``is_fresh_reset`` is set by ``reset_session``).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.session_reset_policy")

__version__ = "0.2.0"

_META_KEY = "srp_last_user_inbound"


def _read_session_reset_policy(profile: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the ``session_reset`` block for *profile*, or None when unset/invalid.

    ``pre_gateway_dispatch`` fires before the per-profile runtime scope is
    installed, so the ambient ``load_config_readonly()`` would read the launch
    profile. Instead the profile's home (``<hermes_home>/profiles/<name>``) is
    installed as a context-local hermes-home override — the same mechanism the
    gateway uses for profile scoping — and the config cache is keyed by the
    resolved config path, so each profile's block is read from its own
    config.yaml. The returned dict is the SHARED cache object (readonly
    variant): never mutated. Never raises.
    """
    try:
        from hermes_constants import get_hermes_home, set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config import load_config_readonly

        token = None
        if profile and profile != "default":
            profile_home = Path(get_hermes_home()) / "profiles" / profile
            token = set_hermes_home_override(profile_home)
        try:
            config = load_config_readonly()
        finally:
            if token is not None:
                reset_hermes_home_override(token)
        policy = (config or {}).get("session_reset")
        if not isinstance(policy, dict) or not policy:
            return None  # unset/empty block: treat as "none", skip quietly
        return policy
    except Exception:
        logger.debug("session_reset policy read failed; skipping", exc_info=True)
        return None


def _parse_iso(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _last_user_inbound(entry: Any) -> Optional[datetime]:
    """The route's user-activity clock.

    Order: the plugin's own ``srp_last_user_inbound`` metadata (written only by
    messages that reach this hook), then ``updated_at``/``created_at`` as a
    one-time migration fallback for routes that predate the plugin.
    """
    meta = getattr(entry, "metadata", None) or {}
    stamp = _parse_iso(meta.get(_META_KEY)) if isinstance(meta, dict) else None
    if stamp is not None:
        return stamp
    return getattr(entry, "updated_at", None) or getattr(entry, "created_at", None)


def _time_reset_reason(policy: Dict[str, Any], last: Optional[datetime], now: datetime) -> Optional[str]:
    """Return "idle"/"daily" when *last* user activity is past the policy boundary, else None.

    Idle compares the user-activity clock against the threshold; daily fires on
    the first activity after the configured local hour. Fails open (None) on
    any malformed value.
    """
    try:
        mode = str(policy.get("mode") or "none")
        if mode == "none":
            return None
        if last is None:
            return None
        if not isinstance(last, datetime):
            last = _parse_iso(last)
            if last is None:
                return None
        if mode in ("idle", "both"):
            idle_minutes = policy.get("idle_minutes")
            if isinstance(idle_minutes, (int, float)) and idle_minutes > 0:
                if now - last >= timedelta(minutes=float(idle_minutes)):
                    return "idle"
        if mode in ("daily", "both"):
            at_hour = policy.get("at_hour")
            if isinstance(at_hour, int) and 0 <= at_hour <= 23:
                boundary = now.replace(hour=at_hour, minute=0, second=0, microsecond=0)
                if boundary > now:
                    boundary -= timedelta(days=1)
                if last < boundary:
                    return "daily"
    except Exception:
        logger.debug("session_reset boundary evaluation failed; skipping", exc_info=True)
    return None


def _session_in_flight(entry: Any) -> bool:
    """True when the entry's durable active-turn marker says a turn is running.

    ``active_turn_token`` is set under the store's lock when a turn starts and
    CAS-cleared on normal unwind; a stale marker left by a crash is recovered at
    next startup, so treating it as in-flight is the conservative choice.
    """
    return bool(getattr(entry, "active_turn_token", None))


def _maybe_reset(event: Any, gateway: Any, session_store: Any) -> Optional[Dict[str, str]]:
    """pre_gateway_dispatch body: reset the session when the policy says so."""
    if session_store is None:
        return None
    source = getattr(event, "source", None)
    if source is None:
        return None

    profile = getattr(source, "profile", None)
    policy = _read_session_reset_policy(profile)
    if not policy:
        return None

    session_key = None
    entry = None
    try:
        session_key = gateway._session_key_for_source(source)
    except Exception:
        session_key = None  # never wedge dispatch on a key-derivation failure
    if session_key:
        try:
            entry = session_store.lookup_by_session_key(session_key)
        except Exception:
            entry = None
    if entry is None:
        return None  # no route yet -> brand-new conversation anyway

    now = datetime.now()
    reason = _time_reset_reason(policy, _last_user_inbound(entry), now)
    if reason is None:
        # Not past the boundary: this message IS the user activity — advance
        # our own clock (never ``updated_at``; see module docstring).
        try:
            session_store.set_session_metadata(session_key, _META_KEY, now.isoformat())
        except Exception:
            logger.debug("session_reset clock stamp failed", exc_info=True)
        return None
    if _session_in_flight(entry):
        logger.info(
            "session-reset-policy: boundary '%s' reached for %s but a turn is in flight; deferring (clock NOT advanced)",
            reason, session_key,
        )
        return None  # normal dispatch; the next message re-evaluates the same boundary

    try:
        new_entry = session_store.reset_session(session_key)
    except Exception:
        logger.warning(
            "session-reset-policy: reset_session(%s) failed; continuing on the existing session",
            session_key, exc_info=True,
        )
        return None
    new_sid = getattr(new_entry, "session_id", "?") if new_entry is not None else "?"
    logger.info(
        "session-reset-policy: reset %s (reason=%s, new session %s)",
        session_key, reason, new_sid,
    )
    return None  # always allow normal dispatch; the fresh session picks the message up


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _maybe_reset)
    logger.info(
        "session-reset-policy v%s armed (pre_gateway_dispatch)", __version__,
    )


__all__ = ["register", "__version__"]
