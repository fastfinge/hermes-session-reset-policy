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
   resets the conversation through the gateway's OWN ``/new`` reset funnel
   (``gateway._handle_reset_command``, fed a text-free event): the exact path
   a typed ``/new`` takes, so everything a conversation boundary owes is done
   — run-generation bump and cached-agent cleanup/eviction (the compressor's
   previous summary cannot carry over), the conversation-scope clear
   (per-conversation ``/model`` and reasoning overrides), interruption of
   in-flight async delegations, the durable store rotation
   (``SessionStore.reset_session``: route, transcript end, state.db row), the
   ``session:end``/``session:reset`` gateway hooks, and the
   ``on_session_finalize``/``on_session_reset`` plugin hooks (review #2).
   When the funnel is not reachable (older/renamed gateway internals) the
   plugin degrades to a bare ``SessionStore.reset_session()`` plus the
   lifecycle hook pair and the delegation interrupt — a consistent rotation,
   without the gateway-side cache/scope teardown.
5. Otherwise records this message's arrival time as the new clock value
   (via ``set_session_metadata``, which deliberately does not touch
   ``updated_at``).

The event handed to the reset funnel is stripped of its text: the funnel
treats ``event.get_command_args()`` as ``/new <title>``, and on a plain
message that returns the whole utterance — left intact, a policy reset would
title the fresh session with whatever the user happened to send.

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
  with ``session_store``, and ``SessionStore.reset_session``). The full funnel
  path additionally uses ``gateway._handle_reset_command`` (present ever since
  ``/new`` exists; private, per review #2 reused deliberately until a public
  reset funnel exists).
- Session key resolution: the hook payload's ``session_key`` first (when a
  future Hermes provides a public one), else ``gateway._session_key_for_source``
  (review #2: "the right helper today", private but blessed). There is NO bare
  ``build_session_key`` fallback on purpose: under multiplexed profiles that
  key is not profile-namespaced and could collide with ANOTHER profile's
  route — a wrong-key reset would be worse than no reset.
- The reset is an *explicit-style* reset (like ``/new``): the fresh session
  does not carry ``was_auto_reset``/``auto_reset_reason`` bookkeeping, so the
  agent does not receive the auto-reset context note or channel-continuity
  hint that the old core patch produced. First-turn topic/channel skill
  re-injection DOES apply (``is_fresh_reset`` is set by ``reset_session``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.session_reset_policy")

__version__ = "0.3.0"

_META_KEY = "srp_last_user_inbound"


def _read_session_reset_policy(profile: Optional[str]) -> Optional[Dict[str, Any]]:
    """Return the ``session_reset`` block for *profile*, or None when unset/invalid.

    ``pre_gateway_dispatch`` fires before the per-profile runtime scope is
    installed, so the ambient ``load_config_readonly()`` would read the launch
    profile. Instead the profile's home is installed as a context-local
    hermes-home override — the same mechanism the gateway uses for profile
    scoping — and the config cache is keyed by the resolved config path, so
    each profile's block is read from its own config.yaml. The home is
    resolved via :func:`hermes_cli.profiles.get_profile_dir`, which anchors
    names to the Hermes ROOT (not the current home) and handles ``default``
    as the root itself: a standalone gateway may run with ``HERMES_HOME``
    already set to its own profile dir, where ``<home>/profiles/<name>`` would
    double-nest and find nothing (issue #4). The returned dict is the SHARED
    cache object (readonly variant): never mutated. Never raises.
    """
    try:
        from hermes_constants import set_hermes_home_override, reset_hermes_home_override
        from hermes_cli.config import load_config_readonly
        from hermes_cli.profiles import get_profile_dir

        token = None
        if profile:
            token = set_hermes_home_override(get_profile_dir(profile))
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


def _lookup_entry(session_store: Any, session_key: Optional[str]) -> Any:
    """Best-effort route lookup; None on any failure (fail open to "no route")."""
    if not session_key:
        return None
    try:
        return session_store.lookup_by_session_key(session_key)
    except Exception:
        return None


def _resolve_session_key(gateway: Any, source: Any, from_payload: Optional[str]) -> Optional[str]:
    """The routing key for *source*: the hook payload's ``session_key`` when a
    future Hermes exposes one publicly, else ``gateway._session_key_for_source``.

    The gateway helper is private but blessed by review #2 ("the right helper
    today"). Deliberately NO bare ``build_session_key`` fallback: that key is
    not profile-namespaced under multiplexing and could resolve ANOTHER
    profile's route for the same platform/chat — fail closed instead.
    """
    if from_payload:
        return str(from_payload)
    helper = getattr(gateway, "_session_key_for_source", None)
    if not callable(helper):
        logger.debug("session-reset-policy: no session-key helper in hook payload; skipping")
        return None
    try:
        key = helper(source)
        return str(key) if key else None
    except Exception:
        logger.debug("session-reset-policy: session key derivation failed; skipping", exc_info=True)
        return None


class _CommandFreeEvent:
    """Minimal event stand-in: same source, no command arguments."""

    __slots__ = ("source",)

    def __init__(self, source: Any) -> None:
        self.source = source

    def get_command_args(self) -> str:
        return ""


def _command_free_event(event: Any) -> Any:
    """A copy of *event* that reads as a bare command with no arguments.

    The reset funnel treats ``event.get_command_args()`` as ``/new <title>`` and
    would title the fresh session from it — and ``MessageEvent.get_command_args``
    returns the WHOLE message text for a non-command. Stripping the text keeps a
    policy reset identical to a plain ``/new``. Falls back to a source-only
    stand-in for non-dataclass events.
    """
    try:
        return dataclasses.replace(event, text="")
    except Exception:
        return _CommandFreeEvent(getattr(event, "source", None))


def _reset_via_store(source: Any, session_store: Any, session_key: str) -> Any:
    """Degraded reset without the gateway funnel: rotate the store and fire what
    is reachable from the plugin surface — in-flight async delegations end with
    the old conversation (as in ``/new``), then the ``on_session_finalize`` /
    ``on_session_reset`` lifecycle hooks report the boundary. No run-generation
    bump, cached-agent eviction, conversation-scope clear, or gateway
    ``session:end``/``session:reset`` hooks: those live on the funnel path.
    """
    old_entry = _lookup_entry(session_store, session_key)
    old_sid = getattr(old_entry, "session_id", None)
    with contextlib.suppress(Exception):
        from tools.async_delegation import interrupt_for_session
        interrupt_for_session(session_key=session_key, reason="session_reset",
                              parent_session_id=str(old_sid or ""))
    new_entry = session_store.reset_session(session_key)
    new_sid = getattr(new_entry, "session_id", None) if new_entry is not None else None
    try:
        from hermes_cli.lifecycle import invoke_hook
        platform = str(getattr(getattr(source, "platform", None), "value", "") or "")
        invoke_hook("on_session_finalize", session_id=old_sid, platform=platform,
                    reason="new_session", old_session_id=old_sid, new_session_id=new_sid)
        invoke_hook("on_session_reset", session_id=new_sid, platform=platform,
                    reason="new_session", old_session_id=old_sid, new_session_id=new_sid)
    except Exception:
        logger.debug("session-reset-policy: lifecycle boundary hooks failed", exc_info=True)
    return new_entry


async def _reset_session(gateway: Any, event: Any, source: Any, session_store: Any,
                         session_key: str) -> Any:
    """Rotate *session_key* through the gateway's ``/new`` reset funnel when
    reachable, else the degraded store rotation. Returns the new entry or None.
    """
    funnel = getattr(gateway, "_handle_reset_command", None)
    if callable(funnel):
        old_sid = getattr(_lookup_entry(session_store, session_key), "session_id", None)
        try:
            outcome = funnel(_command_free_event(event))
            if inspect.isawaitable(outcome):
                await outcome
            return _lookup_entry(session_store, session_key)
        except Exception:
            # Failure AFTER the rotation (e.g. banner/tip bookkeeping in the
            # funnel's tail) must not rotate a second time.
            new_entry = _lookup_entry(session_store, session_key)
            new_sid = getattr(new_entry, "session_id", None)
            if new_entry is not None and new_sid != old_sid:
                logger.warning(
                    "session-reset-policy: gateway reset funnel failed after rotating %s; "
                    "keeping its result", session_key, exc_info=True)
                return new_entry
            logger.warning(
                "session-reset-policy: gateway reset funnel failed for %s; falling back to a "
                "bare store rotation", session_key, exc_info=True)
    else:
        logger.debug(
            "session-reset-policy: gateway reset funnel not available for %s; using the "
            "degraded store rotation", session_key)
    return _reset_via_store(source, session_store, session_key)


async def _maybe_reset(event: Any, gateway: Any, session_store: Any,
                       session_key: Optional[str] = None, **_extra: Any) -> Optional[Dict[str, str]]:
    """pre_gateway_dispatch body: reset the session when the policy says so.

    ``session_key`` is consumed only if a future Hermes puts a public one in
    the hook payload (the dispatcher forwards just the payload fields a
    callback declares); otherwise resolution falls to the gateway helper.
    """
    if session_store is None:
        return None
    source = getattr(event, "source", None)
    if source is None:
        return None

    profile = getattr(source, "profile", None)
    policy = _read_session_reset_policy(profile)
    if not policy:
        return None
    key = _resolve_session_key(gateway, source, session_key)
    entry = _lookup_entry(session_store, key)
    if key is None or entry is None:
        return None  # no key or no route yet -> brand-new conversation anyway

    now = datetime.now()
    reason = _time_reset_reason(policy, _last_user_inbound(entry), now)
    if reason is None:
        # Not past the boundary: this message IS the user activity — advance
        # our own clock (never ``updated_at``; see module docstring).
        try:
            session_store.set_session_metadata(key, _META_KEY, now.isoformat())
        except Exception:
            logger.debug("session_reset clock stamp failed", exc_info=True)
        return None
    if _session_in_flight(entry):
        logger.info(
            "session-reset-policy: boundary '%s' reached for %s but a turn is in flight; deferring (clock NOT advanced)",
            reason, key,
        )
        return None  # normal dispatch; the next message re-evaluates the same boundary

    try:
        new_entry = await _reset_session(gateway, event, source, session_store, key)
    except Exception:
        logger.warning(
            "session-reset-policy: reset_session(%s) failed; continuing on the existing session",
            key, exc_info=True,
        )
        return None
    new_sid = getattr(new_entry, "session_id", "?") if new_entry is not None else "?"
    logger.info(
        "session-reset-policy: reset %s (reason=%s, new session %s)",
        key, reason, new_sid,
    )
    return None  # always allow normal dispatch; the fresh session picks the message up


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _maybe_reset)
    logger.info(
        "session-reset-policy v%s armed (pre_gateway_dispatch)", __version__,
    )


__all__ = ["register", "__version__"]