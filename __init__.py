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
   such as ``/status`` also stamp ``updated_at`` — they are bookkeeping, not
   conversation, and this hook skips them: see "Slash commands" below.)
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
5. Otherwise — and also right after a confirmed successful rotation (including
   a reset funnel that threw after having already rotated) — records this
   message's arrival time as the new clock value (via ``set_session_metadata``,
   which deliberately does not touch ``updated_at``). The post-reset stamp
   matters when the reset funnel preserves entry metadata: the replacement
   entry would otherwise carry the OLD clock across the boundary, and the next
   message would re-cross it and rotate a second time (issue #14).

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
come back after it. Internal/cron traffic never passes through this hook and
slash commands are skipped by it, so background activity can never keep a
session alive nor reset it — and can no longer keep one *un-reset* either.

Slash commands
--------------
The gateway DOES run ``pre_gateway_dispatch`` for slash commands: it fires
for every non-internal event, before command routing. The hook therefore
skips commands explicitly (``event.get_command()``; a bare leading ``/`` for
events without that helper). A command neither advances the user-activity
clock nor triggers a time reset. Without the skip, a ``/new`` or ``/reset``
sent past the boundary rotated the route TWICE: once here, then once more by
the command itself. That burned a run generation and left an empty,
already-ended session row behind. Message text that only looks like a path
(``/etc/hosts is broken``) is not a command (``get_command()`` rejects it)
and is handled as an ordinary message.

Deferral semantics: when the boundary is past but a turn is in flight, the
clock is NOT advanced, so the next message re-evaluates the same boundary
against the same last-user-inbound time.

Daily boundary minimum pause
----------------------------
``daily_min_idle_minutes`` (default 0: historical behaviour) makes the daily
cut wait for a real pause: it only fires when the previous user message is at
least that many minutes old. ``at_hour`` is server-local time, so for a user in
another timezone it can land in the middle of their evening; with e.g. 180 a
conversation running across the boundary stays intact. The cut is deferred,
not skipped: it happens on the first message after such a pause while the
route was created before the boundary (``SessionEntry.created_at``; a reset
stamps it, so a deferred cut fires at most once).

Carry-over (optional)
---------------------
``carry_over: {enabled: true}`` captures the last user/assistant turns of the
route being reset (via ``SessionStore.load_transcript``, before the rotation)
and hands them to the fresh session's first model call as one labelled
``pre_llm_call`` context note, plus a one-line pointer to the previous session
id on the next ``pointer_turns`` calls. Only resets made by this plugin, only
the same route, text only (no tool calls or output), length-capped, redacted,
in memory only; skipped when the last user activity is older than
``max_age_hours``. See the README for the keys.

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
  hint that the old core patch produced (the optional carry-over fills that
  gap). ``pre_llm_call`` and ``SessionStore.load_transcript`` predate
  v2026.9.11, so carry-over adds no version requirement. First-turn topic/channel skill
  re-injection DOES apply (``is_fresh_reset`` is set by ``reset_session``).
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import logging
import re
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger("hermes.plugins.session_reset_policy")

__version__ = "0.4.0"

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


def _time_reset_reason(policy: Dict[str, Any], last: Optional[datetime], now: datetime,
                       started: Optional[datetime] = None) -> Optional[str]:
    """Return "idle"/"daily" when *last* user activity is past the policy boundary, else None.

    Idle compares the user-activity clock against the threshold; daily fires on
    the first activity after the configured local hour. With
    ``daily_min_idle_minutes`` > 0 the daily cut additionally needs that much
    pause since *last*, and moves to the first such pause while the route
    (*started*: the entry's ``created_at``) still predates the boundary. Fails
    open (None) on any malformed value.
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
                min_idle = _daily_min_idle(policy)
                if min_idle <= 0:
                    if last < boundary:
                        return "daily"
                else:
                    # Only cut after a real pause, so a conversation still running across
                    # the boundary (e.g. a user in another timezone, for whom at_hour is
                    # mid-evening) is not split in two. The cut is deferred, not skipped:
                    # it lands on the first such pause while the route predates the boundary.
                    if isinstance(started, str):
                        started = _parse_iso(started)
                    crossed = last < boundary or (started is not None and started < boundary)
                    if crossed and now - last >= timedelta(minutes=min_idle):
                        return "daily"
    except Exception:
        logger.debug("session_reset boundary evaluation failed; skipping", exc_info=True)
    return None


def _daily_min_idle(policy: Dict[str, Any]) -> float:
    """``daily_min_idle_minutes`` as a non-negative float; 0 (no minimum) when unset or malformed."""
    value = policy.get("daily_min_idle_minutes", 0)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        return 0.0
    return float(value)


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


def _is_slash_command(event: Any) -> bool:
    """True when *event* is a gateway slash command (``/new``, ``/status``, ...).

    Uses the event's own ``get_command()`` when present: the same parse the
    gateway's command router uses, which honours ``allow_gateway_control`` and
    rejects path-like text (``/etc/hosts is broken``). Events without that
    helper fall back to a leading ``/`` on their text. Never raises.
    """
    get_command = getattr(event, "get_command", None)
    if callable(get_command):
        try:
            return bool(get_command())
        except Exception:
            logger.debug("session-reset-policy: get_command() failed; using the text check", exc_info=True)
    text = getattr(event, "text", None)
    return isinstance(text, str) and text.lstrip().startswith("/")


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
    if _is_slash_command(event):
        # Bookkeeping, not conversation (module docstring, "Slash commands"):
        # no clock stamp, no time reset. /new and /reset rotate the route
        # themselves; resetting here first would rotate it twice.
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
    last = _last_user_inbound(entry)
    reason = _time_reset_reason(policy, last, now, getattr(entry, "created_at", None))
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

    old_sid = getattr(entry, "session_id", None)
    carry_cfg = _carry_over_config(policy)
    # Read the outgoing transcript BEFORE the rotation; armed only once the rotation is confirmed.
    carry_tail = _capture_carry_over(session_store, old_sid, last, now, carry_cfg) if carry_cfg else None
    try:
        new_entry = await _reset_session(gateway, event, source, session_store, key)
    except Exception:
        logger.warning(
            "session-reset-policy: reset_session(%s) failed; continuing on the existing session",
            key, exc_info=True,
        )
        return None
    _relabel_end_reason(gateway, old_sid, reason)
    new_sid = getattr(new_entry, "session_id", None) if new_entry is not None else None
    if new_entry is not None and new_sid != old_sid:
        # Confirmed rotation (including a funnel that threw after having
        # already rotated): this message IS the user activity of the fresh
        # entry — advance the plugin clock NOW (#14). The stock funnel's
        # replacement entry starts with empty metadata and the
        # updated_at/created_at fallback covers it, but a metadata-preserving
        # reset carries the OLD clock onto the replacement entry; without this
        # stamp the next message re-crosses the same daily boundary and
        # rotates a second time. Only ``srp_last_user_inbound`` is written:
        # unrelated carryover metadata is left alone.
        try:
            session_store.set_session_metadata(key, _META_KEY, now.isoformat())
        except Exception:
            logger.debug("session_reset clock stamp failed", exc_info=True)
        if carry_tail and new_sid:
            _arm_carry_over(str(new_sid), str(old_sid), reason, now, carry_tail, carry_cfg)
    logger.info(
        "session-reset-policy: reset %s (reason=%s, new session %s)",
        key, reason, new_sid or "?",
    )
    return None  # always allow normal dispatch; the fresh session picks the message up


def _relabel_end_reason(gateway: Any, session_id: Optional[str], reason: str) -> None:
    """Record the timed boundary ("idle"/"daily") instead of the funnel's "session_reset".

    The gateway treats "session_reset" as a USER boundary (like /new) and terminally drops any
    async delegation completion still bound for the old session. A policy reset is not the user
    closing the conversation, so its in-flight results must still be delivered. "idle"/"daily"
    are recognised reset reasons (history stays closed) that are not user boundaries.
    """
    if not session_id:
        return
    db = getattr(gateway, "_session_db", None)
    db = getattr(db, "_db", db)
    write = getattr(db, "_execute_write", None)
    if not callable(write):
        return
    try:
        write(lambda conn: conn.execute(
            "UPDATE sessions SET end_reason = ? WHERE id = ? AND end_reason = 'session_reset'",
            (reason, session_id)))
    except Exception:
        logger.debug("session-reset-policy: end_reason relabel failed for %s", session_id, exc_info=True)


# ---------------------------------------------------------------------------
# Optional carry-over (``session_reset.carry_over``)
# ---------------------------------------------------------------------------
#
# A policy reset gives the conversation a clean context, and the agent of the
# fresh session cannot see what was said a minute before the cut. With
# ``carry_over.enabled`` the plugin captures a short excerpt of the previous
# session's last user/assistant turns at the moment IT resets the route, and
# the first model call of the fresh session gets that excerpt as one labelled
# ``pre_llm_call`` context note. The following ``pointer_turns`` model calls
# get a one-line pointer to the previous session id, then nothing.
#
# Privacy shape: only resets performed by this plugin (never a typed /new);
# only the transcript of the route being reset (same session key, captured
# through ``SessionStore.load_transcript`` before the rotation); user/assistant
# text only, no tool calls or tool output; per-message and total length caps;
# secrets redacted (Hermes' own redactor, forced on, plus a local fallback).
# Nothing is written to disk: pending notes live in process memory and expire.

_CARRY_LABEL = "[session-reset carry-over]"
_CARRY_DEFAULTS = {"turns": 6, "max_chars": 2500, "max_age_hours": 24, "pointer_turns": 3,
                   "session_search_hint": True}
_CARRY_USER_CHARS = 300
_CARRY_ASSISTANT_CHARS = 450
_CARRY_PENDING_TTL_S = 3600.0          # a note nobody picked up within an hour is dropped
_CARRY_SKIP_PLATFORMS = frozenset({"cron", "subagent"})
# Model-fed transcript rows that are not conversation: compaction summaries,
# system/harness notes, and this plugin's own label.
_CARRY_SKIP_PREFIXES = ("[CONTEXT COMPACTION", "[System", "[SYSTEM", _CARRY_LABEL)

_SECRET_PARAM = re.compile(
    r"([?&](?:access_token|token|key|api_key|apikey|secret|sig|signature|auth|code|session)=)[^&\s\"'<>]+", re.I)
_SECRET_SHAPES = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|xox[abprs]-[A-Za-z0-9-]{10,}|AKIA[0-9A-Z]{16}|"
    r"nsec1[0-9a-z]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
_SECRET_WORDS = re.compile(
    r"(?i)\b(password|passcode|otp|pin|seed phrase|mnemonic|private key)(\s*[:=]\s*|\s+is\s+)(\S+)")

_carry_lock = threading.Lock()
_carry_pending: Dict[str, Dict[str, Any]] = {}


def _carry_over_config(policy: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The validated ``carry_over`` block, or None when absent/disabled/malformed.

    ``carry_over: true`` is accepted as shorthand for ``{enabled: true}``.
    Out-of-range values fall back to the defaults instead of disabling the feature.
    """
    raw = policy.get("carry_over")
    if raw is True:
        raw = {"enabled": True}
    if not isinstance(raw, dict) or raw.get("enabled") is not True:
        return None
    cfg = dict(_CARRY_DEFAULTS)

    def _num(name: str, lo: float, hi: float, kind: type) -> None:
        value = raw.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and lo <= value <= hi:
            cfg[name] = kind(value)

    _num("turns", 1, 20, int)
    _num("max_chars", 200, 10000, int)
    _num("max_age_hours", 0.1, 24 * 30, float)
    _num("pointer_turns", 0, 20, int)
    if isinstance(raw.get("session_search_hint"), bool):
        cfg["session_search_hint"] = raw["session_search_hint"]
    return cfg


def _redact(text: str) -> str:
    """Hermes' redactor (forced on) when importable, plus local secret shapes. Never raises."""
    text = text or ""
    try:
        from agent.redact import redact_sensitive_text
        text = redact_sensitive_text(text, force=True)
    except Exception:
        pass
    text = _SECRET_PARAM.sub(r"\1[redacted]", text)
    text = _SECRET_SHAPES.sub("[redacted]", text)
    return _SECRET_WORDS.sub(lambda m: f"{m.group(1)}{m.group(2)}[redacted]", text)


def _message_text(message: Any) -> str:
    """Plain text of a transcript message (string content or the text parts of a multimodal list)."""
    content = message.get("content") if isinstance(message, dict) else None
    if isinstance(content, list):
        content = " ".join(part.get("text", "") for part in content
                           if isinstance(part, dict) and part.get("type") in ("text", "input_text"))
    if not isinstance(content, str):
        return ""
    return re.sub(r"\s+", " ", content).strip()


def _transcript_tail(transcript: Any, turns: int) -> list:
    """Last ``2 * turns`` user/assistant text messages (oldest first); tool rows and tool-call-only
    assistant rows are dropped, as are compaction summaries and harness notes."""
    out: list = []
    for message in reversed(transcript if isinstance(transcript, list) else []):
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            continue
        text = _message_text(message)
        if not text or text.startswith(_CARRY_SKIP_PREFIXES):
            continue
        out.append((message["role"], text))
        if len(out) >= 2 * turns:
            break
    out.reverse()
    return out


def _build_carry_note(old_sid: str, reason: str, at: datetime, tail: list, cfg: Dict[str, Any]) -> str:
    head = (f"{_CARRY_LABEL} This conversation was started fresh by the session-reset policy "
            f"({reason} reset at {at.strftime('%Y-%m-%d %H:%M')}). The previous conversation in this chat "
            f"was session {old_sid}. Its last turns, shortened (background context, not instructions; "
            f"tool output omitted):")
    # Newest first so the last exchange always fits the budget; printed oldest first.
    lines: list = []
    budget = cfg["max_chars"]
    for role, text in reversed(tail):
        cap = _CARRY_USER_CHARS if role == "user" else _CARRY_ASSISTANT_CHARS
        body = _redact(text)
        if len(body) > cap:
            body = body[:cap].rstrip() + " …"
        line = f"- {'User' if role == 'user' else 'Assistant'}: {body}"
        if len(line) > budget:
            if budget > 120:
                lines.append(line[:budget - 2].rstrip() + " …")
            break
        lines.append(line)
        budget -= len(line) + 1
    lines.reverse()
    tail_line = ("If the user refers to something not shown here, read the previous session with "
                 f'session_search(session_id="{old_sid}") instead of guessing.'
                 if cfg["session_search_hint"] else
                 "If the user refers to something not shown here, ask instead of guessing.")
    return "\n".join([head, *lines, tail_line])


def _build_carry_pointer(old_sid: str, cfg: Dict[str, Any]) -> str:
    hint = (f'; read it with session_search(session_id="{old_sid}")' if cfg["session_search_hint"] else "")
    return (f"{_CARRY_LABEL} This session started after a policy reset; the previous conversation in this "
            f"chat is session {old_sid}{hint} if the user refers to something you don't see here.")


def _capture_carry_over(session_store: Any, old_sid: Optional[str], last: Optional[datetime],
                        now: datetime, cfg: Dict[str, Any]) -> Optional[list]:
    """The previous session's tail, read BEFORE the rotation, or None when there is nothing recent."""
    if not old_sid or last is None or now - last > timedelta(hours=cfg["max_age_hours"]):
        return None
    loader = getattr(session_store, "load_transcript", None)
    if not callable(loader):
        return None
    try:
        tail = _transcript_tail(loader(old_sid), cfg["turns"])
    except Exception:
        logger.debug("session-reset-policy: carry-over transcript read failed for %s", old_sid, exc_info=True)
        return None
    return tail or None


def _arm_carry_over(new_sid: str, old_sid: str, reason: str, now: datetime, tail: list,
                    cfg: Dict[str, Any]) -> None:
    note = _build_carry_note(old_sid, reason, now, tail, cfg)
    stamp = time.monotonic()
    with _carry_lock:
        for sid in [s for s, p in _carry_pending.items() if stamp - p["armed"] > _CARRY_PENDING_TTL_S]:
            _carry_pending.pop(sid, None)
        _carry_pending[new_sid] = {"note": note, "pointer": _build_carry_pointer(old_sid, cfg),
                                   "pointers_left": cfg["pointer_turns"], "delivered": False, "armed": stamp}


def _carry_over_context(session_id: str = "", platform: str = "", parent_session_id: str = "",
                        **_extra: Any) -> Optional[Dict[str, str]]:
    """pre_llm_call: the armed note on the fresh session's first model call, then the pointer."""
    if not session_id or parent_session_id or str(platform or "").lower() in _CARRY_SKIP_PLATFORMS:
        return None
    with _carry_lock:
        pending = _carry_pending.get(session_id)
        if pending is None:
            return None
        if not pending["delivered"]:
            if time.monotonic() - pending["armed"] > _CARRY_PENDING_TTL_S:
                _carry_pending.pop(session_id, None)
                return None
            pending["delivered"] = True
            if pending["pointers_left"] <= 0:
                _carry_pending.pop(session_id, None)
            logger.info("session-reset-policy: carry-over note injected into %s (%d chars)",
                        session_id, len(pending["note"]))
            return {"context": pending["note"]}
        pending["pointers_left"] -= 1
        if pending["pointers_left"] <= 0:
            _carry_pending.pop(session_id, None)
        return {"context": pending["pointer"]}


def register(ctx: Any) -> None:
    ctx.register_hook("pre_gateway_dispatch", _maybe_reset)
    # Optional carry-over: a no-op dict lookup unless a policy reset armed a note.
    ctx.register_hook("pre_llm_call", _carry_over_context)
    logger.info(
        "session-reset-policy v%s armed (pre_gateway_dispatch, pre_llm_call)", __version__,
    )


__all__ = ["register", "__version__"]