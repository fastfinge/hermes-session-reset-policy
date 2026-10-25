"""Tests for the hermes-session-reset-policy plugin.

Standalone-suite discipline (per the platform-plugin skill):
- pytest must be able to COLLECT this file without any Hermes import
  succeeding. All hermes imports are therefore behind helpers inside tests,
  never at module scope.
- Real classes from the installed Hermes tree are imported lazily inside
  helpers, so the tests exercise the actual SessionStore/SessionEntry
  behavior (routing transitions, reset_session) rather than mocks.
- The reset path has two layers (review #2): the gateway's /new reset funnel
  (``gateway._handle_reset_command``) when reachable, and a degraded bare
  store rotation otherwise. Both are covered; the funnel layer additionally
  runs the REAL mixin body bound to a minimal runner stub.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"
HERMES_TREE = Path.home() / ".hermes" / "hermes-agent"


def _run(coro):
    """Drive the (async) hook body to completion; the suite runs without pytest-asyncio."""
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def _temp_hermes_home(tmp_path, monkeypatch):
    """Redirect HERMES_HOME for every test.

    The real SessionStore opens <hermes_home>/state.db; Hermes' live-system
    guard correctly refuses any test-context open of the production DB.
    A temporary HERMES_HOME is resolved at call time (hermes_state._default_db_path),
    so the redirect reaches session storage regardless of import order, and
    the guard passes because the temp root is not the platform default.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))
    # Hermetic root inference (issue #4 repro exposed this): on hosts whose
    # TMPDIR sits under the native ~/.hermes (e.g. the Hermes scratch dir),
    # pytest tmp roots read as part of the DEFAULT profile tree — root
    # inference then anchors at ~/.hermes and profile-name resolution
    # (hermes_cli.profiles.get_profile_dir) escapes the sandbox into the LIVE
    # profiles/ dir. Shift the platform-default suffix so every tmp root is a
    # self-contained external root.
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", f"/pytest-{tmp_path.name}")
    try:
        import hermes_constants
        monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
        monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)
    except Exception:
        pass  # tree not importable here: the tests skip anyway


@pytest.fixture(autouse=True)
def hook_recorder(monkeypatch, _temp_hermes_home):
    """Record — never execute — the boundary side effects the reset path fires
    through host modules: ``hermes_cli.lifecycle.invoke_hook`` (the
    on_session_finalize / on_session_reset plugin hooks) and
    ``tools.async_delegation.interrupt_for_session``. Both are patched on the
    source module, and the plugin imports them lazily at call time, so the
    recorders intercept every path (funnel tail and degraded fallback)."""
    rec = SimpleNamespace(lifecycle=[], interrupts=[])
    if HERMES_TREE.is_dir():
        sys.path.insert(0, str(HERMES_TREE))
    try:
        import hermes_cli.lifecycle as lifecycle
        monkeypatch.setattr(
            lifecycle, "invoke_hook",
            lambda name, **kw: rec.lifecycle.append((name, kw)) or [])
    except Exception:
        pass  # tree not importable here: the plugin's own try/suppress covers it
    try:
        import tools.async_delegation as async_delegation
        monkeypatch.setattr(
            async_delegation, "interrupt_for_session",
            lambda **kw: rec.interrupts.append(kw) or 0)
    except Exception:
        pass
    return rec


@pytest.fixture()
def plugin():
    """Load the plugin module standalone (spec_from_file_location shape)."""
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.hermes_session_reset_policy", PLUGIN_INIT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def _import_tree_module(name: str) -> Any:
    """Import *name* from the installed Hermes tree, or skip the test."""
    if not HERMES_TREE.is_dir():
        pytest.skip("Hermes tree not present on this host")
    sys.path.insert(0, str(HERMES_TREE))
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.skip(f"Hermes tree module not importable: {name}: {exc}")


# ---------------------------------------------------------------------------
# _time_reset_reason — boundary math (ported verbatim from the core patch)
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 26, 12, 0, 0)


def _last(updated_minutes_ago):
    """A user-activity clock value N minutes before NOW."""
    if updated_minutes_ago is None:
        return None
    return NOW - timedelta(minutes=updated_minutes_ago)

def _entry(updated_minutes_ago: float, *, token=None):
    from dataclasses import dataclass, field

    @dataclass
    class Entry:
        created_at: datetime = NOW - timedelta(minutes=updated_minutes_ago or 1)
        updated_at: datetime = NOW - timedelta(minutes=updated_minutes_ago or 1)
        active_turn_token: str | None = token
        session_key: str = "agent:main:telegram:dm:test"
        session_id: str = "s-1"

    return Entry()


def test_idle_mode_triggers_past_threshold(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(90), NOW) == "idle"

def test_idle_mode_no_reset_within_threshold(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(30), NOW) is None

def test_idle_mode_exactly_at_threshold_triggers(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _last(60), NOW) == "idle"

def test_daily_mode_boundary(plugin):
    # now=12:00, boundary hour=4 -> boundary today 04:00; last= yesterday -> reset
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 10, 0),
        updated_at=datetime(2026, 9, 25, 10, 0),
    )
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 25, 10, 0), NOW) == "daily"
    # last after today's boundary -> no reset
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 8, 0), NOW) is None

def test_daily_mode_boundary_in_future_rolls_back_a_day(plugin):
    # now=02:00, boundary hour=4 (today 04:00 is in the future) -> boundary
    # yesterday 04:00; last=yesterday 12:00 (after it) -> no reset.
    now = datetime(2026, 9, 26, 2, 0)
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 12, 0),
        updated_at=datetime(2026, 9, 25, 12, 0),
    )
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 25, 12, 0), now) is None
    # last=day-before-yesterday -> reset
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 24, 12, 0), now) == "daily"

def test_daily_min_idle_default_keeps_historical_cut(plugin):
    # No daily_min_idle_minutes: the first message past the boundary cuts, even 1 minute after the last one.
    now = datetime(2026, 9, 26, 4, 1)
    assert plugin._time_reset_reason({"mode": "daily", "at_hour": 4}, datetime(2026, 9, 26, 3, 59), now) == "daily"

def test_daily_min_idle_keeps_conversation_across_boundary(plugin):
    # A conversation still running at the boundary (03:59 -> 04:27) is not cut...
    now = datetime(2026, 9, 26, 4, 27)
    policy = {"mode": "daily", "at_hour": 4, "daily_min_idle_minutes": 180}
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 3, 59), now) is None
    # ...but a real pause before the boundary still is.
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 0, 30), now) == "daily"
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 1, 27), now) == "daily"  # exactly 180 min

def test_daily_min_idle_defers_cut_to_first_real_pause(plugin):
    # Chat ran across 04:00 (route created the evening before, last message 04:27): the cut is deferred
    # to the first message after a >= 180 min pause, not skipped for the day.
    policy = {"mode": "daily", "at_hour": 4, "daily_min_idle_minutes": 180}
    started, last = datetime(2026, 9, 25, 22, 0), datetime(2026, 9, 26, 4, 27)
    assert plugin._time_reset_reason(policy, last, datetime(2026, 9, 26, 5, 0), started) is None
    assert plugin._time_reset_reason(policy, last, datetime(2026, 9, 26, 10, 0), started) == "daily"
    # A route created after the boundary (e.g. by that cut) is not cut again the same day.
    fresh = datetime(2026, 9, 26, 10, 0)
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 10, 5), datetime(2026, 9, 26, 15, 0), fresh) is None
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 10, 5), datetime(2026, 9, 27, 9, 0), fresh) == "daily"

def test_daily_without_min_idle_ignores_route_start(plugin):
    # Default (0): unchanged historical semantics, the route's created_at plays no role.
    policy = {"mode": "daily", "at_hour": 4}
    started = datetime(2026, 9, 25, 22, 0)
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 4, 27), datetime(2026, 9, 26, 10, 0), started) is None

@pytest.mark.parametrize("bad", [-5, "180", None, True, [180], float("nan"), float("inf"), float("-inf")])
def test_daily_min_idle_malformed_means_no_minimum(plugin, bad):
    policy = {"mode": "daily", "at_hour": 4, "daily_min_idle_minutes": bad}
    assert plugin._time_reset_reason(policy, datetime(2026, 9, 26, 3, 59), datetime(2026, 9, 26, 4, 1)) == "daily"

def test_daily_min_idle_does_not_touch_idle_mode(plugin):
    policy = {"mode": "both", "idle_minutes": 30, "at_hour": 4, "daily_min_idle_minutes": 180}
    assert plugin._time_reset_reason(policy, _last(45), NOW) == "idle"

def test_both_mode_idle_wins_first(plugin):
    policy = {"mode": "both", "idle_minutes": 30, "at_hour": 4}
    assert plugin._time_reset_reason(policy, _last(90), NOW) == "idle"

def test_none_mode_never_triggers(plugin):
    policy = {"mode": "none"}
    assert plugin._time_reset_reason(policy, _last(100000), NOW) is None

def test_malformed_policy_fails_open(plugin):
    for policy in ({"mode": "idle", "idle_minutes": "lots"}, {"mode": "daily", "at_hour": 99}, {}):
        assert plugin._time_reset_reason(policy, _last(90), NOW) is None

def test_missing_timestamps_no_reset(plugin):
    policy = {"mode": "idle", "idle_minutes": 1}
    assert plugin._time_reset_reason(policy, None, NOW) is None

def test_created_at_fallback(plugin):
    # Migration path: entry without metadata clock falls back to created_at
    policy = {"mode": "idle", "idle_minutes": 60}
    entry = SimpleNamespace(
        created_at=NOW - timedelta(minutes=90),
        updated_at=None,
        metadata={},
    )
    assert plugin._time_reset_reason(policy, plugin._last_user_inbound(entry), NOW) == "idle"


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

def test_register_registers_hook(plugin):
    ctx = mock.MagicMock()
    plugin.register(ctx)
    assert ctx.register_hook.call_args_list == [
        mock.call("pre_gateway_dispatch", plugin._maybe_reset),
        mock.call("pre_llm_call", plugin._carry_over_context),
    ]


# ---------------------------------------------------------------------------
# _command_free_event — the funnel must never title the session from the
# triggering utterance (/new <title> semantics; get_command_args() on a plain
# message is the whole message text).
# ---------------------------------------------------------------------------

def test_reset_event_strips_message_text(plugin):
    from dataclasses import dataclass

    @dataclass
    class Event:
        source: object = "src"
        text: str = "please start fresh tomorrow morning"
        def get_command_args(self) -> str:
            return self.text  # non-command MessageEvent semantics

    event = Event()
    bare = plugin._command_free_event(event)
    assert bare.get_command_args() == ""
    assert bare.source is event.source
    assert event.text == "please start fresh tomorrow morning"  # original untouched

def test_reset_event_falls_back_for_plain_objects(plugin):
    source = object()
    event = SimpleNamespace(source=source, get_command_args=lambda: "hello there")
    bare = plugin._command_free_event(event)
    assert bare.get_command_args() == ""
    assert bare.source is source


# ---------------------------------------------------------------------------
# _maybe_reset — end-to-end against the REAL SessionStore from the tree
# ---------------------------------------------------------------------------

class _RoutingCfg:
    write_sessions_json = True


def _make_real_store(tmp_path):
    """Build a real SessionStore over a scratch sessions dir."""
    mod = _import_tree_module("gateway.session")
    return mod.SessionStore(sessions_dir=tmp_path / "sessions", config=_RoutingCfg())

def _real_source():
    mod = _import_tree_module("gateway.session")
    cfg = _import_tree_module("gateway.config")
    # A chat platform, not LOCAL: from the Hermes session-authority runtime (v0.23) on, a LOCAL
    # route rotates only under the authority epoch a running gateway installs, so a bare store
    # cannot reset one. Chat routes are what this plugin resets anyway.
    return mod.SessionSource(platform=cfg.Platform.TELEGRAM, chat_id="test-chat", chat_type="dm",
                             user_id="tester")

def _seed_route(store, *, key="agent:main:telegram:dm:test-chat", updated_minutes_ago=0.0, token=None):
    mod = _import_tree_module("gateway.session")
    SessionEntry = mod.SessionEntry
    src = _real_source()
    entry = SessionEntry(
        session_key=key, session_id="seed-sid",
        created_at=datetime.now() - timedelta(minutes=updated_minutes_ago),
        updated_at=datetime.now() - timedelta(minutes=updated_minutes_ago),
        origin=src, platform=None, chat_type="dm",
        active_turn_token=token,
    )
    store._entries[key] = entry
    store._loaded = True
    return key, entry


def test_maybe_reset_store_fallback_calls_real_reset_session(plugin, tmp_path, hook_recorder):
    """Degraded path (gateway double has no reset funnel): a bare store rotation
    PLUS what the plugin surface can reach — the delegation interrupt and the
    on_session_finalize / on_session_reset lifecycle pair."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        result = _run(plugin._maybe_reset(event, gateway, store))

    assert result is None  # allow normal dispatch
    new_entry = store.lookup_by_session_key(key)
    assert new_entry is not None
    assert new_entry.session_id != "seed-sid"
    assert new_entry.is_fresh_reset is True
    # In-flight delegations end with the old conversation (review #2, item 3).
    assert hook_recorder.interrupts
    assert hook_recorder.interrupts[0]["session_key"] == key
    assert hook_recorder.interrupts[0]["parent_session_id"] == "seed-sid"
    # Boundary hooks fire (review #2, item 2): finalize for the old id, reset for the new.
    fired = [name for name, _ in hook_recorder.lifecycle]
    assert "on_session_finalize" in fired
    assert "on_session_reset" in fired
    reset_kw = dict(hook_recorder.lifecycle)["on_session_reset"]
    assert reset_kw["old_session_id"] == "seed-sid"
    assert reset_kw["new_session_id"] == new_entry.session_id

def test_maybe_reset_prefers_the_gateway_funnel(plugin, tmp_path, hook_recorder):
    """When the gateway reset funnel exists it IS the reset — the store is never
    rotated behind its back — and it receives a command-free event so /new's
    title-from-args step cannot title the fresh session from the utterance."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    seen = []

    async def funnel(evt):
        seen.append(evt)

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False,
                            get_command_args=lambda: "and please keep the old context in mind")

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy), \
         mock.patch.object(store, "reset_session",
                           side_effect=AssertionError("store must not rotate behind the funnel")):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert len(seen) == 1
    assert seen[0].get_command_args() == ""  # no title can leak in as /new args
    assert seen[0].source is event.source

def test_maybe_reset_funnel_failure_before_rotation_falls_back(plugin, tmp_path, hook_recorder):
    """A funnel that blows up before rotating degrades to the store rotation;
    the boundary must still happen."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)

    async def funnel(evt):
        raise RuntimeError("funnel exploded")

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry.session_id != "seed-sid"
    assert new_entry.is_fresh_reset is True
    assert hook_recorder.interrupts  # fallback still ends old delegations

def test_maybe_reset_funnel_failure_after_rotation_keeps_result(plugin, tmp_path, hook_recorder):
    """A funnel that rotates and THEN blows up (banner/tip tail) must not trigger
    a second rotation in the fallback."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    rotated_sid = []

    async def funnel(evt):
        rotated_sid.append(store.reset_session(key).session_id)
        raise RuntimeError("banner exploded after the rotation")

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
        _handle_reset_command=funnel,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == rotated_sid[0]

def test_maybe_reset_prefers_payload_session_key(plugin, tmp_path):
    """A public session_key in the hook payload outranks the private gateway
    helper (review #2, item 4: ready for the payload equivalent they offered)."""
    store = _make_real_store(tmp_path)
    payload_key = "agent:main:telegram:dm:payload-chat"
    helper_key = "agent:main:telegram:dm:helper-chat"
    _seed_route(store, key=payload_key, updated_minutes_ago=5)
    _seed_route(store, key=helper_key, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: helper_key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}  # within threshold -> clock-stamp path
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store,
                                        session_key=payload_key)) is None

    assert store.get_session_metadata(payload_key, "srp_last_user_inbound") is not None
    assert store.get_session_metadata(helper_key, "srp_last_user_inbound") is None

def test_maybe_reset_bg_turn_does_not_block_reset(plugin, tmp_path, hook_recorder):
    """The 19:04 bug: a background-review turn stamps ``updated_at`` but must
    NOT reset the plugin's own clock. An old last-user-inbound with a fresh
    ``updated_at`` must still reset."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=1)  # fresh updated_at
    # But the plugin's clock says the last real user inbound was 2h ago:
    store.set_session_metadata(key, "srp_last_user_inbound",
                               (datetime.now() - timedelta(hours=2)).isoformat())

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry.session_id != "seed-sid"
    assert new_entry.is_fresh_reset is True

def test_maybe_reset_stamps_clock_when_within_threshold(plugin, tmp_path):
    """Within the threshold the message must advance the plugin's own clock,
    not ``updated_at``."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    before = datetime.now() - timedelta(seconds=5)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == "seed-sid"
    stamp = store.get_session_metadata(key, "srp_last_user_inbound")
    assert stamp is not None
    assert datetime.fromisoformat(stamp) >= before

def test_maybe_reset_deferral_does_not_advance_clock(plugin, tmp_path):
    """Boundary past + turn in flight: defer AND leave the clock alone, so the
    next message re-evaluates the same boundary."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="tok-123")
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == "seed-sid"
    # No clock stamp: the deferral must not swallow the pending reset.
    assert store.get_session_metadata(key, "srp_last_user_inbound") is None

def test_maybe_reset_skips_when_within_threshold(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    assert store.lookup_by_session_key(key).session_id == "seed-sid"

def test_maybe_reset_defers_when_turn_in_flight(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="tok-123")
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

    # Route unchanged: the in-flight turn keeps its session.
    assert store.lookup_by_session_key(key).session_id == "seed-sid"

def test_maybe_reset_no_policy_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=100000)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=None):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    assert store.lookup_by_session_key(key).session_id == "seed-sid"

def test_maybe_reset_survives_store_errors(plugin, tmp_path):
    class ExplodingStore:
        def lookup_by_session_key(self, key):
            raise RuntimeError("boom")

    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:telegram:dm:x")
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, ExplodingStore())) is None

def test_maybe_reset_no_route_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:telegram:dm:none")
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None

def test_maybe_reset_survives_missing_key_helper(plugin, tmp_path):
    """No payload key and no gateway helper: fail closed (no reset, no stamp).
    A bare build_session_key fallback is deliberately absent — under
    multiplexed profiles it is not profile-namespaced and could resolve
    ANOTHER profile's route (review #2, item 4)."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gateway = SimpleNamespace()  # no _session_key_for_source at all
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    assert store.lookup_by_session_key(key).session_id == "seed-sid"


# ---------------------------------------------------------------------------
# The REAL /new reset funnel — GatewaySessionCommandsMixin bodies bound to a
# minimal runner stub, run against the REAL SessionStore (review #2, items 1-3:
# run-generation bump, cached-agent eviction, conversation-scope clear,
# async-delegation interrupt, session:end/session:reset, boundary hooks).
# ---------------------------------------------------------------------------

def _real_funnel_gateway(store, key):
    """A GatewayRunner stand-in carrying the REAL funnel methods.

    Only the leaf internals the funnel touches are recorders (they are the
    gateway's own per-session caches/hooks, covered by upstream tests);
    ``_handle_reset_command``, ``_fire_session_reset_hooks`` and
    ``_cleanup_old_agent_for_reset`` run for real."""
    session_mod = _import_tree_module("gateway.session")
    slash_mod = _import_tree_module("gateway.slash_commands_session")
    mixin = slash_mod.GatewaySessionCommandsMixin

    calls = SimpleNamespace(invalidated=[], released=[], evicted=[], cleared=[], finalize=[], stopped=[])

    class _Hooks:
        def __init__(self):
            self.emitted = []

        async def emit(self, name, payload):
            self.emitted.append((name, payload))

    async def _finalize_session_off_loop(**kw):
        calls.finalize.append(kw)

    gw = SimpleNamespace()
    gw.session_store = store
    gw.async_session_store = session_mod.AsyncSessionStore(store)
    gw.hooks = _Hooks()
    gw._session_key_for_source = lambda src: key
    gw._cached_agent_for = lambda k: None  # real cleanup helper early-returns on None
    gw._invalidate_session_run_generation = lambda k, reason="": calls.invalidated.append((k, reason))
    gw._release_running_agent_state = lambda k: calls.released.append(k)
    gw._evict_cached_agent = lambda k: calls.evicted.append(k)
    gw._clear_conversation_scope = lambda k, reason="": calls.cleared.append((k, reason))
    gw._finalize_session_off_loop = _finalize_session_off_loop
    gw._telegram_topic_new_header = lambda source: ""
    gw._is_telegram_topic_lane = lambda source: False
    gw._session_db = None
    # v0.23's funnel first stops the route owner's turn through the session authority; with no
    # authority that is a no-op, so a recorder stands in.
    async def _hm_stop_owner_turn(k):
        calls.stopped.append(k)
    gw._hm_stop_owner_turn = _hm_stop_owner_turn
    for name in ("_handle_reset_command", "_fire_session_reset_hooks",
                 "_cleanup_old_agent_for_reset"):
        setattr(gw, name, getattr(mixin, name).__get__(gw))
    return gw, calls, slash_mod


def test_maybe_reset_runs_the_real_new_funnel(plugin, tmp_path, hook_recorder, monkeypatch):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gw, calls, slash_mod = _real_funnel_gateway(store, key)
    monkeypatch.setattr(slash_mod, "t", lambda key_name, **kw: key_name)  # no i18n in tests
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gw, store)) is None

    new_entry = store.lookup_by_session_key(key)
    assert new_entry is not None
    assert new_entry.session_id != "seed-sid"
    assert new_entry.is_fresh_reset is True
    # Item 1: run-generation bump, running-slot release, cached-agent eviction,
    # conversation-scope clear — the exact /new boundary sequence.
    assert calls.invalidated == [(key, "session_reset")]
    assert calls.released == [key]
    assert calls.evicted == [key]
    assert calls.cleared == [(key, "session_reset")]
    # Item 2: gateway session:end / session:reset hooks + on_session_finalize.
    assert [name for name, _ in gw.hooks.emitted] == ["session:end", "session:reset"]
    assert gw.hooks.emitted[0][1]["session_key"] == key
    assert calls.finalize and calls.finalize[0]["session_id"] == "seed-sid"
    assert calls.finalize[0]["new_session_id"] == new_entry.session_id
    fired = dict(hook_recorder.lifecycle)
    assert "on_session_reset" in fired
    assert fired["on_session_reset"]["old_session_id"] == "seed-sid"
    assert fired["on_session_reset"]["new_session_id"] == new_entry.session_id
    # Item 3: in-flight async delegations end with the old conversation.
    assert hook_recorder.interrupts
    assert hook_recorder.interrupts[0]["session_key"] == key
    assert hook_recorder.interrupts[0]["parent_session_id"] == "seed-sid"


# ---------------------------------------------------------------------------
# _read_session_reset_policy — profile resolution via the real config loader
# ---------------------------------------------------------------------------

def _bootstrap_hermes_paths():
    sys.path.insert(0, str(HERMES_TREE))
    return HERMES_TREE.is_dir()

def test_policy_reader_none_on_missing_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    import hermes_constants
    from hermes_cli import config as cfgmod

    # Point the config cache at the temp home (the autouse fixture redirects
    # HERMES_HOME already; ensure no stale cache survives from other tests).
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)

    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)

    assert plugin._read_session_reset_policy(None) is None

def test_policy_reader_reads_named_profile_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    from hermes_cli import config as cfgmod
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    prof = home / "profiles" / "mom"
    prof.mkdir(parents=True)
    (prof / "config.yaml").write_text(
        "session_reset:\n  mode: both\n  idle_minutes: 45\n  at_hour: 4\n"
    )

    policy = plugin._read_session_reset_policy("mom")
    assert policy == {"mode": "both", "idle_minutes": 45, "at_hour": 4}

def test_policy_reader_reads_default_profile_config(plugin, tmp_path, monkeypatch):
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    from hermes_cli import config as cfgmod
    monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
    monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
    home = Path(os.environ["HERMES_HOME"])
    home.mkdir(parents=True, exist_ok=True)
    (home / "config.yaml").write_text(
        "session_reset:\n  mode: idle\n  idle_minutes: 30\n"
    )

    policy = plugin._read_session_reset_policy(None)
    assert policy == {"mode": "idle", "idle_minutes": 30}

def _write_idle_policy(config_dir: Path, minutes: int) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(
        f"session_reset:\n  mode: idle\n  idle_minutes: {minutes}\n", encoding="utf-8"
    )

def test_policy_reader_resolves_profiles_from_root(plugin, tmp_path, monkeypatch):
    """Issue #4 regression: profile tags resolve against the Hermes ROOT under
    BOTH gateway layouts — a multiplexed gateway (HERMES_HOME = <root>) and a
    standalone per-profile gateway (HERMES_HOME = <root>/profiles/<name>). The
    old ``<home>/profiles/<name>`` build double-nested on the standalone layout
    (looking for ``profiles/alpha/profiles/alpha/config.yaml``) and the reader
    silently returned None, so no reset ever fired.

    ``default`` is just a profile name too: its home IS the root, whatever the
    launch home is (same mapping hermes_cli.profiles.get_profile_dir encodes).
    """
    if not _bootstrap_hermes_paths():
        pytest.skip("Hermes tree not present on this host")
    import hermes_constants
    from hermes_cli import config as cfgmod

    # External/custom-root layout (the reporter's per-profile gateways, cf. a
    # Docker /opt/data root): shift the native-home suffix so the tmp root is
    # not absorbed into the default profile tree's root inference.
    monkeypatch.setenv("HERMES_DATA_DIR_SUFFIX", f"/srp-test-{tmp_path.name}")
    monkeypatch.setattr(hermes_constants, "_profile_fallback_warned", True, raising=False)

    root = tmp_path / "root"
    _write_idle_policy(root, 111)                    # the "default" profile
    _write_idle_policy(root / "profiles" / "alpha", 222)
    _write_idle_policy(root / "profiles" / "beta", 333)

    cases = [
        # per-profile home (standalone gateway per profile)
        (root / "profiles" / "alpha", "alpha", 222),
        (root / "profiles" / "alpha", "beta", 333),
        (root / "profiles" / "alpha", None, 222),
        (root / "profiles" / "beta", "beta", 333),
        # root home (multiplexed gateway)
        (root, "alpha", 222),
        (root, "beta", 333),
        (root, "default", 111),
        (root, None, 111),
        # "default" names the root profile's config, even from a profile home
        (root / "profiles" / "alpha", "default", 111),
    ]
    for home, profile, minutes in cases:
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
        monkeypatch.setattr(cfgmod, "_LOAD_CONFIG_CACHE", {})
        monkeypatch.setattr(cfgmod, "_CONFIG_CACHE", {}, raising=False)
        policy = plugin._read_session_reset_policy(profile)
        assert policy == {"mode": "idle", "idle_minutes": minutes}, (
            f"HERMES_HOME={home} profile={profile!r} -> {policy}"
        )


# ---------------------------------------------------------------------------
# Slash commands: skipped by the hook (no clock stamp, no time reset)
# ---------------------------------------------------------------------------

def _real_event(text):
    """A real gateway MessageEvent (its get_command() is the router's parse)."""
    event_mod = _import_tree_module("gateway.platforms.event")
    return event_mod.MessageEvent(text=text, source=_real_source())


@pytest.mark.parametrize("text", ["/new", "/reset", "/status", "  /model some-model"])
def test_maybe_reset_skips_slash_commands_past_boundary(plugin, tmp_path, text):
    """The gateway runs pre_gateway_dispatch for commands too (every
    non-internal event, before command routing): the hook must not rotate."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    funnel = mock.AsyncMock(side_effect=AssertionError("a command must not trigger a time reset"))
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key, _handle_reset_command=funnel)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy), \
         mock.patch.object(store, "reset_session",
                           side_effect=AssertionError("store must not rotate for a command")):
        assert _run(plugin._maybe_reset(_real_event(text), gateway, store)) is None

    assert not funnel.called
    assert store.lookup_by_session_key(key).session_id == "seed-sid"
    # Deferral semantics hold: the clock is untouched, the next real message resets.
    assert store.get_session_metadata(key, "srp_last_user_inbound") is None


def test_maybe_reset_command_does_not_stamp_clock(plugin, tmp_path):
    """Within the threshold a command is bookkeeping, not user activity: a
    periodic /status must not keep an otherwise idle session alive."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(_real_event("/status"), gateway, store)) is None

    assert store.get_session_metadata(key, "srp_last_user_inbound") is None


def test_command_check_falls_back_to_text_without_get_command(plugin):
    """Event stand-ins without get_command(): a leading "/" (after whitespace)."""
    assert plugin._is_slash_command(SimpleNamespace(text="  /new")) is True
    assert plugin._is_slash_command(SimpleNamespace(text="hello /new")) is False
    assert plugin._is_slash_command(SimpleNamespace(text=None)) is False
    assert plugin._is_slash_command(SimpleNamespace()) is False


def test_path_like_message_is_not_a_command(plugin, tmp_path):
    """get_command() rejects path-like text: it is an ordinary message and the
    time reset still applies to it."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    seen = []

    async def funnel(evt):
        seen.append(evt)

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key, _handle_reset_command=funnel)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(_real_event("/etc/hosts is broken"), gateway, store)) is None

    assert len(seen) == 1


def test_new_past_boundary_rotates_once_with_real_funnel(plugin, tmp_path, hook_recorder, monkeypatch):
    """Regression: /new sent past the boundary rotated the route TWICE (the hook
    with reason=idle, then the command), burning a run generation and leaving
    an empty, already-ended session row. Hook first, then the command, as the
    gateway orders them: exactly one rotation."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)
    gw, calls, slash_mod = _real_funnel_gateway(store, key)
    monkeypatch.setattr(slash_mod, "t", lambda key_name, **kw: key_name)  # no i18n in tests
    event = _real_event("/new")

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gw, store)) is None  # pre_gateway_dispatch
        _run(gw._handle_reset_command(event))  # the gateway then routes /new

    assert calls.invalidated == [(key, "session_reset")]
    assert [name for name, _ in gw.hooks.emitted] == ["session:end", "session:reset"]
    assert calls.finalize and calls.finalize[0]["session_id"] == "seed-sid"
    assert store.lookup_by_session_key(key).session_id != "seed-sid"


# ---------------------------------------------------------------------------
# Issue #14: a metadata-preserving reset funnel carries the plugin's clock
# onto the replacement entry. The successful reset branch must advance the
# clock on that replacement entry, or the NEXT message re-crosses the same
# boundary and the route rotates a second time.
# ---------------------------------------------------------------------------

def _freeze_plugin_now(plugin, monkeypatch, at: datetime):
    """Freeze the plugin's ``datetime.now()`` (kept deterministic around the
    daily boundary), leaving fromisoformat parsing behaviour intact."""
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls.fromisoformat(at.isoformat())
    monkeypatch.setattr(plugin, "datetime", _Frozen)

def _metadata_preserving_funnel(store, key, calls, *, explode_after=False):
    """The reporter's owner/carryover reset funnel: rotate the route through the
    store, then carry EVERY metadata entry (unrelated pending-work state AND the
    plugin's own clock) onto the replacement entry."""
    async def funnel(evt):
        old = store.lookup_by_session_key(key)
        carried = dict(getattr(old, "metadata", None) or {})
        new_sid = store.reset_session(key).session_id
        for meta_key, meta_val in carried.items():
            store.set_session_metadata(key, meta_key, meta_val)
        calls.append((new_sid, carried))
        if explode_after:
            raise RuntimeError("banner exploded after the rotation")
    return funnel

def _two_messages_past_daily_boundary(plugin, store, gateway, tmp_path, monkeypatch, *, explode_after):
    """One seeded route, clock at 03:00 behind a 04:00 daily boundary, two
    ordinary inbound messages at 08:00. Returns the metadata-preserving funnel
    calls and the session id after the first message."""
    key, entry = _seed_route(store, updated_minutes_ago=30)
    store.set_session_metadata(key, "srp_last_user_inbound",
                               datetime(2026, 9, 26, 3, 0).isoformat())
    store.set_session_metadata(key, "pending_work", {"queue": "review", "id": 7})
    calls = []
    funnel = _metadata_preserving_funnel(store, key, calls, explode_after=explode_after)
    gateway._handle_reset_command = funnel
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "daily", "at_hour": 4}
    _freeze_plugin_now(plugin, monkeypatch, datetime(2026, 9, 26, 8, 0))
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None  # first message
        first_sid = store.lookup_by_session_key(key).session_id
        assert first_sid != "seed-sid"  # the first message rotated exactly once
        assert _run(plugin._maybe_reset(event, gateway, store)) is None  # second message
    return calls, first_sid

def test_issue14_daily_reset_rotates_once_across_two_messages(plugin, tmp_path, monkeypatch):
    """Two ordinary inbound messages after one daily boundary must rotate the
    route exactly once even when the reset funnel carries the plugin clock onto
    the replacement entry (#14)."""
    store = _make_real_store(tmp_path)
    key = "agent:main:telegram:dm:test-chat"
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    calls, first_sid = _two_messages_past_daily_boundary(
        plugin, store, gateway, tmp_path, monkeypatch, explode_after=False)

    assert store.lookup_by_session_key(key).session_id == first_sid  # no second rotation
    assert len(calls) == 1
    # The successful reset stamped the fresh entry's clock at the inbound time.
    stamp = store.get_session_metadata(key, "srp_last_user_inbound")
    assert datetime.fromisoformat(stamp) >= datetime(2026, 9, 26, 8, 0)
    # Unrelated carryover metadata survives, untouched.
    assert store.get_session_metadata(key, "pending_work") == {"queue": "review", "id": 7}

def test_issue14_clock_stamped_when_funnel_throws_after_rotating(plugin, tmp_path, monkeypatch):
    """Same, through the funnel-throws-after-rotating path (#14): the rotation
    is kept, so it must also clear the pending boundary or the next message
    rotates a second time."""
    store = _make_real_store(tmp_path)
    key = "agent:main:telegram:dm:test-chat"
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    calls, first_sid = _two_messages_past_daily_boundary(
        plugin, store, gateway, tmp_path, monkeypatch, explode_after=True)

    assert store.lookup_by_session_key(key).session_id == first_sid  # no second rotation
    assert len(calls) == 1
    assert store.get_session_metadata(key, "pending_work") == {"queue": "review", "id": 7}


def test_relabel_end_reason_keeps_policy_reset_out_of_user_boundaries(plugin, tmp_path):
    """A policy reset rotates through the /new funnel, which records ``session_reset`` — a user
    boundary that terminally drops in-flight async completions for the old session. The plugin
    relabels it to its own timed reason, and leaves any other end reason alone."""
    hermes_state = _import_tree_module("hermes_state")
    db = hermes_state.SessionDB(tmp_path / "state.db")
    db.create_session("s-reset", "discord")
    db.end_session("s-reset", "session_reset")
    db.create_session("s-exit", "discord")
    db.end_session("s-exit", "user_exit")
    gateway = SimpleNamespace(_session_db=hermes_state.AsyncSessionDB(db))

    plugin._relabel_end_reason(gateway, "s-reset", "idle")
    plugin._relabel_end_reason(gateway, "s-exit", "idle")
    plugin._relabel_end_reason(gateway, None, "idle")
    plugin._relabel_end_reason(SimpleNamespace(), "s-reset", "daily")  # no DB: no-op

    assert db.get_session("s-reset")["end_reason"] == "idle"
    assert db.get_session("s-exit")["end_reason"] == "user_exit"


# ---------------------------------------------------------------------------
# carry_over — optional excerpt of the previous conversation after a policy
# reset (config parsing, transcript filtering, note shape, pre_llm_call
# delivery, and the end-to-end arm-on-reset path against the real store)
# ---------------------------------------------------------------------------

def _carry_policy(**carry):
    return {"mode": "idle", "idle_minutes": 60, "carry_over": dict({"enabled": True}, **carry)}

def _carry_cfg(plugin, **carry):
    return plugin._carry_over_config(_carry_policy(**carry))

_TRANSCRIPT = [
    {"role": "system", "content": "system prompt"},
    {"role": "user", "content": "[CONTEXT COMPACTION] earlier turns summarised"},
    {"role": "user", "content": "Any other restaurants near the station?"},
    {"role": "assistant", "content": None,
     "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "web_search", "arguments": "{}"}}]},
    {"role": "tool", "tool_call_id": "c1", "content": "TOOL OUTPUT must never be carried"},
    {"role": "assistant", "content": "Two more within walking distance: Alpha and Beta."},
    {"role": "user", "content": [{"type": "text", "text": "Which one is"}, {"type": "image_url", "image_url": {}},
                                 {"type": "text", "text": "open late?"}]},
    {"role": "assistant", "content": "Beta, until 23:00. My password: hunter2 is not something to share."},
]

def test_maybe_reset_daily_min_idle_deferred_cut_end_to_end(plugin, tmp_path, monkeypatch):
    """Real store: a route created before 04:00 whose clock was advanced at 04:27 (cut deferred) rotates
    on the first message after a 180-minute pause, and the replacement route is not cut again."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store)
    entry.created_at = datetime(2026, 9, 25, 22, 0)
    store.set_session_metadata(key, "srp_last_user_inbound", datetime(2026, 9, 26, 4, 27).isoformat())
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "daily", "at_hour": 4, "daily_min_idle_minutes": 180}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        _freeze_plugin_now(plugin, monkeypatch, datetime(2026, 9, 26, 10, 0))
        _run(plugin._maybe_reset(event, gateway, store))
        first = store.lookup_by_session_key(key).session_id
        assert first != "seed-sid"
        _freeze_plugin_now(plugin, monkeypatch, datetime(2026, 9, 26, 14, 0))
        _run(plugin._maybe_reset(event, gateway, store))
        assert store.lookup_by_session_key(key).session_id == first

def test_carry_over_config_off_by_default(plugin):
    assert plugin._carry_over_config({"mode": "idle"}) is None
    assert plugin._carry_over_config({"carry_over": {"turns": 6}}) is None  # enabled must be true
    assert plugin._carry_over_config({"carry_over": "yes"}) is None
    assert plugin._carry_over_config({"carry_over": {"enabled": "true"}}) is None

def test_carry_over_config_defaults_and_shorthand(plugin):
    cfg = plugin._carry_over_config({"carry_over": True})
    assert cfg == {"turns": 6, "max_chars": 2500, "max_age_hours": 24.0, "pointer_turns": 3,
                   "session_search_hint": True}
    cfg = _carry_cfg(plugin, turns=2, max_chars=800, max_age_hours=6, pointer_turns=0, session_search_hint=False)
    assert cfg == {"turns": 2, "max_chars": 800, "max_age_hours": 6.0, "pointer_turns": 0,
                   "session_search_hint": False}

def test_carry_over_config_out_of_range_falls_back(plugin):
    cfg = _carry_cfg(plugin, turns=0, max_chars=50, max_age_hours=-1, pointer_turns=True)
    assert (cfg["turns"], cfg["max_chars"], cfg["max_age_hours"], cfg["pointer_turns"]) == (6, 2500, 24, 3)

def test_transcript_tail_is_text_only(plugin):
    tail = plugin._transcript_tail(_TRANSCRIPT, turns=6)
    assert tail == [
        ("user", "Any other restaurants near the station?"),
        ("assistant", "Two more within walking distance: Alpha and Beta."),
        ("user", "Which one is open late?"),
        ("assistant", "Beta, until 23:00. My password: hunter2 is not something to share."),
    ]
    assert plugin._transcript_tail(_TRANSCRIPT, turns=1) == tail[-2:]
    assert plugin._transcript_tail(None, turns=3) == []

def test_carry_note_shape_and_redaction(plugin):
    cfg = _carry_cfg(plugin)
    tail = plugin._transcript_tail(_TRANSCRIPT, turns=6)
    note = plugin._build_carry_note("old-sid", "daily", datetime(2026, 9, 26, 4, 27), tail, cfg)
    assert note.startswith("[session-reset carry-over]")
    assert "daily reset at 2026-09-26 04:27" in note and "session old-sid" in note
    assert "- User: Any other restaurants near the station?" in note
    assert "- Assistant: Beta, until 23:00." in note
    assert "hunter2" not in note and "[redacted]" in note
    assert "TOOL OUTPUT" not in note and "COMPACTION" not in note
    assert note.rstrip().endswith('session_search(session_id="old-sid") instead of guessing.')
    no_hint = plugin._build_carry_note("old-sid", "daily", datetime(2026, 9, 26, 4, 27), tail,
                                       _carry_cfg(plugin, session_search_hint=False))
    assert "session_search" not in no_hint and "ask instead of guessing" in no_hint

def test_carry_note_respects_caps_newest_first(plugin):
    cfg = _carry_cfg(plugin, max_chars=600)
    tail = [("user", "u" * 1000), ("assistant", "a" * 1000), ("user", "LAST " + "z" * 1000)]
    note = plugin._build_carry_note("old-sid", "idle", datetime(2026, 9, 26, 9, 0), tail, cfg)
    lines = [ln for ln in note.splitlines() if ln.startswith("- ")]
    assert lines[-1].startswith("- User: LAST ")  # the newest turn always makes it in
    assert all(len(ln) <= len("- Assistant: ") + plugin._CARRY_ASSISTANT_CHARS + 2 for ln in lines)
    assert sum(len(ln) + 1 for ln in lines) <= 600 + 1
    assert not any(ln.startswith("- User: uuu") for ln in lines)  # oldest dropped past the budget

def test_carry_over_context_note_then_pointers_then_nothing(plugin):
    cfg = _carry_cfg(plugin, pointer_turns=2)
    plugin._arm_carry_over("new-sid", "old-sid", "daily", datetime(2026, 9, 26, 4, 27),
                           [("user", "hello"), ("assistant", "hi")], cfg)
    assert plugin._carry_over_context(session_id="other-sid", platform="telegram") is None
    first = plugin._carry_over_context(session_id="new-sid", platform="telegram")
    assert "- User: hello" in first["context"]
    for _ in range(2):
        ptr = plugin._carry_over_context(session_id="new-sid", platform="telegram")["context"]
        assert "session old-sid" in ptr and "- User" not in ptr
    assert plugin._carry_over_context(session_id="new-sid", platform="telegram") is None
    assert plugin._carry_pending == {}

def test_carry_over_context_skips_subagents_and_cron(plugin):
    plugin._arm_carry_over("new-sid", "old-sid", "idle", datetime(2026, 9, 26, 9, 0), [("user", "x")],
                           _carry_cfg(plugin, pointer_turns=0))
    assert plugin._carry_over_context(session_id="new-sid", platform="cron") is None
    assert plugin._carry_over_context(session_id="new-sid", platform="telegram",
                                      parent_session_id="new-sid") is None  # delegated child turn
    assert plugin._carry_over_context(session_id="new-sid", platform="telegram")["context"]
    assert plugin._carry_over_context(session_id="new-sid", platform="telegram") is None  # pointer_turns=0

def test_carry_over_note_expires_when_never_picked_up(plugin, monkeypatch):
    plugin._arm_carry_over("new-sid", "old-sid", "idle", datetime(2026, 9, 26, 9, 0), [("user", "x")],
                           _carry_cfg(plugin))
    later = plugin.time.monotonic() + plugin._CARRY_PENDING_TTL_S + 1
    monkeypatch.setattr(plugin.time, "monotonic", lambda: later)
    assert plugin._carry_over_context(session_id="new-sid", platform="telegram") is None
    assert "new-sid" not in plugin._carry_pending

def _carry_store(tmp_path, *, updated_minutes_ago=120):
    """Real store + route; the outgoing transcript comes through a recorder on
    the store's public ``load_transcript`` (the plugin's only transcript API)."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=updated_minutes_ago)
    reads = []

    def load_transcript(sid):
        reads.append((sid, store.lookup_by_session_key(key).session_id))
        return list(_TRANSCRIPT)

    store.load_transcript = load_transcript
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    return store, key, gateway, reads

def test_carry_over_armed_by_policy_reset(plugin, tmp_path):
    store, key, gateway, reads = _carry_store(tmp_path)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy()):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    new_sid = store.lookup_by_session_key(key).session_id
    assert new_sid != "seed-sid"
    assert reads == [("seed-sid", "seed-sid")]  # the OUTGOING session, read before the rotation
    note = plugin._carry_over_context(session_id=new_sid, platform="telegram")["context"]
    assert "idle reset" in note and "session seed-sid" in note
    assert "Beta, until 23:00." in note and "TOOL OUTPUT" not in note
    assert "session seed-sid" in plugin._carry_over_context(session_id=new_sid, platform="telegram")["context"]

def test_carry_over_disabled_reads_nothing(plugin, tmp_path):
    store, key, gateway, reads = _carry_store(tmp_path)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy",
                           return_value={"mode": "idle", "idle_minutes": 60}):
        _run(plugin._maybe_reset(event, gateway, store))
    assert store.lookup_by_session_key(key).session_id != "seed-sid"  # reset still happens
    assert reads == [] and plugin._carry_pending == {}

def test_carry_over_skipped_when_previous_activity_too_old(plugin, tmp_path):
    store, key, gateway, reads = _carry_store(tmp_path, updated_minutes_ago=30 * 60)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy(max_age_hours=24)):
        _run(plugin._maybe_reset(event, gateway, store))
    assert store.lookup_by_session_key(key).session_id != "seed-sid"
    assert reads == [] and plugin._carry_pending == {}

def test_carry_over_not_armed_without_rotation(plugin, tmp_path):
    """No confirmed rotation (deferral for an in-flight turn) -> nothing armed."""
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="turn-1")
    store.load_transcript = lambda sid: list(_TRANSCRIPT)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy()):
        _run(plugin._maybe_reset(event, gateway, store))
    assert store.lookup_by_session_key(key).session_id == "seed-sid"
    assert plugin._carry_pending == {}

def test_carry_over_survives_transcript_read_failure(plugin, tmp_path):
    store, key, gateway, reads = _carry_store(tmp_path)

    def broken(sid):
        raise RuntimeError("db locked")

    store.load_transcript = broken
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy()):
        assert _run(plugin._maybe_reset(event, gateway, store)) is None
    assert store.lookup_by_session_key(key).session_id != "seed-sid"  # the reset itself is unaffected
    assert plugin._carry_pending == {}

def test_typed_new_never_arms_carry_over(plugin, tmp_path):
    store, key, gateway, reads = _carry_store(tmp_path)
    event = _real_event("/new")
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy()):
        _run(plugin._maybe_reset(event, gateway, store))
    assert reads == [] and plugin._carry_pending == {}

def test_carry_over_through_the_real_new_funnel(plugin, tmp_path, hook_recorder, monkeypatch):
    store, key, _, reads = _carry_store(tmp_path)
    gw, calls, slash_mod = _real_funnel_gateway(store, key)
    monkeypatch.setattr(slash_mod, "t", lambda key_name, **kw: key_name)  # no i18n in tests
    event = SimpleNamespace(source=_real_source(), internal=False)
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=_carry_policy()):
        assert _run(plugin._maybe_reset(event, gw, store)) is None
    new_sid = store.lookup_by_session_key(key).session_id
    assert new_sid != "seed-sid" and calls.evicted == [key]
    assert reads == [("seed-sid", "seed-sid")]
    assert "session seed-sid" in plugin._carry_over_context(session_id=new_sid, platform="telegram")["context"]
