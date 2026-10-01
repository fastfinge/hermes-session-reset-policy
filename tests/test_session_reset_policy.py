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
        session_key: str = "agent:main:local:dm:test"
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
    ctx.register_hook.assert_called_once_with("pre_gateway_dispatch", plugin._maybe_reset)


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
    return mod.SessionSource(platform=cfg.Platform.LOCAL, chat_id="test-chat", chat_type="dm",
                             user_id="tester")

def _seed_route(store, *, key="agent:main:local:dm:test-chat", updated_minutes_ago=0.0, token=None):
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
    payload_key = "agent:main:local:dm:payload-chat"
    helper_key = "agent:main:local:dm:helper-chat"
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

    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:x")
    event = SimpleNamespace(source=_real_source(), internal=False)
    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert _run(plugin._maybe_reset(event, gateway, ExplodingStore())) is None

def test_maybe_reset_no_route_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:none")
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

    calls = SimpleNamespace(invalidated=[], released=[], evicted=[], cleared=[], finalize=[])

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
