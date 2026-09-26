"""Tests for the hermes-session-reset-policy plugin.

Standalone-suite discipline (per the platform-plugin skill):
- pytest must be able to COLLECT this file without any Hermes import
  succeeding. All hermes imports are therefore behind the
  ``_bootstrap_gateway()`` call inside tests, never at module scope.
- Real classes from the installed Hermes tree are imported lazily inside a
  session fixture, so the tests exercise the actual SessionStore/SessionEntry
  behavior (routing transitions, reset_session) rather than mocks.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

PLUGIN_DIR = Path(__file__).resolve().parent.parent
PLUGIN_INIT = PLUGIN_DIR / "__init__.py"
HERMES_TREE = Path.home() / ".hermes" / "hermes-agent"


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


@pytest.fixture(scope="session")
def hermes_tree():
    return HERMES_TREE


@pytest.fixture(scope="session")
def real_classes(hermes_tree):
    """Real SessionStore/SessionEntry/gateway key helper from the installed tree."""
    if not hermes_tree.is_dir():
        pytest.skip("Hermes tree not present on this host")
    sys.path.insert(0, str(hermes_tree))
    try:
        from gateway.session import SessionStore, SessionEntry  # noqa: F401
        from gateway.session_identity import transport_profile_of  # noqa: F401
        return SimpleNamespace(
            SessionStore=None, SessionEntry=SessionEntry, path=hermes_tree,
        )
    except Exception as exc:
        pytest.skip(f"Hermes tree not importable: {exc}")


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


# ---------------------------------------------------------------------------
# _time_reset_reason — boundary math (ported verbatim from the core patch)
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 26, 12, 0, 0)


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
    assert plugin._time_reset_reason(policy, _entry(90), NOW) == "idle"


def test_idle_mode_no_reset_within_threshold(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _entry(30), NOW) is None


def test_idle_mode_exactly_at_threshold_triggers(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    assert plugin._time_reset_reason(policy, _entry(60), NOW) == "idle"


def test_daily_mode_boundary(plugin):
    # now=12:00, boundary hour=4 -> boundary today 04:00; last= yesterday -> reset
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 10, 0),
        updated_at=datetime(2026, 9, 25, 10, 0),
    )
    assert plugin._time_reset_reason(policy, entry, NOW) == "daily"
    # last after today's boundary -> no reset
    entry2 = SimpleNamespace(
        created_at=datetime(2026, 9, 26, 8, 0),
        updated_at=datetime(2026, 9, 26, 8, 0),
    )
    assert plugin._time_reset_reason(policy, entry2, NOW) is None


def test_daily_mode_boundary_in_future_rolls_back_a_day(plugin):
    # now=02:00, boundary hour=4 (today 04:00 is in the future) -> boundary
    # yesterday 04:00; last=yesterday 12:00 (after it) -> no reset.
    now = datetime(2026, 9, 26, 2, 0)
    policy = {"mode": "daily", "at_hour": 4}
    entry = SimpleNamespace(
        created_at=datetime(2026, 9, 25, 12, 0),
        updated_at=datetime(2026, 9, 25, 12, 0),
    )
    assert plugin._time_reset_reason(policy, entry, now) is None
    # last=day-before-yesterday -> reset
    entry2 = SimpleNamespace(
        created_at=datetime(2026, 9, 24, 12, 0),
        updated_at=datetime(2026, 9, 24, 12, 0),
    )
    assert plugin._time_reset_reason(policy, entry2, now) == "daily"


def test_both_mode_idle_wins_first(plugin):
    policy = {"mode": "both", "idle_minutes": 30, "at_hour": 4}
    assert plugin._time_reset_reason(policy, _entry(90), NOW) == "idle"


def test_none_mode_never_triggers(plugin):
    policy = {"mode": "none"}
    assert plugin._time_reset_reason(policy, _entry(100000), NOW) is None


def test_malformed_policy_fails_open(plugin):
    for policy in ({"mode": "idle", "idle_minutes": "lots"}, {"mode": "daily", "at_hour": 99}, {}):
        assert plugin._time_reset_reason(policy, _entry(90), NOW) is None


def test_missing_timestamps_no_reset(plugin):
    policy = {"mode": "idle", "idle_minutes": 1}
    entry = SimpleNamespace(created_at=None, updated_at=None)
    assert plugin._time_reset_reason(policy, entry, NOW) is None


def test_created_at_fallback(plugin):
    policy = {"mode": "idle", "idle_minutes": 60}
    entry = SimpleNamespace(
        created_at=NOW - timedelta(minutes=90),
        updated_at=None,
    )
    assert plugin._time_reset_reason(policy, entry, NOW) == "idle"


# ---------------------------------------------------------------------------
# register()
# ---------------------------------------------------------------------------

def test_register_registers_hook(plugin):
    ctx = mock.MagicMock()
    plugin.register(ctx)
    ctx.register_hook.assert_called_once_with("pre_gateway_dispatch", plugin._maybe_reset)


# ---------------------------------------------------------------------------
# _maybe_reset — end-to-end against the REAL SessionStore from the tree
# ---------------------------------------------------------------------------

class _RoutingCfg:
    write_sessions_json = True


def _make_real_store(tmp_path):
    """Build a real SessionStore over a scratch sessions dir."""
    if not HERMES_TREE.is_dir():
        pytest.skip("Hermes tree not present on this host")
    sys.path.insert(0, str(HERMES_TREE))
    from gateway.session import SessionStore  # noqa: F401

    store = SessionStore(sessions_dir=tmp_path / "sessions", config=_RoutingCfg())
    return store


def _real_source():
    if not HERMES_TREE.is_dir():
        pytest.skip("Hermes tree not present on this host")
    sys.path.insert(0, str(HERMES_TREE))
    from gateway.config import Platform
    from gateway.session import SessionSource

    return SessionSource(platform=Platform.LOCAL, chat_id="test-chat", chat_type="dm",
                         user_id="tester")


def _seed_route(store, *, updated_minutes_ago=0.0, token=None):
    from gateway.session import SessionEntry

    src = _real_source()
    key = "agent:main:local:dm:test-chat"
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


def test_maybe_reset_calls_real_reset_session(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120)

    gateway = SimpleNamespace(
        _session_key_for_source=lambda src: key,
    )
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy), \
         mock.patch.object(plugin, "datetime") as fake_dt:
        fake_dt.now.return_value = datetime.now()
        result = plugin._maybe_reset(event, gateway, store)

    assert result is None  # allow normal dispatch
    new_entry = store.lookup_by_session_key(key)
    assert new_entry is not None
    assert new_entry.session_id != "seed-sid"
    assert new_entry.is_fresh_reset is True


def test_maybe_reset_skips_when_within_threshold(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=5)

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert plugin._maybe_reset(event, gateway, store) is None

    assert store.lookup_by_session_key(key).session_id == "seed-sid"


def test_maybe_reset_defers_when_turn_in_flight(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=120, token="tok-123")

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert plugin._maybe_reset(event, gateway, store) is None

    # Route unchanged: the in-flight turn keeps its session.
    assert store.lookup_by_session_key(key).session_id == "seed-sid"


def test_maybe_reset_no_policy_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    key, entry = _seed_route(store, updated_minutes_ago=100000)

    gateway = SimpleNamespace(_session_key_for_source=lambda src: key)
    event = SimpleNamespace(source=_real_source(), internal=False)

    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=None):
        assert plugin._maybe_reset(event, gateway, store) is None
    assert store.lookup_by_session_key(key).session_id == "seed-sid"


def test_maybe_reset_survives_store_errors(plugin, tmp_path):
    class ExplodingStore:
        def lookup_by_session_key(self, key):
            raise RuntimeError("boom")

    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:x")
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert plugin._maybe_reset(event, gateway, ExplodingStore()) is None


def test_maybe_reset_no_route_no_op(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    gateway = SimpleNamespace(_session_key_for_source=lambda src: "agent:main:local:dm:none")
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert plugin._maybe_reset(event, gateway, store) is None


def test_maybe_reset_survives_missing_key_helper(plugin, tmp_path):
    store = _make_real_store(tmp_path)
    gateway = SimpleNamespace()  # no _session_key_for_source at all
    event = SimpleNamespace(source=_real_source(), internal=False)

    policy = {"mode": "idle", "idle_minutes": 60}
    with mock.patch.object(plugin, "_read_session_reset_policy", return_value=policy):
        assert plugin._maybe_reset(event, gateway, store) is None


# ---------------------------------------------------------------------------
# _read_session_reset_policy — profile resolution
# ---------------------------------------------------------------------------

def test_policy_reader_none_on_missing_config(plugin, tmp_path, monkeypatch):
    import types

    fake_const = types.ModuleType("hermes_constants")
    fake_const.get_hermes_home = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake_const)

    assert plugin._read_session_reset_policy(None) is None
    assert plugin._read_session_reset_policy("mom") is None


def test_policy_reader_reads_named_profile_config(plugin, tmp_path, monkeypatch):
    import types

    prof = tmp_path / "profiles" / "mom"
    prof.mkdir(parents=True)
    (prof / "config.yaml").write_text(
        "session_reset:\n  mode: both\n  idle_minutes: 45\n  at_hour: 4\n"
    )

    fake_const = types.ModuleType("hermes_constants")
    fake_const.get_hermes_home = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "hermes_constants", fake_const)

    fake_cfg = types.ModuleType("hermes_cli.config")
    fake_cfg.load_config_readonly = lambda path=None: {
        "session_reset": {"mode": "both", "idle_minutes": 45, "at_hour": 4}
    }
    fake_cli = types.ModuleType("hermes_cli")
    fake_cli.config = fake_cfg
    monkeypatch.setitem(sys.modules, "hermes_cli", fake_cli)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", fake_cfg)

    policy = plugin._read_session_reset_policy("mom")
    assert policy == {"mode": "both", "idle_minutes": 45, "at_hour": 4}
