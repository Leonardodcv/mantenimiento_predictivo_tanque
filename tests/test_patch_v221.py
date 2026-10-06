from datetime import datetime, timezone

import pandas as pd

from app.services.model_manager import ModelManager, ModelSnapshot


class _NoopEngine:
    training_summary = {}

    def score(self, raw):
        return raw.copy()


def _snapshot(scored: pd.DataFrame, cycles=None) -> ModelSnapshot:
    return ModelSnapshot(
        source="sqlserver",
        built_at=datetime(2026, 10, 6, 14, 0, tzinfo=timezone.utc),
        engine=_NoopEngine(),
        raw_full=scored.copy(),
        scored_full=scored.copy(),
        summary={},
        cycles=list(cycles or []),
        coverage=[],
    )


def _persistent_event_frame() -> pd.DataFrame:
    start = pd.Timestamp("2026-10-02 10:53:36")
    indices = [10, 85, 86, 87, 55, 55, 55, 55, 55, 55, 55]
    return pd.DataFrame(
        {
            "fecha_hora": [start + pd.Timedelta(seconds=i) for i in range(len(indices))],
            "indice_anomalia": indices,
            "indice_reglas": [0.0] * len(indices),
            "fase_operativa": ["OPERACION_ESTABLE"] * len(indices),
            "razones": [[] for _ in indices],
            "advertencias": [[] for _ in indices],
        }
    )


def test_v221_cycles_latest_returns_last_snapshot_cycles_without_recent_sql_read(monkeypatch):
    manager = ModelManager()
    manager._snapshots["sqlserver"] = _snapshot(
        pd.DataFrame(),
        cycles=[{"ciclo_id": 10}, {"ciclo_id": 11}, {"ciclo_id": 12}],
    )

    def _must_not_read_latest(*args, **kwargs):
        raise AssertionError("cycles_latest no debe depender de la ventana SQL reciente")

    monkeypatch.setattr("app.services.model_manager.load_latest", _must_not_read_latest)

    cycles = manager.cycles_latest("sqlserver", limit=2)
    assert [item["ciclo_id"] for item in cycles] == [11, 12]


def test_v221_events_full_uses_snapshot_complete_without_recent_sql_read(monkeypatch):
    manager = ModelManager()
    manager._snapshots["sqlserver"] = _snapshot(_persistent_event_frame())

    def _must_not_read_latest(*args, **kwargs):
        raise AssertionError("scope=full no debe consultar la ventana SQL reciente")

    monkeypatch.setattr("app.services.model_manager.load_latest", _must_not_read_latest)

    events = manager.events("sqlserver", threshold=80, scope="full", limit_rows=5000)
    assert len(events) == 1
    assert events[0]["criterio_apertura"] == "persistencia_estadistica"


def test_v221_events_recent_may_be_empty_when_machine_was_idle(monkeypatch):
    manager = ModelManager()
    manager._snapshots["sqlserver"] = _snapshot(_persistent_event_frame())

    idle_start = pd.Timestamp("2026-10-06 00:00:00")
    idle = pd.DataFrame(
        {
            "fecha_hora": [idle_start + pd.Timedelta(seconds=i * 5) for i in range(20)],
            "indice_anomalia": [10.0] * 20,
            "indice_reglas": [0.0] * 20,
            "fase_operativa": ["REPOSO"] * 20,
            "razones": [[] for _ in range(20)],
            "advertencias": [[] for _ in range(20)],
        }
    )

    monkeypatch.setattr("app.services.model_manager.load_latest", lambda *args, **kwargs: idle)

    events = manager.events("sqlserver", threshold=80, scope="recent", limit_rows=5000)
    assert events == []
