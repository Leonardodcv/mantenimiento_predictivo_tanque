from types import SimpleNamespace

import pandas as pd

from app import main
from app.services import data_source
from app.services.model_manager import ModelManager


def _rows():
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "fecha_hora": pd.to_datetime(
                [
                    "2026-10-08 07:00:00",
                    "2026-10-08 08:00:00",
                    "2026-10-08 09:00:00",
                    "2026-10-08 10:00:00",
                ]
            ),
            "flujo_instantaneo": [18.0, 18.1, 18.2, 18.3],
            "presion_relativa": [8.5, 8.6, 8.7, 8.8],
        }
    )


def test_v502_file_time_window_uses_latest_record_as_end_and_keeps_context(monkeypatch):
    frame = _rows()
    monkeypatch.setattr(data_source, "read_file", lambda limit=None: frame.copy())

    data, start, end = data_source.read_file_time_window(hours=1.0, context_rows=1)

    assert end == pd.Timestamp("2026-10-08 10:00:00")
    assert start == pd.Timestamp("2026-10-08 09:00:00")
    # 08:00 is retained only as context; 09:00 and 10:00 are the requested window.
    assert data["id"].tolist() == [2, 3, 4]


def test_v502_model_manager_filters_context_out_of_returned_timeline(monkeypatch):
    raw = pd.DataFrame(
        {
            "id": [10, 11, 12],
            "fecha_hora": pd.to_datetime(
                ["2026-10-08 08:59:59", "2026-10-08 09:00:00", "2026-10-08 10:00:00"]
            ),
        }
    )

    class FakeEngine:
        def score(self, df):
            out = df.copy()
            out["indice_anomalia"] = [5.0, 20.0, 90.0]
            out["nivel_anomalia"] = ["BAJO", "BAJO", "CRITICO"]
            return out

    manager = ModelManager()
    monkeypatch.setattr(
        manager,
        "get",
        lambda source: SimpleNamespace(engine=FakeEngine()),
    )
    monkeypatch.setattr(
        "app.services.model_manager.load_time_window",
        lambda source, hours, context_rows=0: (
            raw.copy(),
            pd.Timestamp("2026-10-08 09:00:00"),
            pd.Timestamp("2026-10-08 10:00:00"),
        ),
    )

    scored, start, end = manager.score_time_window("sqlserver", hours=1.0)

    assert start == pd.Timestamp("2026-10-08 09:00:00")
    assert end == pd.Timestamp("2026-10-08 10:00:00")
    assert scored["id"].tolist() == [11, 12]
    assert scored["indice_anomalia"].tolist() == [20.0, 90.0]


def test_v502_timeline_endpoint_returns_moment_and_anomaly_index(monkeypatch):
    scored = pd.DataFrame(
        {
            "id": [100, 101],
            "fecha_hora": pd.to_datetime(
                ["2026-10-08 09:59:59", "2026-10-08 10:00:00"]
            ),
            "indice_anomalia": [14.25, 82.5],
            "nivel_anomalia": ["BAJO", "ALTO"],
        }
    )
    monkeypatch.setattr(
        main.model_manager,
        "score_time_window",
        lambda source, hours: (
            scored.copy(),
            pd.Timestamp("2026-10-08 09:00:00"),
            pd.Timestamp("2026-10-08 10:00:00"),
        ),
    )

    response = main.anomaly_timeline(hours=1.0, source="sqlserver")

    assert response["api_version"] == "5.0.2"
    assert response["horas_solicitadas"] == 1.0
    assert response["ventana"]["hasta"] == "2026-10-08T10:00:00"
    assert response["count"] == 2
    assert response["data"][0]["momento_comparacion"] == "2026-10-08T09:59:59"
    assert response["data"][0]["indice_anomalia"] == 14.25
    assert response["data"][1]["indice_anomalia"] == 82.5
