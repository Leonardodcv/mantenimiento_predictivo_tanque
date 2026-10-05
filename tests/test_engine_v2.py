from pathlib import Path

import pandas as pd

from app.services.anomaly_engine import (
    AnomalyEngine,
    EXCLUDED_FROM_MODEL,
    build_cycles,
    build_summary,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "lecturas_mixtas_demo.tsv"


def load_df():
    return pd.read_csv(DATA, sep="\t")


def test_mixed_history_legacy_and_enriched_rows_are_supported():
    df = load_df()
    engine = AnomalyEngine().fit(df)
    scored = engine.score(df)
    assert len(scored) == len(df)
    assert (scored["contexto_version"] == "LEGACY").any()
    assert (scored["contexto_version"] == "ENRIQUECIDO").any()
    assert scored["indice_anomalia"].between(0, 100).all()


def test_energy_and_manual_start_counter_are_not_ml_features():
    assert "energia_aparente_l1" in EXCLUDED_FROM_MODEL
    assert "numero_arranques" in EXCLUDED_FROM_MODEL
    df = load_df()
    engine = AnomalyEngine().fit(df)
    used = {f for model in engine.models.values() for f in model.features}
    assert "energia_aparente_l1" not in used
    assert "numero_arranques" not in used


def test_known_hydraulic_mismatch_is_detected():
    df = load_df()
    engine = AnomalyEngine().fit(df)
    scored = engine.score(df)
    scored["fecha_hora"] = pd.to_datetime(scored["fecha_hora"])
    window = scored[
        scored["fecha_hora"].between(
            pd.Timestamp("2026-10-02 10:53:38"),
            pd.Timestamp("2026-10-02 10:53:44.999"),
        )
    ]
    assert not window.empty
    assert float(window["indice_anomalia"].max()) >= 95


def test_pressure_high_is_warning_not_direct_failure():
    df = load_df()
    engine = AnomalyEngine().fit(df)
    scored = engine.score(df)
    candidates = scored[
        scored["presion_alta"].fillna(0).astype(bool)
        & (scored["indice_reglas"] == 0)
        & (scored["indice_ml"] < 40)
    ]
    assert not candidates.empty
    assert (candidates["indice_advertencias"] > 0).all()
    assert (candidates["indice_anomalia"] < 40).all()


def test_cycles_and_summary_are_built():
    df = load_df()
    engine = AnomalyEngine().fit(df)
    scored = engine.score(df)
    cycles = build_cycles(scored)
    summary = build_summary(scored)
    assert len(cycles) > 10
    assert summary["total_registros"] == len(df)
    assert summary["conteo_por_contexto"]["LEGACY"] > 0
    assert summary["conteo_por_contexto"]["ENRIQUECIDO"] > 0
