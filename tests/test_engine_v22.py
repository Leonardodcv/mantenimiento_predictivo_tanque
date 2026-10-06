import pandas as pd

from app.services.anomaly_engine import (
    AnomalyEngine,
    build_cycles,
    build_events,
    variable_coverage,
)


def _event_frame(indices, rule_indices=None):
    if rule_indices is None:
        rule_indices = [0.0] * len(indices)
    start = pd.Timestamp("2026-10-06 08:00:00")
    return pd.DataFrame(
        {
            "fecha_hora": [start + pd.Timedelta(seconds=i) for i in range(len(indices))],
            "indice_anomalia": indices,
            "indice_reglas": rule_indices,
            "fase_operativa": ["OPERACION_ESTABLE"] * len(indices),
            "razones": [[] for _ in indices],
            "advertencias": [[] for _ in indices],
        }
    )


def test_v22_isolated_statistical_peak_does_not_open_event():
    df = _event_frame([10, 10, 95, 10, 10, 10, 10, 10, 10])
    assert build_events(df, threshold=80) == []


def test_v22_persistent_statistical_signal_opens_event():
    df = _event_frame([10, 85, 86, 87, 65, 55, 55, 55, 55, 55, 55])
    events = build_events(df, threshold=80)
    assert len(events) == 1
    assert events[0]["criterio_apertura"] == "persistencia_estadistica"
    assert events[0]["registros"] >= 3


def test_v22_strong_physical_rule_opens_event_immediately():
    df = _event_frame([10, 96, 10, 10, 10, 10, 10, 10], [0, 96, 0, 0, 0, 0, 0, 0])
    events = build_events(df, threshold=80)
    assert len(events) == 1
    assert events[0]["criterio_apertura"] == "regla_fisica_inmediata"


def test_v22_cycle_uses_p95_not_single_instant_peak():
    start = pd.Timestamp("2026-10-06 09:00:00")
    rows = []
    for i in range(21):
        moving = i < 20
        rows.append(
            {
                "ciclo_id": 1.0,
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "en_movimiento": moving,
                "fase_operativa": "OPERACION_ESTABLE" if moving else "POST_PARO",
                "nivel_tanque": 40 + i * 2,
                "flujo_instantaneo": 18.2 if moving else 0.0,
                "presion_relativa": 8.6 if moving else 0.1,
                "velocidad": 1197 if moving else 0,
                "indice_anomalia": 100.0 if i == 10 else 10.0,
                "indice_reglas": 0.0,
                "duracion_presion_alta_seg": float(i if moving else 0),
                "contexto_version": "ENRIQUECIDO",
                "razones": [],
                "advertencias": ["primer_umbral_presion_alta_activo"] if moving else [],
                "observaciones": [],
            }
        )
    cycles = build_cycles(pd.DataFrame(rows))
    assert len(cycles) == 1
    cycle = cycles[0]
    assert cycle["ciclo_completo"] is True
    assert cycle["indice_maximo_instantaneo"] == 100.0
    assert cycle["tuvo_pico_atipico"] is True
    assert cycle["indice_ciclo"] < 70
    assert cycle["nivel_ciclo"] in {"BAJO", "MEDIO"}


def test_v22_pressure_high_is_separate_warning_and_not_final_index():
    row = pd.Series(
        {
            "fase_operativa": "OPERACION_ESTABLE",
            "velocidad": 1197,
            "flujo_instantaneo": 18.2,
            "presion_relativa": 8.6,
            "segundos_desde_arranque": 30,
            "presion_alta": 1,
            "duracion_presion_alta_seg": 25,
            "falta_presion": 0,
            "bajo_flujo": 0,
            "falla_vfd": 0,
            "estado_vfd": 2,
        }
    )
    hard, warning, reasons, warnings, observations = AnomalyEngine()._context_rules(row)
    assert hard == 0
    assert warning > 0
    assert reasons == []
    assert "umbral_presion_alta_persistente" in warnings
    assert observations == []


def test_v22_variable_status_includes_all_new_context_signals():
    names = {item["variable"] for item in variable_coverage(pd.DataFrame([{}]))}
    for expected in {
        "paro_emergencia_raw",
        "frecuencia_vfd_hz",
        "manual_auto_s1",
        "manual_auto_s2",
        "recirculacion_automatica",
        "arranque_paro_llenado",
        "arranque_paro_vaciado",
    }:
        assert expected in names
