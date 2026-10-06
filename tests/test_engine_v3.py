import pandas as pd

from app.services.anomaly_engine_v3 import (
    AnomalyEngine,
    _cycle_signature_detail,
    build_cycles,
    build_events,
)


def test_v3_rest_substate_detects_drain_from_level_slope():
    start = pd.Timestamp("2026-10-06 08:00:00")
    df = pd.DataFrame(
        {
            "fecha_hora": [start + pd.Timedelta(seconds=i * 5) for i in range(4)],
            "velocidad": [0, 0, 0, 0],
            "voltaje_salida": [0, 0, 0, 0],
            "nivel_tanque": [82.0, 80.5, 79.0, 77.5],
            "flujo_instantaneo": [0, 0, 0, 0],
            "presion_relativa": [0, 0, 0, 0],
        }
    )
    prepared = AnomalyEngine()._prepare(df)
    assert prepared.iloc[1]["fase_operativa"] == "REPOSO"
    assert prepared.iloc[1]["fase_modelo"] == "VACIADO"
    assert prepared.iloc[1]["subestado_reposo"] == "VACIADO"


def test_v3_long_stop_marks_restart_then_stabilization_cycles():
    rows = []
    t = pd.Timestamp("2026-10-06 00:00:00")

    def cycle(at):
        return [
            {"fecha_hora": at, "velocidad": 1200, "voltaje_salida": 155, "nivel_tanque": 40},
            {"fecha_hora": at + pd.Timedelta(seconds=2), "velocidad": 0, "voltaje_salida": 0, "nivel_tanque": 41},
            {"fecha_hora": at + pd.Timedelta(seconds=20), "velocidad": 0, "voltaje_salida": 0, "nivel_tanque": 40},
        ]

    rows += cycle(t)
    t = t + pd.Timedelta(hours=3)
    rows += cycle(t)  # REANUDACION
    for _ in range(5):
        t = t + pd.Timedelta(minutes=4)
        rows += cycle(t)

    prepared = AnomalyEngine()._prepare(pd.DataFrame(rows))
    starts = prepared.loc[prepared["en_movimiento"], ["contexto_ciclo", "numero_ciclo_estabilizacion"]]
    contexts = starts["contexto_ciclo"].tolist()
    assert contexts[0] == "INICIO_HISTORIAL"
    assert contexts[1] == "REANUDACION"
    assert contexts[2:6] == ["ESTABILIZACION"] * 4
    assert contexts[6] == "NORMAL"


def test_v3_engineering_tolerance_blocks_tiny_but_statistically_large_shift():
    # Similar al arranque del 6-Oct: +~2 % de flujo y -~3.5 % de presion.
    flow = _cycle_signature_detail(18.58, 18.2183, 0.02, 0.5, 0.03)
    pressure = _cycle_signature_detail(8.3659, 8.6697, 0.02, 0.35, 0.04)
    assert flow["z_robusto"] > 4
    assert pressure["z_robusto"] > 4
    assert flow["score"] == 0
    assert pressure["score"] == 0


def test_v3_engineering_tolerance_still_flags_large_physical_shift():
    detail = _cycle_signature_detail(12.0, 18.2, 0.03, 0.5, 0.03)
    assert detail["score"] >= 90
    assert detail["ratio_tolerancia"] > 3


def test_v3_statistical_rest_does_not_open_event_but_physical_rule_does():
    start = pd.Timestamp("2026-10-06 08:00:00")
    rest = pd.DataFrame(
        {
            "fecha_hora": [start + pd.Timedelta(seconds=i * 5) for i in range(6)],
            "indice_anomalia": [95.0] * 6,
            "indice_reglas": [0.0] * 6,
            "fase_operativa": ["REPOSO"] * 6,
            "fase_modelo": ["REPOSO_ESTATICO"] * 6,
            "contexto_ciclo": ["NO_APLICA"] * 6,
            "razones": [[] for _ in range(6)],
            "advertencias": [[] for _ in range(6)],
            "observaciones": [[] for _ in range(6)],
        }
    )
    assert build_events(rest, threshold=80) == []

    rest.loc[2, "indice_reglas"] = 96.0
    events = build_events(rest, threshold=80)
    assert len(events) == 1
    assert events[0]["criterio_apertura"] == "regla_fisica_inmediata"


def test_v3_restart_cycle_keeps_raw_signature_but_caps_effective_cycle_score():
    rows = []
    base = pd.Timestamp("2026-10-05 12:00:00")

    def add_cycle(cid, start, duration, level_start, level_end, flow, pressure, context, p95=18.0):
        rows.extend(
            [
                {
                    "ciclo_id": float(cid),
                    "fecha_hora": start,
                    "en_movimiento": True,
                    "fase_operativa": "ARRANQUE",
                    "nivel_tanque": level_start,
                    "flujo_instantaneo": flow,
                    "presion_relativa": pressure,
                    "velocidad": 1198,
                    "indice_anomalia": p95,
                    "indice_ml_crudo": 95.0 if context == "REANUDACION" else p95,
                    "indice_reglas": 0.0,
                    "duracion_presion_alta_seg": 0,
                    "contexto_version": "ENRIQUECIDO",
                    "contexto_ciclo": context,
                    "horas_paro_previas_al_ciclo": 12.0 if context == "REANUDACION" else 0.03,
                    "numero_ciclo_estabilizacion": 0 if context == "REANUDACION" else None,
                    "razones": [],
                    "advertencias": [],
                    "observaciones": [],
                },
                {
                    "ciclo_id": float(cid),
                    "fecha_hora": start + pd.Timedelta(seconds=duration),
                    "en_movimiento": True,
                    "fase_operativa": "OPERACION_ESTABLE",
                    "nivel_tanque": level_end,
                    "flujo_instantaneo": flow,
                    "presion_relativa": pressure,
                    "velocidad": 1198,
                    "indice_anomalia": p95,
                    "indice_ml_crudo": 95.0 if context == "REANUDACION" else p95,
                    "indice_reglas": 0.0,
                    "duracion_presion_alta_seg": duration,
                    "contexto_version": "ENRIQUECIDO",
                    "contexto_ciclo": context,
                    "horas_paro_previas_al_ciclo": 12.0 if context == "REANUDACION" else 0.03,
                    "numero_ciclo_estabilizacion": 0 if context == "REANUDACION" else None,
                    "razones": [],
                    "advertencias": [],
                    "observaciones": [],
                },
                {
                    "ciclo_id": float(cid),
                    "fecha_hora": start + pd.Timedelta(seconds=duration + 1),
                    "en_movimiento": False,
                    "fase_operativa": "POST_PARO",
                    "nivel_tanque": level_end,
                    "flujo_instantaneo": 0,
                    "presion_relativa": 0.1,
                    "velocidad": 0,
                    "indice_anomalia": 10.0,
                    "indice_ml_crudo": 10.0,
                    "indice_reglas": 0.0,
                    "duracion_presion_alta_seg": 0,
                    "contexto_version": "ENRIQUECIDO",
                    "contexto_ciclo": context,
                    "horas_paro_previas_al_ciclo": 12.0 if context == "REANUDACION" else 0.03,
                    "numero_ciclo_estabilizacion": 0 if context == "REANUDACION" else None,
                    "razones": [],
                    "advertencias": [],
                    "observaciones": [],
                },
            ]
        )

    for cid in range(1, 16):
        jitter = ((cid % 5) - 2)
        add_cycle(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            98 + jitter * 0.3,
            39.8 + jitter * 0.03,
            82.5 + jitter * 0.04,
            18.22 + jitter * 0.01,
            8.67 + jitter * 0.01,
            "NORMAL",
        )
    add_cycle(16, base + pd.Timedelta(hours=12), 21, 76.2, 83.2, 18.84, 7.62, "REANUDACION", p95=39.0)

    cycles = build_cycles(pd.DataFrame(rows))
    restart = cycles[-1]
    assert restart["contexto_ciclo"] == "REANUDACION"
    assert restart["indice_firma_ciclo_crudo"] >= 90
    assert restart["indice_firma_ciclo"] <= 39
    assert restart["indice_ciclo"] <= 39
    assert restart["nivel_ciclo"] == "BAJO"
