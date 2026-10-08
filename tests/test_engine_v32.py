import pandas as pd

from app.services.anomaly_engine_v32 import (
    AnomalyEngine,
    BASELINE_EXCLUDED,
    BASELINE_QUARANTINE,
    BASELINE_TRUSTED,
    PARTIAL_CYCLE_TYPE,
    build_cycles,
    create_cycle_baseline_reference,
)


def _cycle_rows(
    cycle_id,
    start,
    *,
    duration=98,
    level_start=39.75,
    level_end=82.5,
    flow=18.27,
    pressure=8.62,
    score=18.0,
    setpoint=80.0,
):
    rows = []
    n = max(8, int(duration) + 1)
    for i in range(n):
        frac = i / max(1, n - 1)
        rows.append(
            {
                "ciclo_id": float(cycle_id),
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "en_movimiento": True,
                "fase_operativa": "ARRANQUE" if i < 4 else "OPERACION_ESTABLE",
                "nivel_tanque": level_start + (level_end - level_start) * frac,
                "setpoint_llenado": setpoint,
                "flujo_instantaneo": flow,
                "presion_relativa": pressure,
                "velocidad": 1198,
                "indice_anomalia": score,
                "indice_ml_crudo": score,
                "indice_reglas": 0.0,
                "duracion_presion_alta_seg": float(i),
                "contexto_version": "ENRIQUECIDO",
                "contexto_ciclo": "NORMAL",
                "horas_paro_previas_al_ciclo": 0.04,
                "numero_ciclo_estabilizacion": None,
                "razones": [],
                "advertencias": [],
                "observaciones": [],
            }
        )
    rows.append(
        {
            "ciclo_id": float(cycle_id),
            "fecha_hora": start + pd.Timedelta(seconds=n),
            "en_movimiento": False,
            "fase_operativa": "POST_PARO",
            "nivel_tanque": level_end,
            "setpoint_llenado": setpoint,
            "flujo_instantaneo": 0.0,
            "presion_relativa": 0.1,
            "velocidad": 0,
            "indice_anomalia": 10.0,
            "indice_ml_crudo": 10.0,
            "indice_reglas": 0.0,
            "duracion_presion_alta_seg": 0.0,
            "contexto_version": "ENRIQUECIDO",
            "contexto_ciclo": "NORMAL",
            "horas_paro_previas_al_ciclo": 0.04,
            "numero_ciclo_estabilizacion": None,
            "razones": [],
            "advertencias": [],
            "observaciones": [],
        }
    )
    return rows


def test_v32_protected_baseline_quarantines_shift_and_detects_new_regime():
    base = pd.Timestamp("2026-10-06 08:00:00")
    normal_rows = []
    for cid in range(1, 21):
        jitter = ((cid % 5) - 2)
        normal_rows += _cycle_rows(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            duration=98 + jitter * 0.2,
            flow=18.27 + jitter * 0.01,
            pressure=8.62 + jitter * 0.01,
        )
    first_cycles = build_cycles(pd.DataFrame(normal_rows))
    reference = create_cycle_baseline_reference(first_cycles)
    assert reference["global"]["metrics"]["presion_estable_mediana"]["median"] > 8.5

    all_rows = list(normal_rows)
    for offset, cid in enumerate(range(21, 29)):
        all_rows += _cycle_rows(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            duration=89,
            flow=19.16,
            pressure=7.70,
        )
    cycles = build_cycles(pd.DataFrame(all_rows), baseline_reference=reference)
    shifted = cycles[-8:]
    assert all(c["estado_baseline"] == BASELINE_QUARANTINE for c in shifted)
    assert all(c["nuevo_regimen_candidato"] for c in shifted)
    assert len({c["regimen_candidato_id"] for c in shifted}) == 1
    # La referencia congelada no se desplaza hacia 7.7.
    assert reference["global"]["metrics"]["presion_estable_mediana"]["median"] > 8.5


def test_v32_explanation_uses_physical_scale_floor_when_mad_is_zero():
    start = pd.Timestamp("2026-10-06 10:00:00")
    rows = []
    for i in range(100):
        rows.append(
            {
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "flujo_instantaneo": 18.27,
                "presion_relativa": 8.62,
                "nivel_tanque": 45 + i * 0.3,
                "frecuencia_salida": 40,
                "velocidad": 1198,
                "voltaje_salida": 155,
                "voltaje_bus_dc": 293,
                "tension_l1_n": 264,
                "corriente_l1": 0.145,
                "potencia_activa_l1": 26.0,
                "potencia_aparente_l1": 38.5,
                "falla_vfd": False,
                "falta_presion": False,
                "bajo_flujo": False,
                "presion_alta": True,
                "estado_vfd": 2,
            }
        )
    engine = AnomalyEngine().fit(pd.DataFrame(rows))
    row = engine._prepare(
        pd.DataFrame(
            [
                {
                    **rows[-1],
                    "fecha_hora": start + pd.Timedelta(seconds=101),
                    "voltaje_salida": 164,
                }
            ]
        )
    ).iloc[0]
    items = engine._feature_explanations(row, "OPERACION_ESTABLE", 90.0)
    voltage = next(x for x in items if x["variable"] == "voltaje_salida")
    assert voltage["escala_limitada_por_piso_fisico"] is True
    assert voltage["piso_escala_fisica"] == 2.0
    assert voltage["z_robusto_aprox"] <= 4.5


def test_v32_partial_cycle_near_fill_setpoint_is_not_compared_to_full_cycle_baseline():
    base = pd.Timestamp("2026-10-06 08:00:00")
    rows = []
    for cid in range(1, 16):
        rows += _cycle_rows(cid, base + pd.Timedelta(minutes=4 * cid))
    baseline_cycles = build_cycles(pd.DataFrame(rows))
    reference = create_cycle_baseline_reference(baseline_cycles)

    rows += _cycle_rows(
        16,
        base + pd.Timedelta(hours=2),
        duration=16,
        level_start=79.5,
        level_end=83.3,
        flow=19.18,
        pressure=6.8,
        score=95.0,
        setpoint=80.0,
    )
    cycle = build_cycles(pd.DataFrame(rows), baseline_reference=reference)[-1]
    assert cycle["es_ciclo_parcial_nivel_inicial"] is True
    assert cycle["tipo_ciclo"] == PARTIAL_CYCLE_TYPE
    assert cycle["estado_baseline"] == BASELINE_EXCLUDED
    assert cycle["indice_firma_ciclo"] is None
    assert cycle["indice_ciclo"] is None
    assert cycle["nivel_ciclo"] == "NO_COMPARABLE"


def test_v32_clean_cycle_can_be_trusted_by_protected_baseline():
    base = pd.Timestamp("2026-10-06 08:00:00")
    rows = []
    for cid in range(1, 16):
        rows += _cycle_rows(cid, base + pd.Timedelta(minutes=4 * cid))
    cycles = build_cycles(pd.DataFrame(rows))
    assert sum(c["estado_baseline"] == BASELINE_TRUSTED for c in cycles) >= 12
