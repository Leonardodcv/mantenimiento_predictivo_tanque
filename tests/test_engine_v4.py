import pandas as pd

from app.services.anomaly_engine_v4 import (
    AnomalyEngine,
    COMPLETE_CYCLE_TYPE,
    build_cycles,
    build_events,
    create_cycle_baseline_reference,
    cycle_baseline_summary,
)
from app.services.equipment_context import EQUIPMENT_CONTEXT, pump_expected_head_from_nameplate


def _cycle_rows(
    cycle_id,
    start,
    *,
    duration=98,
    level_start=39.75,
    level_end=82.5,
    flow=18.25,
    pressure=8.63,
    score=18.0,
    freq=40.0,
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
                "fase_modelo": "ARRANQUE" if i < 4 else "OPERACION_ESTABLE",
                "nivel_tanque": level_start + (level_end - level_start) * frac,
                "setpoint_llenado": 80.0,
                "flujo_instantaneo": flow,
                "presion_relativa": pressure,
                "velocidad": 1198,
                "velocidad_vfd_raw": 1198,
                "frecuencia_contexto_hz": freq,
                "ratio_presion_flujo": pressure / flow,
                "presion_bomba_teorica_aprox_psi": 10.5,
                "residuo_presion_vs_modelo_teorico_psi": pressure - 10.5,
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
            "fase_modelo": "POST_PARO",
            "nivel_tanque": level_end,
            "setpoint_llenado": 80.0,
            "flujo_instantaneo": 0.0,
            "presion_relativa": 0.1,
            "velocidad": 0,
            "velocidad_vfd_raw": 0,
            "frecuencia_contexto_hz": freq,
            "ratio_presion_flujo": None,
            "presion_bomba_teorica_aprox_psi": None,
            "residuo_presion_vs_modelo_teorico_psi": None,
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


def _raw_active_rows(start, pressure, flow=18.25, seconds=100):
    rows = []
    for i in range(seconds):
        rows.append(
            {
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "flujo_instantaneo": flow,
                "presion_relativa": pressure,
                "temperatura_tanque": 23.0,
                "nivel_tanque": 40.0 + i * 0.3,
                "frecuencia_salida": 40,
                "frecuencia_vfd_hz": 40.0,
                "velocidad": 1198,
                "voltaje_salida": 155,
                "voltaje_bus_dc": 293,
                "tension_l1_n": 264,
                "corriente_l1": 0.14,
                "potencia_activa_l1": 26.0,
                "potencia_aparente_l1": 38.0,
                "falla_vfd": False,
                "falta_presion": False,
                "bajo_flujo": False,
                "presion_alta": True,
                "estado_vfd": 2,
                "manual_auto_vfd": True,
            }
        )
    return rows


def test_v4_equipment_context_captures_new_photo_information():
    assert EQUIPMENT_CONTEXT["pump"]["model"] == "PK 60"
    assert EQUIPMENT_CONTEXT["pump"]["nameplate"]["nominal_frequency_hz"] == 60.0
    assert EQUIPMENT_CONTEXT["manual_valves"]["main_red"]["normal_operating_position"] == "ABIERTA"
    assert EQUIPMENT_CONTEXT["manual_valves"]["blue_drain_valves"]["count"] == 2
    assert EQUIPMENT_CONTEXT["sensors"]["level"]["model"] == "Prosonic T FMU30"
    assert EQUIPMENT_CONTEXT["units_and_scaling"]["velocity_column"]["rpm_confirmed"] is False


def test_v4_pump_reference_is_context_only_and_reasonable_at_40hz():
    ref = pump_expected_head_from_nameplate(18.25, 40.0)
    assert ref["aplicable"] is True
    assert ref["solo_contexto"] is True
    assert 6.5 < ref["altura_teorica_aprox_m"] < 8.5
    assert 9.0 < ref["presion_teorica_aprox_psi"] < 12.5


def test_v4_prepare_exposes_velocity_as_raw_and_process_context():
    rows = _raw_active_rows(pd.Timestamp("2026-10-07 10:00:00"), 8.63, seconds=40)
    engine = AnomalyEngine()
    prepared = engine._prepare(pd.DataFrame(rows))
    row = prepared.iloc[-1]
    assert row["velocidad_vfd_raw"] == 1198
    assert bool(row["velocidad_rpm_confirmada"]) is False
    assert row["velocidad_unidad"] == "RAW_VFD_NO_CONFIRMADA"
    assert row["modo_operacion_contextual"] == "AUTOMATICO"
    assert row["frecuencia_contexto_hz"] == 40.0
    assert bool(row["modelo_bomba_fisico_solo_contexto"]) is True


def test_v4_event_explanations_do_not_mix_regimes_for_same_variable():
    start = pd.Timestamp("2026-10-07 11:00:00")
    rows = []
    for i in range(6):
        regime = "ARRANQUE" if i < 3 else "OPERACION_ESTABLE"
        rows.append(
            {
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "indice_anomalia": 90.0,
                "indice_ml": 90.0,
                "indice_reglas": 0.0,
                "fase_operativa": regime,
                "fase_modelo": regime,
                "contexto_ciclo": "NORMAL",
                "razones": ["patron_multivariable_atipico_para_el_regimen"],
                "advertencias": [],
                "observaciones": [],
                "variables_mas_atipicas": [
                    {
                        "variable": "nivel_tanque",
                        "grupo": "HIDRAULICA",
                        "regimen_origen": regime,
                        "score_explicacion": 90.0,
                        "z_robusto_aprox": 5.4,
                        "desviacion_pct": 20.0,
                        "valor": 60.0,
                        "mediana_regimen": 40.0 if regime == "ARRANQUE" else 60.0,
                        "direccion": "ALTA",
                    }
                ],
            }
        )
    events = build_events(pd.DataFrame(rows), threshold=80)
    assert len(events) == 1
    top = events[0]["variables_mas_atipicas"]
    level_entries = [x for x in top if x["variable"] == "nivel_tanque"]
    assert {x["regimen_origen"] for x in level_entries} == {"ARRANQUE", "OPERACION_ESTABLE"}


def test_v4_separate_episodes_with_same_signature_share_regime_family():
    base = pd.Timestamp("2026-10-05 08:00:00")
    normal_rows = []
    for cid in range(1, 21):
        jitter = ((cid % 5) - 2)
        normal_rows += _cycle_rows(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            duration=98 + jitter * 0.4,
            flow=18.25 + jitter * 0.02,
            pressure=8.63 + jitter * 0.02,
        )
    initial = build_cycles(pd.DataFrame(normal_rows))
    reference = create_cycle_baseline_reference(initial)

    all_rows = list(normal_rows)
    # Episodio 1.
    for cid in range(21, 27):
        all_rows += _cycle_rows(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            duration=89,
            flow=19.18,
            pressure=7.68,
        )
    # Ciclos normales que cortan el episodio.
    for cid in range(27, 30):
        all_rows += _cycle_rows(cid, base + pd.Timedelta(minutes=4 * cid))
    # Episodio 2 con la misma firma.
    for cid in range(30, 36):
        all_rows += _cycle_rows(
            cid,
            base + pd.Timedelta(minutes=4 * cid),
            duration=89,
            flow=19.19,
            pressure=7.67,
        )

    cycles = build_cycles(pd.DataFrame(all_rows), baseline_reference=reference)
    episodes = {c["episodio_regimen_id"] for c in cycles if c.get("nuevo_regimen_candidato")}
    families = {c["familia_regimen_id"] for c in cycles if c.get("nuevo_regimen_candidato")}
    family_names = {c["familia_regimen_probable"] for c in cycles if c.get("nuevo_regimen_candidato")}
    assert episodes == {"EP-001", "EP-002"}
    assert families == {"RF-001"}
    assert family_names == {"CAMBIO_PUNTO_OPERACION_HIDRAULICO"}
    summary = cycle_baseline_summary(cycles, reference)
    assert len(summary["familias_regimen_candidatas"]) == 1
    assert summary["familias_regimen_candidatas"][0]["numero_episodios"] == 2


def test_v4_complete_cycle_type_no_longer_calls_structure_normal():
    rows = _cycle_rows(1, pd.Timestamp("2026-10-07 12:00:00"))
    cycle = build_cycles(pd.DataFrame(rows))[0]
    assert cycle["tipo_ciclo"] == COMPLETE_CYCLE_TYPE


def test_v4_protected_active_ml_training_uses_only_trusted_cycle_ids():
    start = pd.Timestamp("2026-10-07 13:00:00")
    rows = []
    # Ciclo 1: sirve solo para que ciclo 2 ya no sea INICIO_HISTORIAL.
    rows += _raw_active_rows(start, 8.63, seconds=40)
    for i in range(15):
        rows.append(
            {
                **_raw_active_rows(start, 8.63, seconds=1)[0],
                "fecha_hora": start + pd.Timedelta(seconds=40 + i),
                "velocidad": 0,
                "voltaje_salida": 0,
                "flujo_instantaneo": 0.0,
                "presion_relativa": 0.1,
                "presion_alta": False,
            }
        )
    # Ciclo 2 confiable.
    rows += _raw_active_rows(start + pd.Timedelta(seconds=55), 8.63, seconds=80)
    for i in range(15):
        rows.append(
            {
                **_raw_active_rows(start, 8.63, seconds=1)[0],
                "fecha_hora": start + pd.Timedelta(seconds=135 + i),
                "velocidad": 0,
                "voltaje_salida": 0,
                "flujo_instantaneo": 0.0,
                "presion_relativa": 0.1,
                "presion_alta": False,
            }
        )
    # Ciclo 3 desplazado, que no debe entrar al fit protegido.
    rows += _raw_active_rows(start + pd.Timedelta(seconds=150), 7.0, flow=19.2, seconds=80)

    protected = AnomalyEngine().fit(pd.DataFrame(rows), trusted_cycle_ids={2})
    bundle = protected.models["OPERACION_ESTABLE"]
    assert bundle.feature_stats["presion_relativa"]["median"] > 8.5
    assert protected.training_summary["v4_protected_ml_training"]["applied"] is True
    assert protected.training_summary["v4_protected_ml_training"]["trusted_cycle_count"] == 1

