import pandas as pd

from app.services.anomaly_engine_v5 import (
    AnomalyEngine,
    BASELINE_EXCLUDED,
    build_cycles,
    build_summary,
)
from app.services.controlled_trials import (
    annotate_controlled_trials,
    load_controlled_trials_catalog,
    validate_controlled_trials,
)
from app.services.equipment_context import EQUIPMENT_CONTEXT


def _raw_row(ts, *, speed=1198, flow=18.2, pressure=8.6, level=50.0):
    return {
        "fecha_hora": pd.Timestamp(ts),
        "flujo_instantaneo": flow,
        "presion_relativa": pressure,
        "temperatura_tanque": 23.0,
        "nivel_tanque": level,
        "frecuencia_salida": 40.0,
        "frecuencia_vfd_hz": 40.0,
        "velocidad": speed,
        "voltaje_salida": 155.0 if speed else 0.0,
        "voltaje_bus_dc": 290.0,
        "tension_l1_n": 260.0,
        "corriente_l1": 0.14 if speed else 0.0,
        "potencia_activa_l1": 25.0 if speed else 0.0,
        "potencia_aparente_l1": 35.0 if speed else 0.0,
        "falla_vfd": False,
        "falta_presion": False,
        "bajo_flujo": False,
        "presion_alta": pressure > 6,
        "estado_vfd": 2 if speed else 1,
        "manual_auto_vfd": True,
    }


def test_v5_equipment_context_uses_only_upper_pressure_channel_currently():
    pressure = EQUIPMENT_CONTEXT["sensors"]["pressure"]
    assert pressure["physical_sensor_count_confirmed"] == 2
    assert pressure["active_data_channel_count"] == 1
    assert pressure["sensors"]["sensor_superior"]["location"] == "entre_valvula_azul_superior_y_tanque"
    assert pressure["sensors"]["sensor_superior"]["database_column"] == "presion_relativa"
    assert pressure["sensors"]["sensor_superior"]["data_available"] is True
    assert pressure["sensors"]["sensor_bomba"]["location"] == "entre_bomba_y_valvula_superior_naranja"
    assert pressure["sensors"]["sensor_bomba"]["data_available"] is False
    assert pressure["sensors"]["sensor_bomba"]["use_in_current_model"] is False
    assert pressure["database_mapping"]["differential_pressure_available"] is False
    assert EQUIPMENT_CONTEXT["manual_valves"]["upper_blue"]["used_to_induce_cavitation_reported"] is True
    assert EQUIPMENT_CONTEXT["manual_valves"]["tank_valve"]["hydraulic_effect"] == "recirculacion_corta"


def test_v5_pressure_context_maps_presion_relativa_to_upper_sensor_only():
    engine = AnomalyEngine()
    out = engine._add_v5_pressure_topology_context(pd.DataFrame({"presion_relativa": [8.25]}))
    row = out.iloc[0]
    assert row["sensor_presion_superior_columna"] == "presion_relativa"
    assert row["sensor_presion_superior_estado"] == "DISPONIBLE"
    assert row["presion_sensor_superior"] == 8.25
    assert row["sensor_presion_bomba_estado"] == "NO_DISPONIBLE"
    assert pd.isna(row["presion_sensor_bomba"])
    assert bool(row["presion_doble_canal_disponible"]) is False
    assert pd.isna(row["delta_presion_bomba_a_superior"])


def test_v5_catalog_preserves_minute_precision_and_no_assumed_recovery():
    catalog = load_controlled_trials_catalog()
    assert catalog["principles"]["do_not_assume_recovery_between_annotations"] is True
    session = catalog["sessions"][0]
    assert session["exclude_from_normal_ml_training"] is True
    assert session["exclude_from_cycle_baseline"] is True
    cavitation = next(x for x in catalog["annotations"] if x["id"] == "CT-0846")
    assert cavitation["precision"] == "MINUTO"
    assert set(cavitation["manipulations"]) == {
        "CIERRE_VALVULA_AZUL_SUPERIOR",
        "CIERRE_VALVULA_TANQUE",
    }


def test_v5_annotation_marks_whole_session_for_exclusion_but_only_logged_minute_as_ground_truth():
    df = pd.DataFrame(
        [
            _raw_row("2026-10-08 08:15:10"),  # sesion, minuto no anotado
            _raw_row("2026-10-08 08:46:20"),  # cavitacion anotada
            _raw_row("2026-10-08 09:29:10"),  # fuera de sesion
        ]
    )
    out = annotate_controlled_trials(df)
    assert bool(out.iloc[0]["es_sesion_pruebas_controladas"]) is True
    assert bool(out.iloc[0]["excluir_entrenamiento_normal"]) is True
    assert out.iloc[0]["ground_truth_ids"] == []
    assert out.iloc[1]["ground_truth_ids"] == ["CT-0846"]
    assert "CAVITACION_CONTROLADA_REPORTADA" in out.iloc[1]["ground_truth_tipos"]
    assert bool(out.iloc[2]["es_sesion_pruebas_controladas"]) is False


def test_v5_sensor_manipulation_is_data_quality_not_machine_failure_label():
    df = pd.DataFrame([_raw_row("2026-10-08 08:59:15")])
    out = annotate_controlled_trials(df).iloc[0]
    assert bool(out["ground_truth_manipulacion_sensor"]) is True
    assert "CALIDAD_DATO_SENSOR" in out["ground_truth_categorias"]
    assert out["ground_truth_falla_real"] is False


def test_v5_training_mask_excludes_controlled_session():
    rows = []
    # Historico normal fuera de la sesion.
    for i in range(50):
        rows.append(_raw_row(pd.Timestamp("2026-10-08 07:00:00") + pd.Timedelta(seconds=i)))
    # Prueba controlada con una presion extrema; no debe sesgar la mediana de entrenamiento.
    for i in range(50):
        rows.append(
            _raw_row(
                pd.Timestamp("2026-10-08 08:46:00") + pd.Timedelta(seconds=i),
                pressure=1.0,
                flow=2.0,
            )
        )
    engine = AnomalyEngine().fit(pd.DataFrame(rows))
    # Puede entrenar ARRANQUE/OPERACION_ESTABLE segun la derivacion temporal; al menos
    # el resumen debe confirmar que las 50 filas de prueba quedaron fuera.
    assert engine.training_summary["v5_controlled_ground_truth"]["records_excluded_from_normal_training"] == 50
    for bundle in engine.models.values():
        if "presion_relativa" in bundle.feature_stats:
            assert bundle.feature_stats["presion_relativa"]["median"] > 8.0


def _scored_cycle_rows(start: pd.Timestamp, cycle_id: int = 1):
    rows = []
    for i in range(12):
        rows.append(
            {
                "ciclo_id": float(cycle_id),
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "en_movimiento": True,
                "fase_operativa": "OPERACION_ESTABLE",
                "fase_modelo": "OPERACION_ESTABLE",
                "nivel_tanque": 40.0 + i * 3.5,
                "setpoint_llenado": 80.0,
                "flujo_instantaneo": 18.2,
                "presion_relativa": 8.6,
                "velocidad": 1198,
                "velocidad_vfd_raw": 1198,
                "frecuencia_contexto_hz": 40.0,
                "ratio_presion_flujo": 8.6 / 18.2,
                "presion_bomba_teorica_aprox_psi": 10.5,
                "residuo_presion_vs_modelo_teorico_psi": -1.9,
                "indice_anomalia": 20.0,
                "indice_ml_crudo": 20.0,
                "indice_reglas": 0.0,
                "duracion_presion_alta_seg": 0.0,
                "contexto_version": "ENRIQUECIDO",
                "contexto_ciclo": "NORMAL",
                "horas_paro_previas_al_ciclo": 0.1,
                "numero_ciclo_estabilizacion": None,
                "razones": [],
                "advertencias": [],
                "observaciones": [],
                "es_sesion_pruebas_controladas": True,
                "excluir_baseline_normal": True,
                "ground_truth_ids": ["CT-0846"],
                "ground_truth_tipos": ["CAVITACION_CONTROLADA_REPORTADA"],
                "ground_truth_categorias": ["PRUEBA_HIDRAULICA"],
            }
        )
    rows.append(
        {
            **rows[-1],
            "fecha_hora": start + pd.Timedelta(seconds=12),
            "en_movimiento": False,
            "fase_operativa": "POST_PARO",
            "fase_modelo": "POST_PARO",
        }
    )
    return rows


def test_v5_controlled_cycle_is_excluded_from_baseline():
    cycle = build_cycles(pd.DataFrame(_scored_cycle_rows(pd.Timestamp("2026-10-08 08:46:00"))))[0]
    assert cycle["estado_baseline"] == BASELINE_EXCLUDED
    assert "prueba_controlada_ground_truth" in cycle["motivos_cuarentena_baseline"]
    assert cycle["clasificacion_ciclo_operativa"].startswith("PRUEBA_CONTROLADA")


def test_v5_validation_compares_detector_without_turning_test_into_failure():
    scored = pd.DataFrame(
        [
            {
                "fecha_hora": pd.Timestamp("2026-10-08 08:46:20"),
                "indice_anomalia": 96.0,
                "indice_ml": 70.0,
                "indice_reglas": 96.0,
                "flujo_instantaneo": 0.2,
                "presion_relativa": 0.3,
                "nivel_tanque": 70.0,
                "en_movimiento": True,
            }
        ]
    )
    validation = validate_controlled_trials(scored, events=[], detection_threshold=70.0)
    row = next(x for x in validation["results"] if x["id"] == "CT-0846")
    assert row["validation_status"] == "RESPUESTA_DETECTADA_EN_VENTANA"
    assert row["detected_anomaly"] is True
    assert row["presion_fuente_sensor"] == "SENSOR_SUPERIOR"
    assert row["presion_sensor_bomba_disponible"] is False
    assert validation["pressure_validation_scope"]["database_column"] == "presion_relativa"
    assert validation["pressure_validation_scope"]["pump_sensor_available"] is False
    assert validation["ground_truth_policy"].startswith("Las etiquetas solo validan")


def test_v5_summary_does_not_count_normal_rows_as_sin_clasificar():
    scored = pd.DataFrame(
        [
            {
                "fecha_hora": pd.Timestamp("2026-10-08 07:00:00"),
                "nivel_anomalia": "BAJO",
                "fase_operativa": "REPOSO",
                "fase_modelo": "REPOSO_ESTATICO",
                "contexto_version": "ENRIQUECIDO",
                "contexto_ciclo": "NO_APLICA",
                "subestado_reposo": "REPOSO_ESTATICO",
                "origen_deteccion": "NORMAL",
                "familia_probable": None,
                "estado_proceso_contextual": "REPOSO_ESTATICO",
                "modo_operacion_contextual": "AUTOMATICO",
                "indice_anomalia": 5.0,
                "advertencias": [],
            },
            {
                "fecha_hora": pd.Timestamp("2026-10-08 07:00:01"),
                "nivel_anomalia": "ALTO",
                "fase_operativa": "OPERACION_ESTABLE",
                "fase_modelo": "OPERACION_ESTABLE",
                "contexto_version": "ENRIQUECIDO",
                "contexto_ciclo": "NORMAL",
                "subestado_reposo": "NO_APLICA",
                "origen_deteccion": "ML_MULTIVARIABLE",
                "familia_probable": None,
                "estado_proceso_contextual": "LLENADO_INFERIDO_POR_NIVEL",
                "modo_operacion_contextual": "AUTOMATICO",
                "indice_anomalia": 80.0,
                "advertencias": [],
            },
        ]
    )
    summary = build_summary(scored)
    assert summary["conteo_por_familia_probable"] == {"SIN_CLASIFICAR": 1}
