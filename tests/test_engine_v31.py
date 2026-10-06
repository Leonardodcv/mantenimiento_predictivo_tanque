import numpy as np
import pandas as pd

from app.services.anomaly_engine_v31 import AnomalyEngine, build_cycles, build_events


def _active_training_frame(n=160):
    start = pd.Timestamp("2026-10-06 10:00:00")
    rows = []
    for i in range(n):
        jitter = ((i % 7) - 3) / 10.0
        rows.append(
            {
                "id": i + 1,
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "flujo_instantaneo": 18.28 + jitter * 0.02,
                "presion_relativa": 8.60 + jitter * 0.01,
                "nivel_tanque": 50 + i * 0.4,
                "frecuencia_salida": 40,
                "velocidad": 1198 + (i % 3 - 1),
                "voltaje_salida": 155,
                "voltaje_bus_dc": 293 + (i % 5 - 2),
                "tension_l1_n": 264 + (i % 5 - 2) * 0.8,
                "corriente_l1": 0.146 + jitter * 0.001,
                "potencia_activa_l1": 26.2 + jitter * 0.1,
                "potencia_aparente_l1": 38.8 + jitter * 0.15,
                "falla_vfd": False,
                "falta_presion": False,
                "bajo_flujo": False,
                "presion_alta": True,
                "estado_vfd": 2,
            }
        )
    return pd.DataFrame(rows)


def test_v31_explanation_identifies_electrical_excursion_like_oct6_event():
    engine = AnomalyEngine().fit(_active_training_frame())
    row = engine._prepare(
        pd.DataFrame(
            [
                {
                    "fecha_hora": pd.Timestamp("2026-10-06 11:08:38"),
                    "flujo_instantaneo": 18.27,
                    "presion_relativa": 8.72,
                    "nivel_tanque": 70.27,
                    "frecuencia_salida": 40,
                    "velocidad": 1276,
                    "voltaje_salida": 164,
                    "voltaje_bus_dc": 333,
                    "tension_l1_n": 298.0,
                    "corriente_l1": 0.155,
                    "potencia_activa_l1": 29.67,
                    "potencia_aparente_l1": 46.42,
                    "presion_alta": True,
                    "falta_presion": False,
                    "bajo_flujo": False,
                    "falla_vfd": False,
                    "estado_vfd": 2,
                }
            ]
        )
    ).iloc[0]
    variables = engine._feature_explanations(row, "OPERACION_ESTABLE", 90.0)
    names = {item["variable"] for item in variables}
    assert "voltaje_bus_dc" in names
    assert "tension_l1_n" in names
    assert "potencia_aparente_l1" in names
    diagnostic = engine._row_diagnostic(
        ml_index=90.0,
        rule_index=0.0,
        warning_index=35.0,
        reasons=["patron_multivariable_atipico_para_el_regimen"],
        variables=variables,
    )
    assert diagnostic["origen_deteccion"] == "ML_MULTIVARIABLE"
    assert diagnostic["familia_probable"] in {
        "TRANSITORIO_ELECTRICO_VFD",
        "TRANSITORIO_ELECTRICO_VFD_CON_RESPUESTA_FISICA",
    }
    assert diagnostic["falla_confirmada"] is False
    assert diagnostic["explicacion_ml_es_causal"] is False


def _cycle_rows(cycle_id, start, scores, *, flow=18.25, pressure=8.63, level_start=39.75, level_end=82.5):
    rows = []
    n = len(scores)
    for i, score in enumerate(scores):
        frac = i / max(1, n - 1)
        rows.append(
            {
                "ciclo_id": float(cycle_id),
                "fecha_hora": start + pd.Timedelta(seconds=i),
                "en_movimiento": True,
                "fase_operativa": "ARRANQUE" if i < 4 else "OPERACION_ESTABLE",
                "nivel_tanque": level_start + (level_end - level_start) * frac,
                "flujo_instantaneo": flow,
                "presion_relativa": pressure,
                "velocidad": 1198,
                "indice_anomalia": float(score),
                "indice_ml_crudo": float(score),
                "indice_reglas": 0.0,
                "duracion_presion_alta_seg": float(i),
                "contexto_version": "ENRIQUECIDO",
                "contexto_ciclo": "NORMAL",
                "horas_paro_previas_al_ciclo": 0.04,
                "numero_ciclo_estabilizacion": None,
                "razones": ["patron_multivariable_atipico_para_el_regimen"] if score >= 70 else [],
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


def test_v31_short_ml_event_does_not_make_whole_cycle_high():
    base = pd.Timestamp("2026-10-06 08:00:00")
    rows = []
    # Baseline suficiente de ciclos fisicamente iguales y sanos.
    for cid in range(1, 16):
        rows += _cycle_rows(cid, base + pd.Timedelta(minutes=4 * cid), [18.0] * 99)
    # 13 s altos dentro de ~99 s: se conserva como evento transitorio, no salud global ALTA.
    target_scores = [18.0] * 68 + [90.0] * 13 + [18.0] * 18
    rows += _cycle_rows(16, base + pd.Timedelta(hours=2), target_scores)
    cycle = build_cycles(pd.DataFrame(rows))[-1]
    assert cycle["indice_p95"] >= 80
    assert cycle["ml_persistente_en_ciclo"] is False
    assert cycle["tuvo_evento_transitorio_ml"] is True
    assert cycle["indice_ml_ciclo_efectivo"] <= 39
    assert cycle["nivel_ciclo"] == "BAJO"
    assert "evento_multivariable_transitorio_sin_desviacion_integral_del_ciclo" in cycle["observaciones"]


def test_v31_persistent_ml_deviation_can_raise_cycle_health():
    base = pd.Timestamp("2026-10-06 08:00:00")
    rows = []
    for cid in range(1, 16):
        rows += _cycle_rows(cid, base + pd.Timedelta(minutes=4 * cid), [18.0] * 99)
    target_scores = [90.0] * 40 + [18.0] * 59
    rows += _cycle_rows(16, base + pd.Timedelta(hours=2), target_scores)
    cycle = build_cycles(pd.DataFrame(rows))[-1]
    assert cycle["ml_persistente_en_ciclo"] is True
    assert cycle["fraccion_muestras_ml_altas_pct"] >= 25
    assert cycle["indice_ciclo"] >= 70
    assert cycle["nivel_ciclo"] in {"ALTO", "MUY_ALTO"}


def test_v31_event_exposes_detection_origin_and_probable_family():
    start = pd.Timestamp("2026-10-06 11:08:37")
    atypical = [
        {
            "variable": "voltaje_bus_dc",
            "grupo": "ELECTRICA_VFD",
            "score_explicacion": 95.0,
            "z_robusto_aprox": 8.0,
            "desviacion_pct": 13.0,
            "valor": 333,
            "mediana_regimen": 293,
            "direccion": "ALTA",
        },
        {
            "variable": "tension_l1_n",
            "grupo": "ELECTRICA_VFD",
            "score_explicacion": 93.0,
            "z_robusto_aprox": 7.5,
            "desviacion_pct": 12.0,
            "valor": 298,
            "mediana_regimen": 264,
            "direccion": "ALTA",
        },
        {
            "variable": "velocidad",
            "grupo": "MECANICA",
            "score_explicacion": 80.0,
            "z_robusto_aprox": 5.0,
            "desviacion_pct": 6.0,
            "valor": 1276,
            "mediana_regimen": 1198,
            "direccion": "ALTA",
        },
    ]
    df = pd.DataFrame(
        {
            "fecha_hora": [start + pd.Timedelta(seconds=i) for i in range(6)],
            "indice_anomalia": [88, 91, 90, 89, 88, 87],
            "indice_ml": [88, 91, 90, 89, 88, 87],
            "indice_reglas": [0.0] * 6,
            "fase_operativa": ["OPERACION_ESTABLE"] * 6,
            "fase_modelo": ["OPERACION_ESTABLE"] * 6,
            "contexto_ciclo": ["NORMAL"] * 6,
            "razones": [["patron_multivariable_atipico_para_el_regimen"] for _ in range(6)],
            "advertencias": [[] for _ in range(6)],
            "observaciones": [[] for _ in range(6)],
            "variables_mas_atipicas": [atypical for _ in range(6)],
        }
    )
    event = build_events(df, threshold=80)[0]
    assert event["origen_deteccion"] == "ML_MULTIVARIABLE"
    assert event["clasificacion_operativa"] == "DESVIACION_TRANSITORIA_MULTIVARIABLE"
    assert event["familia_probable"] == "TRANSITORIO_ELECTRICO_VFD_CON_RESPUESTA_FISICA"
    assert event["impacto_mecanico"] is True
    assert event["falla_confirmada"] is False
    assert len(event["variables_mas_atipicas"]) >= 2
