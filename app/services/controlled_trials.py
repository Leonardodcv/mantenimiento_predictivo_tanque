from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.config import settings


CONTROLLED_TRIALS_VERSION = "v5.0.1-minute-ground-truth-single-pressure-channel"


@lru_cache(maxsize=1)
def load_controlled_trials_catalog() -> dict[str, Any]:
    path: Path = settings.controlled_trials_file_path
    if not path.exists():
        return {
            "catalog_version": CONTROLLED_TRIALS_VERSION,
            "time_basis": "SQL_LOCAL_NAIVE",
            "principles": {},
            "sessions": [],
            "annotations": [],
            "warning": f"No existe el catalogo configurado: {path}",
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "catalog_version": CONTROLLED_TRIALS_VERSION,
            "time_basis": "SQL_LOCAL_NAIVE",
            "principles": {},
            "sessions": [],
            "annotations": [],
            "warning": f"No se pudo leer el catalogo: {exc}",
        }
    if not isinstance(payload, dict):
        raise ValueError("El catalogo de pruebas controladas debe ser un objeto JSON.")
    payload.setdefault("catalog_version", CONTROLLED_TRIALS_VERSION)
    payload.setdefault("sessions", [])
    payload.setdefault("annotations", [])
    return payload


def clear_controlled_trials_cache() -> None:
    load_controlled_trials_catalog.cache_clear()


def _timestamp(value: Any) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
        return None if pd.isna(ts) else ts
    except (TypeError, ValueError):
        return None


def controlled_session_mask(df: pd.DataFrame) -> pd.Series:
    if df.empty or "fecha_hora" not in df.columns or not settings.controlled_trials_enabled:
        return pd.Series(False, index=df.index, dtype=bool)
    times = pd.to_datetime(df["fecha_hora"], errors="coerce")
    mask = pd.Series(False, index=df.index, dtype=bool)
    catalog = load_controlled_trials_catalog()
    for session in catalog.get("sessions", []):
        start = _timestamp(session.get("start"))
        end = _timestamp(session.get("end"))
        if start is None or end is None:
            continue
        if not session.get("exclude_from_normal_ml_training", False):
            continue
        mask |= times.between(start, end, inclusive="both")
    return mask.fillna(False)


def annotate_controlled_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Superpone ground truth sin alterar el indice de anomalia.

    La sesion completa se puede excluir de entrenamiento/baseline porque el usuario
    confirmo que las pruebas fueron continuas y no hubo recuperacion normal asumible
    entre anotaciones. Las etiquetas por minuto son solo ventanas de observacion:
    no se interpretan como inicio/fin exactos de la maniobra.
    """
    if df.empty or "fecha_hora" not in df.columns:
        return df.copy()
    out = df.copy()
    times = pd.to_datetime(out["fecha_hora"], errors="coerce")

    out["es_sesion_pruebas_controladas"] = False
    out["sesion_prueba_controlada_id"] = None
    out["excluir_entrenamiento_normal"] = False
    out["excluir_baseline_normal"] = False
    out["ground_truth_ids"] = [[] for _ in range(len(out))]
    out["ground_truth_tipos"] = [[] for _ in range(len(out))]
    out["ground_truth_categorias"] = [[] for _ in range(len(out))]
    out["ground_truth_precision"] = None
    out["ground_truth_falla_real"] = None
    out["ground_truth_manipulacion_sensor"] = False
    out["ground_truth_disponible"] = False

    if not settings.controlled_trials_enabled:
        return out

    catalog = load_controlled_trials_catalog()
    for session in catalog.get("sessions", []):
        start = _timestamp(session.get("start"))
        end = _timestamp(session.get("end"))
        if start is None or end is None:
            continue
        mask = times.between(start, end, inclusive="both")
        if not bool(mask.any()):
            continue
        out.loc[mask, "es_sesion_pruebas_controladas"] = True
        out.loc[mask, "sesion_prueba_controlada_id"] = session.get("session_id")
        if settings.controlled_trials_exclude_session_from_training and session.get(
            "exclude_from_normal_ml_training", False
        ):
            out.loc[mask, "excluir_entrenamiento_normal"] = True
        if settings.controlled_trials_exclude_session_from_baseline and session.get(
            "exclude_from_cycle_baseline", False
        ):
            out.loc[mask, "excluir_baseline_normal"] = True
        out.loc[mask, "ground_truth_falla_real"] = False

    # Se usan listas porque puede haber varias anotaciones en el mismo minuto (09:13).
    ids = [[] for _ in range(len(out))]
    types = [[] for _ in range(len(out))]
    categories = [[] for _ in range(len(out))]
    precisions: list[str | None] = [None for _ in range(len(out))]
    sensor_flags = [False for _ in range(len(out))]

    for annotation in catalog.get("annotations", []):
        start = _timestamp(annotation.get("start"))
        end = _timestamp(annotation.get("end"))
        if start is None or end is None:
            continue
        mask = times.between(start, end, inclusive="both").fillna(False)
        indices = np.flatnonzero(mask.to_numpy())
        for pos in indices:
            ids[pos].append(str(annotation.get("id")))
            types[pos].append(str(annotation.get("type")))
            categories[pos].append(str(annotation.get("category")))
            precisions[pos] = str(annotation.get("precision") or "MINUTO")
            if annotation.get("category") == "CALIDAD_DATO_SENSOR":
                sensor_flags[pos] = True

    out["ground_truth_ids"] = ids
    out["ground_truth_tipos"] = types
    out["ground_truth_categorias"] = categories
    out["ground_truth_precision"] = precisions
    out["ground_truth_manipulacion_sensor"] = sensor_flags
    out["ground_truth_disponible"] = [bool(v) for v in ids]
    return out


def _window_frame(scored: pd.DataFrame, annotation: dict[str, Any]) -> pd.DataFrame:
    if scored.empty or "fecha_hora" not in scored.columns:
        return scored.iloc[0:0].copy()
    start = _timestamp(annotation.get("start"))
    end = _timestamp(annotation.get("end"))
    if start is None or end is None:
        return scored.iloc[0:0].copy()
    times = pd.to_datetime(scored["fecha_hora"], errors="coerce")
    return scored.loc[times.between(start, end, inclusive="both")].copy()


def _safe_metric(frame: pd.DataFrame, column: str, stat: str) -> float | None:
    if frame.empty or column not in frame.columns:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna().astype(float)
    if values.empty:
        return None
    if stat == "min":
        value = values.min()
    elif stat == "max":
        value = values.max()
    elif stat == "median":
        value = values.median()
    elif stat == "p95":
        value = values.quantile(0.95)
    elif stat == "std":
        value = values.std(ddof=0)
    else:
        raise ValueError(stat)
    return round(float(value), 4)


def validate_controlled_trials(
    scored: pd.DataFrame,
    events: list[dict[str, Any]] | None = None,
    detection_threshold: float | None = None,
) -> dict[str, Any]:
    """Evalua el detector contra la bitacora sin usar la etiqueta para puntuar.

    La exactitud temporal esta limitada a minutos. El resultado mide si hubo una
    respuesta del detector dentro de la ventana anotada, no diagnostico causal.
    """
    catalog = load_controlled_trials_catalog()
    threshold = float(
        settings.controlled_trials_detection_threshold
        if detection_threshold is None
        else detection_threshold
    )
    events = list(events or [])
    results: list[dict[str, Any]] = []

    for annotation in catalog.get("annotations", []):
        frame = _window_frame(scored, annotation)
        use_for_validation = bool(annotation.get("use_for_validation", False))
        expected = str(annotation.get("expected_evaluation") or "NO_EVALUAR")
        max_index = _safe_metric(frame, "indice_anomalia", "max")
        p95_index = _safe_metric(frame, "indice_anomalia", "p95")
        max_rule = _safe_metric(frame, "indice_reglas", "max")
        max_ml = _safe_metric(frame, "indice_ml", "max")
        detected = bool(
            max_index is not None
            and (
                max_index >= threshold
                or (max_rule or 0.0) >= float(settings.event_immediate_rule_threshold)
            )
        )

        start = _timestamp(annotation.get("start"))
        end = _timestamp(annotation.get("end"))
        overlapping_events: list[dict[str, Any]] = []
        if start is not None and end is not None:
            for event in events:
                event_start = _timestamp(event.get("inicio"))
                event_end = _timestamp(event.get("fin"))
                if event_start is None or event_end is None:
                    continue
                if event_start <= end and event_end >= start:
                    overlapping_events.append(
                        {
                            "evento": event.get("evento"),
                            "inicio": event.get("inicio"),
                            "fin": event.get("fin"),
                            "indice_maximo": event.get("indice_maximo"),
                            "origen_deteccion": event.get("origen_deteccion"),
                            "familia_probable": event.get("familia_probable"),
                        }
                    )

        if frame.empty:
            status = "SIN_DATOS"
        elif not use_for_validation or expected == "NO_EVALUAR":
            status = "NO_EVALUADA"
        elif expected == "NO_ANOMALIA":
            status = "ALERTA_EN_VENTANA_CONTROL_REVISAR" if detected else "CONTROL_SIN_ALERTA_ALTA"
        elif expected == "ANOMALIA_HIDRAULICA":
            status = "RESPUESTA_DETECTADA_EN_VENTANA" if detected else "SIN_RESPUESTA_EN_VENTANA"
        elif expected == "CALIDAD_DATO":
            status = "CALIDAD_DATO_SEPARADA"
        elif expected in {"CAMBIO_REGIMEN", "TRANSICION_RECUPERACION"}:
            status = "RESPUESTA_DETECTADA" if detected else "SIN_RESPUESTA_ALTA"
        else:
            status = "NO_EVALUADA"

        moving_pct = None
        if not frame.empty and "en_movimiento" in frame.columns:
            moving = frame["en_movimiento"].astype(bool)
            moving_pct = round(float(moving.mean() * 100.0), 2)

        results.append(
            {
                "id": annotation.get("id"),
                "type": annotation.get("type"),
                "category": annotation.get("category"),
                "precision": annotation.get("precision"),
                "start": annotation.get("start"),
                "end": annotation.get("end"),
                "expected_evaluation": expected,
                "use_for_validation": use_for_validation,
                "records": int(len(frame)),
                "detected_anomaly": detected,
                "validation_status": status,
                "indice_maximo": max_index,
                "indice_p95": p95_index,
                "indice_reglas_max": max_rule,
                "indice_ml_max": max_ml,
                "flujo_min": _safe_metric(frame, "flujo_instantaneo", "min"),
                "flujo_max": _safe_metric(frame, "flujo_instantaneo", "max"),
                "flujo_mediana": _safe_metric(frame, "flujo_instantaneo", "median"),
                "flujo_std": _safe_metric(frame, "flujo_instantaneo", "std"),
                "presion_fuente_sensor": "SENSOR_SUPERIOR",
                "presion_fuente_columna": "presion_relativa",
                "presion_sensor_bomba_disponible": False,
                "presion_min": _safe_metric(frame, "presion_relativa", "min"),
                "presion_max": _safe_metric(frame, "presion_relativa", "max"),
                "presion_mediana": _safe_metric(frame, "presion_relativa", "median"),
                "presion_std": _safe_metric(frame, "presion_relativa", "std"),
                "nivel_min": _safe_metric(frame, "nivel_tanque", "min"),
                "nivel_max": _safe_metric(frame, "nivel_tanque", "max"),
                "movimiento_pct": moving_pct,
                "eventos_solapados": overlapping_events,
                "description": annotation.get("description"),
                "note": annotation.get("note"),
            }
        )

    evaluated = [r for r in results if r["use_for_validation"] and r["validation_status"] != "SIN_DATOS"]
    hydraulic_expected = [r for r in evaluated if r["expected_evaluation"] == "ANOMALIA_HIDRAULICA"]
    normal_expected = [r for r in evaluated if r["expected_evaluation"] == "NO_ANOMALIA"]
    detected_hydraulic = sum(r["validation_status"] == "RESPUESTA_DETECTADA_EN_VENTANA" for r in hydraulic_expected)
    control_alert_windows = sum(r["validation_status"] == "ALERTA_EN_VENTANA_CONTROL_REVISAR" for r in normal_expected)

    return {
        "catalog_version": catalog.get("catalog_version"),
        "detection_threshold": threshold,
        "time_precision_warning": (
            "Las horas fueron registradas al minuto; cada ventana es aproximada y no representa "
            "la duracion exacta de la maniobra."
        ),
        "ground_truth_policy": (
            "Las etiquetas solo validan el detector y excluyen la sesion del aprendizaje normal; "
            "no modifican el indice de anomalia ni convierten una prueba en falla real."
        ),
        "pressure_validation_scope": {
            "active_sensor": "SENSOR_SUPERIOR",
            "database_column": "presion_relativa",
            "location": "entre_valvula_azul_superior_y_tanque",
            "pump_sensor_available": False,
            "pump_sensor_reason": "PLC_SECUNDARIO_NO_TRANSMITE_DATOS_ACTUALMENTE",
            "note": (
                "Las observaciones humanas que mencionan ambos sensores se conservan, pero la validacion "
                "numerica actual solo puede comprobar el sensor superior."
            ),
        },
        "summary": {
            "annotations_total": len(results),
            "annotations_with_data": sum(r["records"] > 0 for r in results),
            "hydraulic_expected": len(hydraulic_expected),
            "hydraulic_detected": detected_hydraulic,
            "hydraulic_detection_rate_pct": (
                None
                if not hydraulic_expected
                else round(100.0 * detected_hydraulic / len(hydraulic_expected), 2)
            ),
            "normal_controls": len(normal_expected),
            "normal_control_alert_windows_to_review": control_alert_windows,
        },
        "results": results,
    }


def controlled_trials_context_summary(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty or "es_sesion_pruebas_controladas" not in scored.columns:
        return {
            "records_in_controlled_sessions": 0,
            "records_with_ground_truth": 0,
            "records_sensor_manipulation": 0,
        }
    return {
        "records_in_controlled_sessions": int(scored["es_sesion_pruebas_controladas"].astype(bool).sum()),
        "records_with_ground_truth": int(scored["ground_truth_disponible"].astype(bool).sum()),
        "records_sensor_manipulation": int(scored["ground_truth_manipulacion_sensor"].astype(bool).sum()),
    }
