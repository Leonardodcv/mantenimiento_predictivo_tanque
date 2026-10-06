from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.anomaly_engine_v31 import MODEL_VERSION, dataframe_records
from app.services.data_source import DataSourceError, sqlserver_metadata
from app.services.model_manager import model_manager


app = FastAPI(
    title=settings.app_name,
    version="3.1.0",
    description=(
        "Backend v3.1 para mantenimiento predictivo del tanque. Agrega explicabilidad "
        "heuristica por regimen, clasificacion de la evidencia y separa eventos transitorios "
        "del estado integral de cada ciclo. Conserva soporte de historial LEGACY con NULL."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_VERSION = "3.1.0"
SOURCE_PATTERN = "^(file|sqlserver)$"
EVENT_SCOPE_PATTERN = "^(recent|full)$"


def _cols_for_frontend(df: pd.DataFrame) -> list[str]:
    preferred = [
        "id",
        "fecha_hora",
        "indice_anomalia",
        "nivel_anomalia",
        "indice_ml_crudo",
        "indice_ml",
        "indice_reglas",
        "indice_advertencias",
        "fase_operativa",
        "fase_modelo",
        "subestado_reposo",
        "en_movimiento",
        "ciclo_id",
        "contexto_ciclo",
        "horas_paro_previas_al_ciclo",
        "numero_ciclo_estabilizacion",
        "segundos_desde_arranque",
        "segundos_desde_paro",
        "pendiente_nivel_por_seg",
        "duracion_presion_alta_seg",
        "contexto_version",
        "contexto_nuevas_variables_disponibles",
        "confianza_evaluacion",
        "razones",
        "advertencias",
        "observaciones",
        "origen_deteccion",
        "clasificacion_operativa",
        "familia_probable",
        "impacto_hidraulico",
        "impacto_mecanico",
        "falla_confirmada",
        "metodo_explicacion_ml",
        "explicacion_ml_es_causal",
        "variables_mas_atipicas",
        "flujo_instantaneo",
        "presion_relativa",
        "temperatura_tanque",
        "nivel_tanque",
        "frecuencia_salida",
        "frecuencia_vfd_hz",
        "velocidad",
        "corriente",
        "torque",
        "pwr_actual",
        "kwh_total",
        "voltaje_bus_dc",
        "referencia",
        "potencia_nominal",
        "voltaje_salida",
        "tension_l1_n",
        "corriente_l1",
        "potencia_reactiva_l1",
        "energia_aparente_l1",
        "potencia_activa_l1",
        "potencia_aparente_l1",
        "presion_alta",
        "falta_presion",
        "bajo_flujo",
        "falla_vfd",
        "estado_vfd",
        "manual_auto_vfd",
        "control",
        "start_stop_vfd",
        "manual_auto_s1",
        "start_stop_s1",
        "estado_s1",
        "manual_auto_s2",
        "start_stop_s2",
        "estado_s2",
        "recirculacion_automatica",
        "setpoint_llenado",
        "arranque_paro_llenado",
        "setpoint_vaciado",
        "arranque_paro_vaciado",
        "encendido_ia",
        "encendido_local",
        "horas_marcha",
        "numero_arranques",
        "en_mantenimiento",
    ]
    return [c for c in preferred if c in df.columns]


def _snapshot(source: str):
    try:
        return model_manager.get(source)
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/v3/health/")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "default_source": settings.data_source,
        "supports_legacy_null_context": True,
        "supports_restart_context": True,
        "supports_rest_substates": True,
        "uses_engineering_tolerances": True,
        "supports_ml_explainability": True,
        "separates_transient_events_from_cycle_health": True,
        "warning": "El indice es anomalia/rareza, no probabilidad de falla.",
    }


@app.get("/api/v3/model/status/")
def model_status(source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN)):
    return model_manager.status(source)


@app.post("/api/v3/model/rebuild/")
def model_rebuild(source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN)):
    """Relee todo el historial y reconstruye modelos, ciclos y baseline v3.1."""
    try:
        snap = model_manager.rebuild(source)
        response = {
            "status": "ok",
            "source": source,
            "built_at": snap.built_at.isoformat(),
            "model_version": MODEL_VERSION,
            "training_summary": snap.engine.training_summary,
            "summary": snap.summary,
            "cycles_detected": len(snap.cycles),
            "cycle_baseline": snap.cycle_baseline,
        }
        if source == "sqlserver":
            response["database"] = sqlserver_metadata()
        return response
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v3/anomalies/summary/")
def anomaly_summary(source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN)):
    snap = _snapshot(source)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "built_at": snap.built_at.isoformat(),
        "warning": "El indice es anomalia/rareza, no probabilidad de falla.",
        "summary": snap.summary,
        "training_summary": snap.engine.training_summary,
        "cycle_baseline": snap.cycle_baseline,
    }


@app.get("/api/v3/anomalies/latest/")
def anomaly_latest(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    limit: int = Query(default=settings.default_limit, ge=1, le=10000),
    min_index: float = Query(default=0.0, ge=0.0, le=100.0),
):
    try:
        scored = model_manager.score_latest(source, limit=max(limit, 1))
        filtered = scored.loc[scored["indice_anomalia"] >= min_index].tail(limit)
        cols = _cols_for_frontend(filtered)
        return {
            "model_version": MODEL_VERSION,
            "source": source,
            "count": int(len(filtered)),
            "data": dataframe_records(filtered[cols]),
        }
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v3/anomalies/history/")
def anomaly_history(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    desde: datetime | None = Query(default=None),
    hasta: datetime | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=50000),
    min_index: float = Query(default=0.0, ge=0.0, le=100.0),
):
    snap = _snapshot(source)
    data = snap.scored_full
    if desde is not None:
        data = data.loc[data["fecha_hora"] >= pd.Timestamp(desde)]
    if hasta is not None:
        data = data.loc[data["fecha_hora"] <= pd.Timestamp(hasta)]
    data = data.loc[data["indice_anomalia"] >= min_index].tail(limit)
    cols = _cols_for_frontend(data)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "count": int(len(data)),
        "data": dataframe_records(data[cols]),
    }


@app.get("/api/v3/anomalies/explain/{record_id}")
def anomaly_explain(
    record_id: int,
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
):
    snap = _snapshot(source)
    if "id" not in snap.scored_full.columns:
        raise HTTPException(status_code=404, detail="La fuente no contiene identificadores de registro.")
    matches = snap.scored_full.loc[pd.to_numeric(snap.scored_full["id"], errors="coerce").eq(record_id)]
    if matches.empty:
        raise HTTPException(status_code=404, detail=f"No se encontro el registro {record_id} en el snapshot.")
    row = matches.tail(1)
    cols = _cols_for_frontend(row)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "snapshot_built_at": snap.built_at.isoformat(),
        "record": dataframe_records(row[cols])[0],
        "explanation_note": (
            "variables_mas_atipicas es una explicacion heuristica contra el baseline robusto "
            "del regimen; no es una atribucion causal exacta del Isolation Forest."
        ),
    }


@app.get("/api/v3/anomalies/events/")
def anomaly_events(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    threshold: float = Query(default=settings.event_open_threshold, ge=0.0, le=100.0),
    scope: str = Query(default="recent", pattern=EVENT_SCOPE_PATTERN),
    recent_rows: int = Query(default=5000, ge=100, le=50000),
):
    try:
        events = model_manager.events(source, threshold=threshold, scope=scope, limit_rows=recent_rows)
        response = {
            "model_version": MODEL_VERSION,
            "source": source,
            "scope": scope,
            "threshold": threshold,
            "event_policy": {
                "min_consecutive": settings.event_min_consecutive,
                "min_duration_seconds": settings.event_min_duration_seconds,
                "close_threshold": settings.event_close_threshold,
                "close_seconds": settings.event_close_seconds,
                "immediate_rule_threshold": settings.event_immediate_rule_threshold,
                "statistical_active_only": settings.event_statistical_active_only,
            },
            "count": len(events),
            "events": events,
        }
        if scope == "recent":
            response["recent_rows"] = recent_rows
            response["scope_note"] = (
                "Ventana reciente. Los estados pasivos de reposo no abren eventos "
                "estadisticos por si solos; las reglas fisicas fuertes siguen activas."
            )
        else:
            snap = _snapshot(source)
            response["snapshot_built_at"] = snap.built_at.isoformat()
            response["scope_note"] = "Historial completo contenido en el ultimo model/rebuild."
        return response
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v3/cycles/latest/")
def cycles_latest(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    limit: int = Query(default=settings.cycle_default_limit, ge=1, le=200),
):
    try:
        snap = _snapshot(source)
        cycles = model_manager.cycles_latest(source, limit=limit)
        return {
            "model_version": MODEL_VERSION,
            "source": source,
            "scope": "snapshot_full_history",
            "snapshot_built_at": snap.built_at.isoformat(),
            "cycles_detected_snapshot": len(snap.cycles),
            "count": len(cycles),
            "cycles": cycles,
        }
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v3/cycles/baseline/")
def cycles_baseline(source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN)):
    snap = _snapshot(source)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "snapshot_built_at": snap.built_at.isoformat(),
        "baseline": snap.cycle_baseline,
    }


@app.get("/api/v3/variables/status/")
def variables_status(source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN)):
    snap = _snapshot(source)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "total_registros": int(len(snap.raw_full)),
        "variables": snap.coverage,
    }
