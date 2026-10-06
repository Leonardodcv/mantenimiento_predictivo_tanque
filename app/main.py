from __future__ import annotations

from datetime import datetime

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.services.anomaly_engine import MODEL_VERSION, dataframe_records
from app.services.data_source import DataSourceError, sqlserver_metadata
from app.services.model_manager import model_manager


app = FastAPI(
    title=settings.app_name,
    version="2.2.1",
    description=(
        "Backend v2.2.1 para indice de anomalia 0-100 del tanque. Soporta historial "
        "mixto: registros antiguos con NULL en variables nuevas y registros v3 enriquecidos."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allowed_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

API_VERSION = "2.2.1"
SOURCE_PATTERN = "^(file|sqlserver)$"
EVENT_SCOPE_PATTERN = "^(recent|full)$"


def _cols_for_frontend(df: pd.DataFrame) -> list[str]:
    preferred = [
        "id",
        "fecha_hora",
        "indice_anomalia",
        "nivel_anomalia",
        "indice_ml",
        "indice_reglas",
        "indice_advertencias",
        "fase_operativa",
        "en_movimiento",
        "ciclo_id",
        "segundos_desde_arranque",
        "segundos_desde_paro",
        "duracion_presion_alta_seg",
        "contexto_version",
        "contexto_nuevas_variables_disponibles",
        "confianza_evaluacion",
        "razones",
        "advertencias",
        "observaciones",
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


@app.get("/api/v2/health/")
def health():
    return {
        "status": "ok",
        "service": settings.app_name,
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
        "default_source": settings.data_source,
        "supports_legacy_null_context": True,
        "warning": "El indice es anomalia/rareza, no probabilidad de falla.",
    }


@app.get("/api/v2/model/status/")
def model_status(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
):
    return model_manager.status(source)


@app.post("/api/v2/model/rebuild/")
def model_rebuild(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
):
    """Relee TODO el historial y reconstruye la linea base/modelos v2."""
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
        }
        if source == "sqlserver":
            response["database"] = sqlserver_metadata()
        return response
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v2/anomalies/summary/")
def anomaly_summary(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
):
    snap = _snapshot(source)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "built_at": snap.built_at.isoformat(),
        "warning": "El indice es anomalia/rareza, no probabilidad de falla.",
        "summary": snap.summary,
        "training_summary": snap.engine.training_summary,
    }


@app.get("/api/v2/anomalies/latest/")
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


@app.get("/api/v2/anomalies/history/")
def anomaly_history(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    desde: datetime | None = Query(default=None),
    hasta: datetime | None = Query(default=None),
    limit: int = Query(default=5000, ge=1, le=50000),
    min_index: float = Query(default=0.0, ge=0.0, le=100.0),
):
    """Consulta el historial completo cargado durante el ultimo rebuild."""
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


@app.get("/api/v2/anomalies/events/")
def anomaly_events(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    threshold: float = Query(default=settings.event_open_threshold, ge=0.0, le=100.0),
    scope: str = Query(default="recent", pattern=EVENT_SCOPE_PATTERN),
    recent_rows: int = Query(default=5000, ge=100, le=50000),
):
    """
    Agrupa eventos persistentes.

    - scope=recent: evalua una ventana reciente de SQL; puede devolver 0 si la maquina
      estuvo detenida durante ese intervalo.
    - scope=full: evalua el snapshot historico completo generado por el ultimo rebuild.
    """
    try:
        events = model_manager.events(
            source,
            threshold=threshold,
            scope=scope,
            limit_rows=recent_rows,
        )
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
            },
            "count": len(events),
            "events": events,
        }
        if scope == "recent":
            response["recent_rows"] = recent_rows
            response["scope_note"] = (
                "Ventana reciente. Un count=0 es esperado si no hubo operacion/anomalias "
                "persistentes durante el intervalo reciente."
            )
        else:
            snap = _snapshot(source)
            response["snapshot_built_at"] = snap.built_at.isoformat()
            response["scope_note"] = (
                "Historial completo contenido en el snapshot del ultimo model/rebuild."
            )
        return response
    except DataSourceError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v2/cycles/latest/")
def cycles_latest(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
    limit: int = Query(default=settings.cycle_default_limit, ge=1, le=200),
):
    """
    Devuelve los ultimos ciclos DETECTADOS en el snapshot del ultimo rebuild.

    No depende de que la maquina haya operado en las ultimas horas. Si el equipo estuvo
    detenido durante la noche, se siguen devolviendo los ultimos ciclos existentes.
    """
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


@app.get("/api/v2/variables/status/")
def variables_status(
    source: str = Query(default=settings.data_source, pattern=SOURCE_PATTERN),
):
    snap = _snapshot(source)
    return {
        "model_version": MODEL_VERSION,
        "source": source,
        "total_registros": int(len(snap.raw_full)),
        "variables": snap.coverage,
    }
