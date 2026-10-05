from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import Any

import pandas as pd

from app.config import settings
from app.services.anomaly_engine import (
    AnomalyEngine,
    build_cycles,
    build_events,
    build_summary,
    variable_coverage,
)
from app.services.data_source import load_full, load_latest


@dataclass
class ModelSnapshot:
    source: str
    built_at: datetime
    engine: AnomalyEngine
    raw_full: pd.DataFrame
    scored_full: pd.DataFrame
    summary: dict[str, Any]
    cycles: list[dict[str, Any]]
    coverage: list[dict[str, Any]]


class ModelManager:
    """Mantiene un modelo en memoria y lo reconstruye de forma explicita."""

    def __init__(self):
        self._lock = RLock()
        self._snapshots: dict[str, ModelSnapshot] = {}

    def rebuild(self, source: str) -> ModelSnapshot:
        source = source.strip().lower()
        with self._lock:
            raw = load_full(source)
            engine = AnomalyEngine().fit(raw)
            scored = engine.score(raw)
            snapshot = ModelSnapshot(
                source=source,
                built_at=datetime.now(timezone.utc),
                engine=engine,
                raw_full=raw,
                scored_full=scored,
                summary=build_summary(scored),
                cycles=build_cycles(scored),
                coverage=variable_coverage(raw),
            )
            self._snapshots[source] = snapshot
            return snapshot

    def get(self, source: str, rebuild_if_missing: bool = True) -> ModelSnapshot:
        source = source.strip().lower()
        with self._lock:
            snapshot = self._snapshots.get(source)
        if snapshot is None and rebuild_if_missing:
            return self.rebuild(source)
        if snapshot is None:
            raise RuntimeError("El modelo aun no ha sido construido.")
        return snapshot

    def status(self, source: str) -> dict[str, Any]:
        source = source.strip().lower()
        with self._lock:
            snapshot = self._snapshots.get(source)
        if snapshot is None:
            return {"source": source, "ready": False}
        return {
            "source": source,
            "ready": True,
            "built_at": snapshot.built_at.isoformat(),
            "training_summary": snapshot.engine.training_summary,
            "summary": snapshot.summary,
        }

    def score_latest(self, source: str, limit: int) -> pd.DataFrame:
        snapshot = self.get(source)
        if source == "file":
            return snapshot.scored_full.tail(limit).reset_index(drop=True)

        context = max(settings.latest_context_rows, limit)
        raw = load_latest(source, limit=limit, context_rows=context)
        scored = snapshot.engine.score(raw)
        return scored.tail(limit).reset_index(drop=True)

    def events_latest(self, source: str, threshold: float, limit_rows: int = 5000):
        snapshot = self.get(source)
        if source == "file":
            scored = snapshot.scored_full
        else:
            raw = load_latest(source, limit=limit_rows, context_rows=settings.latest_context_rows)
            scored = snapshot.engine.score(raw)
        return build_events(scored, threshold=threshold)

    def cycles_latest(self, source: str, limit: int):
        snapshot = self.get(source)
        if source == "file":
            cycles = snapshot.cycles
        else:
            raw = load_latest(
                source,
                limit=max(settings.latest_context_rows, limit * 250),
                context_rows=settings.latest_context_rows,
            )
            scored = snapshot.engine.score(raw)
            cycles = build_cycles(scored)
        return cycles[-limit:]


model_manager = ModelManager()
