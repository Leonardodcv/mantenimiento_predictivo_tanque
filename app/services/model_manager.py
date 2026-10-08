from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any

import pandas as pd

from app.config import settings
from app.services.anomaly_engine_v5 import (
    AnomalyEngine,
    MODEL_VERSION,
    build_cycles,
    build_events,
    build_summary,
    create_cycle_baseline_reference,
    cycle_baseline_summary,
    variable_coverage,
)
from app.services.data_source import load_full, load_latest, load_time_window
from app.services.controlled_trials import validate_controlled_trials


@dataclass
class ModelSnapshot:
    source: str
    built_at: datetime
    engine: AnomalyEngine
    raw_full: pd.DataFrame
    scored_full: pd.DataFrame
    summary: dict[str, Any]
    cycles: list[dict[str, Any]]
    cycle_baseline: dict[str, Any] = field(default_factory=dict)
    cycle_baseline_reference: dict[str, Any] = field(default_factory=dict)
    coverage: list[dict[str, Any]] = field(default_factory=list)


class ModelManager:
    """Snapshot v5: v4 protegido + exclusion/validacion de pruebas controladas."""

    def __init__(self):
        self._lock = RLock()
        self._snapshots: dict[str, ModelSnapshot] = {}

    @staticmethod
    def _baseline_file(source: str) -> Path:
        safe_source = "".join(ch for ch in source.lower() if ch.isalnum() or ch in {"-", "_"})
        return settings.cycle_baseline_dir_path / f"cycle_baseline_{safe_source}_v50.json"

    @staticmethod
    def _legacy_v40_baseline_file(source: str) -> Path:
        safe_source = "".join(ch for ch in source.lower() if ch.isalnum() or ch in {"-", "_"})
        return settings.cycle_baseline_dir_path / f"cycle_baseline_{safe_source}_v40.json"

    @staticmethod
    def _legacy_v32_baseline_file(source: str) -> Path:
        safe_source = "".join(ch for ch in source.lower() if ch.isalnum() or ch in {"-", "_"})
        return settings.cycle_baseline_dir_path / f"cycle_baseline_{safe_source}_v32.json"

    def _load_cycle_baseline(self, source: str) -> dict[str, Any] | None:
        path = self._baseline_file(source)
        migrated_from: Path | None = None
        if not path.exists():
            for legacy in [self._legacy_v40_baseline_file(source), self._legacy_v32_baseline_file(source)]:
                if legacy.exists():
                    migrated_from = legacy
                    path = legacy
                    break
            else:
                return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(data, dict) or not data.get("global"):
            return None
        if migrated_from is not None:
            # v5 conserva la referencia protegida de v4/v3.2 en lugar de
            # aprender de nuevo con la sesion de pruebas controladas. Se guarda una copia v50.
            migrated = dict(data)
            migrated["migrated_from_file"] = migrated_from.name
            migrated["model_version_migrated_to"] = MODEL_VERSION
            data = self._save_cycle_baseline(source, migrated)
        return data

    def _save_cycle_baseline(self, source: str, reference: dict[str, Any]) -> dict[str, Any]:
        path = self._baseline_file(source)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(reference)
        payload["created_at"] = payload.get("created_at") or datetime.now(timezone.utc).isoformat()
        payload["source"] = source
        payload["file_name"] = path.name
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return payload

    def _delete_cycle_baseline(self, source: str) -> None:
        # reset explicito: elimina referencias v5/v4/v3.2 para evitar que la
        # migracion automatica restaure inmediatamente un baseline antiguo.
        for path in [
            self._baseline_file(source),
            self._legacy_v40_baseline_file(source),
            self._legacy_v32_baseline_file(source),
        ]:
            if path.exists():
                path.unlink()

    def rebuild(self, source: str, reset_cycle_baseline: bool = False) -> ModelSnapshot:
        source = source.strip().lower()
        with self._lock:
            if reset_cycle_baseline:
                self._delete_cycle_baseline(source)
            baseline_reference = None
            if settings.cycle_baseline_mode == "protected_frozen":
                baseline_reference = self._load_cycle_baseline(source)

            raw = load_full(source)

            # Primera pasada: necesaria para reconstruir ciclos y aplicar el baseline protegido.
            # Esta pasada NO es el modelo final cuando ML_PROTECTED_ACTIVE_TRAINING=true.
            first_engine = AnomalyEngine().fit(raw)
            first_scored = first_engine.score(raw)

            if baseline_reference is None:
                # Primer rebuild v5.0 sin baseline previo: bootstrap robusto y congelacion. Las futuras
                # reconstrucciones no incorporan automaticamente ciclos en cuarentena.
                bootstrap_cycles = build_cycles(first_scored)
                baseline_reference = create_cycle_baseline_reference(bootstrap_cycles)
                baseline_reference = self._save_cycle_baseline(source, baseline_reference)

            first_cycles = build_cycles(first_scored, baseline_reference=baseline_reference)
            trusted_cycle_ids = {
                int(c["ciclo_id"])
                for c in first_cycles
                if c.get("estado_baseline") == "CONFIABLE_BASELINE"
            }

            if settings.ml_protected_active_training and trusted_cycle_ids:
                # Segunda pasada v5.0: ARRANQUE/OPERACION_ESTABLE/DESACELERACION solo
                # aprenden de ciclos confiables. Esto impide que NR/EP persistentes
                # terminen normalizandose por repetidos rebuilds.
                engine = AnomalyEngine().fit(raw, trusted_cycle_ids=trusted_cycle_ids)
                scored = engine.score(raw)
                cycles = build_cycles(scored, baseline_reference=baseline_reference)
            else:
                engine = first_engine
                scored = first_scored
                cycles = first_cycles

            snapshot = ModelSnapshot(
                source=source,
                built_at=datetime.now(timezone.utc),
                engine=engine,
                raw_full=raw,
                scored_full=scored,
                summary=build_summary(scored),
                cycles=cycles,
                cycle_baseline=cycle_baseline_summary(cycles, baseline_reference),
                cycle_baseline_reference=baseline_reference,
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
            baseline = self._load_cycle_baseline(source)
            return {
                "source": source,
                "ready": False,
                "model_version": MODEL_VERSION,
                "cycle_baseline_frozen_available": baseline is not None,
                "cycle_baseline_created_at": None if baseline is None else baseline.get("created_at"),
            }
        return {
            "source": source,
            "ready": True,
            "model_version": MODEL_VERSION,
            "built_at": snapshot.built_at.isoformat(),
            "training_summary": snapshot.engine.training_summary,
            "summary": snapshot.summary,
            "cycles_detected": len(snapshot.cycles),
            "cycle_baseline": snapshot.cycle_baseline,
        }

    def score_latest(self, source: str, limit: int) -> pd.DataFrame:
        snapshot = self.get(source)
        if source == "file":
            return snapshot.scored_full.tail(limit).reset_index(drop=True)

        context = max(settings.latest_context_rows, limit)
        raw = load_latest(source, limit=limit, context_rows=context)
        scored = snapshot.engine.score(raw)
        return scored.tail(limit).reset_index(drop=True)

    def score_time_window(
        self, source: str, hours: float
    ) -> tuple[pd.DataFrame, pd.Timestamp | None, pd.Timestamp | None]:
        """Puntua cada lectura de las ultimas `hours` horas contra el modelo protegido actual."""
        source = source.strip().lower()
        snapshot = self.get(source)
        raw, start, end = load_time_window(
            source, hours=hours, context_rows=settings.latest_context_rows
        )
        if raw.empty or start is None or end is None:
            return raw, start, end

        scored = snapshot.engine.score(raw)
        mask = (scored["fecha_hora"] >= start) & (scored["fecha_hora"] <= end)
        return scored.loc[mask].reset_index(drop=True), start, end

    def events(
        self,
        source: str,
        threshold: float,
        scope: str = "recent",
        limit_rows: int = 5000,
    ):
        source = source.strip().lower()
        scope = scope.strip().lower()
        snapshot = self.get(source)

        if scope == "full":
            scored = snapshot.scored_full
        elif scope == "recent":
            if source == "file":
                scored = snapshot.scored_full.tail(limit_rows).reset_index(drop=True)
            else:
                raw = load_latest(source, limit=limit_rows, context_rows=settings.latest_context_rows)
                scored = snapshot.engine.score(raw)
        else:
            raise ValueError("scope debe ser 'recent' o 'full'.")
        return build_events(scored, threshold=threshold)

    def cycles_latest(self, source: str, limit: int):
        snapshot = self.get(source)
        return snapshot.cycles[-limit:]

    def new_regime_candidates(self, source: str) -> list[dict[str, Any]]:
        snapshot = self.get(source)
        return list(snapshot.cycle_baseline.get("nuevos_regimenes_candidatos", []))

    def new_regime_families(self, source: str) -> list[dict[str, Any]]:
        snapshot = self.get(source)
        return list(snapshot.cycle_baseline.get("familias_regimen_candidatas", []))

    def controlled_trials_validation(
        self,
        source: str,
        detection_threshold: float | None = None,
    ) -> dict[str, Any]:
        snapshot = self.get(source)
        events = build_events(snapshot.scored_full, threshold=settings.event_open_threshold)
        return validate_controlled_trials(
            snapshot.scored_full,
            events=events,
            detection_threshold=detection_threshold,
        )

    def controlled_trials_rows(self, source: str) -> pd.DataFrame:
        snapshot = self.get(source)
        scored = snapshot.scored_full
        if "es_sesion_pruebas_controladas" not in scored.columns:
            return scored.iloc[0:0].copy()
        return scored.loc[scored["es_sesion_pruebas_controladas"].fillna(False).astype(bool)].copy()


model_manager = ModelManager()
