from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.config import settings


MODEL_VERSION = "v3.0-regime-aware-engineering-tolerance-restart-context"

REST_FEATURES = [
    "flujo_instantaneo",
    "presion_relativa",
    "temperatura_tanque",
    "nivel_tanque",
    "voltaje_bus_dc",
    "tension_l1_n",
]
ACTIVE_FEATURES = [
    "flujo_instantaneo",
    "presion_relativa",
    "nivel_tanque",
    "frecuencia_salida",
    "velocidad",
    "voltaje_salida",
    "voltaje_bus_dc",
    "tension_l1_n",
    "corriente_l1",
    "potencia_activa_l1",
    "potencia_aparente_l1",
]

# v3.0 separa los estados que antes se mezclaban dentro de REPOSO.
PHASE_FEATURES: dict[str, list[str]] = {
    "REPOSO_ESTATICO": REST_FEATURES,
    "VACIADO": REST_FEATURES,
    "ESPERA_CICLO": REST_FEATURES,
    "REPOSO_TRANSITORIO": REST_FEATURES,
    "ARRANQUE": ACTIVE_FEATURES,
    "OPERACION_ESTABLE": ACTIVE_FEATURES,
    "DESACELERACION": ACTIVE_FEATURES,
    "POST_PARO": REST_FEATURES,
}

NUMERIC_COLUMNS = sorted({c for cols in PHASE_FEATURES.values() for c in cols})

BIT_COLUMNS = [
    "nivel_alto_alto",
    "nivel_alto",
    "nivel_bajo_bajo",
    "nivel_bajo",
    "presion_alta",
    "falta_presion",
    "bajo_flujo",
    "falla_vfd",
    "vfd_mantenimiento",
    "solenoide_superior_mantenimiento",
    "solenoide_inferior_mantenimiento",
    "manual_auto_vfd",
    "control",
    "paro_emergencia_raw",
    "start_stop_vfd",
    "manual_auto_s1",
    "start_stop_s1",
    "manual_auto_s2",
    "start_stop_s2",
    "recirculacion_automatica",
    "arranque_paro_llenado",
    "arranque_paro_vaciado",
    "encendido_ia",
    "encendido_local",
]

ENRICHED_COLUMNS = [
    "paro_emergencia_raw",
    "frecuencia_vfd_hz",
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
]

OPTIONAL_NUMERIC_CONTEXT = [
    "frecuencia_vfd_hz",
    "estado_s1",
    "estado_s2",
    "setpoint_llenado",
    "setpoint_vaciado",
    "horas_marcha",
    "numero_arranques",
]

MAINTENANCE_COLUMNS = [
    "vfd_mantenimiento",
    "solenoide_superior_mantenimiento",
    "solenoide_inferior_mantenimiento",
]

EXCLUDED_FROM_MODEL = {
    "energia_aparente_l1": "valor congelado en PLC; pendiente de correccion",
    "numero_arranques": "contador de arranques manuales, no ciclos automaticos",
    "start_stop_vfd": "significado operativo aun pendiente de confirmar",
    "start_stop_s1": "contexto de valvula; se usa como contexto, no como predictor numerico",
    "start_stop_s2": "contexto de valvula; se usa como contexto, no como predictor numerico",
    "estado_s1": "contexto automatico del ciclo",
    "estado_s2": "contexto automatico del ciclo",
    "horas_marcha": "contexto de desgaste acumulado, no rareza instantanea",
}

PASSIVE_MODEL_PHASES = {"REPOSO_ESTATICO", "VACIADO", "ESPERA_CICLO", "REPOSO_TRANSITORIO"}
ACTIVE_EVENT_PHASES = {"ARRANQUE", "OPERACION_ESTABLE", "DESACELERACION"}
RESTART_CONTEXTS = {"INICIO_HISTORIAL", "REANUDACION", "ESTABILIZACION"}


@dataclass
class ModelBundle:
    scaler: RobustScaler
    model: IsolationForest
    sorted_raw_scores: np.ndarray
    features: list[str]
    medians: pd.Series
    training_rows: int


class AnomalyEngine:
    """Motor v3.0: regimenes, reanudacion y tolerancias fisicas."""

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: dict[str, ModelBundle] = {}
        self.training_summary: dict[str, Any] = {}

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        try:
            if pd.isna(value):
                return default
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        try:
            if pd.isna(value):
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optional_bool(value: Any) -> bool | None:
        if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
            return None
        if isinstance(value, str):
            text = value.strip().lower()
            if text in {"1", "true", "yes", "on"}:
                return True
            if text in {"0", "false", "no", "off"}:
                return False
            return None
        try:
            return bool(int(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _truthy(cls, value: Any) -> bool:
        return cls._optional_bool(value) is True

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df.copy()

        out = df.copy()
        if "fecha_hora" not in out.columns:
            raise ValueError("La fuente no contiene fecha_hora.")
        out["fecha_hora"] = pd.to_datetime(out["fecha_hora"], errors="coerce")
        out = out.dropna(subset=["fecha_hora"])
        if "id" in out.columns:
            out["id"] = pd.to_numeric(out["id"], errors="coerce")
            out = out.sort_values(["fecha_hora", "id"], kind="stable")
        else:
            out = out.sort_values("fecha_hora", kind="stable")
        out = out.reset_index(drop=True)

        for col in NUMERIC_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")

        for col in ENRICHED_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        for col in BIT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        for col in ["estado_vfd", *OPTIONAL_NUMERIC_CONTEXT]:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")

        key_context = ["setpoint_llenado", "setpoint_vaciado", "encendido_ia", "horas_marcha"]
        context_count = out[key_context].notna().sum(axis=1)
        out["contexto_version"] = np.where(context_count > 0, "ENRIQUECIDO", "LEGACY")
        out["contexto_nuevas_variables_disponibles"] = out[ENRICHED_COLUMNS].notna().sum(axis=1)

        derived = self._derive_temporal_context(out)
        for name, values in derived.items():
            out[name] = values
        return out

    def _derive_temporal_context(self, df: pd.DataFrame) -> dict[str, list[Any]]:
        phases: list[str] = []
        model_phases: list[str] = []
        rest_states: list[str] = []
        moving_values: list[bool] = []
        cycle_ids: list[float] = []
        sec_from_start_values: list[float] = []
        sec_from_stop_values: list[float] = []
        pressure_high_durations: list[float] = []
        level_slopes: list[float] = []
        cycle_contexts: list[str] = []
        cycle_gap_hours: list[float] = []
        cycle_after_restart_numbers: list[float] = []

        movement_threshold = settings.movement_speed_threshold
        stable_threshold = settings.stable_speed_threshold
        post_stop_seconds = settings.post_stop_seconds

        cycle_counter = 0
        current_cycle: int | None = None
        movement_started_at: pd.Timestamp | None = None
        last_movement_at: pd.Timestamp | None = None
        reached_stable = False
        prev_moving = False

        current_cycle_context = "NO_APLICA"
        current_gap_hours = np.nan
        current_restart_number = np.nan
        stabilization_remaining = 0
        stabilization_index = 0

        pressure_high_started_at: pd.Timestamp | None = None
        prev_ts: pd.Timestamp | None = None
        prev_level: float | None = None

        for _, row in df.iterrows():
            ts = row["fecha_hora"]
            speed = self._safe_float(row.get("velocidad"), 0.0)
            vout = self._safe_float(row.get("voltaje_salida"), 0.0)
            moving = speed > movement_threshold or vout > 5.0

            level = self._optional_float(row.get("nivel_tanque"))
            slope = np.nan
            if prev_ts is not None and prev_level is not None and level is not None:
                dt = (ts - prev_ts).total_seconds()
                if 0 < dt <= settings.rest_max_derivative_gap_seconds:
                    slope = (level - prev_level) / dt

            if moving and not prev_moving:
                cycle_counter += 1
                current_cycle = cycle_counter
                movement_started_at = ts
                reached_stable = False

                if last_movement_at is None:
                    current_cycle_context = "INICIO_HISTORIAL"
                    current_gap_hours = np.nan
                    current_restart_number = np.nan
                else:
                    gap_hours = max(0.0, (ts - last_movement_at).total_seconds() / 3600.0)
                    current_gap_hours = gap_hours
                    if gap_hours >= settings.restart_long_stop_hours:
                        current_cycle_context = "REANUDACION"
                        stabilization_remaining = max(0, settings.restart_stabilization_cycles)
                        stabilization_index = 0
                        current_restart_number = 0.0
                    elif stabilization_remaining > 0:
                        stabilization_index += 1
                        current_cycle_context = "ESTABILIZACION"
                        current_restart_number = float(stabilization_index)
                        stabilization_remaining -= 1
                    else:
                        current_cycle_context = "NORMAL"
                        current_restart_number = np.nan

            if moving:
                last_movement_at = ts
                if speed >= stable_threshold:
                    reached_stable = True
                    phase = "OPERACION_ESTABLE"
                elif reached_stable:
                    phase = "DESACELERACION"
                else:
                    phase = "ARRANQUE"
                model_phase = phase
                rest_state = "NO_APLICA"
                sec_from_start = (
                    max(0.0, (ts - movement_started_at).total_seconds())
                    if movement_started_at is not None
                    else 0.0
                )
                sec_from_stop = np.nan
                cycle_id = float(current_cycle) if current_cycle is not None else np.nan
                row_cycle_context = current_cycle_context
                row_gap_hours = current_gap_hours
                row_restart_number = current_restart_number
            else:
                sec_from_start = np.nan
                if last_movement_at is not None:
                    sec = max(0.0, (ts - last_movement_at).total_seconds())
                else:
                    sec = np.inf

                if sec <= post_stop_seconds:
                    phase = "POST_PARO"
                    model_phase = "POST_PARO"
                    rest_state = "POST_PARO"
                    sec_from_stop = sec
                    cycle_id = float(current_cycle) if current_cycle is not None else np.nan
                    row_cycle_context = current_cycle_context
                    row_gap_hours = current_gap_hours
                    row_restart_number = current_restart_number
                else:
                    phase = "REPOSO"
                    sec_from_stop = sec if np.isfinite(sec) else np.nan
                    cycle_id = np.nan
                    row_cycle_context = "NO_APLICA"
                    row_gap_hours = np.nan
                    row_restart_number = np.nan
                    current_cycle = None
                    movement_started_at = None
                    reached_stable = False
                    current_cycle_context = "NO_APLICA"
                    current_gap_hours = np.nan
                    current_restart_number = np.nan

                    setpoint_empty = self._optional_float(row.get("setpoint_vaciado"))
                    recirc = self._optional_bool(row.get("recirculacion_automatica"))
                    near_empty_setpoint = (
                        level is not None
                        and setpoint_empty is not None
                        and level <= setpoint_empty + settings.rest_wait_setpoint_margin
                    )
                    if np.isfinite(slope) and slope <= settings.rest_drain_slope_threshold:
                        rest_state = "VACIADO"
                    elif near_empty_setpoint and recirc is not False:
                        rest_state = "ESPERA_CICLO"
                    elif np.isfinite(slope) and abs(slope) <= settings.rest_static_slope_threshold:
                        rest_state = "REPOSO_ESTATICO"
                    elif not np.isfinite(slope):
                        rest_state = "REPOSO_ESTATICO"
                    else:
                        rest_state = "REPOSO_TRANSITORIO"
                    model_phase = rest_state

            pressure_high = self._truthy(row.get("presion_alta"))
            if pressure_high:
                if (
                    pressure_high_started_at is None
                    or (prev_ts is not None and (ts - prev_ts).total_seconds() > 15)
                ):
                    pressure_high_started_at = ts
                ph_duration = max(0.0, (ts - pressure_high_started_at).total_seconds())
            else:
                pressure_high_started_at = None
                ph_duration = 0.0

            phases.append(phase)
            model_phases.append(model_phase)
            rest_states.append(rest_state)
            moving_values.append(moving)
            cycle_ids.append(cycle_id)
            sec_from_start_values.append(sec_from_start)
            sec_from_stop_values.append(sec_from_stop)
            pressure_high_durations.append(ph_duration)
            level_slopes.append(slope)
            cycle_contexts.append(row_cycle_context)
            cycle_gap_hours.append(row_gap_hours)
            cycle_after_restart_numbers.append(row_restart_number)

            prev_moving = moving
            prev_ts = ts
            if level is not None:
                prev_level = level

        return {
            "fase_operativa": phases,
            "fase_modelo": model_phases,
            "subestado_reposo": rest_states,
            "en_movimiento": moving_values,
            "ciclo_id": cycle_ids,
            "segundos_desde_arranque": sec_from_start_values,
            "segundos_desde_paro": sec_from_stop_values,
            "duracion_presion_alta_seg": pressure_high_durations,
            "pendiente_nivel_por_seg": level_slopes,
            "contexto_ciclo": cycle_contexts,
            "horas_paro_previas_al_ciclo": cycle_gap_hours,
            "numero_ciclo_estabilizacion": cycle_after_restart_numbers,
        }

    def _training_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)
        for col in ["falla_vfd", *MAINTENANCE_COLUMNS]:
            if col in df.columns:
                mask &= ~df[col].apply(self._truthy)

        if "estado_vfd" in df.columns:
            estado = pd.to_numeric(df["estado_vfd"], errors="coerce")
            mask &= ~estado.isin([3, 4])

        seconds = pd.to_numeric(df.get("segundos_desde_arranque"), errors="coerce")
        after_grace = seconds.fillna(0).gt(settings.startup_grace_seconds)
        for col in ["falta_presion", "bajo_flujo"]:
            if col in df.columns:
                flag = df[col].apply(self._truthy)
                mask &= ~(flag & after_grace)

        # La linea base principal no aprende los primeros ciclos tras un paro prolongado.
        if "contexto_ciclo" in df.columns:
            mask &= ~df["contexto_ciclo"].isin(RESTART_CONTEXTS)
        return mask

    def fit(self, df: pd.DataFrame) -> "AnomalyEngine":
        data = self._prepare(df)
        self.models = {}
        base_mask = self._training_mask(data)

        per_phase: dict[str, Any] = {}
        for phase, features in PHASE_FEATURES.items():
            phase_mask = data["fase_modelo"].eq(phase)
            train = data.loc[phase_mask & base_mask, features].copy()
            used_fallback = False
            if len(train) < 30:
                train = data.loc[phase_mask, features].copy()
                used_fallback = True
            if len(train) < 15:
                per_phase[phase] = {"training_rows": int(len(train)), "trained": False}
                continue

            train = train.apply(pd.to_numeric, errors="coerce")
            usable = [c for c in features if train[c].notna().any()]
            if not usable:
                per_phase[phase] = {"training_rows": int(len(train)), "trained": False}
                continue
            train = train[usable]
            medians = train.median(numeric_only=True)
            train = train.fillna(medians).fillna(0.0)

            scaler = RobustScaler()
            x = scaler.fit_transform(train)
            model = IsolationForest(
                n_estimators=240,
                contamination="auto",
                random_state=self.random_state,
                n_jobs=-1,
                max_samples="auto",
            )
            model.fit(x)
            raw = -model.score_samples(x)
            self.models[phase] = ModelBundle(
                scaler=scaler,
                model=model,
                sorted_raw_scores=np.sort(raw),
                features=usable,
                medians=medians,
                training_rows=int(len(train)),
            )
            per_phase[phase] = {
                "training_rows": int(len(train)),
                "trained": True,
                "features": usable,
                "fallback_incluye_contexto_reanudacion": used_fallback,
            }

        self.training_summary = {
            "total_rows": int(len(data)),
            "legacy_rows": int(data["contexto_version"].eq("LEGACY").sum()),
            "enriched_rows": int(data["contexto_version"].eq("ENRIQUECIDO").sum()),
            "regimenes_modelo": dict(Counter(data["fase_modelo"].astype(str))),
            "contextos_ciclo": dict(Counter(data["contexto_ciclo"].astype(str))),
            "phases": per_phase,
            "excluded_variables": EXCLUDED_FROM_MODEL,
            "restart_policy": {
                "long_stop_hours": settings.restart_long_stop_hours,
                "stabilization_cycles": settings.restart_stabilization_cycles,
                "ml_cap": settings.restart_ml_cap,
            },
        }
        return self

    @staticmethod
    def _percentile_to_index(percentile: np.ndarray | float) -> np.ndarray | float:
        p = np.asarray(percentile, dtype=float)
        low = np.clip(p / 0.90 * 20.0, 0.0, 20.0)
        high = 20.0 + np.clip((p - 0.90) / 0.10, 0.0, 1.0) * 80.0
        out = np.where(p <= 0.90, low, high)
        if np.isscalar(percentile):
            return float(out)
        return out

    @staticmethod
    def _severity(index: float) -> str:
        if index >= 90:
            return "MUY_ALTO"
        if index >= 70:
            return "ALTO"
        if index >= 40:
            return "MEDIO"
        return "BAJO"

    @staticmethod
    def _confidence(phase_model: str, context_version: str, maintenance: bool, cycle_context: str) -> str:
        if maintenance:
            return "BAJA"
        if cycle_context in RESTART_CONTEXTS:
            return "MEDIA"
        base = {
            "OPERACION_ESTABLE": 3,
            "ARRANQUE": 2,
            "DESACELERACION": 2,
            "POST_PARO": 1,
            "VACIADO": 2,
            "ESPERA_CICLO": 2,
            "REPOSO_ESTATICO": 2,
            "REPOSO_TRANSITORIO": 1,
        }.get(phase_model, 1)
        if context_version == "ENRIQUECIDO":
            base += 1
        return {1: "BAJA", 2: "MEDIA", 3: "ALTA", 4: "ALTA"}.get(base, "BAJA")

    def _ml_indices(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(0.0, index=data.index, dtype=float)
        for phase, bundle in self.models.items():
            idx = data.index[data["fase_modelo"].eq(phase)]
            if len(idx) == 0:
                continue
            x = data.loc[idx, bundle.features].apply(pd.to_numeric, errors="coerce")
            x = x.fillna(bundle.medians).fillna(0.0)
            scaled = bundle.scaler.transform(x)
            raw = -bundle.model.score_samples(scaled)
            p = np.searchsorted(bundle.sorted_raw_scores, raw, side="right") / max(
                1, bundle.sorted_raw_scores.size
            )
            result.loc[idx] = self._percentile_to_index(p)
        return result

    def _context_rules(
        self, row: pd.Series
    ) -> tuple[float, float, list[str], list[str], list[str]]:
        hard = 0.0
        warning = 0.0
        reasons: list[str] = []
        warnings: list[str] = []
        observations: list[str] = []

        phase = str(row.get("fase_operativa", ""))
        speed = self._safe_float(row.get("velocidad"), 0.0)
        flow = self._safe_float(row.get("flujo_instantaneo"), 0.0)
        pressure = self._safe_float(row.get("presion_relativa"), 0.0)
        sec_start = self._optional_float(row.get("segundos_desde_arranque"))

        if phase in {"ARRANQUE", "OPERACION_ESTABLE"} and speed >= 500 and flow < 1.0:
            hard = max(hard, 96.0)
            reasons.append("velocidad_alta_sin_respuesta_de_flujo")
        if phase in {"ARRANQUE", "OPERACION_ESTABLE"} and speed >= 700 and pressure < 0.5:
            hard = max(hard, 96.0)
            reasons.append("velocidad_alta_sin_respuesta_de_presion")

        if self._truthy(row.get("falta_presion")):
            if sec_start is not None and sec_start <= settings.startup_grace_seconds:
                observations.append("falta_presion_durante_gracia_de_arranque")
            elif speed >= 500:
                hard = max(hard, 85.0)
                reasons.append("falta_presion_persistente_con_bomba_en_movimiento")
            else:
                warning = max(warning, 30.0)
                warnings.append("senal_falta_presion_activa")

        if self._truthy(row.get("bajo_flujo")):
            if sec_start is not None and sec_start <= settings.startup_grace_seconds:
                observations.append("bajo_flujo_durante_gracia_de_arranque")
            elif speed >= 500:
                hard = max(hard, 85.0)
                reasons.append("bajo_flujo_persistente_con_bomba_en_movimiento")
            else:
                warning = max(warning, 30.0)
                warnings.append("senal_bajo_flujo_activa")

        if self._truthy(row.get("presion_alta")):
            duration = self._safe_float(row.get("duracion_presion_alta_seg"), 0.0)
            if duration >= settings.pressure_warning_persistent_seconds:
                warning = max(warning, 35.0)
                warnings.append("umbral_presion_alta_persistente")
            else:
                warning = max(warning, 20.0)
                warnings.append("primer_umbral_presion_alta_activo")

        if self._truthy(row.get("falla_vfd")):
            hard = 100.0
            reasons.append("senal_falla_vfd_activa")

        estado_vfd = self._optional_float(row.get("estado_vfd"))
        if estado_vfd == 3:
            hard = 100.0
            reasons.append("estado_vfd_error")

        return hard, warning, reasons, warnings, observations

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        data = self._prepare(df)
        if data.empty:
            return data
        if not self.models:
            self.fit(data)

        ml_raw = self._ml_indices(data)
        rows: list[dict[str, Any]] = []
        for row_index, row in data.iterrows():
            phase = str(row["fase_operativa"])
            phase_model = str(row["fase_modelo"])
            cycle_context = str(row.get("contexto_ciclo", "NO_APLICA"))
            maintenance = any(self._truthy(row.get(c)) for c in MAINTENANCE_COLUMNS)
            if self._optional_float(row.get("estado_vfd")) == 4:
                maintenance = True

            raw_ml_index = float(ml_raw.loc[row_index])
            rule_index, warning_index, reasons, warnings, observations = self._context_rules(row)

            effective_ml = raw_ml_index
            if cycle_context in RESTART_CONTEXTS and rule_index < settings.event_immediate_rule_threshold:
                effective_ml = min(effective_ml, settings.restart_ml_cap)
                if raw_ml_index > effective_ml:
                    observations.append("rareza_estadistica_limitada_por_reanudacion")
                if cycle_context == "REANUDACION":
                    observations.append("reanudacion_tras_paro_prolongado")
                elif cycle_context == "ESTABILIZACION":
                    observations.append("estabilizacion_post_reanudacion")
                elif cycle_context == "INICIO_HISTORIAL":
                    observations.append("primer_ciclo_del_historial_sin_contexto_previo")

            index = max(effective_ml, rule_index)
            if effective_ml >= 70:
                reasons.append("patron_multivariable_atipico_para_el_regimen")
            if maintenance:
                observations.append("equipo_en_mantenimiento")

            context_version = str(row.get("contexto_version", "LEGACY"))
            rows.append(
                {
                    **row.to_dict(),
                    "indice_anomalia": round(float(index), 2),
                    "nivel_anomalia": self._severity(index),
                    "confianza_evaluacion": self._confidence(
                        phase_model, context_version, maintenance, cycle_context
                    ),
                    "razones": sorted(set(reasons)),
                    "advertencias": sorted(set(warnings)),
                    "observaciones": sorted(set(observations)),
                    "indice_ml_crudo": round(float(raw_ml_index), 2),
                    "indice_ml": round(float(effective_ml), 2),
                    "indice_reglas": round(float(rule_index), 2),
                    "indice_advertencias": round(float(warning_index), 2),
                    "en_mantenimiento": maintenance,
                }
            )
        return pd.DataFrame(rows)


def dataframe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for key, value in record.items():
            if isinstance(value, pd.Timestamp):
                clean[key] = value.isoformat()
            elif isinstance(value, np.integer):
                clean[key] = int(value)
            elif isinstance(value, np.floating):
                clean[key] = None if np.isnan(value) else float(value)
            elif isinstance(value, np.bool_):
                clean[key] = bool(value)
            elif not isinstance(value, (list, dict)) and pd.isna(value):
                clean[key] = None
            else:
                clean[key] = value
        out.append(clean)
    return out


def build_summary(scored: pd.DataFrame) -> dict[str, Any]:
    if scored.empty:
        return {
            "total_registros": 0,
            "indice_actual": None,
            "indice_maximo": None,
            "indice_promedio": None,
            "conteo_por_nivel": {},
            "conteo_por_fase": {},
            "conteo_por_regimen_modelo": {},
            "conteo_por_contexto": {},
        }

    levels = Counter(scored["nivel_anomalia"].astype(str))
    phases = Counter(scored["fase_operativa"].astype(str))
    model_phases = Counter(scored["fase_modelo"].astype(str))
    contexts = Counter(scored["contexto_version"].astype(str))
    cycle_contexts = Counter(scored["contexto_ciclo"].astype(str))
    rest_states = Counter(scored.loc[scored["fase_operativa"].eq("REPOSO"), "subestado_reposo"].astype(str))
    latest = scored.iloc[-1]
    return {
        "total_registros": int(len(scored)),
        "desde": scored["fecha_hora"].iloc[0].isoformat(),
        "hasta": scored["fecha_hora"].iloc[-1].isoformat(),
        "indice_actual": round(float(latest["indice_anomalia"]), 2),
        "indice_maximo": round(float(scored["indice_anomalia"].max()), 2),
        "indice_promedio": round(float(scored["indice_anomalia"].mean()), 2),
        "registros_indice_alto": int(levels.get("ALTO", 0)),
        "registros_indice_muy_alto": int(levels.get("MUY_ALTO", 0)),
        "registros_indice_alto_o_superior": int(levels.get("ALTO", 0) + levels.get("MUY_ALTO", 0)),
        "conteo_por_nivel": dict(levels),
        "conteo_por_fase": dict(phases),
        "conteo_por_regimen_modelo": dict(model_phases),
        "conteo_por_contexto": dict(contexts),
        "conteo_por_contexto_ciclo": dict(cycle_contexts),
        "conteo_subestado_reposo": dict(rest_states),
        "advertencias_presion_alta": int(
            scored.get("advertencias", pd.Series([[]] * len(scored), index=scored.index))
            .apply(lambda x: any("presion_alta" in r for r in (x or [])))
            .sum()
        ),
    }


def build_events(
    scored: pd.DataFrame,
    threshold: float | None = None,
    max_gap_seconds: float | None = None,
) -> list[dict[str, Any]]:
    if scored.empty:
        return []

    open_threshold = float(settings.event_open_threshold if threshold is None else threshold)
    max_gap = float(settings.event_max_gap_seconds if max_gap_seconds is None else max_gap_seconds)
    close_threshold = min(float(settings.event_close_threshold), open_threshold)
    min_consecutive = max(1, int(settings.event_min_consecutive))
    min_duration = max(0.0, float(settings.event_min_duration_seconds))
    close_seconds = max(0.0, float(settings.event_close_seconds))
    immediate_rule = float(settings.event_immediate_rule_threshold)

    work = scored.sort_values("fecha_hora").reset_index(drop=True)
    events_frames: list[tuple[pd.DataFrame, str]] = []
    pending: list[pd.Series] = []
    active: list[pd.Series] = []
    recovery: list[pd.Series] = []
    active_open_reason = "persistencia_estadistica"
    previous_ts: pd.Timestamp | None = None

    def eligible_statistical(row: pd.Series) -> bool:
        if not settings.event_statistical_active_only:
            return True
        return str(row.get("fase_operativa", "")) in ACTIVE_EVENT_PHASES

    def flush_active():
        nonlocal active, recovery, active_open_reason
        if active:
            events_frames.append((pd.DataFrame(active), active_open_reason))
        active = []
        recovery = []
        active_open_reason = "persistencia_estadistica"

    for _, row in work.iterrows():
        ts = row["fecha_hora"]
        index = float(row.get("indice_anomalia", 0.0) or 0.0)
        rule_index = float(row.get("indice_reglas", 0.0) or 0.0)
        immediate = rule_index >= immediate_rule
        eligible = eligible_statistical(row)

        if previous_ts is not None and (ts - previous_ts).total_seconds() > max_gap:
            if active:
                flush_active()
            pending = []
            recovery = []

        if not active:
            if immediate:
                active = [row]
                active_open_reason = "regla_fisica_inmediata"
                pending = []
            elif eligible and index >= open_threshold:
                pending.append(row)
                if len(pending) > 1:
                    gap = (pending[-1]["fecha_hora"] - pending[-2]["fecha_hora"]).total_seconds()
                    if gap > max_gap:
                        pending = [row]
                elapsed = (pending[-1]["fecha_hora"] - pending[0]["fecha_hora"]).total_seconds()
                if len(pending) >= min_consecutive or elapsed >= min_duration:
                    active = list(pending)
                    active_open_reason = "persistencia_estadistica"
                    pending = []
            else:
                pending = []
        else:
            if immediate or (eligible and index >= close_threshold):
                if recovery:
                    active.extend(recovery)
                    recovery = []
                active.append(row)
            else:
                recovery.append(row)
                recovery_elapsed = (
                    recovery[-1]["fecha_hora"] - recovery[0]["fecha_hora"]
                ).total_seconds()
                if recovery_elapsed >= close_seconds:
                    flush_active()
                    pending = []

        previous_ts = ts

    if active:
        flush_active()

    events: list[dict[str, Any]] = []
    for idx, (frame, criterio_apertura) in enumerate(events_frames, start=1):
        if frame.empty:
            continue
        reason_counts = Counter(
            reason for reasons in frame.get("razones", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )
        warning_counts = Counter(
            reason for reasons in frame.get("advertencias", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )
        observation_counts = Counter(
            reason for reasons in frame.get("observaciones", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )
        phase_counts = Counter(frame["fase_operativa"].astype(str))
        regime_counts = Counter(frame.get("fase_modelo", frame["fase_operativa"]).astype(str))
        context_counts = Counter(frame.get("contexto_ciclo", pd.Series(["NO_APLICA"] * len(frame))).astype(str))
        start = frame["fecha_hora"].iloc[0]
        end = frame["fecha_hora"].iloc[-1]
        events.append(
            {
                "evento": idx,
                "inicio": start.isoformat(),
                "fin": end.isoformat(),
                "duracion_seg": round((end - start).total_seconds(), 2),
                "registros": int(len(frame)),
                "criterio_apertura": criterio_apertura,
                "indice_maximo": round(float(frame["indice_anomalia"].max()), 2),
                "indice_p95": round(float(frame["indice_anomalia"].quantile(0.95)), 2),
                "indice_promedio": round(float(frame["indice_anomalia"].mean()), 2),
                "fases": dict(phase_counts),
                "regimenes": dict(regime_counts),
                "contextos_ciclo": dict(context_counts),
                "razones": [item for item, _ in reason_counts.most_common(8)],
                "advertencias": [item for item, _ in warning_counts.most_common(8)],
                "observaciones": [item for item, _ in observation_counts.most_common(8)],
            }
        )
    return events


def _metric_tolerances(metric: str) -> tuple[float, float]:
    mapping = {
        "duracion_movimiento_seg": (
            settings.cycle_tol_duration_abs_seconds,
            settings.cycle_tol_duration_pct,
        ),
        "nivel_inicio": (settings.cycle_tol_level_start_abs, settings.cycle_tol_level_start_pct),
        "nivel_fin": (settings.cycle_tol_level_end_abs, settings.cycle_tol_level_end_pct),
        "flujo_estable_mediana": (settings.cycle_tol_flow_abs, settings.cycle_tol_flow_pct),
        "presion_estable_mediana": (
            settings.cycle_tol_pressure_abs,
            settings.cycle_tol_pressure_pct,
        ),
        "velocidad_estable_mediana": (settings.cycle_tol_speed_abs, settings.cycle_tol_speed_pct),
    }
    return mapping[metric]


def _cycle_signature_detail(
    value: float | None,
    median: float | None,
    mad: float | None,
    abs_tolerance: float,
    pct_tolerance: float,
) -> dict[str, float | None]:
    if value is None or median is None:
        return {
            "score": 0.0,
            "z_robusto": None,
            "desviacion_abs": None,
            "desviacion_pct": None,
            "tolerancia_ingenieria": None,
            "ratio_tolerancia": None,
        }

    delta = abs(float(value) - float(median))
    pct = None if abs(float(median)) <= 1e-12 else delta / abs(float(median))
    tolerance = max(float(abs_tolerance), abs(float(median)) * float(pct_tolerance))
    ratio = delta / tolerance if tolerance > 1e-12 else np.inf

    robust_z = None
    if mad is not None and np.isfinite(mad) and mad > 1e-12:
        robust_z = delta / (1.4826 * float(mad))

    z0 = float(settings.cycle_signature_z_start)
    z1 = max(z0 + 0.1, float(settings.cycle_signature_z_high))
    if robust_z is None or robust_z <= z0 or ratio <= 1.0:
        score = 0.0
    else:
        stat_score = 40.0 + np.clip((robust_z - z0) / (z1 - z0), 0.0, 1.0) * 60.0
        high_ratio = max(1.1, float(settings.cycle_signature_engineering_high_ratio))
        eng_score = 40.0 + np.clip((ratio - 1.0) / (high_ratio - 1.0), 0.0, 1.0) * 60.0
        # Debe ser raro estadisticamente Y relevante fisicamente.
        score = min(stat_score, eng_score)

    return {
        "score": float(np.clip(score, 0.0, 100.0)),
        "z_robusto": None if robust_z is None else float(robust_z),
        "desviacion_abs": float(delta),
        "desviacion_pct": None if pct is None else float(pct * 100.0),
        "tolerancia_ingenieria": float(tolerance),
        "ratio_tolerancia": None if not np.isfinite(ratio) else float(ratio),
    }


def build_cycles(scored: pd.DataFrame) -> list[dict[str, Any]]:
    if scored.empty or "ciclo_id" not in scored.columns:
        return []
    work = scored.loc[scored["ciclo_id"].notna()].copy()
    if work.empty:
        return []

    raw_cycles: list[dict[str, Any]] = []
    for cycle_id, frame in work.groupby("ciclo_id", sort=True):
        frame = frame.sort_values("fecha_hora")
        moving = frame.loc[frame["en_movimiento"].astype(bool)]
        if moving.empty:
            continue
        start = moving["fecha_hora"].iloc[0]
        end = moving["fecha_hora"].iloc[-1]
        stable = frame.loc[frame["fase_operativa"].eq("OPERACION_ESTABLE")]

        def list_counter(column: str) -> Counter:
            return Counter(
                item
                for items in frame.get(column, pd.Series([[]] * len(frame)))
                for item in (items or [])
            )

        def first_num(name: str):
            if name not in moving.columns:
                return None
            s = pd.to_numeric(moving[name], errors="coerce").dropna()
            return None if s.empty else float(s.iloc[0])

        def last_num(name: str):
            if name not in moving.columns:
                return None
            s = pd.to_numeric(moving[name], errors="coerce").dropna()
            return None if s.empty else float(s.iloc[-1])

        def med_stable(name: str):
            if stable.empty or name not in stable.columns:
                return None
            s = pd.to_numeric(stable[name], errors="coerce").dropna()
            return None if s.empty else float(s.median())

        context_counts = Counter(frame["contexto_version"].astype(str))
        primary_context = context_counts.most_common(1)[0][0] if context_counts else "LEGACY"
        cycle_ctx_series = moving.get("contexto_ciclo", pd.Series(["NORMAL"] * len(moving), index=moving.index))
        cycle_context = str(cycle_ctx_series.iloc[0]) if not cycle_ctx_series.empty else "NORMAL"
        gap_hours = first_num("horas_paro_previas_al_ciclo")
        restart_num = first_num("numero_ciclo_estabilizacion")
        complete = bool(frame["fase_operativa"].eq("POST_PARO").any())
        max_index = float(pd.to_numeric(frame["indice_anomalia"], errors="coerce").fillna(0).max())
        p95_index = float(pd.to_numeric(frame["indice_anomalia"], errors="coerce").fillna(0).quantile(0.95))
        raw_ml = pd.to_numeric(frame.get("indice_ml_crudo", frame["indice_anomalia"]), errors="coerce").fillna(0)
        rule_series = pd.to_numeric(frame["indice_reglas"], errors="coerce").fillna(0)
        raw_instant = np.maximum(raw_ml.to_numpy(), rule_series.to_numpy())
        p95_raw = float(np.quantile(raw_instant, 0.95)) if len(raw_instant) else 0.0
        max_rule = float(rule_series.max())

        raw_cycles.append(
            {
                "ciclo_id": int(cycle_id),
                "inicio": start.isoformat(),
                "fin_movimiento": end.isoformat(),
                "duracion_movimiento_seg": round((end - start).total_seconds(), 2),
                "registros": int(len(frame)),
                "registros_movimiento": int(len(moving)),
                "nivel_inicio": first_num("nivel_tanque"),
                "nivel_fin": last_num("nivel_tanque"),
                "flujo_estable_mediana": med_stable("flujo_instantaneo"),
                "presion_estable_mediana": med_stable("presion_relativa"),
                "velocidad_estable_mediana": med_stable("velocidad"),
                "indice_maximo_instantaneo": round(max_index, 2),
                "indice_maximo": round(max_index, 2),
                "indice_p95": round(p95_index, 2),
                "indice_p95_crudo": round(p95_raw, 2),
                "indice_regla_fisica_max": round(max_rule, 2),
                "ciclo_completo": complete,
                "contexto_principal": primary_context,
                "contexto_ciclo": cycle_context,
                "horas_paro_previas": None if gap_hours is None else round(gap_hours, 3),
                "numero_ciclo_estabilizacion": None if restart_num is None else int(restart_num),
                "max_duracion_presion_alta_seg": round(
                    float(pd.to_numeric(frame["duracion_presion_alta_seg"], errors="coerce").max()), 2
                ),
                "contexto": dict(context_counts),
                "razones": [item for item, _ in list_counter("razones").most_common(8)],
                "advertencias": [item for item, _ in list_counter("advertencias").most_common(8)],
                "observaciones": [item for item, _ in list_counter("observaciones").most_common(8)],
            }
        )

    if not raw_cycles:
        return []

    metric_names = [
        "duracion_movimiento_seg",
        "nivel_inicio",
        "nivel_fin",
        "flujo_estable_mediana",
        "presion_estable_mediana",
        "velocidad_estable_mediana",
    ]

    complete_normal = [
        c
        for c in raw_cycles
        if c["ciclo_completo"]
        and c["contexto_ciclo"] == "NORMAL"
        and c["indice_regla_fisica_max"] < settings.event_immediate_rule_threshold
    ]
    fallback_complete = [
        c
        for c in raw_cycles
        if c["ciclo_completo"]
        and c["contexto_ciclo"] not in RESTART_CONTEXTS
        and c["indice_regla_fisica_max"] < settings.event_immediate_rule_threshold
    ]

    baselines: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    baseline_sources: dict[str, str] = {}
    contexts = sorted({c["contexto_principal"] for c in raw_cycles})
    for context in contexts:
        subset = [c for c in complete_normal if c["contexto_principal"] == context]
        source = "NORMAL_MISMO_CONTEXTO"
        if len(subset) < settings.cycle_signature_min_baseline:
            subset = complete_normal
            source = "NORMAL_TODOS_CONTEXTOS"
        if len(subset) < settings.cycle_signature_min_baseline:
            subset = fallback_complete
            source = "FALLBACK_COMPLETOS_SIN_REANUDACION"
        metrics: dict[str, tuple[float | None, float | None]] = {}
        for metric in metric_names:
            values = np.array([float(c[metric]) for c in subset if c.get(metric) is not None], dtype=float)
            if values.size < settings.cycle_signature_min_baseline:
                metrics[metric] = (None, None)
            else:
                median = float(np.median(values))
                mad = float(np.median(np.abs(values - median)))
                metrics[metric] = (median, mad)
        baselines[context] = metrics
        baseline_sources[context] = source

    cycles: list[dict[str, Any]] = []
    for cycle in raw_cycles:
        if not cycle["ciclo_completo"]:
            cycle.update(
                {
                    "indice_firma_ciclo_crudo": None,
                    "indice_firma_ciclo": None,
                    "indice_ciclo_crudo": None,
                    "indice_ciclo": None,
                    "nivel_ciclo": "EN_CURSO",
                    "nivel_anomalia": "EN_CURSO",
                    "tuvo_pico_atipico": bool(
                        cycle["indice_maximo_instantaneo"] >= settings.cycle_instant_peak_threshold
                    ),
                    "desviaciones_ciclo": [],
                    "baseline_aplicado": baseline_sources.get(cycle["contexto_principal"]),
                }
            )
            cycles.append(cycle)
            continue

        baseline = baselines.get(cycle["contexto_principal"], {})
        raw_signature = 0.0
        deviations: list[dict[str, Any]] = []
        for metric in metric_names:
            median, mad = baseline.get(metric, (None, None))
            abs_tol, pct_tol = _metric_tolerances(metric)
            detail = _cycle_signature_detail(cycle.get(metric), median, mad, abs_tol, pct_tol)
            raw_signature = max(raw_signature, float(detail["score"] or 0.0))
            if detail["score"] and detail["score"] > 0:
                deviations.append(
                    {
                        "variable": metric,
                        "valor": round(float(cycle[metric]), 4),
                        "mediana_baseline": None if median is None else round(float(median), 4),
                        "z_robusto": None if detail["z_robusto"] is None else round(float(detail["z_robusto"]), 2),
                        "desviacion_abs": round(float(detail["desviacion_abs"]), 4),
                        "desviacion_pct": None if detail["desviacion_pct"] is None else round(float(detail["desviacion_pct"]), 2),
                        "tolerancia_ingenieria": round(float(detail["tolerancia_ingenieria"]), 4),
                        "ratio_tolerancia": None if detail["ratio_tolerancia"] is None else round(float(detail["ratio_tolerancia"]), 2),
                        "score_variable": round(float(detail["score"]), 2),
                    }
                )

        effective_signature = raw_signature
        if cycle["contexto_ciclo"] in RESTART_CONTEXTS:
            effective_signature = min(effective_signature, settings.restart_cycle_score_cap)
            if raw_signature > effective_signature:
                cycle["observaciones"] = sorted(
                    set(cycle["observaciones"] + ["firma_ciclo_limitada_por_contexto_reanudacion"])
                )

        raw_cycle_index = max(
            float(cycle["indice_p95_crudo"]),
            float(cycle["indice_regla_fisica_max"]),
            float(raw_signature),
        )
        cycle_index = max(
            float(cycle["indice_p95"]),
            float(cycle["indice_regla_fisica_max"]),
            float(effective_signature),
        )

        if effective_signature >= 70:
            cycle["razones"] = sorted(set(cycle["razones"] + ["firma_ciclo_fuera_de_tolerancia_ingenieria"]))
        elif raw_signature >= 70 and cycle["contexto_ciclo"] in RESTART_CONTEXTS:
            cycle["observaciones"] = sorted(
                set(cycle["observaciones"] + ["firma_atipica_esperable_durante_reanudacion"])
            )

        if cycle["contexto_ciclo"] == "REANUDACION":
            cycle["observaciones"] = sorted(set(cycle["observaciones"] + ["ciclo_reanudacion_tras_paro_prolongado"]))
        elif cycle["contexto_ciclo"] == "ESTABILIZACION":
            cycle["observaciones"] = sorted(set(cycle["observaciones"] + ["ciclo_en_estabilizacion_post_reanudacion"]))

        cycle.update(
            {
                "indice_firma_ciclo_crudo": round(float(raw_signature), 2),
                "indice_firma_ciclo": round(float(effective_signature), 2),
                "indice_ciclo_crudo": round(float(raw_cycle_index), 2),
                "indice_ciclo": round(float(cycle_index), 2),
                "nivel_ciclo": AnomalyEngine._severity(cycle_index),
                "nivel_anomalia": AnomalyEngine._severity(cycle_index),
                "tuvo_pico_atipico": bool(
                    cycle["indice_maximo_instantaneo"] >= settings.cycle_instant_peak_threshold
                ),
                "desviaciones_ciclo": deviations,
                "baseline_aplicado": baseline_sources.get(cycle["contexto_principal"]),
            }
        )
        cycles.append(cycle)
    return cycles


def cycle_baseline_summary(cycles: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        "duracion_movimiento_seg",
        "nivel_inicio",
        "nivel_fin",
        "flujo_estable_mediana",
        "presion_estable_mediana",
        "velocidad_estable_mediana",
    ]
    normal = [
        c
        for c in cycles
        if c.get("ciclo_completo")
        and c.get("contexto_ciclo") == "NORMAL"
        and float(c.get("indice_regla_fisica_max") or 0.0) < settings.event_immediate_rule_threshold
    ]
    out: dict[str, Any] = {
        "ciclos_normales_candidatos": len(normal),
        "restart_long_stop_hours": settings.restart_long_stop_hours,
        "restart_stabilization_cycles": settings.restart_stabilization_cycles,
        "metricas": {},
    }
    for metric in metrics:
        values = np.array([float(c[metric]) for c in normal if c.get(metric) is not None], dtype=float)
        if not len(values):
            out["metricas"][metric] = None
            continue
        median = float(np.median(values))
        abs_tol, pct_tol = _metric_tolerances(metric)
        tolerance = max(abs_tol, abs(median) * pct_tol)
        out["metricas"][metric] = {
            "mediana": round(median, 4),
            "p05": round(float(np.quantile(values, 0.05)), 4),
            "p95": round(float(np.quantile(values, 0.95)), 4),
            "tolerancia_ingenieria": round(float(tolerance), 4),
            "muestras": int(len(values)),
        }
    return out


def variable_coverage(df: pd.DataFrame) -> list[dict[str, Any]]:
    roles: dict[str, tuple[str, str]] = {
        "flujo_instantaneo": ("PREDICTORA", "ML y respuesta hidraulica"),
        "presion_relativa": ("PREDICTORA", "ML y respuesta hidraulica"),
        "temperatura_tanque": ("PREDICTORA", "ML"),
        "nivel_tanque": ("PREDICTORA", "ML, ciclos y deteccion de vaciado"),
        "frecuencia_salida": ("PREDICTORA", "ML"),
        "velocidad": ("PREDICTORA", "ML, movimiento y ciclos"),
        "voltaje_salida": ("PREDICTORA", "ML y movimiento"),
        "corriente_l1": ("PREDICTORA", "ML electrico"),
        "potencia_activa_l1": ("PREDICTORA", "ML electrico"),
        "potencia_aparente_l1": ("PREDICTORA", "ML electrico"),
        "presion_alta": ("ADVERTENCIA", "primer umbral; no eleva indice por si sola"),
        "falta_presion": ("REGLA", "desviacion si persiste tras gracia de arranque"),
        "bajo_flujo": ("REGLA", "desviacion si persiste tras gracia de arranque"),
        "falla_vfd": ("REGLA", "falla explicita"),
        "estado_vfd": ("CONTEXTO", "estado del VFD"),
        "manual_auto_vfd": ("CONTEXTO", "modo manual/automatico"),
        "control": ("CONTEXTO", "compatibilidad; mismo bit historico que Encendido-IA"),
        "paro_emergencia_raw": ("CONTEXTO", "bit crudo; polaridad pendiente de confirmar"),
        "frecuencia_vfd_hz": ("CONTEXTO", "consigna de frecuencia del VFD"),
        "manual_auto_s1": ("CONTEXTO", "modo de la valvula S1"),
        "start_stop_s1": ("CONTEXTO", "valvula cambia segun llenado/vaciado"),
        "estado_s1": ("CONTEXTO", "estado automatico de valvula S1"),
        "manual_auto_s2": ("CONTEXTO", "modo de la valvula S2"),
        "start_stop_s2": ("CONTEXTO", "valvula cambia segun llenado/vaciado"),
        "estado_s2": ("CONTEXTO", "estado automatico de valvula S2"),
        "recirculacion_automatica": ("CONTEXTO", "modo automatico del ciclo"),
        "setpoint_llenado": ("CONTEXTO", "setpoint del ciclo"),
        "arranque_paro_llenado": ("CONTEXTO", "mando/estado asociado a llenado"),
        "setpoint_vaciado": ("CONTEXTO", "setpoint del ciclo"),
        "arranque_paro_vaciado": ("CONTEXTO", "mando/estado asociado a vaciado"),
        "encendido_ia": ("CONTEXTO", "mando IA/edge"),
        "encendido_local": ("CONTEXTO", "mando local"),
        "horas_marcha": ("CONTEXTO", "uso acumulado"),
        "numero_arranques": ("EXCLUIDA_ML", "contador de arranques manuales"),
        "energia_aparente_l1": ("EXCLUIDA_ML", "congelada actualmente en el PLC"),
        "start_stop_vfd": ("EXCLUIDA_ML", "semantica operativa pendiente"),
    }
    total = max(1, len(df))
    result: list[dict[str, Any]] = []
    for name, (role, note) in roles.items():
        non_null = int(df[name].notna().sum()) if name in df.columns else 0
        result.append(
            {
                "variable": name,
                "rol": role,
                "nota": note,
                "registros_con_valor": non_null,
                "cobertura_pct": round(non_null / total * 100.0, 2),
            }
        )
    return result
