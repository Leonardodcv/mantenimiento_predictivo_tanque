from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from app.config import settings


MODEL_VERSION = "v2.2-mixed-history-cycle-aware-persistent-events"

# Solo se usan en ML variables que existen desde el esquema historico. Las variables
# nuevas se usan como contexto/reglas cuando existen, pero sus NULL antiguos no se
# convierten en cero ni se imputan como si hubieran sido medidos.
PHASE_FEATURES: dict[str, list[str]] = {
    "REPOSO": [
        "flujo_instantaneo",
        "presion_relativa",
        "temperatura_tanque",
        "nivel_tanque",
        "voltaje_bus_dc",
        "tension_l1_n",
    ],
    "ARRANQUE": [
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
    ],
    "OPERACION_ESTABLE": [
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
    ],
    "DESACELERACION": [
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
    ],
    "POST_PARO": [
        "flujo_instantaneo",
        "presion_relativa",
        "nivel_tanque",
        "voltaje_bus_dc",
        "tension_l1_n",
    ],
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

MAINTENANCE_COLUMNS = [
    "vfd_mantenimiento",
    "solenoide_superior_mantenimiento",
    "solenoide_inferior_mantenimiento",
]

# Se conservan y se exponen, pero deliberadamente NO participan del modelo v2.
EXCLUDED_FROM_MODEL = {
    "energia_aparente_l1": "valor congelado en PLC; pendiente de correccion",
    "numero_arranques": "contador de arranques manuales, no ciclos automaticos",
    "start_stop_vfd": "significado operativo aun pendiente de confirmar",
    "start_stop_s1": "contexto de valvula; no predictor numerico por ahora",
    "start_stop_s2": "contexto de valvula; no predictor numerico por ahora",
    "estado_s1": "contexto automatico del ciclo",
    "estado_s2": "contexto automatico del ciclo",
    "horas_marcha": "contexto de desgaste acumulado, no rareza instantanea",
}


@dataclass
class ModelBundle:
    scaler: RobustScaler
    model: IsolationForest
    sorted_raw_scores: np.ndarray
    features: list[str]
    medians: pd.Series
    training_rows: int


class AnomalyEngine:
    """
    Motor v2 para historial mixto.

    - Los registros antiguos con NULL en variables nuevas siguen siendo validos.
    - El ML usa solamente variables historicas comunes.
    - Las variables nuevas enriquecen contexto y explicabilidad cuando existen.
    - El indice 0-100 sigue siendo un indice de anomalia/rareza, NO una probabilidad
      de falla.
    """

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
        out["fecha_hora"] = pd.to_datetime(out["fecha_hora"], errors="coerce")
        out = out.dropna(subset=["fecha_hora"])
        if "id" in out.columns:
            out["id"] = pd.to_numeric(out["id"], errors="coerce")
            out = out.sort_values(["fecha_hora", "id"], kind="stable")
        else:
            out = out.sort_values("fecha_hora", kind="stable")
        out = out.reset_index(drop=True)

        # Columnas ML: si faltan totalmente en un archivo, se crean como NaN y
        # despues se imputan con mediana de la fase. En la tabla real existen.
        for col in NUMERIC_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan
            out[col] = pd.to_numeric(out[col], errors="coerce")

        # Contexto opcional: NUNCA convertir NULL historico a 0.
        for col in ENRICHED_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        for col in BIT_COLUMNS:
            if col not in out.columns:
                out[col] = np.nan

        for col in ["estado_vfd", "estado_s1", "estado_s2", "horas_marcha", "numero_arranques"]:
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
        moving_values: list[bool] = []
        cycle_ids: list[float] = []
        sec_from_start_values: list[float] = []
        sec_from_stop_values: list[float] = []
        pressure_high_durations: list[float] = []

        movement_threshold = settings.movement_speed_threshold
        stable_threshold = settings.stable_speed_threshold
        post_stop_seconds = settings.post_stop_seconds

        cycle_counter = 0
        current_cycle: int | None = None
        movement_started_at: pd.Timestamp | None = None
        last_movement_at: pd.Timestamp | None = None
        reached_stable = False
        prev_moving = False

        pressure_high_started_at: pd.Timestamp | None = None
        prev_ts: pd.Timestamp | None = None

        for _, row in df.iterrows():
            ts = row["fecha_hora"]
            speed = self._safe_float(row.get("velocidad"), 0.0)
            vout = self._safe_float(row.get("voltaje_salida"), 0.0)
            moving = speed > movement_threshold or vout > 5.0

            if moving and not prev_moving:
                cycle_counter += 1
                current_cycle = cycle_counter
                movement_started_at = ts
                reached_stable = False

            if moving:
                last_movement_at = ts
                if speed >= stable_threshold:
                    reached_stable = True
                    phase = "OPERACION_ESTABLE"
                elif reached_stable:
                    phase = "DESACELERACION"
                else:
                    phase = "ARRANQUE"
                sec_from_start = (
                    max(0.0, (ts - movement_started_at).total_seconds())
                    if movement_started_at is not None
                    else 0.0
                )
                sec_from_stop = np.nan
                cycle_id = float(current_cycle) if current_cycle is not None else np.nan
            else:
                sec_from_start = np.nan
                if last_movement_at is not None:
                    sec = max(0.0, (ts - last_movement_at).total_seconds())
                else:
                    sec = np.inf

                if sec <= post_stop_seconds:
                    phase = "POST_PARO"
                    sec_from_stop = sec
                    cycle_id = float(current_cycle) if current_cycle is not None else np.nan
                else:
                    phase = "REPOSO"
                    sec_from_stop = sec if np.isfinite(sec) else np.nan
                    cycle_id = np.nan
                    current_cycle = None
                    movement_started_at = None
                    reached_stable = False

            pressure_high = self._truthy(row.get("presion_alta"))
            if pressure_high:
                # Si hay una discontinuidad grande, reinicia el contador para no
                # inventar persistencia durante huecos de adquisicion.
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
            moving_values.append(moving)
            cycle_ids.append(cycle_id)
            sec_from_start_values.append(sec_from_start)
            sec_from_stop_values.append(sec_from_stop)
            pressure_high_durations.append(ph_duration)
            prev_moving = moving
            prev_ts = ts

        return {
            "fase_operativa": phases,
            "en_movimiento": moving_values,
            "ciclo_id": cycle_ids,
            "segundos_desde_arranque": sec_from_start_values,
            "segundos_desde_paro": sec_from_stop_values,
            "duracion_presion_alta_seg": pressure_high_durations,
        }

    def _training_mask(self, df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(True, index=df.index)

        for col in ["falla_vfd", *MAINTENANCE_COLUMNS]:
            if col in df.columns:
                mask &= ~df[col].apply(self._truthy)

        if "estado_vfd" in df.columns:
            estado = pd.to_numeric(df["estado_vfd"], errors="coerce")
            mask &= ~estado.isin([3, 4])

        grace = settings.startup_grace_seconds
        seconds = pd.to_numeric(df.get("segundos_desde_arranque"), errors="coerce")
        after_grace = seconds.fillna(0).gt(grace)

        # Falta de presion/bajo flujo durante los primeros segundos de arranque se
        # conserva como comportamiento transitorio normal; si persiste se excluye
        # de la linea base normal.
        for col in ["falta_presion", "bajo_flujo"]:
            if col in df.columns:
                flag = df[col].apply(self._truthy)
                mask &= ~(flag & after_grace)
        return mask

    def fit(self, df: pd.DataFrame) -> "AnomalyEngine":
        data = self._prepare(df)
        self.models = {}
        base_mask = self._training_mask(data)

        per_phase: dict[str, Any] = {}
        for phase, features in PHASE_FEATURES.items():
            phase_mask = data["fase_operativa"].eq(phase)
            train = data.loc[phase_mask & base_mask, features].copy()
            if len(train) < 30:
                train = data.loc[phase_mask, features].copy()
            if len(train) < 15:
                per_phase[phase] = {"training_rows": int(len(train)), "trained": False}
                continue

            train = train.apply(pd.to_numeric, errors="coerce")
            # Elimina variables totalmente vacias dentro de la fase.
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
                n_estimators=220,
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
            }

        self.training_summary = {
            "total_rows": int(len(data)),
            "legacy_rows": int(data["contexto_version"].eq("LEGACY").sum()),
            "enriched_rows": int(data["contexto_version"].eq("ENRIQUECIDO").sum()),
            "phases": per_phase,
            "excluded_variables": EXCLUDED_FROM_MODEL,
        }
        return self

    @staticmethod
    def _percentile_to_index(percentile: np.ndarray | float) -> np.ndarray | float:
        """Mapeo menos agresivo que v1: el 90 % tipico queda en 0-20."""
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
    def _confidence(phase: str, context_version: str, maintenance: bool) -> str:
        if maintenance:
            return "BAJA"
        base = {
            "OPERACION_ESTABLE": 3,
            "REPOSO": 2,
            "ARRANQUE": 2,
            "DESACELERACION": 2,
            "POST_PARO": 1,
        }.get(phase, 1)
        if context_version == "ENRIQUECIDO":
            base += 1
        return {1: "BAJA", 2: "MEDIA", 3: "ALTA", 4: "ALTA"}.get(base, "BAJA")

    def _ml_indices(self, data: pd.DataFrame) -> pd.Series:
        result = pd.Series(0.0, index=data.index, dtype=float)
        for phase, bundle in self.models.items():
            idx = data.index[data["fase_operativa"].eq(phase)]
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
        """
        Devuelve:
        - indice de reglas fisicas duras
        - indice informativo de advertencias
        - razones de anomalia (pueden abrir evento)
        - advertencias (NO elevan por si solas indice_anomalia)
        - observaciones contextuales (NO elevan indice)
        """
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

        # Desacoplamiento mecanico-hidraulico: regla fuerte e inmediata.
        if phase in {"ARRANQUE", "OPERACION_ESTABLE"} and speed >= 500 and flow < 1.0:
            hard = max(hard, 96.0)
            reasons.append("velocidad_alta_sin_respuesta_de_flujo")
        if phase in {"ARRANQUE", "OPERACION_ESTABLE"} and speed >= 700 and pressure < 0.5:
            hard = max(hard, 96.0)
            reasons.append("velocidad_alta_sin_respuesta_de_presion")

        falta_presion = self._truthy(row.get("falta_presion"))
        if falta_presion:
            if sec_start is not None and sec_start <= settings.startup_grace_seconds:
                # Normal en el arranque observado: se informa, pero aporta 0 al indice.
                observations.append("falta_presion_durante_gracia_de_arranque")
            elif speed >= 500:
                hard = max(hard, 85.0)
                reasons.append("falta_presion_persistente_con_bomba_en_movimiento")
            else:
                warning = max(warning, 30.0)
                warnings.append("senal_falta_presion_activa")

        bajo_flujo = self._truthy(row.get("bajo_flujo"))
        if bajo_flujo:
            if sec_start is not None and sec_start <= settings.startup_grace_seconds:
                observations.append("bajo_flujo_durante_gracia_de_arranque")
            elif speed >= 500:
                hard = max(hard, 85.0)
                reasons.append("bajo_flujo_persistente_con_bomba_en_movimiento")
            else:
                warning = max(warning, 30.0)
                warnings.append("senal_bajo_flujo_activa")

        # v2.2: presion_alta es el primer umbral de advertencia.
        # Se expone y se mide su persistencia, pero NO sube indice_anomalia.
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

        ml_indices = self._ml_indices(data)
        rows: list[dict[str, Any]] = []
        for row_index, row in data.iterrows():
            phase = str(row["fase_operativa"])
            maintenance = any(self._truthy(row.get(c)) for c in MAINTENANCE_COLUMNS)
            estado_vfd = self._optional_float(row.get("estado_vfd"))
            if estado_vfd == 4:
                maintenance = True

            ml_index = float(ml_indices.loc[row_index])
            rule_index, warning_index, reasons, warnings, observations = self._context_rules(row)

            # v2.2: las advertencias NO elevan el indice final.
            # El indice instantaneo representa rareza ML o una regla fisica real.
            index = max(ml_index, rule_index)

            if ml_index >= 70:
                reasons.append("patron_multivariable_atipico_para_la_fase")
            if maintenance:
                observations.append("equipo_en_mantenimiento")

            context_version = str(row.get("contexto_version", "LEGACY"))
            rows.append(
                {
                    **row.to_dict(),
                    "indice_anomalia": round(float(index), 2),
                    "nivel_anomalia": self._severity(index),
                    "confianza_evaluacion": self._confidence(
                        phase, context_version, maintenance
                    ),
                    "razones": sorted(set(reasons)),
                    "advertencias": sorted(set(warnings)),
                    "observaciones": sorted(set(observations)),
                    "indice_ml": round(float(ml_index), 2),
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
            elif isinstance(value, (np.integer,)):
                clean[key] = int(value)
            elif isinstance(value, (np.floating,)):
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
            "conteo_por_contexto": {},
        }

    levels = Counter(scored["nivel_anomalia"].astype(str))
    phases = Counter(scored["fase_operativa"].astype(str))
    contexts = Counter(scored["contexto_version"].astype(str))
    latest = scored.iloc[-1]
    return {
        "total_registros": int(len(scored)),
        "desde": scored["fecha_hora"].iloc[0].isoformat(),
        "hasta": scored["fecha_hora"].iloc[-1].isoformat(),
        "indice_actual": round(float(latest["indice_anomalia"]), 2),
        "indice_maximo": round(float(scored["indice_anomalia"].max()), 2),
        "indice_promedio": round(float(scored["indice_anomalia"].mean()), 2),
        # Conteos exactos para evitar ambiguedad con el acumulado >=70.
        "registros_indice_alto": int(levels.get("ALTO", 0)),
        "registros_indice_muy_alto": int(levels.get("MUY_ALTO", 0)),
        "registros_indice_alto_o_superior": int(
            levels.get("ALTO", 0) + levels.get("MUY_ALTO", 0)
        ),
        "conteo_por_nivel": dict(levels),
        "conteo_por_fase": dict(phases),
        "conteo_por_contexto": dict(contexts),
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
    """
    Construye eventos v2.2 usando persistencia e histeresis.

    - Un pico estadistico aislado no abre evento.
    - Se abre por >= EVENT_MIN_CONSECUTIVE muestras consecutivas o por
      EVENT_MIN_DURATION_SECONDS sobre el umbral.
    - Una regla fisica fuerte abre de inmediato.
    - Se cierra cuando el indice se mantiene por debajo de EVENT_CLOSE_THRESHOLD
      durante EVENT_CLOSE_SECONDS.
    """
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

    def _flush_active():
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

        if previous_ts is not None and (ts - previous_ts).total_seconds() > max_gap:
            # Un hueco grande rompe continuidad temporal.
            if active:
                _flush_active()
            pending = []
            recovery = []

        if not active:
            if immediate:
                active = [row]
                active_open_reason = "regla_fisica_inmediata"
                pending = []
            elif index >= open_threshold:
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
            if immediate or index >= close_threshold:
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
                    # No incluye la cola de recuperacion en las metricas del evento.
                    _flush_active()
                    pending = []

        previous_ts = ts

    if active:
        _flush_active()

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
        phase_counts = Counter(frame["fase_operativa"].astype(str))
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
                "razones": [item for item, _ in reason_counts.most_common(8)],
                "advertencias": [item for item, _ in warning_counts.most_common(8)],
            }
        )
    return events


def _cycle_signature_score(
    value: float | None,
    median: float | None,
    mad: float | None,
) -> tuple[float, float | None]:
    if value is None or median is None or mad is None or not np.isfinite(mad) or mad <= 1e-12:
        return 0.0, None
    robust_z = abs(value - median) / (1.4826 * mad)
    z0 = float(settings.cycle_signature_z_start)
    z1 = max(z0 + 0.1, float(settings.cycle_signature_z_high))
    if robust_z <= z0:
        score = 0.0
    elif robust_z >= z1:
        score = 100.0
    else:
        score = 40.0 + (robust_z - z0) / (z1 - z0) * 50.0
    return float(np.clip(score, 0.0, 100.0)), float(robust_z)


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
        reason_counts = Counter(
            reason for reasons in frame.get("razones", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )
        warning_counts = Counter(
            reason for reasons in frame.get("advertencias", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )
        observation_counts = Counter(
            reason for reasons in frame.get("observaciones", pd.Series([[]] * len(frame))) for reason in (reasons or [])
        )

        def first_num(series_name: str):
            if series_name not in moving.columns:
                return None
            s = pd.to_numeric(moving[series_name], errors="coerce").dropna()
            return None if s.empty else float(s.iloc[0])

        def last_num(series_name: str):
            if series_name not in moving.columns:
                return None
            s = pd.to_numeric(moving[series_name], errors="coerce").dropna()
            return None if s.empty else float(s.iloc[-1])

        def med_stable(series_name: str):
            if stable.empty or series_name not in stable.columns:
                return None
            s = pd.to_numeric(stable[series_name], errors="coerce").dropna()
            return None if s.empty else float(s.median())

        context_counts = Counter(frame["contexto_version"].astype(str))
        primary_context = context_counts.most_common(1)[0][0] if context_counts else "LEGACY"
        complete = bool(frame["fase_operativa"].eq("POST_PARO").any())
        max_index = float(frame["indice_anomalia"].max())
        p95_index = float(frame["indice_anomalia"].quantile(0.95))
        max_rule = float(pd.to_numeric(frame["indice_reglas"], errors="coerce").fillna(0).max())

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
                "indice_maximo": round(max_index, 2),  # compatibilidad v2.1
                "indice_p95": round(p95_index, 2),
                "indice_regla_fisica_max": round(max_rule, 2),
                "ciclo_completo": complete,
                "contexto_principal": primary_context,
                "max_duracion_presion_alta_seg": round(
                    float(pd.to_numeric(frame["duracion_presion_alta_seg"], errors="coerce").max()),
                    2,
                ),
                "contexto": dict(context_counts),
                "razones": [item for item, _ in reason_counts.most_common(8)],
                "advertencias": [item for item, _ in warning_counts.most_common(8)],
                "observaciones": [item for item, _ in observation_counts.most_common(8)],
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

    # Baseline robusto por tipo de contexto para no mezclar muestreo LEGACY y ENRIQUECIDO.
    baselines: dict[str, dict[str, tuple[float | None, float | None]]] = {}
    complete_cycles = [c for c in raw_cycles if c["ciclo_completo"]]
    contexts = sorted({c["contexto_principal"] for c in complete_cycles})
    for context in contexts:
        subset = [c for c in complete_cycles if c["contexto_principal"] == context]
        if len(subset) < settings.cycle_signature_min_baseline:
            subset = complete_cycles
        metrics: dict[str, tuple[float | None, float | None]] = {}
        for metric in metric_names:
            values = np.array(
                [float(c[metric]) for c in subset if c.get(metric) is not None],
                dtype=float,
            )
            if values.size < settings.cycle_signature_min_baseline:
                metrics[metric] = (None, None)
                continue
            median = float(np.median(values))
            mad = float(np.median(np.abs(values - median)))
            metrics[metric] = (median, mad)
        baselines[context] = metrics

    cycles: list[dict[str, Any]] = []
    for cycle in raw_cycles:
        if not cycle["ciclo_completo"]:
            cycle.update(
                {
                    "indice_firma_ciclo": None,
                    "indice_ciclo": None,
                    "nivel_ciclo": "EN_CURSO",
                    "nivel_anomalia": "EN_CURSO",
                    "tuvo_pico_atipico": bool(
                        cycle["indice_maximo_instantaneo"] >= settings.cycle_instant_peak_threshold
                    ),
                    "desviaciones_ciclo": [],
                }
            )
            cycles.append(cycle)
            continue

        baseline = baselines.get(cycle["contexto_principal"], {})
        signature_score = 0.0
        deviations: list[dict[str, Any]] = []
        for metric in metric_names:
            median, mad = baseline.get(metric, (None, None))
            score, robust_z = _cycle_signature_score(cycle.get(metric), median, mad)
            signature_score = max(signature_score, score)
            if robust_z is not None and robust_z > settings.cycle_signature_z_start:
                deviations.append(
                    {
                        "variable": metric,
                        "valor": round(float(cycle[metric]), 4),
                        "mediana_baseline": round(float(median), 4),
                        "z_robusto": round(float(robust_z), 2),
                    }
                )

        # El p95 describe el ciclo. Una regla fisica fuerte puede elevarlo de inmediato.
        cycle_index = max(
            float(cycle["indice_p95"]),
            float(cycle["indice_regla_fisica_max"]),
            float(signature_score),
        )
        if signature_score >= 70:
            cycle["razones"] = sorted(set(cycle["razones"] + ["firma_ciclo_fuera_de_baseline"]))

        cycle.update(
            {
                "indice_firma_ciclo": round(float(signature_score), 2),
                "indice_ciclo": round(float(cycle_index), 2),
                "nivel_ciclo": AnomalyEngine._severity(cycle_index),
                "nivel_anomalia": AnomalyEngine._severity(cycle_index),  # compatibilidad
                "tuvo_pico_atipico": bool(
                    cycle["indice_maximo_instantaneo"] >= settings.cycle_instant_peak_threshold
                ),
                "desviaciones_ciclo": deviations,
            }
        )
        cycles.append(cycle)
    return cycles


def variable_coverage(df: pd.DataFrame) -> list[dict[str, Any]]:
    roles: dict[str, tuple[str, str]] = {
        "flujo_instantaneo": ("PREDICTORA", "ML y respuesta hidraulica"),
        "presion_relativa": ("PREDICTORA", "ML y respuesta hidraulica"),
        "temperatura_tanque": ("PREDICTORA", "ML"),
        "nivel_tanque": ("PREDICTORA", "ML y ciclos"),
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
        if name in df.columns:
            non_null = int(df[name].notna().sum())
        else:
            non_null = 0
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

