from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Mantenimiento Predictivo Tanque API v4.0")
    app_env: str = os.getenv("APP_ENV", "development")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = _env_int("APP_PORT", 8002)
    cors_allowed_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5174,http://127.0.0.1:5174",
        ).split(",")
        if item.strip()
    )

    data_source: str = os.getenv("DATA_SOURCE", "file").strip().lower()
    data_file: str = os.getenv("DATA_FILE", "data/lecturas_mixtas_demo.tsv")

    # 0 = sin limite. El rebuild usa todo el historial por defecto.
    model_max_rows: int = _env_int("MODEL_MAX_ROWS", 0)
    latest_context_rows: int = _env_int("LATEST_CONTEXT_ROWS", 2000)
    default_limit: int = _env_int("ANOMALY_DEFAULT_LIMIT", 500)
    cycle_default_limit: int = _env_int("CYCLE_DEFAULT_LIMIT", 20)

    # Fases principales.
    movement_speed_threshold: float = _env_float("MOVEMENT_SPEED_THRESHOLD", 20.0)
    stable_speed_threshold: float = _env_float("STABLE_SPEED_THRESHOLD", 1100.0)
    post_stop_seconds: float = _env_float("POST_STOP_SECONDS", 10.0)
    startup_grace_seconds: float = _env_float("STARTUP_GRACE_SECONDS", 5.0)

    # v3.0: subestados de reposo derivados principalmente por pendiente de nivel.
    rest_drain_slope_threshold: float = _env_float("REST_DRAIN_SLOPE_THRESHOLD", -0.04)
    rest_static_slope_threshold: float = _env_float("REST_STATIC_SLOPE_THRESHOLD", 0.02)
    rest_wait_setpoint_margin: float = _env_float("REST_WAIT_SETPOINT_MARGIN", 2.5)
    rest_max_derivative_gap_seconds: float = _env_float(
        "REST_MAX_DERIVATIVE_GAP_SECONDS", 30.0
    )

    # Presion_alta sigue siendo advertencia contextual.
    pressure_warning_persistent_seconds: float = _env_float(
        "PRESSURE_WARNING_PERSISTENT_SECONDS", 15.0
    )

    # Eventos persistentes + histeresis.
    event_open_threshold: float = _env_float("EVENT_OPEN_THRESHOLD", 80.0)
    event_min_consecutive: int = _env_int("EVENT_MIN_CONSECUTIVE", 3)
    event_min_duration_seconds: float = _env_float("EVENT_MIN_DURATION_SECONDS", 5.0)
    event_close_threshold: float = _env_float("EVENT_CLOSE_THRESHOLD", 60.0)
    event_close_seconds: float = _env_float("EVENT_CLOSE_SECONDS", 5.0)
    event_immediate_rule_threshold: float = _env_float(
        "EVENT_IMMEDIATE_RULE_THRESHOLD", 85.0
    )
    event_max_gap_seconds: float = _env_float("EVENT_MAX_GAP_SECONDS", 12.0)
    # Evita abrir eventos estadisticos solo por rareza durante reposo/vaciado.
    event_statistical_active_only: bool = _env_bool("EVENT_STATISTICAL_ACTIVE_ONLY", True)

    # v3.0: reanudacion y estabilizacion tras paro prolongado.
    restart_long_stop_hours: float = _env_float("RESTART_LONG_STOP_HOURS", 2.0)
    restart_stabilization_cycles: int = _env_int("RESTART_STABILIZATION_CYCLES", 4)
    restart_ml_cap: float = _env_float("RESTART_ML_CAP", 39.0)
    restart_cycle_score_cap: float = _env_float("RESTART_CYCLE_SCORE_CAP", 39.0)

    # Firma de ciclo: estadistica robusta + tolerancia minima de ingenieria.
    cycle_signature_min_baseline: int = _env_int("CYCLE_SIGNATURE_MIN_BASELINE", 12)
    cycle_signature_z_start: float = _env_float("CYCLE_SIGNATURE_Z_START", 2.5)
    cycle_signature_z_high: float = _env_float("CYCLE_SIGNATURE_Z_HIGH", 6.0)
    cycle_signature_engineering_high_ratio: float = _env_float(
        "CYCLE_SIGNATURE_ENGINEERING_HIGH_RATIO", 3.0
    )
    cycle_instant_peak_threshold: float = _env_float("CYCLE_INSTANT_PEAK_THRESHOLD", 90.0)

    # v3.1/v3.2: explicabilidad heuristica del ML por regimen.
    explain_min_index: float = _env_float("EXPLAIN_MIN_INDEX", 70.0)
    explain_top_features: int = _env_int("EXPLAIN_TOP_FEATURES", 8)
    explain_min_robust_z: float = _env_float("EXPLAIN_MIN_ROBUST_Z", 1.5)
    explain_z_full_scale: float = _env_float("EXPLAIN_Z_FULL_SCALE", 6.0)

    # v3.2: piso fisico de escala para evitar z robustos absurdos cuando MAD ~= 0.
    explain_min_scale_flow: float = _env_float("EXPLAIN_MIN_SCALE_FLOW", 0.20)
    explain_min_scale_pressure: float = _env_float("EXPLAIN_MIN_SCALE_PRESSURE", 0.10)
    explain_min_scale_level: float = _env_float("EXPLAIN_MIN_SCALE_LEVEL", 1.0)
    explain_min_scale_frequency: float = _env_float("EXPLAIN_MIN_SCALE_FREQUENCY", 1.0)
    explain_min_scale_speed: float = _env_float("EXPLAIN_MIN_SCALE_SPEED", 10.0)
    explain_min_scale_output_voltage: float = _env_float("EXPLAIN_MIN_SCALE_OUTPUT_VOLTAGE", 2.0)
    explain_min_scale_dc_bus: float = _env_float("EXPLAIN_MIN_SCALE_DC_BUS", 5.0)
    explain_min_scale_l1_voltage: float = _env_float("EXPLAIN_MIN_SCALE_L1_VOLTAGE", 5.0)
    explain_min_scale_l1_current: float = _env_float("EXPLAIN_MIN_SCALE_L1_CURRENT", 0.005)
    explain_min_scale_active_power: float = _env_float("EXPLAIN_MIN_SCALE_ACTIVE_POWER", 1.0)
    explain_min_scale_apparent_power: float = _env_float("EXPLAIN_MIN_SCALE_APPARENT_POWER", 1.0)

    # v3.1: el estado global del ciclo no se hereda automaticamente de un
    # evento corto. El evento se conserva por separado.
    cycle_ml_integral_threshold: float = _env_float("CYCLE_ML_INTEGRAL_THRESHOLD", 70.0)
    cycle_ml_persistence_fraction: float = _env_float("CYCLE_ML_PERSISTENCE_FRACTION", 0.25)
    cycle_ml_persistence_seconds: float = _env_float("CYCLE_ML_PERSISTENCE_SECONDS", 30.0)
    cycle_transient_ml_cap: float = _env_float("CYCLE_TRANSIENT_ML_CAP", 39.0)

    cycle_tol_duration_abs_seconds: float = _env_float(
        "CYCLE_TOL_DURATION_ABS_SECONDS", 6.0
    )
    cycle_tol_duration_pct: float = _env_float("CYCLE_TOL_DURATION_PCT", 0.08)
    cycle_tol_level_start_abs: float = _env_float("CYCLE_TOL_LEVEL_START_ABS", 2.5)
    cycle_tol_level_start_pct: float = _env_float("CYCLE_TOL_LEVEL_START_PCT", 0.04)
    cycle_tol_level_end_abs: float = _env_float("CYCLE_TOL_LEVEL_END_ABS", 2.5)
    cycle_tol_level_end_pct: float = _env_float("CYCLE_TOL_LEVEL_END_PCT", 0.04)
    cycle_tol_flow_abs: float = _env_float("CYCLE_TOL_FLOW_ABS", 0.5)
    cycle_tol_flow_pct: float = _env_float("CYCLE_TOL_FLOW_PCT", 0.03)
    cycle_tol_pressure_abs: float = _env_float("CYCLE_TOL_PRESSURE_ABS", 0.35)
    cycle_tol_pressure_pct: float = _env_float("CYCLE_TOL_PRESSURE_PCT", 0.04)
    cycle_tol_speed_abs: float = _env_float("CYCLE_TOL_SPEED_ABS", 25.0)
    cycle_tol_speed_pct: float = _env_float("CYCLE_TOL_SPEED_PCT", 0.02)

    # v3.2: baseline de ciclo protegido y cuarentena de regimenes nuevos.
    cycle_baseline_mode: str = os.getenv("CYCLE_BASELINE_MODE", "protected_frozen").strip().lower()
    cycle_baseline_dir: str = os.getenv("CYCLE_BASELINE_DIR", "artifacts")
    cycle_baseline_accept_max_signature: float = _env_float(
        "CYCLE_BASELINE_ACCEPT_MAX_SIGNATURE", 39.0
    )
    cycle_baseline_max_ml_fraction: float = _env_float(
        "CYCLE_BASELINE_MAX_ML_FRACTION", 0.20
    )
    cycle_baseline_bootstrap_min_cycles: int = _env_int(
        "CYCLE_BASELINE_BOOTSTRAP_MIN_CYCLES", 20
    )

    new_regime_min_consecutive_cycles: int = _env_int(
        "NEW_REGIME_MIN_CONSECUTIVE_CYCLES", 5
    )
    new_regime_signature_top_metrics: int = _env_int(
        "NEW_REGIME_SIGNATURE_TOP_METRICS", 2
    )
    new_regime_max_gap_minutes: float = _env_float("NEW_REGIME_MAX_GAP_MINUTES", 15.0)

    # v3.2: ciclo parcial por arranque cerca del setpoint superior.
    cycle_partial_start_setpoint_margin: float = _env_float(
        "CYCLE_PARTIAL_START_SETPOINT_MARGIN", 5.0
    )
    cycle_partial_end_setpoint_margin: float = _env_float(
        "CYCLE_PARTIAL_END_SETPOINT_MARGIN", 3.0
    )
    cycle_partial_max_duration_seconds: float = _env_float(
        "CYCLE_PARTIAL_MAX_DURATION_SECONDS", 60.0
    )


    # v4.0: contexto fisico y entrenamiento ML protegido.
    # La velocidad del VFD se conserva como RAW hasta confirmar su escalado en TIA/Modbus.
    velocity_rpm_confirmed: bool = _env_bool("VELOCITY_RPM_CONFIRMED", False)
    flow_unit_l_min_confirmed: bool = _env_bool("FLOW_UNIT_L_MIN_CONFIRMED", False)
    pressure_unit_psi_confirmed: bool = _env_bool("PRESSURE_UNIT_PSI_CONFIRMED", False)
    pump_physics_context_enabled: bool = _env_bool("PUMP_PHYSICS_CONTEXT_ENABLED", True)
    pump_physics_use_for_rules: bool = _env_bool("PUMP_PHYSICS_USE_FOR_RULES", False)

    # Segunda pasada de entrenamiento: las fases activas aprenden solo de ciclos
    # que el baseline protegido considero confiables. Evita que un regimen nuevo
    # persistente termine normalizandose en Isolation Forest tras varios rebuilds.
    ml_protected_active_training: bool = _env_bool("ML_PROTECTED_ACTIVE_TRAINING", True)

    # v4.0: familia de regimenes. Episodios separados con la misma firma se agrupan
    # bajo una misma familia (por ejemplo flujo alto + presion baja).
    regime_family_enabled: bool = _env_bool("REGIME_FAMILY_ENABLED", True)

    # SQL Server.
    sql_driver: str = os.getenv("SQL_DRIVER", "ODBC Driver 17 for SQL Server")
    sql_server: str = os.getenv("SQL_SERVER", r"USER4710-PC\SQLEXPRESS").strip()
    sql_port: str = os.getenv("SQL_PORT", "").strip()
    sql_database: str = os.getenv("SQL_DATABASE", "MantenimientoPredictivo")
    sql_table: str = os.getenv("SQL_TABLE", "dbo.LecturasTanque")
    sql_trusted_connection: bool = _env_bool("SQL_TRUSTED_CONNECTION", False)
    sql_username: str = os.getenv("SQL_USERNAME", "")
    sql_password: str = os.getenv("SQL_PASSWORD", "")
    sql_encrypt: str = os.getenv("SQL_ENCRYPT", "no")
    sql_trust_server_certificate: str = os.getenv("SQL_TRUST_SERVER_CERTIFICATE", "yes")
    sql_connection_timeout: int = _env_int("SQL_CONNECTION_TIMEOUT", 8)
    sql_query_timeout: int = _env_int("SQL_QUERY_TIMEOUT", 60)

    @property
    def data_file_path(self) -> Path:
        p = Path(self.data_file)
        return p if p.is_absolute() else BASE_DIR / p

    @property
    def cycle_baseline_dir_path(self) -> Path:
        p = Path(self.cycle_baseline_dir)
        return p if p.is_absolute() else BASE_DIR / p


settings = Settings()
