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


@dataclass(frozen=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Mantenimiento Predictivo API v2")
    app_env: str = os.getenv("APP_ENV", "development")
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = _env_int("APP_PORT", 8002)
    cors_allowed_origins: tuple[str, ...] = tuple(
        item.strip()
        for item in os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if item.strip()
    )

    # file = demo local, sqlserver = tabla completa.
    data_source: str = os.getenv("DATA_SOURCE", "file").strip().lower()
    data_file: str = os.getenv("DATA_FILE", "data/lecturas_mixtas_demo.tsv")

    # 0 = sin limite. Para el rebuild se consulta la base completa por defecto.
    model_max_rows: int = _env_int("MODEL_MAX_ROWS", 0)
    latest_context_rows: int = _env_int("LATEST_CONTEXT_ROWS", 2000)
    default_limit: int = _env_int("ANOMALY_DEFAULT_LIMIT", 500)
    cycle_default_limit: int = _env_int("CYCLE_DEFAULT_LIMIT", 20)

    # Parametros del motor v2.
    movement_speed_threshold: float = _env_float("MOVEMENT_SPEED_THRESHOLD", 20.0)
    stable_speed_threshold: float = _env_float("STABLE_SPEED_THRESHOLD", 1100.0)
    post_stop_seconds: float = _env_float("POST_STOP_SECONDS", 10.0)
    startup_grace_seconds: float = _env_float("STARTUP_GRACE_SECONDS", 5.0)
    pressure_warning_persistent_seconds: float = _env_float(
        "PRESSURE_WARNING_PERSISTENT_SECONDS", 15.0
    )

    # SQL Server.
    sql_driver: str = os.getenv("SQL_DRIVER", "ODBC Driver 18 for SQL Server")
    sql_server: str = os.getenv("SQL_SERVER", "10.10.17.13")
    sql_port: int = _env_int("SQL_PORT", 1433)
    sql_database: str = os.getenv("SQL_DATABASE", "MantenimientoPredictivo")
    sql_table: str = os.getenv("SQL_TABLE", "dbo.LecturasTanque")
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


settings = Settings()
