from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from app.config import settings


class DataSourceError(RuntimeError):
    pass


_TABLE_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?$")


def _validated_table() -> str:
    table = settings.sql_table.strip()
    if not _TABLE_RE.fullmatch(table):
        raise DataSourceError("SQL_TABLE contiene un nombre no valido.")
    return table


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "fecha_hora" not in df.columns:
        raise DataSourceError("La fuente no contiene la columna fecha_hora.")

    out = df.copy()
    out["fecha_hora"] = pd.to_datetime(out["fecha_hora"], errors="coerce")
    out = out.dropna(subset=["fecha_hora"])
    if "id" in out.columns:
        out["id"] = pd.to_numeric(out["id"], errors="coerce")
        out = out.sort_values(["fecha_hora", "id"], kind="stable")
    else:
        out = out.sort_values("fecha_hora", kind="stable")
    return out.reset_index(drop=True)


def read_file(limit: int | None = None) -> pd.DataFrame:
    path: Path = settings.data_file_path
    if not path.exists():
        raise DataSourceError(f"No existe el archivo de datos: {path}")
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise DataSourceError(f"No se pudo leer {path}: {exc}") from exc

    df = normalize_frame(df)
    if limit and limit > 0:
        df = df.tail(int(limit)).reset_index(drop=True)
    return df


def _sql_connection_string() -> str:
    if not settings.sql_username or not settings.sql_password:
        raise DataSourceError(
            "SQL_USERNAME y SQL_PASSWORD deben configurarse localmente para usar SQL Server."
        )
    return (
        f"DRIVER={{{settings.sql_driver}}};"
        f"SERVER={settings.sql_server},{settings.sql_port};"
        f"DATABASE={settings.sql_database};"
        f"UID={settings.sql_username};"
        f"PWD={settings.sql_password};"
        f"Encrypt={settings.sql_encrypt};"
        f"TrustServerCertificate={settings.sql_trust_server_certificate};"
        f"Connection Timeout={settings.sql_connection_timeout};"
    )


def _connect():
    try:
        import pyodbc
    except ImportError as exc:
        raise DataSourceError(
            "pyodbc no esta instalado. Ejecuta: pip install pyodbc"
        ) from exc

    try:
        conn = pyodbc.connect(_sql_connection_string())
        conn.timeout = settings.sql_query_timeout
        return conn
    except Exception as exc:
        raise DataSourceError(f"No se pudo conectar a SQL Server: {exc}") from exc


def read_sqlserver_full(max_rows: int = 0) -> pd.DataFrame:
    """Lee la tabla completa. max_rows=0 significa sin limite."""
    table = _validated_table()
    if max_rows and max_rows > 0:
        # Toma los registros mas recientes, pero los devuelve cronologicamente.
        query = (
            f"SELECT * FROM (SELECT TOP ({int(max_rows)}) * FROM {table} "
            "ORDER BY id DESC) AS q ORDER BY id ASC;"
        )
    else:
        query = f"SELECT * FROM {table} ORDER BY id ASC;"

    try:
        with _connect() as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        raise DataSourceError(f"Error consultando SQL Server: {exc}") from exc
    return normalize_frame(df)


def read_sqlserver_latest(limit: int, context_rows: int = 0) -> pd.DataFrame:
    table = _validated_table()
    take = max(1, int(limit)) + max(0, int(context_rows))
    query = f"SELECT TOP ({take}) * FROM {table} ORDER BY id DESC;"
    try:
        with _connect() as conn:
            df = pd.read_sql_query(query, conn)
    except Exception as exc:
        raise DataSourceError(f"Error consultando SQL Server: {exc}") from exc
    return normalize_frame(df)


def sqlserver_metadata() -> dict:
    table = _validated_table()
    query = (
        f"SELECT COUNT_BIG(*) AS total, MIN(id) AS min_id, MAX(id) AS max_id, "
        f"MIN(fecha_hora) AS desde, MAX(fecha_hora) AS hasta FROM {table};"
    )
    try:
        with _connect() as conn:
            row = pd.read_sql_query(query, conn).iloc[0]
    except Exception as exc:
        raise DataSourceError(f"Error consultando metadatos SQL Server: {exc}") from exc

    def dt(value):
        if pd.isna(value):
            return None
        return pd.Timestamp(value).isoformat()

    return {
        "total": int(row["total"] or 0),
        "min_id": None if pd.isna(row["min_id"]) else int(row["min_id"]),
        "max_id": None if pd.isna(row["max_id"]) else int(row["max_id"]),
        "desde": dt(row["desde"]),
        "hasta": dt(row["hasta"]),
    }


def load_full(source: str) -> pd.DataFrame:
    source = (source or settings.data_source).strip().lower()
    if source == "file":
        return read_file(limit=settings.model_max_rows or None)
    if source == "sqlserver":
        return read_sqlserver_full(max_rows=settings.model_max_rows)
    raise DataSourceError("source debe ser 'file' o 'sqlserver'.")


def load_latest(source: str, limit: int, context_rows: int = 0) -> pd.DataFrame:
    source = (source or settings.data_source).strip().lower()
    if source == "file":
        return read_file(limit=max(1, int(limit)) + max(0, int(context_rows)))
    if source == "sqlserver":
        return read_sqlserver_latest(limit=limit, context_rows=context_rows)
    raise DataSourceError("source debe ser 'file' o 'sqlserver'.")
