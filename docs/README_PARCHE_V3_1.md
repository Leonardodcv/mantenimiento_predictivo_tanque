# Parche v3.1 sobre v3.0

Este parche actualiza el backend v3.0 a v3.1.

## Que modifica

- `app/config.py`
- `app/main.py`
- `app/services/model_manager.py`
- agrega `app/services/anomaly_engine_v31.py`
- agrega `tests/test_engine_v31.py`
- actualiza `.env.example` solo como referencia
- agrega documentacion v3.1

No reemplaza tu `.env` real, no modifica SQL Server y no modifica Node-RED.

## Instalacion

Desde la carpeta donde descomprimiste este parche:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v3_1.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

El script crea un respaldo antes de copiar archivos.

Despues:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

La entrega validada produce `24 passed`.

Reconstruye el modelo:

```text
POST /api/v3/model/rebuild/?source=sqlserver
```

Pruebas recomendadas:

```text
GET /api/v3/anomalies/events/?source=sqlserver&threshold=80&scope=recent
GET /api/v3/cycles/latest/?source=sqlserver&limit=20
GET /api/v3/anomalies/explain/84485?source=sqlserver
```

El ultimo endpoint usa `84485` solo como ejemplo del registro analizado durante la validacion del 6-Oct; puedes sustituirlo por cualquier `id` presente en el snapshot.

## .env

Tu `.env` actual puede seguir funcionando sin cambios porque las opciones nuevas tienen valores predeterminados.

Opcionalmente agrega:

```env
APP_NAME=Mantenimiento Predictivo Tanque API v3.1

EXPLAIN_MIN_INDEX=70
EXPLAIN_TOP_FEATURES=8
EXPLAIN_MIN_ROBUST_Z=1.5
EXPLAIN_Z_FULL_SCALE=6

CYCLE_ML_INTEGRAL_THRESHOLD=70
CYCLE_ML_PERSISTENCE_FRACTION=0.25
CYCLE_ML_PERSISTENCE_SECONDS=30
CYCLE_TRANSIENT_ML_CAP=39
```

## Cambio de rutas

v3.1 mantiene `/api/v3/...` y elimina los alias `/api/v2/...`, ya que el proyecto aun no tiene un frontend que dependa de esos endpoints.
