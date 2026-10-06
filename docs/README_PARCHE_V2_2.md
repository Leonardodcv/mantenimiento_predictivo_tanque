# Parche v2.2 para mantenimiento_predictivo_tanque v2.1

Este parche actualiza el backend v2.1 a v2.2 sin tocar la base SQL ni Node-RED.

## Archivos reemplazados

- `app/config.py`
- `app/main.py`
- `app/services/anomaly_engine.py`
- `app/services/model_manager.py`
- `.env.example`

## Archivo agregado

- `tests/test_engine_v22.py`
- `CHANGELOG_V2_2.md`

El script `aplicar_parche_v2_2.ps1` crea una copia de seguridad antes de copiar los archivos.
No modifica tu `.env` real.

## Aplicar desde PowerShell

Descomprime este parche y ejecuta:

```powershell
powershell -ExecutionPolicy Bypass -File .\aplicar_parche_v2_2.ps1 -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Luego:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

No hay requerimientos Python nuevos.

## Recomendacion para `.env`

La v2.2 funciona con valores por defecto aunque tu `.env` v2.1 no tenga las nuevas variables. Para hacer la configuracion explicita, agrega:

```env
EVENT_OPEN_THRESHOLD=80
EVENT_MIN_CONSECUTIVE=3
EVENT_MIN_DURATION_SECONDS=5
EVENT_CLOSE_THRESHOLD=60
EVENT_CLOSE_SECONDS=5
EVENT_IMMEDIATE_RULE_THRESHOLD=85
EVENT_MAX_GAP_SECONDS=12
CYCLE_SIGNATURE_MIN_BASELINE=12
CYCLE_SIGNATURE_Z_START=2.5
CYCLE_SIGNATURE_Z_HIGH=4.5
CYCLE_INSTANT_PEAK_THRESHOLD=90
```

Opcionalmente cambia:

```env
APP_NAME=Mantenimiento Predictivo Tanque API v2.2
```

## Despues de reiniciar

Ejecuta:

```text
POST /api/v2/model/rebuild/?source=sqlserver
```

Y valida:

```text
GET /api/v2/health/
GET /api/v2/model/status/?source=sqlserver
GET /api/v2/anomalies/events/?source=sqlserver&threshold=80
GET /api/v2/cycles/latest/?source=sqlserver&limit=20
GET /api/v2/variables/status/?source=sqlserver
```
