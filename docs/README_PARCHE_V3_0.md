# Parche v3.0 sobre v2.2.1

Este parche actualiza `mantenimiento_predictivo_tanque` desde v2.2.1 a v3.0 sin tocar el archivo `.env` real ni la base SQL Server.

## Cambios

- Reanudacion despues de paro prolongado.
- Primer ciclo = `REANUDACION`; siguientes ciclos configurables = `ESTABILIZACION`.
- Subestados de reposo: `REPOSO_ESTATICO`, `VACIADO`, `ESPERA_CICLO`, `REPOSO_TRANSITORIO`.
- Tolerancias minimas de ingenieria ademas del z robusto.
- Los estados pasivos no abren eventos solo por rareza estadistica.
- Mantiene reglas fisicas fuertes sin limitar.
- API principal `/api/v3`, con alias `/api/v2` para compatibilidad.
- Nuevo endpoint `/api/v3/cycles/baseline/`.

## Aplicacion

Desde la carpeta donde descomprimiste este parche:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v3_0.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

El script crea un respaldo antes de sustituir archivos.

Despues:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

No se agregaron dependencias nuevas.

## .env

El parche NO reemplaza tu `.env`. Revisa `.env.example` y agrega las variables v3.0 si quieres dejarlas explicitas. Si no las agregas, el codigo usa los valores predeterminados.

Recomendado:

```env
APP_NAME=Mantenimiento Predictivo Tanque API v3.0
RESTART_LONG_STOP_HOURS=2
RESTART_STABILIZATION_CYCLES=4
RESTART_ML_CAP=39
RESTART_CYCLE_SCORE_CAP=39
EVENT_STATISTICAL_ACTIVE_ONLY=true
CYCLE_TOL_FLOW_ABS=0.5
CYCLE_TOL_FLOW_PCT=0.03
CYCLE_TOL_PRESSURE_ABS=0.35
CYCLE_TOL_PRESSURE_PCT=0.04
```

## Despues de reiniciar

El modelo vive en memoria. Debes reconstruirlo:

```text
POST /api/v3/model/rebuild/?source=sqlserver
```

Luego revisa:

```text
GET /api/v3/cycles/latest/?source=sqlserver&limit=20
GET /api/v3/cycles/baseline/?source=sqlserver
GET /api/v3/anomalies/events/?source=sqlserver&threshold=80&scope=recent
GET /api/v3/anomalies/events/?source=sqlserver&threshold=80&scope=full
```
