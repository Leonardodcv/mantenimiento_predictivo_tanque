# Parche v5.0 sobre backend v4.0

Este parche actualiza una instalacion v4.0 a v5.0 sin tocar `.env`, `artifacts/`, SQL Server ni Node-RED.

## Aplicacion automatica

Desde PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v5_0.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

El script crea una carpeta de respaldo dentro del proyecto y copia los archivos v5.0.

Despues:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

Reconstruccion recomendada:

```text
POST /api/v5/model/rebuild/?source=sqlserver
```

No uses `reset_cycle_baseline=true` en una actualizacion normal. v5 migra el baseline v4 (`cycle_baseline_*_v40.json`) a v5 (`cycle_baseline_*_v50.json`).

## .env

El parche no reemplaza tu `.env`. Las nuevas variables son opcionales y tienen valores seguros por defecto. Puedes agregar:

```env
CONTROLLED_TRIALS_ENABLED=true
CONTROLLED_TRIALS_FILE=data/controlled_trials_2026-10-08.json
CONTROLLED_TRIALS_EXCLUDE_SESSION_FROM_TRAINING=true
CONTROLLED_TRIALS_EXCLUDE_SESSION_FROM_BASELINE=true
CONTROLLED_TRIALS_DETECTION_THRESHOLD=70

PRESSURE_SENSOR_PUMP_COLUMN=
PRESSURE_SENSOR_TANK_COLUMN=
```

Deja vacias las dos columnas de presion hasta confirmar el mapeo fisico de ambos canales en Node-RED/SQL.
