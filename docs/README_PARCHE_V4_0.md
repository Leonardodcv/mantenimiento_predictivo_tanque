# Parche v4.0 sobre v3.2

Este parche actualiza el backend de v3.2 a v4.0 sin modificar:

- `.env` real
- SQL Server
- `dbo.LecturasTanque`
- Node-RED
- datos historicos

El baseline protegido v3.2 se migra automaticamente a una referencia v4 la primera vez que se usa.

## Aplicacion

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v4_0.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Luego:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

Reconstruir:

```text
POST /api/v4/model/rebuild/?source=sqlserver
```

No use `reset_cycle_baseline=true` durante la actualizacion normal.
