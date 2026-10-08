# Parche v5.0.1 sobre v5.0

Parche pequeno para corregir el mapeo de sensores de presion de la demo actual.

## Aplicar

Desde PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v5_0_1.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Luego:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

No es necesario modificar SQL Server ni Node-RED. El parche no modifica `.env` ni `artifacts`.

## Resultado esperado

`GET /api/v5/context/equipment/` debe indicar:

- sensor superior: disponible, columna `presion_relativa`;
- sensor cercano a bomba: no disponible, PLC secundario sin transmision;
- presion diferencial: no disponible.

Las pruebas controladas siguen excluidas del entrenamiento normal y del baseline.
