# Parche v3.2 sobre v3.1

Este parche implementa cuatro correcciones:

1. baseline de ciclos protegido/congelado;
2. cuarentena y deteccion de `NUEVO_REGIMEN_CANDIDATO`;
3. piso de escala fisica para la explicabilidad del ML;
4. `CICLO_PARCIAL_NIVEL_INICIAL` para arranques cerca del setpoint superior.

No modifica tu `.env` real, SQL Server ni Node-RED.

## Aplicacion

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v3_2.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Luego:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

Resultado esperado:

```text
28 passed
```

Finalmente:

```text
POST /api/v3/model/rebuild/?source=sqlserver
```

El primer rebuild v3.2 crea un archivo protegido dentro de `artifacts/`. Los rebuild siguientes lo reutilizan.

## Importante

No uses esto salvo que hayas validado operacionalmente que deseas redefinir lo que significa "normal":

```text
POST /api/v3/model/rebuild/?source=sqlserver&reset_cycle_baseline=true
```

Ese parametro reemplaza deliberadamente la referencia protegida.
