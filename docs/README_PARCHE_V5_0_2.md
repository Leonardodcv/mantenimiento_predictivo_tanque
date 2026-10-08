# Parche v5.0.2 - timeline de indice de anomalia

Agrega un endpoint que recibe una cantidad de horas y devuelve un punto por cada lectura registrada en esa ventana temporal:

```text
GET /api/v5/anomalies/timeline/?source=sqlserver&hours=6
```

La ventana termina en el ultimo `fecha_hora` disponible. Cada punto contiene `momento_comparacion` e `indice_anomalia`, mas `id` y `nivel_anomalia` cuando estan disponibles.

La ventana no reentrena el modelo ni modifica baseline, ground truth, SQL Server o Node-RED. El backend usa registros previos solo como contexto de calculo y no los devuelve.

Aplicacion:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v5_0_2.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Despues:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python run.py
```

No es necesario usar `reset_cycle_baseline=true` ni ejecutar un rebuild exclusivamente por este parche. El modelo existente se utiliza para puntuar la ventana.
