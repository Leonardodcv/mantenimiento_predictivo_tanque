# Parche v2.2.1

Este paquete actualiza un proyecto `mantenimiento_predictivo_tanque` que ya se encuentre en v2.2.

## Por que existe este parche

La maquina estuvo detenida durante el intervalo nocturno. Por eso era correcto que una consulta limitada a la ventana SQL reciente no encontrara ciclos nuevos. Sin embargo, el endpoint llamado `cycles/latest` debe ser util para consultar los ultimos ciclos que realmente existieron, aunque hayan ocurrido horas antes.

La v2.2.1 separa esas dos necesidades:

- Los ciclos `latest` salen del snapshot historico del ultimo rebuild.
- Los eventos permiten elegir entre ventana reciente e historial completo.

## Archivos que reemplaza

```text
app/main.py
app/services/model_manager.py
```

Tambien agrega:

```text
tests/test_patch_v221.py
CHANGELOG_V2_2_1.md
```

El script de instalacion hace una copia de respaldo antes de reemplazar archivos.

## Aplicar el parche

Deten primero el backend con `Ctrl+C`.

Desde PowerShell, dentro de la carpeta donde descomprimiste este parche:

```powershell
powershell -ExecutionPolicy Bypass -File ".\aplicar_parche_v2_2_1.ps1" -ProjectPath "C:\ruta\mantenimiento_predictivo_tanque"
```

Luego:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pytest -q
```

El resultado esperado es:

```text
14 passed
```

Inicia de nuevo:

```powershell
python run.py
```

Como el snapshot vive en memoria, reconstruye el modelo despues del reinicio:

```text
POST /api/v2/model/rebuild/?source=sqlserver
```

## Pruebas recomendadas

### 1. Version

```text
GET /api/v2/health/
```

Debe indicar:

```json
{
  "api_version": "2.2.1",
  "model_version": "v2.2-mixed-history-cycle-aware-persistent-events"
}
```

Es intencional que `model_version` siga diciendo v2.2: el parche no cambia el algoritmo.

Si tu `.env` aun tiene un nombre anterior, puedes cambiar solamente:

```env
APP_NAME=Mantenimiento Predictivo Tanque API v2.2.1
```

### 2. Ultimos ciclos existentes

```text
GET /api/v2/cycles/latest/?source=sqlserver&limit=20
```

Ahora debe consultar los ciclos almacenados en el snapshot completo del rebuild. Si el rebuild reporta 260 ciclos, el endpoint puede devolver los ultimos 20 aunque la maquina no haya funcionado durante la noche.

La respuesta incluye:

```text
scope = snapshot_full_history
snapshot_built_at
cycles_detected_snapshot
```

### 3. Eventos recientes

```text
GET /api/v2/anomalies/events/?source=sqlserver&threshold=80&scope=recent
```

Si la maquina estuvo detenida durante la noche, `count=0` puede ser totalmente correcto.

### 4. Eventos del historial completo

```text
GET /api/v2/anomalies/events/?source=sqlserver&threshold=80&scope=full
```

Este modo usa todos los registros ya puntuados en el snapshot del ultimo rebuild y sirve para validar pruebas/anomalias historicas.

## Lo que NO cambia

- Isolation Forest por fase.
- Reglas fisicas.
- Politica de persistencia/histeresis de eventos.
- Score de ciclo de v2.2.
- `presion_alta` continua siendo una advertencia y no eleva por si sola el indice final.
- Compatibilidad LEGACY + ENRIQUECIDO.
- Conexion SQL Server.
- Tabla `dbo.LecturasTanque`.
- Flujo Node-RED.
