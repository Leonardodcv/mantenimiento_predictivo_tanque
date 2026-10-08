# Mantenimiento Predictivo del Tanque - Backend v3.2

Backend FastAPI para analisis de anomalias, eventos persistentes y salud de ciclos del banco hidraulico.

La v3.2 mantiene los regimenes, reglas fisicas, tolerancias de ingenieria y explicabilidad de v3.1, pero agrega proteccion contra contaminacion del baseline y separa un nuevo regimen persistente de una referencia normal ya validada.

## Cambios principales

### Baseline protegido

En el primer `model/rebuild` de v3.2 se crea una referencia de ciclos confiables:

```text
artifacts/cycle_baseline_sqlserver_v32.json
```

A partir de ese momento los rebuild normales reutilizan el mismo baseline. Una desviacion persistente no puede convertirse automaticamente en normal solo porque se acumulen mas registros.

Los ciclos se etiquetan como:

```text
CONFIABLE_BASELINE
CUARENTENA_BASELINE
EXCLUIDO_BASELINE
```

`CUARENTENA_BASELINE` significa: el ciclo se conserva, se puntua y se muestra, pero no se utiliza para modificar la referencia normal.

Para regenerar deliberadamente el baseline despues de validar un cambio de operacion:

```text
POST /api/v3/model/rebuild/?source=sqlserver&reset_cycle_baseline=true
```

No uses `reset_cycle_baseline=true` de manera rutinaria.

### Nuevo regimen candidato

Si al menos varios ciclos consecutivos en cuarentena comparten las mismas desviaciones dominantes, v3.2 los agrupa como:

```text
NUEVO_REGIMEN_CANDIDATO
```

Esto sirve para el caso observado el 6-Oct, donde varios ciclos consecutivos presentaron aproximadamente mas flujo, menos presion y menor duracion respecto al baseline anterior.

Consulta:

```text
GET /api/v3/cycles/regime-candidates/?source=sqlserver
```

La clasificacion no declara automaticamente que el regimen sea una falla ni que sea normal. Requiere validacion operacional antes de promoverlo a baseline.

### Explicabilidad fisicamente escalada

La explicacion heuristica sigue sin ser una atribucion causal del Isolation Forest. La diferencia en v3.2 es que las variables con dispersion estadistica casi nula usan un piso de escala fisica.

Ejemplo:

```json
{
  "variable": "voltaje_salida",
  "valor": 164,
  "mediana_regimen": 155,
  "escala_estadistica": 0.000155,
  "piso_escala_fisica": 2.0,
  "escala_explicacion": 2.0,
  "escala_limitada_por_piso_fisico": true,
  "z_robusto_aprox": 4.5
}
```

De esta forma no se producen valores de `z` de decenas de miles solo porque la variable historicamente haya sido casi constante.

### Ciclo parcial por nivel inicial alto

Si el llenado comienza cerca del setpoint superior, termina cerca del setpoint y dura poco, el ciclo se marca:

```text
CICLO_PARCIAL_NIVEL_INICIAL
```

Por defecto no se compara contra la firma de un ciclo completo. Si no existe una regla fisica fuerte:

```text
nivel_ciclo = NO_COMPARABLE
indice_ciclo = null
```

Los eventos internos del ciclo siguen disponibles en `/api/v3/anomalies/events/`.

## Instalacion / actualizacion

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python run.py
```

Resultado esperado:

```text
28 passed
```

Luego ejecuta una sola vez:

```text
POST /api/v3/model/rebuild/?source=sqlserver
```

Si no existe un baseline v3.2, ese rebuild lo crea y lo congela. Los siguientes rebuild lo reutilizan.

## Endpoints principales

```text
GET  /api/v3/health/
GET  /api/v3/model/status/
POST /api/v3/model/rebuild/

GET  /api/v3/anomalies/summary/
GET  /api/v3/anomalies/latest/
GET  /api/v3/anomalies/history/
GET  /api/v3/anomalies/explain/{record_id}
GET  /api/v3/anomalies/events/

GET  /api/v3/cycles/latest/
GET  /api/v3/cycles/baseline/
GET  /api/v3/cycles/regime-candidates/

GET  /api/v3/variables/status/
```

Swagger:

```text
http://127.0.0.1:8002/docs
```

## Configuracion nueva v3.2

```env
CYCLE_BASELINE_MODE=protected_frozen
CYCLE_BASELINE_DIR=artifacts
CYCLE_BASELINE_ACCEPT_MAX_SIGNATURE=39
CYCLE_BASELINE_MAX_ML_FRACTION=0.20
CYCLE_BASELINE_BOOTSTRAP_MIN_CYCLES=20

NEW_REGIME_MIN_CONSECUTIVE_CYCLES=5
NEW_REGIME_SIGNATURE_TOP_METRICS=2
NEW_REGIME_MAX_GAP_MINUTES=15

CYCLE_PARTIAL_START_SETPOINT_MARGIN=5
CYCLE_PARTIAL_END_SETPOINT_MARGIN=3
CYCLE_PARTIAL_MAX_DURATION_SECONDS=60
```

Los pisos de escala de explicabilidad tambien estan documentados en `.env.example`.

## Interpretacion importante

`indice_anomalia` sigue representando rareza/anomalia. No es probabilidad de falla.

Un ciclo puede estar en cuarentena o formar parte de un `NUEVO_REGIMEN_CANDIDATO` sin ser una falla confirmada. La confirmacion de falla se reserva para evidencia fisica/PLC suficientemente fuerte.
