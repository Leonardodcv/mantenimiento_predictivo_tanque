# Backend de Mantenimiento Predictivo v2

Segunda version del backend para el proyecto del tanque/bomba.

La mejora principal de v2 es que **puede entrenarse con todo el historial de `dbo.LecturasTanque`, incluyendo los dias antiguos en los que las columnas agregadas en la v3 de Node-RED tienen `NULL`**.

> El indice `0-100` sigue siendo un indice de anomalia/rareza y contexto. No representa una probabilidad de falla.

## Que cambia frente a v1

- Usa la tabla completa durante el `rebuild` del modelo (`MODEL_MAX_ROWS=0`).
- Acepta historial mixto:
  - `LEGACY`: filas antiguas sin las variables nuevas.
  - `ENRIQUECIDO`: filas nuevas con setpoints, valvulas, mando IA, horas de marcha, etc.
- **No rellena los NULL antiguos de variables nuevas con cero**. Un NULL significa "esta variable no se registro todavia".
- El Isolation Forest usa solamente variables disponibles historicamente para que los dias viejos sigan siendo utiles.
- Las nuevas variables se usan como contexto cuando existen.
- Separa cinco fases:
  - `REPOSO`
  - `ARRANQUE`
  - `OPERACION_ESTABLE`
  - `DESACELERACION`
  - `POST_PARO`
- Detecta ciclos automaticamente a partir del movimiento real de la bomba.
- `presion_alta` se trata como **primer umbral de advertencia**, no como falla directa.
- `falta_presion` y `bajo_flujo` tienen una gracia inicial de arranque para no castigar los primeros segundos normales del ciclo.
- `energia_aparente_l1` se excluye temporalmente del ML porque actualmente esta congelada en el propio PLC.
- `numero_arranques` se conserva como contexto porque corresponde a arranques manuales, no a cada ciclo automatico.
- `start_stop_s1`, `start_stop_s2`, `estado_s1` y `estado_s2` se conservan como contexto de valvulas/ciclo, no como predictores numericos directos en esta version.
- `start_stop_vfd` se conserva pero queda fuera del modelo hasta confirmar su significado operativo.

## Arquitectura v2

```text
SQL Server / archivo
        |
        v
Historial completo
        |
        +--> filas LEGACY -----------+
        |                            |
        +--> filas ENRIQUECIDAS -----+----> preparacion temporal
                                              |
                                              +--> fase operativa
                                              +--> ciclo automatico
                                              +--> persistencia presion_alta
                                              |
                                              v
                                  RobustScaler + IsolationForest
                                  por fase (variables historicas)
                                              |
                              +---------------+---------------+
                              |                               |
                        reglas fisicas                  advertencias
                  falta presion persistente       presion_alta = umbral 1
                  bajo flujo persistente
                  falla VFD
                  velocidad sin flujo/presion
                              |                               |
                              +---------------+---------------+
                                              v
                                  indice de anomalia 0-100
```

## Variables excluidas temporalmente del ML

Estas columnas siguen apareciendo en la API y se guardan en SQL, pero no entran al Isolation Forest v2:

- `energia_aparente_l1`: congelada actualmente en el PLC.
- `numero_arranques`: contador de arranques manuales.
- `start_stop_vfd`: semantica pendiente.
- `start_stop_s1`, `start_stop_s2`: contexto de valvulas.
- `estado_s1`, `estado_s2`: contexto automatico del ciclo.
- `horas_marcha`: uso acumulado; se conserva para futuras comparaciones por desgaste.

## 1. Crear entorno

En PowerShell:

```powershell
cd C:\ruta\backend_mantenimiento_predictivo_v2
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Para SQL Server necesitas tener instalado un ODBC Driver compatible, por ejemplo **ODBC Driver 18 for SQL Server**.

## 2. Configurar

```powershell
Copy-Item .env.example .env
```

La demo arranca con:

```text
DATA_SOURCE=file
```

El archivo `data/lecturas_mixtas_demo.tsv` contiene filas antiguas sin las variables nuevas y filas recientes enriquecidas, para probar precisamente la compatibilidad mixta.

Para trabajar con tu base real:

```text
DATA_SOURCE=sqlserver
SQL_SERVER=10.10.17.13
SQL_PORT=1433
SQL_DATABASE=MantenimientoPredictivo
SQL_TABLE=dbo.LecturasTanque
SQL_USERNAME=TU_USUARIO_SQL
SQL_PASSWORD=TU_PASSWORD_SQL
```

Deja:

```text
MODEL_MAX_ROWS=0
```

para usar **toda la tabla** durante el entrenamiento/rebuild.

No guardes credenciales reales en Git.

## 3. Ejecutar

```powershell
python run.py
```

API:

```text
http://127.0.0.1:8002
```

Swagger:

```text
http://127.0.0.1:8002/docs
```

## 4. Primer paso con SQL Server

Una vez configurado `.env`, ejecuta:

```text
POST /api/v2/model/rebuild/?source=sqlserver
```

Ese endpoint:

1. lee la tabla completa;
2. conserva filas viejas aunque las columnas nuevas sean NULL;
3. clasifica fases y ciclos;
4. entrena un Isolation Forest por fase;
5. puntua todo el historial;
6. guarda el modelo y el resumen en memoria.

El `rebuild` no debe ejecutarse cada segundo. Hazlo cuando quieras incorporar un nuevo bloque significativo de historial o reentrenar la linea base.

Los endpoints `latest` usan el modelo ya construido para puntuar registros recientes.

## Endpoints

### Salud

```text
GET /api/v2/health/
```

### Estado del modelo

```text
GET /api/v2/model/status/?source=sqlserver
```

### Reconstruir con TODO el historial

```text
POST /api/v2/model/rebuild/?source=sqlserver
```

### Resumen del historial usado por el modelo

```text
GET /api/v2/anomalies/summary/?source=sqlserver
```

Incluye conteos `LEGACY` y `ENRIQUECIDO`.

### Ultimas lecturas para React

```text
GET /api/v2/anomalies/latest/?source=sqlserver&limit=500
```

Solo indices altos:

```text
GET /api/v2/anomalies/latest/?source=sqlserver&limit=500&min_index=70
```

### Historial ya analizado

```text
GET /api/v2/anomalies/history/?source=sqlserver&limit=5000
```

Con periodo:

```text
GET /api/v2/anomalies/history/?source=sqlserver&desde=2026-10-02T00:00:00&hasta=2026-10-05T23:59:59
```

### Eventos anormales agrupados

```text
GET /api/v2/anomalies/events/?source=sqlserver&threshold=70
```

### Ciclos detectados

```text
GET /api/v2/cycles/latest/?source=sqlserver&limit=20
```

Cada ciclo devuelve, entre otros:

- duracion de movimiento;
- nivel inicial y final;
- mediana de flujo estable;
- mediana de presion estable;
- mediana de velocidad estable;
- maximo indice de anomalia;
- P95 del indice;
- tiempo maximo continuo de `presion_alta`;
- razones principales.

### Cobertura de variables

```text
GET /api/v2/variables/status/?source=sqlserver
```

Este endpoint es importante para comprobar que las variables nuevas pueden tener una cobertura parcial sin romper el modelo.

## Como se maneja el historial antiguo

Ejemplo conceptual:

```text
2026-10-02  flujo=18  presion=8.2  velocidad=1197  encendido_ia=NULL
2026-10-05  flujo=18  presion=8.6  velocidad=1198  encendido_ia=1
```

Ambas filas participan del modelo estadistico porque flujo, presion, velocidad y las demas variables historicas existen en ambas.

La primera se etiqueta:

```text
contexto_version = LEGACY
```

La segunda:

```text
contexto_version = ENRIQUECIDO
```

No se hace esto:

```text
encendido_ia NULL -> 0   # incorrecto
```

porque `NULL` no significa que la IA estuviera apagada; significa que esa variable todavia no se almacenaba.

## Tratamiento de `presion_alta`

En v2 se considera una advertencia temprana:

```text
presion_alta = 1
    -> indice_advertencias aumenta
    -> razon: primer_umbral_presion_alta_activo
```

Si permanece activa:

```text
umbral_presion_alta_persistente
```

Pero por si sola no eleva el indice final a nivel de falla. El frontend puede mostrar por separado:

```text
indice_anomalia
indice_advertencias
razones
```

## Respuesta de ejemplo

```json
{
  "id": 12345,
  "fecha_hora": "2026-10-05T14:09:39.435383800",
  "indice_anomalia": 18.2,
  "nivel_anomalia": "BAJO",
  "indice_ml": 12.3,
  "indice_reglas": 0.0,
  "indice_advertencias": 20.0,
  "fase_operativa": "ARRANQUE",
  "ciclo_id": 63,
  "contexto_version": "ENRIQUECIDO",
  "razones": [
    "primer_umbral_presion_alta_activo"
  ]
}
```

## Recomendacion de uso

- Usa `POST /model/rebuild/` para entrenar/recalibrar con el historial completo.
- Usa `/anomalies/latest/` desde React para la vista en tiempo casi real.
- Usa `/cycles/latest/` para comparar un ciclo contra los anteriores.
- Cuando los programadores corrijan `energia_aparente_l1`, no la agregues automaticamente al modelo: primero verifica que evolucione correctamente y luego se incorpora en una v2.x/v3.
