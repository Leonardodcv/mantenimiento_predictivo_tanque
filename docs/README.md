# Mantenimiento Predictivo del Tanque - Backend v5.0.2

Backend FastAPI para deteccion de anomalias, analisis de ciclos y validacion de pruebas controladas del banco hidraulico.

La v5.0.2 conserva todo lo desarrollado en v5.0.1 y agrega una consulta temporal para obtener el indice de anomalia de cada lectura de las ultimas N horas. La capa formal de **ground truth** sigue basada en la bitacora de pruebas del 8 de octubre de 2026. Esas pruebas son maniobras deliberadas: se usan para validar el detector, pero **no se consideran fallas reales** y no pueden contaminar el entrenamiento normal ni el baseline protegido.

## Principios

- `indice_anomalia` = rareza/anomalia, no probabilidad de falla.
- Las pruebas controladas no alteran el score del modelo.
- Toda la sesion 08:10-09:28 se excluye de entrenamiento normal y del baseline porque las maniobras fueron continuas.
- No se inventan segundos, porcentajes de apertura ni recuperaciones no registradas.
- 08:58-09:00 es manipulacion deliberada del sensor ultrasonico y se trata como calidad de dato, no falla de maquina.
- `presion_alta` sigue siendo una advertencia contextual, no una falla directa.
- `velocidad` sigue siendo RAW del VFD hasta confirmar su escalado.
- La referencia teorica de la bomba Pedrollo PK60 sigue siendo diagnostica, no causal.
- CMS/SM1281/vibraciones siguen fuera de este proyecto.

## Ground truth de pruebas controladas

Catalogo incluido:

```text
data/controlled_trials_2026-10-08.json
```

Contiene la sesion:

```text
CT-20261008-AM
2026-10-08 08:10 -> 09:28
precision temporal: MINUTO
```

Cada registro del backend puede exponer:

```text
es_sesion_pruebas_controladas
sesion_prueba_controlada_id
ground_truth_disponible
ground_truth_ids
ground_truth_tipos
ground_truth_categorias
ground_truth_precision
ground_truth_falla_real
ground_truth_manipulacion_sensor
excluir_entrenamiento_normal
excluir_baseline_normal
```

Las etiquetas solo aparecen cuando el timestamp cae dentro de una ventana anotada. La exclusion de entrenamiento/baseline se aplica a toda la sesion.

## Serie temporal del indice de anomalia

Para graficar la evolucion del score por hora/ventana:

```text
GET /api/v5/anomalies/timeline/?source=sqlserver&hours=6
```

La ventana termina en el ultimo `fecha_hora` disponible, no en la hora del servidor. El backend usa registros anteriores como contexto para calcular correctamente fases y transiciones, pero la respuesta contiene unicamente las lecturas incluidas en las horas solicitadas.

Respuesta simplificada:

```json
{
  "horas_solicitadas": 6.0,
  "ventana": {
    "desde": "2026-10-08T04:00:00",
    "hasta": "2026-10-08T10:00:00",
    "referencia_hasta": "ULTIMO_REGISTRO_DISPONIBLE"
  },
  "base_comparacion": "MODELO_PROTEGIDO_ACTUAL",
  "data": [
    {
      "momento_comparacion": "2026-10-08T09:59:58",
      "indice_anomalia": 12.4,
      "id": 131255,
      "nivel_anomalia": "BAJO"
    }
  ]
}
```

`indice_anomalia` sigue siendo rareza/distancia respecto al comportamiento aprendido, no probabilidad de falla. La ventana solicitada no reentrena ni redefine el baseline. El maximo por defecto es 168 horas y se puede ajustar con `ANOMALY_TIMELINE_MAX_HOURS`.

## Validacion del detector

```text
GET /api/v5/controlled-tests/catalog/
GET /api/v5/controlled-tests/validation/?source=sqlserver
GET /api/v5/controlled-tests/rows/?source=sqlserver&only_annotated=true
```

`validation` compara el detector con la bitacora y devuelve por prueba:

```text
indice_maximo / p95
indice_ml_max
indice_reglas_max
flujo min/max/mediana/std
presion min/max/mediana/std
nivel min/max
porcentaje de muestras en movimiento
eventos solapados
estado de validacion
```

No se usa la etiqueta para generar el score; por eso la validacion sigue siendo independiente del detector.

## Contexto hidraulico v5

Topologia confirmada:

```text
bomba -> sensor_presion_bomba -> valvula_superior_naranja
valvula_azul_superior -> sensor_presion_superior -> tanque
valvula_tanque -> tanque_inferior  (recirculacion corta)
valvula_roja_principal -> paso principal bomba-hacia-tanque
```

La valvula azul superior se utilizo para restringir flujo/presion y, junto con la valvula de tanque, para las pruebas reportadas como cavitacion.

Consulta:

```text
GET /api/v5/context/equipment/
```

## Sensores de presion en la demo actual

Fisicamente existen dos sensores, pero solo uno esta disponible para el backend:

```text
sensor_superior
ubicacion: entre valvula azul superior y tanque
PLC: principal
columna SQL: presion_relativa
estado: DISPONIBLE

sensor_bomba
ubicacion: entre bomba y valvula superior naranja
PLC: secundario
estado: NO_DISPONIBLE
motivo: PLC secundario no transmite datos actualmente
```

Por ello `presion_relativa` se mapea directamente a `presion_sensor_superior`. `presion_sensor_bomba` y `delta_presion_bomba_a_superior` permanecen nulos y no participan en scoring, reglas ni validacion numerica. Las notas humanas de la bitacora que mencionan ambos sensores se conservan, pero actualmente solo pueden comprobarse con el sensor superior.

## Baseline y ML protegidos

v5 conserva las protecciones anteriores:

1. baseline de ciclo congelado;
2. cuarentena de nuevos regimenes;
3. segunda pasada de Isolation Forest usando ciclos confiables;
4. exclusion adicional de todas las filas de pruebas controladas.

Los ciclos que intersectan la sesion controlada se etiquetan como `PRUEBA_CONTROLADA` y quedan fuera del baseline.

### Migracion

Si existe:

```text
artifacts/cycle_baseline_sqlserver_v40.json
```

o, en su defecto:

```text
artifacts/cycle_baseline_sqlserver_v32.json
```

v5 lo migra a:

```text
artifacts/cycle_baseline_sqlserver_v50.json
```

No uses `reset_cycle_baseline=true` durante una actualizacion normal.

## Endpoints principales

```text
GET  /api/v5/health/
GET  /api/v5/model/status/
POST /api/v5/model/rebuild/

GET  /api/v5/anomalies/summary/
GET  /api/v5/anomalies/latest/
GET  /api/v5/anomalies/history/
GET  /api/v5/anomalies/explain/{record_id}
GET  /api/v5/anomalies/events/

GET  /api/v5/cycles/latest/
GET  /api/v5/cycles/baseline/
GET  /api/v5/cycles/regime-candidates/
GET  /api/v5/cycles/regime-families/

GET  /api/v5/context/equipment/
GET  /api/v5/context/pump-reference/
GET  /api/v5/variables/status/

GET  /api/v5/controlled-tests/catalog/
GET  /api/v5/controlled-tests/validation/
GET  /api/v5/controlled-tests/rows/
```

## Actualizacion recomendada desde v4

Despues de aplicar el parche:

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python run.py
```

Luego reconstruye sin resetear baseline:

```text
POST /api/v5/model/rebuild/?source=sqlserver
```

Y revisa primero:

```text
GET /api/v5/controlled-tests/validation/?source=sqlserver
GET /api/v5/cycles/baseline/?source=sqlserver
GET /api/v5/anomalies/events/?source=sqlserver&threshold=80&scope=full
```

Swagger:

```text
http://127.0.0.1:8002/docs
```
