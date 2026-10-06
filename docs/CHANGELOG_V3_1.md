# Changelog v3.1

## Modelo

- `MODEL_VERSION`: `v3.1-explainable-transient-aware-cycle-health`.
- Se mantiene Isolation Forest por regimen y las reglas fisicas de v3.0.
- Se agregan estadisticas robustas por variable y regimen para explicabilidad heuristica.

## Explicabilidad

Cada registro puede incluir:

- `variables_mas_atipicas`;
- `origen_deteccion`;
- `clasificacion_operativa`;
- `familia_probable`;
- `impacto_hidraulico`;
- `impacto_mecanico`;
- `falla_confirmada`;
- `metodo_explicacion_ml`;
- `explicacion_ml_es_causal=false`.

Nuevo endpoint:

```text
GET /api/v3/anomalies/explain/{record_id}
```

La explicacion es robusta contra el baseline del regimen; no es una atribucion causal exacta del Isolation Forest.

## Eventos

Los eventos agregan las variables atipicas mas repetidas/intensas y clasifican su evidencia como ML, regla fisica o combinacion de ambas.

Se introducen familias diagnosticas conservadoras, entre ellas:

- `TRANSITORIO_ELECTRICO_VFD`;
- `TRANSITORIO_ELECTRICO_VFD_CON_RESPUESTA_FISICA`;
- `DESACOPLAMIENTO_HIDRAULICO`;
- `FALLA_VFD_EXPLICITA`.

## Ciclos

Se corrige la propagacion de eventos cortos al estado integral del ciclo.

Campos nuevos:

- `fraccion_muestras_ml_altas_pct`;
- `fraccion_muestras_ml_muy_altas_pct`;
- `duracion_ml_alta_max_seg`;
- `ml_persistente_en_ciclo`;
- `tuvo_evento_transitorio_ml`;
- `indice_ml_ciclo_efectivo`.

Un pico/evento ML >=70 solo eleva el estado integral del ciclo si es persistente segun fraccion o duracion configurada. El evento sigue existiendo independientemente.

## API

- `API_VERSION=3.1.0`.
- Se mantienen rutas `/api/v3/...`.
- Se eliminan alias `/api/v2/...` del router principal porque aun no existe un frontend dependiente del contrato.

## Configuracion nueva

```env
EXPLAIN_MIN_INDEX=70
EXPLAIN_TOP_FEATURES=8
EXPLAIN_MIN_ROBUST_Z=1.5
EXPLAIN_Z_FULL_SCALE=6

CYCLE_ML_INTEGRAL_THRESHOLD=70
CYCLE_ML_PERSISTENCE_FRACTION=0.25
CYCLE_ML_PERSISTENCE_SECONDS=30
CYCLE_TRANSIENT_ML_CAP=39
```

## Compatibilidad de datos

No requiere migracion SQL ni cambios de Node-RED.

Se mantienen como exclusiones ML:

- `energia_aparente_l1` mientras siga congelada en PLC;
- `numero_arranques` como contador manual;
- `start_stop_vfd` hasta confirmar su semantica;
- S1/S2 como contexto, no predictor numerico directo;
- `horas_marcha` como contexto acumulado.
