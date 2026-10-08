# CHANGELOG v5.0

## Objetivo

La v5.0 incorpora la bitacora del 8 de octubre de 2026 como **ground truth de pruebas controladas** y agrega la topologia hidraulica confirmada de las valvulas manuales y los dos puntos de presion. Las pruebas controladas no se convierten en fallas reales ni se usan para enseñar normalidad al modelo.

## Cambios principales

### 1. Ground truth de pruebas controladas

Se agrega `data/controlled_trials_2026-10-08.json` con la sesion `CT-20261008-AM` (08:10-09:28) y 28 anotaciones derivadas de la bitacora. La precision temporal se conserva como `MINUTO` o `RANGO_MINUTOS`; no se inventan segundos ni porcentajes de apertura.

Las etiquetas incluyen controles normales, cavitacion reportada, recirculacion, restricciones, perdida de flujo/presion, presion alta, obstruccion reportada y manipulacion deliberada del sensor ultrasonico.

### 2. No contaminacion del aprendizaje

Toda la sesion controlada se marca como:

- `excluir_entrenamiento_normal=true`
- `excluir_baseline_normal=true`

La razon es que las pruebas se realizaron de forma continua y no se puede asumir recuperacion normal entre anotaciones. El fallback de entrenamiento tambien conserva esta exclusion.

### 3. Validacion separada del scoring

Las etiquetas de bitacora no modifican `indice_anomalia`, `indice_ml`, `indice_reglas` ni `familia_probable`. Se usan exclusivamente para comparar la respuesta del detector mediante:

```text
GET /api/v5/controlled-tests/validation/?source=sqlserver
```

El resultado indica, por ventana anotada, si hubo respuesta del detector y resume flujo, presion, nivel, ML, reglas y eventos solapados.

### 4. Manipulacion del sensor ultrasonico

El intervalo 08:58-09:00 se clasifica como `CALIDAD_DATO_SENSOR`. Puede generar rareza, pero queda explicitamente separado de una falla hidraulica o mecanica real.

### 5. Topologia de valvulas y sensores de presion

Se incorpora el contexto confirmado:

- Valvula roja principal: paso de la bomba hacia el tanque.
- Valvula azul superior: antes del sensor de presion superior; puede restringir flujo/presion y se uso en pruebas.
- Valvula de tanque: deriva el flujo al tanque inferior y produce recirculacion corta.
- Sensor de presion superior: entre valvula azul superior y tanque.
- Sensor de presion de bomba: entre bomba y valvula superior naranja.

Los dos sensores se registran como Endress+Hauser Cerabar M PMP51 segun fotografias/contexto proporcionado.

### 6. Dos canales de presion preparados, no inventados

El dataset actual solo expone `presion_relativa`, por lo que v5 no fuerza cual sensor fisico corresponde a esa columna. Se agregan:

```env
PRESSURE_SENSOR_PUMP_COLUMN=
PRESSURE_SENSOR_TANK_COLUMN=
```

Cuando existan dos columnas SQL confirmadas, v5 calcula como contexto:

```text
presion_sensor_bomba
presion_sensor_superior
delta_presion_bomba_a_superior
```

El diferencial no es regla de falla por defecto.

### 7. Ciclos durante pruebas controladas

Los ciclos que intersectan la sesion quedan `EXCLUIDO_BASELINE` con motivo `prueba_controlada_ground_truth`. Su clasificacion operativa pasa a `PRUEBA_CONTROLADA` o `PRUEBA_CONTROLADA_CON_ANOMALIA_FISICA` segun corresponda.

### 8. Eventos con contexto de bitacora

Los eventos incluyen:

```text
es_sesion_pruebas_controladas
ground_truth_ids
ground_truth_tipos
ground_truth_falla_real
contiene_manipulacion_sensor
```

La etiqueta sigue siendo externa al detector.

### 9. Correccion de resumen

Se corrige el conteo `conteo_por_familia_probable`: filas `NORMAL` o `ADVERTENCIA` sin familia ya no inflan `SIN_CLASIFICAR`.

### 10. Baseline protegido

v5 usa `cycle_baseline_<source>_v50.json` y migra automaticamente referencias v4.0 o v3.2 existentes. No se recomienda `reset_cycle_baseline=true` durante una actualizacion normal.

## Pruebas

La suite completa de v5.0 contiene 43 pruebas automatizadas y cubre compatibilidad v2-v4, ground truth, exclusion de entrenamiento/baseline, topologia fisica y validacion de pruebas controladas.
