# Changelog v3.2

Modelo: `v3.2-protected-baseline-new-regime-physical-scale-partial-cycle`

La v3.2 corrige cuatro problemas encontrados al validar la v3.1 con los ciclos reales del 6-Oct-2026.

## 1. Baseline de ciclos protegido y congelado

En v3.1 todos los ciclos completos con contexto `NORMAL` podian terminar participando en el baseline del siguiente rebuild. Eso permitia que una desviacion persistente se normalizara con el tiempo.

v3.2 crea un archivo de referencia protegido en el primer rebuild:

```text
artifacts/cycle_baseline_<source>_v32.json
```

Los rebuild posteriores reutilizan esa referencia. Los ciclos nuevos se clasifican como:

```text
CONFIABLE_BASELINE
CUARENTENA_BASELINE
EXCLUIDO_BASELINE
```

Un ciclo en cuarentena no modifica automaticamente la referencia normal.

Solo si el operador decide que el nuevo comportamiento es normal se debe ejecutar:

```text
POST /api/v3/model/rebuild/?source=sqlserver&reset_cycle_baseline=true
```

Este parametro elimina y reconstruye explicitamente el baseline. No debe usarse como parte del rebuild rutinario.

## 2. Cuarentena y nuevo regimen candidato

Las desviaciones consecutivas con una firma fisica semejante se agrupan como `NUEVO_REGIMEN_CANDIDATO` si alcanzan el minimo configurado.

Campos nuevos por ciclo:

```text
estado_baseline
motivos_cuarentena_baseline
nuevo_regimen_candidato
regimen_candidato_id
firma_nuevo_regimen_candidato
ciclos_consecutivos_regimen_candidato
clasificacion_ciclo_operativa
```

Endpoint nuevo:

```text
GET /api/v3/cycles/regime-candidates/
```

Un nuevo regimen candidato permanece en cuarentena: no se considera automaticamente falla ni normalidad.

## 3. Explicabilidad con piso de escala fisica

v3.1 podia producir valores como `z_robusto_aprox=58064` cuando una variable discreta tenia MAD practicamente cero.

v3.2 calcula:

```text
escala_explicacion = max(escala_estadistica, piso_escala_fisica)
```

Se agregan:

```text
escala_estadistica
piso_escala_fisica
escala_explicacion
escala_limitada_por_piso_fisico
```

Con los valores predeterminados, una salida de VFD de `155 -> 164` utiliza un piso de 2 V y produce una desviacion interpretable, no decenas de miles de desviaciones robustas.

## 4. Ciclos parciales por nivel inicial alto

Un ciclo que arranca cerca del setpoint superior y termina rapidamente ya no se compara directamente contra un ciclo completo 40 -> 80.

Se identifica como:

```text
CICLO_PARCIAL_NIVEL_INICIAL
```

Si no existe una regla fisica fuerte:

```text
indice_firma_ciclo = null
indice_ciclo = null
nivel_ciclo = NO_COMPARABLE
```

El ciclo permanece disponible y sus eventos instantaneos siguen siendo analizables.

## Ajuste adicional

Los conteos de `familia_probable` usan `SIN_CLASIFICAR` cuando existe una deteccion sin familia suficiente, evitando diferencias silenciosas entre el total detectado y el total por familia.

## Pruebas

```text
28 passed
```
