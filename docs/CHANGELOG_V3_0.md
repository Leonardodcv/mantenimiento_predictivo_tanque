# Changelog v3.0

## Motor

- Nuevo `MODEL_VERSION`: `v3.0-regime-aware-engineering-tolerance-restart-context`.
- Mantiene historial mixto LEGACY + ENRIQUECIDO.
- Se agrega `indice_ml_crudo` y se conserva `indice_ml` efectivo.

## Reanudacion

- Detecta paro prolongado antes de un ciclo.
- Clasifica ciclos como `REANUDACION`, `ESTABILIZACION` o `NORMAL`.
- Por defecto usa 2 horas como paro prolongado y 4 ciclos de estabilizacion.
- El contexto de reanudacion limita rareza estadistica, pero nunca una regla fisica fuerte.
- Los ciclos de reanudacion/estabilizacion se excluyen del baseline normal principal.

## Reposo

- Se agregan `fase_modelo`, `subestado_reposo` y `pendiente_nivel_por_seg`.
- Subestados: `REPOSO_ESTATICO`, `VACIADO`, `ESPERA_CICLO`, `REPOSO_TRANSITORIO`.
- El vaciado se deriva principalmente por caida del nivel; no se asume una semantica no confirmada de S1/S2.

## Ciclos

- La firma usa estadistica robusta + tolerancias minimas de ingenieria.
- Se agregan `indice_firma_ciclo_crudo`, `indice_ciclo_crudo` y detalle de tolerancia fisica.
- Variaciones pequenas no se convierten en 100 solo porque el MAD sea muy pequeno.
- Los ciclos incompletos siguen devolviendo `EN_CURSO`.

## Eventos

- Por defecto los estados pasivos de reposo no abren eventos solo por rareza estadistica.
- Las reglas fisicas fuertes siguen abriendo de inmediato.
- `scope=recent` y `scope=full` se conservan.

## API

- API principal en `/api/v3/...`.
- Alias `/api/v2/...` mantenidos para compatibilidad.
- Nuevo endpoint `/api/v3/cycles/baseline/`.

## Validacion

- 20 pruebas automaticas aprobadas.
- El dataset demo reduce de forma importante los eventos falsos de reposo y mantiene la deteccion de la prueba conocida de velocidad sin respuesta hidraulica.
