# Changelog v5.0.2

## Nuevo endpoint temporal de indice de anomalia

Se agrega:

```text
GET /api/v5/anomalies/timeline/?source=sqlserver&hours=6
```

El endpoint recibe una cantidad de horas y toma todas las lecturas existentes dentro de esa ventana, terminando en el ultimo `fecha_hora` disponible en la fuente. Cada lectura se puntua con el modelo protegido ya entrenado y se devuelve cronologicamente con:

- `momento_comparacion`
- `indice_anomalia`
- `id` cuando existe
- `nivel_anomalia`

La ventana NO se utiliza para reentrenar el modelo ni para redefinir normalidad. El objetivo es mostrar la evolucion del indice a lo largo del tiempo usando la misma referencia protegida del backend.

## Contexto previo

Para calcular correctamente fases y transiciones en el borde inicial de la ventana, el backend lee `LATEST_CONTEXT_ROWS` registros anteriores. Esos registros solo sirven como contexto de calculo y no aparecen en la respuesta final.

## Referencia temporal

`ventana.hasta` es el ultimo registro disponible en SQL/archivo, no la hora del servidor. Esto permite usar el endpoint tanto en vivo como con historicos.

## Limite de seguridad

Por defecto se aceptan hasta 168 horas, configurable con:

```env
ANOMALY_TIMELINE_MAX_HOURS=168
```

No se aplica downsampling: se devuelve un punto por cada lectura registrada.

## Compatibilidad

No requiere cambios en SQL Server, Node-RED, baseline, ground truth ni modelo de Isolation Forest. La API pasa a version `5.0.2`; el modelo estadistico sigue siendo el de v5.0.1 porque su logica de scoring no cambia.
