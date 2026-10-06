# Mantenimiento Predictivo Tanque - parche v2.2.1

Fecha: 2026-10-06

## Objetivo

Este parche no cambia el algoritmo de anomalias de v2.2. Corrige la forma en que se consultan ciclos y permite distinguir eventos recientes de eventos del historial completo.

## Cambios

1. API version actualizada a `2.2.1`.
2. El `model_version` permanece en `v2.2-mixed-history-cycle-aware-persistent-events` porque el algoritmo de scoring no cambia.
3. `GET /api/v2/cycles/latest/` ahora devuelve los ultimos ciclos detectados en el snapshot completo generado por el ultimo `model/rebuild`.
   - Ya no depende de encontrar ciclos dentro de una ventana SQL reciente.
   - Si la maquina estuvo detenida durante la noche, seguira devolviendo los ultimos ciclos existentes.
4. `GET /api/v2/anomalies/events/` agrega el parametro `scope`:
   - `scope=recent`: comportamiento operacional reciente. Puede devolver 0 si la maquina estuvo detenida o no hubo eventos persistentes.
   - `scope=full`: analiza el historial completo ya puntuado durante el ultimo rebuild.
5. Se conserva `events_latest()` internamente como compatibilidad con v2.2.
6. No se modifica SQL Server, Node-RED, `.env`, `requirements.txt` ni el modelo matematico.

## Validacion

La suite completa del proyecto v2.2 mas las pruebas del parche termina con:

```text
14 passed
```
