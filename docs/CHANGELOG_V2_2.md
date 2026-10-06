# Cambios v2.2

La v2.2 se genera a partir de la v2.1 y mantiene compatibilidad con el historial mixto de SQL Server.

## Correcciones

- `presion_alta` ya no incrementa directamente `indice_anomalia`; se reporta como advertencia.
- `falta_presion` y `bajo_flujo` durante la gracia de arranque pasan a observaciones informativas.
- Los eventos requieren persistencia/histeresis, salvo reglas fisicas fuertes.
- Los ciclos se califican con `indice_p95`, reglas fisicas y firma robusta; no por un unico pico maximo.
- Los ciclos incompletos se marcan `EN_CURSO`.
- Los conteos `ALTO` y `MUY_ALTO` del resumen son exactos y se agrega `registros_indice_alto_o_superior`.
- `/variables/status/` incluye todas las nuevas variables contextuales relevantes.
- `model_version` cambia a `v2.2-mixed-history-cycle-aware-persistent-events`.

## Sin cambios

- No cambia el esquema SQL.
- No cambia Node-RED.
- No hay dependencias Python nuevas.
- Los registros LEGACY con variables nuevas en `NULL` siguen participando en el modelo.
