# Parche v5.0.1 - canal de presion activo

Este parche corrige el contexto de presion de la demo sin cambiar la arquitectura general de v5.0.

## Cambios

- `presion_relativa` queda confirmada como la senal del **sensor superior**, ubicado entre la valvula azul superior y el tanque.
- El sensor de presion cercano a la bomba se mantiene documentado fisicamente, pero queda marcado `NO_DISPONIBLE` porque su PLC secundario no transmite datos actualmente.
- El sensor cercano a la bomba no participa en scoring, reglas, ground truth numerico ni calculos fisicos actuales.
- `presion_sensor_superior` se obtiene directamente de `presion_relativa`.
- `presion_sensor_bomba` y `delta_presion_bomba_a_superior` permanecen nulos.
- Se elimina la necesidad actual de `PRESSURE_SENSOR_PUMP_COLUMN` y `PRESSURE_SENSOR_TANK_COLUMN` en `.env.example`.
- La validacion de pruebas controladas declara explicitamente que las metricas de presion corresponden solo al sensor superior.
- Las notas humanas que mencionan ambos sensores se conservan como bitacora, pero no se intentan comprobar contra un canal no disponible.
- API/model metadata se actualiza a `5.0.1` sin cambiar las rutas `/api/v5/...`.

## Compatibilidad

No requiere cambios en SQL Server, Node-RED ni en el baseline congelado. No uses `reset_cycle_baseline=true` para aplicar este parche.

## Pruebas

Suite completa: `44 passed`.
