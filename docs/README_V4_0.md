# Mantenimiento Predictivo del Tanque - Backend v4.0

Backend FastAPI para deteccion de anomalias y analisis de ciclos del banco hidraulico.

La v4.0 incorpora contexto real del banco obtenido del proyecto TIA Portal y de fotografias de la instalacion: bomba Pedrollo PK 60, sensores Endress+Hauser, valvulas manuales y automatizacion de llenado/vaciado. Tambien protege el entrenamiento ML activo contra la normalizacion de regimenes nuevos persistentes.

## Principios de la v4.0

- El indice 0-100 representa rareza/anomalia, no probabilidad de falla.
- Reglas fisicas fuertes siguen teniendo prioridad sobre ML.
- `presion_alta` sigue siendo advertencia contextual, no falla directa.
- `energia_aparente_l1` sigue excluida mientras el PLC mantenga el valor congelado.
- `numero_arranques` sigue tratandose como contador de arranques manuales.
- `start_stop_vfd` sigue fuera del ML hasta confirmar su semantica exacta.
- `velocidad` se trata como RAW del VFD; no se presenta como rpm hasta confirmar escalado TIA/Modbus.
- La referencia teorica de la PK60 es informativa y no genera reglas de falla.
- CMS/SM1281/vibraciones no forman parte de este backend.

## Contexto de hardware incorporado

Consultar:

```text
GET /api/v4/context/equipment/
```

El registro incluye:

- Bomba Pedrollo PK 60.
- Sensor de nivel Endress+Hauser Prosonic T FMU30.
- Transmisor de presion Endress+Hauser Cerabar M PMP51.
- Sensor de temperatura Endress+Hauser iTHERM ModuLine TM131.
- Valvula roja manual: paso principal abierto durante operacion.
- Dos valvulas azules: drenaje manual, no instrumentadas.
- S1 = solenoide inferior.
- S2 = solenoide superior.
- Llenado/vaciado automaticos.

Lo que todavia no se confirma se devuelve explicitamente como pendiente, en lugar de inferirse como hecho.

## Cambio importante: velocidad VFD

La placa de la bomba indica 3450 min^-1 a 60 Hz, mientras el historico contiene valores cercanos a `1198` con ~40 Hz. Por ello v4 expone:

```json
{
  "velocidad_vfd_raw": 1198,
  "velocidad_rpm_confirmada": false,
  "velocidad_unidad": "RAW_VFD_NO_CONFIRMADA"
}
```

Los umbrales historicos de movimiento siguen usando el valor RAW porque han demostrado separar reposo/movimiento de forma consistente.

## Referencia fisica de bomba

Endpoint:

```text
GET /api/v4/context/pump-reference/?flow_l_min=18.25&frequency_hz=40
```

Se usan solo los extremos Q/H visibles en la placa de la PK60 y leyes de afinidad. El resultado no compensa perdidas, cota, posicion de valvulas ni ubicacion del transmisor de presion.

Variables derivadas por registro:

```text
ratio_presion_flujo
altura_bomba_teorica_aprox_m
presion_bomba_teorica_aprox_psi
residuo_presion_vs_modelo_teorico_psi
```

Por defecto:

```env
PUMP_PHYSICS_USE_FOR_RULES=false
```

## Estado de proceso contextual

La API agrega:

```text
modo_operacion_contextual
estado_proceso_contextual
estado_proceso_fuente
frecuencia_contexto_hz
fuente_frecuencia_contexto
```

Cuando los bits PLC de llenado/vaciado estan disponibles, se prefieren. Si no, la inferencia queda marcada como `INFERIDO`.

No se asigna todavia un significado a `estado_s1=1/2` o `estado_s2=1/2` hasta disponer de exportacion de redes/operandos TIA.

## Baseline protegido + entrenamiento ML protegido

v3.2 protegia la firma de ciclos. v4 agrega una segunda proteccion:

1. Primera pasada de ML para reconstruir ciclos.
2. Se determina cuales ciclos son `CONFIABLE_BASELINE`.
3. ARRANQUE / OPERACION_ESTABLE / DESACELERACION se vuelven a entrenar usando solo esos ciclos confiables.
4. El scoring final se realiza con ese modelo protegido.

Configuracion:

```env
ML_PROTECTED_ACTIVE_TRAINING=true
```

Esto evita que un patron persistente en cuarentena se vuelva normal solamente por ejecutar `model/rebuild` repetidamente.

## Migracion del baseline v3.2

Si existe:

```text
artifacts/cycle_baseline_sqlserver_v32.json
```

v4 lo copia/migra automaticamente a:

```text
artifacts/cycle_baseline_sqlserver_v40.json
```

No uses `reset_cycle_baseline=true` salvo que quieras redefinir explicitamente el comportamiento normal.

## Episodios y familias de regimen

Un episodio persistente se identifica como:

```text
EP-001
EP-002
```

Episodios separados con la misma firma se agrupan en una familia:

```text
RF-001
```

Ejemplo de firma:

```text
presion_estable_mediana:BAJA
flujo_estable_mediana:ALTA
```

Interpretacion candidata:

```text
CAMBIO_PUNTO_OPERACION_HIDRAULICO
```

Esto NO significa falla confirmada. Puede corresponder a menor resistencia hidraulica, una condicion de valvulas/trayectoria, un cambio manual no instrumentado o un escalado que aun deba validarse.

Endpoints:

```text
GET /api/v4/cycles/regime-candidates/?source=sqlserver
GET /api/v4/cycles/regime-families/?source=sqlserver
```

## Punto de operacion de cada ciclo

Los ciclos completos agregan:

```text
punto_operacion_hidraulico
```

con:

- flujo/presion observados,
- frecuencia contextual,
- velocidad VFD RAW,
- baseline protegido,
- desviaciones porcentuales,
- ratio presion/flujo,
- referencia teorica PK60 cuando es aplicable,
- interpretacion del desplazamiento del punto de operacion.

## Explicabilidad de eventos

`variables_mas_atipicas` agrega `regimen_origen`.

Por ello un mismo evento puede mostrar por separado:

```text
nivel_tanque + ARRANQUE
nivel_tanque + OPERACION_ESTABLE
```

sin mezclar medianas de fases distintas.

## Tipos de ciclo

La estructura ya no usa `CICLO_COMPLETO_NORMAL`.

Ahora:

```text
CICLO_COMPLETO
CICLO_PARCIAL_NIVEL_INICIAL
REANUDACION
ESTABILIZACION
EN_CURSO
```

La salud se expresa por campos separados como `nivel_ciclo`, `estado_baseline` y `clasificacion_ciclo_operativa`.

## Instalacion

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
python run.py
```

Swagger:

```text
http://127.0.0.1:8002/docs
```

## Endpoints principales v4

```text
GET  /api/v4/health/
GET  /api/v4/model/status/
POST /api/v4/model/rebuild/

GET  /api/v4/anomalies/summary/
GET  /api/v4/anomalies/latest/
GET  /api/v4/anomalies/history/
GET  /api/v4/anomalies/explain/{record_id}
GET  /api/v4/anomalies/events/

GET  /api/v4/cycles/latest/
GET  /api/v4/cycles/baseline/
GET  /api/v4/cycles/regime-candidates/
GET  /api/v4/cycles/regime-families/

GET  /api/v4/context/equipment/
GET  /api/v4/context/pump-reference/
GET  /api/v4/variables/status/
```

## Rebuild recomendado

```text
POST /api/v4/model/rebuild/?source=sqlserver
```

Despues revisar:

```text
GET /api/v4/cycles/baseline/?source=sqlserver
GET /api/v4/cycles/regime-candidates/?source=sqlserver
GET /api/v4/cycles/regime-families/?source=sqlserver
GET /api/v4/cycles/latest/?source=sqlserver&limit=30
GET /api/v4/anomalies/events/?source=sqlserver&threshold=80&scope=recent
GET /api/v4/anomalies/explain/84485?source=sqlserver
GET /api/v4/context/equipment/
```

## Base de datos y Node-RED

No hay migracion SQL para v4.0 y no hay que modificar el flujo Node-RED v3 para usar este backend.
