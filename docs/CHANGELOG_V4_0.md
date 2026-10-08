# Changelog v4.0

Modelo: `v4.0-physical-context-protected-ml-regime-families`
API: `4.0.0`

La v4.0 incorpora la nueva informacion obtenida del proyecto TIA Portal y de las fotografias reales del banco hidraulico.

## Contexto fisico confirmado

- Bomba: Pedrollo PK 60, 0.37 kW / 0.5 HP, 60 Hz, 3450 min^-1 nominales, Q 5-40 L/min, H 38-5 m.
- Nivel: Endress+Hauser Prosonic T FMU30, 4-20 mA, 2 hilos.
- Presion: Endress+Hauser Cerabar M PMP51, 4-20 mA HART. El rango configurado real sigue pendiente de confirmar en TIA.
- Temperatura: Endress+Hauser iTHERM ModuLine TM131.
- Valvula manual roja: abierta para habilitar el flujo principal del circuito.
- Dos valvulas azules: drenaje manual del circuito. No estan instrumentadas.
- S1: solenoide inferior; S2: solenoide superior. La codificacion exacta de sus estados sigue pendiente.
- El llenado y vaciado son automaticos.

## 1. `velocidad` deja de presentarse como rpm

La columna SQL/PLC `velocidad` se conserva para ML y deteccion de movimiento, pero v4 la expone tambien como:

- `velocidad_vfd_raw`
- `velocidad_rpm_confirmada=false`
- `velocidad_unidad=RAW_VFD_NO_CONFIRMADA`

No se convertira a rpm hasta confirmar el escalado VFD/TIA.

## 2. Contexto fisico de la bomba PK60

Se agrega una referencia teorica aproximada usando los extremos Q/H de la placa y leyes de afinidad. Es solo contexto diagnostico:

- `altura_bomba_teorica_aprox_m`
- `presion_bomba_teorica_aprox_psi`
- `residuo_presion_vs_modelo_teorico_psi`
- `ratio_presion_flujo`

Por defecto `PUMP_PHYSICS_USE_FOR_RULES=false`: esta referencia NO genera fallas.

Nuevo endpoint:

```text
GET /api/v4/context/pump-reference/?flow_l_min=18.25&frequency_hz=40
```

## 3. Contexto del equipo

Nuevo endpoint:

```text
GET /api/v4/context/equipment/
```

Expone la informacion confirmada y, de forma explicita, lo que sigue pendiente de confirmar.

## 4. Estado de proceso contextual

Cada registro incorpora:

- `modo_operacion_contextual`
- `estado_proceso_contextual`
- `estado_proceso_fuente`
- `frecuencia_contexto_hz`
- `fuente_frecuencia_contexto`

Se prefieren bits PLC de llenado/vaciado cuando estan activos; si no estan disponibles, la inferencia por nivel/fase queda marcada como `INFERIDO`.

## 5. Isolation Forest protegido contra normalizacion de regimenes nuevos

El baseline de ciclos ya estaba protegido en v3.2. v4 agrega una segunda pasada de entrenamiento:

1. Primera pasada: reconstruye ciclos y determina `CONFIABLE_BASELINE`, cuarentena y exclusiones.
2. Segunda pasada: `ARRANQUE`, `OPERACION_ESTABLE` y `DESACELERACION` se entrenan solo con ciclos confiables.

Asi un regimen persistente en cuarentena no termina convirtiendose en normal simplemente por hacer rebuilds repetidos.

Configuracion:

```env
ML_PROTECTED_ACTIVE_TRAINING=true
```

## 6. Migracion del baseline v3.2

Si existe:

```text
artifacts/cycle_baseline_<source>_v32.json
```

v4 lo migra automaticamente a:

```text
artifacts/cycle_baseline_<source>_v40.json
```

Esto evita aprender de nuevo una referencia que ya habia sido protegida.

## 7. Episodio y familia de regimen

Los antiguos NR pasan conceptualmente a episodios:

```text
EP-001
EP-002
...
```

Episodios separados con la misma firma se agrupan en una familia:

```text
RF-001
```

Para la firma:

```text
presion_estable_mediana:BAJA
flujo_estable_mediana:ALTA
```

la familia se interpreta como:

```text
CAMBIO_PUNTO_OPERACION_HIDRAULICO
```

Esto sigue siendo una hipotesis operativa, no una falla confirmada.

Nuevo endpoint:

```text
GET /api/v4/cycles/regime-families/?source=sqlserver
```

## 8. Explicabilidad de eventos por variable + regimen

Los eventos ya no mezclan la misma variable entre ARRANQUE, OPERACION_ESTABLE y DESACELERACION.

`variables_mas_atipicas` agrega:

```text
regimen_origen
```

por lo que un valor de nivel en ARRANQUE se compara y presenta con la referencia de ARRANQUE, no con otra fase del mismo evento.

## 9. Nombre estructural del ciclo

`CICLO_COMPLETO_NORMAL` se reemplaza por:

```text
CICLO_COMPLETO
```

La palabra `NORMAL` queda reservada para evaluacion de salud, no para indicar que el ciclo simplemente termino completo.

## 10. Punto de operacion por ciclo

Los ciclos completos agregan `punto_operacion_hidraulico`, con valores observados, baseline, desviaciones porcentuales y una interpretacion como:

```text
MAYOR_CAUDAL_MENOR_PRESION_A_VELOCIDAD_RAW_SIMILAR
```

La velocidad se mantiene como RAW y el modelo PK60 permanece solo como contexto.

## Compatibilidad

- No requiere cambios en `dbo.LecturasTanque`.
- No requiere cambios en Node-RED.
- No modifica el `.env` real al aplicar el parche.
- La API principal cambia de `/api/v3/...` a `/api/v4/...` porque todavia no existe un frontend que dependa del contrato anterior.
- CMS/SM1281/vibraciones siguen fuera de este proyecto.
