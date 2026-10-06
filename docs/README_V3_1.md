# Mantenimiento Predictivo del Tanque - Backend v3.1

Backend FastAPI para deteccion de anomalias, eventos persistentes y analisis de ciclos del banco hidraulico.

La v3.1 parte de la v3.0 y conserva:

- historial mixto `LEGACY` + `ENRIQUECIDO`;
- regimenes de reposo y `VACIADO`;
- contexto `REANUDACION` / `ESTABILIZACION` tras paro prolongado;
- tolerancias minimas de ingenieria para la firma del ciclo;
- reglas fisicas fuertes para desacoplamiento hidraulico y falla explicita de VFD;
- `presion_alta` como advertencia, no como falla directa.

## Por que existe v3.1

La validacion real del 6-Oct mostro un evento de aproximadamente 12 s dentro de un ciclo que, globalmente, termino con duracion, nivel, flujo y presion normales.

Durante ese intervalo cambiaron de forma coordinada variables como:

```text
voltaje_bus_dc
tension_l1_n
voltaje_salida
velocidad
potencia_activa_l1
potencia_aparente_l1
flujo_instantaneo
presion_relativa
```

El Isolation Forest hizo bien en conservar el evento multivariable, pero v3.0 tenia dos limitaciones:

1. `patron_multivariable_atipico_para_el_regimen` no explicaba que variables estaban provocando la rareza.
2. Un evento corto podia elevar el `indice_p95` del ciclo y hacer que todo el ciclo apareciera como `ALTO`, aunque su firma integral fuera normal.

La v3.1 corrige esas dos situaciones sin reducir la sensibilidad de las reglas fisicas.

## 1. Explicabilidad heuristica del ML

Los registros con rareza suficiente incluyen ahora:

```text
variables_mas_atipicas
origen_deteccion
clasificacion_operativa
familia_probable
impacto_hidraulico
impacto_mecanico
falla_confirmada
metodo_explicacion_ml
explicacion_ml_es_causal
```

Ejemplo conceptual:

```json
{
  "indice_anomalia": 90.5,
  "origen_deteccion": "ML_MULTIVARIABLE",
  "clasificacion_operativa": "DESVIACION_MULTIVARIABLE",
  "familia_probable": "TRANSITORIO_ELECTRICO_VFD_CON_RESPUESTA_FISICA",
  "falla_confirmada": false,
  "variables_mas_atipicas": [
    {
      "variable": "voltaje_bus_dc",
      "grupo": "ELECTRICA_VFD",
      "valor": 333,
      "mediana_regimen": 293,
      "desviacion_pct": 13.65,
      "z_robusto_aprox": 6.8,
      "direccion": "ALTA"
    }
  ]
}
```

### Importante sobre la explicacion

`variables_mas_atipicas` NO es SHAP ni una atribucion causal exacta del Isolation Forest.

El metodo utilizado es:

```text
desviacion_robusta_vs_baseline_del_regimen
```

Compara cada variable del registro contra la distribucion robusta del mismo regimen (`OPERACION_ESTABLE`, `ARRANQUE`, etc.) y sirve para explicar que variables acompañaron la deteccion.

Por eso:

```json
"explicacion_ml_es_causal": false
```

siempre permanece explicito.

Configuracion:

```env
EXPLAIN_MIN_INDEX=70
EXPLAIN_TOP_FEATURES=8
EXPLAIN_MIN_ROBUST_Z=1.5
EXPLAIN_Z_FULL_SCALE=6
```

## 2. Clasificacion de la evidencia

`origen_deteccion` distingue:

```text
NORMAL
ADVERTENCIA
ML_MULTIVARIABLE
REGLA_FISICA
REGLA_FISICA_Y_ML
```

`clasificacion_operativa` distingue:

```text
NORMAL
ADVERTENCIA
DESVIACION_MULTIVARIABLE
DESVIACION_TRANSITORIA_MULTIVARIABLE   # en eventos
ANOMALIA_FISICA
FALLA_CONFIRMADA
```

`falla_confirmada=true` se reserva para evidencia explicita, actualmente:

```text
falla_vfd = 1
estado_vfd = 3
```

Una regla fisica como velocidad alta sin flujo/presion sigue siendo una anomalia fisica fuerte, pero no se presenta automaticamente como causa de falla confirmada.

## 3. Familia probable del evento

La v3.1 agrega una clasificacion diagnostica conservadora basada en reglas fisicas y en las variables atipicas predominantes.

Familias posibles incluyen:

```text
FALLA_VFD_EXPLICITA
DESACOPLAMIENTO_HIDRAULICO
TRANSITORIO_ELECTRICO_VFD
TRANSITORIO_ELECTRICO_VFD_CON_RESPUESTA_FISICA
DESVIACION_ELECTROMECANICA
DESVIACION_HIDRAULICA
DESVIACION_MULTIVARIABLE
```

`familia_probable` ayuda al frontend/LLM a describir el tipo de evidencia. No sustituye diagnostico de causa raiz.

## 4. Evento transitorio != salud integral del ciclo

La v3.0 utilizaba el `p95` ML directamente dentro de `indice_ciclo`. Un evento de 10-15 s podia hacer que un ciclo de ~98 s quedara `ALTO` aun cuando:

```text
indice_firma_ciclo = 0
indice_regla_fisica_max = 0
```

v3.1 calcula adicionalmente:

```text
fraccion_muestras_ml_altas_pct
fraccion_muestras_ml_muy_altas_pct
duracion_ml_alta_max_seg
ml_persistente_en_ciclo
tuvo_evento_transitorio_ml
indice_ml_ciclo_efectivo
```

Por defecto, la rareza ML se considera integral para la salud del ciclo solo si cumple al menos una de estas condiciones:

```env
CYCLE_ML_INTEGRAL_THRESHOLD=70
CYCLE_ML_PERSISTENCE_FRACTION=0.25
CYCLE_ML_PERSISTENCE_SECONDS=30
```

Es decir, debe afectar aproximadamente >=25 % de las muestras del ciclo o durar >=30 s.

Si el ML supera 70 pero no cumple persistencia, el evento NO desaparece. Se conserva en `/anomalies/events/`, pero su contribucion al estado global del ciclo se limita:

```env
CYCLE_TRANSIENT_ML_CAP=39
```

Por tanto puede ocurrir correctamente:

```text
EVENTO:
  indice_p95 = 90
  desviacion transitoria real

CICLO COMPLETO:
  firma fisica normal
  regla fisica = 0
  nivel_ciclo = BAJO
  tuvo_evento_transitorio_ml = true
```

Esto no es una contradiccion: describe dos escalas temporales diferentes.

## 5. Reglas fisicas conservadas

La v3.1 no limita estas evidencias:

```text
velocidad alta + flujo casi nulo
velocidad alta + presion casi nula
falta_presion persistente despues de la gracia de arranque
bajo_flujo persistente despues de la gracia de arranque
falla_vfd activa
estado_vfd = error
```

La prueba historica conocida del 2-Oct sigue siendo detectada por regla fisica.

## 6. Regimenes y reanudacion

Se mantienen:

```text
REPOSO_ESTATICO
VACIADO
ESPERA_CICLO
REPOSO_TRANSITORIO
POST_PARO
ARRANQUE
OPERACION_ESTABLE
DESACELERACION
```

`ESPERA_CICLO` es un regimen opcional: si el historial no contiene muestras que satisfagan su definicion puede aparecer con `training_rows=0` y `trained=false`. Eso no invalida los demas modelos.

Contextos de ciclo:

```text
INICIO_HISTORIAL
NORMAL
REANUDACION
ESTABILIZACION
```

## API v3.1

Swagger:

```text
http://127.0.0.1:8002/docs
```

Endpoints:

```text
GET  /api/v3/health/
GET  /api/v3/model/status/
POST /api/v3/model/rebuild/

GET  /api/v3/anomalies/summary/
GET  /api/v3/anomalies/latest/
GET  /api/v3/anomalies/history/
GET  /api/v3/anomalies/events/
GET  /api/v3/anomalies/explain/{record_id}

GET  /api/v3/cycles/latest/
GET  /api/v3/cycles/baseline/
GET  /api/v3/variables/status/
```

Como aun no existe frontend dependiente del contrato, v3.1 elimina los alias historicos `/api/v2/...` del router principal.

### Explicar un registro concreto

Ejemplo:

```text
GET /api/v3/anomalies/explain/84485?source=sqlserver
```

Devuelve el registro puntuado, su clasificacion y las variables que mas se alejaron del baseline de su regimen.

## SQL Server recomendado

```env
DATA_SOURCE=sqlserver
MODEL_MAX_ROWS=0
SQL_DRIVER=ODBC Driver 17 for SQL Server
SQL_SERVER=USER4710-PC\SQLEXPRESS
SQL_PORT=
SQL_DATABASE=MantenimientoPredictivo
SQL_TABLE=dbo.LecturasTanque
SQL_TRUSTED_CONNECTION=yes
SQL_USERNAME=
SQL_PASSWORD=
SQL_ENCRYPT=no
SQL_TRUST_SERVER_CERTIFICATE=yes
```

No requiere cambios en `dbo.LecturasTanque` ni en Node-RED.

## Instalacion

```powershell
cd C:\ruta\mantenimiento_predictivo_tanque
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Si PowerShell bloquea la activacion:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Primer arranque

```powershell
python run.py
```

Luego:

```text
GET  /api/v3/health/
POST /api/v3/model/rebuild/?source=sqlserver
GET  /api/v3/cycles/latest/?source=sqlserver&limit=20
GET  /api/v3/cycles/baseline/?source=sqlserver
GET  /api/v3/anomalies/events/?source=sqlserver&threshold=80&scope=recent
GET  /api/v3/anomalies/events/?source=sqlserver&threshold=80&scope=full
```

## Pruebas

```powershell
python -m pytest -q
```

La entrega v3.1 incluye 24 pruebas automaticas.

## Importante

`indice_anomalia` sigue siendo un indice de rareza/anomalia de 0 a 100. No es probabilidad calibrada de falla ni RUL.

`familia_probable` y `variables_mas_atipicas` son ayudas diagnosticas. No deben presentarse como causalidad confirmada.
