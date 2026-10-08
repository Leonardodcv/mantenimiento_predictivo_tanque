from __future__ import annotations

from typing import Any


EQUIPMENT_CONTEXT_VERSION = "2026-10-07-photo-tia-context-1"
M_H2O_TO_PSI = 1.4223343308


EQUIPMENT_CONTEXT: dict[str, Any] = {
    "context_version": EQUIPMENT_CONTEXT_VERSION,
    "scope": "tanque_hidraulico",
    "project_boundary": {
        "includes": ["tanque", "bomba", "VFD", "hidraulica", "instrumentacion", "solenoides"],
        "excludes": ["CMS", "SM1281", "vibraciones"],
        "note": "La demostracion de vibraciones/CMS es un proyecto separado y no se mezcla en este backend.",
    },
    "pump": {
        "manufacturer": "Pedrollo",
        "model": "PK 60",
        "source": "fotografia_placa",
        "nameplate": {
            "flow_range_l_min": [5.0, 40.0],
            "head_range_m": [38.0, 5.0],
            "head_max_m": 40.0,
            "power_kw": 0.37,
            "power_hp": 0.5,
            "nominal_frequency_hz": 60.0,
            "nominal_speed_min_1": 3450.0,
            "supply_220_v_delta_current_a": 2.1,
            "supply_440_v_star_current_a": 1.2,
            "service": "S1",
            "max_temperature_c": 60.0,
        },
        "physics_model": {
            "method": "linearizacion_extremos_placa_mas_leyes_afinidad",
            "diagnostic_only": True,
            "reason": (
                "La placa aporta extremos Q/H, no la curva completa. Ademas la ubicacion de la toma "
                "de presion y las perdidas de la instalacion aun no estan modeladas."
            ),
        },
    },
    "sensors": {
        "temperature": {
            "manufacturer": "Endress+Hauser",
            "family_model": "iTHERM ModuLine TM131",
            "source": "fotografia_placa",
            "configured_range_confirmed": False,
        },
        "level": {
            "manufacturer": "Endress+Hauser",
            "model": "Prosonic T FMU30",
            "order_code": "FMU30-AAHGAARGF",
            "source": "fotografia_placa",
            "signal": "4-20 mA, 2-wire",
            "supply_vdc": [14.0, 35.0],
            "configured_range_confirmed": False,
        },
        "pressure": {
            "manufacturer": "Endress+Hauser",
            "family_model": "Cerabar M PMP51",
            "source": "fotografia_placa",
            "signal": "4-20 mA HART",
            "supply_vdc": [11.5, 45.0],
            "mwp_psi": 400.5,
            "nameplate_span_psi": [7.5, 150.0],
            "configured_range_confirmed": False,
            "note": "El span de placa no se toma automaticamente como rango configurado del PLC.",
        },
        "flow": {
            "source": "fotografia_instalacion",
            "model_confirmed": False,
            "configured_range_confirmed": False,
            "note": "El caudalimetro es visible, pero su placa/modelo aun no esta confirmado.",
        },
    },
    "manual_valves": {
        "instrumented": False,
        "main_red": {
            "role": "habilitar_paso_principal_del_circuito",
            "normal_operating_position": "ABIERTA",
            "source": "confirmacion_usuario",
        },
        "blue_drain_valves": {
            "count": 2,
            "role": "drenaje_manual_del_circuito",
            "source": "confirmacion_usuario",
            "normal_operating_position_confirmed": False,
        },
        "diagnostic_note": (
            "Como las valvulas manuales no estan instrumentadas, un cambio de posicion puede alterar "
            "el punto hidraulico sin quedar registrado en SQL."
        ),
    },
    "automation": {
        "fill_and_drain": "AUTOMATICO",
        "source": "confirmacion_usuario_y_proyecto_TIA",
        "solenoid_s1": {"role": "solenoide_inferior", "state_encoding_confirmed": False},
        "solenoid_s2": {"role": "solenoide_superior", "state_encoding_confirmed": False},
        "start_stop_vfd_semantics_confirmed": False,
        "note": (
            "El proyecto TIA contiene logica automatica de llenado/vaciado, motor VFD y solenoides. "
            "La codificacion exacta de estado_s1/estado_s2 y start_stop_vfd sigue pendiente de exportar/confirmar."
        ),
    },
    "units_and_scaling": {
        "velocity_column": {
            "database_column": "velocidad",
            "interpretation": "VFD_RAW",
            "rpm_confirmed": False,
            "reason": (
                "La bomba nominal es 3450 min^-1 a 60 Hz, mientras el historico muestra ~1198 con ~40 Hz. "
                "Hasta confirmar el escalado Modbus/TIA, el valor no se publica como rpm."
            ),
        },
        "flow_l_min_confirmed": False,
        "pressure_psi_confirmed": False,
        "note": (
            "El modelo fisico de bomba usa provisionalmente las magnitudes de flujo/presion tal como se han "
            "interpretado en el banco, pero no las convierte en reglas de falla hasta confirmar escalados TIA."
        ),
    },
}


def pump_expected_head_from_nameplate(flow_l_min: float | None, frequency_hz: float | None) -> dict[str, Any]:
    """Referencia teorica aproximada; nunca confirma por si sola una falla.

    Usa los extremos de placa PK60 (Q 5..40 L/min, H 38..5 m a 60 Hz) y leyes de afinidad.
    No sustituye una curva oficial completa ni compensa perdidas/ubicacion del sensor de presion.
    """
    result: dict[str, Any] = {
        "modelo": "PK60_EXTREMOS_PLACA_AFINIDAD",
        "aplicable": False,
        "solo_contexto": True,
        "frecuencia_nominal_hz": 60.0,
        "caudal_nominal_equivalente_l_min": None,
        "altura_teorica_aprox_m": None,
        "presion_teorica_aprox_psi": None,
        "nota": "Referencia teorica, no regla de falla.",
    }
    try:
        if flow_l_min is None or frequency_hz is None:
            return result
        q = float(flow_l_min)
        f = float(frequency_hz)
    except (TypeError, ValueError):
        return result
    if q <= 0 or f <= 0:
        return result

    ratio = f / 60.0
    if ratio <= 0:
        return result
    q_nom = q / ratio
    result["caudal_nominal_equivalente_l_min"] = round(q_nom, 4)
    if not 5.0 <= q_nom <= 40.0:
        result["nota"] = "Caudal equivalente fuera de los extremos Q/H impresos en la placa."
        return result

    # Interpolacion lineal SOLO entre los extremos visibles de la placa.
    h_nom = 38.0 + (q_nom - 5.0) * (5.0 - 38.0) / (40.0 - 5.0)
    h_scaled = h_nom * ratio * ratio
    result.update(
        {
            "aplicable": True,
            "altura_teorica_aprox_m": round(h_scaled, 4),
            "presion_teorica_aprox_psi": round(h_scaled * M_H2O_TO_PSI, 4),
            "nota": (
                "Aproximacion por extremos de placa y afinidad. No compensa cota, perdidas, valvulas, "
                "toma de presion ni curva real completa."
            ),
        }
    )
    return result


def hydraulic_point_context(
    flow_value: float | None,
    pressure_value: float | None,
    frequency_hz: float | None,
) -> dict[str, Any]:
    model = pump_expected_head_from_nameplate(flow_value, frequency_hz)
    ratio = None
    residual = None
    try:
        flow = None if flow_value is None else float(flow_value)
        pressure = None if pressure_value is None else float(pressure_value)
        if flow is not None and flow > 1e-12 and pressure is not None:
            ratio = pressure / flow
        theoretical = model.get("presion_teorica_aprox_psi")
        if pressure is not None and theoretical is not None:
            residual = pressure - float(theoretical)
    except (TypeError, ValueError):
        pass
    return {
        "ratio_presion_flujo": None if ratio is None else round(float(ratio), 6),
        "altura_bomba_teorica_aprox_m": model.get("altura_teorica_aprox_m"),
        "presion_bomba_teorica_aprox_psi": model.get("presion_teorica_aprox_psi"),
        "residuo_presion_vs_modelo_teorico_psi": None if residual is None else round(float(residual), 4),
        "modelo_bomba_fisico_aplicable": bool(model.get("aplicable")),
        "modelo_bomba_fisico_solo_contexto": True,
        "modelo_bomba_fisico_nota": model.get("nota"),
    }
