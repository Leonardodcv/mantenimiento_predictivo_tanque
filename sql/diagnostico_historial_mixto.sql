USE [MantenimientoPredictivo];
GO

-- Verifica que el historial anterior a la v3 siga presente y que los NULL
-- de variables nuevas se distingan de los registros enriquecidos.
SELECT
    COUNT_BIG(*) AS total_registros,
    SUM(CASE WHEN encendido_ia IS NULL THEN 1 ELSE 0 END) AS registros_legacy,
    SUM(CASE WHEN encendido_ia IS NOT NULL THEN 1 ELSE 0 END) AS registros_enriquecidos,
    MIN(fecha_hora) AS desde,
    MAX(fecha_hora) AS hasta
FROM dbo.LecturasTanque;
GO

-- Cobertura de algunas variables nuevas.
SELECT
    COUNT_BIG(*) AS total,
    COUNT(encendido_ia) AS con_encendido_ia,
    COUNT(setpoint_llenado) AS con_setpoint_llenado,
    COUNT(start_stop_s1) AS con_start_stop_s1,
    COUNT(start_stop_s2) AS con_start_stop_s2,
    COUNT(horas_marcha) AS con_horas_marcha,
    COUNT(numero_arranques) AS con_numero_arranques
FROM dbo.LecturasTanque;
GO

-- Ejemplo de historial cronologico completo para validacion.
SELECT *
FROM dbo.LecturasTanque
ORDER BY id ASC;
GO
