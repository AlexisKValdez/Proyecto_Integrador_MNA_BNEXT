# Cambios

## v6.1 · Octubre y promociones con sólo fechas (4-nov-2026)
- `MESES` hasta `2026-10`; cuatro bloques de prueba (1-sep a 26-oct); eventos de octubre a diciembre.
- Calendario: `categoria_id = TODAS` con `tipo_categoria`, imputación del descuento histórico, bloque comentado para el
  catálogo de Bnext; plantilla con columna `tipo_categoria` y ejemplo de sólo fechas.
- Tablero: fracción de series con promoción (observada/planeada) en la serie de tiempo.

## v6.0 · Avance 6 (noviembre de 2026)
- Operación: `22_pronostico_operativo` (inferencia con el modelo registrado), `23_monitoreo` (acierto, PSI, alertas),
  `24_promocion_modelo` (`@challenger` → `@champion` con aprobaciones en tabla).
- Calendario de promociones editable (`13_calendario_promociones_plan`, `calendario/`) y variables `f_*` sin fuga.
- Avance 6 corregido: `30` what-if re-puntuando con el modelo, `31` reparto a producto con fila OTROS y reconciliación.
- `40_salidas_dashboard` y tablero AI/BI importable (7 vistas).
- Jobs J1–J5 (`jobs/`), parámetros como widgets, compuerta de datos con paro duro, constraints y comentarios en tablas,
  publicación idempotente, bitácoras de corridas y pronósticos.
- Documentación: auditoría de código, runbook, diccionario de datos, pendientes con Bnext. Pruebas locales y CI.

## v5.0 · Avance 5 (octubre de 2026)
- `Avance5.43`: ajuste fino, ensambles homogéneos y heterogéneos, prueba en datos no vistos (septiembre), intervalos,
  compuerta, registro en Unity Catalog y pronóstico operativo. Pipeline 00–20 con septiembre.
