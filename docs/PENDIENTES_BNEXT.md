# Pendientes con Bnext — Equipo 43 (noviembre de 2026)

| # | Qué se necesita | Para qué | Estado |
| --- | --- | --- | --- |
| 1 | **Calendario de promociones planeadas**: hoy sólo hay fechas; faltan categoría (o tipo), descuento y mecánica | Con sólo fechas se captura `categoria_id = TODAS` y el descuento se imputa; con el detalle el efecto deja de estar diluido | parcial (sólo fechas) |
| 2 | **Noviembre de 2026** (y meses siguientes) en el volumen `maestria/2026-11` | Medir en producción el pronóstico de noviembre; reentrenar con más historia | octubre entregado; noviembre pendiente |
| 3 | Entrega faltante del **16 al 31 de julio** | Cerrar el hueco (hoy los cortes lo evitan) | pendiente desde octubre |
| 4 | Significado de los **estatus de tienda** 8 / 17 / 4 | Confirmar qué tiendas se pronostican | pendiente |
| 5 | Confirmación de **GENERICO** y de las uniones de categorías (YOGURTH→YOGURT, AGUA→AGUAS, …) | Aprobación `alcance_categoria` en `mlops.aprobaciones` | pendiente |
| 6 | **Política de servicios**: ¿se pronostican o se excluyen las categorías de servicio? | Aprobación `politica_servicios`; parámetro `ALCANCE_SERVICIOS` | pendiente |
| 7 | **Atributos al pronosticar**: tipo de tienda, zona, región, jefatura y estatus disponibles para tiendas nuevas | Aprobación `atributos_point_in_time` | pendiente |
| 8 | **SQL warehouse** serverless en el workspace | Tablero AI/BI y alertas SQL | por confirmar |
| 9 | Permisos: **crear Jobs** y **registrar modelos** (`CREATE MODEL` / `EXECUTE` en `maestria`) para el usuario del equipo | Jobs J1–J5 y alias `@champion` | por confirmar |
| 10 | Inventario en tienda (existencias) por producto, si existe | Convertir el techo de demanda en pedido sugerido (31) | opcional |
| 11 | Región de las tiendas que no la traen en los datos (hoy se imputa con la zona) | Mejor segmentación del monitoreo y el tablero | opcional |
