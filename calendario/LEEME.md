# Calendario de promociones planeadas (archivo editable)

`calendario_promociones_plan.xlsx` (y su equivalente `calendario_promociones_plan.csv`) es **lo único que hay que
editar** para que el modelo, el pronóstico operativo y el what-if conozcan las promociones futuras. Hoy contiene un
**placeholder inventado** (filas con `fuente = placeholder`) que hay que sustituir por el plan real de Bnext.

## Cómo se usa

1. Abrir el Excel, hoja `promociones`. Una fila por promoción × categoría × alcance (la misma `promo_id` puede
   repetirse en varias categorías o alcances).
2. Llenar las columnas obligatorias: `promo_id`, `inicio_planeado`, `fin_planeado` (AAAA-MM-DD) y `categoria_id` **o**
   `categoria_nombre` (el nombre se traduce con el catálogo; el notebook 13 deja `catalogo_categorias.csv` junto al
   archivo con los identificadores válidos).
3. Alcance: dejar `tienda`, `zona` y `region` vacíos = nacional; llenar sólo el nivel que aplique.
4. `descuento_planeado` en fracción (0.15 = 15 %; 2x1 = 0.50). `fecha_publicacion` = desde cuándo se conoce el plan
   (vacío = 14 días antes del inicio). `fuente = bnext` para el plan real.
5. Guardar y subir el archivo al volumen `/Volumes/modeladoprescriptivoventas/maestria/outputs/calendario/`
   (Catalog → Volumes → outputs → calendario → Upload, reemplazando el anterior).
6. Correr `1_datos/13_calendario_promociones_plan` (o el Job **J2 features**) y después **J4 inferencia**: el pronóstico y
   los escenarios toman el plan nuevo sin reentrenar.

El notebook 13 valida el archivo (fechas, categorías, alcances, descuentos), reporta las filas con error y las omite,
construye el plan histórico a partir de las promociones observadas y escribe las tablas
`silver.calendario_promociones_plan` y `silver.calendario_promociones_plan_dia`.

## Si sólo se conocen las fechas

Pon `categoria_id = TODAS` (y `tipo_categoria = mercancia` si aplica sólo a mercancía), deja `descuento_planeado` vacío
y `fuente = bnext`. El notebook 13 expande la fila a todas las categorías de ese tipo y le pone a cada una su descuento
histórico mediano (`descuento_imputado = 1`). El efecto queda diluido (el modelo aprendió promociones por categoría);
cuando llegue el detalle, sustituye esas filas por una por categoría. La fila `PH-2026-11-SOLO-FECHAS` de la plantilla es
el ejemplo.

## Cuando llegue el catálogo completo de Bnext

Dos caminos: (1) vaciarlo a este Excel con las columnas del contrato, o (2) descomentar el bloque «CUANDO BNEXT ENTREGUE
SU CATÁLOGO» al inicio de la sección 2 del notebook 13, indicar dónde está (tabla, parquet o CSV) y mapear sus nombres de
columna en `MAPEO_COLUMNAS_BNEXT`. Todo lo demás (validación, histórico, tablas, pronóstico, what-if) sigue igual.

## Por qué importa `fecha_publicacion`

El modelo sólo usa, en cada fecha de pronóstico, las promociones que ya estaban publicadas. Así las variables del
calendario no filtran información del futuro en los cortes históricos ni en la prueba no vista, y lo que se mide
es lo que el negocio realmente sabría al pronosticar.

## Regenerar la plantilla

`python herramientas/generar_plantilla_calendario.py calendario` vuelve a escribir el Excel y el CSV con el placeholder.
