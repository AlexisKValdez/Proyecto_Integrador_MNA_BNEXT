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

## Por qué importa `fecha_publicacion`

El modelo sólo usa, en cada fecha de pronóstico, las promociones que ya estaban publicadas. Así las variables del
calendario no filtran información del futuro en los cortes históricos ni en la prueba de septiembre, y lo que se mide
es lo que el negocio realmente sabría al pronosticar.

## Regenerar la plantilla

`python herramientas/generar_plantilla_calendario.py calendario` vuelve a escribir el Excel y el CSV con el placeholder.
