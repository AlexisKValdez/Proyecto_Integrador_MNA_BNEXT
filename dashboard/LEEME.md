# Tablero AI/BI «Demanda Tiendas SIX» — Equipo 43

Tablero de Databricks (AI/BI Dashboards, antes Lakeview) con siete vistas sobre las tablas `gold.dash_*` que calcula
`40_salidas_dashboard`. **Los datos no salen de Databricks**: se comparte dentro del workspace con el equipo y con
Bnext, y los importes que muestra son los del catálogo privado.

| Vista | Pregunta de negocio | Tablas |
| --- | --- | --- |
| **Resumen** | ¿Cuánto se espera vender en los próximos 14 días y cómo se compara con las dos semanas anteriores, por región y tipo? | `gold_dash_resumen_ejecutivo` |
| **Pronóstico** | ¿Cómo viene la venta de una región, zona, categoría o tienda: historia de 8 semanas y 14 días con banda P10–P90? | `gold_dash_serie_tiempo` |
| **Precisión** | ¿Cuánto acierta el modelo (prueba de septiembre y producción) y cómo se compara con el ingenuo y con los demás modelos? | `gold_dash_precision`, `gold_dash_precision_modelos` |
| **Surtido** | ¿Qué productos y cuántas unidades cubren la demanda de cada tienda × categoría esta semana y la siguiente? | `gold_surtido_sku_semana` |
| **Promociones** | ¿Qué promociones vienen, cuánto aporta el plan y qué pasaría con otra profundidad de descuento? | `gold_dash_promociones`, `gold_escenarios_whatif_resumen` |
| **Alertas** | ¿Qué series caen o suben fuerte, cuáles tienen alta incertidumbre, qué tiendas están inactivas y qué dice el monitoreo? | `gold_dash_alertas` |
| **Estado** | ¿Están frescos los datos, cuál es el modelo vigente, pasó la compuerta, cuándo corrió cada notebook? | `gold_dash_estado_pipeline` |

## Requisitos

1. Un **SQL warehouse** (serverless) en el workspace: los tableros AI/BI consultan con él.
2. Haber corrido `22` → `30` → `31` → `40` (Job J4) y, para la vista Precisión, `Avance5.43` completo y `23`.

## Importar el tablero

1. Databricks → **Dashboards** → menú **⋮** (o botón *Create*) → **Import dashboard from file** → elegir
   `mna_equipo43_demanda.lvdash.json`.
2. Abrirlo, elegir el SQL warehouse y pulsar **Publish**. Compartir con *Can view* a quien deba consultarlo.
3. Si el catálogo o esquema fueran otros, regenerar el archivo:
   `python herramientas/generar_dashboard.py dashboard modeladoprescriptivoventas.maestria`.

Si la importación rechazara el archivo (el formato cambia entre versiones de Databricks), el tablero se arma a mano
en 20–30 minutos con las consultas de `consultas.sql`: cada bloque es un *dataset*; las visualizaciones por vista están
en la tabla de arriba (contadores, barras por región y tipo, líneas de venta real vs. pronóstico coloreadas por
`tipo_dato`, tablas con filtros por región, zona, tipo, categoría y tienda).

## Tablero AI/BI, Databricks App o React: qué es cada cosa

| Opción | Qué es | Qué se necesita | Cuándo conviene |
| --- | --- | --- | --- |
| **Tablero AI/BI** (lo entregado) | Tablero nativo de Databricks: se construye con consultas SQL, sin programar; filtros, gráficas, tablas; se publica y comparte dentro del workspace; se refresca al consultar | SQL warehouse; importar el `.lvdash.json` | Siempre primero: cubre las 7 vistas, cero código de interfaz, datos dentro de Databricks |
| **Databricks App (Streamlit o Dash, en Python)** | Aplicación web alojada en el workspace que lee las tablas `gold` por el SQL warehouse; permite controles propios (p. ej. un what-if con deslizadores que re-puntúe el modelo) | Databricks Apps habilitado por el administrador, SQL warehouse, un *service principal* con permiso de lectura | Cuando el tablero se queda corto en interacción (what-if a la medida, formularios) |
| **Databricks App con React** | Lo mismo, con la interfaz escrita en React (JavaScript) y un *backend* (FastAPI o Node) que consulta Databricks | Lo anterior + alguien que programe React; compilar la app y desplegarla | Cuando se quiere una interfaz de producto (marca, navegación propia); es la opción más costosa y no aporta más datos ni más modelo |

React no es automatización: es sólo la capa visual. La automatización del proyecto son los **Jobs** (ingesta,
features, entrenamiento, inferencia, monitoreo) y las tablas que dejan listas; cualquiera de las tres interfaces lee las
mismas tablas `gold.dash_*`. Para el curso, el tablero AI/BI es suficiente; una App de Streamlit es el siguiente paso
natural si hay tiempo (un día de trabajo); React sólo si el negocio lo pide.

## Genie (opcional)

Con las mismas tablas se puede crear un **Genie space** (Databricks → Genie → New) para preguntar en lenguaje natural
(«¿qué categorías crecen más en la región norte la próxima semana?»). Los comentarios de columna que dejan los
notebooks (`COMMENT ON COLUMN`) son lo que Genie usa para entender cada campo.

## Actualización

El tablero se refresca solo al consultar; las tablas se actualizan cuando corre J4 (nuevo pronóstico) o J5 (monitoreo).
Para una captura pública (reporte, GitHub) usar las vistas sin importes (Precisión, Estado, Alertas) o índices.
