# Pipeline en Azure Databricks — Equipo 43

Notebooks del proyecto en formato fuente de Databricks. Se importan en **`/Workspace/Shared/mna_equipo43/`** y se
ejecutan con cómputo **serverless**. Cada notebook trae su configuración y funciones de apoyo dentro (no hay paquetes
aparte). Las tablas del equipo se guardan en `modeladoprescriptivoventas.maestria` con prefijo de capa
(`bronze_*`, `silver_*`, `gold_*`, `mlops_*`).

## Notebooks

| # | Notebook | Qué hace |
| --- | --- | --- |
| 00 | `1_datos/00_entorno_y_permisos` | Revisa cómputo, bibliotecas y permisos de escritura |
| 01 | `1_datos/01_inventario_catalogo` | Inventario del catálogo: tablas, volúmenes, archivos por mes, experimentos |
| 02 | `1_datos/02_carga_bronce` | Carga los parquet mensuales a Bronce con trazabilidad; `MESES_A_CARGAR` permite cargar sólo un mes |
| 04–07 | `1_datos/04…07_eda_*` | Análisis exploratorio: calidad, tiempo y tiendas, variables y promociones, fuentes externas |
| 08 | `1_datos/08_limpieza_plata` | Reglas de limpieza, deduplicación entre meses, homologación de categorías y regiones |
| 09 | `1_datos/09_panel_diario` | Panel tienda × categoría × día: ceros comerciales, tienda inactiva, días sin datos, promociones |
| 10 | `1_datos/10_eda_series_y_linea_base` | Series y línea base ingenua con los cortes walk-forward fijos (desarrollo y prueba) |
| 11 | `1_datos/11_validacion_calidad` | Reglas de calidad de Bronce, Plata, panel y línea base (`mlops_validaciones`) |
| 20 | `2_modelos/20_features_oro` | Tabla Oro: una fila por serie × origen × horizonte, variables sólo con días ≤ origen y prueba anti-fuga |
| 21 | `2_modelos/21_avance4_modelos_alternativos` | Avance 4: torneo de modelos individuales y ajuste fino |
| — | `2_modelos/Avance5.43` | Avance 5: modelo final (ver `Avance5/README.md`) |

Los notebooks de auditoría del trabajo previo (`03`) y de evidencia de calidad para Bnext (`12`) contienen cifras del
negocio y sólo viven en el espacio privado de Databricks; no se publican en este repositorio.

## Configuración común

La primera celda de cada notebook es la misma: catálogo y esquema, `MESES` (un volumen por mes, de `2026-03` a
`2026-09`) y los **cortes walk-forward fijos**:

| Bloque | Origen | Uso |
| --- | --- | --- |
| `ajuste` | 2026-05-20 | ajuste de hiperparámetros |
| `corte_1` … `corte_4` | 06-03, 06-17, 07-01, 08-17 | desarrollo (los del Avance 4) |
| `prueba_1`, `prueba_2` | 08-31, 09-14 | datos no vistos (septiembre) |

Un corte sólo se usa si su bloque de 14 días y los 14 días previos tienen datos (el hueco de entrega del 16 al 31 de
julio queda fuera). Si llega un mes nuevo, se agrega a `MESES` y los cortes de desarrollo no cambian.

## Orden para incorporar un mes nuevo (septiembre)

1. `02_carga_bronce` con `MESES_A_CARGAR = ["2026-09"]`.
2. `08_limpieza_plata` con `MESES_A_LIMPIAR = MESES` (todos: la deduplicación compara entre meses).
3. `09_panel_diario` → `10_eda_series_y_linea_base` → `11_validacion_calidad` → `20_features_oro`.
4. `Avance5.43`: primero `MODO_RAPIDO = True`; después la corrida completa como Job.

`20_features_oro` declara en las propiedades de la tabla (`mna.s_promos_vigentes_t0 = causal_7d`) que la variable de
promociones vigentes se calcula sólo con días ≤ origen; el notebook del Avance 5 la usa únicamente si encuentra esa
propiedad.
