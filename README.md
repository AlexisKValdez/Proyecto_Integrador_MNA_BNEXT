# Proyecto Integrador MNA · Pronóstico de demanda para Tiendas SIX (Bnext) — Equipo 43

**Maestría en Inteligencia Artificial Aplicada (MNA) · Tecnológico de Monterrey**
**Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez

Pronóstico de la venta diaria en pesos de cada **tienda × categoría** para los **14 días siguientes**, construido sobre
los datos de punto de venta de Tiendas SIX en Azure Databricks: limpieza en capas (Bronce → Plata → Oro), modelos de
aprendizaje automático y ensambles, evaluación en datos no vistos y prácticas de MLOps (versiones fijas de datos,
MLflow, Unity Catalog, compuerta de aceptación, pronóstico operativo y monitoreo).

> **Confidencialidad.** Este repositorio contiene sólo código, documentación y entregables. Los datos de Bnext viven
> en el catálogo privado de Databricks y nunca se suben aquí. Los notebooks muestran sus salidas sin importes ni
> volúmenes (`MODO_PUBLICO = True`).

---

## Estructura

```
.
├── README.md                                   ← este archivo
├── Semana 5 - Avance 3_ … (Equipo 43).pdf      ← Avance 3: línea base y validación walk-forward
├── Semana 6 - Avance 4_ … (Equipo 43).pdf      ← Avance 4: modelos alternativos
├── Avance5/                                    ← Avance 5: modelo final (ensambles)
│   ├── Avance5.43.ipynb                        ← libreta del entregable (se reemplaza por la versión ejecutada)
│   ├── Avance5.43.py                           ← la misma libreta en formato fuente de Databricks
│   └── README.md                               ← qué hace, cómo correrla, cambios y rúbrica
├── databricks/                                 ← pipeline completo para Databricks
│   ├── README.md                               ← orden de ejecución y qué hace cada notebook
│   ├── mna_equipo43/                           ← notebooks (se importan en /Workspace/Shared/mna_equipo43)
│   │   ├── LEEME.py · ESTADO_DEL_PROYECTO.py
│   │   ├── 1_datos/                            ← 00 entorno … 11 validación de calidad
│   │   └── 2_modelos/                          ← 20 tabla Oro · 21 Avance 4 · Avance5.43
│   └── mna_equipo43.zip                        ← los mismos notebooks para importar de una vez
└── herramientas/
    └── py_a_ipynb.py                           ← convierte un notebook fuente de Databricks a .ipynb
```

## Avances

| Avance | Tema | Dónde |
| --- | --- | --- |
| 3 | Línea base ingenua y validación walk-forward | PDF en la raíz |
| 4 | Modelos alternativos (torneo de modelos individuales y ajuste fino) | PDF en la raíz · `databricks/mna_equipo43/2_modelos/21_avance4_modelos_alternativos.py` |
| 5 | Modelo final: ensambles homogéneos y heterogéneos, prueba en datos no vistos (septiembre), intervalos, compuerta y pronóstico operativo | [`Avance5/`](Avance5/) |

## Arquitectura en Databricks

```
volúmenes mensuales (parquet)
  └─ 02 Bronce: carga fiel con trazabilidad de archivo
      └─ 08 Plata: reglas de rango POS, deduplicación, homologación de categorías y regiones
          └─ 09 Panel diario tienda × categoría (ceros comerciales vs. tienda inactiva, vigencia de promociones)
              ├─ 10 Línea base ingenua (cortes walk-forward fijos)  ·  11 Reglas de calidad
              └─ 20 Tabla Oro: una fila por serie × origen × horizonte, variables sólo con días ≤ origen, prueba anti-fuga
                  └─ Avance5.43: ajuste fino, ensambles, selección, prueba no vista, intervalos, compuerta
                      ├─ MLflow (parámetros, métricas, gráficas, modelo pyfunc) → Unity Catalog demanda_categoria @challenger
                      ├─ gold.pronostico_14d (pronóstico operativo con P10/P50/P90)
                      └─ mlops.monitoreo_modelo (error, sesgo y cobertura por semana × región × tipo)
```

## Cómo usar este repositorio en Databricks

**Opción recomendada (Git folder).** Workspace → **Create** → **Git folder** → URL de este repositorio. Los notebooks
quedan versionados: después de cambiarlos se hace *commit* y *push* desde Databricks.

**Opción rápida (importar).** Workspace → Shared → **Import** → `databricks/mna_equipo43.zip`; y para el entregable,
**Import** → `Avance5/Avance5.43.py`.

El orden de ejecución y los parámetros de cada notebook están en [`databricks/README.md`](databricks/README.md) y en
[`Avance5/README.md`](Avance5/README.md).
