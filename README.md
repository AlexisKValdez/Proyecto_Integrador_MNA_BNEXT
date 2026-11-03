# Proyecto Integrador MNA · Pronóstico de demanda para Tiendas SIX (Bnext) — Equipo 43

**Maestría en Inteligencia Artificial Aplicada (MNA) · Tecnológico de Monterrey**
**Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez

Pronóstico de la venta diaria en pesos de cada **tienda × categoría** para los **14 días siguientes**, construido sobre
los datos de punto de venta de Tiendas SIX en Azure Databricks: limpieza en capas (Bronce → Plata → Oro), modelos de
aprendizaje automático y ensambles, evaluación en datos no vistos y **operación con MLOps**: datos versionados,
compuerta de calidad, registro de modelos, inferencia separada del entrenamiento, monitoreo con alertas, escenarios de
promoción, reparto a producto y tablero.

> **Confidencialidad.** Este repositorio contiene sólo código y documentación. Los datos de Bnext viven en el
> catálogo privado de Databricks y nunca se suben aquí. Los notebooks muestran sus salidas sin importes ni
> volúmenes (`MODO_PUBLICO = True`).

---

## Estructura

```
.
├── README.md
├── Semana 5 - Avance 3_ … (Equipo 43).pdf      ← Avance 3: línea base y validación walk-forward
├── Semana 6 - Avance 4_ … (Equipo 43).pdf      ← Avance 4: modelos alternativos
├── Avance5/                                    ← Avance 5: modelo final (ensambles, prueba no vista)
│   ├── Avance5.43.ipynb · Avance5.43.py · README.md
├── Avance6/                                    ← Avance 6: operación con MLOps, escenarios y tablero
│   └── README.md                               ← qué se entrega, arquitectura, cómo correrlo, límites
├── databricks/                                 ← pipeline completo para Databricks
│   ├── README.md                               ← notebooks, parámetros y orden
│   ├── mna_equipo43_avance6/                   ← 1_datos (00–13) · 2_modelos (20, 21, Avance5.43) · 3_operacion (22–40)
│   └── mna_equipo43_avance6.zip                ← los mismos notebooks para importar de una vez
├── calendario/                                 ← calendario de promociones editable (Excel/CSV) + instrucciones
├── jobs/                                       ← definiciones JSON de los 5 Jobs de Workflows + guía
├── dashboard/                                  ← tablero AI/BI importable (.lvdash.json), consultas SQL y guía
├── docs/                                       ← auditoría de código, runbook, diccionario de datos, pendientes con Bnext
├── pruebas/                                    ← datos sintéticos y harness para correr todo en Spark local
└── herramientas/                               ← conversión .py → .ipynb, generadores de plantilla, tablero y diccionario
```

## Avances

| Avance | Tema | Dónde |
| --- | --- | --- |
| 3 | Línea base ingenua y validación walk-forward | PDF en la raíz |
| 4 | Modelos alternativos (torneo de modelos individuales y ajuste fino) | PDF en la raíz · `databricks/…/2_modelos/21_avance4_modelos_alternativos.py` |
| 5 | Modelo final: ensambles, prueba en datos no vistos (septiembre), intervalos, compuerta y pronóstico operativo | [`Avance5/`](Avance5/) |
| 6 | Operación con MLOps: Jobs, inferencia, monitoreo, promoción de modelo, calendario de promociones, what-if, reparto a producto y tablero | [`Avance6/`](Avance6/) |

## Arquitectura en Databricks

```
volúmenes mensuales (parquet)
  └─ 02 Bronce → 08 Plata → 09 Panel diario → 10 Línea base → 11 Compuerta de datos          (Job J1)
      └─ 13 Calendario de promociones (Excel editable) → 20 Tabla Oro (anti-fuga)             (Job J2)
          ├─ Avance5.43: ajuste fino, ensambles, prueba no vista, modelo final → UC @challenger (Job J3) → 24 → @champion
          ├─ 22 Pronóstico con el modelo registrado → 30 What-if → 31 Reparto a producto → 40 Tablero (Job J4)
          └─ 23 Monitoreo: acierto en producción, deriva, alertas (Job J5)
```

## Cómo usar este repositorio en Databricks

**Opción recomendada (Git folder).** Workspace → **Create** → **Git folder** → URL de este repositorio. Los notebooks
quedan versionados y los Jobs apuntan a `databricks/mna_equipo43_avance6/…`.

**Opción rápida (importar).** Workspace → **Import** → `databricks/mna_equipo43_avance6.zip`.

Después: `databricks/README.md` (orden y parámetros), `jobs/LEEME.md` (Jobs), `dashboard/LEEME.md` (tablero) y
`docs/RUNBOOK.md` (operación).
