# Avance 6 · Producto de difusión: operación con MLOps, escenarios y tablero — Equipo 43

**Proyecto Integrador · Maestría en Inteligencia Artificial Aplicada (MNA) · Tecnológico de Monterrey**
**Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez
**Socio formador:** Tiendas SIX (Bnext) · **Semana 8**

El Avance 5 dejó un modelo final evaluado en datos no vistos. El Avance 6 lo convierte en un **sistema que opera**:
llega un mes → se valida → se construyen las variables → se pronostica con el modelo registrado → se reparte a
producto → se simulan promociones → se monitorea → se ve en un tablero. Todo dentro de Databricks (los datos no salen).

---

## 1. Qué se entrega

| Pieza | Dónde | Qué es |
| --- | --- | --- |
| Pipeline completo (25 notebooks) | `databricks/mna_equipo43_avance6/` (y `.zip`) | `1_datos` 00–13 · `2_modelos` 20, 21, Avance5.43 · `3_operacion` 22, 23, 24, 30, 31, 40 |
| Calendario de promociones editable | `calendario/` | Excel/CSV con placeholder + instrucciones; el notebook 13 lo convierte en tablas |
| Jobs de Workflows | `jobs/` | J1 ingesta · J2 features · J3 entrenamiento · J4 inferencia · J5 monitoreo (JSON + guía) |
| Tablero AI/BI | `dashboard/` | `mna_equipo43_demanda.lvdash.json` importable, consultas SQL y guía (7 vistas) |
| Documentación | `docs/` | Auditoría de código, runbook de operación, diccionario de datos, pendientes con Bnext |
| Pruebas locales | `pruebas/` | Generador de datos sintéticos y harness que corre toda la cadena en Spark local (sin datos de Bnext) |

## 2. Arquitectura de operación

```
Volúmenes de Bnext (parquet mensual)
  └─ J1 ingesta_mes:   02 Bronce → 08 Plata → 09 Panel → 10 Línea base → 11 Compuerta de datos (se detiene si falla)
      └─ J2 features:  13 Calendario de promociones (Excel editable → tablas) → 20 Tabla Oro (anti-fuga)
          ├─ J3 entrenamiento (mensual / por alerta): Avance5.43 → MLflow → Unity Catalog demanda_categoria @challenger
          │     └─ 24 promoción: compuerta técnica + prueba no vista + aprobaciones (mlops.aprobaciones) → @champion
          ├─ J4 inferencia (cada carga o cambio de calendario, minutos):
          │     22 pronóstico con @champion → gold.pronostico_14d (P10/P50/P90, techo)
          │     30 escenarios what-if (re-puntúa con el modelo) → gold.escenarios_whatif[_resumen]
          │     31 reparto a producto (participación 28 d, fila OTROS) → gold.predicciones_sku, gold.surtido_sku_semana
          │     40 tablas gold.dash_* → tablero AI/BI
          └─ J5 monitoreo (semanal): 23 acierto en producción, PSI, frescura → mlops.monitoreo_modelo, mlops.alertas
```

## 3. Novedades respecto al Avance 5

1. **Inferencia separada del entrenamiento** (`22`): pronosticar tarda minutos y usa el modelo registrado; antes
   obligaba a reentrenar 4–6 h.
2. **Calendario de promociones editable** (`13` + `calendario/`): el negocio captura el plan en un Excel; el modelo lo
   usa con variables `f_*` **sin fuga** (cada promoción cuenta sólo desde su `fecha_publicacion`). El histórico se
   construye con las promociones observadas (supuesto declarado: se conocían 14 días antes).
3. **What-if con el modelo** (`30`): cada escenario cambia las variables del calendario y re-puntúa; el uplift es lo que
   el modelo aprendió, no una elasticidad escrita a mano. Referencia empírica `gold.uplift_promocional_historico`
   declarada como cota superior.
4. **Reparto a producto corregido** (`31`): fila OTROS, reconciliación verificada, techo de demanda en unidades y vista
   semanal; ya no fabrica pronósticos si falta la tabla oficial.
5. **Monitoreo y alertas** (`23`): acierto en producción cuando llega la venta real, PSI, frescura, regla de reentrenar.
6. **Gobierno del modelo** (`24`): promoción `@challenger → @champion` con evidencia y aprobaciones registradas.
7. **Jobs parametrizados**: los mismos notebooks corren en interactivo o en Workflows sin editar celdas.
8. **Calidad como contrato**: constraints Delta, comentarios por columna (diccionario en el catálogo), publicación
   idempotente, bitácoras de corridas y pronósticos.
9. **Tablero AI/BI** de siete vistas sobre tablas precalculadas.

## 4. Cómo correrlo (demostración de punta a punta)

| Paso | Qué | Tiempo |
| --- | --- | --- |
| 0 | Importar `databricks/mna_equipo43_avance6.zip` en el workspace (o conectar el repositorio como Git folder) y subir `calendario/calendario_promociones_plan.xlsx` a `outputs/calendario/` del volumen (opcional: el 13 lo crea si no está) | 10 min |
| 1 | **J1 ingesta_mes** con `MES = 2026-09` (o correr 02 → 08 → 09 → 10 → 11 a mano) | 1.5–4 h |
| 2 | **J2 features** (13 → 20) | 40–90 min |
| 3 | **J3 entrenamiento** (`Avance5.43` completo). Antes, una prueba de humo con `MODO_RAPIDO = True` | 4–6 h |
| 4 | `24_promocion_modelo` con `ACCION = evaluar` (y `promover` si procede) | 2 min |
| 5 | **J4 inferencia** (22 → 30 → 31 → 40) | 30–80 min |
| 6 | **J5 monitoreo** (23 → 40) | 10–20 min |
| 7 | Importar el tablero (`dashboard/LEEME.md`) y publicarlo | 10 min |

Detalles de cada Job y parámetros: `jobs/LEEME.md`. Qué hacer en cada situación: `docs/RUNBOOK.md`.

## 5. Valor para el negocio (lo que el tablero responde)

- **Resumen**: cuánto se venderá en 14 días y cómo se compara con las dos semanas anteriores, por región y tipo.
- **Pronóstico**: historia de 8 semanas y 14 días con banda P10–P90 hasta tienda × categoría.
- **Precisión**: WMAPE, sesgo y cobertura (prueba de septiembre y producción) y comparativa contra el ingenuo.
- **Surtido**: productos y unidades que cubren la demanda (P50 y techo P90) por tienda × categoría × semana.
- **Promociones**: plan vigente con su aporte estimado y escenarios de 10 / 20 / 30 % de descuento.
- **Alertas**: series con caída o subida fuerte, alta incertidumbre, tiendas inactivas, deriva del modelo.
- **Estado**: frescura de datos, modelo vigente, compuerta, última corrida de cada notebook.

## 6. Límites declarados

- El calendario trae un **placeholder inventado**: las cifras del what-if sólo valen con el plan real de Bnext.
- El histórico de promociones «planeadas» se aproxima con la vigencia observada (supuesto: se conocían 14 días antes).
- El reparto a producto es *top-down* por participación: P10/P90 por producto son asignaciones consistentes, no cuantiles
  propios; sin inventario, el techo de demanda no es un pedido sugerido.
- El acierto en producción se mide cuando llegue octubre; hasta entonces la referencia es la prueba de septiembre.
- El registro en Unity Catalog y los Jobs dependen de permisos del workspace (`docs/PENDIENTES_BNEXT.md`).

## 7. Confidencialidad

Este repositorio es público: contiene sólo código y documentación. Los notebooks imprimen métricas relativas
(`MODO_PUBLICO = True`); las tablas con importes viven en el catálogo privado y el tablero se comparte dentro del
workspace. `03_auditoria_trabajo_previo` y `12_evidencia_calidad` no se publican.
