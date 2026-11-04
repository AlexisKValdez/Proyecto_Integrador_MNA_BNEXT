
# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Salida operativa y monitoreo
# MAGIC **Pronóstico operativo.** El modelo final, reentrenado con todos los datos, pronostica los 14 días siguientes al
# MAGIC último día con datos para cada tienda × categoría evaluable y lo guarda en `gold.pronostico_14d` con el pronóstico
# MAGIC puntual, P10, P50, P90 y el techo operativo (P90 por defecto). Cada corrida reemplaza sólo su origen, así que la tabla
# MAGIC conserva la historia de pronósticos para medir después su acierto. En `MODO_RAPIDO` no se escribe.
# MAGIC
# MAGIC **Monitoreo.** `mlops.monitoreo_modelo` guarda, por semana × región × tipo de categoría, el WMAPE, el sesgo y la
# MAGIC cobertura del modelo final con alertas: error 5 puntos por encima del de desarrollo, |sesgo| > 5 % o cobertura fuera
# MAGIC de 75–85 %. En operación, un Job semanal compara `gold.pronostico_14d` contra la venta real y agrega filas a esta
# MAGIC tabla; dos semanas seguidas con alerta disparan el reentrenamiento (además del reentrenamiento mensual programado).

# COMMAND ----------

titulo("8. SALIDA OPERATIVA Y MONITOREO")
OPERATIVO_C = json.loads(RESUMEN_C["salida_operativa"]) if isinstance(RESUMEN_C.get("salida_operativa"), str) else {}
if OPERATIVO_C:
    mostrar_publico(pd.DataFrame([OPERATIVO_C]), "Pronóstico operativo de esta corrida")
    if not MODO_RAPIDO and existe(TABLA_PRONOSTICO):
        rev = (spark.table(TABLA_PRONOSTICO).where(F.col("run_id") == C["run_id"])
               .agg(F.avg(((F.col("p10") <= F.col("prediccion_puntual")) & (F.col("prediccion_puntual") <= F.col("p90"))).cast("double")).alias("dentro"),
                    F.expr("percentile_approx((p90 - p10) / greatest(prediccion_puntual, 1.0), 0.5)").alias("ancho_relativo_mediano"),
                    F.min("fecha_objetivo").alias("desde"), F.max("fecha_objetivo").alias("hasta")).collect()[0].asDict())
        mostrar(pd.DataFrame([rev]), "Revisión de la tabla de pronóstico (sin importes)", formatos={"dentro": "pct", "ancho_relativo_mediano": "dec2"})
        hallazgo(f"Pronóstico operativo guardado en {TABLA_PRONOSTICO}: {rev['desde']} a {rev['hasta']}; el {rev['dentro']:.0%} de las filas cumple "
                 f"P10 ≤ pronóstico ≤ P90; techo operativo = {NIVEL_TECHO_OPERATIVO.upper()}.")
else:
    print("La Oro no tiene conjunto 'pronostico' (o no se reentrenó el modelo final): no hay salida operativa en esta corrida.")
if existe(TABLA_MONITOREO):
    MON = spark.table(TABLA_MONITOREO).where(F.col("run_id") == C["run_id"]).toPandas()
    if len(MON):
        alertas = MON[["alerta_error", "alerta_sesgo", "alerta_cobertura"]].sum()
        mostrar_publico(MON.sort_values(["semana", "region", "tipo_categoria"]).head(30), "Monitoreo por semana × región × tipo (prueba)",
                        formatos={"wmape": "dec4", "sesgo_pct": "dec2", "cobertura": "pct"})
        hallazgo(f"Monitoreo de la prueba no vista: {len(MON)} combinaciones semana × región × tipo; alertas de error {int(alertas['alerta_error'])}, "
                 f"de sesgo {int(alertas['alerta_sesgo'])} y de cobertura {int(alertas['alerta_cobertura'])}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calendario de promociones planeadas
# MAGIC **Lo que el modelo veía antes.** Sólo promociones **pasadas**: `s_tasa_descuento_28`, `s_intensidad_promo_28` y
# MAGIC `s_promos_en_venta_28` promedian los 28 días anteriores a T y `s_promos_vigentes_t0` cuenta las promociones con venta
# MAGIC en la última semana (≤ T). Ninguna mira los 14 días que se pronostican, así que una promoción que empieza dentro de
# MAGIC la ventana no se anticipaba.
# MAGIC
# MAGIC **Lo que ve ahora.** Con `USAR_CALENDARIO_PROMOS = True` entran `f_promos_planeadas`, `f_descuento_planeado` y
# MAGIC `f_dias_promo_restantes`, calculadas desde `silver.calendario_promociones_plan_dia`. Esa tabla la escribe
# MAGIC `13_calendario_promociones_plan` a partir de dos fuentes: (1) el archivo editable
# MAGIC `outputs/calendario/calendario_promociones_plan.xlsx` del volumen (promociones futuras: hoy un *placeholder*
# MAGIC inventado que se sustituye por el plan de Bnext editando el Excel) y (2) la vigencia observada de las promociones
# MAGIC pasadas (`silver.calendario_promociones` del 09) como aproximación del plan histórico, suponiendo que el plan se
# MAGIC conocía `DIAS_ANTICIPACION_HISTORICO` días antes de empezar. Esa suposición queda declarada en la tabla (`fuente`).
# MAGIC
# MAGIC **Sin fuga.** Cada promoción trae `fecha_publicacion`; en cada origen T sólo cuentan las publicadas en o antes de T.
# MAGIC Por eso las variables `f_*` son legítimas en los cortes históricos y en la prueba no vista (septiembre–octubre).
# MAGIC
# MAGIC **Para actualizar el plan**: editar el Excel, volver a subirlo al volumen y correr `13`; el pronóstico operativo
# MAGIC (`22_pronostico_operativo`) y el what-if (`30`) toman el plan nuevo sin reentrenar.

# COMMAND ----------

titulo("9. CALENDARIO DE PROMOCIONES PLANEADAS")
mostrar(pd.DataFrame([{"columna": k, "tipo y significado": v} for k, v in ESQUEMA_CALENDARIO_PROMOS.items()]),
        "Contrato del archivo editable (hoja promociones)")
print(f"USAR_CALENDARIO_PROMOS = {USAR_CALENDARIO_PROMOS} → ", end="")
print("variables f_* EN USO por los modelos." if FUENTE_AV5._promos_futuras is not None else "variables f_* NO usadas en esta corrida.")
mostrar(pd.DataFrame([FUENTE_AV5.info_calendario]), "Estado del calendario (tabla tienda × categoría × día)")
if existe(TABLA_CALENDARIO_PLAN):
    plan = spark.table(TABLA_CALENDARIO_PLAN)
    revision = validar_calendario_promociones(plan)
    mostrar(pd.DataFrame([revision]), "Revisión del plan (promoción × alcance)")
    por_fuente = plan.groupBy("fuente").agg(F.countDistinct("promo_id").alias("promociones"), F.min("inicio_planeado").alias("desde"),
                                           F.max("fin_planeado").alias("hasta")).orderBy("fuente").toPandas()
    mostrar(por_fuente, "Promociones por fuente (placeholder = inventadas, sustituir por el plan de Bnext)")
    if (por_fuente["fuente"] == "placeholder").any():
        aviso("El calendario contiene promociones placeholder (inventadas): sustituirlas por el plan real de Bnext en el Excel del volumen.")
else:
    print(f"Todavía no existe {TABLA_CALENDARIO_PLAN}: corre 13_calendario_promociones_plan para crearla desde el Excel del volumen.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Resumen para el reporte
# MAGIC Todo lo que se necesita para redactar el reporte del Avance 5, sin importes ni volúmenes.

# COMMAND ----------

titulo("10. RESUMEN PARA EL REPORTE (sólo métricas relativas)")
print(f"Corrida {C['run_id']} · modo {RESUMEN_C['modo']} · tabla Oro versión {RESUMEN_C['version_oro']} · "
      f"{int(RESUMEN_C['cortes'])} cortes de desarrollo · {int(RESUMEN_C['bloques_prueba'])} bloques de prueba · "
      f"{float(RESUMEN_C['minutos_modelado']):.0f} min de modelado")
print("\nComparativa de desarrollo (ordenada por WMAPE medio de los cortes):")
print(COMP[["posicion", "etiqueta", "tipo", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo", "sesgo_medio_pct",
            "wmape_mercancia", "min_entrenamiento_por_corte"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if HAY_PRUEBA:
    print("\nPrueba no vista (septiembre–octubre):")
    print(NV.drop(columns=["modelo"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if "ajuste_resumen" in C:
    print("\nAjuste fino:")
    print(C["ajuste_resumen"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nMatriz de decisión:")
print(DEC.drop(columns=[c for c in ("modelo_final", "argumento") if c in DEC]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n{ARGUMENTO_FINAL}")
if "cuantiles" in C:
    print("\nIntervalos (prueba no vista):")
    print(GQ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if COMPUERTA_C:
    print(f"\nCompuerta: {COMPUERTA_C['desarrollo_status']} · registro: {COMPUERTA_C['registry_action']} · motivos de bloqueo: "
          + "; ".join(COMPUERTA_C["blocking_reasons"]))
if len(C.get("importancia", pd.DataFrame())):
    print("\nVariables más importantes: " + ", ".join(f"{r.variable} {r.importancia:.1%}" for r in C["importancia"].head(8).itertuples()))
for i in INTERPRETACIONES:
    print(f"\nGráfico {i['grafico']} · {i['titulo']}: {i['interpretacion']}")
cerrar("Avance5.43")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Declaración sobre el uso de herramientas de inteligencia artificial
# MAGIC *Celda obligatoria del curso. Sustituir este texto por la plantilla oficial con la información del equipo antes de
# MAGIC entregar.*
# MAGIC
# MAGIC | Herramienta | Versión / proveedor | Para qué se usó | En qué partes | Cómo se verificó |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | *(completar)* | *(completar)* | *(completar)* | *(completar)* | *(completar)* |
# MAGIC
# MAGIC *(Completar con la declaración del equipo.)*
