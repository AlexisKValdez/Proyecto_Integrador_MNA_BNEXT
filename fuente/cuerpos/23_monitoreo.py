# Databricks notebook source
# MAGIC %md
# MAGIC # 23 · Monitoreo del modelo en operación
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Cierra el ciclo de MLOps: cuando llega la venta real de los días ya pronosticados, mide cuánto acertó el pronóstico
# MAGIC guardado (`gold.pronostico_14d`), vigila que las variables no se hayan movido respecto al entrenamiento (deriva) y
# MAGIC deja **alertas** que dicen cuándo reentrenar. Corre después de cada carga de datos (Job `J5_monitoreo`).
# MAGIC
# MAGIC 1. **Acierto en producción.** Pronósticos con venta real disponible → WMAPE, sesgo y cobertura del intervalo P10–P90 por
# MAGIC    semana × región × tipo de categoría (misma tabla y reglas que la fase de prueba del Avance 5) y por horizonte.
# MAGIC 2. **Deriva de variables (PSI).** Distribución de las variables principales en el conjunto que se pronostica contra la
# MAGIC    muestra de entrenamiento de la Oro: PSI > 0.20 = alerta, > 0.10 = aviso.
# MAGIC 3. **Frescura.** Días desde la última venta cargada y si el pronóstico operativo ya quedó atrás.
# MAGIC 4. **Alertas** en `mlops.alertas` con la regla, el valor y el umbral; dos semanas seguidas con alerta de error o sesgo
# MAGIC    = reentrenar (además del reentrenamiento mensual).
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.pronostico_14d` · `silver.panel_categoria_dia` · `gold.features_origen` · `mlops.av5_comparativa` · `mlops.monitoreo_modelo` |
# MAGIC | **Escribe** | `mlops.monitoreo_modelo` (fase `produccion`) · `mlops.monitoreo_horizonte` · `mlops.deriva_variables` · `mlops.alertas` |
# MAGIC | **Parámetros** | `UMBRAL_ERROR_PP`, `UMBRAL_SESGO_PCT`, `BANDA_COBERTURA`, `UMBRAL_PSI_ALERTA`, `DIAS_FRESCURA` |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
UMBRAL_ERROR_PP = parametro("UMBRAL_ERROR_PP", 0.05)       # alerta si el WMAPE supera al de desarrollo por más de 5 puntos
UMBRAL_SESGO_PCT = parametro("UMBRAL_SESGO_PCT", 5.0)      # alerta si |sesgo| > 5 %
BANDA_COBERTURA = (0.75, 0.85)                             # cobertura esperada del intervalo P10–P90 (80 % ± 5)
UMBRAL_PSI_AVISO, UMBRAL_PSI_ALERTA = 0.10, parametro("UMBRAL_PSI_ALERTA", 0.20)
DIAS_FRESCURA = parametro("DIAS_FRESCURA", 7)              # aviso si la venta más reciente tiene más de 7 días
VARIABLES_DERIVA = ["s_ma7", "s_ma28", "s_ma56", "s_desv28", "s_tasa_venta_28", "s_tasa_descuento_28", "s_intensidad_promo_28",
                    "s_promos_en_venta_28", "s_part_tienda_28", "s_dias_desde_ultima_venta", "s_lag_dow_1", "s_media_dow_4"]
# PSI sólo en variables que varían entre series. Las de contexto (tienda, zona, país) toman un valor por tienda o por
# categoría y origen: ahí se vigila el cambio de la media, no la distribución. dias_historia y antiguedad_tienda crecen
# con el calendario por construcción y no se vigilan.
VARIABLES_CONTEXTO = ["t_ma28", "t_tickets_ma28", "zc_ma28", "pc_ma28"]
UMBRAL_CAMBIO_CONTEXTO = parametro("UMBRAL_CAMBIO_CONTEXTO", 0.20)   # aviso si la media cambia más de 20 %
DIAS_REFERENCIA_DERIVA = parametro("DIAS_REFERENCIA_DERIVA", 56)
# la referencia son las filas de entrenamiento con origen en las últimas 8 semanas antes del origen pronosticado: compara
# el régimen reciente que vio el modelo contra lo que se pronostica (deriva real), no el crecimiento natural de la serie
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_PANEL = tabla("silver", "panel_categoria_dia")
TABLA_ORO = tabla("gold", "features_origen")
TABLA_MONITOREO = tabla("mlops", "monitoreo_modelo")
TABLA_MONITOREO_H = tabla("mlops", "monitoreo_horizonte")
TABLA_DERIVA = tabla("mlops", "deriva_variables")
TABLA_ALERTAS = tabla("mlops", "alertas")
AHORA = pd.Timestamp.now()
ALERTAS = []


def alerta(tipo, severidad, mensaje, valor=None, umbral=None, origen=None, run_id=None):
    ALERTAS.append({"fecha": AHORA, "tipo": tipo, "severidad": severidad, "mensaje": mensaje, "valor": None if valor is None else float(valor),
                    "umbral": None if umbral is None else float(umbral), "origen": None if origen is None else str(origen), "run_id": run_id,
                    "atendida": False})
    (aviso if severidad == "alta" else print)(f"[{severidad.upper()}] {tipo}: {mensaje}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pronósticos con venta real disponible

# COMMAND ----------

titulo("1. ACIERTO EN PRODUCCIÓN")
PANEL = leer_tabla(TABLA_PANEL, "09_panel_diario")
ULTIMO_DIA_VENTA = PANEL.where(F.col("sin_datos") == 0).agg(F.max("fecha")).collect()[0][0]
if not existe(TABLA_PRONOSTICO):
    aviso(f"No existe {TABLA_PRONOSTICO}: corre 22_pronostico_operativo (o Avance5.43 con REENTRENAR_FINAL = True). No hay nada que monitorear.")
    PRON = None
else:
    PRON = spark.table(TABLA_PRONOSTICO)
    PRON = PRON.withColumn("fase", F.coalesce(F.col("fase"), F.lit("operativo")) if "fase" in PRON.columns else F.lit("operativo"))
    resumen_origenes = (PRON.groupBy("origen", "fase", "modelo", "run_id").agg(F.count("*").alias("filas"), F.min("fecha_objetivo").alias("desde"),
                                                                              F.max("fecha_objetivo").alias("hasta")).orderBy("origen").toPandas())
    resumen_origenes["venta_real_disponible"] = resumen_origenes["hasta"] <= ULTIMO_DIA_VENTA
    mostrar(resumen_origenes, f"Pronósticos guardados (última venta cargada: {ULTIMO_DIA_VENTA})", formatos={"filas": "int"})
EVAL = None
if PRON is not None:
    real = (PANEL.where((F.col("sin_datos") == 0) & (F.col("tienda_inactiva") == 0) & F.col("venta").isNotNull())
            .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                    F.col("fecha").alias("fecha_objetivo"), F.col("venta").alias("real")))
    EVAL = (PRON.where(F.col("fecha_objetivo") <= F.lit(ULTIMO_DIA_VENTA))
            .withColumn("tienda", F.col("tienda").cast("string")).withColumn("categoria_id", F.col("categoria_id").cast("string"))
            .join(real, ["tienda", "categoria_id", "fecha_objetivo"], "inner")
            .withColumn("semana", F.date_trunc("week", F.col("fecha_objetivo")).cast("date"))
            .withColumn("h_bloque", F.when(F.col("h") <= 7, 1).otherwise(2))
            .withColumn("err_abs", F.abs(F.col("real") - F.col("prediccion_puntual")))
            .withColumn("cubierto", ((F.col("real") >= F.col("p10")) & (F.col("real") <= F.col("p90"))).cast("double")))
    n_eval = EVAL.count()
    if n_eval == 0:
        print("Todavía no hay pronósticos cuyos días ya tengan venta real: el acierto en producción se medirá cuando llegue el mes siguiente "
              "(J1 → J5). Mientras tanto la referencia es la fase de prueba (septiembre) del Avance 5.")
        EVAL = None
    else:
        print(f"{n_eval:,} filas pronosticadas ya tienen venta real.")

# COMMAND ----------

# ---- referencia: WMAPE de desarrollo del modelo (de la corrida que lo entrenó)
def wmape_referencia(modelo, run_id):
    t = tabla("mlops", "av5_comparativa")
    if not existe(t):
        return None
    c = spark.table(t).where(F.col("modelo") == modelo)
    if run_id:
        c2 = c.where(F.col("run_id") == run_id)
        if c2.limit(1).count():
            c = c2
    fila = c.orderBy(F.col("fecha_publicacion").desc()).select("wmape_media_cortes").limit(1).collect()
    return float(fila[0][0]) if fila else None


MON_PROD, MON_H = pd.DataFrame(), pd.DataFrame()
if EVAL is not None:
    agg = lambda df, dims: (df.groupBy(*dims).agg(F.count("*").alias("n_filas"), F.sum("real").alias("ventas_reales"), F.sum("prediccion_puntual").alias("ventas_predichas"),
                                                  F.sum("err_abs").alias("err_abs"), F.avg("cubierto").alias("cobertura")).toPandas())
    def metricas(m):
        m["wmape"] = m["err_abs"] / m["ventas_reales"].replace(0, np.nan)
        m["sesgo_pct"] = (m["ventas_predichas"] / m["ventas_reales"].replace(0, np.nan) - 1) * 100
        return m
    MON_PROD = metricas(agg(EVAL, ["origen", "fase", "modelo", "run_id", "semana", "region", "tipo_categoria"]))
    MON_H = metricas(agg(EVAL, ["origen", "fase", "modelo", "run_id", "h"]))
    GLOBAL = metricas(agg(EVAL, ["origen", "fase", "modelo", "run_id"]))
    refs = {(r.modelo, r.run_id): wmape_referencia(r.modelo, r.run_id) for r in GLOBAL[["modelo", "run_id"]].drop_duplicates().itertuples()}
    GLOBAL["wmape_desarrollo"] = [refs.get((r.modelo, r.run_id)) for r in GLOBAL.itertuples()]
    MON_PROD["wmape_desarrollo"] = [refs.get((r.modelo, r.run_id)) for r in MON_PROD.itertuples()]
    MON_PROD["alerta_error"] = MON_PROD["wmape"] > MON_PROD["wmape_desarrollo"].fillna(0.553) + UMBRAL_ERROR_PP
    MON_PROD["alerta_sesgo"] = MON_PROD["sesgo_pct"].abs() > UMBRAL_SESGO_PCT
    MON_PROD["alerta_cobertura"] = ~MON_PROD["cobertura"].between(*BANDA_COBERTURA)
    MON_PROD["fase"] = "produccion_" + MON_PROD["fase"].astype(str)
    MON_PROD["fecha_calculo"] = AHORA
    mostrar(GLOBAL.drop(columns=["ventas_reales", "ventas_predichas", "err_abs"]), "Acierto global por origen pronosticado",
            formatos={"n_filas": "int", "wmape": "dec4", "sesgo_pct": "dec2", "cobertura": "pct", "wmape_desarrollo": "dec4"})
    mostrar(MON_PROD.drop(columns=["ventas_reales", "ventas_predichas", "err_abs"]).sort_values(["origen", "semana", "region", "tipo_categoria"]),
            "Por semana × región × tipo", formatos={"n_filas": "int", "wmape": "dec4", "sesgo_pct": "dec2", "cobertura": "pct", "wmape_desarrollo": "dec4"}, max_filas=40)
    mostrar(MON_H.drop(columns=["ventas_reales", "ventas_predichas", "err_abs"]).sort_values(["origen", "h"]), "Por horizonte",
            formatos={"n_filas": "int", "wmape": "dec4", "sesgo_pct": "dec2", "cobertura": "pct"}, max_filas=30)
    for r in GLOBAL.itertuples():
        ref = r.wmape_desarrollo if pd.notna(r.wmape_desarrollo) else 0.553
        if r.wmape > ref + UMBRAL_ERROR_PP:
            alerta("error", "alta", f"WMAPE {r.wmape:.4f} del origen {r.origen} supera el de desarrollo ({ref:.4f}) por más de {UMBRAL_ERROR_PP:.2f}: revisar y reentrenar.",
                   r.wmape, ref + UMBRAL_ERROR_PP, r.origen, r.run_id)
        if abs(r.sesgo_pct) > UMBRAL_SESGO_PCT:
            alerta("sesgo", "alta", f"Sesgo {r.sesgo_pct:+.1f} % del origen {r.origen} fuera de ±{UMBRAL_SESGO_PCT:.0f} %.", r.sesgo_pct, UMBRAL_SESGO_PCT, r.origen, r.run_id)
        if not (BANDA_COBERTURA[0] <= r.cobertura <= BANDA_COBERTURA[1]):
            alerta("cobertura", "media", f"Cobertura {r.cobertura:.1%} del intervalo P10–P90 en el origen {r.origen} fuera de la banda 75–85 %: recalibrar intervalos.",
                   r.cobertura, BANDA_COBERTURA[1], r.origen, r.run_id)
        hallazgo(f"Origen {r.origen} ({r.fase}): WMAPE {r.wmape:.4f} (desarrollo {ref:.4f}), sesgo {r.sesgo_pct:+.1f} %, cobertura {r.cobertura:.1%} en {r.n_filas:,} filas.")
    # persistir (idempotente por origen + fase)
    for t, df_ in ((TABLA_MONITOREO, MON_PROD), (TABLA_MONITOREO_H, MON_H.assign(fecha_calculo=AHORA))):
        if existe(t):
            for o in df_["origen"].unique():
                try:
                    spark.sql(f"DELETE FROM {t} WHERE origen = DATE'{o}' AND fase LIKE 'produccion%'")
                except Exception as e:
                    print(f"(no se pudo limpiar {t} para {o}: {type(e).__name__})")
        guardar(df_, "mlops", t.split(".")[-1].replace("mlops_", ""), "Acierto del pronostico en produccion (23_monitoreo)", modo="append")
else:
    ultimo_prueba = spark.table(TABLA_MONITOREO).where(F.col("fase") == "prueba") if existe(TABLA_MONITOREO) else None
    if ultimo_prueba is not None and ultimo_prueba.limit(1).count():
        rp = ultimo_prueba.groupBy("modelo", "run_id").agg(F.sum("n_filas").alias("n"), F.avg("wmape").alias("wmape_medio"),
                                                              F.avg("cobertura").alias("cobertura_media")).orderBy(F.desc("n")).limit(3).toPandas()
        mostrar(rp, "Última referencia disponible: fase de prueba (septiembre) del Avance 5", formatos={"n": "int", "wmape_medio": "dec4", "cobertura_media": "pct"})
        hallazgo("Sin pronósticos operativos con venta real todavía; la referencia vigente es la prueba de septiembre del Avance 5.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Deriva de variables (PSI) entre el entrenamiento y lo que se pronostica

# COMMAND ----------

titulo("2. DERIVA DE VARIABLES")
DERIVA = pd.DataFrame()
if existe(TABLA_ORO):
    ORO = spark.table(TABLA_ORO)
    o_pron = ORO.where(F.col("conjunto") == "pronostico").agg(F.max("origen")).collect()[0][0]
    if o_pron is None:
        print("La Oro no tiene conjunto 'pronostico': no se mide deriva.")
    else:
        variables = [v for v in VARIABLES_DERIVA if v in ORO.columns]
        contexto = [v for v in VARIABLES_CONTEXTO if v in ORO.columns]
        desde_ref = (pd.Timestamp(o_pron) - pd.Timedelta(days=DIAS_REFERENCIA_DERIVA)).date()
        ventana = (F.col("h") == 1) & (F.col("origen") >= F.lit(desde_ref)) & (F.col("origen") < F.lit(o_pron))
        # referencia: los bloques de evaluación recientes (todas las series, misma población que el pronóstico); si no
        # los hay, la muestra de entrenamiento reciente (estratificada: puede diferir en composición) y, si no, toda
        ref, origen_ref = ORO.where((F.col("conjunto") == "evaluacion") & ventana).select(*variables, *contexto), "evaluación reciente"
        if ref.limit(1).count() == 0:
            ref, origen_ref = ORO.where((F.col("conjunto") == "entrenamiento") & ventana).select(*variables, *contexto), "entrenamiento reciente"
        if ref.limit(1).count() == 0:
            ref, origen_ref = ORO.where((F.col("conjunto") == "entrenamiento") & (F.col("h") == 1)).select(*variables, *contexto), "entrenamiento completo"
        print(f"Referencia de deriva: {origen_ref} (orígenes desde {desde_ref})")
        act = ORO.where((F.col("conjunto") == "pronostico") & (F.col("origen") == F.lit(o_pron)) & (F.col("h") == 1)).select(*variables, *contexto)
        n_ref = ref.count()
        if n_ref > 2_000_000:
            ref = ref.sample(fraction=2_000_000 / n_ref, seed=43)
        filas = []
        for v in variables:
            distintos = ref.select(v).distinct().count()
            if distintos <= 12:                      # pocos valores (conteos, fracciones de 7 días): PSI sobre los valores exactos
                bucket = lambda df: df.select(F.coalesce(F.col(v), F.lit(-1.0)).cast("double").alias("b")).groupBy("b").count().toPandas().set_index("b")["count"]
            else:                                    # continua: deciles de la referencia
                cortes = sorted(set(ref.approxQuantile(v, [i / 10 for i in range(1, 10)], 0.01)))
                if len(cortes) < 2:
                    continue
                bucket = lambda df, cortes=cortes: df.select(F.when(F.col(v).isNull(), F.lit(-1)).otherwise(
                    F.expr("CASE " + " ".join(f"WHEN {v} <= {c} THEN {i}" for i, c in enumerate(cortes)) + f" ELSE {len(cortes)} END")).alias("b")) \
                    .groupBy("b").count().toPandas().set_index("b")["count"]
            pr, pa = bucket(ref), bucket(act)
            idx = sorted(set(pr.index) | set(pa.index))
            pr = (pr.reindex(idx).fillna(0) + 0.5); pa = (pa.reindex(idx).fillna(0) + 0.5)
            pr, pa = pr / pr.sum(), pa / pa.sum()
            psi = float(((pa - pr) * np.log(pa / pr)).sum())
            m_ref, m_act = ref.agg(F.avg(v)).collect()[0][0], act.agg(F.avg(v)).collect()[0][0]
            filas.append({"origen": o_pron, "variable": v, "psi": psi, "media_entrenamiento": m_ref, "media_actual": m_act, "referencia": origen_ref,
                          "estado": "alerta" if psi > UMBRAL_PSI_ALERTA else "aviso" if psi > UMBRAL_PSI_AVISO else "ok", "fecha_calculo": AHORA})
        for v in contexto:          # contexto: cambio relativo de la media (una sola cifra por tienda o categoría y origen)
            m_ref, m_act = ref.agg(F.avg(v)).collect()[0][0], act.agg(F.avg(v)).collect()[0][0]
            cambio = abs(m_act / m_ref - 1) if m_ref else 0.0
            filas.append({"origen": o_pron, "variable": v, "psi": None, "media_entrenamiento": m_ref, "media_actual": m_act, "referencia": origen_ref,
                          "cambio_media": cambio, "estado": "aviso" if cambio > UMBRAL_CAMBIO_CONTEXTO else "ok", "fecha_calculo": AHORA})
        DERIVA = pd.DataFrame(filas)
        if "cambio_media" not in DERIVA:
            DERIVA["cambio_media"] = np.nan
        mostrar(DERIVA.sort_values("psi", ascending=False), f"PSI por variable (pronóstico del {o_pron} vs. {origen_ref}, {DIAS_REFERENCIA_DERIVA // 7} semanas previas)",
                formatos={"psi": "dec3", "media_entrenamiento": "dec2", "media_actual": "dec2", "cambio_media": "pct"})
        for r in DERIVA[DERIVA["estado"] == "alerta"].itertuples():
            alerta("deriva", "alta", f"La variable {r.variable} cambió de distribución (PSI {r.psi:.2f} > {UMBRAL_PSI_ALERTA:.2f}): revisar datos y reentrenar.",
                   r.psi, UMBRAL_PSI_ALERTA, o_pron)
        for r in DERIVA[DERIVA["estado"] == "aviso"].itertuples():
            if pd.notna(r.psi):
                alerta("deriva", "media", f"La variable {r.variable} se movió (PSI {r.psi:.2f}).", r.psi, UMBRAL_PSI_AVISO, o_pron)
            else:
                alerta("deriva", "media", f"La media de {r.variable} cambió {r.cambio_media:.0%} respecto a la referencia.", r.cambio_media, UMBRAL_CAMBIO_CONTEXTO, o_pron)
        hallazgo(f"Deriva: {int((DERIVA['estado'] == 'alerta').sum())} variables en alerta y {int((DERIVA['estado'] == 'aviso').sum())} en aviso de {len(DERIVA)} vigiladas.")
        if existe(TABLA_DERIVA):
            try:
                spark.sql(f"DELETE FROM {TABLA_DERIVA} WHERE origen = DATE'{o_pron}'")
            except Exception:
                pass
        if len(DERIVA):
            guardar(DERIVA, "mlops", "deriva_variables", "PSI de las variables principales: pronostico vs entrenamiento (23_monitoreo)", modo="append")
else:
    print("No existe la Oro: corre 20_features_oro.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Frescura de datos y vigencia del pronóstico

# COMMAND ----------

titulo("3. FRESCURA")
hoy = AHORA.date()
dias_sin_datos = (hoy - ULTIMO_DIA_VENTA).days if ULTIMO_DIA_VENTA else None
print(f"Última venta cargada: {ULTIMO_DIA_VENTA} ({dias_sin_datos} días antes de hoy)")
if dias_sin_datos is not None and dias_sin_datos > DIAS_FRESCURA:
    alerta("frescura", "media", f"La venta más reciente tiene {dias_sin_datos} días (umbral {DIAS_FRESCURA}): pedir a Bnext el mes siguiente y correr J1_ingesta_mes.",
           dias_sin_datos, DIAS_FRESCURA)
if PRON is not None:
    ultimo_origen = PRON.where(F.col("fase") == "operativo").agg(F.max("origen")).collect()[0][0]
    print(f"Último origen del pronóstico operativo: {ultimo_origen}")
    if ultimo_origen is None or ultimo_origen < ULTIMO_DIA_VENTA:
        alerta("pronostico_desactualizado", "alta", f"Hay venta hasta {ULTIMO_DIA_VENTA} pero el pronóstico operativo es del {ultimo_origen}: correr 20_features_oro y 22_pronostico_operativo.",
               origen=ultimo_origen)
# reentrenar si hay dos semanas seguidas con alerta de error o sesgo (en producción)
if existe(TABLA_MONITOREO):
    hist = (spark.table(TABLA_MONITOREO).where(F.col("fase").startswith("produccion"))
            .groupBy("semana").agg(F.max(F.col("alerta_error").cast("int")).alias("e"), F.max(F.col("alerta_sesgo").cast("int")).alias("s"))
            .orderBy(F.desc("semana")).limit(2).toPandas())
    if len(hist) == 2 and ((hist["e"] == 1).all() or (hist["s"] == 1).all()):
        alerta("reentrenar", "alta", "Dos semanas seguidas con alerta de error o sesgo en producción: lanzar J3_entrenamiento y evaluar al challenger (24).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Alertas

# COMMAND ----------

titulo("4. ALERTAS")
AL = pd.DataFrame(ALERTAS)
if len(AL):
    mostrar(AL.drop(columns=["fecha"]), "Alertas de esta corrida")
    guardar(AL, "mlops", "alertas", "Alertas del monitoreo: tipo, severidad, valor, umbral y si fue atendida (23_monitoreo)", modo="append")
    hallazgo(f"{len(AL)} alertas registradas en {TABLA_ALERTAS}: " + ", ".join(f"{t} ({n})" for t, n in AL["tipo"].value_counts().items()) + ".")
else:
    hallazgo("Sin alertas: el modelo y los datos están dentro de los umbrales.")

# COMMAND ----------

cerrar("23_monitoreo", siguiente="24_promocion_modelo (si hay challenger) · 40_salidas_dashboard")
