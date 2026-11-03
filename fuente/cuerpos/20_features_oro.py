# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Tabla Oro: variables por origen y horizonte (sin fuga de información)
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Fase 2 (modelos)**
# MAGIC
# MAGIC **Para qué sirve.** Construye la tabla con la que se entrenan y evalúan **todos** los modelos de la fase 2:
# MAGIC una fila por **serie (tienda × categoría), origen T y horizonte h = 1…14**, con el objetivo en T + h y variables
# MAGIC calculadas **únicamente con información disponible hasta T**. Así ningún modelo ve el futuro y todos se comparan
# MAGIC con la misma entrada.
# MAGIC
# MAGIC | Grupo | Variables (todas con datos ≤ T) |
# MAGIC | --- | --- |
# MAGIC | Historia de la serie | venta en T; medias móviles 7/14/28/56; desviación y máximo 28; días activos y con venta en 7 y 28; días desde la última venta; días con venta en toda la historia; unidades, tickets, descuento y promoción en 28 días |
# MAGIC | Mismo día de la semana | rezagos del mismo día de la semana que el día objetivo (1 a 4 semanas antes de T) y sus medias a 4 y 8 semanas; lo mismo para la categoría en la zona y en el país |
# MAGIC | Contexto | afluencia de la tienda (tickets) y venta total de la tienda; venta media por tienda de la categoría en su zona y en el país; participación de la serie en su tienda |
# MAGIC | Calendario del día objetivo | día de la semana, día y semana del mes, quincena, eventos y festivos, días al evento más próximo, horizonte h |
# MAGIC | Atributos | tipo, zona, región, jefatura, estatus y antigüedad de la tienda; tipo de categoría |
# MAGIC
# MAGIC **Tres conjuntos.** `entrenamiento`: una muestra estratificada de series (por clase ABC) en orígenes cada
# MAGIC `PASO_ORIGENES` días, hasta el último origen cuyos 14 días ya se observaron. `evaluacion`: **todas** las series
# MAGIC relevantes en los orígenes de los cortes walk-forward fijos de la configuración (los cuatro de desarrollo del Avance 4
# MAGIC y, cuando ya hay datos, los dos de **prueba** de septiembre: datos no vistos), más el origen de **ajuste** (20 de mayo)
# MAGIC para afinar hiperparámetros sin gastar los cortes. `pronostico`: el último día con datos como origen, con los 14 días
# MAGIC siguientes sin objetivo (`y` nulo): son las filas que el modelo final pronostica para operar.
# MAGIC Una muestra de series lleva `en_muestra_eval = 1`: es el conjunto común en el que se ordena la comparativa de modelos.
# MAGIC
# MAGIC **Promociones.** Todas las variables de promoción usan sólo días ≤ T. `s_promos_vigentes_t0` ahora es el máximo de
# MAGIC promociones con venta en un mismo día de la última semana (antes usaba la vigencia reconstruida con la última venta de
# MAGIC cada promoción, que depende de días posteriores a T). La tabla lo declara en sus propiedades
# MAGIC (`mna.s_promos_vigentes_t0 = causal_7d`) para que los notebooks de modelos sepan que pueden usarla.
# MAGIC
# MAGIC **Prueba anti-fuga.** Al final se reconstruyen las variables del primer corte usando sólo el panel truncado en su
# MAGIC origen y se comparan con las guardadas: deben ser idénticas. Si no lo son, el notebook se detiene.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `silver.panel_categoria_dia` · `silver.catalogo_tiendas` · `silver.catalogo_categorias` · `mlops.linea_base_ingenua` |
# MAGIC | **Escribe** | `gold.features_origen` (partición `conjunto`) · `gold.calendario` · `mlops.oro_resumen` |
# MAGIC | **Tiempo** | 30–70 min en serverless (con septiembre hay dos bloques de evaluación y un origen de pronóstico más) |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@DICCIONARIO

# COMMAND ----------

# @@EVENTOS

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
H_MAX = 14                     # horizonte máximo (días después del origen)
VENTANA = 56                   # días de historia que se resumen en cada origen (8 semanas)
DIAS_MINIMOS_HISTORIA = 28     # el primer origen de entrenamiento es INICIO + 28 días
PASO_ORIGENES = 2              # un origen de entrenamiento cada 2 días
N_CORTES = len(ORIGENES_CORTES)   # cortes de desarrollo fijos (configuración); DIAS_CORTE también viene de la configuración
GENERAR_PRONOSTICO = parametro("GENERAR_PRONOSTICO", True)   # agrega el conjunto 'pronostico': origen = último día con datos, h = 1…14 sin objetivo
MIN_DIAS_CON_VENTA = 5         # serie relevante en un origen (igual que el 10): ≥ 5 días con venta antes del origen
SERIES_ENTRENAMIENTO = {"A": 5000, "B": 3000, "C": 2000}   # muestra estratificada de series para entrenar (máximo por clase ABC)
MUESTRA_EVAL_POR_MIL = 30      # 3 % de las series relevantes forman la muestra de evaluación común (todas si hay pocas series)
SEMILLA = 43
EVENTOS_EXTRA = [{"fecha": "2026-04-30", "nombre": "Día del Niño", "tipo": "comercial"},
                 {"fecha": "2026-08-24", "nombre": "Regreso a clases (aprox.)", "tipo": "comercial"}]

TABLA_ORO = tabla("gold", "features_origen")
INICIO = pd.Timestamp(min(MESES) + "-01")
FIN = pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)
CLAVES = ["tienda", "categoria_id"]
K_MAX = VENTANA - 1
P = leer_tabla(tabla("silver", "panel_categoria_dia"), "09_panel_diario")
TIE = leer_tabla(tabla("silver", "catalogo_tiendas"), "08_limpieza_plata")
CAT = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata")
print("Periodo del panel:", INICIO.date(), "→", FIN.date(), "· destino:", TABLA_ORO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Orígenes: cortes de evaluación, origen de ajuste, orígenes de entrenamiento y de pronóstico
# MAGIC **Qué hace.** Toma los cortes fijos de la configuración (iguales a los del notebook 10): cuatro de desarrollo y dos de
# MAGIC prueba en septiembre cuando sus días ya están cargados. Define el origen de ajuste, los orígenes de entrenamiento
# MAGIC (hasta el último origen con sus 14 días observados) y el origen de pronóstico (el último día con datos).

# COMMAND ----------

titulo("1. ORÍGENES")
dias_con_datos = {r[0] for r in P.where(F.col("sin_datos") == 0).select("fecha").distinct().collect()}
FIN_DATOS = pd.Timestamp(max(dias_con_datos))                    # último día con datos (p. ej., 30 de septiembre)
CORTES = cortes_walk_forward(dias_con_datos)                    # desarrollo + prueba (si ya hay datos de septiembre)
CORTES_DEV = [c for c in CORTES if c["fase"] == "desarrollo"]
if len(CORTES_DEV) < N_CORTES:
    aviso(f"Sólo {len(CORTES_DEV)} cortes de desarrollo tienen datos completos (se esperaban {N_CORTES}).")
if not any(c["fase"] == "prueba" for c in CORTES):
    aviso("No hay bloques de prueba (septiembre): la tabla Oro se construye sin datos no vistos. Carga el mes y vuelve a correr 02 → 20.")

# origen de ajuste: el de la configuración (si ese día no tuviera datos, retrocede hasta uno con datos)
origen_ajuste = pd.Timestamp(ORIGEN_AJUSTE)
while origen_ajuste.date() not in dias_con_datos and origen_ajuste > INICIO + pd.Timedelta(days=DIAS_MINIMOS_HISTORIA):
    origen_ajuste -= pd.Timedelta(days=1)
AJUSTE = {"corte": "ajuste", "origen": origen_ajuste.date(), "desde": (origen_ajuste + pd.Timedelta(days=1)).date(),
          "hasta": (origen_ajuste + pd.Timedelta(days=DIAS_CORTE)).date(), "fase": "ajuste"}
ORIGENES_EVAL = pd.DataFrame(CORTES + [AJUSTE])[["corte", "fase", "origen", "desde", "hasta"]]
mostrar(ORIGENES_EVAL, "Orígenes de evaluación (desarrollo = Avance 4, prueba = septiembre, ajuste de hiperparámetros)")

# orígenes de entrenamiento: cada PASO_ORIGENES días hasta el último origen cuyos 14 días ya se observaron
ultimo_origen = (FIN_DATOS - pd.Timedelta(days=H_MAX)).date()
ORIGENES_TRAIN = [d.date() for d in pd.date_range(INICIO + pd.Timedelta(days=DIAS_MINIMOS_HISTORIA), pd.Timestamp(ultimo_origen),
                                                  freq=f"{PASO_ORIGENES}D") if d.date() in dias_con_datos]
ORIGEN_PRONOSTICO = FIN_DATOS.date() if GENERAR_PRONOSTICO else None
hallazgo(f"{len(ORIGENES_TRAIN)} orígenes de entrenamiento ({ORIGENES_TRAIN[0]} → {ORIGENES_TRAIN[-1]}, cada {PASO_ORIGENES} días, "
         f"sólo días con datos) y {len(ORIGENES_EVAL)} orígenes de evaluación: " +
         ", ".join(f"{r.corte} = {r.origen}" for r in ORIGENES_EVAL.itertuples()) +
         (f". Origen de pronóstico: {ORIGEN_PRONOSTICO} (14 días siguientes sin objetivo)." if ORIGEN_PRONOSTICO else "."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Series: universo, clase ABC, muestra de entrenamiento y muestra de evaluación
# MAGIC **Qué hace.** Universo = series relevantes del panel (≥ 5 días con venta). La clase ABC (80 / 95 % de la venta
# MAGIC acumulada) sólo sirve para **estratificar la muestra de entrenamiento**; no es una variable del modelo (se calcula con
# MAGIC todo el periodo). La muestra de evaluación común es un 3 % de las series, elegido por hash (reproducible).

# COMMAND ----------

titulo("2. SERIES Y MUESTRAS")
SER = (P.where((F.col("serie_relevante") == 1) & F.col("venta").isNotNull()).groupBy(*CLAVES)
       .agg(F.sum("venta").alias("venta_total"), F.sum((F.col("venta") > 0).cast("int")).alias("dias_con_venta")))
total = SER.agg(F.sum("venta_total")).collect()[0][0] or 1.0
w_abc = Window.orderBy(F.desc("venta_total"), *CLAVES).rowsBetween(Window.unboundedPreceding, -1)
SER = (SER.withColumn("acum_antes", F.coalesce(F.sum("venta_total").over(w_abc), F.lit(0.0)) / F.lit(total))
       .withColumn("abc", F.when(F.col("acum_antes") < 0.80, "A").when(F.col("acum_antes") < 0.95, "B").otherwise("C"))
       .withColumn("hash", F.abs(F.xxhash64(F.col("tienda").cast("string"), F.col("categoria_id").cast("string"), F.lit(SEMILLA))))
       .drop("acum_antes"))
n_series = SER.count()
por_mil = 1000 if n_series <= 50_000 else MUESTRA_EVAL_POR_MIL
SER = SER.withColumn("en_muestra_eval", ((F.col("hash") % 1000) < por_mil).cast("int"))
SER = SER.withColumn("rn", F.row_number().over(Window.partitionBy("abc").orderBy("hash")))
cupo = (F.when(F.col("abc") == "A", SERIES_ENTRENAMIENTO["A"]).when(F.col("abc") == "B", SERIES_ENTRENAMIENTO["B"])
        .otherwise(SERIES_ENTRENAMIENTO["C"]))
SER = SER.withColumn("en_entrenamiento", (F.col("rn") <= cupo).cast("int")).drop("rn")
SER = persistir(SER)
RES_SER = (SER.groupBy("abc").agg(F.count("*").alias("series"), F.sum("en_entrenamiento").alias("en_entrenamiento"),
                                  F.sum("en_muestra_eval").alias("en_muestra_eval"), F.sum("venta_total").alias("venta"))
           .orderBy("abc").toPandas())
RES_SER["pct_venta"] = RES_SER["venta"] / RES_SER["venta"].sum()
mostrar(RES_SER[["abc", "series", "en_entrenamiento", "en_muestra_eval", "pct_venta"]], "Series relevantes por clase ABC y muestras",
        formatos={"series": "int", "en_entrenamiento": "int", "en_muestra_eval": "int", "pct_venta": "pct"})
n_train, n_eval = int(RES_SER["en_entrenamiento"].sum()), int(RES_SER["en_muestra_eval"].sum())
hallazgo(f"{n_series:,} series relevantes; muestra de entrenamiento {n_train:,} series (máximo A {SERIES_ENTRENAMIENTO['A']}, "
         f"B {SERIES_ENTRENAMIENTO['B']}, C {SERIES_ENTRENAMIENTO['C']}); muestra de evaluación común {n_eval:,} series ({por_mil / 10:.0f} %).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Construcción de la tabla Oro
# MAGIC **Qué hace.** Para cada par (serie, origen) une los 56 días anteriores al origen y resume la historia en una sola
# MAGIC fila (medias, conteos y un vector con los 56 valores diarios para sacar los rezagos del mismo día de la semana).
# MAGIC Lo mismo para la tienda, para la categoría en la zona y para la categoría en el país. Luego abre el horizonte
# MAGIC h = 1…14, une el objetivo del día T + h y el calendario de ese día. Toda la lógica está en `construir_oro`, cuyas
# MAGIC piezas se vuelven a usar en la prueba anti-fuga.

# COMMAND ----------

titulo("3. CONSTRUCCIÓN")
# ---- calendario del día objetivo (se guarda como tabla propia)
cal = pd.DataFrame({"fecha": pd.date_range(INICIO, FIN + pd.Timedelta(days=60), freq="D")})
cal["dow"] = cal["fecha"].dt.dayofweek + 1                      # 1 = lunes … 7 = domingo
cal["fin_de_semana"] = (cal["dow"] >= 6).astype(int)
cal["dia_mes"] = cal["fecha"].dt.day
cal["semana_mes"] = ((cal["dia_mes"] - 1) // 7 + 1).astype(int)
ultimo_dia_mes = cal["fecha"].dt.days_in_month
cal["quincena"] = ((cal["dia_mes"].between(14, 16)) | (cal["dia_mes"] >= ultimo_dia_mes - 1) | (cal["dia_mes"] <= 1)).astype(int)
eventos = pd.DataFrame(EVENTOS + EVENTOS_EXTRA)
eventos["fecha"] = pd.to_datetime(eventos["fecha"])
tipos_evento = {"ninguno": 0, "festivo": 1, "comercial": 2, "futbol": 3}
ev_fechas = eventos.sort_values("fecha")["fecha"].to_numpy()
cal["es_evento"] = cal["fecha"].isin(eventos["fecha"]).astype(int)
cal["tipo_evento"] = cal["fecha"].map(eventos.drop_duplicates("fecha").set_index("fecha")["tipo"]).fillna("ninguno").map(tipos_evento).astype(int)
def _dias_a_evento(f):
    futuros = ev_fechas[ev_fechas >= f]
    return int(min((futuros[0] - f).days, 30)) if len(futuros) else 30
def _dias_desde_evento(f):
    pasados = ev_fechas[ev_fechas <= f]
    return int(min((f - pasados[-1]).days, 30)) if len(pasados) else 30
cal["dias_a_evento"] = cal["fecha"].map(_dias_a_evento)
cal["dias_desde_evento"] = cal["fecha"].map(_dias_desde_evento)
cal["en_mundial"] = ((cal["fecha"] >= pd.Timestamp(PERIODOS[0]["inicio"])) & (cal["fecha"] <= pd.Timestamp(PERIODOS[0]["fin"]))).astype(int)
cal["fecha"] = cal["fecha"].dt.date
guardar(cal, "gold", "calendario", "Calendario diario con dia de la semana, quincena, eventos y dias al evento (20_features_oro)")
CALEN = F.broadcast(spark.table(tabla("gold", "calendario")))

# ---- atributos de tienda y categoría
ATR_TIENDA = (TIE.select("tienda", "tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "primera_venta")
              .withColumn("tipo_tienda", F.coalesce(F.col("tipo_tienda").cast("string"), F.lit("SIN TIPO")))
              .withColumn("zona", F.coalesce(F.col("zona").cast("string"), F.lit("SIN ZONA")))
              .withColumn("region", F.coalesce(F.col("region").cast("string"), F.lit("SIN REGIÓN")))
              .withColumn("jefatura", F.coalesce(F.col("jefatura").cast("string"), F.lit("SIN JEFATURA")))
              .withColumn("estatus_tienda", F.coalesce(F.col("estatus_tienda").cast("string"), F.lit("SIN ESTATUS"))))
ATR_CAT = CAT.select("categoria_id", "tipo_categoria")
ZONA_DE = ATR_TIENDA.select("tienda", "zona")
w_hist = Window.partitionBy(*CLAVES).orderBy("fecha").rowsBetween(Window.unboundedPreceding, 0)
vector = F.expr(f"transform(sequence(0, {K_MAX}), j -> try_element_at(mapa, j))")


def resumir_historia(pares, diario, grupo, valor, prefijo, extras=None):
    """Para cada grupo (claves…, origen) une los `VENTANA` días ≤ origen de la serie `diario` y los resume:
    medias móviles, conteos y el vector de los 56 valores diarios (posición k = días antes del origen).
    `grupo` son las columnas de `pares` que identifican la fila; las claves de unión con `diario` son las que comparten."""
    claves = [c for c in grupo if c in diario.columns and c != "origen"]
    j = pares.join(diario, claves).where(F.col("fecha").between(F.date_sub(F.col("origen"), K_MAX), F.col("origen")))
    j = j.withColumn("k", F.datediff(F.col("origen"), F.col("fecha")))
    v = F.col(valor)
    aggs = [F.max(F.when(F.col("k") == 0, v)).alias(f"{prefijo}_t0"),
            F.avg(F.when(F.col("k") < 7, v)).alias(f"{prefijo}_ma7"), F.avg(F.when(F.col("k") < 14, v)).alias(f"{prefijo}_ma14"),
            F.avg(F.when(F.col("k") < 28, v)).alias(f"{prefijo}_ma28"), F.avg(v).alias(f"{prefijo}_ma56"),
            F.count(F.when(F.col("k") < 7, v)).alias(f"{prefijo}_dias_activos_7"),
            F.count(F.when(F.col("k") < 28, v)).alias(f"{prefijo}_dias_activos_28"),
            F.map_from_entries(F.collect_list(F.struct(F.col("k"), v))).alias("mapa")]
    aggs.extend(extras or [])
    return j.groupBy(*grupo).agg(*aggs).withColumn(f"{prefijo}_vec", vector).drop("mapa")


def rezagos_dow(df, prefijo, n_medias=(4, 8)):
    """Del vector de 56 valores saca los rezagos del mismo día de la semana que el día objetivo (T + h):
    posición k = 7·⌈h/7⌉ − h es el mismo día de la semana más reciente ≤ T; k + 7, k + 14, … las semanas anteriores."""
    vec = F.col(f"{prefijo}_vec")
    k0 = F.expr("7 * ceil(h / 7) - h").cast("int")
    cols = [F.element_at(vec, (k0 + 1 + 7 * j).cast("int")).alias(f"_{prefijo}_dow{j}") for j in range(8)]
    df = df.select("*", *cols)
    df = df.withColumn(f"{prefijo}_lag_dow_1", F.col(f"_{prefijo}_dow0")).withColumn(f"{prefijo}_lag_dow_2", F.col(f"_{prefijo}_dow1"))
    if prefijo == "s":
        df = df.withColumn("s_lag_dow_3", F.col("_s_dow2")).withColumn("s_lag_dow_4", F.col("_s_dow3"))
    for n in n_medias:
        suma = sum((F.coalesce(F.col(f"_{prefijo}_dow{j}"), F.lit(0.0)) for j in range(n)), F.lit(0.0))
        cuenta = sum((F.col(f"_{prefijo}_dow{j}").isNotNull().cast("int") for j in range(n)), F.lit(0))
        df = df.withColumn(f"{prefijo}_media_dow_{n}", F.when(cuenta > 0, suma / cuenta))
        if prefijo == "s" and n == 8:
            df = df.withColumn("s_n_dow_8", cuenta)
    return df.drop(*[f"_{prefijo}_dow{j}" for j in range(8)], f"{prefijo}_vec")


GRUPO_SERIE = [*CLAVES, "origen", "conjunto", "corte", "en_muestra_eval"]
EXTRAS_SERIE = [F.stddev_pop(F.when(F.col("k") < 28, F.col("venta"))).alias("s_desv28"),
                F.max(F.when(F.col("k") < 28, F.col("venta"))).alias("s_max28"),
                F.sum(F.when((F.col("k") < 28) & (F.col("venta") > 0), 1).otherwise(0)).alias("s_dias_con_venta_28"),
                F.min(F.when(F.col("venta") > 0, F.col("k"))).alias("s_dias_desde_ultima_venta"),
                F.avg(F.when(F.col("k") < 28, F.col("unidades"))).alias("s_unidades_ma28"),
                F.avg(F.when(F.col("k") < 28, F.col("tickets"))).alias("s_tickets_ma28"),
                F.avg(F.when(F.col("k") < 28, F.col("tasa_descuento"))).alias("s_tasa_descuento_28"),
                F.avg(F.when(F.col("k") < 28, F.col("intensidad_promo"))).alias("s_intensidad_promo_28"),
                # máximo de promociones con venta en un mismo día de la última semana (sólo días ≤ T; antes se usaba la
                # vigencia reconstruida con la última venta de cada promoción, que depende de días posteriores a T)
                F.max(F.when(F.col("k") < 7, F.col("promos_en_venta"))).alias("s_promos_vigentes_t0"),
                F.avg(F.when(F.col("k") < 28, F.col("promos_en_venta"))).alias("s_promos_en_venta_28")]
EXTRAS_TIENDA = [F.avg(F.when(F.col("k") < 7, F.col("tickets_tienda"))).alias("t_tickets_ma7"),
                 F.avg(F.when(F.col("k") < 28, F.col("tickets_tienda"))).alias("t_tickets_ma28")]


def estado_series(pares, panel):
    """Variables de estado de cada (serie, origen): historia de la serie, de la tienda, de la categoría en la zona y en el país,
    y atributos. Sólo usa filas del panel con fecha ≤ origen."""
    pv = panel.where(F.col("venta").isNotNull())
    diario = pv.select(*CLAVES, "fecha", "venta", "unidades", "tickets", "tasa_descuento", "intensidad_promo", "promos_en_venta")
    S = resumir_historia(pares, diario, GRUPO_SERIE, "venta", "s", EXTRAS_SERIE)
    td = pv.groupBy("tienda", "fecha").agg(F.sum("venta").alias("venta_tienda"), F.max("tickets_tienda").alias("tickets_tienda"))
    T_ = (resumir_historia(pares.select("tienda", "origen").distinct(), td, ["tienda", "origen"], "venta_tienda", "t", EXTRAS_TIENDA)
          .select("tienda", "origen", "t_ma7", "t_ma28", "t_tickets_ma7", "t_tickets_ma28"))
    zcd = pv.join(ZONA_DE, "tienda").groupBy("zona", "categoria_id", "fecha").agg((F.sum("venta") / F.count("venta")).alias("zc_prom"))
    pares_z = pares.join(ZONA_DE, "tienda").select("zona", "categoria_id", "origen").distinct()
    Z_ = (resumir_historia(pares_z, zcd, ["zona", "categoria_id", "origen"], "zc_prom", "zc")
          .select("zona", "categoria_id", "origen", "zc_ma7", "zc_ma28", "zc_vec"))
    pcd = pv.groupBy("categoria_id", "fecha").agg((F.sum("venta") / F.count("venta")).alias("pc_prom"))
    C_ = (resumir_historia(pares.select("categoria_id", "origen").distinct(), pcd, ["categoria_id", "origen"], "pc_prom", "pc")
          .select("categoria_id", "origen", "pc_ma7", "pc_ma28", "pc_vec"))
    hist = (panel.select(*CLAVES, "fecha", "venta")
            .withColumn("dias_con_venta_hist", F.sum(F.when(F.col("venta") > 0, 1).otherwise(0)).over(w_hist))
            .withColumn("dias_historia", F.datediff("fecha", F.min("fecha").over(Window.partitionBy(*CLAVES))))
            .select(*CLAVES, F.col("fecha").alias("origen"), "dias_con_venta_hist", "dias_historia"))
    S = (S.join(hist, [*CLAVES, "origen"], "left").join(T_, ["tienda", "origen"], "left")
         .join(ATR_TIENDA, "tienda", "left").join(Z_, ["zona", "categoria_id", "origen"], "left")
         .join(C_, ["categoria_id", "origen"], "left").join(ATR_CAT, "categoria_id", "left"))
    return (S.withColumn("s_dias_desde_ultima_venta", F.coalesce(F.col("s_dias_desde_ultima_venta"), F.lit(VENTANA)))
            .withColumn("s_tasa_venta_28", F.when(F.col("s_dias_activos_28") > 0, F.col("s_dias_con_venta_28") / F.col("s_dias_activos_28")))
            .withColumn("s_part_tienda_28", F.when(F.col("t_ma28") > 0, F.col("s_ma28") / F.col("t_ma28")))
            .withColumn("antiguedad_tienda", F.datediff("origen", "primera_venta")).drop("primera_venta"))


def abrir_horizonte(S):
    """Abre h = 1…14, calcula los rezagos del mismo día de la semana del día objetivo y une su calendario."""
    O = (S.withColumn("h", F.explode(F.sequence(F.lit(1), F.lit(H_MAX))))
         .withColumn("fecha_objetivo", F.date_add("origen", F.col("h")))
         .withColumn("h_bloque", F.when(F.col("h") <= 7, 1).otherwise(2)))
    O = rezagos_dow(O, "s")
    O = rezagos_dow(O, "zc", n_medias=(4,))
    O = rezagos_dow(O, "pc", n_medias=(4,))
    O = O.join(CALEN.withColumnRenamed("fecha", "fecha_objetivo"), "fecha_objetivo", "left")
    return O.withColumn("ingenuo_estacional", F.col("s_lag_dow_1"))


def construir_oro(pares, panel, con_objetivo=True):
    """pares: (tienda, categoria_id, origen, conjunto, corte, en_muestra_eval). Devuelve una fila por (serie, origen, h)
    con el objetivo observado en T + h (las filas sin objetivo —tienda inactiva o día sin datos— no se generan).
    Con con_objetivo=False (conjunto 'pronostico') los 14 días son futuros: el objetivo queda nulo."""
    O = abrir_horizonte(estado_series(pares, panel))
    objetivo = panel.select(*CLAVES, F.col("fecha").alias("fecha_objetivo"), F.col("venta").alias("y"),
                            F.col("unidades").alias("y_unidades"), F.col("dia_atipico").alias("dia_atipico_obj"))
    if con_objetivo:
        return O.join(objetivo, [*CLAVES, "fecha_objetivo"], "inner").where(F.col("y").isNotNull())
    tipos = {f.name: f.dataType for f in objetivo.schema.fields}
    return (O.withColumn("y", F.lit(None).cast(tipos["y"])).withColumn("y_unidades", F.lit(None).cast(tipos["y_unidades"]))
             .withColumn("dia_atipico_obj", F.lit(None).cast(tipos["dia_atipico_obj"])))


# ---- pares (serie, origen) de los dos conjuntos
ser_train = SER.where(F.col("en_entrenamiento") == 1).select(*CLAVES)
ser_todas = SER.select(*CLAVES, "en_muestra_eval")
orig_train = spark.createDataFrame(pd.DataFrame({"origen": ORIGENES_TRAIN}))
orig_eval = spark.createDataFrame(ORIGENES_EVAL[["corte", "origen"]])
PARES_TRAIN = (ser_train.crossJoin(orig_train).join(ser_todas, CLAVES)
               .withColumn("conjunto", F.lit("entrenamiento")).withColumn("corte", F.lit(None).cast("string")))
PARES_EVAL = (ser_todas.crossJoin(orig_eval)
              .where((F.col("corte") != "ajuste") | (F.col("en_muestra_eval") == 1))      # el origen de ajuste sólo con la muestra
              .withColumn("conjunto", F.lit("evaluacion")))
PARES = PARES_TRAIN.select(*GRUPO_SERIE).unionByName(PARES_EVAL.select(*GRUPO_SERIE))
print(f"Pares (serie, origen): entrenamiento ≈ {n_train * len(ORIGENES_TRAIN):,}; evaluación ≈ {n_series * len(CORTES) + n_eval:,}"
      + (f"; pronóstico ≈ {n_series:,}" if ORIGEN_PRONOSTICO else ""))

t0 = time.time()
ORO = construir_oro(PARES, P)
if ORIGEN_PRONOSTICO:
    PARES_PRON = (ser_todas.crossJoin(spark.createDataFrame(pd.DataFrame({"origen": [ORIGEN_PRONOSTICO]})))
                  .withColumn("conjunto", F.lit("pronostico")).withColumn("corte", F.lit("pronostico")).select(*GRUPO_SERIE))
COLUMNAS_ORO = [*CLAVES, "origen", "h", "fecha_objetivo", "conjunto", "corte", "en_muestra_eval", "y", "y_unidades", "dia_atipico_obj",
                "ingenuo_estacional",
                # historia de la serie
                "s_t0", "s_ma7", "s_ma14", "s_ma28", "s_ma56", "s_desv28", "s_max28", "s_dias_activos_7", "s_dias_activos_28",
                "s_dias_con_venta_28", "s_tasa_venta_28", "s_dias_desde_ultima_venta", "dias_con_venta_hist", "dias_historia",
                "s_unidades_ma28", "s_tickets_ma28", "s_tasa_descuento_28", "s_intensidad_promo_28", "s_promos_vigentes_t0",
                "s_promos_en_venta_28", "s_lag_dow_1", "s_lag_dow_2", "s_lag_dow_3", "s_lag_dow_4", "s_media_dow_4", "s_media_dow_8", "s_n_dow_8",
                # contexto
                "t_ma7", "t_ma28", "t_tickets_ma7", "t_tickets_ma28", "s_part_tienda_28",
                "zc_ma7", "zc_ma28", "zc_lag_dow_1", "zc_lag_dow_2", "zc_media_dow_4", "pc_ma7", "pc_ma28", "pc_lag_dow_1", "pc_lag_dow_2", "pc_media_dow_4",
                # calendario del día objetivo
                "dow", "fin_de_semana", "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento", "dias_a_evento", "dias_desde_evento",
                "en_mundial", "h_bloque",
                # atributos
                "tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "antiguedad_tienda", "tipo_categoria"]
NO_VARIABLES = [*CLAVES, "origen", "h", "fecha_objetivo", "conjunto", "corte", "en_muestra_eval", "y", "y_unidades", "dia_atipico_obj"]
if ORIGEN_PRONOSTICO:
    ORO = ORO.select(*COLUMNAS_ORO).unionByName(construir_oro(PARES_PRON, P, con_objetivo=False).select(*COLUMNAS_ORO))
guardar(ORO.select(*COLUMNAS_ORO), "gold", "features_origen",
        "Una fila por serie, origen T y horizonte h con el objetivo en T+h y variables calculadas solo con datos <= T (20_features_oro)",
        particion="conjunto")
print(f"Construida en {(time.time() - t0) / 60:.1f} min")
G = spark.table(TABLA_ORO)
try:
    spark.sql(f"OPTIMIZE {TABLA_ORO} ZORDER BY (origen, tienda)")
except Exception:
    pass
# contrato para los notebooks de modelos: qué versión de variables tiene la tabla
try:
    spark.sql(f"ALTER TABLE {TABLA_ORO} SET TBLPROPERTIES ('mna.s_promos_vigentes_t0' = 'causal_7d', "
              f"'mna.version_variables' = '2', 'mna.origen_pronostico' = '{ORIGEN_PRONOSTICO or ''}')")
except Exception as e:
    aviso(f"No se pudieron escribir las propiedades de la tabla Oro ({type(e).__name__}); el notebook de modelos excluirá s_promos_vigentes_t0.")
# diccionario de datos en la propia tabla (comentarios por columna) y restricciones que Delta vigila en cada escritura
n_com = comentar_columnas(TABLA_ORO, {**DESCRIPCION_METADATOS, **{k: v for k, v in DESCRIPCION_VARIABLES.items() if k in COLUMNAS_ORO}})
asegurar_restricciones(TABLA_ORO, {"oro_h_valido": "h BETWEEN 1 AND 14", "oro_objetivo_no_negativo": "y IS NULL OR y >= 0",
                                   "oro_conjunto_valido": "conjunto IN ('entrenamiento', 'evaluacion', 'pronostico')"})
print(f"Comentarios escritos en {n_com} columnas de la Oro; restricciones CHECK verificadas.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resumen de la tabla y reproducción de la línea base
# MAGIC **Qué mirar.** Filas por conjunto y por corte; completitud de las variables; y que el ingenuo estacional calculado
# MAGIC desde la Oro (`ingenuo_estacional`, con los mismos filtros del notebook 10) reproduzca el WMAPE de la línea base.

# COMMAND ----------

titulo("4. RESUMEN Y LÍNEA BASE REPRODUCIDA")
RES = (G.groupBy("conjunto", "corte").agg(F.count("*").alias("filas"), F.countDistinct(*CLAVES).alias("series"),
                                          F.countDistinct("origen").alias("origenes"), F.min("origen").alias("primer_origen"),
                                          F.max("origen").alias("ultimo_origen"), F.avg((F.col("y") == 0).cast("double")).alias("pct_ceros"))
       .orderBy("conjunto", "corte").toPandas())
mostrar(RES, "Filas de la tabla Oro por conjunto y corte", formatos={"filas": "int", "series": "int", "origenes": "int", "pct_ceros": "pct"})
# filtros de evaluación idénticos al 10: serie relevante en el origen, rezago estacional disponible, ≥ 5 días activos en la última semana
EVALUABLE = ((F.col("dias_con_venta_hist") >= MIN_DIAS_CON_VENTA) & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))
LB = (G.where((F.col("conjunto") == "evaluacion") & (F.col("corte") != "ajuste") & EVALUABLE)
      .groupBy("corte").agg(F.count("*").alias("filas"), F.sum("y").alias("real"),
                            F.sum(F.abs(F.col("y") - F.col("ingenuo_estacional"))).alias("e"), F.sum("ingenuo_estacional").alias("p"))
      .orderBy("corte").toPandas())
LB["wmape_ingenuo"] = LB["e"] / LB["real"]
LB["sesgo_ingenuo"] = LB["p"] / LB["real"] - 1
if existe(tabla("mlops", "linea_base_ingenua")):
    ref = (spark.table(tabla("mlops", "linea_base_ingenua")).toPandas().sort_values("fecha_calculo").groupby("corte").tail(1)
           .set_index("corte")["wmape_lag7"])
    LB["wmape_10"] = LB["corte"].map(ref)
    mostrar(LB[["corte", "filas", "wmape_ingenuo", "wmape_10", "sesgo_ingenuo"]], "Ingenuo estacional desde la Oro contra el notebook 10",
            formatos={"filas": "int", "wmape_ingenuo": "dec4", "wmape_10": "dec4", "sesgo_ingenuo": "pct"})
    dif = float((LB["wmape_ingenuo"] - LB["wmape_10"]).abs().max())
    hallazgo(f"Ingenuo estacional desde la Oro: WMAPE medio {LB['wmape_ingenuo'].mean():.4f} (notebook 10: {LB['wmape_10'].mean():.4f}; "
             f"diferencia máxima por corte {dif:.4f}). " + ("La Oro reproduce la línea base." if dif < 0.002 else
             "REVISAR: la Oro no reproduce la línea base del 10."))
    if dif >= 0.002:
        aviso("La línea base reproducida desde la Oro difiere de la del notebook 10 en más de 0.002 (¿filtros distintos?).")
else:
    mostrar(LB[["corte", "filas", "wmape_ingenuo", "sesgo_ingenuo"]], "Ingenuo estacional desde la Oro (filtros del 10)",
            formatos={"filas": "int", "wmape_ingenuo": "dec4", "sesgo_ingenuo": "pct"})
    hallazgo(f"Ingenuo estacional desde la Oro: WMAPE medio {LB['wmape_ingenuo'].mean():.4f} en {len(LB)} cortes "
             "(no se encontró la tabla del 10 para comparar).")
num_cols = [c for c in COLUMNAS_ORO if c not in NO_VARIABLES and c not in ("tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "tipo_categoria")]
NUL = G.where(F.col("conjunto") == "evaluacion").agg(*[F.avg(F.col(c).isNull().cast("double")).alias(c) for c in num_cols]).toPandas().T
NUL.columns = ["pct_nulos"]
NUL = NUL[NUL["pct_nulos"] > 0].sort_values("pct_nulos", ascending=False).reset_index().rename(columns={"index": "variable"})
mostrar(NUL, "Variables con nulos en evaluación (los modelos de árboles los aceptan; los lineales los imputan)", formatos={"pct_nulos": "pct2"})
r_tr = RES[RES["conjunto"] == "entrenamiento"]
hallazgo(f"Tabla Oro: {int(RES['filas'].sum()):,} filas y {len(COLUMNAS_ORO) - len(NO_VARIABLES)} variables; entrenamiento "
         f"{int(r_tr['filas'].sum()):,} filas ({int(r_tr['series'].max()):,} series × {int(r_tr['origenes'].max())} orígenes × {H_MAX} h); "
         f"evaluación {int(RES[RES['conjunto'] == 'evaluacion']['filas'].sum()):,} filas.")
guardar(RES.assign(fecha_calculo=pd.Timestamp.now()), "mlops", "oro_resumen", "Filas de la tabla Oro por conjunto y corte (20_features_oro)", modo="append")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Prueba anti-fuga
# MAGIC **Qué hace.** Toma el primer corte y la muestra de evaluación, reconstruye sus variables con el panel **truncado en el
# MAGIC origen** (como si el futuro no existiera) y las compara, fila por fila, con las guardadas. Deben coincidir todas.

# COMMAND ----------

titulo("5. PRUEBA ANTI-FUGA")
origen_prueba = CORTES[0]["origen"]
pares_prueba = PARES_EVAL.where((F.col("corte") == CORTES[0]["corte"]) & (F.col("en_muestra_eval") == 1)).select(*GRUPO_SERIE)
P_trunc = P.where(F.col("fecha") <= F.lit(origen_prueba))
VARS_ESTADO = [c for c in COLUMNAS_ORO if c not in NO_VARIABLES and c not in
               ("ingenuo_estacional", "h_bloque", "dow", "fin_de_semana", "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento",
                "dias_a_evento", "dias_desde_evento", "en_mundial") and "lag_dow" not in c and "media_dow" not in c and c != "s_n_dow_8"]
VARS_H = [c for c in COLUMNAS_ORO if "lag_dow" in c or "media_dow" in c or c == "s_n_dow_8"]
# reconstrucción con el panel truncado: estado + rezagos (sin objetivo, porque el futuro no existe en el panel truncado)
RECON = abrir_horizonte(estado_series(pares_prueba, P_trunc)).select(*CLAVES, "origen", "h", *VARS_ESTADO, *VARS_H)
GUARDADA = (G.where((F.col("conjunto") == "evaluacion") & (F.col("corte") == CORTES[0]["corte"]) & (F.col("en_muestra_eval") == 1))
            .select(*CLAVES, "origen", "h", *VARS_ESTADO, *VARS_H))
CATEG = ["tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "tipo_categoria"]
numericas = [c for c in VARS_ESTADO + VARS_H if c not in CATEG]
categoricas = [c for c in VARS_ESTADO if c in CATEG]
# Comparación fila por fila con tolerancia numérica: Spark suma en paralelo y el orden de la suma cambia entre corridas, así que
# dos cálculos idénticos pueden diferir en el último decimal. Una fuga real cambia el valor, no el décimo dígito.
TOL_ABS, TOL_REL = 1e-4, 1e-6
gs = GUARDADA.select(*CLAVES, "origen", "h", *[F.col(c).alias(f"g__{c}") for c in numericas + categoricas])
rs = RECON.select(*CLAVES, "origen", "h", F.lit(1).alias("r__existe"), *[F.col(c).alias(f"r__{c}") for c in numericas + categoricas])
J = gs.join(rs, [*CLAVES, "origen", "h"], "left")
aggs = [F.count("*").alias("filas"), F.sum(F.when(F.col("r__existe").isNull(), 1).otherwise(0)).alias("sin_reconstruccion")]
for c in numericas:
    g, r = F.col(f"g__{c}").cast("double"), F.col(f"r__{c}").cast("double")
    nulo_distinto = g.isNull() != r.isNull()
    dif = F.abs(g - r)
    fuera = F.coalesce(nulo_distinto | (dif > F.lit(TOL_ABS) + F.lit(TOL_REL) * F.abs(g)), F.lit(False))
    exacta = F.coalesce(nulo_distinto | (F.round(g, 4) != F.round(r, 4)), F.lit(False))
    aggs += [F.sum(fuera.cast("int")).alias(f"fuera__{c}"), F.sum(exacta.cast("int")).alias(f"exacta__{c}"),
             F.max(dif / F.greatest(F.abs(g), F.lit(1.0))).alias(f"rel__{c}")]
for c in categoricas:
    distinta = ~F.col(f"g__{c}").eqNullSafe(F.col(f"r__{c}"))
    aggs += [F.sum(distinta.cast("int")).alias(f"fuera__{c}"), F.sum(distinta.cast("int")).alias(f"exacta__{c}"), F.lit(0.0).alias(f"rel__{c}")]
R = J.agg(*aggs).collect()[0].asDict()
n_filas, sin_recon = int(R["filas"]), int(R["sin_reconstruccion"])
DIF = pd.DataFrame([{"variable": c, "filas_distintas_al_4o_decimal": int(R[f"exacta__{c}"] or 0),
                     "filas_fuera_de_tolerancia": int(R[f"fuera__{c}"] or 0), "max_diferencia_relativa": float(R[f"rel__{c}"] or 0.0)}
                    for c in numericas + categoricas])
DIF = DIF[(DIF["filas_distintas_al_4o_decimal"] > 0) | (DIF["filas_fuera_de_tolerancia"] > 0)].sort_values(
    ["filas_fuera_de_tolerancia", "filas_distintas_al_4o_decimal"], ascending=False)
n_fuera = int(DIF["filas_fuera_de_tolerancia"].sum()) if len(DIF) else 0
n_exactas = int(DIF["filas_distintas_al_4o_decimal"].sum()) if len(DIF) else 0
print(f"Filas comparadas: {n_filas:,} (origen {origen_prueba}, muestra de evaluación, {len(numericas)} variables numéricas y "
      f"{len(categoricas)} categóricas) · sin reconstrucción: {sin_recon:,} · diferencias fuera de tolerancia: {n_fuera:,} · "
      f"diferencias sólo en el 4.º decimal: {n_exactas:,}")
if len(DIF):
    mostrar(DIF, "Variables con alguna diferencia entre el panel completo y el truncado",
            formatos={"filas_distintas_al_4o_decimal": "int", "filas_fuera_de_tolerancia": "int", "max_diferencia_relativa": "dec4"})
if n_filas > 0 and n_fuera == 0 and sin_recon == 0:
    detalle = (f" {n_exactas:,} valores difieren sólo en el cuarto decimal (máx. diferencia relativa "
               f"{DIF['max_diferencia_relativa'].max():.1e}), por el orden en que Spark suma en paralelo; no es fuga." if n_exactas else "")
    hallazgo(f"Prueba anti-fuga en PASA: las {len(numericas) + len(categoricas)} variables del origen {origen_prueba} coinciden, calculadas con el "
             f"panel completo y con el panel truncado en el origen, en las {n_filas:,} filas de la muestra (tolerancia {TOL_ABS:g} + {TOL_REL:g}·|x|)." + detalle)
else:
    aviso(f"Prueba anti-fuga en FALLA: {n_fuera:,} valores fuera de tolerancia y {sin_recon:,} filas sin reconstrucción en {origen_prueba}.")
    raise RuntimeError("Fuga de información en la tabla Oro: revisar estado_series / abrir_horizonte antes de entrenar.")
liberar(SER)

# COMMAND ----------

cerrar("20_features_oro", siguiente="Avance5.43 (2_modelos)")
