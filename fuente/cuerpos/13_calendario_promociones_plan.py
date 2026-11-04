# Databricks notebook source
# MAGIC %md
# MAGIC # 13 · Calendario de promociones planeadas
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Convierte el **archivo editable** del calendario de promociones (Excel o CSV en el volumen `outputs/calendario/`) en las
# MAGIC tablas que usan los modelos y el what-if. El equipo (o Bnext) sólo edita el archivo; este notebook lo valida y lo
# MAGIC publica.
# MAGIC
# MAGIC **Fuentes del plan.**
# MAGIC 1. `calendario_promociones_plan.xlsx` (hoja `promociones`): promociones **futuras** — hoy un *placeholder* inventado con
# MAGIC    `fuente = placeholder`, que se sustituye por el plan real de Bnext (`fuente = bnext`) editando el archivo. Si el
# MAGIC    archivo no existe, este notebook escribe la plantilla en el volumen para que se edite.
# MAGIC 2. La vigencia **observada** de las promociones pasadas (`silver.calendario_promociones` del 09) como aproximación del
# MAGIC    plan histórico: se supone que cada promoción se conocía `DIAS_ANTICIPACION_HISTORICO` días antes de empezar
# MAGIC    (`fuente = historico_observado`). Con eso el modelo aprende el efecto de una promoción anunciada.
# MAGIC
# MAGIC **Sin fuga.** Cada promoción lleva `fecha_publicacion`; al construir variables para un origen T sólo cuentan las
# MAGIC publicadas en o antes de T (`unir_promociones_planeadas`).
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | el archivo del volumen · `silver.calendario_promociones`, `silver.promociones`, `silver.catalogo_categorias`, `silver.catalogo_tiendas` |
# MAGIC | **Escribe** | `silver.calendario_promociones_plan` (promoción × categoría × alcance) · `silver.calendario_promociones_plan_dia` (tienda × categoría × día) · `mlops.calendario_plan_versiones` · `outputs/calendario/catalogo_categorias.csv` |
# MAGIC | **Parámetros** | `ARCHIVO_CALENDARIO`, `INCLUIR_HISTORICO_OBSERVADO`, `DIAS_ANTICIPACION_DEFECTO`, `DIAS_ANTICIPACION_HISTORICO`, `FALLAR_SI_INVALIDO` |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# MAGIC %pip install "openpyxl>=3.1,<4"

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# @@PROMOCIONES

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
CARPETA_CALENDARIO = volumen("outputs", "calendario")
ARCHIVO_CALENDARIO = parametro("ARCHIVO_CALENDARIO", CARPETA_CALENDARIO + "/calendario_promociones_plan.xlsx")
# Excel (.xlsx, hoja 'promociones') o CSV con las columnas del contrato ESQUEMA_CALENDARIO_PROMOS.
INCLUIR_HISTORICO_OBSERVADO = parametro("INCLUIR_HISTORICO_OBSERVADO", True)
# True → las promociones pasadas observadas (09) entran como plan histórico; False → sólo el archivo.
DIAS_ANTICIPACION_DEFECTO = parametro("DIAS_ANTICIPACION_DEFECTO", 14)      # fecha_publicacion si el archivo no la trae
DIAS_ANTICIPACION_HISTORICO = parametro("DIAS_ANTICIPACION_HISTORICO", 14)  # cuántos días antes se "conocía" una promoción observada
FALLAR_SI_INVALIDO = parametro("FALLAR_SI_INVALIDO", False)                 # True en Jobs: detenerse si el archivo tiene errores
IMPUTAR_DESCUENTO = parametro("IMPUTAR_DESCUENTO", True)
# True → si una fila no trae descuento_planeado, se usa el descuento histórico mediano de la categoría (de las promociones
#        observadas); así una promoción de la que sólo se conocen las fechas entra al modelo con un descuento plausible.
TIPO_TODAS_DEFECTO = parametro("TIPO_TODAS_DEFECTO", "mercancia")
# A qué tipo de categoría se expande una fila con categoria = TODAS cuando no indica tipo_categoria ("" = todos los tipos).
TABLA_VERSIONES = tabla("mlops", "calendario_plan_versiones")
print("Archivo del plan:", ARCHIVO_CALENDARIO)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leer el archivo editable (o escribir la plantilla si no existe)

# COMMAND ----------

import hashlib
import io

COLUMNAS_PLAN = list(ESQUEMA_CALENDARIO_PROMOS)
# Placeholder INVENTADO para arrancar (las mismas filas de calendario/calendario_promociones_plan.xlsx del repositorio).
PLANTILLA_PLACEHOLDER = [
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "REFRESCOS", "", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA: cae dentro del horizonte del pronóstico de octubre"),
    ("PH-2026-10-OTONO", "Arranque de otoño refrescos y cerveza", "2026-10-02", "2026-10-11", "", "CERVEZA", "", "", "", "", 0.10, "descuento", "2026-09-20", "placeholder", "INVENTADA"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "BOTANAS", "", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA: sustituir por el plan real"),
    ("PH-2026-10-HALLOWEEN", "Halloween botanas y dulces", "2026-10-26", "2026-10-31", "", "CACAHUATE Y SEMILLAS", "", "", "", "", 0.10, "descuento", "2026-10-10", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "PAN Y GALLETA", "", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-DIA-MUERTOS", "Día de Muertos pan y café", "2026-10-30", "2026-11-02", "", "CAFES", "", "", "", "", 0.15, "descuento", "2026-10-15", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "CERVEZA", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA; fechas del Buen Fin por confirmar"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "REFRESCOS", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "BOTANAS", "", "", "", "", 0.20, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-BUEN-FIN", "El Buen Fin 2026", "2026-11-13", "2026-11-16", "", "AGUAS", "", "", "", "", 0.15, "descuento", "2026-10-20", "placeholder", "INVENTADA"),
    ("PH-2026-11-2X1-YOGURT", "2x1 yogurt noviembre", "2026-11-17", "2026-11-30", "", "YOGURT", "", "", "", "", 0.50, "lleva 2 paga 1", "2026-11-01", "placeholder", "INVENTADA; 2x1 ≈ 50 % de descuento"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "CERVEZA", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "BOTANAS", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-POSADAS", "Posadas cerveza y botanas", "2026-12-12", "2026-12-24", "", "REFRESCOS", "", "", "", "", 0.10, "descuento", "2026-11-25", "placeholder", "INVENTADA"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CERVEZA", "", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA; se traslapa con posadas a propósito"),
    ("PH-2026-12-NAVIDAD", "Navidad y Año Nuevo", "2026-12-20", "2026-12-31", "", "CHICHARRONES", "", "", "", "", 0.15, "descuento", "2026-12-01", "placeholder", "INVENTADA"),
    ("PH-2026-11-REGIONAL-EJEMPLO", "Ejemplo de alcance regional", "2026-11-20", "2026-11-22", "", "PALOMITAS", "", "", "", "NORTE", 0.20, "descuento", "2026-11-05", "placeholder", "INVENTADA: ejemplo de alcance por región"),
    ("PH-2026-11-SOLO-FECHAS", "Promoción de la que sólo se conocen las fechas (ejemplo)", "2026-11-05", "2026-11-08", "TODAS", "", "mercancia", "", "", "", "", "descuento", "2026-10-25", "placeholder", "INVENTADA: ejemplo de fechas sin categoría ni descuento; se expande a toda la mercancía con descuento imputado"),
]


def escribir_plantilla(ruta):
    """Escribe la plantilla (Excel o CSV según la extensión) con el placeholder y las instrucciones."""
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    datos = pd.DataFrame(PLANTILLA_PLACEHOLDER, columns=COLUMNAS_PLAN)
    if ruta.lower().endswith(".csv"):
        datos.to_csv(ruta, index=False)
        return
    instrucciones = pd.DataFrame({"instrucciones": [
        "Cada fila es una promoción × categoría × alcance; la misma promo_id puede repetirse.",
        "Obligatorias: promo_id, inicio_planeado, fin_planeado (AAAA-MM-DD) y categoria_id o categoria_nombre.",
        "categoria_nombre se traduce con silver.catalogo_categorias (ver catalogo_categorias.csv en esta carpeta).",
        "Alcance: tienda, zona y region vacíos = nacional; llena sólo el nivel que aplique.",
        "descuento_planeado en fracción (0.15 = 15 %); 2x1 = 0.50. fecha_publicacion = desde cuándo se conoce el plan.",
        "fuente: bnext (plan real) o placeholder (inventada: sustituir). Guardar, subir al volumen y correr 13.",
    ]})
    with pd.ExcelWriter(ruta, engine="openpyxl") as w:
        datos.to_excel(w, sheet_name="promociones", index=False)
        instrucciones.to_excel(w, sheet_name="instrucciones", index=False)


def leer_archivo(ruta):
    if ruta.lower().endswith(".csv"):
        return pd.read_csv(ruta, dtype=str, keep_default_na=False)
    return pd.read_excel(ruta, sheet_name="promociones", dtype=str, keep_default_na=False)


titulo("1. ARCHIVO DEL CALENDARIO")
if not os.path.exists(ARCHIVO_CALENDARIO):
    escribir_plantilla(ARCHIVO_CALENDARIO)
    aviso(f"No existía el archivo del calendario: se escribió la plantilla con promociones placeholder (inventadas) en {ARCHIVO_CALENDARIO}. "
          "Edítala con el plan real de Bnext y vuelve a correr este notebook.")
with open(ARCHIVO_CALENDARIO, "rb") as f:
    contenido = f.read()
VERSION_PLAN = hashlib.sha256(contenido).hexdigest()[:12]
PLAN_CRUDO = leer_archivo(ARCHIVO_CALENDARIO)
PLAN_CRUDO.columns = [str(c).strip().lower() for c in PLAN_CRUDO.columns]
for c in COLUMNAS_PLAN:
    if c not in PLAN_CRUDO.columns:
        PLAN_CRUDO[c] = ""
PLAN_CRUDO = PLAN_CRUDO[COLUMNAS_PLAN].map(lambda v: str(v).strip() if v is not None else "") if hasattr(PLAN_CRUDO, "map") else PLAN_CRUDO[COLUMNAS_PLAN].applymap(lambda v: str(v).strip())
PLAN_CRUDO = PLAN_CRUDO[(PLAN_CRUDO["promo_id"] != "") | (PLAN_CRUDO["inicio_planeado"] != "")]
print(f"Archivo: {os.path.basename(ARCHIVO_CALENDARIO)} · versión {VERSION_PLAN} · {len(PLAN_CRUDO)} filas · "
      f"fuentes: {dict(PLAN_CRUDO['fuente'].replace('', 'sin_fuente').value_counts())}")

# COMMAND ----------

# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# CUANDO BNEXT ENTREGUE SU CATÁLOGO DE PROMOCIONES COMO TABLA O ARCHIVO (en lugar del Excel): descomentar y ajustar
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════════
# 1. Indicar dónde está el catálogo (tabla de Unity Catalog, parquet o CSV en el volumen):
# ORIGEN_CATALOGO_BNEXT = tabla("silver", "promociones_bnext")          # o volumen("outputs", "calendario", "catalogo_bnext.csv")
# 2. Mapear los nombres de columnas de Bnext a las del contrato (ESQUEMA_CALENDARIO_PROMOS); las que no existan se dejan "":
# MAPEO_COLUMNAS_BNEXT = {
#     "promo_id": "id_promocion",            # identificador de la promoción en Bnext
#     "nombre": "nombre_promocion",
#     "inicio_planeado": "fecha_inicio",
#     "fin_planeado": "fecha_fin",
#     "categoria_id": "id_categoria",        # o "categoria_nombre": "categoria" si viene el nombre; "TODAS" si aplica a todas
#     "tienda": "",                          # vacío = nacional; o "store_code" si viene por tienda
#     "zona": "", "region": "",
#     "descuento_planeado": "porcentaje_descuento",   # 0–1 o 0–100 (el notebook lo normaliza); vacío = se imputa
#     "tipo_mecanica": "mecanica",
#     "fecha_publicacion": "fecha_aprobacion",        # vacío = inicio − DIAS_ANTICIPACION_DEFECTO
#     "comentario": "",
# }
# 3. Leer, renombrar y usar en lugar del Excel (todo lo demás del notebook sigue igual):
# if ORIGEN_CATALOGO_BNEXT.count(".") == 2:
#     crudo = spark.table(ORIGEN_CATALOGO_BNEXT).toPandas()
# elif ORIGEN_CATALOGO_BNEXT.lower().endswith(".csv"):
#     crudo = pd.read_csv(ORIGEN_CATALOGO_BNEXT, dtype=str, keep_default_na=False)
# else:
#     crudo = spark.read.parquet(ORIGEN_CATALOGO_BNEXT).toPandas()
# PLAN_CRUDO = pd.DataFrame({c: (crudo[origen].astype(str) if origen in crudo.columns else "") for c, origen in MAPEO_COLUMNAS_BNEXT.items()})
# for c in COLUMNAS_PLAN:
#     if c not in PLAN_CRUDO.columns:
#         PLAN_CRUDO[c] = ""
# PLAN_CRUDO["fuente"] = "bnext"
# PLAN_CRUDO = PLAN_CRUDO[COLUMNAS_PLAN]
# VERSION_PLAN = hashlib.sha256(PLAN_CRUDO.to_csv(index=False).encode("utf-8")).hexdigest()[:12]
# print(f"Catálogo de Bnext: {len(PLAN_CRUDO)} filas · versión {VERSION_PLAN}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Validación y traducción de categorías y alcances
# MAGIC Se traducen `categoria_nombre` → `categoria_id` con el catálogo, se completan `fecha_publicacion` y `descuento_planeado`,
# MAGIC se revisan fechas, rangos y alcances (región, zona o tienda que existan). Los errores se listan sin datos sensibles;
# MAGIC las filas con error se omiten (o detienen el notebook si `FALLAR_SI_INVALIDO = True`).
# MAGIC
# MAGIC **Si sólo se conocen las fechas** de una promoción: `categoria_id = TODAS` (opcionalmente `tipo_categoria = mercancia`)
# MAGIC la expande a todas las categorías de ese tipo y, si no trae descuento, se le imputa el descuento histórico mediano de
# MAGIC cada categoría (`descuento_imputado = 1`). Es una aproximación declarada: cuando llegue el detalle por categoría, se
# MAGIC sustituyen esas filas.

# COMMAND ----------

titulo("2. VALIDACIÓN")
CAT = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata").select(
    F.col("categoria_id").cast("string").alias("categoria_id"), F.upper(F.trim(F.col("nombre"))).alias("nombre"), "tipo_categoria").toPandas()
TIENDAS = leer_tabla(tabla("silver", "catalogo_tiendas"), "08_limpieza_plata").select(
    F.col("tienda").cast("string").alias("tienda"), F.col("zona").cast("string").alias("zona"),
    F.col("region").cast("string").alias("region")).toPandas()
try:      # catálogo de referencia junto al archivo, para que quien edite el Excel encuentre los identificadores
    CAT.sort_values("nombre").to_csv(CARPETA_CALENDARIO + "/catalogo_categorias.csv", index=False)
    print(f"Catálogo de categorías de referencia escrito en {CARPETA_CALENDARIO}/catalogo_categorias.csv ({len(CAT)} categorías)")
except Exception as e:
    print(f"(no se pudo escribir el catálogo de referencia: {type(e).__name__})")

nombre_a_id = dict(zip(CAT["nombre"], CAT["categoria_id"]))
ids_validos = set(CAT["categoria_id"])
plan = PLAN_CRUDO.copy()
plan["fila"] = range(2, len(plan) + 2)      # número de fila en el archivo (encabezado = 1)
# ---- TODAS: una promoción de la que sólo se conocen las fechas se expande a todas las categorías (del tipo indicado)
es_todas = (plan["categoria_id"].str.upper() == "TODAS") | ((plan["categoria_id"] == "") & (plan["categoria_nombre"].str.upper() == "TODAS"))
if es_todas.any():
    tipo_filtro = plan["tipo_categoria"].str.lower().where(plan["tipo_categoria"] != "", TIPO_TODAS_DEFECTO)
    expandidas = []
    for i in plan.index[es_todas]:
        cats = CAT[CAT["tipo_categoria"].str.lower() == tipo_filtro[i]] if tipo_filtro[i] else CAT
        for cid in cats["categoria_id"]:
            fila = plan.loc[i].copy()
            fila["categoria_id"], fila["categoria_nombre"] = cid, ""
            expandidas.append(fila)
    n_todas = int(es_todas.sum())
    plan = pd.concat([plan[~es_todas], pd.DataFrame(expandidas)], ignore_index=True)
    hallazgo(f"{n_todas} filas con categoría TODAS se expandieron a {len(expandidas)} filas promoción × categoría "
             f"(tipo por defecto: {TIPO_TODAS_DEFECTO or 'todos'}).")
plan["categoria_id"] = plan["categoria_id"].where(plan["categoria_id"] != "", plan["categoria_nombre"].str.upper().map(nombre_a_id).fillna(""))
plan["descuento_planeado"] = pd.to_numeric(plan["descuento_planeado"].replace("", np.nan), errors="coerce")
plan.loc[plan["descuento_planeado"] > 1, "descuento_planeado"] = plan.loc[plan["descuento_planeado"] > 1, "descuento_planeado"] / 100  # 15 → 0.15
plan["descuento_imputado"] = 0
if IMPUTAR_DESCUENTO and plan["descuento_planeado"].isna().any() and existe(tabla("silver", "calendario_promociones")):
    hist = (spark.table(tabla("silver", "calendario_promociones")).select("promo_id", F.col("categoria_id").cast("string").alias("categoria_id"))
            .join(spark.table(tabla("silver", "promociones")).select("promo_id", "tasa_media"), "promo_id").where(F.col("tasa_media") > 0)
            .groupBy("categoria_id").agg(F.expr("percentile_approx(tasa_media, 0.5)").alias("descuento_hist")).toPandas())
    desc_cat = dict(zip(hist["categoria_id"], hist["descuento_hist"]))
    desc_global = float(hist["descuento_hist"].median()) if len(hist) else 0.0
    faltante = plan["descuento_planeado"].isna()
    plan.loc[faltante, "descuento_planeado"] = plan.loc[faltante, "categoria_id"].map(desc_cat).fillna(desc_global)
    plan.loc[faltante, "descuento_imputado"] = 1
    hallazgo(f"{int(faltante.sum())} filas sin descuento: se imputó el descuento histórico mediano de su categoría "
             f"(global {desc_global:.0%}); quedan marcadas con descuento_imputado = 1.")
plan["descuento_planeado"] = plan["descuento_planeado"].fillna(0.0).clip(0, 1)
for c in ("inicio_planeado", "fin_planeado", "fecha_publicacion"):
    plan[c] = pd.to_datetime(plan[c].replace("", np.nan), errors="coerce")
plan["fecha_publicacion"] = plan["fecha_publicacion"].fillna(plan["inicio_planeado"] - pd.Timedelta(days=DIAS_ANTICIPACION_DEFECTO))
plan["fuente"] = plan["fuente"].replace("", "bnext").str.lower()
plan["tipo_mecanica"] = plan["tipo_mecanica"].replace("", "descuento")

errores = []
def marcar(mascara, motivo):
    for fila in plan.loc[mascara, "fila"]:
        errores.append({"fila": int(fila), "motivo": motivo})
marcar(plan["promo_id"] == "", "promo_id vacío")
marcar(plan["inicio_planeado"].isna() | plan["fin_planeado"].isna(), "fecha de inicio o fin vacía o inválida (usar AAAA-MM-DD)")
marcar(plan["fin_planeado"] < plan["inicio_planeado"], "fin anterior al inicio")
marcar((plan["categoria_id"] == "") | ~plan["categoria_id"].isin(ids_validos), "categoría no encontrada en silver.catalogo_categorias (revisar categoria_id o categoria_nombre)")
marcar((plan["region"] != "") & ~plan["region"].str.upper().isin(set(TIENDAS["region"].dropna().str.upper())), "región no existe en el catálogo de tiendas")
marcar((plan["zona"] != "") & ~plan["zona"].str.upper().isin(set(TIENDAS["zona"].dropna().str.upper())), "zona no existe en el catálogo de tiendas")
marcar((plan["tienda"] != "") & ~plan["tienda"].isin(set(TIENDAS["tienda"])), "tienda no existe en el catálogo de tiendas")
marcar((plan["fin_planeado"] - plan["inicio_planeado"]).dt.days > 120, "promoción de más de 120 días (revisar fechas)")
ERRORES = pd.DataFrame(errores)
if len(ERRORES):
    mostrar(ERRORES.groupby("motivo").agg(filas=("fila", lambda s: ", ".join(str(v) for v in sorted(s)[:15])), n=("fila", "size")).reset_index(),
            "Filas con error (se omiten)")
    if FALLAR_SI_INVALIDO:
        raise ValueError(f"El calendario tiene {ERRORES['fila'].nunique()} filas con error; corrígelas en el archivo.")
    aviso(f"{ERRORES['fila'].nunique()} filas del archivo tienen errores y se omiten (ver tabla); el resto del plan sí se publica.")
plan = plan[~plan["fila"].isin(set(ERRORES["fila"]) if len(ERRORES) else set())].copy()
# traslapes de la misma promoción × categoría × alcance
plan["alcance"] = np.where(plan["tienda"] != "", "tienda", np.where(plan["zona"] != "", "zona", np.where(plan["region"] != "", "region", "nacional")))
dup = plan.duplicated(subset=["promo_id", "categoria_id", "tienda", "zona", "region"], keep=False)
if dup.any():
    aviso(f"{int(dup.sum())} filas repiten promo_id × categoría × alcance: se conservan todas (cuentan como una sola promoción por día).")
resumen_plan = (plan.groupby("fuente").agg(promociones=("promo_id", "nunique"), filas=("promo_id", "size"),
                                            desde=("inicio_planeado", "min"), hasta=("fin_planeado", "max"),
                                            descuento_medio=("descuento_planeado", "mean")).reset_index())
mostrar(resumen_plan, "Plan válido por fuente", formatos={"descuento_medio": "pct"})
mostrar(plan.groupby("alcance").agg(filas=("promo_id", "size")).reset_index(), "Filas por tipo de alcance")
if (plan["fuente"] == "placeholder").any():
    aviso(f"{int((plan['fuente'] == 'placeholder').sum())} filas son placeholder (inventadas): sustituirlas por el plan real de Bnext en {os.path.basename(ARCHIVO_CALENDARIO)}.")
hallazgo(f"Calendario versión {VERSION_PLAN}: {plan['promo_id'].nunique()} promociones válidas del archivo en {plan['categoria_id'].nunique()} categorías "
         f"({plan['inicio_planeado'].min().date() if len(plan) else '—'} → {plan['fin_planeado'].max().date() if len(plan) else '—'}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Tabla del plan (promoción × categoría × alcance) y plan histórico observado

# COMMAND ----------

titulo("3. TABLAS DEL PLAN")
plan_tabla = plan.assign(inicio_planeado=plan["inicio_planeado"].dt.date, fin_planeado=plan["fin_planeado"].dt.date,
                         fecha_publicacion=plan["fecha_publicacion"].dt.date, version_plan=VERSION_PLAN, fecha_carga=pd.Timestamp.now())[
    ["promo_id", "nombre", "inicio_planeado", "fin_planeado", "categoria_id", "tienda", "zona", "region", "alcance",
     "descuento_planeado", "descuento_imputado", "tipo_mecanica", "fecha_publicacion", "fuente", "comentario", "version_plan", "fecha_carga"]]
guardar(plan_tabla, "silver", "calendario_promociones_plan",
        "Plan de promociones: una fila por promocion x categoria x alcance, leido del archivo editable del volumen (13)", modo="overwrite")
comentar_columnas(TABLA_CALENDARIO_PLAN, {k: v for k, v in ESQUEMA_CALENDARIO_PROMOS.items()})
PLAN_SDF = spark.table(TABLA_CALENDARIO_PLAN).withColumn("tienda", F.coalesce(F.col("tienda"), F.lit(""))) \
    .withColumn("zona", F.coalesce(F.col("zona"), F.lit(""))).withColumn("region", F.coalesce(F.col("region"), F.lit("")))

# ---- expansión del archivo a tiendas según el alcance (nacional / región / zona / tienda)
tiendas_sdf = spark.createDataFrame(TIENDAS[["tienda", "zona", "region"]].fillna(""))
archivo_dia = (PLAN_SDF.alias("p").join(tiendas_sdf.alias("t"),
                                       ((F.col("p.tienda") == "") | (F.col("p.tienda") == F.col("t.tienda")))
                                       & ((F.col("p.zona") == "") | (F.upper(F.col("p.zona")) == F.upper(F.col("t.zona"))))
                                       & ((F.col("p.region") == "") | (F.upper(F.col("p.region")) == F.upper(F.col("t.region")))))
               .select(F.col("t.tienda").alias("tienda"), F.col("p.categoria_id").alias("categoria_id"), F.col("p.promo_id").alias("promo_id"),
                       F.col("p.inicio_planeado").alias("inicio"), F.col("p.fin_planeado").alias("fin"),
                       F.col("p.fecha_publicacion").alias("fecha_publicacion"), F.col("p.descuento_planeado").alias("descuento"),
                       F.col("p.fuente").alias("fuente")))

# ---- histórico observado (09): la vigencia real de cada promoción por tienda × categoría, "publicada" N días antes
if INCLUIR_HISTORICO_OBSERVADO and existe(tabla("silver", "calendario_promociones")):
    promos_arch = {str(v) for v in plan["promo_id"].unique()}
    obs = (spark.table(tabla("silver", "calendario_promociones"))
           .join(spark.table(tabla("silver", "promociones")).select("promo_id", F.col("tasa_media").alias("descuento")), "promo_id", "left")
           .where(~F.col("promo_id").cast("string").isin(list(promos_arch)) if promos_arch else F.lit(True))
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   F.col("promo_id").cast("string").alias("promo_id"), F.col("inicio").cast("date").alias("inicio"), F.col("fin").cast("date").alias("fin"),
                   F.date_sub(F.col("inicio").cast("date"), int(DIAS_ANTICIPACION_HISTORICO)).alias("fecha_publicacion"),
                   F.coalesce(F.col("descuento").cast("double"), F.lit(0.0)).alias("descuento"), F.lit("historico_observado").alias("fuente")))
    n_obs = obs.agg(F.countDistinct("promo_id").alias("p"), F.count("*").alias("f"), F.min("inicio").alias("d"), F.max("fin").alias("h")).collect()[0]
    hallazgo(f"Plan histórico observado: {n_obs['p']:,} promociones ({n_obs['d']} → {n_obs['h']}) con fecha de publicación supuesta "
             f"{DIAS_ANTICIPACION_HISTORICO} días antes de su inicio.")
    todo = archivo_dia.unionByName(obs)
else:
    if INCLUIR_HISTORICO_OBSERVADO:
        aviso("No existe silver.calendario_promociones (corre 09): el plan sólo tiene lo que trae el archivo.")
    todo = archivo_dia

# ---- tienda × categoría × día con las promociones que lo cubren (lo que une la Oro)
dias = (todo.withColumn("fecha", F.explode(F.sequence(F.col("inicio"), F.col("fin"))))
        .withColumn("restantes", F.datediff(F.col("fin"), F.col("fecha"))))
PLAN_DIA = (dias.groupBy("tienda", "categoria_id", "fecha")
            .agg(F.collect_list(F.struct(F.col("promo_id"), F.col("fecha_publicacion"), F.col("descuento"), F.col("restantes"), F.col("fuente"))).alias("promos"),
                 F.countDistinct("promo_id").alias("n_promos"), F.max("descuento").alias("descuento_max"),
                 F.max(F.when(F.col("fuente") != "historico_observado", 1).otherwise(0)).alias("del_archivo"))
            .withColumn("version_plan", F.lit(VERSION_PLAN)).withColumn("mes", F.date_format("fecha", "yyyy-MM")))
guardar(PLAN_DIA, "silver", "calendario_promociones_plan_dia",
        "Promociones planeadas por tienda x categoria x dia (archivo + historico observado); cada promo trae su fecha_publicacion", particion="mes")
try:
    spark.sql(f"ALTER TABLE {TABLA_CALENDARIO_PLAN_DIA} SET TBLPROPERTIES ('mna.version_plan' = '{VERSION_PLAN}', "
              f"'mna.dias_anticipacion_historico' = '{DIAS_ANTICIPACION_HISTORICO}')")
    spark.sql(f"OPTIMIZE {TABLA_CALENDARIO_PLAN_DIA} ZORDER BY (tienda, categoria_id)")
except Exception as e:
    print(f"(propiedades u OPTIMIZE no disponibles: {type(e).__name__})")
comentar_columnas(TABLA_CALENDARIO_PLAN_DIA, {
    "tienda": "Tienda a la que aplica.", "categoria_id": "Categoría canónica.", "fecha": "Día cubierto por al menos una promoción planeada.",
    "promos": "Promociones que cubren el día: promo_id, fecha_publicacion, descuento (0-1), dias restantes, fuente.",
    "n_promos": "Promociones distintas ese día.", "descuento_max": "Mayor descuento planeado ese día.",
    "del_archivo": "1 si alguna viene del archivo editable (plan de Bnext o placeholder); 0 si todas son historico observado.",
    "version_plan": "Huella del archivo con el que se construyo.", "mes": "Particion AAAA-MM."})
r = spark.table(TABLA_CALENDARIO_PLAN_DIA).agg(F.count("*").alias("filas"), F.countDistinct("tienda").alias("tiendas"),
                                                F.countDistinct("categoria_id").alias("categorias"), F.min("fecha").alias("desde"), F.max("fecha").alias("hasta"),
                                                F.sum("del_archivo").alias("dias_del_archivo")).collect()[0].asDict()
mostrar(pd.DataFrame([r]), "Tabla tienda × categoría × día", formatos={"filas": "int", "tiendas": "int", "categorias": "int", "dias_del_archivo": "int"})
hallazgo(f"{TABLA_CALENDARIO_PLAN_DIA}: {r['filas']:,} filas tienda × categoría × día ({r['desde']} → {r['hasta']}); "
         f"{r['dias_del_archivo']:,} provienen del archivo editable.")
cobertura = (archivo_dia.groupBy("promo_id", "fuente").agg(F.countDistinct("tienda").alias("tiendas"), F.countDistinct("categoria_id").alias("categorias"))
             .orderBy("promo_id").toPandas())
mostrar(cobertura, "Alcance de cada promoción del archivo (tiendas y categorías a las que aplica)", formatos={"tiendas": "int", "categorias": "int"})
sin_tiendas = cobertura[cobertura["tiendas"] == 0]
if len(sin_tiendas):
    aviso(f"Promociones del archivo sin tiendas en su alcance: {', '.join(sin_tiendas['promo_id'])} (revisar región/zona/tienda).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Bitácora de versiones del plan

# COMMAND ----------

titulo("4. BITÁCORA")
bitacora = pd.DataFrame([{"version_plan": VERSION_PLAN, "archivo": os.path.basename(ARCHIVO_CALENDARIO), "fecha_carga": pd.Timestamp.now(),
                          "promociones_archivo": int(plan["promo_id"].nunique()), "filas_archivo": int(len(plan)),
                          "filas_con_error": int(ERRORES["fila"].nunique()) if len(ERRORES) else 0,
                          "placeholder": int((plan["fuente"] == "placeholder").sum()), "historico_observado": bool(INCLUIR_HISTORICO_OBSERVADO),
                          "filas_plan_dia": int(r["filas"]), "desde": str(r["desde"]), "hasta": str(r["hasta"])}])
guardar(bitacora, "mlops", "calendario_plan_versiones", "Versiones del calendario de promociones planeadas cargadas (13)", modo="append")
mostrar(bitacora.T.reset_index().rename(columns={"index": "campo", 0: "valor"}), "Esta carga")

# COMMAND ----------

cerrar("13_calendario_promociones_plan", siguiente="20_features_oro (las variables f_* se unen al entrenar y pronosticar)")
