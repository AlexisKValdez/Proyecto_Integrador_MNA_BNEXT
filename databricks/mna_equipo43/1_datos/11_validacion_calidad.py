# Databricks notebook source
# MAGIC %md
# MAGIC # 11 · Validación de calidad (contratos de datos)
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Es la compuerta antes de construir variables y entrenar: revisa reglas que **siempre** deben
# MAGIC cumplirse en Bronce, Plata y el panel. En la etapa de MLOps este notebook corre automáticamente antes de cada
# MAGIC reentrenamiento; si una regla falla, el pipeline se detiene.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `bronze.ventas_lineas` · `silver.ventas_lineas` · catálogos de Plata · `silver.panel_categoria_dia` · `mlops.bronce_cargas` · `mlops.conciliacion_plata` · `mlops.linea_base_ingenua` |
# MAGIC | **Escribe** | `mlops.validaciones` (una fila por regla y corrida) |
# MAGIC | **Tiempo** | 5–15 min |

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PROYECTO · es la misma en los 11 notebooks (si cambias algo, cámbialo en todos)
# ════════════════════════════════════════════════════════════════════════════════════════════════
CATALOGO       = "modeladoprescriptivoventas"     # catálogo de Unity Catalog del proyecto
ESQUEMA_DATOS  = "maestria"                       # esquema donde están los volúmenes de Bnext
RAIZ_VOLUMENES = "/Volumes"
MESES = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09"]   # un volumen por mes

# Cortes del walk-forward. Son FIJOS (los del Avance 4): así, al llegar un mes nuevo, los modelos se siguen comparando
# en las mismas ventanas. Los bloques de prueba son datos no vistos: no intervienen en el ajuste ni en la elección del
# modelo y sólo existen cuando ya están cargados sus días (septiembre).
DIAS_CORTE = 14                                           # días que se pronostican desde cada origen
ORIGEN_AJUSTE = "2026-05-20"                              # bloque 21-may…03-jun: sólo para afinar hiperparámetros
ORIGENES_CORTES = {"corte_1": "2026-06-03", "corte_2": "2026-06-17",      # desarrollo: con estos se compara y se elige
                   "corte_3": "2026-07-01", "corte_4": "2026-08-17"}
ORIGENES_PRUEBA = {"prueba_1": "2026-08-31", "prueba_2": "2026-09-14"}    # prueba no vista: 01–14 y 15–28 de septiembre

# Dónde se guardan las tablas del equipo:
#   False → dentro de maestria con prefijo de capa: modeladoprescriptivoventas.maestria.bronze_ventas_lineas,
#           maestria.silver_ventas_lineas, maestria.mlops_hallazgos, … (es lo que permiten hoy los permisos del equipo)
#   True  → esquemas propios bronze / silver / gold / mlops, si algún día Bnext da permiso de CREATE SCHEMA en el catálogo
ESQUEMAS_PROPIOS = False

# Qué columna define el día de la venta y en qué zona horaria vienen sus horas. El notebook 05 compara las candidatas;
# si recomienda otra columna o detecta horas en UTC, cámbialo aquí en todos los notebooks y vuelve a correr del 05 en adelante.
COLUMNA_FECHA_VENTA = "finish_order"
HORAS_EN_UTC = False

VOLUMENES = f"{RAIZ_VOLUMENES}/{CATALOGO}/{ESQUEMA_DATOS}"      # /Volumes/modeladoprescriptivoventas/maestria

if ESQUEMAS_PROPIOS:      # si los esquemas propios no existen, se usa maestria (así nunca falla por esquema inexistente)
    try:
        _existentes = {r[0] for r in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect()}
    except Exception:
        _existentes = set()
    if not {"bronze", "silver", "gold", "mlops"} <= _existentes:
        ESQUEMAS_PROPIOS = False
        print("(los esquemas bronze/silver/gold/mlops no existen: las tablas se guardan en maestria con prefijo)")


def tabla(capa: str, nombre: str) -> str:
    """Nombre completo de una tabla del equipo. capa = bronze, silver, gold o mlops.
    tabla("bronze", "ventas_lineas") → modeladoprescriptivoventas.bronze.ventas_lineas"""
    if ESQUEMAS_PROPIOS:
        return f"{CATALOGO}.{capa}.{nombre}"
    return f"{CATALOGO}.{ESQUEMA_DATOS}.{capa}_{nombre}"


def volumen(*partes: str) -> str:
    """Ruta dentro de los volúmenes de maestria. volumen("2026-03") → /Volumes/modeladoprescriptivoventas/maestria/2026-03"""
    return "/".join([VOLUMENES, *partes])


def cortes_walk_forward(dias_con_datos, incluir_prueba=True):
    """Cortes configurados arriba como lista de {corte, origen, desde, hasta, fase}. Un corte sólo se usa si su bloque de
    14 días y los 14 días previos a su origen tienen datos (así ninguno cae en un hueco de entrega como julio 16–31).
    Los de prueba aparecen únicamente cuando sus días ya están cargados."""
    import pandas as pd
    dias = {pd.Timestamp(d).date() for d in dias_con_datos}
    fuentes = list(ORIGENES_CORTES.items()) + (list(ORIGENES_PRUEBA.items()) if incluir_prueba else [])
    salida = []
    for nombre, origen in fuentes:
        o = pd.Timestamp(origen)
        necesarios = {(o + pd.Timedelta(days=k)).date() for k in range(-DIAS_CORTE + 1, DIAS_CORTE + 1)}
        if not necesarios <= dias:
            if nombre.startswith("corte_"):
                print(f"⚠ {nombre} (origen {o.date()}): faltan días con datos en su bloque o en los 14 días previos; se omite")
            continue
        salida.append({"corte": nombre, "origen": o.date(), "desde": (o + pd.Timedelta(days=1)).date(),
                       "hasta": (o + pd.Timedelta(days=DIAS_CORTE)).date(),
                       "fase": "desarrollo" if nombre.startswith("corte_") else "prueba"})
    return salida


print("Catálogo         :", CATALOGO)
print("Datos crudos     :", ", ".join(volumen(m) for m in MESES[:2]), "…", volumen(MESES[-1]))
print("Tablas del equipo:", tabla("bronze", "…"), "|", tabla("silver", "…"), "|", tabla("mlops", "…"))

# COMMAND ----------

# ─────────────────────── funciones de apoyo (iguales en todos los notebooks; no hace falta editarlas) ───────────────────────
import datetime as dt
import json
import os
import re
import time

import numpy as np
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import functions as F

import logging
import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message=".*No Partition Defined for Window.*")   # ventanas globales pequeñas (ABC, rangos)
logging.getLogger("pyspark.sql.connect.logging").setLevel(logging.CRITICAL)   # serverless: no repetir errores ya manejados
logging.getLogger("py4j").setLevel(logging.CRITICAL)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 80)
def conf_set(clave, valor):
    """Cambia una configuración de Spark si el cómputo lo permite (serverless sólo acepta algunas). Devuelve True/False."""
    try:
        spark.conf.set(clave, valor)
        return True
    except Exception:
        return False


def persistir(df):
    """Guarda en memoria/disco un resultado que se usa varias veces. En serverless no está permitido: se sigue sin guardar."""
    try:
        from pyspark import StorageLevel
        return df.persist(StorageLevel.MEMORY_AND_DISK)
    except Exception:
        return df


def liberar(df):
    try:
        df.unpersist()
    except Exception:
        pass


conf_set("spark.sql.session.timeZone", "America/Mexico_City")
_INICIO_NOTEBOOK = time.time()
HALLAZGOS = []          # lo que este notebook encontró; se imprime al final y se guarda en mlops.hallazgos


def titulo(texto):
    print("\n" + "═" * 110 + f"\n{texto}\n" + "═" * 110)


def subtitulo(texto):
    print(f"\n── {texto} " + "─" * max(0, 104 - len(texto)))


def hallazgo(texto):
    """Un resultado que importa para decidir algo. Se repite en el resumen final."""
    HALLAZGOS.append(texto)
    print(f"► {texto}")


def aviso(texto):
    """Algo que hay que revisar o decidir antes de seguir."""
    HALLAZGOS.append(f"REVISAR: {texto}")
    print(f"⚠ {texto}")


_FMT = {"int": lambda v: f"{v:,.0f}", "pct": lambda v: f"{v:.1%}", "pct2": lambda v: f"{v:.2%}",
        "dinero": lambda v: f"${v:,.0f}", "dec1": lambda v: f"{v:,.1f}", "dec2": lambda v: f"{v:,.2f}",
        "dec3": lambda v: f"{v:,.3f}", "dec4": lambda v: f"{v:,.4f}",
        "auto": lambda v: f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.4f}"}


def _vacio(v):
    return v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NaT)


def mostrar(df, titulo_tabla=None, formatos=None, max_filas=60):
    """Imprime una tabla (pandas o Spark) completa y alineada, lista para copiar.
    formatos = {"columna": "int" | "pct" | "pct2" | "dinero" | "dec1" | "dec2" | "dec3" | "dec4" | "auto"}"""
    if titulo_tabla:
        subtitulo(titulo_tabla)
    if df is None:
        print("(sin datos)")
        return
    if not isinstance(df, pd.DataFrame):
        df = df.limit(max_filas + 1).toPandas()
    if df.empty:
        print("(sin filas)")
        return
    vista = df.head(max_filas).copy()
    for col, f in (formatos or {}).items():
        if col in vista.columns:
            vista[col] = vista[col].map(lambda v: "" if _vacio(v) else _FMT[f](float(v)))
    print(vista.to_string(index=False))
    if len(df) > max_filas:
        print(f"… y {len(df) - max_filas:,} filas más")


def _para_spark(pdf):
    """pandas → (pandas limpio, esquema de Spark): texto mixto a str, nulos de pandas a None, tipos explícitos
    (así una columna toda nula no rompe la creación de la tabla)."""
    from pyspark.sql import types as T
    pdf = pdf.copy()
    pdf.columns = [str(c) for c in pdf.columns]
    campos = []
    for c in pdf.columns:
        s = pdf[c]
        d = str(s.dtype)
        if d.startswith("datetime64"):
            if getattr(s.dt, "tz", None) is not None:
                pdf[c] = s.dt.tz_localize(None)
            tipo = T.TimestampType()
        elif d.startswith(("int", "uint")):
            tipo = T.LongType()
        elif d.startswith(("float", "Int", "Float", "UInt")):
            pdf[c] = s.astype("float64")
            tipo = T.DoubleType()
        elif d == "bool":
            tipo = T.BooleanType()
        elif d == "object" and len(s.dropna()) and all(isinstance(v, dt.date) and not isinstance(v, dt.datetime)
                                                       for v in s.dropna().head(200)):
            pdf[c] = pd.Series([None if _vacio(v) else v for v in s], index=s.index, dtype=object)
            tipo = T.DateType()
        else:
            pdf[c] = pd.Series([None if _vacio(v) or v is pd.NA else str(v) for v in s], index=s.index, dtype=object)
            tipo = T.StringType()
        campos.append(T.StructField(c, tipo, True))
    return pdf, T.StructType(campos)


def guardar(df, capa, nombre, comentario=None, modo="overwrite", particion=None):
    """Guarda un DataFrame (pandas o Spark) como tabla Delta del equipo y devuelve su nombre completo."""
    nombre_tabla = tabla(capa, nombre)
    if isinstance(df, pd.DataFrame):
        limpio, esquema = _para_spark(df)
        sdf = spark.createDataFrame(limpio, schema=esquema)
    else:
        sdf = df
    w = sdf.write.format("delta").mode(modo)
    w = w.option("overwriteSchema", "true") if modo == "overwrite" else w.option("mergeSchema", "true")
    if particion:
        w = w.partitionBy(particion)
    w.saveAsTable(nombre_tabla)
    if comentario:
        try:
            spark.sql(f"COMMENT ON TABLE {nombre_tabla} IS '{comentario.replace(chr(39), '')}'")
        except Exception:
            pass
    print(f"   → guardada {nombre_tabla}")
    return nombre_tabla


def tipos_de(df):
    """{columna: tipo de Spark} de un DataFrame."""
    return {f.name: f.dataType.simpleString() for f in df.schema.fields}


def leer_parquet(ruta):
    """Lee un parquet (archivo o carpeta de un volumen) con Spark, aunque traiga timestamps en nanosegundos
    (Spark no los lee directo): los convierte a timestamp conservando la hora del reloj."""
    import pyarrow.parquet as pq
    muestra = ruta
    if os.path.isdir(ruta):
        partes = sorted(os.path.join(b, a) for b, _, fs in os.walk(ruta) for a in fs if a.endswith(".parquet"))
        muestra = partes[0] if partes else ruta
    try:
        nanos = {f.name: f.type.tz is not None for f in pq.read_schema(muestra) if str(f.type).startswith("timestamp[ns")}
    except Exception:
        nanos = {}
    conf_set("spark.sql.parquet.inferTimestampNTZ.enabled", "true")   # horas sin zona = hora del reloj
    if nanos and not conf_set("spark.sql.legacy.parquet.nanosAsLong", "true"):
        # serverless no permite esa configuración: esquema explícito con los nanosegundos como número (sí se puede leer así)
        from pyspark.sql.pandas.types import from_arrow_type
        from pyspark.sql import types as T
        import pyarrow as pa
        campos = []
        for f in pq.read_schema(muestra):
            t = f.type.value_type if pa.types.is_dictionary(f.type) else f.type
            if pa.types.is_null(t):
                continue
            try:
                campos.append(T.StructField(f.name, T.LongType() if f.name in nanos else from_arrow_type(t), True))
            except Exception:
                pass
        df = spark.read.schema(T.StructType(campos)).parquet(ruta)
    else:
        df = spark.read.parquet(ruta)
    leidos = dict(df.dtypes)
    for c, con_zona in nanos.items():
        if leidos.get(c) != "bigint":        # INT96 u otro: Spark ya lo leyó como timestamp
            continue
        ts = F.expr(f"timestamp_micros(cast(`{c}` / 1000 as bigint))")
        df = df.withColumn(c, ts if con_zona else F.to_utc_timestamp(ts, spark.conf.get("spark.sql.session.timeZone")))
    return df


SENSIBLE = re.compile(r"lat|lon|geo|direcc|address|calle|tel|phone|correo|email|rfc|curp|cliente|customer|usuario|user_|nombre_", re.I)


def json_seguro(obj, prof=0):
    """Copia de un JSON sin llaves sensibles (ubicación, contacto, personas) y con listas recortadas, para imprimir."""
    if prof > 4:
        return "…"
    if isinstance(obj, dict):
        return {k: json_seguro(v, prof + 1) for k, v in obj.items() if not SENSIBLE.search(str(k))}
    if isinstance(obj, list):
        return [json_seguro(v, prof + 1) for v in obj[:10]] + (["…"] if len(obj) > 10 else [])
    return obj


def existe(nombre_tabla):
    try:
        return spark.catalog.tableExists(nombre_tabla)
    except Exception:
        try:
            spark.table(nombre_tabla).schema
            return True
        except Exception:
            return False


def leer_tabla(nombre_tabla, notebook_que_la_crea):
    """spark.table con un mensaje claro si la tabla todavía no existe."""
    if not existe(nombre_tabla):
        raise RuntimeError(f"No existe {nombre_tabla}. Corre antes el notebook {notebook_que_la_crea}.")
    return spark.table(nombre_tabla)


def cerrar(notebook, siguiente=None):
    """Imprime el resumen para revisión y guarda los hallazgos en mlops.hallazgos."""
    titulo(f"RESUMEN PARA REVISIÓN · {notebook} · {dt.datetime.now():%Y-%m-%d %H:%M} · "
           f"{(time.time() - _INICIO_NOTEBOOK) / 60:.1f} min")
    for i, h in enumerate(HALLAZGOS, 1):
        print(f"{i:>2}. {h}")
    if not HALLAZGOS:
        print("(sin hallazgos registrados)")
    try:
        if HALLAZGOS:
            ahora = dt.datetime.now()
            filas = [(notebook, ahora, i, h) for i, h in enumerate(HALLAZGOS, 1)]
            (spark.createDataFrame(filas, "notebook string, fecha_corrida timestamp, orden int, hallazgo string")
             .write.format("delta").mode("append").option("mergeSchema", "true")
             .saveAsTable(tabla("mlops", "hallazgos")))
            print(f"\n(hallazgos guardados en {tabla('mlops', 'hallazgos')})")
    except Exception as e:
        print(f"\n(no se pudieron guardar los hallazgos: {type(e).__name__}: {str(e)[:120]})")
    if siguiente:
        print(f"\nSiguiente notebook: {siguiente}")


def resumir_fechas(fechas, max_items=6):
    """Lista de fechas como rangos consecutivos: '2026-07-16 a 2026-07-31' en lugar de 16 fechas sueltas."""
    fechas = sorted(pd.Timestamp(f).date() for f in fechas)
    rangos, ini, ant = [], None, None
    for f in fechas:
        if ini is None:
            ini = ant = f
        elif (f - ant).days == 1:
            ant = f
        else:
            rangos.append((ini, ant)); ini = ant = f
    if ini is not None:
        rangos.append((ini, ant))
    txt = [f"{a}" if a == b else f"{a} a {b}" for a, b in rangos]
    return ", ".join(txt[:max_items]) + (f" y {len(txt) - max_items} rangos más" if len(txt) > max_items else "")

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
DETENER_SI_FALLA = False     # True en el pipeline automático: el notebook termina con error si una regla crítica falla
BANDA_CEROS = (0.30, 0.75)   # proporción de ceros comerciales esperada por mes en el panel
MIN_FILAS_DIA = 0.50         # un día con menos de esta fracción de las líneas de la mediana de su mes es sospechoso
VENTA_MAX_LINEA, CANTIDAD_MAX_LINEA, COSTO_MAX_LINEA = 15_000, 1_000, 5_000   # rango POS de Bnext
RESULTADOS = []


def regla(capa, nombre, valor, cumple, critica=True, detalle=""):
    RESULTADOS.append({"capa": capa, "regla": nombre, "valor": valor, "estado": "PASA" if cumple else "FALLA",
                       "critica": critica, "detalle": detalle})
    print(f"{'✓' if cumple else '✗'} [{capa}] {nombre}: {valor}" + (f" — {detalle}" if detalle else ""))

# COMMAND ----------

titulo("1. BRONCE")
B = leer_tabla(tabla("bronze", "ventas_lineas"), "02_carga_bronce")
meses_b = sorted(r[0] for r in B.select("anio_mes").distinct().collect())
regla("bronze", "están todos los meses", ", ".join(meses_b), set(MESES) <= set(meses_b))
try:
    cargas = spark.table(tabla("mlops", "bronce_cargas")).toPandas().sort_values("fecha_carga").groupby("anio_mes").tail(1)
    malos = cargas[cargas["estado"] != "ok"]
    regla("bronze", "conciliación de filas contra metadatos", f"{len(cargas) - len(malos)}/{len(cargas)} meses ok", len(malos) == 0)
except Exception as e:
    regla("bronze", "conciliación de filas contra metadatos", f"sin bitácora ({type(e).__name__})", False)

# COMMAND ----------

titulo("2. PLATA: LÍNEAS")
S = leer_tabla(tabla("silver", "ventas_lineas"), "08_limpieza_plata")
r = S.agg(F.count("*").alias("filas"), F.countDistinct("id_linea").alias("ids"),
          F.sum(F.col("venta").isNull().cast("int")).alias("venta_nula"),
          F.sum(((F.col("venta") < 0) | (F.col("venta") > VENTA_MAX_LINEA)).cast("int")).alias("venta_fuera"),
          F.sum(((F.col("cantidad") > CANTIDAD_MAX_LINEA) | (F.col("costo_unitario") > COSTO_MAX_LINEA)).cast("int")).alias("pos_fuera"),
          F.sum((F.col("venta_neta") > F.col("venta") + 0.01).cast("int")).alias("neta_mayor"),
          F.sum(F.col("fecha_venta").isNull().cast("int")).alias("fecha_nula"),
          F.sum(((F.col("tasa_descuento") < 0) | (F.col("tasa_descuento") > 1)).cast("int")).alias("tasa_fuera"),
          F.sum(F.col("tienda").isNull().cast("int")).alias("tienda_nula"),
          F.sum(F.col("categoria_id").isNull().cast("int")).alias("categoria_nula"),
          F.min("fecha_venta").alias("desde"), F.max("fecha_venta").alias("hasta")).collect()[0]
regla("silver", "order_line_id único", f"{r['filas']:,} filas · {r['ids']:,} ids", r["filas"] == r["ids"])
regla("silver", "venta no nula", f"{r['venta_nula']:,}", r["venta_nula"] == 0)
regla("silver", f"venta de línea entre 0 y {VENTA_MAX_LINEA:,}", f"{r['venta_fuera']:,} fuera", r["venta_fuera"] == 0)
regla("silver", f"cantidad ≤ {CANTIDAD_MAX_LINEA:,} y costo ≤ {COSTO_MAX_LINEA:,} (rango POS)", f"{r['pos_fuera']:,} fuera", r["pos_fuera"] == 0)
regla("silver", "venta neta de impuestos ≤ venta", f"{r['neta_mayor']:,} líneas con impuestos negativos", r["neta_mayor"] == 0,
      critica=False)
regla("silver", "fecha de venta no nula", f"{r['fecha_nula']:,}", r["fecha_nula"] == 0)
regla("silver", "tasa de descuento en [0, 1]", f"{r['tasa_fuera']:,} fuera", r["tasa_fuera"] == 0)
regla("silver", "tienda y categoría no nulas", f"{r['tienda_nula']:,} / {r['categoria_nula']:,}", r["tienda_nula"] + r["categoria_nula"] == 0)
inicio, fin = pd.Timestamp(min(MESES) + "-01").date(), (pd.Timestamp(max(MESES) + "-01") + pd.offsets.MonthEnd(0)).date()
regla("silver", "fechas dentro del periodo", f"{r['desde']} → {r['hasta']}", r["desde"] >= inicio and r["hasta"] <= fin, critica=False)
dia = S.groupBy("fecha_venta").count().withColumn("mes", F.date_format("fecha_venta", "yyyy-MM")).toPandas()
dia["mediana_mes"] = dia.groupby("mes")["count"].transform("median")
bajos = dia[dia["count"] < MIN_FILAS_DIA * dia["mediana_mes"]]
regla("silver", f"ningún día con menos del {MIN_FILAS_DIA:.0%} de las líneas de la mediana de su mes", f"{len(bajos)} días",
      len(bajos) == 0, critica=False, detalle=", ".join(str(d) for d in bajos["fecha_venta"].head(10)))
try:
    conc = spark.table(tabla("mlops", "conciliacion_plata")).toPandas()
    ok = (conc.sort_values("fecha_limpieza", ascending=False).drop_duplicates("anio_mes")["cuadra"] == "sí").all()
    regla("silver", "Plata = Bronce − filas quitadas (cada mes)", "sí" if ok else "no", ok)
except Exception as e:
    regla("silver", "Plata = Bronce − filas quitadas (cada mes)", f"sin conciliación ({type(e).__name__})", False)
CATALOGO_CAT = spark.table(tabla("silver", "catalogo_categorias"))
cat = CATALOGO_CAT.select("categoria_id")
sin_cat = S.select("categoria_id").distinct().join(cat, "categoria_id", "left_anti").count()
regla("silver", "toda categoría está en el catálogo", f"{sin_cat} sin catálogo", sin_cat == 0)
cp = CATALOGO_CAT.select("categoria_id", "nombre", *(["tipo_categoria"] if "tipo_categoria" in CATALOGO_CAT.columns else [])).toPandas()
cp["clave"] = cp["nombre"].fillna("").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True).str.rstrip("SH")
dup = cp[cp["clave"] != ""].groupby("clave").filter(lambda g: g["categoria_id"].nunique() > 1)
regla("silver", "sin categorías con el mismo nombre en dos identificadores", f"{dup['clave'].nunique()} grupos",
      dup.empty, critica=False, detalle=", ".join(f"{c}: {sorted(g['categoria_id'].tolist())}" for c, g in dup.groupby("clave"))[:300])
sin_nombre = int(cp["nombre"].isna().sum() + (cp["nombre"] == "").sum())
regla("silver", "toda categoría tiene nombre", f"{sin_nombre} sin nombre", sin_nombre == 0, critica=False)
if "tipo_categoria" in cp:
    tipos = cp["tipo_categoria"].value_counts().to_dict()
    regla("silver", "toda categoría tiene tipo (mercancía, servicio o genérico)", str(tipos), cp["tipo_categoria"].notna().all(),
          critica=False)
tiendas_cat = spark.table(tabla("silver", "catalogo_tiendas")).select("tienda")
sin_tienda = S.select("tienda").distinct().join(tiendas_cat, "tienda", "left_anti").count()
regla("silver", "toda tienda está en el catálogo de tiendas", f"{sin_tienda} sin catálogo", sin_tienda == 0)

# COMMAND ----------

titulo("3. PLATA: PANEL")
P = leer_tabla(tabla("silver", "panel_categoria_dia"), "09_panel_diario")
r = P.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id", "fecha").alias("llaves"),
          F.sum((F.col("venta") < 0).cast("int")).alias("negativa"),
          F.sum((F.col("venta").isNull() & (F.col("tienda_inactiva") == 0) & (F.col("sin_datos") == 0)).cast("int")).alias("nula_activa"),
          F.countDistinct(F.when(F.col("sin_datos") == 1, F.col("fecha"))).alias("dias_sin_datos"),
          F.sum((F.col("venta").isNotNull() & (F.col("tienda_inactiva") == 1)).cast("int")).alias("valor_inactiva"),
          F.sum((F.col("promos_vigentes") < 0).cast("int")).alias("promos_neg"),
          F.sum(F.col("tipo_categoria").isNull().cast("int")).alias("sin_tipo"),
          F.countDistinct(F.when(F.col("dia_atipico") == 1, F.col("fecha"))).alias("dias_atipicos")).collect()[0]
regla("panel", "llave tienda-categoría-fecha única", f"{r['filas']:,} filas · {r['llaves']:,} llaves", r["filas"] == r["llaves"])
regla("panel", "venta ≥ 0", f"{r['negativa']:,} negativas", r["negativa"] == 0)
regla("panel", "venta nula sólo si la tienda está inactiva o el día no tiene datos", f"{r['nula_activa']:,} / {r['valor_inactiva']:,}",
      r["nula_activa"] == 0 and r["valor_inactiva"] == 0)
dias_sd = [x[0] for x in P.where(F.col("sin_datos") == 1).select("fecha").distinct().collect()]
regla("panel", "todos los días del periodo tienen datos", f"{r['dias_sin_datos']} días sin datos", r["dias_sin_datos"] == 0,
      critica=False, detalle=(f"{resumir_fechas(dias_sd)}: pedir a Bnext la entrega faltante" if dias_sd else ""))
dias_at = [x[0] for x in P.where(F.col("dia_atipico") == 1).select("fecha").distinct().collect()]
regla("panel", "días atípicos (con datos pero menos tiendas) marcados", f"{r['dias_atipicos']} días", True, critica=False,
      detalle=resumir_fechas(dias_at) if dias_at else "")
regla("panel", "promociones vigentes ≥ 0", f"{r['promos_neg']:,} negativas", r["promos_neg"] == 0)
regla("panel", "toda fila tiene tipo de categoría", f"{r['sin_tipo']:,} sin tipo", r["sin_tipo"] == 0, critica=False)
T = (S.where(F.col("fecha_venta").between(inicio, fin)).groupBy("tienda")
     .agg(F.min("fecha_venta").alias("primera_venta"), F.max("fecha_venta").alias("ultima_venta")))
fuera = P.join(T, "tienda").where((F.col("fecha") < F.col("primera_venta")) | (F.col("fecha") > F.col("ultima_venta"))).count()
regla("panel", "ninguna fila fuera de la ventana de su tienda", f"{fuera:,}", fuera == 0)
ceros = (P.where("tienda_inactiva = 0 AND sin_datos = 0 AND en_warmup = 0 AND serie_relevante = 1").groupBy("mes")
         .agg(F.avg((F.col("venta") == 0).cast("double")).alias("p")).orderBy("mes").toPandas())
fuera_banda = ceros[(ceros["p"] < BANDA_CEROS[0]) | (ceros["p"] > BANDA_CEROS[1])]
regla("panel", f"ceros comerciales por mes entre {BANDA_CEROS[0]:.0%} y {BANDA_CEROS[1]:.0%}",
      ", ".join(f"{m} {p:.1%}" for m, p in zip(ceros["mes"], ceros["p"])), len(fuera_banda) == 0, critica=False)
v_s = S.where(F.col("fecha_venta").between(inicio, fin)).agg(F.sum("venta")).collect()[0][0] or 0.0
v_p = P.agg(F.sum("venta")).collect()[0][0] or 0.0
regla("panel", "venta del panel = venta de Plata", f"${v_p:,.2f} vs ${v_s:,.2f}", abs(v_p - v_s) <= max(1.0, 1e-6 * v_s))

# COMMAND ----------

titulo("4. LÍNEA BASE")
try:
    LB = spark.table(tabla("mlops", "linea_base_ingenua")).toPandas()
    LB = LB[LB["fecha_calculo"] == LB["fecha_calculo"].max()].sort_values("corte")
    sd = set(dias_sd)
    toca = []
    for c in LB.itertuples():
        origen = pd.Timestamp(c.origen).date()
        necesarios = {origen - dt.timedelta(days=k) for k in range(14)} | \
                     {d.date() for d in pd.date_range(c.desde, c.hasta, freq="D")}
        if necesarios & sd:
            toca.append(c.corte)
    n_dev = int(LB["corte"].astype(str).str.startswith("corte_").sum())
    n_prueba = int(LB["corte"].astype(str).str.startswith("prueba_").sum())
    regla("línea base", "cuatro cortes walk-forward de desarrollo", f"{n_dev} cortes de desarrollo · {n_prueba} de prueba",
          n_dev == len(ORIGENES_CORTES), critica=False,
          detalle=", ".join(f"{c.corte}: {c.desde} a {c.hasta}" for c in LB.itertuples()))
    regla("línea base", "ningún corte ni sus 14 días previos tocan días sin datos", f"{len(toca)} cortes", not toca,
          detalle=", ".join(toca))
    regla("línea base", "WMAPE de la línea base calculado", f"estacional {LB['wmape_lag7'].mean():.4f} · media móvil {LB['wmape_ma7'].mean():.4f}",
          LB[["wmape_lag7", "wmape_ma7"]].notna().all().all())
except Exception as e:
    regla("línea base", "tabla de la línea base", f"no disponible ({type(e).__name__}); corre el 10", False)

# COMMAND ----------

titulo("5. RESULTADO")
RES = pd.DataFrame(RESULTADOS)
mostrar(RES, "Reglas", max_filas=100)
guardar(RES.assign(fecha_validacion=pd.Timestamp.now()), "mlops", "validaciones", "Resultado de las reglas de calidad por corrida", modo="append")
criticas = RES[(RES["estado"] == "FALLA") & RES["critica"]]
avisos_ = RES[(RES["estado"] == "FALLA") & ~RES["critica"]]
if len(criticas):
    aviso(f"{len(criticas)} reglas críticas fallan: {', '.join(criticas['regla'])}.")
else:
    hallazgo(f"Las {int(RES['critica'].sum())} reglas críticas pasan; {len(avisos_)} reglas de aviso por revisar.")
for a in avisos_.itertuples():
    aviso(f"{a.regla}: {a.valor}")
if DETENER_SI_FALLA and len(criticas):
    raise AssertionError("Validación de calidad fallida: " + "; ".join(criticas["regla"]))

# COMMAND ----------

cerrar("11_validacion_calidad", siguiente="ESTADO_DEL_PROYECTO (en la carpeta principal)")
