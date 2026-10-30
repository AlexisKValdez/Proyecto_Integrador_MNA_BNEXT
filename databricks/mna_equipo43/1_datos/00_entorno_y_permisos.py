# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Entorno y permisos
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Tiendas SIX / Bnext**
# MAGIC
# MAGIC **Para qué sirve.** Antes de leer un solo dato, confirma que el clúster y los permisos alcanzan para todo el proyecto:
# MAGIC 1. Qué cómputo se está usando (serverless o clúster), su memoria y qué librerías trae (LightGBM, MLflow, etc.).
# MAGIC 2. Que existen el catálogo `modeladoprescriptivoventas`, el esquema `maestria` y los seis volúmenes mensuales.
# MAGIC 3. Si puedes **crear tablas** donde se guardará el trabajo del equipo (por defecto en `maestria`, con prefijo de capa).
# MAGIC 4. Si están disponibles MLflow y el registro de modelos en Unity Catalog (los usaremos en la etapa de modelos).
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | Sólo metadatos del catálogo |
# MAGIC | **Crea** | Nada (sólo una tabla de prueba que borra enseguida) |
# MAGIC | **Tiempo** | 1–2 min |
# MAGIC | **Cómo correr** | Conectar el clúster (arriba a la derecha) → **Run all** |
# MAGIC
# MAGIC Al final imprime un **RESUMEN PARA REVISIÓN**: cópialo y pégalo en el chat, o exporta el notebook a HTML
# MAGIC (File → Export → HTML) y guárdalo en tu carpeta `Databricks/salidas/`.

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

# MAGIC %md
# MAGIC ## 1. Clúster y librerías
# MAGIC **Qué mirar.** Runtime **ML** (trae LightGBM, MLflow, scikit-learn). Memoria del nodo: para entrenar con los seis meses
# MAGIC conviene 64 GB o más. Si falta una librería marcada como necesaria, se instala con `%pip install` en el notebook que la use.

# COMMAND ----------

import importlib
import platform

titulo("1. CLÚSTER Y LIBRERÍAS")
def conf(clave, defecto="n/d"):
    try:
        return spark.conf.get(clave)
    except Exception:
        return defecto

USUARIO = spark.sql("SELECT current_user() AS u").collect()[0]["u"]
RUNTIME = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
NODO = conf("spark.databricks.clusterUsageTags.clusterNodeType", "")
SERVERLESS = not NODO                     # en serverless no hay datos del clúster
print("Usuario           :", USUARIO)
print("Spark             :", spark.version)
print("Cómputo           :", "serverless" if SERVERLESS else f"clúster {conf('spark.databricks.clusterUsageTags.clusterName')} · nodo {NODO}")
print("Runtime           :", RUNTIME or "n/d")
print("Python            :", platform.python_version())
try:
    memoria_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    print(f"Memoria del driver: {memoria_gb:.0f} GB · núcleos: {os.cpu_count()}")
except Exception:
    memoria_gb = None

LIBRERIAS = [("pandas", True), ("numpy", True), ("pyarrow", True), ("scipy", True), ("sklearn", True),
             ("matplotlib", True), ("lightgbm", True), ("mlflow", True), ("statsmodels", False),
             ("xgboost", False), ("catboost", False), ("optuna", False), ("holidays", False),
             ("shap", False), ("polars", False)]
faltan = []
for nombre, necesaria in LIBRERIAS:
    try:
        m = importlib.import_module(nombre)
        print(f"  [ok] {nombre:12s} {getattr(m, '__version__', '')}")
    except Exception:
        print(f"  [--] {nombre:12s} no instalada" + ("  ← necesaria" if necesaria else ""))
        if necesaria:
            faltan.append(nombre)
if faltan:
    aviso(f"Faltan librerías para la fase de modelos: {faltan}. En serverless se instalan con %pip al inicio del notebook "
          "que las use (o en el panel Environment); en un clúster, con un runtime ML.")
if SERVERLESS:
    hallazgo(f"Cómputo serverless · Spark {spark.version} · Python {platform.python_version()} · "
             f"{memoria_gb:.0f} GB en el driver. Sirve para toda la fase de datos (el trabajo pesado lo hace Spark)." if memoria_gb
             else f"Cómputo serverless · Spark {spark.version}.")
else:
    if "ml" not in RUNTIME.lower():
        aviso(f"El runtime ({RUNTIME or 'desconocido'}) no parece ser ML; para la fase de modelos conviene 15.4 LTS ML o superior.")
    hallazgo(f"Clúster con nodo {NODO} · {memoria_gb:.0f} GB de memoria." if memoria_gb else f"Clúster con nodo {NODO}.")
if memoria_gb and memoria_gb < 60:
    print(f"Nota: {memoria_gb:.0f} GB alcanzan para la fase de datos; para entrenar con seis meses se usará una muestra de "
          "series o un clúster más grande (se decide en la fase 2).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Catálogo, esquemas y volúmenes
# MAGIC **Qué mirar.** Que aparezcan los seis volúmenes mensuales (`2026-03` … `2026-08`) más `processed` y `outputs`.
# MAGIC Los esquemas que no son `maestria` pueden ser de tus compañeros: el notebook `01_inventario_catalogo` los revisa a fondo.

# COMMAND ----------

titulo("2. CATÁLOGO, ESQUEMAS Y VOLÚMENES")
try:
    esquemas = sorted(r[0] for r in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect())
except Exception as e:
    esquemas = []
    aviso(f"No se pudo listar el catálogo {CATALOGO}: {type(e).__name__}. ¿El clúster tiene Unity Catalog?")
print(f"Esquemas en {CATALOGO} ({len(esquemas)}):", esquemas)

def listar_volumenes(esquema):
    try:
        return sorted(r[1] for r in spark.sql(f"SHOW VOLUMES IN {CATALOGO}.{esquema}").collect())
    except Exception:
        try:
            return sorted(f.name.rstrip("/") for f in dbutils.fs.ls(f"{RAIZ_VOLUMENES}/{CATALOGO}/{esquema}") if f.name.endswith("/"))
        except Exception:
            return []

VOLS = listar_volumenes(ESQUEMA_DATOS)
print(f"Volúmenes en {ESQUEMA_DATOS} ({len(VOLS)}):", VOLS)
faltantes = [m for m in MESES if m not in VOLS]
extra_meses = sorted(v for v in VOLS if re.fullmatch(r"\d{4}-\d{2}", v) and v not in MESES)
if faltantes:
    aviso(f"No encuentro los volúmenes de {faltantes}. Revisa MESES en la configuración.")
else:
    hallazgo(f"Los {len(MESES)} volúmenes mensuales existen ({MESES[0]} a {MESES[-1]}).")
if extra_meses:
    aviso(f"Hay volúmenes mensuales que no están en MESES: {extra_meses}. Si son datos nuevos, agrégalos a MESES en todos los notebooks.")
for v in ("processed", "outputs"):
    print(f"  {v:10s}: {'existe' if v in VOLS else 'NO existe'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Permisos
# MAGIC **Qué hace.** Prueba crear (y borra enseguida) una tabla donde se guardará el trabajo del equipo. Con la configuración
# MAGIC por defecto (`ESQUEMAS_PROPIOS = False`) es `maestria`, con prefijo de capa: `maestria.bronze_ventas_lineas`,
# MAGIC `maestria.silver_ventas_lineas`, `maestria.mlops_hallazgos`… Si algún día Bnext da permiso de crear esquemas, se cambia
# MAGIC a `True` en todos los notebooks y se usan esquemas `bronze`, `silver`, `gold` y `mlops`.
# MAGIC
# MAGIC **Qué mirar.** Que la fila de `maestria` diga `ok` en `crear_tabla`.

# COMMAND ----------

titulo("3. PERMISOS")
COMENTARIOS = {"bronze": "Equipo 43 · datos crudos de Bnext tal como llegan (una tabla por fuente)",
               "silver": "Equipo 43 · datos limpios, deduplicados y homologados; panel diario",
               "gold": "Equipo 43 · variables para modelos, predicciones y escenarios",
               "mlops": "Equipo 43 · bitácoras, conciliaciones, hallazgos de EDA, validaciones y monitoreo"}
resultado_permisos = []

def probar_escritura(nombre_tabla):
    try:
        spark.sql(f"CREATE TABLE IF NOT EXISTS {nombre_tabla} (x INT) USING DELTA")
        spark.sql(f"DROP TABLE IF EXISTS {nombre_tabla}")
        return "ok"
    except Exception as e:
        return f"falla: {type(e).__name__}: {str(e).splitlines()[0][:120]}"

if ESQUEMAS_PROPIOS:
    for esq, comentario in COMENTARIOS.items():
        try:
            spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOGO}.{esq} COMMENT '{comentario}'")
            creado = "ok"
        except Exception as e:
            creado = f"falla: {type(e).__name__}: {str(e).splitlines()[0][:120]}"
        escritura = probar_escritura(f"{CATALOGO}.{esq}._prueba_equipo43") if creado == "ok" else "—"
        resultado_permisos.append({"esquema": f"{CATALOGO}.{esq}", "crear_esquema": creado, "crear_tabla": escritura})
resultado_permisos.append({"esquema": f"{CATALOGO}.{ESQUEMA_DATOS}", "crear_esquema": "(ya existe)",
                           "crear_tabla": probar_escritura(f"{CATALOGO}.{ESQUEMA_DATOS}._prueba_equipo43")})
PERMISOS = pd.DataFrame(resultado_permisos)
mostrar(PERMISOS, "Resultado")

propios_ok = ESQUEMAS_PROPIOS and all(r["crear_tabla"] == "ok" for r in resultado_permisos[:-1])
maestria_ok = resultado_permisos[-1]["crear_tabla"] == "ok"
if propios_ok:
    hallazgo("Permisos completos: esquemas bronze, silver, gold y mlops listos (ESQUEMAS_PROPIOS = True).")
elif maestria_ok and not ESQUEMAS_PROPIOS:
    hallazgo(f"Las tablas del equipo se guardan en {CATALOGO}.{ESQUEMA_DATOS} con prefijo de capa "
             f"(p. ej. {tabla('bronze', 'ventas_lineas')}); hay permiso de escritura.")
elif maestria_ok:
    aviso("No se pudieron crear (o escribir en) los esquemas propios, pero sí en maestria: "
          "pon ESQUEMAS_PROPIOS = False en la configuración de todos los notebooks.")
else:
    aviso("No hay permiso para crear tablas ni en esquemas propios ni en maestria. Pide al dueño del catálogo "
          "(jreyes@bnext.mx) USE CATALOG, CREATE SCHEMA o, al menos, CREATE TABLE en maestria.")

try:
    privs = spark.sql(f"SHOW GRANTS ON SCHEMA {CATALOGO}.{ESQUEMA_DATOS}").toPandas()
    if len(privs):
        cols = [c for c in privs.columns if c.lower() in ("principal", "actiontype", "action_type", "objecttype")]
        mostrar(privs[cols] if cols else privs, f"Permisos explícitos en {CATALOGO}.{ESQUEMA_DATOS}", max_filas=30)
except Exception as e:
    print(f"(no se pudieron leer los permisos explícitos: {type(e).__name__})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. MLflow y registro de modelos
# MAGIC **Qué mirar.** Que MLflow responda y que el registro sea de Unity Catalog (`databricks-uc`): ahí registraremos el modelo
# MAGIC campeón con alias `champion`. No se crea nada en esta celda.

# COMMAND ----------

titulo("4. MLFLOW")
try:
    import mlflow
    print("MLflow             :", mlflow.__version__)
    print("Tracking URI       :", mlflow.get_tracking_uri())
    try:
        mlflow.set_registry_uri("databricks-uc")
        print("Registro de modelos: databricks-uc (Unity Catalog)")
        hallazgo("MLflow disponible con registro de modelos en Unity Catalog.")
    except Exception as e:
        aviso(f"No se pudo usar el registro de Unity Catalog: {type(e).__name__}")
    try:
        exps = mlflow.search_experiments(max_results=200)
        print(f"Experimentos visibles: {len(exps)}")
    except Exception as e:
        print(f"(no se pudieron listar experimentos: {type(e).__name__})")
except ImportError:
    aviso("MLflow no está instalado.")

# COMMAND ----------

cerrar("00_entorno_y_permisos", siguiente="01_inventario_catalogo")
