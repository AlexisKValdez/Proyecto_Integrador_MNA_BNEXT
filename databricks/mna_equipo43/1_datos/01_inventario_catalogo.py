# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Inventario de todo lo que hay en Databricks
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Antes de construir nada, saber con exactitud qué existe hoy y **quién lo hizo**: el equipo anterior,
# MAGIC tus compañeros o Bnext. Responde:
# MAGIC 1. ¿Qué tablas hay en el catálogo, quién las creó, cuándo se modificaron por última vez y cuántas filas tienen?
# MAGIC 2. ¿Qué archivos hay en cada volumen (meses de ventas, `processed`, `outputs`) y de cuándo son?
# MAGIC 3. ¿Cuántos archivos, días, filas y columnas trae cada mes?
# MAGIC 4. ¿Qué notebooks, experimentos de MLflow, modelos registrados y jobs existen en el workspace?
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | Metadatos del catálogo `modeladoprescriptivoventas`, los volúmenes de `maestria`, el workspace y MLflow |
# MAGIC | **Escribe** | `mlops.inventario_tablas`, `mlops.inventario_archivos`, `mlops.inventario_workspace` |
# MAGIC | **Tiempo** | 5–20 min (depende del número de archivos) |
# MAGIC | **No hace** | No modifica ni borra nada; no imprime registros de ventas |

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
CONTAR_FILAS = True            # cuenta las filas de cada tabla (rápido en Delta)
MAX_TABLAS = 300               # tope de tablas a detallar
MAX_ARCHIVOS_VOLUMEN = 50_000  # tope de archivos a listar por volumen
HILOS = 16                     # lecturas de metadatos de parquet en paralelo
YO = spark.sql("SELECT current_user() AS u").collect()[0]["u"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Tablas del catálogo y quién las creó
# MAGIC **Qué hace.** Lee `information_schema.tables` (dueño, quién la creó y cuándo, quién la modificó) y, para cada tabla,
# MAGIC su tamaño, filas y la última operación de su historial Delta.
# MAGIC
# MAGIC **Qué mirar.** Las tablas creadas por tus compañeros o por el equipo anterior: son las que el notebook
# MAGIC `03_auditoria_trabajo_previo` revisa. Las de `mlops`, `bronze`, `silver` que ya existan antes de correr el `02` no son nuestras.

# COMMAND ----------

titulo("1. TABLAS DEL CATÁLOGO")
try:
    TABLAS = spark.sql(f"""
        SELECT table_schema AS esquema, table_name AS tabla, table_type AS tipo, table_owner AS duenio,
               created_by AS creada_por, created AS creada, last_altered_by AS modificada_por,
               last_altered AS modificada, comment AS comentario
        FROM {CATALOGO}.information_schema.tables
        WHERE table_schema <> 'information_schema'
        ORDER BY esquema, tabla""").toPandas()
except Exception as e:
    print(f"(information_schema no disponible: {type(e).__name__}; se usa SHOW TABLES)")
    filas = []
    for esq in [r[0] for r in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect()]:
        if esq == "information_schema":
            continue
        try:
            for r in spark.sql(f"SHOW TABLES IN {CATALOGO}.{esq}").collect():
                if not r["isTemporary"]:
                    filas.append({"esquema": esq, "tabla": r["tableName"], "tipo": None, "duenio": None,
                                  "creada_por": None, "creada": None, "modificada_por": None,
                                  "modificada": None, "comentario": None})
        except Exception:
            pass
    TABLAS = pd.DataFrame(filas, columns=["esquema", "tabla", "tipo", "duenio", "creada_por", "creada",
                                          "modificada_por", "modificada", "comentario"])

print(f"Tablas y vistas encontradas: {len(TABLAS):,}")
detalle = []
for _, r in TABLAS.head(MAX_TABLAS).iterrows():
    nombre = f"{CATALOGO}.{r['esquema']}.{r['tabla']}"
    d = {"esquema": r["esquema"], "tabla": r["tabla"], "filas": None, "columnas": None, "mb": None,
         "formato": None, "particion": None, "ultima_operacion": None, "ultimo_usuario": None, "ultima_fecha": None}
    try:
        df_t = spark.table(nombre)
        d["columnas"] = len(df_t.columns)
        if CONTAR_FILAS:
            d["filas"] = df_t.count()
    except Exception as e:
        d["ultima_operacion"] = f"ilegible: {type(e).__name__}"
    try:
        det = spark.sql(f"DESCRIBE DETAIL {nombre}").collect()[0]
        d["formato"], d["mb"] = det["format"], (det["sizeInBytes"] or 0) / 1024**2
        d["particion"] = ",".join(det["partitionColumns"] or [])
    except Exception:
        pass
    try:
        h = spark.sql(f"DESCRIBE HISTORY {nombre} LIMIT 1").collect()[0]
        d["ultima_operacion"], d["ultimo_usuario"], d["ultima_fecha"] = h["operation"], h["userName"], h["timestamp"]
    except Exception:
        pass
    detalle.append(d)
COLS_DETALLE = ["filas", "columnas", "mb", "formato", "particion", "ultima_operacion", "ultimo_usuario", "ultima_fecha"]
INV_TABLAS = (TABLAS.merge(pd.DataFrame(detalle), on=["esquema", "tabla"], how="left") if detalle
              else TABLAS.assign(**{c: None for c in COLS_DETALLE}))
if len(INV_TABLAS):
    INV_TABLAS["creada_por"] = INV_TABLAS["creada_por"].where(INV_TABLAS["creada_por"].notna(), INV_TABLAS["ultimo_usuario"])

mostrar(INV_TABLAS[["esquema", "tabla", "tipo", "creada_por", "creada", "modificada", "filas", "columnas", "mb",
                    "ultima_operacion", "ultimo_usuario"]],
        "Tablas por esquema", formatos={"filas": "int", "columnas": "int", "mb": "dec1"}, max_filas=MAX_TABLAS)
if len(INV_TABLAS):
    por_autor = (INV_TABLAS.assign(autor=INV_TABLAS["creada_por"].fillna("(desconocido)"))
                 .groupby(["autor", "esquema"]).agg(tablas=("tabla", "count"), filas=("filas", "sum")).reset_index())
    mostrar(por_autor, "¿Quién creó qué?", formatos={"tablas": "int", "filas": "int"})
    otros = INV_TABLAS[INV_TABLAS["creada_por"].notna() & (INV_TABLAS["creada_por"] != YO)]
    hallazgo(f"{len(INV_TABLAS):,} tablas en {INV_TABLAS['esquema'].nunique()} esquemas; "
             f"{len(otros):,} creadas por otras personas ({', '.join(sorted(otros['creada_por'].unique())[:6]) or 'ninguna'}).")
else:
    hallazgo("El catálogo no tiene tablas todavía: todo está en volúmenes (archivos).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Volúmenes: qué archivos hay y de cuándo son
# MAGIC **Qué hace.** Recorre cada volumen de cada esquema y resume archivos, tamaño, extensiones y fechas de modificación.
# MAGIC Para `processed` y `outputs` (lo que dejó el equipo anterior y lo que han escrito tus compañeros) lista cada archivo.
# MAGIC
# MAGIC **Qué mirar.** Archivos de `processed`/`outputs` modificados después del 1 de octubre: son trabajo de tu equipo
# MAGIC (por ejemplo `outputs/avance3_equipo43/`). Los de junio o antes vienen del equipo anterior.

# COMMAND ----------

titulo("2. VOLÚMENES")

def listar(ruta, prof=0, max_prof=5, acumulado=None):
    """Lista archivos bajo una ruta de volumen (recursivo) con tamaño y fecha de modificación."""
    acumulado = [] if acumulado is None else acumulado
    try:
        entradas = dbutils.fs.ls(ruta)
    except Exception as e:
        print(f"  (no se pudo listar {ruta}: {type(e).__name__})")
        return acumulado
    for f in entradas:
        if len(acumulado) >= MAX_ARCHIVOS_VOLUMEN:
            break
        if f.name.endswith("/"):
            if prof < max_prof:
                listar(f.path, prof + 1, max_prof, acumulado)
        else:
            mod = getattr(f, "modificationTime", None)
            acumulado.append({"ruta": f.path.replace("dbfs:", ""), "archivo": f.name, "bytes": int(f.size),
                              "modificado": pd.to_datetime(mod, unit="ms") if mod else pd.NaT})
    return acumulado

try:
    ESQUEMAS = sorted(r[0] for r in spark.sql(f"SHOW SCHEMAS IN {CATALOGO}").collect() if r[0] != "information_schema")
except Exception:
    ESQUEMAS = [ESQUEMA_DATOS]
volumenes = []
for esq in ESQUEMAS:
    try:
        vs = [r[1] for r in spark.sql(f"SHOW VOLUMES IN {CATALOGO}.{esq}").collect()]
    except Exception:
        vs = []
        if esq == ESQUEMA_DATOS:
            try:
                vs = [f.name.rstrip("/") for f in dbutils.fs.ls(VOLUMENES) if f.name.endswith("/")]
            except Exception:
                pass
    volumenes += [(esq, v) for v in vs]

ARCHIVOS = []
resumen_vol = []
for esq, v in sorted(volumenes):
    t0 = time.time()
    arch = listar(f"{RAIZ_VOLUMENES}/{CATALOGO}/{esq}/{v}")
    for a in arch:
        a.update({"esquema": esq, "volumen": v})
    ARCHIVOS += arch
    ext = pd.Series([a["archivo"].rsplit(".", 1)[-1].lower() if "." in a["archivo"] else "(sin ext.)" for a in arch])
    fechas = pd.Series([a["modificado"] for a in arch], dtype="datetime64[ns]")
    resumen_vol.append({"esquema": esq, "volumen": v, "archivos": len(arch),
                        "gb": sum(a["bytes"] for a in arch) / 1024**3,
                        "extensiones": ", ".join(f"{k}×{n}" for k, n in ext.value_counts().head(5).items()),
                        "mod_min": fechas.min(), "mod_max": fechas.max(),
                        "tope": "sí" if len(arch) >= MAX_ARCHIVOS_VOLUMEN else ""})
RESUMEN_VOL = pd.DataFrame(resumen_vol)
mostrar(RESUMEN_VOL, "Volúmenes", formatos={"archivos": "int", "gb": "dec2"}, max_filas=100)
INV_ARCHIVOS = pd.DataFrame(ARCHIVOS)

for v in ("processed", "outputs"):
    sub = INV_ARCHIVOS[INV_ARCHIVOS["volumen"] == v] if len(INV_ARCHIVOS) else pd.DataFrame()
    if len(sub):
        relativa = sub["ruta"].str.slice(len(volumen(v)) + 1)
        vista = sub.assign(mb=sub["bytes"] / 1024**2,
                           carpeta=[r.rsplit("/", 1)[0] if "/" in r else "(raíz)" for r in relativa])
        mostrar(vista.sort_values("ruta")[["carpeta", "archivo", "mb", "modificado"]],
                f"Archivos en {v} ({len(sub):,})", formatos={"mb": "dec1"}, max_filas=200)
        recientes = sub[sub["modificado"] >= pd.Timestamp("2026-10-01")]
        if len(recientes):
            hallazgo(f"{v}: {len(recientes)} archivos modificados desde el 1 de octubre (trabajo reciente del equipo): "
                     + ", ".join(sorted(recientes["archivo"].head(8))))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Meses de ventas: archivos, días, filas y columnas
# MAGIC **Qué hace.** Para cada volumen mensual cuenta los parquet, los días que indica el nombre del archivo y, leyendo sólo
# MAGIC los metadatos de cada parquet (sin leer los datos), las filas y las columnas.
# MAGIC
# MAGIC **Qué mirar.** Que los seis meses tengan tamaño parecido; un mes con menos filas o días faltantes es una entrega
# MAGIC incompleta. Columnas que aparecen o desaparecen entre meses, o que cambian de tipo, se resuelven en `02_carga_bronce`.

# COMMAND ----------

titulo("3. MESES DE VENTAS")
from concurrent.futures import ThreadPoolExecutor
import pyarrow.parquet as pq

def meta(ruta):
    try:
        m = pq.read_metadata(ruta)
        return ruta, int(m.num_rows), [(c.name, str(c.type)) for c in m.schema.to_arrow_schema()], None
    except Exception as e:
        return ruta, 0, [], f"{type(e).__name__}"

def dia_del_nombre(nombre, mes):
    m = re.search(r"(20\d{2})-?(\d{2})-?(\d{2})", nombre)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    m = re.match(r"^(\d{1,2})\.parquet$", nombre, re.I)
    return f"{mes}-{int(m.group(1)):02d}" if m else None

filas_mes, columnas_mes = [], {}
for mes in MESES:
    arch = [a for a in ARCHIVOS if a["volumen"] == mes and a["archivo"].lower().endswith(".parquet")
            and not a["archivo"].startswith(("_", "."))]
    with ThreadPoolExecutor(HILOS) as ex:
        metas = list(ex.map(meta, [a["ruta"] for a in arch]))
    dias = sorted({d for d in (dia_del_nombre(a["archivo"], mes) for a in arch) if d})
    y, m = map(int, mes.split("-"))
    ndias = (dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)).day
    esperados = {f"{mes}-{d:02d}" for d in range(1, ndias + 1)}
    faltan = sorted(esperados - set(dias)) if dias else []
    cols = {}
    for _, _, campos, err in metas:
        for c, t in campos:     # large_string y string son el mismo tipo para Spark
            cols.setdefault(c, set()).add(t.replace("large_string", "string").replace("large_binary", "binary"))
    columnas_mes[mes] = cols
    filas_mes.append({"mes": mes, "archivos": len(arch), "gb": sum(a["bytes"] for a in arch) / 1024**3,
                      "filas": sum(f for _, f, _, _ in metas), "columnas": len(cols),
                      "ilegibles": sum(1 for *_, e in metas if e),
                      "dias_por_nombre": len(dias), "dias_mes": ndias,
                      "dias_faltantes": ", ".join(d[-2:] for d in faltan[:15]) + (" …" if len(faltan) > 15 else "")})
MESES_DF = pd.DataFrame(filas_mes)
mostrar(MESES_DF, "Resumen por mes", formatos={"archivos": "int", "gb": "dec2", "filas": "int", "columnas": "int"})
total = int(MESES_DF["filas"].sum())
hallazgo(f"{total:,} filas en {int(MESES_DF['archivos'].sum()):,} archivos parquet de {len(MESES)} meses "
         f"(por mes entre {MESES_DF['filas'].min():,} y {MESES_DF['filas'].max():,}).")
if MESES_DF["filas"].max() > 0 and MESES_DF["filas"].min() < 0.6 * MESES_DF["filas"].median():
    aviso("Algún mes tiene mucho menos filas que la mediana: posible entrega incompleta (ver tabla).")
for r in MESES_DF[MESES_DF["dias_por_nombre"] < MESES_DF["dias_mes"]].itertuples():
    if r.dias_por_nombre:
        aviso(f"{r.mes}: sólo hay archivos de {r.dias_por_nombre} de {r.dias_mes} días (faltan los días {r.dias_faltantes}). "
              "Pedir a Bnext la entrega completa.")
if (MESES_DF["ilegibles"] > 0).any():
    aviso(f"Hay {int(MESES_DF['ilegibles'].sum())} archivos que no se pudieron leer.")

todas = sorted(set().union(*[set(c) for c in columnas_mes.values()])) if columnas_mes else []
matriz = pd.DataFrame([{"columna": c, **{m: ("✓" if c in columnas_mes[m] else "·") for m in MESES},
                        "tipos": " | ".join(sorted(set().union(*[columnas_mes[m].get(c, set()) for m in MESES])))}
                       for c in todas])
mostrar(matriz, "Columnas por mes (✓ presente · ausente) y tipos encontrados", max_filas=200)
cambian = matriz[matriz["tipos"].str.contains(r"\|")] if len(matriz) else matriz
if len(cambian):
    hallazgo(f"{len(cambian)} columnas cambian de tipo entre archivos: {', '.join(cambian['columna'].head(10))}. "
             "02_carga_bronce las unifica.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Workspace, MLflow, modelos y jobs
# MAGIC **Qué mirar.** Notebooks en `/Shared` y en las carpetas de usuario que puedas ver (trabajo de tus compañeros),
# MAGIC experimentos de MLflow con sus corridas, modelos registrados y jobs programados.

# COMMAND ----------

titulo("4. WORKSPACE, MLFLOW, MODELOS Y JOBS")
INV_WS = []
try:
    from databricks.sdk import WorkspaceClient
    w = WorkspaceClient()

    def recorrer_ws(ruta, prof=0):
        """Lista notebooks y archivos bajo una carpeta del workspace (sin depender de list(recursive=...))."""
        if prof > 5 or len(INV_WS) >= 5000:
            return
        try:
            objetos = list(w.workspace.list(ruta))
        except Exception as e:
            if prof == 0:
                print(f"  (no se pudo listar {ruta}: {type(e).__name__})")
            return
        for o in objetos:
            tipo = str(o.object_type).split(".")[-1]
            if tipo in ("DIRECTORY", "REPO"):
                recorrer_ws(o.path, prof + 1)
            elif tipo in ("NOTEBOOK", "FILE") and "/.git/" not in o.path and "/.db_internal/" not in o.path:
                INV_WS.append({"ruta": o.path, "tipo": tipo, "lenguaje": str(o.language).split(".")[-1] if o.language else "",
                               "modificado": pd.to_datetime(o.modified_at, unit="ms") if o.modified_at else pd.NaT})

    for raiz in ["/Shared", "/Users", "/Repos"]:
        recorrer_ws(raiz)
except Exception as e:
    print(f"(databricks-sdk no disponible: {type(e).__name__}; se usa el sistema de archivos /Workspace)")
    for raiz in ["/Workspace/Shared", "/Workspace/Users"]:
        for base, dirs, archivos in os.walk(raiz):
            if base.count("/") > 6:
                dirs[:] = []
            for a in archivos:
                p = os.path.join(base, a)
                try:
                    INV_WS.append({"ruta": p, "tipo": "?", "lenguaje": "", "modificado": pd.to_datetime(os.path.getmtime(p), unit="s")})
                except Exception:
                    pass
INV_WS = pd.DataFrame(INV_WS, columns=["ruta", "tipo", "lenguaje", "modificado"])
if len(INV_WS):
    INV_WS["carpeta"] = INV_WS["ruta"].str.rsplit("/", n=1).str[0]
    mostrar(INV_WS.sort_values("ruta")[["ruta", "tipo", "modificado"]], f"Notebooks y archivos ({len(INV_WS):,})", max_filas=150)
    hallazgo(f"{(INV_WS['tipo'] == 'NOTEBOOK').sum():,} notebooks visibles en el workspace, en "
             f"{INV_WS['carpeta'].nunique():,} carpetas.")

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    try:
        exps = mlflow.search_experiments(max_results=50, order_by=["last_update_time DESC"])
    except Exception:
        exps = mlflow.search_experiments(max_results=50)
    filas = []
    for e in exps:
        try:
            runs = mlflow.search_runs([e.experiment_id], max_results=1000, output_format="pandas")
        except Exception:
            runs = pd.DataFrame()
        usuarios = sorted(runs["tags.mlflow.user"].dropna().unique()) if "tags.mlflow.user" in runs else []
        metricas = sorted(c[8:] for c in runs.columns if c.startswith("metrics."))[:6] if len(runs) else []
        filas.append({"experimento": e.name, "corridas": len(runs),
                      "creado": pd.to_datetime(e.creation_time, unit="ms") if e.creation_time else pd.NaT,
                      "ultima": pd.to_datetime(e.last_update_time, unit="ms") if e.last_update_time else pd.NaT,
                      "usuarios": ", ".join(usuarios[:3]), "metricas": ", ".join(metricas)})
    EXPS = pd.DataFrame(filas)
    mostrar(EXPS.sort_values("ultima", ascending=False) if len(EXPS) else EXPS, f"Experimentos de MLflow ({len(EXPS)})",
            formatos={"corridas": "int"}, max_filas=60)
    try:
        cli = MlflowClient(registry_uri="databricks-uc")
        modelos = list(cli.search_registered_models(filter_string=f"catalog_name = '{CATALOGO}'"))
        def _alias(m):
            a = getattr(m, "aliases", None) or []
            return ", ".join(sorted(a.keys()) if isinstance(a, dict) else [getattr(x, "alias", str(x)) for x in a])
        mostrar(pd.DataFrame([{"modelo": m.name, "alias": _alias(m),
                               "actualizado": pd.to_datetime(m.last_updated_timestamp, unit="ms") if m.last_updated_timestamp else pd.NaT}
                              for m in modelos]), f"Modelos registrados en {CATALOGO} ({len(modelos)})")
    except Exception as e:
        print(f"(no se pudieron listar modelos registrados: {type(e).__name__})")
except Exception as e:
    print(f"(MLflow no disponible: {type(e).__name__})")

try:
    from databricks.sdk import WorkspaceClient
    jobs = list(WorkspaceClient().jobs.list(limit=100))
    mostrar(pd.DataFrame([{"job": j.settings.name if j.settings else j.job_id, "creador": j.creator_user_name,
                           "creado": pd.to_datetime(j.created_time, unit="ms") if j.created_time else pd.NaT,
                           "programado": bool(j.settings and j.settings.schedule)} for j in jobs]),
            f"Jobs ({len(jobs)})")
except Exception as e:
    print(f"(no se pudieron listar jobs: {type(e).__name__})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Guardar el inventario
# MAGIC Se guarda en tablas para compararlo en el futuro (qué cambió entre una entrega y otra).

# COMMAND ----------

titulo("5. GUARDAR EL INVENTARIO")
sello = pd.Timestamp.now()
if len(INV_TABLAS):
    guardar(INV_TABLAS.assign(fecha_inventario=sello), "mlops", "inventario_tablas",
            "Inventario de tablas del catalogo: autor, fechas, filas, tamano")
if len(INV_ARCHIVOS):
    guardar(INV_ARCHIVOS.assign(fecha_inventario=sello), "mlops", "inventario_archivos",
            "Inventario de archivos en los volumenes: ruta, tamano, fecha de modificacion")
if len(INV_WS):
    guardar(INV_WS.assign(fecha_inventario=sello), "mlops", "inventario_workspace",
            "Notebooks y archivos visibles en el workspace")

# COMMAND ----------

cerrar("01_inventario_catalogo", siguiente="02_carga_bronce")
