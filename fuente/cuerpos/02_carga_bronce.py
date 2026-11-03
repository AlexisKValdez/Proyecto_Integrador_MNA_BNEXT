# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Carga de los meses a Bronce
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC **Para qué sirve.** Lee **todos** los parquet de los volúmenes mensuales y los guarda, tal como llegan, en una sola
# MAGIC tabla Delta. Bronce no limpia nada: sólo resuelve lo indispensable para que archivos distintos quepan en una tabla.
# MAGIC
# MAGIC **Por qué no basta con `spark.read.parquet(carpeta)`.** Los archivos de Bnext no son homogéneos: hay columnas cuyo
# MAGIC tipo cambia entre archivos (entero en uno y decimal o texto en otro), columnas que vienen todas nulas, nombres con
# MAGIC espacios o símbolos (`$ Pago con puntos`) y fechas en nanosegundos. Spark falla o pierde datos al mezclarlos. Aquí:
# MAGIC 1. Se leen sólo los metadatos de cada parquet (columnas, tipos, filas).
# MAGIC 2. Se arma un **plan de columnas**: un nombre válido y un tipo común por columna, igual para todos los meses.
# MAGIC 3. Cada mes se lee por grupos de archivos con tipos idénticos y se lleva al plan.
# MAGIC 4. Se agregan columnas de trazabilidad: `_archivo` (de dónde vino cada fila), `_fecha_archivo`, `_fecha_carga`, `anio_mes`.
# MAGIC 5. Se escribe la partición del mes (repetir la carga de un mes **no** duplica filas) y se **concilia**: filas escritas
# MAGIC    por archivo = filas que dicen sus metadatos.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `/Volumes/modeladoprescriptivoventas/maestria/2026-03` … `2026-08` (todos los `.parquet`) |
# MAGIC | **Escribe** | `bronze.ventas_lineas` (partición `anio_mes`) · `mlops.bronce_columnas` (plan) · `mlops.bronce_cargas` (bitácora) |
# MAGIC | **Tiempo** | 20–60 min la primera vez |

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# ----------------------------------------------------------------- parámetros de este notebook
MESES_A_CARGAR = parametro("MESES_A_CARGAR", MESES)     # o una lista, p. ej. ["2026-09"], para cargar sólo ese mes (también como parámetro de Job)
RECARGA_TOTAL = False      # True = borra y reescribe la tabla completa (si cambió el tipo de alguna columna)
HILOS = 16                 # lecturas de metadatos en paralelo
OPTIMIZAR = True           # compacta los archivos de la tabla al terminar
TABLA = tabla("bronze", "ventas_lineas")
print("Tabla destino :", TABLA)
print("Meses a cargar:", MESES_A_CARGAR)

# COMMAND ----------

# ----------------------------------------------------------------- funciones de la carga (no editar)
import unicodedata
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from functools import reduce

import pyarrow.parquet as pq

TRAZA = ["_archivo", "_fecha_archivo", "_fecha_carga", "anio_mes"]


def ruta_local(ruta):
    """dbfs:/Volumes/x o file:/Volumes/x → /Volumes/x (lo que entiende PyArrow)."""
    for prefijo in ("dbfs:", "file:"):
        if ruta.startswith(prefijo):
            return ruta[len(prefijo):]
    return ruta


def sanear(nombre):
    """'$ Pago con puntos' → 'pago_con_puntos'; 'Categoría' → 'categoria'."""
    t = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", t.strip().lower()).strip("_")) or "columna"
    return f"c_{t}" if t[0].isdigit() else t


def familia(tipo):
    """Tipo de PyArrow → familia: string, bool, int, float, decimal, date, timestamp, timestamp_ns, null, binary."""
    t = str(tipo).lower()
    if t in ("string", "large_string", "utf8", "large_utf8") or t.startswith("dictionary<values=string"):
        return "string"
    if t in ("bool", "boolean"):
        return "bool"
    if re.fullmatch(r"u?int(8|16|32|64)", t):
        return "int"
    if t in ("float", "double", "halffloat", "float16", "float32", "float64"):
        return "float"
    for pref in ("decimal", "date", "timestamp[ns", "timestamp", "null", "binary", "large_binary"):
        if t.startswith(pref):
            return {"timestamp[ns": "timestamp_ns", "large_binary": "binary"}.get(pref, pref)
    return "otro"


def tipo_exacto(tipo):
    """Tipo tal como lo lee Spark (int32 ≠ int64; timestamp con zona ≠ sin zona). Archivos con el mismo tipo exacto
    en todas las columnas se pueden leer juntos."""
    t = str(tipo).lower()
    if familia(t) == "string":
        return "string"
    if t.startswith("timestamp[ns"):
        return "timestamp_ns"
    if t.startswith("timestamp"):
        return "timestamp_tz" if "tz=" in t else "timestamp_ntz"
    return t


def tipo_comun(familias):
    """Tipo de Spark en Bronce dadas las familias de una columna en todos los archivos.
    Se ignoran los archivos donde viene toda nula; números mezclados se ensanchan (bigint o double);
    fechas con timestamps → timestamp; cualquier otra mezcla → string (no se pierde nada)."""
    f = {x for x in familias if x != "null"}
    if not f:
        return "string"
    if len(f) == 1:
        return {"string": "string", "bool": "boolean", "int": "bigint", "float": "double", "decimal": "double",
                "date": "date", "timestamp": "timestamp", "timestamp_ns": "timestamp", "binary": "binary"}.get(f.pop(), "string")
    if f <= {"int", "float", "decimal", "bool"}:
        return "double" if f & {"float", "decimal"} else "bigint"
    if f <= {"date", "timestamp", "timestamp_ns"}:
        return "timestamp"
    return "string"


def listar_parquet(ruta, max_prof=6):
    """Parquet bajo una carpeta (recursivo), sin _SUCCESS, .crc ni archivos ocultos."""
    salida, otros = [], []
    def recorrer(r, prof):
        for f in dbutils.fs.ls(r):
            if f.name.startswith(("_", ".")) or f.name.endswith(".crc"):
                continue
            if f.name.endswith("/"):
                if prof < max_prof:
                    recorrer(f.path, prof + 1)
            elif f.name.lower().endswith(".parquet"):
                salida.append({"ruta": f.path, "archivo": f.name, "bytes": int(f.size)})
            else:
                otros.append(f.name)
    recorrer(ruta, 0)
    return sorted(salida, key=lambda a: a["ruta"]), otros


def metadatos(archivo):
    """Columnas, tipos y filas de un parquet leyendo sólo su pie (no lee datos)."""
    try:
        m = pq.read_metadata(ruta_local(archivo["ruta"]))
        return {**archivo, "filas": int(m.num_rows),
                "columnas": [(c.name, str(c.type)) for c in m.schema.to_arrow_schema()], "error": None}
    except Exception as e:
        return {**archivo, "filas": 0, "columnas": [], "error": f"{type(e).__name__}: {str(e)[:150]}"}


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Archivos por mes
# MAGIC **Qué mirar.** Número de archivos y tamaño parecidos entre meses; un mes con la mitad de tamaño es una carga incompleta.

# COMMAND ----------

titulo("1. ARCHIVOS POR MES")
ARCH = {}
filas = []
for mes in MESES:                      # se listan todos los meses: el plan de columnas debe ser igual para todos
    try:
        parquet, otros = listar_parquet(volumen(mes))
    except Exception as e:
        aviso(f"No se pudo listar {volumen(mes)}: {type(e).__name__}")
        parquet, otros = [], []
    for a in parquet:
        a["mes"] = mes
    ARCH[mes] = parquet
    filas.append({"mes": mes, "archivos": len(parquet), "gb": sum(a["bytes"] for a in parquet) / 1024**3,
                  "otros_archivos": len(otros), "ejemplo": ", ".join(a["archivo"] for a in parquet[:2])})
mostrar(pd.DataFrame(filas), "Parquet por volumen", formatos={"archivos": "int", "gb": "dec2"})
if not any(ARCH.values()):
    raise RuntimeError("No encontré archivos parquet en los volúmenes mensuales. Revisa MESES y VOLUMENES en la configuración.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Plan de columnas
# MAGIC **Qué hace.** Para cada columna decide el nombre en Bronce (minúsculas, sin acentos ni símbolos) y el tipo común.
# MAGIC **Qué mirar.** Las columnas con conflicto de tipo y las que sólo aparecen en algunos meses: se revisan en el EDA de calidad.

# COMMAND ----------

titulo("2. PLAN DE COLUMNAS")
t0 = time.time()
with ThreadPoolExecutor(HILOS) as ex:
    METAS = list(ex.map(metadatos, [a for m in MESES for a in ARCH[m]]))
errores = [m for m in METAS if m["error"]]
print(f"Metadatos de {len(METAS):,} archivos en {time.time() - t0:.0f} s · con error: {len(errores)}")
for m in errores[:20]:
    print("  ✗", m["ruta"], m["error"])
if errores:
    aviso(f"{len(errores)} archivos no se pudieron leer y quedan fuera de Bronce.")
validos = [m for m in METAS if not m["error"]]

orden, fams, exactos, meses_col = [], {}, {}, {}
for m in validos:
    for nombre, tipo in m["columnas"]:
        if nombre not in fams:
            orden.append(nombre)
            fams[nombre], exactos[nombre], meses_col[nombre] = {}, {}, set()
        fams[nombre][familia(tipo)] = fams[nombre].get(familia(tipo), 0) + 1
        exactos[nombre][tipo_exacto(tipo)] = exactos[nombre].get(tipo_exacto(tipo), 0) + 1
        meses_col[nombre].add(m["mes"])
usados, PLAN = set(TRAZA), []
for nombre in orden:
    destino, k = sanear(nombre), 2
    base = destino
    while destino in usados:
        destino, k = f"{base}_{k}", k + 1
    usados.add(destino)
    PLAN.append({"columna_original": nombre, "columna_bronce": destino, "tipo_bronce": tipo_comun(fams[nombre]),
                 "tipos_origen": ", ".join(f"{t}×{n}" for t, n in sorted(exactos[nombre].items())),
                 "archivos": sum(fams[nombre].values()), "meses": ", ".join(sorted(meses_col[nombre])),
                 "conflicto": len(exactos[nombre]) > 1, "nanosegundos": "timestamp_ns" in exactos[nombre]})
PLAN_DF = pd.DataFrame(PLAN)
n_arch = len(validos)
print(f"Columnas: {len(PLAN)} · con tipo distinto entre archivos: {int(PLAN_DF['conflicto'].sum())} · "
      f"renombradas: {int((PLAN_DF['columna_original'] != PLAN_DF['columna_bronce']).sum())} · "
      f"ausentes en algún archivo: {int((PLAN_DF['archivos'] < n_arch).sum())}")
mostrar(PLAN_DF[["columna_original", "columna_bronce", "tipo_bronce", "tipos_origen", "archivos", "conflicto"]],
        "Plan completo", formatos={"archivos": "int"}, max_filas=200)
if PLAN_DF["conflicto"].any():
    hallazgo(f"{int(PLAN_DF['conflicto'].sum())} columnas cambian de tipo entre archivos "
             f"({', '.join(PLAN_DF[PLAN_DF['conflicto']]['columna_original'].head(8))}); se leen por grupos y se unifican.")
parciales = PLAN_DF[PLAN_DF["archivos"] < n_arch]
if len(parciales):
    hallazgo(f"{len(parciales)} columnas no vienen en todos los archivos: {', '.join(parciales['columna_original'].head(8))}.")
guardar(PLAN_DF, "mlops", "bronce_columnas", "Plan de columnas de bronze.ventas_lineas: nombre y tipo en Bronce por columna original")
PLAN_POR_NOMBRE = {p["columna_original"]: p for p in PLAN}
CONFLICTIVAS = [p["columna_original"] for p in PLAN if p["conflicto"]]

if existe(TABLA) and not RECARGA_TOTAL:
    actuales = tipos_de(spark.table(TABLA))
    cambios = [f"{p['columna_bronce']}: tabla={actuales[p['columna_bronce']]} plan={p['tipo_bronce']}"
               for p in PLAN if p["columna_bronce"] in actuales and actuales[p["columna_bronce"]] != p["tipo_bronce"]]
    if cambios:
        raise RuntimeError("El tipo de algunas columnas cambió respecto a la tabla existente: " + "; ".join(cambios)
                           + ". Corre de nuevo con RECARGA_TOTAL = True.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Carga y conciliación por mes
# MAGIC **Qué mirar.** En la bitácora, `diferencia` = 0 y `estado` = `ok` en todos los meses.

# COMMAND ----------

titulo("3. CARGA Y CONCILIACIÓN")
HAY_NANOS = any(p["nanosegundos"] for p in PLAN)
NANOS_COMO_NUMERO = conf_set("spark.sql.legacy.parquet.nanosAsLong", "true") if HAY_NANOS else False
if HAY_NANOS and not NANOS_COMO_NUMERO:
    print("Columnas en nanosegundos (" + ", ".join(p["columna_original"] for p in PLAN if p["nanosegundos"])
          + "): este cómputo no permite la configuración para leerlas; se leen con esquema explícito como número y se convierten.")
# Un timestamp guardado sin zona horaria se lee como hora del reloj (TIMESTAMP_NTZ) y al llevarlo a timestamp conserva la
# hora local; sin esto, algunos runtimes lo leen como UTC y las ventas de 0 a 6 h pasarían al día anterior.
conf_set("spark.sql.parquet.inferTimestampNTZ.enabled", "true")
from pyspark.sql.pandas.types import from_arrow_type
from pyspark.sql import types as T
import pyarrow as pa


def esquema_nanos_como_numero(meta):
    """Esquema explícito del archivo con los timestamps en nanosegundos como número (bigint); después se convierten."""
    campos = []
    for f in pq.read_schema(ruta_local(meta["ruta"])):
        t = f.type.value_type if pa.types.is_dictionary(f.type) else f.type
        if pa.types.is_null(t):
            continue
        try:
            tipo = T.LongType() if tipo_exacto(str(f.type)) == "timestamp_ns" else from_arrow_type(t)
            campos.append(T.StructField(f.name, tipo, True))
        except Exception:
            pass
    return T.StructType(campos)


def leer_grupo(metas, mes):
    """Lee archivos con tipos idénticos y los lleva al plan de columnas, con trazabilidad."""
    hay_ns = any(tipo_exacto(t) == "timestamp_ns" for m in metas for _, t in m["columnas"])
    if hay_ns and not NANOS_COMO_NUMERO:
        d = spark.read.schema(esquema_nanos_como_numero(metas[0])).parquet(*[m["ruta"] for m in metas])
    else:
        d = spark.read.option("mergeSchema", "true").parquet(*[m["ruta"] for m in metas])
    d = d.select("*", F.col("_metadata.file_path").alias("__ruta"))
    presentes = set(d.columns)
    leidos = dict(d.dtypes)
    # nanosegundos que Spark leyó como número (con nanosAsLong); un INT96 ya llega como timestamp y sólo se convierte el tipo
    nanos = {n: "tz=" in str(t) for m in metas for n, t in m["columnas"]
             if tipo_exacto(t) == "timestamp_ns" and leidos.get(n) == "bigint"}
    cols = []
    for p in PLAN:
        o = p["columna_original"]
        if o in presentes:
            if o in nanos:            # nanosegundos: a timestamp conservando la hora del reloj si no traía zona
                c = F.expr(f"timestamp_micros(cast(`{o}` / 1000 as bigint))")
                c = c if nanos[o] else F.to_utc_timestamp(c, spark.conf.get("spark.sql.session.timeZone"))
            else:
                c = F.col(f"`{o}`")
            cols.append(c.cast(p["tipo_bronce"]).alias(p["columna_bronce"]))
        else:
            cols.append(F.lit(None).cast(p["tipo_bronce"]).alias(p["columna_bronce"]))
    nombre = F.regexp_extract(F.col("__ruta"), r"([^/]+)$", 1)
    iso = F.regexp_extract(nombre, r"(20\d{2}-\d{2}-\d{2})", 1)
    compacto = F.regexp_extract(nombre, r"(20\d{2}\d{2}\d{2})", 1)
    dia = F.regexp_extract(nombre, r"(?i)^(\d{1,2})\.parquet$", 1)
    fecha_archivo = F.coalesce(
        F.expr("try_to_timestamp(nullif(__iso, ''), 'yyyy-MM-dd')"),
        F.expr("try_to_timestamp(nullif(__compacto, ''), 'yyyyMMdd')"),
        F.expr(f"try_to_timestamp(concat('{mes}-', lpad(nullif(__dia, ''), 2, '0')), 'yyyy-MM-dd')")).cast("date")
    d = d.select(*cols, F.col("__ruta").alias("_archivo"), iso.alias("__iso"), compacto.alias("__compacto"), dia.alias("__dia"))
    return d.select(*[p["columna_bronce"] for p in PLAN], "_archivo", fecha_archivo.alias("_fecha_archivo"),
                    F.current_timestamp().alias("_fecha_carga"), F.lit(mes).alias("anio_mes"))


def clave(ruta):
    return urllib.parse.unquote(ruta_local(ruta))


bitacora, fallidos = [], []
primera = True
for mes in MESES_A_CARGAR:
    t0 = time.time()
    metas = [m for m in validos if m["mes"] == mes]
    if not metas:
        aviso(f"{mes}: sin archivos legibles; no se carga")
        continue
    grupos = {}
    for m in metas:
        tipos = {n: tipo_exacto(t) for n, t in m["columnas"]}
        grupos.setdefault(tuple(tipos.get(c, "ausente") for c in CONFLICTIVAS), []).append(m)
    partes = []
    for gm in grupos.values():
        try:
            d = leer_grupo(gm, mes)
            d.schema
            partes.append(d)
        except Exception as e:
            print(f"  grupo de {len(gm)} archivos ilegible ({type(e).__name__}); se reintenta archivo por archivo")
            for m in gm:
                try:
                    d = leer_grupo([m], mes)
                    d.limit(1).collect()
                    partes.append(d)
                except Exception as e2:
                    fallidos.append({"mes": mes, "archivo": m["ruta"], "error": f"{type(e2).__name__}: {str(e2)[:150]}"})
    if not partes:
        aviso(f"{mes}: ningún grupo legible")
        continue
    df_mes = reduce(lambda a, b: a.unionByName(b), partes)
    escritor = df_mes.write.format("delta")
    if (RECARGA_TOTAL and primera) or not existe(TABLA):
        escritor.mode("overwrite").option("overwriteSchema", "true").partitionBy("anio_mes").saveAsTable(TABLA)
    else:
        escritor.mode("overwrite").option("replaceWhere", f"anio_mes = '{mes}'").option("mergeSchema", "true").saveAsTable(TABLA)
    primera = False
    escritas = spark.table(TABLA).where(F.col("anio_mes") == mes).groupBy("_archivo").count().toPandas()
    escritas["clave"] = escritas["_archivo"].map(clave)
    esperadas = pd.DataFrame({"clave": [clave(m["ruta"]) for m in metas], "filas_meta": [m["filas"] for m in metas]})
    conc = esperadas.merge(escritas[["clave", "count"]], on="clave", how="outer").fillna(0)
    descuadre = conc[conc["filas_meta"] != conc["count"]]
    fila = {"fecha_carga": pd.Timestamp.now(), "anio_mes": mes, "archivos": len(metas),
            "grupos_de_tipos": len(grupos), "filas_metadatos": int(esperadas["filas_meta"].sum()),
            "filas_escritas": int(escritas["count"].sum()), "archivos_fallidos": sum(f["mes"] == mes for f in fallidos),
            "archivos_descuadrados": len(descuadre), "segundos": round(time.time() - t0)}
    fila["diferencia"] = fila["filas_escritas"] - fila["filas_metadatos"]
    fila["estado"] = ("ok" if fila["diferencia"] == 0 and not fila["archivos_fallidos"] and not fila["archivos_descuadrados"]
                      else "revisar")
    bitacora.append(fila)
    print(f"{mes}: {fila['filas_escritas']:,} filas de {fila['filas_metadatos']:,} esperadas · "
          f"{len(grupos)} grupo(s) de tipos · {fila['segundos']} s · {fila['estado']}")
    if len(descuadre):
        mostrar(descuadre.head(10), f"{mes}: archivos descuadrados", formatos={"filas_meta": "int", "count": "int"})

BITACORA = pd.DataFrame(bitacora)
mostrar(BITACORA.drop(columns=["fecha_carga"]), "Bitácora de la carga",
        formatos={"filas_metadatos": "int", "filas_escritas": "int", "diferencia": "int"})
if fallidos:
    mostrar(pd.DataFrame(fallidos), "Archivos que Spark no pudo leer")
if len(BITACORA):
    guardar(BITACORA, "mlops", "bronce_cargas", "Bitacora de cargas a bronze.ventas_lineas", modo="append")
    if (BITACORA["estado"] == "ok").all():
        hallazgo(f"Bronce: {int(BITACORA['filas_escritas'].sum()):,} líneas cargadas de {len(BITACORA)} meses; "
                 "conciliación exacta contra los metadatos de cada archivo.")
    else:
        aviso("Hay meses con diferencias entre filas escritas y metadatos (ver bitácora).")
if NANOS_COMO_NUMERO:
    try:
        spark.conf.unset("spark.sql.legacy.parquet.nanosAsLong")
    except Exception:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificación de la tabla

# COMMAND ----------

titulo("4. VERIFICACIÓN")
B = spark.table(TABLA)
por_mes = (B.groupBy("anio_mes").agg(F.count("*").alias("filas"), F.countDistinct("_archivo").alias("archivos"),
                                      F.avg(F.col("_fecha_archivo").isNotNull().cast("double")).alias("pct_con_fecha_en_nombre"))
           .orderBy("anio_mes").toPandas())
mostrar(por_mes, "Filas por partición", formatos={"filas": "int", "archivos": "int", "pct_con_fecha_en_nombre": "pct"})
if OPTIMIZAR:
    try:
        spark.sql(f"OPTIMIZE {TABLA}")
        print("OPTIMIZE ejecutado")
    except Exception as e:
        print(f"(OPTIMIZE no disponible: {type(e).__name__})")
try:
    det = spark.sql(f"DESCRIBE DETAIL {TABLA}").collect()[0]
    print(f"Archivos Delta: {det['numFiles']:,} · tamaño: {det['sizeInBytes'] / 1024**3:.2f} GB · columnas: {len(B.columns)}")
    spark.sql(f"COMMENT ON TABLE {TABLA} IS 'Lineas de venta de Bnext tal como llegan (sin limpiar), una particion por mes'")
except Exception:
    pass
mostrar(pd.DataFrame([{"columna": n, "tipo": t} for n, t in tipos_de(B).items()]), "Esquema final de Bronce", max_filas=200)

# COMMAND ----------

cerrar("02_carga_bronce", siguiente="03_auditoria_trabajo_previo")
