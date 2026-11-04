# Databricks notebook source
# MAGIC %md
# MAGIC # 31 · Desagregación del pronóstico a producto (SKU) para el surtido en tienda
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Avance 6**
# MAGIC
# MAGIC El modelo pronostica **tienda × categoría** (el grano viable: el producto directo tiene 83 % de ceros). Para que el
# MAGIC pronóstico sirva al surtido, aquí se reparte a los productos de cada categoría en cada tienda con su **participación
# MAGIC móvil de 28 días** (*top-down*), anclada en el origen T del pronóstico (sin mirar días posteriores):
# MAGIC
# MAGIC 1. Participación de cada producto en la venta de su categoría × tienda en los 28 días que terminan en T.
# MAGIC 2. Se conservan los `TOP_SKUS_POR_CATEGORIA` productos con más venta y el resto se agrupa en una fila **OTROS** (así la
# MAGIC    suma de productos reproduce exactamente la categoría y la cola no se asigna en silencio a los líderes).
# MAGIC 3. P50, P10 y P90 de la categoría se reparten proporcionalmente (es una asignación consistente del intervalo, no el
# MAGIC    cuantil propio de cada producto) y se convierten a **unidades** con el precio medio observado del producto:
# MAGIC    `techo_demanda_unidades_p90`. Sin datos de inventario no es un pedido sugerido: es la demanda que cubre el P90.
# MAGIC 4. Se **verifica** la reconciliación (Σ productos = categoría) y se publica una vista **semanal** (h 1–7 y 8–14) que es
# MAGIC    lo que usa el tablero y lo que una tienda revisa al pedir.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.pronostico_14d` (origen operativo más reciente) · `silver.ventas_lineas` · `silver.catalogo_categorias` · `silver.catalogo_tiendas` |
# MAGIC | **Escribe** | `gold.predicciones_sku` (reemplaza su origen) · `gold.surtido_sku_semana` |
# MAGIC | **Parámetros** | `VENTANA_DIAS_SHARE`, `TOP_SKUS_POR_CATEGORIA`, `MODO_PUBLICO` |
# MAGIC | **Tiempo** | 10–30 min |
# MAGIC
# MAGIC **No fabrica pronósticos**: si `gold.pronostico_14d` no existe, se detiene y dice qué correr (`22_pronostico_operativo`).

# COMMAND ----------

# ════════════════════════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DEL PROYECTO · es la misma en los 11 notebooks (si cambias algo, cámbialo en todos)
# ════════════════════════════════════════════════════════════════════════════════════════════════
CATALOGO       = "modeladoprescriptivoventas"     # catálogo de Unity Catalog del proyecto
ESQUEMA_DATOS  = "maestria"                       # esquema donde están los volúmenes de Bnext
RAIZ_VOLUMENES = "/Volumes"
MESES = ["2026-03", "2026-04", "2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"]   # un volumen por mes

# Cortes del walk-forward. Son FIJOS (los del Avance 4): así, al llegar un mes nuevo, los modelos se siguen comparando
# en las mismas ventanas. Los bloques de prueba son datos no vistos: no intervienen en el ajuste ni en la elección del
# modelo y sólo existen cuando ya están cargados sus días (septiembre y octubre).
DIAS_CORTE = 14                                           # días que se pronostican desde cada origen
ORIGEN_AJUSTE = "2026-05-20"                              # bloque 21-may…03-jun: sólo para afinar hiperparámetros
ORIGENES_CORTES = {"corte_1": "2026-06-03", "corte_2": "2026-06-17",      # desarrollo: con estos se compara y se elige
                   "corte_3": "2026-07-01", "corte_4": "2026-08-17"}
ORIGENES_PRUEBA = {"prueba_1": "2026-08-31", "prueba_2": "2026-09-14",    # prueba no vista: 01–14 y 15–28 de septiembre
                   "prueba_3": "2026-09-28", "prueba_4": "2026-10-12"}    # y 29-sep–12-oct, 13–26 de octubre (modelos congelados al 31-ago)

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


def _parametro_a_texto(valor):
    if valor is None:
        return ""
    if isinstance(valor, bool):
        return "True" if valor else "False"
    if isinstance(valor, (list, tuple)):
        return ",".join(str(v) for v in valor)
    return str(valor)


def _parametro_desde_texto(texto, defecto):
    """Convierte el texto del widget al tipo del valor por defecto. Vacío = valor por defecto (o None si el defecto es None)."""
    texto = (texto or "").strip()
    if texto == "" or texto.lower() in ("none", "null"):
        return defecto if defecto is not None else None
    if isinstance(defecto, bool):
        if texto.lower() in ("true", "1", "sí", "si", "yes", "verdadero"):
            return True
        if texto.lower() in ("false", "0", "no", "falso"):
            return False
        raise ValueError(f"El parámetro esperaba True/False y recibió {texto!r}")
    if isinstance(defecto, int) and not isinstance(defecto, bool):
        return int(float(texto))
    if isinstance(defecto, float):
        return float(texto)
    if isinstance(defecto, (list, tuple)):
        if texto.startswith("["):
            import json
            return list(json.loads(texto))
        return [v.strip() for v in texto.split(",") if v.strip()]
    return texto


def parametro(nombre, defecto, descripcion=None):
    """Parámetro del notebook con el mismo nombre que la variable. Si el notebook corre en un Job (o se escribió el valor
    en el widget de la parte superior), toma ese valor; si no, usa el valor por defecto escrito en el código. El tipo
    (True/False, número, lista separada por comas, texto) se deduce del valor por defecto. Así el mismo notebook sirve en
    interactivo y en Workflows sin editar celdas."""
    try:
        dbutils.widgets.text(nombre, _parametro_a_texto(defecto), descripcion or nombre)
        crudo = dbutils.widgets.get(nombre)
    except Exception:
        return defecto
    valor = _parametro_desde_texto(crudo, defecto)
    if _parametro_a_texto(valor) != _parametro_a_texto(defecto):
        print(f"(parámetro {nombre} = {valor!r}, tomado del widget o del Job; por defecto {defecto!r})")
    return valor

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


def comentar_columnas(nombre_tabla, descripciones):
    """Escribe el comentario de cada columna de una tabla (queda visible en Catalog Explorer y lo usa Genie)."""
    hechos = 0
    for col, texto in (descripciones or {}).items():
        try:
            spark.sql(f"ALTER TABLE {nombre_tabla} ALTER COLUMN `{col}` COMMENT '{str(texto).replace(chr(39), '')}'")
            hechos += 1
        except Exception:
            pass
    return hechos


def asegurar_restricciones(nombre_tabla, restricciones):
    """Agrega restricciones CHECK de Delta que aún no existan. restricciones = {"nombre": "expresión SQL"}. Delta valida
    la regla sobre los datos existentes: si no se cumple, la rechaza y aquí queda como aviso (la tabla sigue sin ella)."""
    try:
        existentes = {r["key"].replace("delta.constraints.", "") for r in spark.sql(f"SHOW TBLPROPERTIES {nombre_tabla}").collect()
                      if str(r["key"]).startswith("delta.constraints.")}
    except Exception:
        existentes = set()
    for nombre, expresion in (restricciones or {}).items():
        if nombre in existentes:
            continue
        try:
            spark.sql(f"ALTER TABLE {nombre_tabla} ADD CONSTRAINT {nombre} CHECK ({expresion})")
        except Exception as e:
            aviso(f"la restricción {nombre} ({expresion}) no se pudo agregar a {nombre_tabla}: {str(e)[:140]}")


def borrar_corrida(nombre_tabla, valor, columna="run_id"):
    """Borra de una tabla las filas de una corrida antes de volver a escribirlas (idempotencia: repetir o reanudar una
    corrida no duplica resultados). Devuelve True si borró o no había nada que borrar."""
    if not existe(nombre_tabla):
        return True
    try:
        if columna in spark.table(nombre_tabla).columns:
            spark.sql(f"DELETE FROM {nombre_tabla} WHERE {columna} = '{str(valor).replace(chr(39), '')}'")
        return True
    except Exception as e:
        print(f"(no se pudo borrar {columna} = {valor} de {nombre_tabla}: {type(e).__name__}: {str(e)[:100]})")
        return False


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
    try:      # bitácora de corridas: qué notebook corrió, cuándo y cuánto tardó (la lee el tablero de estado del pipeline)
        fila = [(notebook, dt.datetime.fromtimestamp(_INICIO_NOTEBOOK), dt.datetime.now(),
                 float((time.time() - _INICIO_NOTEBOOK) / 60), len(HALLAZGOS),
                 sum(1 for h in HALLAZGOS if h.startswith("REVISAR:")), "OK")]
        (spark.createDataFrame(fila, "notebook string, inicio timestamp, fin timestamp, minutos double, hallazgos int, avisos int, estado string")
         .write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(tabla("mlops", "corridas")))
    except Exception as e:
        print(f"(no se pudo registrar la corrida en mlops.corridas: {type(e).__name__})")
    if siguiente:
        print(f"\nSiguiente notebook: {siguiente}")


class Cronometro:
    """Mide los segundos de un bloque de código: `with Cronometro() as reloj: ...` y después `reloj.segundos`."""

    def __enter__(self):
        self._inicio = time.perf_counter()
        self.segundos = 0.0
        return self

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self._inicio
        return False


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

# ----------------------------------------------------------------- parámetros de este notebook (también como parámetros de Job)
VENTANA_DIAS_SHARE = parametro("VENTANA_DIAS_SHARE", 28)         # días que terminan en T para medir la participación
TOP_SKUS_POR_CATEGORIA = parametro("TOP_SKUS_POR_CATEGORIA", 15)  # productos líderes por categoría × tienda; el resto es OTROS
MODO_PUBLICO = parametro("MODO_PUBLICO", True)
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_LINEAS = tabla("silver", "ventas_lineas")
TABLA_SKU = tabla("gold", "predicciones_sku")
TABLA_SKU_SEMANA = tabla("gold", "surtido_sku_semana")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pronóstico de categoría y ventana de participación

# COMMAND ----------

titulo("1. PRONÓSTICO BASE")
PRON = leer_tabla(TABLA_PRONOSTICO, "22_pronostico_operativo (o Avance5.43 con REENTRENAR_FINAL = True)")
if "fase" in PRON.columns:
    PRON = PRON.where(F.coalesce(F.col("fase"), F.lit("operativo")) == "operativo")
ORIGEN = PRON.agg(F.max("origen")).collect()[0][0]
if ORIGEN is None:
    raise RuntimeError(f"{TABLA_PRONOSTICO} no tiene pronósticos operativos: corre 22_pronostico_operativo.")
CAT_RUN = (PRON.where(F.col("origen") == F.lit(ORIGEN))
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   "origen", "fecha_objetivo", "h", "tipo_categoria", "region", "prediccion_puntual", "p10", "p50", "p90", "techo_operativo",
                   "modelo", "run_id"))
DESDE = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=VENTANA_DIAS_SHARE - 1)).date()
MESES_VENTANA = sorted({pd.Timestamp(DESDE).strftime("%Y-%m"), pd.Timestamp(ORIGEN).strftime("%Y-%m")})
n = CAT_RUN.agg(F.count("*").alias("filas"), F.countDistinct("tienda", "categoria_id").alias("series")).collect()[0]
print(f"Origen {ORIGEN} · {n['filas']:,} filas de categoría · {n['series']:,} series · ventana de participación {DESDE} → {ORIGEN} ({VENTANA_DIAS_SHARE} días, meses {MESES_VENTANA})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Participación de cada producto (28 días que terminan en T) con fila OTROS

# COMMAND ----------

titulo("2. PARTICIPACIÓN POR PRODUCTO")
L = leer_tabla(TABLA_LINEAS, "08_limpieza_plata")
ventana = (L.where(F.col("anio_mes").isin(MESES_VENTANA) & (F.col("fecha_venta") >= F.lit(DESDE)) & (F.col("fecha_venta") <= F.lit(ORIGEN))
                   & (F.col("venta") > 0) & F.col("producto_id").isNotNull())
           .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                   F.col("producto_id").cast("string").alias("producto_id"), "venta", "cantidad", "fecha_venta"))
sku = (ventana.groupBy("tienda", "categoria_id", "producto_id")
       .agg(F.sum("venta").alias("venta_28d"), F.sum("cantidad").alias("unidades_28d"),
            F.countDistinct("fecha_venta").alias("dias_con_venta_28d"), F.max("fecha_venta").alias("ultima_venta")))
w_cat = Window.partitionBy("tienda", "categoria_id")
w_rank = Window.partitionBy("tienda", "categoria_id").orderBy(F.desc("venta_28d"), F.desc("unidades_28d"), "producto_id")
sku = (sku.withColumn("total_cat_venta", F.sum("venta_28d").over(w_cat)).withColumn("total_cat_unidades", F.sum("unidades_28d").over(w_cat))
       .withColumn("rank_sku", F.row_number().over(w_rank)).withColumn("es_lider", F.col("rank_sku") <= TOP_SKUS_POR_CATEGORIA))
lideres = sku.where("es_lider").select("tienda", "categoria_id", "producto_id", "rank_sku", "venta_28d", "unidades_28d", "dias_con_venta_28d", "ultima_venta", "total_cat_venta")
otros = (sku.where(~F.col("es_lider")).groupBy("tienda", "categoria_id")
         .agg(F.lit("OTROS").alias("producto_id"), F.lit(TOP_SKUS_POR_CATEGORIA + 1).cast("int").alias("rank_sku"), F.sum("venta_28d").alias("venta_28d"),
              F.sum("unidades_28d").alias("unidades_28d"), F.max("dias_con_venta_28d").alias("dias_con_venta_28d"), F.max("ultima_venta").alias("ultima_venta"),
              F.max("total_cat_venta").alias("total_cat_venta"), F.countDistinct("producto_id").alias("productos_en_otros")))
SHARE = (lideres.withColumn("productos_en_otros", F.lit(0).cast("long")).unionByName(otros.withColumn("productos_en_otros", F.col("productos_en_otros").cast("long")))
         .withColumn("share_venta", F.col("venta_28d") / F.col("total_cat_venta"))
         .withColumn("precio_medio", F.col("venta_28d") / F.greatest(F.col("unidades_28d"), F.lit(1e-6))))
r = SHARE.agg(F.countDistinct("tienda", "categoria_id").alias("series"), F.count("*").alias("filas"),
              F.sum(F.when(F.col("producto_id") == "OTROS", F.col("share_venta"))).alias("share_otros_total"),
              F.sum(F.when(F.col("producto_id") == "OTROS", 1).otherwise(0)).alias("series_con_otros")).collect()[0].asDict()
share_otros = (SHARE.where(F.col("producto_id") == "OTROS").agg(F.avg("share_venta")).collect()[0][0]) or 0.0
mostrar(pd.DataFrame([{**r, "share_medio_otros": share_otros}]), "Participaciones calculadas",
        formatos={"series": "int", "filas": "int", "series_con_otros": "int", "share_medio_otros": "pct"})
hallazgo(f"Participación por producto en {r['series']:,} series (ventana de {VENTANA_DIAS_SHARE} días); los {TOP_SKUS_POR_CATEGORIA} líderes cubren en promedio "
         f"el {1 - share_otros:.0%} de la venta y la fila OTROS el {share_otros:.0%}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reparto del pronóstico y techo de demanda en unidades

# COMMAND ----------

titulo("3. REPARTO A PRODUCTO")
con_share = CAT_RUN.join(SHARE, ["tienda", "categoria_id"], "inner")
sin_share = (CAT_RUN.join(SHARE.select("tienda", "categoria_id").distinct(), ["tienda", "categoria_id"], "left_anti")
             .withColumn("producto_id", F.lit("SIN_HISTORIA_SKU")).withColumn("rank_sku", F.lit(0).cast("int")).withColumn("share_venta", F.lit(1.0))
             .withColumn("precio_medio", F.lit(None).cast("double")).withColumn("dias_con_venta_28d", F.lit(0).cast("long")).withColumn("ultima_venta", F.lit(None).cast("date"))
             .withColumn("productos_en_otros", F.lit(0).cast("long")))
SKU = (con_share.select(*CAT_RUN.columns, "producto_id", "rank_sku", "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros")
       .unionByName(sin_share.select(*CAT_RUN.columns, "producto_id", "rank_sku", "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros"))
       .withColumn("prediccion_sku_p50", F.col("prediccion_puntual") * F.col("share_venta"))
       .withColumn("p10_sku", F.col("p10") * F.col("share_venta")).withColumn("p90_sku", F.col("p90") * F.col("share_venta"))
       .withColumn("techo_demanda_unidades_p90", F.when(F.col("precio_medio") > 0, F.ceil(F.col("p90_sku") / F.col("precio_medio"))).cast("int"))
       .withColumn("demanda_unidades_p50", F.when(F.col("precio_medio") > 0, F.col("prediccion_sku_p50") / F.col("precio_medio")))
       .withColumn("semana_horizonte", F.when(F.col("h") <= 7, 1).otherwise(2))
       .withColumn("fecha_desagregacion", F.current_timestamp())
       .select("tienda", "categoria_id", "origen", "fecha_objetivo", "h", "semana_horizonte", "tipo_categoria", "region", "producto_id", "rank_sku",
               "share_venta", "precio_medio", "dias_con_venta_28d", "ultima_venta", "productos_en_otros",
               "prediccion_sku_p50", "p10_sku", "p90_sku", "demanda_unidades_p50", "techo_demanda_unidades_p90", "modelo", "run_id", "fecha_desagregacion"))
escritor = SKU.write.format("delta").mode("overwrite")
escritor = (escritor.option("replaceWhere", f"origen = DATE'{ORIGEN}'").option("mergeSchema", "true") if existe(TABLA_SKU)
            else escritor.option("overwriteSchema", "true"))
escritor.partitionBy("origen").saveAsTable(TABLA_SKU)
try:
    spark.sql(f"COMMENT ON TABLE {TABLA_SKU} IS 'Pronostico de categoria repartido a producto por participacion de 28 dias (top-down) con fila OTROS; techo de demanda en unidades al P90 (31)'")
except Exception:
    pass
comentar_columnas(TABLA_SKU, {"producto_id": "Producto; OTROS agrupa la cola fuera del top; SIN_HISTORIA_SKU = serie sin ventas en la ventana.",
                              "share_venta": "Participacion del producto en la venta de su categoria x tienda en la ventana.",
                              "prediccion_sku_p50": "Pronostico de la categoria x participacion (pesos).", "p10_sku": "P10 de la categoria x participacion.",
                              "p90_sku": "P90 de la categoria x participacion (asignacion proporcional, no el cuantil propio del producto).",
                              "techo_demanda_unidades_p90": "Unidades que cubren el P90 del producto (p90_sku / precio medio); no descuenta inventario.",
                              "demanda_unidades_p50": "Unidades esperadas (p50).", "semana_horizonte": "1 = h 1-7, 2 = h 8-14."})
print(f"   → guardada {TABLA_SKU}")
# ---- reconciliación verificada: la suma de productos debe igualar la categoría en cada tienda × categoría × día
rec = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN)).groupBy("tienda", "categoria_id", "fecha_objetivo")
       .agg(F.sum("prediccion_sku_p50").alias("suma_sku"))
       .join(CAT_RUN.select("tienda", "categoria_id", "fecha_objetivo", F.col("prediccion_puntual").alias("cat_pron")), ["tienda", "categoria_id", "fecha_objetivo"])
       .withColumn("dif_rel", F.abs(F.col("suma_sku") - F.col("cat_pron")) / F.greatest(F.col("cat_pron"), F.lit(1.0))))
rr = rec.agg(F.count("*").alias("grupos"), F.max("dif_rel").alias("dif_rel_max"), F.avg("dif_rel").alias("dif_rel_media"),
             F.sum((F.col("dif_rel") > 0.001).cast("int")).alias("grupos_fuera")).collect()[0].asDict()
mostrar(pd.DataFrame([rr]), "Reconciliación Σ productos vs categoría (pronóstico P50)", formatos={"grupos": "int", "dif_rel_max": "dec4", "dif_rel_media": "dec4", "grupos_fuera": "int"})
if rr["grupos_fuera"]:
    aviso(f"{rr['grupos_fuera']:,} de {rr['grupos']:,} tienda × categoría × día no reconcilian (diferencia máxima {rr['dif_rel_max']:.2%}).")
else:
    hallazgo(f"Reconciliación verificada: la suma de productos iguala la categoría en los {rr['grupos']:,} tienda × categoría × día (diferencia máxima {rr['dif_rel_max']:.4%}).")
stats = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN))
         .agg(F.count("*").alias("filas"), F.countDistinct("producto_id").alias("productos"),
              F.sum((F.col("producto_id") == "SIN_HISTORIA_SKU").cast("int")).alias("filas_sin_historia"),
              F.sum((F.col("producto_id") == "OTROS").cast("int")).alias("filas_otros")).collect()[0].asDict())
mostrar(pd.DataFrame([stats]), "Tabla por producto", formatos={k: "int" for k in stats})
hallazgo(f"{TABLA_SKU}: {stats['filas']:,} filas, {stats['productos']:,} productos distintos (+ OTROS); {stats['filas_sin_historia']:,} filas de series sin historia de producto.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Vista semanal para el pedido (h 1–7 y h 8–14)

# COMMAND ----------

titulo("4. VISTA SEMANAL")
SEM = (spark.table(TABLA_SKU).where(F.col("origen") == F.lit(ORIGEN))
       .groupBy("tienda", "categoria_id", "tipo_categoria", "region", "producto_id", "rank_sku", "origen", "semana_horizonte")
       .agg(F.min("fecha_objetivo").alias("desde"), F.max("fecha_objetivo").alias("hasta"), F.max("share_venta").alias("share_venta"),
            F.max("precio_medio").alias("precio_medio"), F.sum("prediccion_sku_p50").alias("venta_p50_semana"), F.sum("p90_sku").alias("venta_p90_semana"),
            F.sum("demanda_unidades_p50").alias("unidades_p50_semana"), F.sum("techo_demanda_unidades_p90").alias("techo_unidades_p90_semana"),
            F.max("modelo").alias("modelo"), F.max("run_id").alias("run_id"))
       .withColumn("fecha_calculo", F.current_timestamp()))
guardar(SEM, "gold", "surtido_sku_semana", "Demanda por producto agregada a semana del horizonte (h 1-7 y 8-14): unidades P50 y techo P90 (31)")
comentar_columnas(TABLA_SKU_SEMANA, {"techo_unidades_p90_semana": "Suma de los techos diarios P90 (conservador: los cuantiles no son aditivos).",
                                     "unidades_p50_semana": "Unidades esperadas en la semana.", "semana_horizonte": "1 = dias 1-7, 2 = dias 8-14 desde el origen."})
top = (SEM.where(F.col("producto_id") != "OTROS").groupBy("tipo_categoria", "semana_horizonte")
       .agg(F.countDistinct("tienda").alias("tiendas"), F.countDistinct("producto_id").alias("productos"), F.avg("share_venta").alias("share_medio")).toPandas())
mostrar(top, "Cobertura de la vista semanal", formatos={"tiendas": "int", "productos": "int", "share_medio": "pct"})
if not MODO_PUBLICO:
    mostrar(SEM.groupBy("semana_horizonte").agg(F.sum("venta_p50_semana").alias("venta_p50"), F.sum("techo_unidades_p90_semana").alias("unidades_p90")).toPandas(),
            "Totales (pesos y unidades)", formatos={"venta_p50": "dinero", "unidades_p90": "int"})

# COMMAND ----------

cerrar("31_desagregacion_predicciones_sku", siguiente="40_salidas_dashboard")
