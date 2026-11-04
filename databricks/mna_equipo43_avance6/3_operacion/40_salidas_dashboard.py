# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Salidas para el tablero (tablas `gold.dash_*`)
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Avance 6**
# MAGIC
# MAGIC Prepara, ya calculadas, las tablas que lee el tablero AI/BI de Databricks (`dashboard/mna_equipo43_demanda.lvdash.json`).
# MAGIC El tablero sólo consulta y filtra: ningún cálculo pesado vive en él y los datos no salen de Databricks.
# MAGIC
# MAGIC | Tabla | Vista del tablero | Qué responde |
# MAGIC | --- | --- | --- |
# MAGIC | `gold.dash_resumen_ejecutivo` | Resumen | ¿Cuánto se espera vender en los próximos 14 días y cómo se compara con las dos semanas anteriores, por región y tipo? |
# MAGIC | `gold.dash_serie_tiempo` | Pronóstico | Historia reciente (8 semanas) + 14 días de pronóstico con banda P10–P90, desde nacional hasta tienda × categoría |
# MAGIC | `gold.dash_precision` y `gold.dash_precision_modelos` | Precisión | ¿Qué tan bien acierta el modelo (prueba y producción) y contra el ingenuo? |
# MAGIC | `gold.surtido_sku_semana` (31) | Surtido | ¿Qué productos y cuántas unidades cubren la demanda de cada tienda × categoría por semana? |
# MAGIC | `gold.dash_promociones` y `gold.escenarios_whatif_resumen` (30) | Promociones | ¿Qué promociones vienen, cuánto aportan y qué pasaría con otra profundidad? |
# MAGIC | `gold.dash_alertas` | Alertas | Series con cambio fuerte, alta incertidumbre, tiendas inactivas y alertas del monitoreo |
# MAGIC | `gold.dash_estado_pipeline` | Estado | Última corrida de cada notebook, frescura de datos, compuerta, modelo vigente |
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.pronostico_14d`, `silver.panel_categoria_dia`, `silver.catalogo_tiendas`, `mlops.monitoreo_modelo`, `mlops.av5_*`, `mlops.alertas`, `mlops.corridas`, `mlops.validaciones`, `silver.calendario_promociones_plan`, `gold.escenarios_whatif_resumen` |
# MAGIC | **Escribe** | `gold.dash_resumen_ejecutivo`, `gold.dash_serie_tiempo`, `gold.dash_precision`, `gold.dash_precision_modelos`, `gold.dash_promociones`, `gold.dash_alertas`, `gold.dash_estado_pipeline` |
# MAGIC | **Tiempo** | 5–15 min |

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

# ----------------------------------------------------------------- parámetros de este notebook
DIAS_HISTORIA = parametro("DIAS_HISTORIA", 56)                 # semanas de historia real que muestra el tablero (8)
UMBRAL_CAMBIO_SERIE = parametro("UMBRAL_CAMBIO_SERIE", 0.30)   # alerta de serie si el pronóstico de 7 días difiere > 30 % de los 7 días reales previos
UMBRAL_INCERTIDUMBRE = parametro("UMBRAL_INCERTIDUMBRE", 3.0)  # alerta si P90 / P50 > 3 en la semana
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")
TABLA_PANEL = tabla("silver", "panel_categoria_dia")
AHORA = pd.Timestamp.now()
PRON = leer_tabla(TABLA_PRONOSTICO, "22_pronostico_operativo")
if "fase" in PRON.columns:
    PRON = PRON.where(F.coalesce(F.col("fase"), F.lit("operativo")) == "operativo")
ORIGEN = PRON.agg(F.max("origen")).collect()[0][0]
if ORIGEN is None:
    raise RuntimeError("No hay pronóstico operativo: corre 22_pronostico_operativo.")
P = (PRON.where(F.col("origen") == F.lit(ORIGEN))
     .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"), "origen", "fecha_objetivo", "h",
             "tipo_categoria", "region", "prediccion_puntual", "p10", "p90", "modelo", "run_id",
             *([F.col("f_promos_planeadas")] if "f_promos_planeadas" in PRON.columns else [F.lit(0.0).alias("f_promos_planeadas")])))
TIENDAS = leer_tabla(tabla("silver", "catalogo_tiendas"), "08_limpieza_plata").select(F.col("tienda").cast("string").alias("tienda"), F.col("zona").cast("string").alias("zona"),
                                                                                        F.col("jefatura").cast("string").alias("jefatura"), F.col("tipo_tienda").cast("string").alias("tipo_tienda"),
                                                                                        F.col("region").cast("string").alias("region"))
CAT = leer_tabla(tabla("silver", "catalogo_categorias"), "08_limpieza_plata").select(F.col("categoria_id").cast("string").alias("categoria_id"), F.col("nombre").alias("categoria_nombre"))
P = P.join(TIENDAS.drop("region"), "tienda", "left").join(CAT, "categoria_id", "left")
PANEL = leer_tabla(TABLA_PANEL, "09_panel_diario")
DESDE_HIST = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=DIAS_HISTORIA - 1)).date()
REAL = (PANEL.where((F.col("fecha") >= F.lit(DESDE_HIST)) & (F.col("fecha") <= F.lit(ORIGEN)) & (F.col("sin_datos") == 0))
        .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"), "fecha",
                F.col("venta").alias("real"), "tipo_categoria", F.col("tienda_inactiva"),
                (F.coalesce(F.col("promos_vigentes"), F.lit(0)) > 0).cast("double").alias("con_promo"))
        .join(TIENDAS, "tienda", "left").join(CAT, "categoria_id", "left"))
print(f"Origen {ORIGEN} · historia desde {DESDE_HIST}")
NIVELES = {"nacional": [], "region": ["region"], "tipo_categoria": ["tipo_categoria"], "region_tipo": ["region", "tipo_categoria"],
           "zona": ["zona"], "categoria": ["categoria_id", "categoria_nombre", "tipo_categoria"], "tienda": ["tienda", "region", "zona"],
           "serie": ["tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria"]}
LLAVES = ["tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria"]


def con_llaves(df, dims):
    for c in LLAVES:
        if c not in dims:
            df = df.withColumn(c, F.lit(None).cast("string"))
    return df.select(*LLAVES, *[c for c in df.columns if c not in LLAVES])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Serie de tiempo: historia + pronóstico con banda, en todos los niveles

# COMMAND ----------

titulo("1. SERIE DE TIEMPO")
partes = []
for nivel, dims in NIVELES.items():
    h = (REAL.groupBy(*dims, "fecha").agg(F.sum("real").alias("real"), F.countDistinct("tienda").alias("tiendas"),
                                          F.avg("con_promo").alias("pct_series_con_promo"))
         .withColumn("pronostico", F.lit(None).cast("double")).withColumn("p10", F.lit(None).cast("double")).withColumn("p90", F.lit(None).cast("double"))
         .withColumn("tipo_dato", F.lit("real")))
    p = (P.groupBy(*dims, F.col("fecha_objetivo").alias("fecha")).agg(F.sum("prediccion_puntual").alias("pronostico"), F.sum("p10").alias("p10"), F.sum("p90").alias("p90"),
                                                                     F.countDistinct("tienda").alias("tiendas"),
                                                                     F.avg((F.col("f_promos_planeadas") > 0).cast("double")).alias("pct_series_con_promo"))
         .withColumn("real", F.lit(None).cast("double")).withColumn("tipo_dato", F.lit("pronostico")))
    partes.append(con_llaves(h.unionByName(p, allowMissingColumns=True).withColumn("nivel", F.lit(nivel)), dims))
SERIE = partes[0]
for q in partes[1:]:
    SERIE = SERIE.unionByName(q)
SERIE = SERIE.withColumn("origen", F.lit(ORIGEN)).withColumn("fecha_calculo", F.lit(AHORA))
guardar(SERIE, "gold", "dash_serie_tiempo", "Historia reciente y pronostico con banda P10-P90 por nivel (nacional, region, zona, tipo, categoria, tienda, serie) (40)", particion="nivel")
comentar_columnas(tabla("gold", "dash_serie_tiempo"), {"nivel": "Nivel de agregacion: nacional, region, tipo_categoria, region_tipo, zona, categoria, tienda, serie.",
                                                       "tipo_dato": "real (venta observada) o pronostico.", "p10": "Suma de P10 (banda conservadora al agregar).",
                                                       "p90": "Suma de P90 (banda conservadora al agregar).", "tiendas": "Tiendas que aportan a la fila.",
                                                       "pct_series_con_promo": "Fraccion de series con promocion ese dia: observada (real) o planeada (pronostico)."})
n = spark.table(tabla("gold", "dash_serie_tiempo")).groupBy("nivel").count().toPandas()
mostrar(n, "Filas por nivel", formatos={"count": "int"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Resumen ejecutivo: 14 días pronosticados vs. 14 días reales anteriores

# COMMAND ----------

titulo("2. RESUMEN EJECUTIVO")
prev14 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=13)).date()
prev28 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=27)).date()
partes = []
for nivel, dims in NIVELES.items():
    if nivel in ("serie", "tienda"):
        continue
    f = (P.groupBy(*dims).agg(F.sum("prediccion_puntual").alias("venta_pronostico_14d"), F.sum("p10").alias("p10_14d"), F.sum("p90").alias("p90_14d"),
                              F.countDistinct("tienda", "categoria_id").alias("series"), F.countDistinct("tienda").alias("tiendas"),
                              F.avg((F.col("f_promos_planeadas") > 0).cast("double")).alias("pct_filas_con_promo_planeada")))
    r = (REAL.where(F.col("fecha") > F.lit(prev14)).groupBy(*dims).agg(F.sum("real").alias("venta_real_14d_previos")))
    r2 = (REAL.where((F.col("fecha") > F.lit(prev28)) & (F.col("fecha") <= F.lit(prev14))).groupBy(*dims).agg(F.sum("real").alias("venta_real_14d_anteriores")))
    j = f.join(r, dims, "left") if dims else f.crossJoin(r)
    j = j.join(r2, dims, "left") if dims else j.crossJoin(r2)
    partes.append(con_llaves(j.withColumn("nivel", F.lit(nivel)), dims))
RES = partes[0]
for q in partes[1:]:
    RES = RES.unionByName(q)
RES = (RES.withColumn("variacion_vs_14d_previos", F.col("venta_pronostico_14d") / F.col("venta_real_14d_previos") - 1)
       .withColumn("variacion_14d_previos_vs_anteriores", F.col("venta_real_14d_previos") / F.col("venta_real_14d_anteriores") - 1)
       .withColumn("ancho_relativo_banda", (F.col("p90_14d") - F.col("p10_14d")) / F.col("venta_pronostico_14d"))
       .withColumn("origen", F.lit(ORIGEN)).withColumn("desde", F.date_add(F.lit(ORIGEN), 1)).withColumn("hasta", F.date_add(F.lit(ORIGEN), 14))
       .withColumn("fecha_calculo", F.lit(AHORA)))
guardar(RES, "gold", "dash_resumen_ejecutivo", "Venta esperada en los 14 dias siguientes vs 14 dias reales previos por nivel (40)")
comentar_columnas(tabla("gold", "dash_resumen_ejecutivo"), {"variacion_vs_14d_previos": "Pronostico 14 dias / venta real de los 14 dias previos - 1.",
                                                            "ancho_relativo_banda": "(P90 - P10) / pronostico: incertidumbre relativa.",
                                                            "pct_filas_con_promo_planeada": "Fraccion de tienda x categoria x dia con promocion planeada en el horizonte."})
vista = spark.table(tabla("gold", "dash_resumen_ejecutivo")).where(F.col("nivel").isin("nacional", "region", "tipo_categoria")) \
    .select("nivel", "region", "tipo_categoria", "series", "variacion_vs_14d_previos", "ancho_relativo_banda", "pct_filas_con_promo_planeada").toPandas()
mostrar(vista, "Resumen (sin importes)", formatos={"series": "int", "variacion_vs_14d_previos": "pct", "ancho_relativo_banda": "dec2", "pct_filas_con_promo_planeada": "pct"})
nac = vista[vista["nivel"] == "nacional"].iloc[0]
hallazgo(f"Próximos 14 días ({ORIGEN} + 1…14): venta esperada {nac['variacion_vs_14d_previos']:+.1%} respecto a los 14 días reales previos, "
         f"en {int(nac['series']):,} series; {nac['pct_filas_con_promo_planeada']:.0%} de las filas con promoción planeada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Precisión: monitoreo (prueba y producción) y comparativa de modelos

# COMMAND ----------

titulo("3. PRECISIÓN")
if existe(tabla("mlops", "monitoreo_modelo")):
    M = spark.table(tabla("mlops", "monitoreo_modelo"))
    ult = M.groupBy("fase").agg(F.max("fecha_calculo").alias("f")).collect()
    M = M.join(spark.createDataFrame([(r["fase"], r["f"]) for r in ult], "fase string, f timestamp"), "fase").where(F.col("fecha_calculo") == F.col("f")).drop("f")
    guardar(M.withColumn("fecha_tablero", F.lit(AHORA)), "gold", "dash_precision", "WMAPE, sesgo y cobertura por semana x region x tipo en prueba y produccion (ultima corrida de cada fase) (40)")
    mostrar(M.groupBy("fase", "modelo").agg(F.count("*").alias("filas"), F.avg("wmape").alias("wmape_medio"), F.avg("cobertura").alias("cobertura_media")).toPandas(),
            "Monitoreo disponible", formatos={"filas": "int", "wmape_medio": "dec4", "cobertura_media": "pct"})
else:
    print("Sin tabla de monitoreo todavía (corre Avance5.43 completo y 23_monitoreo).")
if existe(tabla("mlops", "av5_comparativa")):
    C = spark.table(tabla("mlops", "av5_comparativa"))
    ultimo_run = C.orderBy(F.col("fecha_publicacion").desc()).select("run_id").limit(1).collect()[0][0]
    cols = [c for c in ("run_id", "modelo", "tipo", "estrategia", "wmape_media_cortes", "mejora_vs_ingenuo", "mejora_vs_mejor_individual", "sesgo_abs_medio_pct",
                        "wmape_mercancia", "segundos_entrenamiento", "fase") if c in C.columns]
    partes = [C.where(F.col("run_id") == ultimo_run).select(*cols)]
    if existe(tabla("mlops", "av5_comparativa_prueba")):
        CP = spark.table(tabla("mlops", "av5_comparativa_prueba"))
        partes.append(CP.where(F.col("run_id") == ultimo_run).select(*[c for c in cols if c in CP.columns]))
    CM = partes[0].unionByName(partes[1], allowMissingColumns=True) if len(partes) > 1 else partes[0]
    guardar(CM.withColumn("fecha_tablero", F.lit(AHORA)), "gold", "dash_precision_modelos", "Comparativa de modelos de la ultima corrida (desarrollo y prueba) (40)")
    mostrar(CM.orderBy("fase", "wmape_media_cortes").toPandas().head(12), "Comparativa (última corrida)", formatos={"wmape_media_cortes": "dec4", "mejora_vs_ingenuo": "pct"})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Promociones próximas con el aporte estimado

# COMMAND ----------

titulo("4. PROMOCIONES")
if existe(tabla("silver", "calendario_promociones_plan")):
    PL = spark.table(tabla("silver", "calendario_promociones_plan")).where(F.col("fin_planeado") >= F.lit(ORIGEN)).join(CAT, "categoria_id", "left")
    if existe(tabla("gold", "escenarios_whatif_resumen")):
        E = spark.table(tabla("gold", "escenarios_whatif_resumen")).where(F.col("origen") == F.lit(ORIGEN))
        base = E.where(F.col("id_escenario") == "S_PLAN").groupBy("categoria_id").agg(F.sum("venta_escenario").alias("con_plan"))
        sinp = E.where(F.col("id_escenario") == "S0_SIN_PROMO").groupBy("categoria_id").agg(F.sum("venta_escenario").alias("sin_promo"))
        aporte = base.join(sinp, "categoria_id").withColumn("aporte_plan_pct", (F.col("con_plan") / F.col("sin_promo") - 1) * 100).select("categoria_id", "aporte_plan_pct")
        PL = PL.join(aporte, "categoria_id", "left")
    else:
        PL = PL.withColumn("aporte_plan_pct", F.lit(None).cast("double"))
    PL = (PL.withColumn("dias_para_inicio", F.datediff(F.col("inicio_planeado"), F.lit(ORIGEN)))
          .withColumn("en_horizonte", (F.col("inicio_planeado") <= F.date_add(F.lit(ORIGEN), 14)) & (F.col("fin_planeado") > F.lit(ORIGEN)))
          .withColumn("fecha_calculo", F.lit(AHORA)))
    guardar(PL, "gold", "dash_promociones", "Promociones planeadas vigentes o proximas con el aporte estimado del plan por categoria (40)")
    mostrar(PL.groupBy("fuente", "en_horizonte").agg(F.countDistinct("promo_id").alias("promociones"), F.avg("aporte_plan_pct").alias("aporte_medio_pct")).toPandas(),
            "Promociones por fuente", formatos={"promociones": "int", "aporte_medio_pct": "dec1"})
else:
    print("Sin calendario de promociones (corre 13).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Alertas para el negocio y del monitoreo

# COMMAND ----------

titulo("5. ALERTAS")
prev7 = (pd.Timestamp(ORIGEN) - pd.Timedelta(days=6)).date()
real7 = REAL.where(F.col("fecha") >= F.lit(prev7)).groupBy("tienda", "categoria_id").agg(F.sum("real").alias("real_7d"), F.sum("tienda_inactiva").alias("dias_inactiva_7d"))
pron7 = (P.where(F.col("h") <= 7).groupBy("tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria")
         .agg(F.sum("prediccion_puntual").alias("pronostico_7d"), F.sum("p90").alias("p90_7d"), F.sum("prediccion_puntual").alias("p50_7d")))
S = (pron7.join(real7, ["tienda", "categoria_id"], "left")
     .withColumn("cambio", F.when(F.col("real_7d") > 0, F.col("pronostico_7d") / F.col("real_7d") - 1))
     .withColumn("incertidumbre", F.when(F.col("p50_7d") > 0, F.col("p90_7d") / F.col("p50_7d"))))
alertas_series = (S.select("tienda", "categoria_id", "categoria_nombre", "region", "zona", "tipo_categoria",
                           F.when(F.col("cambio") > UMBRAL_CAMBIO_SERIE, "subida_fuerte").when(F.col("cambio") < -UMBRAL_CAMBIO_SERIE, "caida_fuerte")
                           .when(F.col("incertidumbre") > UMBRAL_INCERTIDUMBRE, "alta_incertidumbre")
                           .when(F.col("dias_inactiva_7d") >= 3, "tienda_inactiva").alias("tipo"),
                           F.coalesce(F.col("cambio"), F.lit(0.0)).alias("cambio_7d"), F.col("incertidumbre"), F.col("dias_inactiva_7d"))
                  .where(F.col("tipo").isNotNull())
                  .withColumn("severidad", F.when(F.col("tipo").isin("caida_fuerte", "tienda_inactiva"), "alta").otherwise("media"))
                  .withColumn("mensaje", F.concat(F.lit("Serie "), F.col("categoria_nombre"), F.lit(": "), F.col("tipo"), F.lit(" ("), F.round(F.col("cambio_7d") * 100, 0).cast("int").cast("string"), F.lit(" % vs 7 dias previos)")))
                  .withColumn("origen", F.lit(ORIGEN)).withColumn("fuente", F.lit("series")).withColumn("fecha", F.lit(AHORA)))
if existe(tabla("mlops", "alertas")):
    A = (spark.table(tabla("mlops", "alertas")).where(~F.col("atendida"))
         .select(F.lit(None).cast("string").alias("tienda"), F.lit(None).cast("string").alias("categoria_id"), F.lit(None).cast("string").alias("categoria_nombre"),
                 F.lit(None).cast("string").alias("region"), F.lit(None).cast("string").alias("zona"), F.lit(None).cast("string").alias("tipo_categoria"),
                 "tipo", F.lit(None).cast("double").alias("cambio_7d"), F.lit(None).cast("double").alias("incertidumbre"), F.lit(None).cast("long").alias("dias_inactiva_7d"),
                 "severidad", "mensaje", F.col("origen").cast("date").alias("origen"), F.lit("monitoreo").alias("fuente"), F.col("fecha")))
    alertas_series = alertas_series.withColumn("dias_inactiva_7d", F.col("dias_inactiva_7d").cast("long")).unionByName(A)
guardar(alertas_series, "gold", "dash_alertas", "Alertas para el negocio (series con cambio fuerte, incertidumbre, tiendas inactivas) y del monitoreo (40)")
resumen_al = spark.table(tabla("gold", "dash_alertas")).groupBy("fuente", "tipo", "severidad").count().orderBy("fuente", "tipo").toPandas()
mostrar(resumen_al, "Alertas por tipo", formatos={"count": "int"})
hallazgo("Alertas del tablero: " + ", ".join(f"{t} {int(n):,}" for t, n in zip(resumen_al["tipo"], resumen_al["count"])) + ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Estado del pipeline

# COMMAND ----------

titulo("6. ESTADO DEL PIPELINE")
filas = []
def indicador(nombre, valor, detalle="", estado="ok"):
    filas.append({"indicador": nombre, "valor": str(valor), "detalle": str(detalle)[:300], "estado": estado, "fecha_calculo": AHORA})
ultimo_dia = PANEL.where(F.col("sin_datos") == 0).agg(F.max("fecha")).collect()[0][0]
indicador("ultima_venta_cargada", ultimo_dia, f"{(AHORA.date() - ultimo_dia).days} días antes de hoy", "ok" if (AHORA.date() - ultimo_dia).days <= 7 else "aviso")
indicador("origen_pronostico_operativo", ORIGEN, f"pronostica {pd.Timestamp(ORIGEN).date() + pd.Timedelta(days=1)} a {pd.Timestamp(ORIGEN).date() + pd.Timedelta(days=14)}",
          "ok" if ORIGEN >= ultimo_dia else "alerta")
m = P.agg(F.first("modelo").alias("m"), F.first("run_id").alias("r"), F.count("*").alias("n")).collect()[0]
indicador("modelo_en_uso", m["m"], f"corrida {m['r']} · {m['n']:,} filas pronosticadas")
if existe(tabla("mlops", "av5_compuerta")):
    g = spark.table(tabla("mlops", "av5_compuerta")).orderBy(F.col("fecha_publicacion").desc()).limit(1).toPandas().iloc[0]
    indicador("compuerta_tecnica", "cumple" if g["technical_pass"] else "no cumple", f"{g['estado_desarrollo']} · registro: {g['registry_action']}", "ok" if g["technical_pass"] else "alerta")
    indicador("produccion_bloqueada", bool(g["production_blocked"]), "faltan aprobaciones de negocio (mlops.aprobaciones)" if g["production_blocked"] else "", "aviso" if g["production_blocked"] else "ok")
if existe(tabla("mlops", "validaciones")):
    V = spark.table(tabla("mlops", "validaciones"))
    ult = V.agg(F.max("fecha_validacion")).collect()[0][0]
    V = V.where(F.col("fecha_validacion") == F.lit(ult))
    c = V.agg(F.sum((F.col("critica") & (F.col("estado") == "FALLA")).cast("int")).alias("cf"), F.sum((F.col("estado") == "FALLA").cast("int")).alias("f"), F.count("*").alias("n")).collect()[0]
    indicador("reglas_de_calidad", f"{c['n'] - c['f']}/{c['n']} pasan", f"{c['cf']} críticas fallan · validación del {ult}", "ok" if c["cf"] == 0 else "alerta")
if existe(tabla("mlops", "corridas")):
    R = spark.table(tabla("mlops", "corridas")).groupBy("notebook").agg(F.max("fin").alias("fin"), F.max_by("minutos", "fin").alias("minutos")).toPandas()
    for r in R.sort_values("notebook").itertuples():
        indicador(f"notebook {r.notebook}", pd.Timestamp(r.fin).strftime("%Y-%m-%d %H:%M"), f"{r.minutos:.1f} min")
if existe(tabla("mlops", "alertas")):
    na = spark.table(tabla("mlops", "alertas")).where(~F.col("atendida")).count()
    indicador("alertas_monitoreo_abiertas", na, "", "ok" if na == 0 else "aviso")
EST = pd.DataFrame(filas)
guardar(EST, "gold", "dash_estado_pipeline", "Indicadores de estado del pipeline y del modelo para el tablero (40)")
mostrar(EST, "Estado del pipeline")

# COMMAND ----------

cerrar("40_salidas_dashboard", siguiente="importar dashboard/mna_equipo43_demanda.lvdash.json en Databricks (ver dashboard/LEEME.md)")
