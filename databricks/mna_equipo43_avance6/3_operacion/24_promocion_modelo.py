# Databricks notebook source
# MAGIC %md
# MAGIC # 24 · Promoción del modelo: de `@challenger` a `@champion`
# MAGIC **Proyecto Integrador MNA · Equipo 43**
# MAGIC
# MAGIC Gobierno del registro de modelos en Unity Catalog (`demanda_categoria`). Cada corrida completa de `Avance5.43` que
# MAGIC pasa la compuerta técnica registra una versión nueva con alias `@challenger`. Este notebook decide, con evidencia, si
# MAGIC esa versión pasa a `@champion` (la que usa `22_pronostico_operativo` en producción):
# MAGIC
# MAGIC 1. **Compuerta técnica** de la corrida (`mlops.av5_compuerta`): WMAPE ≤ línea base, sesgo, Diebold–Mariano, cobertura.
# MAGIC 2. **Comparación con el campeón vigente**: el retador no puede ser peor en la prueba no vista (WMAPE) más allá de la tolerancia.
# MAGIC 3. **Aprobaciones de negocio** registradas en la tabla `mlops.aprobaciones` (quién, cuándo, qué): alcance por categoría,
# MAGIC    política de servicios y disponibilidad de atributos al momento de pronosticar. Son datos, no constantes en el código.
# MAGIC
# MAGIC Con `ACCION = "evaluar"` sólo informa; con `ACCION = "promover"` mueve el alias (y conserva al campeón anterior como
# MAGIC `@anterior`) y deja constancia en `mlops.av5_ejecuciones` y `mlops.registro_modelos`.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | registro UC `demanda_categoria` · `mlops.av5_compuerta` · `mlops.av5_comparativa_prueba` · `mlops.aprobaciones` |
# MAGIC | **Escribe** | alias en el registro · `mlops.aprobaciones` (si no existe) · `mlops.registro_modelos` · `mlops.av5_ejecuciones` |
# MAGIC | **Parámetros** | `ACCION` (evaluar / promover), `TOLERANCIA_WMAPE`, `FORZAR` |
# MAGIC | **Tiempo** | 1–3 min |

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
ACCION = parametro("ACCION", "evaluar")                 # "evaluar" (sólo informa) o "promover" (mueve el alias si procede)
TOLERANCIA_WMAPE = parametro("TOLERANCIA_WMAPE", 0.0)  # el retador puede ser peor que el campeón hasta por esto (0 = no)
FORZAR = parametro("FORZAR", False)                     # True: promueve aunque falte evidencia (queda registrado que fue forzado)
NOMBRE_MODELO_UC = f"{CATALOGO}.{ESQUEMA_DATOS}.demanda_categoria"
TABLA_APROBACIONES = tabla("mlops", "aprobaciones")
TABLA_REGISTRO = tabla("mlops", "registro_modelos")
TABLA_EJECUCIONES = tabla("mlops", "av5_ejecuciones")
ASPECTOS = {"alcance_categoria": "Bnext confirma que el grano tienda × categoría y las categorías incluidas son los correctos",
            "politica_servicios": "Bnext confirma si las categorías de servicio se pronostican o se excluyen",
            "atributos_point_in_time": "Bnext confirma que los atributos de tienda y los calendarios estarán disponibles al pronosticar"}
USUARIO = None
try:
    USUARIO = spark.sql("SELECT current_user()").collect()[0][0]
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Versiones en el registro

# COMMAND ----------

titulo("1. REGISTRO DE MODELOS")
REGISTRO_OK, VERSIONES, CLIENTE = False, pd.DataFrame(), None
try:
    import mlflow
    from mlflow import MlflowClient
    mlflow.set_registry_uri("databricks-uc")
    CLIENTE = MlflowClient()
    filas = []
    for mv in CLIENTE.search_model_versions(f"name = '{NOMBRE_MODELO_UC}'"):
        etq = dict(mv.tags or {})
        filas.append({"version": int(mv.version), "aliases": ",".join(sorted(mv.aliases or [])), "modelo": etq.get("modelo"),
                      "run_id_av5": etq.get("run_id_av5"), "wmape_desarrollo": etq.get("wmape_desarrollo"), "wmape_prueba": etq.get("wmape_prueba"),
                      "creada": pd.to_datetime(mv.creation_timestamp, unit="ms"), "run_id_mlflow": mv.run_id, "estado": mv.status})
    VERSIONES = pd.DataFrame(filas).sort_values("version", ascending=False) if filas else pd.DataFrame()
    REGISTRO_OK = True
except Exception as e:
    aviso(f"No se pudo leer el registro {NOMBRE_MODELO_UC} ({type(e).__name__}: {str(e)[:120]}). Sin registro no hay promoción: "
          "revisar permisos (USE SCHEMA, EXECUTE/CREATE MODEL) o que Avance5.43 haya registrado el modelo.")
if REGISTRO_OK and VERSIONES.empty:
    aviso(f"El registro {NOMBRE_MODELO_UC} no tiene versiones: corre Avance5.43 en modo completo (la compuerta técnica registra @challenger).")
if len(VERSIONES):
    mostrar(VERSIONES, "Versiones registradas")


def version_con(alias):
    if VERSIONES.empty:
        return None
    m = VERSIONES[VERSIONES["aliases"].str.split(",").map(lambda a: alias in a)]
    return m.iloc[0] if len(m) else None


CHALLENGER, CHAMPION = version_con("challenger"), version_con("champion")
print("Retador :", None if CHALLENGER is None else f"versión {CHALLENGER['version']} · {CHALLENGER['modelo']} · corrida {CHALLENGER['run_id_av5']}")
print("Campeón :", None if CHAMPION is None else f"versión {CHAMPION['version']} · {CHAMPION['modelo']} · corrida {CHAMPION['run_id_av5']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Evidencia: compuerta técnica, prueba no vista y aprobaciones de negocio

# COMMAND ----------

titulo("2. EVIDENCIA")
EVIDENCIA = {"compuerta_tecnica": None, "mejor_o_igual_que_campeon": None, "aprobaciones_negocio": None}
if CHALLENGER is not None:
    t = tabla("mlops", "av5_compuerta")
    if existe(t):
        comp = spark.table(t).where(F.col("run_id") == CHALLENGER["run_id_av5"]).orderBy(F.col("fecha_publicacion").desc()).limit(1).toPandas()
        if len(comp):
            EVIDENCIA["compuerta_tecnica"] = bool(comp["technical_pass"].iloc[0])
            det = json.loads(comp["detalle_json"].iloc[0]) if isinstance(comp["detalle_json"].iloc[0], str) else {}
            mostrar(pd.DataFrame([{"comprobacion": k, "cumple": v} for k, v in det.get("checks", {}).items()]), "Compuerta técnica del retador")
            print("Motivos de bloqueo registrados en la corrida:", det.get("blocking_reasons"))
    w_ch = pd.to_numeric(pd.Series([CHALLENGER["wmape_prueba"], CHALLENGER["wmape_desarrollo"]]), errors="coerce")
    if CHAMPION is None:
        EVIDENCIA["mejor_o_igual_que_campeon"] = True
        print("No hay campeón: el retador sólo necesita la compuerta técnica y las aprobaciones.")
    else:
        w_cp = pd.to_numeric(pd.Series([CHAMPION["wmape_prueba"], CHAMPION["wmape_desarrollo"]]), errors="coerce")
        ref_ch, ref_cp = w_ch.dropna().iloc[0] if w_ch.notna().any() else np.nan, w_cp.dropna().iloc[0] if w_cp.notna().any() else np.nan
        EVIDENCIA["mejor_o_igual_que_campeon"] = bool(np.isnan(ref_cp) or (not np.isnan(ref_ch) and ref_ch <= ref_cp + TOLERANCIA_WMAPE))
        print(f"WMAPE retador {ref_ch} vs campeón {ref_cp} (tolerancia {TOLERANCIA_WMAPE}) → {'cumple' if EVIDENCIA['mejor_o_igual_que_campeon'] else 'NO cumple'}")
# aprobaciones de negocio como datos
if not existe(TABLA_APROBACIONES):
    base = spark.createDataFrame([(k, v, False, None, None, "pendiente") for k, v in ASPECTOS.items()],
                                 "aspecto string, descripcion string, aprobado boolean, aprobado_por string, fecha timestamp, comentario string")
    guardar(base, "mlops", "aprobaciones", "Aprobaciones de negocio necesarias para promover un modelo a produccion (24)")
    print(f"Se creó {TABLA_APROBACIONES} con los tres aspectos pendientes. Para aprobar uno:\n"
          f"  UPDATE {TABLA_APROBACIONES} SET aprobado = true, aprobado_por = '<nombre>', fecha = current_timestamp(), comentario = '<evidencia>' WHERE aspecto = 'alcance_categoria';")
APROB = spark.table(TABLA_APROBACIONES).toPandas()
mostrar(APROB, "Aprobaciones de negocio")
EVIDENCIA["aprobaciones_negocio"] = bool(len(APROB) and APROB["aprobado"].fillna(False).astype(bool).all())
mostrar(pd.DataFrame([{"criterio": k, "cumple": v} for k, v in EVIDENCIA.items()]), "Resumen de la evidencia")
PUEDE_PROMOVER = CHALLENGER is not None and all(bool(v) for v in EVIDENCIA.values())
hallazgo("Promoción " + ("PROCEDE: " if PUEDE_PROMOVER else "NO procede todavía: ")
         + ", ".join(f"{k} = {'sí' if v else 'no' if v is not None else 'sin dato'}" for k, v in EVIDENCIA.items()) + ".")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Acción

# COMMAND ----------

titulo("3. ACCIÓN")
if ACCION == "promover" and CHALLENGER is not None and REGISTRO_OK and (PUEDE_PROMOVER or FORZAR):
    v = int(CHALLENGER["version"])
    try:
        if CHAMPION is not None and int(CHAMPION["version"]) != v:
            CLIENTE.set_registered_model_alias(NOMBRE_MODELO_UC, "anterior", int(CHAMPION["version"]))
        CLIENTE.set_registered_model_alias(NOMBRE_MODELO_UC, "champion", v)
        try:
            CLIENTE.delete_registered_model_alias(NOMBRE_MODELO_UC, "challenger")
        except Exception:
            pass
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promovido_por", str(USUARIO))
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promovido_en", str(pd.Timestamp.now()))
        CLIENTE.set_model_version_tag(NOMBRE_MODELO_UC, v, "promocion_forzada", str(bool(FORZAR and not PUEDE_PROMOVER)))
        registro = pd.DataFrame([{"fecha": pd.Timestamp.now(), "modelo_uc": NOMBRE_MODELO_UC, "version": v, "accion": "promover_a_champion",
                                  "version_anterior": None if CHAMPION is None else int(CHAMPION["version"]), "run_id_av5": CHALLENGER["run_id_av5"],
                                  "usuario": USUARIO, "forzada": bool(FORZAR and not PUEDE_PROMOVER), "evidencia": json.dumps(EVIDENCIA)}])
        guardar(registro, "mlops", "registro_modelos", "Promociones y cambios de alias del modelo en Unity Catalog (24)", modo="append")
        evento = pd.DataFrame([{"run_id": CHALLENGER["run_id_av5"], "estado": "PROMOVIDO", "fecha_utc": dt.datetime.utcnow(),
                                "info_json": json.dumps({"version": v, "usuario": USUARIO, "forzada": bool(FORZAR and not PUEDE_PROMOVER)})}])
        guardar(evento, "mlops", "av5_ejecuciones", None, modo="append")
        hallazgo(f"Versión {v} promovida a @champion de {NOMBRE_MODELO_UC}" + (" (forzada)" if FORZAR and not PUEDE_PROMOVER else "") + ".")
    except Exception as e:
        aviso(f"No se pudo mover el alias: {type(e).__name__}: {str(e)[:160]}")
elif ACCION == "promover":
    print("No se promueve: " + ("no hay retador" if CHALLENGER is None else "falta evidencia (usa FORZAR = True sólo con acuerdo del equipo)"))
else:
    print("ACCION = evaluar: no se cambia nada en el registro.")

# COMMAND ----------

cerrar("24_promocion_modelo", siguiente="22_pronostico_operativo (con ALIAS_MODELO = champion)")
