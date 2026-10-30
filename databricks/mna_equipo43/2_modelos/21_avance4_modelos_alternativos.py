# Databricks notebook source
# MAGIC %md
# MAGIC # Avance 4 · Modelos alternativos: torneo, ajuste fino y modelo individual final
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Fase 2 (modelos) · entregable `Avance4.43`**
# MAGIC
# MAGIC **Qué hace esta libreta.** Construye y compara **nueve modelos individuales de siete familias** (más la referencia
# MAGIC ingenua) para pronosticar la venta diaria por tienda y categoría a 14 días, con un mismo protocolo de evaluación
# MAGIC (cuatro cortes walk-forward, mismas series, misma función de métricas); ordena la comparativa por la métrica
# MAGIC principal (WMAPE) con métricas secundarias y tiempos; **ajusta los dos mejores** con Optuna; y **elige el modelo
# MAGIC individual final** con una matriz de decisión que pesa error, estabilidad, sesgo, costo, interpretabilidad y
# MAGIC compatibilidad con la siguiente etapa.
# MAGIC
# MAGIC | Sección | Criterio de la rúbrica |
# MAGIC | --- | --- |
# MAGIC | §1 Datos · §2 Protocolo · §3 Modelos · §4 Comparativa | Comparativa: ≥ 6 modelos individuales con algoritmos variados, ordenados por la métrica principal, con otras métricas y los tiempos de entrenamiento |
# MAGIC | §5 Ajuste fino | Ajuste de hiperparámetros con espacio y número de pruebas suficientes; mejora observable |
# MAGIC | §6 Modelo final | Elección argumentada más allá de la métrica (trade-offs, interpretabilidad, negocio, cómputo) |
# MAGIC
# MAGIC **Entrada.** La tabla Oro `gold.features_origen` (notebook 20): una fila por serie, origen T y horizonte h con el
# MAGIC objetivo en T + h y variables calculadas sólo con datos ≤ T (prueba anti-fuga en PASA). Los modelos tabulares
# MAGIC reciben exactamente la misma matriz; los modelos de series reciben la serie diaria; la referencia ingenua, sólo la serie.
# MAGIC
# MAGIC **Confidencialidad.** La libreta imprime únicamente métricas de error (relativas), tiempos, hiperparámetros y
# MAGIC nombres de variables; no imprime registros, identificadores, importes ni volúmenes de venta. Los datos residen en el
# MAGIC catálogo privado de la empresa y no se incluyen en el repositorio.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` · `silver.panel_categoria_dia` · `mlops.linea_base_ingenua` |
# MAGIC | **Escribe** | `mlops.torneo_modelos` · `mlops.torneo_predicciones_muestra` · `mlops.ajuste_fino_pruebas` · `mlops.evaluacion_completa` · `mlops.decision_modelo_final` · modelo en MLflow (y en el registro de Unity Catalog si hay permiso) |
# MAGIC | **Tiempo** | `MODO_RAPIDO = True`: 20–40 min · completo: 3–5 h (conviene correrlo como Job) |
# MAGIC
# MAGIC **Cómo correrla.** De principio a fin («Run all»). Primero con `MODO_RAPIDO = True` para verificar que todo corre;
# MAGIC después con `MODO_RAPIDO = False` para las cifras finales. Cada librería se instala en su propia celda: si una
# MAGIC falla, las demás sí quedan. Si alguna no se puede instalar, el notebook usa la alternativa de la misma familia y lo
# MAGIC dice en la tabla de librerías y en la columna `implementacion` de la comparativa. Si ninguna se instala (sin acceso
# MAGIC a PyPI), agrégalas en el panel **Environment** de la derecha → *Dependencies* y vuelve a correr.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %pip install optuna shap

# COMMAND ----------

# MAGIC %pip install catboost

# COMMAND ----------

# MAGIC %pip install statsforecast

# COMMAND ----------

try:
    dbutils.library.restartPython()      # necesario tras %pip en Databricks; fuera de Databricks no existe y se ignora
except Exception:
    pass

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
MODO_RAPIDO = False               # True: muestras pequeñas y pocas pruebas (20–40 min) para verificar que todo corre
SEMILLA = 43
VERSION_TORNEO = "avance4_v1"     # etiqueta de esta corrida en las tablas mlops.*
N_PRUEBAS_AJUSTE = 6 if MODO_RAPIDO else 40          # pruebas de Optuna por finalista (la rúbrica pide profundidad de búsqueda)
TIEMPO_MAX_AJUSTE_MIN = 15 if MODO_RAPIDO else 45    # tope de tiempo por finalista en el ajuste fino
MAX_FILAS_AJUSTE = 300_000 if MODO_RAPIDO else 600_000     # filas de entrenamiento por prueba de Optuna
TOPE_AJUSTE = {"M3_bosque_aleatorio": 200_000, "M6_perceptron_multicapa": 200_000, "M5_catboost": 400_000}   # modelos lentos: menos filas por prueba
FRACCION_RAPIDO = 0.25            # fracción de series que se usan en MODO_RAPIDO
MAX_FILAS_CARGA_TRAIN = 1_000_000 if MODO_RAPIDO else 3_500_000   # filas de entrenamiento que se traen a memoria (muestra aleatoria si hay más)
EVALUAR_CONJUNTO_COMPLETO = True  # finalistas y modelo final también en TODAS las series (la cifra que se compara con 0.553)
MIN_DIAS_CON_VENTA = 5            # serie relevante en el origen (igual que los notebooks 10 y 20)
EXPERIMENTO_MLFLOW = "/Shared/mna_equipo43/avance4_torneo"
NOMBRE_MODELO_UC = "demanda_categoria"            # nombre del modelo en el registro de Unity Catalog (catalogo.esquema.nombre)
# qué modelos corren (M7 requiere `%pip install neuralforecast`, que instala PyTorch: apagado por defecto)
INCLUIR = {"M1_elastic_net": True, "M2_glm_tweedie": True, "M3_bosque_aleatorio": True, "M4_lightgbm": True, "M5_catboost": True,
           "M6_perceptron_multicapa": True, "M7_nhits": False, "M8_theta": True, "M9_tsb": True}
# tope de filas de entrenamiento por modelo (los lentos se entrenan con una submuestra aleatoria; la evaluación es siempre completa)
MAX_FILAS = {"M1_elastic_net": 2_000_000, "M2_glm_tweedie": 2_000_000, "M3_bosque_aleatorio": 500_000, "M4_lightgbm": None,
             "M5_catboost": 1_000_000, "M6_perceptron_multicapa": 500_000}
if MODO_RAPIDO:
    MAX_FILAS = {k: (min(v, 400_000) if v else 400_000) for k, v in MAX_FILAS.items()}
PESOS_DECISION = {"error": 0.35, "estabilidad": 0.10, "sesgo": 0.10, "error_donde_cuesta": 0.10, "costo": 0.10,
                  "interpretabilidad": 0.10, "compatibilidad": 0.15}
N_HILOS = max(1, (os.cpu_count() or 4))
np.random.seed(SEMILLA)
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["figure.dpi"] = 110
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
TOPE_PREDICCION_FACTOR = 3.0      # ninguna predicción supera 3 × el percentil 99.9 de la venta observada en entrenamiento (protege contra divergencias)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Datos: la tabla Oro, los cortes y las muestras
# MAGIC **Qué hace.** Lee la Oro, codifica las variables categóricas, trae a memoria la muestra de evaluación común (todos
# MAGIC los orígenes) y el conjunto de entrenamiento, y reproduce la referencia ingenua para comprobar que los filtros son
# MAGIC los del notebook 10. Las filas evaluables son las mismas de la línea base: serie con ≥ 5 días de venta antes del
# MAGIC origen, rezago estacional disponible y ≥ 5 días activos en la última semana.

# COMMAND ----------

titulo("1. DATOS")
G = leer_tabla(tabla("gold", "features_origen"), "20_features_oro")
CLAVES = ["tienda", "categoria_id"]
NO_VARIABLES = [*CLAVES, "origen", "h", "fecha_objetivo", "conjunto", "corte", "en_muestra_eval", "y", "y_unidades", "dia_atipico_obj"]
CATEGORICAS = ["tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "tipo_categoria"]
NUM_VARS = [c for c in G.columns if c not in NO_VARIABLES and c not in CATEGORICAS and c != "ingenuo_estacional"] + ["h"]   # h sí es variable
COD_VARS = [f"cod_{c}" for c in CATEGORICAS] + ["cod_categoria"]
X_VARS = NUM_VARS + COD_VARS
EVALUABLE = ((F.col("dias_con_venta_hist") >= MIN_DIAS_CON_VENTA) & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))

ORIGENES = (G.where(F.col("conjunto") == "evaluacion").select("corte", "origen").distinct().toPandas()
            .sort_values("origen").reset_index(drop=True))
CORTES = [c for c in ORIGENES["corte"] if c != "ajuste"]
ORIGEN_DE = dict(zip(ORIGENES["corte"], pd.to_datetime(ORIGENES["origen"])))
ORIGEN_AJUSTE = ORIGEN_DE.get("ajuste")
mostrar(ORIGENES, "Orígenes de evaluación en la Oro (cuatro cortes + origen de ajuste)")

# códigos de las variables categóricas (los mismos en entrenamiento y evaluación; 0 = desconocido)
CODIGOS, ETIQUETAS = {}, {}
for c in CATEGORICAS + ["categoria_id"]:
    vals = sorted(v for v in G.select(F.col(c).cast("string")).distinct().toPandas()[c].tolist() if v is not None)
    CODIGOS[c] = {v: i + 1 for i, v in enumerate(vals)}
    ETIQUETAS[c] = {i + 1: v for i, v in enumerate(vals)}
print("Niveles por variable categórica: " + ", ".join(f"{c} {len(v)}" for c, v in CODIGOS.items()))


def con_codigos(df):
    for c in CATEGORICAS + ["categoria_id"]:
        mapa = F.create_map(*[x for k, v in CODIGOS[c].items() for x in (F.lit(k), F.lit(v))])
        nombre = "cod_categoria" if c == "categoria_id" else f"cod_{c}"
        df = df.withColumn(nombre, F.coalesce(F.try_element_at(mapa, F.col(c).cast("string")), F.lit(0)).cast("int"))
    return df


def cargar(df, columnas_extra=()):
    """Trae a pandas las columnas necesarias con tipos compactos (float32 en variables, códigos enteros)."""
    sel = [*CLAVES, "origen", "h", "fecha_objetivo", "corte", F.col("y").cast("double").alias("y"),
           F.col("ingenuo_estacional").cast("double").alias("ingenuo_estacional"), "dia_atipico_obj", "en_muestra_eval",
           *[F.col(c).cast("float").alias(c) for c in NUM_VARS if c != "h"], *COD_VARS, *columnas_extra]
    pdf = con_codigos(df).select(*sel).toPandas()
    pdf["fecha_objetivo"] = pd.to_datetime(pdf["fecha_objetivo"])
    pdf["origen"] = pd.to_datetime(pdf["origen"])
    for c in COD_VARS:
        pdf[c] = pdf[c].astype("int32")
    return pdf


en_rapido = (F.abs(F.xxhash64(F.col("tienda").cast("string"), F.col("categoria_id").cast("string"), F.lit(SEMILLA))) % 100 < FRACCION_RAPIDO * 100)
filtro_eval = (F.col("conjunto") == "evaluacion") & (F.col("en_muestra_eval") == 1) & EVALUABLE
filtro_train = (F.col("conjunto") == "entrenamiento") & EVALUABLE
if MODO_RAPIDO:
    filtro_eval, filtro_train = filtro_eval & en_rapido, filtro_train & en_rapido
t0 = time.time()
EVAL = cargar(G.where(filtro_eval))
n_train_total = G.where(filtro_train).count()
fraccion_train = min(1.0, MAX_FILAS_CARGA_TRAIN / max(n_train_total, 1))
TRAIN = cargar(G.where(filtro_train & (F.rand(SEMILLA) < fraccion_train)) if fraccion_train < 1 else G.where(filtro_train))
if fraccion_train < 1:
    print(f"(entrenamiento: {n_train_total:,} filas en la Oro; se carga una muestra aleatoria del {fraccion_train:.0%} por memoria del driver)")
print(f"En memoria: evaluación (muestra común) {len(EVAL):,} filas · entrenamiento {len(TRAIN):,} filas · "
      f"{len(X_VARS)} variables · {(time.time() - t0) / 60:.1f} min")
n_series_eval = EVAL[CLAVES].drop_duplicates().shape[0]
n_series_train = TRAIN[CLAVES].drop_duplicates().shape[0]
res_eval = EVAL.groupby("corte").agg(filas=("y", "size"), pct_ceros=("y", lambda s: float((s == 0).mean()))).reset_index()
mostrar(res_eval, "Muestra de evaluación común por origen", formatos={"filas": "int", "pct_ceros": "pct"})
hallazgo(f"Muestra de evaluación común: {n_series_eval:,} series, {len(EVAL):,} filas evaluables en {len(ORIGENES)} orígenes; "
         f"entrenamiento: {n_series_train:,} series, {len(TRAIN):,} filas (serie × origen × horizonte), {len(X_VARS)} variables"
         + (" · MODO_RAPIDO" if MODO_RAPIDO else "") + ".")

# referencia ingenua en la muestra común contra la del notebook 10 (todas las series)
_d = EVAL[EVAL["corte"].isin(CORTES)].assign(e=lambda d: (d["y"] - d["ingenuo_estacional"]).abs())
ing = _d.groupby("corte").agg(e=("e", "sum"), y=("y", "sum")).reset_index()
ing["wmape_muestra"] = ing["e"] / ing["y"]
ing = ing[["corte", "wmape_muestra"]]
if existe(tabla("mlops", "linea_base_ingenua")):
    ref = (spark.table(tabla("mlops", "linea_base_ingenua")).toPandas().sort_values("fecha_calculo").groupby("corte").tail(1)
           .set_index("corte")["wmape_lag7"])
    ing["wmape_todas_las_series_10"] = ing["corte"].map(ref)
mostrar(ing, "Ingenuo estacional: muestra común contra todas las series (notebook 10)", formatos={c: "dec4" for c in ing.columns if c != "corte"})
WMAPE_INGENUO_TODAS = float(ing["wmape_todas_las_series_10"].mean()) if "wmape_todas_las_series_10" in ing else float("nan")
hallazgo(f"Referencia ingenua: WMAPE {ing['wmape_muestra'].mean():.4f} en la muestra común" +
         (f" y {WMAPE_INGENUO_TODAS:.4f} en todas las series (notebook 10)." if np.isfinite(WMAPE_INGENUO_TODAS) else "."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Protocolo común: métricas, prueba de significancia, registro y preparación de variables
# MAGIC **Qué hace.** Una sola función calcula las métricas de cualquier modelo sobre las mismas filas: WMAPE (principal),
# MAGIC sesgo, MAE y RMSE (se reportan relativos a la referencia ingenua), WMAPE entre semana y por bloque de horizonte.
# MAGIC Diebold–Mariano compara, día a día, el error de un modelo con el de la referencia. Cada corrida se registra en
# MAGIC MLflow y en `mlops.torneo_modelos`; la comparativa se genera desde ahí.

# COMMAND ----------

titulo("2. PROTOCOLO")
from scipy.stats import norm


def metricas(y, pred, dow, h):
    y, pred = np.asarray(y, dtype=np.float64), np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    e = np.abs(y - pred)
    real = y.sum()
    def w(mask):
        return float(e[mask].sum() / y[mask].sum()) if mask.any() and y[mask].sum() > 0 else float("nan")
    dow, h = np.asarray(dow), np.asarray(h)
    return {"wmape": float(e.sum() / real) if real > 0 else float("nan"), "sesgo": float(pred.sum() / real - 1) if real > 0 else float("nan"),
            "mae": float(e.mean()), "rmse": float(np.sqrt(((y - pred) ** 2).mean())),
            "wmape_entre_semana": w(dow <= 5), "wmape_fin_semana": w(dow >= 6), "wmape_h1_7": w(h <= 7), "wmape_h8_14": w(h >= 8),
            "filas": int(len(y))}


def diebold_mariano(e_modelo, e_referencia, fechas, rezagos=3):
    """Prueba DM sobre la pérdida absoluta agregada por día: d_t = |e_ref| − |e_modelo| (positivo = el modelo gana).
    Varianza con Newey–West (Bartlett, `rezagos`). Devuelve el estadístico y el valor p bilateral."""
    d = (pd.DataFrame({"f": np.asarray(fechas), "d": np.abs(e_referencia) - np.abs(e_modelo)}).groupby("f")["d"].mean().sort_index().to_numpy())
    T = len(d)
    if T < 8:
        return float("nan"), float("nan")
    dbar = d.mean()
    gamma = [float(np.mean((d[k:] - dbar) * (d[:T - k] - dbar))) for k in range(rezagos + 1)]
    var = gamma[0] + 2 * sum((1 - k / (rezagos + 1)) * gamma[k] for k in range(1, rezagos + 1))
    if var <= 0:
        return float("nan"), float("nan")
    dm = dbar / np.sqrt(var / T)
    return float(dm), float(2 * (1 - norm.cdf(abs(dm))))


# ---- MLflow (opcional: si no está disponible, la libreta sigue y lo dice una vez)
MLFLOW_OK = False
try:
    import mlflow
    try:
        mlflow.autolog(disable=True)
    except Exception:
        pass
    try:
        mlflow.set_experiment(EXPERIMENTO_MLFLOW)
        print(f"MLflow: experimento {EXPERIMENTO_MLFLOW}")
    except Exception as e:
        try:
            usuario = spark.sql("SELECT current_user()").collect()[0][0]
            EXPERIMENTO_MLFLOW = f"/Users/{usuario}/mna_equipo43_avance4_torneo"
            mlflow.set_experiment(EXPERIMENTO_MLFLOW)
            print(f"MLflow: experimento {EXPERIMENTO_MLFLOW} (no hubo permiso para crearlo en /Shared)")
        except Exception:
            print(f"(mlflow: no se pudo fijar un experimento propio: {type(e).__name__}; se usa el experimento de la libreta)")
    MLFLOW_OK = True
except Exception as e:
    print(f"(mlflow no disponible: {type(e).__name__}; las corridas sólo quedan en las tablas mlops)")


def registrar_mlflow(nombre, params, mets, tags=None):
    if not MLFLOW_OK:
        return None
    try:
        with mlflow.start_run(run_name=nombre) as run:
            mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            mlflow.log_metrics({k: float(v) for k, v in mets.items() if isinstance(v, (int, float)) and np.isfinite(v)})
            mlflow.set_tags({"version_torneo": VERSION_TORNEO, **(tags or {})})
            return run.info.run_id
    except Exception as e:
        print(f"(mlflow: {type(e).__name__}: {str(e)[:120]})")
        return None


RESULTADOS = []                                  # una fila por modelo × corte (muestra común)
PRED = {}                                        # predicciones en la muestra común: modelo → Series alineada a EVAL
FAMILIA = {"M0_ingenuo_estacional": "referencia", "M1_elastic_net": "lineal", "M2_glm_tweedie": "lineal generalizado",
           "M3_bosque_aleatorio": "árboles (bagging)", "M4_lightgbm": "boosting de gradiente", "M5_catboost": "boosting de gradiente (ordenado)",
           "M6_perceptron_multicapa": "red neuronal tabular", "M7_nhits": "red neuronal de series", "M8_theta": "estadístico clásico",
           "M9_tsb": "demanda intermitente"}
FAMILIA_GRUESA = {"M1_elastic_net": "lineal", "M2_glm_tweedie": "lineal generalizado", "M3_bosque_aleatorio": "árboles (bagging)",
                  "M4_lightgbm": "boosting de gradiente", "M5_catboost": "boosting de gradiente", "M6_perceptron_multicapa": "red neuronal tabular",
                  "M7_nhits": "red neuronal de series", "M8_theta": "estadístico por serie", "M9_tsb": "estadístico por serie"}
INTERPRETABILIDAD = {"M0_ingenuo_estacional": 1.0, "M1_elastic_net": 1.0, "M2_glm_tweedie": 1.0, "M3_bosque_aleatorio": 0.7, "M4_lightgbm": 0.7,
                     "M5_catboost": 0.7, "M6_perceptron_multicapa": 0.4, "M7_nhits": 0.3, "M8_theta": 0.8, "M9_tsb": 0.8}
COMPATIBILIDAD = {"M0_ingenuo_estacional": 0.2, "M1_elastic_net": 0.5, "M2_glm_tweedie": 0.5, "M3_bosque_aleatorio": 0.6, "M4_lightgbm": 1.0,
                  "M5_catboost": 1.0, "M6_perceptron_multicapa": 0.5, "M7_nhits": 0.8, "M8_theta": 0.6, "M9_tsb": 0.3}


def registrar(modelo, corte, ev, pred, params, t_entrenamiento, t_inferencia, filas_entrenamiento, etiqueta="defecto"):
    pred = np.clip(np.asarray(pred, dtype=np.float64), 0, None)
    m = metricas(ev["y"], pred, ev["dow"], ev["h"])
    m_ing = metricas(ev["y"], ev["ingenuo_estacional"], ev["dow"], ev["h"])
    fila = {"modelo": modelo, "familia": FAMILIA.get(modelo, ""), "configuracion": etiqueta, "corte": corte, **m,
            "mae_relativo": m["mae"] / m_ing["mae"] if m_ing["mae"] else float("nan"),
            "rmse_relativo": m["rmse"] / m_ing["rmse"] if m_ing["rmse"] else float("nan"),
            "wmape_ingenuo": m_ing["wmape"], "t_entrenamiento_s": float(t_entrenamiento), "t_inferencia_s": float(t_inferencia),
            "filas_entrenamiento": int(filas_entrenamiento), "version": VERSION_TORNEO}
    RESULTADOS.append(fila)
    if etiqueta == "defecto":
        PRED.setdefault(modelo, pd.Series(np.nan, index=EVAL.index, dtype="float64"))
        PRED[modelo].loc[ev.index] = pred
    registrar_mlflow(f"{modelo}·{corte}·{etiqueta}", {"modelo": modelo, "corte": corte, "configuracion": etiqueta, **params},
                     {k: v for k, v in fila.items() if isinstance(v, (int, float))}, {"familia": FAMILIA.get(modelo, "")})
    print(f"   {modelo:28s} {corte:8s} {etiqueta:9s} WMAPE {m['wmape']:.4f} (ingenuo {m_ing['wmape']:.4f}) · sesgo {m['sesgo']:+.1%} · "
          f"entrena {t_entrenamiento / 60:.1f} min · infiere {t_inferencia:.0f} s · filas {filas_entrenamiento:,}")
    return m


# ---- preparación de variables (la misma para todos los modelos de cada familia)
BAJA_CARDINALIDAD = [c for c in COD_VARS if c != "cod_categoria" and len(ETIQUETAS[c.replace("cod_", "")]) <= 50]
ONEHOT = {c: sorted(ETIQUETAS[c.replace("cod_", "")].keys()) for c in BAJA_CARDINALIDAD}
ONEHOT["dow"] = list(range(1, 8))
ONEHOT["tipo_evento"] = [0, 1, 2, 3]
LINEAL_NUM = [c for c in NUM_VARS if c not in ("dow", "tipo_evento")]


def X_arbol(df, rellenar=None):
    """Matriz para modelos de árboles: variables numéricas (float32, con NaN) y códigos enteros de las categóricas."""
    X = df[X_VARS]
    return X.fillna(rellenar) if rellenar is not None else X


class PreparadorLineal:
    """log1p en variables sesgadas, imputación por mediana, estandarización y one-hot de las categóricas de baja cardinalidad.
    Se ajusta con el conjunto de entrenamiento del corte y se aplica igual a la evaluación."""
    def ajustar(self, df):
        A = df[LINEAL_NUM].astype("float32")
        self.log_cols = [c for c in LINEAL_NUM if float(np.nanmax(A[c].to_numpy())) > 50 and float(np.nanmin(A[c].to_numpy())) >= 0]
        A[self.log_cols] = np.log1p(A[self.log_cols].clip(lower=0))
        self.mediana = A.median().fillna(0.0)
        A = A.fillna(self.mediana)
        self.mu, self.sd = A.mean().fillna(0.0), A.std().replace(0, 1.0).fillna(1.0)
        return self
    def transformar(self, df):
        A = df[LINEAL_NUM].astype("float32")
        A[self.log_cols] = np.log1p(A[self.log_cols].clip(lower=0))
        A = ((A.fillna(self.mediana) - self.mu) / self.sd).to_numpy(dtype=np.float32)
        bloques = [A]
        for c, niveles in ONEHOT.items():
            codigos = df[c].to_numpy()
            bloques.append((codigos[:, None] == np.asarray(niveles)[None, :]).astype(np.float32))
        return np.concatenate(bloques, axis=1)
    def nombres(self):
        return LINEAL_NUM + [f"{c}={k}" for c, niveles in ONEHOT.items() for k in niveles]


def submuestra(df, tope):
    if tope is None or len(df) <= tope:
        return df
    return df.sample(n=int(tope), random_state=SEMILLA)


def factor_sesgo(y, pred_log):
    """Corrección al regresar de log(1 + y): escala para que el total pronosticado iguale al total observado en entrenamiento."""
    p = np.expm1(pred_log).clip(min=0).sum()
    return float(y.sum() / p) if p > 0 else 1.0


print(f"Variables para árboles: {len(X_VARS)} · para lineales y red: {len(PreparadorLineal().nombres())} (con one-hot)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Los modelos: nueve individuales de siete familias (más la referencia)
# MAGIC **Qué hace.** Define cada modelo como una función `entrenar_predecir(params, entrenamiento, evaluación)` y los corre
# MAGIC en los cuatro cortes: se entrena con las filas cuyo objetivo es anterior o igual al origen del corte y se predice el
# MAGIC bloque de 14 días. Los modelos lentos usan una submuestra aleatoria de filas de entrenamiento (`MAX_FILAS`); la
# MAGIC evaluación es siempre sobre todas las filas evaluables de la muestra común.
# MAGIC
# MAGIC | # | Modelo | Familia | Pérdida | Entrada |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | M0 | Ingenuo estacional | referencia | — | mismo día de la semana más reciente |
# MAGIC | M1 | Elastic Net | lineal | MSE sobre log(1 + y), con corrección de sesgo | matriz lineal |
# MAGIC | M2 | GLM Tweedie (enlace log) | lineal generalizado | devianza Tweedie | matriz lineal |
# MAGIC | M3 | Bosque aleatorio | árboles con bagging | MSE | matriz de árboles |
# MAGIC | M4 | LightGBM | boosting de gradiente | Tweedie | matriz de árboles (categóricas nativas) |
# MAGIC | M5 | CatBoost (alternativa XGBoost) | boosting ordenado | Tweedie | matriz de árboles (categóricas nativas) |
# MAGIC | M6 | Perceptrón multicapa | red neuronal tabular | MSE sobre log(1 + y), con corrección de sesgo | matriz lineal |
# MAGIC | M7 | N-HiTS (opcional) | red neuronal de series | MAE | serie diaria |
# MAGIC | M8 | Theta | estadístico clásico | verosimilitud | serie diaria |
# MAGIC | M9 | TSB | demanda intermitente | suavizamiento | serie diaria |

# COMMAND ----------

titulo("3. MODELOS")
DISPONIBLE = {}
try:
    import lightgbm as lgb
    DISPONIBLE["lightgbm"] = lgb.__version__
except Exception:
    lgb = None
try:
    import catboost as cb
    DISPONIBLE["catboost"] = cb.__version__
except Exception:
    cb = None
try:
    import xgboost as xgb
    DISPONIBLE["xgboost"] = xgb.__version__
except Exception:
    xgb = None
try:
    from statsforecast import StatsForecast
    from statsforecast.models import Theta, TSB, Naive
    import statsforecast
    DISPONIBLE["statsforecast"] = statsforecast.__version__
except Exception:
    StatsForecast = None
import sklearn
from sklearn.linear_model import ElasticNet, TweedieRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
DISPONIBLE["scikit-learn"] = sklearn.__version__
try:
    import optuna as _op
    DISPONIBLE["optuna"] = _op.__version__
except Exception:
    pass
try:
    import shap as _sh
    DISPONIBLE["shap"] = _sh.__version__
except Exception:
    pass
LIB = pd.DataFrame([{"libreria": k, "version": DISPONIBLE.get(k, "NO DISPONIBLE")}
                    for k in ("lightgbm", "catboost", "xgboost", "statsforecast", "optuna", "shap", "scikit-learn")])
mostrar(LIB, "Librerías disponibles")
faltan_libs = [k for k in ("lightgbm", "catboost", "statsforecast", "optuna") if k not in DISPONIBLE]
if faltan_libs:
    aviso(f"No se instalaron: {', '.join(faltan_libs)}. Revisa la salida de las celdas %pip (o agrégalas en el panel Environment). "
          "El notebook sigue con alternativas y lo indica en la columna `implementacion`.")
IMPLEMENTACION = {
    "M0_ingenuo_estacional": "mismo día de la semana más reciente",
    "M1_elastic_net": f"scikit-learn {sklearn.__version__} ElasticNet",
    "M2_glm_tweedie": f"scikit-learn {sklearn.__version__} TweedieRegressor",
    "M3_bosque_aleatorio": f"scikit-learn {sklearn.__version__} RandomForestRegressor",
    "M4_lightgbm": f"lightgbm {DISPONIBLE['lightgbm']}" if lgb is not None else "scikit-learn HistGradientBoosting (sustituto: lightgbm no disponible)",
    "M5_catboost": (f"catboost {DISPONIBLE['catboost']}" if cb is not None else
                    f"xgboost {DISPONIBLE['xgboost']} (sustituto: catboost no disponible)" if xgb is not None else "no disponible"),
    "M6_perceptron_multicapa": f"scikit-learn {sklearn.__version__} MLPRegressor",
    "M7_nhits": "neuralforecast",
    "M8_theta": (f"statsforecast {DISPONIBLE['statsforecast']} Theta" if StatsForecast is not None else
                 "numpy: método Theta estándar, implementación propia (statsforecast no disponible)"),
    "M9_tsb": (f"statsforecast {DISPONIBLE['statsforecast']} TSB" if StatsForecast is not None else
               "numpy: TSB, implementación propia (statsforecast no disponible)"),
}
if cb is None and xgb is None and INCLUIR.get("M5_catboost"):
    INCLUIR["M5_catboost"] = False
    aviso("M5 (boosting ordenado) no se corre: ni catboost ni xgboost están instalados.")

PARAMS_DEFECTO = {
    "M1_elastic_net": {"alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 300},
    "M2_glm_tweedie": {"power": 1.5, "alpha": 1e-3, "max_iter": 100},
    "M3_bosque_aleatorio": {"n_estimators": 100, "max_depth": 14, "min_samples_leaf": 50, "max_features": 0.33},
    "M4_lightgbm": {"objective": "tweedie", "tweedie_variance_power": 1.5, "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
                    "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0},
    "M5_catboost": {"loss_function": "Tweedie:variance_power=1.5", "iterations": 500, "learning_rate": 0.08, "depth": 8, "l2_leaf_reg": 3.0},
    "M6_perceptron_multicapa": {"hidden_layer_sizes": (256, 128), "learning_rate_init": 1e-3, "batch_size": 4096, "max_iter": 15, "alpha": 1e-4},
    "M7_nhits": {"input_size": 56, "max_steps": 300},
    "M8_theta": {"season_length": 7, "ventana": 112},
    "M9_tsb": {"alpha_d": 0.2, "alpha_p": 0.2},
}
if MODO_RAPIDO:
    PARAMS_DEFECTO["M4_lightgbm"]["n_estimators"] = 200
    PARAMS_DEFECTO["M5_catboost"]["iterations"] = 200
    PARAMS_DEFECTO["M6_perceptron_multicapa"]["max_iter"] = 8
    PARAMS_DEFECTO["M3_bosque_aleatorio"]["n_estimators"] = 40
IDX_CAT = [X_VARS.index(c) for c in COD_VARS]


def entrenar_predecir(modelo, params, tr, ev, prep_lineal=None):
    """Entrena `modelo` con `tr` y predice `ev`. Devuelve (predicción, segundos de entrenamiento, segundos de inferencia, filas usadas, objeto).
    El objeto devuelto incluye el tope de predicción (3 × percentil 99.9 de y en entrenamiento) para aplicarlo igual en cualquier inferencia."""
    tr = submuestra(tr, MAX_FILAS.get(modelo))
    y = tr["y"].to_numpy(dtype=np.float64)
    tope = float(np.quantile(y, 0.999) * TOPE_PREDICCION_FACTOR) if len(y) else np.inf
    t0 = time.perf_counter()
    if modelo in ("M1_elastic_net", "M2_glm_tweedie", "M6_perceptron_multicapa"):
        prep = prep_lineal or PreparadorLineal().ajustar(tr)
        Xtr, Xev = prep.transformar(tr), prep.transformar(ev)
        if modelo == "M1_elastic_net":
            est = ElasticNet(alpha=params["alpha"], l1_ratio=params["l1_ratio"], max_iter=params["max_iter"], random_state=SEMILLA).fit(Xtr, np.log1p(y))
            n = min(len(Xtr), 500_000)
            f = factor_sesgo(y[:n], est.predict(Xtr[:n]))
            t1 = time.perf_counter(); pred = np.expm1(est.predict(Xev)).clip(min=0) * f
        elif modelo == "M2_glm_tweedie":
            try:
                est = TweedieRegressor(power=params["power"], link="log", alpha=params["alpha"], max_iter=params["max_iter"],
                                       solver="newton-cholesky").fit(Xtr, y)
            except (TypeError, ValueError):
                est = TweedieRegressor(power=params["power"], link="log", alpha=params["alpha"], max_iter=params["max_iter"]).fit(Xtr, y)
            f = 1.0
            t1 = time.perf_counter(); pred = est.predict(Xev)
        else:
            est = MLPRegressor(hidden_layer_sizes=tuple(params["hidden_layer_sizes"]), learning_rate_init=params["learning_rate_init"],
                               batch_size=int(params["batch_size"]), max_iter=int(params["max_iter"]), alpha=params["alpha"],
                               early_stopping=True, validation_fraction=0.1, n_iter_no_change=4, random_state=SEMILLA).fit(Xtr, np.log1p(y))
            n = min(len(Xtr), 500_000)
            f = factor_sesgo(y[:n], est.predict(Xtr[:n]))
            t1 = time.perf_counter(); pred = np.expm1(est.predict(Xev)).clip(min=0) * f
        objeto = (est, prep, f, tope)
    elif modelo == "M3_bosque_aleatorio":
        Xtr, Xev = X_arbol(tr, rellenar=-1.0), X_arbol(ev, rellenar=-1.0)
        est = RandomForestRegressor(n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
                                    min_samples_leaf=int(params["min_samples_leaf"]), max_features=params["max_features"],
                                    n_jobs=N_HILOS, random_state=SEMILLA).fit(Xtr, y)
        t1 = time.perf_counter(); pred = est.predict(Xev); objeto = (est, tope)
    elif modelo == "M4_lightgbm":
        Xtr, Xev = X_arbol(tr), X_arbol(ev)
        if lgb is not None:
            est = lgb.LGBMRegressor(**params, n_jobs=N_HILOS, random_state=SEMILLA, verbose=-1)
            est.fit(Xtr, y, categorical_feature=IDX_CAT)
        else:
            est = HistGradientBoostingRegressor(loss="poisson", max_iter=int(params["n_estimators"]), learning_rate=params["learning_rate"],
                                                max_leaf_nodes=int(params["num_leaves"]), min_samples_leaf=int(params["min_child_samples"]),
                                                l2_regularization=params["reg_lambda"], random_state=SEMILLA).fit(Xtr, y)
        t1 = time.perf_counter(); pred = est.predict(Xev); objeto = (est, tope)
    elif modelo == "M5_catboost":
        if cb is not None:
            Xtr, Xev = X_arbol(tr), X_arbol(ev)
            est = cb.CatBoostRegressor(**params, cat_features=IDX_CAT, thread_count=N_HILOS, random_seed=SEMILLA, verbose=0, allow_writing_files=False)
            est.fit(Xtr, y)
        elif xgb is not None:
            Xtr, Xev = X_arbol(tr), X_arbol(ev)
            est = xgb.XGBRegressor(objective="reg:tweedie", tweedie_variance_power=1.5, n_estimators=int(params["iterations"]),
                                   learning_rate=params["learning_rate"], max_depth=int(params["depth"]), reg_lambda=params["l2_leaf_reg"],
                                   subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=N_HILOS, random_state=SEMILLA).fit(Xtr, y)
        else:
            raise ImportError("ni catboost ni xgboost disponibles")
        t1 = time.perf_counter(); pred = est.predict(Xev); objeto = (est, tope)
    else:
        raise ValueError(modelo)
    t2 = time.perf_counter()
    return np.clip(np.asarray(pred, dtype=np.float64), 0, tope), t1 - t0, t2 - t1, len(tr), objeto


def predecir_con(modelo, objeto, df):
    """Inferencia con un objeto devuelto por entrenar_predecir (misma preparación y mismo tope)."""
    if modelo in ("M1_elastic_net", "M2_glm_tweedie", "M6_perceptron_multicapa"):
        est, prep, f, tope = objeto
        Xev = prep.transformar(df)
        p = est.predict(Xev) if modelo == "M2_glm_tweedie" else np.expm1(est.predict(Xev)).clip(min=0) * f
    else:
        est, tope = objeto
        p = est.predict(X_arbol(df, rellenar=-1.0) if modelo == "M3_bosque_aleatorio" else X_arbol(df))
    return np.clip(np.asarray(p, dtype=np.float64), 0, tope)


# ---- modelos de series (statsforecast) sobre la muestra común: historia diaria de cada serie hasta el origen
HIST_SERIES = None
if (INCLUIR["M8_theta"] or INCLUIR["M9_tsb"] or INCLUIR["M7_nhits"]):
    P = leer_tabla(tabla("silver", "panel_categoria_dia"), "09_panel_diario")
    ser_eval = spark.createDataFrame(EVAL[CLAVES].drop_duplicates())
    hs = (P.join(ser_eval, CLAVES).select(*CLAVES, "fecha", "venta").toPandas().sort_values([*CLAVES, "fecha"]))
    hs["unique_id"] = hs["tienda"].astype(str) + "|" + hs["categoria_id"].astype(str)
    hs["ds"] = pd.to_datetime(hs["fecha"])
    # días sin dato (tienda inactiva o sin entrega): mismo día de la semana anterior; si no hay, la media de la serie; si no, 0
    g = hs.groupby("unique_id")["venta"]
    hs["y"] = hs["venta"].fillna(g.shift(7)).fillna(g.transform("mean")).fillna(0.0).astype("float64")
    HIST_SERIES = hs[["unique_id", "ds", "y"]]
    print(f"Historias para modelos de series: {HIST_SERIES['unique_id'].nunique():,} series, {len(HIST_SERIES):,} días")


def _matriz_historia(hist, origen, ventana):
    """Historia diaria en forma ancha (series × días) de los últimos `ventana` días hasta el origen; NaN antes del inicio de la serie."""
    h = hist[hist["ds"] > origen - pd.Timedelta(days=int(ventana))]
    W = h.pivot(index="unique_id", columns="ds", values="y")
    W = W.reindex(columns=pd.date_range(origen - pd.Timedelta(days=int(ventana) - 1), origen, freq="D"))
    return W


def theta_numpy(W, origen, alfas=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Método Theta estándar (Assimakopoulos y Nikolopoulos, 2000) vectorizado: índice estacional semanal multiplicativo,
    suavizamiento exponencial simple de la serie desestacionalizada (α elegido por serie con el error a un paso) y la mitad de
    la pendiente lineal como deriva. Devuelve una matriz series × 14."""
    Y = W.to_numpy(dtype=np.float64)
    dows = W.columns.dayofweek.to_numpy()
    media = np.nanmean(Y, axis=1, keepdims=True)
    idx = np.ones((Y.shape[0], 7))
    for d in range(7):
        col = np.nanmean(Y[:, dows == d], axis=1) if (dows == d).any() else np.full(Y.shape[0], np.nan)
        idx[:, d] = np.where(media[:, 0] > 0, col / np.where(media[:, 0] > 0, media[:, 0], 1.0), 1.0)
    idx = np.where(np.isfinite(idx) & (idx > 0.05), idx, 1.0)
    D = Y / idx[:, dows]
    n = np.sum(np.isfinite(D), axis=1)
    t = np.arange(D.shape[1], dtype=np.float64)[None, :]
    m = np.isfinite(D)
    tm = np.where(m, t, np.nan)
    b = np.nansum((tm - np.nanmean(tm, axis=1, keepdims=True)) * (D - np.nanmean(D, axis=1, keepdims=True)), axis=1) / \
        np.maximum(np.nansum((tm - np.nanmean(tm, axis=1, keepdims=True)) ** 2, axis=1), 1e-9)
    mejor_sse, mejor_l, mejor_a = np.full(D.shape[0], np.inf), np.zeros(D.shape[0]), np.full(D.shape[0], alfas[0])
    for a in alfas:
        l = np.full(D.shape[0], np.nan)
        sse = np.zeros(D.shape[0])
        for j in range(D.shape[1]):
            x = D[:, j]
            ok = np.isfinite(x)
            inicio = ok & np.isnan(l)
            l = np.where(inicio, x, l)
            seguir = ok & ~inicio
            sse = sse + np.where(seguir, (x - l) ** 2, 0.0)
            l = np.where(seguir, a * x + (1 - a) * l, l)
        mejor = sse < mejor_sse
        mejor_sse, mejor_l, mejor_a = np.where(mejor, sse, mejor_sse), np.where(mejor, l, mejor_l), np.where(mejor, a, mejor_a)
    hs = np.arange(1, 15)[None, :]
    deriva = 0.5 * b[:, None] * (hs - 1 + 1 / mejor_a[:, None] - (1 - mejor_a[:, None]) ** n[:, None] / mejor_a[:, None])
    dows_f = pd.date_range(origen + pd.Timedelta(days=1), periods=14, freq="D").dayofweek.to_numpy()
    F_ = (np.nan_to_num(mejor_l)[:, None] + deriva) * idx[:, dows_f]
    return np.clip(F_, 0, None)


def tsb_numpy(W, alpha_d, alpha_p):
    """TSB (Teunter, Syntetos y Babai, 2011) vectorizado: tamaño de la demanda z y probabilidad de demanda p, actualizada cada día;
    el pronóstico es p · z, igual para los 14 días."""
    Y = W.to_numpy(dtype=np.float64)
    pos = np.where(np.isfinite(Y), Y > 0, False)
    z = np.where(pos.any(axis=1), np.nansum(np.where(pos, Y, 0), axis=1) / np.maximum(pos.sum(axis=1), 1), 0.0)
    p = np.nanmean(np.where(np.isfinite(Y), pos.astype(float), np.nan), axis=1)
    z0, p0 = z.copy(), np.nan_to_num(p)
    z, p = z0.copy(), p0.copy()
    for j in range(Y.shape[1]):
        x = Y[:, j]
        ok = np.isfinite(x)
        z = np.where(ok & (x > 0), z + alpha_d * (x - z), z)
        p = np.where(ok, p + alpha_p * ((x > 0).astype(float) - p), p)
    return np.repeat(np.clip(p * z, 0, None)[:, None], 14, axis=1)


def pronostico_series(modelo, params, origen, ev):
    """Theta o TSB por serie (statsforecast si está instalado; si no, implementación propia en numpy). Las series cuya historia no
    llega al origen usan la referencia ingenua."""
    t0 = time.perf_counter()
    ids = (ev["tienda"].astype(str) + "|" + ev["categoria_id"].astype(str))
    hist = HIST_SERIES[(HIST_SERIES["ds"] <= origen) & HIST_SERIES["unique_id"].isin(set(ids))]
    ultimo = hist.groupby("unique_id")["ds"].max()
    completas = set(ultimo[ultimo == origen].index)
    hist = hist[hist["unique_id"].isin(completas)]
    ventana = int(params.get("ventana", 112))
    if StatsForecast is not None:
        hist = hist[hist["ds"] > origen - pd.Timedelta(days=ventana)]
        if modelo == "M8_theta":
            modelos_sf = [Theta(season_length=int(params["season_length"]))]
        else:
            modelos_sf = [TSB(alpha_d=float(params["alpha_d"]), alpha_p=float(params["alpha_p"]))]
        sf = StatsForecast(models=modelos_sf, freq="D", n_jobs=N_HILOS, fallback_model=Naive())
        fc = sf.forecast(df=hist, h=14)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        col = [c for c in fc.columns if c not in ("unique_id", "ds")][0]
        fc = fc[["unique_id", "ds", col]].rename(columns={col: "pred"})
    else:
        W = _matriz_historia(hist, origen, ventana if modelo == "M8_theta" else max(ventana, 168))
        M = theta_numpy(W, origen) if modelo == "M8_theta" else tsb_numpy(W, float(params["alpha_d"]), float(params["alpha_p"]))
        fechas = pd.date_range(origen + pd.Timedelta(days=1), periods=14, freq="D")
        fc = pd.DataFrame({"unique_id": np.repeat(W.index.to_numpy(), 14), "ds": np.tile(fechas.to_numpy(), len(W)), "pred": M.reshape(-1)})
    t1 = time.perf_counter()
    base = pd.DataFrame({"unique_id": ids.to_numpy(), "ds": ev["fecha_objetivo"].to_numpy()})
    pred = base.merge(fc, on=["unique_id", "ds"], how="left")["pred"].to_numpy(dtype=np.float64)
    faltan = int(np.isnan(pred).sum())
    pred = np.where(np.isnan(pred), ev["ingenuo_estacional"].to_numpy(dtype=np.float64), pred)
    return np.clip(pred, 0, None), t1 - t0, time.perf_counter() - t1, len(completas), faltan


# ---- torneo: cada modelo en cada corte
ACTIVOS = [m for m, on in INCLUIR.items() if on and m not in ("M7_nhits", "M8_theta", "M9_tsb")]
SERIES_ACTIVOS = [m for m in ("M8_theta", "M9_tsb") if INCLUIR[m]]
for corte in CORTES:
    origen = ORIGEN_DE[corte]
    ev = EVAL[EVAL["corte"] == corte]
    tr = TRAIN[TRAIN["fecha_objetivo"] <= origen]
    subtitulo(f"{corte} · origen {origen.date()} · entrenamiento {len(tr):,} filas (objetivo ≤ origen) · evaluación {len(ev):,} filas")
    registrar("M0_ingenuo_estacional", corte, ev, ev["ingenuo_estacional"].to_numpy(), {}, 0.0, 0.0, 0)
    prep = None
    for modelo in ACTIVOS:
        try:
            if modelo in ("M1_elastic_net", "M2_glm_tweedie", "M6_perceptron_multicapa") and prep is None:
                prep = PreparadorLineal().ajustar(submuestra(tr, MAX_FILAS[modelo]))
            pred, t_tr, t_inf, n, _ = entrenar_predecir(modelo, PARAMS_DEFECTO[modelo], tr, ev, prep_lineal=prep)
            registrar(modelo, corte, ev, pred, PARAMS_DEFECTO[modelo], t_tr, t_inf, n)
        except Exception as e:
            aviso(f"{modelo} no se pudo entrenar en {corte}: {type(e).__name__}: {str(e)[:140]}")
    for modelo in SERIES_ACTIVOS:
        try:
            pred, t_tr, t_inf, n, faltan = pronostico_series(modelo, PARAMS_DEFECTO[modelo], origen, ev)
            registrar(modelo, corte, ev, pred, PARAMS_DEFECTO[modelo], t_tr, t_inf, n)
            if faltan:
                print(f"      ({faltan:,} filas sin historia hasta el origen usaron la referencia ingenua)")
        except Exception as e:
            aviso(f"{modelo} no se pudo correr en {corte}: {type(e).__name__}: {str(e)[:140]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Modelo neuronal de series (N-HiTS, opcional)
# MAGIC Requiere `neuralforecast` (PyTorch). Se corre sólo si `INCLUIR["M7_nhits"] = True` y la librería está instalada;
# MAGIC si falla, la libreta lo dice y sigue. Entrena una red global sobre las series de la muestra común con la historia
# MAGIC hasta cada origen y pronostica los 14 días.

# COMMAND ----------

if INCLUIR["M7_nhits"]:
    try:
        from neuralforecast import NeuralForecast
        from neuralforecast.models import NHITS
        for corte in CORTES:
            origen = ORIGEN_DE[corte]
            ev = EVAL[EVAL["corte"] == corte]
            ids = (ev["tienda"].astype(str) + "|" + ev["categoria_id"].astype(str))
            hist = HIST_SERIES[(HIST_SERIES["ds"] <= origen) & HIST_SERIES["unique_id"].isin(set(ids))]
            ultimo = hist.groupby("unique_id")["ds"].max()
            hist = hist[hist["unique_id"].isin(set(ultimo[ultimo == origen].index))]
            t0 = time.perf_counter()
            nf = NeuralForecast(models=[NHITS(h=14, input_size=int(PARAMS_DEFECTO["M7_nhits"]["input_size"]),
                                              max_steps=int(PARAMS_DEFECTO["M7_nhits"]["max_steps"]), loss=None, random_seed=SEMILLA,
                                              scaler_type="robust", enable_progress_bar=False, logger=False)], freq="D")
            nf.fit(df=hist)
            t1 = time.perf_counter()
            fc = nf.predict().reset_index()
            fc = fc if "unique_id" in fc.columns else fc.reset_index()
            base = pd.DataFrame({"unique_id": ids.to_numpy(), "ds": ev["fecha_objetivo"].to_numpy()})
            pred = base.merge(fc[["unique_id", "ds", "NHITS"]], on=["unique_id", "ds"], how="left")["NHITS"].to_numpy(dtype=np.float64)
            pred = np.where(np.isnan(pred), ev["ingenuo_estacional"].to_numpy(dtype=np.float64), pred)
            registrar("M7_nhits", corte, ev, pred, PARAMS_DEFECTO["M7_nhits"], t1 - t0, time.perf_counter() - t1, hist["unique_id"].nunique())
    except Exception as e:
        aviso(f"M7_nhits no se pudo correr: {type(e).__name__}: {str(e)[:160]}. Se reporta como no disponible.")
else:
    print("M7_nhits apagado (INCLUIR['M7_nhits'] = False).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparativa
# MAGIC **Qué mirar.** La tabla se ordena por la métrica principal (WMAPE medio de los cuatro cortes, con su desviación) e
# MAGIC incluye sesgo, MAE y RMSE relativos a la referencia, WMAPE entre semana y en h = 8–14, tiempos de entrenamiento e
# MAGIC inferencia, filas usadas y la prueba de Diebold–Mariano contra la referencia. Los dos primeros (sin contar la
# MAGIC referencia) pasan al ajuste fino.

# COMMAND ----------

titulo("4. COMPARATIVA")
TORNEO = pd.DataFrame(RESULTADOS)
TORNEO["fecha_calculo"] = pd.Timestamp.now()
guardar(TORNEO, "mlops", "torneo_modelos", "Metricas de cada modelo en cada corte walk-forward, muestra comun (Avance 4)", modo="append")
T0 = TORNEO[TORNEO["configuracion"] == "defecto"]
COMP = (T0.groupby(["modelo", "familia"])
        .agg(wmape=("wmape", "mean"), wmape_desv=("wmape", "std"), sesgo=("sesgo", "mean"), mae_relativo=("mae_relativo", "mean"),
             rmse_relativo=("rmse_relativo", "mean"), wmape_entre_semana=("wmape_entre_semana", "mean"), wmape_h8_14=("wmape_h8_14", "mean"),
             cortes=("corte", "nunique"), t_entrenamiento_min=("t_entrenamiento_s", lambda s: s.sum() / 60),
             t_inferencia_s=("t_inferencia_s", "sum"), filas_entrenamiento=("filas_entrenamiento", "max"))
        .reset_index().sort_values("wmape").reset_index(drop=True))
wmape_ing_muestra = float(T0[T0["modelo"] == "M0_ingenuo_estacional"]["wmape"].mean())
COMP["mejora_vs_ingenuo"] = 1 - COMP["wmape"] / wmape_ing_muestra
COMP["implementacion"] = COMP["modelo"].map(IMPLEMENTACION)
# Diebold–Mariano contra la referencia (días de los cuatro cortes)
ev4 = EVAL[EVAL["corte"].isin(CORTES)]
e_ref = (ev4["y"] - ev4["ingenuo_estacional"]).to_numpy()
dm_rows = {}
for modelo, serie in PRED.items():
    p = serie.loc[ev4.index].to_numpy()
    ok = ~np.isnan(p)
    dm, pval = diebold_mariano((ev4["y"].to_numpy()[ok] - p[ok]), e_ref[ok], ev4["fecha_objetivo"].to_numpy()[ok])
    dm_rows[modelo] = (dm, pval)
COMP["dm_estadistico"] = COMP["modelo"].map(lambda m: dm_rows.get(m, (np.nan, np.nan))[0])
COMP["dm_valor_p"] = COMP["modelo"].map(lambda m: dm_rows.get(m, (np.nan, np.nan))[1])
mostrar(COMP[["modelo", "familia", "wmape", "wmape_desv", "mejora_vs_ingenuo", "sesgo", "mae_relativo", "rmse_relativo", "wmape_entre_semana",
              "wmape_h8_14", "dm_valor_p", "t_entrenamiento_min", "t_inferencia_s", "filas_entrenamiento", "cortes", "implementacion"]],
        "Comparativa de modelos individuales (muestra común, promedio de los cuatro cortes; ordenada por WMAPE)",
        formatos={"wmape": "dec4", "wmape_desv": "dec4", "mejora_vs_ingenuo": "pct", "sesgo": "pct", "mae_relativo": "dec3", "rmse_relativo": "dec3",
                  "wmape_entre_semana": "dec4", "wmape_h8_14": "dec4", "dm_valor_p": "dec3", "t_entrenamiento_min": "dec1",
                  "t_inferencia_s": "dec1", "filas_entrenamiento": "int", "cortes": "int"})
POR_CORTE = T0.pivot_table(index="modelo", columns="corte", values="wmape").reindex(COMP["modelo"]).reset_index()
mostrar(POR_CORTE, "WMAPE por corte", formatos={c: "dec4" for c in POR_CORTE.columns if c != "modelo"})
n_modelos = int((COMP["modelo"] != "M0_ingenuo_estacional").sum())
n_familias = len({FAMILIA_GRUESA.get(m, m) for m in COMP["modelo"] if m != "M0_ingenuo_estacional"})
mejor = COMP.iloc[0]
hallazgo(f"Torneo: {n_modelos} modelos individuales ({n_familias} familias de algoritmos) en {len(CORTES)} cortes. Mejor: {mejor['modelo']} con WMAPE "
         f"{mejor['wmape']:.4f} ± {mejor['wmape_desv']:.4f} ({mejor['mejora_vs_ingenuo']:+.1%} contra la referencia ingenua, {wmape_ing_muestra:.4f}; "
         f"Diebold–Mariano p = {mejor['dm_valor_p']:.3f}).")
peores = COMP[(COMP["modelo"] != "M0_ingenuo_estacional") & (COMP["wmape"] >= wmape_ing_muestra)]["modelo"].tolist()
if peores:
    hallazgo(f"No superan a la referencia ingenua en la muestra común: {', '.join(peores)}.")
FINALISTAS = COMP[COMP["modelo"] != "M0_ingenuo_estacional"]["modelo"].head(2).tolist()
if len(FINALISTAS) < 2:
    raise RuntimeError("Hacen falta al menos dos modelos entrenados para el ajuste fino; revisa los avisos de la sección 3.")
hallazgo(f"Finalistas para el ajuste fino: {FINALISTAS[0]} y {FINALISTAS[1]}.")

# error por horizonte y por día de la semana (muestra común) de los mejores cuatro y la referencia
def wmape_por(col, modelos):
    filas = []
    for m in modelos:
        p = PRED[m].loc[ev4.index].to_numpy()
        ok = ~np.isnan(p)
        d = pd.DataFrame({"g": ev4[col].to_numpy()[ok], "e": np.abs(ev4["y"].to_numpy()[ok] - p[ok]), "y": ev4["y"].to_numpy()[ok]})
        r = d.groupby("g").agg(e=("e", "sum"), y=("y", "sum"))
        filas.append((r["e"] / r["y"]).rename(m))
    return pd.concat(filas, axis=1)

top = ["M0_ingenuo_estacional"] + [m for m in COMP["modelo"] if m != "M0_ingenuo_estacional"][:4]
POR_H = wmape_por("h", top)
POR_DOW = wmape_por("dow", top)
POR_DOW.index = POR_DOW.index.map({1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"})
mostrar(POR_H.reset_index().rename(columns={"g": "h"}), "WMAPE por horizonte (muestra común)", formatos={m: "dec4" for m in top})
mostrar(POR_DOW.reset_index().rename(columns={"g": "dia"}), "WMAPE por día de la semana del día pronosticado (muestra común)", formatos={m: "dec4" for m in top})
fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
for m in top:
    axes[0].plot(POR_H.index, POR_H[m], marker="o", ms=3, lw=1.6 if m != "M0_ingenuo_estacional" else 2.2,
                 ls="--" if m == "M0_ingenuo_estacional" else "-", label=m)
    axes[1].plot(POR_DOW.index, POR_DOW[m], marker="o", ms=3, lw=1.6 if m != "M0_ingenuo_estacional" else 2.2,
                 ls="--" if m == "M0_ingenuo_estacional" else "-", label=m)
axes[0].set_title("WMAPE por horizonte"); axes[0].set_xlabel("h (días después del origen)"); axes[0].set_xticks(range(1, 15))
axes[1].set_title("WMAPE por día de la semana pronosticado")
for ax in axes:
    ax.grid(alpha=0.3); ax.set_ylabel("WMAPE")
axes[1].legend(fontsize=8, frameon=False)
plt.tight_layout(); plt.show()
# predicciones de la muestra común (para análisis posteriores; quedan en Databricks)
PRED_DF = EVAL[[*CLAVES, "origen", "h", "fecha_objetivo", "corte", "y", "ingenuo_estacional"]].copy()
for m, s in PRED.items():
    PRED_DF[m] = s
guardar(PRED_DF.assign(version=VERSION_TORNEO), "mlops", "torneo_predicciones_muestra",
        "Predicciones de cada modelo en la muestra comun de evaluacion, cuatro cortes (Avance 4)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ajuste fino de los dos finalistas (Optuna)
# MAGIC **Qué hace.** Para cada finalista, Optuna (muestreador TPE) explora un espacio amplio de hiperparámetros con
# MAGIC `N_PRUEBAS_AJUSTE` pruebas. La afinación usa un **origen de ajuste anterior al primer corte**: se entrena con las filas
# MAGIC cuyo objetivo es ≤ ese origen y se mide el WMAPE en sus 14 días siguientes, de modo que los cuatro cortes de prueba
# MAGIC se usan una sola vez, al final (Géron, «Fine-Tune Your Model»). Después, la configuración ganadora se evalúa en los
# MAGIC cuatro cortes y se compara con la configuración por defecto.

# COMMAND ----------

titulo("5. AJUSTE FINO")
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
except Exception as e:
    optuna = None
    aviso(f"optuna no disponible ({type(e).__name__}): el ajuste fino se hace con búsqueda aleatoria.")


def espacio(trial, modelo):
    """Espacio de búsqueda de cada modelo (amplio a propósito; la historia de la optimización muestra si se agotó)."""
    if modelo == "M4_lightgbm":
        return {"objective": "tweedie", "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
                "n_estimators": trial.suggest_int("n_estimators", 200, 1500, log=True), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True), "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1, "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0), "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
                "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255])}
    if modelo == "M5_catboost":
        return {"loss_function": f"Tweedie:variance_power={trial.suggest_float('variance_power', 1.1, 1.9):.3f}",
                "iterations": trial.suggest_int("iterations", 200, 1500, log=True), "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "depth": trial.suggest_int("depth", 4, 10), "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0), "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
                "one_hot_max_size": trial.suggest_int("one_hot_max_size", 2, 64, log=True)}
    if modelo == "M3_bosque_aleatorio":
        return {"n_estimators": trial.suggest_int("n_estimators", 50, 300, log=True), "max_depth": trial.suggest_int("max_depth", 6, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 20, 500, log=True),
                "max_features": trial.suggest_categorical("max_features", [0.2, 0.33, 0.5, 0.7])}
    if modelo == "M6_perceptron_multicapa":
        capas = trial.suggest_int("capas", 1, 4)
        unidades = trial.suggest_int("unidades", 64, 1024, log=True)
        return {"hidden_layer_sizes": tuple([unidades] * capas), "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
                "batch_size": trial.suggest_categorical("batch_size", [1024, 4096, 16384]), "max_iter": trial.suggest_int("max_iter", 10, 50),
                "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True)}
    if modelo == "M2_glm_tweedie":
        return {"power": trial.suggest_float("power", 1.1, 1.9), "alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True), "max_iter": 100}
    if modelo == "M1_elastic_net":
        return {"alpha": trial.suggest_float("alpha", 1e-5, 1.0, log=True), "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0), "max_iter": 300}
    if modelo == "M8_theta":
        return {"season_length": 7, "ventana": trial.suggest_categorical("ventana", [56, 84, 112, 140, 168])}
    if modelo == "M9_tsb":
        return {"alpha_d": trial.suggest_float("alpha_d", 0.02, 0.6), "alpha_p": trial.suggest_float("alpha_p", 0.02, 0.6)}
    raise ValueError(modelo)


ev_aj = EVAL[EVAL["corte"] == "ajuste"] if ORIGEN_AJUSTE is not None else EVAL[EVAL["corte"] == CORTES[0]]
origen_aj = ORIGEN_AJUSTE if ORIGEN_AJUSTE is not None else ORIGEN_DE[CORTES[0]]
tr_aj = submuestra(TRAIN[TRAIN["fecha_objetivo"] <= origen_aj], MAX_FILAS_AJUSTE)
print(f"Afinación: origen {origen_aj.date()} · entrenamiento {len(tr_aj):,} filas · validación {len(ev_aj):,} filas · "
      f"{N_PRUEBAS_AJUSTE} pruebas por finalista · tope {TIEMPO_MAX_AJUSTE_MIN} min")
PRUEBAS, MEJORES = [], {}


def correr_prueba(modelo, params):
    if modelo in ("M8_theta", "M9_tsb"):
        pred, *_ = pronostico_series(modelo, params, origen_aj, ev_aj)
    else:
        pred, *_ = entrenar_predecir(modelo, params, submuestra(tr_aj, TOPE_AJUSTE.get(modelo, MAX_FILAS_AJUSTE)), ev_aj)
    return metricas(ev_aj["y"], pred, ev_aj["dow"], ev_aj["h"])["wmape"]


for modelo in FINALISTAS:
    subtitulo(f"Optuna · {modelo}")
    base = correr_prueba(modelo, PARAMS_DEFECTO[modelo])
    print(f"   configuración por defecto en el origen de ajuste: WMAPE {base:.4f}")
    t_ini = time.time()

    def objetivo(trial, modelo=modelo):
        params = espacio(trial, modelo)
        t0 = time.perf_counter()
        try:
            valor = correr_prueba(modelo, params)
        except Exception as e:
            print(f"      prueba {trial.number} falló: {type(e).__name__}: {str(e)[:80]}")
            valor = float("nan")
        PRUEBAS.append({"modelo": modelo, "prueba": trial.number, "wmape": valor, "segundos": time.perf_counter() - t0,
                        "params": json.dumps({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in params.items()})})
        if not np.isfinite(valor):
            raise optuna.TrialPruned() if optuna is not None else ValueError("prueba inválida")
        return valor

    if optuna is not None:
        estudio = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEMILLA, n_startup_trials=min(8, N_PRUEBAS_AJUSTE // 2)),
                                      study_name=f"{VERSION_TORNEO}_{modelo}")
        estudio.optimize(objetivo, n_trials=N_PRUEBAS_AJUSTE, timeout=TIEMPO_MAX_AJUSTE_MIN * 60, gc_after_trial=True)
        completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if completas:
            mejor_t = estudio.best_trial
            MEJORES[modelo] = {"params": espacio(optuna.trial.FixedTrial(mejor_t.params), modelo), "wmape_ajuste": float(mejor_t.value),
                               "wmape_defecto_ajuste": base, "pruebas": len(completas), "minutos": (time.time() - t_ini) / 60}
            print(f"   {len(completas)} pruebas en {(time.time() - t_ini) / 60:.1f} min · mejor WMAPE {mejor_t.value:.4f} "
                  f"({mejor_t.value / base - 1:+.1%} contra la configuración por defecto)")
            try:
                imp = optuna.importance.get_param_importances(estudio)
                mostrar(pd.DataFrame({"hiperparametro": list(imp.keys()), "importancia": list(imp.values())}), f"Importancia de hiperparámetros · {modelo}",
                        formatos={"importancia": "pct"})
            except Exception as e:
                print(f"   (importancia de hiperparámetros no disponible: {type(e).__name__})")
            hist_opt = pd.DataFrame({"prueba": [t.number for t in completas], "wmape": [t.value for t in completas]})
            hist_opt["mejor_hasta_ahora"] = hist_opt["wmape"].cummin()
            fig, ax = plt.subplots(figsize=(7, 3.2))
            ax.scatter(hist_opt["prueba"], hist_opt["wmape"], s=14, alpha=0.6, label="prueba")
            ax.plot(hist_opt["prueba"], hist_opt["mejor_hasta_ahora"], color="#A08A5B", lw=2, label="mejor hasta ahora")
            ax.axhline(base, color="gray", ls="--", lw=1, label="por defecto")
            ax.set_title(f"Historia de la optimización · {modelo}"); ax.set_xlabel("prueba"); ax.set_ylabel("WMAPE (origen de ajuste)")
            ax.grid(alpha=0.3); ax.legend(fontsize=8, frameon=False); plt.tight_layout(); plt.show()
        else:
            aviso(f"Ninguna prueba de Optuna terminó para {modelo}; se conserva la configuración por defecto.")
    else:   # búsqueda aleatoria de respaldo
        rng = np.random.default_rng(SEMILLA)
        class _T:
            def __init__(s): s.number = len(PRUEBAS)
            def suggest_float(s, n, a, b, log=False): return float(np.exp(rng.uniform(np.log(a), np.log(b))) if log else rng.uniform(a, b))
            def suggest_int(s, n, a, b, log=False): return int(round(np.exp(rng.uniform(np.log(a), np.log(b))))) if log else int(rng.integers(a, b + 1))
            def suggest_categorical(s, n, ops): return ops[int(rng.integers(0, len(ops)))]
        mejor_v, mejor_p = base, PARAMS_DEFECTO[modelo]
        for _ in range(N_PRUEBAS_AJUSTE):
            t = _T(); params = espacio(t, modelo)
            try:
                v = correr_prueba(modelo, params)
            except Exception:
                continue
            PRUEBAS.append({"modelo": modelo, "prueba": t.number, "wmape": v, "segundos": 0.0, "params": json.dumps({k: str(x) for k, x in params.items()})})
            if v < mejor_v:
                mejor_v, mejor_p = v, params
        MEJORES[modelo] = {"params": mejor_p, "wmape_ajuste": mejor_v, "wmape_defecto_ajuste": base, "pruebas": N_PRUEBAS_AJUSTE, "minutos": (time.time() - t_ini) / 60}
    if modelo not in MEJORES:
        MEJORES[modelo] = {"params": PARAMS_DEFECTO[modelo], "wmape_ajuste": base, "wmape_defecto_ajuste": base, "pruebas": 0, "minutos": 0.0}
if PRUEBAS:
    guardar(pd.DataFrame(PRUEBAS).assign(version=VERSION_TORNEO, fecha_calculo=pd.Timestamp.now()), "mlops", "ajuste_fino_pruebas",
            "Pruebas de Optuna por finalista: hiperparametros y WMAPE en el origen de ajuste (Avance 4)", modo="append")
mostrar(pd.DataFrame([{"modelo": m, "pruebas": v["pruebas"], "minutos": v["minutos"], "wmape_defecto": v["wmape_defecto_ajuste"],
                       "wmape_ajustado": v["wmape_ajuste"], "mejora": 1 - v["wmape_ajuste"] / v["wmape_defecto_ajuste"]} for m, v in MEJORES.items()]),
        "Ajuste fino en el origen de ajuste", formatos={"pruebas": "int", "minutos": "dec1", "wmape_defecto": "dec4", "wmape_ajustado": "dec4", "mejora": "pct"})
for m, v in MEJORES.items():
    print(f"   {m}: " + ", ".join(f"{k}={round(x, 4) if isinstance(x, float) else x}" for k, x in v["params"].items()))

# ---- configuración ajustada en los cuatro cortes: muestra común y, con el mismo modelo entrenado, todas las series
SEGMENTOS = ["corte", "fecha_objetivo", "h", "dow", "cod_tipo_categoria", "cod_region", "dia_atipico_obj"]
ACUM = []            # sumas por segmento: cualquier métrica del conjunto completo sin guardar las predicciones
TIEMPO_INFERENCIA = {}


def acumular(chunk, preds):
    base = chunk[SEGMENTOS].copy()
    base["y"] = chunk["y"].to_numpy()
    base["n"] = 1
    for nombre, p in preds.items():
        p = np.clip(np.asarray(p, dtype=np.float64), 0, None)
        base[f"e_{nombre}"] = np.abs(chunk["y"].to_numpy() - p)
        base[f"p_{nombre}"] = p
        base[f"e2_{nombre}"] = (chunk["y"].to_numpy() - p) ** 2
    ACUM.append(base.groupby(SEGMENTOS, dropna=False).sum(numeric_only=True).reset_index())


def metricas_acumuladas(A, nombre, por=None):
    g = A.groupby(por).sum(numeric_only=True) if por else A.sum(numeric_only=True).to_frame().T
    return pd.DataFrame({"wmape": g[f"e_{nombre}"] / g["y"], "sesgo": g[f"p_{nombre}"] / g["y"] - 1,
                         "mae": g[f"e_{nombre}"] / g["n"], "rmse": np.sqrt(g[f"e2_{nombre}"] / g["n"]), "filas": g["n"]})


def evaluar_completo(corte, objetos):
    """Predice con los modelos ya entrenados del corte TODAS las series evaluables del corte, por bloques de un horizonte."""
    filtro = (F.col("conjunto") == "evaluacion") & (F.col("corte") == corte) & EVALUABLE
    if MODO_RAPIDO:
        filtro = filtro & en_rapido
    for h in range(1, 15):
        chunk = cargar(G.where(filtro & (F.col("h") == h)))
        if chunk.empty:
            continue
        preds = {"ingenuo": chunk["ingenuo_estacional"].to_numpy()}
        for modelo, obj in objetos.items():
            t0 = time.perf_counter()
            preds[f"{modelo}|ajustado"] = predecir_con(modelo, obj, chunk)
            TIEMPO_INFERENCIA[modelo] = TIEMPO_INFERENCIA.get(modelo, 0.0) + time.perf_counter() - t0
        acumular(chunk, preds)
        del chunk


subtitulo("Finalistas ajustados en los cuatro cortes (muestra común" + (" y todas las series)" if EVALUAR_CONJUNTO_COMPLETO else ")"))
if EVALUAR_CONJUNTO_COMPLETO and any(m in ("M8_theta", "M9_tsb") for m in FINALISTAS):
    print("   (los métodos por serie se evalúan sólo en la muestra común: en todas las series requerirían la historia diaria completa en memoria)")
t_total = time.time()
for corte in CORTES:
    origen = ORIGEN_DE[corte]
    ev = EVAL[EVAL["corte"] == corte]
    tr = TRAIN[TRAIN["fecha_objetivo"] <= origen]
    objetos = {}
    for modelo in FINALISTAS:
        try:
            if modelo in ("M8_theta", "M9_tsb"):
                pred, t_tr, t_inf, n, _ = pronostico_series(modelo, MEJORES[modelo]["params"], origen, ev)
            else:
                pred, t_tr, t_inf, n, obj = entrenar_predecir(modelo, MEJORES[modelo]["params"], tr, ev)
                objetos[modelo] = obj
            registrar(modelo, corte, ev, pred, MEJORES[modelo]["params"], t_tr, t_inf, n, etiqueta="ajustado")
        except Exception as e:
            aviso(f"{modelo} ajustado falló en {corte}: {type(e).__name__}: {str(e)[:120]}")
    if EVALUAR_CONJUNTO_COMPLETO:
        evaluar_completo(corte, objetos)
        print(f"   {corte}: todas las series evaluadas · {(time.time() - t_total) / 60:.1f} min acumulados")
    del objetos, tr
if TIEMPO_INFERENCIA:
    print("   Tiempo de inferencia en todas las series (cuatro cortes): " + ", ".join(f"{m} {t / 60:.1f} min" for m, t in TIEMPO_INFERENCIA.items()))
TORNEO = pd.DataFrame(RESULTADOS)
AJ = (TORNEO[TORNEO["modelo"].isin(FINALISTAS)].groupby(["modelo", "configuracion"])
      .agg(wmape=("wmape", "mean"), wmape_desv=("wmape", "std"), sesgo=("sesgo", "mean"), wmape_entre_semana=("wmape_entre_semana", "mean"),
           wmape_h8_14=("wmape_h8_14", "mean"), t_entrenamiento_min=("t_entrenamiento_s", lambda s: s.sum() / 60)).reset_index())
mostrar(AJ, "Por defecto contra ajustado, cuatro cortes (muestra común)",
        formatos={"wmape": "dec4", "wmape_desv": "dec4", "sesgo": "pct", "wmape_entre_semana": "dec4", "wmape_h8_14": "dec4", "t_entrenamiento_min": "dec1"})
for m in FINALISTAS:
    a = AJ[(AJ["modelo"] == m)].set_index("configuracion")["wmape"]
    if "ajustado" in a and "defecto" in a:
        hallazgo(f"Ajuste fino de {m}: WMAPE {a['defecto']:.4f} → {a['ajustado']:.4f} en los cuatro cortes ({1 - a['ajustado'] / a['defecto']:+.1%}); "
                 f"{MEJORES[m]['pruebas']} pruebas en {MEJORES[m]['minutos']:.0f} min.")
guardar(TORNEO[TORNEO["configuracion"] == "ajustado"].assign(fecha_calculo=pd.Timestamp.now()), "mlops", "torneo_modelos",
        "Metricas de cada modelo en cada corte walk-forward, muestra comun (Avance 4)", modo="append")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Modelo individual final
# MAGIC **Qué hace.** (a) Resume a los dos finalistas ajustados en **todas** las series (evaluadas en la sección 5 con el
# MAGIC mismo modelo de cada corte; es la cifra comparable con la línea base 0.553); (b) llena la matriz de decisión con pesos fijados de antemano: error 35 %, estabilidad 10 %, sesgo
# MAGIC 10 %, error donde más cuesta (entre semana y h = 8–14) 10 %, costo computacional 10 %, interpretabilidad 10 % y
# MAGIC compatibilidad con la siguiente etapa (cuantiles, registro, inferencia distribuida, tiendas nuevas) 15 %; (c) elige el
# MAGIC modelo, lo reentrena con todo el histórico, lo registra en MLflow (y en Unity Catalog si hay permiso) y explica sus
# MAGIC variables (SHAP o coeficientes) y su error por día de la semana, horizonte, tipo de categoría y región.

# COMMAND ----------

titulo("6. MODELO INDIVIDUAL FINAL")
if EVALUAR_CONJUNTO_COMPLETO and ACUM:
    A = pd.concat(ACUM, ignore_index=True)
    nombres_pred = sorted({c[2:] for c in A.columns if c.startswith("e_")})
    COMPLETO = []
    for nombre in nombres_pred:
        por_corte = metricas_acumuladas(A, nombre, ["corte"])
        tot = metricas_acumuladas(A, nombre)
        entre_semana = metricas_acumuladas(A[A["dow"] <= 5], nombre)
        h8 = metricas_acumuladas(A[A["h"] >= 8], nombre)
        COMPLETO.append({"prediccion": nombre, "wmape": float(por_corte["wmape"].mean()), "wmape_desv": float(por_corte["wmape"].std()),
                         "sesgo": float(por_corte["sesgo"].mean()), "wmape_entre_semana": float(entre_semana["wmape"].iloc[0]),
                         "wmape_h8_14": float(h8["wmape"].iloc[0]), "mae_relativo": float(tot["mae"].iloc[0] / metricas_acumuladas(A, "ingenuo")["mae"].iloc[0]),
                         "rmse_relativo": float(tot["rmse"].iloc[0] / metricas_acumuladas(A, "ingenuo")["rmse"].iloc[0]),
                         "filas": int(tot["filas"].iloc[0]), **{f"wmape_{c}": float(v) for c, v in por_corte["wmape"].items()}})
    COMPLETO = pd.DataFrame(COMPLETO).sort_values("wmape").reset_index(drop=True)
    wmape_ing_completo = float(COMPLETO.loc[COMPLETO["prediccion"] == "ingenuo", "wmape"].iloc[0])
    COMPLETO["mejora_vs_ingenuo"] = 1 - COMPLETO["wmape"] / wmape_ing_completo
    mostrar(COMPLETO[["prediccion", "wmape", "wmape_desv", "mejora_vs_ingenuo", "sesgo", "wmape_entre_semana", "wmape_h8_14", "mae_relativo", "rmse_relativo"]
                     + [f"wmape_{c}" for c in CORTES]],
            "Conjunto completo (todas las series relevantes): finalistas ajustados contra la referencia ingenua",
            formatos={c: "dec4" for c in COMPLETO.columns if c.startswith("wmape")} | {"mejora_vs_ingenuo": "pct", "sesgo": "pct", "mae_relativo": "dec3", "rmse_relativo": "dec3"})
    guardar(COMPLETO.assign(version=VERSION_TORNEO, fecha_calculo=pd.Timestamp.now()), "mlops", "evaluacion_completa",
            "Finalistas en todas las series relevantes, cuatro cortes (Avance 4)", modo="append")
    hallazgo(f"Conjunto completo: referencia ingenua WMAPE {wmape_ing_completo:.4f} (notebook 10: {WMAPE_INGENUO_TODAS:.4f}); " +
             "; ".join(f"{r.prediccion} {r.wmape:.4f} ({r.mejora_vs_ingenuo:+.1%})" for r in COMPLETO.itertuples() if r.prediccion != "ingenuo") + ".")
else:
    COMPLETO, A, wmape_ing_completo = None, None, float("nan")

# COMMAND ----------

# ---- matriz de decisión (pesos fijados antes de ver resultados)
subtitulo("Matriz de decisión")
def fila_decision(modelo):
    if COMPLETO is not None and f"{modelo}|ajustado" in set(COMPLETO["prediccion"]):
        r = COMPLETO.set_index("prediccion").loc[f"{modelo}|ajustado"]
        fuente = "todas las series"
    else:
        T1 = TORNEO[(TORNEO["modelo"] == modelo) & (TORNEO["configuracion"] == "ajustado")]
        r = pd.Series({"wmape": T1["wmape"].mean(), "wmape_desv": T1["wmape"].std(), "sesgo": T1["sesgo"].mean(),
                       "wmape_entre_semana": T1["wmape_entre_semana"].mean(), "wmape_h8_14": T1["wmape_h8_14"].mean()})
        fuente = "muestra común"
    T1 = TORNEO[(TORNEO["modelo"] == modelo) & (TORNEO["configuracion"] == "ajustado")]
    costo = float(T1["t_entrenamiento_s"].mean() + T1["t_inferencia_s"].mean())
    return {"modelo": modelo, "fuente": fuente, "error": float(r["wmape"]), "estabilidad": float(r["wmape_desv"]), "sesgo": abs(float(r["sesgo"])),
            "error_donde_cuesta": float((r["wmape_entre_semana"] + r["wmape_h8_14"]) / 2), "costo": costo,
            "interpretabilidad": INTERPRETABILIDAD.get(modelo, 0.5), "compatibilidad": COMPATIBILIDAD.get(modelo, 0.5)}

DEC = pd.DataFrame([fila_decision(m) for m in FINALISTAS])
MENOR_MEJOR = ["error", "estabilidad", "sesgo", "error_donde_cuesta", "costo"]
puntos = pd.DataFrame({"modelo": DEC["modelo"]})
for c in PESOS_DECISION:
    v = DEC[c].astype(float)
    if c in MENOR_MEJOR:
        s = (v.min() / v).where(v > 0, 1.0)          # el mejor vale 1; los demás, proporcionalmente
    else:
        s = v / v.max() if v.max() > 0 else pd.Series(1.0, index=v.index)
    puntos[c] = s.fillna(1.0) * PESOS_DECISION[c]
puntos["total"] = puntos[list(PESOS_DECISION)].sum(axis=1)
DEC = DEC.merge(puntos[["modelo", "total"]], on="modelo").sort_values("total", ascending=False).reset_index(drop=True)
mostrar(DEC, "Matriz de decisión: valores de cada criterio (métricas de la configuración ajustada) y puntuación ponderada",
        formatos={"error": "dec4", "estabilidad": "dec4", "sesgo": "pct", "error_donde_cuesta": "dec4", "costo": "dec1", "interpretabilidad": "dec2",
                  "compatibilidad": "dec2", "total": "dec3"})
mostrar(puntos.set_index("modelo").loc[DEC["modelo"]].reset_index(), "Puntos por criterio (peso × puntuación relativa)", formatos={c: "dec3" for c in puntos.columns if c != "modelo"})
# significancia entre finalistas (muestra común, configuración por defecto: misma base de comparación)
p1, p2 = PRED[FINALISTAS[0]].loc[ev4.index].to_numpy(), PRED[FINALISTAS[1]].loc[ev4.index].to_numpy()
ok = ~np.isnan(p1) & ~np.isnan(p2)
dm_f, p_f = diebold_mariano(ev4["y"].to_numpy()[ok] - p1[ok], ev4["y"].to_numpy()[ok] - p2[ok], ev4["fecha_objetivo"].to_numpy()[ok])
MODELO_FINAL = DEC.iloc[0]["modelo"]
if DEC["total"].iloc[0] - DEC["total"].iloc[1] < 0.02 and (np.isnan(p_f) or p_f > 0.05):
    # empate práctico y sin diferencia significativa: gana el más simple y barato
    MODELO_FINAL = DEC.sort_values(["costo", "interpretabilidad"], ascending=[True, False]).iloc[0]["modelo"]
    hallazgo(f"Los finalistas empatan en la matriz ({DEC['total'].iloc[0]:.3f} contra {DEC['total'].iloc[1]:.3f}) y la prueba de Diebold–Mariano entre "
             f"ellos no es significativa (p = {p_f:.3f}): se elige el más simple y barato, {MODELO_FINAL}.")
else:
    hallazgo(f"Modelo individual final: {MODELO_FINAL} (puntuación {DEC['total'].iloc[0]:.3f} contra {DEC['total'].iloc[1]:.3f}; "
             f"Diebold–Mariano entre finalistas p = {p_f:.3f}).")
RETADOR = [m for m in FINALISTAS if m != MODELO_FINAL][0]
guardar(DEC.assign(modelo_final=MODELO_FINAL, retador=RETADOR, dm_valor_p=p_f, version=VERSION_TORNEO, fecha_calculo=pd.Timestamp.now()),
        "mlops", "decision_modelo_final", "Matriz de decision del modelo individual final (Avance 4)", modo="append")

# COMMAND ----------

# ---- reentrenamiento final con todo el histórico, registro y explicación
subtitulo(f"Reentrenamiento final de {MODELO_FINAL} con todo el histórico de entrenamiento")
params_final = MEJORES[MODELO_FINAL]["params"]
ultimo_origen = max(ORIGEN_DE[c] for c in CORTES)
ev_ult = EVAL[EVAL["corte"] == CORTES[-1]]
RUN_FINAL, OBJ_FINAL = None, None
if MODELO_FINAL in ("M8_theta", "M9_tsb"):
    print("   El modelo final es un método por serie: no hay un objeto único que registrar; se registran sus parámetros.")
    RUN_FINAL = registrar_mlflow(f"{MODELO_FINAL}·final", {"modelo": MODELO_FINAL, **params_final}, {"wmape_muestra_comun": float(AJ[(AJ['modelo'] == MODELO_FINAL) & (AJ['configuracion'] == 'ajustado')]['wmape'].iloc[0])},
                                 {"etapa": "modelo_final"})
else:
    tr_all = TRAIN                      # todo el histórico disponible: es el modelo que pasa a la siguiente etapa
    pred_f, t_tr, t_inf, n_f, OBJ_FINAL = entrenar_predecir(MODELO_FINAL, params_final, tr_all, ev_ult)
    print(f"   entrenado con {n_f:,} filas en {t_tr / 60:.1f} min (variables: {len(X_VARS)})")
    if MLFLOW_OK:
        try:
            with mlflow.start_run(run_name=f"{MODELO_FINAL}·final") as run:
                mlflow.log_params({k: str(v)[:250] for k, v in params_final.items()})
                mlflow.set_tags({"etapa": "modelo_final", "version_torneo": VERSION_TORNEO, "familia": FAMILIA[MODELO_FINAL], "retador": RETADOR})
                if COMPLETO is not None:
                    r = COMPLETO.set_index("prediccion").loc[f"{MODELO_FINAL}|ajustado"]
                    mlflow.log_metrics({"wmape_todas_las_series": float(r["wmape"]), "sesgo": float(r["sesgo"]), "wmape_ingenuo": wmape_ing_completo})
                est = OBJ_FINAL[0]
                ejemplo = X_arbol(ev_ult.head(5)) if len(OBJ_FINAL) == 2 else None
                if MODELO_FINAL == "M4_lightgbm" and lgb is not None:
                    import mlflow.lightgbm
                    mlflow.lightgbm.log_model(est, "modelo", input_example=ejemplo)
                elif MODELO_FINAL == "M5_catboost" and cb is not None:
                    import mlflow.catboost
                    mlflow.catboost.log_model(est, "modelo", input_example=ejemplo)
                else:
                    import mlflow.sklearn
                    mlflow.sklearn.log_model(est, "modelo")
                RUN_FINAL = run.info.run_id
            print(f"   modelo registrado en MLflow (run {RUN_FINAL})")
            try:
                mlflow.set_registry_uri("databricks-uc")
                nombre_uc = f"{CATALOGO}.{ESQUEMA_DATOS}.{NOMBRE_MODELO_UC}"
                version = mlflow.register_model(f"runs:/{RUN_FINAL}/modelo", nombre_uc)
                from mlflow import MlflowClient
                MlflowClient().set_registered_model_alias(nombre_uc, "challenger", version.version)
                hallazgo(f"Modelo final registrado en Unity Catalog como {nombre_uc} versión {version.version} con alias @challenger.")
            except Exception as e:
                print(f"   (registro en Unity Catalog no disponible: {type(e).__name__}: {str(e)[:120]}; el modelo queda en el run de MLflow)")
        except Exception as e:
            print(f"   (no se pudo registrar el modelo en MLflow: {type(e).__name__}: {str(e)[:120]})")

# ---- explicación: SHAP (árboles), coeficientes (lineales) o importancia por permutación (red)
subtitulo("Qué variables usa el modelo final")
IMPORTANCIA = None
try:
    muestra_x = ev_ult.sample(n=min(5_000 if MODELO_FINAL == "M3_bosque_aleatorio" else 20_000, len(ev_ult)), random_state=SEMILLA)
    if MODELO_FINAL in ("M4_lightgbm", "M5_catboost", "M3_bosque_aleatorio") and OBJ_FINAL is not None:
        import shap
        Xs = X_arbol(muestra_x, rellenar=-1.0) if MODELO_FINAL == "M3_bosque_aleatorio" else X_arbol(muestra_x)
        explicador = shap.TreeExplainer(OBJ_FINAL[0])
        valores = explicador.shap_values(Xs)
        imp = np.abs(np.asarray(valores)).mean(axis=0)
        IMPORTANCIA = pd.DataFrame({"variable": X_VARS, "importancia_shap_media": imp})
        IMPORTANCIA["participacion"] = IMPORTANCIA["importancia_shap_media"] / IMPORTANCIA["importancia_shap_media"].sum()
        metodo = f"SHAP (TreeExplainer, media de |valor SHAP| en {len(Xs):,} filas)"
    elif MODELO_FINAL in ("M1_elastic_net", "M2_glm_tweedie") and OBJ_FINAL is not None:
        est, prep = OBJ_FINAL[0], OBJ_FINAL[1]
        IMPORTANCIA = pd.DataFrame({"variable": prep.nombres(), "coeficiente_estandarizado": est.coef_})
        IMPORTANCIA["importancia_shap_media"] = IMPORTANCIA["coeficiente_estandarizado"].abs()
        IMPORTANCIA["participacion"] = IMPORTANCIA["importancia_shap_media"] / IMPORTANCIA["importancia_shap_media"].sum()
        metodo = "coeficientes sobre variables estandarizadas"
    elif MODELO_FINAL == "M6_perceptron_multicapa" and OBJ_FINAL is not None:
        from sklearn.inspection import permutation_importance
        est, prep = OBJ_FINAL[0], OBJ_FINAL[1]
        Xs = prep.transformar(muestra_x)
        pi = permutation_importance(est, Xs, np.log1p(muestra_x["y"].to_numpy()), n_repeats=3, random_state=SEMILLA, n_jobs=N_HILOS)
        IMPORTANCIA = pd.DataFrame({"variable": prep.nombres(), "importancia_shap_media": pi.importances_mean})
        IMPORTANCIA["participacion"] = IMPORTANCIA["importancia_shap_media"].clip(lower=0) / IMPORTANCIA["importancia_shap_media"].clip(lower=0).sum()
        metodo = "importancia por permutación (3 repeticiones, 20 mil filas)"
    else:
        metodo = None
    if IMPORTANCIA is not None:
        IMPORTANCIA = IMPORTANCIA.sort_values("importancia_shap_media", ascending=False).reset_index(drop=True)
        mostrar(IMPORTANCIA.head(15)[["variable", "participacion"]], f"Las 15 variables con más peso · {metodo}", formatos={"participacion": "pct"})
        fig, ax = plt.subplots(figsize=(7, 4.5))
        topv = IMPORTANCIA.head(15).iloc[::-1]
        ax.barh(topv["variable"], topv["participacion"], color="#A08A5B")
        ax.set_title(f"Variables con más peso en {MODELO_FINAL}"); ax.set_xlabel("participación en la importancia total"); ax.grid(axis="x", alpha=0.3)
        plt.tight_layout(); plt.show()
        grupos = {"calendario": ["dow", "fin_de_semana", "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento", "dias_a_evento", "dias_desde_evento",
                                 "en_mundial", "h", "h_bloque"], "historia de la serie": [c for c in X_VARS if c.startswith("s_") or c in ("dias_con_venta_hist", "dias_historia")],
                  "contexto (tienda, zona, país)": [c for c in X_VARS if c.startswith(("t_", "zc_", "pc_"))],
                  "atributos": [c for c in X_VARS if c.startswith("cod_") or c == "antiguedad_tienda"]}
        por_grupo = {g: float(IMPORTANCIA[IMPORTANCIA["variable"].str.split("=").str[0].isin(cols)]["participacion"].sum()) for g, cols in grupos.items()}
        hallazgo(f"Variables del modelo final ({metodo}): " + ", ".join(f"{g} {v:.0%}" for g, v in sorted(por_grupo.items(), key=lambda x: -x[1])) +
                 f". Las tres con más peso: {', '.join(IMPORTANCIA['variable'].head(3))}.")
    else:
        print("   El modelo final no usa variables (método por serie): no aplica SHAP.")
except Exception as e:
    print(f"   (explicación no disponible: {type(e).__name__}: {str(e)[:120]})")

# ---- análisis de error del modelo final (conjunto completo si existe; si no, muestra común)
subtitulo("Dónde acierta y dónde falla el modelo final (contra la referencia ingenua)")
def tabla_error(por, etiquetas=None):
    nombre = f"{MODELO_FINAL}|ajustado"
    if COMPLETO is not None and nombre in set(COMPLETO["prediccion"]):
        m = metricas_acumuladas(A, nombre, por)[["wmape", "sesgo", "filas"]]
        i = metricas_acumuladas(A, "ingenuo", por)[["wmape"]].rename(columns={"wmape": "wmape_ingenuo"})
        t = m.join(i)
    else:
        p = PRED[MODELO_FINAL].loc[ev4.index].to_numpy()
        ok = ~np.isnan(p)
        d = pd.DataFrame({**{c: ev4[c].to_numpy()[ok] for c in por}, "e": np.abs(ev4["y"].to_numpy()[ok] - p[ok]),
                          "ei": np.abs(ev4["y"].to_numpy()[ok] - ev4["ingenuo_estacional"].to_numpy()[ok]), "y": ev4["y"].to_numpy()[ok], "p": p[ok]})
        g = d.groupby(por).sum()
        t = pd.DataFrame({"wmape": g["e"] / g["y"], "sesgo": g["p"] / g["y"] - 1, "filas": d.groupby(por).size(), "wmape_ingenuo": g["ei"] / g["y"]})
    t["mejora"] = 1 - t["wmape"] / t["wmape_ingenuo"]
    t = t.reset_index()
    if etiquetas:
        for c, mapa in etiquetas.items():
            t[c] = t[c].map(mapa)
    return t

mostrar(tabla_error(["h"]), "Por horizonte", formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "sesgo": "pct", "mejora": "pct", "filas": "int"})
mostrar(tabla_error(["dow"], {"dow": {1: "lun", 2: "mar", 3: "mié", 4: "jue", 5: "vie", 6: "sáb", 7: "dom"}}), "Por día de la semana",
        formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "sesgo": "pct", "mejora": "pct", "filas": "int"})
mostrar(tabla_error(["cod_tipo_categoria"], {"cod_tipo_categoria": ETIQUETAS["tipo_categoria"]}), "Por tipo de categoría",
        formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "sesgo": "pct", "mejora": "pct", "filas": "int"})
TE_REG = tabla_error(["cod_region"], {"cod_region": ETIQUETAS["region"]}).sort_values("filas", ascending=False)
mostrar(TE_REG, "Por región", formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "sesgo": "pct", "mejora": "pct", "filas": "int"})
mostrar(tabla_error(["dia_atipico_obj"]), "Días atípicos (1) contra normales (0)", formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "sesgo": "pct", "mejora": "pct", "filas": "int"})
te_dow = tabla_error(["dow"]).set_index("dow")
hallazgo(f"El modelo final mejora a la referencia {te_dow.loc[te_dow.index <= 5, 'mejora'].mean():+.1%} entre semana y "
         f"{te_dow.loc[te_dow.index >= 6, 'mejora'].mean():+.1%} en fin de semana; por horizonte, "
         f"{tabla_error(['h']).set_index('h').loc[1:7, 'mejora'].mean():+.1%} en h = 1–7 y {tabla_error(['h']).set_index('h').loc[8:14, 'mejora'].mean():+.1%} en h = 8–14.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Resumen para el reporte y siguientes pasos
# MAGIC Lo que pasa a la siguiente etapa: el modelo final como campeón provisional (`@challenger` hasta que exista una
# MAGIC compuerta), el segundo finalista como retador, los cuantiles P10/P90 con el mismo pipeline, el registro con
# MAGIC compuerta y los jobs de reentrenamiento e inferencia.

# COMMAND ----------

titulo("RESUMEN PARA EL REPORTE (sólo métricas; sin datos de negocio)")
print(f"Modelos individuales comparados: {n_modelos} de {n_familias} familias · cortes: {len(CORTES)} · muestra común: {n_series_eval:,} series")
print(f"Referencia ingenua: WMAPE {wmape_ing_muestra:.4f} (muestra común) · {WMAPE_INGENUO_TODAS:.4f} (todas las series, notebook 10)")
print("\nComparativa (muestra común, promedio de cuatro cortes):")
print(COMP[["modelo", "familia", "wmape", "wmape_desv", "mejora_vs_ingenuo", "sesgo", "mae_relativo", "rmse_relativo", "wmape_entre_semana", "wmape_h8_14",
            "dm_valor_p", "t_entrenamiento_min", "t_inferencia_s", "filas_entrenamiento", "implementacion"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nFinalistas: {FINALISTAS} · pruebas de Optuna: " + ", ".join(f"{m} {v['pruebas']} ({v['minutos']:.0f} min)" for m, v in MEJORES.items()))
print(AJ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if COMPLETO is not None:
    print("\nTodas las series relevantes (finalistas ajustados contra la referencia):")
    print(COMPLETO[["prediccion", "wmape", "wmape_desv", "mejora_vs_ingenuo", "sesgo", "wmape_entre_semana", "wmape_h8_14"] + [f"wmape_{c}" for c in CORTES]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\nModelo individual final: {MODELO_FINAL} · retador: {RETADOR} · DM entre finalistas p = {p_f:.3f}")
print(DEC.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("Hiperparámetros finales: " + ", ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in params_final.items()))
if IMPORTANCIA is not None:
    print("Variables con más peso: " + ", ".join(f"{r.variable} {r.participacion:.1%}" for r in IMPORTANCIA.head(10).itertuples()))
cerrar("21_avance4_modelos_alternativos", siguiente="22_cuantiles_y_compuerta (fase 2)")
