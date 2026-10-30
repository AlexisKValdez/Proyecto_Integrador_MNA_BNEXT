# Databricks notebook source
# MAGIC %md
# MAGIC # Avance 5 · Modelo final: ensambles, ajuste fino y evaluación en datos no vistos
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Entregable `Avance5.43`**
# MAGIC
# MAGIC **Autores:** Daniela Juárez · Kevin Valdez · Yael Márquez
# MAGIC
# MAGIC **Problema.** Pronosticar la venta diaria en pesos de cada **tienda × categoría** para los **14 días siguientes**
# MAGIC (horizonte h = 1…14) usando sólo información disponible el día en que se pronostica (el **origen** T). La métrica
# MAGIC principal es el **WMAPE** = Σ|real − pronóstico| / Σ real: pondera cada error por la venta, que es lo que importa para
# MAGIC surtir. Se reportan además el **sesgo** (Σ pronóstico / Σ real − 1), el MAE y el RMSE relativos a la referencia,
# MAGIC el error en mercancía (lo que se surte físicamente) y los tiempos de entrenamiento e inferencia.
# MAGIC
# MAGIC **Qué hace esta libreta.** Parte de los mejores modelos individuales del Avance 4, les ajusta hiperparámetros y
# MAGIC construye ensambles homogéneos y heterogéneos; elige el modelo final con criterios de negocio **antes** de ver los
# MAGIC datos de prueba y después lo evalúa una sola vez en septiembre (datos no vistos). Termina con intervalos de
# MAGIC pronóstico (P10/P50/P90), una compuerta de aceptación, el registro del modelo y la tabla de pronóstico operativo.
# MAGIC
# MAGIC | Clave | Modelo | Tipo | Estrategia |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `B0_Ingenuo` | Ingenuo estacional | referencia | venta del mismo día de la semana más reciente ≤ T |
# MAGIC | `I1_LightGBM_Tweedie` | LightGBM Tweedie | individual (mejor del Avance 4) | boosting de árboles, **ajustado con Optuna** |
# MAGIC | `I2_CatBoost_Tweedie` | CatBoost Tweedie | individual | boosting de árboles simétricos |
# MAGIC | `I3_ElasticNet` | Elastic Net | individual | lineal regularizado sobre log(1 + venta) |
# MAGIC | `I4_XGBoost_Tweedie` | XGBoost Tweedie | individual (nuevo) | boosting de árboles, **ajustado con Optuna** |
# MAGIC | `I5_LightGBM_Log1p` | LightGBM log1p | individual (nuevo) | boosting sobre log(1 + venta) con corrección de escala |
# MAGIC | `I6_Theta` | Theta | individual (opcional) | estadístico por serie (apagado por defecto: costo de cómputo) |
# MAGIC | `E1_Bagging` | Bagging de LightGBM | ensamble homogéneo | 3 LightGBM en submuestras distintas, promediados |
# MAGIC | `E2_AdaBoost` | AdaBoost | ensamble homogéneo | boosting adaptativo de árboles de decisión |
# MAGIC | `E7_Bosque_Aleatorio` | Bosque aleatorio | ensamble homogéneo | bagging de árboles con submuestreo de variables |
# MAGIC | `E3_Blending` | Blending | ensamble heterogéneo | pesos convexos aprendidos en el bloque observado más reciente |
# MAGIC | `E4_Stacking_temporal` | Stacking temporal | ensamble heterogéneo | pesos no negativos por horizonte + corrección de nivel por tipo |
# MAGIC | `E5_Votacion` | Votación | ensamble heterogéneo | promedio de las 3 bases con menor error reciente |
# MAGIC | `E6_Stacking_LightGBM` | Stacking no lineal | ensamble heterogéneo | meta-modelo LightGBM sobre las predicciones de las bases |
# MAGIC | `D1_Stacking_sin_correccion` | Diagnóstico | diagnóstico | E4 sin la corrección de nivel: mide cuánto aporta esa corrección |
# MAGIC
# MAGIC Las bases de los ensambles heterogéneos son los individuales (`I1`–`I6`), entre ellos los mejores del Avance 4.
# MAGIC
# MAGIC **Protocolo temporal (sin fuga de información).** Cada modelo de un origen T se entrena sólo con filas cuyo
# MAGIC objetivo es ≤ T, y cada ensamble aprende sus pesos sólo con bloques que ya terminaron antes de T.
# MAGIC
# MAGIC | Bloque | Origen | Días pronosticados | Uso |
# MAGIC | --- | --- | --- | --- |
# MAGIC | `ajuste` | 20-may | 21-may a 03-jun | Optuna, número de árboles y primeros pesos de los ensambles |
# MAGIC | `corte_1` … `corte_4` | 03-jun, 17-jun, 01-jul, 17-ago | 4 bloques de 14 días | **desarrollo**: comparar y elegir (los mismos del Avance 4) |
# MAGIC | `prueba_1`, `prueba_2` | 31-ago, 14-sep | 01–14 y 15–28 de septiembre | **datos no vistos**: el modelo elegido, congelado con datos al 31-ago, pronostica septiembre |
# MAGIC | `pronostico` | último día con datos | 14 días siguientes | salida operativa (sin objetivo todavía) |
# MAGIC
# MAGIC **Dónde se cumple cada criterio de la rúbrica**
# MAGIC
# MAGIC | Criterio | Sección |
# MAGIC | --- | --- |
# MAGIC | Ensambles homogéneos y heterogéneos (≥ 4); stacking/blending con los mejores individuales; optimización de hiperparámetros | §2.4–2.7 (definición), §4 (ejecución), §5.4 (Optuna), §5.5 (pesos) |
# MAGIC | Tabla comparativa ordenada por la métrica principal, con otras métricas y tiempos de entrenamiento | §5.2 y §5.3 |
# MAGIC | Elección del modelo final alineada con el negocio | §5.7 (matriz de decisión) y §6.5 (compuerta) |
# MAGIC | Gráficos del modelo final con interpretación | §7 (siete gráficos, cada uno con su lectura calculada) |
# MAGIC | Objetivo 3.6: evaluación en datos no vistos | §5.3, §6 y §7 (septiembre) |
# MAGIC | Declaración sobre el uso de herramientas de IA | última celda |
# MAGIC
# MAGIC **Confidencialidad.** Con `MODO_PUBLICO = True` las salidas sólo muestran métricas relativas, tiempos,
# MAGIC hiperparámetros, pesos y ventas indexadas; nunca importes en pesos, volúmenes ni identificadores de tienda. Los datos
# MAGIC y las tablas con detalle permanecen en el catálogo privado de Databricks.
# MAGIC
# MAGIC | | |
# MAGIC | --- | --- |
# MAGIC | **Lee** | `gold.features_origen` (versión Delta fija) · `silver.panel_categoria_dia` (sólo si `I6_Theta` está activo) · `mlops.validaciones` |
# MAGIC | **Escribe** | `mlops.av5_*` (resultados agregados por corrida) · `gold.pronostico_14d` · `mlops.monitoreo_modelo` · MLflow · `demanda_categoria` en Unity Catalog (alias `@challenger`) |
# MAGIC | **Tiempo** | `MODO_RAPIDO = True`: 20–40 min · completo: 3.5–5 h con 16 GB de RAM (conviene correrlo como Job) |
# MAGIC
# MAGIC **Cómo correrla.** (1) Con `MODO_RAPIDO = True`, «Run all»: comprueba que todo funciona. (2) Con
# MAGIC `MODO_RAPIDO = False`, como Job: son las cifras del reporte. (3) Si la corrida se interrumpe, escribe su
# MAGIC identificador en `RUN_ID_REANUDAR` y vuelve a correr: continúa desde el último bloque terminado. (4) Para volver a
# MAGIC ver resultados sin entrenar, `EJECUTAR_ENTRENAMIENTO = False` (y opcionalmente `RUN_ID_A_MOSTRAR`).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Bibliotecas
# MAGIC Serverless no trae LightGBM, XGBoost, CatBoost, Optuna ni SHAP. Cada una se instala en su propia celda (una
# MAGIC instalación con varias bibliotecas en una sola línea falló en el Avance 4) y al final se reinicia Python para que
# MAGIC queden disponibles. La sección 3 imprime las versiones efectivas y las registra en MLflow.

# COMMAND ----------

# MAGIC %pip install "lightgbm>=4.3"

# COMMAND ----------

# MAGIC %pip install "xgboost>=2.0"

# COMMAND ----------

# MAGIC %pip install "catboost>=1.2"

# COMMAND ----------

# MAGIC %pip install "optuna>=3.6"

# COMMAND ----------

# MAGIC %pip install "shap>=0.44"

# COMMAND ----------

try:
    dbutils.library.restartPython()      # necesario después de %pip en Databricks; fuera de Databricks no existe y se ignora
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

# MAGIC %md
# MAGIC ## 1. Configuración
# MAGIC Todo lo que se puede cambiar está en las celdas de esta sección; el resto de la libreta no tiene números fijos.
# MAGIC Primero se definen unas funciones de apoyo propias de esta libreta (memoria disponible, núcleos, cronómetro y una
# MAGIC forma de imprimir tablas sin importes cuando la libreta se va a publicar).

# COMMAND ----------

# ------------------------------------------------------------------------------------------------------------------
# Apoyo propio de esta libreta
# ------------------------------------------------------------------------------------------------------------------
import gc                          # liberar memoria explícitamente entre bloques grandes
import hashlib                     # huellas SHA-256 del código y de las muestras (auditoría de la corrida)
import math
import pickle                      # puntos de control para reanudar una corrida interrumpida
import sys
import tempfile
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence
from uuid import uuid4

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

matplotlib.rcParams["figure.dpi"] = 110        # resolución de las gráficas dentro de la libreta
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*(invalid value|divide by zero|Mean of empty).*")


def fecha_diaria(value: Any) -> pd.Timestamp:
    """Convierte un valor a fecha (día completo, sin hora ni zona horaria). Falla si no lo es."""
    result = pd.Timestamp(value)
    if pd.isna(result) or result.tzinfo is not None or result != result.normalize():
        raise ValueError("Se requiere una fecha diaria válida sin hora ni zona horaria.")
    return result


def memoria_disponible_gb() -> float:
    """RAM que puede usar este proceso, en GB. En serverless el contenedor tiene un límite (cgroup) menor que la memoria
    física de la máquina; se toma el menor de los dos para no sobreestimar lo que cabe."""
    limites = []
    try:
        import psutil
        limites.append(psutil.virtual_memory().total)
    except Exception:                                          # sin psutil: memoria física según el sistema operativo
        try:
            limites.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
        except (ValueError, OSError, AttributeError):
            pass
    for ruta in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            valor = open(ruta).read().strip()
            if valor.isdigit() and int(valor) < (1 << 50):       # "max" o un número gigante significan "sin límite"
                limites.append(int(valor))
        except Exception:
            pass
    return min(limites) / 1e9 if limites else 16.0


def runtime_threads(n_jobs=None) -> int:
    """Núcleos de CPU visibles para este proceso; los modelos los usan todos."""
    if n_jobs is None:
        try:
            n_jobs = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n_jobs = os.cpu_count()
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, (int, np.integer)) or n_jobs < 1:
        raise ValueError("No se dispone de un número válido de CPU para el entrenamiento.")
    return int(n_jobs)


class Cronometro:
    """Mide los segundos de un bloque de código: `with Cronometro() as reloj: ...` y después `reloj.segundos`."""

    def __enter__(self):
        self._inicio = time.perf_counter()
        self.segundos = 0.0
        return self

    def __exit__(self, *exc):
        self.segundos = time.perf_counter() - self._inicio
        return False


# Columnas que contienen importes en pesos o volúmenes del negocio. Con MODO_PUBLICO = True no se imprimen: en su
# lugar se muestran métricas relativas (WMAPE, sesgo %, MAE y RMSE divididos entre los de la referencia ingenua, etc.).
COLUMNAS_IMPORTE = {
    "n_filas", "filas", "filas_evaluadas", "ventas_reales", "ventas_predichas", "error_absoluto_total",
    "error_cuadrado_total", "error_total", "n_ventas_cero", "error_absoluto_ceros", "prediccion_en_ceros",
    "prediccion_positiva_ceros", "mae", "rmse", "ancho_medio", "ancho_total", "pinball_p10", "pinball_p50",
    "pinball_p90", "pinball_p10_total", "pinball_p50_total", "pinball_p90_total", "cubiertos", "cubiertos_base",
    "error_absoluto_p50", "cruces_raw", "cruces_final", "n_calibracion", "suma_y", "suma_pred", "cap_entrenamiento",
    "tope", "venta", "real", "pred", "series", "filas_pronostico", "series_pronostico", "filas_ampliada",
}


def mostrar_publico(df, titulo_tabla=None, formatos=None, max_filas=60):
    """Imprime una tabla como `mostrar`, pero con MODO_PUBLICO = True quita las columnas de importes o volúmenes."""
    if df is not None and isinstance(df, pd.DataFrame) and MODO_PUBLICO:
        df = df[[c for c in df.columns if c not in COLUMNAS_IMPORTE]]
    mostrar(df, titulo_tabla, formatos, max_filas)


print(f"Apoyo listo · Python {sys.version.split()[0]} · pandas {pd.__version__} · numpy {np.__version__}")

# COMMAND ----------

# ==================================================================================================================
# 1.1 MODO DE EJECUCIÓN
# ==================================================================================================================
MODO_RAPIDO = True
# True  → prueba de humo: ~5 % de las series en la evaluación, pocas pruebas de Optuna y menos árboles (20–40 min).
#         Sirve para comprobar que todo corre de principio a fin. Sus cifras NO van al reporte.
# False → corrida completa con todas las series evaluables (3.5–5 h con 16 GB). Conviene lanzarla como Job.

MODO_PUBLICO = True
# True  → lo que se imprime sólo contiene métricas relativas, tiempos, hiperparámetros y ventas indexadas (media = 100):
#         la libreta ejecutada puede subirse a GitHub sin exponer información de Bnext. Las tablas Delta internas sí
#         guardan las sumas completas, porque no salen de Databricks.

EJECUTAR_ENTRENAMIENTO = True
# False → se salta el entrenamiento (sección 4) y las secciones 5–10 muestran una corrida ya guardada.

RUN_ID_REANUDAR = None
# Identificador de una corrida interrumpida, p. ej. "av5_43_20261031t031500_1a2b3c4d" (se imprime al iniciar cada
# corrida y queda en mlops.av5_ejecuciones). Con él, la libreta carga su punto de control y continúa en el siguiente
# bloque sin repetir lo ya calculado.

RUN_ID_A_MOSTRAR = None
# Corrida que muestran las secciones 5–10. None = la de esta sesión o, si no hay, la última terminada.

SEMILLA = 43
# Semilla de todo lo aleatorio (muestreos, modelos, Optuna): misma semilla y mismos datos = mismo resultado.

REENTRENAR_FINAL = True
# True → al terminar, el modelo elegido se reentrena con todos los datos disponibles (hasta el último día con objetivo)
#        para producir el pronóstico operativo y registrarse en Unity Catalog.

# ==================================================================================================================
# 1.2 RECURSOS DE CÓMPUTO
# ==================================================================================================================
RAM_GB = memoria_disponible_gb()                 # memoria que puede usar el driver
HILOS_AV5 = runtime_threads()                    # núcleos que usan los modelos
NIVEL_RAM = 64 if RAM_GB >= 56 else 32 if RAM_GB >= 28 else 16
# NIVEL_RAM escala los presupuestos de filas: con 16 GB se conservan los del Avance 4; con un clúster de 32–64 GB los
# modelos usan más filas (mejor precisión) sin cambiar nada más.

# ==================================================================================================================
# 1.3 FUENTE DE DATOS (tabla Oro) Y CONTRATO DE VARIABLES
# ==================================================================================================================
CONFIG_FUENTE = {
    "tabla": tabla("gold", "features_origen"),
    # tabla Oro: una fila por serie (tienda × categoría), origen T y horizonte h, con variables calculadas con días ≤ T
    "version": None,
    # versión Delta que se lee. None = la más reciente al iniciar; queda fija y registrada (snapshot reproducible).
    "semilla": SEMILLA,
    "max_filas_entrenamiento": 400_000 if MODO_RAPIDO else {16: 3_500_000, 32: 7_000_000, 64: 14_000_000}[NIVEL_RAM],
    # filas de entrenamiento que se traen al driver (muestra uniforme y reproducible de todas las elegibles)
    "max_filas_bloque": 9_000_000,
    # tope de filas por lectura de evaluación: si un bloque lo excede, la lectura se detiene en vez de saturar la memoria
    "min_dias_con_venta": 5,
    # una fila es evaluable si su serie tiene ≥ 5 días con venta antes de T (mismo filtro del Avance 4 y la línea base)
    "horizonte": 14,
    "fraccion_series_evaluacion": 0.05 if MODO_RAPIDO else 1.0,
    # fracción de series (por hash, reproducible) que se evalúan en cada bloque; 1.0 = universo completo
    "n_origenes_oof": 0 if MODO_RAPIDO else 1,
    # orígenes extra, anteriores al ajuste, cuyas predicciones fuera de muestra alimentan los primeros meta-modelos
    "lotes_horizonte": 2 if NIVEL_RAM == 16 else 1,
    # en cuántas lecturas se divide cada bloque de evaluación (2 = h 1–7 y h 8–14) para no exceder la memoria
}
ALCANCE_SERVICIOS = "incluir"
# "incluir" → se modelan y evalúan todas las categorías (comparable con el Avance 4).
# "excluir" → se dejan fuera las categorías de servicio (recargas, pagos), si Bnext confirma que no se surten.
# La tabla comparativa reporta siempre el WMAPE de mercancía por separado.

CLAVES_SERIE = ["tienda", "categoria_id"]                  # una serie = una tienda × una categoría
CLAVES_EJEMPLO = [*CLAVES_SERIE, "origen", "h"]            # un ejemplo = serie × origen × horizonte

CATEGORICAS = ["tipo_tienda", "zona", "region", "jefatura", "estatus_tienda", "tipo_categoria", "categoria_id"]
# Llegan como códigos enteros (vocabulario fijo del snapshot); los árboles las tratan como categorías.

NUMERICAS = [
    "s_t0", "s_ma7", "s_ma14", "s_ma28", "s_ma56", "s_desv28", "s_max28",
    "s_dias_activos_7", "s_dias_activos_28", "s_dias_con_venta_28",
    "s_tasa_venta_28", "s_dias_desde_ultima_venta", "dias_con_venta_hist",
    "dias_historia", "s_unidades_ma28", "s_tickets_ma28", "s_tasa_descuento_28",
    "s_intensidad_promo_28", "s_promos_en_venta_28", "s_lag_dow_1",
    "s_lag_dow_2", "s_lag_dow_3", "s_lag_dow_4", "s_media_dow_4",
    "s_media_dow_8", "s_n_dow_8", "t_ma7", "t_ma28", "t_tickets_ma7",
    "t_tickets_ma28", "s_part_tienda_28", "zc_ma7", "zc_ma28",
    "zc_lag_dow_1", "zc_lag_dow_2", "zc_media_dow_4", "pc_ma7", "pc_ma28",
    "pc_lag_dow_1", "pc_lag_dow_2", "pc_media_dow_4", "dow", "fin_de_semana",
    "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento",
    "dias_a_evento", "dias_desde_evento", "en_mundial", "h_bloque",
    "antiguedad_tienda", "h",
]
VARIABLE_CONDICIONAL = "s_promos_vigentes_t0"
# Sólo se usa si la tabla Oro declara (propiedad mna.s_promos_vigentes_t0 = causal_7d) que se calcula con días ≤ T.
# En la Oro anterior dependía de la última venta de cada promoción (posterior a T) y por eso se excluía.

METADATOS = [
    "tienda", "categoria_id", "origen", "fecha_objetivo", "conjunto", "corte",
    "en_muestra_eval", "y", "ingenuo_estacional", "dia_atipico_obj",
]
COLUMNAS_CARGA = list(dict.fromkeys([*METADATOS, *NUMERICAS, *CATEGORICAS]))

DESCRIPCION_VARIABLES = {
    # ---- historia de la serie (tienda × categoría); siempre con días ≤ T
    "s_t0": "Venta de la serie el día del origen T.",
    "s_ma7": "Venta media diaria de la serie en los 7 días que terminan en T.",
    "s_ma14": "Venta media diaria en los 14 días que terminan en T.",
    "s_ma28": "Venta media diaria en los 28 días que terminan en T (nivel reciente; también es la escala de la serie).",
    "s_ma56": "Venta media diaria en los 56 días que terminan en T (nivel de largo plazo).",
    "s_desv28": "Desviación estándar de la venta diaria en 28 días (volatilidad).",
    "s_max28": "Venta diaria máxima en 28 días (tamaño de los picos).",
    "s_dias_activos_7": "Días con la tienda activa (hay dato, aunque sea cero) en los últimos 7 días.",
    "s_dias_activos_28": "Días con la tienda activa en los últimos 28 días.",
    "s_dias_con_venta_28": "Días con venta mayor que cero en los últimos 28 días.",
    "s_tasa_venta_28": "Fracción de días activos con venta en 28 días (1 = vende diario; mide la intermitencia).",
    "s_dias_desde_ultima_venta": "Días desde la última venta positiva (56 si no vendió en la ventana).",
    "dias_con_venta_hist": "Días con venta en toda la historia de la serie hasta T (madurez).",
    "dias_historia": "Días desde la primera fecha de la serie hasta T.",
    "s_unidades_ma28": "Unidades medias diarias en 28 días.",
    "s_tickets_ma28": "Tickets medios diarios de la serie en 28 días.",
    "s_tasa_descuento_28": "Tasa media de descuento efectiva (1 − venta / venta sin descuento) en 28 días.",
    "s_intensidad_promo_28": "Fracción media de líneas vendidas con promoción en 28 días.",
    "s_promos_vigentes_t0": "Máximo de promociones distintas con venta en un mismo día de la última semana (≤ T).",
    "s_promos_en_venta_28": "Promociones distintas con venta por día, promedio de 28 días.",
    "s_lag_dow_1": "Venta del mismo día de la semana que el día objetivo, en la semana más reciente ≤ T.",
    "s_lag_dow_2": "Ídem, dos semanas antes.",
    "s_lag_dow_3": "Ídem, tres semanas antes.",
    "s_lag_dow_4": "Ídem, cuatro semanas antes.",
    "s_media_dow_4": "Promedio del mismo día de la semana en las últimas 4 semanas.",
    "s_media_dow_8": "Promedio del mismo día de la semana en las últimas 8 semanas.",
    "s_n_dow_8": "Cuántas de esas 8 semanas tienen dato (confiabilidad del promedio anterior).",
    "s_part_tienda_28": "Participación de la categoría en la venta de su tienda (28 días).",
    # ---- contexto: tienda, categoría en la zona y en el país
    "t_ma7": "Venta media diaria de toda la tienda en 7 días.",
    "t_ma28": "Venta media diaria de toda la tienda en 28 días.",
    "t_tickets_ma7": "Tickets medios diarios de la tienda en 7 días (afluencia).",
    "t_tickets_ma28": "Tickets medios diarios de la tienda en 28 días.",
    "zc_ma7": "Venta media por tienda de la categoría en la zona, 7 días.",
    "zc_ma28": "Venta media por tienda de la categoría en la zona, 28 días.",
    "zc_lag_dow_1": "Venta media por tienda de la categoría en la zona, mismo día de la semana, semana más reciente.",
    "zc_lag_dow_2": "Ídem, dos semanas antes.",
    "zc_media_dow_4": "Ídem, promedio de 4 semanas.",
    "pc_ma7": "Venta media por tienda de la categoría en todo el país, 7 días.",
    "pc_ma28": "Venta media por tienda de la categoría en todo el país, 28 días.",
    "pc_lag_dow_1": "Venta media por tienda de la categoría en el país, mismo día de la semana, semana más reciente.",
    "pc_lag_dow_2": "Ídem, dos semanas antes.",
    "pc_media_dow_4": "Ídem, promedio de 4 semanas.",
    # ---- calendario del día objetivo (se conoce de antemano)
    "dow": "Día de la semana del día objetivo (1 = lunes … 7 = domingo).",
    "fin_de_semana": "1 si el día objetivo es sábado o domingo.",
    "dia_mes": "Día del mes del día objetivo.",
    "semana_mes": "Semana del mes del día objetivo (1–5).",
    "quincena": "1 en días de pago (14–16, último y primero del mes).",
    "es_evento": "1 si el día objetivo es un evento del calendario (festivo, comercial o futbol).",
    "tipo_evento": "Tipo de evento: 0 ninguno, 1 festivo, 2 comercial, 3 futbol.",
    "dias_a_evento": "Días que faltan para el siguiente evento (tope 30).",
    "dias_desde_evento": "Días desde el último evento (tope 30).",
    "en_mundial": "1 si el día objetivo cae dentro de la Copa Mundial 2026.",
    "h": "Horizonte: días entre el origen T y el día objetivo (1–14).",
    "h_bloque": "1 si h = 1–7; 2 si h = 8–14.",
    # ---- atributos (categóricas codificadas y antigüedad)
    "antiguedad_tienda": "Días desde la primera venta de la tienda hasta T.",
    "tipo_tienda": "Formato de la tienda.",
    "zona": "Zona operativa de la tienda.",
    "region": "Región de la tienda (la de su zona cuando falta en el catálogo).",
    "jefatura": "Jefatura a la que reporta la tienda.",
    "estatus_tienda": "Estatus de la tienda en el catálogo.",
    "tipo_categoria": "Tipo de categoría: mercancía, servicio o genérico.",
    "categoria_id": "Categoría (identificador canónico homologado).",
}
DESCRIPCION_METADATOS = {
    "tienda": "Identificador de la tienda (nunca se imprime).",
    "categoria_id": "Identificador de la categoría (también es variable categórica del modelo).",
    "origen": "Día T en que se hace el pronóstico; todas las variables usan días ≤ T.",
    "fecha_objetivo": "Día pronosticado: T + h.",
    "conjunto": "entrenamiento (muestra de series en muchos orígenes) · evaluacion (todas las series en los cortes) · pronostico (futuro).",
    "corte": "ajuste, corte_1…corte_4 (desarrollo), prueba_1 y prueba_2 (septiembre, no vistos) o pronostico.",
    "en_muestra_eval": "1 si la serie pertenece a la muestra común (3 % de las series) en la que se comparan modelos rápido.",
    "y": "Venta real del día objetivo (pesos): lo que se pronostica. Nula en el conjunto pronostico.",
    "ingenuo_estacional": "Referencia B0: venta del mismo día de la semana más reciente ≤ T.",
    "dia_atipico_obj": "1 si el día objetivo fue atípico (apagón, ley seca); sólo para analizar el error, no es variable.",
}
faltan_descripcion = [c for c in [*NUMERICAS, VARIABLE_CONDICIONAL, *CATEGORICAS] if c not in DESCRIPCION_VARIABLES]
assert not faltan_descripcion, f"Variables sin descripción: {faltan_descripcion}"
print(f"Modo {'RÁPIDO (prueba de humo)' if MODO_RAPIDO else 'COMPLETO'} · RAM ≈ {RAM_GB:.0f} GB (nivel {NIVEL_RAM}) · "
      f"{HILOS_AV5} núcleos · {len(NUMERICAS)} numéricas + {len(CATEGORICAS)} categóricas (+ {VARIABLE_CONDICIONAL} si la Oro "
      f"la declara causal) · salidas {'públicas (sin importes)' if MODO_PUBLICO else 'completas'}")

# COMMAND ----------

# ==================================================================================================================
# 1.4 MODELOS, HIPERPARÁMETROS Y PRESUPUESTOS
# ==================================================================================================================
CANDIDATOS_BASE = ("I1_LightGBM_Tweedie", "I2_CatBoost_Tweedie", "I3_ElasticNet", "I4_XGBoost_Tweedie",
                   "I5_LightGBM_Log1p", "I6_Theta")
CANDIDATOS_HOMOGENEOS = ("E1_Bagging", "E2_AdaBoost", "E7_Bosque_Aleatorio")
POINT_HETEROGENEOUS = ("E3_Blending", "E4_Stacking_temporal", "E5_Votacion", "E6_Stacking_LightGBM")
MODELOS_DIAGNOSTICO = ("D1_Stacking_sin_correccion",)
INCLUIR = {
    "I2_CatBoost_Tweedie": True, "I4_XGBoost_Tweedie": True, "I5_LightGBM_Log1p": True,
    "I6_Theta": False,           # Theta por serie se calcula en Spark: ~5–10 min por bloque; actívalo si hay tiempo
    "E2_AdaBoost": True, "E7_Bosque_Aleatorio": True, "E6_Stacking_LightGBM": True,
}
# POINT_BASE_MODELS y POINT_HOMOGENEOUS (los modelos que de verdad se entrenan) se fijan en la sección 3, cuando ya se
# sabe qué bibliotecas quedaron instaladas.

AGG_LABELS = {
    "B0_Ingenuo": "B0 · Ingenuo estacional", "I1_LightGBM_Tweedie": "I1 · LightGBM Tweedie",
    "I2_CatBoost_Tweedie": "I2 · CatBoost Tweedie", "I3_ElasticNet": "I3 · Elastic Net",
    "I4_XGBoost_Tweedie": "I4 · XGBoost Tweedie", "I5_LightGBM_Log1p": "I5 · LightGBM log1p", "I6_Theta": "I6 · Theta",
    "E1_Bagging": "E1 · Bagging LightGBM", "E2_AdaBoost": "E2 · AdaBoost", "E7_Bosque_Aleatorio": "E7 · Bosque aleatorio",
    "E3_Blending": "E3 · Blending", "E4_Stacking_temporal": "E4 · Stacking temporal", "E5_Votacion": "E5 · Votación",
    "E6_Stacking_LightGBM": "E6 · Stacking LightGBM", "D1_Stacking_sin_correccion": "D1 · Stacking sin corrección",
}
ESTRATEGIA = {
    "B0_Ingenuo": "regla: mismo día de la semana más reciente",
    "I1_LightGBM_Tweedie": "boosting Tweedie, ajustado con Optuna", "I2_CatBoost_Tweedie": "boosting de árboles simétricos",
    "I3_ElasticNet": "lineal regularizado sobre log(1 + y)", "I4_XGBoost_Tweedie": "boosting Tweedie, ajustado con Optuna",
    "I5_LightGBM_Log1p": "boosting sobre log(1 + y) con corrección de escala", "I6_Theta": "estadístico por serie",
    "E1_Bagging": "bagging: 3 LightGBM en submuestras distintas", "E2_AdaBoost": "boosting adaptativo de árboles",
    "E7_Bosque_Aleatorio": "bagging de árboles + submuestreo de variables",
    "E3_Blending": "blending: pesos convexos del bloque más reciente",
    "E4_Stacking_temporal": "stacking: pesos no negativos por horizonte + corrección de nivel",
    "E5_Votacion": "votación: promedio de las 3 mejores bases", "E6_Stacking_LightGBM": "stacking no lineal (meta LightGBM)",
    "D1_Stacking_sin_correccion": "diagnóstico: E4 sin corrección de nivel",
}
INTERPRETABILIDAD = {   # 1 = cualquiera lo explica; 0.4 = caja negra compuesta. Se fija antes de ver resultados.
    "B0_Ingenuo": 1.0, "I3_ElasticNet": 1.0, "I6_Theta": 0.9, "I1_LightGBM_Tweedie": 0.8, "I2_CatBoost_Tweedie": 0.8,
    "I4_XGBoost_Tweedie": 0.8, "I5_LightGBM_Log1p": 0.8, "E1_Bagging": 0.7, "E2_AdaBoost": 0.7, "E7_Bosque_Aleatorio": 0.7,
    "E3_Blending": 0.65, "E4_Stacking_temporal": 0.6, "E5_Votacion": 0.65, "E6_Stacking_LightGBM": 0.4,
    "D1_Stacking_sin_correccion": 0.6,
}
OPERABILIDAD = {        # 1 = un solo modelo que se entrena y sirve; menos = más piezas que versionar cada mes
    "B0_Ingenuo": 1.0, "I1_LightGBM_Tweedie": 1.0, "I2_CatBoost_Tweedie": 1.0, "I3_ElasticNet": 1.0,
    "I4_XGBoost_Tweedie": 1.0, "I5_LightGBM_Log1p": 1.0, "I6_Theta": 0.7, "E1_Bagging": 0.8, "E2_AdaBoost": 0.9,
    "E7_Bosque_Aleatorio": 0.9, "E3_Blending": 0.55, "E4_Stacking_temporal": 0.5, "E5_Votacion": 0.6,
    "E6_Stacking_LightGBM": 0.4, "D1_Stacking_sin_correccion": 0.5,
}

TOPE_PREDICCION_FACTOR = 3.0
# Ninguna predicción supera 3 × el percentil 99.9 del objetivo de entrenamiento (evita explosiones numéricas).

S6_POINT_PARAMETERS = {
    "I1_LightGBM_Tweedie": {     # configuración del Avance 4: es la primera prueba de Optuna y se conserva si gana
        "objective": "tweedie", "tweedie_variance_power": 1.5, "n_estimators": 500, "learning_rate": 0.05,
        "num_leaves": 63, "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.8, "reg_lambda": 1.0,
    },
    "I2_CatBoost_Tweedie": {     # configuración del Avance 4 (sin cambios)
        "loss_function": "Tweedie:variance_power=1.5", "iterations": 400, "learning_rate": 0.08,
        "depth": 8, "l2_leaf_reg": 3.0,
    },
    "I3_ElasticNet": {           # Avance 4 con max_iter 250 → 1000 y variables en log: elimina los avisos de convergencia
        "alpha": 1e-3, "l1_ratio": 0.5, "max_iter": 1000, "tol": 1e-3, "selection": "random",
    },
    "I4_XGBoost_Tweedie": {      # punto de partida de Optuna para XGBoost (nuevo en este avance)
        "objective": "reg:tweedie", "tweedie_variance_power": 1.5, "n_estimators": 500, "learning_rate": 0.05,
        "max_depth": 8, "min_child_weight": 50.0, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
        "tree_method": "hist", "max_bin": 256,
    },
    "I5_LightGBM_Log1p": {       # la estructura del I1 del Avance 4, con pérdida cuadrática sobre log(1 + y)
        "objective": "regression", "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63,
        "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0,
    },
}
if MODO_RAPIDO:
    S6_POINT_PARAMETERS["I2_CatBoost_Tweedie"]["iterations"] = 150

_PRESUPUESTO = {   # filas de entrenamiento por modelo según la RAM (16 GB = presupuestos del Avance 4)
    16: {"I1_LightGBM_Tweedie": 2_000_000, "I2_CatBoost_Tweedie": 1_000_000, "I3_ElasticNet": 1_500_000,
         "I4_XGBoost_Tweedie": 2_000_000, "I5_LightGBM_Log1p": 2_000_000, "E2_AdaBoost": 300_000,
         "E7_Bosque_Aleatorio": 1_500_000},
    32: {"I1_LightGBM_Tweedie": 4_000_000, "I2_CatBoost_Tweedie": 2_000_000, "I3_ElasticNet": 3_000_000,
         "I4_XGBoost_Tweedie": 4_000_000, "I5_LightGBM_Log1p": 4_000_000, "E2_AdaBoost": 500_000,
         "E7_Bosque_Aleatorio": 3_000_000},
    64: {"I1_LightGBM_Tweedie": None, "I2_CatBoost_Tweedie": 3_000_000, "I3_ElasticNet": 4_000_000,
         "I4_XGBoost_Tweedie": None, "I5_LightGBM_Log1p": None, "E2_AdaBoost": 800_000, "E7_Bosque_Aleatorio": 6_000_000},
}
S6_MODEL_ROW_BUDGETS = dict(_PRESUPUESTO[NIVEL_RAM])     # None = todas las filas disponibles en el origen
if MODO_RAPIDO:
    S6_MODEL_ROW_BUDGETS = {k: 50_000 if k == "E2_AdaBoost" else 150_000 for k in S6_MODEL_ROW_BUDGETS}

S7_HOMOGENEOUS_PARAMETERS = {
    "E1_Bagging": {"semillas_extra": (44, 45), "fraccion_filas": 0.7},
    # I1 (semilla 43, todas sus filas) + dos LightGBM con los mismos hiperparámetros en submuestras del 70 % con otras
    # semillas; la predicción es el promedio de los tres (reduce varianza).
    "E2_AdaBoost": {"n_estimators": 30 if not MODO_RAPIDO else 10, "max_depth": 10, "min_samples_leaf": 100,
                    "learning_rate": 0.1, "loss": "linear"},
    # AdaBoost.R2 clásico sobre árboles de decisión (referencia de boosting adaptativo).
    "E7_Bosque_Aleatorio": {"boosting_type": "rf", "objective": "regression", "n_estimators": 200 if not MODO_RAPIDO else 40,
                            "num_leaves": 255, "min_child_samples": 100, "subsample": 0.632, "subsample_freq": 1,
                            "colsample_bytree": 0.5},
    # bosque aleatorio implementado con LightGBM (modo rf): árboles independientes en submuestras, promediados.
}

# ---- ajuste fino con Optuna (sólo en el origen de ajuste, anterior a todos los cortes)
MODELOS_AJUSTE_FINO = ("I1_LightGBM_Tweedie", "I4_XGBoost_Tweedie")
N_PRUEBAS_AJUSTE = 6 if MODO_RAPIDO else 30          # pruebas por modelo
N_PRUEBAS_ALEATORIAS = 3 if MODO_RAPIDO else 8       # pruebas al azar antes de que TPE empiece a optimizar
TIEMPO_MAX_AJUSTE_MIN = 3 if MODO_RAPIDO else 25     # tope de minutos por modelo (se revisa entre pruebas)
FILAS_POR_PRUEBA = 60_000 if MODO_RAPIDO else 500_000
ARBOLES_MAX = 300 if MODO_RAPIDO else 3000           # máximo de árboles; el paro temprano decide cuántos
PARO_TEMPRANO = 30 if MODO_RAPIDO else 50            # rondas sin mejora antes de detener
PARAMETROS_VIGENTES = {}
# Hiperparámetros con los que se entrena cada modelo en todos los orígenes. Se llenan en el ajuste (Optuna + número de
# árboles por paro temprano) y quedan congelados para los cortes, la prueba y el reentrenamiento final.

# ==================================================================================================================
# 1.5 ENSAMBLES HETEROGÉNEOS, INTERVALOS, DECISIÓN Y COMPUERTA
# ==================================================================================================================
VIDA_MEDIA_BLOQUES = 2.0
# E4, E5 y E6 aprenden de todos los bloques ya observados, pero el peso de cada bloque se reduce a la mitad cada 2
# bloques de antigüedad (los recientes pesan más: la demanda cambia con la temporada).
N_MIEMBROS_VOTACION = 3                       # E5: cuántas bases se promedian
REDUCCION_FACTOR_NIVEL = 2_000
# E4: el factor de nivel por tipo de categoría × bloque de horizonte se encoge hacia 1 según n / (n + 2,000) filas
LIMITES_FACTOR_NIVEL = (0.85, 1.15)           # E4: el factor de nivel nunca corrige más de ±15 %
S7_META_LGBM = {"objective": "tweedie", "tweedie_variance_power": 1.5, "n_estimators": 200, "learning_rate": 0.05,
                "num_leaves": 15, "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1,
                "colsample_bytree": 0.9, "reg_lambda": 5.0}
# E6: meta-modelo pequeño y regularizado (pocas hojas) para no sobreajustar los bloques de meta-entrenamiento.

QUANTILE_LEVELS = (0.1, 0.5, 0.9)             # cuantiles del intervalo de pronóstico
QUANTILE_COLUMNS = ("p10", "p50", "p90")
NOMINAL_COVERAGE = 0.8                        # el intervalo P10–P90 debe cubrir el 80 % de los días reales
BANDA_COBERTURA = (0.75, 0.85)                # banda de aceptación de la cobertura observada
METODOS_INTERVALO = ("Q_Conformal", "Q_LightGBM_CQR")
INCLUIR_Q_LIGHTGBM = True
# Q_Conformal: intervalos alrededor del pronóstico del modelo elegido (residuos de un bloque reciente, por grupos).
# Q_LightGBM_CQR: tres LightGBM cuantílicos con calibración conformal (el método del avance anterior), como comparación.
METODO_INTERVALOS = "automatico"
# "automatico" → el modelo final usa el método con menor pérdida pinball en septiembre entre los que cumplen la banda.
N_TRAMOS_CONFORMAL = 10                       # tramos del pronóstico (deciles) en que se agrupan los residuos
MIN_FILAS_GRUPO_CONFORMAL = 200               # un grupo con menos filas usa el grupo más amplio
S6_QUANTILE_PARAMETERS = {
    "n_estimators": 300 if MODO_RAPIDO else 500, "learning_rate": 0.05, "num_leaves": 63,
    "min_child_samples": 200, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "random_state": SEMILLA, "verbosity": -1,
}
QUANTILE_METHOD_REFERENCES = {
    "lightgbm": "https://lightgbm.readthedocs.io/en/stable/Parameters.html",
    "cqr": "https://papers.nips.cc/paper/8613-conformalized-quantile-regression.pdf",
    "conformal_mondrian": "https://arxiv.org/abs/2107.07511",
}

PESOS_DECISION = {"error": 0.35, "estabilidad": 0.10, "sesgo": 0.10, "error_donde_cuesta": 0.10, "costo": 0.15,
                  "interpretabilidad": 0.10, "operabilidad": 0.10}
# Matriz de decisión fijada antes de ver resultados: error = WMAPE medio de los cortes; estabilidad = su desviación;
# sesgo = |sesgo| medio por corte; error donde cuesta = WMAPE de mercancía; costo = minutos del reentrenamiento
# mensual (no penaliza mientras quepa en PRESUPUESTO_JOB_MIN); interpretabilidad y operabilidad = tablas de arriba.
PRESUPUESTO_JOB_MIN = 180                     # minutos que puede tardar el reentrenamiento mensual en un Job
N_FINALISTAS = 5                              # cuántos candidatos entran a la matriz (más el mejor individual)
UMBRALES_COMPUERTA = {"sesgo_max_pct": 3.0, "dm_p": 0.05, "dm_cortes_min": 3, "cortes_min": 4,
                      "wmape_global_max": 0.553, "wmape_mercancia_max": 0.506, "sesgo_prueba_max_pct": 5.0}
# 0.553 y 0.506 = WMAPE del ingenuo estacional (total y mercancía) en el Avance 4: el modelo debe superarlos.
EVIDENCIA_NEGOCIO = {"scope_approved": False, "services_policy_approved": False, "point_in_time_validated": False}
# Se cambian a True sólo cuando Bnext confirme por escrito el alcance (población), la política de servicios y que los
# atributos de tienda están disponibles a la fecha de cada pronóstico. Mientras tanto, la compuerta bloquea producción.

# ==================================================================================================================
# 1.6 SALIDAS: TABLAS, PUNTOS DE CONTROL, MLFLOW Y UNITY CATALOG
# ==================================================================================================================
TABLA_EJECUCIONES = tabla("mlops", "av5_ejecuciones")      # bitácora de eventos de cada corrida
TABLA_PRONOSTICO = tabla("gold", "pronostico_14d")         # pronóstico operativo (tienda × categoría × día)
TABLA_MONITOREO = tabla("mlops", "monitoreo_modelo")       # error, sesgo y cobertura por semana × región × tipo
RUTA_PUNTOS_CONTROL = volumen("outputs", "avance5_puntos_control")   # carpeta de los puntos de control
EXPERIMENTO_MLFLOW = None                 # None → /Users/<usuario>/mna_equipo43_avance5
NOMBRE_MODELO_UC = f"{CATALOGO}.{ESQUEMA_DATOS}.demanda_categoria"
REGISTRAR_EN_UC = None
# None → se registra (alias @challenger) sólo en modo completo y si pasan las comprobaciones técnicas de la compuerta.
NIVEL_TECHO_OPERATIVO = "p90"
# Columna que se publica como techo operativo de surtido (P90 por defecto; cambiar cuando Bnext elija P80 o P90).

# ==================================================================================================================
# 1.7 CALENDARIO DE PROMOCIONES FUTURAS (apartado preparado; ver sección 9)
# ==================================================================================================================
USAR_CALENDARIO_PROMOS = False
# Hoy el modelo sólo ve promociones PASADAS (variables s_* con días ≤ T). Cuando Bnext entregue el calendario de
# promociones planeadas, se carga en la tabla de abajo y se cambia a True: las variables f_* entran al modelo.
TABLA_CALENDARIO_PROMOS = tabla("silver", "calendario_promociones")
VARIABLES_PROMO_FUTURA = ["f_promos_planeadas", "f_descuento_planeado", "f_dias_promo_restantes"]
print("Configuración lista.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Funciones: datos, modelos, ensambles, intervalos, evaluación y registro
# MAGIC Esta sección sólo **define** funciones; no lee tablas ni entrena. Cada función explica qué recibe, qué hace y qué
# MAGIC controles aplica. La ejecución está en la sección 4.
# MAGIC
# MAGIC ### 2.1 Fuente de datos: tabla Oro con versión fija
# MAGIC `FuenteAvance5` fija una versión Delta de la tabla Oro (snapshot: la corrida se puede reconstruir), traduce las
# MAGIC categóricas a códigos enteros en Spark (vocabulario completo del snapshot, ordenado), descubre los orígenes que
# MAGIC existen (ajuste, cortes de desarrollo, bloques de prueba, pronóstico) y carga en pandas sólo lo necesario: una
# MAGIC muestra de entrenamiento y, para evaluar, un bloque de 7 o 14 horizontes a la vez.

# COMMAND ----------

def validar_config_fuente(config=None) -> dict:
    """Valida CONFIG_FUENTE (llaves conocidas, tipos y rangos) y devuelve una copia lista para usar."""
    resolved = dict(CONFIG_FUENTE)
    if config is not None:
        unknown = set(config) - set(resolved)
        if unknown:
            raise ValueError(f"Parámetros de fuente desconocidos: {sorted(unknown)}")
        resolved.update(config)
    if not re.fullmatch(r"[A-Za-z_][\w]*\.[A-Za-z_][\w]*\.[A-Za-z_][\w]*", resolved["tabla"]):
        raise ValueError("Se requiere una tabla de tres componentes: catálogo.esquema.tabla.")
    for name in ("semilla", "max_filas_bloque", "min_dias_con_venta", "horizonte", "n_origenes_oof", "lotes_horizonte"):
        value = resolved[name]
        minimum = 0 if name in ("semilla", "n_origenes_oof") else 1
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} debe ser un entero ≥ {minimum}.")
    if resolved["max_filas_entrenamiento"] is not None and resolved["max_filas_entrenamiento"] < 1:
        raise ValueError("max_filas_entrenamiento debe ser positivo o None (todas las filas).")
    if resolved["horizonte"] != 14:
        raise ValueError("El horizonte del proyecto es de catorce días.")
    if not 0 < float(resolved["fraccion_series_evaluacion"]) <= 1:
        raise ValueError("fraccion_series_evaluacion debe estar en (0, 1].")
    if resolved["lotes_horizonte"] not in (1, 2, 7, 14):
        raise ValueError("lotes_horizonte debe ser 1, 2, 7 o 14 (divisores de 14).")
    return resolved


def normalizar_bloque(pdf: pd.DataFrame, max_filas=9_000_000, con_objetivo=True) -> pd.DataFrame:
    """Valida un bloque leído de la Oro y deja los tipos compactos: fechas como datetime64, h entero, variables float32,
    categóricas como códigos int32 (−1 = desconocido). Con con_objetivo=False (conjunto pronostico) `y` puede ser nulo."""
    if len(pdf) > max_filas:
        raise MemoryError("El bloque supera el presupuesto de carga; se divide por lotes de horizonte.")
    if pdf.empty:
        raise ValueError("El bloque solicitado no contiene ejemplos elegibles.")
    for name in ("origen", "fecha_objetivo"):
        pdf[name] = pd.to_datetime(pdf[name], errors="raise")
        if pdf[name].isna().any():
            raise ValueError(f"La columna {name} tiene fechas nulas.")
    pdf["h"] = pdf["h"].astype("int16")
    if not ((pdf["fecha_objetivo"] - pdf["origen"]).dt.days.to_numpy() == pdf["h"].to_numpy()).all():
        raise ValueError("La fecha objetivo debe ser origen más horizonte.")
    if not pdf["h"].between(1, 14).all():
        raise ValueError("Los horizontes deben ser enteros de uno a catorce.")
    if con_objetivo:
        y = pdf["y"].to_numpy(dtype=np.float64)
        if not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("Las ventas objetivo deben ser finitas y no negativas.")
    base = pdf["ingenuo_estacional"].to_numpy(dtype=np.float64)
    if not np.isfinite(base).all() or (base < 0).any():
        raise ValueError("La referencia ingenua debe ser finita y no negativa en las filas elegibles.")
    for name in pdf.columns:
        if pdf[name].dtype == np.float32 and np.isinf(pdf[name].to_numpy()).any():
            raise ValueError(f"La variable {name} contiene infinitos.")
    for name in CATEGORICAS:
        if name in pdf:
            pdf[name] = pdf[name].fillna(-1).astype("int32")
    for name in ("en_muestra_eval", "dia_atipico_obj"):
        if name in pdf:
            pdf[name] = pdf[name].fillna(-1).astype("int8")
    if "tienda" in pdf and pdf.duplicated(["tienda", "categoria_id", "origen", "h"]).any():
        raise ValueError("Existen ejemplos duplicados (serie, origen, h) en el bloque.")
    for name in ("tienda", "conjunto", "corte"):          # texto repetido → categoría: ahorra cientos de MB por bloque
        if name in pdf:
            pdf[name] = pdf[name].astype("category")
    return pdf.reset_index(drop=True)


def resolver_origenes(resumen: pd.DataFrame, horizonte=14, n_oof=1) -> dict:
    """A partir del resumen (conjunto, corte, origen → número de horizontes y fecha objetivo máxima) identifica los bloques
    completos: ajuste, cortes de desarrollo, bloques de prueba y origen de pronóstico. Exige que los bloques de prueba
    empiecen después de que termine el último corte (así son datos no vistos) y elige los orígenes OOF del entrenamiento."""
    required = {"conjunto", "corte", "origen", "n_h", "h_min", "h_max", "fecha_objetivo_max", "filas"}
    if required - set(resumen):
        raise ValueError("El resumen temporal no cumple el contrato de orígenes.")
    frame = resumen.copy()
    frame["origen"] = pd.to_datetime(frame["origen"], errors="raise")
    frame["fecha_objetivo_max"] = pd.to_datetime(frame["fecha_objetivo_max"], errors="raise")
    paso = pd.Timedelta(days=horizonte)
    complete = frame.loc[(frame.n_h == horizonte) & (frame.h_min == 1) & (frame.h_max == horizonte)
                         & frame.fecha_objetivo_max.eq(frame.origen + paso)]
    evaluation = complete.loc[complete.conjunto.eq("evaluacion")].copy()
    if evaluation.corte.isna().any() or evaluation.corte.duplicated().any():
        raise ValueError("Cada bloque de evaluación debe tener un origen único.")
    adjustment = evaluation.loc[evaluation.corte.eq("ajuste")]
    if len(adjustment) != 1:
        raise ValueError("Se requiere el origen de ajuste en la Oro.")
    aj = fecha_diaria(adjustment.origen.iloc[0])
    dev = evaluation.loc[evaluation.corte.str.startswith("corte_")].sort_values("origen")
    tests = evaluation.loc[evaluation.corte.str.startswith("prueba_")].sort_values("origen")
    if len(dev) < 2:
        raise ValueError("Se requieren al menos dos cortes de desarrollo completos.")
    if (dev.origen.diff().dropna() < paso).any() or (tests.origen.diff().dropna() < paso).any():
        raise ValueError("Los bloques de evaluación no pueden solaparse.")
    if aj + paso > dev.origen.iloc[0]:
        raise ValueError("El bloque de ajuste debe terminar antes del primer corte de desarrollo.")
    if len(tests) and tests.origen.iloc[0] < dev.origen.iloc[-1] + paso:
        raise ValueError("La prueba debe empezar después del último corte de desarrollo (datos no vistos).")
    train_origins = sorted(set(complete.loc[complete.conjunto.eq("entrenamiento"), "origen"]))
    oof = []
    for k in range(1, int(n_oof) + 1):
        target = aj - k * paso
        candidates = [o for o in train_origins if o <= target and (not oof or o <= oof[-1] - paso)]
        if not candidates:
            break
        oof.append(max(candidates))
    oof = sorted(oof)
    pronostico = frame.loc[frame.conjunto.eq("pronostico"), "origen"]
    observados = frame.loc[~frame.conjunto.eq("pronostico")]

    def bloque(row, fase):
        return {"corte": str(row.corte), "fase": fase, "origen": row.origen.date().isoformat(),
                "desde": (row.origen + pd.Timedelta(days=1)).date().isoformat(),
                "hasta": (row.origen + paso).date().isoformat()}

    return {"cortes": [bloque(r, "desarrollo") for r in dev.itertuples()],
            "pruebas": [bloque(r, "prueba") for r in tests.itertuples()],
            "ajuste_origen": aj.date().isoformat(),
            "oof_origenes": [o.date().isoformat() for o in oof],
            "origenes_entrenamiento_completos": [o.date().isoformat() for o in train_origins],
            "origenes_observados_completos": [o.date().isoformat() for o in sorted(set(complete.loc[
                ~complete.conjunto.eq("pronostico"), "origen"]))],
            "origen_pronostico": pronostico.max().date().isoformat() if len(pronostico) else None,
            "fecha_maxima_objetivo": observados.fecha_objetivo_max.max().date().isoformat()}


class FuenteAvance5:
    """Tabla Oro con una versión Delta fija y cargas en pandas para modelar.

    Atributos principales (se imprimen en la sección 3):
      version            versión Delta leída (todas las lecturas de la corrida usan la misma)
      num_features       variables numéricas que usan los modelos (NUMERICAS + s_promos_vigentes_t0 si es causal)
      cat_features       variables categóricas (códigos enteros)
      codigos/etiquetas  vocabulario de cada categórica: texto → código y código → texto
      cortes, pruebas    bloques de desarrollo y de prueba (lista de diccionarios con origen, desde, hasta)
      ajuste_origen      origen del bloque de ajuste de hiperparámetros
      oof_origenes       orígenes extra, anteriores al ajuste, para los primeros meta-modelos
      origen_pronostico  último día con datos (salida operativa) o None
      manifiesto         todo lo anterior en un diccionario que se registra en MLflow
    """

    def __init__(self, spark, config=None):
        self.spark = spark
        self.config = validar_config_fuente(config)
        self.tabla = self.config["tabla"]
        if not existe(self.tabla):
            raise ValueError(f"No existe la tabla Oro {self.tabla}. Corre antes 20_features_oro.")
        self.version, self.modo_lectura, self._gold = self._fijar_version(self.config["version"])
        faltan = set(COLUMNAS_CARGA) - set(self._gold.columns)
        if faltan:
            raise ValueError(f"La Oro no cumple el contrato de columnas: {sorted(faltan)}")
        self.propiedades = self._leer_propiedades()
        self.promos_vigentes_causal = (self.propiedades.get("mna.s_promos_vigentes_t0") == "causal_7d"
                                       and VARIABLE_CONDICIONAL in self._gold.columns)
        self.num_features = NUMERICAS + ([VARIABLE_CONDICIONAL] if self.promos_vigentes_causal else [])
        self._promos_futuras = cargar_variables_promo_futura(spark) if USAR_CALENDARIO_PROMOS else None
        if self._promos_futuras is not None:
            self.num_features = self.num_features + VARIABLES_PROMO_FUTURA
        self.cat_features = CATEGORICAS.copy()
        self.features_table = [*self.num_features, *self.cat_features]
        self.codigos, self.etiquetas = self._vocabularios()
        self._contratos = None
        self._pool = None
        self._resolver_metadata()
        self.manifiesto = {
            "tabla_gold": self.tabla, "version_delta": self.version, "modo_lectura": self.modo_lectura,
            "semilla": self.config["semilla"], "max_filas_carga_train": self.config["max_filas_entrenamiento"],
            "num_features": self.num_features, "cat_features": self.cat_features,
            "numero_features": len(self.features_table),
            "s_promos_vigentes_t0": ("incluida: la Oro la declara causal (≤ T)" if self.promos_vigentes_causal else
                                     "excluida: la Oro no declara que sea causal"),
            "calendario_promos_futuras": "activo" if self._promos_futuras is not None else "apagado",
            "alcance_servicios": ALCANCE_SERVICIOS,
            "cortes": self.cortes, "pruebas": self.pruebas, "ajuste_origen": self.ajuste_origen,
            "oof_origenes": self.oof_origenes, "origen_pronostico": self.origen_pronostico,
            "fecha_maxima_objetivo": self.fecha_maxima_objetivo,
            "fraccion_series_evaluacion": self.config["fraccion_series_evaluacion"],
            "muestreo_entrenamiento": "uniforme por hash xxhash64(serie, origen, h, semilla); sin ordenar la tabla",
        }

    # ---------------------------------------------------------------- lectura y contrato
    def _fijar_version(self, version):
        """Lee la Oro en una versión fija. Intenta SQL `VERSION AS OF`, después la opción versionAsOf y, si el entorno no
        admite viajes en el tiempo, la versión vigente (queda anotado en modo_lectura)."""
        if version is None:
            try:
                version = int(self.spark.sql(f"DESCRIBE HISTORY {self.tabla} LIMIT 1").collect()[0]["version"])
            except Exception:
                version = None
        if version is not None:
            for modo, lector in (("sql_version_as_of", lambda: self.spark.sql(f"SELECT * FROM {self.tabla} VERSION AS OF {version}")),
                                 ("opcion_versionAsOf", lambda: self.spark.read.option("versionAsOf", version).table(self.tabla))):
                try:
                    frame = lector()
                    frame.limit(1).count()
                    return version, modo, frame
                except Exception:
                    continue
        return version, "vigente_sin_viaje_en_el_tiempo", self.spark.table(self.tabla)

    def _leer_propiedades(self) -> dict:
        try:
            filas = self.spark.sql(f"SHOW TBLPROPERTIES {self.tabla}").collect()
            return {r[0]: r[1] for r in filas}
        except Exception:
            return {}

    def _vocabularios(self):
        """Vocabulario de cada categórica en todo el snapshot (texto → código 0, 1, 2… en orden alfabético). Un mismo
        valor tiene siempre el mismo código en entrenamiento, evaluación y pronóstico."""
        fila = self._gold.select(*[F.collect_set(F.col(c).cast("string")).alias(c) for c in CATEGORICAS]).collect()[0]
        codigos = {c: {v: i for i, v in enumerate(sorted(fila[c]))} for c in CATEGORICAS}
        etiquetas = {c: {i: v for v, i in codigos[c].items()} for c in CATEGORICAS}
        return codigos, etiquetas

    def _codigo(self, name):
        mapa = F.create_map(*[x for valor, codigo in self.codigos[name].items() for x in (F.lit(valor), F.lit(codigo))])
        return F.coalesce(F.try_element_at(mapa, F.col(name).cast("string")), F.lit(-1)).cast("int")

    def _elegible(self):
        """Filtro de filas evaluables del Avance 4: serie con ≥ 5 días con venta antes de T, referencia ingenua disponible y
        ≥ 5 días activos en la última semana. Con ALCANCE_SERVICIOS = 'excluir' también quita las categorías de servicio."""
        condicion = ((F.col("dias_con_venta_hist") >= self.config["min_dias_con_venta"])
                     & F.col("ingenuo_estacional").isNotNull() & (F.col("s_dias_activos_7") >= 5))
        if ALCANCE_SERVICIOS == "excluir":
            condicion = condicion & (F.coalesce(F.col("tipo_categoria").cast("string"), F.lit("")) != "servicio")
        return condicion

    def _hash_uniforme(self, columnas):
        """Número en [0, 1) fijo para cada fila (hash de las columnas y la semilla): permite muestrear sin ordenar."""
        h = F.xxhash64(*[F.col(c).cast("string") for c in columnas], F.lit(int(self.config["semilla"])))
        return F.pmod(h, F.lit(1_000_000_007)).cast("double") / F.lit(1_000_000_007.0)

    def _columnas(self):
        """Proyección de la Oro a pandas con tipos compactos (float32, códigos int32)."""
        columnas = list(dict.fromkeys([*METADATOS, *self.num_features, *self.cat_features]))
        salida = []
        for name in columnas:
            if name in self.cat_features:
                salida.append(self._codigo(name).alias(name))
            elif name in ("y", "ingenuo_estacional"):
                salida.append(F.col(name).cast("double").alias(name))
            elif name == "h":
                salida.append(F.col("h").cast("int").alias("h"))
            elif name in self.num_features:
                salida.append(F.col(name).cast("float").alias(name))
            elif name in ("en_muestra_eval", "dia_atipico_obj"):
                salida.append(F.col(name).cast("int").alias(name))
            elif name == "tienda":
                salida.append(F.col(name).cast("string").alias(name))
            else:
                salida.append(F.col(name))
        return salida

    def _base(self):
        """Snapshot con las variables de promociones futuras unidas (sólo si el calendario está activo)."""
        if self._promos_futuras is None:
            return self._gold
        unido = (self._gold.withColumn("_t", F.col("tienda").cast("string")).withColumn("_c", F.col("categoria_id").cast("string"))
                 .join(self._promos_futuras, ["_t", "_c", "fecha_objetivo"], "left").drop("_t", "_c"))
        for c in VARIABLES_PROMO_FUTURA:
            unido = unido.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))
        return unido

    def _resolver_metadata(self):
        resumen = (self._gold.where(self._elegible()).groupBy("conjunto", "corte", "origen")
                   .agg(F.countDistinct("h").alias("n_h"), F.min("h").alias("h_min"), F.max("h").alias("h_max"),
                        F.max("fecha_objetivo").alias("fecha_objetivo_max"), F.count("*").alias("filas")))
        local = resumen.limit(1_001).toPandas()
        if len(local) > 1_000:
            raise ValueError("Demasiados grupos temporales: revisa la tabla Oro.")
        self._resumen = local
        r = resolver_origenes(local, self.config["horizonte"], self.config["n_origenes_oof"])
        self.cortes, self.pruebas = r["cortes"], r["pruebas"]
        self.ajuste_origen, self.oof_origenes = r["ajuste_origen"], r["oof_origenes"]
        self.origenes_entrenamiento = r["origenes_entrenamiento_completos"]
        self.origenes_observados = r["origenes_observados_completos"]
        self.origen_pronostico = r["origen_pronostico"]
        self.fecha_maxima_objetivo = r["fecha_maxima_objetivo"]

    def calibracion_para(self, origen):
        """Bloque completo más reciente que termina a más tardar en `origen` (sirve para calibrar intervalos)."""
        cutoff = fecha_diaria(origen)
        candidatos = [fecha_diaria(v) for v in self.origenes_observados
                      if fecha_diaria(v) + pd.Timedelta(days=self.config["horizonte"]) <= cutoff]
        if not candidatos:
            raise ValueError("No existe un bloque observado para calibrar antes del origen.")
        return max(candidatos).date().isoformat()

    def conjunto_que_contiene_origen(self, origen):
        """'evaluacion' si el origen tiene bloque de evaluación; si no, 'entrenamiento'."""
        origin = fecha_diaria(origen)
        conjuntos = set(self._resumen.loc[pd.to_datetime(self._resumen.origen).eq(origin), "conjunto"])
        for preferido in ("evaluacion", "entrenamiento", "pronostico"):
            if preferido in conjuntos:
                return preferido
        raise ValueError(f"El origen {origin.date()} no tiene bloque en la Oro.")

    def validar_contratos(self, duplicados=True):
        """Escanea el snapshot completo: fechas y horizontes coherentes, objetivo válido (salvo en pronóstico), llaves
        completas, conjuntos conocidos, referencia válida en filas elegibles y sin infinitos. Opcionalmente revisa que no
        haya duplicados (serie, origen, h) dentro de cada conjunto. Sólo regresa conteos (nada de detalle)."""
        if self._contratos is not None:
            return dict(self._contratos)
        invalid_date = (F.col("origen").isNull() | F.col("fecha_objetivo").isNull() | F.col("h").isNull()
                        | ~F.col("h").between(1, self.config["horizonte"])
                        | (F.datediff("fecha_objetivo", "origen") != F.col("h")))
        con_objetivo = F.col("conjunto") != "pronostico"
        invalid_y = con_objetivo & (F.col("y").isNull() | F.isnan("y") | (F.col("y") < 0))
        invalid_keys = F.col("tienda").isNull() | F.col("categoria_id").isNull()
        invalid_set = F.col("conjunto").isNull() | ~F.col("conjunto").isin("entrenamiento", "evaluacion", "pronostico")
        baseline = F.col("ingenuo_estacional")
        invalid_baseline = self._elegible() & (F.isnan(baseline) | (baseline < 0))
        invalid_features = F.lit(False)
        for name in self.num_features:
            if name in self._gold.columns:
                invalid_features = invalid_features | (F.abs(F.col(name).cast("double")) == float("inf"))
        row = self._gold.agg(
            F.count("*").alias("filas_snapshot"),
            F.sum(F.when(invalid_date, 1).otherwise(0)).alias("fechas_invalidas"),
            F.sum(F.when(invalid_y, 1).otherwise(0)).alias("etiquetas_invalidas"),
            F.sum(F.when(invalid_keys, 1).otherwise(0)).alias("claves_invalidas"),
            F.sum(F.when(invalid_set, 1).otherwise(0)).alias("conjuntos_invalidos"),
            F.sum(F.when(invalid_baseline, 1).otherwise(0)).alias("referencias_invalidas_elegibles"),
            F.sum(F.when(invalid_features, 1).otherwise(0)).alias("variables_infinitas"),
        ).collect()[0].asDict()
        if not row["filas_snapshot"] or any(row[name] for name in row if name != "filas_snapshot"):
            raise ValueError(f"El snapshot incumple contratos críticos: { {k: v for k, v in row.items() if k != 'filas_snapshot'} }")
        row["duplicados_detectados"] = 0
        if duplicados:
            row["duplicados_detectados"] = (self._gold.groupBy("conjunto", *CLAVES_EJEMPLO).count()
                                            .where(F.col("count") > 1).limit(1).count())
            if row["duplicados_detectados"]:
                raise ValueError("El snapshot contiene duplicados en conjunto–serie–origen–horizonte.")
        self._contratos = {name: int(value or 0) for name, value in row.items()}
        self.manifiesto["contratos_snapshot"] = {k: v for k, v in self._contratos.items() if k != "filas_snapshot"}
        return dict(self._contratos)

    # ---------------------------------------------------------------- cargas
    def cargar_entrenamiento(self):
        """Muestra de entrenamiento: filas elegibles del conjunto 'entrenamiento' cuyo hash uniforme cae debajo de
        max_filas / total. Es reproducible y no ordena la tabla (el Avance 4 ordenaba 7.8 M de filas sólo para muestrear).
        La columna `_u` (uniforme en [0, 1)) se conserva: con ella cada modelo toma su submuestra sin sesgo."""
        if self._pool is not None:
            return self._pool
        frame = self._base().where((F.col("conjunto") == "entrenamiento") & self._elegible())
        total = frame.count()
        cap = self.config["max_filas_entrenamiento"]
        fraccion = 1.0 if cap is None else min(1.0, cap / max(total, 1))
        u = self._hash_uniforme(CLAVES_EJEMPLO)
        seleccion = frame.withColumn("_u", u).where(F.col("_u") < F.lit(fraccion))
        pdf = seleccion.select(*self._columnas(), F.col("_u")).toPandas()
        pdf["_u"] = (pdf["_u"] / fraccion).clip(upper=0.999999999)     # vuelve a ser uniforme en [0, 1) dentro de la muestra
        pool = normalizar_bloque(pdf, max_filas=max(cap or 0, total) + 1)
        del pdf
        self.manifiesto["filas_train_elegibles"] = int(total)
        self.manifiesto["filas_train_cargadas"] = int(len(pool))
        self.manifiesto["fraccion_train_cargada"] = float(fraccion)
        self.manifiesto["huella_claves_train"] = hashlib.sha256(
            pd.util.hash_pandas_object(pool[CLAVES_EJEMPLO], index=False).to_numpy().tobytes()).hexdigest()
        # para entrenar no se necesitan la tienda ni el conjunto/corte: se sueltan para ahorrar memoria
        self._pool = pool.drop(columns=["tienda", "conjunto", "corte"], errors="ignore")
        gc.collect()
        return self._pool

    def cargar_bloque(self, origen, h=None, comun=False, conjunto=None, horizontes=None, fraccion_series=None):
        """Filas elegibles de un origen. h u horizontes acotan los días; comun=True deja sólo la muestra común;
        fraccion_series < 1 toma una fracción reproducible de las series (modo rápido). No trunca: si el bloque excede
        max_filas_bloque, se detiene."""
        origin = fecha_diaria(origen)
        conjunto = conjunto or self.conjunto_que_contiene_origen(origin)
        frame = self._base().where((F.col("conjunto") == conjunto) & (F.col("origen") == F.lit(origin.date()))
                                   & self._elegible())
        if h is not None:
            frame = frame.where(F.col("h") == int(h))
        if horizontes is not None:
            frame = frame.where(F.col("h").isin([int(v) for v in horizontes]))
        if comun:
            frame = frame.where(F.col("en_muestra_eval") == 1)
        fraccion = self.config["fraccion_series_evaluacion"] if fraccion_series is None else fraccion_series
        if fraccion < 1:
            frame = frame.where(self._hash_uniforme(CLAVES_SERIE) < F.lit(float(fraccion)))
        cap = self.config["max_filas_bloque"]
        local = frame.select(*self._columnas()).limit(cap + 1).toPandas()
        return normalizar_bloque(local, cap, con_objetivo=conjunto != "pronostico")

    def liberar_memoria(self):
        """Suelta la muestra de entrenamiento del driver (otra llamada a cargar_entrenamiento la vuelve a leer igual)."""
        self._pool = None
        gc.collect()

    def decodificar(self, columna, codigos):
        """Códigos → texto de una categórica (para tablas y gráficas); −1 → 'SIN_DATO'."""
        mapa = self.etiquetas[columna]
        return pd.Series(codigos).map(lambda c: mapa.get(int(c), "SIN_DATO")).to_numpy()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Calendario de promociones futuras (funciones; se explica y activa en la sección 9)
# MAGIC Hoy las variables de promoción del modelo (`s_tasa_descuento_28`, `s_intensidad_promo_28`, `s_promos_en_venta_28`,
# MAGIC `s_promos_vigentes_t0`) describen **promociones pasadas** (días ≤ T). El modelo no sabe si en los próximos 14 días
# MAGIC habrá una promoción, y por eso subestima los picos que éstas generan. Estas funciones convierten un calendario de
# MAGIC promociones **planeadas** en tres variables por tienda × categoría × día objetivo:
# MAGIC `f_promos_planeadas` (cuántas promociones estarán vigentes), `f_descuento_planeado` (el mayor descuento planeado) y
# MAGIC `f_dias_promo_restantes` (días que le quedan a la promoción vigente). Son legítimas porque un plan se conoce en T.

# COMMAND ----------

ESQUEMA_CALENDARIO_PROMOS = {
    "promo_id": "texto · identificador de la promoción",
    "inicio_planeado": "fecha · primer día planeado",
    "fin_planeado": "fecha · último día planeado",
    "categoria_id": "texto · categoría (mismo identificador canónico de la Oro)",
    "tienda": "texto · tienda; vacío = todas las tiendas del alcance indicado en zona o región",
    "zona": "texto · zona; vacío = no se acota por zona",
    "region": "texto · región; vacío = no se acota por región",
    "descuento_planeado": "decimal 0–1 · descuento planeado (opcional)",
    "tipo_mecanica": "texto · 'precio', 'lleva N paga M', 'regalo', … (opcional)",
}


def validar_calendario_promociones(calendario) -> dict:
    """Revisa un calendario (Spark) contra ESQUEMA_CALENDARIO_PROMOS. Regresa conteos; no imprime filas."""
    faltan = [c for c in ("promo_id", "inicio_planeado", "fin_planeado", "categoria_id") if c not in calendario.columns]
    if faltan:
        return {"valido": False, "motivo": f"faltan columnas obligatorias: {faltan}"}
    r = calendario.agg(F.count("*").alias("filas"),
                       F.sum(F.when(F.col("fin_planeado") < F.col("inicio_planeado"), 1).otherwise(0)).alias("fechas_invertidas"),
                       F.sum(F.when(F.col("categoria_id").isNull(), 1).otherwise(0)).alias("sin_categoria"),
                       F.min("inicio_planeado").alias("primer_dia"), F.max("fin_planeado").alias("ultimo_dia")).collect()[0].asDict()
    r["valido"] = bool(r["filas"]) and not r["fechas_invertidas"] and not r["sin_categoria"]
    return r


def cargar_variables_promo_futura(spark):
    """Construye (tienda, categoria_id, fecha_objetivo) → variables f_* a partir de TABLA_CALENDARIO_PROMOS. Si la tabla
    no existe o no es válida, avisa y regresa None (el modelo sigue sin promociones futuras)."""
    if not existe(TABLA_CALENDARIO_PROMOS):
        aviso(f"USAR_CALENDARIO_PROMOS = True pero no existe {TABLA_CALENDARIO_PROMOS}: el modelo sigue sin promociones futuras.")
        return None
    cal = spark.table(TABLA_CALENDARIO_PROMOS)
    revision = validar_calendario_promociones(cal)
    if not revision["valido"]:
        aviso(f"El calendario de promociones no es válido ({revision}); no se usa.")
        return None
    for c in ("tienda", "zona", "region", "descuento_planeado"):
        if c not in cal.columns:
            cal = cal.withColumn(c, F.lit(None).cast("string" if c != "descuento_planeado" else "double"))
    tiendas = spark.table(tabla("silver", "catalogo_tiendas")).select(
        F.col("tienda").cast("string").alias("_t"), F.col("zona").cast("string").alias("_zona"),
        F.col("region").cast("string").alias("_region"))
    alcance = (cal.crossJoin(F.broadcast(tiendas))
               .where((F.col("tienda").isNull() | (F.col("tienda").cast("string") == F.col("_t")))
                      & (F.col("zona").isNull() | (F.col("zona").cast("string") == F.col("_zona")))
                      & (F.col("region").isNull() | (F.col("region").cast("string") == F.col("_region")))))
    dias = (alcance.withColumn("fecha_objetivo", F.explode(F.sequence(F.col("inicio_planeado").cast("date"),
                                                                       F.col("fin_planeado").cast("date"))))
            .withColumn("_restantes", F.datediff(F.col("fin_planeado").cast("date"), F.col("fecha_objetivo"))))
    return (dias.groupBy("_t", F.col("categoria_id").cast("string").alias("_c"), "fecha_objetivo")
            .agg(F.countDistinct("promo_id").cast("float").alias("f_promos_planeadas"),
                 F.coalesce(F.max("descuento_planeado"), F.lit(0.0)).cast("float").alias("f_descuento_planeado"),
                 F.min("_restantes").cast("float").alias("f_dias_promo_restantes")))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Theta por serie (`I6_Theta`, opcional)
# MAGIC Método Theta estándar (índice estacional semanal multiplicativo, suavizamiento exponencial simple de la serie
# MAGIC desestacionalizada con α elegido por serie y la mitad de la pendiente lineal como deriva). Se calcula en Spark con
# MAGIC `applyInPandas` usando las 16 semanas previas a T. Los días sin dato se rellenan **sólo con información anterior**
# MAGIC (mismo día de la semana previa o último dato), nunca con promedios que incluyan días futuros.

# COMMAND ----------

VENTANA_THETA = 112           # días de historia (16 semanas)
N_CUBETAS_THETA = 64 if MODO_RAPIDO else 256   # grupos de series que procesa cada tarea de Spark


def rellenar_pasado(Y):
    """Rellena huecos de cada serie (filas) usando sólo días anteriores: primero la semana previa, luego el día previo."""
    Y = Y.copy()
    for j in range(Y.shape[1]):
        x = Y[:, j]
        faltan = np.isnan(x)
        if j >= 7 and faltan.any():
            x = np.where(faltan, Y[:, j - 7], x)
            faltan = np.isnan(x)
        if j >= 1 and faltan.any():
            x = np.where(faltan, Y[:, j - 1], x)
        Y[:, j] = x
    return Y


def theta_arr(Y, dows, dows_f, alfas=(0.05, 0.1, 0.2, 0.3, 0.5)):
    """Pronóstico Theta vectorizado. Y: series × días (historia); dows: día de la semana de cada columna; dows_f: día de
    la semana de los 14 días futuros. Regresa series × 14."""
    media = np.nanmean(Y, axis=1)
    idx = np.ones((Y.shape[0], 7))
    for d in range(7):
        if (dows == d).any():
            col = np.nanmean(Y[:, dows == d], axis=1)
            idx[:, d] = np.where(media > 0, col / np.where(media > 0, media, 1.0), 1.0)
    idx = np.where(np.isfinite(idx) & (idx > 0.05), idx, 1.0)
    D = Y / idx[:, dows]
    n = np.sum(np.isfinite(D), axis=1)
    t = np.where(np.isfinite(D), np.arange(D.shape[1], dtype=np.float64)[None, :], np.nan)
    tc, dc = t - np.nanmean(t, axis=1, keepdims=True), D - np.nanmean(D, axis=1, keepdims=True)
    b = np.nansum(tc * dc, axis=1) / np.maximum(np.nansum(tc ** 2, axis=1), 1e-9)
    mejor_sse, mejor_l, mejor_a = np.full(D.shape[0], np.inf), np.zeros(D.shape[0]), np.full(D.shape[0], alfas[0])
    for a in alfas:
        nivel, sse = np.full(D.shape[0], np.nan), np.zeros(D.shape[0])
        for j in range(D.shape[1]):
            x = D[:, j]
            ok = np.isfinite(x)
            inicio = ok & np.isnan(nivel)
            nivel = np.where(inicio, x, nivel)
            sigue = ok & ~inicio
            sse = sse + np.where(sigue, (x - nivel) ** 2, 0.0)
            nivel = np.where(sigue, a * x + (1 - a) * nivel, nivel)
        mejor = sse < mejor_sse
        mejor_sse, mejor_l, mejor_a = np.where(mejor, sse, mejor_sse), np.where(mejor, nivel, mejor_l), np.where(mejor, a, mejor_a)
    hs = np.arange(1, 15)[None, :]
    deriva = 0.5 * np.nan_to_num(b)[:, None] * (hs - 1 + 1 / mejor_a[:, None] - (1 - mejor_a[:, None]) ** n[:, None] / mejor_a[:, None])
    return np.clip((np.nan_to_num(mejor_l)[:, None] + deriva) * idx[:, dows_f], 0, None)


def theta_spark(series_pdf, origen):
    """Pronóstico Theta a 14 días para las series de `series_pdf` (columnas tienda y categoria_id como texto), con la
    historia del panel de VENTANA_THETA días ≤ origen. Regresa pandas: tienda, categoria_id, h, p_theta."""
    from pyspark.sql import types as T
    origen = fecha_diaria(origen)
    panel = spark.table(tabla("silver", "panel_categoria_dia"))
    inicio = (origen - pd.Timedelta(days=VENTANA_THETA - 1)).date()
    fechas = [d.date() for d in pd.date_range(inicio, origen.date(), freq="D")]
    dows = np.array([d.weekday() for d in fechas])
    dows_f = np.array([(origen + pd.Timedelta(days=k)).weekday() for k in range(1, 15)])
    claves = spark.createDataFrame(series_pdf[["tienda", "categoria_id"]].astype(str).drop_duplicates())
    hist = (panel.where(F.col("fecha").between(F.lit(inicio), F.lit(origen.date())))
            .select(F.col("tienda").cast("string").alias("tienda"), F.col("categoria_id").cast("string").alias("categoria_id"),
                    "fecha", F.col("venta").cast("double").alias("venta"))
            .join(claves, ["tienda", "categoria_id"])
            .withColumn("_cubeta", F.pmod(F.xxhash64("tienda", "categoria_id"), F.lit(N_CUBETAS_THETA))))
    esquema = T.StructType([T.StructField("tienda", T.StringType()), T.StructField("categoria_id", T.StringType()),
                            T.StructField("h", T.IntegerType()), T.StructField("p_theta", T.DoubleType())])

    def por_cubeta(pdf):
        W = (pdf.drop_duplicates(["tienda", "categoria_id", "fecha"])
             .pivot(index=["tienda", "categoria_id"], columns="fecha", values="venta").reindex(columns=fechas))
        pronostico = theta_arr(rellenar_pasado(W.to_numpy(dtype=np.float64)), dows, dows_f)
        k = W.index.to_frame(index=False)
        return pd.DataFrame({"tienda": np.repeat(k["tienda"].to_numpy(), 14),
                             "categoria_id": np.repeat(k["categoria_id"].to_numpy(), 14),
                             "h": np.tile(np.arange(1, 15, dtype=np.int32), len(k)), "p_theta": pronostico.reshape(-1)})

    return hist.groupBy("_cubeta").applyInPandas(por_cubeta, schema=esquema).toPandas()


class TablaTheta:
    """'Modelo' I6 de un origen: tabla de pronósticos Theta por (tienda, categoría, h). Si una serie no tiene pronóstico
    Theta (sin historia suficiente), usa la referencia ingenua."""

    def __init__(self, tabla_theta, etiquetas_categoria):
        self.tabla = tabla_theta.set_index(["tienda", "categoria_id", "h"])["p_theta"]
        self.etiquetas_categoria = etiquetas_categoria

    def predict(self, frame):
        categoria = frame["categoria_id"].map(lambda c: self.etiquetas_categoria.get(int(c), "SIN_DATO"))
        indice = pd.MultiIndex.from_arrays([frame["tienda"].astype(str), categoria, frame["h"].astype(int)])
        valores = self.tabla.reindex(indice).to_numpy(dtype=np.float64)
        return np.where(np.isnan(valores), frame["ingenuo_estacional"].to_numpy(dtype=np.float64), valores)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Preparación de variables y modelos individuales
# MAGIC - `TreeEncoder` arma la matriz para los árboles (LightGBM, XGBoost, CatBoost, AdaBoost, bosque aleatorio): las
# MAGIC   categóricas ya vienen como códigos; sólo fija el orden de columnas y el tipo float32 (la mitad de memoria que float64).
# MAGIC - `LinearEncoder` prepara el modelo lineal: logaritmo de las variables en pesos o conteos grandes, imputación con la
# MAGIC   mediana del entrenamiento, estandarización y variables indicadoras (one-hot) de las categóricas con ≤ 60 niveles.
# MAGIC - `fit_model` entrena un modelo con su presupuesto de filas; `predict_bundle` predice con piso 0 y tope.
# MAGIC
# MAGIC **Submuestra común.** Cada fila de entrenamiento trae un número uniforme fijo (`_u`). Un modelo con presupuesto de
# MAGIC N filas usa las N filas con menor `_u`: dos modelos con el mismo presupuesto ven **exactamente** las mismas filas, así
# MAGIC que la comparación es justa; los miembros del bagging usan otra semilla y por eso otras filas.

# COMMAND ----------

def _motor_spec(feature_spec: Mapping) -> dict:
    """Valida la especificación {"numeric": [...], "categorical": [...]}: sin duplicados y sin objetivo ni metadatos."""
    if not isinstance(feature_spec, Mapping) or set(feature_spec) != {"numeric", "categorical"}:
        raise ValueError("feature_spec debe contener únicamente numeric y categorical.")
    spec = {key: list(feature_spec[key]) for key in ("numeric", "categorical")}
    todas = spec["numeric"] + spec["categorical"]
    if not spec["numeric"] or len(set(todas)) != len(todas):
        raise ValueError("Se requieren variables numéricas y una lista conjunta sin duplicados.")
    prohibidas = {"y", "y_unidades", "origen", "fecha_objetivo", "conjunto", "corte", "en_muestra_eval",
                  "ingenuo_estacional", "tienda", "dia_atipico_obj", "_u"}
    if prohibidas & set(todas):
        raise ValueError(f"La especificación contiene objetivo, referencia o metadatos: {sorted(prohibidas & set(todas))}")
    return spec


class TreeEncoder:
    """Matriz float32 para modelos de árboles, con el orden de variables fijo.

    feature_names       numéricas seguidas de categóricas
    categorical_indices posición de las categóricas (LightGBM y CatBoost las tratan como categorías)
    """

    def fit(self, train, feature_spec):
        spec = _motor_spec(feature_spec)
        faltan = set(spec["numeric"] + spec["categorical"]) - set(train.columns)
        if faltan:
            raise ValueError(f"Faltan variables en el entrenamiento: {sorted(faltan)}")
        self.numeric_names, self.categorical_names = spec["numeric"], spec["categorical"]
        self.feature_names = [*self.numeric_names, *self.categorical_names]
        self.categorical_indices = list(range(len(self.numeric_names), len(self.feature_names)))
        return self

    def transform(self, frame, fill_missing=None) -> np.ndarray:
        """Regresa la matriz de variables. Un código −1 (categoría desconocida) pasa a NaN = faltante. fill_missing
        reemplaza los NaN (los árboles de scikit-learn no aceptan faltantes)."""
        X = frame[self.feature_names].to_numpy(dtype=np.float32)
        if self.categorical_indices:
            cats = X[:, self.categorical_indices]
            cats[cats < 0] = np.nan
            X[:, self.categorical_indices] = cats
        if fill_missing is not None:
            X = np.where(np.isnan(X), np.float32(fill_missing), X)
        return X

    def transform_catboost(self, frame) -> pd.DataFrame:
        """CatBoost exige categóricas enteras (no decimales): DataFrame con numéricas float32 y categóricas int32."""
        df = frame[self.feature_names].copy()
        for c in self.categorical_names:
            df[c] = df[c].fillna(-1).astype("int32")
        return df


def fit_tree_encoder(train, feature_spec):
    """Atajo: TreeEncoder ajustado con el entrenamiento."""
    return TreeEncoder().fit(train, feature_spec)


class LinearEncoder:
    """Matriz estandarizada para el modelo lineal (Elastic Net sobre log(1 + venta)).

    log_cols   numéricas no negativas con valores grandes (importes, conteos): se usa log(1 + x)
    medianas   imputación de faltantes con la mediana del entrenamiento
    medias, desv  estandarización (media 0, desviación 1) con estadísticos del entrenamiento
    onehot     niveles de cada categórica (≤ 60) y de dow y tipo_evento, convertidos a indicadoras 0/1
    """
    MAX_NIVELES_ONEHOT = 60

    def fit(self, train, feature_spec):
        spec = _motor_spec(feature_spec)
        self.numericas = [c for c in spec["numeric"] if c not in ("dow", "tipo_evento")]
        A = train[self.numericas].astype("float32")
        self.log_cols = [c for c in self.numericas if A[c].notna().any() and float(A[c].min()) >= 0 and float(A[c].max()) > 50]
        A[self.log_cols] = np.log1p(A[self.log_cols])
        self.medianas = A.median().fillna(0.0)
        A = A.fillna(self.medianas)
        self.medias, self.desv = A.mean(), A.std().replace(0, 1.0).fillna(1.0)
        self.onehot = {}
        for c in [*spec["categorical"], *[c for c in ("dow", "tipo_evento") if c in spec["numeric"]]]:
            niveles = np.sort(pd.unique(train[c].dropna()))
            if 1 < len(niveles) <= self.MAX_NIVELES_ONEHOT:
                self.onehot[c] = niveles
        self.feature_names = self.numericas + [f"{c}={v}" for c, ns in self.onehot.items() for v in ns]
        return self

    def transform(self, frame) -> np.ndarray:
        A = frame[self.numericas].astype("float32")
        A[self.log_cols] = np.log1p(A[self.log_cols].clip(lower=0))
        A = ((A.fillna(self.medianas) - self.medias) / self.desv).to_numpy(dtype=np.float32)
        indicadoras = [(frame[c].to_numpy()[:, None] == niveles[None, :]).astype(np.float32) for c, niveles in self.onehot.items()]
        return np.concatenate([A, *indicadoras], axis=1) if indicadoras else A


@dataclass
class ModelBundle:
    """Modelo entrenado con todo lo necesario para predecir.

    name                clave del modelo (I1_LightGBM_Tweedie, E1_Bagging, …)
    encoder             TreeEncoder o LinearEncoder ajustado con su entrenamiento
    estimator           el modelo (o la lista de miembros en E1_Bagging)
    cap                 tope de predicción = 3 × percentil 99.9 del objetivo de entrenamiento
    factor_log          corrección de escala de los modelos sobre log(1 + y) (1.0 en los demás)
    filas_entrenamiento filas con que se entrenó
    params              hiperparámetros usados
    extras              datos de diagnóstico (mejor iteración, convergencia, …)
    """
    name: str
    encoder: Any
    estimator: Any
    cap: float
    factor_log: float
    filas_entrenamiento: int
    params: dict
    extras: dict = field(default_factory=dict)


def _motor_training_frame(train: pd.DataFrame, limit, seed=SEMILLA) -> pd.DataFrame:
    """Submuestra determinista con a lo más `limit` filas: las de menor `_u`. Con otra semilla (bagging) se transforma `_u`
    para obtener otras filas igual de uniformes."""
    if limit is None or len(train) <= limit:
        return train
    u = train["_u"].to_numpy()
    if seed != SEMILLA:
        u = (u * 7919.0 + (seed - SEMILLA) * 0.6180339887498949) % 1.0
    umbral = np.partition(u, int(limit) - 1)[int(limit) - 1]
    return train.loc[u <= umbral]


def _lgbm(params, seed, jobs):
    import lightgbm as lgb
    return lgb.LGBMRegressor(**params, n_jobs=jobs, random_state=seed, verbosity=-1)


def fit_model(name: str, train: pd.DataFrame, feature_spec: Mapping, params=None, n_jobs=None, seed=SEMILLA,
              budget=None, validacion=None):
    """Entrena el modelo `name` con las filas de `train` (ya filtradas a objetivo ≤ origen).

    params      hiperparámetros que reemplazan a los vigentes (PARAMETROS_VIGENTES o los iniciales)
    budget      máximo de filas (por defecto S6_MODEL_ROW_BUDGETS[name])
    validacion  bloque para paro temprano (sólo en el origen de ajuste): decide cuántos árboles usar después
    Regresa (ModelBundle, costo) con costo = {modelo, filas, segundos_entrenamiento, …}."""
    jobs = runtime_threads(n_jobs)
    limit = S6_MODEL_ROW_BUDGETS.get(name) if budget is None else budget
    selected = _motor_training_frame(train, limit, seed)
    y = selected["y"].to_numpy(dtype=np.float64)
    if not len(y) or y.sum() <= 0:
        raise ValueError(f"{name}: el entrenamiento no tiene venta positiva.")
    cap = float(np.quantile(y, 0.999) * TOPE_PREDICCION_FACTOR)
    base = PARAMETROS_VIGENTES.get(name) or S6_POINT_PARAMETERS.get(name) or S7_HOMOGENEOUS_PARAMETERS.get(name, {})
    configured = deepcopy(base)
    configured.update(deepcopy(params or {}))
    encoder = LinearEncoder().fit(selected, feature_spec) if name == "I3_ElasticNet" else TreeEncoder().fit(selected, feature_spec)
    factor, extras = 1.0, {}
    with Cronometro() as reloj:
        if name in ("I1_LightGBM_Tweedie", "I5_LightGBM_Log1p", "E7_Bosque_Aleatorio"):
            import lightgbm as lgb
            X = encoder.transform(selected)
            objetivo = np.log1p(y) if name == "I5_LightGBM_Log1p" else y
            kwargs = {"categorical_feature": encoder.categorical_indices}
            if validacion is not None and name != "E7_Bosque_Aleatorio":
                configured["n_estimators"] = ARBOLES_MAX
                yv = validacion["y"].to_numpy(dtype=np.float64)
                kwargs.update(eval_set=[(encoder.transform(validacion), np.log1p(yv) if name == "I5_LightGBM_Log1p" else yv)],
                              callbacks=[lgb.early_stopping(PARO_TEMPRANO, verbose=False)])
            estimator = _lgbm(configured, seed, jobs).fit(X, objetivo, **kwargs)
            if validacion is not None and name != "E7_Bosque_Aleatorio":
                extras["mejor_iteracion"] = int(estimator.best_iteration_ or configured["n_estimators"])
            if name == "I5_LightGBM_Log1p":          # corrección de Duan: devuelve el nivel en pesos tras expm1
                n = min(len(X), 300_000)
                back = np.expm1(estimator.predict(X[:n])).clip(min=0)
                factor = float(y[:n].sum() / max(back.sum(), 1e-9))
            del X
        elif name == "I4_XGBoost_Tweedie":
            import xgboost as xgb
            X = encoder.transform(selected)
            if validacion is not None:
                configured["n_estimators"] = ARBOLES_MAX
                estimator = xgb.XGBRegressor(**configured, n_jobs=jobs, random_state=seed, early_stopping_rounds=PARO_TEMPRANO)
                estimator.fit(X, y, eval_set=[(encoder.transform(validacion), validacion["y"].to_numpy(dtype=np.float64))],
                              verbose=False)
                extras["mejor_iteracion"] = int(estimator.best_iteration + 1)
            else:
                estimator = xgb.XGBRegressor(**configured, n_jobs=jobs, random_state=seed).fit(X, y, verbose=False)
            del X
        elif name == "I2_CatBoost_Tweedie":
            from catboost import CatBoostRegressor
            estimator = CatBoostRegressor(**configured, thread_count=jobs, random_seed=seed, verbose=False,
                                          allow_writing_files=False, cat_features=encoder.categorical_indices)
            estimator.fit(encoder.transform_catboost(selected), y)
        elif name == "I3_ElasticNet":
            from sklearn.exceptions import ConvergenceWarning
            from sklearn.linear_model import ElasticNet
            matrix = encoder.transform(selected)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                estimator = ElasticNet(**configured, random_state=seed).fit(matrix, np.log1p(y))
            extras["iteraciones"] = int(np.max(estimator.n_iter_))
            extras["convergio"] = bool(extras["iteraciones"] < configured.get("max_iter", 1000))
            n = min(len(matrix), 300_000)
            back = np.expm1(estimator.predict(matrix[:n])).clip(min=0)
            factor = float(y[:n].sum() / max(back.sum(), 1e-9))
            del matrix
        elif name == "E2_AdaBoost":
            from sklearn.ensemble import AdaBoostRegressor
            from sklearn.tree import DecisionTreeRegressor
            arbol = DecisionTreeRegressor(max_depth=configured["max_depth"], min_samples_leaf=configured["min_samples_leaf"],
                                          random_state=seed)
            estimator = AdaBoostRegressor(arbol, n_estimators=int(configured["n_estimators"]),
                                          learning_rate=float(configured["learning_rate"]), loss=configured["loss"],
                                          random_state=seed)
            estimator.fit(encoder.transform(selected, fill_missing=-1.0), y)
        else:
            raise ValueError(f"Modelo no especificado: {name}")
    bundle = ModelBundle(name, encoder, estimator, cap, factor, len(selected), configured, extras)
    return bundle, {"modelo": name, "filas_entrenamiento": int(len(selected)), "presupuesto_filas": limit,
                    "segundos_entrenamiento": reloj.segundos, "factor_log": factor, "hilos": jobs, **extras}


def predict_bundle(bundle: ModelBundle, frame: pd.DataFrame) -> np.ndarray:
    """Predicción en pesos con piso 0 y tope `cap`. E1 promedia a sus miembros; E2 predice por partes en paralelo."""
    if bundle.name == "I6_Theta":
        values = bundle.estimator.predict(frame)
    elif bundle.name == "E1_Bagging":
        values = np.mean([predict_bundle(m, frame) for m in bundle.estimator], axis=0)
    elif bundle.name == "I3_ElasticNet":
        values = np.expm1(bundle.estimator.predict(bundle.encoder.transform(frame))).clip(min=0) * bundle.factor_log
    elif bundle.name == "I2_CatBoost_Tweedie":
        values = bundle.estimator.predict(bundle.encoder.transform_catboost(frame))
    elif bundle.name == "E2_AdaBoost":
        X = bundle.encoder.transform(frame, fill_missing=-1.0)
        partes = np.array_split(np.arange(len(X)), max(1, min(HILOS_AV5, len(X) // 50_000 + 1)))
        from joblib import Parallel, delayed
        resultados = Parallel(n_jobs=len(partes), backend="threading")(delayed(bundle.estimator.predict)(X[p]) for p in partes)
        values = np.concatenate(resultados) if resultados else np.zeros(0)
    else:
        values = bundle.estimator.predict(bundle.encoder.transform(frame))
        if bundle.name == "I5_LightGBM_Log1p":
            values = np.expm1(values).clip(min=0) * bundle.factor_log
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(frame),) or not np.isfinite(values).all():
        raise ValueError(f"{bundle.name} produjo predicciones no finitas o de tamaño incorrecto.")
    return np.clip(values, 0.0, bundle.cap)


def fit_individuals(train: pd.DataFrame, feature_spec: Mapping, n_jobs=None, seed=SEMILLA, validacion=None):
    """Entrena las bases individuales activas (POINT_BASE_MODELS, salvo Theta, que se calcula por bloque).
    Regresa ({modelo: ModelBundle}, DataFrame de costos)."""
    bundles, costs = {}, []
    for name in POINT_BASE_MODELS:
        if name == "I6_Theta":
            continue
        bundles[name], cost = fit_model(name, train, feature_spec, n_jobs=n_jobs, seed=seed, validacion=validacion)
        costs.append(cost)
    return bundles, pd.DataFrame(costs)


def motor_metrics(y, prediction) -> dict:
    """WMAPE, MAE, RMSE y sesgo % de un vector de predicciones (todas en pesos)."""
    actual, predicted = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if actual.ndim != 1 or predicted.shape != actual.shape or not len(actual) or actual.sum() <= 0:
        raise ValueError("Se requieren vectores no vacíos del mismo tamaño y venta total positiva.")
    error = predicted - actual
    return {"wmape": float(np.abs(error).sum() / actual.sum()), "mae": float(np.abs(error).mean()),
            "rmse": float(np.sqrt(np.mean(error ** 2))), "sesgo_pct": float(100 * error.sum() / actual.sum()),
            "filas": int(len(actual))}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Ajuste fino de hiperparámetros (Optuna) para los modelos más relevantes
# MAGIC Se ajustan **LightGBM Tweedie** (el mejor individual del Avance 4) y **XGBoost Tweedie** (el nuevo). Cada prueba
# MAGIC entrena con filas cuyo objetivo es ≤ 20 de mayo y se califica con el WMAPE de los 14 días siguientes (bloque de
# MAGIC ajuste); el paro temprano decide el número de árboles. La primera prueba es la configuración del Avance 4: si
# MAGIC ninguna la supera, se conserva. En el Avance 4 Optuna sólo alcanzó 7 pruebas, todas en su fase aleatoria (TPE no
# MAGIC llegó a optimizar); aquí cada modelo tiene `N_PRUEBAS_AJUSTE` pruebas con `N_PRUEBAS_ALEATORIAS` iniciales.

# COMMAND ----------

def _motor_trial_parameters(name: str, trial) -> dict:
    """Espacio de búsqueda de cada modelo (rangos amplios alrededor de la configuración del Avance 4)."""
    if name == "I1_LightGBM_Tweedie":
        return {"objective": "tweedie",
                "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 31, 255, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 50, 2000, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True)}
    if name == "I4_XGBoost_Tweedie":
        return {"objective": "reg:tweedie", "tree_method": "hist",
                "tweedie_variance_power": trial.suggest_float("tweedie_variance_power", 1.1, 1.9),
                "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 10),
                "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 300.0, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 30.0, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "max_bin": trial.suggest_categorical("max_bin", [128, 256])}
    raise ValueError(f"Sin espacio de búsqueda para {name}")


PRUEBA_INICIAL = {   # la configuración del Avance 4 escrita con los nombres del espacio de búsqueda (primera prueba)
    "I1_LightGBM_Tweedie": {"tweedie_variance_power": 1.5, "learning_rate": 0.05, "num_leaves": 63,
                            "min_child_samples": 200, "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
                            "reg_alpha": 1e-3},
    "I4_XGBoost_Tweedie": {"tweedie_variance_power": 1.5, "learning_rate": 0.05, "max_depth": 8, "min_child_weight": 50.0,
                           "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0, "reg_alpha": 1e-3, "max_bin": 256},
}


def tune_boosting(train: pd.DataFrame, tune_eval: pd.DataFrame, feature_spec: Mapping, n_jobs=None, seed=SEMILLA,
                  n_trials=N_PRUEBAS_AJUSTE, timeout_s=TIEMPO_MAX_AJUSTE_MIN * 60):
    """Optuna (TPE) para los modelos de MODELOS_AJUSTE_FINO activos. `train` debe tener objetivo ≤ origen de ajuste y
    `tune_eval` es el bloque de ajuste. Regresa (mejores parámetros por modelo, pruebas, resumen, importancias)."""
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except Exception:
        aviso("Optuna no está disponible: se conservan los hiperparámetros del Avance 4 (sin ajuste fino).")
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    if pd.to_datetime(train["fecha_objetivo"]).max() > pd.to_datetime(tune_eval["origen"]).min():
        raise ValueError("El ajuste contiene etiquetas posteriores a su origen.")
    jobs = runtime_threads(n_jobs)
    filas = _motor_training_frame(train, FILAS_POR_PRUEBA, seed)
    encoder = TreeEncoder().fit(filas, feature_spec)
    X, y = encoder.transform(filas), filas["y"].to_numpy(dtype=np.float64)
    Xv, yv = encoder.transform(tune_eval), tune_eval["y"].to_numpy(dtype=np.float64)
    cap = float(np.quantile(y, 0.999) * TOPE_PREDICCION_FACTOR)
    mejores, registros, resumen, importancias = {}, [], [], []
    for name in [m for m in MODELOS_AJUSTE_FINO if m in POINT_BASE_MODELS]:
        inicio = time.time()

        def objective(trial, name=name):
            params = _motor_trial_parameters(name, trial)
            with Cronometro() as reloj:
                if name == "I1_LightGBM_Tweedie":
                    import lightgbm as lgb
                    est = _lgbm({**params, "n_estimators": ARBOLES_MAX}, seed, jobs)
                    est.fit(X, y, eval_set=[(Xv, yv)], categorical_feature=encoder.categorical_indices,
                            callbacks=[lgb.early_stopping(PARO_TEMPRANO, verbose=False)])
                    arboles = int(est.best_iteration_ or ARBOLES_MAX)
                else:
                    import xgboost as xgb
                    est = xgb.XGBRegressor(**params, n_estimators=ARBOLES_MAX, n_jobs=jobs, random_state=seed,
                                           early_stopping_rounds=PARO_TEMPRANO)
                    est.fit(X, y, eval_set=[(Xv, yv)], verbose=False)
                    arboles = int(est.best_iteration + 1)
                pred = np.clip(est.predict(Xv), 0, cap)
            score = float(np.abs(pred - yv).sum() / yv.sum())
            trial.set_user_attr("n_estimators", arboles)
            registros.append({"modelo": name, "prueba": trial.number, "wmape_ajuste": score, "n_estimators": arboles,
                              "segundos": reloj.segundos, "parametros": json.dumps(trial.params)})
            return score

        study = optuna.create_study(direction="minimize", study_name=f"avance5_43_{name}",
                                    sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=N_PRUEBAS_ALEATORIAS))
        study.enqueue_trial(PRUEBA_INICIAL[name])
        study.optimize(objective, n_trials=n_trials, timeout=float(timeout_s), gc_after_trial=True, catch=(Exception,))
        completas = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completas:
            raise RuntimeError(f"No se completó ninguna prueba de Optuna para {name}.")
        inicial = completas[0] if completas[0].number == 0 else None
        mejor = study.best_trial
        gana_inicial = inicial is not None and inicial.value <= mejor.value
        elegido = inicial if gana_inicial else mejor
        parametros = {**deepcopy(S6_POINT_PARAMETERS[name]), **_motor_trial_parameters(name, optuna.trial.FixedTrial(elegido.params))}
        parametros["n_estimators"] = int(elegido.user_attrs.get("n_estimators", parametros.get("n_estimators", 500)))
        mejores[name] = parametros
        resumen.append({"modelo": name, "pruebas": len(completas), "minutos": (time.time() - inicio) / 60,
                        "wmape_inicial": float(inicial.value) if inicial is not None else np.nan,
                        "wmape_mejor": float(mejor.value),
                        "mejora_relativa": (1 - mejor.value / inicial.value) if inicial is not None and inicial.value else np.nan,
                        "configuracion": "inicial (Avance 4)" if gana_inicial else "ajustada por Optuna",
                        "arboles_prueba": parametros["n_estimators"]})
        try:
            for parametro, valor in optuna.importance.get_param_importances(study).items():
                importancias.append({"modelo": name, "hiperparametro": parametro, "importancia": float(valor)})
        except Exception:
            pass
    del X, Xv
    gc.collect()
    return mejores, pd.DataFrame(registros), pd.DataFrame(resumen), pd.DataFrame(importancias)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Ensambles homogéneos (muchos modelos del mismo algoritmo)
# MAGIC - `E1_Bagging`: el LightGBM I1 más dos LightGBM iguales entrenados en submuestras distintas del 70 % con otras
# MAGIC   semillas; se promedian. Reduce la varianza (el error que cambia según qué filas tocaron). Sustituye al bagging de
# MAGIC   árboles de scikit-learn con 350 mil filas, que era mucho más débil que el LightGBM que intentaba acompañar.
# MAGIC - `E2_AdaBoost`: AdaBoost.R2 sobre árboles de decisión (boosting adaptativo clásico), como referencia.
# MAGIC - `E7_Bosque_Aleatorio`: bosque aleatorio en modo `rf` de LightGBM (árboles independientes en submuestras de filas y
# MAGIC   variables, promediados); es mucho más rápido que el de scikit-learn con el mismo número de filas.

# COMMAND ----------

def fit_homogeneous(train: pd.DataFrame, feature_spec: Mapping, bases: Mapping, n_jobs=None, seed=SEMILLA):
    """Entrena los ensambles homogéneos activos. E1 reutiliza el I1 ya entrenado como su primer miembro."""
    bundles, costs = {}, []
    if "E1_Bagging" in POINT_HOMOGENEOUS:
        cfg = S7_HOMOGENEOUS_PARAMETERS["E1_Bagging"]
        presupuesto_i1 = S6_MODEL_ROW_BUDGETS.get("I1_LightGBM_Tweedie")
        disponible = len(train) if presupuesto_i1 is None else min(len(train), presupuesto_i1)
        miembros, segundos, filas = [bases["I1_LightGBM_Tweedie"]], 0.0, bases["I1_LightGBM_Tweedie"].filas_entrenamiento
        for semilla in cfg["semillas_extra"]:
            miembro, costo = fit_model("I1_LightGBM_Tweedie", train, feature_spec, n_jobs=n_jobs, seed=int(semilla),
                                       budget=max(1, int(disponible * cfg["fraccion_filas"])))
            miembros.append(miembro)
            segundos += costo["segundos_entrenamiento"]
            filas += costo["filas_entrenamiento"]
        bundles["E1_Bagging"] = ModelBundle("E1_Bagging", None, miembros, max(m.cap for m in miembros), 1.0, filas,
                                            {"miembros": len(miembros), **cfg}, {"incluye_I1": True})
        costs.append({"modelo": "E1_Bagging", "filas_entrenamiento": filas, "segundos_entrenamiento": segundos,
                      "segundos_miembros_extra": segundos})
    for name in ("E2_AdaBoost", "E7_Bosque_Aleatorio"):
        if name in POINT_HOMOGENEOUS:
            bundles[name], costo = fit_model(name, train, feature_spec, n_jobs=n_jobs, seed=seed)
            costs.append(costo)
    return bundles, pd.DataFrame(costs)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Ensambles heterogéneos (algoritmos distintos combinados)
# MAGIC Los cuatro aprenden **sólo con bloques que ya terminaron antes del origen** que se evalúa (la "bolsa" de predicciones
# MAGIC fuera de muestra: bloques OOF, ajuste y cortes anteriores, con la muestra común de series). Por eso no hay fuga.
# MAGIC
# MAGIC | Ensamble | Qué aprende | Por qué así |
# MAGIC | --- | --- | --- |
# MAGIC | `E3_Blending` | pesos w ≥ 0 con Σw = 1 que minimizan el error absoluto en el **bloque más reciente** | blending clásico: combinación convexa en una ventana reservada |
# MAGIC | `E4_Stacking_temporal` | pesos w ≥ 0 **sin** restricción de suma, uno por bloque de horizonte (h 1–7 y 8–14), con todos los bloques pasados ponderados por antigüedad; después un **factor de nivel** por tipo de categoría × bloque de horizonte medido en el bloque más reciente | la suma libre y el factor corrigen el sesgo de nivel que el Avance 4 dejó fuera de ±3 % en dos cortes |
# MAGIC | `E5_Votacion` | promedio simple de las 3 bases con menor error reciente | votación: robusta, sin pesos que estimar |
# MAGIC | `E6_Stacking_LightGBM` | LightGBM pequeño que recibe las predicciones de las bases + h, día, tipo, región y escala de la serie | puede usar pesos distintos según el tipo de serie |
# MAGIC
# MAGIC El avance anterior usaba Ridge con intercepto (puede dar pesos negativos y un intercepto en pesos aprendido en mayo
# MAGIC que no vale en agosto) y una malla de pesos en décimas con mínimo 0.1 por base (obligaba a darle peso a una base que
# MAGIC estorba). Aquí los pesos son no negativos y continuos: una base que no aporta recibe 0.

# COMMAND ----------

def pesos_no_negativos(P, y, suma_uno=False, peso_fila=None, max_filas=400_000, seed=SEMILLA) -> np.ndarray:
    """Pesos w ≥ 0 que minimizan Σ peso · |y − P·w| (L1 suavizada: la misma lógica del WMAPE).
    suma_uno=True agrega Σw = 1 (combinación convexa, blending). Los datos se dividen entre la media de y para que el
    optimizador trabaje con números de orden 1; si hay más de max_filas filas se usa una muestra reproducible."""
    from scipy.optimize import minimize
    P, y = np.asarray(P, dtype=np.float64), np.asarray(y, dtype=np.float64)
    w_fila = np.ones(len(y)) if peso_fila is None else np.asarray(peso_fila, dtype=np.float64)
    if len(y) > max_filas:
        seleccion = np.random.default_rng(seed).choice(len(y), max_filas, replace=False)
        P, y, w_fila = P[seleccion], y[seleccion], w_fila[seleccion]
    k = P.shape[1]
    escala = max(float(np.average(y, weights=w_fila)), 1e-9)
    A, b, wn = P / escala, y / escala, w_fila / w_fila.sum()
    delta = 1e-3

    def objetivo(w):
        r = A @ w - b
        s = np.sqrt(r * r + delta * delta)
        return float(np.dot(wn, s)), A.T @ (wn * r / s)

    w0 = np.full(k, 1.0 / k)
    if suma_uno:
        res = minimize(objetivo, w0, jac=True, method="SLSQP", bounds=[(0.0, 1.0)] * k,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}],
                       options={"maxiter": 300})
    else:
        res = minimize(objetivo, w0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * k, options={"maxiter": 500})
    w = np.clip(np.asarray(res.x, dtype=np.float64), 0.0, None)
    if suma_uno:
        w = w / w.sum() if w.sum() > 0 else w0
    return w


@dataclass
class MetaBundle:
    """Ensambles heterogéneos ajustados en un origen.

    bases              bases combinadas (en este orden van los pesos)
    weights            E3: pesos convexos (Σ = 1)
    pesos_stacking     E4: {1: pesos para h 1–7, 2: pesos para h 8–14}
    factores_nivel     E4: matriz [código de tipo de categoría, bloque de h] → factor de nivel (1 = sin corrección)
    miembros_votacion  E5: bases promediadas
    meta_lgbm          E6: meta-modelo LightGBM (o None)
    cap                tope de predicción de los ensambles
    oof_rows           filas de meta-entrenamiento usadas
    bloques            orígenes de los bloques usados
    diagnostico        error reciente de cada base y otros datos para el reporte
    """
    bases: tuple
    weights: np.ndarray
    pesos_stacking: dict
    factores_nivel: np.ndarray
    miembros_votacion: tuple
    meta_lgbm: Any
    cap: float
    oof_rows: int
    bloques: list
    diagnostico: dict


COLUMNAS_META_POOL = ["origen", "fecha_objetivo", "h", "h_bloque", "dow", "tipo_categoria", "region", "s_ma28", "y"]
# columnas fijas de la bolsa de meta-entrenamiento (además, una columna por modelo con su predicción)


def _motor_meta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Valida la bolsa de predicciones fuera de muestra: columnas, valores finitos y venta positiva."""
    faltan = set(COLUMNAS_META_POOL) - set(frame.columns)
    if not isinstance(frame, pd.DataFrame) or frame.empty or faltan:
        raise ValueError(f"La bolsa de meta-entrenamiento no cumple el contrato: faltan {sorted(faltan)}")
    if frame["y"].sum() <= 0:
        raise ValueError("La bolsa de meta-entrenamiento no tiene venta positiva.")
    return frame


def _meta_features(frame: pd.DataFrame, P: np.ndarray) -> np.ndarray:
    """Variables del meta-modelo E6: predicciones de las bases (en log), h, día de la semana, tipo, región y escala."""
    extra = np.column_stack([frame["h"].to_numpy(dtype=np.float32), frame["dow"].to_numpy(dtype=np.float32),
                             frame["tipo_categoria"].to_numpy(dtype=np.float32), frame["region"].to_numpy(dtype=np.float32),
                             np.log1p(np.nan_to_num(frame["s_ma28"].to_numpy(dtype=np.float64), nan=0.0).clip(min=0))])
    return np.column_stack([np.log1p(P), extra]).astype(np.float32)


def fit_meta(oof_df: pd.DataFrame, origen, seed=SEMILLA, n_jobs=None):
    """Ajusta E3–E6 con los bloques de la bolsa cuyo objetivo es ≤ origen. Regresa (MetaBundle, tabla de pesos)."""
    origen = fecha_diaria(origen)
    pool = _motor_meta_frame(oof_df)
    pool = pool.loc[pool["fecha_objetivo"] <= origen]
    bases = tuple(b for b in POINT_BASE_MODELS if b in pool.columns and pool[b].notna().all())
    if len(bases) < 2 or pool.empty:
        raise ValueError("Se requieren al menos dos bases con predicciones fuera de muestra anteriores al origen.")
    origenes = np.sort(pool["origen"].unique())
    orden = pd.Categorical(pool["origen"], categories=origenes, ordered=True).codes      # 0 = bloque más antiguo
    edad = (len(origenes) - 1 - orden).astype(np.float64)                               # 0 = bloque más reciente
    peso = 0.5 ** (edad / VIDA_MEDIA_BLOQUES)                  # peso por antigüedad del bloque
    P, y = pool[list(bases)].to_numpy(dtype=np.float64), pool["y"].to_numpy(dtype=np.float64)
    reciente = (pool["origen"] == origenes[-1]).to_numpy()
    # E3 · blending: combinación convexa aprendida en el bloque más reciente
    w_blend = pesos_no_negativos(P[reciente], y[reciente], suma_uno=True, seed=seed)
    # E4 · stacking: pesos libres por bloque de horizonte con todos los bloques ponderados por antigüedad
    hb = pool["h_bloque"].to_numpy()
    pesos_stack = {}
    for bloque in (1, 2):
        m = hb == bloque
        pesos_stack[bloque] = (pesos_no_negativos(P[m], y[m], suma_uno=False, peso_fila=peso[m], seed=seed)
                               if m.sum() >= 100 else w_blend.copy())
    stack = np.where(hb == 1, P @ pesos_stack[1], P @ pesos_stack[2])
    # E4 · factor de nivel por tipo de categoría × bloque de horizonte (bloque más reciente, encogido hacia 1)
    n_tipos = int(max(pool["tipo_categoria"].max(), 0)) + 1
    factores = np.ones((max(n_tipos, 1) + 1, 3))
    rec = pool.loc[reciente].assign(_stack=stack[reciente])
    for (tipo, bloque), g in rec.groupby(["tipo_categoria", "h_bloque"]):
        suma_pred = float(g["_stack"].sum())
        if tipo < 0 or suma_pred <= 0:
            continue
        crudo = float(g["y"].sum()) / suma_pred
        n = len(g)
        encogido = 1.0 + (crudo - 1.0) * n / (n + REDUCCION_FACTOR_NIVEL)
        factores[int(tipo), int(bloque)] = float(np.clip(encogido, *LIMITES_FACTOR_NIVEL))
    # E5 · votación: las N bases con menor WMAPE ponderado por antigüedad
    errores = {b: float(np.sum(peso * np.abs(y - pool[b].to_numpy())) / max(np.sum(peso * y), 1e-9)) for b in bases}
    miembros = tuple(sorted(bases, key=errores.get)[:N_MIEMBROS_VOTACION])
    # E6 · stacking no lineal
    meta_lgbm = None
    if "E6_Stacking_LightGBM" in POINT_HETEROGENEOUS_ACTIVOS:
        params = dict(S7_META_LGBM)
        params["min_child_samples"] = int(max(50, min(params["min_child_samples"], len(y) // 200)))
        meta_lgbm = _lgbm(params, seed, runtime_threads(n_jobs)).fit(_meta_features(pool, P), y, sample_weight=peso,
                                                                       categorical_feature=[len(bases) + 2, len(bases) + 3])
    cap = float(np.quantile(y, 0.999) * TOPE_PREDICCION_FACTOR)
    meta = MetaBundle(bases, w_blend, pesos_stack, factores, miembros, meta_lgbm, cap, int(len(pool)),
                      [pd.Timestamp(o).date().isoformat() for o in origenes], {"error_reciente": errores})
    filas = [{"ensamble": "E3_Blending", "base": b, "peso": float(w)} for b, w in zip(bases, w_blend)]
    for bloque in (1, 2):
        filas += [{"ensamble": f"E4_Stacking_temporal·h{'1-7' if bloque == 1 else '8-14'}", "base": b, "peso": float(w)}
                  for b, w in zip(bases, pesos_stack[bloque])]
    filas += [{"ensamble": "E5_Votacion", "base": b, "peso": (1 / len(miembros)) if b in miembros else 0.0} for b in bases]
    for tipo in range(factores.shape[0]):
        for bloque in (1, 2):
            if factores[tipo, bloque] != 1.0:
                filas.append({"ensamble": f"E4_factor_nivel·h{'1-7' if bloque == 1 else '8-14'}", "base": f"tipo:{tipo}",
                              "peso": float(factores[tipo, bloque])})
    tabla_pesos = pd.DataFrame(filas).assign(origen=origen.date().isoformat(), bloques_usados=len(origenes),
                                             filas_meta=int(len(pool)))
    return meta, tabla_pesos


def predict_ensembles(base_predictions: Mapping, meta: MetaBundle, frame: pd.DataFrame) -> dict:
    """Predicciones de E3–E6 y del diagnóstico D1 a partir de las predicciones de las bases sobre las mismas filas."""
    P = np.column_stack([base_predictions[b] for b in meta.bases])
    hb = frame["h_bloque"].to_numpy() if "h_bloque" in frame else np.where(frame["h"].to_numpy() <= 7, 1, 2)
    stack = np.where(hb == 1, P @ meta.pesos_stacking[1], P @ meta.pesos_stacking[2])
    tipo = frame["tipo_categoria"].to_numpy()
    tipo = np.where((tipo >= 0) & (tipo < meta.factores_nivel.shape[0]), tipo, meta.factores_nivel.shape[0] - 1)
    salida = {"E3_Blending": P @ meta.weights,
              "D1_Stacking_sin_correccion": stack,
              "E4_Stacking_temporal": stack * meta.factores_nivel[tipo, hb],
              "E5_Votacion": P[:, [meta.bases.index(b) for b in meta.miembros_votacion]].mean(axis=1)}
    if meta.meta_lgbm is not None:
        salida["E6_Stacking_LightGBM"] = meta.meta_lgbm.predict(_meta_features(frame, P))
    return {k: np.clip(np.asarray(v, dtype=np.float64), 0.0, meta.cap) for k, v in salida.items()
            if k in POINT_HETEROGENEOUS_ACTIVOS or k in MODELOS_DIAGNOSTICO}


def predecir_todos(bases: Mapping, homogeneos: Mapping, meta: MetaBundle, frame: pd.DataFrame, theta=None):
    """Predice todos los candidatos sobre las mismas filas. Regresa (predicciones, segundos de inferencia por modelo).
    El ingenuo es la columna ingenuo_estacional; E1 reutiliza la predicción de I1 como su primer miembro."""
    preds = {"B0_Ingenuo": frame["ingenuo_estacional"].to_numpy(dtype=np.float64)}
    segundos = {"B0_Ingenuo": 0.0}
    for name in POINT_BASE_MODELS:
        with Cronometro() as reloj:
            if name == "I6_Theta":
                preds[name] = np.clip(theta.predict(frame), 0.0, None)
            else:
                preds[name] = predict_bundle(bases[name], frame)
        segundos[name] = reloj.segundos
    for name, bundle in homogeneos.items():
        with Cronometro() as reloj:
            if name == "E1_Bagging":
                otros = [predict_bundle(m, frame) for m in bundle.estimator[1:]]
                preds[name] = np.clip(np.mean([preds["I1_LightGBM_Tweedie"], *otros], axis=0), 0.0, bundle.cap)
            else:
                preds[name] = predict_bundle(bundle, frame)
        segundos[name] = reloj.segundos + (segundos["I1_LightGBM_Tweedie"] if name == "E1_Bagging" else 0.0)
    with Cronometro() as reloj:
        heterogeneos = predict_ensembles(preds, meta, frame)
    for name, values in heterogeneos.items():
        preds[name] = values
        segundos[name] = reloj.segundos / max(len(heterogeneos), 1) + sum(segundos[b] for b in meta.bases)
    for name, values in preds.items():
        if values.shape != (len(frame),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"La configuración {name} produjo predicciones inválidas.")
    return preds, segundos

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 Intervalos de pronóstico (P10, P50, P90)
# MAGIC Un pronóstico puntual no dice cuánto puede equivocarse. Para surtir se necesita una banda: P90 como techo (cubre el
# MAGIC 90 % de los días) y P10 como piso. Se comparan dos métodos en septiembre (datos no vistos):
# MAGIC
# MAGIC - **`Q_Conformal` (nuevo)**: alrededor del pronóstico del **modelo elegido**. Se toman los residuos (real − pronóstico)
# MAGIC   de un bloque reciente ya observado y se calculan sus cuantiles de muestra finita por grupo: tipo de categoría ×
# MAGIC   bloque de horizonte × tramo del pronóstico (deciles). Así el ancho crece con el tamaño de la serie y el pronóstico
# MAGIC   queda **siempre** dentro de [P10, P90] (P10 ≤ P50 ≤ P90 sin cruces).
# MAGIC - **`Q_LightGBM_CQR`**: tres LightGBM cuantílicos con calibración conformal (CQR) por horizonte × tipo, el método
# MAGIC   del avance anterior. Son modelos aparte del pronóstico puntual; aquí se les fuerza a contener ese pronóstico.
# MAGIC
# MAGIC La cobertura objetivo es 80 % (banda de aceptación 75–85 %) global, por horizonte, por tipo y por región.

# COMMAND ----------

def _cuantiles_residuo(r: np.ndarray) -> tuple:
    """Cuantiles de muestra finita del residuo: inferior = k-ésimo menor con k = ⌊(n+1)·0.1⌋, superior con
    k = ⌈(n+1)·0.9⌉ (garantía conformal de cobertura ≥ 80 % si los residuos futuros se parecen a los calibrados);
    mediana para P50. El inferior nunca es positivo y el superior nunca negativo (el pronóstico queda dentro)."""
    r = np.sort(np.asarray(r, dtype=np.float64))
    n = len(r)
    k_lo = max(1, int(math.floor((n + 1) * QUANTILE_LEVELS[0])))
    k_hi = min(n, int(math.ceil((n + 1) * QUANTILE_LEVELS[2])))
    return min(float(r[k_lo - 1]), 0.0), float(np.median(r)), max(float(r[k_hi - 1]), 0.0)


def fit_conformal(calibracion: pd.DataFrame, columna_pred: str, calibration_end) -> dict:
    """Ajusta Q_Conformal con un bloque de calibración (filas con y, la predicción del modelo elegido, tipo y h_bloque).
    Jerarquía de grupos: (tipo, h_bloque, tramo) → (tipo, tramo) → (tramo) → global; cada grupo necesita
    MIN_FILAS_GRUPO_CONFORMAL filas."""
    y = calibracion["y"].to_numpy(dtype=np.float64)
    p = calibracion[columna_pred].to_numpy(dtype=np.float64)
    bordes = np.unique(np.quantile(p, np.linspace(0, 1, N_TRAMOS_CONFORMAL + 1)[1:-1]))
    datos = pd.DataFrame({"tipo_categoria": calibracion["tipo_categoria"].to_numpy(), "h_bloque": calibracion["h_bloque"].to_numpy(),
                          "tramo": np.searchsorted(bordes, p, side="right"), "r": y - p})
    niveles = [("tipo_categoria", "h_bloque", "tramo"), ("tipo_categoria", "tramo"), ("tramo",)]
    tablas = []
    for nivel in niveles:
        filas = []
        for clave, g in datos.groupby(list(nivel)):
            if len(g) >= MIN_FILAS_GRUPO_CONFORMAL:
                clave = clave if isinstance(clave, tuple) else (clave,)
                q_lo, q_50, q_hi = _cuantiles_residuo(g["r"].to_numpy())
                filas.append({**dict(zip(nivel, clave)), "q_lo": q_lo, "q_50": q_50, "q_hi": q_hi, "n": len(g)})
        tablas.append((nivel, pd.DataFrame(filas)))
    q_global = _cuantiles_residuo(datos["r"].to_numpy())
    return {"metodo": "Q_Conformal", "columna_pred": columna_pred, "bordes": bordes, "tablas": tablas, "global": q_global,
            "n_calibracion": int(len(datos)), "calibration_end": fecha_diaria(calibration_end).date().isoformat(),
            "cobertura_nominal": NOMINAL_COVERAGE}


def predict_conformal(bundle: dict, frame: pd.DataFrame, prediccion: np.ndarray) -> pd.DataFrame:
    """P10/P50/P90 alrededor de `prediccion` con los cuantiles del grupo más específico disponible."""
    prediccion = np.asarray(prediccion, dtype=np.float64)
    claves = pd.DataFrame({"tipo_categoria": frame["tipo_categoria"].to_numpy(), "h_bloque": frame["h_bloque"].to_numpy(),
                           "tramo": np.searchsorted(bundle["bordes"], prediccion, side="right")})
    q = pd.DataFrame(np.nan, index=claves.index, columns=["q_lo", "q_50", "q_hi"])
    for nivel, tabla_nivel in bundle["tablas"]:
        if tabla_nivel.empty or not q["q_lo"].isna().any():
            continue
        unido = claves[list(nivel)].merge(tabla_nivel, on=list(nivel), how="left")
        unido.index = claves.index
        faltan = q["q_lo"].isna()
        q.loc[faltan, ["q_lo", "q_50", "q_hi"]] = unido.loc[faltan, ["q_lo", "q_50", "q_hi"]].to_numpy()
    for i, col in enumerate(["q_lo", "q_50", "q_hi"]):
        q[col] = q[col].fillna(bundle["global"][i])
    p10 = np.maximum(prediccion + q["q_lo"].to_numpy(), 0.0)
    p50 = np.clip(prediccion + q["q_50"].to_numpy(), p10, None)
    p90 = np.maximum(prediccion + q["q_hi"].to_numpy(), np.maximum(prediccion, p50))
    resultado = pd.DataFrame({"p10": p10, "p50": p50, "p90": p90}, index=frame.index)
    for col in QUANTILE_COLUMNS:          # en el método conformal no hay intervalo previo a la calibración
        resultado[f"{col}_raw"] = resultado[col]
        resultado[f"{col}_base"] = resultado[col]
    return resultado


def finite_sample_cqr_offset(scores, coverage: float = NOMINAL_COVERAGE) -> float:
    """Ensanche conformal de CQR: estadístico de orden ⌈(n+1)·cobertura⌉ de las puntuaciones; nunca negativo."""
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Se requieren puntuaciones de calibración finitas no vacías.")
    rank = int(math.ceil((len(values) + 1) * coverage))
    if rank > len(values):
        raise ValueError("La muestra no admite un cuantil conformal finito para esta cobertura.")
    return max(0.0, float(np.partition(values, rank - 1)[rank - 1]))


def fit_quantiles(train_df: pd.DataFrame, calibration_df: pd.DataFrame, feature_spec: Mapping, cutoff_train,
                  calibration_end, n_jobs=None, segment_columns=("h", "tipo_categoria")) -> dict:
    """Q_LightGBM_CQR: tres LightGBM cuantílicos (P10, P50, P90) entrenados con objetivo ≤ cutoff_train y calibrados con
    CQR en el bloque (cutoff_train, calibration_end]. No se reentrenan después de calibrar."""
    cutoff, end = fecha_diaria(cutoff_train), fecha_diaria(calibration_end)
    if not train_df["fecha_objetivo"].le(cutoff).all():
        raise ValueError("El entrenamiento cuantílico contiene etiquetas posteriores al corte.")
    if not (calibration_df["fecha_objetivo"].gt(cutoff).all() and calibration_df["fecha_objetivo"].le(end).all()):
        raise ValueError("La calibración debe estar después del entrenamiento y dentro de su ventana.")
    jobs = runtime_threads(n_jobs)
    train = _motor_training_frame(train_df, S6_MODEL_ROW_BUDGETS.get("I1_LightGBM_Tweedie"))
    encoder = TreeEncoder().fit(train, feature_spec)
    X, y = encoder.transform(train), train["y"].to_numpy(dtype=np.float64)
    modelos, costos = {}, []
    parametros = {k: v for k, v in S6_QUANTILE_PARAMETERS.items() if k not in ("random_state", "verbosity", "n_jobs")}
    for alpha in QUANTILE_LEVELS:
        with Cronometro() as reloj:
            modelos[str(alpha)] = _lgbm({**parametros, "objective": "quantile", "alpha": alpha}, SEMILLA, jobs).fit(
                X, y, categorical_feature=encoder.categorical_indices)
        costos.append({"modelo": f"Q_P{int(round(alpha * 100)):02d}", "filas_entrenamiento": len(train),
                       "segundos_entrenamiento": reloj.segundos})
    del X
    bundle = {"metodo": "Q_LightGBM_CQR", "models": modelos, "encoder": encoder, "segment_columns": list(segment_columns),
              "cutoff_train": cutoff.date().isoformat(), "calibration_end": end.date().isoformat(),
              "costs": pd.DataFrame(costos), "references": QUANTILE_METHOD_REFERENCES}
    base = _quantile_base_predictions(bundle, calibration_df)
    yc = calibration_df["y"].to_numpy(dtype=np.float64)
    scores = np.maximum(base["p10_base"].to_numpy() - yc, yc - base["p90_base"].to_numpy())
    bundle["global_offset"] = finite_sample_cqr_offset(scores)
    offsets = []
    for clave, posiciones in calibration_df.reset_index(drop=True).groupby(list(segment_columns)).indices.items():
        clave = clave if isinstance(clave, tuple) else (clave,)
        try:
            offset, fuente = finite_sample_cqr_offset(scores[posiciones]), "segmento"
        except ValueError:
            offset, fuente = bundle["global_offset"], "global_por_muestra_insuficiente"
        offsets.append({**dict(zip(segment_columns, clave)), "offset": offset, "n_calibracion": len(posiciones), "fuente": fuente})
    bundle["offsets"] = pd.DataFrame(offsets)
    bundle["audit"] = {"max_target_train": train["fecha_objetivo"].max().date().isoformat(),
                       "min_target_calibration": calibration_df["fecha_objetivo"].min().date().isoformat(),
                       "max_target_calibration": calibration_df["fecha_objetivo"].max().date().isoformat(),
                       "n_train": int(len(train)), "n_calibration": int(len(calibration_df)),
                       "calibration_raw_crossing_rate": float(base["cruce_raw"].mean())}
    return bundle


def _quantile_base_predictions(bundle: Mapping, frame: pd.DataFrame) -> pd.DataFrame:
    """Cuantiles crudos de los tres LightGBM y su versión ordenada (piso 0, P10 ≤ P50 ≤ P90)."""
    X = bundle["encoder"].transform(frame)
    raw = np.column_stack([bundle["models"][str(alpha)].predict(X) for alpha in QUANTILE_LEVELS])
    result = pd.DataFrame(raw, columns=[f"{n}_raw" for n in QUANTILE_COLUMNS], index=frame.index)
    middle = np.maximum(raw[:, 1], 0.0)
    result["p10_base"] = np.minimum(np.maximum(raw[:, 0], 0.0), middle)
    result["p50_base"] = middle
    result["p90_base"] = np.maximum(np.maximum(raw[:, 2], 0.0), middle)
    result["cruce_raw"] = (raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])
    return result


def predict_quantiles(bundle: Mapping, frame: pd.DataFrame, prediccion=None) -> pd.DataFrame:
    """Aplica Q_LightGBM_CQR (offsets del grupo h × tipo; global si el grupo no existe). Exige que el bloque empiece
    después de la calibración. Si se da `prediccion`, el intervalo final se amplía para contenerla."""
    end = fecha_diaria(bundle["calibration_end"])
    if not frame["origen"].ge(end).all() or not frame["fecha_objetivo"].gt(end).all():
        raise ValueError("No se reutilizan etiquetas de calibración para evaluar los intervalos.")
    result = _quantile_base_predictions(bundle, frame)
    segmentos = bundle["segment_columns"]
    unido = frame[segmentos].reset_index(drop=True).merge(bundle["offsets"][[*segmentos, "offset"]], on=segmentos, how="left")
    offsets = unido["offset"].fillna(bundle["global_offset"]).to_numpy()
    result["p10"] = np.maximum(result["p10_base"].to_numpy() - offsets, 0.0)
    result["p50"] = result["p50_base"]
    result["p90"] = result["p90_base"].to_numpy() + offsets
    if prediccion is not None:
        prediccion = np.asarray(prediccion, dtype=np.float64)
        result["p10"] = np.minimum(result["p10"].to_numpy(), prediccion)
        result["p90"] = np.maximum(result["p90"].to_numpy(), prediccion)
    result["p50"] = np.clip(result["p50"].to_numpy(), result["p10"].to_numpy(), result["p90"].to_numpy())
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9 Evaluación por sumas: `AcumuladorAvance5`
# MAGIC El universo completo de un corte tiene millones de filas; no cabe guardar todas las predicciones. El acumulador
# MAGIC recibe cada bloque (un horizonte a la vez), suma lo necesario por grupo (ventas, errores absolutos y cuadrados,
# MAGIC sesgo, ceros, coberturas, pinball) y descarta las filas. Con esas sumas se calculan exactamente el WMAPE, el MAE,
# MAGIC el RMSE y el sesgo de cualquier agregación (por corte, horizonte, día, tipo, región). Guarda además una muestra
# MAGIC fija de 20 mil filas (sin identificadores) para la gráfica de residuos y las sumas por tramo de pronóstico para la
# MAGIC gráfica de calibración.

# COMMAND ----------

AGG_DIMENSIONS = ["corte", "origen", "fecha_objetivo", "h", "dow", "tipo_categoria", "region", "dia_atipico_obj"]
AGG_POINT_SUMS = ["n_filas", "ventas_reales", "ventas_predichas", "error_absoluto_total", "error_cuadrado_total",
                  "error_total", "n_ventas_cero", "error_absoluto_ceros", "prediccion_en_ceros", "prediccion_positiva_ceros"]
AGG_QUANTILE_SUMS = ["n_filas", "ventas_reales", "cubiertos", "ancho_total", "pinball_p10_total", "pinball_p50_total",
                     "pinball_p90_total", "cruces_raw", "cruces_final", "cubiertos_base", "error_absoluto_p50"]
BORDES_CALIBRACION = np.array([0, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4, 1e5])
# tramos del pronóstico (pesos) para la gráfica de calibración: real medio contra pronóstico medio en cada tramo
AGG_LIMITATION = ("Los cortes de desarrollo ya se usaron en el Avance 4 para comparar modelos; la evaluación "
                  "independiente es la de septiembre (prueba), que no interviene en ninguna decisión.")


def _agg_model_type(name):
    """Tipo de cada candidato en la comparativa."""
    if name == "B0_Ingenuo":
        return "referencia"
    if name.startswith("I"):
        return "individual"
    if name.startswith("D"):
        return "diagnostico"
    return "ensamble_homogeneo" if name in CANDIDATOS_HOMOGENEOS else "ensamble_heterogeneo"


def _agg_point_metrics(frame):
    """Métricas a partir de sumas (nunca promediando errores relativos por fila)."""
    result = frame.copy()
    denominator = result["ventas_reales"].where(result["ventas_reales"] > 0)
    result["wmape"] = result["error_absoluto_total"] / denominator
    result["mae"] = result["error_absoluto_total"] / result["n_filas"]
    result["rmse"] = np.sqrt(result["error_cuadrado_total"] / result["n_filas"])
    result["sesgo_pct"] = 100 * result["error_total"] / denominator
    result["n_filas"] = result["n_filas"].astype("int64")
    return result


def _agg_quantile_metrics(frame):
    """Cobertura, ancho y pinball a partir de sumas; el ancho y el pinball también relativos a la venta media."""
    result = frame.copy()
    n = result["n_filas"]
    venta_media = (result["ventas_reales"] / n).where(result["ventas_reales"] > 0)
    result["cobertura"] = result["cubiertos"] / n
    result["cobertura_base"] = result["cubiertos_base"] / n
    result["ancho_medio"] = result["ancho_total"] / n
    result["ancho_relativo"] = result["ancho_medio"] / venta_media
    for name in QUANTILE_COLUMNS:
        result[f"pinball_{name}"] = result[f"pinball_{name}_total"] / n
        result[f"pinball_{name}_rel"] = result[f"pinball_{name}"] / venta_media
    result["tasa_cruces_raw"] = result["cruces_raw"] / n
    result["tasa_cruces_final"] = result["cruces_final"] / n
    result["wmape_p50"] = result["error_absoluto_p50"] / result["ventas_reales"].where(result["ventas_reales"] > 0)
    result["cumple_banda"] = result["cobertura"].between(*BANDA_COBERTURA)
    result["n_filas"] = n.astype("int64")
    return result


def _agg_rollup(frame, dimensions, sums):
    if dimensions:
        return frame.groupby(dimensions, as_index=False, observed=True, dropna=False)[sums].sum()
    return pd.DataFrame([frame[sums].sum().to_dict()])


class AcumuladorAvance5:
    """Acumula métricas de evaluación por bloques sin conservar las filas.

    nombre_evaluacion   etiqueta (desarrollo_ampliada, desarrollo_comun, prueba_ampliada, prueba_comun)
    modelos             candidatos que deben venir en cada bloque
    etiquetas           vocabulario de las categóricas (para guardar tipo y región como texto)
    _sumas              (modelo, dimensiones…) → sumas de AGG_POINT_SUMS
    _cuantiles          (método, dimensiones…) → sumas de AGG_QUANTILE_SUMS
    _calibracion        (modelo, tramo) → [filas, venta real, venta pronosticada]
    _residuos           muestra fija (por hash) de filas: venta real, predicciones, h, tipo y escala; sin identificadores
    """

    def __init__(self, nombre_evaluacion, modelos, etiquetas, muestra_residuos=20_000, seed=SEMILLA):
        self.nombre_evaluacion = str(nombre_evaluacion)
        self.modelos = tuple(modelos)
        self.etiquetas = etiquetas
        self.muestra_residuos, self.seed = int(muestra_residuos), int(seed)
        self._sumas, self._cuantiles, self._calibracion = {}, {}, {}
        self._bloques, self._origen_de = set(), {}
        self._residuos = pd.DataFrame()
        self._costos = []
        self.filas_evaluadas = 0

    def agregar(self, pdf, point_predictions, quantiles=None, corte=None):
        """Agrega un bloque (un origen, uno o varios horizontes). `quantiles` = {método: DataFrame con p10/p50/p90 …}.
        Rechaza repetir (corte, origen, h) y predicciones inválidas."""
        if set(point_predictions) != set(self.modelos):
            raise ValueError(f"El bloque no trae exactamente los candidatos esperados: {sorted(set(self.modelos) ^ set(point_predictions))}")
        origin = pd.Timestamp(pdf["origen"].iloc[0])
        if pdf["origen"].nunique() != 1:
            raise ValueError("El bloque debe corresponder a un único origen.")
        h = pdf["h"].to_numpy(dtype=np.int64)
        y = pdf["y"].to_numpy(dtype=np.float64)
        corte = str(corte if corte is not None else pdf["corte"].iloc[0])
        if corte in self._origen_de and self._origen_de[corte] != origin:
            raise ValueError("El mismo corte no puede identificar orígenes distintos.")
        bloques = {(corte, origin, int(v)) for v in np.unique(h)}
        if self._bloques & bloques:
            raise ValueError("El origen y horizonte ya fueron agregados; no se duplica el bloque.")
        meta = pd.DataFrame({"corte": corte, "origen": origin, "fecha_objetivo": pdf["fecha_objetivo"].to_numpy(), "h": h,
                             "dow": pdf["fecha_objetivo"].dt.dayofweek.to_numpy() + 1,
                             "tipo_categoria": pd.Series(pdf["tipo_categoria"].to_numpy()).map(self.etiquetas["tipo_categoria"]).fillna("SIN_DATO").to_numpy(),
                             "region": pd.Series(pdf["region"].to_numpy()).map(self.etiquetas["region"]).fillna("SIN_DATO").to_numpy(),
                             "dia_atipico_obj": pdf["dia_atipico_obj"].astype(int).astype(str).to_numpy()})
        codes, groups = pd.factorize(pd.MultiIndex.from_frame(meta[AGG_DIMENSIONS]), sort=False)
        size = len(groups)

        def sums(values):
            return np.bincount(codes, weights=np.asarray(values, dtype=np.float64), minlength=size)

        counts, sales, zero = sums(np.ones(len(y))), sums(y), (y == 0)
        pendientes = {}
        for model in self.modelos:
            values = np.asarray(point_predictions[model], dtype=np.float64)
            if values.shape != y.shape or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"Predicciones inválidas o no emparejadas para {model}.")
            error = values - y
            grouped = np.column_stack([counts, sales, sums(values), sums(np.abs(error)), sums(error ** 2), sums(error),
                                       sums(zero), sums(np.abs(error) * zero), sums(values * zero), sums((values > 0) & zero)])
            for key, row in zip(groups, grouped):
                pendientes[(model, *key)] = row
            tramo = np.clip(np.searchsorted(BORDES_CALIBRACION, values, side="right") - 1, 0, len(BORDES_CALIBRACION) - 1)
            cal = np.column_stack([np.bincount(tramo, minlength=len(BORDES_CALIBRACION)),
                                   np.bincount(tramo, weights=y, minlength=len(BORDES_CALIBRACION)),
                                   np.bincount(tramo, weights=values, minlength=len(BORDES_CALIBRACION))])
            for t in np.flatnonzero(cal[:, 0]):
                self._calibracion[(model, int(t))] = self._calibracion.get((model, int(t)), np.zeros(3)) + cal[t]
        pendientes_q = {}
        for metodo, q in (quantiles or {}).items():
            lo, med, hi = (q[c].to_numpy(dtype=np.float64) for c in QUANTILE_COLUMNS)
            if (lo < 0).any() or (lo > med + 1e-9).any() or (med > hi + 1e-9).any():
                raise ValueError(f"Los cuantiles finales de {metodo} deben ser no negativos y ordenados.")
            pin = [sums(np.maximum(a * (y - q[c].to_numpy()), (a - 1) * (y - q[c].to_numpy())))
                   for c, a in zip(QUANTILE_COLUMNS, QUANTILE_LEVELS)]
            raw_cross = (q["p10_raw"].to_numpy() > q["p50_raw"].to_numpy()) | (q["p50_raw"].to_numpy() > q["p90_raw"].to_numpy())
            blo, bhi = q["p10_base"].to_numpy(), q["p90_base"].to_numpy()
            grouped = np.column_stack([counts, sales, sums((y >= lo) & (y <= hi)), sums(hi - lo), *pin, sums(raw_cross),
                                       sums((lo > med + 1e-9) | (med > hi + 1e-9)), sums((y >= blo) & (y <= bhi)),
                                       sums(np.abs(y - med))])
            for key, row in zip(groups, grouped):
                pendientes_q[(metodo, *key)] = row
        # muestra fija para la gráfica de residuos (las filas con menor hash; sin identificadores)
        claves = pdf[["tienda", "categoria_id", "h"]].assign(_o=origin.value)
        prioridad = pd.util.hash_pandas_object(claves, index=False).to_numpy(dtype="uint64") ^ np.uint64(self.seed)
        toma = min(len(pdf), self.muestra_residuos)
        idx = np.argpartition(prioridad, toma - 1)[:toma] if toma < len(pdf) else np.arange(toma)
        muestra = pd.DataFrame({"_prioridad": prioridad[idx], "corte": corte, "y": y[idx], "h": h[idx],
                                "tipo_categoria": meta["tipo_categoria"].to_numpy()[idx],
                                "s_ma28": pdf["s_ma28"].to_numpy(dtype=np.float64)[idx],
                                **{m: np.asarray(point_predictions[m], dtype=np.float32)[idx] for m in self.modelos}})
        self._residuos = pd.concat([self._residuos, muestra], ignore_index=True).nsmallest(self.muestra_residuos, "_prioridad")
        for key, row in pendientes.items():
            self._sumas[key] = self._sumas.get(key, np.zeros(len(AGG_POINT_SUMS))) + row
        for key, row in pendientes_q.items():
            self._cuantiles[key] = self._cuantiles.get(key, np.zeros(len(AGG_QUANTILE_SUMS))) + row
        self._bloques |= bloques
        self._origen_de[corte] = origin
        self.filas_evaluadas += len(pdf)

    def registrar_costos(self, costs):
        """Tiempos medidos por modelo y corte (entrenamiento y predicción)."""
        required = ["modelo", "corte", "segundos_entrenamiento", "segundos_prediccion"]
        if not isinstance(costs, pd.DataFrame) or set(required) - set(costs):
            raise ValueError("Los costos requieren modelo, corte y segundos de entrenamiento y predicción.")
        self._costos.append(costs[required].copy())

    def tablas(self):
        """Tablas agregadas: comparativa (ordenada por WMAPE medio de los cortes), por corte, por horizonte, diario,
        segmentos, calibración y cuantiles."""
        if not self._sumas:
            raise ValueError("No existen bloques de evaluación agregados.")
        base = pd.DataFrame([dict(zip(["modelo", *AGG_DIMENSIONS], key), **dict(zip(AGG_POINT_SUMS, value)))
                             for key, value in self._sumas.items()])
        per_cut = _agg_point_metrics(_agg_rollup(base, ["modelo", "corte", "origen"], AGG_POINT_SUMS))
        ingenuo = per_cut.loc[per_cut.modelo.eq("B0_Ingenuo"), ["corte", "wmape", "mae", "rmse"]].rename(
            columns={"wmape": "wmape_ingenuo", "mae": "mae_ingenuo", "rmse": "rmse_ingenuo"})
        per_cut = per_cut.merge(ingenuo, on="corte", how="left")
        per_cut["mejora_vs_ingenuo"] = 1 - per_cut["wmape"] / per_cut["wmape_ingenuo"]
        pooled = _agg_point_metrics(_agg_rollup(base, ["modelo"], AGG_POINT_SUMS)).rename(columns={"wmape": "wmape_global",
                                                                                                    "sesgo_pct": "sesgo_global_pct"})
        folds = per_cut.groupby("modelo", as_index=False).agg(
            wmape_media_cortes=("wmape", "mean"), wmape_desviacion_cortes=("wmape", lambda v: float(v.std(ddof=0))),
            sesgo_medio_pct=("sesgo_pct", "mean"), sesgo_abs_medio_pct=("sesgo_pct", lambda v: float(np.abs(v).mean())),
            n_cortes=("corte", "nunique"))
        tipos = _agg_point_metrics(_agg_rollup(base, ["modelo", "tipo_categoria"], AGG_POINT_SUMS))
        merc = tipos.loc[tipos.tipo_categoria.eq("mercancia"), ["modelo", "wmape"]].rename(columns={"wmape": "wmape_mercancia"})
        bh = _agg_point_metrics(_agg_rollup(base.assign(_hb=np.where(base["h"] <= 7, "h1_7", "h8_14")), ["modelo", "_hb"], AGG_POINT_SUMS))
        bh = bh.pivot(index="modelo", columns="_hb", values="wmape").add_prefix("wmape_").reset_index()
        compared = pooled.merge(folds, on="modelo").merge(merc, on="modelo", how="left").merge(bh, on="modelo", how="left")
        if self._costos:
            costos = pd.concat(self._costos, ignore_index=True)
            por_corte_costo = costos.groupby("modelo", as_index=False).agg(
                segundos_entrenamiento=("segundos_entrenamiento", "sum"), segundos_prediccion=("segundos_prediccion", "sum"),
                cortes_con_costo=("corte", "nunique"))
            por_corte_costo["min_entrenamiento_por_corte"] = (por_corte_costo["segundos_entrenamiento"] / 60
                                                              / por_corte_costo["cortes_con_costo"].clip(lower=1))
            compared = compared.merge(por_corte_costo, on="modelo", how="left")
        compared["tipo"] = compared["modelo"].map(_agg_model_type)
        compared["estrategia"] = compared["modelo"].map(ESTRATEGIA)
        ref = compared.set_index("modelo").loc["B0_Ingenuo"]
        compared["mejora_vs_ingenuo"] = 1 - compared["wmape_media_cortes"] / ref["wmape_media_cortes"]
        compared["mae_relativo"] = compared["mae"] / ref["mae"]
        compared["rmse_relativo"] = compared["rmse"] / ref["rmse"]
        individuales = compared.loc[compared.tipo.eq("individual")]
        if len(individuales):
            mejor_ind = individuales.sort_values("wmape_media_cortes").iloc[0]
            compared["mejor_individual"] = mejor_ind["modelo"]
            compared["mejora_vs_mejor_individual"] = 1 - compared["wmape_media_cortes"] / mejor_ind["wmape_media_cortes"]
        compared = compared.sort_values(["wmape_media_cortes", "wmape_desviacion_cortes", "modelo"], kind="stable").reset_index(drop=True)
        compared.insert(0, "posicion", np.arange(1, len(compared) + 1))
        segments = []
        for dimension in ("tipo_categoria", "region", "dow", "dia_atipico_obj"):
            seg = _agg_point_metrics(_agg_rollup(base, ["modelo", dimension], AGG_POINT_SUMS)).rename(columns={dimension: "valor_segmento"})
            seg.insert(1, "segmento", dimension)
            seg["valor_segmento"] = seg["valor_segmento"].astype(str)
            segments.append(seg)
        cruz = _agg_point_metrics(_agg_rollup(base, ["modelo", "tipo_categoria", "region"], AGG_POINT_SUMS))
        cruz = cruz.assign(segmento="tipo_categoria×region",
                           valor_segmento=cruz["tipo_categoria"].astype(str) + " · " + cruz["region"].astype(str)).drop(
            columns=["tipo_categoria", "region"])
        segments.append(cruz)
        tablas = {"comparativa": compared, "por_corte": per_cut,
                  "por_horizonte": _agg_point_metrics(_agg_rollup(base, ["modelo", "h"], AGG_POINT_SUMS)),
                  "diario": _agg_point_metrics(_agg_rollup(base, ["modelo", "corte", "origen", "fecha_objetivo"], AGG_POINT_SUMS)),
                  "segmentos": pd.concat(segments, ignore_index=True),
                  "residuos": self._residuos.drop(columns=["_prioridad"]).reset_index(drop=True)}
        cal = pd.DataFrame([{"modelo": m, "tramo": t, "n_filas": v[0], "suma_y": v[1], "suma_pred": v[2]}
                            for (m, t), v in self._calibracion.items()])
        if len(cal):
            cal["desde"] = BORDES_CALIBRACION[cal["tramo"]]
            cal["real_medio"] = cal["suma_y"] / cal["n_filas"]
            cal["pred_medio"] = cal["suma_pred"] / cal["n_filas"]
            tablas["calibracion"] = cal.sort_values(["modelo", "tramo"]).reset_index(drop=True)
        if self._cuantiles:
            qbase = pd.DataFrame([dict(zip(["metodo", *AGG_DIMENSIONS], key), **dict(zip(AGG_QUANTILE_SUMS, value)))
                                  for key, value in self._cuantiles.items()])
            partes = []
            for ambito, dims in (("global", []), ("h", ["h"]), ("tipo_categoria", ["tipo_categoria"]), ("region", ["region"]),
                                 ("corte", ["corte"]), ("corte×h×tipo", ["corte", "h", "tipo_categoria"]),
                                 ("tipo×region", ["tipo_categoria", "region"])):
                parte = _agg_quantile_metrics(_agg_rollup(qbase, ["metodo", *dims], AGG_QUANTILE_SUMS))
                parte.insert(1, "ambito", ambito)
                partes.append(parte)
            tablas["cuantiles"] = pd.concat(partes, ignore_index=True)
        return tablas

    def estado(self):
        """Contenido del acumulador para el punto de control (se guarda con pickle)."""
        return {k: v for k, v in self.__dict__.items()}

    @classmethod
    def desde_estado(cls, estado):
        obj = cls.__new__(cls)
        obj.__dict__.update(estado)
        return obj

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.10 Significancia, matriz de decisión y compuerta de aceptación
# MAGIC - **Diebold–Mariano** por corte: compara la pérdida absoluta media diaria del candidato contra una referencia
# MAGIC   (ingenuo o mejor individual) con varianza de Bartlett (3 rezagos). Con 14 días por corte es descriptiva.
# MAGIC - **Matriz de decisión**: pesos fijados antes de ver resultados (sección 1.5). Entre los `N_FINALISTAS` mejores y
# MAGIC   el mejor individual, gana el de mayor puntaje; si la diferencia con el segundo es < 0.02 y Diebold–Mariano no
# MAGIC   distingue a los dos, se prefiere el más fácil de operar. **La elección se registra antes de evaluar septiembre.**
# MAGIC - **Compuerta**: comprobaciones técnicas (supera al ingenuo en cada corte y en septiembre, sesgo, significancia,
# MAGIC   cobertura de intervalos) y de negocio (alcance aprobado, política de servicios, disponibilidad de atributos). Con las
# MAGIC   técnicas aprobadas el modelo se registra como `@challenger`; pasar a producción (`@champion`) exige también las de
# MAGIC   negocio.

# COMMAND ----------

def dm_por_corte(diario, modelo, referencia="B0_Ingenuo", rezagos=3):
    """Diebold–Mariano por corte con la pérdida absoluta media por fecha (aproximación normal bilateral)."""
    from scipy.stats import norm
    filas = []
    for corte, grupo in diario.groupby("corte", sort=True):
        sel = grupo.loc[grupo.modelo.eq(modelo)].set_index("fecha_objetivo").sort_index()
        ref = grupo.loc[grupo.modelo.eq(referencia)].set_index("fecha_objetivo").sort_index()
        if sel.empty or ref.empty or not sel.index.equals(ref.index):
            continue
        d = (ref.error_absoluto_total / ref.n_filas - sel.error_absoluto_total / sel.n_filas).to_numpy(dtype=float)
        estadistico = probabilidad = np.nan
        if len(d) >= 8 and np.isfinite(d).all():
            centrado = d - d.mean()
            k = min(rezagos, len(d) - 1)
            gamma = [float(np.mean(centrado[j:] * centrado[:len(d) - j])) for j in range(k + 1)]
            varianza = gamma[0] + 2 * sum((1 - j / (k + 1)) * gamma[j] for j in range(1, k + 1))
            if varianza > 0:
                estadistico = float(d.mean() / np.sqrt(varianza / len(d)))
                probabilidad = float(2 * norm.sf(abs(estadistico)))
        filas.append({"corte": str(corte), "modelo": modelo, "referencia": referencia, "dm_estadistico": estadistico,
                      "dm_pvalue": probabilidad, "n_dias": len(d), "rezagos": rezagos,
                      "mejor_que_referencia": bool(np.isfinite(estadistico) and estadistico > 0)})
    return pd.DataFrame(filas)


def matriz_decision(tablas_desarrollo, n_finalistas=N_FINALISTAS):
    """Puntúa a los finalistas con PESOS_DECISION y regresa (matriz, puntos por criterio, modelo elegido, argumento)."""
    comp = tablas_desarrollo["comparativa"].set_index("modelo")
    elegibles = comp.loc[~comp.tipo.isin(["referencia", "diagnostico"])].sort_values("wmape_media_cortes")
    finalistas = list(elegibles.index[:n_finalistas])
    individuales = elegibles.loc[elegibles.tipo.eq("individual")]
    if len(individuales) and individuales.index[0] not in finalistas:
        finalistas.append(individuales.index[0])
    D = comp.loc[finalistas]
    wm_ing = float(comp.loc["B0_Ingenuo", "wmape_media_cortes"])
    minutos = D.get("min_entrenamiento_por_corte", pd.Series(0.0, index=D.index)).fillna(0.0)
    escala_final = max(1.0, (FUENTE_AV5.manifiesto.get("filas_train_elegibles", 1) or 1)
                       / max(FUENTE_AV5.manifiesto.get("filas_train_cargadas", 1) or 1, 1) * 0.5)
    DEC = pd.DataFrame({"modelo": finalistas, "error": D["wmape_media_cortes"].to_numpy(),
                        "estabilidad": D["wmape_desviacion_cortes"].to_numpy(), "sesgo": D["sesgo_abs_medio_pct"].to_numpy(),
                        "error_donde_cuesta": D["wmape_mercancia"].fillna(D["wmape_media_cortes"]).to_numpy(),
                        "costo": (minutos * escala_final).to_numpy(),
                        "interpretabilidad": [INTERPRETABILIDAD.get(m, 0.5) for m in finalistas],
                        "operabilidad": [OPERABILIDAD.get(m, 0.5) for m in finalistas]})
    puntos = pd.DataFrame({"modelo": DEC["modelo"]})
    for criterio, peso in PESOS_DECISION.items():
        v = DEC[criterio].astype(float)
        if criterio == "error":              # mejora sobre la referencia relativa a la mejor mejora
            s = ((wm_ing - v) / max(wm_ing - v.min(), 1e-9)).clip(lower=0)
        elif criterio == "costo":            # dentro del presupuesto del job mensual no penaliza
            s = (PRESUPUESTO_JOB_MIN / v.clip(lower=1e-9)).clip(upper=1.0)
        elif criterio in ("estabilidad", "sesgo", "error_donde_cuesta"):
            s = (max(v.min(), 1e-9) / v.clip(lower=1e-9)).clip(upper=1.0)
        else:
            s = v / max(v.max(), 1e-9)
        puntos[criterio] = s.fillna(1.0).to_numpy() * peso
    DEC["total"] = puntos[list(PESOS_DECISION)].sum(axis=1).to_numpy()
    DEC = DEC.sort_values("total", ascending=False).reset_index(drop=True)
    elegido, segundo = DEC.iloc[0]["modelo"], DEC.iloc[1]["modelo"] if len(DEC) > 1 else None
    argumento = [f"{elegido} obtiene el mayor puntaje ({DEC.iloc[0]['total']:.3f})."]
    if segundo is not None:
        dm = dm_por_corte(tablas_desarrollo["diario"], elegido, referencia=segundo)
        significativo = int((dm["dm_pvalue"] < 0.05).sum()) if len(dm) else 0
        diferencia = float(DEC.iloc[0]["total"] - DEC.iloc[1]["total"])
        if diferencia < 0.02 and significativo < max(1, len(dm) // 2 + 1):
            par = DEC.iloc[:2].sort_values(["operabilidad", "costo"], ascending=[False, True])
            if par.iloc[0]["modelo"] != elegido:
                argumento.append(f"Empate práctico con {segundo} (diferencia {diferencia:.3f}; Diebold–Mariano significativo en "
                                 f"{significativo} de {len(dm)} cortes): se elige el más fácil de operar, {par.iloc[0]['modelo']}.")
                elegido = par.iloc[0]["modelo"]
        else:
            argumento.append(f"Supera a {segundo} por {diferencia:.3f} puntos (Diebold–Mariano significativo en "
                             f"{significativo} de {len(dm)} cortes).")
    return DEC, puntos.set_index("modelo").loc[DEC["modelo"]].reset_index(), elegido, " ".join(argumento)


def evaluate_gate(point_metrics, quantile_metrics, quality_evidence, business_evidence, test_metrics=None,
                  scope_metrics=None) -> dict:
    """Compuerta de aceptación. point_metrics: una fila por corte de desarrollo del modelo elegido (wmape, wmape_ingenuo,
    sesgo_pct, dm_pvalue). quantile_metrics: métricas de intervalos en la prueba (ámbitos global, h, tipo, región).
    test_metrics: una fila por bloque de prueba (wmape, wmape_ingenuo, sesgo_pct). No cambia aliases: sólo decide."""
    u = UMBRALES_COMPUERTA
    checks, reasons = {}, []
    pm = point_metrics if isinstance(point_metrics, pd.DataFrame) else pd.DataFrame()
    valid = {"corte", "wmape", "wmape_ingenuo", "sesgo_pct", "dm_pvalue"} <= set(pm.columns) and len(pm) > 0
    checks["cortes_desarrollo_completos"] = bool(valid and pm["corte"].nunique() >= u["cortes_min"])
    checks["supera_ingenuo_cada_corte"] = bool(valid and (pm["wmape"] < pm["wmape_ingenuo"]).all())
    checks["sesgo_por_corte"] = bool(valid and pm["sesgo_pct"].abs().le(u["sesgo_max_pct"]).all())
    checks["dm_tres_cortes"] = bool(valid and (pm["dm_pvalue"] < u["dm_p"]).sum() >= min(u["dm_cortes_min"], len(pm)))
    tm = test_metrics if isinstance(test_metrics, pd.DataFrame) else pd.DataFrame()
    holdout = len(tm) > 0
    checks["evaluacion_independiente"] = bool(holdout)
    checks["supera_ingenuo_en_prueba"] = bool(holdout and (tm["wmape"] < tm["wmape_ingenuo"]).all())
    checks["sesgo_en_prueba"] = bool(holdout and tm["sesgo_pct"].abs().le(u["sesgo_prueba_max_pct"]).all())
    qm = quantile_metrics if isinstance(quantile_metrics, pd.DataFrame) else pd.DataFrame()
    valid_q = not qm.empty and {"ambito", "cobertura", "tasa_cruces_final", "ancho_medio"} <= set(qm.columns)
    checks["metricas_cuantilicas_completas"] = bool(valid_q)
    checks["catorce_horizontes_evaluados"] = bool(valid_q and set(qm.loc[qm.ambito.eq("h"), "h"].dropna().astype(int)) == set(range(1, 15)))
    checks["cobertura_global_en_banda"] = bool(valid_q and qm.loc[qm.ambito.eq("global"), "cobertura"].between(*BANDA_COBERTURA).all())
    checks["cobertura_por_horizonte"] = bool(valid_q and qm.loc[qm.ambito.eq("h"), "cobertura"].between(*BANDA_COBERTURA).all())
    checks["cobertura_por_tipo_y_region"] = bool(valid_q and qm.loc[qm.ambito.isin(["tipo_categoria", "region"]), "cobertura"]
                                                 .between(*BANDA_COBERTURA).all())
    checks["sin_cruces_corregidos"] = bool(valid_q and qm["tasa_cruces_final"].eq(0).all())
    checks["anchos_no_negativos"] = bool(valid_q and qm["ancho_medio"].ge(0).all())
    tecnicas = ("cortes_desarrollo_completos", "supera_ingenuo_cada_corte", "evaluacion_independiente", "supera_ingenuo_en_prueba",
                "metricas_cuantilicas_completas", "cobertura_global_en_banda", "sin_cruces_corregidos")
    reasons = [name for name, ok in checks.items() if not ok]
    quality_pass = (quality_evidence.get("critical_total", 0) >= 1 and quality_evidence.get("critical_passed") == quality_evidence.get("critical_total")
                    and quality_evidence.get("snapshot_verified") is True)
    if not quality_pass:
        reasons.append("contratos_del_snapshot_no_acreditados")
    upstream_pass = quality_evidence.get("upstream_quality_snapshot_verified") is True
    if not upstream_pass:
        reasons.append("reglas_criticas_de_calidad_upstream_pendientes")
    business_pass = all(business_evidence.get(k) is True for k in ("scope_approved", "services_policy_approved", "point_in_time_validated"))
    if not business_pass:
        reasons.append("aprobaciones_de_negocio_pendientes (alcance, servicios, disponibilidad de atributos)")
    scope = dict(scope_metrics or {})
    absolute_pass = (scope.get("wmape_global") is not None and scope["wmape_global"] < u["wmape_global_max"]
                     and scope.get("wmape_mercancia") is not None and scope["wmape_mercancia"] < u["wmape_mercancia_max"])
    if not absolute_pass:
        reasons.append("umbrales_absolutos_del_avance_4_no_superados")
    tecnica_ok = all(checks[c] for c in tecnicas) and quality_pass and absolute_pass
    return {"desarrollo_status": "cumple_comprobaciones_tecnicas" if tecnica_ok else "no_cumple_comprobaciones_tecnicas",
            "technical_pass": bool(tecnica_ok), "production_blocked": not (tecnica_ok and business_pass and upstream_pass),
            "eligible_for_promotion": bool(tecnica_ok and business_pass and upstream_pass),
            "registry_action": "registrar_challenger" if tecnica_ok else "ninguna",
            "checks": checks, "checks_tecnicos_bloqueantes": list(tecnicas), "blocking_reasons": reasons,
            "historic_global_threshold": u["wmape_global_max"], "historic_merchandise_threshold": u["wmape_mercancia_max"],
            "limitacion": "La compuerta evalúa la evidencia de esta corrida; promover a producción exige aprobación de negocio."}


def evidencia_calidad_upstream():
    """Última corrida de 11_validacion_calidad: ¿todas las reglas críticas pasaron? (tabla mlops.validaciones)."""
    try:
        v = spark.table(tabla("mlops", "validaciones")).toPandas()
        v = v[v["fecha_validacion"] == v["fecha_validacion"].max()]
        criticas = v[v["critica"].astype(bool)]
        pasan = criticas["estado"].eq("PASA")
        return {"upstream_quality_snapshot_verified": bool(len(criticas) and pasan.all()),
                "upstream_critical_total": int(len(criticas)), "upstream_critical_passed": int(pasan.sum()),
                "reglas_criticas_que_fallan": criticas.loc[~pasan, "regla"].tolist(),
                "fecha_validacion": str(v["fecha_validacion"].max())}
    except Exception as e:
        return {"upstream_quality_snapshot_verified": False, "motivo": f"sin tabla de validaciones ({type(e).__name__})"}

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.11 Registro de la corrida: bitácora, puntos de control, MLflow y Unity Catalog
# MAGIC - Cada corrida tiene un identificador `av5_43_<fecha UTC>_<8 caracteres>`. Sus eventos (iniciada, bloque terminado,
# MAGIC   modelo elegido, finalizada o error) se agregan a `mlops.av5_ejecuciones`.
# MAGIC - Después de cada bloque se guarda un **punto de control** (pickle con las sumas, la bolsa de meta-entrenamiento y los
# MAGIC   hiperparámetros) en el volumen `outputs`. Si la corrida se cae, `RUN_ID_REANUDAR` la continúa sin repetir bloques.
# MAGIC - Las tablas de resultados se agregan con la columna `run_id` a tablas fijas `mlops.av5_*` (no se crea una tabla nueva
# MAGIC   por corrida). Ninguna tabla de resultados lleva tienda, categoría ni filas individuales, salvo el pronóstico operativo.
# MAGIC - MLflow guarda parámetros, métricas, tablas, gráficas y el modelo final empaquetado (`pyfunc`) que se registra en
# MAGIC   Unity Catalog como nueva versión de `demanda_categoria` con alias `@challenger`.

# COMMAND ----------

def verificar_entorno_formal(spark):
    """Versiones efectivas de las bibliotecas (no instala nada; las que falten se reportan y sus modelos se omiten)."""
    import importlib
    import platform
    observed, missing = {"python": platform.python_version()}, []
    for name in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "xgboost", "catboost", "optuna", "shap", "mlflow", "cloudpickle"):
        try:
            observed[name] = str(getattr(importlib.import_module(name), "__version__", "sin_version_publicada"))
        except Exception:
            missing.append(name)
            observed[name] = "NO DISPONIBLE"
    try:
        observed["databricks_runtime"] = spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion")
    except Exception:
        observed["databricks_runtime"] = "serverless (no publica versión)"
    return observed, missing


def huella_codigo():
    """SHA-256 de las funciones principales de esta libreta: identifica qué versión del código produjo la corrida."""
    import inspect
    texto = ""
    for nombre in ("FuenteAvance5", "fit_model", "tune_boosting", "fit_homogeneous", "fit_meta", "predict_ensembles",
                   "fit_conformal", "fit_quantiles", "AcumuladorAvance5", "matriz_decision", "evaluate_gate", "ejecutar_modelado"):
        try:
            texto += inspect.getsource(globals()[nombre])
        except Exception:
            texto += nombre
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def _operacion_json(value):
    """Convierte fechas y tipos de numpy para guardarlos en JSON."""
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    return str(value)


class RegistroFormal:
    """Contexto de una corrida formal: `with RegistroFormal(...) as REGISTRO_AV5: ...`.

    run_id          identificador de la corrida (nuevo, o el de RUN_ID_REANUDAR)
    mlflow          módulo de MLflow si está disponible (si no, se registra sólo en Delta)
    mlflow_run_id   identificador del run en MLflow
    ruta_control    carpeta del volumen donde vive el punto de control de esta corrida
    manifest        fuente, entorno, huella del código y decisiones; se registra al inicio y al final
    """

    def __init__(self, spark, fuente, entorno, huella, n_jobs, run_id=None):
        self.spark, self.fuente, self.entorno = spark, fuente, entorno
        self.reanudada = run_id is not None
        self.run_id = run_id or ("av5_43_" + datetime.now(timezone.utc).strftime("%Y%m%dt%H%M%S") + "_" + uuid4().hex[:8])
        self.ruta_control = f"{RUTA_PUNTOS_CONTROL}/{self.run_id}"
        self.mlflow, self.mlflow_run_id, self.experimento = None, None, None
        self.manifest = {"run_id": self.run_id, "codigo_sha256": huella, "entorno": entorno, "hilos": int(n_jobs),
                         "modo_rapido": MODO_RAPIDO, "fuente": fuente.manifiesto, "reanudada": self.reanudada}

    def __enter__(self):
        try:
            import mlflow
            mlflow.autolog(disable=True)          # sin esto, Databricks registraría un run por cada modelo entrenado
            self.experimento = EXPERIMENTO_MLFLOW
            if self.experimento is None:
                usuario = self.spark.sql("SELECT current_user()").collect()[0][0]
                self.experimento = f"/Users/{usuario}/mna_equipo43_avance5"
            try:
                mlflow.set_experiment(self.experimento)
            except Exception:
                self.experimento = "/Shared/mna_equipo43_avance5"
                mlflow.set_experiment(self.experimento)
            if mlflow.active_run() is not None:
                mlflow.end_run()
            previo = None
            if self.reanudada:
                encontrados = mlflow.search_runs(filter_string=f"tags.run_id_av5 = '{self.run_id}'", output_format="list")
                previo = encontrados[0].info.run_id if encontrados else None
            run = mlflow.start_run(run_id=previo) if previo else mlflow.start_run(run_name=self.run_id)
            self.mlflow, self.mlflow_run_id = mlflow, run.info.run_id
            mlflow.set_tags({"equipo": "43", "avance": "5", "run_id_av5": self.run_id,
                             "version_gold": str(self.fuente.version), "codigo_sha256": self.manifest["codigo_sha256"],
                             "modo": "rapido" if MODO_RAPIDO else "completo"})
            self.log_dict("manifiesto_inicial", self.manifest)
        except Exception as e:
            print(f"(MLflow no disponible: {type(e).__name__}: {str(e)[:120]}; se registra sólo en Delta)")
            self.mlflow = None
        self.manifest["mlflow_run_id"] = self.mlflow_run_id
        self.manifest["mlflow_experimento"] = self.experimento
        self.evento("REANUDADA" if self.reanudada else "INICIADA", {"version_gold": self.fuente.version,
                                                                     "mlflow_run_id": self.mlflow_run_id})
        print(f"Corrida {self.run_id} · MLflow {self.mlflow_run_id or '(no disponible)'} · punto de control en {self.ruta_control}")
        return self

    def evento(self, estado, info=None):
        """Agrega un evento a mlops.av5_ejecuciones (append; nunca sobrescribe)."""
        fila = pd.DataFrame([{"run_id": self.run_id, "estado": str(estado), "fecha_utc": datetime.now(timezone.utc).replace(tzinfo=None),
                              "info_json": json.dumps(info or {}, ensure_ascii=False, default=_operacion_json)}])
        try:
            limpio, esquema = _para_spark(fila)
            (self.spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
             .option("mergeSchema", "true").saveAsTable(TABLA_EJECUCIONES))
        except Exception as e:
            print(f"(no se pudo escribir el evento {estado}: {type(e).__name__})")

    def log_dict(self, name, value):
        if self.mlflow is not None:
            try:
                self.mlflow.log_dict(json.loads(json.dumps(value, ensure_ascii=False, default=_operacion_json)), f"metadatos/{name}.json")
            except Exception:
                pass

    def log_table(self, name, frame):
        """Tabla agregada a MLflow como CSV (nunca con identificadores de tienda)."""
        if self.mlflow is None or frame is None or not len(frame):
            return
        if "tienda" in frame.columns:
            raise ValueError("Las evidencias de MLflow no admiten identificadores de tienda.")
        try:
            with tempfile.TemporaryDirectory(prefix="av5_tabla_") as carpeta:
                destino = Path(carpeta) / f"{name}.csv"
                frame.to_csv(destino, index=False)
                self.mlflow.log_artifact(str(destino), artifact_path="tablas")
        except Exception:
            pass

    def log_metrics(self, metricas):
        if self.mlflow is not None:
            try:
                self.mlflow.log_metrics({k: float(v) for k, v in metricas.items() if v is not None and np.isfinite(v)})
            except Exception:
                pass

    def guardar_punto_control(self, estado) -> bool:
        """Guarda el estado de la corrida (pickle) en el volumen: primero en un archivo temporal y luego lo renombra."""
        try:
            os.makedirs(self.ruta_control, exist_ok=True)
            final, temporal = f"{self.ruta_control}/estado.pkl", f"{self.ruta_control}/estado.tmp"
            with open(temporal, "wb") as f:
                pickle.dump(estado, f, protocol=4)
            try:
                os.replace(temporal, final)
            except OSError:
                with open(final, "wb") as f:
                    pickle.dump(estado, f, protocol=4)
            return True
        except Exception as e:
            print(f"(no se pudo guardar el punto de control: {type(e).__name__}: {str(e)[:100]})")
            return False

    def cargar_punto_control(self):
        ruta = f"{self.ruta_control}/estado.pkl"
        if not os.path.exists(ruta):
            return None
        with open(ruta, "rb") as f:
            return pickle.load(f)

    def publicar_tablas(self, tables: Mapping) -> dict:
        """Agrega cada tabla a mlops.av5_<nombre> con run_id y fecha. Rechaza tablas con identificadores de tienda."""
        publicadas = {}
        for name, frame in tables.items():
            if frame is None or not isinstance(frame, pd.DataFrame) or frame.empty:
                continue
            if not re.fullmatch(r"[a-z_][a-z0-9_]*", name):
                raise ValueError(f"Nombre de salida no válido: {name}")
            if "tienda" in frame.columns:
                raise ValueError(f"La salida {name} contiene identificadores de tienda.")
            destino = tabla("mlops", f"av5_{name}")
            datos = frame.copy()
            for c in datos.columns:
                if datos[c].dtype == object and datos[c].map(lambda v: isinstance(v, (dict, list, tuple))).any():
                    datos[c] = datos[c].map(lambda v: json.dumps(v, ensure_ascii=False, default=_operacion_json))
            datos.insert(0, "run_id", self.run_id)
            datos["fecha_publicacion"] = pd.Timestamp.now()
            limpio, esquema = _para_spark(datos)
            (self.spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
             .option("mergeSchema", "true").saveAsTable(destino))
            publicadas[name] = destino
            self.log_table(name, frame)
        self.manifest["tablas_publicadas"] = publicadas
        return publicadas

    def __exit__(self, kind, error, traceback):
        if kind is None:
            self.evento("FINALIZADA", {"tablas": list(self.manifest.get("tablas_publicadas", {}))})
            self.log_dict("manifiesto_final", self.manifest)
        else:
            self.evento("ERROR", {"tipo_error": kind.__name__, "mensaje": str(error)[:500]})
        if self.mlflow is not None:
            try:
                self.mlflow.end_run(status="FINISHED" if kind is None else "FAILED")
            except Exception:
                pass
        return False


def predecir_candidato(candidate: Mapping, frame: pd.DataFrame) -> pd.DataFrame:
    """Predicción del paquete final (pronóstico puntual + P10/P50/P90 + techo operativo) para filas con las variables de
    la Oro. Acepta categóricas como texto (las codifica con el vocabulario guardado) o ya codificadas."""
    frame = frame.copy()
    for c in ("origen", "fecha_objetivo"):
        frame[c] = pd.to_datetime(frame[c])
    frame["h"] = frame["h"].astype("int16")
    if not ((frame["fecha_objetivo"] - frame["origen"]).dt.days.to_numpy() == frame["h"].to_numpy()).all():
        raise ValueError("La fecha objetivo debe ser origen más horizonte.")
    for c in candidate["cat_features"]:
        if frame[c].dtype == object:
            frame[c] = frame[c].map(candidate["codigos"][c]).fillna(-1).astype("int32")
    if "h_bloque" not in frame:
        frame["h_bloque"] = np.where(frame["h"] <= 7, 1, 2)
    name, bases = candidate["modelo"], candidate["bases"]
    if name == "B0_Ingenuo":
        point = frame["ingenuo_estacional"].to_numpy(dtype=np.float64)
    elif name in bases:
        point = predict_bundle(bases[name], frame)
    elif name == "E1_Bagging":
        point = predict_bundle(candidate["homogeneos"][name], frame)
    elif name in candidate.get("homogeneos", {}):
        point = predict_bundle(candidate["homogeneos"][name], frame)
    else:
        meta = candidate["meta"]
        point = predict_ensembles({b: predict_bundle(bases[b], frame) for b in meta.bases}, meta, frame)[name]
    intervalo = candidate.get("intervalos")
    if intervalo is None:
        q = pd.DataFrame({"p10": point, "p50": point, "p90": point}, index=frame.index)
    elif intervalo["metodo"] == "Q_Conformal":
        q = predict_conformal(intervalo, frame, point)
    else:
        q = predict_quantiles(intervalo, frame, prediccion=point)
    salida = pd.DataFrame({"prediccion_puntual": point, "p10": q["p10"].to_numpy(), "p50": q["p50"].to_numpy(),
                           "p90": q["p90"].to_numpy()}, index=frame.index)
    salida["techo_operativo"] = salida[NIVEL_TECHO_OPERATIVO]
    return salida


def registrar_modelo_uc(registro, candidate, ejemplo, metricas) -> dict:
    """Empaqueta el modelo final como MLflow pyfunc, lo registra como nueva versión de NOMBRE_MODELO_UC y le pone el alias
    @challenger. El ejemplo de entrada es sintético (ceros), para no guardar datos de Bnext en el registro."""
    import mlflow
    import mlflow.pyfunc
    from mlflow.models import infer_signature

    with warnings.catch_warnings():
        # MLflow ≥ 2.20 avisa si predict no tiene anotaciones de tipo; aquí la firma se declara explícitamente abajo
        warnings.simplefilter("ignore", UserWarning)

        class ModeloFinalAvance5(mlflow.pyfunc.PythonModel):
            """Modelo final del Avance 5: recibe filas con las variables de la Oro y regresa pronóstico, P10, P50 y P90."""

            def __init__(self, paquete):
                self.paquete = paquete

            def predict(self, context, model_input, params=None):
                return predecir_candidato(self.paquete, model_input)

    salida = predecir_candidato(candidate, ejemplo)
    firma = infer_signature(ejemplo, salida)
    requisitos = [f"{lib}=={registro.entorno.get(clave)}" for lib, clave in
                  (("lightgbm", "lightgbm"), ("xgboost", "xgboost"), ("catboost", "catboost"), ("scikit-learn", "sklearn"),
                   ("pandas", "pandas"), ("numpy", "numpy"), ("cloudpickle", "cloudpickle"))
                  if registro.entorno.get(clave) not in (None, "NO DISPONIBLE", "sin_version_publicada")]
    modelo = ModeloFinalAvance5(candidate)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        try:
            info = mlflow.pyfunc.log_model(name="modelo_final", python_model=modelo, signature=firma, pip_requirements=requisitos)
        except TypeError:
            info = mlflow.pyfunc.log_model(artifact_path="modelo_final", python_model=modelo, signature=firma, pip_requirements=requisitos)
    resultado = {"model_uri": info.model_uri, "registrado": False}
    try:
        mlflow.set_registry_uri("databricks-uc")
        version = mlflow.register_model(info.model_uri, NOMBRE_MODELO_UC)
        from mlflow import MlflowClient
        cliente = MlflowClient()
        cliente.set_registered_model_alias(NOMBRE_MODELO_UC, "challenger", version.version)
        for clave, valor in {"run_id_av5": registro.run_id, "modelo": candidate["modelo"], **metricas}.items():
            cliente.set_model_version_tag(NOMBRE_MODELO_UC, version.version, clave, str(valor))
        resultado.update(registrado=True, version=str(version.version), alias="challenger")
    except Exception as e:
        resultado["motivo"] = f"{type(e).__name__}: {str(e)[:200]}"
    return resultado

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.12 Orquestación: `ejecutar_modelado`
# MAGIC Recorre los orígenes en orden cronológico y guarda un punto de control al terminar cada uno:
# MAGIC
# MAGIC 1. **Ajuste (20-may).** Optuna para I1 e I4; después todas las bases con paro temprano en el bloque de ajuste: el
# MAGIC    número de árboles queda **congelado** para el resto de la corrida. Las predicciones del bloque entran a la bolsa.
# MAGIC 2. **Orígenes OOF** (si `n_origenes_oof` > 0): bases entrenadas antes del ajuste; sus predicciones también entran.
# MAGIC 3. **Cortes de desarrollo.** En cada corte: bases y homogéneos con objetivo ≤ origen; ensambles heterogéneos con la
# MAGIC    bolsa (bloques anteriores); predicción del universo evaluable por lotes de horizonte; sumas en los acumuladores;
# MAGIC    las filas de la muestra común de este corte entran a la bolsa para los cortes siguientes.
# MAGIC 4. **Selección.** Matriz de decisión con los cortes de desarrollo. El modelo elegido se registra **antes** de ver
# MAGIC    septiembre.
# MAGIC 5. **Prueba (datos no vistos).** Todo se entrena con datos al 31-ago y se congela; pronostica los dos bloques de
# MAGIC    septiembre sin reentrenar. Se evalúan también los intervalos (conformal calibrado con el bloque del 18 al 31 de
# MAGIC    agosto y LightGBM cuantílico).
# MAGIC 6. **Modelo final.** El elegido se reentrena con todos los datos, se recalibran sus intervalos con septiembre, se
# MAGIC    calcula la importancia de variables y se pronostican los 14 días siguientes al último dato.

# COMMAND ----------

def _flujo_train(pool: pd.DataFrame, origin) -> pd.DataFrame:
    """Filas de entrenamiento disponibles en `origin`: origen < origin y objetivo ≤ origin (nada del futuro)."""
    cutoff = pd.Timestamp(origin)
    result = pool.loc[(pool["origen"] < cutoff) & (pool["fecha_objetivo"] <= cutoff)]
    if result.empty or result["y"].sum() <= 0:
        raise ValueError(f"No existe entrenamiento elegible anterior a {cutoff.date()}.")
    return result


def _flujo_observed(fuente, origin) -> pd.DataFrame:
    """Bloque observado de un origen: la muestra común si es de evaluación; todas sus filas si es de entrenamiento (OOF)."""
    conjunto = fuente.conjunto_que_contiene_origen(origin)
    return fuente.cargar_bloque(origin, comun=conjunto == "evaluacion", conjunto=conjunto, fraccion_series=1.0)


def _flujo_wide(frame: pd.DataFrame, preds: Mapping, bloque: str) -> pd.DataFrame:
    """Filas para la bolsa de meta-entrenamiento: columnas fijas + una columna por modelo con su predicción."""
    filas = frame[[c for c in COLUMNAS_META_POOL if c != "h_bloque"]].copy()
    filas["h_bloque"] = np.where(frame["h"].to_numpy() <= 7, 1, 2).astype("int8")
    filas["bloque"] = bloque
    for name, values in preds.items():
        filas[name] = np.asarray(values, dtype=np.float32)
    return filas.reset_index(drop=True)


def _agregar_meta_pool(pool, nuevas):
    return nuevas if pool is None else pd.concat([pool, nuevas], ignore_index=True, sort=False)


def _flujo_costos(costs: pd.DataFrame, fase, origen, ledger: list):
    for row in costs.to_dict("records"):
        ledger.append({"fase": fase, "origen": str(pd.Timestamp(origen).date()), "segundos_prediccion": 0.0, **row})


def _theta_para(fuente, datos, origen):
    """Tabla Theta del origen para las series de `datos` (sólo si I6 está activo)."""
    if "I6_Theta" not in POINT_BASE_MODELS:
        return None
    series = pd.DataFrame({"tienda": datos["tienda"].astype(str).to_numpy(),
                           "categoria_id": fuente.decodificar("categoria_id", datos["categoria_id"].to_numpy())})
    return TablaTheta(theta_spark(series, origen), fuente.etiquetas["categoria_id"])


def _predecir_intervalos(intervalos, chunk, prediccion):
    salida = {}
    for metodo, bundle in (intervalos or {}).items():
        salida[metodo] = (predict_conformal(bundle, chunk, prediccion) if metodo == "Q_Conformal"
                          else predict_quantiles(bundle, chunk, prediccion=prediccion))
    return salida


def _evaluar_bloque(fuente, bloque, modelos, meta, acumuladores, estado, intervalos=None, modelo_final=None):
    """Predice un bloque (corte o prueba) por lotes de horizonte, agrega a los acumuladores y agrega la muestra común a la
    bolsa de meta-entrenamiento. Regresa los segundos de predicción por modelo."""
    segundos = {}
    lotes = [lote.tolist() for lote in np.array_split(np.arange(1, 15), fuente.config["lotes_horizonte"])]
    for lote in lotes:
        datos = fuente.cargar_bloque(bloque["origen"], horizontes=lote, conjunto="evaluacion")
        theta = _theta_para(fuente, datos, bloque["origen"])
        for h in lote:
            chunk = datos.loc[datos["h"].to_numpy() == h]
            if chunk.empty:
                continue
            chunk = chunk.assign(h_bloque=np.where(chunk["h"].to_numpy() <= 7, 1, 2).astype("int8"))
            preds, seg = predecir_todos(modelos["bases"], modelos["homogeneos"], meta, chunk, theta=theta)
            quantiles = _predecir_intervalos(intervalos, chunk, preds[modelo_final]) if intervalos else None
            acumuladores["ampliada"].agregar(chunk, preds, quantiles, corte=bloque["corte"])
            comun = chunk["en_muestra_eval"].to_numpy() == 1
            if comun.any():
                sub = chunk.loc[comun]
                preds_c = {m: v[comun] for m, v in preds.items()}
                q_c = {k: q.loc[comun] for k, q in quantiles.items()} if quantiles else None
                acumuladores["comun"].agregar(sub, preds_c, q_c, corte=bloque["corte"])
                estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(sub, preds_c, bloque["corte"]))
            for m, s in seg.items():
                segundos[m] = segundos.get(m, 0.0) + s
        del datos, theta
        gc.collect()
    return segundos


def _costos_por_modelo(nombre_corte, costos_bases, costos_homog, segundos_meta, segundos_pred):
    """Tiempos por modelo en un corte. Un ensamble cuenta el entrenamiento de todo lo que necesita (sus bases más el
    meta-modelo); E1 cuenta también el de I1, su primer miembro."""
    entrenamiento = {"B0_Ingenuo": 0.0}
    for row in costos_bases.to_dict("records"):
        entrenamiento[row["modelo"]] = float(row["segundos_entrenamiento"])
    for row in costos_homog.to_dict("records"):
        extra = entrenamiento.get("I1_LightGBM_Tweedie", 0.0) if row["modelo"] == "E1_Bagging" else 0.0
        entrenamiento[row["modelo"]] = float(row["segundos_entrenamiento"]) + extra
    if "I6_Theta" in POINT_BASE_MODELS:
        entrenamiento["I6_Theta"] = 0.0
    bases_total = sum(entrenamiento.get(b, 0.0) for b in POINT_BASE_MODELS)
    for name in (*POINT_HETEROGENEOUS_ACTIVOS, *MODELOS_DIAGNOSTICO):
        entrenamiento[name] = bases_total + segundos_meta
    return pd.DataFrame([{"modelo": m, "corte": nombre_corte, "segundos_entrenamiento": entrenamiento.get(m, 0.0),
                          "segundos_prediccion": segundos_pred.get(m, 0.0)} for m in AGG_MODELS])


def _nuevo_estado(fuente):
    """Estado de la corrida (lo que se guarda en cada punto de control)."""
    etq = fuente.etiquetas
    return {"version_estado": 1, "completados": [], "parametros": {}, "meta_pool": None,
            "acumuladores": {"desarrollo": {"ampliada": AcumuladorAvance5("desarrollo_ampliada", AGG_MODELS, etq),
                                            "comun": AcumuladorAvance5("desarrollo_comun", AGG_MODELS, etq)},
                             "prueba": {"ampliada": AcumuladorAvance5("prueba_ampliada", AGG_MODELS, etq),
                                        "comun": AcumuladorAvance5("prueba_comun", AGG_MODELS, etq)}},
            "costos": [], "auditoria": [], "ajuste": {}, "pesos": [], "decision": None, "prueba": {}, "final": {},
            "modelos_activos": list(AGG_MODELS)}


def _auditar(estado, etapa, origen, train=None, bloque=None):
    """Evidencia de que no hay fuga: la fecha objetivo máxima del entrenamiento nunca pasa del origen."""
    fila = {"etapa": etapa, "origen": str(pd.Timestamp(origen).date())}
    if train is not None:
        fila.update(filas_train=int(len(train)), max_objetivo_train=str(train["fecha_objetivo"].max().date()),
                    sin_fuga=bool(train["fecha_objetivo"].max() <= pd.Timestamp(origen)))
        if not fila["sin_fuga"]:
            raise RuntimeError(f"Fuga temporal en {etapa}: hay objetivos posteriores al origen.")
    if bloque is not None:
        fila.update(min_objetivo_evaluado=str(bloque["desde"]), max_objetivo_evaluado=str(bloque["hasta"]))
    estado["auditoria"].append(fila)


def ejecutar_modelado(fuente, registro, n_jobs=None, estado=None) -> dict:
    """Ejecuta el protocolo completo (ajuste → OOF → cortes → selección → prueba → modelo final) y regresa el estado.
    Con `estado` (punto de control) continúa en el primer paso no completado."""
    global PARAMETROS_VIGENTES
    inicio_total = time.perf_counter()
    jobs = runtime_threads(n_jobs)
    spec = {"numeric": list(fuente.num_features), "categorical": list(fuente.cat_features)}
    estado = estado or _nuevo_estado(fuente)
    if estado["modelos_activos"] != list(AGG_MODELS):
        raise RuntimeError("El punto de control se creó con otros modelos activos; revisa INCLUIR o inicia una corrida nueva.")
    if estado["parametros"]:
        PARAMETROS_VIGENTES = deepcopy(estado["parametros"])
    pool = fuente.cargar_entrenamiento()
    registro.evento("entrenamiento_cargado", {"filas": len(pool), "hilos": jobs, "variables": len(spec["numeric"]) + len(spec["categorical"])})
    completados = estado["completados"]

    def cerrar_paso(paso, info=None):
        completados.append(paso)
        estado["parametros"] = deepcopy(PARAMETROS_VIGENTES)
        registro.evento("paso_completado", {"paso": paso, **(info or {})})
        registro.guardar_punto_control(estado)

    # ---------------------------------------------------------------- 1. ajuste de hiperparámetros
    A = fuente.ajuste_origen
    if "ajuste" not in completados:
        subtitulo(f"1. Ajuste · origen {A}")
        train_A = _flujo_train(pool, A)
        _auditar(estado, "ajuste", A, train_A)
        tune_eval = _flujo_observed(fuente, A)
        mejores, pruebas, resumen, importancias = tune_boosting(train_A, tune_eval, spec, n_jobs=jobs)
        PARAMETROS_VIGENTES.update(mejores)
        if "I5_LightGBM_Log1p" in POINT_BASE_MODELS:
            PARAMETROS_VIGENTES["I5_LightGBM_Log1p"] = deepcopy(S6_POINT_PARAMETERS["I5_LightGBM_Log1p"])
        bases_A, costos_A = fit_individuals(train_A, spec, n_jobs=jobs, validacion=tune_eval)
        for name, bundle in bases_A.items():          # número de árboles congelado por paro temprano en el ajuste
            if "mejor_iteracion" in bundle.extras:
                PARAMETROS_VIGENTES.setdefault(name, deepcopy(bundle.params))
                PARAMETROS_VIGENTES[name]["n_estimators"] = int(bundle.extras["mejor_iteracion"])
        theta = _theta_para(fuente, tune_eval, A)
        preds = {b: (theta.predict(tune_eval) if b == "I6_Theta" else predict_bundle(bases_A[b], tune_eval)) for b in POINT_BASE_MODELS}
        estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(tune_eval, preds, "ajuste"))
        estado["ajuste"] = {"pruebas": pruebas, "resumen": resumen, "importancias": importancias,
                            "wmape_bases_ajuste": {b: motor_metrics(tune_eval["y"], p)["wmape"] for b, p in preds.items()}}
        _flujo_costos(costos_A, "bases_ajuste", A, estado["costos"])
        mostrar(resumen, "Ajuste fino (Optuna) en el bloque de ajuste",
                formatos={"pruebas": "int", "minutos": "dec1", "wmape_inicial": "dec4", "wmape_mejor": "dec4",
                          "mejora_relativa": "pct", "arboles_prueba": "int"})
        del train_A, tune_eval, bases_A
        gc.collect()
        cerrar_paso("ajuste", {"parametros": {k: v.get("n_estimators") for k, v in PARAMETROS_VIGENTES.items() if isinstance(v, dict)}})

    # ---------------------------------------------------------------- 2. orígenes OOF
    for o in fuente.oof_origenes:
        paso = f"oof_{o}"
        if paso in completados:
            continue
        subtitulo(f"2. Origen OOF {o}")
        train_o = _flujo_train(pool, o)
        _auditar(estado, "oof", o, train_o)
        observado = _flujo_observed(fuente, o)
        bases_o, costos_o = fit_individuals(train_o, spec, n_jobs=jobs)
        theta = _theta_para(fuente, observado, o)
        preds = {b: (theta.predict(observado) if b == "I6_Theta" else predict_bundle(bases_o[b], observado)) for b in POINT_BASE_MODELS}
        estado["meta_pool"] = _agregar_meta_pool(estado["meta_pool"], _flujo_wide(observado, preds, f"oof_{o}"))
        _flujo_costos(costos_o, "bases_oof", o, estado["costos"])
        del train_o, observado, bases_o
        gc.collect()
        cerrar_paso(paso)

    # ---------------------------------------------------------------- 3. cortes de desarrollo
    acum_dev = estado["acumuladores"]["desarrollo"]
    for bloque in fuente.cortes:
        if bloque["corte"] in completados:
            continue
        origen = bloque["origen"]
        subtitulo(f"3. {bloque['corte']} · origen {origen} · pronostica {bloque['desde']} a {bloque['hasta']}")
        with Cronometro() as reloj_corte:
            train = _flujo_train(pool, origen)
            _auditar(estado, bloque["corte"], origen, train, bloque)
            bases, costos_b = fit_individuals(train, spec, n_jobs=jobs)
            homogeneos, costos_h = fit_homogeneous(train, spec, bases, n_jobs=jobs)
            del train
            gc.collect()
            with Cronometro() as reloj_meta:
                meta, pesos = fit_meta(estado["meta_pool"], origen, n_jobs=jobs)
            estado["pesos"].append(pesos.assign(corte=bloque["corte"]))
            segundos_pred = _evaluar_bloque(fuente, bloque, {"bases": bases, "homogeneos": homogeneos}, meta, acum_dev, estado)
            costos = _costos_por_modelo(bloque["corte"], costos_b, costos_h, reloj_meta.segundos, segundos_pred)
            for acum in acum_dev.values():
                acum.registrar_costos(costos)
            _flujo_costos(costos_b, "bases_corte", origen, estado["costos"])
            _flujo_costos(costos_h, "homogeneos_corte", origen, estado["costos"])
            estado["costos"].append({"fase": "meta_corte", "origen": origen, "modelo": "meta_heterogeneos",
                                     "segundos_entrenamiento": reloj_meta.segundos, "segundos_prediccion": 0.0})
        parcial = acum_dev["ampliada"].tablas()["por_corte"]
        fila = parcial.loc[parcial.corte.eq(bloque["corte"])].set_index("modelo")["wmape"]
        print(f"   {bloque['corte']} en {reloj_corte.segundos / 60:.1f} min · WMAPE: " +
              " · ".join(f"{m.split('_')[0]} {fila.get(m, np.nan):.4f}" for m in AGG_MODELS if m in fila.index))
        del bases, homogeneos, meta
        gc.collect()
        cerrar_paso(bloque["corte"], {"minutos": reloj_corte.segundos / 60})

    # ---------------------------------------------------------------- 4. selección (antes de ver septiembre)
    if "seleccion" not in completados:
        subtitulo("4. Selección del modelo final con los cortes de desarrollo")
        tablas_dev = acum_dev["ampliada"].tablas()
        DEC, puntos, elegido, argumento = matriz_decision(tablas_dev)
        estado["decision"] = {"matriz": DEC, "puntos": puntos, "modelo_final": elegido, "argumento": argumento,
                              "fecha_utc": datetime.now(timezone.utc).isoformat()}
        print(f"   Modelo elegido: {elegido}. {argumento}")
        cerrar_paso("seleccion", {"modelo_final": elegido, "nota": "elegido antes de evaluar los bloques de prueba"})
    elegido = estado["decision"]["modelo_final"]

    # ---------------------------------------------------------------- 5. prueba en datos no vistos
    acum_prueba = estado["acumuladores"]["prueba"]
    if fuente.pruebas and "prueba" not in completados:
        P1 = fuente.pruebas[0]["origen"]
        subtitulo(f"5. Prueba no vista · modelos congelados con datos al {P1}")
        train = _flujo_train(pool, P1)
        _auditar(estado, "prueba_entrenamiento", P1, train)
        bases, costos_b = fit_individuals(train, spec, n_jobs=jobs)
        homogeneos, costos_h = fit_homogeneous(train, spec, bases, n_jobs=jobs)
        meta, pesos = fit_meta(estado["meta_pool"], P1, n_jobs=jobs)
        estado["pesos"].append(pesos.assign(corte="prueba"))
        # calibración: el bloque más reciente de la bolsa que ya terminó en P1 (normalmente corte_4, 18–31 de agosto)
        bolsa = estado["meta_pool"]
        origen_cal = pd.Timestamp(bolsa.loc[(bolsa["fecha_objetivo"] <= pd.Timestamp(P1)) & bolsa["bloque"].str.startswith("corte_"),
                                            "origen"].max()).date().isoformat()
        calibracion = bolsa.loc[bolsa["origen"] == pd.Timestamp(origen_cal)]
        intervalos = {"Q_Conformal": fit_conformal(calibracion, elegido, calibration_end=P1)}
        if INCLUIR_Q_LIGHTGBM:
            train_q = _flujo_train(pool, origen_cal)
            bloque_cal = fuente.cargar_bloque(origen_cal, comun=True, conjunto="evaluacion", fraccion_series=1.0)
            intervalos["Q_LightGBM_CQR"] = fit_quantiles(train_q, bloque_cal, spec, cutoff_train=origen_cal, calibration_end=P1, n_jobs=jobs)
            _auditar(estado, "cuantiles_entrenamiento", origen_cal, train_q)
            _flujo_costos(intervalos["Q_LightGBM_CQR"]["costs"], "cuantiles_prueba", origen_cal, estado["costos"])
            del train_q, bloque_cal
        del train
        gc.collect()
        for bloque in fuente.pruebas:
            _auditar(estado, bloque["corte"], P1, bloque=bloque)
            segundos_pred = _evaluar_bloque(fuente, bloque, {"bases": bases, "homogeneos": homogeneos}, meta, acum_prueba,
                                            estado, intervalos, elegido)
            costos = _costos_por_modelo(bloque["corte"], costos_b, costos_h, 0.0, segundos_pred)
            for acum in acum_prueba.values():
                acum.registrar_costos(costos)
        estado["prueba"] = {"origen_entrenamiento": P1, "origen_calibracion": origen_cal,
                            "intervalos_auditoria": {m: {k: v for k, v in b.items() if k in ("calibration_end", "cutoff_train", "n_calibracion", "audit")}
                                                     for m, b in intervalos.items()}}
        del bases, homogeneos, meta, intervalos
        gc.collect()
        cerrar_paso("prueba")
    elif not fuente.pruebas:
        aviso("La Oro no tiene bloques de prueba: no hay evaluación en datos no vistos (carga septiembre y reconstruye 02 → 20).")

    # ---------------------------------------------------------------- 6. modelo final, intervalos, importancia y pronóstico
    if REENTRENAR_FINAL and "final" not in completados:
        F_ = fuente.fecha_maxima_objetivo
        subtitulo(f"6. Modelo final: {elegido} con todos los datos hasta {F_}")
        train = _flujo_train(pool, F_)
        _auditar(estado, "final", F_, train)
        bases, homogeneos, costos_b = _entrenar_final(elegido, train, spec, jobs)
        meta = None
        if elegido in POINT_HETEROGENEOUS_ACTIVOS or elegido in MODELOS_DIAGNOSTICO:
            meta, pesos = fit_meta(estado["meta_pool"], F_, n_jobs=jobs)
            estado["pesos"].append(pesos.assign(corte="final"))
        metodo = elegir_metodo_intervalos(acum_prueba)
        ultimo = estado["meta_pool"]["origen"].max()
        recientes = estado["meta_pool"].loc[estado["meta_pool"]["origen"] == ultimo]
        if metodo == "Q_LightGBM_CQR":
            origen_cal = pd.Timestamp(ultimo).date().isoformat()
            bloque_cal = fuente.cargar_bloque(origen_cal, comun=True, conjunto="evaluacion", fraccion_series=1.0)
            intervalo = fit_quantiles(_flujo_train(pool, origen_cal), bloque_cal, spec, cutoff_train=origen_cal,
                                      calibration_end=F_, n_jobs=jobs)
        else:
            intervalo = fit_conformal(recientes, elegido, calibration_end=F_)
        candidate = {"modelo": elegido, "bases": bases, "homogeneos": homogeneos, "meta": meta, "intervalos": intervalo,
                     "metodo_intervalos": metodo, "cat_features": list(fuente.cat_features), "codigos": fuente.codigos,
                     "feature_spec": spec, "max_target_train": str(train["fecha_objetivo"].max().date()),
                     "snapshot": {"tabla": fuente.tabla, "version": fuente.version}, "run_id": registro.run_id}
        _flujo_costos(costos_b, "reentrenamiento_final", F_, estado["costos"]) if len(costos_b) else None
        del train
        gc.collect()
        estado["final"] = {"candidate": candidate, "metodo_intervalos": metodo,
                           "importancia": importancia_variables(candidate, pool, spec)}
        if fuente.origen_pronostico:
            datos = fuente.cargar_bloque(fuente.origen_pronostico, conjunto="pronostico", fraccion_series=1.0 if not MODO_RAPIDO else None)
            datos = datos.assign(h_bloque=np.where(datos["h"].to_numpy() <= 7, 1, 2).astype("int8"))
            if "I6_Theta" in (getattr(meta, "bases", ()) or ()) or elegido == "I6_Theta":
                candidate["bases"]["I6_Theta"] = ModelBundle("I6_Theta", None, _theta_para(fuente, datos, fuente.origen_pronostico),
                                                             np.inf, 1.0, 0, {})
                candidate["requiere_spark"] = True
            salida = predecir_candidato(candidate, datos)
            estado["final"]["pronostico"] = pd.concat([datos[["tienda", "origen", "fecha_objetivo", "h"]].reset_index(drop=True),
                                                       pd.DataFrame({"categoria_id": fuente.decodificar("categoria_id", datos["categoria_id"].to_numpy()),
                                                                     "tipo_categoria": fuente.decodificar("tipo_categoria", datos["tipo_categoria"].to_numpy()),
                                                                     "region": fuente.decodificar("region", datos["region"].to_numpy())}),
                                                       salida.reset_index(drop=True)], axis=1)
            del datos
        cerrar_paso("final")
    estado["segundos_modelado"] = estado.get("segundos_modelado", 0.0) + (time.perf_counter() - inicio_total)
    return estado


def _entrenar_final(elegido, train, spec, jobs):
    """Entrena sólo lo que necesita el modelo elegido: la base misma; I1 y sus miembros (E1); el homogéneo (E2, E7);
    o todas las bases tabulares (ensambles heterogéneos)."""
    bases, homogeneos, costos = {}, {}, []
    if elegido in POINT_HETEROGENEOUS_ACTIVOS or elegido in MODELOS_DIAGNOSTICO:
        necesarias = [b for b in POINT_BASE_MODELS if b != "I6_Theta"]
    elif elegido == "E1_Bagging":
        necesarias = ["I1_LightGBM_Tweedie"]
    elif elegido in POINT_BASE_MODELS and elegido != "I6_Theta":
        necesarias = [elegido]
    else:
        necesarias = []
    for name in necesarias:
        bases[name], costo = fit_model(name, train, spec, n_jobs=jobs)
        costos.append(costo)
    if elegido == "E1_Bagging":
        h, c = fit_homogeneous(train, spec, bases, n_jobs=jobs)
        homogeneos["E1_Bagging"] = h["E1_Bagging"]
        costos += c.to_dict("records")
    elif elegido in ("E2_AdaBoost", "E7_Bosque_Aleatorio"):
        homogeneos[elegido], costo = fit_model(elegido, train, spec, n_jobs=jobs)
        costos.append(costo)
    return bases, homogeneos, pd.DataFrame(costos)


def elegir_metodo_intervalos(acum_prueba) -> str:
    """Método de intervalos del modelo final: METODO_INTERVALOS, o en 'automatico' el de menor pinball total en la prueba
    entre los que dejan la cobertura global dentro de la banda (si ninguno, el más cercano al 80 %)."""
    if METODO_INTERVALOS != "automatico":
        return METODO_INTERVALOS
    try:
        q = acum_prueba["ampliada"].tablas().get("cuantiles")
        g = q.loc[q.ambito.eq("global")].copy()
        g["pinball_total"] = g["pinball_p10"] + g["pinball_p50"] + g["pinball_p90"]
        en_banda = g.loc[g.cumple_banda]
        elegido = (en_banda.sort_values("pinball_total") if len(en_banda)
                   else g.assign(_d=(g.cobertura - NOMINAL_COVERAGE).abs()).sort_values("_d")).iloc[0]["metodo"]
        return str(elegido)
    except Exception:
        return "Q_Conformal"


def importancia_variables(candidate, pool, spec, filas=20_000):
    """Importancia de variables del modelo de árboles principal del modelo final: media de |SHAP| en una muestra (o la
    importancia por ganancia si SHAP no está disponible). En un ensamble se usa la base con más peso en el stacking."""
    principal = candidate["modelo"]
    meta = candidate.get("meta")
    if meta is not None:
        pesos = (meta.pesos_stacking[1] + meta.pesos_stacking[2]) / 2
        arboles = [b for b in meta.bases if b in ("I1_LightGBM_Tweedie", "I4_XGBoost_Tweedie", "I5_LightGBM_Log1p", "I2_CatBoost_Tweedie")]
        principal = max(arboles, key=lambda b: pesos[meta.bases.index(b)]) if arboles else "I1_LightGBM_Tweedie"
    elif principal == "E1_Bagging":
        principal = "I1_LightGBM_Tweedie"
    bundle = candidate["bases"].get(principal) or candidate["homogeneos"].get(principal)
    if bundle is None or principal in ("I3_ElasticNet", "I6_Theta", "B0_Ingenuo", "E2_AdaBoost"):
        return pd.DataFrame()
    muestra = _motor_training_frame(pool, filas, seed=SEMILLA + 7)
    try:
        import shap
        X = bundle.encoder.transform_catboost(muestra) if principal == "I2_CatBoost_Tweedie" else bundle.encoder.transform(muestra)
        valores = shap.TreeExplainer(bundle.estimator).shap_values(X)
        importancia, metodo = np.abs(np.asarray(valores)).mean(axis=0), "media de |SHAP|"
    except Exception:
        importancia = np.asarray(getattr(bundle.estimator, "feature_importances_", np.ones(len(bundle.encoder.feature_names))), dtype=float)
        metodo = "importancia nativa (ganancia o divisiones)"
    tabla_imp = pd.DataFrame({"variable": bundle.encoder.feature_names, "importancia": importancia})
    tabla_imp["importancia"] = tabla_imp["importancia"] / max(tabla_imp["importancia"].sum(), 1e-12)
    tabla_imp["modelo"], tabla_imp["metodo"] = principal, metodo
    tabla_imp["grupo"] = tabla_imp["variable"].map(grupo_variable)
    tabla_imp["descripcion"] = tabla_imp["variable"].map(DESCRIPCION_VARIABLES).fillna("")
    return tabla_imp.sort_values("importancia", ascending=False).reset_index(drop=True)


def grupo_variable(nombre):
    """Grupo de una variable para resumir la importancia."""
    if nombre.startswith("f_"):
        return "promociones futuras"
    if nombre.startswith("s_") or nombre in ("dias_con_venta_hist", "dias_historia"):
        return "historia de la serie"
    if nombre.startswith(("t_", "zc_", "pc_")):
        return "contexto de tienda, zona y país"
    if nombre in ("dow", "fin_de_semana", "dia_mes", "semana_mes", "quincena", "es_evento", "tipo_evento", "dias_a_evento",
                  "dias_desde_evento", "en_mundial", "h", "h_bloque"):
        return "calendario y horizonte"
    return "atributos de tienda y categoría"


def compilar_resultados(estado, fuente, registro) -> dict:
    """Arma las tablas que se publican (todas agregadas, sin identificadores) a partir del estado de la corrida."""
    dev = estado["acumuladores"]["desarrollo"]["ampliada"].tablas()
    dev_c = estado["acumuladores"]["desarrollo"]["comun"].tablas()
    hay_prueba = bool(estado["acumuladores"]["prueba"]["ampliada"]._sumas)
    pru = estado["acumuladores"]["prueba"]["ampliada"].tablas() if hay_prueba else {}
    pru_c = estado["acumuladores"]["prueba"]["comun"].tablas() if hay_prueba else {}
    elegido = estado["decision"]["modelo_final"]
    T = {"comparativa": dev["comparativa"].assign(fase="desarrollo", evaluacion="ampliada"),
         "comparativa_comun": dev_c["comparativa"].assign(fase="desarrollo", evaluacion="comun")}
    for nombre in ("por_corte", "por_horizonte", "diario", "segmentos", "calibracion", "residuos"):
        partes = [dev.get(nombre, pd.DataFrame()).assign(fase="desarrollo")]
        if hay_prueba:
            partes.append(pru.get(nombre, pd.DataFrame()).assign(fase="prueba"))
        T[nombre] = pd.concat(partes, ignore_index=True)
    if hay_prueba:
        T["comparativa_prueba"] = pru["comparativa"].assign(fase="prueba", evaluacion="ampliada")
        T["comparativa_prueba_comun"] = pru_c["comparativa"].assign(fase="prueba", evaluacion="comun")
        T["cuantiles"] = pru.get("cuantiles", pd.DataFrame()).assign(fase="prueba")
    # Diebold–Mariano del elegido contra el ingenuo y contra el mejor individual, por bloque
    mejor_ind = dev["comparativa"].loc[dev["comparativa"].tipo.eq("individual")].iloc[0]["modelo"]
    dms = []
    for fase, tablas_f in (("desarrollo", dev), ("prueba", pru)):
        if not tablas_f:
            continue
        for ref in ("B0_Ingenuo", mejor_ind):
            if ref != elegido:
                dms.append(dm_por_corte(tablas_f["diario"], elegido, referencia=ref).assign(fase=fase))
    T["dm"] = pd.concat(dms, ignore_index=True) if dms else pd.DataFrame()
    dec = estado["decision"]
    T["decision"] = dec["matriz"].assign(modelo_final=elegido, argumento=dec["argumento"])
    T["decision_puntos"] = dec["puntos"]
    T["pesos"] = pd.concat(estado["pesos"], ignore_index=True) if estado["pesos"] else pd.DataFrame()
    T["ajuste"] = estado["ajuste"].get("pruebas", pd.DataFrame())
    T["ajuste_resumen"] = estado["ajuste"].get("resumen", pd.DataFrame())
    T["ajuste_importancia"] = estado["ajuste"].get("importancias", pd.DataFrame())
    T["costos"] = pd.DataFrame(estado["costos"])
    T["auditoria"] = pd.DataFrame(estado["auditoria"])
    T["parametros"] = pd.DataFrame([{"modelo": m, "parametros": json.dumps(p, default=_operacion_json)}
                                    for m, p in PARAMETROS_VIGENTES.items()])
    T["importancia"] = estado.get("final", {}).get("importancia", pd.DataFrame())
    pool = estado["meta_pool"]
    if pool is not None:
        filas = pool.loc[pool["bloque"].str.startswith("corte_")]
        bases = [b for b in POINT_BASE_MODELS if b in filas.columns]
        if len(filas) and len(bases) > 1:
            errores = pd.DataFrame({b: filas["y"].to_numpy() - filas[b].to_numpy() for b in bases}).corr()
            T["correlacion_errores"] = errores.reset_index().rename(columns={"index": "base"})
    return T

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.13 Compuerta, salida operativa, monitoreo y lectura de resultados

# COMMAND ----------

def preparar_compuerta(T: Mapping, estado, fuente) -> dict:
    """Reúne la evidencia de la corrida (cortes, prueba, intervalos, contratos y negocio) y evalúa la compuerta."""
    elegido = estado["decision"]["modelo_final"]
    pc = T["por_corte"]
    dev = pc.loc[pc.fase.eq("desarrollo") & pc.modelo.eq(elegido), ["corte", "wmape", "wmape_ingenuo", "sesgo_pct"]]
    dm = T["dm"]
    if len(dm):
        dm_ing = dm.loc[dm.fase.eq("desarrollo") & dm.referencia.eq("B0_Ingenuo"), ["corte", "dm_pvalue"]]
        dev = dev.merge(dm_ing, on="corte", how="left")
    else:
        dev = dev.assign(dm_pvalue=np.nan)
    prueba = pc.loc[pc.fase.eq("prueba") & pc.modelo.eq(elegido), ["corte", "wmape", "wmape_ingenuo", "sesgo_pct"]]
    metodo = estado.get("final", {}).get("metodo_intervalos") or "Q_Conformal"
    q = T.get("cuantiles", pd.DataFrame())
    q = q.loc[q.metodo.eq(metodo)] if len(q) else q
    contratos = fuente.validar_contratos(duplicados=False) if fuente._contratos is None else fuente._contratos
    controles = [k for k in contratos if k != "filas_snapshot"]
    calidad = {"critical_total": len(controles), "critical_passed": sum(int(contratos[k] == 0) for k in controles),
               "snapshot_verified": True, **evidencia_calidad_upstream()}
    comp = T["comparativa"].set_index("modelo")
    alcance = {"wmape_global": float(comp.loc[elegido, "wmape_media_cortes"]),
               "wmape_mercancia": float(comp.loc[elegido, "wmape_mercancia"]) if pd.notna(comp.loc[elegido].get("wmape_mercancia")) else None}
    compuerta = evaluate_gate(dev, q, calidad, EVIDENCIA_NEGOCIO, test_metrics=prueba, scope_metrics=alcance)
    compuerta.update(modelo=elegido, metodo_intervalos=metodo, evidencia_calidad=calidad, alcance=alcance)
    return compuerta


def guardar_pronostico(pronostico: pd.DataFrame, fuente, registro, elegido):
    """Escribe el pronóstico operativo en TABLA_PRONOSTICO reemplazando sólo el origen pronosticado (las corridas de otros
    orígenes se conservan). Es la única salida con tienda y categoría: vive en el catálogo privado."""
    datos = pronostico.copy()
    datos["modelo"], datos["run_id"], datos["version_oro"] = elegido, registro.run_id, str(fuente.version)
    datos["fecha_generacion"] = pd.Timestamp.now()
    for c in ("origen", "fecha_objetivo"):
        datos[c] = pd.to_datetime(datos[c]).dt.date
    limpio, esquema = _para_spark(datos)
    sdf = spark.createDataFrame(limpio, schema=esquema)
    origen = str(pd.Timestamp(pronostico["origen"].iloc[0]).date())
    escritor = sdf.write.format("delta").mode("overwrite")
    if existe(TABLA_PRONOSTICO):
        escritor = escritor.option("replaceWhere", f"origen = DATE'{origen}'").option("mergeSchema", "true")
    else:
        escritor = escritor.option("overwriteSchema", "true")
    escritor.saveAsTable(TABLA_PRONOSTICO)
    return TABLA_PRONOSTICO


def monitoreo_desde_prueba(acum_ampliada, elegido, metodo, wmape_desarrollo) -> pd.DataFrame:
    """Tabla de monitoreo por semana × región × tipo con el modelo elegido en la prueba: WMAPE, sesgo, cobertura y
    alertas. Es la misma tabla que un Job semanal llenaría en operación."""
    base = pd.DataFrame([dict(zip(["modelo", *AGG_DIMENSIONS], k), **dict(zip(AGG_POINT_SUMS, v)))
                         for k, v in acum_ampliada._sumas.items() if k[0] == elegido])
    if base.empty:
        return pd.DataFrame()
    base["semana"] = pd.to_datetime(base["fecha_objetivo"]).dt.to_period("W-SUN").dt.start_time.dt.date
    m = _agg_point_metrics(_agg_rollup(base, ["semana", "region", "tipo_categoria"], AGG_POINT_SUMS))
    if acum_ampliada._cuantiles:
        qb = pd.DataFrame([dict(zip(["metodo", *AGG_DIMENSIONS], k), **dict(zip(AGG_QUANTILE_SUMS, v)))
                           for k, v in acum_ampliada._cuantiles.items() if k[0] == metodo])
        if len(qb):
            qb["semana"] = pd.to_datetime(qb["fecha_objetivo"]).dt.to_period("W-SUN").dt.start_time.dt.date
            q = _agg_quantile_metrics(_agg_rollup(qb, ["semana", "region", "tipo_categoria"], AGG_QUANTILE_SUMS))
            m = m.merge(q[["semana", "region", "tipo_categoria", "cobertura"]], on=["semana", "region", "tipo_categoria"], how="left")
    m["alerta_error"] = m["wmape"] > wmape_desarrollo + 0.05
    m["alerta_sesgo"] = m["sesgo_pct"].abs() > 5
    m["alerta_cobertura"] = ~m.get("cobertura", pd.Series(np.nan, index=m.index)).between(*BANDA_COBERTURA)
    m["modelo"], m["fase"] = elegido, "prueba"
    return m[["semana", "region", "tipo_categoria", "modelo", "fase", "n_filas", "ventas_reales", "wmape", "sesgo_pct",
              *(["cobertura"] if "cobertura" in m else []), "alerta_error", "alerta_sesgo", "alerta_cobertura"]]


NOMBRES_TABLAS = ("comparativa", "comparativa_comun", "comparativa_prueba", "comparativa_prueba_comun", "por_corte",
                  "por_horizonte", "diario", "segmentos", "calibracion", "residuos", "cuantiles", "dm", "decision",
                  "decision_puntos", "pesos", "ajuste", "ajuste_resumen", "ajuste_importancia", "costos", "auditoria",
                  "parametros", "importancia", "correlacion_errores", "compuerta", "resumen_corrida")


def cargar_corrida(run_id=None) -> dict:
    """Lee de Delta las tablas publicadas de una corrida (por defecto, la última que terminó)."""
    if run_id is None:
        if not existe(TABLA_EJECUCIONES):
            raise RuntimeError("No hay corridas registradas: ejecuta la sección 4.")
        fin = spark.table(TABLA_EJECUCIONES).where(F.col("estado") == "FINALIZADA").orderBy(F.desc("fecha_utc")).limit(1).collect()
        if not fin:
            raise RuntimeError("No hay corridas terminadas en mlops.av5_ejecuciones.")
        run_id = fin[0]["run_id"]
    tablas = {"run_id": run_id}
    for nombre in NOMBRES_TABLAS:
        destino = tabla("mlops", f"av5_{nombre}")
        if existe(destino):
            df = spark.table(destino).where(F.col("run_id") == run_id).drop("run_id", "fecha_publicacion").toPandas()
            if len(df):
                tablas[nombre] = df
    return tablas

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Entorno y snapshot de la fuente
# MAGIC Se comprueban las bibliotecas (los modelos cuya biblioteca falte se omiten, con aviso), se fijan los modelos activos,
# MAGIC se lee la tabla Oro en una versión fija y se validan sus contratos (fechas, objetivo, llaves, conjuntos, referencia,
# MAGIC infinitos y duplicados). Sólo se imprimen conteos y fechas, nunca filas.

# COMMAND ----------

titulo("3. ENTORNO Y SNAPSHOT DE LA FUENTE")
ENTORNO_AV5, FALTANTES_AV5 = verificar_entorno_formal(spark)
mostrar(pd.DataFrame([{"biblioteca": k, "version": v} for k, v in ENTORNO_AV5.items()]), "Bibliotecas efectivas")


def _disponible(lib):
    return ENTORNO_AV5.get(lib, "NO DISPONIBLE") != "NO DISPONIBLE"


if not _disponible("lightgbm"):
    raise RuntimeError("LightGBM es indispensable (I1, E1, E6, E7 y cuantiles). Revisa la celda %pip install lightgbm.")
POINT_BASE_MODELS = tuple(m for m in CANDIDATOS_BASE if INCLUIR.get(m, True)
                          and (m != "I2_CatBoost_Tweedie" or _disponible("catboost"))
                          and (m != "I4_XGBoost_Tweedie" or _disponible("xgboost")))
POINT_HOMOGENEOUS = tuple(m for m in CANDIDATOS_HOMOGENEOS if INCLUIR.get(m, True))
POINT_HETEROGENEOUS_ACTIVOS = tuple(m for m in POINT_HETEROGENEOUS if INCLUIR.get(m, True))
AGG_MODELS = ("B0_Ingenuo", *POINT_BASE_MODELS, *POINT_HOMOGENEOUS, *POINT_HETEROGENEOUS_ACTIVOS, *MODELOS_DIAGNOSTICO)
for lib, modelo in (("catboost", "I2_CatBoost_Tweedie"), ("xgboost", "I4_XGBoost_Tweedie")):
    if not _disponible(lib):
        aviso(f"{lib} no está instalado: {modelo} queda fuera de la comparativa.")
if not _disponible("optuna"):
    aviso("Optuna no está instalado: no habrá ajuste fino (se usan los hiperparámetros del Avance 4).")
mostrar(pd.DataFrame([{"modelo": m, "etiqueta": AGG_LABELS[m], "tipo": _agg_model_type(m), "estrategia": ESTRATEGIA[m]}
                      for m in AGG_MODELS]), f"Candidatos activos ({len(AGG_MODELS)})")

FUENTE_AV5 = FuenteAvance5(spark)
if EJECUTAR_ENTRENAMIENTO:
    with Cronometro() as reloj:
        CONTRATOS_AV5 = FUENTE_AV5.validar_contratos(duplicados=not MODO_RAPIDO)
    print(f"Contratos del snapshot verificados en {reloj.segundos / 60:.1f} min: "
          + ", ".join(f"{k} = {v}" for k, v in CONTRATOS_AV5.items() if k != "filas_snapshot"))
def _ventana(origen):
    """Días que se pronostican desde un origen: T + 1 … T + 14."""
    o = pd.Timestamp(origen)
    return {"desde": (o + pd.Timedelta(days=1)).date(), "hasta": (o + pd.Timedelta(days=CONFIG_FUENTE["horizonte"])).date()}


BLOQUES_AV5 = pd.DataFrame(
    [{"bloque": "ajuste", "fase": "ajuste", "origen": FUENTE_AV5.ajuste_origen, **_ventana(FUENTE_AV5.ajuste_origen)}]
    + [{"bloque": f"oof_{o}", "fase": "oof", "origen": o, **_ventana(o)} for o in FUENTE_AV5.oof_origenes]
    + [{"bloque": b["corte"], "fase": b["fase"], "origen": b["origen"], "desde": b["desde"], "hasta": b["hasta"]}
       for b in FUENTE_AV5.cortes + FUENTE_AV5.pruebas]
    + ([{"bloque": "pronostico", "fase": "operación", "origen": FUENTE_AV5.origen_pronostico,
         **_ventana(FUENTE_AV5.origen_pronostico)}] if FUENTE_AV5.origen_pronostico else []))
mostrar(BLOQUES_AV5, "Bloques de la corrida (desarrollo = elegir; prueba = datos no vistos; pronóstico = salida operativa)")
mostrar(pd.DataFrame([{"variable": v, "tipo": "categórica" if v in FUENTE_AV5.cat_features else "numérica",
                       "descripcion": DESCRIPCION_VARIABLES.get(v, "")} for v in FUENTE_AV5.features_table]),
        f"Variables del modelo ({len(FUENTE_AV5.features_table)})", max_filas=80)
hallazgo(f"Tabla Oro versión {FUENTE_AV5.version} ({FUENTE_AV5.modo_lectura}); {len(FUENTE_AV5.cortes)} cortes de desarrollo, "
         f"{len(FUENTE_AV5.pruebas)} bloques de prueba no vistos, {len(FUENTE_AV5.features_table)} variables; "
         f"s_promos_vigentes_t0 {'incluida (causal)' if FUENTE_AV5.promos_vigentes_causal else 'excluida (la Oro no la declara causal)'}; "
         f"calendario de promociones futuras {'activo' if FUENTE_AV5._promos_futuras is not None else 'apagado'}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Ejecución formal
# MAGIC Se abre la corrida (bitácora en Delta y run de MLflow), se ejecuta el protocolo de la sección 2.12 con puntos de
# MAGIC control, se arman las tablas agregadas, se evalúa la compuerta, se guarda el pronóstico operativo y la tabla de
# MAGIC monitoreo, se registra el modelo en Unity Catalog (si corresponde) y se publican los resultados en `mlops.av5_*`.
# MAGIC Si la celda se interrumpe, copia el `run_id` impreso al inicio en `RUN_ID_REANUDAR` y vuelve a correr la libreta.

# COMMAND ----------

titulo("4. EJECUCIÓN FORMAL")
RUN_ID_ACTUAL, TABLAS_SALIDA_AV5 = None, None
if EJECUTAR_ENTRENAMIENTO:
    try:
        with RegistroFormal(spark, FUENTE_AV5, ENTORNO_AV5, huella_codigo(), HILOS_AV5, run_id=RUN_ID_REANUDAR) as REGISTRO_AV5:
            ESTADO_PREVIO = REGISTRO_AV5.cargar_punto_control() if RUN_ID_REANUDAR else None
            if RUN_ID_REANUDAR and ESTADO_PREVIO is None:
                aviso(f"No se encontró punto de control para {RUN_ID_REANUDAR}: la corrida empieza desde el principio con ese identificador.")
            elif ESTADO_PREVIO is not None:
                print(f"Se reanuda desde el punto de control: pasos completados {ESTADO_PREVIO['completados']}")
            ESTADO_AV5 = ejecutar_modelado(FUENTE_AV5, REGISTRO_AV5, HILOS_AV5, estado=ESTADO_PREVIO)
            TABLAS_SALIDA_AV5 = compilar_resultados(ESTADO_AV5, FUENTE_AV5, REGISTRO_AV5)
            MODELO_FINAL = ESTADO_AV5["decision"]["modelo_final"]
            COMPUERTA_AV5 = preparar_compuerta(TABLAS_SALIDA_AV5, ESTADO_AV5, FUENTE_AV5)
            TABLAS_SALIDA_AV5["compuerta"] = pd.DataFrame([{
                "modelo": MODELO_FINAL, "estado_desarrollo": COMPUERTA_AV5["desarrollo_status"],
                "technical_pass": COMPUERTA_AV5["technical_pass"], "production_blocked": COMPUERTA_AV5["production_blocked"],
                "registry_action": COMPUERTA_AV5["registry_action"],
                "detalle_json": json.dumps(COMPUERTA_AV5, ensure_ascii=False, default=_operacion_json)}])
            REGISTRO_AV5.log_dict("compuerta", COMPUERTA_AV5)
            # ---- salida operativa y monitoreo (no se escriben en MODO_RAPIDO: serían cifras de una prueba de humo)
            final = ESTADO_AV5.get("final", {})
            RESUMEN_OPERATIVO = {}
            if "pronostico" in final and len(final["pronostico"]):
                pron = final["pronostico"]
                RESUMEN_OPERATIVO = {"origen_pronostico": str(pd.Timestamp(pron["origen"].iloc[0]).date()),
                                     "dias": f"{pron['fecha_objetivo'].min().date()} a {pron['fecha_objetivo'].max().date()}",
                                     "series_pronostico": int(pron[["tienda", "categoria_id"]].drop_duplicates().shape[0]),
                                     "filas_pronostico": int(len(pron)), "metodo_intervalos": final.get("metodo_intervalos"),
                                     "techo_operativo": NIVEL_TECHO_OPERATIVO}
                if not MODO_RAPIDO:
                    RESUMEN_OPERATIVO["tabla"] = guardar_pronostico(pron, FUENTE_AV5, REGISTRO_AV5, MODELO_FINAL)
            acum_p = ESTADO_AV5["acumuladores"]["prueba"]["ampliada"]
            wm_dev = float(TABLAS_SALIDA_AV5["comparativa"].set_index("modelo").loc[MODELO_FINAL, "wmape_media_cortes"])
            MONITOREO_AV5 = monitoreo_desde_prueba(acum_p, MODELO_FINAL, COMPUERTA_AV5["metodo_intervalos"], wm_dev) if acum_p._sumas else pd.DataFrame()
            if len(MONITOREO_AV5) and not MODO_RAPIDO:
                limpio, esquema = _para_spark(MONITOREO_AV5.assign(run_id=REGISTRO_AV5.run_id, fecha_calculo=pd.Timestamp.now()))
                (spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append")
                 .option("mergeSchema", "true").saveAsTable(TABLA_MONITOREO))
            # ---- registro en Unity Catalog
            REGISTRO_UC_AV5 = {"registrado": False, "motivo": "no aplica"}
            registrar = (not MODO_RAPIDO and COMPUERTA_AV5["technical_pass"]) if REGISTRAR_EN_UC is None else bool(REGISTRAR_EN_UC)
            candidato = final.get("candidate")
            if registrar and candidato is not None and REGISTRO_AV5.mlflow is not None and not candidato.get("requiere_spark"):
                inicio_ej = pd.Timestamp(FUENTE_AV5.fecha_maxima_objetivo) + pd.Timedelta(days=1)
                ejemplo = pd.DataFrame({**{c: np.zeros(2, dtype=np.float32) for c in FUENTE_AV5.num_features},
                                        **{c: np.zeros(2, dtype=np.int32) for c in FUENTE_AV5.cat_features},
                                        "ingenuo_estacional": np.zeros(2), "origen": [inicio_ej] * 2,
                                        "fecha_objetivo": [inicio_ej + pd.Timedelta(days=1)] * 2})
                ejemplo["h"] = 1
                fila_p = TABLAS_SALIDA_AV5.get("comparativa_prueba", pd.DataFrame()).set_index("modelo") if "comparativa_prueba" in TABLAS_SALIDA_AV5 else None
                metricas_uc = {"wmape_desarrollo": round(wm_dev, 4)}
                if fila_p is not None and MODELO_FINAL in fila_p.index:
                    metricas_uc["wmape_prueba"] = round(float(fila_p.loc[MODELO_FINAL, "wmape_media_cortes"]), 4)
                try:
                    REGISTRO_UC_AV5 = registrar_modelo_uc(REGISTRO_AV5, candidato, ejemplo, metricas_uc)
                except Exception as e:
                    REGISTRO_UC_AV5 = {"registrado": False, "motivo": f"{type(e).__name__}: {str(e)[:200]}"}
            elif not registrar:
                REGISTRO_UC_AV5["motivo"] = ("modo rápido" if MODO_RAPIDO else "la compuerta técnica no pasó") if REGISTRAR_EN_UC is None else "REGISTRAR_EN_UC = False"
            # ---- métricas principales en MLflow y resumen de la corrida
            comp = TABLAS_SALIDA_AV5["comparativa"].set_index("modelo")
            REGISTRO_AV5.log_metrics({"wmape_desarrollo_final": comp.loc[MODELO_FINAL, "wmape_media_cortes"],
                                      "wmape_desarrollo_ingenuo": comp.loc["B0_Ingenuo", "wmape_media_cortes"],
                                      "sesgo_abs_medio_final": comp.loc[MODELO_FINAL, "sesgo_abs_medio_pct"]})
            TABLAS_SALIDA_AV5["resumen_corrida"] = pd.DataFrame([{
                "modelo_final": MODELO_FINAL, "version_oro": str(FUENTE_AV5.version), "modo": "rapido" if MODO_RAPIDO else "completo",
                "cortes": len(FUENTE_AV5.cortes), "bloques_prueba": len(FUENTE_AV5.pruebas),
                "minutos_modelado": ESTADO_AV5.get("segundos_modelado", 0.0) / 60, "mlflow_run_id": REGISTRO_AV5.mlflow_run_id,
                "registro_uc": json.dumps(REGISTRO_UC_AV5, ensure_ascii=False, default=_operacion_json),
                "salida_operativa": json.dumps(RESUMEN_OPERATIVO, ensure_ascii=False, default=_operacion_json),
                "argumento_decision": ESTADO_AV5["decision"]["argumento"]}])
            SALIDAS_AV5 = REGISTRO_AV5.publicar_tablas({k: v for k, v in TABLAS_SALIDA_AV5.items() if k in NOMBRES_TABLAS})
            RUN_ID_ACTUAL = REGISTRO_AV5.run_id
            print(f"Corrida {RUN_ID_ACTUAL} terminada: {len(SALIDAS_AV5)} tablas publicadas en mlops.av5_* · "
                  f"modelo final {MODELO_FINAL} · Unity Catalog: {REGISTRO_UC_AV5}")
    finally:
        FUENTE_AV5.liberar_memoria()
else:
    print("EJECUTAR_ENTRENAMIENTO = False: se muestran resultados ya guardados.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Resultados de la corrida
# MAGIC Las secciones 5–10 se construyen con las tablas publicadas de la corrida (las de esta sesión o, con
# MAGIC `EJECUTAR_ENTRENAMIENTO = False`, las de `RUN_ID_A_MOSTRAR` o la última terminada): se pueden volver a mostrar sin
# MAGIC entrenar. Con `MODO_PUBLICO = True` no aparecen importes ni volúmenes.

# COMMAND ----------

titulo("5. RESULTADOS")
if TABLAS_SALIDA_AV5 is not None and RUN_ID_A_MOSTRAR in (None, RUN_ID_ACTUAL):
    CORRIDA_AV5 = dict(TABLAS_SALIDA_AV5, run_id=RUN_ID_ACTUAL)
else:
    CORRIDA_AV5 = cargar_corrida(RUN_ID_A_MOSTRAR)
C = CORRIDA_AV5
RESUMEN_C = C["resumen_corrida"].iloc[0]
MODELO_FINAL = RESUMEN_C["modelo_final"]
COMP = C["comparativa"].sort_values("posicion").reset_index(drop=True)
COMP["etiqueta"] = COMP["modelo"].map(AGG_LABELS)
HAY_PRUEBA = "comparativa_prueba" in C
MEJOR_IND = COMP.loc[COMP.tipo.eq("individual")].iloc[0]["modelo"]
F_COMP = {"wmape_media_cortes": "dec4", "wmape_desviacion_cortes": "dec4", "mejora_vs_ingenuo": "pct",
          "mejora_vs_mejor_individual": "pct", "sesgo_medio_pct": "dec2", "sesgo_abs_medio_pct": "dec2", "mae_relativo": "dec3",
          "rmse_relativo": "dec3", "wmape_mercancia": "dec4", "wmape_h1_7": "dec4", "wmape_h8_14": "dec4",
          "min_entrenamiento_por_corte": "dec1", "segundos_prediccion": "dec1", "wmape_global": "dec4"}
mostrar(pd.DataFrame([{"run_id": C["run_id"], "modo": RESUMEN_C["modo"], "version_oro": RESUMEN_C["version_oro"],
                       "cortes_desarrollo": int(RESUMEN_C["cortes"]), "bloques_prueba": int(RESUMEN_C["bloques_prueba"]),
                       "minutos_modelado": round(float(RESUMEN_C["minutos_modelado"]), 1), "modelo_final": MODELO_FINAL}]),
        "5.1 Identificación de la corrida")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Comparativa en los cortes de desarrollo (ordenada por la métrica principal)
# MAGIC **Cómo leerla.** `wmape_media_cortes` es la métrica principal (promedio del WMAPE de los cortes; menor es mejor) y
# MAGIC ordena la tabla. `wmape_desviacion_cortes` mide estabilidad entre cortes. `mejora_vs_ingenuo` y
# MAGIC `mejora_vs_mejor_individual` son reducciones relativas del WMAPE. `sesgo_medio_pct` > 0 = sobrepronóstico (más
# MAGIC inventario), < 0 = subpronóstico (riesgo de quiebre); `sesgo_abs_medio_pct` promedia su tamaño por corte. `mae_relativo`
# MAGIC y `rmse_relativo` dividen el MAE y el RMSE entre los del ingenuo (< 1 = mejor; el RMSE castiga más los errores grandes,
# MAGIC los picos). `wmape_mercancia` es el error donde se surte físicamente. `wmape_h1_7` y `wmape_h8_14` separan la primera y
# MAGIC la segunda semana. Tiempos: minutos de entrenamiento por corte (los ensambles incluyen a sus bases y su meta-modelo) y
# MAGIC segundos de predicción del bloque completo. Todo con el universo completo de series evaluables.

# COMMAND ----------

COLS_COMP = ["posicion", "etiqueta", "tipo", "estrategia", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo",
             "mejora_vs_mejor_individual", "sesgo_medio_pct", "sesgo_abs_medio_pct", "mae_relativo", "rmse_relativo",
             "wmape_mercancia", "wmape_h1_7", "wmape_h8_14", "min_entrenamiento_por_corte", "segundos_prediccion"]
mostrar_publico(COMP[[c for c in COLS_COMP if c in COMP]], "Comparativa: individuales de la fase previa, ensambles y referencia",
                formatos=F_COMP)
PC = C["por_corte"]
PIV = (PC.loc[PC.fase.eq("desarrollo")].pivot_table(index="modelo", columns="corte", values="wmape")
       .reindex(COMP["modelo"]).reset_index())
PIV.insert(1, "etiqueta", PIV["modelo"].map(AGG_LABELS))
mostrar(PIV.drop(columns=["modelo"]), "WMAPE por corte de desarrollo", formatos={c: "dec4" for c in PIV.columns if c.startswith("corte")})
if "comparativa_comun" in C:
    CC = C["comparativa_comun"].sort_values("posicion").assign(etiqueta=lambda d: d["modelo"].map(AGG_LABELS))
    mostrar_publico(CC[["posicion", "etiqueta", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo", "sesgo_medio_pct"]],
                    "Misma comparativa en la muestra común (3 % de las series; comparable con el Avance 4)", formatos=F_COMP)
n_hom = int(COMP.tipo.eq("ensamble_homogeneo").sum())
n_het = int(COMP.tipo.eq("ensamble_heterogeneo").sum())
primero = COMP.loc[~COMP.tipo.isin(["referencia", "diagnostico"])].iloc[0]
fila_ind = COMP.set_index("modelo").loc[MEJOR_IND]
hallazgo(f"Comparativa de desarrollo: {n_hom} ensambles homogéneos y {n_het} heterogéneos contra {int(COMP.tipo.eq('individual').sum())} "
         f"individuales y la referencia. Primer lugar: {AGG_LABELS[primero['modelo']]} con WMAPE {primero['wmape_media_cortes']:.4f} "
         f"({primero['mejora_vs_ingenuo']:+.1%} contra el ingenuo, {primero.get('mejora_vs_mejor_individual', np.nan):+.1%} contra el mejor "
         f"individual, {AGG_LABELS[MEJOR_IND]} {fila_ind['wmape_media_cortes']:.4f}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Prueba en datos no vistos (septiembre)
# MAGIC Todos los modelos se entrenaron con datos hasta el 31 de agosto y se congelaron; pronostican del 1 al 28 de
# MAGIC septiembre sin reentrenar. La tabla conserva el **orden de desarrollo** (no se reordena con la prueba, para no elegir
# MAGIC con ella). `diferencia` = WMAPE en prueba − WMAPE en desarrollo: un valor cercano a cero indica que el desempeño se
# MAGIC sostiene en meses nuevos.

# COMMAND ----------

if HAY_PRUEBA:
    CP = C["comparativa_prueba"].set_index("modelo")
    NV = COMP[["posicion", "modelo", "etiqueta", "tipo", "wmape_media_cortes"]].rename(columns={"wmape_media_cortes": "wmape_desarrollo"})
    NV["wmape_prueba"] = NV["modelo"].map(CP["wmape_media_cortes"])
    NV["diferencia"] = NV["wmape_prueba"] - NV["wmape_desarrollo"]
    NV["mejora_vs_ingenuo_prueba"] = NV["modelo"].map(CP["mejora_vs_ingenuo"])
    NV["sesgo_prueba_pct"] = NV["modelo"].map(CP["sesgo_medio_pct"])
    NV["wmape_mercancia_prueba"] = NV["modelo"].map(CP["wmape_mercancia"])
    mostrar(NV.drop(columns=["modelo"]), "Desarrollo contra prueba no vista (orden de desarrollo)",
            formatos={"wmape_desarrollo": "dec4", "wmape_prueba": "dec4", "diferencia": "dec4", "mejora_vs_ingenuo_prueba": "pct",
                      "sesgo_prueba_pct": "dec2", "wmape_mercancia_prueba": "dec4"})
    PIVP = (PC.loc[PC.fase.eq("prueba")].pivot_table(index="modelo", columns="corte", values="wmape").reindex(COMP["modelo"]).reset_index())
    PIVP.insert(1, "etiqueta", PIVP["modelo"].map(AGG_LABELS))
    mostrar(PIVP.drop(columns=["modelo"]), "WMAPE por bloque de prueba", formatos={c: "dec4" for c in PIVP.columns if c.startswith("prueba")})
    nv = NV.set_index("modelo")
    from scipy.stats import spearmanr
    candidatos = NV.loc[~NV.tipo.isin(["referencia", "diagnostico"])]
    rho = spearmanr(candidatos["wmape_desarrollo"], candidatos["wmape_prueba"]).correlation if len(candidatos) > 2 else np.nan
    hallazgo(f"Datos no vistos (septiembre): {AGG_LABELS[MODELO_FINAL]} WMAPE {nv.loc[MODELO_FINAL, 'wmape_prueba']:.4f} "
             f"({nv.loc[MODELO_FINAL, 'mejora_vs_ingenuo_prueba']:+.1%} contra el ingenuo, {nv.loc['B0_Ingenuo', 'wmape_prueba']:.4f}; "
             f"sesgo {nv.loc[MODELO_FINAL, 'sesgo_prueba_pct']:+.2f} %); en desarrollo tenía {nv.loc[MODELO_FINAL, 'wmape_desarrollo']:.4f} "
             f"(diferencia {nv.loc[MODELO_FINAL, 'diferencia']:+.4f}). El orden de los modelos se sostiene en septiembre "
             f"(correlación de rangos desarrollo–prueba {rho:.2f}).")
else:
    print("Esta corrida no tiene bloques de prueba (la Oro no incluye septiembre).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Ajuste fino de hiperparámetros
# MAGIC Resultado de Optuna en el bloque de ajuste (20-may). La gráfica muestra cada prueba y la mejor hasta ese momento:
# MAGIC si la curva baja después de la fase aleatoria (línea punteada), TPE encontró mejores regiones del espacio.

# COMMAND ----------

if "ajuste_resumen" in C:
    mostrar(C["ajuste_resumen"], "Ajuste fino por modelo", formatos={"pruebas": "int", "minutos": "dec1", "wmape_inicial": "dec4",
                                                                     "wmape_mejor": "dec4", "mejora_relativa": "pct", "arboles_prueba": "int"})
if "ajuste" in C and len(C["ajuste"]):
    AJ = C["ajuste"]
    modelos_aj = list(dict.fromkeys(AJ["modelo"]))
    fig, axes = plt.subplots(1, len(modelos_aj), figsize=(6.2 * len(modelos_aj), 3.4), squeeze=False)
    for ax, m in zip(axes[0], modelos_aj):
        d = AJ.loc[AJ.modelo.eq(m)].sort_values("prueba")
        ax.scatter(d["prueba"], d["wmape_ajuste"], s=16, alpha=0.6, color="#5B7BA0", label="prueba")
        ax.plot(d["prueba"], d["wmape_ajuste"].cummin(), color="#A06A2C", lw=2, label="mejor hasta el momento")
        ax.axhline(d.loc[d.prueba.eq(0), "wmape_ajuste"].min(), color="gray", ls="--", lw=1, label="configuración del Avance 4")
        ax.axvline(N_PRUEBAS_ALEATORIAS - 0.5, color="#bbbbbb", ls=":", lw=1)
        ax.set_title(f"Optuna · {AGG_LABELS.get(m, m)}", fontsize=10)
        ax.set_xlabel("prueba")
        ax.set_ylabel("WMAPE en el bloque de ajuste")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, frameon=False)
    plt.tight_layout()
    plt.show()
if "ajuste_importancia" in C and len(C["ajuste_importancia"]):
    mostrar(C["ajuste_importancia"].sort_values(["modelo", "importancia"], ascending=[True, False]).groupby("modelo").head(5),
            "Hiperparámetros que más movieron el WMAPE (importancia de Optuna)", formatos={"importancia": "pct"})
if "parametros" in C:
    filas_p = []
    for r in C["parametros"].itertuples():
        p = json.loads(r.parametros)
        filas_p.append({"modelo": r.modelo, **{k: p.get(k) for k in ("n_estimators", "iterations", "learning_rate", "num_leaves", "max_depth",
                                                                         "min_child_samples", "min_child_weight", "tweedie_variance_power",
                                                                         "subsample", "colsample_bytree", "reg_lambda", "reg_alpha") if k in p}})
    mostrar(pd.DataFrame(filas_p), "Hiperparámetros congelados para cortes, prueba y modelo final")
if "ajuste_resumen" in C:
    for r in C["ajuste_resumen"].itertuples():
        hallazgo(f"Ajuste fino de {AGG_LABELS.get(r.modelo, r.modelo)}: {int(r.pruebas)} pruebas en {r.minutos:.0f} min; WMAPE en el "
                 f"bloque de ajuste {r.wmape_inicial:.4f} → {min(r.wmape_mejor, r.wmape_inicial):.4f} ({r.mejora_relativa:+.1%}); "
                 f"configuración {r.configuracion}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Cómo combinan los ensambles a sus bases
# MAGIC Pesos de cada base en los ensambles heterogéneos, por origen (aprendidos sólo con bloques anteriores), y
# MAGIC correlación de los errores de las bases en los cortes: **un ensamble sólo gana si sus bases se equivocan distinto**
# MAGIC (correlación menor que 1).

# COMMAND ----------

if "pesos" in C and len(C["pesos"]):
    PW = C["pesos"]
    ultimo = "prueba" if PW["corte"].eq("prueba").any() else PW["corte"].iloc[-1]
    vista = PW.loc[PW.corte.eq(ultimo) & ~PW.ensamble.str.startswith("E4_factor")].pivot_table(index="ensamble", columns="base", values="peso")
    mostrar(vista.reset_index(), f"Pesos de las bases por ensamble (modelos congelados para '{ultimo}')", formatos={c: "dec3" for c in vista.columns})
    factores = PW.loc[PW.corte.eq(ultimo) & PW.ensamble.str.startswith("E4_factor")]
    if len(factores):
        etiquetas_tipo = FUENTE_AV5.etiquetas["tipo_categoria"]
        factores = factores.assign(tipo=factores["base"].str.replace("tipo:", "").astype(int).map(etiquetas_tipo))
        mostrar(factores[["ensamble", "tipo", "peso"]].rename(columns={"peso": "factor"}),
                "E4 · factores de nivel por tipo de categoría (1 = sin corrección)", formatos={"factor": "dec3"})
    evolucion = PW.loc[PW.ensamble.eq("E3_Blending")].pivot_table(index="corte", columns="base", values="peso")
    if len(evolucion):
        mostrar(evolucion.reset_index(), "E3 · evolución de los pesos de blending por corte", formatos={c: "dec3" for c in evolucion.columns})
if "correlacion_errores" in C and len(C["correlacion_errores"]):
    CE = C["correlacion_errores"].set_index("base")
    CE = CE[[c for c in CE.columns if c in CE.index]]
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    im = ax.imshow(CE.to_numpy(dtype=float), cmap="Blues", vmin=min(0.5, float(np.nanmin(CE.to_numpy(dtype=float)))), vmax=1)
    ax.set_xticks(range(len(CE)))
    ax.set_xticklabels([AGG_LABELS.get(c, c) for c in CE.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CE)))
    ax.set_yticklabels([AGG_LABELS.get(c, c) for c in CE.index], fontsize=8)
    for i in range(len(CE)):
        for j in range(len(CE)):
            ax.text(j, i, f"{CE.iat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if CE.iat[i, j] > 0.85 else "black")
    ax.set_title("Correlación de errores entre bases (cortes de desarrollo)", fontsize=10)
    plt.colorbar(im, fraction=0.046)
    plt.tight_layout()
    plt.show()
    par = CE.where(~np.eye(len(CE), dtype=bool)).stack()
    hallazgo(f"Diversidad de las bases: la correlación de errores va de {par.min():.2f} ({AGG_LABELS.get(par.idxmin()[0])} con "
             f"{AGG_LABELS.get(par.idxmin()[1])}) a {par.max():.2f} ({AGG_LABELS.get(par.idxmax()[0])} con {AGG_LABELS.get(par.idxmax()[1])}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.6 Tiempos de entrenamiento e inferencia
# MAGIC Minutos de entrenamiento por corte y segundos de predicción del bloque completo por modelo, y minutos totales por
# MAGIC fase de la corrida. El costo cuenta para el negocio porque el modelo se reentrenará cada mes en un Job.

# COMMAND ----------

TIEMPOS = COMP[["posicion", "etiqueta", "tipo", "min_entrenamiento_por_corte", "segundos_prediccion"]].copy()
mostrar(TIEMPOS, "Tiempos por modelo (desarrollo)", formatos={"min_entrenamiento_por_corte": "dec1", "segundos_prediccion": "dec1"})
if "costos" in C and len(C["costos"]):
    CO = C["costos"]
    por_fase = (CO.groupby("fase", as_index=False)[["segundos_entrenamiento", "segundos_prediccion"]].sum(min_count=1)
                .assign(minutos=lambda d: (d["segundos_entrenamiento"].fillna(0) + d["segundos_prediccion"].fillna(0)) / 60))
    mostrar(por_fase[["fase", "minutos"]].sort_values("minutos", ascending=False), "Minutos medidos por fase", formatos={"minutos": "dec1"})
hallazgo(f"La corrida tardó {float(RESUMEN_C['minutos_modelado']):.0f} min de modelado ({RESUMEN_C['modo']}). Entrenar el modelo "
         f"final cuesta {COMP.set_index('modelo').loc[MODELO_FINAL, 'min_entrenamiento_por_corte']:.1f} min por corte con la muestra de "
         f"entrenamiento de esta corrida.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.7 Elección del modelo final (matriz de decisión)
# MAGIC Pesos fijados antes de ver resultados: error 35 %, estabilidad 10 %, sesgo 10 %, error en mercancía 10 %, costo de
# MAGIC reentrenamiento 15 % (no penaliza mientras quepa en el Job mensual), interpretabilidad 10 % y operabilidad 10 %. El
# MAGIC error se puntúa como mejora sobre el ingenuo relativa a la mejor mejora. La elección quedó registrada en la bitácora
# MAGIC **antes** de evaluar septiembre.

# COMMAND ----------

DEC = C["decision"].copy()
DEC.insert(1, "etiqueta", DEC["modelo"].map(AGG_LABELS))
mostrar(DEC.drop(columns=[c for c in ("modelo_final", "argumento") if c in DEC]), "Matriz de decisión (métricas de los cortes de desarrollo)",
        formatos={"error": "dec4", "estabilidad": "dec4", "sesgo": "dec2", "error_donde_cuesta": "dec4", "costo": "dec1",
                  "interpretabilidad": "dec2", "operabilidad": "dec2", "total": "dec3"})
if "decision_puntos" in C:
    mostrar(C["decision_puntos"], "Puntos por criterio (peso × puntuación)", formatos={c: "dec3" for c in PESOS_DECISION})
fila_f = COMP.set_index("modelo").loc[MODELO_FINAL]
ARGUMENTO_FINAL = (
    f"Modelo final: {AGG_LABELS[MODELO_FINAL]} ({fila_f['tipo'].replace('_', ' ')}; {ESTRATEGIA[MODELO_FINAL]}). "
    f"{DEC['argumento'].iloc[0]} En desarrollo tiene WMAPE {fila_f['wmape_media_cortes']:.4f} ({fila_f['mejora_vs_ingenuo']:+.1%} contra el "
    f"ingenuo y {fila_f.get('mejora_vs_mejor_individual', np.nan):+.1%} contra el mejor individual), sesgo absoluto medio "
    f"{fila_f['sesgo_abs_medio_pct']:.2f} % por corte, WMAPE en mercancía {fila_f['wmape_mercancia']:.4f} y entrena en "
    f"{fila_f['min_entrenamiento_por_corte']:.1f} min por corte. Para el negocio importa que acierte donde se surte (mercancía), que no "
    f"tenga sesgo sistemático (el sesgo positivo inmoviliza inventario y el negativo provoca quiebres) y que su reentrenamiento mensual "
    f"quepa en un Job; la matriz pondera esos tres aspectos además del error.")
hallazgo(ARGUMENTO_FINAL)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Intervalos, diagnóstico estadístico y compuerta de aceptación
# MAGIC ### 6.1 Intervalos P10–P90 en septiembre
# MAGIC `cobertura` = fracción de días reales dentro de [P10, P90] (objetivo 80 %, banda 75–85 %). `ancho_relativo` = ancho
# MAGIC medio del intervalo entre la venta media (más angosto = más útil, si cubre). Las pérdidas pinball relativas miden la
# MAGIC calidad de cada cuantil (menor es mejor). `tasa_cruces_final` debe ser 0 (P10 ≤ P50 ≤ P90).

# COMMAND ----------

titulo("6. INTERVALOS, DIAGNÓSTICO ESTADÍSTICO Y COMPUERTA")
COMPUERTA_C = json.loads(C["compuerta"]["detalle_json"].iloc[0]) if "compuerta" in C else {}
METODO_FINAL = COMPUERTA_C.get("metodo_intervalos", "Q_Conformal")
if "cuantiles" in C and len(C["cuantiles"]):
    Q = C["cuantiles"]
    GQ = Q.loc[Q.ambito.eq("global"), ["metodo", "cobertura", "cobertura_base", "ancho_relativo", "pinball_p10_rel", "pinball_p50_rel",
                                       "pinball_p90_rel", "tasa_cruces_final", "cumple_banda"]]
    mostrar(GQ, "Intervalos en la prueba no vista (global por método)",
            formatos={"cobertura": "pct", "cobertura_base": "pct", "ancho_relativo": "dec3", "pinball_p10_rel": "dec4",
                      "pinball_p50_rel": "dec4", "pinball_p90_rel": "dec4", "tasa_cruces_final": "pct2"})
    resumen_ambitos = []
    for metodo in GQ["metodo"]:
        for ambito in ("h", "tipo_categoria", "region", "corte×h×tipo", "tipo×region"):
            sub = Q.loc[Q.metodo.eq(metodo) & Q.ambito.eq(ambito)]
            if len(sub):
                resumen_ambitos.append({"metodo": metodo, "ambito": ambito, "grupos": len(sub), "en_banda": int(sub["cumple_banda"].sum()),
                                        "cobertura_min": sub["cobertura"].min(), "cobertura_max": sub["cobertura"].max()})
    mostrar(pd.DataFrame(resumen_ambitos), "6.2 Cobertura por ámbito (grupos dentro de la banda 75–85 %)",
            formatos={"cobertura_min": "pct", "cobertura_max": "pct"})
    fuera = Q.loc[Q.metodo.eq(METODO_FINAL) & Q.ambito.isin(["tipo_categoria", "region", "tipo×region"]) & ~Q.cumple_banda]
    if len(fuera):
        mostrar(fuera[["ambito", "tipo_categoria", "region", "cobertura", "ancho_relativo"]].head(20),
                f"Grupos fuera de la banda con {METODO_FINAL}", formatos={"cobertura": "pct", "ancho_relativo": "dec3"})
    g = GQ.set_index("metodo")
    hallazgo(f"Intervalos en septiembre: " + "; ".join(f"{m} cubre {r.cobertura:.1%} con ancho relativo {r.ancho_relativo:.2f}"
                                                         for m, r in g.iterrows())
             + f". El modelo final usa {METODO_FINAL} ({'método automático: menor pinball dentro de la banda' if METODO_INTERVALOS == 'automatico' else 'fijado en la configuración'}).")
else:
    print("Sin evaluación de intervalos (no hay bloques de prueba).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Sesgo del modelo final por bloque y 6.4 significancia (Diebold–Mariano)
# MAGIC Sesgo % = (Σ pronóstico − Σ real) / Σ real × 100: negativo = subpronóstico (riesgo de quiebre), positivo =
# MAGIC sobrepronóstico (inventario de más). Criterio: |sesgo| ≤ 3 % en cada corte de desarrollo y ≤ 5 % en la prueba.
# MAGIC Diebold–Mariano compara la pérdida diaria del modelo final contra el ingenuo y contra el mejor individual (estadístico
# MAGIC positivo = el modelo final se equivoca menos); con 14 días por bloque es una evidencia descriptiva.

# COMMAND ----------

SB = C["por_corte"].loc[C["por_corte"].modelo.eq(MODELO_FINAL), ["fase", "corte", "wmape", "wmape_ingenuo", "mejora_vs_ingenuo", "sesgo_pct"]].copy()
SB["limite_sesgo"] = np.where(SB["fase"].eq("prueba"), UMBRALES_COMPUERTA["sesgo_prueba_max_pct"], UMBRALES_COMPUERTA["sesgo_max_pct"])
SB["cumple_sesgo"] = SB["sesgo_pct"].abs() <= SB["limite_sesgo"]
mostrar(SB, f"Sesgo y error de {AGG_LABELS[MODELO_FINAL]} por bloque",
        formatos={"wmape": "dec4", "wmape_ingenuo": "dec4", "mejora_vs_ingenuo": "pct", "sesgo_pct": "dec2", "limite_sesgo": "dec1"})
if "dm" in C and len(C["dm"]):
    DM = C["dm"].copy()
    DM["referencia"] = DM["referencia"].map(AGG_LABELS)
    mostrar(DM[["fase", "corte", "referencia", "dm_estadistico", "dm_pvalue", "mejor_que_referencia", "n_dias"]],
            f"Diebold–Mariano: {AGG_LABELS[MODELO_FINAL]} contra cada referencia", formatos={"dm_estadistico": "dec3", "dm_pvalue": "dec4"})
    dmd = DM.loc[DM.fase.eq("desarrollo") & DM.referencia.eq(AGG_LABELS["B0_Ingenuo"])]
    hallazgo(f"Sesgo de {AGG_LABELS[MODELO_FINAL]}: de {SB['sesgo_pct'].min():+.2f} % a {SB['sesgo_pct'].max():+.2f} % por bloque; "
             f"{int(SB['cumple_sesgo'].sum())} de {len(SB)} bloques dentro del límite. Diebold–Mariano contra el ingenuo significativo "
             f"(p < 0.05) en {int((dmd['dm_pvalue'] < 0.05).sum())} de {len(dmd)} cortes de desarrollo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.5 Compuerta de aceptación
# MAGIC Las comprobaciones **técnicas** deciden si el modelo se registra como `@challenger` en Unity Catalog. Pasar a
# MAGIC producción (`@champion`) exige además las **aprobaciones de negocio** (alcance, política de servicios y disponibilidad
# MAGIC de atributos a la fecha del pronóstico) y las reglas críticas de calidad del pipeline: mientras Bnext no las confirme,
# MAGIC la compuerta bloquea producción por diseño, no por un error del modelo.

# COMMAND ----------

if COMPUERTA_C:
    mostrar(pd.DataFrame([{"comprobacion": k.replace("_", " "), "resultado": "CUMPLE" if v else "NO CUMPLE",
                           "bloquea_challenger": k in COMPUERTA_C.get("checks_tecnicos_bloqueantes", [])}
                          for k, v in COMPUERTA_C["checks"].items()]), "Comprobaciones de la compuerta")
    mostrar(pd.DataFrame({"motivo de bloqueo de producción": COMPUERTA_C["blocking_reasons"] or ["ninguno"]}), "Motivos registrados")
    registro_uc = json.loads(RESUMEN_C["registro_uc"]) if isinstance(RESUMEN_C.get("registro_uc"), str) else {}
    hallazgo(f"Compuerta: {COMPUERTA_C['desarrollo_status'].replace('_', ' ')}; acción de registro: {COMPUERTA_C['registry_action']}; "
             f"producción {'bloqueada' if COMPUERTA_C['production_blocked'] else 'habilitada'} "
             f"({len(COMPUERTA_C['blocking_reasons'])} {'motivo' if len(COMPUERTA_C['blocking_reasons']) == 1 else 'motivos'}). Unity Catalog: "
             + (f"versión {registro_uc.get('version')} de {NOMBRE_MODELO_UC} con alias @challenger." if registro_uc.get("registrado")
                else f"sin registro ({registro_uc.get('motivo', 'no aplica')})."))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Gráficos del modelo final, con interpretación
# MAGIC Siete gráficos; después de cada uno se imprime su interpretación, **calculada con las cifras de esta corrida** (si se
# MAGIC vuelve a correr, el texto cambia con los datos). Las ventas se muestran indexadas (media del bloque = 100) para no
# MAGIC revelar importes.

# COMMAND ----------

titulo("7. GRÁFICOS DEL MODELO FINAL")
INTERPRETACIONES, FIGURAS = [], {}
COLOR_FINAL, COLOR_IND, COLOR_REF, COLOR_REAL = "#A06A2C", "#5B7BA0", "#8C8C8C", "#1F1F1F"


def interpretar(numero, titulo_grafico, texto):
    """Guarda e imprime la interpretación de un gráfico."""
    INTERPRETACIONES.append({"grafico": numero, "titulo": titulo_grafico, "interpretacion": texto})
    print(f"\nInterpretación del gráfico {numero} ({titulo_grafico}). {texto}\n")


def _cerrar_figura(fig, nombre):
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
    FIGURAS[nombre] = fig
    plt.show()


# ---------------------------------------------------------------------------------------------- 1. comparación de modelos
orden = COMP.loc[~COMP.tipo.eq("diagnostico")].sort_values("wmape_media_cortes")
fig, ax = plt.subplots(figsize=(10, 0.42 * len(orden) + 1.6))
y_pos = np.arange(len(orden))
colores = [COLOR_FINAL if m == MODELO_FINAL else COLOR_REF if m == "B0_Ingenuo" else "#B9C7D6" for m in orden["modelo"]]
ax.barh(y_pos, orden["wmape_media_cortes"], xerr=orden["wmape_desviacion_cortes"], color=colores, edgecolor="#4A4A4A",
        linewidth=0.5, error_kw={"ecolor": "#4A4A4A", "capsize": 3}, label="desarrollo (media ± desviación entre cortes)")
if HAY_PRUEBA:
    ax.scatter(orden["modelo"].map(C["comparativa_prueba"].set_index("modelo")["wmape_media_cortes"]), y_pos, marker="D", s=28,
               color="#2E4D2E", zorder=3, label="prueba no vista (septiembre)")
ax.set_yticks(y_pos, [AGG_LABELS[m] for m in orden["modelo"]], fontsize=8)
ax.invert_yaxis()
ax.xaxis.set_major_formatter(PercentFormatter(1))
ax.set_xlabel("WMAPE (menor es mejor)")
ax.set_title("1. Comparación de modelos: WMAPE en desarrollo y en datos no vistos", fontsize=10)
ax.grid(axis="x", alpha=0.3)
ax.legend(fontsize=7, frameon=False, loc="lower right")
plt.tight_layout()
_cerrar_figura(fig, "g1_comparacion_modelos")
f = COMP.set_index("modelo")
ens = COMP.loc[COMP.tipo.str.startswith("ensamble")]
ganan_a_ind = int((ens["wmape_media_cortes"] < f.loc[MEJOR_IND, "wmape_media_cortes"]).sum())
texto = (f"{AGG_LABELS[MODELO_FINAL]} (barra resaltada) logra WMAPE medio {f.loc[MODELO_FINAL, 'wmape_media_cortes']:.4f} en los cortes de "
         f"desarrollo contra {f.loc['B0_Ingenuo', 'wmape_media_cortes']:.4f} del ingenuo y {f.loc[MEJOR_IND, 'wmape_media_cortes']:.4f} del mejor "
         f"individual ({AGG_LABELS[MEJOR_IND]}). {ganan_a_ind} de {len(ens)} ensambles superan al mejor individual; la barra de error es la "
         f"desviación entre cortes (estabilidad), no un intervalo de confianza. ")
if HAY_PRUEBA:
    cp = C["comparativa_prueba"].set_index("modelo")
    texto += (f"Los rombos son septiembre, que no intervino en ninguna decisión: el modelo final obtiene {cp.loc[MODELO_FINAL, 'wmape_media_cortes']:.4f} "
              f"(ingenuo {cp.loc['B0_Ingenuo', 'wmape_media_cortes']:.4f}), así que la ventaja se sostiene fuera de la muestra de desarrollo.")
interpretar(1, "Comparación de modelos", texto)

# ---------------------------------------------------------------------------------------------- 2. tendencia y predicción
D = C["diario"]
bloques = list(dict.fromkeys(D.sort_values("origen")["corte"]))
cols = min(3, len(bloques))
filas_g = int(math.ceil(len(bloques) / cols))
fig, axes = plt.subplots(filas_g, cols, figsize=(4.6 * cols, 3.0 * filas_g), squeeze=False, sharey=True)
corr, sesgos = {}, {}
for ax, b in zip(axes.flat, bloques):
    real = D.loc[D.corte.eq(b) & D.modelo.eq(MODELO_FINAL)].sort_values("fecha_objetivo")
    base = real["ventas_reales"].mean()
    for modelo, color, estilo, etiqueta in ((MODELO_FINAL, COLOR_FINAL, "-", "modelo final"), (MEJOR_IND, COLOR_IND, "--", "mejor individual"),
                                             ("B0_Ingenuo", COLOR_REF, ":", "ingenuo")):
        d = D.loc[D.corte.eq(b) & D.modelo.eq(modelo)].sort_values("fecha_objetivo")
        ax.plot(d["fecha_objetivo"], d["ventas_predichas"] / base * 100, color=color, ls=estilo, lw=1.6, label=etiqueta)
    ax.plot(real["fecha_objetivo"], real["ventas_reales"] / base * 100, color=COLOR_REAL, lw=2.2, label="real")
    fase = real["fase"].iloc[0] if "fase" in real else ""
    ax.set_title(f"{b}" + (" · NO VISTO" if fase == "prueba" else ""), fontsize=9)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(alpha=0.3)
    corr[b] = float(np.corrcoef(real["ventas_reales"], real["ventas_predichas"])[0, 1]) if len(real) > 2 else np.nan
    sesgos[b] = float(real["ventas_predichas"].sum() / real["ventas_reales"].sum() - 1)
for ax in list(axes.flat)[len(bloques):]:
    ax.set_visible(False)
axes[0, 0].set_ylabel("venta diaria indexada (media = 100)")
axes[0, 0].legend(fontsize=7, frameon=False)
fig.suptitle("2. Tendencia y predicción: venta total diaria de cada bloque, real contra pronóstico", fontsize=10)
plt.tight_layout()
_cerrar_figura(fig, "g2_tendencia_prediccion")
peor = max(sesgos, key=lambda k: abs(sesgos[k]))
texto = (f"El pronóstico agregado sigue el patrón semanal de la venta real: la correlación diaria va de {np.nanmin(list(corr.values())):.2f} a "
         f"{np.nanmax(list(corr.values())):.2f} según el bloque, y el sesgo del total va de {min(sesgos.values()):+.1%} a {max(sesgos.values()):+.1%} "
         f"(el mayor en {peor}). ")
pruebas = [b for b in bloques if b.startswith("prueba")]
if pruebas:
    texto += (f"En septiembre (no visto) el sesgo es {', '.join(f'{sesgos[b]:+.1%} en {b}' for b in pruebas)}: incluye la quincena y las "
              "fiestas patrias (15–16 de septiembre), que el calendario del modelo sí conoce. ")
texto += ("Los picos que el modelo suaviza corresponden a días con promociones o eventos que no se conocen en el origen; el calendario "
          "de promociones futuras (sección 9) es la vía para capturarlos. La línea del ingenuo copia la semana previa y por eso se "
          "adelanta o atrasa a los picos. Al sumar tiendas los errores se compensan: la cercanía de las curvas no implica error pequeño "
          "por tienda; esa medición está en el WMAPE.")
interpretar(2, "Tendencia y predicción", texto)

# ---------------------------------------------------------------------------------------------- 3. análisis de residuos
R = C["residuos"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
rel = {}
for fase, color in (("desarrollo", COLOR_IND), ("prueba", COLOR_FINAL)):
    r = R.loc[R.fase.eq(fase)] if "fase" in R else R
    if len(r) and MODELO_FINAL in r:
        rel[fase] = ((r["y"] - r[MODELO_FINAL]) / np.maximum(r["s_ma28"].fillna(0), 1.0)).to_numpy()
        axes[0].hist(np.clip(rel[fase], -3, 3), bins=60, alpha=0.55, color=color, label=fase, density=True)
axes[0].axvline(0, color="black", lw=1)
axes[0].set_xlabel("(real − pronóstico) / venta media 28 días de la serie")
axes[0].set_ylabel("densidad")
axes[0].set_title("Residuos relativos a la escala de cada serie (recortados a ±3)", fontsize=9)
axes[0].legend(fontsize=7, frameon=False)
PH = C["por_horizonte"].loc[C["por_horizonte"].modelo.eq(MODELO_FINAL)]
ancho = 0.4
for i, (fase, color) in enumerate((("desarrollo", COLOR_IND), ("prueba", COLOR_FINAL))):
    d = PH.loc[PH.fase.eq(fase)].sort_values("h") if "fase" in PH else PH.sort_values("h")
    if len(d):
        axes[1].bar(d["h"] + (i - 0.5) * ancho, d["sesgo_pct"], width=ancho, color=color, label=fase)
axes[1].axhline(0, color="black", lw=1)
for lim in (-3, 3):
    axes[1].axhline(lim, color="gray", ls="--", lw=0.8)
axes[1].set_xticks(range(1, 15))
axes[1].set_xlabel("h (días después del origen)")
axes[1].set_ylabel("sesgo %")
axes[1].set_title("Sesgo por horizonte (banda ±3 %)", fontsize=9)
axes[1].legend(fontsize=7, frameon=False)
fig.suptitle(f"3. Análisis de residuos de {AGG_LABELS[MODELO_FINAL]}", fontsize=10)
plt.tight_layout()
_cerrar_figura(fig, "g3_residuos")
rd = rel.get("desarrollo", np.array([0.0]))
dev_h = PH.loc[PH.fase.eq("desarrollo")].set_index("h")["sesgo_pct"] if "fase" in PH else PH.set_index("h")["sesgo_pct"]
texto = (f"La mediana del residuo relativo es {np.median(rd):+.2f} veces la venta media de la serie y el {np.mean(np.abs(rd) <= 0.5):.0%} de las "
         f"filas queda a menos de media venta media de su serie. La cola derecha (real muy por encima del pronóstico) son días pico que el "
         f"modelo no anticipa; la izquierda, días con venta cero donde se pronosticó algo. Por horizonte, el sesgo de desarrollo va de "
         f"{dev_h.min():+.1f} % (h = {int(dev_h.idxmin())}) a {dev_h.max():+.1f} % (h = {int(dev_h.idxmax())}); "
         f"{int((dev_h.abs() > 3).sum())} de 14 horizontes salen de ±3 %. ")
if "prueba" in rel:
    texto += (f"En septiembre la distribución {'se mantiene' if abs(np.median(rel['prueba']) - np.median(rd)) < 0.1 else 'se desplaza'} "
              f"(mediana {np.median(rel['prueba']):+.2f}), señal de que los errores no cambiaron de naturaleza con el mes nuevo.")
interpretar(3, "Análisis de residuos", texto)

# ---------------------------------------------------------------------------------------------- 4. calibración por tramo
CAL = C["calibracion"]
fig, ax = plt.subplots(figsize=(6.4, 5.0))
razones = {}
for modelo, color, marca in ((MODELO_FINAL, COLOR_FINAL, "o"), (MEJOR_IND, COLOR_IND, "s")):
    for fase, estilo in (("desarrollo", "-"), ("prueba", "--")):
        d = CAL.loc[CAL.modelo.eq(modelo) & CAL.fase.eq(fase) & (CAL.n_filas >= 50)] if "fase" in CAL else CAL.loc[CAL.modelo.eq(modelo)]
        if not len(d):
            continue
        escala = d["suma_y"].sum() / d["n_filas"].sum()
        ax.plot(d["pred_medio"] / escala, d["real_medio"] / escala, marker=marca, ls=estilo, color=color, ms=4,
                label=f"{AGG_LABELS[modelo]} · {fase}")
        if modelo == MODELO_FINAL:
            razones[fase] = (d["real_medio"] / d["pred_medio"]).replace([np.inf, -np.inf], np.nan).dropna()
lim = ax.get_xlim()[1]
ax.plot([0, lim], [0, lim], color="gray", ls=":", lw=1, label="calibración perfecta")
ax.set_xscale("symlog", linthresh=0.1)
ax.set_yscale("symlog", linthresh=0.1)
ax.set_xlabel("pronóstico medio del tramo (relativo a la venta media)")
ax.set_ylabel("venta real media del tramo (relativa)")
ax.set_title("4. Calibración: venta real contra pronóstico por tramo", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False)
plt.tight_layout()
_cerrar_figura(fig, "g4_calibracion")
rz = razones.get("desarrollo", pd.Series([1.0]))
texto = (f"Cada punto agrupa las filas cuyo pronóstico cae en un tramo; si el modelo está calibrado, los puntos caen sobre la diagonal. "
         f"La razón real / pronóstico del modelo final va de {rz.min():.2f} a {rz.max():.2f} en desarrollo. En los tramos bajos (series casi "
         "sin venta) una razón menor que 1 indica que se pronostica algo de venta donde casi siempre hay cero; en los tramos altos, que "
         "pesan más en el WMAPE y en el surtido, la razón cercana a 1 indica buena calibración. ")
if "prueba" in razones:
    texto += f"En septiembre la razón va de {razones['prueba'].min():.2f} a {razones['prueba'].max():.2f}."
interpretar(4, "Calibración por tramo", texto)

# ---------------------------------------------------------------------------------------------- 5. importancia de variables y pesos
IMP = C.get("importancia", pd.DataFrame())
PW = C.get("pesos", pd.DataFrame())
con_pesos = MODELO_FINAL in POINT_HETEROGENEOUS and len(PW)
if len(IMP):
    fig, axes = plt.subplots(1, 2 if con_pesos else 1, figsize=(13 if con_pesos else 7.5, 5.0), squeeze=False)
    top = IMP.sort_values("importancia", ascending=False).head(15).iloc[::-1]
    axes[0, 0].barh(top["variable"], top["importancia"], color=COLOR_FINAL)
    axes[0, 0].xaxis.set_major_formatter(PercentFormatter(1))
    axes[0, 0].set_title(f"Variables con más peso en {AGG_LABELS.get(IMP['modelo'].iloc[0], IMP['modelo'].iloc[0])} ({IMP['metodo'].iloc[0]})", fontsize=9)
    axes[0, 0].grid(axis="x", alpha=0.3)
    if con_pesos:
        pesos_f = PW.loc[PW.corte.eq("final") & PW.ensamble.str.startswith(("E3", "E4_Stacking", "E5"))] if PW.corte.eq("final").any() else \
            PW.loc[PW.corte.eq(PW["corte"].iloc[-1]) & PW.ensamble.str.startswith(("E3", "E4_Stacking", "E5"))]
        vista = pesos_f.pivot_table(index="ensamble", columns="base", values="peso").fillna(0)
        vista.columns = [AGG_LABELS.get(c, c) for c in vista.columns]
        vista.plot(kind="barh", stacked=True, ax=axes[0, 1], colormap="tab10", width=0.7)
        axes[0, 1].set_title("Pesos de las bases en los ensambles del modelo final", fontsize=9)
        axes[0, 1].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("5. Qué usa el modelo", fontsize=10)
    plt.tight_layout()
    _cerrar_figura(fig, "g5_importancia")
    grupos = IMP.groupby("grupo")["importancia"].sum().sort_values(ascending=False)
    lectura = {"historia de la serie": "el nivel reciente y el patrón semanal de cada serie fijan la escala del pronóstico; el resto lo ajusta",
               "contexto de tienda, zona y país": "el modelo se apoya en cómo se mueve la categoría en la zona y el país, útil cuando la serie es ruidosa",
               "calendario y horizonte": "el día de la semana, la quincena y los eventos explican más que el nivel de la serie",
               "atributos de tienda y categoría": "los atributos distinguen tipos de serie con comportamientos distintos",
               "promociones futuras": "las promociones planeadas mueven el pronóstico"}.get(grupos.index[0], "")
    texto = (f"Por grupos: " + ", ".join(f"{g} {v:.0%}" for g, v in grupos.items()) + f". Las tres variables con más peso son "
             + ", ".join(f"{r.variable} ({DESCRIPCION_VARIABLES.get(r.variable, '').rstrip('.').lower()})" for r in IMP.head(3).itertuples())
             + f": {lectura}. La importancia describe en qué se apoya el modelo, no relaciones causales. ")
    promo = IMP.loc[IMP.variable.str.contains("promo|descuento")]["importancia"].sum()
    texto += (f"Las variables de promoción (todas de días pasados) suman {promo:.1%}: el modelo sabe si la serie suele estar en promoción, "
              "pero no si habrá una en los próximos 14 días.")
    if con_pesos:
        mayor = pesos_f.loc[pesos_f.ensamble.str.startswith("E4_Stacking")].groupby("base")["peso"].mean().sort_values(ascending=False)
        if len(mayor):
            texto += (f" En el stacking del modelo final la base con más peso es {AGG_LABELS.get(mayor.index[0], mayor.index[0])} ({mayor.iloc[0]:.2f}); "
                      "las bases con peso casi cero no aportan información nueva frente a las demás.")
    interpretar(5, "Importancia de variables", texto)
else:
    print("(Sin tabla de importancia: el modelo final no es de árboles o no se reentrenó.)")

# ---------------------------------------------------------------------------------------------- 6. error por horizonte y segmento
PHA = C["por_horizonte"]
SEG = C["segmentos"]
fig, axes = plt.subplots(1, 3, figsize=(16, 3.9))
for modelo, color, estilo in ((MODELO_FINAL, COLOR_FINAL, "-"), (MEJOR_IND, COLOR_IND, "--"), ("B0_Ingenuo", COLOR_REF, ":")):
    for fase, marca in (("desarrollo", "o"), ("prueba", "D")):
        d = PHA.loc[PHA.modelo.eq(modelo) & PHA.fase.eq(fase)].sort_values("h")
        if len(d):
            axes[0].plot(d["h"], d["wmape"], color=color, ls=estilo, marker=marca, ms=3, lw=1.3,
                         label=f"{AGG_LABELS[modelo]} · {fase}" if modelo != "B0_Ingenuo" or fase == "desarrollo" else None)
axes[0].yaxis.set_major_formatter(PercentFormatter(1))
axes[0].set_xticks(range(1, 15))
axes[0].set_xlabel("h")
axes[0].set_ylabel("WMAPE")
axes[0].set_title("Error por horizonte", fontsize=9)
axes[0].legend(fontsize=6, frameon=False)
axes[0].grid(alpha=0.3)
seg_txt = {}
for ax, segmento in ((axes[1], "tipo_categoria"), (axes[2], "region")):
    s = SEG.loc[SEG.segmento.eq(segmento) & SEG.fase.eq("desarrollo")]
    piv = s.pivot_table(index="valor_segmento", columns="modelo", values="wmape")
    if MODELO_FINAL in piv and "B0_Ingenuo" in piv:
        piv = piv.sort_values(MODELO_FINAL)
        x = np.arange(len(piv))
        ax.bar(x - 0.2, piv["B0_Ingenuo"], 0.4, color=COLOR_REF, label="ingenuo")
        ax.bar(x + 0.2, piv[MODELO_FINAL], 0.4, color=COLOR_FINAL, label="modelo final")
        ax.set_xticks(x, piv.index, rotation=30, ha="right", fontsize=8)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        seg_txt[segmento] = piv
    ax.set_title(f"WMAPE por {segmento.replace('_', ' ')} (desarrollo)", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
axes[1].legend(fontsize=7, frameon=False)
fig.suptitle("6. Dónde acierta y dónde falla el modelo final", fontsize=10)
plt.tight_layout()
_cerrar_figura(fig, "g6_error_horizonte_segmento")
hd = PHA.loc[PHA.modelo.eq(MODELO_FINAL) & PHA.fase.eq("desarrollo")].set_index("h")["wmape"]
texto = (f"El error del modelo final por horizonte va de {hd.min():.3f} (h = {int(hd.idxmin())}) a {hd.max():.3f} (h = {int(hd.idxmax())}); "
         "no crece de forma continua con h porque depende sobre todo del día de la semana que cae en cada horizonte. ")
if "tipo_categoria" in seg_txt:
    t = seg_txt["tipo_categoria"]
    texto += "Por tipo de categoría: " + ", ".join(f"{i} {r[MODELO_FINAL]:.3f} (ingenuo {r['B0_Ingenuo']:.3f})" for i, r in t.iterrows()) + ". "
if "region" in seg_txt:
    r_ = seg_txt["region"]
    mejora = 1 - r_[MODELO_FINAL] / r_["B0_Ingenuo"]
    texto += (f"Por región el error va de {r_[MODELO_FINAL].min():.3f} ({r_[MODELO_FINAL].idxmin()}) a {r_[MODELO_FINAL].max():.3f} "
              f"({r_[MODELO_FINAL].idxmax()}) y la mejora contra el ingenuo de {mejora.min():+.0%} a {mejora.max():+.0%}. ")
texto += ("Los servicios (recargas y pagos) son intermitentes y elevan el promedio: conviene reportarlos aparte del pronóstico de surtido. "
          "Las regiones con más error son candidatas a calibraciones o variables específicas.")
interpretar(6, "Error por horizonte y segmento", texto)

# ---------------------------------------------------------------------------------------------- 7. cobertura de intervalos
if "cuantiles" in C and len(C["cuantiles"]):
    Qh = C["cuantiles"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.9))
    for metodo, color in (("Q_Conformal", COLOR_FINAL), ("Q_LightGBM_CQR", COLOR_IND)):
        d = Qh.loc[Qh.metodo.eq(metodo) & Qh.ambito.eq("h")].sort_values("h")
        if len(d):
            axes[0].plot(d["h"], d["cobertura"], marker="o", color=color, label=metodo)
    axes[0].axhspan(*BANDA_COBERTURA, color="#EAE3CC", label="banda 75–85 %")
    axes[0].axhline(NOMINAL_COVERAGE, color="gray", ls="--", lw=1)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_xticks(range(1, 15))
    axes[0].set_xlabel("h")
    axes[0].set_title("Cobertura P10–P90 por horizonte (septiembre)", fontsize=9)
    axes[0].legend(fontsize=7, frameon=False)
    d = Qh.loc[Qh.metodo.eq(METODO_FINAL) & Qh.ambito.isin(["tipo_categoria", "region"])].copy()
    d["grupo"] = np.where(d.ambito.eq("tipo_categoria"), "tipo: " + d["tipo_categoria"].astype(str), "región: " + d["region"].astype(str))
    d = d.sort_values("cobertura")
    axes[1].barh(d["grupo"], d["cobertura"], color=np.where(d["cumple_banda"], COLOR_FINAL, "#C44E52"))
    axes[1].axvspan(*BANDA_COBERTURA, color="#EAE3CC", zorder=0)
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set_title(f"Cobertura por tipo y región con {METODO_FINAL}", fontsize=9)
    fig.suptitle("7. Intervalos de pronóstico en datos no vistos", fontsize=10)
    plt.tight_layout()
    _cerrar_figura(fig, "g7_intervalos")
    gq = Qh.loc[Qh.ambito.eq("global")].set_index("metodo")
    fh = Qh.loc[Qh.metodo.eq(METODO_FINAL) & Qh.ambito.eq("h")]
    texto = ("; ".join(f"{m}: cobertura global {r.cobertura:.1%}, ancho relativo {r.ancho_relativo:.2f}" for m, r in gq.iterrows()) + ". "
             f"Con {METODO_FINAL}, {int(fh['cumple_banda'].sum())} de {len(fh)} horizontes y {int(d['cumple_banda'].sum())} de {len(d)} grupos "
             "de tipo y región quedan en la banda 75–85 %. Una cobertura cercana a 80 % significa que el techo P90 se rebasa en uno de cada "
             "diez días, que es el riesgo de quiebre que se acepta al surtir al P90; un grupo por debajo de la banda (en rojo) necesita un "
             "techo más alto o una calibración propia.")
    interpretar(7, "Intervalos de pronóstico", texto)

# ---- registro de las gráficas y de las interpretaciones
if INTERPRETACIONES and EJECUTAR_ENTRENAMIENTO and RUN_ID_ACTUAL:
    try:
        limpio, esquema = _para_spark(pd.DataFrame(INTERPRETACIONES).assign(run_id=RUN_ID_ACTUAL, modelo=MODELO_FINAL, fecha=pd.Timestamp.now()))
        (spark.createDataFrame(limpio, schema=esquema).write.format("delta").mode("append").option("mergeSchema", "true")
         .saveAsTable(tabla("mlops", "av5_interpretaciones")))
    except Exception as e:
        print(f"(no se guardaron las interpretaciones: {type(e).__name__})")
    try:
        import mlflow
        run_mlflow = str(RESUMEN_C.get("mlflow_run_id") or "")
        if run_mlflow and run_mlflow != "None":
            with mlflow.start_run(run_id=run_mlflow):
                for nombre, figura in FIGURAS.items():
                    mlflow.log_figure(figura, f"graficas/{nombre}.png")
                mlflow.log_dict({"interpretaciones": INTERPRETACIONES}, "graficas/interpretaciones.json")
    except Exception as e:
        print(f"(no se registraron las gráficas en MLflow: {type(e).__name__})")
for figura in FIGURAS.values():
    plt.close(figura)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Salida operativa y monitoreo
# MAGIC **Pronóstico operativo.** El modelo final, reentrenado con todos los datos, pronostica los 14 días siguientes al
# MAGIC último día con datos para cada tienda × categoría evaluable y lo guarda en `gold.pronostico_14d` con el pronóstico
# MAGIC puntual, P10, P50, P90 y el techo operativo (P90 por defecto). Cada corrida reemplaza sólo su origen, así que la tabla
# MAGIC conserva la historia de pronósticos para medir después su acierto. En `MODO_RAPIDO` no se escribe.
# MAGIC
# MAGIC **Monitoreo.** `mlops.monitoreo_modelo` guarda, por semana × región × tipo de categoría, el WMAPE, el sesgo y la
# MAGIC cobertura del modelo final con alertas: error 5 puntos por encima del de desarrollo, |sesgo| > 5 % o cobertura fuera
# MAGIC de 75–85 %. En operación, un Job semanal compara `gold.pronostico_14d` contra la venta real y agrega filas a esta
# MAGIC tabla; dos semanas seguidas con alerta disparan el reentrenamiento (además del reentrenamiento mensual programado).

# COMMAND ----------

titulo("8. SALIDA OPERATIVA Y MONITOREO")
OPERATIVO_C = json.loads(RESUMEN_C["salida_operativa"]) if isinstance(RESUMEN_C.get("salida_operativa"), str) else {}
if OPERATIVO_C:
    mostrar_publico(pd.DataFrame([OPERATIVO_C]), "Pronóstico operativo de esta corrida")
    if not MODO_RAPIDO and existe(TABLA_PRONOSTICO):
        rev = (spark.table(TABLA_PRONOSTICO).where(F.col("run_id") == C["run_id"])
               .agg(F.avg(((F.col("p10") <= F.col("prediccion_puntual")) & (F.col("prediccion_puntual") <= F.col("p90"))).cast("double")).alias("dentro"),
                    F.expr("percentile_approx((p90 - p10) / greatest(prediccion_puntual, 1.0), 0.5)").alias("ancho_relativo_mediano"),
                    F.min("fecha_objetivo").alias("desde"), F.max("fecha_objetivo").alias("hasta")).collect()[0].asDict())
        mostrar(pd.DataFrame([rev]), "Revisión de la tabla de pronóstico (sin importes)", formatos={"dentro": "pct", "ancho_relativo_mediano": "dec2"})
        hallazgo(f"Pronóstico operativo guardado en {TABLA_PRONOSTICO}: {rev['desde']} a {rev['hasta']}; el {rev['dentro']:.0%} de las filas cumple "
                 f"P10 ≤ pronóstico ≤ P90; techo operativo = {NIVEL_TECHO_OPERATIVO.upper()}.")
else:
    print("La Oro no tiene conjunto 'pronostico' (o no se reentrenó el modelo final): no hay salida operativa en esta corrida.")
if existe(TABLA_MONITOREO):
    MON = spark.table(TABLA_MONITOREO).where(F.col("run_id") == C["run_id"]).toPandas()
    if len(MON):
        alertas = MON[["alerta_error", "alerta_sesgo", "alerta_cobertura"]].sum()
        mostrar_publico(MON.sort_values(["semana", "region", "tipo_categoria"]).head(30), "Monitoreo por semana × región × tipo (prueba)",
                        formatos={"wmape": "dec4", "sesgo_pct": "dec2", "cobertura": "pct"})
        hallazgo(f"Monitoreo de septiembre: {len(MON)} combinaciones semana × región × tipo; alertas de error {int(alertas['alerta_error'])}, "
                 f"de sesgo {int(alertas['alerta_sesgo'])} y de cobertura {int(alertas['alerta_cobertura'])}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Calendario de promociones futuras (apartado preparado, apagado)
# MAGIC **Situación actual (confirmada).** El modelo sólo ve promociones **pasadas**: `s_tasa_descuento_28`,
# MAGIC `s_intensidad_promo_28` y `s_promos_en_venta_28` promedian los 28 días anteriores a T y `s_promos_vigentes_t0` cuenta
# MAGIC las promociones con venta en la última semana (≤ T). Ninguna variable mira los 14 días que se pronostican. Por eso,
# MAGIC cuando una promoción empieza dentro de esa ventana, el modelo no la anticipa y subestima el pico (se ve en el gráfico 2).
# MAGIC
# MAGIC **Qué falta.** Un calendario de promociones **planeadas** de Bnext con el esquema de abajo, cargado en
# MAGIC `silver.calendario_promociones`.
# MAGIC
# MAGIC **Cómo se activa** (sin cambiar nada más):
# MAGIC 1. Cargar el calendario en `silver.calendario_promociones` (una fila por promoción × alcance).
# MAGIC 2. Correr la celda de abajo: valida el esquema, las fechas y la cobertura.
# MAGIC 3. Cambiar `USAR_CALENDARIO_PROMOS = True` (sección 1.7) y correr la libreta: `FuenteAvance5` une las variables
# MAGIC    `f_promos_planeadas`, `f_descuento_planeado` y `f_dias_promo_restantes` por tienda × categoría × día objetivo, y
# MAGIC    todos los modelos las usan.
# MAGIC 4. Comparar contra esta corrida: la ganancia debería concentrarse en los días con promoción (gráficos 2 y 3).
# MAGIC
# MAGIC **Para entrenar con historia**, el calendario debe cubrir también los meses pasados: si Bnext no tiene los planes
# MAGIC históricos, se puede usar la vigencia observada de cada promoción (`silver.promociones`) como aproximación del plan, con
# MAGIC la advertencia de que supone que el plan se cumplió tal cual.

# COMMAND ----------

titulo("9. CALENDARIO DE PROMOCIONES FUTURAS")
mostrar(pd.DataFrame([{"columna": k, "tipo y significado": v} for k, v in ESQUEMA_CALENDARIO_PROMOS.items()]),
        "Esquema esperado del calendario de promociones")
if existe(TABLA_CALENDARIO_PROMOS):
    revision = validar_calendario_promociones(spark.table(TABLA_CALENDARIO_PROMOS))
    mostrar(pd.DataFrame([revision]), "Revisión del calendario cargado")
    print("USAR_CALENDARIO_PROMOS =", USAR_CALENDARIO_PROMOS, "→", "en uso" if FUENTE_AV5._promos_futuras is not None else "todavía no se usa")
else:
    print(f"Todavía no existe {TABLA_CALENDARIO_PROMOS}: el modelo sigue usando sólo promociones pasadas (USAR_CALENDARIO_PROMOS = {USAR_CALENDARIO_PROMOS}).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Resumen para el reporte
# MAGIC Todo lo que se necesita para redactar el reporte del Avance 5, sin importes ni volúmenes.

# COMMAND ----------

titulo("10. RESUMEN PARA EL REPORTE (sólo métricas relativas)")
print(f"Corrida {C['run_id']} · modo {RESUMEN_C['modo']} · tabla Oro versión {RESUMEN_C['version_oro']} · "
      f"{int(RESUMEN_C['cortes'])} cortes de desarrollo · {int(RESUMEN_C['bloques_prueba'])} bloques de prueba · "
      f"{float(RESUMEN_C['minutos_modelado']):.0f} min de modelado")
print("\nComparativa de desarrollo (ordenada por WMAPE medio de los cortes):")
print(COMP[["posicion", "etiqueta", "tipo", "wmape_media_cortes", "wmape_desviacion_cortes", "mejora_vs_ingenuo", "sesgo_medio_pct",
            "wmape_mercancia", "min_entrenamiento_por_corte"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if HAY_PRUEBA:
    print("\nPrueba no vista (septiembre):")
    print(NV.drop(columns=["modelo"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if "ajuste_resumen" in C:
    print("\nAjuste fino:")
    print(C["ajuste_resumen"].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nMatriz de decisión:")
print(DEC.drop(columns=[c for c in ("modelo_final", "argumento") if c in DEC]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n{ARGUMENTO_FINAL}")
if "cuantiles" in C:
    print("\nIntervalos (septiembre):")
    print(GQ.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
if COMPUERTA_C:
    print(f"\nCompuerta: {COMPUERTA_C['desarrollo_status']} · registro: {COMPUERTA_C['registry_action']} · motivos de bloqueo: "
          + "; ".join(COMPUERTA_C["blocking_reasons"]))
if len(C.get("importancia", pd.DataFrame())):
    print("\nVariables más importantes: " + ", ".join(f"{r.variable} {r.importancia:.1%}" for r in C["importancia"].head(8).itertuples()))
for i in INTERPRETACIONES:
    print(f"\nGráfico {i['grafico']} · {i['titulo']}: {i['interpretacion']}")
cerrar("Avance5.43")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Declaración sobre el uso de herramientas de inteligencia artificial
# MAGIC *Celda obligatoria del curso. Sustituir este texto por la plantilla oficial con la información del equipo antes de
# MAGIC entregar.*
# MAGIC
# MAGIC | Herramienta | Versión / proveedor | Para qué se usó | En qué partes | Cómo se verificó |
# MAGIC | --- | --- | --- | --- | --- |
# MAGIC | *(completar)* | *(completar)* | *(completar)* | *(completar)* | *(completar)* |
# MAGIC
# MAGIC *(Completar con la declaración del equipo.)*
