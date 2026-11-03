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

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Clúster y librerías
# MAGIC **Qué mirar.** Runtime **ML** (trae LightGBM, MLflow, scikit-learn). Memoria del nodo: para entrenar con todos los meses
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
    print(f"Nota: {memoria_gb:.0f} GB alcanzan para la fase de datos; para entrenar con todos los meses se usará una muestra de "
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
