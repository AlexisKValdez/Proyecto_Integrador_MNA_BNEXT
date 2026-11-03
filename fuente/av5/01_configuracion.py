
# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

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
MODO_RAPIDO = parametro("MODO_RAPIDO", True)
# True  → prueba de humo: ~5 % de las series en la evaluación, pocas pruebas de Optuna y menos árboles (20–40 min).
#         Sirve para comprobar que todo corre de principio a fin. Sus cifras NO van al reporte.
# False → corrida completa con todas las series evaluables (3.5–5 h con 16 GB). Conviene lanzarla como Job.

MODO_PUBLICO = parametro("MODO_PUBLICO", True)
# True  → lo que se imprime sólo contiene métricas relativas, tiempos, hiperparámetros y ventas indexadas (media = 100):
#         la libreta ejecutada puede subirse a GitHub sin exponer información de Bnext. Las tablas Delta internas sí
#         guardan las sumas completas, porque no salen de Databricks.

EJECUTAR_ENTRENAMIENTO = parametro("EJECUTAR_ENTRENAMIENTO", True)
# False → se salta el entrenamiento (sección 4) y las secciones 5–10 muestran una corrida ya guardada.

RUN_ID_REANUDAR = parametro("RUN_ID_REANUDAR", None)
# Identificador de una corrida interrumpida, p. ej. "av5_43_20261031t031500_1a2b3c4d" (se imprime al iniciar cada
# corrida y queda en mlops.av5_ejecuciones). Con él, la libreta carga su punto de control y continúa en el siguiente
# bloque sin repetir lo ya calculado.

RUN_ID_A_MOSTRAR = parametro("RUN_ID_A_MOSTRAR", None)
# Corrida que muestran las secciones 5–10. None = la de esta sesión o, si no hay, la última terminada.

SEMILLA = 43
# Semilla de todo lo aleatorio (muestreos, modelos, Optuna): misma semilla y mismos datos = mismo resultado.

REENTRENAR_FINAL = parametro("REENTRENAR_FINAL", True)
# True → al terminar, el modelo elegido se reentrena con todos los datos disponibles (hasta el último día con objetivo)
#        para producir el pronóstico operativo y registrarse en Unity Catalog.

# Todos los interruptores de esta sección se pueden fijar desde un Job (parámetros con el mismo nombre) o desde los
# widgets de la parte superior del notebook; si no se tocan, valen lo que dice el código.

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

# @@DICCIONARIO
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
INCLUIR_Q_LIGHTGBM = parametro("INCLUIR_Q_LIGHTGBM", True)
# Q_Conformal: intervalos alrededor del pronóstico del modelo elegido (residuos de un bloque reciente, por grupos).
# Q_LightGBM_CQR: tres LightGBM cuantílicos con calibración conformal (el método del avance anterior), como comparación.
# False ahorra ~40 min en modo completo: el modelo final usa entonces Q_Conformal.
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
REGISTRAR_EN_UC = parametro("REGISTRAR_EN_UC", None)
if isinstance(REGISTRAR_EN_UC, str):
    REGISTRAR_EN_UC = REGISTRAR_EN_UC.strip().lower() in ("true", "1", "si", "sí")
# None → se registra (alias @challenger) sólo en modo completo y si pasan las comprobaciones técnicas de la compuerta.
# Como parámetro de Job: True / False / vacío (= None).
NIVEL_TECHO_OPERATIVO = "p90"
# Columna que se publica como techo operativo de surtido (P90 por defecto; cambiar cuando Bnext elija P80 o P90).

# ==================================================================================================================
# 1.7 CALENDARIO DE PROMOCIONES FUTURAS (apartado preparado; ver sección 9)
# ==================================================================================================================
USAR_CALENDARIO_PROMOS = parametro("USAR_CALENDARIO_PROMOS", True)
# True → entran al modelo las variables f_* del calendario de promociones planeadas (tabla que escribe
#        13_calendario_promociones_plan a partir del Excel editable del volumen). Si la tabla no existe, se avisa y el
#        modelo sigue sólo con promociones pasadas (variables s_* con días ≤ T). Sin fuga: en cada origen T sólo cuentan
#        las promociones publicadas en o antes de T.
# False → se ignora el calendario (comparable con la corrida sin promociones futuras).
print("Configuración lista.")
