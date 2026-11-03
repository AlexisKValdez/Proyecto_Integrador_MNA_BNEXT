
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
        self._promos_futuras, self.info_calendario = (cargar_variables_promo_futura(spark) if USAR_CALENDARIO_PROMOS
                                                      else (None, {"activo": False, "motivo": "USAR_CALENDARIO_PROMOS = False"}))
        if USAR_CALENDARIO_PROMOS and self._promos_futuras is None:
            aviso(f"Calendario de promociones planeadas no disponible ({self.info_calendario.get('motivo')}): el modelo sigue sólo con promociones pasadas.")
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
            "calendario_promos_futuras": self.info_calendario,
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
        """Snapshot con las variables de promociones planeadas unidas (sólo si el calendario está activo). En cada fila
        cuentan únicamente las promociones publicadas en o antes de su origen T (unir_promociones_planeadas)."""
        if self._promos_futuras is None:
            return self._gold
        return unir_promociones_planeadas(self._gold, self._promos_futuras)

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
# MAGIC ### 2.2 Calendario de promociones planeadas (contrato compartido; se explica en la sección 9)
# MAGIC Las variables de promoción `s_*` del modelo describen **promociones pasadas** (días ≤ T). Las variables `f_*`
# MAGIC traen el **plan**: `f_promos_planeadas` (promociones que cubren el día objetivo), `f_descuento_planeado` (el mayor
# MAGIC descuento planeado) y `f_dias_promo_restantes` (días que le quedan a la promoción). Salen de la tabla
# MAGIC `silver.calendario_promociones_plan_dia` que escribe `13_calendario_promociones_plan` a partir del Excel editable
# MAGIC del volumen. Para que no haya fuga, en cada origen T sólo cuentan las promociones cuya `fecha_publicacion` es ≤ T:
# MAGIC es exactamente lo que el negocio sabía al pronosticar. Estas funciones son las mismas en 13, 20, 22 y 30.

# COMMAND ----------

# @@PROMOCIONES

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
