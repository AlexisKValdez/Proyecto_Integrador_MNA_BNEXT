
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
