
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
