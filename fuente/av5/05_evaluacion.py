
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
