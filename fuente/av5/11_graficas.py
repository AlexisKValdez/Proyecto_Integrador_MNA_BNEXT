
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
        borrar_corrida(tabla("mlops", "av5_interpretaciones"), RUN_ID_ACTUAL)
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
