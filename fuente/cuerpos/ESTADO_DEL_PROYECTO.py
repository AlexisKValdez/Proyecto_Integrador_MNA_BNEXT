# Databricks notebook source
# MAGIC %md
# MAGIC # Estado del proyecto
# MAGIC **Proyecto Integrador MNA · Equipo 43 · Tiendas SIX / Bnext**
# MAGIC
# MAGIC Página de consulta para el equipo, la asesora y Bnext: **sólo lee**. Muestra qué etapas se han corrido, cuándo, qué
# MAGIC tablas existen con cuántas filas y los hallazgos más recientes de cada notebook. Se puede abrir y dar *Run all* en
# MAGIC cualquier momento.

# COMMAND ----------

# @@CONFIG

# COMMAND ----------

# @@APOYO

# COMMAND ----------

ETAPAS = [
    ("00_entorno_y_permisos", "Entorno y permisos", None),
    ("01_inventario_catalogo", "Inventario de Databricks", ("mlops", "inventario_tablas")),
    ("02_carga_bronce", "Carga de los meses (Bronce)", ("bronze", "ventas_lineas")),
    ("03_auditoria_trabajo_previo", "Auditoría del trabajo previo", ("mlops", "auditoria_previo")),
    ("04_eda_calidad", "EDA de calidad", ("mlops", "eda_calidad_reglas")),
    ("05_eda_tiempo_y_tiendas", "EDA de tiempo y tiendas", ("mlops", "eda_tiendas_actividad")),
    ("06_eda_variables_y_promociones", "EDA de variables y promociones", ("mlops", "eda_promociones_mensual")),
    ("07_eda_externas", "EDA de fuentes externas", ("mlops", "eda_externas_archivos")),
    ("08_limpieza_plata", "Limpieza (Plata)", ("silver", "ventas_lineas")),
    ("09_panel_diario", "Panel diario tienda × categoría", ("silver", "panel_categoria_dia")),
    ("10_eda_series_y_linea_base", "Series y línea base ingenua", ("mlops", "linea_base_ingenua")),
    ("11_validacion_calidad", "Validación de calidad", ("mlops", "validaciones")),
    ("12_evidencia_calidad", "Evidencia de calidad para Bnext (parquets)", ("mlops", "evidencia_calidad_inventario")),
    ("20_features_oro", "Fase 2 · Tabla Oro (variables por origen y horizonte)", ("gold", "features_origen")),
    ("21_avance4_modelos_alternativos", "Fase 2 · Avance 4: torneo, ajuste fino y modelo final", ("mlops", "torneo_modelos")),
    ("13_calendario_promociones_plan", "Calendario de promociones planeadas (archivo editable)", ("silver", "calendario_promociones_plan_dia")),
    ("Avance5.43", "Fase 2 · Avance 5: ensambles, prueba no vista y modelo final", ("mlops", "av5_ejecuciones")),
    ("22_pronostico_operativo", "Fase 3 · Pronóstico operativo con el modelo registrado", ("gold", "pronostico_14d")),
    ("23_monitoreo", "Fase 3 · Monitoreo: acierto, deriva y alertas", ("mlops", "alertas")),
    ("24_promocion_modelo", "Fase 3 · Promoción challenger → champion", ("mlops", "aprobaciones")),
    ("30_simulacion_escenarios_whatif", "Avance 6 · Escenarios what-if de promociones", ("gold", "escenarios_whatif_resumen")),
    ("31_desagregacion_predicciones_sku", "Avance 6 · Reparto a producto y techo de demanda", ("gold", "surtido_sku_semana")),
    ("40_salidas_dashboard", "Avance 6 · Tablas del tablero", ("gold", "dash_estado_pipeline")),
]
titulo("1. ETAPAS")
try:
    H = spark.table(tabla("mlops", "hallazgos")).toPandas()
except Exception:
    H = pd.DataFrame(columns=["notebook", "fecha_corrida", "orden", "hallazgo"])
ultima = H.groupby("notebook")["fecha_corrida"].max() if len(H) else pd.Series(dtype="datetime64[ns]")
filas = []
for nb, nombre, t in ETAPAS:
    filas_t = None
    if t and existe(tabla(*t)):
        try:
            filas_t = spark.table(tabla(*t)).count()
        except Exception:
            pass
    corrida = ultima.get(nb)
    filas.append({"etapa": nombre, "notebook": nb, "última corrida": corrida,
                  "estado": "hecho" if corrida is not None and not pd.isna(corrida) else "pendiente",
                  "tabla principal": tabla(*t) if t else "", "filas": filas_t})
ESTADO = pd.DataFrame(filas)
mostrar(ESTADO, "Avance por etapa", formatos={"filas": "int"})
hechas = (ESTADO["estado"] == "hecho").sum()
print(f"\n{hechas} de {len(ESTADO)} etapas completadas (13 de la fase de datos, 3 de la fase de modelos).")

# COMMAND ----------

titulo("2. CIFRAS CLAVE")
def cifra(capa, nombre, consulta):
    t = tabla(capa, nombre)
    if not existe(t):
        return None
    try:
        return consulta(spark.table(t))
    except Exception:
        return None

cifras = [
    ("Líneas en Bronce", cifra("bronze", "ventas_lineas", lambda d: f"{d.count():,}")),
    ("Líneas limpias en Plata", cifra("silver", "ventas_lineas", lambda d: f"{d.count():,}")),
    ("Venta limpia (MXN)", cifra("silver", "ventas_lineas", lambda d: f"${d.agg(F.sum('venta')).collect()[0][0]:,.0f}")),
    ("Periodo", cifra("silver", "ventas_lineas", lambda d: " → ".join(str(x) for x in d.agg(F.min('fecha_venta'), F.max('fecha_venta')).collect()[0]))),
    ("Tiendas", cifra("silver", "catalogo_tiendas", lambda d: f"{d.count():,}")),
    ("Categorías canónicas", cifra("silver", "catalogo_categorias", lambda d: f"{d.count():,}")),
    ("Filas del panel diario", cifra("silver", "panel_categoria_dia", lambda d: f"{d.count():,}")),
    ("Línea base: ingenuo estacional a 14 días (WMAPE, 4 cortes)",
     cifra("mlops", "linea_base_ingenua", lambda d: f"{d.toPandas()['wmape_lag7'].mean():.4f}")),
    ("Línea base: media móvil 7 d a 14 días (WMAPE, 4 cortes)",
     cifra("mlops", "linea_base_ingenua", lambda d: f"{d.toPandas()['wmape_ma7'].mean():.4f}")),
    ("Promociones en el calendario", cifra("silver", "promociones", lambda d: f"{d.count():,}")),
    ("Modelo heredado en su ventana, a un paso (WMAPE contra venta limpia)",
     cifra("mlops", "comparacion_modelo_heredado", lambda d: (lambda p: f"{p.loc[p['pronostico'].str.contains('venta limpia'), 'wmape'].iloc[0]:.4f}")(d.toPandas()))),
    ("Mejor ingenuo a un paso en esa misma ventana (WMAPE)",
     cifra("mlops", "comparacion_modelo_heredado", lambda d: (lambda p: f"{p[p['pronostico'].str.startswith('ingenuo')]['wmape'].min():.4f}")(d.toPandas()))),
    ("Reglas de calidad (última corrida)",
     cifra("mlops", "validaciones", lambda d: (lambda p: f"{(p['estado'] == 'PASA').sum()} de {len(p)} pasan")(
         d.toPandas().sort_values("fecha_validacion").groupby("regla").tail(1)))),
]
mostrar(pd.DataFrame([{"indicador": k, "valor": v if v is not None else "—"} for k, v in cifras]), "Cifras")

# COMMAND ----------

titulo("3. HALLAZGOS MÁS RECIENTES POR NOTEBOOK")
if len(H):
    H["fecha_corrida"] = pd.to_datetime(H["fecha_corrida"])
    recientes = H.merge(H.groupby("notebook")["fecha_corrida"].max().rename("ultima"), left_on="notebook", right_index=True)
    recientes = recientes[recientes["fecha_corrida"] == recientes["ultima"]].sort_values(["notebook", "orden"])
    for nb, g in recientes.groupby("notebook", sort=True):
        print(f"\n■ {nb} · {g['fecha_corrida'].iloc[0]:%Y-%m-%d %H:%M}")
        for h in g["hallazgo"]:
            print(f"   - {h}")
else:
    print("Todavía no hay hallazgos registrados.")
