# ─────────────────────── calendario de promociones planeadas (contrato compartido por 13, 20, Avance5.43, 22 y 30) ───────────────────────
TABLA_CALENDARIO_PLAN = tabla("silver", "calendario_promociones_plan")          # una fila por promoción × alcance (lo que se edita)
TABLA_CALENDARIO_PLAN_DIA = tabla("silver", "calendario_promociones_plan_dia")  # tienda × categoría × día con las promociones que lo cubren
VARIABLES_PROMO_FUTURA = ["f_promos_planeadas", "f_descuento_planeado", "f_dias_promo_restantes"]
ESQUEMA_CALENDARIO_PROMOS = {
    "promo_id": "texto · identificador de la promoción (obligatorio)",
    "nombre": "texto · nombre comercial (opcional)",
    "inicio_planeado": "fecha · primer día planeado (obligatorio)",
    "fin_planeado": "fecha · último día planeado (obligatorio)",
    "categoria_id": "texto · categoría canónica; vacío si se da categoria_nombre; TODAS = todas las categorías (o las del tipo_categoria indicado)",
    "categoria_nombre": "texto · nombre de la categoría tal como está en silver.catalogo_categorias (se traduce a categoria_id); TODAS = todas",
    "tipo_categoria": "texto · mercancia / generico / servicio; con TODAS acota a ese tipo (vacío = todos los tipos)",
    "tienda": "texto · tienda; vacío = todas las tiendas del alcance indicado en zona o región",
    "zona": "texto · zona; vacío = no se acota por zona",
    "region": "texto · región; vacío = no se acota por región",
    "descuento_planeado": "decimal 0–1 · descuento planeado (0.15 = 15 %); vacío = desconocido (se imputa el descuento histórico de la categoría)",
    "tipo_mecanica": "texto · 'descuento', 'lleva N paga M', 'precio fijo', 'regalo', … (opcional)",
    "fecha_publicacion": "fecha · desde cuándo se conoce el plan; vacío = inicio_planeado − DIAS_ANTICIPACION_DEFECTO",
    "fuente": "texto · 'bnext' (plan real), 'placeholder' (inventada, sustituir) o 'historico_observado' (derivada del 09)",
    "comentario": "texto · libre (opcional)",
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
    """Tabla tienda × categoría × día con las promociones planeadas que lo cubren (la escribe 13_calendario_promociones_plan).
    Regresa (DataFrame listo para unir, información) o (None, motivo) si no existe o no es válida."""
    if not existe(TABLA_CALENDARIO_PLAN_DIA):
        return None, {"activo": False, "motivo": f"no existe {TABLA_CALENDARIO_PLAN_DIA}: corre 13_calendario_promociones_plan"}
    dia = spark.table(TABLA_CALENDARIO_PLAN_DIA)
    faltan = {"tienda", "categoria_id", "fecha", "promos"} - set(dia.columns)
    if faltan:
        return None, {"activo": False, "motivo": f"{TABLA_CALENDARIO_PLAN_DIA} no tiene las columnas {sorted(faltan)}"}
    resumen = dia.agg(F.count("*").alias("filas"), F.min("fecha").alias("primer_dia"), F.max("fecha").alias("ultimo_dia"),
                      F.max("version_plan").alias("version_plan")).collect()[0].asDict()
    if not resumen["filas"]:
        return None, {"activo": False, "motivo": f"{TABLA_CALENDARIO_PLAN_DIA} está vacía"}
    listo = dia.select(F.col("tienda").cast("string").alias("_t"), F.col("categoria_id").cast("string").alias("_c"),
                       F.col("fecha").alias("fecha_objetivo"), F.col("promos"))
    return listo, {"activo": True, **{k: (str(v) if k != "filas" else int(v)) for k, v in resumen.items()}}


def unir_promociones_planeadas(df, dia):
    """Agrega a un DataFrame de la Oro (tienda, categoria_id, origen, fecha_objetivo) las variables f_*: sólo cuentan las
    promociones cuya fecha_publicacion es ≤ origen (lo que se sabía al pronosticar), por eso no hay fuga en los cortes."""
    if dia is None:
        for c in VARIABLES_PROMO_FUTURA:
            df = df.withColumn(c, F.lit(0.0).cast("float"))
        return df
    unido = (df.withColumn("_t", F.col("tienda").cast("string")).withColumn("_c", F.col("categoria_id").cast("string"))
             .join(dia, ["_t", "_c", "fecha_objetivo"], "left").drop("_t", "_c")
             .withColumn("_vis", F.expr("filter(promos, p -> p.fecha_publicacion <= origen)")))
    sin = F.col("promos").isNull()
    return (unido.withColumn("f_promos_planeadas", F.when(sin, F.lit(0)).otherwise(F.size("_vis")).cast("float"))
            .withColumn("f_descuento_planeado", F.coalesce(F.when(sin, F.lit(0.0)).otherwise(F.expr("array_max(transform(_vis, p -> p.descuento))")), F.lit(0.0)).cast("float"))
            .withColumn("f_dias_promo_restantes", F.coalesce(F.when(sin, F.lit(0)).otherwise(F.expr("array_min(transform(_vis, p -> p.restantes))")), F.lit(0)).cast("float"))
            .drop("promos", "_vis"))
