# ─────────────────────── columnas del proyecto (iguales en todos los notebooks) ───────────────────────
# Nombre lógico → nombres posibles en los parquet de Bnext. Se usa el primero que exista en la tabla.
# Si Bnext cambia un nombre, basta con agregarlo aquí (en todos los notebooks).
CANDIDATAS = {
    "id_linea": ["order_line_id"], "id_ticket": ["order_id"],
    "tienda": ["store_code", "store_id"], "tipo_tienda": ["store_type"], "zona": ["zone_id", "zone"], "region": ["region"],
    "categoria": ["category_id"], "categoria_nombre": ["category"],
    "marca": ["brand_id"], "marca_nombre": ["brand"],
    "producto": ["product_code", "product_id"], "producto_nombre": ["product_name"],
    "cantidad": ["order_line_qty_ordered"], "precio_unitario": ["order_line_unit_price"],
    "costo_unitario": ["order_line_unit_cost"],
    "total_con_descuento": ["order_line_total_with_discount"],
    "total_sin_descuento": ["order_line_total_without_discount"],
    "pct_descuento": ["order_line_discount_percent"], "descuento_sumasix": ["order_line_sumasix_discount"],
    "pago_puntos": ["pago_con_puntos"], "promo_id": ["order_line_promo_id"], "promo_nombre": ["order_line_promo_name"],
    "fecha_ticket": ["finish_order"], "fecha_orden": ["date_orders", "date_ordered"], "fecha_sync": ["sync_date"],
    "estatus_linea": ["order_line_status_id"], "estatus_orden": ["order_status_id"],
    "impuestos": ["order_line_total_taxes"], "subtotal": ["order_line_subtotal"],
    "estatus_tienda": ["store_status_id"], "jefatura": ["jefatura_six"],
}
TRAZA = ["_archivo", "_fecha_archivo", "_fecha_carga", "anio_mes"]     # columnas que agrega 02_carga_bronce


def _sanear(nombre):
    import unicodedata
    t = unicodedata.normalize("NFKD", str(nombre)).encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"_+", "_", re.sub(r"[^0-9a-zA-Z]+", "_", t.strip().lower()).strip("_")) or "columna"
    return f"c_{t}" if t[0].isdigit() else t


def resolver_columnas(columnas):
    """{nombre_lógico: columna real o None}"""
    disponibles = {_sanear(c): c for c in columnas}
    return {k: next((disponibles[_sanear(o)] for o in v if _sanear(o) in disponibles), None)
            for k, v in CANDIDATAS.items()}


def requiere(C, *logicos):
    """True si existen todas las columnas lógicas; si falta alguna lo avisa y devuelve False."""
    faltan = [l for l in logicos if not C.get(l)]
    if faltan:
        aviso(f"faltan columnas {faltan}; se omite esta parte (agrega su nombre real en CANDIDATAS)")
        return False
    return True


def num(columna):
    """Columna como número; un valor mal capturado se vuelve nulo en lugar de romper la consulta."""
    return F.expr(f"try_cast(`{columna}` as double)")


FORMATOS_FECHA = ["yyyy-MM-dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss.SSS", "yyyy-MM-dd\\'T\\'HH:mm:ss",
                  "dd/MM/yyyy HH:mm:ss", "dd/MM/yyyy HH:mm", "dd/MM/yyyy", "yyyy-MM-dd", "yyyyMMdd"]


FECHA_MINIMA, FECHA_MAXIMA = "2000-01-01", "2100-01-01"   # fuera de este rango la fecha es un valor centinela


def a_timestamp(columna, tipo_spark, acotar=True):
    """Convierte una columna de fecha (texto, número o timestamp) en timestamp sin fallar.
    Con acotar=True, las fechas fuera de 2000–2099 (centinelas como 0001-01-01 o años de cinco cifras, que además no caben
    en una fecha de Python) quedan nulas."""
    t = tipo_spark.lower()
    c = f"`{columna}`"
    if t.startswith("timestamp"):
        ts = F.col(columna)
    elif t == "date":
        ts = F.col(columna).cast("timestamp")
    elif t in ("bigint", "int", "integer", "smallint", "tinyint", "double", "float") or t.startswith("decimal"):
        ts = F.expr(f"CASE WHEN abs({c}) < 1e13 THEN timestamp_seconds(CASE WHEN {c} > 1e11 THEN {c} / 1000 ELSE {c} END) END")
    else:
        intentos = [f"try_to_timestamp(cast({c} as string))"] + \
                   [f"try_to_timestamp(cast({c} as string), '{fmt}')" for fmt in FORMATOS_FECHA]
        ts = F.expr(f"coalesce({', '.join(intentos)})")
    if not acotar:
        return ts
    return F.when((ts >= F.lit(FECHA_MINIMA).cast("timestamp")) & (ts < F.lit(FECHA_MAXIMA).cast("timestamp")), ts)


RANGO_POS = {"costo_max": 5_000, "cantidad_max": 1_000, "venta_min": 0, "venta_max": 15_000}   # reglas de rango POS de Bnext


def fuera_de_rango_pos(C):
    """True si la línea viola alguna regla de rango POS de Bnext (costo, cantidad o importe imposibles): son errores de
    captura, como una línea de mil millones de pesos, que deforman cualquier suma de venta."""
    fuera = F.lit(False)
    if C.get("costo_unitario"):
        fuera = fuera | F.coalesce(num(C["costo_unitario"]) > RANGO_POS["costo_max"], F.lit(False))
    if C.get("cantidad"):
        fuera = fuera | F.coalesce(num(C["cantidad"]) > RANGO_POS["cantidad_max"], F.lit(False))
    if C.get("total_con_descuento"):
        v = num(C["total_con_descuento"])
        fuera = fuera | F.coalesce((v < RANGO_POS["venta_min"]) | (v > RANGO_POS["venta_max"]), F.lit(False))
    return fuera


def fecha_venta(df):
    """Fecha (DATE) de la venta: COLUMNA_FECHA_VENTA convertida sin fallar y, si HORAS_EN_UTC, pasada a hora de México."""
    tipos = {f.name: f.dataType.simpleString() for f in df.schema.fields}
    if COLUMNA_FECHA_VENTA not in tipos:
        raise KeyError(f"No existe la columna de fecha {COLUMNA_FECHA_VENTA!r}; revisa COLUMNA_FECHA_VENTA en la configuración")
    ts = a_timestamp(COLUMNA_FECHA_VENTA, tipos[COLUMNA_FECHA_VENTA])
    if HORAS_EN_UTC:
        ts = F.from_utc_timestamp(ts, "America/Mexico_City")
    return F.to_date(ts)


def expr_promo(C, tipos):
    """(identificador de promoción como texto, ¿la línea tiene promoción?). Bnext manda el identificador como número
    (double con nulos); un nulo, NaN o 0 significa sin promoción."""
    if not C.get("promo_id"):
        return F.lit(None).cast("string"), F.lit(False)
    c = C["promo_id"]
    t = tipos[c]
    if t in ("double", "float") or t.startswith("decimal") or t in ("bigint", "int", "smallint", "tinyint"):
        valor = F.expr(f"try_cast(`{c}` as double)")
        tiene = valor.isNotNull() & ~F.isnan(valor) & (valor != 0)
        return F.when(tiene, F.expr(f"try_cast(try_cast(`{c}` as decimal(38,0)) as string)")), tiene
    texto = F.trim(F.col(c).cast("string"))
    tiene = texto.isNotNull() & (texto != "") & ~texto.isin("0", "0.0", "null", "None", "NaN", "nan")
    return F.when(tiene, texto), tiene
