-- Consultas de los conjuntos de datos del tablero (una por vista). Esquema: modeladoprescriptivoventas.maestria

-- ds_resumen · Resumen ejecutivo
SELECT nivel, region, tipo_categoria, zona, series, tiendas, venta_pronostico_14d, p10_14d, p90_14d,
       venta_real_14d_previos, venta_real_14d_anteriores, variacion_vs_14d_previos, variacion_14d_previos_vs_anteriores,
       ancho_relativo_banda, pct_filas_con_promo_planeada, origen, desde, hasta
FROM modeladoprescriptivoventas.maestria.gold_dash_resumen_ejecutivo;

-- ds_serie · Serie de tiempo
SELECT nivel, tienda, categoria_id, categoria_nombre, region, zona, tipo_categoria, fecha, tipo_dato,
       real, pronostico, p10, p90, COALESCE(real, pronostico) AS venta, tiendas, pct_series_con_promo
FROM modeladoprescriptivoventas.maestria.gold_dash_serie_tiempo;

-- ds_precision · Precisión del modelo
SELECT fase, semana, region, tipo_categoria, modelo, n_filas, wmape, sesgo_pct, cobertura,
       alerta_error, alerta_sesgo, alerta_cobertura
FROM modeladoprescriptivoventas.maestria.gold_dash_precision;

-- ds_modelos · Comparativa de modelos
SELECT fase, modelo, tipo, estrategia, wmape_media_cortes, mejora_vs_ingenuo, sesgo_abs_medio_pct, wmape_mercancia
FROM modeladoprescriptivoventas.maestria.gold_dash_precision_modelos;

-- ds_surtido · Surtido por producto
SELECT s.tienda, CAST(s.categoria_id AS STRING) AS categoria_id, c.nombre AS categoria_nombre, s.tipo_categoria, s.region,
       s.producto_id, s.rank_sku, s.semana_horizonte, s.desde, s.hasta, s.share_venta, s.precio_medio,
       s.venta_p50_semana, s.unidades_p50_semana, s.techo_unidades_p90_semana
FROM modeladoprescriptivoventas.maestria.gold_surtido_sku_semana s
LEFT JOIN modeladoprescriptivoventas.maestria.silver_catalogo_categorias c ON CAST(s.categoria_id AS STRING) = CAST(c.categoria_id AS STRING);

-- ds_promos · Promociones planeadas
SELECT promo_id, nombre, CAST(categoria_id AS STRING) AS categoria_id, categoria_nombre, inicio_planeado, fin_planeado, alcance,
       region, zona, descuento_planeado, tipo_mecanica, fuente, aporte_plan_pct, dias_para_inicio, en_horizonte
FROM modeladoprescriptivoventas.maestria.gold_dash_promociones;

-- ds_whatif · Escenarios what-if
SELECT id_escenario, descripcion_escenario, descuento_escenario, region, tipo_categoria, CAST(categoria_id AS STRING) AS categoria_id,
       series, venta_base, venta_escenario, uplift_pesos, uplift_pct, modo_simulacion, origen
FROM modeladoprescriptivoventas.maestria.gold_escenarios_whatif_resumen;

-- ds_alertas · Alertas
SELECT fuente, tipo, severidad, mensaje, tienda, categoria_nombre, region, zona, tipo_categoria, cambio_7d, incertidumbre,
       dias_inactiva_7d, origen, fecha
FROM modeladoprescriptivoventas.maestria.gold_dash_alertas;

-- ds_estado · Estado del pipeline
SELECT indicador, valor, detalle, estado, fecha_calculo
FROM modeladoprescriptivoventas.maestria.gold_dash_estado_pipeline;

