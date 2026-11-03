# Diccionario de datos — Equipo 43

Nombre real de cada tabla: `modeladoprescriptivoventas.maestria.<capa>_<nombre>` (por ejemplo `gold.pronostico_14d` → `gold_pronostico_14d`).
Los comentarios de columna también están escritos en las tablas (Catalog Explorer).

## Tablas

| Tabla | La escribe | Contenido |
| --- | --- | --- |
| `bronze.ventas_lineas` | 02 | Líneas de venta tal como llegan (partición anio_mes) con archivo de origen y fecha de carga. |
| `silver.ventas_lineas` | 08 | Líneas limpias, únicas y homologadas (partición anio_mes). |
| `silver.catalogo_categorias / catalogo_marcas / catalogo_tiendas` | 08 | Dimensiones homologadas; tiendas con zona, región (imputada por zona si falta), jefatura, estatus y ventana activa. |
| `silver.panel_categoria_dia` | 09 | Una fila por tienda × categoría × día: venta, unidades, tickets, descuento, promociones vigentes, banderas de tienda inactiva, día sin datos y atípico. |
| `silver.promociones / silver.calendario_promociones` | 09 | Promociones observadas: vigencia (inicio, fin, censura) y sus tiendas × categorías. |
| `silver.calendario_promociones_plan` | 13 | Plan de promociones: una fila por promoción × categoría × alcance, con fecha_publicacion y fuente (bnext / placeholder / historico_observado). |
| `silver.calendario_promociones_plan_dia` | 13 | Tienda × categoría × día con las promociones planeadas que lo cubren (arreglo de promo_id, fecha_publicacion, descuento, restantes, fuente). |
| `gold.features_origen` | 20 | Tabla Oro: serie × origen × horizonte con objetivo y variables (sólo días ≤ origen); conjuntos entrenamiento / evaluacion / pronostico. |
| `gold.pronostico_14d` | 22 / Avance5.43 | Pronóstico operativo por tienda × categoría × día con P10/P50/P90 y techo; reemplaza sólo su origen. |
| `gold.escenarios_whatif / gold.escenarios_whatif_resumen` | 30 | Escenarios de promoción (plan, sin promo, 10/20/30 %) por serie × día y agregados por región × tipo × categoría. |
| `gold.uplift_promocional_historico` | 30 | Venta con promoción / sin promoción por categoría y profundidad (misma tienda, mes y día de semana); referencia descriptiva. |
| `gold.predicciones_sku / gold.surtido_sku_semana` | 31 | Pronóstico repartido a producto por participación de 28 días (fila OTROS) y vista semanal con unidades P50 y techo P90. |
| `gold.dash_*` | 40 | Tablas del tablero: resumen ejecutivo, serie de tiempo por nivel, precisión, promociones, alertas, estado del pipeline. |
| `mlops.hallazgos / mlops.corridas` | todos | Resumen de cada corrida de cada notebook y sus tiempos. |
| `mlops.validaciones` | 11 | Resultado de cada regla de calidad (PASA / FALLA, crítica o aviso) por corrida. |
| `mlops.av5_*` | Avance5.43 | Resultados del Avance 5 por run_id: comparativas, por corte, por horizonte, diario, segmentos, calibración, residuos, cuantiles, DM, decisión, pesos, ajuste, costos, auditoría, parámetros, importancia, compuerta, resumen, interpretaciones. |
| `mlops.av5_ejecuciones` | Avance5.43 / 24 | Bitácora de eventos de cada corrida (INICIADA, paso_completado, FINALIZADA, ERROR, REANUDADA, PROMOVIDO). |
| `mlops.monitoreo_modelo / mlops.monitoreo_horizonte` | Avance5.43 / 23 | WMAPE, sesgo y cobertura por semana × región × tipo (fase prueba y produccion) y por horizonte. |
| `mlops.deriva_variables` | 23 | PSI de las variables principales entre el entrenamiento y el conjunto que se pronostica. |
| `mlops.alertas` | 23 | Alertas con tipo, severidad, valor, umbral y si fueron atendidas. |
| `mlops.aprobaciones` | 24 | Aprobaciones de negocio (alcance, servicios, atributos) con responsable y fecha. |
| `mlops.registro_modelos` | 24 | Promociones y cambios de alias del modelo en Unity Catalog. |
| `mlops.pronosticos_bitacora` | 22 | Cada corrida de pronóstico: origen, fase, modelo, versión, filas. |
| `mlops.calendario_plan_versiones` | 13 | Versiones cargadas del archivo del calendario. |

## Variables de la tabla Oro (`gold.features_origen`)

### Metadatos

| Columna | Significado |
| --- | --- |
| `tienda` | Identificador de la tienda (nunca se imprime). |
| `categoria_id` | Identificador de la categoría (también es variable categórica del modelo). |
| `origen` | Día T en que se hace el pronóstico; todas las variables usan días ≤ T. |
| `fecha_objetivo` | Día pronosticado: T + h. |
| `conjunto` | entrenamiento (muestra de series en muchos orígenes) · evaluacion (todas las series en los cortes) · pronostico (futuro). |
| `corte` | ajuste, corte_1…corte_4 (desarrollo), prueba_1 y prueba_2 (septiembre, no vistos) o pronostico. |
| `en_muestra_eval` | 1 si la serie pertenece a la muestra común (3 % de las series) en la que se comparan modelos rápido. |
| `y` | Venta real del día objetivo (pesos): lo que se pronostica. Nula en el conjunto pronostico. |
| `ingenuo_estacional` | Referencia B0: venta del mismo día de la semana más reciente ≤ T. |
| `dia_atipico_obj` | 1 si el día objetivo fue atípico (apagón, ley seca); sólo para analizar el error, no es variable. |

### Variables del modelo

| Variable | Significado |
| --- | --- |
| `s_t0` | Venta de la serie el día del origen T. |
| `s_ma7` | Venta media diaria de la serie en los 7 días que terminan en T. |
| `s_ma14` | Venta media diaria en los 14 días que terminan en T. |
| `s_ma28` | Venta media diaria en los 28 días que terminan en T (nivel reciente; también es la escala de la serie). |
| `s_ma56` | Venta media diaria en los 56 días que terminan en T (nivel de largo plazo). |
| `s_desv28` | Desviación estándar de la venta diaria en 28 días (volatilidad). |
| `s_max28` | Venta diaria máxima en 28 días (tamaño de los picos). |
| `s_dias_activos_7` | Días con la tienda activa (hay dato, aunque sea cero) en los últimos 7 días. |
| `s_dias_activos_28` | Días con la tienda activa en los últimos 28 días. |
| `s_dias_con_venta_28` | Días con venta mayor que cero en los últimos 28 días. |
| `s_tasa_venta_28` | Fracción de días activos con venta en 28 días (1 = vende diario; mide la intermitencia). |
| `s_dias_desde_ultima_venta` | Días desde la última venta positiva (56 si no vendió en la ventana). |
| `dias_con_venta_hist` | Días con venta en toda la historia de la serie hasta T (madurez). |
| `dias_historia` | Días desde la primera fecha de la serie hasta T. |
| `s_unidades_ma28` | Unidades medias diarias en 28 días. |
| `s_tickets_ma28` | Tickets medios diarios de la serie en 28 días. |
| `s_tasa_descuento_28` | Tasa media de descuento efectiva (1 − venta / venta sin descuento) en 28 días. |
| `s_intensidad_promo_28` | Fracción media de líneas vendidas con promoción en 28 días. |
| `s_promos_vigentes_t0` | Máximo de promociones distintas con venta en un mismo día de la última semana (≤ T). |
| `s_promos_en_venta_28` | Promociones distintas con venta por día, promedio de 28 días. |
| `s_lag_dow_1` | Venta del mismo día de la semana que el día objetivo, en la semana más reciente ≤ T. |
| `s_lag_dow_2` | Ídem, dos semanas antes. |
| `s_lag_dow_3` | Ídem, tres semanas antes. |
| `s_lag_dow_4` | Ídem, cuatro semanas antes. |
| `s_media_dow_4` | Promedio del mismo día de la semana en las últimas 4 semanas. |
| `s_media_dow_8` | Promedio del mismo día de la semana en las últimas 8 semanas. |
| `s_n_dow_8` | Cuántas de esas 8 semanas tienen dato (confiabilidad del promedio anterior). |
| `s_part_tienda_28` | Participación de la categoría en la venta de su tienda (28 días). |
| `t_ma7` | Venta media diaria de toda la tienda en 7 días. |
| `t_ma28` | Venta media diaria de toda la tienda en 28 días. |
| `t_tickets_ma7` | Tickets medios diarios de la tienda en 7 días (afluencia). |
| `t_tickets_ma28` | Tickets medios diarios de la tienda en 28 días. |
| `zc_ma7` | Venta media por tienda de la categoría en la zona, 7 días. |
| `zc_ma28` | Venta media por tienda de la categoría en la zona, 28 días. |
| `zc_lag_dow_1` | Venta media por tienda de la categoría en la zona, mismo día de la semana, semana más reciente. |
| `zc_lag_dow_2` | Ídem, dos semanas antes. |
| `zc_media_dow_4` | Ídem, promedio de 4 semanas. |
| `pc_ma7` | Venta media por tienda de la categoría en todo el país, 7 días. |
| `pc_ma28` | Venta media por tienda de la categoría en todo el país, 28 días. |
| `pc_lag_dow_1` | Venta media por tienda de la categoría en el país, mismo día de la semana, semana más reciente. |
| `pc_lag_dow_2` | Ídem, dos semanas antes. |
| `pc_media_dow_4` | Ídem, promedio de 4 semanas. |
| `dow` | Día de la semana del día objetivo (1 = lunes … 7 = domingo). |
| `fin_de_semana` | 1 si el día objetivo es sábado o domingo. |
| `dia_mes` | Día del mes del día objetivo. |
| `semana_mes` | Semana del mes del día objetivo (1–5). |
| `quincena` | 1 en días de pago (14–16, último y primero del mes). |
| `es_evento` | 1 si el día objetivo es un evento del calendario (festivo, comercial o futbol). |
| `tipo_evento` | Tipo de evento: 0 ninguno, 1 festivo, 2 comercial, 3 futbol. |
| `dias_a_evento` | Días que faltan para el siguiente evento (tope 30). |
| `dias_desde_evento` | Días desde el último evento (tope 30). |
| `en_mundial` | 1 si el día objetivo cae dentro de la Copa Mundial 2026. |
| `h` | Horizonte: días entre el origen T y el día objetivo (1–14). |
| `h_bloque` | 1 si h = 1–7; 2 si h = 8–14. |
| `antiguedad_tienda` | Días desde la primera venta de la tienda hasta T. |
| `tipo_tienda` | Formato de la tienda. |
| `zona` | Zona operativa de la tienda. |
| `region` | Región de la tienda (la de su zona cuando falta en el catálogo). |
| `jefatura` | Jefatura a la que reporta la tienda. |
| `estatus_tienda` | Estatus de la tienda en el catálogo. |
| `tipo_categoria` | Tipo de categoría: mercancía, servicio o genérico. |
| `categoria_id` | Categoría (identificador canónico homologado). |
| `f_promos_planeadas` | Promociones planeadas que cubren el día objetivo y ya estaban publicadas en el origen T (0 si ninguna). |
| `f_descuento_planeado` | Mayor descuento planeado (0–1) entre esas promociones; 0 si no hay promoción planeada. |
| `f_dias_promo_restantes` | Días que le quedan a la promoción vigente en el día objetivo (0 si no hay promoción). |

## Contrato del calendario de promociones (archivo editable)

| Columna | Tipo y significado |
| --- | --- |
| `promo_id` | texto · identificador de la promoción (obligatorio) |
| `nombre` | texto · nombre comercial (opcional) |
| `inicio_planeado` | fecha · primer día planeado (obligatorio) |
| `fin_planeado` | fecha · último día planeado (obligatorio) |
| `categoria_id` | texto · categoría canónica; vacío si se da categoria_nombre |
| `categoria_nombre` | texto · nombre de la categoría tal como está en silver.catalogo_categorias (se traduce a categoria_id) |
| `tienda` | texto · tienda; vacío = todas las tiendas del alcance indicado en zona o región |
| `zona` | texto · zona; vacío = no se acota por zona |
| `region` | texto · región; vacío = no se acota por región |
| `descuento_planeado` | decimal 0–1 · descuento planeado (0.15 = 15 %); vacío = desconocido (se usa 0) |
| `tipo_mecanica` | texto · 'descuento', 'lleva N paga M', 'precio fijo', 'regalo', … (opcional) |
| `fecha_publicacion` | fecha · desde cuándo se conoce el plan; vacío = inicio_planeado − DIAS_ANTICIPACION_DEFECTO |
| `fuente` | texto · 'bnext' (plan real), 'placeholder' (inventada, sustituir) o 'historico_observado' (derivada del 09) |
| `comentario` | texto · libre (opcional) |

## Pronóstico operativo (`gold.pronostico_14d`)

| Columna | Significado |
| --- | --- |
| `tienda` | Identificador de la tienda. |
| `origen` | Día T con cuyos datos se hizo el pronóstico. |
| `fecha_objetivo` | Día pronosticado (T + h). |
| `h` | Horizonte en días (1–14). |
| `categoria_id` | Categoría canónica. |
| `tipo_categoria` | mercancia, servicio o generico. |
| `region` | Región de la tienda. |
| `prediccion_puntual` | Pronóstico de venta del día (pesos). |
| `p10` | Límite inferior del intervalo (10 %). |
| `p50` | Mediana del intervalo. |
| `p90` | Límite superior del intervalo (90 %). |
| `techo_operativo` | Cantidad que cubre la demanda con el nivel elegido (P90 por defecto): techo de surtido. |
| `modelo` | Modelo que produjo el pronóstico. |
| `run_id` | Corrida de entrenamiento (mlops.av5_ejecuciones). |
| `version_oro` | Versión Delta de la tabla Oro usada. |
| `fecha_generacion` | Cuándo se calculó. |
| `version_modelo` | Versión del modelo en Unity Catalog (o run de MLflow). |
| `alias_modelo` | champion / challenger / run_mlflow. |
| `fase` | operativo (futuro) o retrospectivo (origen pasado, para medir el acierto). |
| `f_promos_planeadas` | Promociones planeadas que cubren el día (según el calendario vigente al pronosticar). |
