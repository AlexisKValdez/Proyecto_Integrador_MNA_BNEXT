# Pruebas locales con datos sintéticos

Permiten correr cualquier notebook del pipeline **sin datos de Bnext**: `generar_datos2.py` inventa siete meses de líneas
de venta (tiendas que abren y cierran, un apagón, días atípicos, campañas promocionales con efecto en la demanda,
errores de escala en el descuento, duplicados entre meses) y `correr2.py` ejecuta los notebooks en Spark local con Delta,
emulando `spark`, `dbutils` (incluidos los widgets de parámetros), `display` y los volúmenes.

```bash
python3 -m venv venv && source venv/bin/activate && pip install -r pruebas/requirements_local.txt
export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64 SIMULAR_SERVERLESS=1 MLFLOW_TRACKING_URI=file:///tmp/mlruns
# cadena de datos
pruebas/correr_local.sh /tmp/mna_local --nuevo 00_entorno_y_permisos 02_carga_bronce 08_limpieza_plata 09_panel_diario \
    10_eda_series_y_linea_base 11_validacion_calidad 13_calendario_promociones_plan 20_features_oro
# modelo (modo rápido) guardando el modelo en MLflow
PARAMETROS_NOTEBOOK='{"REGISTRAR_EN_UC": "True"}' pruebas/correr_local.sh /tmp/mna_local Avance5.43
# operación
pruebas/correr_local.sh /tmp/mna_local 22_pronostico_operativo 23_monitoreo 24_promocion_modelo \
    30_simulacion_escenarios_whatif 31_desagregacion_predicciones_sku 40_salidas_dashboard
```

Lo que no se puede probar en local: Unity Catalog (registro de modelos, alias), tiempos reales y la importación del
tablero. En local los notebooks avisan y continúan (p. ej. `22` toma el modelo del run de MLflow).

`test_funciones.py` (pytest) prueba las funciones puras de la configuración y del diccionario de variables.
