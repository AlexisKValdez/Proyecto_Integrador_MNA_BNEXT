# Runbook de operación · Pronóstico de demanda Tiendas SIX — Equipo 43

Qué hacer, en qué orden y qué revisar en cada situación. Los nombres entre comillas son Jobs de `jobs/LEEME.md`; los
números son notebooks de `mna_equipo43_avance6`.

## 1. Llega un mes nuevo de Bnext

1. Confirmar que el volumen `/Volumes/modeladoprescriptivoventas/maestria/<AAAA-MM>/` tiene los parquet.
2. Agregar el mes a `MESES` en la celda de configuración de los notebooks **sólo si todavía no está** (hoy llega hasta
   `2026-10`; para noviembre: añadir `"2026-11"` en `fuente/partes/config.py` y reconstruir, o editar la celda en todos
   los notebooks). Los bloques de prueba (`ORIGENES_PRUEBA`) son fijos; un mes nuevo sólo extiende la historia y mueve
   el origen del pronóstico operativo.
3. Correr **J1 ingesta_mes** con `MES = AAAA-MM`. El Job falla si `11` encuentra una regla crítica rota: leer el
   RESUMEN PARA REVISIÓN del 11, corregir (o pedir a Bnext la entrega faltante) y repetir.
4. Correr **J2 features** (calendario + Oro). Revisar en el 20 la prueba anti-fuga (`PASA`) y en `mlops.oro_resumen`
   que el origen de pronóstico sea el último día con datos.
5. Correr **J4 inferencia**: pronóstico con el modelo vigente, escenarios, surtido y tablero.
6. Correr **J5 monitoreo**: con el mes nuevo ya se puede medir el acierto de los pronósticos anteriores.

## 2. Reentrenar

Cuándo: cada mes con datos nuevos, o cuando `23` deja la alerta `reentrenar` (dos semanas seguidas con error o sesgo
fuera de umbral) o `deriva` en alta.

1. **J3 entrenamiento** (`Avance5.43` completo, 4–6 h). Si se interrumpe: copiar el `run_id` del inicio de la
   sección 4 en el parámetro `RUN_ID_REANUDAR` y volver a lanzar; continúa desde el último punto de control.
2. Al terminar: la compuerta técnica decide si registra `@challenger`. Revisar §5–§7 de la libreta.
3. `24_promocion_modelo` con `ACCION = evaluar`: muestra la evidencia (compuerta, prueba no vista vs. campeón,
   aprobaciones). Si procede, `ACCION = promover`.
4. **J4 inferencia** con `ALIAS_MODELO = champion` (si no hay campeón usa `@challenger`).

## 3. Aprobaciones de negocio

Las tres aprobaciones viven en `mlops.aprobaciones` (alcance por categoría, política de servicios, atributos
disponibles al pronosticar). Se registran con SQL, con nombre y fecha:

```sql
UPDATE modeladoprescriptivoventas.maestria.mlops_aprobaciones
SET aprobado = true, aprobado_por = 'Nombre Apellido (Bnext)', fecha = current_timestamp(), comentario = 'correo del 5-nov'
WHERE aspecto = 'alcance_categoria';
```

## 4. Cambia el calendario de promociones

**Si Bnext sólo da fechas** (sin categoría ni descuento): una fila por promoción con `categoria_id = TODAS`, el
`tipo_categoria` al que aplica (`mercancia` si no se sabe), `descuento_planeado` vacío y `fuente = bnext`. El notebook 13
la expande a todas las categorías de ese tipo y le imputa el descuento histórico mediano de cada categoría
(`descuento_imputado = 1`). Cuando llegue el detalle, se sustituyen esas filas por una por categoría.

**Cuando Bnext entregue su catálogo completo** (tabla o archivo con categoría, descuento y mecánica): o se vacía al Excel
con las columnas del contrato, o se usa el bloque comentado al inicio de la sección 2 del notebook 13
(`ORIGEN_CATALOGO_BNEXT` + `MAPEO_COLUMNAS_BNEXT`): se descomenta, se mapean los nombres de columna de Bnext y el resto del
notebook sigue igual. Después: 13 → J4; y J3 cuando ya exista historia con el plan real.

1. Editar `outputs/calendario/calendario_promociones_plan.xlsx` (hoja `promociones`); poner `fuente = bnext` en las
   filas reales y borrar las `placeholder`. Los identificadores de categoría están en `catalogo_categorias.csv` de la
   misma carpeta.
2. Subir el archivo al volumen (Catalog → Volumes → `outputs` → `calendario` → Upload, reemplazar).
3. Correr `13` (o **J2**) y después **J4**. El pronóstico y el what-if toman el plan nuevo **sin reentrenar**.
4. Para que el modelo *aprenda* del plan real también hay que reentrenar (J3) cuando ya exista historia con plan de Bnext.

## 5. Salta una alerta (`mlops.alertas`, vista Alertas del tablero)

| Tipo | Qué significa | Qué hacer |
| --- | --- | --- |
| `error` / `sesgo` | el acierto en producción se degradó | revisar `mlops.monitoreo_modelo` por región/tipo; si se repite dos semanas, reentrenar |
| `cobertura` | el intervalo P10–P90 cubre menos (o más) del 75–85 % | reentrenar (los intervalos se recalibran con el modelo final) |
| `deriva` | una variable cambió de distribución (PSI > 0.2) | revisar la carga del mes (11), luego reentrenar |
| `frescura` | hace más de 7 días que no hay venta nueva | pedir el mes a Bnext; correr J1 |
| `pronostico_desactualizado` | hay venta más reciente que el origen del pronóstico | correr J2 y J4 |
| `reentrenar` | dos semanas seguidas con error/sesgo | J3 → 24 → J4 |

Marcar como atendida: `UPDATE ...mlops_alertas SET atendida = true WHERE tipo = 'frescura' AND fecha = '<fecha>'`.

## 6. Prueba de humo (antes de una corrida completa o tras cambiar código)

`Avance5.43` con `MODO_RAPIDO = True` (20–40 min) y después `22` con `PERMITIR_CHALLENGER = True`. Las cifras del
modo rápido no van a ningún reporte y no escriben pronóstico operativo.

## 7. Dónde ver el estado

- Notebook `ESTADO_DEL_PROYECTO` (Run all).
- Vista **Estado** del tablero (`gold.dash_estado_pipeline`): última venta, origen del pronóstico, compuerta, reglas,
  última corrida por notebook, alertas abiertas.
- `mlops.corridas` (tiempos), `mlops.hallazgos` (resúmenes), `mlops.av5_ejecuciones` (bitácora de entrenamientos),
  `mlops.pronosticos_bitacora` (pronósticos), `mlops.registro_modelos` (promociones de modelo).
