# Jobs (Workflows) del pipeline — Equipo 43

Cinco Jobs serverless que encadenan los notebooks de `mna_equipo43_avance6`. Cada tarea es un notebook con sus
parámetros (los mismos nombres que las variables del código: `MESES_A_CARGAR`, `MODO_RAPIDO`, …); dentro del
notebook, `parametro()` toma el valor del Job y, si no hay, el del código.

| Job | Tareas (en orden) | Cuándo | Tiempo aprox. |
| --- | --- | --- | --- |
| **J1 ingesta_mes** | `02` (sólo el mes nuevo) → `08` (todos los meses) → `09` → `10` → `11` (**se detiene si una regla crítica falla**) | cuando Bnext entrega un mes | 1.5–4 h |
| **J2 features** | `13` (calendario de promociones del archivo) → `20` (tabla Oro) | después de J1, o cuando cambia el calendario | 40–90 min |
| **J3 entrenamiento** | `Avance5.43` completo (ajuste, ensambles, prueba, modelo final, `@challenger`, pronóstico) | mensual, o cuando el monitoreo lo pida | 4–6 h |
| **J4 inferencia** | `22` (pronóstico con `@champion`/`@challenger`) → `30` y `31` → `40` (tablero) | después de J2 o de cambiar el calendario | 30–80 min |
| **J5 monitoreo** | `23` → `40` | semanal (programado los lunes 7:00, en pausa hasta activarlo) | 10–20 min |

Demo de punta a punta («llega un mes → corre todo → predicciones y tablero»): J1 → J2 → J3 → J4 → J5.
Operación normal sin reentrenar: J1 → J2 → J4 → J5.

## Crear los Jobs

**Opción A — interfaz.** Workflows → Create job → por cada tarea: *Type* Notebook, *Path* el notebook, *Compute*
Serverless, *Parameters* los de la tabla `base_parameters` del JSON correspondiente, *Depends on* la tarea previa.
En *Environment* (serverless) agrega las dependencias de `environments[].spec.dependencies` del JSON (J3 y J4
necesitan `lightgbm`, `xgboost`, `catboost`, `optuna`, `shap`). En *Notifications* agrega un correo en *On failure*.
Recomendado para J3: *Timeout* 8 h, sin reintentos.

**Opción B — CLI.** Con la CLI de Databricks autenticada (`databricks auth login`):

```bash
./crear_jobs.sh "/Workspace/Users/<usuario>/mna_equipo43_avance6" "correo@dominio.com"
```

El script sustituye la ruta de los notebooks y el correo en los JSON y llama a `databricks jobs create`. Si el
workspace usa otra versión de entorno serverless, cambia `"client"` en `environments` (la que aparece en el panel
*Environment* del notebook).

## Parámetros útiles

| Job | Parámetro | Para qué |
| --- | --- | --- |
| J1 | `MES` (parámetro del Job, p. ej. `2026-10`) | el mes que se carga en Bronce; `08` siempre relimpia todos |
| J3 | `RUN_ID_REANUDAR` | reanudar una corrida interrumpida desde su punto de control |
| J3 | `MODO_RAPIDO = True` | prueba de humo (20–40 min) antes de la corrida completa |
| J4 | `ALIAS_MODELO = challenger` | pronosticar con el retador para compararlo |
| J4 | `ORIGEN_PRONOSTICO = 2026-08-31` (en `22`) | pronóstico retrospectivo para medir el acierto en `23` |

## Reglas de operación

- J1 falla a propósito si `11` encuentra una regla crítica rota: nada se entrena ni se pronostica con datos malos.
- `Avance5.43` registra el modelo como `@challenger`; `24_promocion_modelo` lo pasa a `@champion` con evidencia y las
  aprobaciones de `mlops.aprobaciones`. `22` usa `@champion` y, si no existe, `@challenger`.
- Cada corrida escribe en `mlops.corridas` (tiempos) y `mlops.hallazgos`; el tablero (`40`) los muestra.
