#!/bin/bash
# Corre notebooks del pipeline en Spark local con datos sintéticos (sin datos de Bnext).
# uso: ./correr_local.sh <carpeta_de_trabajo> [--nuevo] <notebook> ...
#   --nuevo  regenera los datos sintéticos y borra el almacén local
#   PARAMETROS_NOTEBOOK='{"MODO_RAPIDO": "True"}'  emula los parámetros de un Job
#   SIMULAR_SERVERLESS=1  bloquea persist/cache y configuraciones no permitidas en serverless
set -e
AQUI="$(cd "$(dirname "$0")" && pwd)"
RAIZ="$(cd "$AQUI/.." && pwd)"
NOTEBOOKS="$RAIZ/databricks/mna_equipo43_avance6"
BASE="${1:?carpeta de trabajo}"; shift
PY="${PYTHON:-python3}"
if [ "$1" == "--nuevo" ]; then shift; rm -rf "$BASE"; mkdir -p "$BASE"; "$PY" "$AQUI/generar_datos2.py" "$BASE"; fi
NBS=""
for nb in "$@"; do
  if [ -f "$nb" ]; then NBS="$NBS $nb"; continue; fi
  for d in 1_datos 2_modelos 3_operacion .; do
    if [ -f "$NOTEBOOKS/$d/$nb.py" ]; then NBS="$NBS $NOTEBOOKS/$d/$nb.py"; break; fi
  done
done
cd "$AQUI" && "$PY" correr2.py "$BASE" $NBS
