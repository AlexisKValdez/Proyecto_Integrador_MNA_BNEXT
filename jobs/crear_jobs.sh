#!/bin/bash
# Crea los cinco Jobs en Databricks con la CLI (databricks CLI v0.2xx, autenticada contra el workspace).
# uso: ./crear_jobs.sh "/Workspace/Users/<usuario>/mna_equipo43_avance6" "correo@dominio.com"
set -e
RUTA="${1:?ruta de la carpeta de notebooks en el workspace}"
CORREO="${2:?correo para avisos de fallo}"
for j in J1_ingesta_mes J2_features J3_entrenamiento J4_inferencia J5_monitoreo; do
  sed -e "s|__RUTA_NOTEBOOKS__|$RUTA|g" -e "s|__CORREO__|$CORREO|g" "$j.json" > "/tmp/$j.json"
  echo "→ $j"; databricks jobs create --json "@/tmp/$j.json"
done
