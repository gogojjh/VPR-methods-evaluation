#!/bin/bash
# Usage:
#   bash scripts/run_main.sh
#
# Data:    <repo>/assets/database  &  <repo>/assets/queries
# Results: <repo>/logs/<method>_assets/<timestamp>/

set -e

PROJECT_PATH=$(cd "$(dirname "$0")/.." && pwd)
DATABASE_FOLDER=${PROJECT_PATH}/assets/database
QUERIES_FOLDER=${PROJECT_PATH}/assets/queries
DATASET=assets

cd "${PROJECT_PATH}"

ARGS="--recall_values 1 5 10 --num_preds_to_save=3 --batch_size=8"

run_model() {
    METHOD=$1; shift
    echo "--- Running: ${METHOD} ---"
    python3 main.py --method=${METHOD} \
        --database_folder=${DATABASE_FOLDER} \
        --queries_folder=${QUERIES_FOLDER} \
        --log_dir=${METHOD}_${DATASET} \
        --no_labels \
        ${ARGS} \
        "$@"
}

run_model cosplace    --backbone=ResNet18 --descriptors_dimension=256
run_model eigenplaces --backbone=ResNet18 --descriptors_dimension=256
run_model megaloc     --backbone=Dinov2   --descriptors_dimension=8448 --image_size 224 224

echo ""
echo "=== All runs complete ==="
ls logs/*${DATASET}*/info.log 2>/dev/null | sort
