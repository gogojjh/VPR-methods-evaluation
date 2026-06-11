#!/bin/bash
# Usage:
#   bash scripts/run_main.sh              # quick mode: online descriptors, R@1,5,10, save top-3 preds
#   bash scripts/run_main.sh benchmark    # benchmark mode: save descriptors, R@1,5,10, save top-3 preds
#
# Data:    <repo>/assets/database  &  <repo>/assets/queries
# Results: <repo>/logs/<method>_assets/<timestamp>/

set -e

BENCHMARK=false
if [ "$1" == "benchmark" ]; then
    BENCHMARK=true
fi

PROJECT_PATH=$(cd "$(dirname "$0")/.." && pwd)
DATABASE_FOLDER=${PROJECT_PATH}/assets/database
QUERIES_FOLDER=${PROJECT_PATH}/assets/queries
DATASET=assets

cd "${PROJECT_PATH}"

NUM_PREDS_TO_SAVE=3

if [ "$BENCHMARK" = true ]; then
    BATCH_SIZE=16
    EXTRA_ARGS="--recall_values 1 5 10 --save_descriptors --num_preds_to_save=${NUM_PREDS_TO_SAVE}"
    echo "=== Mode: BENCHMARK (save_descriptors=true, recall@1,5,10, preds=${NUM_PREDS_TO_SAVE}) ==="
else
    BATCH_SIZE=8
    EXTRA_ARGS="--recall_values 1 5 10 --num_preds_to_save=${NUM_PREDS_TO_SAVE}"
    echo "=== Mode: QUICK (online descriptors, recall@1,5,10, preds=${NUM_PREDS_TO_SAVE}) ==="
fi

run_model() {
    METHOD=$1; shift
    echo "--- Running: ${METHOD} ---"
    python3 main.py --method=${METHOD} \
        --database_folder=${DATABASE_FOLDER} \
        --queries_folder=${QUERIES_FOLDER} \
        --log_dir=${METHOD}_${DATASET} \
        --batch_size=${BATCH_SIZE} \
        --no_labels \
        ${EXTRA_ARGS} \
        "$@"
}

run_model cosplace    --backbone=ResNet18 --descriptors_dimension=256
run_model eigenplaces --backbone=ResNet18 --descriptors_dimension=256
run_model megaloc     --backbone=Dinov2   --descriptors_dimension=8448 --image_size 224 224

echo ""
echo "=== All runs complete (BENCHMARK=${BENCHMARK}) ==="
ls logs/*${DATASET}*/info.log 2>/dev/null | sort
