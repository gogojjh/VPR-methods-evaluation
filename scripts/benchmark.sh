#!/bin/bash

export DATASET=wildscenes # pitts30k, wildscenes
export DATABASE_FOLDER=../VPR-datasets-downloader/datasets/${DATASET}/images/test/database
export QUERIES_FOLDER=../VPR-datasets-downloader/datasets/${DATASET}/images/test/queries
export PROJECT_PATH=/Titan/code/robohike_ws/src/VPR-methods-evaluation
export SAVE_PREDS=True

# Running different methods
if [ "$SAVE_PREDS" = "True" ]; then
    python3 ${PROJECT_PATH}/main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=2048 \
        --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cosplace_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=eigenplaces --backbone=ResNet50 --descriptors_dimension=2048 \
        --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=eigenplaces_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
        --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=clique-mining --backbone=Dinov2 --descriptors_dimension=8448 \
        --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=clique-mining_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=anyloc-unstructured --backbone=Dinov2 --descriptors_dimension=49152 \
        --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=anyloc-unstructured_${DATASET}
else
    python3 ${PROJECT_PATH}/main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=2048 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cosplace_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=eigenplaces --backbone=ResNet50 --descriptors_dimension=2048 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=eigenplaces_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=clique-mining --backbone=Dinov2 --descriptors_dimension=8448 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=clique-mining_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=anyloc-unstructured --backbone=Dinov2 --descriptors_dimension=49152 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=anyloc-unstructured_${DATASET}
fi