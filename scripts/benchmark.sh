#!/bin/bash

if [ -z "$1" ]; then
    export DATASET=msls # st_lucia, msls, pitts30k, wildscenes, botanicgarden
else
    export DATASET=$1
fi

export SAVE_PREDS=True
export DATABASE_FOLDER=../VPR-datasets-downloader/datasets/${DATASET}/images/test/database
export QUERIES_FOLDER=../VPR-datasets-downloader/datasets/${DATASET}/images/test/queries
export PROJECT_PATH=/Titan/code/robohike_ws/src/VPR-methods-evaluation

# Running different methods
if [ "$SAVE_PREDS" = "True" ]; then
    # python3 ${PROJECT_PATH}/main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=2048 \
    #     --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cosplace_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=eigenplaces --backbone=ResNet50 --descriptors_dimension=2048 \
    #     --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=eigenplaces_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
        --num_preds_to_save=2 --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=clique-mining --backbone=Dinov2 --descriptors_dimension=8448 \
    #     --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=clique-mining_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=anyloc-unstructured --backbone=Dinov2 --descriptors_dimension=49152 \
    #     --num_preds_to_save=5 --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=anyloc-unstructured_${DATASET}
else
    python3 ${PROJECT_PATH}/main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=2048 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cosplace_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=eigenplaces --backbone=ResNet50 --descriptors_dimension=2048 \
    #     --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=eigenplaces_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=salad --backbone=Dinov2 --descriptors_dimension=8448 \
    #     --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=salad_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=clique-mining --backbone=Dinov2 --descriptors_dimension=8448 \
    #     --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=clique-mining_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=anyloc-urban --backbone=Dinov2 --descriptors_dimension=49152 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=anyloc-urban_${DATASET}

    # python3 ${PROJECT_PATH}/main.py --method=selavpr-msls --backbone=Dinov2 --descriptors_dimension=1024 \
    #     --batch_size=8 --image_size 224 224 \
    #     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=selavpr-msls_${DATASET}

    python3 ${PROJECT_PATH}/main.py --method=supervlad --backbone=Dinov2 --descriptors_dimension=3072 \
        --batch_size=8 --image_size 224 224 \
        --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=supervlad_${DATASET}
fi

# ls logs/*${DATASET}*recalls.txt
cat logs/anyloc-urban*${DATASET}*recalls.txt
cat logs/cosplace*${DATASET}*recalls.txt
cat logs/megaloc*${DATASET}*recalls.txt
cat logs/supervlad*${DATASET}*recalls.txt