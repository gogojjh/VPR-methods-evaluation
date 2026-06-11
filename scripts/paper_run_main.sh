#!/bin/bash

# ./paper_run_main.sh wildscenes test

if [ -z "$1" ]; then
    export DATASET=wildscenes # pitts250k, msls, st_lucia, nordland, botanicgarden, wildscenes, opennavmap_ucl_campus, opennavmap_hkust | atlasvpr
else
    export DATASET=$1
fi

if [ -z "$2" ]; then
    export SPLIT=test # train, val, test
else
    export SPLIT=$2
fi

DATASET_DIR=/Rocket_ssd/dataset/data_vpr/${DATASET} # ../VPR-datasets-downloader/datasets/${DATASET}

export PROJECT_PATH=/Titan/code/robohike_ws/src/VPR-methods-evaluation
export DATABASE_FOLDER=${DATASET_DIR}/images/${SPLIT}/database
export QUERIES_FOLDER=${DATASET_DIR}/images/${SPLIT}/queries

# --- Positive Distance Threshold ---
POS_DIST_THRESHOLD=25

# --- Methods ---
# --- Single Image Matching (NetVLAD Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (cosplace Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=2048 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cosplace_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (eigenplaces Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=eigenplaces --backbone=ResNet50 --descriptors_dimension=2048 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=eigenplaces_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (anyloc Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=anyloc-urban --descriptors_dimension=49152 \
#     --batch_size=32 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=anyloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (salad Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=salad --descriptors_dimension=8448 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=salad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (cricavpr Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=cricavpr --backbone=Dinov2 --descriptors_dimension=10752 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cricavpr_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (SuperVLAD Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=supervlad --backbone=Dinov2 --descriptors_dimension=3072 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=supervlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}

# --- Single Image Matching (megaloc Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD}    

# --- Multi-View Matching ---
# --- SeqNet ---
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=3 --mm_method=seqnet \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/seqnet/Jun03_15-22-44_l10_w5/checkpoints/checkpoint.pth.tar

# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=5 --mm_method=seqnet \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/seqnet/Jun03_15-22-44_l10_w5/checkpoints/checkpoint.pth.tar

# --- SeqVLAD ---
###### sequence_length=3 #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=3 --mm_method=seqvlad \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth

##### sequence_length=5 #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=5 --mm_method=seqvlad \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth
    
####### sequence_length=3/5/7/9 with MegaLoc #######
# --- DeltaNet ---
# python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=3 --mm_method=deltanet

# python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --multiview_matching --sequence_length=5 --mm_method=deltanet

# --- VLAD Pooling ---
python3 ${PROJECT_PATH}/test/test_megaloc.py \
    --batch_size=64 --image_size 224 224 \
    --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT}_1 \
    --positive_dist_threshold ${POS_DIST_THRESHOLD} \
    --sequence_length=1 --mm_method=vlad_pooling --save_only_wrong_preds --num_preds_to_save 5

python3 ${PROJECT_PATH}/test/test_megaloc.py \
    --batch_size=64 --image_size 224 224 \
    --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT}_3 \
    --positive_dist_threshold ${POS_DIST_THRESHOLD} \
    --sequence_length=3 --mm_method=vlad_pooling --save_only_wrong_preds --num_preds_to_save 5

# python3 ${PROJECT_PATH}/test/test_megaloc.py \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --sequence_length=5 --mm_method=vlad_pooling

# python3 ${PROJECT_PATH}/test/test_megaloc.py \
#     --batch_size=20 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --sequence_length=7 --mm_method=vlad_pooling

# python3 ${PROJECT_PATH}/test/test_megaloc.py \
#     --batch_size=20 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET}_${SPLIT} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} \
#     --sequence_length=9 --mm_method=vlad_pooling

# --- Display Results ---
echo -e "\n======================= All Runs Complete ========================"
ls -l logs/megaloc_${DATASET}*recalls.txt
cat logs/megaloc_${DATASET}*recalls.txt
# ls -l logs/*${DATASET}*recalls.txt
# cat logs/*${DATASET}*recalls.txt
# ls -l logs/*${DATASET}*time.txt
# cat logs/*${DATASET}*time.txt
echo "=================================================================="
