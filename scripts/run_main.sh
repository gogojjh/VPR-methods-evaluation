#!/bin/bash

if [ -z "$1" ]; then
    export DATASET=wildscenes # st_lucia, msls, pitts30k, wildscenes, botanicgarden, nordland, opennavmap / atlasvpr
else
    export DATASET=$1
fi

DATASET_DIR=/Rocket_ssd/dataset/data_vpr/${DATASET} # ../VPR-datasets-downloader/datasets/${DATASET}

export PROJECT_PATH=/Titan/code/robohike_ws/src/VPR-methods-evaluation
export DATABASE_FOLDER=${DATASET_DIR}/images/test/database
export QUERIES_FOLDER=${DATASET_DIR}/images/test/queries

# --- Positive Distance Threshold ---
# if [ "$DATASET" == "wildscenes" ] || [ "$DATASET" == "botanicgarden" ]; then
#     POS_DIST_THRESHOLD=10
# else
#     POS_DIST_THRESHOLD=25
# fi
POS_DIST_THRESHOLD=25

# --- Methods ---
# --- Single Image Matching (megaloc Baseline) ---
python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
    --batch_size=64 --image_size 224 224 \
    --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
    --positive_dist_threshold ${POS_DIST_THRESHOLD} --mm_method=none

# --- Single Image Matching (cricavpr Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=cricavpr --backbone=Dinov2 --descriptors_dimension=10752 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=cricavpr_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --mm_method=none

# # --- Single Image Matching (SuperVLAD Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=supervlad --backbone=Dinov2 --descriptors_dimension=3072 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=supervlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --mm_method=none
    
# # --- Single Image Matching (NetVLAD Baseline) ---
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --mm_method=none

# --- Multi-View Matching ---
MM_ARGS="--multiview_matching --sequence_length=5"

# # --- SeqNet ---
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} ${MM_ARGS} --mm_method=seqnet \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/seqnet/Jun03_15-22-44_l10_w5/checkpoints/checkpoint.pth.tar

# # --- SeqVLAD ---
# ###### sequence_length=1 #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --multiview_matching --sequence_length=1 --mm_method=seqvlad \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth

# ##### sequence_length=5 #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} ${MM_ARGS} --mm_method=seqvlad \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth

# ##### sequence_length=5 with Permutation #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} ${MM_ARGS} --mm_method=seqvlad --mm_permute \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth

# ###### sequence_length=5 PCA #######
# python3 ${PROJECT_PATH}/main.py --method=netvlad --backbone=VGG16 --descriptors_dimension=4096 \
#     --batch_size=64 --image_size 384 384 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=netvlad_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} ${MM_ARGS} --mm_method=seqvlad --pca_outdim=4096 \
#     --mm_resume=/Rocket_ssd/image_matching_model_weights/msls_cct384_tr8fz1__seqvlad_seq5.pth

####### sequence_length=3/5/7/9 with MegaLoc #######
# --- DeltaNet ---
# python3 ${PROJECT_PATH}/main.py --method=megaloc --backbone=Dinov2 --descriptors_dimension=8448 \
#     --batch_size=64 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} ${MM_ARGS} --mm_method=deltanet

python3 ${PROJECT_PATH}/test/test_megaloc.py \
    --batch_size=64 --image_size 224 224 \
    --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
    --positive_dist_threshold ${POS_DIST_THRESHOLD} --sequence_length=3 --mm_method=mean_pooling

python3 ${PROJECT_PATH}/test/test_megaloc.py \
    --batch_size=64 --image_size 224 224 \
    --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
    --positive_dist_threshold ${POS_DIST_THRESHOLD} --sequence_length=5 --mm_method=mean_pooling

# python3 ${PROJECT_PATH}/test/test_megaloc.py \
#     --batch_size=20 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --sequence_length=7 --mm_method=mean_pooling

# python3 ${PROJECT_PATH}/test/test_megaloc.py \
#     --batch_size=20 --image_size 224 224 \
#     --database_folder=${DATABASE_FOLDER} --queries_folder=${QUERIES_FOLDER} --log_dir=megaloc_${DATASET} \
#     --positive_dist_threshold ${POS_DIST_THRESHOLD} --sequence_length=9 --mm_method=mean_pooling

# --- Display Results ---
echo -e "\n======================= All Runs Complete ========================"
# ls -l logs/megaloc_${DATASET}*recalls.txt
# cat logs/megaloc_${DATASET}*recalls.txt
ls -l logs/*${DATASET}*recalls.txt
cat logs/*${DATASET}*recalls.txt
echo "=================================================================="
