# ============
# Prepare data
# ============

set -e


data_dir=$(pwd)/data/processed
MODEL=$1


echo Preparing Agnews
poetry run python ./scripts/prepare_data.py \
    --dataset_path $data_dir/agnews \
    --model $MODEL \
    --output_name agnews-business-.01 \
    --downsample_prop 0.01 \
    --class_to_binarize business


echo Preparing WikiToxic
poetry run python ./scripts/prepare_data.py \
    --dataset_path $data_dir/wikitoxic \
    --model $MODEL \
    --output_name wikitoxic-.01 \
    --downsample_prop 0.01 \
    --downsample_test_size 5000


echo Preparing AmazonCat-13k
poetry run python ./scripts/prepare_data.py \
    --dataset_path $data_dir/amazoncat-13k \
    --model $MODEL \
    --output_name amazoncat-agri \
    --downsample_test_size 5000 \
    --class_to_binarize agriculture


echo Preparing AmazonCat-13k
poetry run python ./scripts/prepare_data.py \
    --dataset_path $data_dir/amazoncat-13k \
    --model $MODEL \
    --output_name amazoncat-multi \
    --downsample_test_size 5000 \
    --class_to_binarize all