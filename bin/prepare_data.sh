# ============
# Prepare data
# ============

set -e


data_dir=$(pwd)/data
MODEL=bert-tiny


echo Preparing AmazonCat-13k
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --downsample_test_size 25000 \
    --dataset amazoncat-13k-agri

echo Preparing Agnews
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --downsample_prop 0.02 \
    --dataset agnews


echo Preparing Eurlex-57k
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --dataset eurlex-57k


echo Preparing Pubmed-200k-rct
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --dataset pubmed-200k-rct


echo Preparing AmazonCat-13k
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --downsample_test_size 25000 \
    --dataset amazoncat-13k


echo Preparing WikiToxic
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --downsample_test_size 20000 \
    --dataset wiki_toxic


