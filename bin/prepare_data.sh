# ============
# Prepare data
# ============

set -e


data_dir=$(pwd)/data
MODEL=bert-tiny


# echo Preparing Agnews
# poetry run python ./scripts/prepare_data.py \
#     --data_dir=$data_dir \
#     --model $MODEL \
#     --dataset agnews


# echo Preparing Eurlex-57k
# poetry run python ./scripts/prepare_data.py \
#     --data_dir=$data_dir \
#     --model $MODEL \
#     --dataset eurlex-57k


echo Preparing Pubmed-200k-rct
poetry run python ./scripts/prepare_data.py \
    --data_dir=$data_dir \
    --model $MODEL \
    --dataset pubmed-200k-rct