# ============
# Prepare data
# ============

set -e


data_dir=$(pwd)/data
MODEL=bert-tiny


# echo Preparing Agnews
# poetry run python ./scripts/prepare_data.py \
#     --data_dir $data_dir \
#     --model $MODEL \
#     --downsample_prop 0.01 \
#     --dataset agnews


# echo Preparing Eurlex-57k
# poetry run python ./scripts/prepare_data.py \
#     --data_dir $data_dir \
#     --model $MODEL \
#     --dataset eurlex-57k


# echo Preparing Pubmed-200k-rct
# poetry run python ./scripts/prepare_data.py \
#     --data_dir $data_dir \
#     --model $MODEL \
#     --dataset pubmed-200k-rct


echo Preparing AmazonCat-13k
poetry run python ./scripts/prepare_data.py \
    --data_dir $data_dir \
    --model $MODEL \
    --downsample_test_size 30000 \
    --dataset amazoncat-13k


