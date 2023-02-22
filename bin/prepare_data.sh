# ============
# Prepare data
# ============

set -e

echo 'Preparing AG-News'
poetry run python ./scripts/prepare_data.py \
    --dataset_name='agnews' \
    --input_dir='./data/processed' \
    --output_dir='./data/prepared' \
    --name_or_path='google/bert_uncased_L-2_H-128_A-2' \
    --name_or_path_alias='bert_tiny'


echo 'Preparing Civil Comments'
poetry run python ./scripts/prepare_data.py \
    --dataset_name=civil_comments \
    --input_dir='./data/processed/' \
    --output_dir='./data/prepared/' \
    --name_or_path='google/bert_uncased_L-2_H-128_A-2' \
    --name_or_path_alias='bert_tiny'


poetry run python ./scripts/prepare_data.py \
    --dataset_name='agnews' \
    --input_dir='./data/processed' \
    --output_dir='./data/prepared' \
    --name_or_path='bert-base-uncased' \
    --name_or_path_alias='bert_base'