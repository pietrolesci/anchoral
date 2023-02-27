# ============
# Prepare data
# ============

set -e

echo 'Preparing AGNEWS'
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


echo 'Preparing IMDB'
poetry run python ./scripts/prepare_data.py \
    --dataset_name='imdb' \
    --input_dir='./data/processed' \
    --output_dir='./data/prepared' \
    --name_or_path='google/bert_uncased_L-2_H-128_A-2' \
    --name_or_path_alias='bert_tiny'



poetry run python ./scripts/prepare_data.py \
    --dataset_name='agnews' \
    --input_dir='./data/processed' \
    --output_dir='./data/prepared' \
    --name_or_path='bert-base-uncased' \
    --name_or_path_alias='bert_base'


poetry run python ./scripts/prepare_data.py \
    --dataset_name='agnews' \
    --input_dir='./data/processed' \
    --output_dir='./data/prepared' \
    --name_or_path='distilbert-base-uncased' \
    --name_or_path_alias='distilbert_base'