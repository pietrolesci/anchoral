# ============
# Prepare data
# ============

set -e

echo 'Preparing AG-News'
poetry run python ./scripts/prepare_data.py \
    --input_dir='./data/processed/ag_news' \
    --output_dir='./data/prepared/ag_news' \
    --name_or_path='google/bert_uncased_L-2_H-128_A-2'


echo 'Preparing Civil Comments'
poetry run python ./scripts/prepare_data.py \
    --input_dir='./data/processed/civil_comments' \
    --output_dir='./data/prepared/civil_comments' \
    --name_or_path='google/bert_uncased_L-2_H-128_A-2'
