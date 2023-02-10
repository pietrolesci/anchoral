# ============
# Process data
# ============


set -e


# AG-News
echo 'Processing AG-News'

poetry run python ./scripts/process_agnews.py \
    --output_dir='./data/processed/ag_news' \
    --seed=1994


# Civil Comments
echo 'Processing Civil Comments'

poetry run python ./scripts/process_civil_comments.py \
    --input_dir='./data/raw/civil_comments' \
    --output_dir='./data/processed/civil_comments' \
    --seed=1994 \
    --min_chars=10 \
    --train_samples=100_000 \
    --test_samples=50_000

poetry run python ./scripts/embed_data.py \
    --data_path='./data/processed/civil_comments' \
    --model_name='all-mpnet-base-v2'