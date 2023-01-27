# ============
# Process data
# ============


set -e


# AG-News
echo 'Processing AG-News'
poetry run python ./scripts/process_agnews.py \
    --output_dir='./data/processed/ag_news' \
    --seed=1994