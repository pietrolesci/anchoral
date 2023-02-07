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
    --input_dir='./data/raw/jigsaw-unintended-bias-in-toxicity-classification' \
    --output_dir='./data/processed/civil_comments' \
    --seed=1994