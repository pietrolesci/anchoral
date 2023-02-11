# ===========
# Experiments
# ===========

set -e

echo 'Normal training'
poetry run python scripts/train.py \
    dataset_name=civil_comments \
    limit_batches=10