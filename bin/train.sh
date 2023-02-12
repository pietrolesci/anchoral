# ===========
# Experiments
# ===========

set -e

echo 'Normal training'
poetry run python scripts/train.py \
    dataset_name=civil_comments \
    limit_batches=10 \
    fit.loss_fn=cross_entropy \
    fit.loss_fn_kwargs='{weight: true}'