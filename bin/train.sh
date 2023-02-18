# ===========
# Experiments
# ===========

set -e

# Civil Comments
echo 'Training with unweighted loss'
poetry run python scripts/train.py \
    dataset_name=civil_comments

echo 'Training with weighted loss'
poetry run python scripts/train.py \
    dataset_name=civil_comments \
    fit.loss_fn=cross_entropy \
    fit.loss_fn_kwargs='{weight: true}'


# AG-News
echo 'Training with unweighted loss'
poetry run python scripts/train.py \
    dataset_name=ag_news


# ===== Active learning =====
poetry run python scripts/active_train.py \
    dataset_name=ag_news \
    strategy=uncertainty_based \
    strategy.score_fn=margin


