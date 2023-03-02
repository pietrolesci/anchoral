# check variability due to data ordering and model initialization
poetry run python ./scripts/train.py \
    experiment_group=seed_variability \
    dataset_name=agnews_bert_tiny \
    train_val_split=0.05 \
    fit.num_epochs=3 \
    fit.validation_interval=5 \
    fit.learning_rate=0.0002 \
    model.seed=0,42,1994,6006,2023 \
    data.seed=0,42,1994,6006,2023 \
    -m


poetry run python scripts/active_train.py \
    dataset_name=agnews_bert_tiny \
    strategy=random \
    fit.validation_interval=5 \
    active_fit.num_rounds=2 \
    limit_batches=10



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


#############################
# ======== TESTING ======== #
#############################
poetry run python scripts/train.py \
    dataset_name=agnews_bert_tiny \
    limit_batches=10 \
    fit.validation_interval=1 \
    +callbacks=model_checkpoint \
    train_val_split=0.05


poetry run python scripts/active_train.py \
    dataset_name=imdb_bert_tiny \
    strategy=random \
    fit.validation_interval=3 \
    fit.min_steps=50 \
    active_fit.num_rounds=50 \
    active_fit.val_perc=0.1 \
    active_fit.query_size=25 \
    +callbacks=model_checkpoint \
    active_data.budget=100 \
    active_data.val_perc=0.5

