# ====================
# Run main experiments
# ====================

EXPERIMENT_GROUP=main

# all datasets
DATASET=agnews_business,amazon_agri,amazon_multi,wikitoxic

# all strategies
STRATEGY=anchoral_badge,anchoral_entropy,anchoral_ftbertkm,seals_badge,seals_entropy,seals_ftbertkm,randomsubset_badge,randomsubset_entropy,randomsubset_ftbertkm

# for bert-base-uncased
MODELS=bert-base-uncased

# two seeds per each source of randomness
SEEDS=654321,123456

poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=$MODELS


# This will create an `outputs/multirun/main` folder. 
# After it finishes, move it to `outputs/main` to make analysis easier with the provided notebooks