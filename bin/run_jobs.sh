DATASET=agnews_business
STRATEGY=anchoral_badge,anchoral_ftbertkm,seals_badge,seals_ftbertkm,randomsubset_badge,randomsubset_ftbertkm
TIMEOUT_MIN=360
SEEDS=654321,123456
EXPERIMENT_GROUP=missing_runs_agnews
poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=bert-base-uncased \
    +launcher=slurm \
    hydra.launcher.timeout_min=$TIMEOUT_MIN
