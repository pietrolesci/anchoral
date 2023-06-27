poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=testing \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=4 \
    dataset=eurlex,wiki_toxic,agnews,pubmed \
    strategy=tyrogue,randomsubset,seals,anchoral;


poetry run python ./scripts/active_train.py \
    +experiment=basic \
    experiment_group=delete \
    active_fit.max_rounds=36 \
    dataset=eurlex \
    strategy=tyrogue