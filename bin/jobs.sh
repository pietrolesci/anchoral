poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=testing_no_uncertainty \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=4 \
    dataset=eurlex,agnews,pubmed,amazon,wiki_toxic \
    strategy=anchoral \
    strategy.anchor_strategy=kmeans,random \
    strategy.subpool_sampling_strategy=topk \
    strategy.subpool_size=25;
