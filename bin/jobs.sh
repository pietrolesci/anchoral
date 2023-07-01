poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=testing_only_minority \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=4 \
    dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
    strategy=anchoral \
    strategy.only_minority=false;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=testing_only_minority \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=4 \
    dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
    strategy=anchoral \
    strategy.anchor_strategy=kmeans \
    strategy.only_minority=false,true;

# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=testing_subpool_sampling_strategy \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=4 \
#     dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
#     strategy=anchoral \
#     strategy.num_neighbours=500 \
#     strategy.subpool_size=10000 \
#     strategy.subpool_sampling_strategy=uniform,importance,topk;

# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=testing_subpool_sampling_strategy \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=4 \
#     dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
#     strategy=anchoral \
#     strategy.num_neighbours=2000 \
#     strategy.subpool_size=10000 \
#     strategy.subpool_sampling_strategy=uniform,topk;

# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=testing_subpool_sampling_strategy \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=4 \
#     dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
#     strategy=anchoral \
#     strategy.num_neighbours=2000 \
#     strategy.subpool_size=2500 \
#     strategy.subpool_sampling_strategy=uniform,importance,topk;

# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=testing_anchorstrategy \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=4 \
#     dataset=eurlex,wiki_toxic,agnews,pubmed,amazon \
#     strategy=anchoral \
#     strategy.num_neighbours=500 \
#     strategy.subpool_size=2500 \
#     strategy.only_minority=false,true \
#     strategy.anchor_strategy=all,random,kmeans;