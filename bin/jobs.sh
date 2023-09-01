# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=ablations \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=3 \
#     dataset=amazon_agri \
#     strategy=anchoral \
#     strategy.anchor_strategy=diversified,diversified_rampup \
#     strategy.only_minority=false  \
#     strategy.subpool_sampling_strategy=topk \
#     strategy.subpool_size=25;

# poetry run python ./scripts/active_train.py -m \
#     +experiment=basic \
#     experiment_group=ablations \
#     active_fit.max_rounds=36 \
#     data.seed=42,0 \
#     model.seed=42,0 \
#     active_data.seed=42,0 \
#     +launcher=joblib \
#     hydra.launcher.n_jobs=3 \
#     dataset=amazon_agri \
#     strategy=anchoral \
#     strategy.anchor_strategy=diversified,diversified_rampup \
#     strategy.only_minority=false  \
#     strategy.subpool_sampling_strategy=topk,importance \
#     strategy.subpool_size=10000;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=ablations \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=3 \
    dataset=amazon_agri \
    strategy=anchoral2 \
    strategy.anchor_strategy_minority=kmeans_sil,uncertainty,random \
    strategy.anchor_strategy_majority=random \
    strategy.subpool_size=25,10000;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=ablations \
    active_fit.max_rounds=36 \
    data.seed=42,0 \
    model.seed=42,0 \
    active_data.seed=42,0 \
    +launcher=joblib \
    hydra.launcher.n_jobs=3 \
    dataset=amazon_agri \
    strategy=anchoral2 \
    strategy.anchor_strategy_minority=random \
    strategy.anchor_strategy_majority=kmeans_sil,uncertainty,random \
    strategy.subpool_size=25,10000;


poetry run python ./scripts/active_train.py \
    data.seed=42 \
    model.seed=42 \
    active_data.seed=42 \
    dataset=amazon_agri \
    strategy=randomsubset_leastconf
