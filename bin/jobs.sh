# VARIABILITY TO THE RETRIEVER
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=retriever \
    index_metric=all-MiniLM-L12-v2_cosine \
    dataset=amazon,pubmed \
    strategy=anchorswithperclasssampling \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib \
    hydra.launcher.n_jobs=3;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=retriever \
    index_metric=all-MiniLM-L12-v2_cosine \
    dataset=eurlex,wiki_toxic,agnews \
    strategy=anchorswithperclasssampling \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;


# VARIABILITY TO ANCHOR STRATEGY
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=anchor_strategy \
    dataset=amazon,pubmed \
    strategy=anchorswithperclasssampling \
    strategy.anchor_strategy=random,kmeans \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=anchor_strategy \
    dataset=eurlex,wiki_toxic,agnews \
    strategy=anchorswithperclasssampling \
    strategy.anchor_strategy=random,kmeans \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib \
    hydra.launcher.n_jobs=3;



# VARIABILITY TO NUMBER OF MINORITY INSTANCES IN THE SEED SET
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=initial_set \
    dataset=amazon,pubmed \
    active_data.positive_budget=1 \
    strategy=random,randomsubset,seals \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib \
    hydra.launcher.n_jobs=3;

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=initial_set \
    dataset=eurlex,wiki_toxic,agnews \
    active_data.positive_budget=1 \
    strategy=random,randomsubset,seals \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;