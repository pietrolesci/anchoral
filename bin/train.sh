


########
# OURS #
########

# DEFAULT RUN
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=seed \
    dataset=eurlex,wiki_toxic,agnews,amazon,pubmed \
    active_data.positive_budget=5 \
    strategy=anchorswithperclasssampling \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;

# VARIABILITY TO NUMBER OF MINORITY INSTANCES IN THE SEED SET
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=initial_set \
    dataset=eurlex,wiki_toxic,agnews,amazon,pubmed \
    active_data.positive_budget=1 \
    strategy=anchorswithperclasssampling \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;

# VARIABILITY TO THE RETRIEVER
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=retriever \
    index_metric=all-MiniLM-L12-v2_cosine \
    dataset=eurlex,wiki_toxic,agnews,amazon,pubmed \
    strategy=anchorswithperclasssampling \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;

# VARIABILITY TO ANCHOR STRATEGY
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=anchor_strategy \
    dataset=eurlex,wiki_toxic,agnews,amazon,pubmed \
    strategy=anchorswithperclasssampling \
    strategy.anchor_strategy=random,kmeans \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib;




# k-means
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    enable_progress_bar=false \
    experiment_group=seeds \
    dataset=pubmed \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=anchorswithperclasssampling \
    strategy.num_anchors=50 \
    strategy.anchor_strategy=kmeans \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib

# all positive
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    enable_progress_bar=false \
    experiment_group=seeds \
    dataset=pubmed,amazon,wiki_toxic,eurlex,agnews \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=anchorswithperclasssampling \
    strategy.num_anchors=50 \
    strategy.anchor_strategy=all_positive \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib


# random or random_subset
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    enable_progress_bar=false \
    experiment_group=seeds \
    dataset=pubmed,amazon,wiki_toxic,eurlex,agnews \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=randomsubset \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib

# seals
poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    enable_progress_bar=false \
    experiment_group=seeds \
    dataset=pubmed,amazon,wiki_toxic,eurlex,agnews \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=seals \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib


CMD='poetry run python ./scripts/active_train.py -m +experiment=basic enable_progress_bar=false experiment_group=seeds dataset=pubmed,amazon,wiki_toxic,eurlex,agnews data.eval_batch_size=256 active_fit.query_size=25 strategy=seals seed=42,0,1994 model.seed=42,0,1994 +launcher=joblib'
echo $CMD | at now



poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=seeds \
    dataset=pubmed \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=random \
    seed=0,1994,2023,42,1234 \
    +launcher=joblib


poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    enable_progress_bar=false \
    experiment_group=seeds \
    dataset=pubmed \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=anchorswithperclasssampling \
    strategy.num_anchors=50 \
    strategy.anchor_strategy=kmeans \
    seed=42,0,1994 \
    model.seed=42,0,1994 \
    +launcher=joblib





poetry run python ./scripts/active_train.py +experiment=basic strategy=randomsubset dataset=wiki_toxic experiment_group=baselines strategy.subset_size=5000




# ours
poetry run python ./scripts/active_train.py \
    +experiment=basic \
    experiment_group=baselines \
    dataset=wiki_toxic \
    strategy=anchorswithperclasssampling \
    strategy.num_neighbours=2000 \
    strategy.subset_size=5000 \
    strategy.negative_dissimilar=false \
    strategy.pad_subset=false \
    strategy.positive_class_subset_prop=1. \
    strategy.temperatures=[0.5,1.]