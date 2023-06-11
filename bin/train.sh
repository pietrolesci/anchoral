
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

poetry run python ./scripts/active_train.py -m \
    +experiment=basic \
    experiment_group=seeds \
    dataset=pubmed \
    data.eval_batch_size=256 \
    active_fit.query_size=25 \
    strategy=random \
    seed=0,1994,2023,42,1234 \
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