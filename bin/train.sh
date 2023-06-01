
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