# ==========================
# Run ablations for anchoral
# ==========================

EXPERIMENT_GROUP=ablations

# one dataset
DATASET=amazon_agri

# one strategy
STRATEGY=anchoral_entropy

# one bert-base-uncased
MODELS=bert-base-uncased

# two seeds per each source of randomness
SEEDS=654321,123456

# Ablation 1: anchor_strategy 
poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP/anchor_strategy \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=$MODELS \
    ++strategy.args.anchor_strategy_majority=entropy \
    ++strategy.args.anchor_strategy_minority=entropy,kmeans_pp_sampling

# Ablation 2: anchor_strategy 
poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP/num_anchors \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=$MODELS \
    ++strategy.args.num_anchors=50,100

# Ablation 3: num_neighbours 
poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP/num_neighbours \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=$MODELS \
    ++strategy.args.num_neighbours=500,5000

# This will create an `outputs/multirun/ablations/*` folder. 
# After it finishes, move it to `outputs/ablations/*` to make analysis easier with the provided notebooks