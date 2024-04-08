# ================================
# Run comparison with other models
# ================================

# Ablation 4: other models
EXPERIMENT_GROUP=other_models

MODELS=albert-base-v2,bert-tiny,deberta-v3-base,gpt2,t5-base

DATASETS=agnews_business,amazon_agri

STRATEGIES=anchoral_entropy,randomsubset_entropy,seals_entropy

poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP \
    dataset=$DATASETS \
    strategy=$STRATEGIES \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS \
    model.name=$MODELS \
    ++strategy.args.num_neighbours=500,5000


# This will create an `outputs/multirun/other_models/*` folder. 
# After it finishes, move it to `outputs/other_models/*` to make analysis easier with the provided notebooks
