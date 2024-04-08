# AnchorAL

Code for the paper "AnchorAL: Computationally Efficient Active Learning for Large and Imbalanced Datasets" published at NAACL 2024



```bash
git clone --recurse-submodules https://github.com/pietrolesci/anchoral.git
```

## Environment

### Install conda

[source](https://educe-ubc.github.io/conda.html)
[source](https://developers.google.com/earth-engine/guides/python_install-conda)

```bash
curl -sL \
  "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" > \
  "Miniconda3.sh"
bash Miniconda3.sh -b -p
rm Miniconda3.sh
source $HOME/miniconda3/bin/activate
conda init zsh
```

### Install poetry

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### Install dependencies
Use poetry to install the environment (if you don't have poetry run )

```bash
conda create -n anchoral python=3.9 -y
conda activate anchoral
poetry install --sync --with dev
```


---


## Data Preparation

### Step 1: Download and process

To download the processed data from the hub run

```bash
./bin/download_data_from_hub.sh
```

To re-process the data locally (this indexes the data with 3 sentence-transformers -- takes a long time)

```bash
./bin/download_and_process_data.sh
```

### Step 2: Create local HNSW index

To speed up the experimentation, we create the HNSW index once and save it to disk. For each embedding in the dataset this will save a different `.bin` file. By default it will create indices using the cosine distance, change this file if you want to experiment with different metrics. 

```bash
./bin/create_index.sh
```

### Step 3: Prepare data for training

Finally, we tokenize and save the dataset so it is ready to go

```bash
./bin/prepare_data.sh bert-base-uncased
```

At the end of data preparations you should have the following folder structure

```bash
./data
├── prepared
│   ├── agnews-business-.01_bert-base-uncased
│   ├── amazoncat-agri_bert-base-uncased
│   ├── amazoncat-multi_bert-base-uncased
│   └── wikitoxic-.01_bert-base-uncased
└── processed
    ├── agnews
    ├── amazoncat-13k
    └── wikitoxic
```


## Step 3: Run experiments

You can replicate our experiments by looking at the files `./bin/run_main_experiments`, `./bin/run_ablations`, and `./bin/run_other_models` (more instruction in those files).
Once you run these experiments, you should obtain the following folder structure

```bash
./outputs
├── ablations
│   ├── anchor_strategy
│   │   ├── bert-base-uncased_anchoral_badge_2023-12-08T18-23-01_36952945_1
│   │   ├── bert-base-uncased_anchoral_badge_2023-12-08T18-23-01_36952945_2
│   │   ├── ...
│   ├── num_anchors
│   └── num_neighbours
├── main
│   ├── agnews-business-.01
│   ├── amazon-agri
│   ├── amazon-multi
│   └── wikitoxic-.01
└── other_models
    ├── albert-base-v2
    ├── bert-tiny
    ├── deberta-v3-base
    ├── gpt2
    └── t5-base
```

Each run has the following folder structure

```bash
├── active_train.log
├── .early_stopping.jsonl
├── hparams.yaml
├── logs
│   ├── labelled_dataset.parquet
│   ├── ... (optionally, based on the strategy)
├── tb_logs
│   └── version_0
│       └── events.out.tfevents.xxx
└── tensorboard_logs.parquet
```

If you limit the time of each run, it can happen that some files are not deleted and tensorboard logs are not exported to parquet.
In those cases, use the `./notebooks/01_check_runs.ipynb` to manually export it and possibly delete other (big) files and folders (e.g., `.checkpoints` and `model_cache`) that are not needed.


If you want to run new experiments, consider that we use hydra for configuration. 
Thus, to run more experiments in parallel, you can assign comma-separate options, e.g. `SEEDS=654321,123456`.

```bash
# select from: 654321, 123456
SEEDS=654321

# select from: agnews_business, amazon_agri,amazon_multi, wikitoxic
DATASET=agnews_business

# select from: bert-base-uncased, bert-tiny, deberta-v3-base, albert-base-v2, gpt2, t5-base
MODEL=bert-base-uncased

# select from: {anchoral, randomsubset, seals}_{entropy, badge, ftbertkm} or random
STRATEGY=anchoral_entropy

# assign a name to the experiment
EXPERIMENT_GROUP=main

# run experiment
poetry run python ./scripts/active_train.py -m \
    experiment_group=$EXPERIMENT_GROUP \
    model.name=$MODEL \
    dataset=$DATASET \
    strategy=$STRATEGY \
    data.seed=$SEEDS \
    model.seed=$SEEDS \
    active_data.seed=$SEEDS
```

You might need to modify the configurations and include the absolute path to your data (if you use slurm, check also the `conf/launcher/slurm.yaml/` file).
The key to edit is `data_path: <add path>`.



## Step 4: Analysis

Once you have run the experiments and created the correct folder structure, as decribed in the previous section, you can run the analysis.

First, run the `notebooks/01_check_runs.ipynb` to make sure that each run has the necessary files to run the analysis.
Importantly, you need the `tensorboard_logs.parquet` files. If your run exited before exporting the tensorboard logs to parquet, you can use the notebook to do that.

Second, once you have all the `tensorboard_logs.parquet` files for each run, you can export the necessary metrics from them.
To do this, use the `notebooks/03_export_experiments.ipynb`. It will create files into the `results/` folder. This artefacts, will be used in the analysis.

Finally, once you have all the artefacts in the `results/` folder, you can run the analysis.
You can do this by running the `notebooks/04_analysis.ipynb` notebooks which creates the tables and plots used in the paper.