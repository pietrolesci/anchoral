# AnchorAL


```bash
git clone --recurse-submodules <repo-URL>
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
./bin/download_data.sh
./bin/process_data.sh
```

### Step 2: Create local HNSW index

To speed up the experimentation, we create the HNSW index once and save it to disk. For each embedding in the dataset this will save a different `.bin` file. By default it will create indices using the cosine distance, change this file if you want to experiment with different metrics. 

```bash
./bin/create_index.sh
```

### Step 3: Prepare data for training

Finally, we tokenize and save the dataset so it is ready to go

```bash
./bin/prepare_data.sh
```


## Logging (HuggingFace classifier experiments)

Logs are split in 3 granularity groups: instance-level, batch-level, epoch-level.

- Instance-level: these are the raw logits

- Batch-level: these are the loss, and the various metrics passes

- Epoch-level: these are the same as batch-level but aggregated across batches



----

### Utils [DELETE]

```bash
exp='num_neighbours'
find ./outputs/multirun/$exp -type d -wholename "*/.model_cache" -exec rm -rf {} +
find ./outputs/multirun/$exp -type d -wholename "*/logs/train" -exec rm -rf {} +
find ./outputs/multirun/$exp -type d -wholename "*/logs/test" -exec rm -rf {} +
find ./outputs/multirun/$exp -type d -wholename "*/logs/pool" -exec rm -rf {} +
find ./outputs/multirun/$exp -type d -wholename "*/.checkpoints" -exec rm -rf {} +
```
