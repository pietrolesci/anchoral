# ALLSET

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
conda create -n allset python=3.9 -y
conda activate allset
poetry install --sync --with dev
```


---


## Download, process, and prepare data

Since it's tricky to download data from Kaggle through the CLI, manually dowload the Civil Comments
dataset from [here](https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data)
and unzip it into `./data/raw/civil_comments`. Then,

```bash
./bin/process_data.sh
./bin/prepare_data.sh
```

## Logging (HuggingFace classifier experiments)

Logs are split in 3 granularity groups: instance-level, batch-level, epoch-level.

- Instance-level: these are the raw logits

- Batch-level: these are the loss, and the various metrics passes

- Epoch-level: these are the same as batch-level but aggregated across batches