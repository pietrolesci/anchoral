from logging import Logger
from typing import Any, Callable, Dict, Generator, List, Optional

import hnswlib as hb
import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features
from omegaconf import OmegaConf
from sentence_transformers import SentenceTransformer
from sklearn.utils import check_random_state
from tabulate import tabulate
from tqdm.auto import tqdm

from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys
from energizer.utilities import sample

SEP_LINE = f"{'=' * 70}"
import json
from pathlib import Path

MODELS = {"bert-tiny": "google/bert_uncased_L-2_H-128_A-2"}


def parse_name(x: Dict) -> str:
    name = x["_target_"].split(".")[-1].lower().removesuffix("strategy")
    if "anchor_strategy" in x:
        name = f"{x['anchor_strategy']}-{name}-{x['agg_fn']}"
    return name


OmegaConf.register_new_resolver("get_name_strategy", parse_name)


def get_stats_from_dataframe(df: pd.DataFrame, target_name: str, names: List[str]) -> str:
    df = pd.concat(
        [df[target_name].value_counts(normalize=True), df[target_name].value_counts()],
        axis=1,
        keys=("perc", "count"),
    )
    df["labels"] = [names[i] for i in df.index]
    df = df.sort_index(ascending=True)[df.columns.tolist()[::-1]]
    return tabulate(df)  # type: ignore


def binarize_labels(dataset_dict: DatasetDict, target_name: str, positive_class: int, logger: Logger) -> DatasetDict:
    features = dict(dataset_dict["train"].features)
    logger.info(f"Binarizing using class {features[target_name].names[positive_class]} as positive")
    for split in ("train", "test"):
        logger.info(f"Original {split} label distribution\n{get_stats_from_dataframe(dataset_dict[split].to_pandas(), target_name, features[target_name].names)}")  # type: ignore

    features[target_name] = ClassLabel(names=["Others", features[target_name].names[positive_class]])  # positive is 1
    logger.info(f"Binarized labels {features[target_name]}")

    dataset_dict = dataset_dict.map(
        lambda ex: {target_name: [int(l == positive_class) for l in ex[target_name]]},
        batched=True,
        features=Features(features),
    )

    for split in ("train", "test"):
        logger.info(f"Binarized {split} label distribution\n{get_stats_from_dataframe(dataset_dict[split].to_pandas(), target_name, features[target_name].names)}")  # type: ignore

    return dataset_dict


def downsample_positive_class(
    dataset_dict: DatasetDict, target_name: str, positive_class: int, proportion: float, seed: int, logger: Logger
) -> DatasetDict:
    features = dataset_dict["train"].features
    df: DataFrame = dataset_dict["train"].to_pandas()  # type: ignore

    current_proportion = (df[target_name] == positive_class).sum() / len(df)
    if current_proportion < proportion:
        logger.info(
            f"Current proportion ({current_proportion:.2%}) < Selected proportion ({proportion:.2%}). Doing nothing"
        )
        return dataset_dict

    n = int(len(df) * proportion)
    logger.info(
        f"Previous train distribution\n{get_stats_from_dataframe(df, target_name, features[target_name].names)}"
    )

    rng = check_random_state(seed)
    ids = df.loc[df[target_name] == positive_class].sample(n, random_state=rng).index
    df = df.loc[(df[target_name] != positive_class) | (df.index.isin(ids))]
    logger.info(f"Current train distribution\n{get_stats_from_dataframe(df, target_name, features[target_name].names)}")

    dataset_dict["train"] = Dataset.from_pandas(df, features=features, preserve_index=False)

    return dataset_dict


def downsample_test_set(
    dataset_dict: DatasetDict, target_name: str, positive_class: int, test_set_size: int, seed: int, logger: Logger
) -> DatasetDict:
    features = dataset_dict["test"].features
    df: DataFrame = dataset_dict["test"].to_pandas()  # type: ignore

    current_size = len(df)
    if current_size < test_set_size:
        logger.info(f"Current test set size ({current_size:.2%}) < Selected size ({test_set_size:.2%}). Doing nothing")
        return dataset_dict

    rng = check_random_state(seed)

    # keep everything of the positive class
    positive_df = df.loc[df[target_name] == positive_class]
    others_df = df.loc[df[target_name] != positive_class]
    ids = sample(
        indices=others_df.index.tolist(),
        size=test_set_size,
        random_state=rng,
        labels=others_df[target_name].tolist(),
        sampling="stratified",
    )
    others_df = others_df.loc[others_df.index.isin(ids)]
    new_df = pd.concat([positive_df, others_df])

    logger.info(f"Previous test distribution\n{get_stats_from_dataframe(df, target_name, features[target_name].names)}")
    logger.info(
        f"Current test distribution\n{get_stats_from_dataframe(new_df, target_name, features[target_name].names)}"
    )

    dataset_dict["test"] = Dataset.from_pandas(new_df, features=features, preserve_index=False)

    return dataset_dict


def get_initial_budget(
    datastore: PandasDataStoreForSequenceClassification,
    positive_budget: int,
    total_budget: int,
    positive_class: int,
    seed: int,
    validation_perc: Optional[float],
    sampling: Optional[str],
    logger: Logger,
) -> None:
    rng = check_random_state(seed)

    pos_uids = (
        datastore.data.loc[datastore.data[InputKeys.TARGET] == positive_class]
        .sample(positive_budget, random_state=rng)[SpecialKeys.ID]
        .tolist()
    )
    other_uids = (
        datastore.data.loc[datastore.data[InputKeys.TARGET] != positive_class]
        .sample(total_budget - positive_budget, random_state=rng)[SpecialKeys.ID]
        .tolist()
    )

    ids = pos_uids + other_uids

    datastore.label(
        indices=ids,
        round=-1,
        validation_perc=validation_perc,
        validation_sampling=sampling,
    )

    stats = get_stats_from_dataframe(
        df=datastore.data.loc[datastore._labelled_mask()],
        target_name=InputKeys.TARGET,
        names=["Negative", "Positive"],
    )

    logger.info(
        f"Labelled size: {datastore.labelled_size()} "
        f"Pool size: {datastore.pool_size()} "
        f"Test size: {datastore.test_size()}\n"
        f"Label distribution:\n{stats}"
    )


"""
Datasets
"""


def load_models(names: List[str]) -> Dict[str, SentenceTransformer]:
    return {name: SentenceTransformer(name) for name in names}


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1


def load_pubmed(path: Path) -> DatasetDict:
    """Load dataset into DatasetDict and create `text` column."""
    data = {}
    for name in ("train", "dev", "test"):
        data[name] = []
        with (path / f"{name}.txt").open("r") as fl:
            for line in tqdm(fl.readlines(), desc="Reading"):
                if not line.startswith("#") and line.strip() != "":
                    label, text = line.split("\t")
                    data[name].append({"labels": label, "text": text})
    data["validation"] = data.pop("dev")

    ds_dict = DatasetDict({k: Dataset.from_pandas(pd.DataFrame(v), preserve_index=False) for k, v in data.items()})
    ds_dict = ds_dict.class_encode_column("labels")

    return ds_dict


def load_eurlex(path: Path) -> DatasetDict:
    """Load dataset into DatasetDict and create `text` column."""
    ds_dict = DatasetDict(
        {
            name: Dataset.from_pandas(
                pd.DataFrame(srsly.read_jsonl(path / f"{name}.jsonl")).assign(  # type: ignore
                    text=lambda _df: _df["title"] + _df["recitals"],
                ),
                preserve_index=False,
            )
            for name in ("train", "dev", "test")
        }
    )
    ds_dict["validation"] = ds_dict.pop("dev")
    return ds_dict


def load_amazoncat(path: Path) -> DatasetDict:
    """Load dataset into DatasetDict and create `text` column."""
    data = {}
    for split in ("trn", "tst"):
        with open(path / f"{split}.json", "r") as fl:
            data[split] = Dataset.from_pandas(
                df=(
                    pd.DataFrame([json.loads(i) for i in fl.readlines()])
                    .assign(text=lambda _df: _df["title"] + "\n" + _df["content"])
                    .rename(columns={"uid": "uid_original"})
                ),
                preserve_index=False,
            )

    data["train"] = data.pop("trn")
    data["test"] = data.pop("tst")

    return DatasetDict(data)


def add_features(ds_dict: DatasetDict, text_col: str, models: List[str]) -> DatasetDict:
    id_generator = sequential_numbers()
    embedders = load_models(models)

    ds_dict = ds_dict.map(
        lambda ex: {
            "uid": [next(id_generator) for _ in range(len(ex[text_col]))],
            **{f"embedding_{k}": v.encode(ex[text_col], device="cuda", batch_size=512) for k, v in embedders.items()},
        },
        batched=True,
        batch_size=1024,
    )
    return ds_dict


def binarize_eurlex(ex: Dict[str, List]) -> Dict:
    """Make `health control` the target label."""
    # return {"labels": ["health-control" if "192" in l else "others" for l in ex["eurovoc_concepts"]]}
    return {"labels": [int("192" in l) for l in ex["eurovoc_concepts"]]}


def binarize_pubmed(ex: Dict[str, List]) -> Dict:
    """Make `OBJECTIVE` the target label."""
    return {"labels": [int(3 == l) for l in ex["labels"]]}


def binarize_agnews(ex: Dict[str, List]) -> Dict:
    """Make `Business` the target label."""
    return {"labels": [int(2 == l) for l in ex["labels"]]}


def binarize_amazon(ex: Dict[str, List]) -> Dict:
    """Make `Business` the target label."""
    return {"labels": [int(2 == l) for l in ex["target_ind"]]}
