import argparse
import math
from os import cpu_count
from pathlib import Path

import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, disable_caching, load_from_disk
from sklearn.utils import check_random_state
from transformers import AutoTokenizer

from src.utilities import MODELS, binarize_agnews, binarize_eurlex, binarize_pubmed, get_stats_from_dataframe

LABEL_FN = {
    "eurlex-57k": binarize_eurlex,
    "pubmed-200k-rct": binarize_pubmed,
    "agnews": binarize_agnews,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--downsample_prop", type=float, default=None)
    parser.add_argument("--downsample_test_size", type=int, default=None)
    args = parser.parse_args()

    # do not cache dataets
    disable_caching()

    # load data and metadata
    data_dir = Path(args.data_dir)
    dataset_dict: DatasetDict = load_from_disk(data_dir / "processed" / args.dataset)  # type: ignore

    # remove validation set
    dataset_dict.pop("validation", None)  # type: ignore

    # create label
    create_label_fn = LABEL_FN.get(args.dataset, None)
    if create_label_fn is not None:
        features = dict(dataset_dict["train"].features)
        features["labels"] = ClassLabel(names=["Negative", "Positive"])
        dataset_dict = dataset_dict.map(
            create_label_fn, desc="Binarizing", batched=True, num_proc=cpu_count(), features=Features(features)
        )

    # select columns
    dataset_dict = dataset_dict.select_columns(["uid", "labels", "text"])

    # downsample positive class
    if args.downsample_prop is not None:
        rng = check_random_state(42)
        df: pd.DataFrame = dataset_dict["train"].to_pandas()  # type: ignore
        n = math.ceil((args.downsample_prop * (df["labels"] == 0).sum()) / (1 - args.downsample_prop))

        print("Train before downsamples\n", get_stats_from_dataframe(df, "labels", dataset_dict["train"].features["labels"].names))  # type: ignore
        print(f"Subsampling positive class to {n}")
        ids = df.loc[df["labels"] == 1].sample(n, random_state=rng).index
        df = df.loc[(df["labels"] != 1) | (df.index.isin(ids))]

        dataset_dict["train"] = Dataset.from_pandas(df, preserve_index=False, features=dataset_dict["train"].features)

    if args.downsample_test_size is not None:
        rng = check_random_state(42)
        df: pd.DataFrame = dataset_dict["test"].to_pandas()  # type: ignore

        print("Test before downsamples\n", get_stats_from_dataframe(df, "labels", dataset_dict["test"].features["labels"].names))  # type: ignore
        print(f"Subsampling test set negative class to {args.downsample_test_size}")
        ids = df.loc[df["labels"] == 0].sample(args.downsample_test_size, random_state=rng).index
        df = df.loc[(df["labels"] != 0) | (df.index.isin(ids))]
        dataset_dict["test"] = Dataset.from_pandas(df, preserve_index=False, features=dataset_dict["test"].features)

    # get stats
    print("Train\n", get_stats_from_dataframe(dataset_dict["train"].to_pandas(), "labels", dataset_dict["train"].features["labels"].names))  # type: ignore
    print("Test\n", get_stats_from_dataframe(dataset_dict["test"].to_pandas(), "labels", dataset_dict["test"].features["labels"].names))  # type: ignore

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    dataset_dict = dataset_dict.map(
        lambda ex: tokenizer(ex["text"], return_token_type_ids=False),
        batched=True,
        desc="Tokenizing",
        num_proc=cpu_count(),
    )

    # sort by length to optimise inference time -- training set will be shuffled anyway later
    new_dataset_dict = {}
    for split, dataset in dataset_dict.items():  # type: ignore
        new_dataset_dict[split] = Dataset.from_pandas(
            df=(
                dataset.to_pandas()
                .assign(length=lambda df_: df_["input_ids"].map(len))
                .sort_values("length")
                .drop(columns="length")
            ),
            features=dataset.features,
            preserve_index=False,
        )
    dataset_dict = DatasetDict(new_dataset_dict)

    # save
    dataset_dict.save_to_disk(data_dir / "prepared" / f"{args.dataset}_{args.model}")
