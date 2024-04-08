import argparse
import math
from os import cpu_count
from pathlib import Path

import pandas as pd
import srsly
from datasets import ClassLabel, Dataset, DatasetDict, Features, load_from_disk
from sklearn.utils import check_random_state
from transformers import AutoTokenizer

from energizer.enums import SpecialKeys
from energizer.utilities import sample
from src.utilities import MODELS, get_stats_from_dataframe

"""
audio = ["audio & video accessories"]
archaeology = ["archaeology", "archaeology & paleontology", "paleontology"]
agriculture = ["agricultural sciences"]
religion = [
    'comparative religion',
    'cultures & religions',
    'earth-based religions',
    'history of religion',
    'other eastern religions',
    'other eastern religions & sacred texts',
    'other religions',
    'religion',
    'religion & spirituality',
    'religions',
    'science & religion'
]
philosophy = [
    'analytic philosophy',
    'eastern philosophy',
    'educational philosophy',
    'history & philosophy',
    'philosophy',
    'philosophy & social aspects',
    'social philosophy',
    'zen philosophy',
]
video_games = ['plug & play video games', 'video games', 'video games & accessories']

# no sex toys
toys = [
    'activities & toys',
    'adult toys & games',
    'baby & toddler toys',
    'basic & life skills toys',
    'bath toys',
    'beach toys',
    'bird toys',
    'building toys',
    'car seat & stroller toys',
    'catnip toys',
    'chew toys',
    'crib toys & attachments',
    'dive rings & toys',
    'electronic toys',
    'feather toys',
    'flying toys',
    'gag toys & practical jokes',
    'games & toys',
    'grown-up toys',
    'hammering & pounding toys',
    'kitchen toys',
    'laser toys',
    'light-up toys',
    'magnets & magnetic toys',
    'mice & animal toys',
    'novelty & gag toys',
    'pool toys',
    'push & pull toys',
    'ride-on toys',
    'slime & putty toys',
    'small animal toys',
    'squeak toys',
    'stacking & nesting toys',
    'stuffed animals & toys',
    'toys',
    'toys & figurines',
    'toys & game room',
    'toys & games',
    'toys & models',
    'wind-up toys',
    'wood toys'
]
"""
AMAZON_CATEGORIES = {
    # "video_games": [9127, 12771, 12772],
    # "religion": [2790, 3241, 3988, 5886, 8445, 8446, 8453, 9904, 9905, 9906, 10371],  # `religion`
    "philosophy": [413, 3999, 4044, 5881, 8900, 8901, 10990, 13319],
    "archaeology": [538, 539, 8601],
    "agriculture": [246],  # `agricultural sciences`
    "audio": [677],
    # "toys": [
    #     138,
    #     209,
    #     746,
    #     960,
    #     1018,
    #     1076,
    #     1296,
    #     1715,
    #     2000,
    #     2163,
    #     2327,
    #     3168,
    #     3698,
    #     4152,
    #     4525,
    #     4831,
    #     5099,
    #     5132,
    #     5475,
    #     5578,
    #     6689,
    #     6830,
    #     7007,
    #     7269,
    #     7624,
    #     8212,
    #     9232,
    #     9626,
    #     10015,
    #     10885,
    #     10912,
    #     11287,
    #     11297,
    #     11538,
    #     12243,
    #     12244,
    #     12245,
    #     12246,
    #     12247,
    #     13112,
    #     13225,
    # ],
}


def binarize_eurlex(ex: dict[str, list], class_to_binarize: str) -> dict:
    cats = {"health": "192"}  # `health control`
    return {"labels": [int(cats[class_to_binarize] in label_list) for label_list in ex["eurovoc_concepts"]]}


def binarize_amazon(ex: dict[str, list], class_to_binarize: str) -> dict:
    return {
        "labels": [
            int(any(c in label_list for c in AMAZON_CATEGORIES[class_to_binarize])) for label_list in ex["target_ind"]
        ]
    }


def binarize_pubmed(ex: dict[str, list], class_to_binarize: str) -> dict:
    cats = {"objective": 3}
    return {"labels": [int(cats[class_to_binarize] == label) for label in ex["labels"]]}


def binarize_agnews(ex: dict[str, list], class_to_binarize: str) -> dict:
    cats = {"business": 2}
    return {"labels": [int(cats[class_to_binarize] == label) for label in ex["labels"]]}


def multiclass_amazon(ex: dict[str, list], labels: list[str]) -> dict:
    def assign_label(label_list: list[int]) -> int:
        _lab = "others"
        for name, ids in AMAZON_CATEGORIES.items():
            for _id in ids:
                if _id in label_list:
                    # NOTE: the order in AMAZON_CATEGORIES is important because here we overwrite
                    _lab = name
        return labels.index(_lab)

    return {"labels": [assign_label(label_list) for label_list in ex["target_ind"]]}


BINARIZE_FN = {
    "eurlex-57k": binarize_eurlex,
    "pubmed-200k-rct": binarize_pubmed,
    "agnews": binarize_agnews,
    "amazoncat-13k": binarize_amazon,
}

MULTICLASS_FN = {"amazoncat-13k": multiclass_amazon}

LABELS_COLUMN_RENAMING_MAP = {"hyperpartisan_news_detection": "bias", "yahoo_answers_topics": "topic"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--output_name", type=str)
    parser.add_argument("--downsample_prop", type=float, default=None)
    parser.add_argument("--downsample_test_size", type=int, default=None)
    parser.add_argument("--class_to_binarize", type=str, default=None)
    parser.add_argument("--stratified", default=False, action="store_true")
    args = parser.parse_args()

    # do not cache datasets
    # disable_caching()

    dataset_path = Path(args.dataset_path)
    output_path = dataset_path.parents[1] / "prepared" / f"{args.output_name}_{args.model}"
    assert not output_path.exists(), f"{output_path} already exists"

    dataset_name = dataset_path.name
    meta_data = {
        "binarised": False,
        "binarised_class": None,
        "train_set_subsampled": False,
        "test_set_subsampled": False,
        "tokenizer": None,
        "train_set_stats": None,
        "test_set_stats": None,
    }

    # load data and metadata
    dataset_dict: DatasetDict = load_from_disk(dataset_path)  # type: ignore

    # remove validation set
    if "test" not in dataset_dict and "validation" in dataset_dict:
        dataset_dict["test"] = dataset_dict["validation"]
    dataset_dict.pop("validation", None)  # type: ignore

    # create label
    if args.class_to_binarize is not None:
        if args.class_to_binarize != "all":
            print("Binarising labels where 0: Negative and 1: Positive")
            binarize_fn = BINARIZE_FN[dataset_name]

            features = dict(dataset_dict["train"].features)
            features["labels"] = ClassLabel(names=["Negative", "Positive"])

            dataset_dict = dataset_dict.map(
                lambda ex: binarize_fn(ex, args.class_to_binarize),
                desc="Binarizing",
                batched=True,
                num_proc=cpu_count(),
                features=Features(features),
            )

            meta_data["binarised"] = True
            meta_data["binarised_class"] = args.class_to_binarize
        else:
            print("Multiclass labels")
            multiclass_fn = MULTICLASS_FN[dataset_name]
            labels = [*sorted(AMAZON_CATEGORIES.keys()), "others"]

            features = dict(dataset_dict["train"].features)
            features["labels"] = ClassLabel(names=labels)

            dataset_dict = dataset_dict.map(
                lambda ex: multiclass_fn(ex, labels),
                desc="Multiclass encoding",
                batched=True,
                num_proc=cpu_count(),
                features=Features(features),
            )

            meta_data["binarised"] = True
            meta_data["binarised_class"] = args.class_to_binarize

    # select columns
    if dataset_name in LABELS_COLUMN_RENAMING_MAP:
        dataset_dict = dataset_dict.rename_column(LABELS_COLUMN_RENAMING_MAP[dataset_name], "labels")
    dataset_dict = dataset_dict.select_columns(["uid", "labels", "text"])

    # downsample positive class
    if args.downsample_prop is not None:
        rng = check_random_state(42)
        df: pd.DataFrame = dataset_dict["train"].to_pandas()  # type: ignore
        n = math.ceil((args.downsample_prop * (df["labels"] == 0).sum()) / (1 - args.downsample_prop))

        stats = get_stats_from_dataframe(df, "labels", dataset_dict["train"].features["labels"].names)
        print(f"Train before downsamples\n{stats}")
        print(f"Subsampling positive class to {n}")

        ids = df.loc[df["labels"] == 1].sample(n, random_state=rng).index
        df = df.loc[(df["labels"] != 1) | (df.index.isin(ids))]
        dataset_dict["train"] = Dataset.from_pandas(df, preserve_index=False, features=dataset_dict["train"].features)

        meta_data["train_set_subsampled"] = True

    if args.downsample_test_size is not None:
        rng = check_random_state(42)
        df: pd.DataFrame = dataset_dict["test"].to_pandas()  # type: ignore

        stats = get_stats_from_dataframe(df, "labels", dataset_dict["test"].features["labels"].names)
        print(f"Test before downsamples\n{stats}")
        print(f"Subsampling test set negative class to {args.downsample_test_size}")

        if args.stratified:
            print("Stratified sampling")
            ids = sample(
                df[SpecialKeys.ID].tolist(),
                size=args.downsample_test_size,
                random_state=rng,
                mode="stratified",
                labels=df["labels"],
            )
            df = df.loc[df[SpecialKeys.ID].isin(ids)]

        else:
            maj_id = (
                0 if args.class_to_binarize != "all" else dataset_dict["train"].features["labels"].names.index("others")
            )
            ids = df.loc[df["labels"] == maj_id].sample(args.downsample_test_size, random_state=rng).index
            df = df.loc[(df["labels"] != maj_id) | (df.index.isin(ids))]

        dataset_dict["test"] = Dataset.from_pandas(df, preserve_index=False, features=dataset_dict["test"].features)

        meta_data["test_set_subsampled"] = True

    # get stats
    for split in ("train", "test"):
        stats = get_stats_from_dataframe(
            df=dataset_dict[split].to_pandas(),  # type: ignore
            target_name="labels",
            names=dataset_dict[split].features["labels"].names,
        )
        meta_data[f"{split}_set_stats"] = stats
        print(f"Final {split} dataset stats:\n{stats}")

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    dataset_dict = dataset_dict.map(
        lambda ex: tokenizer(ex["text"], return_token_type_ids=False, truncation=True, padding=False),
        batched=True,
        desc="Tokenizing",
        num_proc=cpu_count(),
    )
    meta_data["model"] = args.model

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
    dataset_dict.save_to_disk(output_path)
    srsly.write_json(output_path / "meta_data.json", meta_data)
