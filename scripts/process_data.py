import json
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from seaborn import load_dataset
from sentence_transformers import SentenceTransformer

from src.utilities import sequential_numbers


def load_models(names: list[str]) -> dict[str, SentenceTransformer]:
    return {name: SentenceTransformer(name) for name in names}


def add_features(ds_dict: DatasetDict, text_col: str, models: list[str]) -> DatasetDict:
    """Adds embeddings and unique ids."""
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


def load_amazoncat(path: Path) -> DatasetDict:
    """Load dataset into DatasetDict and create `text` column."""
    data = {}
    for split in ("trn", "tst"):
        with open(path / f"{split}.json") as fl:
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


def load_from_hub(name: str) -> DatasetDict:
    dataset_dict: DatasetDict = load_dataset(name)  # type: ignore
    if "wiki_toxic" in name:
        dataset_dict = dataset_dict.rename_column("comment_text", "text")

    return dataset_dict  # type: ignore


LOADING_FN = {"amazoncat-13k": load_amazoncat}
MODELS = ["all-mpnet-base-v2"]  # "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_dir)

    # === LOAD === #
    print("Loading")
    if args.dataset in LOADING_FN:
        ds_dict = LOADING_FN[args.dataset](data_path / "raw" / args.dataset)
    else:
        ds_dict = load_from_hub(args.dataset)

    # === PROCESSING === #
    print(f"Adding unique id and embeddings {MODELS}")
    start_time = time.perf_counter()
    ds_dict = add_features(ds_dict, "text", MODELS)
    print(f"Time required for feature creation={time.perf_counter() - start_time}")

    # === SAVE === #
    print("Saving")

    # parse name
    name = args.dataset
    if "/" in args.dataset:
        args.dataset.split("/")[1]
    name = name.replace("_", "")

    # create embedding subsets
    emb_columns = [i for i in ds_dict["train"].features if i.startswith("embedding")]
    data_columns = [i for i in ds_dict["train"].features if i not in emb_columns and i != "uid"]

    # create data subset
    data_subset = DatasetDict({k: v.remove_columns(emb_columns) for k, v in ds_dict.items()})

    # create embedding subset
    emb_subsets = {
        col: DatasetDict(
            {
                split: dataset.remove_columns(data_columns + [i for i in emb_columns if i != col])
                for split, dataset in ds_dict.items()
            }
        )
        for col in emb_columns
    }

    # push and save data subset
    data_subset.push_to_hub(name)
    data_subset.save_to_disk(data_path / "processed" / name)  # type: ignore

    # push embedding subset
    for k, subset in emb_subsets.items():
        subset.push_to_hub(name, k)
        subset.save_to_disk(data_path / "processed" / name / k)

    if args.dataset == "amazoncat-13k":
        with (data_path / "raw" / args.dataset / "Yf.txt").open(mode="r", encoding="latin-1") as fl:
            df = pd.DataFrame(fl.read().split("\n"), columns=["labels"])
            Dataset.from_pandas(df).push_to_hub(name, "labels")
