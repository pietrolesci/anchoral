import shutil
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


LOADING_FN = ["pubmed-200k-rct", "eurlex-57k", "amazoncat-13k", "agnews"]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset = args.dataset.split("/")[1] if "/" in args.dataset else args.dataset
    data_path = Path(args.data_dir) / "processed" / dataset

    # downlaod
    ds_dict = load_dataset(args.dataset, cache_dir="data_cache")

    # save to appropriate folder
    ds_dict.save_to_disk(data_path)  # type: ignore

    # remove cache
    shutil.rmtree("data_cache")
