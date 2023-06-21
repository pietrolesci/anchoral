import shutil
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    dataset = args.dataset.split("/")[1] if "/" in args.dataset else args.dataset
    data_path = Path(args.data_dir) / "processed" / dataset

    # downlaod
    ds_dict = load_dataset(args.dataset, cache_dir="data_cache")

    if "wiki_toxic" in args.dataset:
        ds_dict = ds_dict.rename_columns({"comment_text": "text"})

    # save to appropriate folder
    ds_dict.save_to_disk(data_path)  # type: ignore

    # remove cache
    shutil.rmtree("data_cache", ignore_errors=True)
