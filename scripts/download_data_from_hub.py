import shutil
from argparse import ArgumentParser
from pathlib import Path

from datasets import load_dataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--embedding", type=str)
    args = parser.parse_args()

    dataset = args.dataset.split("/")[1] if "/" in args.dataset else args.dataset
    data_path = Path(args.data_dir) / dataset

    # download
    ds_dict = load_dataset(args.dataset, cache_dir=".data_cache")
    emb_dict = load_dataset(args.dataset, args.embedding, cache_dir=".data_cache", split="train")

    # save to appropriate folder
    ds_dict.save_to_disk(data_path)  # type: ignore
    emb_dict.save_to_disk(data_path / args.embedding)  # type: ignore

    # remove cache
    shutil.rmtree(".data_cache", ignore_errors=True)
