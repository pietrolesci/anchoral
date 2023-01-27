import argparse
from pathlib import Path

import srsly
from datasets import load_from_disk
from transformers import AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--name_or_path", type=str)
    args = parser.parse_args()

    # load data
    dataset_dict = load_from_disk(args.input_dir)

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    # tokenize
    dataset_dict = dataset_dict.map(lambda ex: tokenizer(ex["text"]), batched=True)

    # save to disk
    dataset_dict.save_to_disk(args.output_dir)

    # metadata
    meta = {"name_or_path": args.name_or_path}
    srsly.write_yaml(Path(args.output_dir) / "metadata.yaml", meta)
