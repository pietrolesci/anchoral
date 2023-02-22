import argparse
from pathlib import Path

import srsly
from datasets import load_from_disk, Dataset, DatasetDict
from transformers import AutoTokenizer



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--name_or_path", type=str)
    parser.add_argument("--name_or_path_alias", type=str)
    args = parser.parse_args()

    # load data and metadata
    input_dir = Path(args.input_dir) / args.dataset_name
    dataset_dict = load_from_disk(input_dir)
    meta = srsly.read_yaml(input_dir / "metadata.yaml")

    # define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.name_or_path)

    # tokenize
    dataset_dict = dataset_dict.map(lambda ex: tokenizer(ex["text"]), batched=True)
    
    # sort by length
    new_dataset_dict = {}
    for split, dataset in dataset_dict.items():
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

    # update metadata
    meta["name_or_path"] = args.name_or_path
    meta["name_or_path_alias"] = args.name_or_path_alias

    # save to disk
    output_dir = Path(args.output_dir) / f"{args.dataset_name}_{args.name_or_path_alias}"
    dataset_dict.save_to_disk(output_dir)
    srsly.write_yaml(output_dir / "metadata.yaml", meta)


