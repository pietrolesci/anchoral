import argparse
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk
from transformers import AutoTokenizer
from src.utilities import binarize_eurlex, MODELS, binarize_pubmed
from datasets import disable_caching
from os import cpu_count


LABEL_FN = {
    "eurlex-57k": binarize_eurlex,
    "pubmed-200k-rct": binarize_pubmed,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    # do not cache dataets
    disable_caching()
    
    # load data and metadata
    data_dir = Path(args.data_dir)
    dataset_dict = load_from_disk(data_dir / "processed" / args.dataset)

    # remove validation set
    dataset_dict.pop("validation", None)  # type: ignore
    
    # create label
    create_label_fn = LABEL_FN.get(args.dataset, None)
    if create_label_fn is not None:
        dataset_dict = dataset_dict.map(create_label_fn, desc="Binarizing", batched=True, num_proc=cpu_count())

    # select columns
    dataset_dict = dataset_dict.select_columns(["uid", "labels", "text"])

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODELS[args.model])
    dataset_dict = dataset_dict.map(lambda ex: tokenizer(ex["text"], return_token_type_ids=False), batched=True, desc="Tokenizing", num_proc=cpu_count())

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
