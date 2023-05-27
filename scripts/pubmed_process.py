from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm


MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


def load_models(names: List[str]) -> Dict[str, SentenceTransformer]:
    return {name: SentenceTransformer(name) for name in names}


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":

    path = Path("data/raw/pubmed-rct/")
    embedders = load_models(MODELS)

    # load data
    data = {}
    for name in ("train", "dev", "test"):
        data[name] = []
        with (path / f"{name}.txt").open("r") as fl:
            for line in tqdm(fl.readlines(), desc="Reading"):
                if not line.startswith('#') and line.strip() != '':
                    label, text = line.split('\t')
                    data[name].append({"label": label, "text": text})
    data["validation"] = data.pop("dev")
    
    # create dataset dict
    ds_dict = DatasetDict({k: Dataset.from_pandas(pd.DataFrame(v), preserve_index=False) for k, v in data.items()})
    ds_dict = ds_dict.class_encode_column("label")
    ds_dict = ds_dict.rename_columns({"label": "labels"})

    # add features
    id_generator = sequential_numbers()
    ds_dict = ds_dict.map(
        lambda ex: {
            "uid": [next(id_generator) for _ in range(len(ex["text"]))],
            **{f"embedding_{k}": v.encode(ex["text"], device="cuda", batch_size=512) for k, v in embedders.items()},
        },
        batched=True,
        batch_size=1024,
    )

    # save
    ds_dict.save_to_disk("data/processed/pubmed-rct")  # type: ignore
    ds_dict.push_to_hub("pubmed-rct-200k_indexed")
