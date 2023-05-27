from pathlib import Path
from typing import Any, Dict, Generator, List

import pandas as pd
import srsly
from datasets import Dataset, DatasetDict
from sentence_transformers import SentenceTransformer

MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


def load_models(names: List[str]) -> Dict[str, SentenceTransformer]:
    return {name: SentenceTransformer(name) for name in names}


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":

    path = Path("data/raw/eurlex/")
    embedders = load_models(MODELS)

    # load data
    ds_dict = DatasetDict(
        {
            name: Dataset.from_pandas(
                pd.DataFrame(srsly.read_jsonl(path / f"{name}.jsonl")).assign(  # type: ignore
                    text=lambda _df: _df["title"] + _df["recitals"]
                ),
                preserve_index=False,
            )
            for name in ("train", "dev", "test")
        }
    )
    ds_dict["validation"] = ds_dict.pop("dev")

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
    ds_dict.save_to_disk("data/processed/eurlex")  # type: ignore
    ds_dict.push_to_hub("eurlex_indexed")

    # save indices
