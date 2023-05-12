from typing import Any, Dict, Generator, List

from datasets import ClassLabel, DatasetDict, load_dataset
from sentence_transformers import SentenceTransformer

DATASETS = [
    (
        "ag_news",
        "text",
        "label",
    ),
    ("imdb", "text", "label"),
    ("OxAISH-AL-LLM/wiki_toxic", "comment_text", "label"),
    ("armanc/pubmed-rct20k", "text", "label"),
    # ("OxAISH-AL-LLM/pubmed_20k_rct", "text", "label"),
    # ("dbpedia_14", "content", "label"),
    # ("DeveloperOats/DBPedia_Classes", "text", ["l1", "l2", "l3"]),
]

MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


def load_models(names: List[str]) -> Dict[str, SentenceTransformer]:
    return {name: SentenceTransformer(name) for name in names}


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1


if __name__ == "__main__":

    embedders = load_models(MODELS)

    for name, text_col, label_col in DATASETS:

        print(f"Processing {name}")

        dataset_dict: DatasetDict = load_dataset(name)  # type: ignore
        for k in list(dataset_dict.keys()):  # type: ignore
            if k not in ("train", "validation", "test"):
                dataset_dict.pop(k)  # type: ignore

        if isinstance(label_col, str):
            if not isinstance(dataset_dict["train"].features[label_col], ClassLabel):
                dataset_dict = dataset_dict.class_encode_column(label_col)
            dataset_dict = dataset_dict.rename_columns({label_col: "labels"})
        else:
            for label in label_col:
                dataset_dict = dataset_dict.class_encode_column(label)

        id_generator = sequential_numbers()
        dataset_dict = dataset_dict.map(
            lambda ex: {
                "uid": [next(id_generator) for _ in range(len(ex[text_col]))],
                **{
                    f"embedding_{k}": v.encode(ex[text_col], device="cuda", batch_size=512)
                    for k, v in embedders.items()
                },
            },
            batched=True,
            batch_size=1024,
        )

        if "/" in name:
            name = name.split("/")[1]

        dataset_dict.save_to_disk(f"data/processed/{name}")  # type: ignore
        dataset_dict.push_to_hub(f"{name}_indexed")
