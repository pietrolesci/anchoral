from argparse import ArgumentParser
from pathlib import Path

from src.utilities import add_features, load_amazoncat, load_eurlex, load_models, load_pubmed

MODELS = ["all-mpnet-base-v2", "multi-qa-mpnet-base-dot-v1", "all-MiniLM-L12-v2"]


LOADING_FN = {
    "pubmed-200k-rct": load_pubmed,
    "eurlex-57k": load_eurlex,
    "amazoncat-13k": load_amazoncat,
}


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    path = data_path / "raw" / args.dataset

    # load
    print("Loading")
    ds_dict = LOADING_FN[args.dataset](path)

    # add uid and embeddings
    print("Processing")
    ds_dict = add_features(ds_dict, "text", MODELS)

    # save
    print("Saving")
    ds_dict.save_to_disk(data_path / "processed" / args.dataset)  # type: ignore
    ds_dict.push_to_hub(f"{args.dataset}")
