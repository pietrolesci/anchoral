import argparse
from pathlib import Path

import hnswlib as hb
import numpy as np
import pandas as pd
import srsly
from datasets import load_from_disk
from tqdm.auto import tqdm

ef_construction: int = 200
ef: int = 200
M: int = 64
num_threads: int = 5

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--index_metric", type=str, default="cosine")
    args = parser.parse_args()

    # load data and metadata
    data_dir = Path(args.data_dir) / "processed" / args.dataset
    df: pd.DataFrame = load_from_disk(data_dir)["train"].to_pandas()  # type: ignore

    uid = df["uid"].tolist()
    for col in tqdm(df.columns[df.columns.str.startswith("embedding")], desc="Creating index"):
        emb = np.stack(df[col].values)  # type: ignore

        index = hb.Index(space=args.index_metric, dim=emb.shape[1])
        index.set_ef(ef)
        index.init_index(max_elements=emb.shape[0], M=M, ef_construction=ef_construction, random_seed=42)
        index.add_items(emb, uid, num_threads=num_threads)

        out_path = str(data_dir / f"{col.split('embedding_')[1]}_{args.index_metric}")
        index.save_index(f"{out_path}.bin")

        meta = {
            "ef_construction": ef_construction,
            "ef": ef,
            "M": M,
            "num_threads": num_threads,
            "dim": emb.shape[1],
            "metric": args.index_metric,
        }
        srsly.write_json(f"{out_path}.json", meta)
