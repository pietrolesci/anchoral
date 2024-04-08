import shutil
from argparse import ArgumentParser
from pathlib import Path

import duckdb
import pandas as pd
import srsly
from tqdm.auto import tqdm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--experiment_dir", type=str)
    parser.add_argument("--out_dir", type=str)
    args = parser.parse_args()

    path = Path(args.experiment_dir)
    out_path = Path(args.out_dir)
    con = duckdb.connect()

    # COLLECT LOGS
    print("Collecting all results")
    tb_path = f"{path}/*/*/tensorboard_logs.parquet"
    query = f"""
    SELECT *
    FROM read_parquet('{tb_path}', filename=True)
    """
    all_df = con.execute(query).df()

    # COLLECT HPARAMS
    hparams = []
    for p in tqdm(list(path.rglob("*tensorboard_logs.parquet")), desc="Collecting Hyper-Parameters"):
        meta: dict = srsly.read_yaml(p.parent / "hparams.yaml")  # type: ignore
        print(meta["strategy"]["args"])
        hparam_dict = {
            "filename": str(p.parent),
            "data_seed": meta["data"]["seed"],
            "model_seed": meta["model"]["seed"],
            "initial_seed": meta["active_data"]["seed"],
            "global_seed": meta["seed"],
            "retriever": meta["index_metric"],
            "dataset_name": meta["dataset"]["name"],
            "model_name": meta["model"]["name"],
            "strategy_name": meta["strategy"]["name"],
            **{f"strategy_{k}": v for k, v in meta["strategy"]["args"].items()},
        }
        hparams.append(hparam_dict)

    hparams_df = pd.DataFrame(hparams)

    # SAVE
    out_path.mkdir(exist_ok=True, parents=True)
    all_df.to_parquet(out_path / "results.parquet", index=False)
    hparams_df.to_parquet(out_path / "hparams.parquet", index=False)

    shutil.make_archive(out_path.name, "zip", out_path)
    shutil.rmtree(out_path)
