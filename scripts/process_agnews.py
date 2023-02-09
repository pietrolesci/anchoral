import argparse
from pathlib import Path

import pandas as pd
import srsly
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.model_selection import train_test_split

from src.enums import InputColumns, RunningStage

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # download data
    dataset_dict = load_dataset("ag_news")
    features = dataset_dict["train"].features

    # cast to dataframe
    df = pd.concat([dataset_dict[split].to_pandas().assign(split=split) for split in dataset_dict])

    # rename "label" to InputColumns.TARGET
    df = df.rename(columns={"label": InputColumns.TARGET})
    features[InputColumns.TARGET] = features.pop("label")

    # train-val split
    train_df = df.loc[df["split"] == "train"].drop(columns=["split"])

    # # compute quantiles to stratify on sequence length
    # train_df["q"] = pd.qcut(train_df["text"].str.len(), 10)

    # # make validation twice as big as test
    # val_size = (df["split"] == "test").sum() * 2

    # # actually train-val split
    # train_ids, val_ids = train_test_split(
    #     train_df.index, stratify=train_df["q"], test_size=val_size, random_state=args.seed
    # )
    # val_df = (
    #     train_df.loc[train_df.index.isin(val_ids)].assign(split=RunningStage.VALIDATION).drop(columns=["q", "split"])
    # )
    # train_df = (
    #     train_df.loc[train_df.index.isin(train_ids)].assign(split=RunningStage.TRAIN).drop(columns=["q", "split"])
    # )
    test_df = df.loc[df["split"] == "test"].drop(columns=["split"])

    # put everything together
    dataset_dict = DatasetDict(
        {
            RunningStage.TRAIN: Dataset.from_pandas(train_df, preserve_index=False, features=features),
            # RunningStage.VALIDATION: Dataset.from_pandas(val_df, preserve_index=False, features=features),
            RunningStage.TEST: Dataset.from_pandas(test_df, preserve_index=False, features=features),
        }
    )

    # save to disk
    dataset_dict.save_to_disk(args.output_dir)

    # metadata
    meta = {"train_val_seed": args.seed}
    srsly.write_yaml(Path(args.output_dir) / "metadata.yaml", meta)
