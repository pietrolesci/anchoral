import argparse
from pathlib import Path

import pandas as pd
import srsly
from datasets import Dataset, DatasetDict
from datasets.features import ClassLabel, Features, Value
from sklearn.model_selection import train_test_split

from src.enums import InputColumns, RunningStage

############################################################################################
# Download data from Kaggle
# https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
############################################################################################


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    # load data
    df = pd.read_csv(Path(args.input_dir) / "all_data.csv")

    # rename columns
    df = df.rename(columns={"comment_text": "text"})

    # select columns
    cols = ["split", "id", "text", "toxicity"]
    df = df[cols].copy()

    # drop null values
    df = df.dropna(subset=["text", "toxicity"])

    # compute unique texts exact
    df["unique_id"] = df.groupby("text").ngroup().astype(int)

    # compute average toxicity across equal texts
    df["avg_toxicity"] = df.groupby("unique_id")["toxicity"].transform("mean")

    # remove duplicates within split
    df = df.drop_duplicates(subset=["split", "text"])

    # remove duplicates across splits
    ddf = df.sort_values(["unique_id", "split"]).drop_duplicates(subset=["unique_id"])

    # check that duplicates are removed from training rather than testing
    assert df["split"].value_counts()["test"] == ddf["split"].value_counts()["test"]

    # select columns
    df = ddf.drop(columns=["toxicity", "unique_id"])

    # binarize labels
    df[InputColumns.TARGET] = (df["avg_toxicity"] >= 0.5).astype(int)

    train_df = df.loc[df["split"] == "train"].drop(columns=["split", "avg_toxicity"])
    test_df = df.loc[df["split"] == "test"].drop(columns=["split", "avg_toxicity"])

    features = Features(
        {
            "id": Value(dtype="int32", id=None),
            InputColumns.TARGET: ClassLabel(num_classes=2, names=["not_toxic", "toxic"], names_file=None, id=None),
            "text": Value(dtype="string", id=None),
        }
    )

    # put everything together
    dataset_dict = DatasetDict(
        {
            RunningStage.TRAIN: Dataset.from_pandas(train_df, preserve_index=False, features=features),
            RunningStage.TEST: Dataset.from_pandas(test_df, preserve_index=False, features=features),
        }
    )

    # save to disk
    dataset_dict.save_to_disk(args.output_dir)

    # metadata
    meta = {"train_val_seed": args.seed}
    srsly.write_yaml(Path(args.output_dir) / "metadata.yaml", meta)
