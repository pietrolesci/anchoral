import argparse
from pathlib import Path

import pandas as pd
import srsly
from datasets import Dataset, DatasetDict
from datasets.features import ClassLabel, Features, Value
from sklearn.utils import resample
from tqdm.auto import tqdm

from src.energizer.enums import InputKeys, RunningStage, SpecialKeys

############################################################################################
# Download data from Kaggle
# https://www.kaggle.com/competitions/jigsaw-unintended-bias-in-toxicity-classification/data
############################################################################################


def subsample(df: pd.DataFrame, n_samples: int, min_chars: int, seed: int) -> pd.DataFrame:
    df["len"] = df[InputKeys.TEXT].str.replace("\W", "", regex=True).str.len()
    df = df.loc[df["len"] >= min_chars].sort_values(["len"])
    ids = resample(df.index, replace=False, stratify=df["labels"], n_samples=n_samples, random_state=seed)
    df = df.loc[df.index.isin(ids)].drop(columns=["len"])
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--input_dir", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train_samples", type=int, default=100_000)
    parser.add_argument("--test_samples", type=int, default=50_000)
    parser.add_argument("--min_chars", type=int, default=10)
    args = parser.parse_args()

    pbar = tqdm(total=4)

    # ============ STEP 1 ============
    pbar.set_description("Loading and tidying")

    # load data
    df = pd.read_csv(Path(args.input_dir) / "all_data.csv")

    # rename columns
    df = df.rename(columns={"comment_text": InputKeys.TEXT})

    # select columns
    cols = ["split", "id", "toxicity", InputKeys.TEXT]
    df = df[cols].copy()

    # drop null values
    df = df.dropna(subset=[InputKeys.TEXT, "toxicity"])

    pbar.update(1)

    # ============ STEP 2 ============
    pbar.set_description("Removing duplicates and aggregating")

    # compute unique texts exact
    df[SpecialKeys.ID] = df.groupby(InputKeys.TEXT).ngroup().astype(int)

    # compute average toxicity across equal texts
    df["avg_toxicity"] = df.groupby(SpecialKeys.ID)["toxicity"].transform("mean")

    # remove duplicates within split
    df = df.drop_duplicates(subset=["split", InputKeys.TEXT])

    # remove duplicates across splits
    ddf = df.sort_values([SpecialKeys.ID, "split"]).drop_duplicates(subset=[SpecialKeys.ID])

    # check that duplicates are removed from training rather than testing
    assert df["split"].value_counts()["test"] == ddf["split"].value_counts()["test"]

    # select columns
    df = ddf.drop(columns=["toxicity", SpecialKeys.ID])

    # binarize labels
    df[InputKeys.TARGET] = (df["avg_toxicity"] >= 0.5).astype(int)

    # rename id column in order to use it
    df = df.rename(columns={"id": SpecialKeys.ID})

    pbar.update(1)

    # ============ STEP 3 ============
    pbar.set_description("Splitting and subsampling")

    # split in train and test
    train_df = df.loc[df["split"] == "train"].drop(columns=["split", "avg_toxicity"])
    test_df = df.loc[df["split"] == "test"].drop(columns=["split", "avg_toxicity"])

    # subsample training set
    train_df = subsample(train_df, n_samples=args.train_samples, min_chars=args.min_chars, seed=args.seed)
    test_df = subsample(test_df, n_samples=args.test_samples, min_chars=args.min_chars, seed=args.seed)

    pbar.update(1)

    # ============ STEP 4 ============
    pbar.set_description("Saving")

    features = Features(
        {
            SpecialKeys.ID: Value(dtype="int32", id=None),
            InputKeys.TARGET: ClassLabel(num_classes=2, names=["not_toxic", "toxic"], names_file=None, id=None),
            InputKeys.TEXT: Value(dtype="string", id=None),
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
    meta = {
        "seed": args.seed,
        "train_samples": args.train_samples,
        "test_samples": args.test_samples,
        "min_chars": args.min_chars,
    }
    srsly.write_yaml(Path(args.output_dir) / "metadata.yaml", meta)

    pbar.update(1)
