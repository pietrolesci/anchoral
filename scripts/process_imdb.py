import argparse
from pathlib import Path

import pandas as pd
import srsly
from datasets import Dataset, DatasetDict, load_dataset
from datasets.features import Features, Value
from tqdm.auto import tqdm

from src.energizer.enums import InputKeys, RunningStage, SpecialKeys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    pbar = tqdm(total=3)

    # ============ STEP 1 ============
    pbar.set_description("Loading and tidying")

    # download data
    dataset_dict = load_dataset("imdb")
    class_labels_feature = dataset_dict["train"].features["label"]  # take if from the original

    # cast to dataframe
    df = pd.concat(
        [dataset_dict[split].to_pandas().assign(split=split) for split in dataset_dict if split in ("train", "test")]
    )

    # rename columns
    df = df.rename(columns={"label": InputKeys.TARGET, "text": InputKeys.TEXT})

    # drop Nans
    pre_len = len(df)
    df = df.dropna(subset=[InputKeys.TEXT, InputKeys.TARGET])
    assert pre_len == len(df)

    # deduplicate making sure each duplicate has the same label
    dups = df.loc[df["text"].duplicated(False)].sort_values(InputKeys.TEXT)
    dups["id"] = dups.groupby(InputKeys.TEXT).ngroup()
    lab_per_dup = dups.groupby("id")[InputKeys.TARGET].nunique()
    assert (lab_per_dup > 1).sum() == 0

    # drop duplicates
    df = df.drop_duplicates(subset=[InputKeys.TEXT])
    assert df[InputKeys.TEXT].duplicated().sum() == 0
    assert (df[InputKeys.TARGET] < 0).sum() == 0

    # add unique_id column
    df[SpecialKeys.ID] = list(range(len(df)))

    pbar.update(1)

    # ============ STEP 2 ============
    pbar.set_description("Splitting")

    # train-val split
    train_df = df.loc[df["split"] == "train"].drop(columns=["split"])
    test_df = df.loc[df["split"] == "test"].drop(columns=["split"])
    pbar.update(1)

    # ============ STEP 3 ============
    pbar.set_description("Saving")

    features = Features(
        {
            SpecialKeys.ID: Value(dtype="int32", id=None),
            InputKeys.TARGET: class_labels_feature,
            InputKeys.TEXT: Value(dtype="string", id=None),
        }
    )

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

    pbar.update(1)
