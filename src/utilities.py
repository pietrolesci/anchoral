import shutil
from collections.abc import Generator
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

# from omegaconf import OmegaConf
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.enums import InputKeys, SpecialKeys

SEP_LINE = f"{'=' * 70}"

MODELS = {
    "bert-tiny": "google/bert_uncased_L-2_H-128_A-2",
    "bert-base-uncased": "bert-base-uncased",
    "deberta-v3-base": "microsoft/deberta-v3-base",
    "albert-base-v2": "albert-base-v2",
    "gpt2": "gpt2",
    "t5-base": "t5-base",
}


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1


def remove_dir(path: Union[str, Path]) -> None:
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)


def get_stats_from_dataframe(df: pd.DataFrame, target_name: str, names: list[str]) -> dict:
    df = pd.concat(
        [df[target_name].value_counts(normalize=True), df[target_name].value_counts()], axis=1, keys=("perc", "count")
    )
    df["labels"] = [names[i] for i in df.index]
    df = df.sort_index(ascending=True)[df.columns.tolist()[::-1]]
    return df.to_dict()


def get_initial_budget(
    datastore: ActivePandasDataStoreForSequenceClassification,
    positive_budget: Optional[int],
    total_budget: int,
    minority_classes: Optional[list[int]],
    seed: int,
) -> list[int]:
    rng = check_random_state(seed)

    if minority_classes is not None and positive_budget is not None:
        pos_uids = []

        for c in minority_classes:
            pos_uids += (
                datastore._train_data.loc[datastore._train_data[InputKeys.LABELS] == c]
                .sample(positive_budget, random_state=rng)[SpecialKeys.ID]
                .tolist()
            )

        other_uids = (
            datastore._train_data.loc[~datastore._train_data[InputKeys.LABELS].isin(minority_classes)]
            .sample(total_budget - len(set(pos_uids)), random_state=rng)[SpecialKeys.ID]
            .tolist()
        )

        return pos_uids + other_uids

    print("`minority_classes` is not passed. Sampling randomly from the pool.")
    return datastore.sample_from_pool(total_budget, random_state=rng)
