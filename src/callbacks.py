from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
from lightning.fabric.wrappers import _FabricModule

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.callbacks import Callback
from energizer.enums import InputKeys, Interval, OutputKeys, RunningStage, SpecialKeys
from energizer.types import METRIC, ROUND_OUTPUT


class SaveOutputs(Callback):
    """All the logic to save artifacts and log."""

    def __init__(self, dirpath: Union[str, Path], instance_level: bool, batch_level: bool, epoch_level: bool) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.instance_level = instance_level
        self.batch_level = batch_level
        self.epoch_level = epoch_level

    def on_epoch_end(
        self,
        stage: Union[str, RunningStage],
        estimator: ActiveEstimator,
        model: _FabricModule,
        output: Dict,
        metrics: METRIC,
    ) -> None:

        # output directory setup
        path = self.dirpath / f"{stage}"
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

        # instance-level output
        if self.instance_level and stage == RunningStage.TEST:
            data = pd.DataFrame(columns=[f"logit_{i}" for i in range(model.num_labels)], data=output[OutputKeys.LOGITS])
            data[SpecialKeys.ID] = output[SpecialKeys.ID]
            data[Interval.ROUND] = getattr(estimator.tracker, "global_round", 0)

            instance_level_path = path / "instance_level.csv"
            data.to_csv(
                instance_level_path,
                index=False,
                header=not instance_level_path.exists(),
                mode="a" if instance_level_path.exists() else "w",
            )

    def on_query_end(
        self,
        estimator: ActiveEstimator,
        model: _FabricModule,
        datastore: ActivePandasDataStoreForSequenceClassification,
        indices: List[int],
    ) -> None:
        counts = dict(datastore.get_by_ids(indices)[InputKeys.TARGET].value_counts())
        for i in (0, 1):
            if 1 not in counts:
                counts[i] = 0  # type: ignore

        estimator.log_dict(
            {f"summary/count_class_{k}": v for k, v in counts.items()},
            step=estimator.tracker.global_round,
        )

    def on_round_end(
        self,
        estimator: ActiveEstimator,
        datastore: ActivePandasDataStoreForSequenceClassification,
        output: ROUND_OUTPUT,
    ) -> None:

        # save partial results
        datastore.save_labelled_dataset(self.dirpath)
        # if getattr(estimator, "_reason_df", None) is not None:
        #     estimator._reason_df.to_parquet(Path(self.dirpath) / "reason_df.parquet")  # type: ignore

        counts = dict(datastore.get_by_ids(datastore.get_train_ids())[InputKeys.TARGET].value_counts())
        if 1 not in counts:
            counts[1] = 0  # type: ignore

        estimator.log_dict(
            {
                **{f"summary/cumulative_count_class_{k}": v for k, v in counts.items()},
                "summary/labelled_size": datastore.labelled_size(),
                "summary/pool_size": datastore.pool_size(),
                "summary/cumulative_minority_ratio": counts[1] / counts[0],
            },
            step=estimator.tracker.global_round,
        )

    def on_active_fit_end(
        self,
        estimator: ActiveEstimator,
        datastore: ActivePandasDataStoreForSequenceClassification,
        output: List[ROUND_OUTPUT],
    ) -> None:
        datastore.save_labelled_dataset(self.dirpath)
