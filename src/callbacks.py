from pathlib import Path
from typing import Union

import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.callbacks import Callback
from energizer.enums import Interval, OutputKeys, RunningStage, SpecialKeys
from energizer.types import METRIC, ROUND_OUTPUT


class SaveOutputs(Callback):
    """All the logic to save artifacts.

    Convinient to put it here and not in SequenceClassificationMixin because I can then
    switch it off when debugging and avoid saving stuff on disk.
    """

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
        output: dict,
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

    def on_round_end(
        self,
        estimator: ActiveEstimator,
        datastore: ActivePandasDataStoreForSequenceClassification,
        output: ROUND_OUTPUT,
    ) -> None:
        # save partial results
        datastore.save_labelled_dataset(self.dirpath)

    def on_active_fit_end(
        self,
        estimator: ActiveEstimator,
        datastore: ActivePandasDataStoreForSequenceClassification,
        output: list[ROUND_OUTPUT],
    ) -> None:
        # at the end save the labelled dataset (overwrites partial savings)
        datastore.save_labelled_dataset(self.dirpath)

    def on_select_pool_subset_end(self, estimator: ActiveEstimator, subpool_ids: list[int]) -> None:
        # NOTE: this is ad-hoc in SequenceClassificationMixin
        data = {"subpool_ids": subpool_ids, SpecialKeys.LABELLING_ROUND: estimator.tracker.global_round}
        srsly.write_jsonl(self.dirpath / "subpool_ids.jsonl", [data], append=True)

    def on_select_search_query_end(self, estimator: ActiveEstimator, search_query_ids: list[int]) -> None:
        # NOTE: this is ad-hoc in SequenceClassificationMixin
        data = {"search_query_ids": search_query_ids, SpecialKeys.LABELLING_ROUND: estimator.tracker.global_round}
        srsly.write_jsonl(self.dirpath / "search_query_ids.jsonl", [data], append=True, append_new_line=False)
