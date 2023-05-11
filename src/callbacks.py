from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from energizer.callbacks import Callback
from energizer.datastores import PandasDataStoreForSequenceClassification
from energizer.enums import Interval, OutputKeys, RunningStage, SpecialKeys
from energizer.estimators.active_estimator import ActiveEstimator
from energizer.types import METRIC, ROUND_OUTPUT
from energizer.utilities import make_dict_json_serializable, move_to_cpu


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
        if self.instance_level:
            data = pd.DataFrame(columns=[f"logit_{i}" for i in range(model.num_labels)], data=output[OutputKeys.LOGITS])
            if SpecialKeys.ID in output:  # FIXME: enforce the ID column in every data store
                data[SpecialKeys.ID] = output[SpecialKeys.ID]

            data[Interval.EPOCH] = estimator.progress_tracker.safe_global_epoch

            # if we are active learning
            if hasattr(estimator.progress_tracker, "global_round"):
                data[Interval.ROUND] = estimator.progress_tracker.global_round
                if OutputKeys.SCORES in output:
                    data[OutputKeys.SCORES] = output[OutputKeys.SCORES]

            instance_level_path = path / "instance_level.csv"
            data.to_csv(
                instance_level_path,
                index=False,
                header=not instance_level_path.exists(),
                mode="a" if instance_level_path.exists() else "w",
            )

        # epoch-level output
        if self.epoch_level and stage != RunningStage.POOL:
            data = {
                "stage": stage,
                Interval.EPOCH: estimator.progress_tracker.safe_global_epoch,
                OutputKeys.LOSS: output[OutputKeys.LOSS],
                **move_to_cpu(metrics.compute()),
            }
            if hasattr(estimator.progress_tracker, "global_round"):
                data[Interval.ROUND] = estimator.progress_tracker.global_round

            # sanitize inputs for JSON serialization
            srsly.write_jsonl(
                path / "epoch_level.jsonl", [make_dict_json_serializable(data)], append=True, append_new_line=False
            )

    def on_round_end(
        self, estimator: ActiveEstimator, datastore: PandasDataStoreForSequenceClassification, output: ROUND_OUTPUT
    ) -> None:
        # save partial results
        datastore.save_labelled_dataset(self.dirpath)

    def on_active_fit_end(
        self,
        estimator: ActiveEstimator,
        datastore: PandasDataStoreForSequenceClassification,
        output: List[ROUND_OUTPUT],
    ) -> None:
        datastore.save_labelled_dataset(self.dirpath)
