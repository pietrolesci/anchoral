from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.energizer.active_learning.active_estimator import ActiveEstimator
from src.energizer.active_learning.callbacks import ActiveLearningCallbackMixin
from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.callbacks.base import Callback
from src.energizer.enums import Interval, OutputKeys, RunningStage, SpecialKeys
from src.energizer.estimator import Estimator
from src.energizer.types import EPOCH_OUTPUT, METRIC, ROUND_OUTPUT
from src.energizer.utilities import make_dict_json_serializable


class SaveOutputs(ActiveLearningCallbackMixin, Callback):
    """All the logic to save artifacts and log."""

    def __init__(self, dirpath: Union[str, Path], instance_level: bool, batch_level: bool, epoch_level: bool) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.instance_level = instance_level
        self.batch_level = batch_level
        self.epoch_level = epoch_level

    """
    Method forwarding
    """

    def on_train_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, RunningStage.TRAIN)

    def on_validation_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, RunningStage.VALIDATION)

    def on_test_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, RunningStage.TEST)

    def on_pool_epoch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.on_epoch_end(estimator, output, RunningStage.POOL)

    """
    Actual methods
    """

    def on_epoch_end(self, estimator: Union[Estimator, ActiveEstimator], output: Dict, stage: RunningStage) -> None:
        # output directory setup
        path = self.dirpath / f"{stage}"
        if not path.exists():
            path.mkdir(exist_ok=True, parents=True)

        # instance-level output
        if self.instance_level:
            data = pd.DataFrame(
                columns=[f"logit_{i}" for i in range(estimator.model.num_labels)], data=output[OutputKeys.LOGITS]
            )
            data[SpecialKeys.ID] = output[SpecialKeys.ID]
            data[Interval.EPOCH] = estimator.progress_tracker.global_epoch

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
                Interval.EPOCH: estimator.progress_tracker.global_epoch,
                OutputKeys.LOSS: output[OutputKeys.LOSS],
                **output[OutputKeys.METRICS],
            }
            if hasattr(estimator.progress_tracker, "global_round"):
                data[Interval.ROUND] = estimator.progress_tracker.global_round

            # sanitize inputs for JSON serialization
            srsly.write_jsonl(
                path / "epoch_level.jsonl", [make_dict_json_serializable(data)], append=True, append_new_line=False
            )

    def on_round_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ROUND_OUTPUT) -> None:
        # save partial results
        datamodule.save_labelled_dataset(self.dirpath)

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: List[ROUND_OUTPUT]
    ) -> None:
        datamodule.save_labelled_dataset(self.dirpath)
