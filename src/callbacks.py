from pathlib import Path
from typing import Dict, List, Union

import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.energizer.active_learning.base import ActiveEstimator
from src.energizer.active_learning.callbacks import ActiveLearningCallbackMixin
from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.callbacks.base import Callback
from src.energizer.enums import OutputKeys, RunningStage, SpecialKeys
from src.energizer.estimator import Estimator
from src.energizer.types import EPOCH_OUTPUT, METRIC, ROUND_OUTPUT


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
            # check that file exists
            epoch = estimator.progress_tracker.get_epoch_num()
            step = "round" if stage in (RunningStage.TEST, RunningStage.POOL) else "epoch"
            instance_level_path = path / f"instance_level_{step}={epoch}.csv"
            if not instance_level_path.exists():
                columns = [f"logit_{i}" for i in range(estimator.model.num_labels)] + [SpecialKeys.ID, step]
                pd.DataFrame(columns=columns).to_csv(instance_level_path, index=False)
            # append data
            (
                pd.DataFrame(output[OutputKeys.LOGITS])
                .assign(unique_id=output[SpecialKeys.ID], epoch=epoch)
                .to_csv(instance_level_path, mode="a", index=False, header=False)
            )

        if self.epoch_level and stage != RunningStage.POOL:
            data = {
                "stage": stage,
                "epoch": estimator.progress_tracker.get_epoch_num(),
                "round": getattr(estimator.progress_tracker, "num_rounds"),
                "loss": output[OutputKeys.LOSS],
                **output[OutputKeys.METRICS],
            }
            srsly.write_jsonl(path / "epoch_level.jsonl", [data], append=True)

    def on_round_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ROUND_OUTPUT) -> None:
        test_out = output.test

        # flatten logs (metrics)
        test_out = {OutputKeys.LOSS: test_out[OutputKeys.LOSS], **test_out[OutputKeys.METRICS]}

        # log using labelled size as the x-axis
        logs = {f"test_end/{k}_vs_budget": v for k, v in test_out.items()}
        estimator.log_dict(logs, step=datamodule.train_size)

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: List[ROUND_OUTPUT]
    ) -> None:
        datamodule.save_labelled_dataset(self.dirpath)
