import time
from pathlib import Path
from typing import Any, Union

import numpy as np
import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.energizer.active_learning.base import ActiveEstimator
from src.energizer.active_learning.callbacks import ActiveLearningCallbackMixin
from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.active_learning.progress_trackers import ActiveProgressTracker
from src.energizer.callbacks import Callback, Timer
from src.energizer.containers import ActiveFitOutput, QueryOutput, RoundOutput
from src.energizer.enums import OutputKeys, RunningStage, SpecialKeys
from src.energizer.estimator import Estimator
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import ld_to_dl, move_to_cpu


class SaveOutputs(ActiveLearningCallbackMixin, Callback):
    """All the logic to sae artifacts and log."""

    def __init__(self, dirpath: Union[str, Path], instance_level: bool, batch_level: bool, epoch_level: bool) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.instance_level = instance_level
        self.batch_level = batch_level
        self.epoch_level = epoch_level

    def on_batch_end(
        self, estimator: Union[Estimator, ActiveEstimator], output: BATCH_OUTPUT, batch_idx: int, stage: RunningStage
    ) -> None:
        # NOTE: only log at the batch level for the training loop
        if stage != RunningStage.TRAIN:
            return output

        # control logging interval
        if estimator.progress_tracker.should_log(batch_idx):
            # NOTE: output is still on device
            logs = {OutputKeys.LOSS: output[OutputKeys.LOSS], **output[OutputKeys.METRICS]}

            # rename and move to cpu
            logs = move_to_cpu({f"{stage}/{k}": v for k, v in logs.items()})

            # log
            estimator.fabric.log_dict(logs, step=estimator.progress_tracker.get_batch_num(stage))

    def on_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        stage: RunningStage,
    ) -> None:
        # output directory setup
        suffix = f"_epoch_{estimator.progress_tracker.get_epoch_num(stage)}"
        path = self.dirpath / f"{stage}"
        path.mkdir(exist_ok=True, parents=True)

        # list of dicts to dict of lists
        data = ld_to_dl(output)

        # instance-level output
        if self.instance_level:
            logits = np.concatenate(data.pop(OutputKeys.LOGITS))
            idx = np.concatenate(data.pop(SpecialKeys.ID))
            instance_level_output = pd.DataFrame(logits, columns=[f"logit_{i}" for i in range(logits.shape[1])]).assign(
                **{SpecialKeys.ID: idx}
            )
            instance_level_output.to_parquet(path / f"instance_level{suffix}.parquet", index=False)

        # batch-level output
        if self.batch_level:
            batch_metrics = data.pop(OutputKeys.METRICS)
            batch_metrics = {k: [dic[k] for dic in batch_metrics] for k in batch_metrics[0]}
            batch_level_output = pd.DataFrame(batch_metrics).assign(loss=data[OutputKeys.LOSS])
            batch_level_output.to_parquet(path / f"batch_level{suffix}.parquet", index=False)

        # epoch-level output
        epoch_level_output = {
            **move_to_cpu(metrics.compute()),
            f"avg_{OutputKeys.LOSS}": round(np.mean(data[OutputKeys.LOSS]), 6),
        }
        if self.epoch_level:
            srsly.write_json(path / f"epoch_level{suffix}.json", epoch_level_output)

        if stage != RunningStage.TEST:
            logs = {f"{stage}_end/{k}": v for k, v in epoch_level_output.items()}
            estimator.fabric.log_dict(logs, step=estimator.progress_tracker.get_epoch_num(stage))

    def on_round_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        logs = {f"test_end/{k}": v for k, v in output.test.output.items()}
        estimator.fabric.log_dict(logs, step=datamodule.train_size)
        estimator.fabric.log_dict(
            {f"{k}_vs_rounds": v for k, v in logs.items()}, step=estimator.progress_tracker.num_rounds
        )

    def on_train_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, metrics, RunningStage.TRAIN)

    def on_validation_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, metrics, RunningStage.VALIDATION)

    def on_test_epoch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        **kwargs,
    ) -> None:
        self.on_epoch_end(estimator, output, metrics, RunningStage.TEST)

    def on_train_batch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.on_batch_end(estimator, output, batch_idx, RunningStage.TRAIN)

    def on_validation_batch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.on_batch_end(estimator, output, batch_idx, RunningStage.VALIDATION)

    def on_test_batch_end(
        self,
        estimator: Union[Estimator, ActiveEstimator],
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        self.on_batch_end(estimator, output, batch_idx, RunningStage.TEST)

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        datamodule.save_labelled_dataset(self.dirpath)

        # NOTE: output.output is a List[RoundOutput]
        # each RoundOutput has a `.test` attribute which is an EvaluationOutput
        # the EvaluationOutput has the attribute `.output`
        test_outputs = ld_to_dl([out.output for out in output.output])

        # NOTE: no aggregation since we aggregate in the `{stage}_epoch_end` methods
        test_outputs = {f"active_learning_end/test_{k}_auc": np.trapz(v) for k, v in test_outputs.items()}
        estimator.fabric.log_dict(test_outputs, step=0)


class Timer(ActiveLearningCallbackMixin, Timer):
    def _batch_step(self, progress_tracker: ActiveProgressTracker, stage: RunningStage, batch_idx: int) -> int:
        return getattr(progress_tracker, f"total_{stage}_batches") + getattr(progress_tracker, f"num_{stage}_batches")

    def _epoch_step(self, progress_tracker: ActiveProgressTracker, stage: RunningStage) -> int:
        if stage == RunningStage.TEST:
            return progress_tracker.round_tracker.current
        return getattr(progress_tracker, "total_epochs") + getattr(progress_tracker, "num_epochs")

    def on_active_fit_start(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        self.active_fit_start = time.perf_counter()

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        self.active_fit_end = time.perf_counter()
        estimator.fabric.log("timer/active_fit_time", self.active_fit_end - self.active_fit_start, step=0)

    def on_round_start(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.round_start = time.perf_counter()

    def on_round_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.round_end = time.perf_counter()
        estimator.fabric.log(
            "timer/round_time", self.round_end - self.round_start, step=estimator.progress_tracker.num_rounds
        )

    def on_query_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        self.query_start = time.perf_counter()

    def on_query_end(self, estimator: ActiveEstimator, model: _FabricModule, output: QueryOutput) -> None:
        self.query_end = time.perf_counter()
        estimator.fabric.log(
            "timer/query_time", self.query_end - self.query_start, step=estimator.progress_tracker.num_rounds
        )

    def on_label_start(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.label_start = time.perf_counter()

    def on_label_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.label_end = time.perf_counter()
        estimator.fabric.log(
            "timer/label_time", self.label_end - self.label_start, step=estimator.progress_tracker.num_rounds
        )

    def on_pool_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.POOL)

    def on_pool_epoch_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.POOL)

    def on_pool_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.POOL)

    def on_pool_batch_end(self, estimator: ActiveEstimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.POOL, batch_idx)
