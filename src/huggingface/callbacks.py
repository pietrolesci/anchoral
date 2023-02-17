import time
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.active_learning.base import ActiveEstimator
from src.active_learning.callbacks import ActiveLearningCallbackMixin
from src.active_learning.data import ActiveDataModule
from src.callbacks import Callback, Timer
from src.containers import ActiveCounter, ActiveFitOutput, QueryOutput, RoundOutput
from src.enums import OutputKeys, RunningStage, SpecialKeys
from src.estimator import Estimator
from src.types import EPOCH_OUTPUT, METRIC
from src.utilities import move_to_cpu


class SaveOutputs(ActiveLearningCallbackMixin, Callback):
    def __init__(self, dirpath: Union[str, Path], instance_level: bool, batch_level: bool, epoch_level: bool) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.instance_level = instance_level
        self.batch_level = batch_level
        self.epoch_level = epoch_level

    def on_epoch_end(
        self,
        estimator: Estimator,
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
        stage: RunningStage,
        **kwargs,
    ) -> None:
        # output directory setup
        suffix = f"_epoch_{estimator.counter.get_epoch_step(stage)}"
        path = self.dirpath / f"{stage}"
        path.mkdir(exist_ok=True, parents=True)

        # list of dicts to dict of lists
        data = {k: [dic[k] for dic in output] for k in output[0]}

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
        if self.epoch_level:
            total_metrics = move_to_cpu(metrics.compute())
            epoch_level_output = {
                OutputKeys.METRICS: total_metrics,
                OutputKeys.LOSS: round(np.mean(data[OutputKeys.LOSS]), 6),
            }
            srsly.write_json(path / f"epoch_level{suffix}.json", epoch_level_output)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.TRAIN, **kwargs)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.VALIDATION, **kwargs)

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        return self.on_epoch_end(estimator, model, output, metrics, RunningStage.TEST, **kwargs)

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        datamodule.save_labelled_dataset(self.dirpath)


class Timer(ActiveLearningCallbackMixin, Timer):
    def _batch_step(self, counter: ActiveCounter, stage: RunningStage, batch_idx: int) -> int:
        return getattr(counter, f"total_{stage}_batches") + getattr(counter, f"num_{stage}_batches")

    def _epoch_step(self, counter: ActiveCounter, stage: RunningStage) -> int:
        if stage == RunningStage.TEST:
            return counter.num_rounds
        return getattr(counter, "total_epochs") + getattr(counter, "num_epochs")

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
        estimator.fabric.log("timer/round_time", self.round_end - self.round_start, step=estimator.counter.num_rounds)

    def on_query_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        self.query_start = time.perf_counter()

    def on_query_end(self, estimator: ActiveEstimator, model: _FabricModule, output: QueryOutput) -> None:
        self.query_end = time.perf_counter()
        estimator.fabric.log("timer/query_time", self.query_end - self.query_start, step=estimator.counter.num_rounds)

    def on_label_start(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.label_start = time.perf_counter()

    def on_label_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        self.label_end = time.perf_counter()
        estimator.fabric.log("timer/label_time", self.label_end - self.label_start, step=estimator.counter.num_rounds)

    def on_pool_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.POOL)

    def on_pool_epoch_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.POOL)

    def on_pool_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.POOL)

    def on_pool_batch_end(self, estimator: ActiveEstimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.POOL, batch_idx)
