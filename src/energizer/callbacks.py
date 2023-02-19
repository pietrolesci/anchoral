import time
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule

from src.energizer.containers import FitOutput
from src.energizer.enums import Interval, RunningStage
from src.energizer.estimator import Estimator
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import move_to_cpu


class Callback:
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def on_fit_start(self, estimator: Estimator, model: _FabricModule, output: FitOutput) -> None:
        """Called when fit begins."""

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: FitOutput) -> None:
        """Called when fit ends."""

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_validation_epoch_start(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs
    ) -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        """Called when the test epoch ends."""

    def on_train_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``train_step``.
        """

    def on_validation_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the test batch ends."""


class Timer(Callback):
    def epoch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_start_time", time.perf_counter())

    def epoch_end(self, estimator: Estimator, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_epoch_end_time") - getattr(self, f"{stage}_epoch_start_time")
        estimator.fabric.log(f"timer/{stage}_epoch_time", runtime, step=estimator.progress_tracker.get_epoch_num(stage))

    def batch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_batch_start_time", time.perf_counter())

    def batch_end(self, estimator: Estimator, stage: RunningStage, batch_idx: int) -> None:
        setattr(self, f"{stage}_batch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_batch_end_time") - getattr(self, f"{stage}_batch_start_time")
        estimator.fabric.log(f"timer/{stage}_batch_time", runtime, step=estimator.progress_tracker.get_batch_num(stage))

    def on_fit_start(self, *args, **kwargs) -> None:
        self.fit_start = time.perf_counter()

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.fit_end = time.perf_counter()
        estimator.fabric.log("timer/fit_time", self.fit_end - self.fit_start, step=0)

    """
    Epoch start
    """

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.TRAIN)

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.VALIDATION)

    def on_test_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.TEST)

    """
    Epoch end
    """

    def on_train_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.TRAIN)

    def on_validation_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.VALIDATION)

    def on_test_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.TEST)

    """
    Batch start
    """

    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.TRAIN)

    def on_validation_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.VALIDATION)

    def on_test_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.TEST)

    """
    Batch end
    """

    def on_train_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.TRAIN, batch_idx)

    def on_validation_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.VALIDATION, batch_idx)

    def on_test_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.TEST, batch_idx)


class PytorchTensorboardProfiler(Callback):
    def __init__(
        self,
        dirpath: Union[str, Path],
        wait: int = 1,
        warmup: int = 1,
        active: int = 1,
        repeat: int = 2,
        **kwargs,
    ) -> None:
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(dirpath)),
            **kwargs,
        )

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        self.prof.start()

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.prof.step()

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        self.prof.stop()


class EarlyStopping(Callback):
    mode_dict = {"min": np.less, "max": np.greater}
    order_dict = {"min": "<", "max": ">"}
    _msg: Optional[str] = None

    def __init__(
        self,
        monitor: str,
        stage: RunningStage,
        interval: Interval = Interval.EPOCH,
        mode: str = "min",
        min_delta=0.00,
        patience=3,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.stage = stage
        self.interval = interval
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold

        self.wait_count = 0
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def check_stopping_criteria(self, output: BATCH_OUTPUT):
        current = move_to_cpu(output[self.monitor])

        should_stop = False
        reason = None
        if not np.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.3f}."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            self.best_score = current
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric `{self.monitor}` did not improve in the last {self.wait_count} {self.interval}"
                    f"{'es' if self.interval == Interval.BATCH else 's'}. "
                    f"Best score: {self.best_score:.3f}."
                )

        return should_stop, reason

    def check(self, estimator: Estimator, output: BATCH_OUTPUT, stage: RunningStage, interval: Interval) -> None:
        if (self.stage == stage and self.interval == interval) and estimator.progress_tracker.is_training:
            should_stop, reason = self.check_stopping_criteria(output)
            if should_stop:
                estimator.progress_tracker.set_stop_training(True)
                self._msg = f"({stage}) -- {reason}"

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.check(estimator, output, RunningStage.TRAIN, Interval.BATCH)

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.check(estimator, output, RunningStage.VALIDATION, Interval.BATCH)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        self.check(estimator, output, RunningStage.TRAIN, Interval.EPOCH)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        self.check(estimator, output, RunningStage.VALIDATION, Interval.EPOCH)

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: FitOutput) -> None:
        if self._msg is not None:
            estimator.fabric.print(f"\nEarly Stopping {self._msg}\n")
