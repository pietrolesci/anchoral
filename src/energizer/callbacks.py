import time
from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule

from src.energizer.enums import Interval, RunningStage, OutputKeys
from src.energizer.estimator import Estimator, FitEpochOutput
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import move_to_cpu


class Callback:
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    """
    Fit
    """

    def on_fit_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: List[FitEpochOutput]) -> None:
        ...

    """
    Epoch
    """

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_validation_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_test_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    """
    Batch
    """

    def on_train_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_validation_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_test_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_test_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...


class Timer(Callback):
    def epoch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_start_time", time.perf_counter())

    def epoch_end(self, estimator: Estimator, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_epoch_end_time") - getattr(self, f"{stage}_epoch_start_time")
        estimator.log(f"timer/{stage}_epoch_time", runtime, step=estimator.progress_tracker.get_epoch_num())

    def batch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_batch_start_time", time.perf_counter())

    def batch_end(self, estimator: Estimator, stage: RunningStage) -> None:
        setattr(self, f"{stage}_batch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_batch_end_time") - getattr(self, f"{stage}_batch_start_time")
        estimator.log(f"timer/{stage}_batch_time", runtime, step=estimator.progress_tracker.get_batch_num())

    def on_fit_start(self, *args, **kwargs) -> None:
        self.fit_start = time.perf_counter()

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.fit_end = time.perf_counter()
        estimator.log("timer/fit_time", self.fit_end - self.fit_start, step=0)

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
        self.batch_end(estimator, RunningStage.TRAIN)

    def on_validation_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.VALIDATION)

    def on_test_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.TEST)


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

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.prof.start()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        self.prof.step()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
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
        verbose: bool = True,
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
        self.verbose = verbose

        self.wait_count = 0
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    def check_stopping_criteria(self, output: BATCH_OUTPUT):
        if self.monitor not in output:
            current = output[OutputKeys.METRICS][self.monitor]
        else:
            current = output[self.monitor]
        
        current = move_to_cpu(current)

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

    def check(self, estimator: Estimator, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT], stage: RunningStage, interval: Interval) -> None:
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
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.check(estimator, output, RunningStage.TRAIN, Interval.EPOCH)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.check(estimator, output, RunningStage.VALIDATION, Interval.EPOCH)

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        if self._msg is not None and self.verbose:
            estimator.fabric.print(f"\nEarly Stopping {self._msg}\n")
