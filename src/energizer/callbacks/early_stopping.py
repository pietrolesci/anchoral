from typing import Any, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule

from src.energizer.callbacks.base import CallbackWithMonitor
from src.energizer.enums import Interval, OutputKeys, RunningStage
from src.energizer.estimator import Estimator
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import move_to_cpu


class EarlyStopping(CallbackWithMonitor):
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

    def check_stopping_criteria(self, output: BATCH_OUTPUT):
        current = self._get_monitor(output)

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

    def _get_monitor(self, output: EPOCH_OUTPUT) -> float:
        if self.monitor in output:
            monitor = output[self.monitor]
        elif self.monitor in output[OutputKeys.METRICS]:
            monitor = output[OutputKeys.METRICS][self.monitor]
        else:
            raise ValueError(f"`{self.monitor}` is not logged.")

        return move_to_cpu(monitor)

    def check(
        self, estimator: Estimator, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT], stage: RunningStage, interval: Interval
    ) -> None:
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
