from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import srsly
from lightning.fabric.wrappers import _FabricModule

from src.energizer.callbacks.base import CallbackWithMonitor
from src.energizer.enums import Interval, RunningStage
from src.energizer.estimator import Estimator
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import make_dict_json_serializable


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

        self.dirpath = Path("./.early_stopping.jsonl")

    def check_stopping_criteria(self, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT]) -> Tuple[bool, str]:
        current = self._get_monitor(output)

        should_stop = False
        reason = None
        if not np.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.6f}."
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
                    f"Best score: {self.best_score:.6f}."
                )

        return should_stop, reason

    def check(
        self, estimator: Estimator, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT], stage: RunningStage, interval: Interval
    ) -> None:
        if (self.stage == stage and self.interval == interval) and estimator.progress_tracker.is_training:
            should_stop, reason = self.check_stopping_criteria(output)
            if should_stop:
                estimator.progress_tracker.set_stop_training(True)
                self._msg = f"stage={stage}_interval={interval}_reason={reason}"
                out = {
                    "reason": reason,
                    "stage": stage,
                    "interval": interval,
                    "step": estimator.progress_tracker.get_epoch_num()
                    if interval == Interval.EPOCH
                    else estimator.progress_tracker.get_batch_num(),
                }
                srsly.write_jsonl(self.dirpath, [make_dict_json_serializable(out)], append=True, append_new_line=False)

    def reset(self) -> None:
        self.wait_count = 0
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_fit_start(self, *args, **kwargs) -> None:
        self.reset()

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
