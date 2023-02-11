import pickle
import time
from pathlib import Path
from typing import Any, Union

from lightning.fabric.wrappers import _FabricModule

from src.containers import ActiveFitOutput, FitOutput, RoundOutput
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC


class Callback:
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def on_fit_start(self, model: _FabricModule, output: FitOutput) -> None:
        """Called when fit begins."""

    def on_fit_end(self, model: _FabricModule, output: FitOutput) -> None:
        """Called when fit ends."""

    def on_train_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        """Called when the train epoch begins."""

    def on_train_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """

    def on_validation_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        """Called when the val epoch begins."""

    def on_validation_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        """Called when the val epoch ends."""

    def on_test_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        """Called when the test epoch begins."""

    def on_test_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        """Called when the test epoch ends."""

    def on_train_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the train batch begins."""

    def on_train_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the train batch ends.

        Note:
            The value ``outputs["loss"]`` here will be the normalized value w.r.t ``accumulate_grad_batches`` of the
            loss returned from ``training_step``.
        """

    def on_validation_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the validation batch begins."""

    def on_validation_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the validation batch ends."""

    def on_test_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the test batch begins."""

    def on_test_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        """Called when the test batch ends."""


class ActiveLearningCallbackMixin:
    def on_active_fit_start(self, output: ActiveFitOutput) -> None:
        """Called when the active fit begins."""

    def on_active_fit_end(self, output: ActiveFitOutput) -> None:
        """Called when the active fit ends."""

    def on_round_start(self, output: RoundOutput) -> None:
        """Called when the round starts."""

    def on_round_end(self, output: RoundOutput) -> None:
        """Called when the round ends."""


class UncertaintyStrategyMixin:
    def on_pool_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT) -> None:
        """Called when the pool epoch begins."""

    def on_pool_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT) -> None:
        """Called when the train epoch ends.

        To access all batch outputs at the end of the epoch, either:

        1. Implement `training_epoch_end` in the `LightningModule` and access outputs via the module OR
        2. Cache data across train batch hooks inside the callback implementation to post-process in this hook.
        """


class Timer(Callback, ActiveLearningCallbackMixin, UncertaintyStrategyMixin):
    def on_fit_start(self, model: _FabricModule, output: FitOutput) -> None:
        self.fit_start = time.perf_counter()

    def on_fit_end(self, model: _FabricModule, output: FitOutput) -> None:
        self.fit_end = time.perf_counter()
        output.time = self.fit_end - self.fit_start

    def on_train_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        self.train_epoch_start = time.perf_counter()

    def on_train_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        self.train_epoch_end = time.perf_counter()
        output.time = self.train_epoch_end - self.train_epoch_start

    def on_validation_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT) -> None:
        self.validation_epoch_start = time.perf_counter()

    def on_validation_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC) -> None:
        self.validation_epoch_end = time.perf_counter()
        output.time = self.validation_epoch_end - self.validation_epoch_start

    def on_test_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        self.test_epoch_start = time.perf_counter()

    def on_test_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs) -> None:
        self.test_epoch_end = time.perf_counter()
        output.time = self.test_epoch_end - self.test_epoch_start

    def on_train_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        self.train_batch_start = time.perf_counter()

    def on_train_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.train_batch_end = time.perf_counter()
        output.time = self.train_batch_end - self.train_batch_start

    def on_validation_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        self.validation_batch_start = time.perf_counter()

    def on_validation_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.validation_batch_end = time.perf_counter()
        output.time = self.validation_batch_end - self.validation_batch_start

    def on_test_batch_start(self, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        self.test_batch_start = time.perf_counter()

    def on_test_batch_end(self, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.test_batch_end = time.perf_counter()
        output.time = self.test_batch_end - self.test_batch_start

    def on_active_fit_start(self, output: ActiveFitOutput) -> None:
        self.active_fit_start = time.perf_counter()

    def on_active_fit_end(self, output: ActiveFitOutput) -> None:
        self.active_fit_end = time.perf_counter()
        output.time = self.active_fit_end - self.active_fit_start

    def on_round_start(self, output: ActiveFitOutput) -> None:
        self.round_start = time.perf_counter()

    def on_round_end(self, output: ActiveFitOutput) -> None:
        self.round_end = time.perf_counter()
        output.time = self.round_end - self.round_start

    def on_train_epoch_start(self, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs) -> None:
        self.pool_epoch_start = time.perf_counter()

    def on_pool_epoch_end(self, model: _FabricModule, output: EPOCH_OUTPUT) -> None:
        self.pool_epoch_end = time.perf_counter()
        output.time = self.pool_epoch_end - self.pool_epoch_start


class SaveRoundOutput(Callback, ActiveLearningCallbackMixin):
    def __init__(self, save_dir: Union[str, Path]) -> None:
        super().__init__()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_round_end(self, output: RoundOutput) -> None:
        with (self.save_dir / f"round_{output.round_idx}.pkl").open("wb") as fl:
            pickle.dump(output, fl)
