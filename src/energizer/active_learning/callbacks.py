from typing import Any

from lightning.fabric.wrappers import _FabricModule

from src.energizer.active_learning.base import ActiveEstimator
from src.energizer.active_learning.data import ActiveDataModule
from src.energizer.containers import ActiveFitOutput, QueryOutput, RoundOutput
from src.energizer.types import EPOCH_OUTPUT, METRIC, POOL_BATCH_OUTPUT


class ActiveLearningCallbackMixin:
    def on_active_fit_start(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        ...

    def on_active_fit_end(
        self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: ActiveFitOutput
    ) -> None:
        ...

    def on_round_start(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        ...

    def on_round_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        ...

    def on_query_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        ...

    def on_query_end(self, estimator: ActiveEstimator, model: _FabricModule, output: QueryOutput) -> None:
        ...

    def on_label_start(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        ...

    def on_label_end(self, estimator: ActiveEstimator, datamodule: ActiveDataModule, output: RoundOutput) -> None:
        ...

    def on_pool_batch_start(self, estimator: ActiveEstimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        """Called when the pool batch begins."""

    def on_pool_batch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: POOL_BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Called when the pool batch ends."""

    def on_pool_epoch_start(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, **kwargs
    ) -> None:
        """Called when the pool epoch begins."""

    def on_pool_epoch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC, **kwargs
    ) -> None:
        """Called when the pool epoch ends."""
