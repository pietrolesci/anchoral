from lightning.fabric.wrappers import _FabricModule

from src.active_learning.base import ActiveEstimator
from src.active_learning.data import ActiveDataModule
from src.callbacks import Callback
from src.containers import ActiveFitOutput, QueryOutput, RoundOutput


class ActiveLearningCallback(Callback):
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
