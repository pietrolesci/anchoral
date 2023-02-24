import os
from pathlib import Path
from typing import Dict, Optional, Union

from lightning.fabric.wrappers import _FabricModule

from src.energizer.callbacks.base import CallbackWithMonitor
from src.energizer.enums import RunningStage
from src.energizer.estimator import Estimator
from src.energizer.types import EPOCH_OUTPUT, METRIC


class ModelCheckpoint(CallbackWithMonitor):
    _best_k_models: Dict[str, float] = {}

    def __init__(
        self,
        dirpath: Union[Path, str] = ".checkpoints",
        monitor: Optional[str] = None,
        stage: Optional[RunningStage] = None,
        mode: str = "min",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        every_n_epochs: Optional[int] = None,
        save_on_train_epoch_end: Optional[bool] = None,
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.stage = stage
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k

        # frequency
        self.every_n_epochs = every_n_epochs
        self.save_on_train_epoch_end = save_on_train_epoch_end

        # prepare directory
        self.dirpath.mkdir(parents=True, exist_ok=True)

    @property
    def best_model_score(self) -> float:
        return self._best_k_models[self.best_model_path]

    @property
    def best_model_path(self) -> Optional[str]:
        if len(self._best_k_models) > 0:
            return self.optim_op(self._best_k_models, key=self._best_k_models.get)

    @property
    def worst_model_path(self) -> Optional[str]:
        if len(self._best_k_models) > 1:
            return self.reverse_optim_op(self._best_k_models, key=self._best_k_models.get)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        if self.check_should_save(estimator, output, RunningStage.TRAIN):
            self.save_checkpoint(estimator, output, RunningStage.TRAIN)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        if self.check_should_save(estimator, output, RunningStage.VALIDATION):
            self.save_checkpoint(estimator, output, RunningStage.VALIDATION)

    def check_should_save(self, estimator: Estimator, output: EPOCH_OUTPUT, stage: RunningStage) -> bool:
        # always save if explicitly requested
        if stage == RunningStage.TRAIN and (
            (self.save_last and estimator.progress_tracker.is_last_epoch()) or self.save_on_train_epoch_end
        ):
            return True

        # if every_n_epochs is not satisfied, return immediately
        if (
            stage == RunningStage.TRAIN == self.stage
            and self.every_n_epochs is not None
            and estimator.progress_tracker.get_epoch_num() > 0
            and self.every_n_epochs % estimator.progress_tracker.get_epoch_num() != 0
        ):
            return False

        should_save = False
        # save based on monitored value
        if self.monitor is not None and self.stage == stage:
            if self.save_top_k is None:
                should_save = True
            else:
                current = self._get_monitor(output)
                if (
                    # not enough monitored yet
                    (len(self._best_k_models) < self.save_top_k)
                    # check whether it's better than the worst
                    or (
                        self.worst_model_path is not None
                        and self.monitor_op(current, self._best_k_models[self.worst_model_path])
                    )
                ):
                    should_save = True

        return should_save

    def save_checkpoint(self, estimator: Estimator, output: EPOCH_OUTPUT, stage: RunningStage) -> None:
        step = "step" if stage == RunningStage.VALIDATION else "epoch"
        name = f"{stage}_ckpt_{step}={estimator.progress_tracker.get_epoch_num()}"

        # monitoring?
        if self.monitor is not None and self.stage == stage and self.save_top_k:
            current = self._get_monitor(output)
            name += f"_{self.monitor}={current}"

            # remove worst so far
            worst_model_path = self.worst_model_path
            if worst_model_path is not None:
                if len(self._best_k_models) >= self.save_top_k:
                    os.remove(self.dirpath / f"{worst_model_path}.pt")
                    self._best_k_models.pop(worst_model_path)

            # add the new checkpoint
            self._best_k_models[name] = current

        estimator.save_state_dict(self.dirpath, f"{name}.pt")

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        if self.monitor is not None:
            estimator.load_state_dict(self.dirpath, self.best_model_path)
            estimator.fabric.print(
                f"Best model with {self.stage}/{self.monitor}={self.best_model_score} "
                f"loaded from `{self.best_model_path}`"
            )
