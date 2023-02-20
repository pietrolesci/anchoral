import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.enums import RunningStage


@dataclass
class Tracker:
    max: Optional[int] = None
    total: int = 0
    current: int = 0
    progress_bar: Optional[tqdm] = None
    show_progress_bar: bool = False
    leave: Optional[bool] = None

    def reset_current(self) -> None:
        self.current = 0

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def max_reached(self) -> bool:
        cond = self.max is not None and self.current >= self.max
        if cond and self.progress_bar is not None:
            self.progress_bar.close()
        return cond

    def make_progress_bar(self) -> None:
        pass

    def initialize(self) -> None:
        self.reset_current()
        if self.max is not None and self.show_progress_bar:
            self.make_progress_bar()


@dataclass
class EpochTracker(Tracker):
    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed epochs",
            dynamic_ncols=True,
            leave=self.leave,
            file=sys.stderr,
        )


@dataclass
class StageTracker(Tracker):
    stage: str = None

    def make_progress_bar(self) -> None:
        leave = self.stage == RunningStage.TEST if self.leave is None else self.leave
        desc = f"Epoch {self.current}".strip() if self.stage == RunningStage.TRAIN else f"{self.stage.title()}"
        self.progress_bar = tqdm(total=self.max, desc=desc, dynamic_ncols=True, leave=leave, file=sys.stderr)


@dataclass
class FitTracker:
    epoch_tracker: EpochTracker
    step_tracker: Tracker
    train_tracker: StageTracker
    validation_tracker: StageTracker
    validation_interval: Optional[List[int]] = None
    _stop_training: bool = False

    @property
    def stop_training(self) -> bool:
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        self._stop_training = value

    def initialize(self) -> None:
        self._stop_training = False

    @classmethod
    def from_hparams(
        cls,
        num_epochs: int,
        min_steps: int,
        max_train_batches: int,
        max_validation_batches: int,
        validation_interval: int,
        progress_bar: bool,
    ) -> Tracker:
        return cls(
            epoch_tracker=EpochTracker(max=num_epochs, show_progress_bar=progress_bar),
            step_tracker=Tracker(max=min_steps, show_progress_bar=False),
            train_tracker=StageTracker(max=max_train_batches, stage=RunningStage.TRAIN, show_progress_bar=progress_bar),
            validation_tracker=StageTracker(
                max=max_validation_batches, stage=RunningStage.VALIDATION, show_progress_bar=progress_bar
            ),
            validation_interval=validation_interval,
        )

    def update_from_hparams(
        self,
        num_epochs: int,
        min_steps: int,
        max_train_batches: int,
        max_validation_batches: int,
        validation_interval: int,
        progress_bar: bool,
    ) -> Tracker:
        self.epoch_tracker.max = num_epochs
        self.epoch_tracker.progress_bar = progress_bar
        self.step_tracker.max = min_steps
        self.train_tracker.max = max_train_batches
        self.train_tracker.progress_bar = progress_bar
        self.validation_tracker.max = max_validation_batches
        self.validation_tracker.progress_bar = progress_bar
        self.validation_interval = validation_interval


@dataclass
class ProgressTracker:
    fit_tracker: FitTracker = None
    validation_tracker: StageTracker = None
    test_tracker: StageTracker = None

    is_training: bool = False
    log_interval: int = 1

    """
    Status
    """

    def is_epoch_progress_done(self) -> bool:
        return self.fit_tracker.epoch_tracker.max_reached() or self.fit_tracker.stop_training

    def is_batch_progress_done(self, stage: RunningStage) -> bool:
        if self.is_training:
            cond = getattr(self.fit_tracker, f"{stage}_tracker").max_reached()
            # if training and min_steps is provided, check whether to stop
            if stage == RunningStage.TRAIN:
                cond = cond or self.fit_tracker.stop_training
            return cond
        return getattr(self, f"{stage}_tracker").max_reached()

    def get_batch_num(self, stage: RunningStage) -> int:
        if self.is_training:
            tracker = getattr(self.fit_tracker, f"{stage}_tracker")
        else:
            tracker = getattr(self, f"{stage}_tracker")
        return tracker.total

    def get_epoch_num(self, stage: RunningStage) -> int:
        if stage == RunningStage.TEST:
            return 0
        return self.fit_tracker.epoch_tracker.total

    def should_log(self, batch_idx: int) -> None:
        return (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0)

    def should_validate(self) -> bool:
        if self.fit_tracker.validation_tracker.max is None:
            return False
        if self.is_batch_progress_done(RunningStage.TRAIN):
            return True
        if self.fit_tracker.validation_interval is not None:
            return self.fit_tracker.train_tracker.current in self.fit_tracker.validation_interval

    """
    Operations
    """

    def set_stop_training(self, value: bool) -> None:
        self.fit_tracker.stop_training = value

    def increment_epoch_progress(self) -> None:
        self.fit_tracker.epoch_tracker.increment()

    def increment_batch_progress(self, stage: RunningStage) -> None:
        if self.is_training:
            getattr(self.fit_tracker, f"{stage}_tracker").increment()
        else:
            getattr(self, f"{stage}_tracker").increment()

    def increment_step_progress(self) -> None:
        self.fit_tracker.step_tracker.increment()

    """
    Initializers
    """

    def initialize_fit_progress(
        self,
        num_epochs: Optional[int],
        min_steps: Optional[int],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        **kwargs,
    ) -> None:
        hparams = self._solve_hparams(num_epochs, min_steps, train_loader, validation_loader, **kwargs)
        self.fit_tracker = FitTracker.from_hparams(**hparams)
        self.fit_tracker.initialize()
        self.is_training = True
        self.log_interval = kwargs.get("log_interval", 1)

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        max_batches = self._solve_num_batches(loader, kwargs.get("limit_batches", None))
        tracker = StageTracker(max=max_batches, stage=stage, show_progress_bar=kwargs.get("progress_bar", True))
        setattr(self, f"{stage}_tracker", tracker)
        self.is_training = False
        self.log_interval = kwargs.get("log_interval", 1)

    def initialize_batch_progress(self, stage: RunningStage) -> None:
        if self.is_training:
            getattr(self.fit_tracker, f"{stage}_tracker").initialize()
        else:
            getattr(self, f"{stage}_tracker").initialize()

    def initialize_epoch_progress(self) -> None:
        self.fit_tracker.epoch_tracker.initialize()
        self.fit_tracker.step_tracker.initialize()

    """
    Helpers
    """

    def _solve_hparams(
        self,
        num_epochs: Optional[int],
        min_steps: Optional[int],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        assert num_epochs is not None or min_steps is not None, "`num_epochs` or `min_steps` must be passed."

        # train: limit batches
        max_train_batches = self._solve_num_batches(train_loader, kwargs.get("limit_train_batches", None))

        # train: epochs and steps
        if num_epochs is None:
            num_epochs = np.ceil(min_steps / max_train_batches)
        if min_steps is not None:
            num_epochs_for_num_steps = int(np.ceil(min_steps / max_train_batches))
            if num_epochs < num_epochs_for_num_steps:
                # if we do not have enough batches across epochs, adjust epoch number
                num_epochs = num_epochs_for_num_steps
            else:
                # if we have enough batches to cover the num steps, do nothing
                min_steps = None

        # validation: limit batches and validation interval
        max_validation_batches = self._solve_num_batches(
            validation_loader, kwargs.get("limit_validation_batches", None)
        )
        validation_interval = kwargs.get("validation_interval", True)
        if max_validation_batches is not None and validation_interval is not None:
            validation_interval = np.linspace(
                max_train_batches / validation_interval, max_train_batches, validation_interval, dtype=int
            ).tolist()[:-1]

        # configure fit_tracker
        progress_bar = kwargs.get("progress_bar", True)

        return {
            "num_epochs": num_epochs,
            "min_steps": min_steps,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "validation_interval": validation_interval,
            "progress_bar": progress_bar,
        }

    def _solve_num_batches(self, loader: Union[DataLoader, None], limit_batches: Optional[int]) -> int:
        if loader is not None:
            max_batches = len(loader)
            if limit_batches is not None:
                max_batches = min(limit_batches, max_batches)
            return max_batches
