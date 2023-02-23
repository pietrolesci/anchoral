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

    def max_reached(self) -> bool:
        return self.max is not None and self.current >= self.max

    def reset_current(self) -> None:
        self.current = 0

    def reset(self) -> None:
        self.reset_current()
        if self.progress_bar is not None:
            self.progress_bar.reset(total=self.max)

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def make_progress_bar(self) -> None:
        pass

    def close_progress_bar(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.refresh()
            self.progress_bar.clear()
            self.progress_bar.close()


@dataclass
class EpochTracker(Tracker):
    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed epochs",
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class StageTracker(Tracker):
    stage: str = None

    def make_progress_bar(self) -> None:
        desc = f"Epoch {self.total}".strip() if self.stage == RunningStage.TRAIN else f"{self.stage.title()}"
        self.progress_bar = tqdm(
            total=self.max,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class FitTracker:
    epoch_tracker: EpochTracker
    step_tracker: Tracker
    train_tracker: StageTracker
    validation_tracker: StageTracker
    validation_interval: Optional[List[int]] = None
    stop_training: bool = False

    def make_progress_bars(self) -> None:
        self.epoch_tracker.make_progress_bar()
        self.train_tracker.make_progress_bar()
        self.validation_tracker.make_progress_bar()

    def close_progress_bars(self) -> None:
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()

    def reset(self) -> None:
        self.stop_training = False
        self.epoch_tracker.reset()
        self.step_tracker.reset()
        self.train_tracker.reset()
        self.validation_tracker.reset()

    @classmethod
    def from_hparams(
        cls,
        num_epochs: int,
        min_steps: int,
        max_train_batches: int,
        max_validation_batches: int,
        validation_interval: int,
    ):
        return cls(
            epoch_tracker=EpochTracker(max=num_epochs),
            step_tracker=Tracker(max=min_steps),
            train_tracker=StageTracker(max=max_train_batches, stage=RunningStage.TRAIN),
            validation_tracker=StageTracker(max=max_validation_batches, stage=RunningStage.VALIDATION),
            validation_interval=validation_interval,
        )

    def update_from_hparams(
        self,
        num_epochs: int,
        min_steps: int,
        max_train_batches: int,
        max_validation_batches: int,
        validation_interval: int,
    ) -> None:
        self.epoch_tracker.max = num_epochs
        self.step_tracker.max = min_steps
        self.train_tracker.max = max_train_batches
        self.validation_tracker.max = max_validation_batches
        self.validation_interval = validation_interval


@dataclass
class ProgressTracker:
    fit_tracker: FitTracker = None
    validation_tracker: StageTracker = None
    test_tracker: StageTracker = None

    is_training: bool = False
    log_interval: int = 1

    current_stage: RunningStage = None

    """
    Status
    """

    def is_fit_done(self) -> bool:
        cond = self.fit_tracker.epoch_tracker.max_reached() or self.fit_tracker.stop_training
        if cond:
            self.fit_tracker.close_progress_bars()
        return cond

    def is_epoch_done(self) -> bool:
        cond = self._get_active_tracker().max_reached()
        if self.current_stage == RunningStage.TRAIN and self.is_training:
            cond = cond or self.fit_tracker.stop_training

        if not self.is_training:
            self._get_active_tracker().close_progress_bar()
        return cond

    def get_batch_num(self) -> int:
        return self._get_active_tracker().total

    def get_epoch_num(self) -> int:
        if self.current_stage == RunningStage.TEST:
            return 0
        elif self.current_stage == RunningStage.VALIDATION and self.is_training:
            return self.get_batch_num()
        return self.fit_tracker.epoch_tracker.total

    def should_log(self) -> None:
        # return batch_idx is None or (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0)
        return self.get_batch_num() % self.log_interval == 0

    def should_validate(self) -> bool:
        if self.fit_tracker.validation_tracker.max is None:
            return False
        if self.is_epoch_done():
            return True
        if self.fit_tracker.validation_interval is not None:
            return self.fit_tracker.train_tracker.current in self.fit_tracker.validation_interval

    """
    Operations
    """

    def set_stop_training(self, value: bool) -> None:
        self.fit_tracker.stop_training = value

    def increment_fit_progress(self) -> None:
        self.fit_tracker.epoch_tracker.increment()
        pbar = self.fit_tracker.train_tracker.progress_bar
        if pbar is not None:
            pbar.set_description(f"Epoch {self.fit_tracker.epoch_tracker.current}")

    def increment_epoch_progress(self) -> None:
        self._get_active_tracker().increment()

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
        self.is_training = True
        self.log_interval = kwargs.get("log_interval", 1)

        hparams = self._solve_hparams(num_epochs, min_steps, train_loader, validation_loader, **kwargs)
        self.fit_tracker = FitTracker.from_hparams(**hparams)
        self.fit_tracker.reset()
        if kwargs.get("progress_bar", True):
            self.fit_tracker.make_progress_bars()

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        self.is_training = False
        self.log_interval = kwargs.get("log_interval", 1)
        self.current_stage = stage

        max_batches = self._solve_num_batches(loader, kwargs.get("limit_batches", None))
        tracker = StageTracker(max=max_batches, stage=stage)
        setattr(self, f"{stage}_tracker", tracker)
        tracker.reset()
        if kwargs.get("progress_bar", True):
            tracker.make_progress_bar()

    def initialize_epoch_progress(self, stage: RunningStage) -> None:
        """Resets the `current` counters in the tracker and optionally their progress bars."""
        self.current_stage = stage
        # if not (stage == RunningStage.VALIDATION and self.is_training):
        self._get_active_tracker().reset()

    def continue_epoch_progress(self, stage: RunningStage) -> None:
        """Resets the `current` counters in the tracker and optionally their progress bars."""
        self.current_stage = stage

    """
    Helpers
    """

    def _get_active_tracker(self) -> StageTracker:
        tracker = self.fit_tracker if self.is_training else self
        return getattr(tracker, f"{self.current_stage}_tracker")

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

        return {
            "num_epochs": num_epochs,
            "min_steps": min_steps,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "validation_interval": validation_interval,
        }

    def _solve_num_batches(self, loader: Union[DataLoader, None], limit_batches: Optional[int]) -> int:
        if loader is not None:
            max_batches = len(loader)
            if limit_batches is not None:
                max_batches = min(limit_batches, max_batches)
            return max_batches
