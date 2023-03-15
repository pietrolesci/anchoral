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
        """If a max is not set, it will never stop."""
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
        self.terminate_progress_bar()
        if self.progress_bar is not None:
            self.progress_bar.clear()
            self.progress_bar.close()

    def terminate_progress_bar(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str("Done!", refresh=True)
            self.progress_bar.refresh()


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
    epoch_tracker: EpochTracker = EpochTracker()
    step_tracker: Tracker = Tracker()
    train_tracker: StageTracker = StageTracker(stage=RunningStage.TRAIN)
    validation_tracker: StageTracker = StageTracker(stage=RunningStage.VALIDATION)

    validation_interval: Optional[List[int]] = None
    stop_training: bool = False
    has_validation: bool = False

    def make_progress_bars(self) -> None:
        self.epoch_tracker.make_progress_bar()
        self.train_tracker.make_progress_bar()
        if self.has_validation:
            self.validation_tracker.make_progress_bar()

    def close_progress_bars(self) -> None:
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()

    def terminate_progress_bars(self) -> None:
        self.epoch_tracker.terminate_progress_bar()
        self.train_tracker.terminate_progress_bar()
        self.validation_tracker.terminate_progress_bar()

    def reset(self) -> None:
        self.stop_training = False
        self.has_validation = False
        self.epoch_tracker.reset()
        self.step_tracker.reset()
        self.train_tracker.reset()
        self.validation_tracker.reset()

    def update_from_hparams(
        self,
        max_epochs: int,
        min_steps: int,
        max_train_batches: int,
        max_validation_batches: int,
        validation_interval: int,
        has_validation: bool,
    ) -> None:
        self.epoch_tracker.max = max_epochs
        self.step_tracker.max = min_steps
        self.train_tracker.max = max_train_batches
        self.validation_tracker.max = max_validation_batches
        self.validation_interval = validation_interval
        self.has_validation = has_validation


@dataclass
class ProgressTracker:
    fit_tracker: FitTracker = FitTracker()
    validation_tracker: StageTracker = StageTracker(stage=RunningStage.VALIDATION)
    test_tracker: StageTracker = StageTracker(stage=RunningStage.TEST)

    is_fitting: bool = False
    log_interval: int = 1

    current_stage: RunningStage = None

    """
    Status
    """

    @property
    def global_step(self) -> int:
        return self.fit_tracker.step_tracker.total

    @property
    def global_batch(self) -> int:
        return self._get_stage_tracker().total
    
    @property
    def global_epoch(self) -> int:
        if self.is_fitting:
            return self.fit_tracker.epoch_tracker.total
        return 0
    
    @property
    def safe_global_epoch(self) -> int:
        if self.is_fitting and self.current_stage == RunningStage.VALIDATION and self.fit_tracker.validation_interval is not None and self.fit_tracker.validation_interval > 1:
            return self.global_batch
        return self.global_epoch

    def is_fit_done(self) -> bool:
        return self.fit_tracker.epoch_tracker.max_reached() or self.fit_tracker.stop_training

    def is_epoch_done(self) -> bool:
        return self._get_stage_tracker().max_reached() or (
            self.current_stage == RunningStage.TRAIN and self.is_fitting and self.fit_tracker.stop_training
        )

    def should_log(self) -> None:
        return (self.global_batch + 1) % self.log_interval == 0

    def should_validate(self) -> bool:
        if self.fit_tracker.validation_tracker.max is None:
            return False
        if self.is_epoch_done():
            return True
        if self.fit_tracker.validation_interval is not None:
            return self.fit_tracker.train_tracker.current in self.fit_tracker.validation_interval

    def initialize_fit_progress(
        self,
        *args,
        **kwargs,
    ) -> None:
        self.is_fitting = True

        self.fit_tracker.reset()  # <- reset current counts and progress bar line
        self.fit_tracker.update_from_hparams(**self._solve_hparams(*args, **kwargs), has_validation=kwargs.get("has_validation"))
        
        if kwargs.get("progress_bar"):
            self.fit_tracker.make_progress_bars()

        self.log_interval = kwargs.get("log_interval", 1)
    
    def increment_fit_progress(self) -> None:
        self.fit_tracker.epoch_tracker.increment()
        pbar = self.fit_tracker.train_tracker.progress_bar
        if pbar is not None:
            pbar.set_description(f"Epoch {self.fit_tracker.epoch_tracker.current}")

    def finalize_fit_progress(self) -> None:
        self.fit_tracker.close_progress_bars()
        self.is_fitting = False

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        self.is_fitting = False
        self.log_interval = kwargs.get("log_interval", 1)

        max_batches = self._solve_num_batches(loader, kwargs.get("limit_batches", None))
        getattr(self, f"{stage}_tracker").max = max_batches
        getattr(self, f"{stage}_tracker").reset()  # <- reset current counts and progress bar line
        if kwargs.get("progress_bar", True):
            getattr(self, f"{stage}_tracker").make_progress_bar()

    def initialize_epoch_progress(self, stage: RunningStage, loader: Optional[DataLoader] = None, **kwargs) -> None:
        """Resets the `current` counters in the tracker and optionally their progress bars."""
        self.current_stage = stage
        if self.is_fitting:
            self._get_stage_tracker().reset()  # <- reset current counts and progress bar line
            return

        name = f"{stage}_tracker"
        if kwargs.get("progress_bar") and getattr(self, name).progress_bar is None:
            getattr(self, name).make_progress_bar()
        getattr(self, name).max = min(len(loader), kwargs.get("limit_batches") or float("Inf"))
        getattr(self, name).reset()  # <- reset current counts and progress bar line

        if kwargs.get("log_interval") is not None:
            self.log_interval = kwargs.get("log_interval")

    def increment_epoch_progress(self) -> None:
        self._get_stage_tracker().increment()

    def continue_epoch_progress(self, stage: RunningStage) -> None:
        """Resets the `current` counters in the tracker and optionally their progress bars."""
        self.current_stage = stage

    def finalize_epoch_progress(self) -> None:
        tracker = self._get_stage_tracker()
        if self.is_fitting:
            tracker.terminate_progress_bar()
        else:
            tracker.close_progress_bar()

    def set_stop_training(self, value: bool) -> None:
        self.fit_tracker.stop_training = value

    def increment_step_progress(self) -> None:
        self.fit_tracker.step_tracker.increment()

    def _get_stage_tracker(self) -> StageTracker:
        tracker = self.fit_tracker if self.is_fitting else self
        return getattr(tracker, f"{self.current_stage}_tracker")

    def _solve_hparams(
        self,
        max_epochs: Optional[int],
        min_steps: Optional[int],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        **kwargs,
    ) -> Dict:
        assert max_epochs is not None or min_steps is not None, "`max_epochs` or `min_steps` must be passed."

        # train: limit batches
        max_train_batches = self._solve_num_batches(train_loader, kwargs.get("limit_train_batches", None))

        # train: epochs and steps
        if max_epochs is None:
            max_epochs = np.ceil(min_steps / max_train_batches)
        if min_steps is not None:
            max_epochs_for_num_steps = int(np.ceil(min_steps / max_train_batches))
            if max_epochs < max_epochs_for_num_steps:
                # if we do not have enough batches across epochs, adjust epoch number
                max_epochs = max_epochs_for_num_steps
            else:
                # if we have enough batches to cover the num steps, do nothing
                min_steps = None

        # validation: limit batches and validation interval
        max_validation_batches = self._solve_num_batches(
            validation_loader, kwargs.get("limit_validation_batches", None)
        )
        validation_interval = kwargs.get("validation_interval", True)
        if (
            max_validation_batches is not None
            and validation_interval is not None
            and max_train_batches > validation_interval
        ):
            validation_interval = np.linspace(
                max_train_batches / validation_interval, max_train_batches, validation_interval, dtype=int
            ).tolist()[:-1]
        else:
            validation_interval = None

        return {
            "max_epochs": max_epochs,
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
