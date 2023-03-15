from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.enums import RunningStage


@dataclass
class Tracker:
    min: Optional[int] = None
    max: Optional[int] = None
    total: int = 0
    current: int = 0
    progress_bar: Optional[tqdm] = None

    def max_reached(self) -> bool:
        """If a max is not set, it will never stop."""
        return self.max is not None and self.current >= self.max

    def reset(self) -> None:
        self.current = 0
        # self.total = 0
        # self.min = None
        # self.max = None
        if self.progress_bar is not None:
            self.progress_bar.reset(total=self.max)
            self.progress_bar.set_postfix_str()

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

    stop_training: bool = False
    has_validation: bool = False
    validate_every_n_batches: Optional[List[int]] = None
    validate_every_n_epochs: Optional[List[int]] = None

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
        self.validate_every_n_batches = None
        self.validate_every_n_epochs = None
        self.epoch_tracker.reset()
        self.step_tracker.reset()
        self.train_tracker.reset()
        self.validation_tracker.reset()

    def update_from_hparams(
        self,
        num_train_batches: int,
        max_epochs: Optional[int],
        min_epochs: Optional[int],
        max_steps: Optional[int],
        min_steps: Optional[int],
        has_validation: Optional[bool],
    ) -> None:
        """Set epoch and step trackers."""
        assert max_epochs is not None or min_steps is not None, "`max_epochs` or `min_steps` must be passed."

        max_epochs_for_num_steps = 0
        if min_steps is not None:
            max_epochs_for_num_steps = int(np.ceil(min_steps / num_train_batches))

        max_epochs = max_epochs or -1
        if max_epochs < max_epochs_for_num_steps:
            # if we do not have enough batches across epochs, adjust epoch number
            max_epochs = max_epochs_for_num_steps
        else:
            # if we have enough batches to cover the num steps, do nothing
            min_steps = None

        self.epoch_tracker.max = max_epochs
        self.epoch_tracker.min = min_epochs
        self.step_tracker.max = max_steps
        self.step_tracker.min = min_steps
        self.has_validation = has_validation

    def initialize_epoch_progress(self, loader: DataLoader, stage: RunningStage, **kwargs) -> None:
        max_batches = min(len(loader), kwargs.get(f"limit_{stage}_batches") or float("Inf"))
        getattr(self, f"{stage}_tracker").max = max_batches
        getattr(self, f"{stage}_tracker").reset()  # <- reset current counts and progress bar line

        if stage == RunningStage.TRAIN:
            if self.train_tracker.progress_bar is not None:
                self.train_tracker.progress_bar.set_description(f"Epoch {self.epoch_tracker.current}")

            validation_frequency = kwargs.get("validation_frequency", None)
            if validation_frequency:
                if validation_frequency < 1:
                    self.validate_every_n_batches = int(max_batches * validation_frequency) or 1
                elif validation_frequency > 1:
                    self.validate_every_n_epochs = int(validation_frequency)

        else:
            if self.train_tracker.progress_bar is not None:
                # self.train_tracker.progress_bar.set_postfix_str("Validating")
                self.train_tracker.progress_bar.refresh()


@dataclass
class ProgressTracker:
    """Tracks epochs and stages."""

    fit_tracker: FitTracker = FitTracker()
    validation_tracker: StageTracker = None
    test_tracker: StageTracker = None

    is_fitting: bool = False
    log_interval: int = 1

    current_stage: RunningStage = None

    @property
    def global_step(self) -> int:
        return self.fit_tracker.step_tracker.total

    @property
    def global_batch(self) -> int:
        "automatically infers the active stage"
        return self._get_stage_tracker().total

    @property
    def global_epoch(self) -> int:
        if self.is_fitting:
            return self.fit_tracker.epoch_tracker.total
        return 0

    """
    Status
    """

    def is_fit_done(self) -> bool:
        return self.fit_tracker.epoch_tracker.max_reached() or self.fit_tracker.stop_training

    def is_epoch_done(self) -> bool:
        return self._get_stage_tracker().max_reached() or (
            self.current_stage == RunningStage.TRAIN and self.is_fitting and self.fit_tracker.stop_training
        )

    def should_log(self) -> bool:
        return (self.global_batch + 1) % self.log_interval == 0

    def should_validate(self) -> bool:
        # no validation
        if not self.fit_tracker.has_validation:
            return False

        # we are fitting and training has finished
        if self.current_stage == RunningStage.TRAIN and self.is_epoch_done():
            if self.fit_tracker.validate_every_n_epochs is not None:
                return self.global_step > 0 and (self.global_epoch + 1) % self.fit_tracker.validate_every_n_epochs == 0
            return True

        # we can validate mid-epoch
        if (
            self.fit_tracker.validate_every_n_batches is not None
            and self.fit_tracker.train_tracker.current > 0
            and self.fit_tracker.train_tracker.current % self.fit_tracker.validate_every_n_batches == 0
        ):
            return True

        return False

    def set_stop_training(self, value: bool) -> None:
        self.fit_tracker.stop_training = value

    """
    Operations
    """

    def initialize_fit_progress(self, progress_bar: bool, log_interval: int, **kwargs) -> None:
        """Set states and progress bars."""
        self.is_fitting = True

        self.fit_tracker.reset()  # <- reset current counts and progress bar line
        self.fit_tracker.update_from_hparams(**kwargs)

        if progress_bar:
            self.fit_tracker.make_progress_bars()
        if log_interval is not None:
            self.log_interval = log_interval

    def increment_fit_progress(self) -> None:
        self.fit_tracker.epoch_tracker.increment()

    def finalize_fit_progress(self) -> None:
        self.fit_tracker.close_progress_bars()
        self.is_fitting = False

    def initialize_epoch_progress(self, loader: DataLoader, stage: RunningStage, **kwargs) -> None:
        """Resets the `current` counters in the tracker and optionally their progress bars."""
        self.current_stage = stage

        if self.is_fitting:
            self.fit_tracker.initialize_epoch_progress(loader, stage, **kwargs)
            return

        name = f"{stage}_tracker"
        if getattr(self, name) is None:
            setattr(self, name, StageTracker(stage=stage))

        tracker = getattr(self, name)
        if kwargs.get("progress_bar"):
            tracker.make_progress_bar()
        tracker.max = min(len(loader), kwargs.get("limit_batches") or float("Inf"))
        tracker.reset()  # <- reset current counts and progress bar line

        if kwargs.get("log_interval") is not None:
            self.log_interval = kwargs.get("log_interval")

    def increment_epoch_progress(self) -> None:
        self._get_stage_tracker().increment()

    def finalize_epoch_progress(self) -> None:
        tracker = self._get_stage_tracker()
        if self.is_fitting:
            tracker.terminate_progress_bar()
        else:
            tracker.close_progress_bar()

    def increment_step_progress(self) -> None:
        self.fit_tracker.step_tracker.increment()

    def _get_stage_tracker(self) -> StageTracker:
        tracker = self.fit_tracker if self.is_fitting else self
        return getattr(tracker, f"{self.current_stage}_tracker")
