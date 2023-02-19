import sys
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.enums import RunningStage


@dataclass
class Tracker:
    max: Optional[int] = None
    total: int = 0
    current: int = 0

    def reset_current(self) -> None:
        self.current = 0

    def increment(self) -> None:
        self.current += 1
        self.total += 1

    def max_reached(self) -> bool:
        return self.max is not None and self.current >= self.max


@dataclass
class EvaluationProgressTracker:
    tracker: Tracker

    progress_bar: bool = True
    evaluation_progress_bar: Optional[tqdm] = None

    """
    Status
    """

    def is_batch_progress_done(self) -> bool:
        cond = self.tracker.max_reached()
        if cond and self.evaluation_progress_bar:
            sys.stdout.flush()
            self.evaluation_progress_bar.close()
        return cond

    """
    Initializers
    """

    def initialize_batch_progress(self, stage: RunningStage) -> None:
        self.tracker.reset_current()
        if self.progress_bar and self.tracker.max is not None:
            self.evaluation_progress_bar = tqdm(
                total=self.tracker.max,
                desc=f"{stage.title()}",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
            )

    """
    Operations
    """

    def increment_batch_progress(self) -> None:
        self.tracker.increment()
        if self.progress_bar is not None:
            self.evaluation_progress_bar.update(1)


@dataclass
class FitProgressTracker:
    epochs_tracker: Tracker
    steps_tracker: Tracker
    train_tracker: Tracker
    validation_tracker: Tracker

    epochs_progress_bar: Optional[tqdm] = None
    train_progress_bar: Optional[tqdm] = None
    validation_progress_bar: Optional[tqdm] = None

    progress_bar: bool = True
    validation_interval: Optional[List[int]] = None

    _stop_training: bool = False

    @property
    def stop_training(self) -> bool:
        return self._stop_training

    @stop_training.setter
    def stop_training(self, value: bool) -> None:
        self._stop_training = value

    """
    Status
    """

    def is_epoch_progress_done(self) -> bool:
        cond = self.epochs_tracker.max_reached() or self.stop_training

        # if training and num_steps is provided, check whether to stop
        if self.steps_tracker.max is not None:
            cond = cond or self.steps_tracker.max_reached()

        if cond and self.progress_bar:
            self.epochs_progress_bar.close()
        return cond

    def is_batch_progress_done(self, stage: RunningStage) -> bool:
        cond = getattr(self, f"{stage}_tracker").max_reached()

        # if training and num_steps is provided, check whether to stop
        if stage == RunningStage.TRAIN:
            cond = cond or self.stop_training
            if self.steps_tracker.max is not None:
                cond = cond or self.steps_tracker.max_reached()

        if cond and self.progress_bar:
            getattr(self, f"{stage}_progress_bar").close()
        return cond

    def should_validate(self) -> bool:
        if self.validation_tracker.max is None:
            return False

        if self.is_batch_progress_done(RunningStage.TRAIN):
            return True

        if self.validation_interval is not None:
            return self.train_tracker.current in self.validation_interval

    """
    Operations
    """

    def increment_epoch_progress(self) -> None:
        self.epochs_tracker.increment()
        if self.epochs_progress_bar is not None:
            self.epochs_progress_bar.update(1)

    def increment_batch_progress(self, stage: RunningStage) -> None:
        getattr(self, f"{stage}_tracker").increment()
        batch_progress = getattr(self, f"{stage}_progress_bar", None)
        if batch_progress is not None:
            batch_progress.update(1)

    def increment_steps(self) -> None:
        self.steps_tracker.increment()

    """
    Initializers
    """

    def initialize_epoch_progress(self) -> None:
        if self.progress_bar:
            self.epochs_progress_bar = tqdm(
                total=self.epochs_tracker.max,
                desc="Completed epochs",
                dynamic_ncols=True,
                leave=True,
                file=sys.stdout,
                colour="green",
            )

    def initialize_batch_progress(self, stage: RunningStage) -> None:
        getattr(self, f"{stage}_tracker").reset_current()
        if self.progress_bar and getattr(self, f"{stage}_tracker").max is not None:
            desc = f"Epoch {self.epochs_tracker.current}".strip() if stage == RunningStage.TRAIN else f"{stage.title()}"
            pbar = tqdm(
                total=getattr(self, f"{stage}_tracker").max, desc=desc, dynamic_ncols=True, leave=False, file=sys.stdout
            )
            setattr(self, f"{stage}_progress_bar", pbar)


@dataclass
class ProgressTracker:
    fit_tracker: FitProgressTracker = None
    validation_tracker: EvaluationProgressTracker = None
    test_tracker: EvaluationProgressTracker = None

    is_training: bool = False
    log_interval: int = 1

    """
    Status
    """

    def is_epoch_progress_done(self) -> bool:
        if self.is_training:
            return self.fit_tracker.is_epoch_progress_done()

    def is_batch_progress_done(self, stage: RunningStage) -> bool:
        if self.is_training:
            return self.fit_tracker.is_batch_progress_done(stage)
        return getattr(self, f"{stage}_tracker").is_batch_progress_done()

    def get_batch_num(self, stage: RunningStage) -> int:
        if self.is_training:
            tracker = getattr(self.fit_tracker, f"{stage}_tracker")
        else:
            tracker = getattr(self, f"{stage}_tracker").tracker
        return tracker.total

    def get_epoch_num(self, stage: RunningStage) -> int:
        if self.is_training:
            return self.fit_tracker.epochs_tracker.total
        return 0

    def should_log(self, batch_idx: int) -> None:
        return (batch_idx == 0) or ((batch_idx + 1) % self.log_interval == 0)

    """
    Operations
    """

    def increment_batch_progress(self, stage: RunningStage) -> None:
        if self.is_training:
            self.fit_tracker.increment_batch_progress(stage)
        else:
            getattr(self, f"{stage}_tracker").increment_batch_progress()

    def increment_epoch_progress(self) -> None:
        self.fit_tracker.increment_epoch_progress()

    def set_stop_training(self, value: bool) -> None:
        self.fit_tracker.stop_training = value

    """
    Initializers
    """

    def initialize_fit_progress(
        self,
        num_epochs: Optional[int],
        num_steps: Optional[int],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        **kwargs,
    ) -> None:
        assert num_epochs is not None or num_steps is not None, "`num_epochs` or `num_steps` must be passed."

        # train: limit batches
        max_train_batches = len(train_loader)
        limit_train_batches = kwargs.get("limit_train_batches", None)
        if limit_train_batches is not None:
            max_train_batches = min(limit_train_batches, max_train_batches)

        # train: epochs and steps
        if num_epochs is None:
            num_epochs = np.ceil(num_steps / max_train_batches)

        if num_steps is not None:
            num_epochs_for_num_steps = int(np.ceil(num_steps / max_train_batches))
            if num_epochs < num_epochs_for_num_steps:
                # if we do not have enough batches across epochs, adjust epoch number
                num_epochs = num_epochs_for_num_steps
            else:
                # if we have enough batches to cover the num steps, do nothing
                num_steps = None

        # validation: limit batches and validation interval
        max_validation_batches = None
        validation_interval = kwargs.get("validation_interval", None)
        if validation_loader is not None:
            max_validation_batches = len(validation_loader)
            limit_validation_batches = kwargs.get("limit_validation_batches", None)
            if limit_validation_batches is not None:
                max_validation_batches = min(limit_train_batches, max_train_batches)

            if validation_interval is not None:
                validation_interval = np.linspace(
                    max_train_batches / validation_interval, max_train_batches, validation_interval, dtype=int
                ).tolist()[:-1]

        self.fit_tracker = FitProgressTracker(
            epochs_tracker=Tracker(max=num_epochs),
            steps_tracker=Tracker(max=num_steps),
            train_tracker=Tracker(max=max_train_batches),
            validation_tracker=Tracker(max=max_validation_batches),
            progress_bar=kwargs.get("progress_bar", True),
            validation_interval=validation_interval,
        )
        self.fit_tracker.initialize_epoch_progress()
        self.is_training = True
        self.log_interval = kwargs.get("log_interval", 1)

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        max_batches = len(loader)
        limit_batches = kwargs.get("limit_batches", None)
        if limit_batches is not None:
            max_batches = min(limit_batches, max_batches)

        tracker = EvaluationProgressTracker(
            tracker=Tracker(max=max_batches), progress_bar=kwargs.get("progress_bar", True)
        )
        setattr(self, f"{stage}_tracker", tracker)
        self.is_training = False
        self.log_interval = kwargs.get("log_interval", 1)

    def initialize_batch_progress(self, stage: RunningStage) -> None:
        if self.is_training:
            self.fit_tracker.initialize_batch_progress(stage)
        else:
            getattr(self, f"{stage}_tracker").initialize_batch_progress(stage)


@dataclass
class ActiveProgressTracker(ProgressTracker):
    num_rounds: int = -1
    total_epochs: int = 0
    total_train_batches: int = 0
    total_validation_batches: int = 0
    total_test_batches: int = 0
    num_pool_batches: int = 0
    total_pool_batches: int = 0

    def reset_rounds(self) -> None:
        self.num_rounds = 0

    def reset_total_epochs(self) -> None:
        self.total_epochs = 0

    def reset_total_batches(self, stage: RunningStage) -> None:
        setattr(self, f"total_{stage}_batches", 0)

    def increment_rounds(self) -> None:
        self.num_rounds += 1

    def increment_total_epochs(self) -> None:
        self.total_epochs += self.num_epochs

    def increment_total_batches(self, stage: RunningStage) -> None:
        current = getattr(self, f"total_{stage}_batches")
        num_batches = getattr(self, f"num_{stage}_batches")
        setattr(self, f"total_{stage}_batches", current + num_batches)

    def get_batch_num(self, stage: RunningStage, batch_idx: int) -> int:
        return getattr(self, f"total_{stage}_batches") + getattr(self, f"num_{stage}_batches")

    def get_epoch_num(self, stage: RunningStage) -> int:
        if stage == RunningStage.TEST:
            return self.num_rounds
        return getattr(self, "total_epochs") + getattr(self, "num_epochs")
