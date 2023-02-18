from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from numpy import ndarray
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.enums import RunningStage


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
class FitProgressTracker:
    epochs_tracker: Tracker
    steps_tracker: Tracker
    train_tracker: Tracker
    validation_tracker: Tracker

    epochs_progress_bar: Optional[tqdm] = None
    train_progress_bar: Optional[tqdm] = None
    validation_progress_bar: Optional[tqdm] = None

    progress_bar: bool = True

    def is_done(self) -> bool:
        cond = self.epochs_tracker.max_reached()
        if cond and self.progress_bar:
            self.epochs_progress_bar.close()
        return cond

    def is_epoch_done(self, stage: RunningStage) -> bool:
        cond = getattr(self, f"{stage}_tracker").max_reached()        
        if cond and self.progress_bar:
            getattr(self, f"{stage}_progress_bar").close()
        return cond

    def initialize_tracking(self) -> None:
        if self.progress_bar:
            self.epochs_progress_bar = tqdm(
                total=self.epochs_tracker.max, desc="Completed epochs", dynamic_ncols=True, leave=True
            )

    def initialize_epoch_tracking(self, stage: RunningStage) -> None:
        getattr(self, f"{stage}_tracker").reset_current()
        if self.progress_bar and getattr(self, f"{stage}_tracker").max is not None:
            desc = f"Epoch {self.epochs_tracker.current}".strip() if stage == RunningStage.TRAIN else f"{stage.title()}"
            pbar = tqdm(total=self.train_tracker.max, desc=desc, dynamic_ncols=True, leave=False)
            setattr(self, f"{stage}_progress_bar", pbar)

    def increment_epochs(self) -> None:
        self.epochs_tracker.increment()
        if self.epochs_progress_bar is not None:
            self.epochs_progress_bar.update(1)

    def increment_batches(self, stage: RunningStage) -> None:
        getattr(self, f"{stage}_tracker").increment()
        batch_progress = getattr(self, f"{stage}_progress_bar", None)
        if batch_progress is not None:
            batch_progress.update(1)

    def increment_steps(self) -> None:
        self.steps_tracker.increment()


@dataclass
class ProgressTracker:
    is_training: bool = False
    fit_tracker: FitProgressTracker = None
    validation_tracker: Tracker = None
    test_tracker: Tracker = None

    def get_batch_num(self, stage: RunningStage) -> int:
        if self.is_training:
            tracker = getattr(self.fit_tracker, f"{stage}_tracker")
        else:
            tracker = getattr(self, f"{stage}_tracker")
        return tracker.total

    def get_epoch_num(self) -> int:
        if self.is_training:
            return self.fit_tracker.epochs_tracker.total
        return 0
    
    def is_done(self) -> bool:
        if self.is_training:
            return self.fit_tracker.is_done()

    def is_epoch_done(self, stage: RunningStage) -> bool:
        if self.is_training:
            return self.fit_tracker.is_epoch_done(stage)
        return getattr(self, f"{stage}_tracker").max_reached()

    def increment_batches(self, stage: RunningStage) -> None:
        if self.is_training:
            self.fit_tracker.increment_batches(stage)
        else:
            getattr(self, f"{stage}_tracker").increment()

    def initialize_fit_tracking(
        self,
        max_epochs: Optional[int],
        train_loader: DataLoader,
        validation_loader: DataLoader,
        **kwargs,
    ) -> None:
        progress_bar = kwargs.get("progress_bar", True)

        # train
        max_train_batches = len(train_loader)
        limit_train_batches = kwargs.get("limit_train_batches", None)
        if limit_train_batches is not None:
            max_train_batches = min(limit_train_batches, max_train_batches)

        # validation
        max_validation_batches = None
        if validation_loader is not None:
            max_validation_batches = len(validation_loader)
            limit_validation_batches = kwargs.get("limit_validation_batches", None)
            if limit_validation_batches is not None:
                max_validation_batches = min(limit_train_batches, max_train_batches)

        self.fit_tracker = FitProgressTracker(
            epochs_tracker=Tracker(max=max_epochs),
            steps_tracker=Tracker(),
            train_tracker=Tracker(max=max_train_batches),
            validation_tracker=Tracker(max=max_validation_batches),
            progress_bar=progress_bar,
        )
        self.fit_tracker.initialize_tracking()
        self.is_training = True

    def initialize_evaluation_tracking(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        max_batches = len(loader)
        limit_batches = kwargs.get("limit_batches", None)
        if limit_batches is not None:
            max_batches = min(limit_batches, max_batches)

        tracker = Tracker(max=max_batches)
        setattr(self, f"{stage}_tracker", tracker)
        self.is_training = False

    def initialize_epoch_tracking(self, stage: RunningStage) -> None:
        if self.is_training:
            self.fit_tracker.initialize_epoch_tracking(stage)
        else:
            getattr(self, f"{stage}_tracker").reset_current()


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
