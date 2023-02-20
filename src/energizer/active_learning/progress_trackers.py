import sys
from dataclasses import dataclass

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.containers import RunningStage
from src.energizer.progress_trackers import ProgressTracker, StageTracker, Tracker


@dataclass
class RoundTracker(Tracker):
    current: int = -1
    total: int = -1

    def reset_current(self) -> None:
        self.current = -1

    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed rounds",
            dynamic_ncols=True,
            leave=True,
            file=sys.stderr,
        )

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None and self.current > 0:
            self.progress_bar.update(1)


@dataclass
class ActiveProgressTracker(ProgressTracker):
    round_tracker: RoundTracker = None
    pool_tracker: StageTracker = None

    """
    Status
    """

    def is_round_progress_done(self) -> None:
        return self.round_tracker.max_reached()

    @property
    def num_rounds(self) -> int:
        return self.round_tracker.total

    def get_epoch_num(self, stage: RunningStage) -> int:
        if stage == RunningStage.TEST:
            return self.num_rounds
        return self.fit_tracker.epoch_tracker.total

    """
    Initializers
    """

    def initialize_active_fit_progress(self, num_rounds: int, **kwargs) -> None:
        self.round_tracker = RoundTracker(max=num_rounds, show_progress_bar=kwargs.get("progress_bar", True))

    def initialize_round_progress(self) -> None:
        self.round_tracker.initialize()

    def initialize_fit_progress(self, *args, **kwargs) -> None:
        if self.fit_tracker is None:
            super().initialize_fit_progress(*args, **kwargs)
            return
        hparams = self._solve_hparams(*args, **kwargs)
        self.fit_tracker.update_from_hparams(**hparams)
        self.fit_tracker.initialize()
        self.fit_tracker.epoch_tracker.leave = False
        self.fit_tracker.train_tracker.leave = False
        self.fit_tracker.validation_tracker.leave = False
        self.is_training = True
        self.log_interval = kwargs.get("log_interval", 1)

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        if getattr(self, f"{stage}_tracker") is None:
            super().initialize_evaluation_progress(stage, loader, **kwargs)
            tracker = getattr(self, f"{stage}_tracker")
            tracker.leave = False
            return

        self.is_training = False
        self.log_interval = kwargs.get("log_interval", 1)

    """
    Operations
    """

    def increment_round_progress(self) -> None:
        self.round_tracker.increment()
