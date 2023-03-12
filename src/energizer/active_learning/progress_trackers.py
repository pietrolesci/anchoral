from dataclasses import dataclass

from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.energizer.enums import RunningStage
from src.energizer.progress_trackers import EpochTracker, FitTracker, ProgressTracker, StageTracker, Tracker


@dataclass
class RoundTracker(Tracker):
    current: int = 0
    total: int = 0

    # def reset_current(self) -> None:
    #     self.current = -1

    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed rounds",
            dynamic_ncols=True,
            leave=True,
            colour="#32a852",
        )

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None and self.current > 0:
            self.progress_bar.update(1)


@dataclass
class BudgetTracker(Tracker):
    query_size: int = None

    def increment(self) -> None:
        self.current += self.query_size
        self.total += self.query_size

    def set_initial_budget(self, initial_budget: int) -> None:
        self.current = initial_budget
        self.total = initial_budget


@dataclass
class ActiveProgressTracker(ProgressTracker):
    round_tracker: RoundTracker = None
    pool_tracker: StageTracker = None
    stop_active_training: bool = False

    """
    Status
    """

    @property
    def num_rounds(self) -> int:
        return self.round_tracker.total

    @property
    def budget(self) -> int:
        return self.budget_tracker.total

    def get_epoch_num(self) -> int:
        return (
            self.num_rounds
            if self.current_stage in (RunningStage.TEST, RunningStage.POOL)
            else self.fit_tracker.epoch_tracker.total
        )

    def is_fit_done(self) -> bool:
        return self.fit_tracker.epoch_tracker.max_reached() or self.fit_tracker.stop_training

    def is_epoch_done(self) -> bool:
        cond = self._get_active_tracker().max_reached()
        if self.current_stage == RunningStage.TRAIN and self.is_training:
            cond = cond or self.fit_tracker.stop_training
        return cond

    def is_active_fit_done(self) -> bool:
        cond = self.round_tracker.max_reached() or self.budget_tracker.max_reached() or self.stop_active_training
        if cond:
            self.round_tracker.close_progress_bar()
            self.fit_tracker.close_progress_bars()
            self.test_tracker.close_progress_bar()
            self.pool_tracker.close_progress_bar()
        return cond

    """
    Initializers
    """

    def initialize_active_fit_progress(
        self, num_rounds: int, max_budget: int, query_size: int, initial_budget: int, **kwargs
    ) -> None:
        self.round_tracker = RoundTracker(max=num_rounds)
        self.budget_tracker = BudgetTracker(max=max_budget, query_size=query_size)
        self.budget_tracker.set_initial_budget(initial_budget)
        self.fit_tracker = FitTracker(
            epoch_tracker=EpochTracker(),
            train_tracker=StageTracker(stage=RunningStage.TRAIN),
            validation_tracker=StageTracker(stage=RunningStage.VALIDATION),
            step_tracker=Tracker(),
        )
        self.test_tracker = StageTracker(stage=RunningStage.TEST)
        self.pool_tracker = StageTracker(stage=RunningStage.POOL)
        if kwargs.get("progress_bar", True):
            self.round_tracker.make_progress_bar()
            self.fit_tracker.make_progress_bars()
            self.test_tracker.make_progress_bar()
            self.pool_tracker.make_progress_bar()

    def initialize_fit_progress(self, *args, **kwargs) -> None:
        self.is_training = True
        self.log_interval = kwargs.get("log_interval", 1)

        # NOTE: here we update (not re-create) the fit_tracker
        # also we do not re-create the progress bar
        self.fit_tracker.update_from_hparams(**self._solve_hparams(*args, **kwargs))
        self.fit_tracker.reset()  # <- reset current counts and progress bar line

    def initialize_evaluation_progress(self, stage: RunningStage, loader: DataLoader, **kwargs) -> None:
        self.is_training = False
        self.log_interval = kwargs.get("log_interval", 1)

        # NOTE: we do not reset the tracker nor the progress bar
        tracker = getattr(self, f"{stage}_tracker")
        tracker.max = self._solve_num_batches(loader, kwargs.get("limit_batches", None))
        tracker.reset()  # <- reset current counts and progress bar line

    """
    Operations
    """

    def increment_active_fit_progress(self) -> None:
        self.round_tracker.increment()
        self.budget_tracker.increment()

    def set_stop_active_training(self, value: bool) -> None:
        self.stop_active_training = value
