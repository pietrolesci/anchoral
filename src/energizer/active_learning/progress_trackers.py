from dataclasses import dataclass
from typing import Optional

from tqdm.auto import tqdm

from src.energizer.enums import RunningStage
from src.energizer.progress_trackers import ProgressTracker, StageTracker, Tracker


@dataclass
class RoundTracker(Tracker):
    current: int = 0
    total: int = 0

    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed rounds",
            dynamic_ncols=True,
            leave=True,
            colour="#f7e302",
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


@dataclass
class ActiveProgressTracker(ProgressTracker):
    round_tracker: RoundTracker = None
    pool_tracker: StageTracker = None
    stop_active_training: bool = False

    @property
    def global_round(self) -> int:
        return self.round_tracker.total

    @property
    def budget(self) -> int:
        return self.budget_tracker.total

    def is_active_fit_done(self) -> bool:
        return self.round_tracker.max_reached() or self.budget_tracker.max_reached() or self.stop_active_training

    def initialize_active_fit_progress(
        self,
        max_rounds: int,
        max_budget: int,
        query_size: int,
        initial_budget: int,
        progress_bar: Optional[bool],
        log_interval: Optional[int],
        has_validation: bool,
        has_pool: bool,
    ) -> None:
        self.round_tracker = RoundTracker(max=max_rounds)
        self.budget_tracker = BudgetTracker(
            max=max_budget, total=initial_budget, current=initial_budget, query_size=query_size
        )
        self.test_tracker = StageTracker(stage=RunningStage.TEST)
        self.fit_tracker.has_validation = has_validation
        
        if has_pool:
            self.pool_tracker = StageTracker(stage=RunningStage.POOL)
        
        if progress_bar:
            self.round_tracker.make_progress_bar()
            self.fit_tracker.make_progress_bars()
            self.test_tracker.make_progress_bar()
            if has_pool:
                self.pool_tracker.make_progress_bar()

        self.log_interval = log_interval

    def increment_active_fit_progress(self) -> None:
        self.round_tracker.increment()
        self.budget_tracker.increment()

    def finalize_active_fit_progress(self) -> None:
        self.round_tracker.close_progress_bar()
        self.fit_tracker.close_progress_bars()
        self.test_tracker.close_progress_bar()
        self.pool_tracker.close_progress_bar()

    def set_stop_active_training(self, value: bool) -> None:
        self.stop_active_training = value

    def finalize_fit_progress(self) -> None:
        self.fit_tracker.terminate_progress_bars()  # do not close prog_bar
        self.is_fitting = False

    def finalize_epoch_progress(self) -> None:
        self._get_stage_tracker().terminate_progress_bar()
