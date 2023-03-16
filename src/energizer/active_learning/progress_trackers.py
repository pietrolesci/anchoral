from dataclasses import dataclass

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

    def max_reached(self) -> bool:
        return self.max < self.query_size + self.total


@dataclass
class ActiveProgressTracker(ProgressTracker):
    round_tracker: RoundTracker = RoundTracker()
    budget_tracker: BudgetTracker = BudgetTracker()
    pool_tracker: StageTracker = StageTracker(stage=RunningStage.POOL)

    @property
    def global_round(self) -> int:
        return self.round_tracker.total

    @property
    def global_budget(self) -> int:
        return self.budget_tracker.total

    @property
    def safe_global_epoch(self) -> int:
        return (
            self.global_round
            if self.current_stage in (RunningStage.TEST, RunningStage.POOL)
            else super().safe_global_epoch
        )

    def is_active_fit_done(self) -> bool:
        return self.round_tracker.max_reached() or self.budget_tracker.max_reached()

    def setup_meta_tracking(
        self,
        max_rounds: int,
        max_budget: int,
        query_size: int,
        initial_budget: int,
        has_pool: bool,
        has_test: bool,
        has_validation: bool,
        **kwargs,
    ) -> None:
        """Create progress bars."""
        
        self.log_interval = kwargs.pop("log_interval", 1)
        self.enable_progress_bar = kwargs.pop("enable_progress_bar", True)
        
        self.round_tracker.reset()
        self.budget_tracker.reset()
        self.round_tracker.max = max_rounds
        self.budget_tracker = BudgetTracker(
            max=max_budget, total=initial_budget, current=initial_budget, query_size=query_size
        )
        
        self.fit_tracker.has_validation = has_validation
        
        if self.enable_progress_bar:
            self.round_tracker.make_progress_bar()
            self.fit_tracker.make_progress_bar()
            if has_test:
                self.test_tracker.make_progress_bar()
            if has_pool:
                self.pool_tracker.make_progress_bar()

    def setup_tracking(self, **kwargs) -> None:
        """Only do the math."""

        self.fit_tracker.setup_tracking(
            max_epochs=kwargs.get("max_epochs"),
            min_steps=kwargs.get("min_steps"),
            num_train_batches=kwargs.get("num_train_batches"),
            num_validation_batches=kwargs.get("num_validation_batches"),
            limit_train_batches=kwargs.get("limit_train_batches"),
            limit_validation_batches=kwargs.get("limit_validation_batches"),
            validation_interval=kwargs.get("validation_interval"),
        )
        self.test_tracker.max = min(kwargs.get("num_test_batches"), kwargs.get("limit_test_batches") or float("Inf"))
        self.pool_tracker.max = min(kwargs.get("num_pool_batches"), kwargs.get("limit_pool_batches") or float("Inf"))        

    def increment_round(self) -> None:
        self.round_tracker.increment()
        self.budget_tracker.increment()

    def end_active_fit(self) -> None:
        self.round_tracker.close_progress_bar()
        self.fit_tracker.close_progress_bar()
        self.test_tracker.close_progress_bar()
        self.pool_tracker.close_progress_bar()

    def end_epoch(self) -> None:
        self._get_stage_tracker().terminate_progress_bar()

    def end_fit(self) -> None:
        self.fit_tracker.terminate_progress_bar()