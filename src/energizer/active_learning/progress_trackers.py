from dataclasses import dataclass, field

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
    round_tracker: RoundTracker = field(default_factory=lambda: RoundTracker())
    budget_tracker: BudgetTracker = field(default_factory=lambda: BudgetTracker())
    pool_tracker: StageTracker = field(default_factory=lambda: StageTracker(stage=RunningStage.POOL))

    has_pool: bool = False
    has_test: bool = False

    """Properties"""

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

    """Super outer loops"""

    def is_active_fit_done(self) -> bool:
        return self.round_tracker.max_reached() or self.budget_tracker.max_reached()

    def end_active_fit(self) -> None:
        self.round_tracker.close_progress_bar()
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()
        self.test_tracker.close_progress_bar()
        self.pool_tracker.close_progress_bar()

    def increment_round(self) -> None:
        self.round_tracker.increment()
        self.budget_tracker.increment()

    """Outer loops"""

    def start_fit(self) -> None:
        super().start_fit()
        if self.enable_progress_bar:
            self.epoch_tracker.progress_bar.set_postfix_str("")

    def end_fit(self) -> None:
        self.epoch_tracker.terminate_progress_bar()
        self.train_tracker.terminate_progress_bar()
        self.validation_tracker.terminate_progress_bar()

    """Stage trackers"""

    def end(self) -> None:
        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training

    """Helpers"""

    def setup(
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

        self.has_validation = has_validation
        self.has_pool = has_pool
        self.has_test = has_test

        if self.enable_progress_bar:
            self.round_tracker.make_progress_bar()
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
            if self.has_test:
                self.test_tracker.make_progress_bar()
            if self.has_pool:
                self.pool_tracker.make_progress_bar()

    def setup_round_tracking(self, **kwargs) -> None:
        self._setup_fit(
            max_epochs=kwargs.get("max_epochs"),
            min_steps=kwargs.get("min_steps"),
            num_train_batches=kwargs.get("num_train_batches"),
            num_validation_batches=kwargs.get("num_validation_batches"),
            limit_train_batches=kwargs.get("limit_train_batches"),
            limit_validation_batches=kwargs.get("limit_validation_batches"),
            validation_interval=kwargs.get("validation_interval"),
        )
        for stage in (RunningStage.TEST, RunningStage.POOL):
            if getattr(self, f"has_{stage}"):
                limit_batches = kwargs.get(f"limit_{stage}_batches") or float("Inf")
                getattr(self, f"{stage}_tracker").max = min(kwargs.get(f"num_{stage}_batches"), limit_batches)
