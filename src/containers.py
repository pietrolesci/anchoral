from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from numpy import ndarray
from torch.utils.data import BatchSampler, DataLoader, SequentialSampler

from src.enums import RunningStage
from src.utilities import move_to_cpu


@dataclass
class Counter:
    # TODO: once accumulate batching is implemented, num_steps will be different
    # from batch_idx
    num_epochs: int = 0
    num_steps: int = 0
    num_train_batches: int = 0
    num_validation_batches: int = 0
    num_test_batches: int = 0

    def reset(self) -> None:
        self.reset_epochs()
        self.reset_steps()
        self.reset_batches(RunningStage.TRAIN)
        self.reset_batches(RunningStage.VALIDATION)
        self.reset_batches(RunningStage.TEST)

    def reset_epochs(self) -> None:
        self.num_epochs = 0

    def reset_steps(self) -> None:
        self.num_steps = 0

    def reset_batches(self, stage: RunningStage) -> None:
        setattr(self, f"num_{stage}_batches", 0)

    def increment_epochs(self) -> None:
        self.num_epochs += 1

    def increment_steps(self) -> None:
        self.num_steps += 1

    def increment_batches(self, stage: RunningStage) -> None:
        key = f"num_{stage}_batches"
        current = getattr(self, key)
        setattr(self, key, current + 1)

    # def register_num_batches(self, stage: RunningStage, total: int) -> None:
    #     setattr(self, f"num_{stage}_batches", total)


@dataclass
class ActiveCounter(Counter):
    num_rounds: int = 0
    total_epochs: int = 0
    total_train_batches: int = 0
    total_validation_batches: int = 0
    total_test_batches: int = 0

    def reset(self) -> None:
        super().reset()
        self.reset_rounds()
        self.reset_total_epochs()
        self.reset_total_batches(RunningStage.TRAIN)
        self.reset_total_batches(RunningStage.VALIDATION)
        self.reset_total_batches(RunningStage.TEST)

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


class EpochOutput(list):
    """This is a simple list that moves things to cpu."""

    def append(self, _x: Any) -> None:
        super().append(move_to_cpu(_x))

    def __add__(self, _x: Any) -> None:
        super().__add__(move_to_cpu(_x))

    def __iadd__(self, _x) -> None:
        super().__iadd__(move_to_cpu(_x))


@dataclass
class FitEpochOutput:
    """Simple container for train and validation outputs for each epoch of fitting.

    This deserves a separate container bacause during fit we obtain EpochOutput's
    from both train and, possibly, validation.
    """

    train: EpochOutput = None
    validation: EpochOutput = None


@dataclass
class MetadataParserMixin:
    """Simple wrapper for outputs that allows to add metadata."""

    hparams: Optional[Dict] = None

    def __post_init__(self) -> None:
        ignore = ["self", "loader"]
        hparams = {k: v for k, v in self.hparams.items() if not any([x in k for x in ignore])}

        for stage in RunningStage:
            loader = self.hparams.get(f"{stage}_loader", None)
            if loader is None:
                continue

            loader_hparams = self._dataloader_hparams(loader)
            if stage == RunningStage.TRAIN:
                hparams = {**hparams, **loader_hparams}
            else:
                hparams["eval_batch_size"] = loader_hparams.get("batch_size")

        self.hparams = hparams

    def _dataloader_hparams(self, loader: DataLoader) -> Dict:
        loader = loader.sampler if isinstance(loader.sampler, BatchSampler) else loader
        return {
            "batch_size": loader.batch_size,
            "drop_last": loader.drop_last,
            "shuffle": not isinstance(loader.sampler, SequentialSampler),
        }

    def __repr__(self) -> str:
        s = f"{self.__class__.__name__}(hparams={self.hparams}, output="
        if self.output is None:
            return f"{s}None)"
        return f"{s} ..{len(self.output)} epochs.. )"


@dataclass
class FitOutput(MetadataParserMixin):
    _cls_output = FitEpochOutput
    output: List[FitEpochOutput] = field(default_factory=list)

    def append(self, _x: _cls_output):
        assert isinstance(_x, self._cls_output), f"You can only append `{self._cls_output.__name__}`s, not {type(_x)}"
        self.output.append(_x)

    def __getitem__(self, idx: int) -> _cls_output:
        return self.output[idx]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(output= ..{len(self.output)} outputs.. , hparams={self.hparams})"


@dataclass
class EvaluationOutput(MetadataParserMixin):
    output: EpochOutput = None


"""
Active learning
"""


@dataclass
class QueryOutput:
    """Output of a run on an entire pool dataloader.

    metrics: Metrics aggregated over the entire pool dataloader.
    output: List of individual outputs at the batch level.
    topk_scores: TopK scores for the pool instances.
    indices: Indices corresponding to the topk instances to query.
    """

    topk_scores: ndarray = None
    indices: List[int] = None
    output: Optional[List] = None


@dataclass
class RoundOutput:
    fit: FitOutput = None
    test: EvaluationOutput = None
    query: QueryOutput = None


@dataclass
class ActiveFitOutput(FitOutput):
    _cls_output = RoundOutput
    output = List[RoundOutput]

    def __post_init__(self) -> None:
        ignore = ["self", "datamodule"]
        hparams = {k: v for k, v in self.hparams.items() if not any([x in k for x in ignore])}

        datamodule = self.hparams.get("active_datamodule")
        self.hparams = {**hparams, **datamodule.hparams}
